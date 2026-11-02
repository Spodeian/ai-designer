//! Candle State-Space Models (SSM) Bridge Layers.
//! Bridges Mamba S6 and Mamba2 SSD recurrence operators into native Candle modules.

use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear_no_bias};

/// Mamba S6 Selective State Space Model recurrent block in Candle.
#[derive(Debug, Clone)]
pub struct CandleMamba {
    pub in_proj: Linear,
    pub out_proj: Linear,
    pub x_proj: Linear,
    pub dt_proj: Linear,
    pub a_log: Tensor,
    pub d: Tensor,
    pub d_model: usize,
    pub d_state: usize,
    pub expand: usize,
}

impl CandleMamba {
    pub fn new(d_model: usize, d_state: usize, expand: usize, vb: VarBuilder) -> Result<Self> {
        let d_inner = d_model * expand;
        let in_proj = linear_no_bias(d_model, d_inner * 2, vb.pp("in_proj"))?;
        let out_proj = linear_no_bias(d_inner, d_model, vb.pp("out_proj"))?;
        let dt_rank = d_model.div_ceil(16);
        let x_proj = linear_no_bias(d_inner, dt_rank + d_state * 2, vb.pp("x_proj"))?;
        let dt_proj = linear_no_bias(dt_rank, d_inner, vb.pp("dt_proj"))?;

        let a_log = vb
            .get((d_inner, d_state), "A_log")
            .unwrap_or_else(|_| Tensor::full(-1.0f32, (d_inner, d_state), vb.device()).unwrap());
        let d = vb.get((d_inner,), "D").unwrap_or_else(|_| {
            Tensor::ones((d_inner,), candle_core::DType::F32, vb.device()).unwrap()
        });

        Ok(Self {
            in_proj,
            out_proj,
            x_proj,
            dt_proj,
            a_log,
            d,
            d_model,
            d_state,
            expand,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: (batch, seq, d_model)
        let (b_sz, seq_len, _d) = x.dims3()?;
        let d_inner = self.d_model * self.expand;

        let projected = self.in_proj.forward(x)?;
        let x_branch = projected.narrow(candle_core::D::Minus1, 0, d_inner)?;
        let res_branch = projected.narrow(candle_core::D::Minus1, d_inner, d_inner)?;

        // Activation gate
        let x_act = candle_nn::ops::silu(&x_branch)?;

        // Discretized recurrence over sequence
        let mut h = Tensor::zeros(
            (b_sz, d_inner, self.d_state),
            candle_core::DType::F32,
            x.device(),
        )?;
        let a = self.a_log.exp()?.neg()?; // (d_inner, d_state)

        let mut y_steps = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let xt = x_act.narrow(1, t, 1)?.squeeze(1)?; // (b, d_inner)
            let params = self.x_proj.forward(&xt)?;
            let dt_rank = self.d_model.div_ceil(16);
            let dt = candle_nn::ops::silu(&self.dt_proj.forward(&params.narrow(1, 0, dt_rank)?)?)?;
            let b_mat = params.narrow(1, dt_rank, self.d_state)?; // (b, d_state)
            let c_mat = params.narrow(1, dt_rank + self.d_state, self.d_state)?; // (b, d_state)

            // Discretize: dA = exp(dt * A)
            let dt_exp = dt.unsqueeze(2)?; // (b, d_inner, 1)
            let da = (dt_exp.broadcast_mul(&a.unsqueeze(0)?)?).exp()?;

            // dB = dt * B
            let db = dt_exp.broadcast_mul(&b_mat.unsqueeze(1)?)?;

            // h = dA * h + dB * x
            let xt_exp = xt.unsqueeze(2)?;
            h = (h.broadcast_mul(&da)? + db.broadcast_mul(&xt_exp)?)?;

            // y = h * C + D * x
            let y_t = (h.broadcast_mul(&c_mat.unsqueeze(1)?)?).sum(2)?;
            let y_skip = (y_t + xt.broadcast_mul(&self.d)?)?;
            y_steps.push(y_skip.unsqueeze(1)?);
        }

        let y = Tensor::cat(&y_steps.iter().collect::<Vec<_>>(), 1)?;
        let y_gated = (y * candle_nn::ops::silu(&res_branch)?)?;
        self.out_proj.forward(&y_gated)
    }
}
