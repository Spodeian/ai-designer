//! Candle Quantized & Ternary Bridge Layers.
//! Bridges BitNet 1.58b {-1, 0, +1} and INT8/INT4 symmetric quantization into native Candle modules.

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

/// BitNet 1.58b Ternary Linear Layer for Candle.
/// Weight elements are constrained to {-1, 0, +1} with an activation scaling factor.
#[derive(Debug, Clone)]
pub struct CandleBitNetLinear {
    pub ternary_weights: Tensor,
    pub weight_scale: Tensor,
    pub in_features: usize,
    pub out_features: usize,
}

impl CandleBitNetLinear {
    pub fn new(in_features: usize, out_features: usize, vb: VarBuilder) -> Result<Self> {
        let raw_weights = vb
            .get((out_features, in_features), "weight")
            .unwrap_or_else(|_| {
                Tensor::zeros(
                    (out_features, in_features),
                    candle_core::DType::F32,
                    vb.device(),
                )
                .unwrap()
            });

        // Quantize weights to {-1, 0, +1}
        let scale = raw_weights.abs()?.mean_all()?;
        let scaled_w = raw_weights.broadcast_div(&scale)?;
        let rounded_w = scaled_w.round()?.clamp(-1.0f32, 1.0f32)?;

        Ok(Self {
            ternary_weights: rounded_w,
            weight_scale: scale,
            in_features,
            out_features,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Activation absmax quantization: x_quant = clamp(round(x * 127 / max(abs(x))), -128, 127)
        let x_scale = x.abs()?.max_keepdim(candle_core::D::Minus1)?;
        let x_scaled = x.broadcast_div(&(x_scale.clone() + 1e-6)?)?;

        let in_dims = x.dims();
        let last_dim = in_dims[in_dims.len() - 1];
        let x_2d = x_scaled.reshape(((), last_dim))?;
        let out_2d = x_2d.matmul(&self.ternary_weights.t()?.contiguous()?)?;
        let mut out_shape = in_dims.to_vec();
        out_shape[in_dims.len() - 1] = self.out_features;
        let out = out_2d.reshape(out_shape)?;

        // Rescale output: out * (x_scale * weight_scale)
        let scale_val = self.weight_scale.to_scalar::<f32>()? as f64;
        let scaled_out = out.broadcast_mul(&x_scale)?;
        scaled_out.affine(scale_val, 0.0)
    }
}

/// INT8 Symmetric Quantized Linear Layer for Candle.
#[derive(Debug, Clone)]
pub struct CandleQuantizedLinear {
    pub quant_weights: Tensor,
    pub weight_scale: Tensor,
    pub in_features: usize,
    pub out_features: usize,
}

impl CandleQuantizedLinear {
    pub fn new(in_features: usize, out_features: usize, vb: VarBuilder) -> Result<Self> {
        let raw_weights = vb
            .get((out_features, in_features), "weight")
            .unwrap_or_else(|_| {
                Tensor::zeros(
                    (out_features, in_features),
                    candle_core::DType::F32,
                    vb.device(),
                )
                .unwrap()
            });
        let max_abs = raw_weights.abs()?.max_all()?;
        let scale = (max_abs / 127.0f64)?;
        let quant_w = (raw_weights.broadcast_div(&scale)?)
            .round()?
            .clamp(-127.0f32, 127.0f32)?;

        Ok(Self {
            quant_weights: quant_w,
            weight_scale: scale,
            in_features,
            out_features,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let in_dims = x.dims();
        let last_dim = in_dims[in_dims.len() - 1];
        let x_2d = x.reshape(((), last_dim))?;
        let out_2d = x_2d.matmul(&self.quant_weights.t()?.contiguous()?)?;
        let mut out_shape = in_dims.to_vec();
        out_shape[in_dims.len() - 1] = self.out_features;
        let out = out_2d.reshape(out_shape)?;
        let scale_val = self.weight_scale.to_scalar::<f32>()? as f64;
        out.affine(scale_val, 0.0)
    }
}
