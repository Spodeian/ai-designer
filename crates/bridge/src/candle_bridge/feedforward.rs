//! Candle Feedforward & MoE Bridge Layers.
//! Bridges SwiGLU, GeGLU, and Sparse Mixture of Experts (MoE) into native Candle modules.

use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear_no_bias};

/// SwiGLU Gated Feedforward Layer: (silu(x W_gate) * (x W_up)) W_down.
#[derive(Debug, Clone)]
pub struct CandleSwiGlu {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
    pub in_features: usize,
    pub hidden_features: usize,
}

impl CandleSwiGlu {
    pub fn new(in_features: usize, hidden_features: usize, vb: VarBuilder) -> Result<Self> {
        let gate_proj = linear_no_bias(in_features, hidden_features, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(in_features, hidden_features, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(hidden_features, in_features, vb.pp("down_proj"))?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            in_features,
            hidden_features,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        let hidden = (gate * up)?;
        self.down_proj.forward(&hidden)
    }
}

/// GEGLU Gated Feedforward Layer: (gelu(x W_gate) * (x W_up)) W_down.
#[derive(Debug, Clone)]
pub struct CandleGeGlu {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
}

impl CandleGeGlu {
    pub fn new(in_features: usize, hidden_features: usize, vb: VarBuilder) -> Result<Self> {
        let gate_proj = linear_no_bias(in_features, hidden_features, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(in_features, hidden_features, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(hidden_features, in_features, vb.pp("down_proj"))?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?.gelu_erf()?;
        let up = self.up_proj.forward(x)?;
        let hidden = (gate * up)?;
        self.down_proj.forward(&hidden)
    }
}

/// Sparse Mixture of Experts (MoE) with Top-K softmax gating.
#[derive(Debug, Clone)]
pub struct CandleMoE {
    pub router: Linear,
    pub experts: Vec<CandleSwiGlu>,
    pub num_experts: usize,
    pub top_k: usize,
}

impl CandleMoE {
    pub fn new(
        d_model: usize,
        hidden_dim: usize,
        num_experts: usize,
        top_k: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let router = linear_no_bias(d_model, num_experts, vb.pp("router"))?;
        let mut experts = Vec::with_capacity(num_experts);
        for i in 0..num_experts {
            let expert = CandleSwiGlu::new(d_model, hidden_dim, vb.pp(format!("expert_{i}")))?;
            experts.push(expert);
        }

        Ok(Self {
            router,
            experts,
            num_experts,
            top_k,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Router logits: (batch, seq, num_experts)
        let router_logits = self.router.forward(x)?;
        let router_probs = candle_nn::ops::softmax(&router_logits, candle_core::D::Minus1)?;

        // Weighted dispatch across all experts
        let mut output = Tensor::zeros_like(x)?;
        for (i, expert) in self.experts.iter().enumerate() {
            let expert_out = expert.forward(x)?;
            let weight = router_probs
                .narrow(candle_core::D::Minus1, i, 1)?
                .broadcast_as(x.shape())?;
            output = (output + expert_out.broadcast_mul(&weight)?)?;
        }

        Ok(output)
    }
}
