//! Candle Optimization-as-Architecture & Associative Memory Bridge Layers.
//! Bridges LISTA (Learned Iterative Shrinkage-Thresholding Algorithm) and Modern Hopfield.

use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear_no_bias};

/// Learned Iterative Shrinkage-Thresholding Algorithm (LISTA) layer for sparse coding.
#[derive(Debug, Clone)]
pub struct CandleLista {
    pub we: Linear,    // Measurement matrix W_e
    pub ws: Linear,    // Self-recurrence W_s
    pub theta: Tensor, // Learnable soft-threshold parameters
    pub num_stages: usize,
}

impl CandleLista {
    pub fn new(dim: usize, num_stages: usize, lambda: f64, vb: VarBuilder) -> Result<Self> {
        let we = linear_no_bias(dim, dim, vb.pp("we"))?;
        let ws = linear_no_bias(dim, dim, vb.pp("ws"))?;
        let theta = vb
            .get((dim,), "theta")
            .unwrap_or_else(|_| Tensor::full(lambda as f32, (dim,), vb.device()).unwrap());

        Ok(Self {
            we,
            ws,
            theta,
            num_stages,
        })
    }

    /// Soft-thresholding operator: sign(x) * max(|x| - theta, 0).
    fn soft_threshold(&self, x: &Tensor) -> Result<Tensor> {
        let sign = x.sign()?;
        let abs = x.abs()?;
        let sub = (abs.broadcast_sub(&self.theta)?).clamp(0.0f32, f32::MAX)?;
        sign.broadcast_mul(&sub)
    }

    pub fn forward(&self, y: &Tensor) -> Result<Tensor> {
        let c = self.we.forward(y)?;
        let mut x = self.soft_threshold(&c)?;

        for _ in 0..self.num_stages {
            let recon = self.ws.forward(&x)?;
            let step = (c.clone() + recon)?;
            x = self.soft_threshold(&step)?;
        }

        Ok(x)
    }
}

/// Continuous Modern Hopfield associative retrieval layer for Candle.
#[derive(Debug, Clone)]
pub struct CandleHopfield {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub beta: f64,
}

impl CandleHopfield {
    pub fn new(dim: usize, beta: f64, vb: VarBuilder) -> Result<Self> {
        let q_proj = linear_no_bias(dim, dim, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(dim, dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(dim, dim, vb.pp("v_proj"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            beta,
        })
    }

    pub fn forward(&self, query: &Tensor, memory: &Tensor) -> Result<Tensor> {
        // Query projection: (b, q_len, dim)
        let q = self.q_proj.forward(query)?;
        // Memory projection: (b, m_len, dim)
        let k = self.k_proj.forward(memory)?;
        let v = self.v_proj.forward(memory)?;

        // Energy retrieval weights: softmax(beta * Q K^T)
        let sim = (q.matmul(&k.t()?)? * self.beta)?;
        let p = candle_nn::ops::softmax(&sim, candle_core::D::Minus1)?;
        p.matmul(&v)
    }
}
