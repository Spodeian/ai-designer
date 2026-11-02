//! Candle Normalization Bridge Layers.
//! Bridges RMSNorm, Reversible Instance Normalization (RevIN), and Spectral Norm into native Candle modules.

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

/// Root Mean Square Layer Normalization (RMSNorm) module for Candle.
#[derive(Debug, Clone)]
pub struct CandleRmsNorm {
    pub weight: Tensor,
    pub eps: f64,
}

impl CandleRmsNorm {
    pub fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get((dim,), "weight").unwrap_or_else(|_| {
            Tensor::ones((dim,), candle_core::DType::F32, vb.device()).unwrap()
        });
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let variance = x.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let rms = (variance + self.eps)?.sqrt()?;
        let normalized = x.broadcast_div(&rms)?;
        normalized.broadcast_mul(&self.weight)
    }
}

/// Reversible Instance Normalization (RevIN) for time series and continuous signals in Candle.
#[derive(Debug, Clone)]
pub struct CandleRevIn {
    pub num_features: usize,
    pub eps: f64,
    pub gamma: Tensor,
    pub beta: Tensor,
}

impl CandleRevIn {
    pub fn new(num_features: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let gamma = vb.get((num_features,), "gamma").unwrap_or_else(|_| {
            Tensor::ones((num_features,), candle_core::DType::F32, vb.device()).unwrap()
        });
        let beta = vb.get((num_features,), "beta").unwrap_or_else(|_| {
            Tensor::zeros((num_features,), candle_core::DType::F32, vb.device()).unwrap()
        });
        Ok(Self {
            num_features,
            eps,
            gamma,
            beta,
        })
    }

    /// Forward normalize: x -> (x - mean) / std * gamma + beta.
    pub fn normalize(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        // x: (batch, seq, features)
        let mean = x.mean_keepdim(1)?;
        let x_centered = x.broadcast_sub(&mean)?;
        let var = x_centered.sqr()?.mean_keepdim(1)?;
        let std = (var + self.eps)?.sqrt()?;
        let x_norm = x_centered.broadcast_div(&std)?;
        let out = (x_norm.broadcast_mul(&self.gamma)? + self.beta.broadcast_as(x.shape())?)?;
        Ok((out, mean, std))
    }

    /// Denormalize back to original scale: y -> (y - beta) / gamma * std + mean.
    pub fn denormalize(&self, y: &Tensor, mean: &Tensor, std: &Tensor) -> Result<Tensor> {
        let y_sub_beta = y.broadcast_sub(&self.beta)?;
        let y_scaled = y_sub_beta.broadcast_div(&self.gamma)?;
        let y_with_std = y_scaled.broadcast_mul(std)?;
        y_with_std.broadcast_add(mean)
    }
}
