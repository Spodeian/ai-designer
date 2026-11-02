//! Candle Attention & Associative Bridge Layers.
//! Bridges MHA, GQA, MLA, DiffAttn, and RoPE operators into native Candle modules.

use candle_core::{Device, IndexOp, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear, linear_no_bias};

/// Rotary Position Embedding (RoPE) operator for Candle.
#[derive(Debug, Clone)]
pub struct CandleRoPE {
    pub dim: usize,
    pub max_seq_len: usize,
    pub base: f64,
    cos_cache: Tensor,
    sin_cache: Tensor,
}

impl CandleRoPE {
    pub fn new(dim: usize, max_seq_len: usize, base: f64, dev: &Device) -> Result<Self> {
        let half_dim = dim / 2;
        let mut inv_freq = Vec::with_capacity(half_dim);
        for i in 0..half_dim {
            let freq = 1.0 / base.powf((2 * i) as f64 / dim as f64);
            inv_freq.push(freq as f32);
        }
        let inv_freq_t = Tensor::from_vec(inv_freq, (half_dim,), dev)?;

        let mut pos = Vec::with_capacity(max_seq_len);
        for p in 0..max_seq_len {
            pos.push(p as f32);
        }
        let pos_t = Tensor::from_vec(pos, (max_seq_len, 1), dev)?;
        let freqs = pos_t.matmul(&inv_freq_t.unsqueeze(0)?)?; // (max_seq_len, half_dim)

        let cos_cache = freqs.cos()?;
        let sin_cache = freqs.sin()?;

        Ok(Self {
            dim,
            max_seq_len,
            base,
            cos_cache,
            sin_cache,
        })
    }

    pub fn apply(&self, x: &Tensor, seq_len: usize) -> Result<Tensor> {
        // x: (batch, seq, heads, dim) or (batch, heads, seq, dim)
        let cos = self.cos_cache.i(..seq_len)?;
        let sin = self.sin_cache.i(..seq_len)?;

        // Broadcast matching dims
        let half_dim = self.dim / 2;
        let x1 = x.narrow(candle_core::D::Minus1, 0, half_dim)?;
        let x2 = x.narrow(candle_core::D::Minus1, half_dim, half_dim)?;

        let neg_x2 = x2.neg()?;
        let rotated = Tensor::cat(&[&neg_x2, &x1], candle_core::D::Minus1)?;

        let cos_full = Tensor::cat(&[&cos, &cos], candle_core::D::Minus1)?;
        let sin_full = Tensor::cat(&[&sin, &sin], candle_core::D::Minus1)?;

        // Unsqueeze to match shape
        let cos_b = cos_full.unsqueeze(0)?.unsqueeze(2)?;
        let sin_b = sin_full.unsqueeze(0)?.unsqueeze(2)?;

        x.broadcast_mul(&cos_b)? + rotated.broadcast_mul(&sin_b)?
    }
}

/// Multi-Head Attention (MHA) module for Candle.
#[derive(Debug, Clone)]
pub struct CandleMHA {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
    pub num_heads: usize,
    pub d_k: usize,
    pub d_v: usize,
    pub scale: f64,
}

impl CandleMHA {
    pub fn new(
        d_model: usize,
        num_heads: usize,
        d_k: usize,
        d_v: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let q_proj = linear(d_model, num_heads * d_k, vb.pp("q_proj"))?;
        let k_proj = linear(d_model, num_heads * d_k, vb.pp("k_proj"))?;
        let v_proj = linear(d_model, num_heads * d_v, vb.pp("v_proj"))?;
        let out_proj = linear(num_heads * d_v, d_model, vb.pp("out_proj"))?;
        let scale = 1.0 / (d_k as f64).sqrt();

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            d_k,
            d_v,
            scale,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, _hidden) = x.dims3()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b_sz, seq_len, self.num_heads, self.d_k))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b_sz, seq_len, self.num_heads, self.d_k))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b_sz, seq_len, self.num_heads, self.d_v))?
            .transpose(1, 2)?
            .contiguous()?;

        // Scaled dot-product: (b, heads, seq, seq)
        let k_t = k.transpose(2, 3)?.contiguous()?;
        let att = (q.matmul(&k_t)? * self.scale)?;
        let att = candle_nn::ops::softmax(&att, candle_core::D::Minus1)?;
        let out = att.matmul(&v)?; // (b, heads, seq, d_v)

        let out = out.transpose(1, 2)?.contiguous()?.reshape((
            b_sz,
            seq_len,
            self.num_heads * self.d_v,
        ))?;
        self.out_proj.forward(&out)
    }
}

/// Grouped-Query Attention (GQA) module for Candle.
#[derive(Debug, Clone)]
pub struct CandleGQA {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub d_k: usize,
    pub scale: f64,
}

impl CandleGQA {
    pub fn new(
        d_model: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        d_k: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let q_proj = linear(d_model, num_q_heads * d_k, vb.pp("q_proj"))?;
        let k_proj = linear(d_model, num_kv_heads * d_k, vb.pp("k_proj"))?;
        let v_proj = linear(d_model, num_kv_heads * d_k, vb.pp("v_proj"))?;
        let out_proj = linear(num_q_heads * d_k, d_model, vb.pp("out_proj"))?;
        let scale = 1.0 / (d_k as f64).sqrt();

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_q_heads,
            num_kv_heads,
            d_k,
            scale,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, _hidden) = x.dims3()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b_sz, seq_len, self.num_q_heads, self.d_k))?
            .transpose(1, 2)?
            .contiguous()?;
        let mut k = k
            .reshape((b_sz, seq_len, self.num_kv_heads, self.d_k))?
            .transpose(1, 2)?
            .contiguous()?;
        let mut v = v
            .reshape((b_sz, seq_len, self.num_kv_heads, self.d_k))?
            .transpose(1, 2)?
            .contiguous()?;

        // Repeat KV heads to match Q heads if grouped
        let group_ratio = self.num_q_heads / self.num_kv_heads;
        if group_ratio > 1 {
            k = k
                .unsqueeze(2)?
                .repeat(&[1, 1, group_ratio, 1, 1])?
                .reshape((b_sz, self.num_q_heads, seq_len, self.d_k))?
                .contiguous()?;
            v = v
                .unsqueeze(2)?
                .repeat(&[1, 1, group_ratio, 1, 1])?
                .reshape((b_sz, self.num_q_heads, seq_len, self.d_k))?
                .contiguous()?;
        }

        let k_t = k.transpose(2, 3)?.contiguous()?;
        let att = (q.matmul(&k_t)? * self.scale)?;
        let att = candle_nn::ops::softmax(&att, candle_core::D::Minus1)?;
        let out = att.matmul(&v)?;

        let out = out.transpose(1, 2)?.contiguous()?.reshape((
            b_sz,
            seq_len,
            self.num_q_heads * self.d_k,
        ))?;
        self.out_proj.forward(&out)
    }
}

/// Multi-Head Latent Attention (MLA) module for Candle (DeepSeek V2/V3 compressed KV).
#[derive(Debug, Clone)]
pub struct CandleMLA {
    pub kv_down: Linear,
    pub kv_up: Linear,
    pub q_proj: Linear,
    pub out_proj: Linear,
    pub num_heads: usize,
    pub d_latent_kv: usize,
    pub d_v: usize,
    pub scale: f64,
}

impl CandleMLA {
    pub fn new(
        d_model: usize,
        num_heads: usize,
        d_latent_kv: usize,
        d_v: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let kv_down = linear_no_bias(d_model, d_latent_kv, vb.pp("kv_down"))?;
        let kv_up = linear_no_bias(d_latent_kv, num_heads * d_v * 2, vb.pp("kv_up"))?;
        let q_proj = linear(d_model, num_heads * d_v, vb.pp("q_proj"))?;
        let out_proj = linear(num_heads * d_v, d_model, vb.pp("out_proj"))?;
        let scale = 1.0 / (d_v as f64).sqrt();

        Ok(Self {
            kv_down,
            kv_up,
            q_proj,
            out_proj,
            num_heads,
            d_latent_kv,
            d_v,
            scale,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, _h) = x.dims3()?;
        let compressed_kv = self.kv_down.forward(x)?;
        let decompressed = self.kv_up.forward(&compressed_kv)?;

        let k = decompressed.narrow(candle_core::D::Minus1, 0, self.num_heads * self.d_v)?;
        let v = decompressed.narrow(
            candle_core::D::Minus1,
            self.num_heads * self.d_v,
            self.num_heads * self.d_v,
        )?;

        let q = self.q_proj.forward(x)?;
        let q = q
            .reshape((b_sz, seq_len, self.num_heads, self.d_v))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b_sz, seq_len, self.num_heads, self.d_v))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b_sz, seq_len, self.num_heads, self.d_v))?
            .transpose(1, 2)?
            .contiguous()?;

        let k_t = k.transpose(2, 3)?.contiguous()?;
        let att = (q.matmul(&k_t)? * self.scale)?;
        let att = candle_nn::ops::softmax(&att, candle_core::D::Minus1)?;
        let out = att.matmul(&v)?;

        let out = out.transpose(1, 2)?.contiguous()?.reshape((
            b_sz,
            seq_len,
            self.num_heads * self.d_v,
        ))?;
        self.out_proj.forward(&out)
    }
}

/// Differential Attention module for Candle (cancels attention noise via subtracted maps).
#[derive(Debug, Clone)]
pub struct CandleDiffAttn {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
    pub lambda_param: Tensor,
    pub num_heads: usize,
    pub d_k: usize,
    pub scale: f64,
}

impl CandleDiffAttn {
    pub fn new(d_model: usize, num_heads: usize, d_k: usize, vb: VarBuilder) -> Result<Self> {
        let q_proj = linear(d_model, num_heads * d_k * 2, vb.pp("q_proj"))?;
        let k_proj = linear(d_model, num_heads * d_k * 2, vb.pp("k_proj"))?;
        let v_proj = linear(d_model, num_heads * d_k, vb.pp("v_proj"))?;
        let out_proj = linear(num_heads * d_k, d_model, vb.pp("out_proj"))?;
        let lambda_param = vb
            .get((num_heads,), "lambda_init")
            .unwrap_or_else(|_| Tensor::full(0.8f32, (num_heads,), vb.device()).unwrap());
        let scale = 1.0 / (d_k as f64).sqrt();

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            lambda_param,
            num_heads,
            d_k,
            scale,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, _h) = x.dims3()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let half_feat = self.num_heads * self.d_k;
        let q1 = q.narrow(candle_core::D::Minus1, 0, half_feat)?;
        let q2 = q.narrow(candle_core::D::Minus1, half_feat, half_feat)?;
        let k1 = k.narrow(candle_core::D::Minus1, 0, half_feat)?;
        let k2 = k.narrow(candle_core::D::Minus1, half_feat, half_feat)?;

        let q1 = q1
            .reshape((b_sz, seq_len, self.num_heads, self.d_k))?
            .transpose(1, 2)?
            .contiguous()?;
        let q2 = q2
            .reshape((b_sz, seq_len, self.num_heads, self.d_k))?
            .transpose(1, 2)?
            .contiguous()?;
        let k1 = k1
            .reshape((b_sz, seq_len, self.num_heads, self.d_k))?
            .transpose(1, 2)?
            .contiguous()?;
        let k2 = k2
            .reshape((b_sz, seq_len, self.num_heads, self.d_k))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b_sz, seq_len, self.num_heads, self.d_k))?
            .transpose(1, 2)?
            .contiguous()?;

        let k1_t = k1.transpose(2, 3)?.contiguous()?;
        let k2_t = k2.transpose(2, 3)?.contiguous()?;
        let a1 =
            candle_nn::ops::softmax(&(q1.matmul(&k1_t)? * self.scale)?, candle_core::D::Minus1)?;
        let a2 =
            candle_nn::ops::softmax(&(q2.matmul(&k2_t)? * self.scale)?, candle_core::D::Minus1)?;

        let lambda = self.lambda_param.unsqueeze(0)?.unsqueeze(2)?.unsqueeze(3)?;
        let diff_att = (a1 - a2.broadcast_mul(&lambda)?)?;
        let out = diff_att.matmul(&v)?;

        let out = out.transpose(1, 2)?.contiguous()?.reshape((
            b_sz,
            seq_len,
            self.num_heads * self.d_k,
        ))?;
        self.out_proj.forward(&out)
    }
}
