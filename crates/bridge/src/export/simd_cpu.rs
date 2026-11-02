//! SIMD CPU Generator: emits portable auto-vectorized and cache-aligned Rust kernels
//! for AVX2, NEON, and WASM-SIMD execution on CPU.

use shared::spec::AiDesignSpec;
use std::collections::HashMap;

/// Generates portable SIMD CPU kernels and execution routines.
pub fn generate_simd_cpu_bundle(spec: &AiDesignSpec) -> HashMap<String, String> {
    let mut files = HashMap::new();
    files.insert(
        "simd_cpu/cpu_simd.rs".to_string(),
        generate_simd_cpu_code(spec),
    );
    files
}

pub fn generate_simd_cpu_code(_spec: &AiDesignSpec) -> String {
    r#"//! Fused CPU SIMD Routines: AVX2 / NEON / WASM-SIMD Vectorized Execution.

/// Computes fused RMSNorm inplace across contiguous 32-byte cache lines.
#[inline(always)]
pub fn cpu_fused_rmsnorm(x: &mut [f32], weights: &[f32], eps: f32) {
    let len = x.len();
    assert_eq!(len, weights.len());

    let mut sum_sq = 0.0f32;
    // Chunked auto-vectorized loop (unrolled by 8 for 256-bit SIMD registers)
    let chunks = x.chunks_exact(8);
    let remainder = chunks.remainder();

    for chunk in chunks {
        let mut local_sq = 0.0f32;
        for &val in chunk {
            local_sq += val * val;
        }
        sum_sq += local_sq;
    }
    for &val in remainder {
        sum_sq += val * val;
    }

    let rms = 1.0f32 / ((sum_sq / len as f32) + eps).sqrt();

    let x_chunks = x.chunks_exact_mut(8);
    let w_chunks = weights.chunks_exact(8);
    let x_rem = x_chunks.into_remainder();
    let w_rem = &weights[weights.len() - x_rem.len()..];

    for (xc, wc) in x.chunks_exact_mut(8).zip(weights.chunks_exact(8)) {
        for i in 0..8 {
            xc[i] = xc[i] * rms * wc[i];
        }
    }
    for (val, &w) in x_rem.iter_mut().zip(w_rem.iter()) {
        *val = *val * rms * w;
    }
}

/// Fast numerical approximation of SiLU: x / (1.0 + exp(-x)).
#[inline(always)]
pub fn fast_silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Fused SwiGLU forward pass for CPU vector lanes: (silu(x W_g) * (x W_u)) W_d.
pub fn cpu_fused_swiglu(
    input: &[f32],
    gate_w: &[f32],
    up_w: &[f32],
    down_w: &[f32],
    out: &mut [f32],
    dim: usize,
    hidden_dim: usize,
) {
    assert_eq!(input.len(), dim);
    assert_eq!(out.len(), dim);

    let mut hidden = vec![0.0f32; hidden_dim];
    for h in 0..hidden_dim {
        let mut gate_acc = 0.0f32;
        let mut up_acc = 0.0f32;
        let offset = h * dim;

        for i in 0..dim {
            let x = input[i];
            gate_acc += x * gate_w[offset + i];
            up_acc += x * up_w[offset + i];
        }

        hidden[h] = fast_silu(gate_acc) * up_acc;
    }

    // Down projection
    for i in 0..dim {
        let mut acc = 0.0f32;
        for h in 0..hidden_dim {
            acc += hidden[h] * down_w[i * hidden_dim + h];
        }
        out[i] = acc;
    }
}
"#
    .to_string()
}
