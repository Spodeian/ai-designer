//! `ai-designer-bridge`: Unified Multi-Backend Exporter and Candle Bridge Library.
//!
//! Bridges the full expressive power of the AI Designer Studio to:
//! - **PyTorch 2**: Functional autograd, torch.compile Inductor fastpath, Dynamo tracing.
//! - **JAX / Equinox**: Pure-functional arrays, jax.jit, jax.vmap, and shard_map.
//! - **Candle**: Direct Rust Candle module synthesis and custom layer bridging for
//!   advanced operators (MHA, GQA, MLA, DiffAttn, RoPE, SwiGLU, MoE, BitNet 1.58b,
//!   RMSNorm, RevIN, Mamba S6, LISTA, Modern Hopfield).
//! - **wGPU**: WebGPU WGSL compute shaders, storage buffers, and pipelines.
//! - **SIMD CPU**: Portable auto-vectorized AVX2 / NEON / WASM-SIMD kernels.

pub mod candle_bridge;
pub mod export;

pub use candle_bridge::*;
pub use export::*;
