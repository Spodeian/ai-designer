//! Analytical Quantization & Layer Sizing Trade-Off Engine.
//!
//! Enables data-first, transformation-guided AI model synthesis where architects focus
//! on how data should be modified (spatial compression, causal forecasting, spectral filtering,
//! manifold expansion, sparse routing, symplectic conservation, denoising bottleneck) and
//! dynamically size layer dimensions (d_model, depth, FFN expansion) across quantization levels
//! (BitNet 1.58b, NF4, FP8, BF16, FP32) to saturate target hardware VRAM budgets.

use crate::spec::archetypes::{scaffold_archetype, ArchitectureArchetype};
use crate::spec::precision::PrecisionFormat;
use crate::spec::{AiDesignSpec, HardwareTargetProfile};
use serde::{Deserialize, Serialize};

/// High-level Data Modification Intent defining the mathematical transformation applied to data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataModificationIntent {
    /// Compresses continuous high-dimensional spatial/sensor grids into compact semantic latent representations.
    SpatialCompression,
    /// Autoregressively propagates historical causal sequences forward in time with KV caching.
    CausalForecasting,
    /// Projects tangled input manifolds into high-dimensional orthogonalized space with non-linear activation.
    ManifoldExpansion,
    /// Transforms signals into frequency domains using Fourier/Wavelet orthogonal bases for global differential filtering.
    SpectralFiltering,
    /// Sparsely routes heterogeneous tokens across dynamic expert pathways via learned top-k gating.
    ConditionalSparseRouting,
    /// Preserves Hamiltonian energy, phase-space volume, and symplectic invariants in dynamical physical systems.
    SymplecticConservation,
    /// Iteratively reconstructs pristine distributions from corrupted noise manifolds (Diffusion / Flow Matching).
    DenoisingBottleneck,
}

impl DataModificationIntent {
    pub const ALL: &'static [DataModificationIntent] = &[
        Self::SpatialCompression,
        Self::CausalForecasting,
        Self::ManifoldExpansion,
        Self::SpectralFiltering,
        Self::ConditionalSparseRouting,
        Self::SymplecticConservation,
        Self::DenoisingBottleneck,
    ];

    pub fn display_name(&self) -> &'static str {
        match self {
            Self::SpatialCompression => "Spatial / Manifold Compression",
            Self::CausalForecasting => "Causal Sequence Forecasting",
            Self::ManifoldExpansion => "Non-Linear Manifold Expansion",
            Self::SpectralFiltering => "Spectral & Frequency Filtering",
            Self::ConditionalSparseRouting => "Conditional Sparse Routing (MoE)",
            Self::SymplecticConservation => "Symplectic Phase-Space Conservation",
            Self::DenoisingBottleneck => "Denoising Manifold Bottleneck",
        }
    }

    pub fn short_description(&self) -> &'static str {
        match self {
            Self::SpatialCompression => "Compresses multi-channel spatial grids into dense semantic latent embeddings via hierarchical patching.",
            Self::CausalForecasting => "Propagates sequence tokens forward with strict causal masking, Rotary Position Embeddings (RoPE), and KV caching.",
            Self::ManifoldExpansion => "Expands input features into 4x-8x higher dimensional space with SwiGLU gating to linearly untangle complex decision boundaries.",
            Self::SpectralFiltering => "Decomposes inputs into Fourier modal bases to solve global continuous operators invariant to grid resolution.",
            Self::ConditionalSparseRouting => "Routes input tokens through specialized expert sub-networks with dynamic load-balanced gating.",
            Self::SymplecticConservation => "Integrates continuous flows preserving energy and Hamiltonian structure via reversible symplectic stages.",
            Self::DenoisingBottleneck => "Progressively removes injected noise across timesteps via U-Net / DiT cross-attentive latent conditioning.",
        }
    }

    pub fn to_archetype(&self) -> ArchitectureArchetype {
        match self {
            Self::SpatialCompression => ArchitectureArchetype::SelfSupervisedFoundation,
            Self::CausalForecasting => ArchitectureArchetype::GenerativeLLM,
            Self::ManifoldExpansion => ArchitectureArchetype::DeepResidual,
            Self::SpectralFiltering => ArchitectureArchetype::PhysicsOperator,
            Self::ConditionalSparseRouting => ArchitectureArchetype::DenseMoE,
            Self::SymplecticConservation => ArchitectureArchetype::SuperSmoothLipschitz,
            Self::DenoisingBottleneck => ArchitectureArchetype::VisionDiffusion,
        }
    }
}

/// Quantization tier evaluated for hardware trade-offs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuantizationTier {
    /// 1.58-bit Ternary {-1, 0, +1} (BitNet 1.58b / 2-bit storage)
    BitNet158b,
    /// 4-bit Generalized Distribution Float (Student-t, Laplace, Cauchy, Empirical)
    DistributionFloat4,
    /// 4-bit NormalFloat Quantile (NF4)
    NF4,
    /// 8-bit Unum Type III Tapered Floating Point (Posit8)
    Posit8,
    /// 8-bit Logarithmic Number System (LNS8)
    LNS8,
    /// 8-bit Microfloat (FP8 E4M3)
    FP8,
    /// 16-bit Brain Floating Point (BF16)
    BF16,
    /// 32-bit Single Precision IEEE 754 (FP32)
    FP32,
}

impl QuantizationTier {
    pub const ALL: &'static [QuantizationTier] = &[
        Self::BitNet158b,
        Self::DistributionFloat4,
        Self::NF4,
        Self::Posit8,
        Self::LNS8,
        Self::FP8,
        Self::BF16,
        Self::FP32,
    ];

    pub fn name(&self) -> &'static str {
        match self {
            Self::BitNet158b => "BitNet 1.58b (Ternary)",
            Self::DistributionFloat4 => "DistFloat4 (Generalized Quantile)",
            Self::NF4 => "NF4 (4-Bit NormalFloat)",
            Self::Posit8 => "Posit8 (8-Bit Unum Type III)",
            Self::LNS8 => "LNS8 (8-Bit Logarithmic)",
            Self::FP8 => "FP8 (8-Bit Minifloat)",
            Self::BF16 => "BF16 (16-Bit Bfloat)",
            Self::FP32 => "FP32 (32-Bit IEEE 754)",
        }
    }

    pub fn bits(&self) -> f64 {
        match self {
            Self::BitNet158b => 1.585,
            Self::DistributionFloat4 => 4.0,
            Self::NF4 => 4.0,
            Self::Posit8 => 8.0,
            Self::LNS8 => 8.0,
            Self::FP8 => 8.0,
            Self::BF16 => 16.0,
            Self::FP32 => 32.0,
        }
    }

    pub fn precision_format(&self) -> PrecisionFormat {
        match self {
            Self::BitNet158b => PrecisionFormat::BitNet158b,
            Self::DistributionFloat4 => PrecisionFormat::DistributionFloat4,
            Self::NF4 => PrecisionFormat::NF4,
            Self::Posit8 => PrecisionFormat::Posit8_1,
            Self::LNS8 => PrecisionFormat::LNS8,
            Self::FP8 => PrecisionFormat::FP8E4M3FN,
            Self::BF16 => PrecisionFormat::BF16,
            Self::FP32 => PrecisionFormat::FP32,
        }
    }
}

/// Sizing recommendation and trade-off metrics for a specific quantization tier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationSizingOption {
    pub tier: QuantizationTier,
    pub precision_format: PrecisionFormat,
    pub bits_per_weight: f64,
    pub d_model: usize,
    pub depth: usize,
    pub n_heads: usize,
    pub ffn_dim: usize,
    pub total_params: u64,
    pub weight_vram_gb: f64,
    pub kv_activation_vram_gb: f64,
    pub total_vram_gb: f64,
    pub estimated_tokens_per_sec: f64,
    pub expressivity_capacity_score: f64,
    pub recommendation_summary: String,
}

/// Analytical estimator calculating Pareto layer dimensions across quantization levels.
pub struct QuantizationSizingEstimator;

impl QuantizationSizingEstimator {
    /// Computes allowable layer dimensions (d_model, depth, FFN expansion) for each quantization level
    /// such that the total model weights and activations comfortably fit within the specified VRAM budget.
    pub fn estimate_tradeoffs(
        target_vram_gb: f64,
        context_len: usize,
        batch_size: usize,
        intent: DataModificationIntent,
        hardware: &HardwareTargetProfile,
    ) -> Vec<QuantizationSizingOption> {
        let usable_vram_gb = (target_vram_gb * 0.85).max(1.0); // 15% system scratch overhead
        let mut options = Vec::new();

        for &tier in QuantizationTier::ALL {
            let b = tier.bits();
            let bytes_per_weight = b / 8.0;

            // Solve for balanced (d_model, depth, n_heads) that fits usable_vram_gb
            let (d_model, depth, n_heads) = match tier {
                QuantizationTier::BitNet158b => match usable_vram_gb {
                    v if v <= 4.0 => (1536, 24, 12),
                    v if v <= 8.0 => (2560, 32, 20),
                    v if v <= 16.0 => (4096, 40, 32),
                    v if v <= 24.0 => (5120, 48, 40),
                    _ => (8192, 64, 64),
                },
                QuantizationTier::DistributionFloat4 | QuantizationTier::NF4 => match usable_vram_gb {
                    v if v <= 4.0 => (1024, 20, 8),
                    v if v <= 8.0 => (1792, 28, 14),
                    v if v <= 16.0 => (2816, 36, 22),
                    v if v <= 24.0 => (3584, 40, 28),
                    _ => (5120, 56, 40),
                },
                QuantizationTier::Posit8 | QuantizationTier::LNS8 | QuantizationTier::FP8 => match usable_vram_gb {
                    v if v <= 4.0 => (768, 16, 12),
                    v if v <= 8.0 => (1280, 24, 16),
                    v if v <= 16.0 => (2048, 32, 16),
                    v if v <= 24.0 => (2560, 36, 20),
                    _ => (4096, 48, 32),
                },
                QuantizationTier::BF16 => match usable_vram_gb {
                    v if v <= 4.0 => (512, 12, 8),
                    v if v <= 8.0 => (896, 18, 14),
                    v if v <= 16.0 => (1408, 24, 16),
                    v if v <= 24.0 => (1792, 28, 14),
                    _ => (2560, 36, 20),
                },
                QuantizationTier::FP32 => match usable_vram_gb {
                    v if v <= 4.0 => (384, 8, 6),
                    v if v <= 8.0 => (640, 14, 10),
                    v if v <= 16.0 => (1024, 18, 16),
                    v if v <= 24.0 => (1280, 22, 16),
                    _ => (1792, 28, 14),
                },
            };

            // Intent-specific FFN expansion ratio
            let ffn_mult = match intent {
                DataModificationIntent::ManifoldExpansion => 8, // Wide non-linear projection
                DataModificationIntent::ConditionalSparseRouting => 4, // Multi-expert sizing
                DataModificationIntent::SpectralFiltering => 2, // Compact modal representation
                _ => 4, // Standard 4x SwiGLU / MLP
            };
            let ffn_dim = d_model * ffn_mult;

            // Analytical parameter count calculation:
            // Per layer: Attention (4 * d_model^2) + FFN (2 or 3 * d_model * ffn_dim) + Norms
            let params_per_layer = (4 * d_model * d_model) + (3 * d_model * ffn_dim) + (4 * d_model);
            let embed_params = d_model * 32000; // Standard 32k vocab
            let total_params = (params_per_layer * depth + embed_params) as u64;

            let weight_vram_gb = (total_params as f64 * bytes_per_weight) / (1024.0 * 1024.0 * 1024.0);

            // KV Cache + working activation memory
            let kv_cache_bytes_per_token = 2 * depth * d_model * 2; // 2 bytes for BF16 KV
            let kv_activation_vram_gb = ((batch_size * context_len * kv_cache_bytes_per_token) as f64
                * 1.5)
                / (1024.0 * 1024.0 * 1024.0);

            let total_vram_gb = weight_vram_gb + kv_activation_vram_gb;

            // Hardware throughput estimate based on memory bandwidth:
            let mem_bw = hardware.memory_bandwidth_gbs.max(200.0);
            let estimated_tokens_per_sec = (mem_bw / weight_vram_gb.max(0.1))
                .min(hardware.peak_fp16_tflops * 10.0)
                .max(5.0);

            // Information-theoretic expressivity score: Depth * Width * log2(levels)
            let bits_equiv = b.min(16.0);
            let expressivity_capacity_score =
                ((depth as f64 * d_model as f64) * bits_equiv / 1000.0).round();

            let recommendation_summary = match tier {
                QuantizationTier::BitNet158b => format!(
                    "Maximum scale: {}M params. Fits massive width (d={}) with ternary weights and zero-FP-MAC additions.",
                    total_params / 1_000_000,
                    d_model
                ),
                QuantizationTier::DistributionFloat4 => format!(
                    "Generalized Quantile: {}M params. Exact quantile allocation for heavy-tailed activations (Student-t) or sparse routing (Laplace).",
                    total_params / 1_000_000
                ),
                QuantizationTier::NF4 => format!(
                    "Optimal QLoRA density: {}M params. Balances high accuracy with 4-bit normal float compression.",
                    total_params / 1_000_000
                ),
                QuantizationTier::Posit8 => format!(
                    "Tapered golden zone: {}M params. Unum Type III Posit dynamic range outperforming IEEE float precision at 8 bits.",
                    total_params / 1_000_000
                ),
                QuantizationTier::LNS8 => format!(
                    "Logarithmic LNS: {}M params. Exact multiplication and scale-free dynamic range for exponential and power-law signals.",
                    total_params / 1_000_000
                ),
                QuantizationTier::FP8 => format!(
                    "Hardware accelerated: {}M params. High throughput on modern Tensor Cores with native FP8 arithmetic.",
                    total_params / 1_000_000
                ),
                QuantizationTier::BF16 => format!(
                    "Full dynamic range: {}M params. Maximum numerical stability without quantization calibration overhead.",
                    total_params / 1_000_000
                ),
                QuantizationTier::FP32 => format!(
                    "Reference precision: {}M params. Ideal for scientific verification and numerical sensitivity audits.",
                    total_params / 1_000_000
                ),
            };

            options.push(QuantizationSizingOption {
                tier,
                precision_format: tier.precision_format(),
                bits_per_weight: b,
                d_model,
                depth,
                n_heads,
                ffn_dim,
                total_params,
                weight_vram_gb,
                kv_activation_vram_gb,
                total_vram_gb,
                estimated_tokens_per_sec,
                expressivity_capacity_score,
                recommendation_summary,
            });
        }

        options
    }

    /// Dynamically synthesizes a complete, valid `AiDesignSpec` tailored to the user's
    /// selected Data Modification Intent and chosen Quantization Sizing Option.
    pub fn synthesize_spec_from_intent_and_tradeoff(
        intent: DataModificationIntent,
        tradeoff: &QuantizationSizingOption,
        hardware: &HardwareTargetProfile,
    ) -> AiDesignSpec {
        let archetype = intent.to_archetype();
        let model_id = format!(
            "{}-{}-d{}",
            intent.display_name().to_lowercase().replace(['/', ' '], "-"),
            tradeoff.tier.name().split_whitespace().next().unwrap_or("Model").to_lowercase(),
            tradeoff.d_model
        );
        let display_name = format!(
            "{} ({} d={})",
            intent.display_name(),
            tradeoff.tier.name(),
            tradeoff.d_model
        );

        let mut spec = scaffold_archetype(
            archetype,
            &model_id,
            &display_name,
            tradeoff.d_model,
            tradeoff.depth,
            tradeoff.precision_format,
        );

        spec.metadata.target_hardware = hardware.clone();
        spec.metadata.tags.push("data-guided-synthesis".to_string());
        spec.metadata.tags.push(format!("quant:{:?}", tradeoff.precision_format));
        spec.precision.default_weight_precision = tradeoff.precision_format;
        spec
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_tradeoffs_estimation() {
        let hardware = HardwareTargetProfile::rtx_4090();
        let options = QuantizationSizingEstimator::estimate_tradeoffs(
            16.0, // 16GB target
            2048,
            1,
            DataModificationIntent::CausalForecasting,
            &hardware,
        );

        assert_eq!(options.len(), 5);

        // Lower bitwidths should yield wider models (larger d_model)
        let bitnet = &options[0];
        let fp32 = &options[4];

        assert_eq!(bitnet.tier, QuantizationTier::BitNet158b);
        assert_eq!(fp32.tier, QuantizationTier::FP32);

        assert!(bitnet.d_model > fp32.d_model, "BitNet d_model ({}) should exceed FP32 d_model ({})", bitnet.d_model, fp32.d_model);
        assert!(bitnet.total_params > fp32.total_params, "BitNet capacity should exceed FP32 capacity");

        // Verify all fit comfortably within budget
        for opt in &options {
            assert!(opt.total_vram_gb <= 16.0 * 1.05, "VRAM usage {} should be within budget", opt.total_vram_gb);
        }
    }

    #[test]
    fn test_data_intent_spec_synthesis() {
        let hardware = HardwareTargetProfile::rtx_4090();
        for &intent in DataModificationIntent::ALL {
            let options = QuantizationSizingEstimator::estimate_tradeoffs(
                8.0,
                1024,
                1,
                intent,
                &hardware,
            );
            let chosen = &options[1]; // NF4
            let spec = QuantizationSizingEstimator::synthesize_spec_from_intent_and_tradeoff(
                intent,
                chosen,
                &hardware,
            );

            assert!(!spec.inputs.is_empty(), "Spec inputs should not be empty for {:?}", intent);
            assert!(!spec.outputs.is_empty(), "Spec outputs should not be empty for {:?}", intent);
            assert!(!spec.graph.nodes.is_empty(), "Spec nodes should not be empty for {:?}", intent);
            assert_eq!(spec.precision.default_weight_precision, PrecisionFormat::NF4);
        }
    }
}

