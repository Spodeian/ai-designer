//! Primitive neural, geometric, spectral, state-space, and unrolled optimization operators.

use serde::{Deserialize, Serialize};

/// Enumeration of all primitive operators available in the Designer catalog.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[allow(non_camel_case_types)]
pub enum OperatorKind {
    // 1. Attention & Associative Memories
    #[serde(rename = "mha")]
    MHA {
        num_heads: usize,
        d_k: usize,
        d_v: usize,
        #[serde(default)]
        dropout: f32,
    },
    #[serde(rename = "gqa")]
    GQA {
        num_q_heads: usize,
        num_kv_heads: usize,
        d_k: usize,
    },
    #[serde(rename = "mla")]
    MLA {
        num_heads: usize,
        d_latent_kv: usize,
        d_rope: usize,
        d_v: usize,
    },
    #[serde(rename = "diff_attn")]
    DiffAttn { num_heads: usize, d_k: usize },
    #[serde(rename = "linear_attention")]
    LinearAttention {
        num_heads: usize,
        d_k: usize,
        feature_map: String, // e.g. "elu+1" or "cos_sim"
    },
    #[serde(rename = "sliding_window_sink")]
    SlidingWindowSink {
        window_size: usize,
        num_sink_tokens: usize,
    },
    #[serde(rename = "evoformer_triangle")]
    EvoformerTriangle { d_pair: usize },
    #[serde(rename = "invariant_point_attention")]
    InvariantPointAttention { num_heads: usize, num_points: usize },

    // 2. State Space, Recurrent & Reservoir
    #[serde(rename = "mamba_s6")]
    MambaS6 {
        d_model: usize,
        d_state: usize,
        d_conv: usize,
        expand: usize,
    },
    #[serde(rename = "mamba2_ssd", alias = "mamba_2_s_s_d", alias = "mamba2")]
    Mamba2SSD {
        d_model: usize,
        d_state: usize,
        d_conv: usize,
        expand: usize,
    },
    #[serde(rename = "s4d", alias = "s4_d")]
    S4D { d_model: usize, d_state: usize },
    #[serde(rename = "lru")]
    LRU { d_model: usize, d_state: usize },
    #[serde(rename = "x_lstm", alias = "xlstm")]
    xLSTM {
        d_model: usize,
        num_heads: usize,
        variant: String, // "mLSTM" or "sLSTM"
    },
    #[serde(rename = "titans_memory")]
    TitansMemory { d_model: usize, memory_depth: usize },
    #[serde(rename = "echo_state_network")]
    EchoStateNetwork {
        reservoir_size: usize,
        spectral_radius: f64,
    },
    #[serde(rename = "next_gen_reservoir")]
    NextGenReservoir {
        delay_taps: usize,
        poly_degree: usize,
    },

    // 3. Optimization-as-Architecture (Deep Unfolding & Implicit)
    #[serde(rename = "lista")]
    LISTA {
        num_unrolled_steps: usize,
        lambda: f64,
    },
    #[serde(rename = "admm_net", alias = "a_d_m_m_net")]
    ADMMNet { num_stages: usize, rho: f64 },
    #[serde(rename = "crate")]
    CRATE {
        d_model: usize,
        num_heads: usize,
        lambda_step: f64,
    },
    #[serde(rename = "deq")]
    DEQ {
        d_model: usize,
        max_iters: usize,
        tolerance: f64,
    },
    #[serde(rename = "ttt")]
    TTT {
        d_model: usize,
        mini_batch_size: usize,
    },
    #[serde(rename = "modern_hopfield")]
    ModernHopfield { num_heads: usize, beta: f64 },

    // 4. Spatial, Spectral, Geometric & Physics
    #[serde(rename = "convnext_v2_block", alias = "conv_ne_xt_v2_block")]
    ConvNeXtV2Block {
        dim: usize,
        kernel_size: usize,
        mlp_ratio: f32,
    },
    #[serde(rename = "wt_conv", alias = "w_t_conv")]
    WTConv {
        in_channels: usize,
        out_channels: usize,
        wavelet_levels: usize,
    },
    #[serde(rename = "fourier_neural_operator", alias = "fno")]
    FourierNeuralOperator { modes: Vec<usize>, width: usize },
    #[serde(rename = "deeponet", alias = "deep_o_net")]
    DeepONet {
        branch_dim: usize,
        trunk_dim: usize,
        basis_dim: usize,
    },
    #[serde(rename = "egnn", alias = "e_g_n_n")]
    EGNN { node_dim: usize, edge_dim: usize },
    #[serde(rename = "cellular_sheaf_conv")]
    CellularSheafConv { node_dim: usize, stalk_dim: usize },
    #[serde(rename = "clifford_geometric_product")]
    CliffordGeometricProduct { p: usize, q: usize },
    #[serde(rename = "gaussian_splatting_3d", alias = "gaussian_splatting3_d")]
    GaussianSplatting3D { num_gaussians: usize },

    // 5. Splines, Feedforwards & Activations
    #[serde(rename = "linear")]
    Linear {
        in_features: usize,
        out_features: usize,
        #[serde(default = "default_true")]
        bias: bool,
    },
    #[serde(rename = "embedding")]
    Embedding {
        num_embeddings: usize,
        embedding_dim: usize,
    },
    #[serde(rename = "activation")]
    Activation {
        #[serde(default)]
        kind: ActivationKind,
    },
    #[serde(rename = "kan_layer", alias = "k_a_n_layer")]
    KANLayer {
        in_features: usize,
        out_features: usize,
        grid_size: usize,
        spline_order: usize,
    },
    #[serde(rename = "swiglu", alias = "swi_g_l_u")]
    SwiGLU {
        in_features: usize,
        hidden_features: usize,
    },
    #[serde(rename = "geglu", alias = "ge_g_l_u")]
    GeGLU {
        in_features: usize,
        hidden_features: usize,
    },
    #[serde(rename = "snake")]
    Snake { frequency: f32 },

    // 6. Normalization & Stability Guards
    #[serde(rename = "rms_norm", alias = "r_m_s_norm")]
    RMSNorm {
        dim: usize,
        #[serde(default = "default_norm_eps")]
        eps: f64,
    },
    #[serde(rename = "layer_norm")]
    LayerNorm {
        dim: usize,
        #[serde(default = "default_norm_eps")]
        eps: f64,
    },
    #[serde(rename = "qk_norm", alias = "q_k_norm")]
    QKNorm { d_head: usize },
    #[serde(rename = "logit_soft_cap")]
    LogitSoftCap { cap_val: f32 },
    #[serde(rename = "rev_in_layer", alias = "rev_i_n_layer")]
    RevINLayer { channels: usize },

    // 7. Latent Gating, Routing & Combinatorial
    #[serde(rename = "deepseek_moe", alias = "deep_seek_mo_e")]
    DeepSeekMoE {
        total_experts: usize,
        routed_k: usize,
        shared_experts: usize,
        d_model: usize,
        d_ffn: usize,
        #[serde(default = "default_true")]
        aux_loss_free_bias: bool,
    },
    #[serde(rename = "slot_attention")]
    SlotAttention {
        num_slots: usize,
        slot_dim: usize,
        iters: usize,
    },
    #[serde(rename = "hard_concrete_l0", alias = "hard_concrete_l_0")]
    HardConcreteL0 { dim: usize },
    #[serde(rename = "sparsemax_gate")]
    SparsemaxGate,
    #[serde(
        rename = "entmax15_gate",
        alias = "entmax_15_gate",
        alias = "entmax1_5_gate"
    )]
    Entmax15Gate,
    #[serde(rename = "soft_sort_permutahedron")]
    SoftSortPermutahedron { num_items: usize },
    #[serde(rename = "mixture_of_depths")]
    MixtureOfDepths { capacity_factor: f32 },

    // 8. Multimodal Diffusion, Multi-Token Prediction & Neuromorphic / Non-Euclidean
    #[serde(rename = "ada_ln_zero_dit", alias = "mmdit_block")]
    AdaLNZeroDiT {
        d_model: usize,
        num_heads: usize,
        cond_dim: usize,
        #[serde(default = "default_true")]
        dual_stream_mmdit: bool,
    },
    #[serde(rename = "multi_token_prediction", alias = "mtp_head")]
    MultiTokenPrediction {
        d_model: usize,
        vocab_size: usize,
        #[serde(default = "default_mtp_tokens")]
        num_predict_tokens: usize,
    },
    #[serde(rename = "spiking_lif")]
    SpikingLIF {
        d_model: usize,
        #[serde(default = "default_tau_mem")]
        tau_mem: f32,
        #[serde(default = "default_v_thresh")]
        v_threshold: f32,
        #[serde(default = "default_surrogate_slope")]
        surrogate_slope: f32,
    },
    #[serde(rename = "liquid_neural_cfc")]
    LiquidNeuralCfC {
        d_model: usize,
        backbone_units: usize,
    },
    #[serde(rename = "hyperbolic_poincare_layer")]
    HyperbolicPoincareLayer {
        dim: usize,
        #[serde(default = "default_curvature")]
        curvature_c: f64,
    },

    // 9. World Models, All-Atom Biomolecular, Late Interaction, Byte-Latent & Quantum/Photonic
    #[serde(rename = "jepa_predictor", alias = "jev_predictor")]
    JEPAPredictor {
        d_model: usize,
        predictor_dim: usize,
        num_heads: usize,
        #[serde(default = "default_ema_momentum")]
        ema_momentum: f64,
    },
    #[serde(rename = "pairformer_block", alias = "alphafold3_pairformer")]
    PairformerBlock {
        d_single: usize,
        d_pair: usize,
        num_heads: usize,
    },
    #[serde(rename = "world_model_rssm", alias = "dreamer_rssm")]
    WorldModelRSSM {
        d_model: usize,
        stochastic_classes: usize,
        stochastic_dim: usize,
    },
    #[serde(rename = "rwkv7_delta_rule", alias = "gated_delta_net")]
    RWKV7DeltaRule { d_model: usize, num_heads: usize },
    #[serde(rename = "entropy_byte_patcher", alias = "blt_patcher")]
    EntropyBytePatcher {
        d_byte: usize,
        d_latent: usize,
        #[serde(default = "default_entropy_thresh")]
        entropy_threshold: f32,
    },
    #[serde(rename = "colpali_max_sim", alias = "late_interaction_max_sim")]
    ColPaliMaxSim { d_model: usize, proj_dim: usize },
    #[serde(rename = "spherical_mesh_gnn", alias = "graphcast_icosahedral")]
    SphericalMeshGNN {
        d_node: usize,
        d_edge: usize,
        #[serde(default = "default_mesh_level")]
        mesh_refinement_level: usize,
    },
    #[serde(rename = "quantum_variational_circuit", alias = "pqc_layer")]
    QuantumVariationalCircuit {
        num_qubits: usize,
        circuit_depth: usize,
        #[serde(default = "default_entanglement")]
        entanglement: String,
    },
    #[serde(rename = "photonic_mzi_mesh", alias = "optical_clements_mesh")]
    PhotonicMZIMesh { num_modes: usize, adc_bits: usize },

    // 10. Topological Simplicial Complexes, Continuous Normalizing Flows, EBM/PCN, Neurosymbolic DNC, Multi-Agent MARL & Bio-Organoid Active Inference
    #[serde(rename = "simplicial_hodge_conv", alias = "hodge_laplacian_conv")]
    SimplicialHodgeConv {
        d_node: usize,
        d_edge: usize,
        d_face: usize,
    },
    #[serde(rename = "continuous_normalizing_flow", alias = "ffjord_cnf")]
    ContinuousNormalizingFlow {
        dim: usize,
        #[serde(default = "default_hutchinson_samples")]
        hutchinson_samples: usize,
    },
    #[serde(rename = "energy_langevin_mcmc", alias = "predictive_coding_ebm")]
    EnergyLangevinMCMC {
        dim: usize,
        #[serde(default = "default_langevin_steps")]
        langevin_steps: usize,
        #[serde(default = "default_langevin_step_size")]
        step_size: f32,
    },
    #[serde(rename = "neurosymbolic_logic_dnc", alias = "ltn_dnc_memory")]
    NeurosymbolicLogicDNC {
        d_model: usize,
        memory_slots: usize,
        #[serde(default = "default_t_norm")]
        t_norm: String,
    },
    #[serde(rename = "multi_agent_qmix", alias = "qmix_cfr_mixer")]
    MultiAgentQMIX {
        num_agents: usize,
        state_dim: usize,
        mixing_embed_dim: usize,
    },
    #[serde(
        rename = "bio_organoid_active_inference",
        alias = "skyrmion_organoid_fep"
    )]
    BioOrganoidActiveInference {
        num_channels: usize,
        #[serde(default = "default_free_energy_iters")]
        free_energy_iters: usize,
    },

    // 11. Attention-Free Gated MLP / Hyena, Raw Bit/Byte Multi-Scale LLMs, Zero-Shot Voice Cloning TTS & Self-Supervised GBDT
    #[serde(rename = "attention_free_gated_mlp", alias = "hyena_gmlp_mixer")]
    AttentionFreeGatedMLP {
        d_model: usize,
        seq_len: usize,
        #[serde(default = "default_hyena_order")]
        filter_order: usize,
    },
    #[serde(rename = "raw_bit_byte_multiscale", alias = "megabyte_mambabyte_bit")]
    RawBitByteMultiScale {
        d_global: usize,
        d_local: usize,
        #[serde(default = "default_patch_bytes")]
        patch_bytes: usize,
        #[serde(default = "default_true")]
        bit_plane_decomposition: bool,
    },
    #[serde(rename = "voice_cloning_codec_tts", alias = "rvq_flow_tts")]
    VoiceCloningCodecTTS {
        d_model: usize,
        #[serde(default = "default_rvq_codebooks")]
        num_rvq_codebooks: usize,
        #[serde(default = "default_speaker_dim")]
        speaker_embed_dim: usize,
    },
    #[serde(rename = "tabular_gbdt_vime_ssl", alias = "gbdt_vime_oblivious_tree")]
    TabularGBDTVimeSSL {
        num_trees: usize,
        tree_depth: usize,
        #[serde(default = "default_vime_mask_prob")]
        vime_corruption_rate: f32,
    },

    // 12. Super-Smooth C^infinity Lipschitz / Sobolev Networks & Learnable Quantization Level / Mixed-Bitwidth QAT
    #[serde(
        rename = "super_smooth_lipschitz_layer",
        alias = "bjorck_lipschitz_icnn"
    )]
    SuperSmoothLipschitzLayer {
        dim: usize,
        #[serde(default = "default_bjorck_iters")]
        bjorck_iters: usize,
        #[serde(default = "default_sobolev_order")]
        sobolev_order: usize,
        #[serde(default = "default_lipschitz_k")]
        spectral_bound_k: f64,
    },
    #[serde(rename = "learnable_quantizer_qat", alias = "lsq_fracbits_quantizer")]
    LearnableQuantizerQAT {
        dim: usize,
        #[serde(default = "default_init_bits")]
        init_bits: f32,
        #[serde(default = "default_min_bits")]
        min_bits: f32,
        #[serde(default = "default_max_bits")]
        max_bits: f32,
        #[serde(default = "default_true")]
        learn_step_size_and_clip: bool,
    },
}

fn default_bjorck_iters() -> usize {
    5
}
fn default_sobolev_order() -> usize {
    2
}
fn default_lipschitz_k() -> f64 {
    1.0
}
fn default_init_bits() -> f32 {
    4.0
}
fn default_min_bits() -> f32 {
    1.58
}
fn default_max_bits() -> f32 {
    8.0
}

fn default_hyena_order() -> usize {
    2
}
fn default_patch_bytes() -> usize {
    8
}
fn default_rvq_codebooks() -> usize {
    8
}
fn default_speaker_dim() -> usize {
    256
}
fn default_vime_mask_prob() -> f32 {
    0.3
}

fn default_hutchinson_samples() -> usize {
    4
}
fn default_langevin_steps() -> usize {
    8
}
fn default_langevin_step_size() -> f32 {
    0.05
}
fn default_t_norm() -> String {
    "lukasiewicz".to_string()
}
fn default_free_energy_iters() -> usize {
    6
}

fn default_true() -> bool {
    true
}
fn default_norm_eps() -> f64 {
    1e-5
}
fn default_mtp_tokens() -> usize {
    2
}
fn default_tau_mem() -> f32 {
    20.0
}
fn default_v_thresh() -> f32 {
    1.0
}
fn default_surrogate_slope() -> f32 {
    25.0
}
fn default_curvature() -> f64 {
    1.0
}
fn default_ema_momentum() -> f64 {
    0.996
}
fn default_entropy_thresh() -> f32 {
    1.8
}
fn default_mesh_level() -> usize {
    6
}
fn default_entanglement() -> String {
    "ring_cnot".to_string()
}

/// Comprehensive mathematical activation functions across modern deep learning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ActivationKind {
    #[default]
    SiLU,
    GELU,
    QuickGELU,
    ReLU,
    SquaredReLU,
    Mish,
    Laplace,
    LeakyReLU,
    ELU,
    Tanh,
    HardTanh,
    Sigmoid,
    HardSigmoid,
    Snake,
    SoftplusBeta,
    RationalPade,
    GroupSort2,
    WendlandC4RBF,
}

impl ActivationKind {
    pub const ALL: &'static [ActivationKind] = &[
        Self::SiLU,
        Self::GELU,
        Self::QuickGELU,
        Self::ReLU,
        Self::SquaredReLU,
        Self::Mish,
        Self::Laplace,
        Self::LeakyReLU,
        Self::ELU,
        Self::Tanh,
        Self::HardTanh,
        Self::Sigmoid,
        Self::HardSigmoid,
        Self::Snake,
        Self::SoftplusBeta,
        Self::RationalPade,
        Self::GroupSort2,
        Self::WendlandC4RBF,
    ];

    /// Returns true if the activation function is continuously differentiable (C^1 or higher)
    /// without non-differentiable kinks or step-discontinuities in its first derivative.
    pub fn is_smooth(&self) -> bool {
        !matches!(
            self,
            Self::ReLU | Self::LeakyReLU | Self::HardTanh | Self::HardSigmoid | Self::GroupSort2
        )
    }

    /// Returns true if the activation function is infinitely continuously differentiable (C^inf).
    pub fn is_c_infinity(&self) -> bool {
        matches!(
            self,
            Self::SiLU
                | Self::GELU
                | Self::Tanh
                | Self::Sigmoid
                | Self::Mish
                | Self::Snake
                | Self::SoftplusBeta
                | Self::RationalPade
        )
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::SiLU => "SiLU / Swish (x · σ(x))",
            Self::GELU => "GELU (Gaussian Error Linear Unit)",
            Self::QuickGELU => "QuickGELU (x · σ(1.702 · x))",
            Self::ReLU => "ReLU (max(0, x))",
            Self::SquaredReLU => "Squared ReLU (ReLU(x)²)",
            Self::Mish => "Mish (x · tanh(softplus(x)))",
            Self::Laplace => "Laplace Gate (0.5 · (1 + sgn(x) · (1 - exp(-|x|))))",
            Self::LeakyReLU => "Leaky ReLU (max(0.01x, x))",
            Self::ELU => "ELU (Exponential Linear Unit)",
            Self::Tanh => "Hyperbolic Tangent Tanh (-1..+1)",
            Self::HardTanh => "Hard Tanh (clamp(x, -1, 1))",
            Self::Sigmoid => "Sigmoid Logistic (0..1)",
            Self::HardSigmoid => "Hard Sigmoid (clamp(x/6 + 0.5, 0, 1))",
            Self::Snake => "Snake Periodic (x + sin²(x))",
            Self::SoftplusBeta => "Softplus-β C^∞ Smooth Convex (ln(1 + exp(βx))/β)",
            Self::RationalPade => "Rational Padé PAU C^∞ (P_m(x) / (1 + |Q_n(x)|))",
            Self::GroupSort2 => "GroupSort-2 (1-Lipschitz Norm-Preserving Sort)",
            Self::WendlandC4RBF => "Wendland C⁴ Compact Super-Smooth RBF",
        }
    }

    pub fn formula(&self) -> &'static str {
        match self {
            Self::SiLU => "f(x) = x · σ(x)",
            Self::GELU => "f(x) = x · Φ(x) ≈ 0.5x(1 + tanh(sqrt(2/π)(x + 0.044715x³)))",
            Self::QuickGELU => "f(x) = x · σ(1.702 · x)",
            Self::ReLU => "f(x) = max(0, x)",
            Self::SquaredReLU => "f(x) = max(0, x)²",
            Self::Mish => "f(x) = x · tanh(ln(1 + eˣ))",
            Self::Laplace => "f(x) = 0.5 · (1 + sgn(x) · (1 - exp(-|x|)))",
            Self::LeakyReLU => "f(x) = max(0.01x, x)",
            Self::ELU => "f(x) = x if x > 0 else exp(x) - 1",
            Self::Tanh => "f(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)",
            Self::HardTanh => "f(x) = clamp(x, -1.0, 1.0)",
            Self::Sigmoid => "f(x) = 1 / (1 + e⁻ˣ)",
            Self::HardSigmoid => "f(x) = clamp(x/6 + 0.5, 0.0, 1.0)",
            Self::Snake => "f(x) = x + sin²(x)",
            Self::SoftplusBeta => "f(x) = (1/β) · ln(1 + exp(β·x)), f''(x) > 0 ∀x",
            Self::RationalPade => "f(x) = (a₀ + a₁x + a₂x² + a₃x³) / (1 + |b₁x + b₂x²|)",
            Self::GroupSort2 => {
                "[y_{2i}, y_{2i+1}] = [min(x_{2i}, x_{2i+1}), max(x_{2i}, x_{2i+1})]"
            }
            Self::WendlandC4RBF => "φ(r) = (1 - r)₊⁶ · (35r² + 18r + 3)",
        }
    }

    pub fn typical_usage(&self) -> &'static str {
        match self {
            Self::SiLU => "Llama 3, Mistral, Flux.1, SD3, DeepSeek-V3/R1 gated MLPs",
            Self::GELU => "GPT-4, BERT, ViT, RoBERTa standard Transformer feedforwards",
            Self::QuickGELU => "CLIP, OpenCLIP, SigLIP vision-language encoders",
            Self::ReLU => "ResNet, AlexNet, standard feedforward baseline networks",
            Self::SquaredReLU => "PaLM, Primer, Nemotron sparse language models",
            Self::Mish => "YOLOv4, YOLOv7 real-time object detection backbones",
            Self::Laplace => "MEGA, TransNormer linear attention exponential gating",
            Self::LeakyReLU => "GAN discriminators, autoencoders preventing dying neurons",
            Self::ELU => "Diffusion model backbones, self-normalizing neural nets",
            Self::Tanh => "Recurrent cells (LSTM, GRU), bounded continuous control policies",
            Self::HardTanh => "Quantization-friendly edge MCUs and hardware inference accelerators",
            Self::Sigmoid => "Binary classification, attention gating, probability masks",
            Self::HardSigmoid => "MobileNetV3, EfficientNet Lite hardware friendly gating",
            Self::Snake => "Audio synthesis, implicit neural representations (SIREN/BigVGAN)",
            Self::SoftplusBeta => {
                "Super-smooth C^∞ PINNs, Brenier OT & Input-Convex Neural Networks (ICNNs)"
            }
            Self::RationalPade => {
                "Padé Activation Units (PAU), Kolmogorov rational approximants & smooth PDE solvers"
            }
            Self::GroupSort2 => {
                "Certified 1-Lipschitz robustness, Wasserstein duals & Björck orthonormal networks"
            }
            Self::WendlandC4RBF => {
                "C⁴ super-smooth meshless PDE collocation, SDF surface reconstruction & kriging"
            }
        }
    }
}

/// Architectural overview and composition guide for an operator node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperatorOverview {
    pub category: &'static str,
    pub complexity_tag: &'static str,
    pub what_it_does: &'static str,
    pub typical_usage: &'static str,
    pub typically_followed_by: &'static str,
}

impl OperatorKind {
    /// Functional category label used for catalog filtering.
    pub fn category_tag(&self) -> &'static str {
        self.overview().category
    }

    /// Computational complexity / paradigm filter tag.
    pub fn complexity_tag(&self) -> &'static str {
        self.overview().complexity_tag
    }

    /// Clickable URL to the corresponding reference topic in the AI-Examples repository.
    pub fn curriculum_url(&self) -> String {
        let slug = self.curriculum_slug();
        if slug.contains('/') {
            format!("https://github.com/Spodeian/AI-Examples/blob/main/markdown/{slug}.md")
        } else {
            format!("https://github.com/Spodeian/AI-Examples/tree/main/notebooks/{slug}")
        }
    }

    /// Architectural overview describing what the operator computes, when to use it, and what typically follows it.
    pub fn overview(&self) -> OperatorOverview {
        match self {
            Self::MHA { .. } => OperatorOverview {
                category: "Attention & Memory",
                complexity_tag: "Quadratic / Global Mixing",
                what_it_does: "Computes scaled dot-product attention across multiple query, key, and value heads in parallel, allowing every token or spatial patch to attend globally.",
                typical_usage: "Standard Transformer blocks, multimodal cross-attention fusion, and short-to-medium sequence encoders/decoders.",
                typically_followed_by: "Residual addition -> Pre-FFN RMSNorm -> SwiGLU Gated MLP or DeepSeek Sparse MoE.",
            },
            Self::GQA { .. } => OperatorOverview {
                category: "Attention & Memory",
                complexity_tag: "KV-Compressed Attention",
                what_it_does: "Shares key and value heads across groups of query heads, reducing KV-cache memory bandwidth by 4x-8x while preserving multi-head expressivity.",
                typical_usage: "High-throughput autoregressive LLMs (Llama-3, Mistral, Qwen) where decode latency is memory-bandwidth bound.",
                typically_followed_by: "Residual addition -> RMSNorm -> SwiGLU Gated MLP.",
            },
            Self::MLA { .. } => OperatorOverview {
                category: "Attention & Memory",
                complexity_tag: "KV-Compressed Attention",
                what_it_does: "Projects keys and values into a low-rank latent vector c_KV with decoupled Rotary Positional Embeddings (RoPE), enabling matrix absorption at inference.",
                typical_usage: "Long-context frontier reasoning models (DeepSeek-V2/V3) requiring minimal KV-cache VRAM footprint.",
                typically_followed_by: "Residual addition -> RMSNorm -> DeepSeek Sparse MoE or SwiGLU MLP.",
            },
            Self::DiffAttn { .. } => OperatorOverview {
                category: "Attention & Memory",
                complexity_tag: "Quadratic / Global Mixing",
                what_it_does: "Subtracts two softmax attention maps to cancel common-mode attention noise and amplify sparse relevant context.",
                typical_usage: "Long-context retrieval, needle-in-a-haystack reasoning, and hallucination mitigation.",
                typically_followed_by: "GroupNorm / RMSNorm per head -> Output Linear projection -> SwiGLU MLP.",
            },
            Self::LinearAttention { .. } => OperatorOverview {
                category: "Attention & Memory",
                complexity_tag: "Linear O(T) Sequence",
                what_it_does: "Replaces the exp(QK^T) kernel with associatively factorized feature maps phi(Q)(phi(K)^T V), reducing time and memory complexity from O(T^2) to O(T).",
                typical_usage: "Ultra-long sequences, streaming audio/video, and hybrid linear-full attention stacks.",
                typically_followed_by: "RMSNorm / Gate -> SwiGLU Gated MLP.",
            },
            Self::SlidingWindowSink { .. } => OperatorOverview {
                category: "Attention & Memory",
                complexity_tag: "Linear O(T) Sequence",
                what_it_does: "Restricts attention to a local sliding window while retaining permanent initial 'sink' tokens to absorb surplus softmax probability mass.",
                typical_usage: "Infinite-length streaming language and audio models (StreamingLLM) with constant KV cache.",
                typically_followed_by: "RMSNorm -> SwiGLU MLP or periodic global MLA layer.",
            },
            Self::EvoformerTriangle { .. } => OperatorOverview {
                category: "Spectral & Physics",
                complexity_tag: "Physics & Geometric",
                what_it_does: "Updates pairwise residue/atom representations z_ij via triangular multiplicative and axial attention updates enforcing 3D triangle inequality consistency.",
                typical_usage: "Biomolecular structure prediction (AlphaFold2/3) and 3D distance geometry networks.",
                typically_followed_by: "SE(3) Invariant Point Attention (IPA) -> E(3)-Equivariant GNN or FAPE output head.",
            },
            Self::InvariantPointAttention { .. } => OperatorOverview {
                category: "Spectral & Physics",
                complexity_tag: "Physics & Geometric",
                what_it_does: "Computes attention in local 3D rigid-body residue frames T_i = (R_i, t_i), guaranteeing exact invariance under global SE(3) rotations and translations.",
                typical_usage: "Protein backbone refinement, molecular docking, and SE(3) robotic manipulation policies.",
                typically_followed_by: "Backbone frame update layer -> Dense Linear projection with ContinuousRotation6D constraint.",
            },
            Self::Mamba2SSD { .. } | Self::MambaS6 { .. } => OperatorOverview {
                category: "State Space & Recurrent",
                complexity_tag: "Linear O(T) Sequence",
                what_it_does: "Evolves a selective latent state h_t = A_t h_{t-1} + B_t x_t using input-dependent step sizes and Structured State Space Duality (SSD) block GEMMs.",
                typical_usage: "Linear-time O(T) sequence modeling across language, raw audio waveforms, genomics, and long-horizon control.",
                typically_followed_by: "RMSNorm -> SwiGLU MLP, or interleaved 7:1 with Multi-Head Latent Attention (Jamba hybrid).",
            },
            Self::S4D { .. } | Self::LRU { .. } => OperatorOverview {
                category: "State Space & Recurrent",
                complexity_tag: "Linear O(T) Sequence",
                what_it_does: "Applies a diagonalized continuous-time linear state-space recurrence initialized via HiPPO/exponential eigenvalues for long-range memory.",
                typical_usage: "Continuous-time signal processing, physiological time series, and long-range arena tasks.",
                typically_followed_by: "Pointwise GLU / SwiGLU channel mixing -> LayerNorm / RMSNorm.",
            },
            Self::xLSTM { .. } => OperatorOverview {
                category: "State Space & Recurrent",
                complexity_tag: "Linear O(T) Sequence",
                what_it_does: "Combines exponential input/forget gating with a matrix-valued covariance memory cell (mLSTM) or scalar recurrent state mixing (sLSTM).",
                typical_usage: "Probabilistic time-series forecasting, financial econometrics, and fast O(1)-state sequence inference.",
                typically_followed_by: "RMSNorm -> SwiGLU or Linear projection head (often paired with RevIN).",
            },
            Self::TitansMemory { .. } => OperatorOverview {
                category: "State Space & Recurrent",
                complexity_tag: "Linear O(T) Sequence",
                what_it_does: "Updates a deep neural long-term memory module at test time using surprise-driven (gradient norm) momentum updates.",
                typical_usage: "Million-token context retention and continual test-time adaptation.",
                typically_followed_by: "Sliding-window or Multi-Head Attention branch -> SwiGLU MLP.",
            },
            Self::EchoStateNetwork { .. } | Self::NextGenReservoir { .. } => OperatorOverview {
                category: "State Space & Recurrent",
                complexity_tag: "Linear O(T) Sequence",
                what_it_does: "Projects time series into a fixed high-dimensional dynamical reservoir or nonlinear time-delay polynomial feature space.",
                typical_usage: "Ultra-low-latency chaotic dynamical system forecasting and edge DSP.",
                typically_followed_by: "Ridge-regressed Dense Linear readout layer.",
            },
            Self::LISTA { .. }
            | Self::ADMMNet { .. }
            | Self::CRATE { .. }
            | Self::DEQ { .. }
            | Self::TTT { .. }
            | Self::ModernHopfield { .. } => OperatorOverview {
                category: "Feedforward & Splines",
                complexity_tag: "Optimization-as-Architecture",
                what_it_does: "Embeds an unrolled proximal optimization solver (ISTA/ADMM/Rate-Reduction) or implicit fixed-point equilibrium directly as a differentiable neural layer.",
                typical_usage: "Compressed sensing, inverse problems, white-box interpretable transformers, and adaptive test-time compute.",
                typically_followed_by: "RMSNorm -> Linear readout or OptNet / CBF-QP constrained output head.",
            },
            Self::ConvNeXtV2Block { .. } => OperatorOverview {
                category: "Spectral & Physics",
                complexity_tag: "Physics & Geometric",
                what_it_does: "Applies 7x7 depthwise spatial convolution followed by inverted bottleneck pointwise MLP with Global Response Normalization (GRN).",
                typical_usage: "High-efficiency 2D/3D vision backbones, dense segmentation, and hybrid vision-robotics encoders.",
                typically_followed_by: "Downsampling / Patchify -> Slot Attention or Multi-Head Attention.",
            },
            Self::WTConv { .. } => OperatorOverview {
                category: "Spectral & Physics",
                complexity_tag: "Physics & Geometric",
                what_it_does: "Performs cascaded discrete wavelet decomposition, applying small kernels across multi-frequency wavelet subbands to achieve an O(2^L) receptive field.",
                typical_usage: "Multi-scale vision, medical imaging, seismic/audio DSP, and DiT patch stems.",
                typically_followed_by: "RMSNorm -> Multi-Head Attention or ConvNeXtV2Block.",
            },
            Self::FourierNeuralOperator { .. } | Self::DeepONet { .. } => OperatorOverview {
                category: "Spectral & Physics",
                complexity_tag: "Physics & Geometric",
                what_it_does: "Learns resolution-invariant operators in function space by multiplying truncated Fourier modes in spectral domain and summing with a local residual bypass.",
                typical_usage: "Zero-shot super-resolution PDE surrogates (Navier-Stokes, weather, MHD plasma, elasticity).",
                typically_followed_by: "Additional FNO spectral blocks -> Linear projection head with CurlFreeVectorField or Sobolev H1 loss.",
            },
            Self::EGNN { .. }
            | Self::CellularSheafConv { .. }
            | Self::CliffordGeometricProduct { .. }
            | Self::GaussianSplatting3D { .. } => OperatorOverview {
                category: "Spectral & Physics",
                complexity_tag: "Physics & Geometric",
                what_it_does: "Propagates messages across graph edges, cellular sheaves, or 3D coordinates while preserving E(3)/rotational equivariance.",
                typical_usage: "Molecular conformation generation, 3D point clouds, physical N-body simulations, and heterophilic graphs.",
                typically_followed_by: "Equivariant coordinate readout or invariant pooling -> Linear output head.",
            },
            Self::Linear { .. } => OperatorOverview {
                category: "Feedforward & Splines",
                complexity_tag: "Dense Projection",
                what_it_does: "Applies an affine matrix transformation y = x W^T + b across the feature dimension (optimized via Muon for internal 2D weights or AdamW for terminal heads).",
                typical_usage: "Input stem projection, attention output projection, classification/regression readout heads, and MLP layers.",
                typically_followed_by: "RMSNorm / Activation (if stem/hidden) or Output Port Loss / Structural Constraint (if terminal head).",
            },
            Self::Embedding { .. } => OperatorOverview {
                category: "Feedforward & Splines",
                complexity_tag: "Dense Projection",
                what_it_does: "Maps discrete token indices [B, T] into continuous dense vectors [B, T, D] via a learnable lookup table (partitioned to AdamW).",
                typical_usage: "Input stem for discrete text, code, genomic sequences, or quantized audio/visual tokens.",
                typically_followed_by: "RMSNorm -> Multi-Head Attention (MHA/GQA/MLA) or Mamba-2 SSD backbone.",
            },
            Self::KANLayer { .. } => OperatorOverview {
                category: "Feedforward & Splines",
                complexity_tag: "Dense Projection",
                what_it_does: "Replaces fixed node activations with learnable univariate spline functions (B-splines / Chebyshev / RBF) on every edge.",
                typical_usage: "Symbolic law discovery, scientific regression, and compact high-accuracy tabular/physics models.",
                typically_followed_by: "LayerNorm / RMSNorm -> next KANLayer or Linear readout head.",
            },
            Self::SwiGLU { .. } | Self::GeGLU { .. } => OperatorOverview {
                category: "Feedforward & Splines",
                complexity_tag: "Dense Projection",
                what_it_does: "Computes a multiplicative bilinear gated feedforward block: W_down(SiLU(W_gate x) * (W_up x)), expanding hidden dimension by ~8/3x.",
                typical_usage: "Primary channel-mixing FFN block following Attention or SSM layers in modern Transformers and Mamba hybrids.",
                typically_followed_by: "Residual stream merge -> next block's RMSNorm or final output Linear projection.",
            },
            Self::Snake { .. } => OperatorOverview {
                category: "Norms & Activations",
                complexity_tag: "Shape-Preserving",
                what_it_does: "Applies the periodic activation x + (1/a) sin^2(a x), providing strong inductive bias for periodic and harmonic signals.",
                typical_usage: "Neural audio codecs (Descript/SoundStream), waveform synthesis, and periodic time series.",
                typically_followed_by: "Causal Mamba-2 SSD or 1D Conv / Linear projection.",
            },
            Self::Activation { kind: _ } => OperatorOverview {
                category: "Norms & Activations",
                complexity_tag: "Shape-Preserving",
                what_it_does: "Applies an elementwise mathematical activation function (SiLU, GELU, QuickGELU, ReLU, SquaredReLU, Mish, Laplace, LeakyReLU, ELU, Tanh, Sigmoid, Snake) to introduce representational nonlinearity.",
                typical_usage: "Stem projections, customized MLP stacks, convolutional feature extractors, and modular modality conformers.",
                typically_followed_by: "Linear projection, RMSNorm / LayerNorm, or downstream Attention/SSM block.",
            },
            Self::RMSNorm { .. } | Self::LayerNorm { .. } | Self::QKNorm { .. } => {
                OperatorOverview {
                    category: "Norms & Activations",
                    complexity_tag: "Shape-Preserving",
                    what_it_does: "Normalizes activation root-mean-square or variance (computed in FP32 reduction precision) to prevent gradient explosion and attention logit growth.",
                    typical_usage: "Pre-Norm guard before every Attention, SSM, or FFN sub-block, and final normalization before the output head.",
                    typically_followed_by: "Multi-Head Attention (MHA/GQA/MLA), Mamba-2 SSD, SwiGLU MLP, or final Linear readout.",
                }
            }
            Self::LogitSoftCap { .. } => OperatorOverview {
                category: "Norms & Activations",
                complexity_tag: "Shape-Preserving",
                what_it_does: "Smoothly bounds logits via C * tanh(x / C) to prevent attention entropy collapse and FP8/BF16 overflow.",
                typical_usage: "Placed immediately before final output CrossEntropy loss or inside attention score computation (Gemma-2 style).",
                typically_followed_by: "Output Port (CrossEntropy / GRPO) or Softmax/Entmax gate.",
            },
            Self::RevINLayer { .. } => OperatorOverview {
                category: "Norms & Activations",
                complexity_tag: "Shape-Preserving",
                what_it_does: "Strips non-stationary instance mean and variance per time-series channel at input and restores exact scale/offset at output.",
                typical_usage: "Non-stationary financial, econometric, and sensor time-series forecasting.",
                typically_followed_by: "xLSTM / Mamba-2 SSD backbone -> symmetric RevIN denormalization.",
            },
            Self::DeepSeekMoE { .. } => OperatorOverview {
                category: "Routing & MoE",
                complexity_tag: "Sparse / Conditional",
                what_it_does: "Routes each token to top-k fine-grained routed SwiGLU experts while always executing shared experts to isolate common knowledge.",
                typical_usage: "High-capacity frontier models where total parameters scale without increasing per-token FLOPs.",
                typically_followed_by: "Residual addition -> next block's RMSNorm + MLA/GQA or final output head.",
            },
            Self::SlotAttention { .. } => OperatorOverview {
                category: "Routing & MoE",
                complexity_tag: "Sparse / Conditional",
                what_it_does: "Iteratively binds spatial/temporal features into K object-centric latent slots via competitive softmax normalization over the query/slot axis.",
                typical_usage: "Object-centric scene decomposition, multimodal fusion, and robotic perception.",
                typically_followed_by: "Cross-Attention policy decoder or per-slot MLP readout.",
            },
            Self::HardConcreteL0 { .. }
            | Self::SparsemaxGate
            | Self::Entmax15Gate
            | Self::SoftSortPermutahedron { .. }
            | Self::MixtureOfDepths { .. } => OperatorOverview {
                category: "Routing & MoE",
                complexity_tag: "Sparse / Conditional",
                what_it_does: "Produces exact-zero sparse probabilities or top-k token routing masks with differentiable gradients, enabling true conditional compute skipping.",
                typical_usage: "Feature selection in tabular ML, dynamic layer skipping (MoD), and sparse attention/routing gates.",
                typically_followed_by: "Gated backbone block (MHA / SwiGLU) or Sparse Fenchel-Young output head.",
            },
            Self::AdaLNZeroDiT { .. } => OperatorOverview {
                category: "Attention & Memory",
                complexity_tag: "Quadratic / Global Mixing",
                what_it_does: "Modulates layer normalization scale/shift/gate (gamma, beta, alpha) from timestep & text conditioning with zero-initialized residual gates, executing joint multimodal text-image attention (MMDiT).",
                typical_usage: "Frontier Rectified Flow & Diffusion Transformers for image/video generation (Flux.1, Stable Diffusion 3, Sora).",
                typically_followed_by: "Next Dual-Stream or Single-Stream AdaLNZeroDiT block -> SwiGLU MLP -> Velocity unpatchify head.",
            },
            Self::MultiTokenPrediction { .. } => OperatorOverview {
                category: "Routing & MoE",
                complexity_tag: "Dense Projection",
                what_it_does: "Predicts D_k sequential future tokens at each position using causal cross-entropy chains and shared embedding projections (DeepSeek-V3 MTP).",
                typical_usage: "Frontier LLMs (DeepSeek-V3/R1) to densify training supervision signals and enable 2x-3x faster speculative self-drafting at inference.",
                typically_followed_by: "Final RMSNorm -> Primary vocabulary readout head.",
            },
            Self::SpikingLIF { .. } => OperatorOverview {
                category: "State Space & Recurrent",
                complexity_tag: "Sparse / Conditional",
                what_it_does: "Integrates synaptic currents into leaky membrane potential V_t = beta V_{t-1} + I_t, emitting sparse binary spikes S_t in {0, 1} when V_t > V_th with fast-sigmoid surrogate gradients.",
                typical_usage: "Ultra-low-power neuromorphic processors (Loihi, Cortex-M), event-based DVS cameras, and sparse temporal pattern recognition.",
                typically_followed_by: "Next SpikingLIF recurrent layer or membrane potential Linear readout head.",
            },
            Self::LiquidNeuralCfC { .. } => OperatorOverview {
                category: "State Space & Recurrent",
                complexity_tag: "Linear O(T) Sequence",
                what_it_does: "Solves closed-form continuous-time (CfC) liquid differential equations dx/dt = -[1/tau + f(x, I)] x + A f(x, I) with input-dependent time constants.",
                typical_usage: "Autonomous drone flight control, irregular medical telemetry, and continuous-time physical robotics under distribution shift.",
                typically_followed_by: "ContinuousRotation6D / CBF-QP safe control readout head.",
            },
            Self::HyperbolicPoincareLayer { .. } => OperatorOverview {
                category: "Spectral & Physics",
                complexity_tag: "Physics & Geometric",
                what_it_does: "Performs Möbius matrix-vector multiplication and gyrovector bias addition inside the negatively curved Poincaré ball manifold B^d_c via logarithmic and exponential maps.",
                typical_usage: "Tree-structured taxonomies, biological phylogenetics, hierarchical knowledge graphs, and scale-free networks with zero distortion.",
                typically_followed_by: "Next HyperbolicPoincareLayer -> Logarithmic map at origin -> Linear or GFlowNet readout head.",
            },
            Self::JEPAPredictor { .. } => OperatorOverview {
                category: "Attention & Memory",
                complexity_tag: "Quadratic / Global Mixing",
                what_it_does: "Predicts target latent embeddings of masked spatiotemporal regions conditioned on visible context tokens and positional mask tokens, aligned against an EMA target encoder in representation space.",
                typical_usage: "Self-supervised visual & video world models (I-JEPA, V-JEPA / JEV) that learn semantic physical dynamics without generative pixel reconstruction.",
                typically_followed_by: "RMSNorm -> VICReg / Smooth-L1 latent representation predictor head.",
            },
            Self::PairformerBlock { .. } => OperatorOverview {
                category: "Spectral & Physics",
                complexity_tag: "Physics & Geometric",
                what_it_does: "AlphaFold 3 Pairformer trunk block that jointly updates single (s_i) and pair (z_{ij}) biomolecular representations via triangle multiplicative updates, triangle self-attention, and pair-biased single attention.",
                typical_usage: "All-atom biomolecular structure prediction (AlphaFold 3) across proteins, RNA/DNA, covalent modifications, and small-molecule ligands.",
                typically_followed_by: "Next PairformerBlock -> AdaLNZeroDiT all-atom diffusion module -> 3D atomic coordinate head.",
            },
            Self::WorldModelRSSM { .. } => OperatorOverview {
                category: "State Space & Recurrent",
                complexity_tag: "Linear O(T) Sequence",
                what_it_does: "Recurrent State Space Model (DreamerV3 / MuZero RSSM) combining deterministic recurrent memory h_t with unimix discrete categorical stochastic states z_t (32x32 straight-through gradients) for latent imagination rollouts.",
                typical_usage: "Model-based reinforcement learning, sample-efficient robotics, and planning in latent imagination space without environment interaction.",
                typically_followed_by: "SwiGLU imagination transition predictor -> Symexp actor-critic & reward readout heads.",
            },
            Self::RWKV7DeltaRule { .. } => OperatorOverview {
                category: "State Space & Recurrent",
                complexity_tag: "Linear O(T) Sequence",
                what_it_does: "Generalized Delta-Rule Linear RNN (RWKV-7 'Goose' / Gated DeltaNet) evolving matrix state S_t = S_{t-1}(diag(w_t) - k_t (a_t * k_t)^T) + v_t k_t^T with vector-valued in-context learning rates.",
                typical_usage: "Constant-memory O(1) inference LLMs capable of full state-tracking and associative recall beyond standard Mamba/linear attention.",
                typically_followed_by: "RMSNorm -> Channel-Mixing SwiGLU / R-Gated MLP.",
            },
            Self::EntropyBytePatcher { .. } => OperatorOverview {
                category: "Routing & MoE",
                complexity_tag: "Sparse / Conditional",
                what_it_does: "Byte Latent Transformer (BLT) dynamic entropy patcher that computes next-byte Shannon entropy H(x_t) and segments raw UTF-8 bytes into variable-length latent patches via cross-attention pooling.",
                typical_usage: "Tokenizer-free byte-level foundation models that allocate compute dynamically to high-surprisal unpredictable byte boundaries.",
                typically_followed_by: "Latent Transformer / RWKV7DeltaRule backbone -> Cross-attention byte unpatcher.",
            },
            Self::ColPaliMaxSim { .. } => OperatorOverview {
                category: "Attention & Memory",
                complexity_tag: "Quadratic / Global Mixing",
                what_it_does: "Projects visual document patch tokens and text query tokens into a low-dimensional L2-normalized multi-vector space and computes Late-Interaction MaxSim scores sum_i max_j <q_i, d_j>.",
                typical_usage: "OCR-free visual document retrieval (ColPali / ColQwen2) over complex PDF layouts, tables, figures, and equations.",
                typically_followed_by: "InfoNCE / SigLIP cross-modal contrastive ranking head.",
            },
            Self::SphericalMeshGNN { .. } => OperatorOverview {
                category: "Spectral & Physics",
                complexity_tag: "Physics & Geometric",
                what_it_does: "Maps latitude-longitude atmospheric fields onto a multi-resolution icosahedral spherical geodesic mesh (Grid2Mesh -> Multi-Mesh Message Passing -> Mesh2Grid) free of polar singularities.",
                typical_usage: "Global medium-range weather and climate foundation models (GraphCast, NeuralGCM) across 37 vertical atmospheric pressure levels.",
                typically_followed_by: "FourierNeuralOperator spectral filter or physical residual projection head.",
            },
            Self::QuantumVariationalCircuit { .. } => OperatorOverview {
                category: "Spectral & Physics",
                complexity_tag: "Physics & Geometric",
                what_it_does: "Simulates a Parameterized Quantum Circuit (PQC / VQE) with data re-uploading angle encoding, trainable SU(2) Pauli rotations (Rx, Ry, Rz), CNOT entanglement, and Pauli-Z expectation readout.",
                typical_usage: "Hybrid quantum-classical neural networks (QNN), molecular ground-state energy estimation (VQE), and quantum kernel classifiers.",
                typically_followed_by: "Classical Linear post-processing head or Hamiltonian energy readout.",
            },
            Self::PhotonicMZIMesh { .. } => OperatorOverview {
                category: "Feedforward & Splines",
                complexity_tag: "Dense Projection",
                what_it_does: "Models a coherent optical Clements mesh of Mach-Zehnder Interferometers (MZIs) parameterizing unitary matrices U Sigma V^H via thermo-optic phase shifters alongside finite-bit ADC/DAC quantization.",
                typical_usage: "Speed-of-light photonic neural accelerators and non-volatile analog memristive crossbar inference engines.",
                typically_followed_by: "Electro-optic photodetector nonlinearity (Snake / SiLU) or next PhotonicMZIMesh stage.",
            },
            Self::SimplicialHodgeConv { .. } => OperatorOverview {
                category: "Spectral & Physics",
                complexity_tag: "Physics & Geometric",
                what_it_does: "Propagates signals across 0-simplices (nodes), 1-simplices (edges), and 2-simplices (triangular faces) using Hodge-k Laplacians Delta_k = B_k^T B_k + B_{k+1} B_{k+1}^T and Helmholtz-Hodge curl/harmonic decompositions.",
                typical_usage: "Topological deep learning on 3D CAD meshes, cortical brain connectomes, vascular flow networks, and cohomology-aware molecular rings.",
                typically_followed_by: "Next SimplicialHodgeConv block or CellularSheafConv stalk diffusion -> Harmonic readout.",
            },
            Self::ContinuousNormalizingFlow { .. } => OperatorOverview {
                category: "Spectral & Physics",
                complexity_tag: "Physics & Geometric",
                what_it_does: "Integrates an invertible continuous-time vector field dz/dt = f_theta(z, t) while tracking exact log-density changes d(log p)/dt = -Tr(df/dz) via the unbiased Hutchinson stochastic trace estimator.",
                typical_usage: "Exact likelihood estimation (FFJORD), Boltzmann generators for statistical mechanics, and high-dimensional financial tail-risk vine copulas.",
                typically_followed_by: "Next ContinuousNormalizingFlow block -> Exact log-determinant Jacobian density & QuantileCRPS head.",
            },
            Self::EnergyLangevinMCMC { .. } => OperatorOverview {
                category: "State Space & Recurrent",
                complexity_tag: "Implicit / Unrolled Solver",
                what_it_does: "Minimizes a scalar energy / variational free-energy landscape E_theta(x, z) via K steps of noisy Langevin MCMC dynamics and local Hebbian Predictive Coding (PCN) error units without global backpropagation.",
                typical_usage: "Biologically plausible predictive coding networks, continuous Hopfield associative memories, and constraint-satisfaction energy models.",
                typically_followed_by: "Equilibrium free-energy readout or Contrastive Divergence / Score Matching output head.",
            },
            Self::NeurosymbolicLogicDNC { .. } => OperatorOverview {
                category: "Attention & Memory",
                complexity_tag: "Quadratic / Global Mixing",
                what_it_does: "Couples a Differentiable Neural Computer (DNC) external addressable memory matrix M in R^{N x W} (content + temporal linkage addressing) with fuzzy First-Order Logic Tensor Network (LTN) t-norm grounding.",
                typical_usage: "Algorithmic program synthesis, multi-hop relational theorem proving, and verifiable neurosymbolic knowledge base reasoning.",
                typically_followed_by: "RMSNorm -> SwiGLU rule deduction block -> Certified logical satisfaction readout.",
            },
            Self::MultiAgentQMIX { .. } => OperatorOverview {
                category: "Routing & MoE",
                complexity_tag: "Sparse / Conditional",
                what_it_does: "Mixes decentralized per-agent utilities Q_a(tau_a, u_a) into a factored joint action-value Q_tot via a hypernetwork with strictly non-negative weights (dQ_tot/dQ_a >= 0, IGM principle) plus CFR regret matching.",
                typical_usage: "Cooperative multi-agent reinforcement learning (CTDE), autonomous drone swarms, and imperfect-information Nash equilibrium game solvers.",
                typically_followed_by: "Joint action-value advantage head or Counterfactual Regret policy head.",
            },
            Self::BioOrganoidActiveInference { .. } => OperatorOverview {
                category: "State Space & Recurrent",
                complexity_tag: "Physics & Geometric",
                what_it_does: "Models closed-loop Variational Free Energy F = D_KL[q(s)||p(s|o)] minimization across High-Density Microelectrode Array (HD-MEA) biological organoid channels and topological spintronic skyrmion reservoirs.",
                typical_usage: "Wetware DishBrain biocomputing interfaces, spintronic skyrmion reservoir substrates, and active inference sensorimotor control.",
                typically_followed_by: "SpikingLIF readout or Safe Control Barrier Function (CBF-QP) electrophysiological actuator head.",
            },
            Self::AttentionFreeGatedMLP { .. } => OperatorOverview {
                category: "Feedforward & Splines",
                complexity_tag: "Linear O(T) Sequence",
                what_it_does: "Replaces quadratic self-attention with a Spatial Gating Unit s(Z) = Z_1 * (W_spatial Z_2 + b) (gMLP / MLP-Mixer) and subquadratic FFT implicit long convolutions (Hyena), preventing the doubly-exponential rank-1 collapse proven by Dong et al. ('Attention is Not All You Need').",
                typical_usage: "Attention-free sequence/vision architectures (gMLP, MLP-Mixer, Hyena Hierarchy) requiring full-rank token mixing without O(T^2) attention matrices.",
                typically_followed_by: "Residual addition -> RMSNorm -> SwiGLU channel-mixing MLP or next AttentionFreeGatedMLP block.",
            },
            Self::RawBitByteMultiScale { .. } => OperatorOverview {
                category: "Attention & Memory",
                complexity_tag: "Linear O(T) Sequence",
                what_it_does: "Tokenizer-free hierarchical bit/byte processor (MegaByte / MambaByte / bGPT) that decomposes raw bytes into 8 binary bit-planes, models global context across P-byte patches in O((T/P)^2), and autoregressively decodes fine local bytes/bits.",
                typical_usage: "Raw binary file modeling, tokenizer-free multilingual/code LLMs, raw network packet analysis, and arbitrary digital bitstream generation.",
                typically_followed_by: "EntropyBytePatcher, Mamba-2 SSD byte mixer, or 256-way raw byte / 8-bit binary readout head.",
            },
            Self::VoiceCloningCodecTTS { .. } => OperatorOverview {
                category: "Spectral & Physics",
                complexity_tag: "Physics & Geometric",
                what_it_does: "Zero-shot voice cloning & neural audio codec synthesizer (F5-TTS / VALL-E 2 / CosyVoice) combining reference speaker timbre prompt extraction, Multi-Stream Residual Vector Quantization (RVQ), and Optimal-Transport Conditional Flow Matching (OT-CFM).",
                typical_usage: "Zero-shot voice cloning from a 3-second reference clip, expressive emotional Text-to-Speech (TTS), and multi-speaker acoustic token generation.",
                typically_followed_by: "Snake periodic waveform vocoder activation or HiFi-GAN / Vocos mel-to-waveform readout head.",
            },
            Self::TabularGBDTVimeSSL { .. } => OperatorOverview {
                category: "Routing & MoE",
                complexity_tag: "Sparse / Conditional",
                what_it_does: "Self-supervised & differentiable Gradient Boosted Decision Tree ensemble (CatBoost Oblivious Trees / NODE) trained via VIME (Value Imputation & Mask Estimation), Hastie synthetic-vs-real marginal discrimination, and unsupervised tree-leaf index embeddings.",
                typical_usage: "Self-supervised representation learning and high-accuracy prediction on heterogeneous tabular data, financial credit risk, and clinical EHR tables.",
                typically_followed_by: "Entmax 1.5 sparse gate, tree-leaf embedding Linear projection, or calibrated classification/regression readout.",
            },
            Self::SuperSmoothLipschitzLayer { .. } => OperatorOverview {
                category: "Spectral & Physics",
                complexity_tag: "Physics & Geometric",
                what_it_does: "Enforces strict K-Lipschitz continuity and C^∞ higher-order Sobolev W^{k,2} smoothness via iterative Björck orthonormalization (W_{k+1} = 1.5 W_k - 0.5 W_k W_k^T W_k), Cayley orthogonal maps, and polyharmonic/GroupSort smooth projections.",
                typical_usage: "Super-smooth C^∞ surrogate models, second/fourth-order PDEs (biharmonic/Navier-Stokes), Brenier Optimal Transport potentials, and certified adversarial robustness.",
                typically_followed_by: "Softplus-β / GroupSort-2 / Rational Padé activation, KAN spline layer, or Sobolev H² physics readout head.",
            },
            Self::LearnableQuantizerQAT { .. } => OperatorOverview {
                category: "Norms & Activations",
                complexity_tag: "Optimization-as-Architecture",
                what_it_does: "Differentiable quantization level, step-size & mixed-bitwidth learner (LSQ+ / PACT / FracBits / LQ-Nets) that optimizes per-layer fractional bitwidth b_l in [min_bits, max_bits], quantization step size s, clipping threshold α, and non-uniform reconstruction levels via STE.",
                typical_usage: "Training the quantization level jointly with model weights under a hardware Bit-Operations (BOPs) or VRAM budget for sub-4-bit/ternary deployment.",
                typically_followed_by: "Quantized Linear/SwiGLU/MHA projection or BitNet 1.58b / INT4 / FP4 hardware readout.",
            },
        }
    }

    /// Returns 2-3 recommended operators that commonly follow `self` for one-click chaining in the Inspector.
    pub fn recommended_next_layers(&self) -> Vec<(&'static str, OperatorKind)> {
        match self {
            Self::RMSNorm { dim, .. } | Self::LayerNorm { dim, .. } => {
                let d = *dim;
                let heads = (d / 64).max(1);
                vec![
                    (
                        "Multi-Head Attention",
                        OperatorKind::MHA {
                            num_heads: heads,
                            d_k: 64,
                            d_v: 64,
                            dropout: 0.0,
                        },
                    ),
                    (
                        "Mamba-2 SSD",
                        OperatorKind::Mamba2SSD {
                            d_model: d,
                            d_state: 64,
                            d_conv: 4,
                            expand: 2,
                        },
                    ),
                    (
                        "SwiGLU Gated MLP",
                        OperatorKind::SwiGLU {
                            in_features: d,
                            hidden_features: (d * 8) / 3,
                        },
                    ),
                ]
            }
            Self::MHA { num_heads, d_v, .. } | Self::MLA { num_heads, d_v, .. } => {
                let d = num_heads * d_v;
                vec![
                    ("RMSNorm Layer", OperatorKind::RMSNorm { dim: d, eps: 1e-5 }),
                    (
                        "SwiGLU Gated MLP",
                        OperatorKind::SwiGLU {
                            in_features: d,
                            hidden_features: (d * 8) / 3,
                        },
                    ),
                    (
                        "DeepSeek Sparse MoE",
                        OperatorKind::DeepSeekMoE {
                            total_experts: 16,
                            routed_k: 4,
                            shared_experts: 2,
                            d_model: d,
                            d_ffn: d * 2,
                            aux_loss_free_bias: true,
                        },
                    ),
                ]
            }
            Self::Mamba2SSD { d_model, .. }
            | Self::MambaS6 { d_model, .. }
            | Self::xLSTM { d_model, .. }
            | Self::LiquidNeuralCfC { d_model, .. }
            | Self::SpikingLIF { d_model, .. } => {
                let d = *d_model;
                vec![
                    ("RMSNorm Layer", OperatorKind::RMSNorm { dim: d, eps: 1e-5 }),
                    (
                        "SwiGLU Gated MLP",
                        OperatorKind::SwiGLU {
                            in_features: d,
                            hidden_features: (d * 8) / 3,
                        },
                    ),
                    (
                        "Dense Linear Head",
                        OperatorKind::Linear {
                            in_features: d,
                            out_features: d,
                            bias: false,
                        },
                    ),
                ]
            }
            Self::AdaLNZeroDiT {
                d_model,
                num_heads,
                cond_dim,
                ..
            } => {
                let d = *d_model;
                vec![
                    (
                        "AdaLN-Zero MMDiT Block",
                        OperatorKind::AdaLNZeroDiT {
                            d_model: d,
                            num_heads: *num_heads,
                            cond_dim: *cond_dim,
                            dual_stream_mmdit: false,
                        },
                    ),
                    (
                        "SwiGLU Gated MLP",
                        OperatorKind::SwiGLU {
                            in_features: d,
                            hidden_features: (d * 8) / 3,
                        },
                    ),
                ]
            }
            Self::SwiGLU { in_features, .. } | Self::GeGLU { in_features, .. } => {
                let d = *in_features;
                vec![
                    ("RMSNorm Layer", OperatorKind::RMSNorm { dim: d, eps: 1e-5 }),
                    (
                        "Dense Linear Head",
                        OperatorKind::Linear {
                            in_features: d,
                            out_features: d,
                            bias: false,
                        },
                    ),
                ]
            }
            Self::Linear { out_features, .. } => {
                let d = *out_features;
                vec![
                    ("RMSNorm Layer", OperatorKind::RMSNorm { dim: d, eps: 1e-5 }),
                    (
                        "SwiGLU Gated MLP",
                        OperatorKind::SwiGLU {
                            in_features: d,
                            hidden_features: (d * 8) / 3,
                        },
                    ),
                ]
            }
            _ => vec![
                (
                    "RMSNorm Layer",
                    OperatorKind::RMSNorm {
                        dim: 512,
                        eps: 1e-5,
                    },
                ),
                (
                    "SwiGLU Gated MLP",
                    OperatorKind::SwiGLU {
                        in_features: 512,
                        hidden_features: 1376,
                    },
                ),
            ],
        }
    }

    /// Nominal operational FLOPs per token / element.
    pub fn compute_flops(&self, seq_len: usize) -> usize {
        match self {
            Self::Linear {
                in_features,
                out_features,
                ..
            } => 2 * in_features * out_features,
            Self::Embedding { embedding_dim, .. } => *embedding_dim,
            Self::MHA {
                num_heads,
                d_k,
                d_v,
                ..
            } => {
                let d_model = num_heads * d_k;
                // QKV projection + Attn dot products + Output proj
                (2 * d_model * (2 * d_k * num_heads + d_v * num_heads))
                    + (2 * seq_len * d_model)
                    + (2 * d_model * d_model)
            }
            Self::GQA {
                num_q_heads,
                num_kv_heads,
                d_k,
            } => {
                let d_model = num_q_heads * d_k;
                let kv_dim = num_kv_heads * d_k;
                (2 * d_model * (d_model + 2 * kv_dim))
                    + (2 * seq_len * d_model)
                    + (2 * d_model * d_model)
            }
            Self::MLA {
                num_heads,
                d_latent_kv,
                d_rope,
                d_v,
            } => {
                let d_model = num_heads * d_v;
                (2 * d_model * (2 * d_latent_kv + d_rope))
                    + (2 * seq_len * (d_latent_kv + d_rope))
                    + (2 * d_model * d_model)
            }
            Self::DiffAttn { num_heads, d_k } => {
                let d_model = num_heads * d_k;
                4 * d_model * d_model + 4 * seq_len * d_model
            }
            Self::AdaLNZeroDiT {
                d_model,
                cond_dim,
                dual_stream_mmdit,
                ..
            } => {
                let stream_mult = if *dual_stream_mmdit { 2 } else { 1 };
                stream_mult
                    * (8 * d_model * d_model + 12 * d_model * cond_dim + 2 * seq_len * d_model)
            }
            Self::MultiTokenPrediction {
                d_model,
                vocab_size: _,
                num_predict_tokens,
            } => num_predict_tokens * (4 * d_model * d_model),
            Self::SwiGLU {
                in_features,
                hidden_features,
            }
            | Self::GeGLU {
                in_features,
                hidden_features,
            } => 6 * in_features * hidden_features,
            Self::DeepSeekMoE {
                routed_k,
                shared_experts,
                d_model,
                d_ffn,
                total_experts,
                ..
            } => {
                let active_experts = routed_k + shared_experts;
                (2 * d_model * total_experts) + active_experts * (6 * d_model * d_ffn)
            }
            Self::KANLayer {
                in_features,
                out_features,
                grid_size,
                spline_order,
            } => 2 * in_features * out_features * (grid_size + spline_order + 1),
            Self::Mamba2SSD {
                d_model,
                d_state,
                expand,
                ..
            }
            | Self::MambaS6 {
                d_model,
                d_state,
                expand,
                ..
            } => {
                let d_inner = d_model * expand;
                6 * d_model * d_inner + 4 * d_inner * d_state
            }
            Self::xLSTM {
                d_model, num_heads, ..
            } => 8 * d_model * d_model + 4 * d_model * num_heads,
            Self::S4D { d_model, d_state } | Self::LRU { d_model, d_state } => {
                4 * d_model * d_state + 2 * d_model * d_model
            }
            Self::TitansMemory {
                d_model,
                memory_depth,
            } => 4 * memory_depth * d_model * d_model,
            Self::LiquidNeuralCfC {
                d_model,
                backbone_units,
            } => 6 * d_model * backbone_units,
            Self::SpikingLIF { d_model, .. } => 2 * d_model * d_model + 4 * d_model,
            Self::HyperbolicPoincareLayer { dim, .. } => 4 * dim * dim + 16 * dim,
            Self::JEPAPredictor {
                d_model,
                predictor_dim,
                ..
            } => 6 * d_model * predictor_dim + 2 * seq_len * predictor_dim,
            Self::PairformerBlock {
                d_single, d_pair, ..
            } => 8 * d_single * d_single + 24 * d_pair * d_pair,
            Self::WorldModelRSSM {
                d_model,
                stochastic_classes,
                stochastic_dim,
            } => {
                let z_dim = stochastic_classes * stochastic_dim;
                6 * d_model * (d_model + z_dim)
            }
            Self::RWKV7DeltaRule { d_model, .. } => 8 * d_model * d_model,
            Self::EntropyBytePatcher {
                d_byte, d_latent, ..
            } => 4 * d_byte * d_latent,
            Self::ColPaliMaxSim { d_model, proj_dim } => {
                2 * d_model * proj_dim + 2 * seq_len * proj_dim
            }
            Self::SphericalMeshGNN {
                d_node,
                d_edge,
                mesh_refinement_level,
            } => (6 * d_node * d_node + 4 * d_node * d_edge) * (*mesh_refinement_level).max(1),
            Self::QuantumVariationalCircuit {
                num_qubits,
                circuit_depth,
                ..
            } => {
                let state_dim = 1usize << (*num_qubits).min(16);
                circuit_depth * num_qubits * state_dim * 4
            }
            Self::PhotonicMZIMesh { num_modes, .. } => 4 * num_modes * num_modes,
            Self::SimplicialHodgeConv {
                d_node,
                d_edge,
                d_face,
            } => 6 * (d_node * d_node + d_edge * d_edge + d_face * d_face),
            Self::ContinuousNormalizingFlow {
                dim,
                hutchinson_samples,
            } => 8 * dim * dim * (*hutchinson_samples).max(1),
            Self::EnergyLangevinMCMC {
                dim,
                langevin_steps,
                ..
            } => 4 * dim * dim * (*langevin_steps).max(1),
            Self::NeurosymbolicLogicDNC {
                d_model,
                memory_slots,
                ..
            } => 6 * d_model * d_model + 4 * d_model * memory_slots,
            Self::MultiAgentQMIX {
                num_agents,
                state_dim,
                mixing_embed_dim,
            } => 4 * num_agents * mixing_embed_dim * state_dim,
            Self::BioOrganoidActiveInference {
                num_channels,
                free_energy_iters,
            } => 6 * num_channels * num_channels * (*free_energy_iters).max(1),
            Self::AttentionFreeGatedMLP {
                d_model,
                seq_len: s_len,
                filter_order,
            } => (4 * d_model * d_model + 2 * d_model * (*s_len).max(1)) * (*filter_order).max(1),
            Self::RawBitByteMultiScale {
                d_global,
                d_local,
                patch_bytes,
                ..
            } => 6 * d_global * d_global + (*patch_bytes).max(1) * 4 * d_local * d_local,
            Self::VoiceCloningCodecTTS {
                d_model,
                num_rvq_codebooks,
                speaker_embed_dim,
            } => {
                8 * d_model * d_model
                    + 4 * d_model * speaker_embed_dim
                    + 2 * d_model * (*num_rvq_codebooks).max(1) * 1024
            }
            Self::TabularGBDTVimeSSL {
                num_trees,
                tree_depth,
                ..
            } => {
                let leaves = 1usize << (*tree_depth).min(12);
                4 * num_trees * leaves * (*tree_depth).max(1)
            }
            Self::SuperSmoothLipschitzLayer {
                dim,
                bjorck_iters,
                sobolev_order,
                ..
            } => 2 * dim * dim * ((*bjorck_iters).max(1) + (*sobolev_order).max(1)),
            Self::LearnableQuantizerQAT { dim, .. } => 6 * dim,
            Self::EchoStateNetwork { reservoir_size, .. } => 2 * reservoir_size * reservoir_size,
            Self::NextGenReservoir {
                delay_taps,
                poly_degree,
            } => 256 * delay_taps * poly_degree,
            Self::ConvNeXtV2Block { dim, mlp_ratio, .. } => {
                let hidden = (*dim as f32 * mlp_ratio) as usize;
                2 * dim * dim + 4 * dim * hidden
            }
            Self::WTConv {
                in_channels,
                out_channels,
                wavelet_levels,
            } => 2 * in_channels * out_channels * wavelet_levels * 9,
            Self::FourierNeuralOperator { modes, width } => {
                let m = modes.first().copied().unwrap_or(16);
                4 * width * width * m
            }
            Self::DeepONet {
                branch_dim,
                trunk_dim,
                basis_dim,
            } => 2 * (branch_dim + trunk_dim) * basis_dim,
            Self::EGNN { node_dim, edge_dim } => 6 * node_dim * node_dim + 4 * node_dim * edge_dim,
            Self::EvoformerTriangle { d_pair } => 12 * d_pair * d_pair,
            Self::InvariantPointAttention {
                num_heads,
                num_points,
            } => 64 * num_heads * num_points * 12,
            Self::CRATE { d_model, .. } | Self::DEQ { d_model, .. } | Self::TTT { d_model, .. } => {
                6 * d_model * d_model
            }
            Self::RMSNorm { dim, .. } | Self::LayerNorm { dim, .. } => 4 * dim,
            _ => 4096, // fallback baseline
        }
    }

    /// Primary reference curriculum topic slug in AI-Examples.
    pub fn curriculum_slug(&self) -> &'static str {
        match self {
            Self::Activation { .. } => "05_architectures/01_mlp_moe_soups_and_kan",
            Self::MHA { .. } => {
                "05_architectures/04_transformers_and_heads/01_attention_variants_mha_mqa_gqa_mla_and_diffattn"
            }
            Self::GQA { .. } | Self::MLA { .. } | Self::DiffAttn { .. } => {
                "05_architectures/04_transformers_and_heads/01_attention_variants_mha_mqa_gqa_mla_and_diffattn"
            }
            Self::AdaLNZeroDiT { .. } => {
                "07_domain_tasks/01_computer_vision_and_3d/02_diffusion_transformers_dit_mmdit_and_sit"
            }
            Self::MultiTokenPrediction { .. } => {
                "15_compression_peft_and_speculative_inference/05_speculative_decoding_medusa_eagle_and_multi_token_prediction"
            }
            Self::MambaS6 { .. } | Self::Mamba2SSD { .. } => {
                "05_architectures/03_recurrent_ssm_and_memory/04_mamba_selective_ssm_and_mamba2_ssd"
            }
            Self::S4D { .. } | Self::LRU { .. } => {
                "05_architectures/03_recurrent_ssm_and_memory/03_structured_state_space_models_s4_and_lru"
            }
            Self::xLSTM { .. } => {
                "05_architectures/03_recurrent_ssm_and_memory/05_xlstm_extended_long_short_term_memory_mlstm_and_slstm"
            }
            Self::LiquidNeuralCfC { .. } => {
                "06_smooth_and_physics_informed_ai/01_neural_odes_liquid_networks_and_continuous_depth"
            }
            Self::SpikingLIF { .. } => {
                "05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/05_spiking_neural_networks_snn_lif_and_surrogate_gradients"
            }
            Self::HyperbolicPoincareLayer { .. } => {
                "01_paradigms_and_dichotomies/09_non_euclidean_geometries_hyperbolic_poincare_spherical_and_spd"
            }
            Self::JEPAPredictor { .. } => {
                "01_paradigms_and_dichotomies/05_joint_embedding_predictive_architectures_ijepa_vjepa_and_world_models"
            }
            Self::PairformerBlock { .. }
            | Self::EvoformerTriangle { .. }
            | Self::InvariantPointAttention { .. } => {
                "07_domain_tasks/04_autonomous_driving_robotics_and_biotech/04_biomolecular_foundation_models_alphafold3_pairformer_and_esm3"
            }
            Self::WorldModelRSSM { .. } => {
                "08_learning_paradigms_rl_and_alignment/05_model_based_rl_dreamerv3_rssm_and_muzero_world_models"
            }
            Self::RWKV7DeltaRule { .. } => {
                "05_architectures/03_recurrent_ssm_and_memory/06_rwkv7_gated_deltanet_and_test_time_training_rnns"
            }
            Self::EntropyBytePatcher { .. } => {
                "05_architectures/04_transformers_and_heads/06_byte_latent_transformers_blt_and_dynamic_token_patching"
            }
            Self::ColPaliMaxSim { .. } => {
                "07_domain_tasks/05_multimodal_and_cross_modal/04_late_interaction_colbert_and_colpali_visual_document_retrieval"
            }
            Self::SphericalMeshGNN { .. } => {
                "06_smooth_and_physics_informed_ai/07_spherical_harmonic_and_icosahedral_weather_models_graphcast_neuralgcm"
            }
            Self::QuantumVariationalCircuit { .. }
            | Self::PhotonicMZIMesh { .. }
            | Self::BioOrganoidActiveInference { .. } => {
                "05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/06_quantum_photonic_and_analog_in_memory_neural_networks"
            }
            Self::SimplicialHodgeConv { .. } => {
                "05_architectures/06_graphs_geometric_and_inr/04_equivariant_topological_and_point_cloud_networks_egnn_mpsn"
            }
            Self::ContinuousNormalizingFlow { .. } => {
                "06_smooth_and_physics_informed_ai/02_continuous_normalizing_flows_ffjord_and_neural_copulas"
            }
            Self::EnergyLangevinMCMC { .. } => {
                "01_paradigms_and_dichotomies/08_energy_based_models_langevin_dynamics_and_predictive_coding"
            }
            Self::NeurosymbolicLogicDNC { .. } => {
                "05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/03_neurosymbolic_ai_logic_tensor_networks_and_differentiable_memory"
            }
            Self::MultiAgentQMIX { .. } => {
                "08_learning_paradigms_rl_and_alignment/06_multi_agent_rl_qmix_mappo_and_cfr_game_theory"
            }
            Self::LISTA { .. } | Self::ADMMNet { .. } | Self::CRATE { .. } => {
                "06_smooth_and_physics_informed_ai/11_optimisation_as_architecture_lista_admm_net_crate_and_mesa_optimizers"
            }
            Self::DEQ { .. } => {
                "06_smooth_and_physics_informed_ai/08_deep_equilibrium_models_deq_implicit_differentiation"
            }
            Self::FourierNeuralOperator { .. } | Self::DeepONet { .. } => {
                "06_smooth_and_physics_informed_ai/06_neural_operators_fno_deeponet_and_meshgraphnets"
            }
            Self::EGNN { .. }
            | Self::CellularSheafConv { .. }
            | Self::CliffordGeometricProduct { .. } => {
                "05_architectures/06_graphs_geometric_and_inr/04_equivariant_topological_and_point_cloud_networks_egnn_mpsn"
            }
            Self::GaussianSplatting3D { .. } => {
                "07_domain_tasks/01_computer_vision_and_3d/04_3d_gaussian_splatting_nerf_and_plucker_ray_fields"
            }
            Self::KANLayer { .. } => {
                "05_architectures/01_mlp_moe_soups_and_kan/05_kolmogorov_arnold_networks_kan_bspline_chebyshev_rbf_and_wavkan"
            }
            Self::DeepSeekMoE { .. } => {
                "05_architectures/01_mlp_moe_soups_and_kan/03_deepseekmoe_fine_grained_experts_and_shared_isolation"
            }
            Self::SlotAttention { .. } => {
                "05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic/04_abstract_representations_slot_attention_and_global_workspace"
            }
            Self::WTConv { .. } => {
                "05_architectures/02_convolutional_and_spatial/04_wavelet_convolutions_wtconv_and_multiscale_wavelet_attention"
            }
            Self::AttentionFreeGatedMLP { .. } => {
                "05_architectures/01_mlp_moe_soups_and_kan/01_gated_mlps_swiglu_geglu_and_mlp_mixer"
            }
            Self::RawBitByteMultiScale { .. } => {
                "05_architectures/04_transformers_and_heads/06_byte_latent_transformers_blt_and_dynamic_token_patching"
            }
            Self::VoiceCloningCodecTTS { .. } => {
                "07_domain_tasks/02_audio_speech_and_music/02_neural_audio_codecs_rvq_and_zero_shot_voice_cloning_tts"
            }
            Self::TabularGBDTVimeSSL { .. } => {
                "02_classical_and_probabilistic_ml/07_differentiable_trees_node_soft_gbdt_and_gam_nam"
            }
            Self::SuperSmoothLipschitzLayer { .. } => {
                "06_smooth_and_physics_informed_ai/01_lipschitz_networks_bjorck_cayley_and_groupsort"
            }
            Self::LearnableQuantizerQAT { .. } => {
                "14_quantisation/04_vector_lattice_qat_and_lowbit/03_qat_ste_lsq_lsq_plus_pact_and_dorefa"
            }
            _ => "05_architectures",
        }
    }
}
