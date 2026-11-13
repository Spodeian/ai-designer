//! Built-in Architecture Archetypes and Scaffolding Engine.
//! Provides mathematical blueprints for starting AI designs from scratch or guided archetypes.

use crate::spec::graph::{NodeSpec, NormPlacement, WiringPattern};
use crate::spec::io::{
    CoordinateEncoding, InputPortSpec, LossPairing, NormalizerSpec, OutputPortSpec, PatchingSpec,
    StructuralConstraint, SymbolicDim, TensorSpace,
};
use crate::spec::operators::OperatorKind;
use crate::spec::precision::PrecisionFormat;
use crate::spec::{AiDesignSpec, HardwareTargetProfile};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ArchitectureArchetype {
    #[default]
    Blank,
    GenerativeLLM,
    VisionDiffusion,
    PhysicsOperator,
    SuperSmoothLipschitz,
    LearnableQuantizationQAT,
    ZeroShotVoiceCloningTTS,
    RawBitByteMultiScaleLLM,
    AttentionFreeGatedMixer,
    SelfSupervisedFoundation,
    TabularGBDTVimeSSL,
    RoboticsPolicy,
    TimeSeriesForecaster,
    TabularPriorNetwork,
    DeepResidual,
    DenseMoE,
    SymbolicAstReasoning,
}

impl ArchitectureArchetype {
    pub const ALL: &'static [ArchitectureArchetype] = &[
        ArchitectureArchetype::Blank,
        ArchitectureArchetype::GenerativeLLM,
        ArchitectureArchetype::VisionDiffusion,
        ArchitectureArchetype::PhysicsOperator,
        ArchitectureArchetype::SuperSmoothLipschitz,
        ArchitectureArchetype::LearnableQuantizationQAT,
        ArchitectureArchetype::ZeroShotVoiceCloningTTS,
        ArchitectureArchetype::RawBitByteMultiScaleLLM,
        ArchitectureArchetype::AttentionFreeGatedMixer,
        ArchitectureArchetype::SelfSupervisedFoundation,
        ArchitectureArchetype::TabularGBDTVimeSSL,
        ArchitectureArchetype::RoboticsPolicy,
        ArchitectureArchetype::TimeSeriesForecaster,
        ArchitectureArchetype::TabularPriorNetwork,
        ArchitectureArchetype::DeepResidual,
        ArchitectureArchetype::DenseMoE,
        ArchitectureArchetype::SymbolicAstReasoning,
    ];

    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Blank => "Blank Canvas (Clean Slate)",
            Self::GenerativeLLM => "Generative LLM / Reasoning Hybrid",
            Self::VisionDiffusion => "Vision & Spatiotemporal DiT Flow",
            Self::PhysicsOperator => "Fourier Neural Operator & PINN",
            Self::SuperSmoothLipschitz => "Super-Smooth C^∞ Lipschitz, Spline & Sobolev Network",
            Self::LearnableQuantizationQAT => {
                "Learnable Quantization Level & Mixed-Bitwidth QAT Network"
            }
            Self::ZeroShotVoiceCloningTTS => "Zero-Shot Voice Cloning & Neural Codec Flow TTS",
            Self::RawBitByteMultiScaleLLM => {
                "Raw Bit & Byte Multi-Scale LLM (MegaByte / MambaByte)"
            }
            Self::AttentionFreeGatedMixer => "Attention-Free gMLP & Hyena Rank-Preserving Mixer",
            Self::SelfSupervisedFoundation => {
                "Self-Supervised Vision / Multimodal Backbone (DINOv2 + I-JEPA + VICReg)"
            }
            Self::TabularGBDTVimeSSL => "Self-Supervised Gradient Boosted Trees (VIME + CatBoost)",
            Self::RoboticsPolicy => "Robotics ACT Diffusion Policy",
            Self::TimeSeriesForecaster => "Patch xLSTM Probabilistic Forecaster",
            Self::TabularPriorNetwork => "TabPFN Monotonic Prior-Data Network",
            Self::DeepResidual => "Deep Pre-LN Residual Network",
            Self::DenseMoE => "Sparse MoE Mixture-of-Experts",
            Self::SymbolicAstReasoning => {
                "Symbolic AST DAG Reasoning (Tree-Transformer + Equality Saturation)"
            }
        }
    }

    pub fn domain(&self) -> &'static str {
        match self {
            Self::Blank => "Custom",
            Self::GenerativeLLM => "NaturalLanguageProcessing",
            Self::VisionDiffusion => "ComputerVision",
            Self::PhysicsOperator => "PhysicsInformedAndGeometric",
            Self::SuperSmoothLipschitz => "SmoothAndCertifiedLipschitzAI",
            Self::LearnableQuantizationQAT => "QuantizationAndMixedPrecision",
            Self::ZeroShotVoiceCloningTTS => "AudioSpeechAndVoiceCloning",
            Self::RawBitByteMultiScaleLLM => "TokenizerFreeBitByteModels",
            Self::AttentionFreeGatedMixer => "SubquadraticSequenceMixers",
            Self::SelfSupervisedFoundation => "SelfSupervisedLearning",
            Self::TabularGBDTVimeSSL => "TabularAndGradientBoostingSSL",
            Self::RoboticsPolicy => "RoboticsAndEmbodiedAI",
            Self::TimeSeriesForecaster => "TimeSeriesAndFinance",
            Self::TabularPriorNetwork => "TabularAndClassicalML",
            Self::DeepResidual => "GeneralDeepLearning",
            Self::DenseMoE => "FrontierArchitectures",
            Self::SymbolicAstReasoning => "SymbolicAndMathematicalReasoning",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::Blank => {
                "Clean slate with customizable I/O ports. Build custom DAG layer-by-layer."
            }
            Self::GenerativeLLM => {
                "Autoregressive decoder with RoPE, RMSNorm, Multi-Head Attention, and SwiGLU."
            }
            Self::VisionDiffusion => {
                "Spatiotemporal DiT transformer with Optimal Transport Flow Matching velocity head."
            }
            Self::PhysicsOperator => {
                "Continuous operator learning with Fourier Neural Operator and Wavelet Convolutions."
            }
            Self::SuperSmoothLipschitz => {
                "Certified K-Lipschitz & C^∞ smooth network with Björck orthonormalization, Softplus-β/Rational Padé activations, KAN splines & Sobolev H² loss."
            }
            Self::LearnableQuantizationQAT => {
                "Jointly trains weights and quantization levels: LSQ+ step-size, PACT activation clipping, and FracBits per-layer fractional bitwidth under BOPs penalty."
            }
            Self::ZeroShotVoiceCloningTTS => {
                "In-context 3s speaker prompt conditioning, 8-codebook RVQ synthesis, OT-CFM DiT velocity estimation, and periodic Snake vocoder filter."
            }
            Self::RawBitByteMultiScaleLLM => {
                "Tokenizer-free raw binary/UTF-8 byte & 8-bit-plane model combining dynamic Shannon entropy patching, Mamba-2 SSD, and MegaByte local decoding."
            }
            Self::AttentionFreeGatedMixer => {
                "Prevents rank collapse without self-attention via gMLP Spatial Gating Units (SGU), Hyena implicit long convolutions, and ConvNeXt-V2 blocks."
            }
            Self::SelfSupervisedFoundation => {
                "Multi-view self-supervised encoder with modular I-JEPA latent mask predictor, DINOv2 Sinkhorn-Knopp prototypes, and VICReg anti-collapse head."
            }
            Self::TabularGBDTVimeSSL => {
                "Oblivious decision tree ensemble (CatBoost/NODE) with VIME mask/imputation heads, Hastie marginal discrimination, and Entmax-1.5 split gates."
            }
            Self::RoboticsPolicy => {
                "Action chunking diffusion policy with differentiable Control Barrier Functions (CBF)."
            }
            Self::TimeSeriesForecaster => {
                "Reversible Instance Norm (RevIN), xLSTM matrix memory, and CRPS quantile readout."
            }
            Self::TabularPriorNetwork => {
                "Prior-fitted network with KAN splines and monotonic feature embeddings."
            }
            Self::DeepResidual => {
                "Pre-LN residual backbone with SwiGLU/Dense activations and layer-scaling."
            }
            Self::DenseMoE => {
                "Multi-Head Latent Attention (MLA) and DeepSeek-style top-k sparse router."
            }
            Self::SymbolicAstReasoning => {
                "Equivalence-aware DAG-NN and Tree-Relative Transformer for mathematical AST embeddings, theorem proving, and term rewriting."
            }
        }
    }

    pub fn icon(&self) -> &'static str {
        match self {
            Self::Blank => "\u{f38b}",                    // bi-file-earmark-text
            Self::GenerativeLLM => "\u{f6b1}",            // bi-robot
            Self::VisionDiffusion => "\u{f341}",          // bi-eye
            Self::PhysicsOperator => "\u{f46c}",          // bi-lightning-charge-fill
            Self::SuperSmoothLipschitz => "\u{f56b}",     // bi-shield-check
            Self::LearnableQuantizationQAT => "\u{f581}", // bi-sliders
            Self::ZeroShotVoiceCloningTTS => "\u{f490}",  // bi-mic-fill
            Self::RawBitByteMultiScaleLLM => "\u{f3ae}",  // bi-file-earmark-binary
            Self::AttentionFreeGatedMixer => "\u{f2e8}",  // bi-diagram-3
            Self::SelfSupervisedFoundation => "\u{f479}", // bi-magic
            Self::TabularGBDTVimeSSL => "\u{f5aa}",       // bi-table
            Self::RoboticsPolicy => "\u{f2d6}",           // bi-cpu
            Self::TimeSeriesForecaster => "\u{f673}",     // bi-graph-up-arrow
            Self::TabularPriorNetwork => "\u{f5aa}",      // bi-table
            Self::DeepResidual => "\u{f45b}",             // bi-layers
            Self::DenseMoE => "\u{f544}",                 // bi-shuffle
            Self::SymbolicAstReasoning => "\u{f6ff}",     // diagram-tree
        }
    }

    pub fn default_d_model(&self) -> usize {
        match self {
            Self::Blank => 256,
            Self::GenerativeLLM => 512,
            Self::VisionDiffusion => 512,
            Self::PhysicsOperator => 64,
            Self::SuperSmoothLipschitz => 256,
            Self::LearnableQuantizationQAT => 512,
            Self::ZeroShotVoiceCloningTTS => 512,
            Self::RawBitByteMultiScaleLLM => 512,
            Self::AttentionFreeGatedMixer => 512,
            Self::SelfSupervisedFoundation => 768,
            Self::TabularGBDTVimeSSL => 128,
            Self::RoboticsPolicy => 256,
            Self::TimeSeriesForecaster => 128,
            Self::TabularPriorNetwork => 64,
            Self::DeepResidual => 256,
            Self::DenseMoE => 512,
            Self::SymbolicAstReasoning => 256,
        }
    }

    pub fn default_depth(&self) -> usize {
        match self {
            Self::Blank => 0,
            Self::GenerativeLLM => 4,
            Self::VisionDiffusion => 4,
            Self::PhysicsOperator => 3,
            Self::SuperSmoothLipschitz => 4,
            Self::LearnableQuantizationQAT => 4,
            Self::ZeroShotVoiceCloningTTS => 3,
            Self::RawBitByteMultiScaleLLM => 3,
            Self::AttentionFreeGatedMixer => 4,
            Self::SelfSupervisedFoundation => 4,
            Self::TabularGBDTVimeSSL => 3,
            Self::RoboticsPolicy => 3,
            Self::TimeSeriesForecaster => 3,
            Self::TabularPriorNetwork => 3,
            Self::DeepResidual => 4,
            Self::DenseMoE => 4,
            Self::SymbolicAstReasoning => 4,
        }
    }
}

/// Scaffolds a complete, mathematically coherent `AiDesignSpec` for designing from scratch or from an archetype.
pub fn scaffold_archetype(
    archetype: ArchitectureArchetype,
    model_id: &str,
    display_name: &str,
    d_model: usize,
    depth: usize,
    precision: PrecisionFormat,
) -> AiDesignSpec {
    let mut spec = AiDesignSpec::new(model_id, display_name);
    spec.metadata.domain = archetype.domain().to_string();
    spec.metadata.target_hardware = HardwareTargetProfile::rtx_4090();

    match archetype {
        ArchitectureArchetype::Blank => {
            spec.metadata.tags = vec!["scratch".to_string(), "custom".to_string()];
            spec.inputs = vec![InputPortSpec {
                id: "x".to_string(),
                name: "Primary Input Feature Stream".to_string(),
                space: TensorSpace::ContinuousVector {
                    dimension: d_model,
                    range: None,
                },
                symbolic_shape: vec![SymbolicDim::from("batch"), SymbolicDim::Known(d_model)],
                coordinate_encoding: CoordinateEncoding::None,
                normalizer: NormalizerSpec::None,
                patching: None,
            }];
            spec.outputs = vec![OutputPortSpec {
                id: "y".to_string(),
                name: "Primary Output Prediction Stream".to_string(),
                space: TensorSpace::ContinuousVector {
                    dimension: d_model,
                    range: None,
                },
                symbolic_shape: vec![SymbolicDim::from("batch"), SymbolicDim::Known(d_model)],
                constraint: StructuralConstraint::DirectLinear,
                loss: LossPairing::MSE,
            }];
            spec.graph.nodes = vec![NodeSpec {
                id: "linear_1".to_string(),
                name: "Dense Linear Layer".to_string(),
                operator: OperatorKind::Linear {
                    in_features: d_model,
                    out_features: d_model,
                    bias: true,
                },
                precision: Some(precision),
                ..Default::default()
            }];
        }
        ArchitectureArchetype::GenerativeLLM => {
            spec.metadata.tags = vec![
                "llm".to_string(),
                "generative".to_string(),
                "rope".to_string(),
                "swiglu".to_string(),
            ];
            spec.inputs = vec![InputPortSpec {
                id: "input_ids".to_string(),
                name: "Token Sequence IDs".to_string(),
                space: TensorSpace::DiscreteTokens {
                    vocab_size: 32000,
                    is_causal: true,
                },
                symbolic_shape: vec![SymbolicDim::from("batch"), SymbolicDim::from("seq_len")],
                coordinate_encoding: CoordinateEncoding::RoPE1D { base: 10000.0 },
                normalizer: NormalizerSpec::None,
                patching: None,
            }];
            spec.outputs = vec![OutputPortSpec {
                id: "logits".to_string(),
                name: "Next Token Logits".to_string(),
                space: TensorSpace::DiscreteTokens {
                    vocab_size: 32000,
                    is_causal: true,
                },
                symbolic_shape: vec![
                    SymbolicDim::from("batch"),
                    SymbolicDim::from("seq_len"),
                    SymbolicDim::Known(32000),
                ],
                constraint: StructuralConstraint::DirectLinear,
                loss: LossPairing::CrossEntropy,
            }];
            spec.graph.topology = WiringPattern::ResidualStream {
                norm_placement: NormPlacement::PreLN,
                layer_scale_init: Some(1e-2),
            };

            let mut nodes = Vec::new();
            nodes.push(NodeSpec {
                id: "token_embed".to_string(),
                name: "Token Embedding Matrix".to_string(),
                operator: OperatorKind::Embedding {
                    num_embeddings: 32000,
                    embedding_dim: d_model,
                },
                precision: Some(precision),
                ..Default::default()
            });

            for i in 1..=depth {
                nodes.push(NodeSpec {
                    id: format!("attn_norm_{i}"),
                    name: format!("Layer {i} Pre-Attention RMSNorm"),
                    operator: OperatorKind::RMSNorm {
                        dim: d_model,
                        eps: 1e-5,
                    },
                    precision: Some(precision),
                    ..Default::default()
                });
                let num_heads = (d_model / 64).max(1);
                nodes.push(NodeSpec {
                    id: format!("mha_{i}"),
                    name: format!("Layer {i} Multi-Head Attention"),
                    operator: OperatorKind::MHA {
                        num_heads,
                        d_k: 64,
                        d_v: 64,
                        dropout: 0.0,
                    },
                    precision: Some(precision),
                    ..Default::default()
                });
                nodes.push(NodeSpec {
                    id: format!("ffn_norm_{i}"),
                    name: format!("Layer {i} Pre-FFN RMSNorm"),
                    operator: OperatorKind::RMSNorm {
                        dim: d_model,
                        eps: 1e-5,
                    },
                    precision: Some(precision),
                    ..Default::default()
                });
                let hidden_dim = (d_model * 8 / 3).max(128);
                nodes.push(NodeSpec {
                    id: format!("swiglu_{i}"),
                    name: format!("Layer {i} SwiGLU MLP"),
                    operator: OperatorKind::SwiGLU {
                        in_features: d_model,
                        hidden_features: hidden_dim,
                    },
                    precision: Some(precision),
                    ..Default::default()
                });
            }

            nodes.push(NodeSpec {
                id: "final_norm".to_string(),
                name: "Final RMSNorm Layer".to_string(),
                operator: OperatorKind::RMSNorm {
                    dim: d_model,
                    eps: 1e-5,
                },
                precision: Some(precision),
                ..Default::default()
            });
            nodes.push(NodeSpec {
                id: "lm_head".to_string(),
                name: "Language Model Readout Head".to_string(),
                operator: OperatorKind::Linear {
                    in_features: d_model,
                    out_features: 32000,
                    bias: false,
                },
                precision: Some(PrecisionFormat::FP32),
                ..Default::default()
            });
            spec.graph.nodes = nodes;
            spec.training.auto_partition_parameters = true;
            spec.training.hygiene.auto_clip = true;
        }
        ArchitectureArchetype::VisionDiffusion => {
            spec.metadata.tags = vec![
                "vision".to_string(),
                "diffusion".to_string(),
                "flow-matching".to_string(),
            ];
            spec.inputs = vec![
                InputPortSpec {
                    id: "pixel_values".to_string(),
                    name: "Input Image RGB Frame".to_string(),
                    space: TensorSpace::ImageRGB {
                        resolution: [224, 224],
                        channels: 3,
                    },
                    symbolic_shape: vec![
                        SymbolicDim::from("batch"),
                        SymbolicDim::Known(3),
                        SymbolicDim::Known(224),
                        SymbolicDim::Known(224),
                    ],
                    coordinate_encoding: CoordinateEncoding::RoPE2D_Dynamic { base: 10000.0 },
                    normalizer: NormalizerSpec::ImageNetNormalize,
                    patching: Some(PatchingSpec {
                        patch_size: [16, 16],
                        flatten_channels: true,
                    }),
                },
                InputPortSpec {
                    id: "timesteps".to_string(),
                    name: "Diffusion Timestep Coordinate".to_string(),
                    space: TensorSpace::ContinuousVector {
                        dimension: 1,
                        range: Some((0.0, 1.0)),
                    },
                    symbolic_shape: vec![SymbolicDim::from("batch"), SymbolicDim::Known(1)],
                    coordinate_encoding: CoordinateEncoding::None,
                    normalizer: NormalizerSpec::None,
                    patching: None,
                },
            ];
            spec.outputs = vec![OutputPortSpec {
                id: "velocity_field".to_string(),
                name: "Flow Velocity Vector Field".to_string(),
                space: TensorSpace::ContinuousVector {
                    dimension: 768,
                    range: None,
                },
                symbolic_shape: vec![SymbolicDim::from("batch"), SymbolicDim::Known(768)],
                constraint: StructuralConstraint::DirectLinear,
                loss: LossPairing::OptimalTransportConditionalFlowMatching,
            }];

            let mut nodes = Vec::new();
            nodes.push(NodeSpec {
                id: "patch_projection".to_string(),
                name: "Linear Patch Projection".to_string(),
                operator: OperatorKind::Linear {
                    in_features: 16 * 16 * 3,
                    out_features: d_model,
                    bias: true,
                },
                precision: Some(precision),
                ..Default::default()
            });
            for i in 1..=depth {
                nodes.push(NodeSpec {
                    id: format!("dit_attn_{i}"),
                    name: format!("DiT Spatial Attention {i}"),
                    operator: OperatorKind::MHA {
                        num_heads: (d_model / 64).max(1),
                        d_k: 64,
                        d_v: 64,
                        dropout: 0.0,
                    },
                    precision: Some(precision),
                    ..Default::default()
                });
                nodes.push(NodeSpec {
                    id: format!("dit_ffn_{i}"),
                    name: format!("DiT SwiGLU Block {i}"),
                    operator: OperatorKind::SwiGLU {
                        in_features: d_model,
                        hidden_features: d_model * 2,
                    },
                    precision: Some(precision),
                    ..Default::default()
                });
            }
            nodes.push(NodeSpec {
                id: "flow_readout".to_string(),
                name: "Flow Velocity Readout Head".to_string(),
                operator: OperatorKind::Linear {
                    in_features: d_model,
                    out_features: 768,
                    bias: true,
                },
                precision: Some(PrecisionFormat::FP32),
                ..Default::default()
            });
            spec.graph.nodes = nodes;
        }
        ArchitectureArchetype::PhysicsOperator => {
            spec.metadata.tags = vec![
                "physics".to_string(),
                "fno".to_string(),
                "spectral".to_string(),
                "pde".to_string(),
            ];
            spec.inputs = vec![InputPortSpec {
                id: "spatial_grid".to_string(),
                name: "Spatial Coordinate Discretization".to_string(),
                space: TensorSpace::CoordinateField {
                    domain_dim: 2,
                    time_dependent: false,
                },
                symbolic_shape: vec![
                    SymbolicDim::from("batch"),
                    SymbolicDim::Known(64),
                    SymbolicDim::Known(64),
                    SymbolicDim::Known(4),
                ],
                coordinate_encoding: CoordinateEncoding::RandomFourierFeatures {
                    num_features: 16,
                    scale: 1.0,
                },
                normalizer: NormalizerSpec::None,
                patching: None,
            }];
            spec.outputs = vec![OutputPortSpec {
                id: "solution_field".to_string(),
                name: "Continuous Field Solution".to_string(),
                space: TensorSpace::CoordinateField {
                    domain_dim: 2,
                    time_dependent: false,
                },
                symbolic_shape: vec![
                    SymbolicDim::from("batch"),
                    SymbolicDim::Known(64),
                    SymbolicDim::Known(64),
                    SymbolicDim::Known(1),
                ],
                constraint: StructuralConstraint::CurlFreeVectorField,
                loss: LossPairing::SobolevH1PhysicsResidual,
            }];

            let mut nodes = Vec::new();
            nodes.push(NodeSpec {
                id: "channel_lifting".to_string(),
                name: "Input Coordinate Lifting Layer".to_string(),
                operator: OperatorKind::Linear {
                    in_features: 4,
                    out_features: d_model,
                    bias: true,
                },
                precision: Some(precision),
                ..Default::default()
            });
            for i in 1..=depth {
                nodes.push(NodeSpec {
                    id: format!("fno_spectral_{i}"),
                    name: format!("Fourier Neural Spectral Conv {i}"),
                    operator: OperatorKind::FourierNeuralOperator {
                        modes: vec![16, 16],
                        width: d_model,
                    },
                    precision: Some(precision),
                    ..Default::default()
                });
                nodes.push(NodeSpec {
                    id: format!("wtconv_{i}"),
                    name: format!("Wavelet WTConv Waveform Filter {i}"),
                    operator: OperatorKind::WTConv {
                        in_channels: d_model,
                        out_channels: d_model,
                        wavelet_levels: 3,
                    },
                    precision: Some(precision),
                    ..Default::default()
                });
            }
            nodes.push(NodeSpec {
                id: "channel_projection".to_string(),
                name: "Solution Projection Layer".to_string(),
                operator: OperatorKind::Linear {
                    in_features: d_model,
                    out_features: 1,
                    bias: true,
                },
                precision: Some(PrecisionFormat::FP32),
                ..Default::default()
            });
            spec.graph.nodes = nodes;
        }
        ArchitectureArchetype::RoboticsPolicy => {
            spec.metadata.tags = vec![
                "robotics".to_string(),
                "diffusion-act".to_string(),
                "cbf".to_string(),
            ];
            spec.inputs = vec![
                InputPortSpec {
                    id: "wrist_cam".to_string(),
                    name: "Wrist Camera Visual Frame".to_string(),
                    space: TensorSpace::ImageRGB {
                        resolution: [224, 224],
                        channels: 3,
                    },
                    symbolic_shape: vec![
                        SymbolicDim::from("batch"),
                        SymbolicDim::Known(3),
                        SymbolicDim::Known(224),
                        SymbolicDim::Known(224),
                    ],
                    coordinate_encoding: CoordinateEncoding::None,
                    normalizer: NormalizerSpec::ImageNetNormalize,
                    patching: None,
                },
                InputPortSpec {
                    id: "proprioception".to_string(),
                    name: "Joint Angles & End-Effector States".to_string(),
                    space: TensorSpace::ContinuousVector {
                        dimension: 14,
                        range: None,
                    },
                    symbolic_shape: vec![SymbolicDim::from("batch"), SymbolicDim::Known(14)],
                    coordinate_encoding: CoordinateEncoding::None,
                    normalizer: NormalizerSpec::None,
                    patching: None,
                },
            ];
            spec.outputs = vec![OutputPortSpec {
                id: "action_chunk".to_string(),
                name: "Predicted Action Chunk Trajectory".to_string(),
                space: TensorSpace::RigidBodyTrajectory {
                    horizon_steps: 16,
                    dof_per_step: 14,
                },
                symbolic_shape: vec![
                    SymbolicDim::from("batch"),
                    SymbolicDim::Known(16),
                    SymbolicDim::Known(14),
                ],
                constraint: StructuralConstraint::ContinuousRotation6D,
                loss: LossPairing::MSE,
            }];

            let mut nodes = Vec::new();
            nodes.push(NodeSpec {
                id: "proprio_trunk".to_string(),
                name: "Proprioception Linear Trunk".to_string(),
                operator: OperatorKind::Linear {
                    in_features: 14,
                    out_features: d_model,
                    bias: true,
                },
                precision: Some(precision),
                ..Default::default()
            });
            for i in 1..=depth {
                nodes.push(NodeSpec {
                    id: format!("action_chunk_attn_{i}"),
                    name: format!("Action Chunking Transformer Layer {i}"),
                    operator: OperatorKind::MHA {
                        num_heads: (d_model / 64).max(1),
                        d_k: 64,
                        d_v: 64,
                        dropout: 0.0,
                    },
                    precision: Some(precision),
                    ..Default::default()
                });
            }
            nodes.push(NodeSpec {
                id: "action_head".to_string(),
                name: "Action Chunking Readout Head".to_string(),
                operator: OperatorKind::Linear {
                    in_features: d_model,
                    out_features: 14,
                    bias: true,
                },
                precision: Some(PrecisionFormat::FP32),
                ..Default::default()
            });
            spec.graph.nodes = nodes;
        }
        ArchitectureArchetype::TimeSeriesForecaster => {
            spec.metadata.tags = vec![
                "time-series".to_string(),
                "xlstm".to_string(),
                "revin".to_string(),
                "crps".to_string(),
            ];
            spec.inputs = vec![InputPortSpec {
                id: "history".to_string(),
                name: "Historical Multivariate Time Series".to_string(),
                space: TensorSpace::TimeSeries {
                    channels: 32,
                    horizon: 96,
                    stationary: false,
                },
                symbolic_shape: vec![
                    SymbolicDim::from("batch"),
                    SymbolicDim::Known(96),
                    SymbolicDim::Known(32),
                ],
                coordinate_encoding: CoordinateEncoding::None,
                normalizer: NormalizerSpec::RevIN {
                    eps: 1e-5,
                    affine: true,
                },
                patching: None,
            }];
            spec.outputs = vec![OutputPortSpec {
                id: "forecast_quantiles".to_string(),
                name: "CRPS Probabilistic Forecast Quantiles".to_string(),
                space: TensorSpace::ContinuousVector {
                    dimension: 3,
                    range: None,
                },
                symbolic_shape: vec![
                    SymbolicDim::from("batch"),
                    SymbolicDim::Known(24),
                    SymbolicDim::Known(3),
                ],
                constraint: StructuralConstraint::QuantileCRPS_GPD {
                    quantiles: vec![0.1, 0.5, 0.9],
                },
                loss: LossPairing::QuantileCRPS,
            }];

            let mut nodes = Vec::new();
            nodes.push(NodeSpec {
                id: "patch_proj".to_string(),
                name: "Temporal Feature Projection".to_string(),
                operator: OperatorKind::Linear {
                    in_features: 32,
                    out_features: d_model,
                    bias: true,
                },
                precision: Some(precision),
                ..Default::default()
            });
            for i in 1..=depth {
                nodes.push(NodeSpec {
                    id: format!("xlstm_matrix_{i}"),
                    name: format!("xLSTM Matrix Memory Layer {i}"),
                    operator: OperatorKind::xLSTM {
                        d_model,
                        num_heads: 4,
                        variant: "mLSTM".to_string(),
                    },
                    precision: Some(precision),
                    ..Default::default()
                });
            }
            nodes.push(NodeSpec {
                id: "quantile_readout".to_string(),
                name: "Probabilistic Quantile Readout Head".to_string(),
                operator: OperatorKind::Linear {
                    in_features: d_model,
                    out_features: 3,
                    bias: true,
                },
                precision: Some(PrecisionFormat::FP32),
                ..Default::default()
            });
            spec.graph.nodes = nodes;
        }
        ArchitectureArchetype::TabularPriorNetwork => {
            spec.metadata.tags = vec![
                "tabular".to_string(),
                "tabpfn".to_string(),
                "kan".to_string(),
            ];
            spec.inputs = vec![InputPortSpec {
                id: "features".to_string(),
                name: "Multivariate Tabular Features".to_string(),
                space: TensorSpace::TabularFeatures {
                    num_continuous: 32,
                    num_categorical: 0,
                    skewness: None,
                },
                symbolic_shape: vec![SymbolicDim::from("batch"), SymbolicDim::Known(32)],
                coordinate_encoding: CoordinateEncoding::None,
                normalizer: NormalizerSpec::PLE { num_bins: 16 },
                patching: None,
            }];
            spec.outputs = vec![OutputPortSpec {
                id: "target_classes".to_string(),
                name: "Classification Target Logits".to_string(),
                space: TensorSpace::DiscreteTokens {
                    vocab_size: 10,
                    is_causal: false,
                },
                symbolic_shape: vec![SymbolicDim::from("batch"), SymbolicDim::Known(10)],
                constraint: StructuralConstraint::Monotonic {
                    positive_weights: true,
                },
                loss: LossPairing::CrossEntropy,
            }];

            let mut nodes = Vec::new();
            nodes.push(NodeSpec {
                id: "feature_linear".to_string(),
                name: "Feature Embedding Layer".to_string(),
                operator: OperatorKind::Linear {
                    in_features: 32,
                    out_features: d_model,
                    bias: true,
                },
                precision: Some(precision),
                ..Default::default()
            });
            for i in 1..=depth {
                nodes.push(NodeSpec {
                    id: format!("kan_spline_{i}"),
                    name: format!("Kolmogorov-Arnold Spline Layer {i}"),
                    operator: OperatorKind::KANLayer {
                        in_features: d_model,
                        out_features: d_model,
                        grid_size: 5,
                        spline_order: 3,
                    },
                    precision: Some(precision),
                    ..Default::default()
                });
            }
            nodes.push(NodeSpec {
                id: "class_head".to_string(),
                name: "Multi-Class Readout Head".to_string(),
                operator: OperatorKind::Linear {
                    in_features: d_model,
                    out_features: 10,
                    bias: true,
                },
                precision: Some(PrecisionFormat::FP32),
                ..Default::default()
            });
            spec.graph.nodes = nodes;
        }
        ArchitectureArchetype::DeepResidual => {
            spec.metadata.tags = vec![
                "deep-residual".to_string(),
                "swiglu".to_string(),
                "layernorm".to_string(),
            ];
            spec.inputs = vec![InputPortSpec {
                id: "x".to_string(),
                name: "Continuous Latent Vectors".to_string(),
                space: TensorSpace::ContinuousVector {
                    dimension: d_model,
                    range: None,
                },
                symbolic_shape: vec![SymbolicDim::from("batch"), SymbolicDim::Known(d_model)],
                coordinate_encoding: CoordinateEncoding::None,
                normalizer: NormalizerSpec::None,
                patching: None,
            }];
            spec.outputs = vec![OutputPortSpec {
                id: "y".to_string(),
                name: "Scalar Regression Target".to_string(),
                space: TensorSpace::ContinuousVector {
                    dimension: 1,
                    range: None,
                },
                symbolic_shape: vec![SymbolicDim::from("batch"), SymbolicDim::Known(1)],
                constraint: StructuralConstraint::DirectLinear,
                loss: LossPairing::MSE,
            }];

            let mut nodes = Vec::new();
            for i in 1..=depth {
                nodes.push(NodeSpec {
                    id: format!("ln_{i}"),
                    name: format!("LayerNorm {i}"),
                    operator: OperatorKind::LayerNorm {
                        dim: d_model,
                        eps: 1e-5,
                    },
                    precision: Some(precision),
                    ..Default::default()
                });
                nodes.push(NodeSpec {
                    id: format!("res_dense_{i}"),
                    name: format!("Dense Residual Block {i}"),
                    operator: OperatorKind::Linear {
                        in_features: d_model,
                        out_features: d_model,
                        bias: true,
                    },
                    precision: Some(precision),
                    ..Default::default()
                });
            }
            nodes.push(NodeSpec {
                id: "output_head".to_string(),
                name: "Scalar Projection Readout".to_string(),
                operator: OperatorKind::Linear {
                    in_features: d_model,
                    out_features: 1,
                    bias: true,
                },
                precision: Some(PrecisionFormat::FP32),
                ..Default::default()
            });
            spec.graph.nodes = nodes;
        }
        ArchitectureArchetype::DenseMoE => {
            spec.metadata.tags = vec!["moe".to_string(), "mla".to_string(), "sparse".to_string()];
            spec.inputs = vec![InputPortSpec {
                id: "input_ids".to_string(),
                name: "Input Token Sequence".to_string(),
                space: TensorSpace::DiscreteTokens {
                    vocab_size: 32000,
                    is_causal: true,
                },
                symbolic_shape: vec![SymbolicDim::from("batch"), SymbolicDim::from("seq_len")],
                coordinate_encoding: CoordinateEncoding::RoPE1D { base: 10000.0 },
                normalizer: NormalizerSpec::None,
                patching: None,
            }];
            spec.outputs = vec![OutputPortSpec {
                id: "logits".to_string(),
                name: "Sparse MoE Vocabulary Logits".to_string(),
                space: TensorSpace::DiscreteTokens {
                    vocab_size: 32000,
                    is_causal: true,
                },
                symbolic_shape: vec![
                    SymbolicDim::from("batch"),
                    SymbolicDim::from("seq_len"),
                    SymbolicDim::Known(32000),
                ],
                constraint: StructuralConstraint::DirectLinear,
                loss: LossPairing::CrossEntropy,
            }];

            let mut nodes = Vec::new();
            nodes.push(NodeSpec {
                id: "embed".to_string(),
                name: "Token Embedding Matrix".to_string(),
                operator: OperatorKind::Embedding {
                    num_embeddings: 32000,
                    embedding_dim: d_model,
                },
                precision: Some(precision),
                ..Default::default()
            });
            for i in 1..=depth {
                nodes.push(NodeSpec {
                    id: format!("mla_{i}"),
                    name: format!("Multi-Head Latent Attention {i}"),
                    operator: OperatorKind::MLA {
                        num_heads: (d_model / 64).max(1),
                        d_latent_kv: d_model / 2,
                        d_rope: 64,
                        d_v: 64,
                    },
                    precision: Some(precision),
                    ..Default::default()
                });
                nodes.push(NodeSpec {
                    id: format!("moe_{i}"),
                    name: format!("DeepSeek Sparse MoE Block {i}"),
                    operator: OperatorKind::DeepSeekMoE {
                        total_experts: 8,
                        routed_k: 2,
                        shared_experts: 1,
                        d_model,
                        d_ffn: d_model * 2,
                        aux_loss_free_bias: true,
                    },
                    precision: Some(precision),
                    ..Default::default()
                });
            }
            nodes.push(NodeSpec {
                id: "moe_final_norm".to_string(),
                name: "Final RMSNorm Layer".to_string(),
                operator: OperatorKind::RMSNorm {
                    dim: d_model,
                    eps: 1e-5,
                },
                precision: Some(precision),
                ..Default::default()
            });
            nodes.push(NodeSpec {
                id: "moe_lm_head".to_string(),
                name: "MoE Logits Projection Head".to_string(),
                operator: OperatorKind::Linear {
                    in_features: d_model,
                    out_features: 32000,
                    bias: false,
                },
                precision: Some(PrecisionFormat::FP32),
                ..Default::default()
            });
            spec.graph.nodes = nodes;
        }
        ArchitectureArchetype::SymbolicAstReasoning => {
            spec.metadata.tags = vec![
                "symbolic".to_string(),
                "ast".to_string(),
                "dag".to_string(),
                "tree_transformer".to_string(),
                "equality_saturation".to_string(),
            ];
            spec.inputs = vec![InputPortSpec {
                id: "ast_tokens".to_string(),
                name: "Canonical AST Operator Tokens".to_string(),
                space: TensorSpace::DiscreteTokens {
                    vocab_size: 1024,
                    is_causal: false,
                },
                symbolic_shape: vec![SymbolicDim::from("batch"), SymbolicDim::from("ast_nodes")],
                coordinate_encoding: CoordinateEncoding::None,
                normalizer: NormalizerSpec::None,
                patching: None,
            }];
            spec.outputs = vec![
                OutputPortSpec {
                    id: "node_embeddings".to_string(),
                    name: "Topological Node Embeddings".to_string(),
                    space: TensorSpace::ContinuousVector {
                        dimension: d_model,
                        range: None,
                    },
                    symbolic_shape: vec![
                        SymbolicDim::from("batch"),
                        SymbolicDim::from("ast_nodes"),
                        SymbolicDim::Known(d_model),
                    ],
                    constraint: StructuralConstraint::DirectLinear,
                    loss: LossPairing::InfoNCEContrastive,
                },
                OutputPortSpec {
                    id: "value_score".to_string(),
                    name: "MCTS State Complexity Value".to_string(),
                    space: TensorSpace::ContinuousVector {
                        dimension: 1,
                        range: Some((0.0, 1.0)),
                    },
                    symbolic_shape: vec![SymbolicDim::from("batch"), SymbolicDim::Known(1)],
                    constraint: StructuralConstraint::DirectLinear,
                    loss: LossPairing::MSE,
                },
            ];

            let mut nodes = Vec::new();
            nodes.push(NodeSpec {
                id: "op_embed".to_string(),
                name: "AST Operator & Variable Embedding".to_string(),
                operator: OperatorKind::Embedding {
                    num_embeddings: 1024,
                    embedding_dim: d_model,
                },
                precision: Some(precision),
                ..Default::default()
            });
            for i in 1..=depth {
                nodes.push(NodeSpec {
                    id: format!("tree_attn_{i}"),
                    name: format!("Tree-Relative Self-Attention Block {i}"),
                    operator: OperatorKind::MHA {
                        num_heads: (d_model / 64).max(1),
                        d_k: 64,
                        d_v: 64,
                        dropout: 0.0,
                    },
                    precision: Some(precision),
                    ..Default::default()
                });
                nodes.push(NodeSpec {
                    id: format!("dag_ffn_{i}"),
                    name: format!("DAG Operator SwiGLU FFN {i}"),
                    operator: OperatorKind::SwiGLU {
                        in_features: d_model,
                        hidden_features: d_model * 2,
                    },
                    precision: Some(precision),
                    ..Default::default()
                });
            }
            nodes.push(NodeSpec {
                id: "ast_final_norm".to_string(),
                name: "AST Final RMSNorm".to_string(),
                operator: OperatorKind::RMSNorm {
                    dim: d_model,
                    eps: 1e-5,
                },
                precision: Some(precision),
                ..Default::default()
            });
            nodes.push(NodeSpec {
                id: "value_head".to_string(),
                name: "MCTS Complexity Value Head".to_string(),
                operator: OperatorKind::Linear {
                    in_features: d_model,
                    out_features: 1,
                    bias: true,
                },
                precision: Some(PrecisionFormat::FP32),
                ..Default::default()
            });
            spec.graph.nodes = nodes;
        }
        ArchitectureArchetype::SuperSmoothLipschitz => {
            spec.metadata.tags = vec![
                "super_smooth".to_string(),
                "lipschitz_1".to_string(),
                "bjorck".to_string(),
                "sobolev_h2".to_string(),
            ];
            spec.inputs = vec![InputPortSpec {
                id: "coords".to_string(),
                name: "Continuous Spatiotemporal Coordinates [B, D]".to_string(),
                space: TensorSpace::ContinuousVector {
                    dimension: d_model,
                    range: Some((-1.0, 1.0)),
                },
                symbolic_shape: vec![SymbolicDim::from("batch"), SymbolicDim::Known(d_model)],
                coordinate_encoding: CoordinateEncoding::RandomFourierFeatures {
                    num_features: d_model / 2,
                    scale: 1.0,
                },
                normalizer: NormalizerSpec::None,
                patching: None,
            }];
            spec.outputs = vec![OutputPortSpec {
                id: "smooth_potential".to_string(),
                name: "C^∞ Super-Smooth Potential & Certified 1-Lipschitz Field".to_string(),
                space: TensorSpace::ContinuousVector {
                    dimension: d_model,
                    range: None,
                },
                symbolic_shape: vec![SymbolicDim::from("batch"), SymbolicDim::Known(d_model)],
                constraint: StructuralConstraint::LipschitzBounded {
                    max_lipschitz_k: 1.0,
                },
                loss: LossPairing::SobolevH2CurvaturePenalty,
            }];
            let mut nodes = Vec::new();
            for i in 1..=depth.max(1) {
                nodes.push(NodeSpec {
                    id: format!("bjorck_lipschitz_{i}"),
                    name: format!("Layer {i} Björck 1-Lipschitz & Sobolev H² Block"),
                    operator: OperatorKind::SuperSmoothLipschitzLayer {
                        dim: d_model,
                        bjorck_iters: 5,
                        sobolev_order: 2,
                        spectral_bound_k: 1.0,
                    },
                    precision: Some(precision),
                    ..Default::default()
                });
                nodes.push(NodeSpec {
                    id: format!("smooth_act_{i}"),
                    name: format!("Layer {i} C^∞ Softplus-β Convex Activation"),
                    operator: OperatorKind::Activation {
                        kind: crate::spec::operators::ActivationKind::SoftplusBeta,
                    },
                    precision: Some(precision),
                    ..Default::default()
                });
            }
            nodes.push(NodeSpec {
                id: "kan_smooth_spline_head".to_string(),
                name: "Cubic C² B-Spline KAN Smooth Readout".to_string(),
                operator: OperatorKind::KANLayer {
                    in_features: d_model,
                    out_features: d_model,
                    grid_size: 8,
                    spline_order: 3,
                },
                precision: Some(PrecisionFormat::FP32),
                ..Default::default()
            });
            spec.graph.nodes = nodes;
            crate::spec::training::TrainingPlanPreset::SuperSmoothSobolevLipschitzCertified
                .apply_to_spec(&mut spec, 1, Some(1024));
        }
        ArchitectureArchetype::LearnableQuantizationQAT => {
            spec.metadata.tags = vec![
                "learnable_quantization".to_string(),
                "lsq_plus".to_string(),
                "fracbits".to_string(),
                "mixed_bitwidth_qat".to_string(),
            ];
            spec.inputs = vec![InputPortSpec {
                id: "input_ids".to_string(),
                name: "Input Token Sequence".to_string(),
                space: TensorSpace::DiscreteTokens {
                    vocab_size: 32000,
                    is_causal: true,
                },
                symbolic_shape: vec![SymbolicDim::from("batch"), SymbolicDim::from("seq_len")],
                coordinate_encoding: CoordinateEncoding::RoPE1D { base: 10000.0 },
                normalizer: NormalizerSpec::None,
                patching: None,
            }];
            spec.outputs = vec![OutputPortSpec {
                id: "logits".to_string(),
                name: "BOPs-Constrained Quantized Logits".to_string(),
                space: TensorSpace::DiscreteTokens {
                    vocab_size: 32000,
                    is_causal: true,
                },
                symbolic_shape: vec![
                    SymbolicDim::from("batch"),
                    SymbolicDim::from("seq_len"),
                    SymbolicDim::Known(32000),
                ],
                constraint: StructuralConstraint::DirectLinear,
                loss: LossPairing::LearnableBitwidthBopsPenalty,
            }];
            let mut nodes = vec![NodeSpec {
                id: "embed".to_string(),
                name: "Token Embedding Matrix".to_string(),
                operator: OperatorKind::Embedding {
                    num_embeddings: 32000,
                    embedding_dim: d_model,
                },
                precision: Some(PrecisionFormat::BF16),
                ..Default::default()
            }];
            for i in 1..=depth.max(1) {
                nodes.push(NodeSpec {
                    id: format!("lsq_fracbits_{i}"),
                    name: format!("Layer {i} LSQ+ & FracBits Learnable Quantizer"),
                    operator: OperatorKind::LearnableQuantizerQAT {
                        dim: d_model,
                        init_bits: 3.5,
                        min_bits: 1.58,
                        max_bits: 8.0,
                        learn_step_size_and_clip: true,
                    },
                    precision: Some(precision),
                    ..Default::default()
                });
                nodes.push(NodeSpec {
                    id: format!("quant_gqa_{i}"),
                    name: format!("Layer {i} Mixed-Bitwidth GQA Attention"),
                    operator: OperatorKind::GQA {
                        num_q_heads: (d_model / 64).max(1),
                        num_kv_heads: (d_model / 128).max(1),
                        d_k: 64,
                    },
                    precision: Some(precision),
                    ..Default::default()
                });
                nodes.push(NodeSpec {
                    id: format!("quant_swiglu_{i}"),
                    name: format!("Layer {i} Mixed-Bitwidth SwiGLU MLP"),
                    operator: OperatorKind::SwiGLU {
                        in_features: d_model,
                        hidden_features: d_model * 2,
                    },
                    precision: Some(precision),
                    ..Default::default()
                });
            }
            nodes.push(NodeSpec {
                id: "lm_head".to_string(),
                name: "FP16 Protected Vocabulary Readout".to_string(),
                operator: OperatorKind::Linear {
                    in_features: d_model,
                    out_features: 32000,
                    bias: false,
                },
                precision: Some(PrecisionFormat::FP16),
                ..Default::default()
            });
            spec.graph.nodes = nodes;
            crate::spec::training::TrainingPlanPreset::LearnableQuantizationMixedBitwidthQat
                .apply_to_spec(&mut spec, 4, Some(4096));
        }
        ArchitectureArchetype::ZeroShotVoiceCloningTTS => {
            spec.metadata.tags = vec![
                "voice_cloning".to_string(),
                "zero_shot_tts".to_string(),
                "rvq_codec".to_string(),
                "ot_cfm".to_string(),
            ];
            spec.inputs = vec![InputPortSpec {
                id: "phoneme_ids".to_string(),
                name: "Phoneme & Prosody Sequence".to_string(),
                space: TensorSpace::DiscreteTokens {
                    vocab_size: 1024,
                    is_causal: false,
                },
                symbolic_shape: vec![SymbolicDim::from("batch"), SymbolicDim::from("seq_len")],
                coordinate_encoding: CoordinateEncoding::None,
                normalizer: NormalizerSpec::None,
                patching: None,
            }];
            spec.outputs = vec![OutputPortSpec {
                id: "cloned_speech_latents".to_string(),
                name: "Speaker-Cloned RVQ & Mel Flow Latents".to_string(),
                space: TensorSpace::ContinuousVector {
                    dimension: d_model,
                    range: None,
                },
                symbolic_shape: vec![
                    SymbolicDim::from("batch"),
                    SymbolicDim::from("seq_len"),
                    SymbolicDim::Known(d_model),
                ],
                constraint: StructuralConstraint::DirectLinear,
                loss: LossPairing::OptimalTransportConditionalFlowMatching,
            }];
            let mut nodes = vec![NodeSpec {
                id: "phoneme_embed".to_string(),
                name: "Phoneme Embedding Conformer".to_string(),
                operator: OperatorKind::Embedding {
                    num_embeddings: 1024,
                    embedding_dim: d_model,
                },
                precision: Some(precision),
                ..Default::default()
            }];
            for i in 1..=depth.max(1) {
                nodes.push(NodeSpec {
                    id: format!("voice_clone_codec_{i}"),
                    name: format!("Layer {i} Speaker-Prompt RVQ & Flow Synthesizer"),
                    operator: OperatorKind::VoiceCloningCodecTTS {
                        d_model,
                        num_rvq_codebooks: 8,
                        speaker_embed_dim: 256,
                    },
                    precision: Some(precision),
                    ..Default::default()
                });
            }
            nodes.push(NodeSpec {
                id: "snake_vocoder".to_string(),
                name: "Periodic Snake Harmonic Vocoder Activation".to_string(),
                operator: OperatorKind::Snake { frequency: 2.0 },
                precision: Some(precision),
                ..Default::default()
            });
            spec.graph.nodes = nodes;
            crate::spec::training::TrainingPlanPreset::ZeroShotVoiceCloningFlowTts.apply_to_spec(
                &mut spec,
                4,
                Some(2048),
            );
        }
        ArchitectureArchetype::RawBitByteMultiScaleLLM => {
            spec.metadata.tags = vec![
                "raw_bits_bytes".to_string(),
                "megabyte".to_string(),
                "mambabyte".to_string(),
                "tokenizer_free".to_string(),
            ];
            spec.inputs = vec![InputPortSpec {
                id: "raw_bytes".to_string(),
                name: "Raw Binary Bit/Byte Stream (0..255)".to_string(),
                space: TensorSpace::DiscreteTokens {
                    vocab_size: 256,
                    is_causal: true,
                },
                symbolic_shape: vec![SymbolicDim::from("batch"), SymbolicDim::from("seq_len")],
                coordinate_encoding: CoordinateEncoding::None,
                normalizer: NormalizerSpec::None,
                patching: None,
            }];
            spec.outputs = vec![OutputPortSpec {
                id: "next_byte_logits".to_string(),
                name: "Autoregressive Next-Byte Octet Logits".to_string(),
                space: TensorSpace::DiscreteTokens {
                    vocab_size: 256,
                    is_causal: true,
                },
                symbolic_shape: vec![
                    SymbolicDim::from("batch"),
                    SymbolicDim::from("seq_len"),
                    SymbolicDim::Known(256),
                ],
                constraint: StructuralConstraint::DirectLinear,
                loss: LossPairing::CrossEntropy,
            }];
            let mut nodes = vec![
                NodeSpec {
                    id: "byte_embed".to_string(),
                    name: "Raw Byte Octet Embedding (256)".to_string(),
                    operator: OperatorKind::Embedding {
                        num_embeddings: 256,
                        embedding_dim: 256,
                    },
                    precision: Some(precision),
                    ..Default::default()
                },
                NodeSpec {
                    id: "entropy_patcher".to_string(),
                    name: "Dynamic Shannon-Entropy Byte Patcher".to_string(),
                    operator: OperatorKind::EntropyBytePatcher {
                        d_byte: 256,
                        d_latent: d_model,
                        entropy_threshold: 1.75,
                    },
                    precision: Some(precision),
                    ..Default::default()
                },
            ];
            for i in 1..=depth.max(1) {
                nodes.push(NodeSpec {
                    id: format!("megabyte_multiscale_{i}"),
                    name: format!("Layer {i} MegaByte / bGPT Bit-Plane & Byte Hierarchy"),
                    operator: OperatorKind::RawBitByteMultiScale {
                        d_global: d_model,
                        d_local: d_model,
                        patch_bytes: 8,
                        bit_plane_decomposition: true,
                    },
                    precision: Some(precision),
                    ..Default::default()
                });
            }
            nodes.push(NodeSpec {
                id: "byte_readout".to_string(),
                name: "Next-Byte Octet Projection Head".to_string(),
                operator: OperatorKind::Linear {
                    in_features: d_model,
                    out_features: 256,
                    bias: false,
                },
                precision: Some(PrecisionFormat::FP32),
                ..Default::default()
            });
            spec.graph.nodes = nodes;
            crate::spec::training::TrainingPlanPreset::RawBitByteMultiScaleLlm.apply_to_spec(
                &mut spec,
                8,
                Some(8192),
            );
        }
        ArchitectureArchetype::AttentionFreeGatedMixer => {
            spec.metadata.tags = vec![
                "attention_is_not_all_you_need".to_string(),
                "gmlp".to_string(),
                "hyena".to_string(),
            ];
            spec.inputs = vec![InputPortSpec {
                id: "tokens".to_string(),
                name: "Input Token Sequence".to_string(),
                space: TensorSpace::DiscreteTokens {
                    vocab_size: 32000,
                    is_causal: true,
                },
                symbolic_shape: vec![SymbolicDim::from("batch"), SymbolicDim::from("seq_len")],
                coordinate_encoding: CoordinateEncoding::None,
                normalizer: NormalizerSpec::None,
                patching: None,
            }];
            spec.outputs = vec![OutputPortSpec {
                id: "logits".to_string(),
                name: "Rank-Preserved Vocabulary Logits".to_string(),
                space: TensorSpace::DiscreteTokens {
                    vocab_size: 32000,
                    is_causal: true,
                },
                symbolic_shape: vec![
                    SymbolicDim::from("batch"),
                    SymbolicDim::from("seq_len"),
                    SymbolicDim::Known(32000),
                ],
                constraint: StructuralConstraint::DirectLinear,
                loss: LossPairing::CrossEntropy,
            }];
            let mut nodes = vec![NodeSpec {
                id: "embed".to_string(),
                name: "Token Embedding".to_string(),
                operator: OperatorKind::Embedding {
                    num_embeddings: 32000,
                    embedding_dim: d_model,
                },
                precision: Some(precision),
                ..Default::default()
            }];
            for i in 1..=depth.max(1) {
                nodes.push(NodeSpec {
                    id: format!("gmlp_hyena_{i}"),
                    name: format!("Layer {i} gMLP Spatial Gating & Hyena Long Conv"),
                    operator: OperatorKind::AttentionFreeGatedMLP {
                        d_model,
                        seq_len: 1024,
                        filter_order: 2,
                    },
                    precision: Some(precision),
                    ..Default::default()
                });
            }
            nodes.push(NodeSpec {
                id: "lm_head".to_string(),
                name: "Vocabulary Readout Head".to_string(),
                operator: OperatorKind::Linear {
                    in_features: d_model,
                    out_features: 32000,
                    bias: false,
                },
                precision: Some(PrecisionFormat::FP32),
                ..Default::default()
            });
            spec.graph.nodes = nodes;
        }
        ArchitectureArchetype::SelfSupervisedFoundation => {
            spec.metadata.tags = vec![
                "self_supervised".to_string(),
                "dinov2".to_string(),
                "ijepa".to_string(),
                "vicreg".to_string(),
            ];
            spec.inputs = vec![InputPortSpec {
                id: "image_views".to_string(),
                name: "Multi-Crop Image Views".to_string(),
                space: TensorSpace::ImageRGB {
                    resolution: [224, 224],
                    channels: 3,
                },
                symbolic_shape: vec![
                    SymbolicDim::from("batch"),
                    SymbolicDim::Known(196),
                    SymbolicDim::Known(768),
                ],
                coordinate_encoding: CoordinateEncoding::RoPE2D_Dynamic { base: 10000.0 },
                normalizer: NormalizerSpec::ImageNetNormalize,
                patching: Some(PatchingSpec {
                    patch_size: [16, 16],
                    flatten_channels: true,
                }),
            }];
            spec.outputs = vec![OutputPortSpec {
                id: "ssl_embeddings".to_string(),
                name: "Collapse-Free Self-Supervised Embeddings".to_string(),
                space: TensorSpace::ContinuousVector {
                    dimension: d_model,
                    range: None,
                },
                symbolic_shape: vec![
                    SymbolicDim::from("batch"),
                    SymbolicDim::Known(196),
                    SymbolicDim::Known(d_model),
                ],
                constraint: StructuralConstraint::DirectLinear,
                loss: LossPairing::VICRegAntiCollapse,
            }];
            let mut nodes = vec![NodeSpec {
                id: "patch_proj".to_string(),
                name: "Patch Projection".to_string(),
                operator: OperatorKind::Linear {
                    in_features: 768,
                    out_features: d_model,
                    bias: true,
                },
                precision: Some(precision),
                ..Default::default()
            }];
            for i in 1..=depth.max(1) {
                nodes.push(NodeSpec {
                    id: format!("ssl_jepa_{i}"),
                    name: format!("Layer {i} I-JEPA / DINOv2 Latent Predictor"),
                    operator: OperatorKind::JEPAPredictor {
                        d_model,
                        predictor_dim: (d_model / 2).max(64),
                        num_heads: (d_model / 64).max(1),
                        ema_momentum: 0.996,
                    },
                    precision: Some(precision),
                    ..Default::default()
                });
            }
            spec.graph.nodes = nodes;
            crate::spec::training::TrainingPlanPreset::SelfSupervisedVisionDinoMae.apply_to_spec(
                &mut spec,
                8,
                Some(256),
            );
        }
        ArchitectureArchetype::TabularGBDTVimeSSL => {
            spec.metadata.tags = vec![
                "tabular_ssl".to_string(),
                "gbdt".to_string(),
                "vime".to_string(),
                "catboost".to_string(),
            ];
            spec.inputs = vec![InputPortSpec {
                id: "tabular_x".to_string(),
                name: "Heterogeneous Tabular Features".to_string(),
                space: TensorSpace::TabularFeatures {
                    num_continuous: 32,
                    num_categorical: 0,
                    skewness: None,
                },
                symbolic_shape: vec![SymbolicDim::from("batch"), SymbolicDim::Known(d_model)],
                coordinate_encoding: CoordinateEncoding::None,
                normalizer: NormalizerSpec::RevIN {
                    eps: 1e-5,
                    affine: true,
                },
                patching: None,
            }];
            spec.outputs = vec![OutputPortSpec {
                id: "boosted_risk".to_string(),
                name: "Calibrated Boosted Log-Odds".to_string(),
                space: TensorSpace::ContinuousVector {
                    dimension: 2,
                    range: None,
                },
                symbolic_shape: vec![SymbolicDim::from("batch"), SymbolicDim::Known(2)],
                constraint: StructuralConstraint::Monotonic {
                    positive_weights: true,
                },
                loss: LossPairing::Focal,
            }];
            let mut nodes = Vec::new();
            for i in 1..=depth.max(1) {
                nodes.push(NodeSpec {
                    id: format!("gbdt_vime_{i}"),
                    name: format!("Stage {i} Oblivious GBDT + VIME Self-Supervised Learner"),
                    operator: OperatorKind::TabularGBDTVimeSSL {
                        num_trees: 128,
                        tree_depth: 6,
                        vime_corruption_rate: 0.3,
                    },
                    precision: Some(PrecisionFormat::FP32),
                    ..Default::default()
                });
            }
            nodes.push(NodeSpec {
                id: "leaf_readout".to_string(),
                name: "Boosted Leaf Readout Head".to_string(),
                operator: OperatorKind::Linear {
                    in_features: d_model,
                    out_features: 2,
                    bias: true,
                },
                precision: Some(PrecisionFormat::FP32),
                ..Default::default()
            });
            spec.graph.nodes = nodes;
            crate::spec::training::TrainingPlanPreset::TabularGbdtVimeSelfSupervised.apply_to_spec(
                &mut spec,
                1,
                Some(128),
            );
        }
    }

    spec
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::{analyze_roofline, infer_shapes};

    #[test]
    fn test_all_archetypes_scaffold_and_analyze() {
        for &archetype in ArchitectureArchetype::ALL {
            let spec = scaffold_archetype(
                archetype,
                "test_model",
                archetype.display_name(),
                archetype.default_d_model(),
                archetype.default_depth(),
                PrecisionFormat::BF16,
            );

            assert_eq!(spec.metadata.model_id, "test_model");
            assert!(
                !spec.inputs.is_empty(),
                "Inputs must be declared for {:?}",
                archetype
            );
            assert!(
                !spec.outputs.is_empty(),
                "Outputs must be declared for {:?}",
                archetype
            );

            // Shape inference and roofline budgeting must compute without panicking
            let _shapes = infer_shapes(&spec);
            let roofline = analyze_roofline(&spec, 4, 128);

            if archetype == ArchitectureArchetype::Blank {
                // Blank canvas starts with a single connecting Linear layer.
                assert_eq!(spec.graph.nodes.len(), 1);
            } else {
                assert!(!spec.graph.nodes.is_empty());
                assert!(roofline.total_flops_per_token > 0);
            }

            // Serialization roundtrip
            let yaml = crate::format::aid_yaml::to_yaml_string(&spec).expect("YAML export failed");
            let restored =
                crate::format::aid_yaml::from_yaml_str(&yaml).expect("YAML import failed");
            assert_eq!(restored.metadata.model_id, spec.metadata.model_id);
            assert_eq!(restored.graph.nodes.len(), spec.graph.nodes.len());
        }
    }
}
