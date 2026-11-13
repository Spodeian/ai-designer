//! Architecture Design & Scaffolding Wizard: guided creation of AI models from scratch,
//! interactive step-by-step model building, architecture archetypes, or industry blueprints.

use crate::components::palette::{TEMPLATES, get_template_by_filename};
use crate::icons::bi;
use crate::studio_state::{CanvasMode, StudioState};
use eframe::egui::{self, Color32, CornerRadius, RichText, ScrollArea, Stroke, Ui, Vec2};
use shared::spec::training::TrainingPlanPreset;
use shared::spec::{
    AiDesignSpec, CoordinateEncoding, HaltingGate, HardwareTargetProfile, InputPortSpec,
    LossPairing, NodeSpec, NormPlacement, NormalizerSpec, OperatorKind, OutputPortSpec,
    PatchingSpec, StructuralConstraint, SymbolicDim, TensorSpace, WiringPattern,
};
use shared::{ArchitectureArchetype, PrecisionFormat, scaffold_archetype};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WizardTab {
    #[default]
    StepByStepBuilder,
    Archetypes,
    StarterTemplates,
    TrainingPlanner,
    BlankSlate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BuilderStep {
    #[default]
    ModalityAndTask,
    DimensionsAndIngress,
    BackboneCore,
    FeedForwardAndMoE,
    NormAndResidual,
    EgressAndLoss,
    HardwareAndReview,
}

impl BuilderStep {
    pub const ALL: &'static [BuilderStep] = &[
        Self::ModalityAndTask,
        Self::DimensionsAndIngress,
        Self::BackboneCore,
        Self::FeedForwardAndMoE,
        Self::NormAndResidual,
        Self::EgressAndLoss,
        Self::HardwareAndReview,
    ];

    pub fn number(&self) -> usize {
        match self {
            Self::ModalityAndTask => 1,
            Self::DimensionsAndIngress => 2,
            Self::BackboneCore => 3,
            Self::FeedForwardAndMoE => 4,
            Self::NormAndResidual => 5,
            Self::EgressAndLoss => 6,
            Self::HardwareAndReview => 7,
        }
    }

    pub fn title(&self) -> &'static str {
        match self {
            Self::ModalityAndTask => "Modality & Task",
            Self::DimensionsAndIngress => "Ingress & Pos-Encoding",
            Self::BackboneCore => "Core Operator",
            Self::FeedForwardAndMoE => "FFN & MoE",
            Self::NormAndResidual => "Norm, Residual & Depth",
            Self::EgressAndLoss => "Readout Head & Loss",
            Self::HardwareAndReview => "Hardware & Assembly",
        }
    }

    pub fn icon(&self) -> &'static str {
        match self {
            Self::ModalityAndTask => bi::JOURNAL_CODE,
            Self::DimensionsAndIngress => bi::BOX_ARROW_IN_RIGHT,
            Self::BackboneCore => bi::CPU,
            Self::FeedForwardAndMoE => bi::GRID_1X2,
            Self::NormAndResidual => bi::LAYERS,
            Self::EgressAndLoss => bi::BOX_ARROW_UP_RIGHT,
            Self::HardwareAndReview => bi::ROCKET_TAKEOFF,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModalityChoice {
    #[default]
    LanguageText,
    VisionImage,
    TimeSeries,
    Tabular,
    ScientificPDE,
    RoboticsAction,
}

impl ModalityChoice {
    pub fn label(&self) -> &'static str {
        match self {
            Self::LanguageText => "Language & Text (Tokens)",
            Self::VisionImage => "Computer Vision (Pixels/Patches)",
            Self::TimeSeries => "Time-Series & Forecasting",
            Self::Tabular => "Tabular & Structured Data",
            Self::ScientificPDE => "Scientific PDEs & Physical Grids",
            Self::RoboticsAction => "Robotics & Action Control",
        }
    }

    pub fn domain_name(&self) -> &'static str {
        match self {
            Self::LanguageText => "NaturalLanguage",
            Self::VisionImage => "ComputerVision",
            Self::TimeSeries => "TimeSeries",
            Self::Tabular => "TabularML",
            Self::ScientificPDE => "PhysicsInformedAI",
            Self::RoboticsAction => "RoboticsPolicy",
        }
    }

    pub fn icon(&self) -> &'static str {
        match self {
            Self::LanguageText => bi::FILE_EARMARK_TEXT,
            Self::VisionImage => bi::EYE,
            Self::TimeSeries => bi::GRAPH_UP_ARROW,
            Self::Tabular => bi::TABLE,
            Self::ScientificPDE => bi::ACTIVITY,
            Self::RoboticsAction => bi::ROBOT,
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::LanguageText => {
                "Discrete token sequences parsed into continuous representations for autoregressive or masked modeling."
            }
            Self::VisionImage => {
                "High-dimensional RGB/multispectral image tensors decomposed via 2D patchification."
            }
            Self::TimeSeries => {
                "Temporal multivariate series stabilized with reversible instance normalization (RevIN)."
            }
            Self::Tabular => {
                "Heterogeneous categorical and continuous numerical features mapped via quantile scalers."
            }
            Self::ScientificPDE => {
                "Infinite-dimensional function spaces on geometric meshes and spatial Fourier grids."
            }
            Self::RoboticsAction => {
                "Proprioceptive joint sensor states coupled with continuous multi-horizon action trajectories."
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TaskObjective {
    #[default]
    AutoregressiveLM,
    SequenceClassification,
    RegressionForecasting,
    DiffusionFlowMatching,
    ActionChunking,
}

impl TaskObjective {
    pub fn label(&self) -> &'static str {
        match self {
            Self::AutoregressiveLM => "Autoregressive Causal Generation",
            Self::SequenceClassification => "Multi-Class Sequence Classification",
            Self::RegressionForecasting => "Continuous Value Estimation / Forecasting",
            Self::DiffusionFlowMatching => "Generative Diffusion / Flow Matching",
            Self::ActionChunking => "Multi-Horizon Robotic Action Chunking",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PosEncodingChoice {
    #[default]
    RoPE,
    ALiBi,
    LearnedAbsolute,
    Fourier2D,
    None,
}

impl PosEncodingChoice {
    pub fn label(&self) -> &'static str {
        match self {
            Self::RoPE => "RoPE (Rotary Position Embeddings)",
            Self::ALiBi => "ALiBi (Attention with Linear Biases)",
            Self::LearnedAbsolute => "Learned Absolute Position Table",
            Self::Fourier2D => "2D Sinusoidal / Fourier Grid Encoding",
            Self::None => "None (Implicit via SSM or Causal Conv)",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackboneOperatorChoice {
    #[default]
    GQA,
    MHA,
    MLA,
    MambaSSM,
    FourierOperator,
    ConvNeXtSpatial,
}

impl BackboneOperatorChoice {
    pub fn label(&self) -> &'static str {
        match self {
            Self::GQA => "Grouped Query Attention (GQA)",
            Self::MHA => "Standard Multi-Head Attention (MHA)",
            Self::MLA => "Multi-Head Latent Attention (MLA)",
            Self::MambaSSM => "Mamba-2 Selective State-Space (SSM)",
            Self::FourierOperator => "Fourier Neural Spectral Operator (FNO)",
            Self::ConvNeXtSpatial => "ConvNeXt-V2 Depthwise Separable Conv",
        }
    }

    pub fn hardware_note(&self) -> &'static str {
        match self {
            Self::GQA => {
                "Ideal for long-context LLMs: cuts KV-cache VRAM by 4x to 8x during generation."
            }
            Self::MHA => {
                "General-purpose attention: highest expressivity for symmetric bi-directional encoders."
            }
            Self::MLA => {
                "DeepSeek V2/V3 innovation: compresses KV states into a low-rank latent vector for ultra-compact VRAM."
            }
            Self::MambaSSM => {
                "O(N) linear time complexity in sequence length; ideal for edge and continuous streaming."
            }
            Self::FourierOperator => {
                "Zero discretization error across resolutions; designed for physics and PDE modeling."
            }
            Self::ConvNeXtSpatial => {
                "Strong 2D inductive bias; fast execution on standard tensor cores without quadratic cost."
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FfnChoice {
    #[default]
    DenseSwiGLU,
    DenseGELU,
    SparseMoE,
}

impl FfnChoice {
    pub fn label(&self) -> &'static str {
        match self {
            Self::DenseSwiGLU => "Dense SwiGLU MLP (Gated SiLU)",
            Self::DenseGELU => "Dense GELU MLP (Standard)",
            Self::SparseMoE => "Sparse Mixture of Experts (MoE)",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NormChoice {
    #[default]
    RMSNorm,
    LayerNorm,
    GroupNorm,
}

impl NormChoice {
    pub fn label(&self) -> &'static str {
        match self {
            Self::RMSNorm => "RMSNorm (Root Mean Square - Fast & Modern)",
            Self::LayerNorm => "LayerNorm (Affine Mean & Variance)",
            Self::GroupNorm => "GroupNorm (Batch-Independent Channels)",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResidualChoice {
    #[default]
    PreLNResidual,
    PostLNResidual,
    LayerScaleResidual,
    ReversibleSplit,
    RecurrentDepth,
}

impl ResidualChoice {
    pub fn label(&self) -> &'static str {
        match self {
            Self::PreLNResidual => "Pre-LN Residual Highway (x + SubLayer(Norm(x)))",
            Self::PostLNResidual => "Post-LN Residual Highway (Norm(x + SubLayer(x)))",
            Self::LayerScaleResidual => {
                "LayerScale Pre-LN (Learnable diagonal scaling alpha = 1e-2)"
            }
            Self::ReversibleSplit => "Reversible Split-Stream (O(1) Activation Memory)",
            Self::RecurrentDepth => "Recurrent Depth Deliberation Loop (Adaptive PonderNet)",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputHeadChoice {
    #[default]
    LanguageModelHead,
    ClassificationHead,
    ContinuousRegressionHead,
    DiffusionNoiseHead,
    RoboticsActionHead,
}

impl OutputHeadChoice {
    pub fn label(&self) -> &'static str {
        match self {
            Self::LanguageModelHead => "Causal Language Modeling Head (Vocab Logits)",
            Self::ClassificationHead => "Multi-Class Classification Head (CrossEntropy)",
            Self::ContinuousRegressionHead => "Continuous Value Estimation Head (MSE/Huber)",
            Self::DiffusionNoiseHead => "Noise / Flow Vector Field Head (Flow Matching)",
            Self::RoboticsActionHead => "Robotics Action Chunking Head (Smooth L1)",
        }
    }
}

pub struct DesignWizardModal {
    pub is_open: bool,
    pub current_tab: WizardTab,

    // --- Interactive Step-by-Step Builder State ---
    pub builder_step: BuilderStep,
    pub step_modality: ModalityChoice,
    pub step_task: TaskObjective,
    pub step_pos_enc: PosEncodingChoice,
    pub rope_base: f64,
    pub step_backbone_op: BackboneOperatorChoice,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub d_k: usize,
    pub d_latent_kv: usize,
    pub ssm_d_state: usize,
    pub attention_dropout: f32,
    pub step_ffn: FfnChoice,
    pub ffn_expansion: f32,
    pub moe_num_experts: usize,
    pub moe_routed_k: usize,
    pub moe_shared_experts: usize,
    pub step_norm: NormChoice,
    pub step_residual: ResidualChoice,
    pub step_output_head: OutputHeadChoice,
    pub tie_word_embeddings: bool,
    pub vocab_size: usize,
    pub num_classes: usize,
    pub num_features: usize,
    pub image_resolution: usize,
    pub image_channels: usize,
    pub patch_size: usize,
    pub action_horizon: usize,
    pub action_dim: usize,
    pub loss_choice: LossPairing,
    pub target_hardware: HardwareTargetProfile,

    // --- Common Archetype & Legacy State ---
    pub selected_archetype: ArchitectureArchetype,
    pub selected_training_preset: TrainingPlanPreset,
    pub training_num_gpus: usize,
    pub training_seq_len: usize,
    pub model_id: String,
    pub display_name: String,
    pub d_model: usize,
    pub depth: usize,
    pub precision: PrecisionFormat,
    pub search_query: String,
    pub selected_template_index: Option<usize>,
}

impl Default for DesignWizardModal {
    fn default() -> Self {
        Self {
            is_open: false,
            current_tab: WizardTab::StepByStepBuilder,

            // Step-by-Step Defaults (Llama-3 modern layout)
            builder_step: BuilderStep::ModalityAndTask,
            step_modality: ModalityChoice::LanguageText,
            step_task: TaskObjective::AutoregressiveLM,
            step_pos_enc: PosEncodingChoice::RoPE,
            rope_base: 500000.0,
            step_backbone_op: BackboneOperatorChoice::GQA,
            num_heads: 8,
            num_kv_heads: 2,
            d_k: 64,
            d_latent_kv: 128,
            ssm_d_state: 16,
            attention_dropout: 0.0,
            step_ffn: FfnChoice::DenseSwiGLU,
            ffn_expansion: 2.667,
            moe_num_experts: 8,
            moe_routed_k: 2,
            moe_shared_experts: 1,
            step_norm: NormChoice::RMSNorm,
            step_residual: ResidualChoice::PreLNResidual,
            step_output_head: OutputHeadChoice::LanguageModelHead,
            tie_word_embeddings: true,
            vocab_size: 32000,
            num_classes: 1000,
            num_features: 64,
            image_resolution: 256,
            image_channels: 3,
            patch_size: 16,
            action_horizon: 16,
            action_dim: 7,
            loss_choice: LossPairing::CrossEntropy,
            target_hardware: HardwareTargetProfile::rtx_4090(),

            // Legacy & Archetype defaults
            selected_archetype: ArchitectureArchetype::GenerativeLLM,
            selected_training_preset: TrainingPlanPreset::FrontierLlm4StagePipeline,
            training_num_gpus: 8,
            training_seq_len: 4096,
            model_id: "custom_frontier_model".to_string(),
            display_name: "Custom Frontier Neural Architecture".to_string(),
            d_model: 512,
            depth: 6,
            precision: PrecisionFormat::BF16,
            search_query: String::new(),
            selected_template_index: None,
        }
    }
}

impl DesignWizardModal {
    pub fn open(&mut self) {
        self.is_open = true;
        self.current_tab = WizardTab::StepByStepBuilder;
        self.builder_step = BuilderStep::ModalityAndTask;
    }

    pub fn open_training_wizard(&mut self) {
        self.is_open = true;
        self.current_tab = WizardTab::TrainingPlanner;
    }

    pub fn apply_archetype_defaults(&mut self, archetype: ArchitectureArchetype) {
        self.selected_archetype = archetype;
        self.d_model = archetype.default_d_model();
        self.depth = archetype.default_depth();
        let slug = archetype
            .display_name()
            .to_lowercase()
            .replace([' ', '/', '(', ')', '-'], "_");
        let clean_slug = slug.trim_matches('_');
        self.model_id = format!("custom_{clean_slug}");
        self.display_name = format!("Custom {}", archetype.display_name());
    }

    // --- Preset Quick Fillers for the Step-by-Step Builder ---

    pub fn apply_step_preset_llama3(&mut self) {
        self.step_modality = ModalityChoice::LanguageText;
        self.step_task = TaskObjective::AutoregressiveLM;
        self.model_id = "llama3_gqa_transformer".to_string();
        self.display_name = "Llama-3 Style GQA Transformer".to_string();
        self.d_model = 768;
        self.depth = 8;
        self.step_pos_enc = PosEncodingChoice::RoPE;
        self.rope_base = 500000.0;
        self.step_backbone_op = BackboneOperatorChoice::GQA;
        self.num_heads = 12;
        self.num_kv_heads = 4;
        self.d_k = 64;
        self.step_ffn = FfnChoice::DenseSwiGLU;
        self.ffn_expansion = 2.667;
        self.step_norm = NormChoice::RMSNorm;
        self.step_residual = ResidualChoice::PreLNResidual;
        self.step_output_head = OutputHeadChoice::LanguageModelHead;
        self.tie_word_embeddings = true;
        self.precision = PrecisionFormat::BF16;
        self.target_hardware = HardwareTargetProfile::rtx_4090();
    }

    pub fn apply_step_preset_vit(&mut self) {
        self.step_modality = ModalityChoice::VisionImage;
        self.step_task = TaskObjective::SequenceClassification;
        self.model_id = "vit_patch_classifier".to_string();
        self.display_name = "Vision Transformer (ViT-Patch16)".to_string();
        self.d_model = 512;
        self.depth = 6;
        self.patch_size = 16;
        self.image_resolution = 256;
        self.image_channels = 3;
        self.step_pos_enc = PosEncodingChoice::Fourier2D;
        self.step_backbone_op = BackboneOperatorChoice::MHA;
        self.num_heads = 8;
        self.d_k = 64;
        self.step_ffn = FfnChoice::DenseGELU;
        self.ffn_expansion = 4.0;
        self.step_norm = NormChoice::LayerNorm;
        self.step_residual = ResidualChoice::PreLNResidual;
        self.step_output_head = OutputHeadChoice::ClassificationHead;
        self.num_classes = 1000;
        self.precision = PrecisionFormat::BF16;
        self.target_hardware = HardwareTargetProfile::rtx_4090();
    }

    pub fn apply_step_preset_mamba(&mut self) {
        self.step_modality = ModalityChoice::LanguageText;
        self.step_task = TaskObjective::AutoregressiveLM;
        self.model_id = "mamba2_selective_ssm".to_string();
        self.display_name = "Mamba-2 Linear Selective SSM".to_string();
        self.d_model = 512;
        self.depth = 8;
        self.step_pos_enc = PosEncodingChoice::None;
        self.step_backbone_op = BackboneOperatorChoice::MambaSSM;
        self.ssm_d_state = 16;
        self.step_ffn = FfnChoice::DenseSwiGLU;
        self.ffn_expansion = 2.0;
        self.step_norm = NormChoice::RMSNorm;
        self.step_residual = ResidualChoice::PreLNResidual;
        self.step_output_head = OutputHeadChoice::LanguageModelHead;
        self.tie_word_embeddings = true;
        self.precision = PrecisionFormat::BF16;
        self.target_hardware = HardwareTargetProfile::rtx_4090();
    }

    pub fn apply_step_preset_deepseek_moe(&mut self) {
        self.step_modality = ModalityChoice::LanguageText;
        self.step_task = TaskObjective::AutoregressiveLM;
        self.model_id = "deepseek_mla_moe".to_string();
        self.display_name = "DeepSeek-Style MLA + Sparse MoE".to_string();
        self.d_model = 1024;
        self.depth = 8;
        self.step_pos_enc = PosEncodingChoice::RoPE;
        self.rope_base = 10000.0;
        self.step_backbone_op = BackboneOperatorChoice::MLA;
        self.num_heads = 16;
        self.d_latent_kv = 256;
        self.d_k = 64;
        self.step_ffn = FfnChoice::SparseMoE;
        self.moe_num_experts = 8;
        self.moe_routed_k = 2;
        self.moe_shared_experts = 1;
        self.ffn_expansion = 2.0;
        self.step_norm = NormChoice::RMSNorm;
        self.step_residual = ResidualChoice::PreLNResidual;
        self.step_output_head = OutputHeadChoice::LanguageModelHead;
        self.tie_word_embeddings = true;
        self.precision = PrecisionFormat::FP8E4M3FN;
        self.target_hardware = HardwareTargetProfile::datacenter_h100();
    }

    pub fn apply_step_preset_scientific_pde(&mut self) {
        self.step_modality = ModalityChoice::ScientificPDE;
        self.step_task = TaskObjective::RegressionForecasting;
        self.model_id = "fno_spectral_pde_solver".to_string();
        self.display_name = "Fourier Neural Operator (FNO-2D)".to_string();
        self.d_model = 256;
        self.depth = 4;
        self.num_features = 2;
        self.step_pos_enc = PosEncodingChoice::Fourier2D;
        self.step_backbone_op = BackboneOperatorChoice::FourierOperator;
        self.step_ffn = FfnChoice::DenseGELU;
        self.ffn_expansion = 2.0;
        self.step_norm = NormChoice::LayerNorm;
        self.step_residual = ResidualChoice::PreLNResidual;
        self.step_output_head = OutputHeadChoice::ContinuousRegressionHead;
        self.loss_choice = LossPairing::MSE;
        self.precision = PrecisionFormat::FP32;
        self.target_hardware = HardwareTargetProfile::rtx_4090();
    }

    pub fn apply_step_preset_robotics_act(&mut self) {
        self.step_modality = ModalityChoice::RoboticsAction;
        self.step_task = TaskObjective::ActionChunking;
        self.model_id = "act_diffusion_policy".to_string();
        self.display_name = "Action Chunking Robotics Policy".to_string();
        self.d_model = 512;
        self.depth = 4;
        self.action_dim = 7;
        self.action_horizon = 16;
        self.step_pos_enc = PosEncodingChoice::LearnedAbsolute;
        self.step_backbone_op = BackboneOperatorChoice::GQA;
        self.num_heads = 8;
        self.num_kv_heads = 4;
        self.d_k = 64;
        self.step_ffn = FfnChoice::DenseGELU;
        self.ffn_expansion = 3.0;
        self.step_norm = NormChoice::LayerNorm;
        self.step_residual = ResidualChoice::PreLNResidual;
        self.step_output_head = OutputHeadChoice::RoboticsActionHead;
        self.precision = PrecisionFormat::FP16;
        self.target_hardware = HardwareTargetProfile::jetson_orin();
    }

    // --- Parameter & VRAM Calculations ---

    pub fn estimate_total_params(&self) -> usize {
        let mut total = 0;

        // Ingress
        match self.step_modality {
            ModalityChoice::LanguageText => {
                total += self.vocab_size * self.d_model;
            }
            ModalityChoice::VisionImage => {
                total += (self.image_channels * self.patch_size * self.patch_size) * self.d_model;
            }
            _ => {
                total += self.num_features.max(1) * self.d_model;
            }
        }

        // Positional Embedding
        if self.step_pos_enc == PosEncodingChoice::LearnedAbsolute {
            total += 4096 * self.d_model;
        }

        // Layer Blocks
        let hidden_dim = ((self.d_model as f32) * self.ffn_expansion) as usize;
        let mut layer_params = 0;

        // Pre-Attn & Pre-FFN Norms
        layer_params += self.d_model * 2;

        // Core Operator
        match self.step_backbone_op {
            BackboneOperatorChoice::GQA => {
                layer_params += self.d_model * (self.num_heads + 2 * self.num_kv_heads) * self.d_k;
                layer_params += (self.num_heads * self.d_k) * self.d_model;
            }
            BackboneOperatorChoice::MHA => {
                layer_params += 4 * self.d_model * self.d_model;
            }
            BackboneOperatorChoice::MLA => {
                layer_params += self.d_model * (self.num_heads * 64 + self.d_latent_kv);
                layer_params += self.d_latent_kv * (self.num_heads * 2 * self.d_k);
            }
            BackboneOperatorChoice::MambaSSM => {
                layer_params += self.d_model * (self.d_model * 2) * 2;
            }
            BackboneOperatorChoice::FourierOperator => {
                layer_params += 16 * 16 * self.d_model * self.d_model * 2;
            }
            BackboneOperatorChoice::ConvNeXtSpatial => {
                layer_params += 7 * 7 * self.d_model + self.d_model * (self.d_model * 4) * 2;
            }
        }

        // FFN or MoE
        match self.step_ffn {
            FfnChoice::DenseSwiGLU => {
                layer_params += 3 * self.d_model * hidden_dim;
            }
            FfnChoice::DenseGELU => {
                layer_params += 2 * self.d_model * hidden_dim;
            }
            FfnChoice::SparseMoE => {
                layer_params += self.d_model * self.moe_num_experts; // router
                layer_params += self.moe_num_experts * (3 * self.d_model * hidden_dim);
            }
        }

        total += layer_params * self.depth;

        // Final Norm
        total += self.d_model;

        // Output Head
        match self.step_output_head {
            OutputHeadChoice::LanguageModelHead => {
                if !self.tie_word_embeddings {
                    total += self.d_model * self.vocab_size;
                }
            }
            OutputHeadChoice::ClassificationHead => {
                total += self.d_model * self.num_classes;
            }
            OutputHeadChoice::ContinuousRegressionHead => {
                total += self.d_model * self.num_features.max(1);
            }
            OutputHeadChoice::RoboticsActionHead => {
                total += self.d_model * (self.action_horizon * self.action_dim);
            }
            OutputHeadChoice::DiffusionNoiseHead => {
                total += self.d_model * self.d_model;
            }
        }

        total
    }

    pub fn estimate_weight_memory_mb(&self) -> f64 {
        let params = self.estimate_total_params() as f64;
        let bytes_per_elem = match self.precision {
            PrecisionFormat::FP32 => 4.0,
            PrecisionFormat::FP16 | PrecisionFormat::BF16 => 2.0,
            PrecisionFormat::FP8E4M3FN | PrecisionFormat::FP8E5M2 => 1.0,
            PrecisionFormat::BitNet158b => 0.25,
            _ => 2.0,
        };
        (params * bytes_per_elem) / (1024.0 * 1024.0)
    }

    pub fn estimate_kv_cache_kb_per_token(&self) -> f64 {
        let kv_heads = match self.step_backbone_op {
            BackboneOperatorChoice::GQA => self.num_kv_heads,
            BackboneOperatorChoice::MHA => self.num_heads,
            BackboneOperatorChoice::MLA => 2, // ultra compressed
            _ => 0,
        };
        if kv_heads == 0 {
            return 0.0;
        }
        // 2 (key + value) * layers * kv_heads * d_k * 2 bytes (FP16/BF16)
        let bytes = 2.0 * (self.depth as f64) * (kv_heads as f64) * (self.d_k as f64) * 2.0;
        bytes / 1024.0
    }

    // --- Synthesis of the AiDesignSpec ---

    pub fn assemble_step_by_step_model(&self) -> AiDesignSpec {
        let mut spec = AiDesignSpec::new(&self.model_id, &self.display_name);
        spec.metadata.domain = self.step_modality.domain_name().to_string();
        spec.metadata.target_hardware = self.target_hardware.clone();
        spec.metadata.tags = vec![
            "step_by_step_wizard".to_string(),
            format!("{:?}", self.step_modality).to_lowercase(),
            format!("{:?}", self.step_backbone_op).to_lowercase(),
            format!("{:?}", self.step_ffn).to_lowercase(),
            format!("{:?}", self.step_norm).to_lowercase(),
            format!("{:?}", self.precision).to_lowercase(),
        ];

        // 1. Ingress Port
        match self.step_modality {
            ModalityChoice::LanguageText => {
                spec.inputs = vec![InputPortSpec {
                    id: "input_ids".to_string(),
                    name: "Token Sequence IDs".to_string(),
                    space: TensorSpace::DiscreteTokens {
                        vocab_size: self.vocab_size,
                        is_causal: true,
                    },
                    symbolic_shape: vec![SymbolicDim::from("batch"), SymbolicDim::from("seq_len")],
                    coordinate_encoding: match self.step_pos_enc {
                        PosEncodingChoice::RoPE => CoordinateEncoding::RoPE1D {
                            base: self.rope_base,
                        },
                        PosEncodingChoice::ALiBi => CoordinateEncoding::ALiBi,
                        PosEncodingChoice::Fourier2D => {
                            CoordinateEncoding::RoPE2D_Dynamic { base: 10000.0 }
                        }
                        _ => CoordinateEncoding::None,
                    },
                    normalizer: NormalizerSpec::None,
                    patching: None,
                }];
            }
            ModalityChoice::VisionImage => {
                spec.inputs = vec![InputPortSpec {
                    id: "image".to_string(),
                    name: "Input Image Pixels".to_string(),
                    space: TensorSpace::ImageRGB {
                        resolution: [self.image_resolution, self.image_resolution],
                        channels: self.image_channels,
                    },
                    symbolic_shape: vec![
                        SymbolicDim::from("batch"),
                        SymbolicDim::Known(self.image_channels),
                        SymbolicDim::Known(self.image_resolution),
                        SymbolicDim::Known(self.image_resolution),
                    ],
                    coordinate_encoding: CoordinateEncoding::RoPE2D_Dynamic { base: 10000.0 },
                    normalizer: NormalizerSpec::ImageNetNormalize,
                    patching: Some(PatchingSpec {
                        patch_size: [self.patch_size, self.patch_size],
                        flatten_channels: true,
                    }),
                }];
            }
            ModalityChoice::TimeSeries => {
                spec.inputs = vec![InputPortSpec {
                    id: "series".to_string(),
                    name: "Historical Multi-Variate Sequence".to_string(),
                    space: TensorSpace::ContinuousVector {
                        dimension: self.num_features,
                        range: None,
                    },
                    symbolic_shape: vec![
                        SymbolicDim::from("batch"),
                        SymbolicDim::from("seq_len"),
                        SymbolicDim::Known(self.num_features),
                    ],
                    coordinate_encoding: CoordinateEncoding::None,
                    normalizer: NormalizerSpec::RevIN {
                        eps: 1e-5,
                        affine: true,
                    },
                    patching: None,
                }];
            }
            ModalityChoice::Tabular => {
                spec.inputs = vec![InputPortSpec {
                    id: "tabular_features".to_string(),
                    name: "Tabular Feature Vector".to_string(),
                    space: TensorSpace::ContinuousVector {
                        dimension: self.num_features,
                        range: None,
                    },
                    symbolic_shape: vec![
                        SymbolicDim::from("batch"),
                        SymbolicDim::Known(self.num_features),
                    ],
                    coordinate_encoding: CoordinateEncoding::None,
                    normalizer: NormalizerSpec::PLE { num_bins: 16 },
                    patching: None,
                }];
            }
            ModalityChoice::ScientificPDE => {
                spec.inputs = vec![InputPortSpec {
                    id: "grid_function".to_string(),
                    name: "Spatial Discretization Initial State a(x)".to_string(),
                    space: TensorSpace::ContinuousVector {
                        dimension: self.d_model,
                        range: None,
                    },
                    symbolic_shape: vec![
                        SymbolicDim::from("batch"),
                        SymbolicDim::from("grid_x"),
                        SymbolicDim::from("grid_y"),
                        SymbolicDim::Known(self.num_features.max(1)),
                    ],
                    coordinate_encoding: CoordinateEncoding::RandomFourierFeatures {
                        num_features: 16,
                        scale: 1.0,
                    },
                    normalizer: NormalizerSpec::None,
                    patching: None,
                }];
            }
            ModalityChoice::RoboticsAction => {
                spec.inputs = vec![InputPortSpec {
                    id: "proprioception".to_string(),
                    name: "Joint Angles & Gripper State".to_string(),
                    space: TensorSpace::ContinuousVector {
                        dimension: self.action_dim,
                        range: None,
                    },
                    symbolic_shape: vec![
                        SymbolicDim::from("batch"),
                        SymbolicDim::Known(self.action_dim),
                    ],
                    coordinate_encoding: CoordinateEncoding::None,
                    normalizer: NormalizerSpec::MinMax {
                        min: -1.0,
                        max: 1.0,
                    },
                    patching: None,
                }];
            }
        }

        // 2. Output Port
        match self.step_output_head {
            OutputHeadChoice::LanguageModelHead => {
                spec.outputs = vec![OutputPortSpec {
                    id: "logits".to_string(),
                    name: "Vocabulary Logits Distribution".to_string(),
                    space: TensorSpace::DiscreteTokens {
                        vocab_size: self.vocab_size,
                        is_causal: true,
                    },
                    symbolic_shape: vec![
                        SymbolicDim::from("batch"),
                        SymbolicDim::from("seq_len"),
                        SymbolicDim::Known(self.vocab_size),
                    ],
                    constraint: StructuralConstraint::DirectLinear,
                    loss: LossPairing::CrossEntropy,
                }];
            }
            OutputHeadChoice::ClassificationHead => {
                spec.outputs = vec![OutputPortSpec {
                    id: "class_probabilities".to_string(),
                    name: "Class Probability Logits".to_string(),
                    space: TensorSpace::ContinuousVector {
                        dimension: self.num_classes,
                        range: None,
                    },
                    symbolic_shape: vec![
                        SymbolicDim::from("batch"),
                        SymbolicDim::Known(self.num_classes),
                    ],
                    constraint: StructuralConstraint::Sparsemax,
                    loss: LossPairing::CrossEntropy,
                }];
            }
            OutputHeadChoice::ContinuousRegressionHead => {
                spec.outputs = vec![OutputPortSpec {
                    id: "predictions".to_string(),
                    name: "Continuous Target Estimate".to_string(),
                    space: TensorSpace::ContinuousVector {
                        dimension: self.num_features.max(1),
                        range: None,
                    },
                    symbolic_shape: vec![
                        SymbolicDim::from("batch"),
                        SymbolicDim::Known(self.num_features.max(1)),
                    ],
                    constraint: StructuralConstraint::DirectLinear,
                    loss: self.loss_choice,
                }];
            }
            OutputHeadChoice::DiffusionNoiseHead => {
                spec.outputs = vec![OutputPortSpec {
                    id: "flow_velocity".to_string(),
                    name: "Target Vector Field / Noise Prediction".to_string(),
                    space: TensorSpace::ContinuousVector {
                        dimension: self.d_model,
                        range: None,
                    },
                    symbolic_shape: vec![
                        SymbolicDim::from("batch"),
                        SymbolicDim::Known(self.d_model),
                    ],
                    constraint: StructuralConstraint::DirectLinear,
                    loss: LossPairing::MSE,
                }];
            }
            OutputHeadChoice::RoboticsActionHead => {
                spec.outputs = vec![OutputPortSpec {
                    id: "action_chunk".to_string(),
                    name: "Multi-Step Action Chunk".to_string(),
                    space: TensorSpace::ContinuousVector {
                        dimension: self.action_dim,
                        range: Some((-1.0, 1.0)),
                    },
                    symbolic_shape: vec![
                        SymbolicDim::from("batch"),
                        SymbolicDim::Known(self.action_horizon),
                        SymbolicDim::Known(self.action_dim),
                    ],
                    constraint: StructuralConstraint::DirectLinear,
                    loss: LossPairing::SmoothL1,
                }];
            }
        }

        // 3. Topology Wiring
        spec.graph.topology = match self.step_residual {
            ResidualChoice::PreLNResidual => WiringPattern::ResidualStream {
                norm_placement: NormPlacement::PreLN,
                layer_scale_init: None,
            },
            ResidualChoice::PostLNResidual => WiringPattern::ResidualStream {
                norm_placement: NormPlacement::PostLN,
                layer_scale_init: None,
            },
            ResidualChoice::LayerScaleResidual => WiringPattern::ResidualStream {
                norm_placement: NormPlacement::PreLN,
                layer_scale_init: Some(1e-2),
            },
            ResidualChoice::ReversibleSplit => WiringPattern::ReversibleSplitStream,
            ResidualChoice::RecurrentDepth => WiringPattern::RecurrentDepthLoop {
                max_steps: self.depth,
                input_anchored: true,
                halting: HaltingGate::PonderNetKL { lambda_prior: 0.1 },
            },
        };

        // 4. Graph Nodes Assembly
        let mut nodes = Vec::new();

        // Ingress Embedding
        match self.step_modality {
            ModalityChoice::LanguageText => {
                nodes.push(NodeSpec {
                    id: "token_embed".to_string(),
                    name: "Token Embedding Table".to_string(),
                    operator: OperatorKind::Embedding {
                        num_embeddings: self.vocab_size,
                        embedding_dim: self.d_model,
                    },
                    precision: Some(self.precision),
                    ..Default::default()
                });
            }
            ModalityChoice::VisionImage => {
                nodes.push(NodeSpec {
                    id: "patch_embed".to_string(),
                    name: "Patch Linear Projection".to_string(),
                    operator: OperatorKind::Linear {
                        in_features: self.image_channels * self.patch_size * self.patch_size,
                        out_features: self.d_model,
                        bias: true,
                    },
                    precision: Some(self.precision),
                    ..Default::default()
                });
            }
            ModalityChoice::TimeSeries
            | ModalityChoice::Tabular
            | ModalityChoice::ScientificPDE => {
                nodes.push(NodeSpec {
                    id: "ingress_proj".to_string(),
                    name: "Feature Lifting Projection".to_string(),
                    operator: OperatorKind::Linear {
                        in_features: self.num_features.max(1),
                        out_features: self.d_model,
                        bias: true,
                    },
                    precision: Some(self.precision),
                    ..Default::default()
                });
            }
            ModalityChoice::RoboticsAction => {
                nodes.push(NodeSpec {
                    id: "proprio_proj".to_string(),
                    name: "Proprioception Lifting Projection".to_string(),
                    operator: OperatorKind::Linear {
                        in_features: self.action_dim,
                        out_features: self.d_model,
                        bias: true,
                    },
                    precision: Some(self.precision),
                    ..Default::default()
                });
            }
        }

        // Positional Embedding Node if learned
        if self.step_pos_enc == PosEncodingChoice::LearnedAbsolute {
            nodes.push(NodeSpec {
                id: "pos_embed".to_string(),
                name: "Learned Absolute Position Embedding".to_string(),
                operator: OperatorKind::Embedding {
                    num_embeddings: 4096,
                    embedding_dim: self.d_model,
                },
                precision: Some(self.precision),
                ..Default::default()
            });
        }

        // Backbone Blocks
        for i in 1..=self.depth {
            // Pre-Attn Norm
            match self.step_norm {
                NormChoice::RMSNorm | NormChoice::GroupNorm => {
                    nodes.push(NodeSpec {
                        id: format!("attn_norm_{i}"),
                        name: format!("Layer {i} Pre-Core RMSNorm"),
                        operator: OperatorKind::RMSNorm {
                            dim: self.d_model,
                            eps: 1e-5,
                        },
                        precision: Some(self.precision),
                        ..Default::default()
                    });
                }
                NormChoice::LayerNorm => {
                    nodes.push(NodeSpec {
                        id: format!("attn_norm_{i}"),
                        name: format!("Layer {i} Pre-Core LayerNorm"),
                        operator: OperatorKind::LayerNorm {
                            dim: self.d_model,
                            eps: 1e-5,
                        },
                        precision: Some(self.precision),
                        ..Default::default()
                    });
                }
            }

            // Core Backbone Operator
            match self.step_backbone_op {
                BackboneOperatorChoice::GQA => {
                    nodes.push(NodeSpec {
                        id: format!("gqa_{i}"),
                        name: format!("Layer {i} Grouped Query Attention"),
                        operator: OperatorKind::GQA {
                            num_q_heads: self.num_heads,
                            num_kv_heads: self.num_kv_heads,
                            d_k: self.d_k,
                        },
                        precision: Some(self.precision),
                        ..Default::default()
                    });
                }
                BackboneOperatorChoice::MHA => {
                    nodes.push(NodeSpec {
                        id: format!("mha_{i}"),
                        name: format!("Layer {i} Multi-Head Attention"),
                        operator: OperatorKind::MHA {
                            num_heads: self.num_heads,
                            d_k: self.d_k,
                            d_v: self.d_k,
                            dropout: self.attention_dropout,
                        },
                        precision: Some(self.precision),
                        ..Default::default()
                    });
                }
                BackboneOperatorChoice::MLA => {
                    nodes.push(NodeSpec {
                        id: format!("mla_{i}"),
                        name: format!("Layer {i} Multi-Head Latent Attention"),
                        operator: OperatorKind::MLA {
                            num_heads: self.num_heads,
                            d_latent_kv: self.d_latent_kv,
                            d_rope: 64,
                            d_v: self.d_k,
                        },
                        precision: Some(self.precision),
                        ..Default::default()
                    });
                }
                BackboneOperatorChoice::MambaSSM => {
                    nodes.push(NodeSpec {
                        id: format!("mamba_{i}"),
                        name: format!("Layer {i} Mamba-2 Selective SSM"),
                        operator: OperatorKind::Mamba2SSD {
                            d_model: self.d_model,
                            d_state: self.ssm_d_state,
                            d_conv: 4,
                            expand: 2,
                        },
                        precision: Some(self.precision),
                        ..Default::default()
                    });
                }
                BackboneOperatorChoice::FourierOperator => {
                    nodes.push(NodeSpec {
                        id: format!("fno_{i}"),
                        name: format!("Layer {i} Fourier Neural Spectral Operator"),
                        operator: OperatorKind::FourierNeuralOperator {
                            modes: vec![16, 16],
                            width: self.d_model,
                        },
                        precision: Some(self.precision),
                        ..Default::default()
                    });
                }
                BackboneOperatorChoice::ConvNeXtSpatial => {
                    nodes.push(NodeSpec {
                        id: format!("convnext_{i}"),
                        name: format!("Layer {i} ConvNeXt Depthwise Block"),
                        operator: OperatorKind::ConvNeXtV2Block {
                            dim: self.d_model,
                            kernel_size: 7,
                            mlp_ratio: 4.0,
                        },
                        precision: Some(self.precision),
                        ..Default::default()
                    });
                }
            }

            // Pre-FFN Norm
            match self.step_norm {
                NormChoice::RMSNorm | NormChoice::GroupNorm => {
                    nodes.push(NodeSpec {
                        id: format!("ffn_norm_{i}"),
                        name: format!("Layer {i} Pre-FFN RMSNorm"),
                        operator: OperatorKind::RMSNorm {
                            dim: self.d_model,
                            eps: 1e-5,
                        },
                        precision: Some(self.precision),
                        ..Default::default()
                    });
                }
                NormChoice::LayerNorm => {
                    nodes.push(NodeSpec {
                        id: format!("ffn_norm_{i}"),
                        name: format!("Layer {i} Pre-FFN LayerNorm"),
                        operator: OperatorKind::LayerNorm {
                            dim: self.d_model,
                            eps: 1e-5,
                        },
                        precision: Some(self.precision),
                        ..Default::default()
                    });
                }
            }

            // FFN or MoE
            let hidden_dim = ((self.d_model as f32) * self.ffn_expansion) as usize;
            match self.step_ffn {
                FfnChoice::DenseSwiGLU => {
                    nodes.push(NodeSpec {
                        id: format!("swiglu_{i}"),
                        name: format!("Layer {i} SwiGLU Gated Feed-Forward"),
                        operator: OperatorKind::SwiGLU {
                            in_features: self.d_model,
                            hidden_features: hidden_dim,
                        },
                        precision: Some(self.precision),
                        ..Default::default()
                    });
                }
                FfnChoice::DenseGELU => {
                    nodes.push(NodeSpec {
                        id: format!("gelu_ffn_{i}"),
                        name: format!("Layer {i} Dense GELU Feed-Forward"),
                        operator: OperatorKind::Linear {
                            in_features: self.d_model,
                            out_features: hidden_dim,
                            bias: true,
                        },
                        precision: Some(self.precision),
                        ..Default::default()
                    });
                }
                FfnChoice::SparseMoE => {
                    nodes.push(NodeSpec {
                        id: format!("moe_{i}"),
                        name: format!("Layer {i} Sparse Mixture of Experts"),
                        operator: OperatorKind::DeepSeekMoE {
                            total_experts: self.moe_num_experts,
                            routed_k: self.moe_routed_k,
                            shared_experts: self.moe_shared_experts,
                            d_model: self.d_model,
                            d_ffn: hidden_dim,
                            aux_loss_free_bias: true,
                        },
                        precision: Some(self.precision),
                        ..Default::default()
                    });
                }
            }
        }

        // Final Norm
        match self.step_norm {
            NormChoice::RMSNorm | NormChoice::GroupNorm => {
                nodes.push(NodeSpec {
                    id: "final_norm".to_string(),
                    name: "Final Output RMSNorm".to_string(),
                    operator: OperatorKind::RMSNorm {
                        dim: self.d_model,
                        eps: 1e-5,
                    },
                    precision: Some(self.precision),
                    ..Default::default()
                });
            }
            NormChoice::LayerNorm => {
                nodes.push(NodeSpec {
                    id: "final_norm".to_string(),
                    name: "Final Output LayerNorm".to_string(),
                    operator: OperatorKind::LayerNorm {
                        dim: self.d_model,
                        eps: 1e-5,
                    },
                    precision: Some(self.precision),
                    ..Default::default()
                });
            }
        }

        // Readout Head
        match self.step_output_head {
            OutputHeadChoice::LanguageModelHead => {
                nodes.push(NodeSpec {
                    id: "lm_head".to_string(),
                    name: "Language Model Logits Head".to_string(),
                    operator: OperatorKind::Linear {
                        in_features: self.d_model,
                        out_features: self.vocab_size,
                        bias: false,
                    },
                    precision: Some(PrecisionFormat::FP32),
                    ..Default::default()
                });
            }
            OutputHeadChoice::ClassificationHead => {
                nodes.push(NodeSpec {
                    id: "classifier_head".to_string(),
                    name: "Classification Projection Head".to_string(),
                    operator: OperatorKind::Linear {
                        in_features: self.d_model,
                        out_features: self.num_classes,
                        bias: true,
                    },
                    precision: Some(PrecisionFormat::FP32),
                    ..Default::default()
                });
            }
            OutputHeadChoice::ContinuousRegressionHead => {
                nodes.push(NodeSpec {
                    id: "regression_head".to_string(),
                    name: "Continuous Regression Head".to_string(),
                    operator: OperatorKind::Linear {
                        in_features: self.d_model,
                        out_features: self.num_features.max(1),
                        bias: true,
                    },
                    precision: Some(PrecisionFormat::FP32),
                    ..Default::default()
                });
            }
            OutputHeadChoice::DiffusionNoiseHead => {
                nodes.push(NodeSpec {
                    id: "flow_head".to_string(),
                    name: "Vector Field Velocity Head".to_string(),
                    operator: OperatorKind::Linear {
                        in_features: self.d_model,
                        out_features: self.d_model,
                        bias: true,
                    },
                    precision: Some(self.precision),
                    ..Default::default()
                });
            }
            OutputHeadChoice::RoboticsActionHead => {
                nodes.push(NodeSpec {
                    id: "action_chunk_head".to_string(),
                    name: "Action Chunk Readout Head".to_string(),
                    operator: OperatorKind::Linear {
                        in_features: self.d_model,
                        out_features: self.action_horizon * self.action_dim,
                        bias: true,
                    },
                    precision: Some(PrecisionFormat::FP32),
                    ..Default::default()
                });
            }
        }

        spec.graph.nodes = nodes;
        spec.training.auto_partition_parameters = true;
        spec.training.hygiene.auto_clip = true;
        spec
    }

    // --- Main Modal Renderer ---

    pub fn render(&mut self, ctx: &egui::Context, studio: &mut StudioState) {
        if !self.is_open {
            return;
        }

        let mut should_close = false;
        let mut should_create = false;

        egui::Window::new(
            RichText::new(format!(
                "{} Interactive AI Architecture & Model Design Wizard",
                bi::MAGIC
            ))
            .size(17.0)
            .strong(),
        )
        .collapsible(false)
        .resizable(true)
        .default_size(Vec2::new(960.0, 680.0))
        .anchor(egui::Align2::CENTER_CENTER, Vec2::ZERO)
        .show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new("Interactively build an optimal neural architecture step-by-step, choose industry blueprints, or auto-configure training plans:")
                        .size(12.5)
                        .color(Color32::from_rgb(195, 205, 220)),
                );
            });

            ui.add_space(8.0);
            ui.horizontal_wrapped(|ui| {
                if ui
                    .selectable_label(
                        self.current_tab == WizardTab::StepByStepBuilder,
                        format!("{} Guided Step-by-Step Builder", bi::SLIDERS),
                    )
                    .clicked()
                {
                    self.current_tab = WizardTab::StepByStepBuilder;
                }
                if ui
                    .selectable_label(
                        self.current_tab == WizardTab::Archetypes,
                        format!(
                            "{} Quick Archetypes ({})",
                            bi::ROBOT,
                            ArchitectureArchetype::ALL.len() - 1
                        ),
                    )
                    .clicked()
                {
                    self.current_tab = WizardTab::Archetypes;
                }
                if ui
                    .selectable_label(
                        self.current_tab == WizardTab::StarterTemplates,
                        format!(
                            "{} Industry Blueprints ({})",
                            bi::COLLECTION,
                            TEMPLATES.len()
                        ),
                    )
                    .clicked()
                {
                    self.current_tab = WizardTab::StarterTemplates;
                }
                if ui
                    .selectable_label(
                        self.current_tab == WizardTab::TrainingPlanner,
                        format!(
                            "{} Training Plan Wizard ({} Presets)",
                            bi::SPEEDOMETER2,
                            TrainingPlanPreset::ALL.len()
                        ),
                    )
                    .clicked()
                {
                    self.current_tab = WizardTab::TrainingPlanner;
                }
                if ui
                    .selectable_label(
                        self.current_tab == WizardTab::BlankSlate,
                        format!("{} Blank Slate", bi::FILE_EARMARK_TEXT),
                    )
                    .clicked()
                {
                    self.current_tab = WizardTab::BlankSlate;
                    self.apply_archetype_defaults(ArchitectureArchetype::Blank);
                }
            });

            ui.add_space(6.0);
            ui.separator();

            match self.current_tab {
                WizardTab::StepByStepBuilder => self.render_step_by_step_tab(ui, &mut should_create),
                WizardTab::Archetypes => self.render_archetypes_tab(ui),
                WizardTab::StarterTemplates => self.render_templates_tab(ui),
                WizardTab::TrainingPlanner => self.render_training_planner_tab(ui, studio),
                WizardTab::BlankSlate => self.render_blank_tab(ui),
            }

            ui.add_space(8.0);
            ui.separator();

            // Bottom Actions
            ui.horizontal(|ui| {
                if ui.button("Cancel").clicked() {
                    should_close = true;
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let action_label = match self.current_tab {
                        WizardTab::StepByStepBuilder => {
                            format!("{} Assemble & Scaffold Custom Model", bi::ROCKET_TAKEOFF)
                        }
                        WizardTab::Archetypes => {
                            format!("{} Create & Scaffold Archetype", bi::ROCKET_TAKEOFF)
                        }
                        WizardTab::StarterTemplates => {
                            format!("{} Load Starter Blueprint", bi::ROCKET_TAKEOFF)
                        }
                        WizardTab::TrainingPlanner => {
                            format!(
                                "{} Apply Training Plan & Open Planner",
                                bi::LIGHTNING_CHARGE_FILL
                            )
                        }
                        WizardTab::BlankSlate => {
                            format!("{} Create Blank Model Canvas", bi::ROCKET_TAKEOFF)
                        }
                    };
                    if ui
                        .button(
                            RichText::new(action_label)
                                .size(13.5)
                                .color(Color32::from_rgb(120, 235, 160))
                                .strong(),
                        )
                        .clicked()
                    {
                        should_create = true;
                    }
                });
            });
        });

        if should_create {
            match self.current_tab {
                WizardTab::StepByStepBuilder => {
                    let spec = self.assemble_step_by_step_model();
                    *studio = StudioState::new_with_starter(spec);
                }
                WizardTab::Archetypes => {
                    let spec = scaffold_archetype(
                        self.selected_archetype,
                        &self.model_id,
                        &self.display_name,
                        self.d_model,
                        self.depth,
                        self.precision,
                    );
                    *studio = StudioState::new_with_starter(spec);
                }
                WizardTab::StarterTemplates => {
                    if let Some(idx) = self.selected_template_index {
                        if let Some(tmpl) = TEMPLATES.get(idx) {
                            if let Some(content) = get_template_by_filename(tmpl.file_name) {
                                if let Ok(spec) = shared::format::aid_yaml::from_yaml_str(content) {
                                    *studio = StudioState::new_with_starter(spec);
                                }
                            }
                        }
                    }
                }
                WizardTab::TrainingPlanner => {
                    self.selected_training_preset.apply_to_spec(
                        &mut studio.spec,
                        self.training_num_gpus,
                        Some(self.training_seq_len),
                    );
                    studio.canvas_mode = CanvasMode::TrainingPlanner;
                    studio.refresh_analyses();
                }
                WizardTab::BlankSlate => {
                    let spec = scaffold_archetype(
                        ArchitectureArchetype::Blank,
                        &self.model_id,
                        &self.display_name,
                        self.d_model,
                        0,
                        self.precision,
                    );
                    *studio = StudioState::new_with_starter(spec);
                }
            }
            self.is_open = false;
        }

        if should_close {
            self.is_open = false;
        }
    }

    // --- Interactive Step-by-Step UI Panels ---

    fn render_step_by_step_tab(&mut self, ui: &mut Ui, should_create: &mut bool) {
        // Step Navigation Bar
        ui.horizontal_wrapped(|ui| {
            for &step in BuilderStep::ALL {
                let is_current = self.builder_step == step;
                let is_past = step.number() < self.builder_step.number();

                let pill_color = if is_current {
                    Color32::from_rgb(45, 75, 120)
                } else if is_past {
                    Color32::from_rgb(25, 45, 35)
                } else {
                    Color32::from_rgb(20, 24, 32)
                };

                let text_color = if is_current {
                    Color32::from_rgb(130, 205, 255)
                } else if is_past {
                    Color32::from_rgb(120, 220, 150)
                } else {
                    Color32::from_rgb(150, 160, 175)
                };

                let border_color = if is_current {
                    Color32::from_rgb(90, 160, 255)
                } else if is_past {
                    Color32::from_rgb(60, 140, 80)
                } else {
                    Color32::from_rgb(40, 45, 58)
                };

                let label_text = format!("{}. {} {}", step.number(), step.icon(), step.title());
                egui::Frame::group(ui.style())
                    .fill(pill_color)
                    .stroke(Stroke::new(1.0, border_color))
                    .corner_radius(CornerRadius::same(5))
                    .show(ui, |ui| {
                        if ui
                            .selectable_label(
                                is_current,
                                RichText::new(label_text)
                                    .size(11.5)
                                    .color(text_color)
                                    .strong(),
                            )
                            .clicked()
                        {
                            self.builder_step = step;
                        }
                    });
            }
        });

        ui.add_space(8.0);
        ui.separator();

        // 2-Column Split: Controls on Left (65%), Telemetry & Trace on Right (35%)
        ui.columns(2, |cols| {
            // LEFT COLUMN: Active Step Controls
            cols[0].vertical(|ui| {
                ScrollArea::vertical()
                    .id_salt("step_controls_scroll")
                    .max_height(420.0)
                    .show(ui, |ui| match self.builder_step {
                        BuilderStep::ModalityAndTask => self.render_step_modality(ui),
                        BuilderStep::DimensionsAndIngress => self.render_step_ingress(ui),
                        BuilderStep::BackboneCore => self.render_step_backbone(ui),
                        BuilderStep::FeedForwardAndMoE => self.render_step_ffn(ui),
                        BuilderStep::NormAndResidual => self.render_step_norm(ui),
                        BuilderStep::EgressAndLoss => self.render_step_egress(ui),
                        BuilderStep::HardwareAndReview => {
                            self.render_step_review(ui, should_create)
                        }
                    });

                ui.add_space(8.0);
                ui.separator();

                // Step Step Navigation Buttons
                ui.horizontal(|ui| {
                    let step_idx = self.builder_step.number();
                    if step_idx > 1 {
                        if ui
                            .button(format!("{} Previous Step", bi::ARROW_LEFT))
                            .clicked()
                        {
                            self.builder_step = BuilderStep::ALL[step_idx - 2];
                        }
                    }
                    if step_idx < BuilderStep::ALL.len() {
                        if ui
                            .button(
                                RichText::new(format!(
                                    "Next Step: {} {}",
                                    BuilderStep::ALL[step_idx].icon(),
                                    BuilderStep::ALL[step_idx].title()
                                ))
                                .color(Color32::from_rgb(110, 200, 255))
                                .strong(),
                            )
                            .clicked()
                        {
                            self.builder_step = BuilderStep::ALL[step_idx];
                        }
                    }
                });
            });

            // RIGHT COLUMN: Live Architecture Blueprint & Telemetry Dock
            cols[1].vertical(|ui| {
                self.render_live_telemetry_dock(ui, should_create);
            });
        });
    }

    fn render_step_modality(&mut self, ui: &mut Ui) {
        ui.label(
            RichText::new("Step 1: Problem Domain, Modality & Objective")
                .size(14.0)
                .strong()
                .color(Color32::from_rgb(100, 190, 255)),
        );
        ui.label(
            RichText::new("Select the input signal representation and high-level learning goal for the architecture:")
                .size(11.5)
                .color(Color32::from_rgb(180, 190, 205)),
        );
        ui.add_space(8.0);

        // Modality Cards
        for &modality in &[
            ModalityChoice::LanguageText,
            ModalityChoice::VisionImage,
            ModalityChoice::TimeSeries,
            ModalityChoice::Tabular,
            ModalityChoice::ScientificPDE,
            ModalityChoice::RoboticsAction,
        ] {
            let is_sel = self.step_modality == modality;
            let bg = if is_sel {
                Color32::from_rgb(32, 48, 72)
            } else {
                Color32::from_rgb(22, 26, 36)
            };
            let stroke = if is_sel {
                Color32::from_rgb(80, 160, 255)
            } else {
                Color32::from_rgb(45, 52, 68)
            };

            egui::Frame::group(ui.style())
                .fill(bg)
                .stroke(Stroke::new(if is_sel { 1.5 } else { 1.0 }, stroke))
                .corner_radius(CornerRadius::same(6))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        if ui
                            .selectable_label(
                                is_sel,
                                RichText::new(format!("{} {}", modality.icon(), modality.label()))
                                    .strong(),
                            )
                            .clicked()
                        {
                            self.step_modality = modality;
                            // Adapt defaults to chosen modality
                            match modality {
                                ModalityChoice::LanguageText => {
                                    self.step_task = TaskObjective::AutoregressiveLM;
                                    self.step_output_head = OutputHeadChoice::LanguageModelHead;
                                    self.step_pos_enc = PosEncodingChoice::RoPE;
                                }
                                ModalityChoice::VisionImage => {
                                    self.step_task = TaskObjective::SequenceClassification;
                                    self.step_output_head = OutputHeadChoice::ClassificationHead;
                                    self.step_pos_enc = PosEncodingChoice::Fourier2D;
                                }
                                ModalityChoice::TimeSeries | ModalityChoice::Tabular => {
                                    self.step_task = TaskObjective::RegressionForecasting;
                                    self.step_output_head =
                                        OutputHeadChoice::ContinuousRegressionHead;
                                    self.step_pos_enc = PosEncodingChoice::None;
                                }
                                ModalityChoice::ScientificPDE => {
                                    self.step_task = TaskObjective::RegressionForecasting;
                                    self.step_output_head =
                                        OutputHeadChoice::ContinuousRegressionHead;
                                    self.step_backbone_op = BackboneOperatorChoice::FourierOperator;
                                    self.step_pos_enc = PosEncodingChoice::Fourier2D;
                                }
                                ModalityChoice::RoboticsAction => {
                                    self.step_task = TaskObjective::ActionChunking;
                                    self.step_output_head = OutputHeadChoice::RoboticsActionHead;
                                    self.step_pos_enc = PosEncodingChoice::LearnedAbsolute;
                                }
                            }
                        }
                    });
                    ui.label(
                        RichText::new(modality.description())
                            .size(11.0)
                            .color(Color32::from_rgb(170, 180, 195)),
                    );
                });
            ui.add_space(4.0);
        }

        ui.add_space(10.0);
        ui.separator();
        ui.label(RichText::new("Model Identity & Naming:").strong());
        egui::Grid::new("step1_naming_grid")
            .num_columns(2)
            .spacing([12.0, 6.0])
            .show(ui, |ui| {
                ui.label("Model Slug ID:");
                ui.text_edit_singleline(&mut self.model_id);
                ui.end_row();

                ui.label("Display Name:");
                ui.text_edit_singleline(&mut self.display_name);
                ui.end_row();
            });

        ui.add_space(10.0);
        ui.separator();
        ui.label(RichText::new("Quick-Fill Best-Practice Presets:").strong());
        ui.horizontal_wrapped(|ui| {
            if ui.button("Llama-3 LLM").clicked() {
                self.apply_step_preset_llama3();
            }
            if ui.button("Vision DiT/ViT").clicked() {
                self.apply_step_preset_vit();
            }
            if ui.button("Mamba-2 SSM").clicked() {
                self.apply_step_preset_mamba();
            }
            if ui.button("DeepSeek MoE").clicked() {
                self.apply_step_preset_deepseek_moe();
            }
            if ui.button("Fourier FNO").clicked() {
                self.apply_step_preset_scientific_pde();
            }
            if ui.button("Robotics ACT").clicked() {
                self.apply_step_preset_robotics_act();
            }
        });
    }

    fn render_step_ingress(&mut self, ui: &mut Ui) {
        ui.label(
            RichText::new("Step 2: Hidden Dimensions & Ingress Encoding")
                .size(14.0)
                .strong()
                .color(Color32::from_rgb(100, 190, 255)),
        );
        ui.label(
            RichText::new("Configure the shared latent width (d_model), tokenization dictionary, and spatial/positional encodings:")
                .size(11.5)
                .color(Color32::from_rgb(180, 190, 205)),
        );
        ui.add_space(8.0);

        egui::Grid::new("step2_grid")
            .num_columns(2)
            .spacing([16.0, 8.0])
            .show(ui, |ui| {
                ui.label("Hidden Latent Dimension (d_model):");
                ui.horizontal(|ui| {
                    ui.add(
                        egui::DragValue::new(&mut self.d_model)
                            .range(64..=8192)
                            .speed(32),
                    );
                    for &dim in &[128, 256, 512, 768, 1024, 2048] {
                        if ui
                            .selectable_label(self.d_model == dim, format!("{dim}"))
                            .clicked()
                        {
                            self.d_model = dim;
                        }
                    }
                });
                ui.end_row();

                match self.step_modality {
                    ModalityChoice::LanguageText => {
                        ui.label("Vocabulary Size (Vocab Tokens):");
                        ui.horizontal(|ui| {
                            ui.add(
                                egui::DragValue::new(&mut self.vocab_size)
                                    .range(256..=256000)
                                    .speed(500),
                            );
                            for &v in &[32000, 50257, 128256] {
                                if ui
                                    .selectable_label(self.vocab_size == v, format!("{v}"))
                                    .clicked()
                                {
                                    self.vocab_size = v;
                                }
                            }
                        });
                        ui.end_row();
                    }
                    ModalityChoice::VisionImage => {
                        ui.label("Input Image Resolution (H x W):");
                        ui.horizontal(|ui| {
                            ui.add(
                                egui::DragValue::new(&mut self.image_resolution)
                                    .range(64..=2048)
                                    .speed(32),
                            );
                            for &res in &[128, 256, 512] {
                                if ui
                                    .selectable_label(
                                        self.image_resolution == res,
                                        format!("{res}px"),
                                    )
                                    .clicked()
                                {
                                    self.image_resolution = res;
                                }
                            }
                        });
                        ui.end_row();

                        ui.label("Square Patch Dimension (P x P):");
                        ui.horizontal(|ui| {
                            ui.add(
                                egui::DragValue::new(&mut self.patch_size)
                                    .range(2..=64)
                                    .speed(2),
                            );
                            for &p in &[8, 14, 16] {
                                if ui
                                    .selectable_label(self.patch_size == p, format!("{p}x{p}"))
                                    .clicked()
                                {
                                    self.patch_size = p;
                                }
                            }
                        });
                        ui.end_row();
                    }
                    ModalityChoice::TimeSeries
                    | ModalityChoice::Tabular
                    | ModalityChoice::ScientificPDE => {
                        ui.label("Input Feature Dimensionality:");
                        ui.add(
                            egui::DragValue::new(&mut self.num_features)
                                .range(1..=4096)
                                .speed(1),
                        );
                        ui.end_row();
                    }
                    ModalityChoice::RoboticsAction => {
                        ui.label("Proprioceptive Sensor Dimension:");
                        ui.add(
                            egui::DragValue::new(&mut self.action_dim)
                                .range(1..=64)
                                .speed(1),
                        );
                        ui.end_row();
                    }
                }

                ui.label("Coordinate & Positional Encoding:");
                egui::ComboBox::from_id_salt("pos_enc_combo")
                    .selected_text(self.step_pos_enc.label())
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.step_pos_enc,
                            PosEncodingChoice::RoPE,
                            PosEncodingChoice::RoPE.label(),
                        );
                        ui.selectable_value(
                            &mut self.step_pos_enc,
                            PosEncodingChoice::ALiBi,
                            PosEncodingChoice::ALiBi.label(),
                        );
                        ui.selectable_value(
                            &mut self.step_pos_enc,
                            PosEncodingChoice::LearnedAbsolute,
                            PosEncodingChoice::LearnedAbsolute.label(),
                        );
                        ui.selectable_value(
                            &mut self.step_pos_enc,
                            PosEncodingChoice::Fourier2D,
                            PosEncodingChoice::Fourier2D.label(),
                        );
                        ui.selectable_value(
                            &mut self.step_pos_enc,
                            PosEncodingChoice::None,
                            PosEncodingChoice::None.label(),
                        );
                    });
                ui.end_row();

                if self.step_pos_enc == PosEncodingChoice::RoPE {
                    ui.label("RoPE Base Frequency (theta):");
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::DragValue::new(&mut self.rope_base)
                                .range(1000.0..=10000000.0)
                                .speed(1000.0),
                        );
                        for &b in &[10000.0, 500000.0, 1000000.0] {
                            if ui
                                .selectable_label(self.rope_base == b, format!("{b:.0}"))
                                .clicked()
                            {
                                self.rope_base = b;
                            }
                        }
                    });
                    ui.end_row();
                }
            });
    }

    fn render_step_backbone(&mut self, ui: &mut Ui) {
        ui.label(
            RichText::new("Step 3: Core Computational Backbone Operator")
                .size(14.0)
                .strong()
                .color(Color32::from_rgb(100, 190, 255)),
        );
        ui.label(
            RichText::new("Choose the mathematical operator that processes representation mixing across sequences, grids, or states:")
                .size(11.5)
                .color(Color32::from_rgb(180, 190, 205)),
        );
        ui.add_space(8.0);

        for &op in &[
            BackboneOperatorChoice::GQA,
            BackboneOperatorChoice::MHA,
            BackboneOperatorChoice::MLA,
            BackboneOperatorChoice::MambaSSM,
            BackboneOperatorChoice::FourierOperator,
            BackboneOperatorChoice::ConvNeXtSpatial,
        ] {
            let is_sel = self.step_backbone_op == op;
            let bg = if is_sel {
                Color32::from_rgb(32, 48, 72)
            } else {
                Color32::from_rgb(22, 26, 36)
            };
            let stroke = if is_sel {
                Color32::from_rgb(80, 160, 255)
            } else {
                Color32::from_rgb(45, 52, 68)
            };

            egui::Frame::group(ui.style())
                .fill(bg)
                .stroke(Stroke::new(if is_sel { 1.5 } else { 1.0 }, stroke))
                .corner_radius(CornerRadius::same(6))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        if ui
                            .selectable_label(is_sel, RichText::new(op.label()).strong())
                            .clicked()
                        {
                            self.step_backbone_op = op;
                        }
                    });
                    ui.label(
                        RichText::new(op.hardware_note())
                            .size(11.0)
                            .color(Color32::from_rgb(165, 195, 230)),
                    );
                });
            ui.add_space(4.0);
        }

        ui.add_space(8.0);
        ui.separator();
        ui.label(RichText::new("Operator Hyperparameters:").strong());

        egui::Grid::new("step3_grid")
            .num_columns(2)
            .spacing([16.0, 8.0])
            .show(ui, |ui| match self.step_backbone_op {
                BackboneOperatorChoice::GQA => {
                    ui.label("Query Heads (num_q_heads):");
                    ui.add(
                        egui::DragValue::new(&mut self.num_heads)
                            .range(1..=64)
                            .speed(1),
                    );
                    ui.end_row();

                    ui.label("Key/Value Heads (num_kv_heads):");
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::DragValue::new(&mut self.num_kv_heads)
                                .range(1..=32)
                                .speed(1),
                        );
                        ui.label(format!(
                            "(Compression ratio: {}:1)",
                            self.num_heads / self.num_kv_heads.max(1)
                        ));
                    });
                    ui.end_row();

                    ui.label("Head Dimension (d_k):");
                    ui.add(
                        egui::DragValue::new(&mut self.d_k)
                            .range(16..=256)
                            .speed(16),
                    );
                    ui.end_row();
                }
                BackboneOperatorChoice::MHA => {
                    ui.label("Attention Heads (num_heads):");
                    ui.add(
                        egui::DragValue::new(&mut self.num_heads)
                            .range(1..=64)
                            .speed(1),
                    );
                    ui.end_row();

                    ui.label("Head Dimension (d_k):");
                    ui.add(
                        egui::DragValue::new(&mut self.d_k)
                            .range(16..=256)
                            .speed(16),
                    );
                    ui.end_row();
                }
                BackboneOperatorChoice::MLA => {
                    ui.label("Attention Heads (num_heads):");
                    ui.add(
                        egui::DragValue::new(&mut self.num_heads)
                            .range(1..=64)
                            .speed(1),
                    );
                    ui.end_row();

                    ui.label("Latent KV Dimension (d_latent_kv):");
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::DragValue::new(&mut self.d_latent_kv)
                                .range(32..=1024)
                                .speed(32),
                        );
                        ui.label("(DeepSeek low-rank KV compression bottleneck)");
                    });
                    ui.end_row();
                }
                BackboneOperatorChoice::MambaSSM => {
                    ui.label("SSM State Dimension (d_state):");
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::DragValue::new(&mut self.ssm_d_state)
                                .range(8..=256)
                                .speed(8),
                        );
                        ui.label("(Controls linear recurrent capacity)");
                    });
                    ui.end_row();
                }
                _ => {}
            });
    }

    fn render_step_ffn(&mut self, ui: &mut Ui) {
        ui.label(
            RichText::new("Step 4: Feed-Forward (FFN) & Mixture of Experts (MoE)")
                .size(14.0)
                .strong()
                .color(Color32::from_rgb(100, 190, 255)),
        );
        ui.label(
            RichText::new("Configure the token-wise channel-mixing blocks, nonlinear activations, or sparse conditional expert routing:")
                .size(11.5)
                .color(Color32::from_rgb(180, 190, 205)),
        );
        ui.add_space(8.0);

        for &ffn in &[
            FfnChoice::DenseSwiGLU,
            FfnChoice::DenseGELU,
            FfnChoice::SparseMoE,
        ] {
            let is_sel = self.step_ffn == ffn;
            let bg = if is_sel {
                Color32::from_rgb(32, 48, 72)
            } else {
                Color32::from_rgb(22, 26, 36)
            };
            let stroke = if is_sel {
                Color32::from_rgb(80, 160, 255)
            } else {
                Color32::from_rgb(45, 52, 68)
            };

            egui::Frame::group(ui.style())
                .fill(bg)
                .stroke(Stroke::new(if is_sel { 1.5 } else { 1.0 }, stroke))
                .corner_radius(CornerRadius::same(6))
                .show(ui, |ui| {
                    if ui
                        .selectable_label(is_sel, RichText::new(ffn.label()).strong())
                        .clicked()
                    {
                        self.step_ffn = ffn;
                    }
                });
            ui.add_space(4.0);
        }

        ui.add_space(8.0);
        ui.separator();
        egui::Grid::new("step4_grid")
            .num_columns(2)
            .spacing([16.0, 8.0])
            .show(ui, |ui| {
                ui.label("FFN Hidden Expansion Ratio:");
                ui.horizontal(|ui| {
                    ui.add(
                        egui::DragValue::new(&mut self.ffn_expansion)
                            .range(1.0..=8.0)
                            .speed(0.1),
                    );
                    let hidden_dim = ((self.d_model as f32) * self.ffn_expansion) as usize;
                    ui.label(format!("(Intermediate width: {hidden_dim} channels)"));
                });
                ui.end_row();

                if self.step_ffn == FfnChoice::SparseMoE {
                    ui.label("Total Expert Networks (total_experts):");
                    ui.add(
                        egui::DragValue::new(&mut self.moe_num_experts)
                            .range(2..=256)
                            .speed(2),
                    );
                    ui.end_row();

                    ui.label("Active Routed Experts (top_k):");
                    ui.add(
                        egui::DragValue::new(&mut self.moe_routed_k)
                            .range(1..=16)
                            .speed(1),
                    );
                    ui.end_row();

                    ui.label("Shared Experts (always active):");
                    ui.add(
                        egui::DragValue::new(&mut self.moe_shared_experts)
                            .range(0..=4)
                            .speed(1),
                    );
                    ui.end_row();
                }
            });
    }

    fn render_step_norm(&mut self, ui: &mut Ui) {
        ui.label(
            RichText::new("Step 5: Normalization, Residual Highways & Depth")
                .size(14.0)
                .strong()
                .color(Color32::from_rgb(100, 190, 255)),
        );
        ui.label(
            RichText::new("Select layer normalization type, residual connection topology, and the total stack depth:")
                .size(11.5)
                .color(Color32::from_rgb(180, 190, 205)),
        );
        ui.add_space(8.0);

        egui::Grid::new("step5_grid")
            .num_columns(2)
            .spacing([16.0, 8.0])
            .show(ui, |ui| {
                ui.label("Stack Depth (Number of Backbone Blocks):");
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut self.depth).range(1..=64).speed(1));
                    for &d in &[2, 4, 6, 8, 12, 16, 24, 32] {
                        if ui
                            .selectable_label(self.depth == d, format!("{d}x"))
                            .clicked()
                        {
                            self.depth = d;
                        }
                    }
                });
                ui.end_row();

                ui.label("Layer Normalization Primitive:");
                egui::ComboBox::from_id_salt("norm_choice_combo")
                    .selected_text(self.step_norm.label())
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.step_norm,
                            NormChoice::RMSNorm,
                            NormChoice::RMSNorm.label(),
                        );
                        ui.selectable_value(
                            &mut self.step_norm,
                            NormChoice::LayerNorm,
                            NormChoice::LayerNorm.label(),
                        );
                        ui.selectable_value(
                            &mut self.step_norm,
                            NormChoice::GroupNorm,
                            NormChoice::GroupNorm.label(),
                        );
                    });
                ui.end_row();

                ui.label("Residual Highway Wiring Topology:");
                egui::ComboBox::from_id_salt("residual_choice_combo")
                    .selected_text(self.step_residual.label())
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.step_residual,
                            ResidualChoice::PreLNResidual,
                            ResidualChoice::PreLNResidual.label(),
                        );
                        ui.selectable_value(
                            &mut self.step_residual,
                            ResidualChoice::PostLNResidual,
                            ResidualChoice::PostLNResidual.label(),
                        );
                        ui.selectable_value(
                            &mut self.step_residual,
                            ResidualChoice::LayerScaleResidual,
                            ResidualChoice::LayerScaleResidual.label(),
                        );
                        ui.selectable_value(
                            &mut self.step_residual,
                            ResidualChoice::ReversibleSplit,
                            ResidualChoice::ReversibleSplit.label(),
                        );
                        ui.selectable_value(
                            &mut self.step_residual,
                            ResidualChoice::RecurrentDepth,
                            ResidualChoice::RecurrentDepth.label(),
                        );
                    });
                ui.end_row();
            });
    }

    fn render_step_egress(&mut self, ui: &mut Ui) {
        ui.label(
            RichText::new("Step 6: Readout Head, Task Objective & Loss")
                .size(14.0)
                .strong()
                .color(Color32::from_rgb(100, 190, 255)),
        );
        ui.label(
            RichText::new("Specify the output projection format, loss function pairing, and structural task constraints:")
                .size(11.5)
                .color(Color32::from_rgb(180, 190, 205)),
        );
        ui.add_space(8.0);

        egui::Grid::new("step6_grid")
            .num_columns(2)
            .spacing([16.0, 8.0])
            .show(ui, |ui| {
                ui.label("Task Output Head Format:");
                egui::ComboBox::from_id_salt("head_combo")
                    .selected_text(self.step_output_head.label())
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.step_output_head,
                            OutputHeadChoice::LanguageModelHead,
                            OutputHeadChoice::LanguageModelHead.label(),
                        );
                        ui.selectable_value(
                            &mut self.step_output_head,
                            OutputHeadChoice::ClassificationHead,
                            OutputHeadChoice::ClassificationHead.label(),
                        );
                        ui.selectable_value(
                            &mut self.step_output_head,
                            OutputHeadChoice::ContinuousRegressionHead,
                            OutputHeadChoice::ContinuousRegressionHead.label(),
                        );
                        ui.selectable_value(
                            &mut self.step_output_head,
                            OutputHeadChoice::DiffusionNoiseHead,
                            OutputHeadChoice::DiffusionNoiseHead.label(),
                        );
                        ui.selectable_value(
                            &mut self.step_output_head,
                            OutputHeadChoice::RoboticsActionHead,
                            OutputHeadChoice::RoboticsActionHead.label(),
                        );
                    });
                ui.end_row();

                if self.step_output_head == OutputHeadChoice::LanguageModelHead {
                    ui.label("Weight Tying with Input Embedding:");
                    ui.checkbox(
                        &mut self.tie_word_embeddings,
                        "Share W_in and W_out weights (saves VRAM)",
                    );
                    ui.end_row();
                }

                if self.step_output_head == OutputHeadChoice::ClassificationHead {
                    ui.label("Target Category Count:");
                    ui.add(
                        egui::DragValue::new(&mut self.num_classes)
                            .range(2..=100000)
                            .speed(10),
                    );
                    ui.end_row();
                }

                if self.step_output_head == OutputHeadChoice::RoboticsActionHead {
                    ui.label("Action Prediction Horizon (Steps):");
                    ui.add(
                        egui::DragValue::new(&mut self.action_horizon)
                            .range(1..=128)
                            .speed(1),
                    );
                    ui.end_row();

                    ui.label("Action Vector Degrees of Freedom:");
                    ui.add(
                        egui::DragValue::new(&mut self.action_dim)
                            .range(1..=32)
                            .speed(1),
                    );
                    ui.end_row();
                }

                ui.label("Loss Function Pairing:");
                egui::ComboBox::from_id_salt("loss_combo")
                    .selected_text(format!("{:?}", self.loss_choice))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.loss_choice,
                            LossPairing::CrossEntropy,
                            "Cross-Entropy (Categorical)",
                        );
                        ui.selectable_value(
                            &mut self.loss_choice,
                            LossPairing::MSE,
                            "Mean Squared Error (L2)",
                        );
                        ui.selectable_value(
                            &mut self.loss_choice,
                            LossPairing::SmoothL1,
                            "Smooth L1 / Huber",
                        );
                        ui.selectable_value(
                            &mut self.loss_choice,
                            LossPairing::Focal,
                            "Focal Loss (Class Imbalance)",
                        );
                        ui.selectable_value(
                            &mut self.loss_choice,
                            LossPairing::OptimalTransportConditionalFlowMatching,
                            "OT-CFM (Flow Matching)",
                        );
                    });
                ui.end_row();
            });
    }

    fn render_step_review(&mut self, ui: &mut Ui, should_create: &mut bool) {
        ui.label(
            RichText::new("Step 7: Target Hardware, Execution Precision & Assembly")
                .size(14.0)
                .strong()
                .color(Color32::from_rgb(100, 190, 255)),
        );
        ui.label(
            RichText::new("Choose execution hardware and numeric precision to finalize the architecture synthesis:")
                .size(11.5)
                .color(Color32::from_rgb(180, 190, 205)),
        );
        ui.add_space(8.0);

        egui::Grid::new("step7_grid")
            .num_columns(2)
            .spacing([16.0, 8.0])
            .show(ui, |ui| {
                ui.label("Target Hardware Deployment Profile:");
                egui::ComboBox::from_id_salt("hw_profile_combo")
                    .selected_text(&self.target_hardware.profile_name)
                    .show_ui(ui, |ui| {
                        if ui
                            .selectable_label(
                                self.target_hardware.profile_name.contains("RTX4090"),
                                "Consumer Workstation (RTX 4090 24GB)",
                            )
                            .clicked()
                        {
                            self.target_hardware = HardwareTargetProfile::rtx_4090();
                        }
                        if ui
                            .selectable_label(
                                self.target_hardware.profile_name.contains("H100"),
                                "Data Center Accelerator (NVIDIA H100 SXM 80GB)",
                            )
                            .clicked()
                        {
                            self.target_hardware = HardwareTargetProfile::datacenter_h100();
                        }
                        if ui
                            .selectable_label(
                                self.target_hardware.profile_name.contains("Jetson"),
                                "Edge Robotics (Jetson Orin 8GB)",
                            )
                            .clicked()
                        {
                            self.target_hardware = HardwareTargetProfile::jetson_orin();
                        }
                    });
                ui.end_row();

                ui.label("Model Execution Precision:");
                egui::ComboBox::from_id_salt("step_precision_combo")
                    .selected_text(format!("{:?}", self.precision))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.precision,
                            PrecisionFormat::BF16,
                            "BF16 (16-bit Brain Float - Standard)",
                        );
                        ui.selectable_value(
                            &mut self.precision,
                            PrecisionFormat::FP16,
                            "FP16 (16-bit Float)",
                        );
                        ui.selectable_value(
                            &mut self.precision,
                            PrecisionFormat::FP8E4M3FN,
                            "FP8 E4M3FN (8-bit Quantized)",
                        );
                        ui.selectable_value(
                            &mut self.precision,
                            PrecisionFormat::BitNet158b,
                            "BitNet 1.58b (Ternary {-1,0,+1})",
                        );
                        ui.selectable_value(
                            &mut self.precision,
                            PrecisionFormat::FP32,
                            "FP32 (32-bit Single Precision)",
                        );
                    });
                ui.end_row();
            });

        ui.add_space(14.0);
        egui::Frame::group(ui.style())
            .fill(Color32::from_rgb(20, 36, 28))
            .stroke(Stroke::new(1.5, Color32::from_rgb(70, 180, 110)))
            .corner_radius(CornerRadius::same(6))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(format!("{} Ready to Assemble Architecture!", bi::CHECK_CIRCLE_FILL)).color(Color32::from_rgb(120, 240, 160)).strong());
                });
                ui.label(RichText::new("Clicking the assemble button will construct the typed input/output ports, generate the residual block DAG, and instantiate the model canvas in the studio.").size(11.5).color(Color32::from_rgb(180, 220, 190)));
                ui.add_space(6.0);
                if ui
                    .button(
                        RichText::new(format!("{} Assemble & Open in Studio Canvas", bi::ROCKET_TAKEOFF))
                            .size(14.0)
                            .color(Color32::from_rgb(120, 245, 165))
                            .strong(),
                    )
                    .clicked()
                {
                    *should_create = true;
                }
            });
    }

    fn render_live_telemetry_dock(&self, ui: &mut Ui, should_create: &mut bool) {
        let params = self.estimate_total_params();
        let memory_mb = self.estimate_weight_memory_mb();
        let kv_kb = self.estimate_kv_cache_kb_per_token();

        egui::Frame::group(ui.style())
            .fill(Color32::from_rgb(20, 24, 34))
            .stroke(Stroke::new(1.0, Color32::from_rgb(45, 55, 75)))
            .corner_radius(CornerRadius::same(6))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(format!("{} Live Architecture Telemetry", bi::ACTIVITY))
                            .size(13.0)
                            .strong()
                            .color(Color32::from_rgb(120, 195, 255)),
                    );
                });
                ui.separator();

                egui::Grid::new("telemetry_dock_grid")
                    .num_columns(2)
                    .spacing([12.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Est. Parameter Count:");
                        let param_str = if params >= 1_000_000_000 {
                            format!("{:.2}B", (params as f64) / 1e9)
                        } else if params >= 1_000_000 {
                            format!("{:.2}M", (params as f64) / 1e6)
                        } else {
                            format!("{:.1}K", (params as f64) / 1e3)
                        };
                        ui.label(
                            RichText::new(param_str)
                                .color(Color32::from_rgb(120, 240, 160))
                                .strong(),
                        );
                        ui.end_row();

                        ui.label("Est. Weight Footprint:");
                        ui.label(
                            RichText::new(format!("{memory_mb:.1} MB ({:?})", self.precision))
                                .color(Color32::from_rgb(240, 210, 120))
                                .strong(),
                        );
                        ui.end_row();

                        if kv_kb > 0.0 {
                            ui.label("KV-Cache / Token:");
                            ui.label(
                                RichText::new(format!("{kv_kb:.1} KB"))
                                    .color(Color32::from_rgb(150, 210, 255))
                                    .strong(),
                            );
                            ui.end_row();
                        }

                        ui.label("Selected Precision:");
                        ui.label(format!("{:?}", self.precision));
                        ui.end_row();

                        ui.label("Target Device:");
                        ui.label(
                            RichText::new(&self.target_hardware.profile_name)
                                .size(11.0)
                                .monospace(),
                        );
                        ui.end_row();
                    });

                ui.add_space(8.0);
                ui.separator();
                ui.label(
                    RichText::new("Layer Pipeline Trace:")
                        .size(12.0)
                        .strong()
                        .color(Color32::from_rgb(190, 205, 225)),
                );

                ScrollArea::vertical()
                    .id_salt("layer_trace_scroll")
                    .max_height(200.0)
                    .show(ui, |ui| {
                        ui.label(
                            RichText::new(format!("Input [{}]", self.step_modality.label()))
                                .size(11.0)
                                .color(Color32::from_rgb(130, 220, 160)),
                        );
                        ui.label(RichText::new("  |").size(10.0).color(Color32::GRAY));
                        ui.label(
                            RichText::new(format!("  v [d_model = {}]", self.d_model))
                                .size(10.5)
                                .color(Color32::from_rgb(140, 180, 220)),
                        );
                        ui.label(RichText::new("  |").size(10.0).color(Color32::GRAY));

                        ui.label(
                            RichText::new(format!("  Backbone Block (x{} depth):", self.depth))
                                .size(11.0)
                                .strong()
                                .color(Color32::from_rgb(220, 190, 110)),
                        );
                        ui.label(
                            RichText::new(format!("    ├── Norm: {:?}", self.step_norm))
                                .size(10.5)
                                .color(Color32::from_rgb(170, 180, 195)),
                        );
                        ui.label(
                            RichText::new(format!(
                                "    ├── Core: {}",
                                self.step_backbone_op.label()
                            ))
                            .size(10.5)
                            .color(Color32::from_rgb(170, 180, 195)),
                        );
                        ui.label(
                            RichText::new(format!("    ├── Residual: {:?}", self.step_residual))
                                .size(10.5)
                                .color(Color32::from_rgb(140, 170, 200)),
                        );
                        ui.label(
                            RichText::new(format!("    ├── Norm: {:?}", self.step_norm))
                                .size(10.5)
                                .color(Color32::from_rgb(170, 180, 195)),
                        );
                        ui.label(
                            RichText::new(format!("    └── FFN: {}", self.step_ffn.label()))
                                .size(10.5)
                                .color(Color32::from_rgb(170, 180, 195)),
                        );

                        ui.label(RichText::new("  |").size(10.0).color(Color32::GRAY));
                        ui.label(
                            RichText::new(format!("Final Norm: {:?}", self.step_norm))
                                .size(10.5)
                                .color(Color32::from_rgb(170, 180, 195)),
                        );
                        ui.label(RichText::new("  |").size(10.0).color(Color32::GRAY));
                        ui.label(
                            RichText::new(format!(
                                "Output Readout: {}",
                                self.step_output_head.label()
                            ))
                            .size(11.0)
                            .color(Color32::from_rgb(130, 210, 255)),
                        );
                        ui.label(
                            RichText::new(format!("Loss: {:?}", self.loss_choice))
                                .size(10.5)
                                .color(Color32::from_rgb(240, 150, 150)),
                        );
                    });

                ui.add_space(8.0);
                if ui
                    .button(
                        RichText::new(format!("{} Assemble Model Now", bi::ROCKET_TAKEOFF))
                            .size(13.0)
                            .color(Color32::from_rgb(120, 240, 160))
                            .strong(),
                    )
                    .clicked()
                {
                    *should_create = true;
                }
            });
    }

    // --- Legacy / Complementary Tabs ---

    fn render_archetypes_tab(&mut self, ui: &mut Ui) {
        ui.label(
            RichText::new("1. Select Architecture Archetype")
                .size(13.0)
                .strong(),
        );
        ui.add_space(4.0);

        ScrollArea::vertical().max_height(240.0).show(ui, |ui| {
            for &archetype in ArchitectureArchetype::ALL {
                if archetype == ArchitectureArchetype::Blank {
                    continue;
                }
                let is_selected = self.selected_archetype == archetype;

                let bg_color = if is_selected {
                    Color32::from_rgb(32, 45, 65)
                } else {
                    Color32::from_rgb(22, 25, 34)
                };
                let stroke_color = if is_selected {
                    Color32::from_rgb(90, 160, 255)
                } else {
                    Color32::from_rgb(45, 52, 68)
                };

                egui::Frame::group(ui.style())
                    .fill(bg_color)
                    .stroke(Stroke::new(
                        if is_selected { 1.5 } else { 1.0 },
                        stroke_color,
                    ))
                    .corner_radius(CornerRadius::same(6))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            let label =
                                format!("{} {}", archetype.icon(), archetype.display_name());
                            if ui
                                .selectable_label(is_selected, RichText::new(label).strong())
                                .clicked()
                            {
                                self.apply_archetype_defaults(archetype);
                            }
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    ui.label(
                                        RichText::new(archetype.domain())
                                            .size(11.0)
                                            .color(Color32::from_rgb(140, 180, 220)),
                                    );
                                },
                            );
                        });
                        ui.label(
                            RichText::new(archetype.description())
                                .size(11.5)
                                .color(Color32::from_rgb(180, 190, 205)),
                        );
                    });
                ui.add_space(4.0);
            }
        });

        ui.add_space(8.0);
        ui.separator();
        ui.label(
            RichText::new("2. Configure Dimensions & Parameters")
                .size(13.0)
                .strong(),
        );
        ui.add_space(6.0);

        egui::Grid::new("archetype_params_grid")
            .num_columns(2)
            .spacing([16.0, 8.0])
            .show(ui, |ui| {
                ui.label("Model Identifier:");
                ui.text_edit_singleline(&mut self.model_id);
                ui.end_row();

                ui.label("Display Name:");
                ui.text_edit_singleline(&mut self.display_name);
                ui.end_row();

                ui.label("Hidden Dimension (d_model):");
                ui.horizontal(|ui| {
                    ui.add(
                        egui::DragValue::new(&mut self.d_model)
                            .range(32..=8192)
                            .speed(32),
                    );
                    ui.label("(e.g., 128, 256, 512, 1024, 2048)");
                });
                ui.end_row();

                ui.label("Depth (Blocks/Layers):");
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut self.depth).range(1..=64).speed(1));
                    ui.label("(Number of sequential backbone blocks)");
                });
                ui.end_row();

                ui.label("Execution Precision:");
                egui::ComboBox::from_id_salt("wizard_precision_select")
                    .selected_text(format!(
                        "{:?} ({} bits)",
                        self.precision,
                        self.precision.bits()
                    ))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.precision,
                            PrecisionFormat::BF16,
                            "BF16 (16-bit Brain Float)",
                        );
                        ui.selectable_value(
                            &mut self.precision,
                            PrecisionFormat::FP16,
                            "FP16 (16-bit IEEE Float)",
                        );
                        ui.selectable_value(
                            &mut self.precision,
                            PrecisionFormat::FP8E4M3FN,
                            "FP8 E4M3FN (8-bit Quantized)",
                        );
                        ui.selectable_value(
                            &mut self.precision,
                            PrecisionFormat::BitNet158b,
                            "BitNet 1.58b (Ternary {-1,0,+1})",
                        );
                        ui.selectable_value(
                            &mut self.precision,
                            PrecisionFormat::FP32,
                            "FP32 (32-bit Single Precision)",
                        );
                    });
                ui.end_row();
            });
    }

    fn render_templates_tab(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.label(RichText::new("Search Templates:").strong());
            ui.text_edit_singleline(&mut self.search_query);
        });
        ui.add_space(6.0);

        let query = self.search_query.to_lowercase();

        ScrollArea::vertical().max_height(340.0).show(ui, |ui| {
            for (idx, tmpl) in TEMPLATES.iter().enumerate() {
                let summary = tmpl.summary();
                let matches_query = query.is_empty()
                    || tmpl.display_name.to_lowercase().contains(&query)
                    || tmpl.file_name.to_lowercase().contains(&query)
                    || summary.to_lowercase().contains(&query);

                if !matches_query {
                    continue;
                }

                let is_selected = self.selected_template_index == Some(idx);
                let bg_color = if is_selected {
                    Color32::from_rgb(32, 45, 65)
                } else {
                    Color32::from_rgb(22, 25, 34)
                };
                let stroke_color = if is_selected {
                    Color32::from_rgb(90, 160, 255)
                } else {
                    Color32::from_rgb(45, 52, 68)
                };

                egui::Frame::group(ui.style())
                    .fill(bg_color)
                    .stroke(Stroke::new(
                        if is_selected { 1.5 } else { 1.0 },
                        stroke_color,
                    ))
                    .corner_radius(CornerRadius::same(6))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            if ui
                                .selectable_label(
                                    is_selected,
                                    RichText::new(tmpl.display_name).strong(),
                                )
                                .clicked()
                            {
                                self.selected_template_index = Some(idx);
                            }
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    let clean_name = tmpl
                                        .file_name
                                        .trim_end_matches(".yaml")
                                        .trim_end_matches(".yaml");
                                    ui.label(
                                        RichText::new(format!("{clean_name}.yaml"))
                                            .size(11.0)
                                            .monospace()
                                            .color(Color32::from_rgb(140, 180, 220)),
                                    );
                                },
                            );
                        });
                        ui.label(
                            RichText::new(summary)
                                .size(11.5)
                                .color(Color32::from_rgb(175, 185, 200)),
                        );
                    });
                ui.add_space(4.0);
            }
        });
    }

    fn render_blank_tab(&mut self, ui: &mut Ui) {
        ui.label(
            RichText::new("Start with an Empty Canvas")
                .size(14.0)
                .strong()
                .color(Color32::from_rgb(100, 190, 255)),
        );
        ui.add_space(4.0);
        ui.label("A clean canvas initialized with standard input (x) and output (y) ports. You can freely assemble your network using the Operator Catalog on the left sidebar.");

        ui.add_space(12.0);
        egui::Grid::new("blank_params_grid")
            .num_columns(2)
            .spacing([16.0, 8.0])
            .show(ui, |ui| {
                ui.label("Model Identifier:");
                ui.text_edit_singleline(&mut self.model_id);
                ui.end_row();

                ui.label("Display Name:");
                ui.text_edit_singleline(&mut self.display_name);
                ui.end_row();

                ui.label("Input Latent Dimension:");
                ui.add(
                    egui::DragValue::new(&mut self.d_model)
                        .range(16..=4096)
                        .speed(16),
                );
                ui.end_row();

                ui.label("Default Precision:");
                egui::ComboBox::from_id_salt("blank_precision_select")
                    .selected_text(format!("{:?}", self.precision))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.precision, PrecisionFormat::BF16, "BF16");
                        ui.selectable_value(&mut self.precision, PrecisionFormat::FP16, "FP16");
                        ui.selectable_value(
                            &mut self.precision,
                            PrecisionFormat::FP8E4M3FN,
                            "FP8 E4M3FN",
                        );
                        ui.selectable_value(
                            &mut self.precision,
                            PrecisionFormat::BitNet158b,
                            "BitNet 1.58b",
                        );
                        ui.selectable_value(&mut self.precision, PrecisionFormat::FP32, "FP32");
                    });
                ui.end_row();
            });
    }

    fn render_training_planner_tab(&mut self, ui: &mut Ui, studio: &StudioState) {
        ui.horizontal(|ui| {
            ui.label(
                RichText::new("1. Select Guided Training Plan Preset for Active Model:")
                    .size(13.0)
                    .strong(),
            );
            ui.label(
                RichText::new(format!(
                    "'{}' ({} parameters)",
                    studio.spec.metadata.display_name, studio.partition_report.total_param_count
                ))
                .size(12.0)
                .color(Color32::from_rgb(120, 210, 255)),
            );
        });
        ui.add_space(4.0);

        ScrollArea::vertical().max_height(250.0).show(ui, |ui| {
            for &preset in TrainingPlanPreset::ALL {
                let is_selected = self.selected_training_preset == preset;
                let bg_color = if is_selected {
                    Color32::from_rgb(32, 45, 65)
                } else {
                    Color32::from_rgb(22, 25, 34)
                };
                let stroke_color = if is_selected {
                    Color32::from_rgb(90, 160, 255)
                } else {
                    Color32::from_rgb(45, 52, 68)
                };

                egui::Frame::group(ui.style())
                    .fill(bg_color)
                    .stroke(Stroke::new(
                        if is_selected { 1.5 } else { 1.0 },
                        stroke_color,
                    ))
                    .corner_radius(CornerRadius::same(6))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            if ui
                                .selectable_label(
                                    is_selected,
                                    RichText::new(preset.display_name()).strong(),
                                )
                                .clicked()
                            {
                                self.selected_training_preset = preset;
                            }
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    ui.label(
                                        RichText::new(preset.domain_tag())
                                            .size(11.0)
                                            .color(Color32::from_rgb(140, 195, 235)),
                                    );
                                },
                            );
                        });
                        ui.label(
                            RichText::new(preset.description())
                                .size(11.5)
                                .color(Color32::from_rgb(180, 190, 205)),
                        );
                    });
                ui.add_space(4.0);
            }
        });

        ui.add_space(8.0);
        ui.separator();
        ui.label(
            RichText::new("2. Configure Cluster Scale & Context Window")
                .size(13.0)
                .strong(),
        );
        ui.add_space(6.0);

        egui::Grid::new("training_wizard_params_grid")
            .num_columns(2)
            .spacing([16.0, 8.0])
            .show(ui, |ui| {
                ui.label("Target Accelerator Count (GPUs/Nodes):");
                ui.horizontal(|ui| {
                    ui.add(
                        egui::DragValue::new(&mut self.training_num_gpus)
                            .range(1..=4096)
                            .speed(1),
                    );
                    for &preset_gpus in &[1usize, 4, 8, 64, 256] {
                        if ui
                            .selectable_label(
                                self.training_num_gpus == preset_gpus,
                                format!("{preset_gpus}x"),
                            )
                            .clicked()
                        {
                            self.training_num_gpus = preset_gpus;
                        }
                    }
                });
                ui.end_row();

                ui.label("Base Training Sequence / Context Length:");
                ui.horizontal(|ui| {
                    ui.add(
                        egui::DragValue::new(&mut self.training_seq_len)
                            .range(64..=131072)
                            .speed(128),
                    );
                    for &seq in &[512usize, 2048, 4096, 8192, 32768] {
                        if ui
                            .selectable_label(self.training_seq_len == seq, format!("{seq}"))
                            .clicked()
                        {
                            self.training_seq_len = seq;
                        }
                    }
                });
                ui.end_row();
            });
    }
}
