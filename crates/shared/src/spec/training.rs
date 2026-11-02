//! Training specification, optimizer parameter partitioning, schedules, and gradient hygiene.

use serde::{Deserialize, Serialize};

/// Optimizer configuration for internal 2D weight matrices (GEMMs) via Newton-Schulz matrix orthogonalization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MuonConfig {
    #[serde(default = "default_muon_lr")]
    pub lr: f64,
    #[serde(default = "default_muon_momentum")]
    pub momentum: f64,
    #[serde(default = "default_muon_steps")]
    pub newton_schulz_steps: usize,
    #[serde(default = "default_weight_decay")]
    pub weight_decay: f64,
}

impl Default for MuonConfig {
    fn default() -> Self {
        Self {
            lr: default_muon_lr(),
            momentum: default_muon_momentum(),
            newton_schulz_steps: default_muon_steps(),
            weight_decay: default_weight_decay(),
        }
    }
}

fn default_muon_lr() -> f64 {
    0.02
}
fn default_muon_momentum() -> f64 {
    0.95
}
fn default_muon_steps() -> usize {
    5
}
fn default_weight_decay() -> f64 {
    0.01
}

/// Optimizer configuration for token embeddings, 1D vectors, and final output heads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdamWConfig {
    #[serde(default = "default_adamw_lr")]
    pub lr: f64,
    #[serde(default = "default_adamw_betas")]
    pub betas: (f64, f64),
    #[serde(default = "default_adamw_eps")]
    pub eps: f64,
    #[serde(default = "default_weight_decay")]
    pub weight_decay: f64,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            lr: default_adamw_lr(),
            betas: default_adamw_betas(),
            eps: default_adamw_eps(),
            weight_decay: default_weight_decay(),
        }
    }
}

fn default_adamw_lr() -> f64 {
    0.001
}
fn default_adamw_betas() -> (f64, f64) {
    (0.9, 0.95)
}
fn default_adamw_eps() -> f64 {
    1e-8
}

/// Optimizer configuration for Lion (EvoLved Sign Momentum — Google Brain).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LionConfig {
    #[serde(default = "default_lion_lr")]
    pub lr: f64,
    #[serde(default = "default_lion_betas")]
    pub betas: (f64, f64),
    #[serde(default = "default_weight_decay")]
    pub weight_decay: f64,
}

impl Default for LionConfig {
    fn default() -> Self {
        Self {
            lr: default_lion_lr(),
            betas: default_lion_betas(),
            weight_decay: default_weight_decay(),
        }
    }
}

fn default_lion_lr() -> f64 {
    0.0003
}
fn default_lion_betas() -> (f64, f64) {
    (0.9, 0.99)
}

/// Optimizer configuration for Sophia-G (Second-Order Clipped Stochastic Hessian — Stanford).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SophiaConfig {
    #[serde(default = "default_sophia_lr")]
    pub lr: f64,
    #[serde(default = "default_sophia_betas")]
    pub betas: (f64, f64),
    #[serde(default = "default_sophia_rho")]
    pub rho: f64,
    #[serde(default = "default_weight_decay")]
    pub weight_decay: f64,
}

impl Default for SophiaConfig {
    fn default() -> Self {
        Self {
            lr: default_sophia_lr(),
            betas: default_sophia_betas(),
            rho: default_sophia_rho(),
            weight_decay: default_weight_decay(),
        }
    }
}

fn default_sophia_lr() -> f64 {
    0.0002
}
fn default_sophia_betas() -> (f64, f64) {
    (0.96, 0.99)
}
fn default_sophia_rho() -> f64 {
    0.04
}

/// Optimizer configuration for L-BFGS quasi-Newton optimization (SciML & PINNs).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LbfgsConfig {
    #[serde(default = "default_lbfgs_lr")]
    pub lr: f64,
    #[serde(default = "default_lbfgs_max_iter")]
    pub max_iter: usize,
    #[serde(default = "default_lbfgs_history_size")]
    pub history_size: usize,
    #[serde(default = "default_lbfgs_tolerance")]
    pub tolerance_grad: f64,
}

impl Default for LbfgsConfig {
    fn default() -> Self {
        Self {
            lr: default_lbfgs_lr(),
            max_iter: default_lbfgs_max_iter(),
            history_size: default_lbfgs_history_size(),
            tolerance_grad: default_lbfgs_tolerance(),
        }
    }
}

fn default_lbfgs_lr() -> f64 {
    1.0
}
fn default_lbfgs_max_iter() -> usize {
    20
}
fn default_lbfgs_history_size() -> usize {
    100
}
fn default_lbfgs_tolerance() -> f64 {
    1e-7
}

/// Primary optimizer algorithm family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerFamily {
    #[default]
    MuonAdamWPartitioned,
    AdamWStandard,
    LionSignMomentum,
    SophiaSecondOrder,
    AdanAdaptiveNesterov,
    LBFGSQuasiNewton,
    ScheduleFreeAdamW,
    PagedAdamW8Bit,
    SGDWithNesterov,
}

impl OptimizerFamily {
    pub const ALL: &'static [OptimizerFamily] = &[
        Self::MuonAdamWPartitioned,
        Self::AdamWStandard,
        Self::LionSignMomentum,
        Self::SophiaSecondOrder,
        Self::AdanAdaptiveNesterov,
        Self::LBFGSQuasiNewton,
        Self::ScheduleFreeAdamW,
        Self::PagedAdamW8Bit,
        Self::SGDWithNesterov,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            Self::MuonAdamWPartitioned => "Muon + AdamW Partitioned (Newton-Schulz 2D GEMM)",
            Self::AdamWStandard => "AdamW (Decoupled Weight Decay)",
            Self::LionSignMomentum => "Lion (EvoLved Sign Momentum — 3x Memory Save)",
            Self::SophiaSecondOrder => "Sophia-G (Second-Order Clipped Stochastic Hessian)",
            Self::AdanAdaptiveNesterov => "Adan (Adaptive Nesterov Momentum)",
            Self::LBFGSQuasiNewton => "L-BFGS (Quasi-Newton Second-Order for SciML / PINNs)",
            Self::ScheduleFreeAdamW => "Schedule-Free AdamW (FAIR — No Decay Schedule Tuning)",
            Self::PagedAdamW8Bit => "Paged 8-bit AdamW (bitsandbytes Blockwise Quantization)",
            Self::SGDWithNesterov => "SGD + Nesterov Momentum (Vision & Robotics)",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::MuonAdamWPartitioned => {
                "Partitions 2D internal weight matrices to Muon (5-step Newton-Schulz orthogonalization) and 1D vectors/embeddings to AdamW, yielding ~1.5x-2x faster training convergence."
            }
            Self::AdamWStandard => {
                "Industry-standard decoupled weight decay optimizer maintaining 1st (momentum) and 2nd (RMS variance) moments per parameter."
            }
            Self::LionSignMomentum => {
                "Google Brain sign-momentum optimizer using only tracking of sign(momentum), reducing optimizer state memory from 8 bytes/param to 4 bytes/param."
            }
            Self::SophiaSecondOrder => {
                "Stanford second-order optimizer estimating diagonal Hessian curvature via lightweight Hutchinson stochastic probes, achieving 2x speedup on language models."
            }
            Self::AdanAdaptiveNesterov => {
                "Adaptive Nesterov momentum algorithm estimating both gradient differences and second moments, accelerating convergence on dense and sparse models."
            }
            Self::LBFGSQuasiNewton => {
                "Limited-memory Broyden-Fletcher-Goldfarb-Shanno quasi-Newton optimizer utilizing two-loop recursion for machine-precision physics residual minimization."
            }
            Self::ScheduleFreeAdamW => {
                "Meta FAIR schedule-free optimizer that eliminates learning rate decay schedule tuning by maintaining dual online and evaluation weights."
            }
            Self::PagedAdamW8Bit => {
                "Bitsandbytes paged 8-bit optimizer storing first and second moments in non-linear dynamic 8-bit blocks, cutting optimizer VRAM by 75%."
            }
            Self::SGDWithNesterov => {
                "Classic Stochastic Gradient Descent with Nesterov accelerated momentum, preferred for high-generalization computer vision and reinforcement learning."
            }
        }
    }
}

/// Learning rate schedule configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SchedulerSpec {
    WarmupStableDecay {
        #[serde(default = "default_wsd_warmup")]
        warmup_ratio: f64,
        #[serde(default = "default_wsd_stable")]
        stable_ratio: f64,
        #[serde(default = "default_wsd_decay")]
        decay_ratio: f64,
        #[serde(default = "default_min_lr_ratio")]
        min_lr_ratio: f64,
    },
    #[serde(rename = "cosine_with_warmup", alias = "cosine")]
    CosineWithWarmup {
        #[serde(default = "default_wsd_warmup")]
        warmup_ratio: f64,
        #[serde(default = "default_min_lr_ratio")]
        min_lr_ratio: f64,
        #[serde(default)]
        warmup_steps: Option<usize>,
        #[serde(default)]
        total_steps: Option<usize>,
        #[serde(default)]
        min_lr: Option<f64>,
    },
    Constant {
        lr: f64,
    },
}

impl Default for SchedulerSpec {
    fn default() -> Self {
        Self::WarmupStableDecay {
            warmup_ratio: default_wsd_warmup(),
            stable_ratio: default_wsd_stable(),
            decay_ratio: default_wsd_decay(),
            min_lr_ratio: default_min_lr_ratio(),
        }
    }
}

fn default_wsd_warmup() -> f64 {
    0.05
}
fn default_wsd_stable() -> f64 {
    0.75
}
fn default_wsd_decay() -> f64 {
    0.20
}
fn default_min_lr_ratio() -> f64 {
    0.01
}

/// Gradient clipping, weight averaging, and numerical stabilization rules.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GradientHygieneSpec {
    /// Adaptive percentile-based gradient clipping (AutoClip).
    #[serde(default = "default_true")]
    pub auto_clip: bool,
    #[serde(default = "default_clip_percentile")]
    pub clip_percentile: f64,
    #[serde(default = "default_clip_max_norm")]
    pub max_norm: f64,
    /// Exponential Moving Average (EMA) weight decay factor.
    #[serde(default = "default_ema_decay")]
    pub ema_decay: f64,
    /// Grokfast gradient filtering accelerating algorithmic generalization.
    #[serde(default)]
    pub grokfast: bool,
    /// Activation checkpointing to trade recompute for VRAM.
    #[serde(default)]
    pub activation_checkpointing: bool,
}

impl Default for GradientHygieneSpec {
    fn default() -> Self {
        Self {
            auto_clip: true,
            clip_percentile: default_clip_percentile(),
            max_norm: default_clip_max_norm(),
            ema_decay: default_ema_decay(),
            grokfast: false,
            activation_checkpointing: false,
        }
    }
}

fn default_true() -> bool {
    true
}
fn default_clip_percentile() -> f64 {
    90.0
}
fn default_clip_max_norm() -> f64 {
    1.0
}
fn default_ema_decay() -> f64 {
    0.999
}

/// Multi-task and multi-objective loss balancing strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum LossBalancingStrategy {
    #[default]
    FixedWeights,
    /// Projected Conflicting Gradients (PCGrad) projecting gradients with negative inner product.
    #[serde(rename = "pcgrad", alias = "p_c_grad")]
    PCGrad,
    /// Neural Tangent Kernel (NTK) eigenvalue balancing for physics-informed neural networks.
    #[serde(rename = "ntk_balancing", alias = "n_t_k_balancing")]
    NTKBalancing,
    /// Dynamic gradient norm balancing.
    #[serde(rename = "grad_norm", alias = "gradnorm")]
    GradNorm,
    /// Homoscedastic task uncertainty weighting (Kendall & Gal).
    #[serde(rename = "uncertainty_kendall_gal")]
    UncertaintyKendallGal,
}

/// Training paradigm for a specific stage in a multi-stage training pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TrainingParadigm {
    #[default]
    Pretraining,
    ContextExtension,
    SupervisedFineTuning,
    PreferenceAlignmentDpo,
    ReinforcementLearningGrpo,
    PhysicsCurriculumSobolev,
    KnowledgeDistillationQat,
    SelfSupervisedContrastive,
    SelfSupervisedMaskedModeling,
    SelfSupervisedJointEmbedding,
    SelfSupervisedAntiCollapse,
    SelfSupervisedStudentTeacher,
    TabularGradientBoostingCv,
    TabularSelfSupervisedVime,
    FlowMatchingRectifiedFlow,
    ReinforcementLearningPpo,
    ContinualLearningEwc,
    LearnableQuantizationBitwidthQat,
    SuperSmoothSobolevLipschitz,
}

impl TrainingParadigm {
    pub const ALL: &'static [TrainingParadigm] = &[
        Self::Pretraining,
        Self::ContextExtension,
        Self::SupervisedFineTuning,
        Self::PreferenceAlignmentDpo,
        Self::ReinforcementLearningGrpo,
        Self::PhysicsCurriculumSobolev,
        Self::KnowledgeDistillationQat,
        Self::SelfSupervisedContrastive,
        Self::SelfSupervisedMaskedModeling,
        Self::SelfSupervisedJointEmbedding,
        Self::SelfSupervisedAntiCollapse,
        Self::SelfSupervisedStudentTeacher,
        Self::TabularGradientBoostingCv,
        Self::TabularSelfSupervisedVime,
        Self::FlowMatchingRectifiedFlow,
        Self::ReinforcementLearningPpo,
        Self::ContinualLearningEwc,
        Self::LearnableQuantizationBitwidthQat,
        Self::SuperSmoothSobolevLipschitz,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            Self::Pretraining => "Pretraining (Foundation Trunk)",
            Self::ContextExtension => "Long-Context Extension (YaRN / Ring)",
            Self::SupervisedFineTuning => "Supervised Fine-Tuning (SFT / Instruction)",
            Self::PreferenceAlignmentDpo => "Preference Alignment (DPO / SimPO)",
            Self::ReinforcementLearningGrpo => "Reinforcement Learning (GRPO / RLVR)",
            Self::PhysicsCurriculumSobolev => "Physics Curriculum (Sobolev / NTK Refinement)",
            Self::KnowledgeDistillationQat => "Distillation & Quantization-Aware (QAT)",
            Self::SelfSupervisedContrastive => "SSL Contrastive (InfoNCE / SimCLR / SigLIP)",
            Self::SelfSupervisedMaskedModeling => "SSL Masked Reconstruction (MAE / BERT MLM)",
            Self::SelfSupervisedJointEmbedding => "SSL Latent Prediction (I-JEPA / V-JEPA)",
            Self::SelfSupervisedAntiCollapse => "SSL Anti-Collapse (VICReg / Barlow Twins)",
            Self::SelfSupervisedStudentTeacher => "SSL Student-Teacher EMA (DINOv2 / iBOT)",
            Self::TabularGradientBoostingCv => "Tabular GBDT (LightGBM / CatBoost OOF CV)",
            Self::TabularSelfSupervisedVime => "Tabular GBDT SSL (VIME Mask & Hastie Synthetic)",
            Self::FlowMatchingRectifiedFlow => "Rectified Flow & Score Matching (OT-CFM / EDM)",
            Self::ReinforcementLearningPpo => "Actor-Critic & World Model Rollouts (PPO / Dreamer)",
            Self::ContinualLearningEwc => "Continual & Meta-Learning (EWC Fisher / MAML)",
            Self::LearnableQuantizationBitwidthQat => {
                "Learnable Quantization Level & Mixed-Bitwidth QAT (LSQ+ / FracBits)"
            }
            Self::SuperSmoothSobolevLipschitz => {
                "Super-Smooth C^∞ Sobolev H² & 1-Lipschitz Certified (Björck / ICNN)"
            }
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::Pretraining => {
                "Causal next-token or foundational representation learning on large-scale unlabeled corpora."
            }
            Self::ContextExtension => {
                "Extends positional embeddings (YaRN / RoPE scaling) and ring-attention context window up to 64K-1M tokens."
            }
            Self::SupervisedFineTuning => {
                "Supervised instruction tuning on curated prompt-response demonstrations with masked prompt loss."
            }
            Self::PreferenceAlignmentDpo => {
                "Direct Preference Optimization (DPO / SimPO / ORPO) aligning policy log-ratios on chosen vs rejected pairs."
            }
            Self::ReinforcementLearningGrpo => {
                "Group Relative Policy Optimization (GRPO / RLVR) sampling group rollouts against verifiable math/code/logic rewards without a critic."
            }
            Self::PhysicsCurriculumSobolev => {
                "Sobolev H1 derivative residual minimization with Neural Tangent Kernel (NTK) eigenvalue balancing."
            }
            Self::KnowledgeDistillationQat => {
                "Teacher-to-student KL divergence distillation combined with fake-quantized QAT steerable scales."
            }
            Self::SelfSupervisedContrastive => {
                "Maximizes mutual information between augmented positive views while repelling in-batch or queue negatives (SimCLR / MoCo / CLIP / SigLIP)."
            }
            Self::SelfSupervisedMaskedModeling => {
                "Masks 75-80% of spatial patches, audio frames, or tokens and reconstructs masked signals via an asymmetric decoder (MAE / SimMIM / MLM)."
            }
            Self::SelfSupervisedJointEmbedding => {
                "Predicts target representations of masked spatiotemporal blocks in latent space (I-JEPA / V-JEPA), avoiding pixel-level noise overfitting."
            }
            Self::SelfSupervisedAntiCollapse => {
                "Negative-free SSL enforcing Variance, Invariance, and Covariance regularization (VICReg / Barlow Twins) to prevent dimensional collapse."
            }
            Self::SelfSupervisedStudentTeacher => {
                "Self-distillation with an EMA teacher using centering and temperature sharpening (DINOv2 / BYOL / iBOT) for emergent segmentation and dense features."
            }
            Self::TabularGradientBoostingCv => {
                "Second-order Newton histogram gradient boosting (LightGBM leaf-wise, XGBoost depth-wise, CatBoost oblivious trees) with Out-of-Fold early stopping."
            }
            Self::TabularSelfSupervisedVime => {
                "Self-supervised learning for Gradient Boosting: VIME mask vector estimation + marginal feature imputation + Hastie synthetic-vs-real discrimination + unsupervised tree-leaf embeddings."
            }
            Self::FlowMatchingRectifiedFlow => {
                "Optimal-Transport Conditional Flow Matching (OT-CFM) learning straight-line velocity vector fields v_t(x) = x_1 - x_0."
            }
            Self::ReinforcementLearningPpo => {
                "Online Proximal Policy Optimization (PPO) with GAE critic value baseline or DreamerV3 RSSM latent world-model imagination rollouts."
            }
            Self::ContinualLearningEwc => {
                "Elastic Weight Consolidation (EWC Fisher penalty) and bi-level MAML adaptation preventing catastrophic forgetting across sequential tasks."
            }
            Self::LearnableQuantizationBitwidthQat => {
                "Jointly trains weights, quantization step sizes s (LSQ/LSQ+), clipping thresholds α (PACT), non-uniform grid levels (LQ-Nets), and per-layer fractional bitwidths b_l (FracBits) under a Bit-Operations (BOPs) penalty."
            }
            Self::SuperSmoothSobolevLipschitz => {
                "Trains C^∞ super-smooth and certified K-Lipschitz architectures using Björck/Cayley orthonormal projections, GroupSort/Softplus-β activations, and higher-order Sobolev W^{k,2} Jacobian/Hessian curvature regularization."
            }
        }
    }
}

/// Parameter freeze / adapter policy for a training stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ParameterFreezePolicy {
    #[default]
    FullTrainable,
    FreezeBackboneTrainHeads,
    LoraDoraAdapters,
    FreezeEmbeddingsAndEncoders,
}

impl ParameterFreezePolicy {
    pub const ALL: &'static [ParameterFreezePolicy] = &[
        Self::FullTrainable,
        Self::FreezeBackboneTrainHeads,
        Self::LoraDoraAdapters,
        Self::FreezeEmbeddingsAndEncoders,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            Self::FullTrainable => "Full Model Trainable (100%)",
            Self::FreezeBackboneTrainHeads => "Freeze Backbone (Train Heads / Routers)",
            Self::LoraDoraAdapters => "LoRA / DoRA Low-Rank Adapters (r=32)",
            Self::FreezeEmbeddingsAndEncoders => "Freeze Embeddings & Modality Encoders",
        }
    }
}

/// A single phase in a multi-stage training curriculum (e.g., Pretrain -> Context Ext -> SFT -> GRPO).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingStageSpec {
    pub stage_id: String,
    pub stage_name: String,
    #[serde(default)]
    pub paradigm: TrainingParadigm,
    #[serde(default = "default_stage_tokens")]
    pub token_budget: u64,
    #[serde(default = "default_seq_len")]
    pub sequence_length: usize,
    #[serde(default = "default_lr_mult")]
    pub lr_multiplier: f64,
    #[serde(default)]
    pub freeze_policy: ParameterFreezePolicy,
}

fn default_stage_tokens() -> u64 {
    10_000_000_000
}
fn default_seq_len() -> usize {
    4096
}
fn default_lr_mult() -> f64 {
    1.0
}

impl Default for TrainingStageSpec {
    fn default() -> Self {
        Self {
            stage_id: "stage_1_pretrain".to_string(),
            stage_name: "Stage 1: Core Foundation Pretraining".to_string(),
            paradigm: TrainingParadigm::Pretraining,
            token_budget: 10_000_000_000,
            sequence_length: 4096,
            lr_multiplier: 1.0,
            freeze_policy: ParameterFreezePolicy::FullTrainable,
        }
    }
}

/// Distributed memory sharding mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ShardingStrategy {
    #[default]
    SingleDevice,
    DdpReplicated,
    Zero2GradAndOptim,
    Fsdp2PerParamZero3,
}

impl ShardingStrategy {
    pub const ALL: &'static [ShardingStrategy] = &[
        Self::SingleDevice,
        Self::DdpReplicated,
        Self::Zero2GradAndOptim,
        Self::Fsdp2PerParamZero3,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            Self::SingleDevice => "Single Accelerator (No Sharding)",
            Self::DdpReplicated => "DDP (Replicated Weights, AllReduce Grads)",
            Self::Zero2GradAndOptim => "ZeRO-2 (Shard Gradients + Optimizer States)",
            Self::Fsdp2PerParamZero3 => "FSDP2 / ZeRO-3 (Per-Parameter DTensor Sharding)",
        }
    }
}

/// Distributed parallelism topology and global batch size planner.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParallelismStrategySpec {
    #[serde(default)]
    pub sharding: ShardingStrategy,
    #[serde(default = "default_one")]
    pub data_parallel_degree: usize,
    #[serde(default = "default_one")]
    pub tensor_parallel_degree: usize,
    #[serde(default = "default_one")]
    pub pipeline_parallel_degree: usize,
    #[serde(default = "default_one")]
    pub context_parallel_degree: usize,
    #[serde(default = "default_one")]
    pub expert_parallel_degree: usize,
    #[serde(default = "default_micro_batch")]
    pub micro_batch_size: usize,
    #[serde(default = "default_grad_accum")]
    pub gradient_accumulation_steps: usize,
    #[serde(default = "default_seq_len")]
    pub sequence_length: usize,
}

fn default_one() -> usize {
    1
}
fn default_micro_batch() -> usize {
    4
}
fn default_grad_accum() -> usize {
    8
}

impl Default for ParallelismStrategySpec {
    fn default() -> Self {
        Self {
            sharding: ShardingStrategy::SingleDevice,
            data_parallel_degree: 1,
            tensor_parallel_degree: 1,
            pipeline_parallel_degree: 1,
            context_parallel_degree: 1,
            expert_parallel_degree: 1,
            micro_batch_size: 4,
            gradient_accumulation_steps: 8,
            sequence_length: 4096,
        }
    }
}

impl ParallelismStrategySpec {
    /// Total number of sequences/samples processed per optimizer step across all DP ranks.
    pub fn global_batch_samples(&self) -> usize {
        self.micro_batch_size
            .saturating_mul(self.gradient_accumulation_steps.max(1))
            .saturating_mul(self.data_parallel_degree.max(1))
    }

    /// Total number of tokens/elements processed per optimizer step.
    pub fn global_batch_tokens(&self) -> usize {
        self.global_batch_samples()
            .saturating_mul(self.sequence_length.max(1))
    }

    /// Total accelerators required by the 4D parallelism mesh (DP * TP * PP * CP).
    pub fn total_accelerators(&self) -> usize {
        self.data_parallel_degree
            .max(1)
            .saturating_mul(self.tensor_parallel_degree.max(1))
            .saturating_mul(self.pipeline_parallel_degree.max(1))
            .saturating_mul(self.context_parallel_degree.max(1))
    }
}

/// Dataset token budget, synthetic data mixing, and checkpoint/eval cadence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DatasetCurriculumSpec {
    #[serde(default = "default_stage_tokens")]
    pub total_training_tokens: u64,
    #[serde(default = "default_synthetic_ratio")]
    pub synthetic_data_ratio: f64,
    #[serde(default = "default_true")]
    pub sequence_packing: bool,
    #[serde(default)]
    pub curriculum_length_ramp: bool,
    #[serde(default = "default_ckpt_interval")]
    pub checkpoint_interval_steps: usize,
    #[serde(default = "default_eval_interval")]
    pub eval_interval_steps: usize,
}

fn default_synthetic_ratio() -> f64 {
    0.20
}
fn default_ckpt_interval() -> usize {
    1000
}
fn default_eval_interval() -> usize {
    250
}

impl Default for DatasetCurriculumSpec {
    fn default() -> Self {
        Self {
            total_training_tokens: 10_000_000_000,
            synthetic_data_ratio: 0.20,
            sequence_packing: true,
            curriculum_length_ramp: false,
            checkpoint_interval_steps: 1000,
            eval_interval_steps: 250,
        }
    }
}

/// Differentiable quantization level and bitwidth learning mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum QuantizationTrainingMode {
    #[default]
    FixedPrecision,
    /// LSQ / LSQ+ Learned Step-Size & Zero-Point Offset via Straight-Through Estimator (STE).
    LsqLearnedStepSize,
    /// PACT Parameterized Clipping Activation + DoReFa weight quantization.
    PactActivationClipping,
    /// FracBits / EdMIPS Differentiable Mixed-Precision Bit-Width Learning (optimizes continuous b_l in [min_bits, max_bits] per layer).
    FracBitsMixedBitwidth,
    /// LQ-Nets / AQLM Learnable Non-Uniform Quantization Levels & Codebook Vectors.
    LqNetLearnableLevels,
    /// BitNet b1.58 Native Ternary {-1, 0, +1} Absmean Weight + Absmax INT8 Activation QAT.
    BitNetTernaryQat,
}

impl QuantizationTrainingMode {
    pub const ALL: &'static [QuantizationTrainingMode] = &[
        Self::FixedPrecision,
        Self::LsqLearnedStepSize,
        Self::PactActivationClipping,
        Self::FracBitsMixedBitwidth,
        Self::LqNetLearnableLevels,
        Self::BitNetTernaryQat,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            Self::FixedPrecision => "Fixed Precision (Standard PTQ / Static Format)",
            Self::LsqLearnedStepSize => "LSQ / LSQ+ (Learnable Step-Size s & Zero-Point β)",
            Self::PactActivationClipping => "PACT + DoReFa (Learnable Activation Clipping α)",
            Self::FracBitsMixedBitwidth => {
                "FracBits / HAWQ (Differentiable Per-Layer Bit-Width b_l)"
            }
            Self::LqNetLearnableLevels => {
                "LQ-Nets / AQLM (Learnable Non-Uniform Reconstruction Levels {q_k})"
            }
            Self::BitNetTernaryQat => "BitNet b1.58 Native Ternary {-1, 0, +1} QAT",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::FixedPrecision => {
                "Standard fixed-precision training without learnable quantization thresholds or dynamic bitwidth allocation."
            }
            Self::LsqLearnedStepSize => {
                "Learns quantization scale step-size s > 0 and asymmetric offset β via gradient descent with 1/sqrt(N*Q_p) gradient scaling (Esser et al., ICLR 2020)."
            }
            Self::PactActivationClipping => {
                "Replaces unbounded activations with a learnable clipping parameter α updated via analytical STE gradients dL/dα."
            }
            Self::FracBitsMixedBitwidth => {
                "Assigns a continuous differentiable bitwidth b_l in [min_bits, max_bits] to every layer, interpolating floor(b_l) and ceil(b_l) quantizers under a hardware BOPs budget."
            }
            Self::LqNetLearnableLevels => {
                "Jointly optimizes non-uniform quantization basis vectors and codebook reconstruction levels {q_1, ..., q_{2^b}} to match heavy-tailed weight distributions."
            }
            Self::BitNetTernaryQat => {
                "Trains multiplier-free ternary weights {-1, 0, +1} via absmean scaling γ = mean(|W|) alongside per-token absmax INT8 activations."
            }
        }
    }
}

/// Configuration for training the quantization level, step size, clipping bounds, and per-layer bitwidths.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantizationTrainingConfig {
    #[serde(default)]
    pub mode: QuantizationTrainingMode,
    /// Target average bitwidth across all quantized layers (e.g. 3.5 bits).
    #[serde(default = "default_target_bits")]
    pub target_avg_bits: f64,
    /// Minimum allowed layer bitwidth during mixed-precision search (e.g. 1.58 or 2.0).
    #[serde(default = "default_min_layer_bits")]
    pub min_layer_bits: f64,
    /// Maximum allowed layer bitwidth during mixed-precision search (e.g. 8.0).
    #[serde(default = "default_max_layer_bits")]
    pub max_layer_bits: f64,
    /// Regularization weight λ_BOPs penalizing Bit-Operations exceeding the target hardware budget.
    #[serde(default = "default_bops_lambda")]
    pub bops_penalty_lambda: f64,
    /// Whether to learn per-layer PACT activation clipping thresholds α.
    #[serde(default)]
    pub learn_activation_clip_alpha: bool,
    /// Whether to learn non-uniform reconstruction grid levels {q_k}.
    #[serde(default)]
    pub learn_non_uniform_grid: bool,
    /// Soft-to-hard annealing temperature τ for differentiable rounding tanh(τ(x - floor(x) - 0.5)).
    #[serde(default = "default_ste_temp")]
    pub ste_temperature: f64,
}

fn default_target_bits() -> f64 {
    4.0
}
fn default_min_layer_bits() -> f64 {
    1.58
}
fn default_max_layer_bits() -> f64 {
    8.0
}
fn default_bops_lambda() -> f64 {
    0.001
}
fn default_ste_temp() -> f64 {
    2.0
}

impl Default for QuantizationTrainingConfig {
    fn default() -> Self {
        Self {
            mode: QuantizationTrainingMode::FixedPrecision,
            target_avg_bits: default_target_bits(),
            min_layer_bits: default_min_layer_bits(),
            max_layer_bits: default_max_layer_bits(),
            bops_penalty_lambda: default_bops_lambda(),
            learn_activation_clip_alpha: false,
            learn_non_uniform_grid: false,
            ste_temperature: default_ste_temp(),
        }
    }
}

impl QuantizationTrainingConfig {
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }

    /// Estimated Bit-Operations (BOPs) compression factor relative to FP16xFP16 (256 BOPs/MAC).
    pub fn estimated_bops_compression(&self) -> f64 {
        let eff_bits = self.target_avg_bits.clamp(1.0, 16.0);
        (16.0 / eff_bits).powi(2)
    }
}

/// Complete training regime and multi-stage training plan specification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingSpec {
    #[serde(default = "default_true")]
    pub auto_partition_parameters: bool,
    #[serde(default)]
    pub optimizer_family: OptimizerFamily,
    #[serde(default)]
    pub muon: MuonConfig,
    #[serde(default)]
    pub adamw: AdamWConfig,
    #[serde(default)]
    pub lion: LionConfig,
    #[serde(default)]
    pub sophia: SophiaConfig,
    #[serde(default)]
    pub lbfgs: LbfgsConfig,
    #[serde(default)]
    pub scheduler: SchedulerSpec,
    #[serde(default)]
    pub hygiene: GradientHygieneSpec,
    #[serde(default)]
    pub loss_balancing: LossBalancingStrategy,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stages: Vec<TrainingStageSpec>,
    #[serde(default)]
    pub parallelism: ParallelismStrategySpec,
    #[serde(default)]
    pub curriculum: DatasetCurriculumSpec,
    #[serde(
        default,
        skip_serializing_if = "QuantizationTrainingConfig::is_default"
    )]
    pub quantization_training: QuantizationTrainingConfig,
}

impl Default for TrainingSpec {
    fn default() -> Self {
        Self {
            auto_partition_parameters: true,
            optimizer_family: OptimizerFamily::default(),
            muon: MuonConfig::default(),
            adamw: AdamWConfig::default(),
            lion: LionConfig::default(),
            sophia: SophiaConfig::default(),
            lbfgs: LbfgsConfig::default(),
            scheduler: SchedulerSpec::default(),
            hygiene: GradientHygieneSpec::default(),
            loss_balancing: LossBalancingStrategy::default(),
            stages: Vec::new(),
            parallelism: ParallelismStrategySpec::default(),
            curriculum: DatasetCurriculumSpec::default(),
            quantization_training: QuantizationTrainingConfig::default(),
        }
    }
}

/// Guided Training Plan Archetypes for the Training Planning Wizard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TrainingPlanPreset {
    #[default]
    FrontierLlm4StagePipeline,
    ChinchillaMuonWsdFastpath,
    GenerativeFlowProgressiveCurriculum,
    SciMlPhysicsNtkTwoStage,
    EmbodiedRoboticsVlaSafeRl,
    ConsumerGpuQloraDpo,
    SelfSupervisedVisionDinoMae,
    TabularGbdtVimeSelfSupervised,
    ZeroShotVoiceCloningFlowTts,
    RawBitByteMultiScaleLlm,
    SuperSmoothSobolevLipschitzCertified,
    LearnableQuantizationMixedBitwidthQat,
}

impl TrainingPlanPreset {
    pub const ALL: &'static [TrainingPlanPreset] = &[
        Self::FrontierLlm4StagePipeline,
        Self::ChinchillaMuonWsdFastpath,
        Self::GenerativeFlowProgressiveCurriculum,
        Self::SciMlPhysicsNtkTwoStage,
        Self::EmbodiedRoboticsVlaSafeRl,
        Self::ConsumerGpuQloraDpo,
        Self::SelfSupervisedVisionDinoMae,
        Self::TabularGbdtVimeSelfSupervised,
        Self::ZeroShotVoiceCloningFlowTts,
        Self::RawBitByteMultiScaleLlm,
        Self::SuperSmoothSobolevLipschitzCertified,
        Self::LearnableQuantizationMixedBitwidthQat,
    ];

    pub fn display_name(&self) -> &'static str {
        match self {
            Self::FrontierLlm4StagePipeline => {
                "Frontier LLM 4-Stage Pipeline (Pretrain -> YaRN 128K -> SFT -> GRPO RLVR)"
            }
            Self::ChinchillaMuonWsdFastpath => {
                "Chinchilla-Optimal Muon+AdamW WSD Fastpath (20 Tokens/Param)"
            }
            Self::GenerativeFlowProgressiveCurriculum => {
                "Generative Flow / Diffusion Progressive Resolution Curriculum (256 -> 512 -> 1024px)"
            }
            Self::SciMlPhysicsNtkTwoStage => {
                "SciML / PINN 2-Stage Physics Curriculum (AdamW Spectral -> Sobolev NTK)"
            }
            Self::EmbodiedRoboticsVlaSafeRl => {
                "Embodied Robotics VLA & World Model Pipeline (BC Demos -> RSSM -> Safe CBF-QP)"
            }
            Self::ConsumerGpuQloraDpo => {
                "Low-VRAM Consumer GPU Plan (Checkpointed LoRA/DoRA SFT -> DPO Alignment)"
            }
            Self::SelfSupervisedVisionDinoMae => {
                "Deep SSL 3-Stage Pipeline (MAE Masked Reconstruction -> DINOv2 + VICReg -> Probe)"
            }
            Self::TabularGbdtVimeSelfSupervised => {
                "Tabular GBDT Self-Supervised Pipeline (VIME Mask/Impute + Hastie Synthetic -> OOF GBDT)"
            }
            Self::ZeroShotVoiceCloningFlowTts => {
                "Zero-Shot Voice Cloning & Codec TTS (RVQ Codec Pretrain -> OT-CFM Flow -> Speaker Adapt)"
            }
            Self::RawBitByteMultiScaleLlm => {
                "Raw Bit & Byte Multi-Scale LLM (Bit-Plane Pretrain -> 64K Byte Window -> GRPO)"
            }
            Self::SuperSmoothSobolevLipschitzCertified => {
                "Super-Smooth C^∞ Sobolev H² & 1-Lipschitz Certified Curriculum (Björck -> L-BFGS)"
            }
            Self::LearnableQuantizationMixedBitwidthQat => {
                "Learnable Quantization Level & Mixed-Bitwidth QAT (LSQ+ -> FracBits BOPs -> BitNet/INT4)"
            }
        }
    }

    pub fn domain_tag(&self) -> &'static str {
        match self {
            Self::FrontierLlm4StagePipeline => "Multi-Stage LLM & Reasoning",
            Self::ChinchillaMuonWsdFastpath => "Compute-Optimal Pretraining",
            Self::GenerativeFlowProgressiveCurriculum => "Vision / Video / 3D / Audio",
            Self::SciMlPhysicsNtkTwoStage => "PDEs / Biomolecular / Weather",
            Self::EmbodiedRoboticsVlaSafeRl => "Robotics / World Models / MARL",
            Self::ConsumerGpuQloraDpo => "Single-GPU / Edge Efficiency",
            Self::SelfSupervisedVisionDinoMae => "Self-Supervised Learning (Neural)",
            Self::TabularGbdtVimeSelfSupervised => "Self-Supervised Learning (GBDT)",
            Self::ZeroShotVoiceCloningFlowTts => "Speech / Audio / Voice Cloning",
            Self::RawBitByteMultiScaleLlm => "Tokenizer-Free Raw Bits & Bytes",
            Self::SuperSmoothSobolevLipschitzCertified => "Super-Smooth C^∞ & Lipschitz ML",
            Self::LearnableQuantizationMixedBitwidthQat => "Learnable Quantization & Mixed-Bit QAT",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::FrontierLlm4StagePipeline => {
                "DeepSeek-V3/R1 & Llama-3 style 4-stage pipeline: Stage 1 Core Pretraining (4K), Stage 2 YaRN/Ring Long-Context Extension (64K), Stage 3 Instruction SFT, and Stage 4 Verifiable GRPO RL."
            }
            Self::ChinchillaMuonWsdFastpath => {
                "Single-stage compute-optimal training plan sizing token budget to 20x model parameters with Newton-Schulz Muon (2D matrices) + AdamW (1D/embeddings) under a Warmup-Stable-Decay (WSD) schedule."
            }
            Self::GenerativeFlowProgressiveCurriculum => {
                "3-stage progressive spatial/temporal curriculum starting at low resolution for global structure, scaling to high resolution with EMA weight averaging (0.9999), and finishing with flow-matching distillation."
            }
            Self::SciMlPhysicsNtkTwoStage => {
                "2-stage scientific ML regime: Stage 1 AdamW + Muon spectral warmup with Sobolev H1 loss, followed by Stage 2 low-LR NTK eigenvalue-balanced fine-tuning for machine-precision conservation laws."
            }
            Self::EmbodiedRoboticsVlaSafeRl => {
                "3-stage embodied control curriculum: Stage 1 Behavior Cloning on teleoperation trajectories, Stage 2 latent World Model imagination rollouts, and Stage 3 CBF-QP safety-certified policy alignment."
            }
            Self::ConsumerGpuQloraDpo => {
                "VRAM-frugal 2-stage fine-tuning plan for 24GB consumer GPUs: enables Activation Checkpointing, AutoClip, and DoRA/LoRA adapters for Stage 1 SFT followed by Stage 2 Direct Preference Optimization (DPO)."
            }
            Self::SelfSupervisedVisionDinoMae => {
                "3-stage neural Self-Supervised Learning curriculum: Stage 1 Masked Autoencoder (MAE 75% patch masking), Stage 2 DINOv2 EMA Student-Teacher + VICReg covariance anti-collapse, and Stage 3 Frozen-Backbone Task Head Tuning."
            }
            Self::TabularGbdtVimeSelfSupervised => {
                "3-stage Gradient Boosting Self-Supervised Learning regime: Stage 1 VIME Mask Vector Estimation & Marginal Imputation + Hastie Synthetic-vs-Real Discrimination, Stage 2 Unsupervised Tree-Leaf Embedding Extraction, and Stage 3 Out-of-Fold GBDT Fine-Tuning."
            }
            Self::ZeroShotVoiceCloningFlowTts => {
                "3-stage Speech & Voice Cloning regime: Stage 1 Multi-Stream RVQ Acoustic Codec Pretraining, Stage 2 Optimal-Transport Conditional Flow Matching (OT-CFM) with In-Context Speaker Prompts, and Stage 3 Expressive Prosody Distillation."
            }
            Self::RawBitByteMultiScaleLlm => {
                "3-stage Tokenizer-Free Bit/Byte curriculum: Stage 1 Raw Bit-Plane & Dynamic Entropy Byte-Patch Pretraining, Stage 2 64K Long-Context Byte Stream Extension, and Stage 3 Binary/Code Verification GRPO."
            }
            Self::SuperSmoothSobolevLipschitzCertified => {
                "3-stage Super-Smooth C^∞ & K-Lipschitz certified pipeline: Stage 1 Björck orthonormal + Softplus-β/GroupSort warmup, Stage 2 Sobolev H² Jacobian/Hessian curvature matching with NTK balancing, and Stage 3 L-BFGS quasi-Newton machine-precision polish."
            }
            Self::LearnableQuantizationMixedBitwidthQat => {
                "3-stage Learnable Quantization Level & Mixed-Bitwidth QAT pipeline: Stage 1 Full-precision teacher warmup, Stage 2 FracBits continuous bitwidth b_l search + LSQ+ step-size & PACT clipping optimization under BOPs penalty, and Stage 3 Hard-discrete BitNet/INT4 calibration."
            }
        }
    }

    /// Applies this guided training plan preset to the given `AiDesignSpec`, sizing token budgets,
    /// learning rates, parallelism degrees, and multi-stage curricula to the model and accelerator count.
    pub fn apply_to_spec(
        &self,
        spec: &mut crate::spec::AiDesignSpec,
        num_gpus: usize,
        seq_len_override: Option<usize>,
    ) {
        let gpus = num_gpus.max(1);
        let partition = crate::analysis::partition_parameters(spec);
        let total_params = partition.total_param_count.max(1_000_000) as u64;
        let chinchilla_tokens = (total_params * 20).max(1_000_000_000);
        let base_seq_len = seq_len_override.unwrap_or(4096).max(64);

        spec.training.auto_partition_parameters = true;

        match self {
            Self::FrontierLlm4StagePipeline => {
                spec.training.optimizer_family = OptimizerFamily::MuonAdamWPartitioned;
                spec.training.muon = MuonConfig {
                    lr: 0.02,
                    momentum: 0.95,
                    newton_schulz_steps: 5,
                    weight_decay: 0.01,
                };
                spec.training.adamw = AdamWConfig {
                    lr: 0.0003,
                    betas: (0.9, 0.95),
                    eps: 1e-8,
                    weight_decay: 0.01,
                };
                spec.training.scheduler = SchedulerSpec::WarmupStableDecay {
                    warmup_ratio: 0.03,
                    stable_ratio: 0.77,
                    decay_ratio: 0.20,
                    min_lr_ratio: 0.05,
                };
                spec.training.hygiene = GradientHygieneSpec {
                    auto_clip: true,
                    clip_percentile: 90.0,
                    max_norm: 1.0,
                    ema_decay: 0.999,
                    grokfast: false,
                    activation_checkpointing: gpus >= 4,
                };
                spec.training.loss_balancing = LossBalancingStrategy::FixedWeights;

                let tp = if gpus >= 8 { 4 } else { 1 };
                let dp = (gpus / tp).max(1);
                spec.training.parallelism = ParallelismStrategySpec {
                    sharding: if gpus > 1 {
                        ShardingStrategy::Fsdp2PerParamZero3
                    } else {
                        ShardingStrategy::SingleDevice
                    },
                    data_parallel_degree: dp,
                    tensor_parallel_degree: tp,
                    pipeline_parallel_degree: 1,
                    context_parallel_degree: 1,
                    expert_parallel_degree: if gpus >= 8 { 4 } else { 1 },
                    micro_batch_size: 4,
                    gradient_accumulation_steps: 16,
                    sequence_length: base_seq_len,
                };
                spec.training.curriculum = DatasetCurriculumSpec {
                    total_training_tokens: chinchilla_tokens * 3,
                    synthetic_data_ratio: 0.25,
                    sequence_packing: true,
                    curriculum_length_ramp: true,
                    checkpoint_interval_steps: 1000,
                    eval_interval_steps: 250,
                };
                spec.training.stages = vec![
                    TrainingStageSpec {
                        stage_id: "stage_1_pretrain".to_string(),
                        stage_name: "Stage 1: Dense & MoE Foundation Pretraining".to_string(),
                        paradigm: TrainingParadigm::Pretraining,
                        token_budget: chinchilla_tokens * 2,
                        sequence_length: base_seq_len,
                        lr_multiplier: 1.0,
                        freeze_policy: ParameterFreezePolicy::FullTrainable,
                    },
                    TrainingStageSpec {
                        stage_id: "stage_2_context_ext".to_string(),
                        stage_name: "Stage 2: YaRN / Ring Long-Context Extension".to_string(),
                        paradigm: TrainingParadigm::ContextExtension,
                        token_budget: chinchilla_tokens / 2,
                        sequence_length: base_seq_len * 8,
                        lr_multiplier: 0.25,
                        freeze_policy: ParameterFreezePolicy::FullTrainable,
                    },
                    TrainingStageSpec {
                        stage_id: "stage_3_sft".to_string(),
                        stage_name: "Stage 3: Curated Instruction & Reasoning SFT".to_string(),
                        paradigm: TrainingParadigm::SupervisedFineTuning,
                        token_budget: (chinchilla_tokens / 20).max(50_000_000),
                        sequence_length: base_seq_len * 4,
                        lr_multiplier: 0.05,
                        freeze_policy: ParameterFreezePolicy::FullTrainable,
                    },
                    TrainingStageSpec {
                        stage_id: "stage_4_grpo_rlvr".to_string(),
                        stage_name: "Stage 4: Verifiable Outcome Reward GRPO Alignment".to_string(),
                        paradigm: TrainingParadigm::ReinforcementLearningGrpo,
                        token_budget: (chinchilla_tokens / 40).max(25_000_000),
                        sequence_length: base_seq_len * 4,
                        lr_multiplier: 0.01,
                        freeze_policy: ParameterFreezePolicy::FullTrainable,
                    },
                ];
            }

            Self::ChinchillaMuonWsdFastpath => {
                spec.training.optimizer_family = OptimizerFamily::MuonAdamWPartitioned;
                spec.training.muon = MuonConfig {
                    lr: 0.02,
                    momentum: 0.95,
                    newton_schulz_steps: 5,
                    weight_decay: 0.01,
                };
                spec.training.adamw = AdamWConfig {
                    lr: 0.001,
                    betas: (0.9, 0.95),
                    eps: 1e-8,
                    weight_decay: 0.01,
                };
                spec.training.scheduler = SchedulerSpec::WarmupStableDecay {
                    warmup_ratio: 0.05,
                    stable_ratio: 0.75,
                    decay_ratio: 0.20,
                    min_lr_ratio: 0.01,
                };
                spec.training.hygiene = GradientHygieneSpec {
                    auto_clip: true,
                    clip_percentile: 90.0,
                    max_norm: 1.0,
                    ema_decay: 0.999,
                    grokfast: true,
                    activation_checkpointing: false,
                };
                spec.training.loss_balancing = LossBalancingStrategy::FixedWeights;
                spec.training.parallelism = ParallelismStrategySpec {
                    sharding: if gpus > 1 {
                        ShardingStrategy::Zero2GradAndOptim
                    } else {
                        ShardingStrategy::SingleDevice
                    },
                    data_parallel_degree: gpus,
                    tensor_parallel_degree: 1,
                    pipeline_parallel_degree: 1,
                    context_parallel_degree: 1,
                    expert_parallel_degree: 1,
                    micro_batch_size: 8,
                    gradient_accumulation_steps: 8,
                    sequence_length: base_seq_len,
                };
                spec.training.curriculum = DatasetCurriculumSpec {
                    total_training_tokens: chinchilla_tokens,
                    synthetic_data_ratio: 0.15,
                    sequence_packing: true,
                    curriculum_length_ramp: false,
                    checkpoint_interval_steps: 1000,
                    eval_interval_steps: 200,
                };
                spec.training.stages = vec![TrainingStageSpec {
                    stage_id: "stage_1_chinchilla_wsd".to_string(),
                    stage_name: "Stage 1: Chinchilla-Optimal Muon+AdamW WSD Run".to_string(),
                    paradigm: TrainingParadigm::Pretraining,
                    token_budget: chinchilla_tokens,
                    sequence_length: base_seq_len,
                    lr_multiplier: 1.0,
                    freeze_policy: ParameterFreezePolicy::FullTrainable,
                }];
            }

            Self::GenerativeFlowProgressiveCurriculum => {
                spec.training.optimizer_family = OptimizerFamily::AdamWStandard;
                spec.training.muon = MuonConfig {
                    lr: 0.015,
                    momentum: 0.95,
                    newton_schulz_steps: 5,
                    weight_decay: 0.01,
                };
                spec.training.adamw = AdamWConfig {
                    lr: 0.0002,
                    betas: (0.9, 0.99),
                    eps: 1e-8,
                    weight_decay: 0.01,
                };
                spec.training.scheduler = SchedulerSpec::CosineWithWarmup {
                    warmup_ratio: 0.05,
                    min_lr_ratio: 0.05,
                    warmup_steps: Some(2000),
                    total_steps: Some(250_000),
                    min_lr: Some(1e-5),
                };
                spec.training.hygiene = GradientHygieneSpec {
                    auto_clip: true,
                    clip_percentile: 90.0,
                    max_norm: 1.0,
                    ema_decay: 0.9999,
                    grokfast: false,
                    activation_checkpointing: true,
                };
                spec.training.loss_balancing = LossBalancingStrategy::FixedWeights;
                spec.training.parallelism = ParallelismStrategySpec {
                    sharding: if gpus > 1 {
                        ShardingStrategy::Fsdp2PerParamZero3
                    } else {
                        ShardingStrategy::SingleDevice
                    },
                    data_parallel_degree: gpus,
                    tensor_parallel_degree: 1,
                    pipeline_parallel_degree: 1,
                    context_parallel_degree: 1,
                    expert_parallel_degree: 1,
                    micro_batch_size: 4,
                    gradient_accumulation_steps: 8,
                    sequence_length: 1024,
                };
                spec.training.curriculum = DatasetCurriculumSpec {
                    total_training_tokens: chinchilla_tokens,
                    synthetic_data_ratio: 0.30,
                    sequence_packing: false,
                    curriculum_length_ramp: true,
                    checkpoint_interval_steps: 2500,
                    eval_interval_steps: 500,
                };
                spec.training.stages = vec![
                    TrainingStageSpec {
                        stage_id: "stage_1_low_res".to_string(),
                        stage_name: "Stage 1: Low-Resolution Global Flow Pretraining (256px)"
                            .to_string(),
                        paradigm: TrainingParadigm::Pretraining,
                        token_budget: (chinchilla_tokens * 6) / 10,
                        sequence_length: 256,
                        lr_multiplier: 1.0,
                        freeze_policy: ParameterFreezePolicy::FullTrainable,
                    },
                    TrainingStageSpec {
                        stage_id: "stage_2_high_res".to_string(),
                        stage_name: "Stage 2: High-Resolution Aspect-Bucketed Flow (1024px)"
                            .to_string(),
                        paradigm: TrainingParadigm::ContextExtension,
                        token_budget: (chinchilla_tokens * 3) / 10,
                        sequence_length: 1024,
                        lr_multiplier: 0.5,
                        freeze_policy: ParameterFreezePolicy::FreezeEmbeddingsAndEncoders,
                    },
                    TrainingStageSpec {
                        stage_id: "stage_3_cfg_distill".to_string(),
                        stage_name: "Stage 3: 4-Step Rectified Trajectory Distillation".to_string(),
                        paradigm: TrainingParadigm::KnowledgeDistillationQat,
                        token_budget: chinchilla_tokens / 10,
                        sequence_length: 1024,
                        lr_multiplier: 0.1,
                        freeze_policy: ParameterFreezePolicy::FullTrainable,
                    },
                ];
            }

            Self::SciMlPhysicsNtkTwoStage => {
                spec.training.optimizer_family = OptimizerFamily::LBFGSQuasiNewton;
                spec.training.muon = MuonConfig {
                    lr: 0.015,
                    momentum: 0.95,
                    newton_schulz_steps: 5,
                    weight_decay: 0.001,
                };
                spec.training.adamw = AdamWConfig {
                    lr: 0.0005,
                    betas: (0.9, 0.999),
                    eps: 1e-8,
                    weight_decay: 0.0001,
                };
                spec.training.scheduler = SchedulerSpec::CosineWithWarmup {
                    warmup_ratio: 0.05,
                    min_lr_ratio: 0.001,
                    warmup_steps: Some(1000),
                    total_steps: Some(100_000),
                    min_lr: Some(1e-6),
                };
                spec.training.hygiene = GradientHygieneSpec {
                    auto_clip: true,
                    clip_percentile: 95.0,
                    max_norm: 1.0,
                    ema_decay: 0.9995,
                    grokfast: true,
                    activation_checkpointing: false,
                };
                spec.training.loss_balancing = LossBalancingStrategy::NTKBalancing;
                spec.training.parallelism = ParallelismStrategySpec {
                    sharding: if gpus > 1 {
                        ShardingStrategy::DdpReplicated
                    } else {
                        ShardingStrategy::SingleDevice
                    },
                    data_parallel_degree: gpus,
                    tensor_parallel_degree: 1,
                    pipeline_parallel_degree: 1,
                    context_parallel_degree: 1,
                    expert_parallel_degree: 1,
                    micro_batch_size: 8,
                    gradient_accumulation_steps: 4,
                    sequence_length: 1024,
                };
                spec.training.curriculum = DatasetCurriculumSpec {
                    total_training_tokens: (chinchilla_tokens / 2).max(500_000_000),
                    synthetic_data_ratio: 0.50,
                    sequence_packing: false,
                    curriculum_length_ramp: true,
                    checkpoint_interval_steps: 1000,
                    eval_interval_steps: 200,
                };
                spec.training.stages = vec![
                    TrainingStageSpec {
                        stage_id: "stage_1_spectral_warmup".to_string(),
                        stage_name: "Stage 1: Muon+AdamW Spectral & Data Collocation Warmup"
                            .to_string(),
                        paradigm: TrainingParadigm::Pretraining,
                        token_budget: (chinchilla_tokens * 4) / 10,
                        sequence_length: 512,
                        lr_multiplier: 1.0,
                        freeze_policy: ParameterFreezePolicy::FullTrainable,
                    },
                    TrainingStageSpec {
                        stage_id: "stage_2_sobolev_ntk".to_string(),
                        stage_name:
                            "Stage 2: Sobolev H1 + NTK Eigenvalue PDE Conservation Refinement"
                                .to_string(),
                        paradigm: TrainingParadigm::PhysicsCurriculumSobolev,
                        token_budget: chinchilla_tokens / 10,
                        sequence_length: 1024,
                        lr_multiplier: 0.05,
                        freeze_policy: ParameterFreezePolicy::FullTrainable,
                    },
                ];
            }

            Self::EmbodiedRoboticsVlaSafeRl => {
                spec.training.optimizer_family = OptimizerFamily::AdamWStandard;
                spec.training.muon = MuonConfig {
                    lr: 0.015,
                    momentum: 0.95,
                    newton_schulz_steps: 5,
                    weight_decay: 0.01,
                };
                spec.training.adamw = AdamWConfig {
                    lr: 0.0003,
                    betas: (0.9, 0.99),
                    eps: 1e-8,
                    weight_decay: 0.01,
                };
                spec.training.scheduler = SchedulerSpec::WarmupStableDecay {
                    warmup_ratio: 0.05,
                    stable_ratio: 0.75,
                    decay_ratio: 0.20,
                    min_lr_ratio: 0.05,
                };
                spec.training.hygiene = GradientHygieneSpec {
                    auto_clip: true,
                    clip_percentile: 90.0,
                    max_norm: 1.0,
                    ema_decay: 0.999,
                    grokfast: false,
                    activation_checkpointing: false,
                };
                spec.training.loss_balancing = LossBalancingStrategy::PCGrad;
                spec.training.parallelism = ParallelismStrategySpec {
                    sharding: if gpus > 1 {
                        ShardingStrategy::Zero2GradAndOptim
                    } else {
                        ShardingStrategy::SingleDevice
                    },
                    data_parallel_degree: gpus,
                    tensor_parallel_degree: 1,
                    pipeline_parallel_degree: 1,
                    context_parallel_degree: 1,
                    expert_parallel_degree: 1,
                    micro_batch_size: 8,
                    gradient_accumulation_steps: 4,
                    sequence_length: 512,
                };
                spec.training.curriculum = DatasetCurriculumSpec {
                    total_training_tokens: chinchilla_tokens,
                    synthetic_data_ratio: 0.40,
                    sequence_packing: false,
                    curriculum_length_ramp: true,
                    checkpoint_interval_steps: 500,
                    eval_interval_steps: 100,
                };
                spec.training.stages = vec![
                    TrainingStageSpec {
                        stage_id: "stage_1_bc_demos".to_string(),
                        stage_name: "Stage 1: Multimodal Teleoperation Behavior Cloning (Flow/ACT)"
                            .to_string(),
                        paradigm: TrainingParadigm::SupervisedFineTuning,
                        token_budget: (chinchilla_tokens * 6) / 10,
                        sequence_length: 512,
                        lr_multiplier: 1.0,
                        freeze_policy: ParameterFreezePolicy::FreezeEmbeddingsAndEncoders,
                    },
                    TrainingStageSpec {
                        stage_id: "stage_2_safe_rl".to_string(),
                        stage_name: "Stage 2: Latent World-Model & Safe CBF-QP Policy Alignment"
                            .to_string(),
                        paradigm: TrainingParadigm::ReinforcementLearningGrpo,
                        token_budget: (chinchilla_tokens * 4) / 10,
                        sequence_length: 512,
                        lr_multiplier: 0.1,
                        freeze_policy: ParameterFreezePolicy::FullTrainable,
                    },
                ];
            }

            Self::ConsumerGpuQloraDpo => {
                spec.training.muon = MuonConfig {
                    lr: 0.01,
                    momentum: 0.95,
                    newton_schulz_steps: 5,
                    weight_decay: 0.01,
                };
                spec.training.adamw = AdamWConfig {
                    lr: 0.0002,
                    betas: (0.9, 0.95),
                    eps: 1e-8,
                    weight_decay: 0.01,
                };
                spec.training.scheduler = SchedulerSpec::CosineWithWarmup {
                    warmup_ratio: 0.05,
                    min_lr_ratio: 0.1,
                    warmup_steps: Some(200),
                    total_steps: Some(10_000),
                    min_lr: Some(2e-5),
                };
                spec.training.hygiene = GradientHygieneSpec {
                    auto_clip: true,
                    clip_percentile: 90.0,
                    max_norm: 0.5,
                    ema_decay: 0.995,
                    grokfast: false,
                    activation_checkpointing: true,
                };
                spec.training.loss_balancing = LossBalancingStrategy::FixedWeights;
                spec.training.parallelism = ParallelismStrategySpec {
                    sharding: ShardingStrategy::SingleDevice,
                    data_parallel_degree: 1,
                    tensor_parallel_degree: 1,
                    pipeline_parallel_degree: 1,
                    context_parallel_degree: 1,
                    expert_parallel_degree: 1,
                    micro_batch_size: 2,
                    gradient_accumulation_steps: 16,
                    sequence_length: 2048,
                };
                spec.training.curriculum = DatasetCurriculumSpec {
                    total_training_tokens: 200_000_000,
                    synthetic_data_ratio: 0.35,
                    sequence_packing: true,
                    curriculum_length_ramp: false,
                    checkpoint_interval_steps: 250,
                    eval_interval_steps: 50,
                };
                spec.training.stages = vec![
                    TrainingStageSpec {
                        stage_id: "stage_1_dora_sft".to_string(),
                        stage_name: "Stage 1: Checkpointed DoRA / QLoRA Instruction SFT"
                            .to_string(),
                        paradigm: TrainingParadigm::SupervisedFineTuning,
                        token_budget: 150_000_000,
                        sequence_length: 2048,
                        lr_multiplier: 1.0,
                        freeze_policy: ParameterFreezePolicy::LoraDoraAdapters,
                    },
                    TrainingStageSpec {
                        stage_id: "stage_2_dpo_align".to_string(),
                        stage_name: "Stage 2: Reference-Free / Contrastive DPO Alignment"
                            .to_string(),
                        paradigm: TrainingParadigm::PreferenceAlignmentDpo,
                        token_budget: 50_000_000,
                        sequence_length: 2048,
                        lr_multiplier: 0.1,
                        freeze_policy: ParameterFreezePolicy::LoraDoraAdapters,
                    },
                ];
            }

            Self::SelfSupervisedVisionDinoMae => {
                spec.training.optimizer_family = OptimizerFamily::AdamWStandard;
                spec.training.adamw = AdamWConfig {
                    lr: 0.0004,
                    betas: (0.9, 0.999),
                    eps: 1e-8,
                    weight_decay: 0.04,
                };
                spec.training.scheduler = SchedulerSpec::CosineWithWarmup {
                    warmup_ratio: 0.08,
                    min_lr_ratio: 0.01,
                    warmup_steps: Some(4000),
                    total_steps: Some(150_000),
                    min_lr: Some(1e-6),
                };
                spec.training.hygiene = GradientHygieneSpec {
                    auto_clip: true,
                    clip_percentile: 90.0,
                    max_norm: 3.0,
                    ema_decay: 0.996,
                    grokfast: false,
                    activation_checkpointing: gpus >= 4,
                };
                spec.training.loss_balancing = LossBalancingStrategy::UncertaintyKendallGal;
                spec.training.parallelism = ParallelismStrategySpec {
                    sharding: if gpus > 1 {
                        ShardingStrategy::Fsdp2PerParamZero3
                    } else {
                        ShardingStrategy::SingleDevice
                    },
                    data_parallel_degree: gpus,
                    tensor_parallel_degree: 1,
                    pipeline_parallel_degree: 1,
                    context_parallel_degree: 1,
                    expert_parallel_degree: 1,
                    micro_batch_size: 16,
                    gradient_accumulation_steps: 4,
                    sequence_length: 256,
                };
                spec.training.curriculum = DatasetCurriculumSpec {
                    total_training_tokens: chinchilla_tokens,
                    synthetic_data_ratio: 0.10,
                    sequence_packing: false,
                    curriculum_length_ramp: true,
                    checkpoint_interval_steps: 1000,
                    eval_interval_steps: 250,
                };
                spec.training.stages = vec![
                    TrainingStageSpec {
                        stage_id: "stage_1_mae_pretrain".to_string(),
                        stage_name: "Stage 1: Masked Autoencoder (MAE 75% Patch Reconstruction)"
                            .to_string(),
                        paradigm: TrainingParadigm::SelfSupervisedMaskedModeling,
                        token_budget: (chinchilla_tokens * 5) / 10,
                        sequence_length: 256,
                        lr_multiplier: 1.0,
                        freeze_policy: ParameterFreezePolicy::FullTrainable,
                    },
                    TrainingStageSpec {
                        stage_id: "stage_2_dinov2_vicreg".to_string(),
                        stage_name:
                            "Stage 2: DINOv2 EMA Student-Teacher + VICReg Covariance Regularization"
                                .to_string(),
                        paradigm: TrainingParadigm::SelfSupervisedStudentTeacher,
                        token_budget: (chinchilla_tokens * 4) / 10,
                        sequence_length: 512,
                        lr_multiplier: 0.5,
                        freeze_policy: ParameterFreezePolicy::FullTrainable,
                    },
                    TrainingStageSpec {
                        stage_id: "stage_3_probe_tune".to_string(),
                        stage_name: "Stage 3: Frozen-Backbone Task Head & Linear Probe Tuning"
                            .to_string(),
                        paradigm: TrainingParadigm::SupervisedFineTuning,
                        token_budget: chinchilla_tokens / 10,
                        sequence_length: 512,
                        lr_multiplier: 0.1,
                        freeze_policy: ParameterFreezePolicy::FreezeBackboneTrainHeads,
                    },
                ];
            }

            Self::TabularGbdtVimeSelfSupervised => {
                spec.training.optimizer_family = OptimizerFamily::AdamWStandard;
                spec.training.adamw = AdamWConfig {
                    lr: 0.001,
                    betas: (0.9, 0.999),
                    eps: 1e-8,
                    weight_decay: 0.001,
                };
                spec.training.scheduler = SchedulerSpec::CosineWithWarmup {
                    warmup_ratio: 0.05,
                    min_lr_ratio: 0.05,
                    warmup_steps: Some(500),
                    total_steps: Some(25_000),
                    min_lr: Some(1e-5),
                };
                spec.training.hygiene = GradientHygieneSpec {
                    auto_clip: true,
                    clip_percentile: 95.0,
                    max_norm: 1.0,
                    ema_decay: 0.995,
                    grokfast: true,
                    activation_checkpointing: false,
                };
                spec.training.loss_balancing = LossBalancingStrategy::UncertaintyKendallGal;
                spec.training.parallelism = ParallelismStrategySpec {
                    sharding: ShardingStrategy::SingleDevice,
                    data_parallel_degree: gpus,
                    tensor_parallel_degree: 1,
                    pipeline_parallel_degree: 1,
                    context_parallel_degree: 1,
                    expert_parallel_degree: 1,
                    micro_batch_size: 64,
                    gradient_accumulation_steps: 2,
                    sequence_length: 128,
                };
                spec.training.curriculum = DatasetCurriculumSpec {
                    total_training_tokens: 500_000_000,
                    synthetic_data_ratio: 0.50,
                    sequence_packing: false,
                    curriculum_length_ramp: false,
                    checkpoint_interval_steps: 500,
                    eval_interval_steps: 100,
                };
                spec.training.stages = vec![
                    TrainingStageSpec {
                        stage_id: "stage_1_vime_hastie_ssl".to_string(),
                        stage_name:
                            "Stage 1: VIME Mask/Imputation + Hastie Synthetic Marginal Discrimination"
                                .to_string(),
                        paradigm: TrainingParadigm::TabularSelfSupervisedVime,
                        token_budget: 300_000_000,
                        sequence_length: 128,
                        lr_multiplier: 1.0,
                        freeze_policy: ParameterFreezePolicy::FullTrainable,
                    },
                    TrainingStageSpec {
                        stage_id: "stage_2_oof_gbdt_boost".to_string(),
                        stage_name:
                            "Stage 2: Tree-Leaf Embedding & 5-Fold Out-of-Fold GBDT Fine-Tuning"
                                .to_string(),
                        paradigm: TrainingParadigm::TabularGradientBoostingCv,
                        token_budget: 200_000_000,
                        sequence_length: 128,
                        lr_multiplier: 0.25,
                        freeze_policy: ParameterFreezePolicy::FullTrainable,
                    },
                ];
            }

            Self::ZeroShotVoiceCloningFlowTts => {
                spec.training.optimizer_family = OptimizerFamily::MuonAdamWPartitioned;
                spec.training.muon = MuonConfig {
                    lr: 0.015,
                    momentum: 0.95,
                    newton_schulz_steps: 5,
                    weight_decay: 0.01,
                };
                spec.training.adamw = AdamWConfig {
                    lr: 0.0002,
                    betas: (0.9, 0.98),
                    eps: 1e-8,
                    weight_decay: 0.01,
                };
                spec.training.scheduler = SchedulerSpec::WarmupStableDecay {
                    warmup_ratio: 0.05,
                    stable_ratio: 0.75,
                    decay_ratio: 0.20,
                    min_lr_ratio: 0.05,
                };
                spec.training.hygiene = GradientHygieneSpec {
                    auto_clip: true,
                    clip_percentile: 90.0,
                    max_norm: 1.0,
                    ema_decay: 0.9999,
                    grokfast: false,
                    activation_checkpointing: true,
                };
                spec.training.loss_balancing = LossBalancingStrategy::GradNorm;
                spec.training.parallelism = ParallelismStrategySpec {
                    sharding: if gpus > 1 {
                        ShardingStrategy::Zero2GradAndOptim
                    } else {
                        ShardingStrategy::SingleDevice
                    },
                    data_parallel_degree: gpus,
                    tensor_parallel_degree: 1,
                    pipeline_parallel_degree: 1,
                    context_parallel_degree: 1,
                    expert_parallel_degree: 1,
                    micro_batch_size: 8,
                    gradient_accumulation_steps: 4,
                    sequence_length: 1500,
                };
                spec.training.curriculum = DatasetCurriculumSpec {
                    total_training_tokens: chinchilla_tokens,
                    synthetic_data_ratio: 0.15,
                    sequence_packing: true,
                    curriculum_length_ramp: true,
                    checkpoint_interval_steps: 1000,
                    eval_interval_steps: 250,
                };
                spec.training.stages = vec![
                    TrainingStageSpec {
                        stage_id: "stage_1_rvq_codec".to_string(),
                        stage_name:
                            "Stage 1: Multi-Stream RVQ Acoustic Codec & Speaker Timbre Pretraining"
                                .to_string(),
                        paradigm: TrainingParadigm::SelfSupervisedMaskedModeling,
                        token_budget: (chinchilla_tokens * 4) / 10,
                        sequence_length: 1024,
                        lr_multiplier: 1.0,
                        freeze_policy: ParameterFreezePolicy::FullTrainable,
                    },
                    TrainingStageSpec {
                        stage_id: "stage_2_ot_cfm_voice".to_string(),
                        stage_name:
                            "Stage 2: In-Context Reference Prompt OT-CFM Flow-Matching Synthesis"
                                .to_string(),
                        paradigm: TrainingParadigm::FlowMatchingRectifiedFlow,
                        token_budget: (chinchilla_tokens * 5) / 10,
                        sequence_length: 1500,
                        lr_multiplier: 0.5,
                        freeze_policy: ParameterFreezePolicy::FullTrainable,
                    },
                    TrainingStageSpec {
                        stage_id: "stage_3_prosody_dpo".to_string(),
                        stage_name: "Stage 3: Speaker Similarity & WER Preference Alignment (DPO)"
                            .to_string(),
                        paradigm: TrainingParadigm::PreferenceAlignmentDpo,
                        token_budget: chinchilla_tokens / 10,
                        sequence_length: 1500,
                        lr_multiplier: 0.05,
                        freeze_policy: ParameterFreezePolicy::FreezeEmbeddingsAndEncoders,
                    },
                ];
            }

            Self::RawBitByteMultiScaleLlm => {
                spec.training.optimizer_family = OptimizerFamily::MuonAdamWPartitioned;
                spec.training.muon = MuonConfig {
                    lr: 0.02,
                    momentum: 0.95,
                    newton_schulz_steps: 5,
                    weight_decay: 0.01,
                };
                spec.training.adamw = AdamWConfig {
                    lr: 0.0004,
                    betas: (0.9, 0.95),
                    eps: 1e-8,
                    weight_decay: 0.01,
                };
                spec.training.scheduler = SchedulerSpec::WarmupStableDecay {
                    warmup_ratio: 0.04,
                    stable_ratio: 0.76,
                    decay_ratio: 0.20,
                    min_lr_ratio: 0.05,
                };
                spec.training.hygiene = GradientHygieneSpec {
                    auto_clip: true,
                    clip_percentile: 90.0,
                    max_norm: 1.0,
                    ema_decay: 0.999,
                    grokfast: false,
                    activation_checkpointing: true,
                };
                spec.training.loss_balancing = LossBalancingStrategy::FixedWeights;
                spec.training.parallelism = ParallelismStrategySpec {
                    sharding: if gpus > 1 {
                        ShardingStrategy::Fsdp2PerParamZero3
                    } else {
                        ShardingStrategy::SingleDevice
                    },
                    data_parallel_degree: gpus,
                    tensor_parallel_degree: 1,
                    pipeline_parallel_degree: 1,
                    context_parallel_degree: 1,
                    expert_parallel_degree: 1,
                    micro_batch_size: 4,
                    gradient_accumulation_steps: 8,
                    sequence_length: 8192,
                };
                spec.training.curriculum = DatasetCurriculumSpec {
                    total_training_tokens: chinchilla_tokens * 2,
                    synthetic_data_ratio: 0.20,
                    sequence_packing: true,
                    curriculum_length_ramp: true,
                    checkpoint_interval_steps: 1000,
                    eval_interval_steps: 250,
                };
                spec.training.stages = vec![
                    TrainingStageSpec {
                        stage_id: "stage_1_raw_bit_byte".to_string(),
                        stage_name:
                            "Stage 1: Raw Bit-Plane & Entropy-Patched Byte Stream Pretraining (8K)"
                                .to_string(),
                        paradigm: TrainingParadigm::Pretraining,
                        token_budget: chinchilla_tokens,
                        sequence_length: 8192,
                        lr_multiplier: 1.0,
                        freeze_policy: ParameterFreezePolicy::FullTrainable,
                    },
                    TrainingStageSpec {
                        stage_id: "stage_2_byte_long_ctx".to_string(),
                        stage_name: "Stage 2: 64K Raw Byte / Binary File Context Extension"
                            .to_string(),
                        paradigm: TrainingParadigm::ContextExtension,
                        token_budget: (chinchilla_tokens * 8) / 10,
                        sequence_length: 65536,
                        lr_multiplier: 0.25,
                        freeze_policy: ParameterFreezePolicy::FullTrainable,
                    },
                    TrainingStageSpec {
                        stage_id: "stage_3_binary_grpo".to_string(),
                        stage_name: "Stage 3: Compiler / Binary Execution Verifiable Reward GRPO"
                            .to_string(),
                        paradigm: TrainingParadigm::ReinforcementLearningGrpo,
                        token_budget: (chinchilla_tokens * 2) / 10,
                        sequence_length: 16384,
                        lr_multiplier: 0.02,
                        freeze_policy: ParameterFreezePolicy::FullTrainable,
                    },
                ];
            }

            Self::SuperSmoothSobolevLipschitzCertified => {
                spec.training.optimizer_family = OptimizerFamily::LBFGSQuasiNewton;
                spec.training.adamw = AdamWConfig {
                    lr: 0.0005,
                    betas: (0.9, 0.999),
                    eps: 1e-10,
                    weight_decay: 0.0001,
                };
                spec.training.scheduler = SchedulerSpec::CosineWithWarmup {
                    warmup_ratio: 0.05,
                    min_lr_ratio: 0.001,
                    warmup_steps: Some(1000),
                    total_steps: Some(50_000),
                    min_lr: Some(1e-7),
                };
                spec.training.hygiene = GradientHygieneSpec {
                    auto_clip: true,
                    clip_percentile: 95.0,
                    max_norm: 1.0,
                    ema_decay: 0.9995,
                    grokfast: true,
                    activation_checkpointing: false,
                };
                spec.training.loss_balancing = LossBalancingStrategy::NTKBalancing;
                spec.training.parallelism = ParallelismStrategySpec {
                    sharding: ShardingStrategy::SingleDevice,
                    data_parallel_degree: gpus,
                    tensor_parallel_degree: 1,
                    pipeline_parallel_degree: 1,
                    context_parallel_degree: 1,
                    expert_parallel_degree: 1,
                    micro_batch_size: 16,
                    gradient_accumulation_steps: 2,
                    sequence_length: base_seq_len,
                };
                spec.training.curriculum = DatasetCurriculumSpec {
                    total_training_tokens: chinchilla_tokens,
                    synthetic_data_ratio: 0.50,
                    sequence_packing: false,
                    curriculum_length_ramp: true,
                    checkpoint_interval_steps: 500,
                    eval_interval_steps: 100,
                };
                spec.training.stages = vec![
                    TrainingStageSpec {
                        stage_id: "stage_1_bjorck_lipschitz_warmup".to_string(),
                        stage_name:
                            "Stage 1: Björck Orthonormal 1-Lipschitz & C^∞ Softplus-β/Padé Warmup"
                                .to_string(),
                        paradigm: TrainingParadigm::SuperSmoothSobolevLipschitz,
                        token_budget: (chinchilla_tokens * 5) / 10,
                        sequence_length: base_seq_len,
                        lr_multiplier: 1.0,
                        freeze_policy: ParameterFreezePolicy::FullTrainable,
                    },
                    TrainingStageSpec {
                        stage_id: "stage_2_sobolev_h2_ntk".to_string(),
                        stage_name:
                            "Stage 2: Higher-Order Sobolev H² Jacobian/Hessian Curvature Alignment"
                                .to_string(),
                        paradigm: TrainingParadigm::PhysicsCurriculumSobolev,
                        token_budget: (chinchilla_tokens * 4) / 10,
                        sequence_length: base_seq_len,
                        lr_multiplier: 0.2,
                        freeze_policy: ParameterFreezePolicy::FullTrainable,
                    },
                    TrainingStageSpec {
                        stage_id: "stage_3_lbfgs_polish".to_string(),
                        stage_name:
                            "Stage 3: Quasi-Newton L-BFGS Machine-Precision Residual Polish"
                                .to_string(),
                        paradigm: TrainingParadigm::SuperSmoothSobolevLipschitz,
                        token_budget: chinchilla_tokens / 10,
                        sequence_length: base_seq_len,
                        lr_multiplier: 0.05,
                        freeze_policy: ParameterFreezePolicy::FullTrainable,
                    },
                ];
            }

            Self::LearnableQuantizationMixedBitwidthQat => {
                spec.training.optimizer_family = OptimizerFamily::MuonAdamWPartitioned;
                spec.training.quantization_training = QuantizationTrainingConfig {
                    mode: QuantizationTrainingMode::FracBitsMixedBitwidth,
                    target_avg_bits: 3.25,
                    min_layer_bits: 1.58,
                    max_layer_bits: 8.0,
                    bops_penalty_lambda: 0.002,
                    learn_activation_clip_alpha: true,
                    learn_non_uniform_grid: true,
                    ste_temperature: 2.5,
                };
                spec.training.scheduler = SchedulerSpec::WarmupStableDecay {
                    warmup_ratio: 0.05,
                    stable_ratio: 0.65,
                    decay_ratio: 0.30,
                    min_lr_ratio: 0.02,
                };
                spec.training.hygiene = GradientHygieneSpec {
                    auto_clip: true,
                    clip_percentile: 90.0,
                    max_norm: 1.0,
                    ema_decay: 0.999,
                    grokfast: false,
                    activation_checkpointing: gpus >= 2,
                };
                spec.training.loss_balancing = LossBalancingStrategy::GradNorm;
                spec.training.parallelism = ParallelismStrategySpec {
                    sharding: if gpus > 1 {
                        ShardingStrategy::Zero2GradAndOptim
                    } else {
                        ShardingStrategy::SingleDevice
                    },
                    data_parallel_degree: gpus,
                    tensor_parallel_degree: 1,
                    pipeline_parallel_degree: 1,
                    context_parallel_degree: 1,
                    expert_parallel_degree: 1,
                    micro_batch_size: 8,
                    gradient_accumulation_steps: 4,
                    sequence_length: base_seq_len,
                };
                spec.training.curriculum = DatasetCurriculumSpec {
                    total_training_tokens: chinchilla_tokens,
                    synthetic_data_ratio: 0.25,
                    sequence_packing: true,
                    curriculum_length_ramp: false,
                    checkpoint_interval_steps: 500,
                    eval_interval_steps: 200,
                };
                spec.training.stages = vec![
                    TrainingStageSpec {
                        stage_id: "stage_1_fp_warmup".to_string(),
                        stage_name:
                            "Stage 1: BF16 Master Weight & LSQ+ Scale Statistics Initialization"
                                .to_string(),
                        paradigm: TrainingParadigm::Pretraining,
                        token_budget: (chinchilla_tokens * 3) / 10,
                        sequence_length: base_seq_len,
                        lr_multiplier: 1.0,
                        freeze_policy: ParameterFreezePolicy::FullTrainable,
                    },
                    TrainingStageSpec {
                        stage_id: "stage_2_fracbits_lsq_search".to_string(),
                        stage_name:
                            "Stage 2: Differentiable FracBits Bitwidth b_l + LSQ+ Step-Size & PACT Search"
                                .to_string(),
                        paradigm: TrainingParadigm::LearnableQuantizationBitwidthQat,
                        token_budget: (chinchilla_tokens * 5) / 10,
                        sequence_length: base_seq_len,
                        lr_multiplier: 0.5,
                        freeze_policy: ParameterFreezePolicy::FullTrainable,
                    },
                    TrainingStageSpec {
                        stage_id: "stage_3_hard_bitwidth_qat".to_string(),
                        stage_name:
                            "Stage 3: Frozen Discrete Bitwidth Allocation & Self-Distilled QAT Fine-Tuning"
                                .to_string(),
                        paradigm: TrainingParadigm::KnowledgeDistillationQat,
                        token_budget: (chinchilla_tokens * 2) / 10,
                        sequence_length: base_seq_len,
                        lr_multiplier: 0.1,
                        freeze_policy: ParameterFreezePolicy::FullTrainable,
                    },
                ];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_training_plan_presets_configure_stages_and_roundtrip() {
        for &preset in TrainingPlanPreset::ALL {
            let mut spec = crate::spec::archetypes::scaffold_archetype(
                crate::spec::archetypes::ArchitectureArchetype::GenerativeLLM,
                "test_training_plan",
                "Test Training Plan",
                512,
                4,
                crate::spec::precision::PrecisionFormat::BF16,
            );
            preset.apply_to_spec(&mut spec, 8, Some(4096));
            assert!(
                !spec.training.stages.is_empty(),
                "Preset {:?} should populate stages",
                preset
            );
            assert!(spec.training.parallelism.global_batch_tokens() > 0);
            let yaml = crate::format::aid_yaml::to_yaml_string(&spec).unwrap();
            let parsed = crate::format::aid_yaml::from_yaml_str(&yaml).unwrap();
            assert_eq!(spec, parsed);
        }
    }
}
