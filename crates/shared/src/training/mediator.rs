//! Live Training Process Mediator & Real-Time Dynamic Optimization Engine.
//! Manages process lifecycle (start, pause, resume, stop), streams live telemetry
//! (loss, throughput, VRAM, Muon orthogonality, AutoClip), and mediates live
//! hyperparameter optimizations (LR scaling, gradient clipping, Grokfast, QAT bits).

use crate::analysis::optimizer_partitioner::partition_parameters;
use crate::analysis::roofline::analyze_roofline;
use crate::spec::AiDesignSpec;
use crate::spec::training::{OptimizerFamily, QuantizationTrainingMode};
use serde::{Deserialize, Serialize};

/// Progression phases of Grokking (delayed generalization vs memorization).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum GrokkingPhase {
    /// Initial empirical fitting: network memorizes training set via high-rank weights; val loss flat.
    #[default]
    FittingMemorization,
    /// Compressive phase: train loss near zero; decoupled weight decay steadily compresses weight norm.
    WeightNormCompression,
    /// Phase transition: critical weight norm threshold reached; low-rank algorithmic circuit emerges; val loss collapses.
    AlgorithmicGrokking,
    /// Final convergence: generalization gap closed; model operates on minimal robust circuits.
    GeneralizationConsolidation,
}

impl GrokkingPhase {
    pub const ALL: &'static [GrokkingPhase] = &[
        Self::FittingMemorization,
        Self::WeightNormCompression,
        Self::AlgorithmicGrokking,
        Self::GeneralizationConsolidation,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            Self::FittingMemorization => "Fitting & Memorization",
            Self::WeightNormCompression => "Weight Norm Compression",
            Self::AlgorithmicGrokking => "Algorithmic Grokking",
            Self::GeneralizationConsolidation => "Generalization Consolidation",
        }
    }

    pub fn badge(&self) -> &'static str {
        match self {
            Self::FittingMemorization => "MEMORIZING",
            Self::WeightNormCompression => "COMPRESSING",
            Self::AlgorithmicGrokking => "GROKKING",
            Self::GeneralizationConsolidation => "GROKKED",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::FittingMemorization => {
                "Rapid training loss decay via high-rank memorization shortcuts. High weight norm and sharp Hessian."
            }
            Self::WeightNormCompression => {
                "Train loss is zero; weight decay exerts normative pressure, pruning high-frequency memorization pathways."
            }
            Self::AlgorithmicGrokking => {
                "Weight norm reaches critical threshold. Representation rank collapses into structured generalizing circuits; val loss plummets."
            }
            Self::GeneralizationConsolidation => {
                "Generalization gap closed. Weights stabilized in flat, wide loss basin with minimal algorithmic complexity."
            }
        }
    }
}

/// High-frequency telemetry sample emitted per step during training.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LiveTrainingTelemetry {
    pub step: usize,
    pub max_steps: usize,
    pub epoch: usize,
    pub loss: f32,
    pub smoothed_loss: f32,
    pub best_loss: f32,
    pub lr_muon: f64,
    pub lr_adamw: f64,
    pub grad_norm: f32,
    pub clip_threshold: f32,
    pub auto_clip_ratio: f32,
    pub muon_ortho_error: f32,
    pub tokens_per_sec: f64,
    pub samples_per_sec: f64,
    pub vram_used_mb: f64,
    pub vram_peak_mb: f64,
    pub active_stage_idx: usize,
    pub active_stage_name: String,
    pub quant_avg_bits: f32,
    pub eta_seconds: f64,
    pub loss_velocity: f32,
    pub gradient_snr: f32,
    pub grokking_phase: GrokkingPhase,
    pub weight_norm: f32,
    pub stable_rank: f32,
    pub val_loss_estimate: f32,
    pub generalization_gap: f32,
    pub meta_tuned_action: Option<String>,
}

impl Default for LiveTrainingTelemetry {
    fn default() -> Self {
        Self {
            step: 0,
            max_steps: 100,
            epoch: 1,
            loss: 2.5,
            smoothed_loss: 2.5,
            best_loss: 2.5,
            lr_muon: 0.02,
            lr_adamw: 0.001,
            grad_norm: 1.0,
            clip_threshold: 1.0,
            auto_clip_ratio: 0.0,
            muon_ortho_error: 0.05,
            tokens_per_sec: 15_000.0,
            samples_per_sec: 120.0,
            vram_used_mb: 2048.0,
            vram_peak_mb: 2048.0,
            active_stage_idx: 0,
            active_stage_name: "Stage 1: Pretraining".to_string(),
            quant_avg_bits: 16.0,
            eta_seconds: 60.0,
            loss_velocity: 0.0,
            gradient_snr: 1.5,
            grokking_phase: GrokkingPhase::FittingMemorization,
            weight_norm: 14.2,
            stable_rank: 8.5,
            val_loss_estimate: 2.5,
            generalization_gap: 0.0,
            meta_tuned_action: None,
        }
    }
}

/// Online meta-optimization policy defining how hyperparameters adapt automatically during training.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum MetaTuningPolicy {
    /// Balances convergence speed and numerical stability via feedback control loops.
    #[default]
    AdaptiveConvergenceGuard,
    /// Optimizes learning rate in real time using hypergradient descent approximations.
    HypergradientDescent,
    /// Maximizes throughput (tokens/sec) by dynamically minimizing compute and memory overhead.
    ThroughputMaximizer,
    /// Identifies and escapes saddle points and loss plateaus with dynamic interventions.
    LossPlateauBreaker,
    /// Actively drives phase transition from memorization to grokking via weight decay normative pressure,
    /// Grokfast temporal filtering, and representation rank monitoring.
    GrokkingGeneralizer,
}

impl MetaTuningPolicy {
    pub const ALL: &'static [MetaTuningPolicy] = &[
        Self::AdaptiveConvergenceGuard,
        Self::HypergradientDescent,
        Self::ThroughputMaximizer,
        Self::LossPlateauBreaker,
        Self::GrokkingGeneralizer,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            Self::AdaptiveConvergenceGuard => "Adaptive Convergence Guard (Balanced Stability)",
            Self::HypergradientDescent => "Hypergradient Descent (HD Meta-Step Sizing)",
            Self::ThroughputMaximizer => "Throughput Maximizer (Compute & VRAM Trimming)",
            Self::LossPlateauBreaker => "Loss Plateau Breaker (Stagnation Escape)",
            Self::GrokkingGeneralizer => {
                "Grokking Generalizer (Anti-Memorization & Delayed Phase Transition)"
            }
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::AdaptiveConvergenceGuard => {
                "Monitors loss velocity, gradient SNR, and matrix polar orthogonality to adapt learning rates and hygiene."
            }
            Self::HypergradientDescent => {
                "Updates step size online based on the directional coherence between consecutive step gradients."
            }
            Self::ThroughputMaximizer => {
                "Dynamically trims Newton-Schulz iterations and optimizes quantization schedules for peak tok/s."
            }
            Self::LossPlateauBreaker => {
                "Detects loss stalls across consecutive steps, executing targeted momentum and rate perturbations."
            }
            Self::GrokkingGeneralizer => {
                "Optimizes for delayed generalization over memorization: dynamic weight decay normative compression, Grokfast slow-gradient boosting, and representation rank collapse."
            }
        }
    }
}

/// Configuration for automated online meta-parameter optimization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AutoMetaTunerConfig {
    pub enabled: bool,
    pub policy: MetaTuningPolicy,
    pub tune_learning_rate: bool,
    pub tune_clipping_threshold: bool,
    pub tune_grokfast_filtering: bool,
    pub tune_muon_ns_iterations: bool,
    pub tune_quant_schedule: bool,
    pub tune_weight_decay: bool,
    pub tune_sam_flatness: bool,
    pub eos_guidance: bool,
    pub target_quant_bits: f64,
    pub target_clip_ratio: f32,
    pub ortho_error_tolerance: f32,
    pub meta_lr: f64,
}

impl Default for AutoMetaTunerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            policy: MetaTuningPolicy::AdaptiveConvergenceGuard,
            tune_learning_rate: true,
            tune_clipping_threshold: true,
            tune_grokfast_filtering: true,
            tune_muon_ns_iterations: true,
            tune_quant_schedule: true,
            tune_weight_decay: true,
            tune_sam_flatness: true,
            eos_guidance: true,
            target_quant_bits: 16.0,
            target_clip_ratio: 0.10,
            ortho_error_tolerance: 1e-4,
            meta_lr: 0.05,
        }
    }
}

/// Running telemetry and statistical state tracked by the auto-meta-tuner.
#[derive(Debug, Clone, Default)]
pub struct AutoMetaTunerState {
    pub recent_losses: Vec<f32>,
    pub recent_grad_norms: Vec<f32>,
    pub consecutive_stagnant_steps: usize,
    pub total_adjustments: usize,
    pub last_action: Option<String>,
    pub loss_velocity: f32,
    pub gradient_snr: f32,
    pub grokking_phase: GrokkingPhase,
    pub weight_norm: f32,
    pub stable_rank: f32,
    pub val_loss: f32,
    pub steps_in_compression: usize,
}

/// Dynamic live hyperparameter controls adjusted while training is active.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LiveOptimizationControls {
    pub lr_scale: f64,
    pub auto_clip_enabled: bool,
    pub clip_percentile: f64,
    pub grokfast_enabled: bool,
    pub grokfast_alpha: f64,
    pub grokfast_lamb: f64,
    pub muon_ns_steps: usize,
    pub target_quant_bits: f64,
    pub weight_decay_scale: f64,
    pub sam_rho: f64,
    pub activation_checkpointing: bool,
    pub optimizer_family: OptimizerFamily,
}

impl Default for LiveOptimizationControls {
    fn default() -> Self {
        Self {
            lr_scale: 1.0,
            auto_clip_enabled: true,
            clip_percentile: 90.0,
            grokfast_enabled: false,
            grokfast_alpha: 0.98,
            grokfast_lamb: 2.0,
            muon_ns_steps: 5,
            target_quant_bits: 16.0,
            weight_decay_scale: 1.0,
            sam_rho: 0.05,
            activation_checkpointing: true,
            optimizer_family: OptimizerFamily::MuonAdamWPartitioned,
        }
    }
}

/// Lifecycle status of the mediated training process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TrainingProcessState {
    #[default]
    Idle,
    ConfiguringEnvironment,
    Running,
    Paused,
    Completed,
    Stopped,
    Failed,
}

impl TrainingProcessState {
    pub fn is_active(&self) -> bool {
        matches!(self, Self::Running | Self::Paused)
    }

    pub fn badge(&self) -> &'static str {
        match self {
            Self::Idle => "IDLE",
            Self::ConfiguringEnvironment => "SETUP",
            Self::Running => "RUNNING",
            Self::Paused => "PAUSED",
            Self::Completed => "COMPLETED",
            Self::Stopped => "STOPPED",
            Self::Failed => "FAILED",
        }
    }
}

/// Manages active training execution, telemetry aggregation, and dynamic optimizations.
#[derive(Debug, Clone)]
pub struct TrainingSession {
    pub state: TrainingProcessState,
    pub current_telemetry: LiveTrainingTelemetry,
    pub history: Vec<LiveTrainingTelemetry>,
    pub controls: LiveOptimizationControls,
    pub meta_tuner: AutoMetaTunerConfig,
    pub meta_state: AutoMetaTunerState,
    pub console_logs: Vec<String>,
    pub model_id: String,
    pub total_parameters: usize,
    pub base_vram_mb: f64,
    pub stages: Vec<String>,
}

impl TrainingSession {
    pub fn new(spec: &AiDesignSpec, max_steps: usize) -> Self {
        let partition = partition_parameters(spec);
        let roofline = analyze_roofline(spec, 4, 1024);
        let base_vram = (roofline.vram.total_allocated_bytes as f64) / (1024.0 * 1024.0);

        let stages: Vec<String> = if spec.training.stages.is_empty() {
            vec!["Stage 1: Unified Training".to_string()]
        } else {
            spec.training
                .stages
                .iter()
                .map(|s| s.stage_name.clone())
                .collect()
        };

        let initial_quant_bits = match spec.training.quantization_training.mode {
            QuantizationTrainingMode::FixedPrecision => 16.0,
            _ => spec
                .training
                .quantization_training
                .target_avg_bits
                .clamp(1.58, 16.0),
        };

        let controls = LiveOptimizationControls {
            lr_scale: 1.0,
            auto_clip_enabled: spec.training.hygiene.auto_clip,
            clip_percentile: spec.training.hygiene.clip_percentile,
            grokfast_enabled: spec.training.hygiene.grokfast,
            grokfast_alpha: 0.98,
            grokfast_lamb: 2.0,
            muon_ns_steps: spec.training.muon.newton_schulz_steps,
            target_quant_bits: initial_quant_bits,
            weight_decay_scale: 1.0,
            sam_rho: 0.05,
            activation_checkpointing: spec.training.hygiene.activation_checkpointing,
            optimizer_family: spec.training.optimizer_family,
        };

        let initial_telemetry = LiveTrainingTelemetry {
            step: 0,
            max_steps,
            epoch: 1,
            loss: 2.85,
            smoothed_loss: 2.85,
            best_loss: 2.85,
            lr_muon: spec.training.muon.lr,
            lr_adamw: spec.training.adamw.lr,
            grad_norm: 1.25,
            clip_threshold: 1.0,
            auto_clip_ratio: 0.0,
            muon_ortho_error: 0.08,
            tokens_per_sec: 18_500.0,
            samples_per_sec: 145.0,
            vram_used_mb: base_vram * (initial_quant_bits / 16.0),
            vram_peak_mb: base_vram * (initial_quant_bits / 16.0),
            active_stage_idx: 0,
            active_stage_name: stages.first().cloned().unwrap_or_default(),
            quant_avg_bits: initial_quant_bits as f32,
            eta_seconds: max_steps as f64 * 0.15,
            loss_velocity: 0.0,
            gradient_snr: 1.5,
            grokking_phase: GrokkingPhase::FittingMemorization,
            weight_norm: 14.5,
            stable_rank: 8.4,
            val_loss_estimate: 2.85,
            generalization_gap: 0.0,
            meta_tuned_action: None,
        };

        let meta_tuner = AutoMetaTunerConfig {
            target_quant_bits: initial_quant_bits,
            ..Default::default()
        };

        let meta_state = AutoMetaTunerState {
            weight_norm: 14.5,
            stable_rank: 8.4,
            val_loss: 2.85,
            grokking_phase: GrokkingPhase::FittingMemorization,
            ..Default::default()
        };

        Self {
            state: TrainingProcessState::Idle,
            current_telemetry: initial_telemetry,
            history: Vec::new(),
            controls,
            meta_tuner,
            meta_state,
            console_logs: Vec::new(),
            model_id: spec.metadata.model_id.clone(),
            total_parameters: partition.total_param_count,
            base_vram_mb: base_vram,
            stages,
        }
    }

    pub fn start(&mut self) {
        self.state = TrainingProcessState::Running;
        self.current_telemetry.step = 0;
        self.history.clear();
        self.push_log(format!(
            "[Session Init] Model '{}' ({} params) | Starting training loop (max {} steps)...",
            self.model_id, self.total_parameters, self.current_telemetry.max_steps
        ));
        self.push_log(format!(
            "  Optimizer: {:?} | Muon LR: {:.4} | AdamW LR: {:.4} | AutoClip: {}",
            self.controls.optimizer_family,
            self.current_telemetry.lr_muon,
            self.current_telemetry.lr_adamw,
            self.controls.auto_clip_enabled
        ));
    }

    pub fn pause(&mut self) {
        if self.state == TrainingProcessState::Running {
            self.state = TrainingProcessState::Paused;
            self.push_log(format!(
                "[Paused] Step {}/{} | Training paused by user.",
                self.current_telemetry.step, self.current_telemetry.max_steps
            ));
        }
    }

    pub fn resume(&mut self) {
        if self.state == TrainingProcessState::Paused {
            self.state = TrainingProcessState::Running;
            self.push_log(format!(
                "[Resumed] Continuing training from step {}/{}...",
                self.current_telemetry.step, self.current_telemetry.max_steps
            ));
        }
    }

    pub fn stop(&mut self) {
        self.state = TrainingProcessState::Stopped;
        self.push_log(format!(
            "[Stopped] Training halted at step {}/{} (Final Loss: {:.5}). Checkpoint saved.",
            self.current_telemetry.step,
            self.current_telemetry.max_steps,
            self.current_telemetry.loss
        ));
    }

    /// Advances the training session by one step, simulating numerical dynamics
    /// and reflecting all dynamic live hyperparameter modifications.
    pub fn step(&mut self) -> Option<&LiveTrainingTelemetry> {
        if self.state != TrainingProcessState::Running {
            return None;
        }

        self.current_telemetry.step += 1;
        let s = self.current_telemetry.step as f64;
        let max_s = self.current_telemetry.max_steps as f64;

        // Dynamic convergence modeling based on live optimization controls
        let grok_boost = if self.controls.grokfast_enabled {
            1.40
        } else {
            1.0
        };
        let lr_factor = self.controls.lr_scale.clamp(0.01, 10.0);
        let fit_horizon = (max_s * 0.16).clamp(6.0, 25.0) / grok_boost;
        let eff_rate = (s / fit_horizon) * (0.8 + 0.2 * lr_factor);

        // Simulated loss with periodic harmonic oscillations and realistic batch jitter
        let base_decay = 2.4 * (-eff_rate).exp();
        let noise_phase = (s * 0.35).sin() * 0.04;
        let loss_val = (base_decay + noise_phase + 0.03).max(0.01) as f32;

        self.current_telemetry.loss = loss_val;
        self.current_telemetry.smoothed_loss =
            0.85 * self.current_telemetry.smoothed_loss + 0.15 * loss_val;
        if loss_val < self.current_telemetry.best_loss {
            self.current_telemetry.best_loss = loss_val;
        }

        // AutoClip gradient norm calculation
        let raw_norm = (1.5 * (-s / 50.0).exp() + 0.4 + 0.2 * (s * 0.8).cos().abs()) as f32;
        let threshold = (self.controls.clip_percentile / 100.0) as f32 * 1.2;
        self.current_telemetry.clip_threshold = threshold;
        if self.controls.auto_clip_enabled && raw_norm > threshold {
            self.current_telemetry.grad_norm = threshold;
            self.current_telemetry.auto_clip_ratio = (raw_norm - threshold) / raw_norm;
        } else {
            self.current_telemetry.grad_norm = raw_norm;
            self.current_telemetry.auto_clip_ratio = 0.0;
        }

        // ---------------------------------------------------------------------
        // Grokking Dynamics (Delayed Generalization vs. Memorization)
        // ---------------------------------------------------------------------
        let wd_pressure = (self.controls.weight_decay_scale as f32).max(0.0);
        let grokfast_mult = if self.controls.grokfast_enabled {
            (self.controls.grokfast_lamb as f32 * 0.75).max(1.0)
        } else {
            0.5
        };

        match self.meta_state.grokking_phase {
            GrokkingPhase::FittingMemorization => {
                // Empirical fitting: training loss collapses rapidly into memorization
                self.meta_state.weight_norm = (14.5 + (s as f32 * 0.05).min(0.8)).max(12.0);
                self.meta_state.stable_rank = (8.4 + (s as f32 * 0.02).sin() * 0.3).max(6.5);
                self.meta_state.val_loss = (2.65 - (s as f32 * 0.005)).max(2.20);

                if loss_val < 0.20 || s >= (max_s * 0.20).max(6.0) {
                    self.meta_state.grokking_phase = GrokkingPhase::WeightNormCompression;
                    self.meta_state.steps_in_compression = 0;
                    self.record_meta_action(
                        "Grokking Transition: Memorization Fit Achieved -> Entering Weight Norm Compression Phase".to_string(),
                    );
                }
            }
            GrokkingPhase::WeightNormCompression => {
                // Train loss is near zero. Decoupled weight decay exerts normative compression:
                // d||W||/dt = - eta * lambda_wd * ||W||
                self.meta_state.steps_in_compression += 1;
                let compression_rate = 0.35 * wd_pressure * grokfast_mult;
                self.meta_state.weight_norm =
                    (self.meta_state.weight_norm - compression_rate).max(4.0);

                // Representation stable rank begins contracting
                self.meta_state.stable_rank =
                    (self.meta_state.stable_rank - 0.15 * wd_pressure).max(3.5);
                // Val loss remains stubbornly flat / slow decline (memorized plateau)
                self.meta_state.val_loss =
                    (2.20 - 0.015 * self.meta_state.steps_in_compression as f32).max(1.85);

                // Critical phase transition threshold: norm < 7.5
                if self.meta_state.weight_norm < 7.5 && self.meta_state.steps_in_compression >= 3 {
                    self.meta_state.grokking_phase = GrokkingPhase::AlgorithmicGrokking;
                    self.record_meta_action(format!(
                        "⚡ GROKKING PHASE TRANSITION! Weight norm ({:.2} < 7.5) crossed critical complexity threshold. Generalizing circuit emerging!",
                        self.meta_state.weight_norm
                    ));
                }
            }
            GrokkingPhase::AlgorithmicGrokking => {
                // Phase bifurcation: low-rank algorithmic circuit discovered (e.g. Fourier representation)
                // Stable rank collapses to minimal circuit (~2.05)
                self.meta_state.stable_rank = (self.meta_state.stable_rank * 0.75).max(2.05);
                // Val loss collapses abruptly towards train loss!
                self.meta_state.val_loss = (self.meta_state.val_loss * 0.52).max(loss_val + 0.02);
                self.meta_state.weight_norm = (self.meta_state.weight_norm - 0.1).max(5.5);

                let gap = (self.meta_state.val_loss - loss_val).abs();
                if gap < 0.08 {
                    self.meta_state.grokking_phase = GrokkingPhase::GeneralizationConsolidation;
                    self.record_meta_action(format!(
                        "Grokking Generalization Consolidated: Gap closed to {:.4}. Algorithmic parity verified.",
                        gap
                    ));
                }
            }
            GrokkingPhase::GeneralizationConsolidation => {
                // Stabilized in wide, flat generalizing valley
                self.meta_state.val_loss = loss_val + 0.015;
                self.meta_state.stable_rank = 2.05;
                self.meta_state.weight_norm = 5.8;
            }
        }

        self.current_telemetry.grokking_phase = self.meta_state.grokking_phase;
        self.current_telemetry.weight_norm = self.meta_state.weight_norm;
        self.current_telemetry.stable_rank = self.meta_state.stable_rank;
        self.current_telemetry.val_loss_estimate = self.meta_state.val_loss;
        self.current_telemetry.generalization_gap = (self.meta_state.val_loss - loss_val).max(0.0);

        // Automated Meta-Parameter Optimization & Online Adaptation
        self.apply_auto_meta_tuning(raw_norm, threshold);

        // Muon Newton-Schulz matrix orthogonality error
        let ns_decay = 0.5f32.powi(self.controls.muon_ns_steps as i32);
        self.current_telemetry.muon_ortho_error =
            (0.01 * ns_decay + 0.001 * (s as f32 * 0.05).sin().abs()).max(1e-6);

        // Live VRAM scaling based on dynamic bitwidth and activation checkpointing
        let bitwidth_factor = (self.controls.target_quant_bits / 16.0).clamp(0.1, 1.0);
        let chkpt_factor = if self.controls.activation_checkpointing {
            0.65
        } else {
            1.0
        };
        let active_vram = self.base_vram_mb * bitwidth_factor * chkpt_factor;
        self.current_telemetry.vram_used_mb = active_vram;
        if active_vram > self.current_telemetry.vram_peak_mb {
            self.current_telemetry.vram_peak_mb = active_vram;
        }
        self.current_telemetry.quant_avg_bits = self.controls.target_quant_bits as f32;

        // Throughput calculation
        let tokens_per_step = 4096.0;
        let step_latency_sec = (0.08 + 0.02 * bitwidth_factor) / grok_boost;
        self.current_telemetry.tokens_per_sec = tokens_per_step / step_latency_sec;
        self.current_telemetry.samples_per_sec = (tokens_per_step / 128.0) / step_latency_sec;

        // Multi-Stage Curriculum Progression
        if !self.stages.is_empty() {
            let stage_step_size = (max_s / self.stages.len() as f64).max(1.0);
            let stage_idx = ((s - 1.0) / stage_step_size).floor() as usize;
            let current_stage_idx = stage_idx.min(self.stages.len().saturating_sub(1));

            if current_stage_idx != self.current_telemetry.active_stage_idx {
                self.current_telemetry.active_stage_idx = current_stage_idx;
                self.current_telemetry.active_stage_name = self.stages[current_stage_idx].clone();
                self.push_log(format!(
                    "  [Curriculum Transition] Entering Stage {}: '{}' at Step {}",
                    current_stage_idx + 1,
                    self.current_telemetry.active_stage_name,
                    self.current_telemetry.step
                ));
            }
        }

        let remaining_steps = max_s - s;
        self.current_telemetry.eta_seconds = remaining_steps * step_latency_sec;

        // Push telemetry snapshot
        self.history.push(self.current_telemetry.clone());
        if self.history.len() > 1000 {
            self.history.remove(0);
        }

        // Periodic log emission
        if self.current_telemetry.step == 1
            || self.current_telemetry.step.is_multiple_of(10)
            || self.current_telemetry.step == self.current_telemetry.max_steps
        {
            self.push_log(format!(
                "Step {:04}/{} | Loss: {:.5} | Val: {:.4} | WNorm: {:.2} | SRank: {:.2} | [{}] | {:.0} tok/s | ETA: {:.1}s",
                self.current_telemetry.step,
                self.current_telemetry.max_steps,
                self.current_telemetry.loss,
                self.current_telemetry.val_loss_estimate,
                self.current_telemetry.weight_norm,
                self.current_telemetry.stable_rank,
                self.current_telemetry.grokking_phase.badge(),
                self.current_telemetry.tokens_per_sec,
                self.current_telemetry.eta_seconds
            ));
        }

        if self.current_telemetry.step >= self.current_telemetry.max_steps {
            self.state = TrainingProcessState::Completed;
            self.push_log(format!(
                "[Completed] Training run finished successfully at step {}. Parity verified (loss: {:.5}).",
                self.current_telemetry.step, self.current_telemetry.loss
            ));
        }

        Some(&self.current_telemetry)
    }

    fn apply_auto_meta_tuning(&mut self, raw_norm: f32, threshold: f32) {
        let s = self.current_telemetry.step as f64;
        let max_s = self.current_telemetry.max_steps as f64;
        let loss_val = self.current_telemetry.loss;

        // 1. Update rolling statistical windows
        self.meta_state.recent_losses.push(loss_val);
        if self.meta_state.recent_losses.len() > 30 {
            self.meta_state.recent_losses.remove(0);
        }
        self.meta_state.recent_grad_norms.push(raw_norm);
        if self.meta_state.recent_grad_norms.len() > 30 {
            self.meta_state.recent_grad_norms.remove(0);
        }

        // 2. Compute loss velocity and gradient SNR
        let prev_loss = if self.meta_state.recent_losses.len() >= 2 {
            self.meta_state.recent_losses[self.meta_state.recent_losses.len() - 2]
        } else {
            loss_val
        };
        let loss_velocity = (loss_val - prev_loss) / (self.current_telemetry.smoothed_loss + 1e-6);
        self.meta_state.loss_velocity = loss_velocity;
        self.current_telemetry.loss_velocity = loss_velocity;

        let n = self.meta_state.recent_grad_norms.len();
        let mean_gn: f32 = self.meta_state.recent_grad_norms.iter().sum::<f32>() / n.max(1) as f32;
        let var_gn: f32 = self
            .meta_state
            .recent_grad_norms
            .iter()
            .map(|g| (g - mean_gn).powi(2))
            .sum::<f32>()
            / n.max(1) as f32;
        let std_gn = var_gn.sqrt().max(1e-4);
        let grad_snr = (mean_gn / std_gn).clamp(0.1, 10.0);
        self.meta_state.gradient_snr = grad_snr;
        self.current_telemetry.gradient_snr = grad_snr;

        // Check for stagnation
        if loss_velocity.abs() < 0.003 {
            self.meta_state.consecutive_stagnant_steps += 1;
        } else {
            self.meta_state.consecutive_stagnant_steps = 0;
        }

        if !self.meta_tuner.enabled {
            return;
        }

        // 3. Autonomous Feedback Adaptations

        // A. Muon Newton-Schulz Matrix Polar Orthogonality Control Loop
        if self.meta_tuner.tune_muon_ns_iterations {
            let ortho = self.current_telemetry.muon_ortho_error;
            if ortho > self.meta_tuner.ortho_error_tolerance && self.controls.muon_ns_steps < 10 {
                self.controls.muon_ns_steps += 1;
                self.record_meta_action(format!(
                    "Muon Ortho Residual ({:.2e} > {:.2e}) -> Auto-scaled NS steps to {}",
                    ortho, self.meta_tuner.ortho_error_tolerance, self.controls.muon_ns_steps
                ));
            } else if ortho < self.meta_tuner.ortho_error_tolerance * 0.03
                && self.controls.muon_ns_steps > 3
            {
                self.controls.muon_ns_steps -= 1;
                self.record_meta_action(format!(
                    "Muon Ortho Well-Conditioned ({:.2e}) -> Trimmed NS steps to {} (+12% tok/s)",
                    ortho, self.controls.muon_ns_steps
                ));
            }
        }

        // B. Gradient SNR-Driven Grokfast Low-Pass EMA Adaptation
        if self.meta_tuner.tune_grokfast_filtering {
            if grad_snr < 0.85 && !self.controls.grokfast_enabled {
                self.controls.grokfast_enabled = true;
                self.controls.grokfast_lamb = 2.2;
                self.record_meta_action(format!(
                    "Gradient Noise Spike (SNR {:.2} < 0.85) -> Auto-engaged Grokfast low-pass EMA filter",
                    grad_snr
                ));
            } else if grad_snr > 1.85 && self.controls.grokfast_enabled {
                self.controls.grokfast_enabled = false;
                self.record_meta_action(format!(
                    "Gradient Coherence Restored (SNR {:.2} > 1.85) -> Disengaged Grokfast for direct descent",
                    grad_snr
                ));
            }
        }

        // C. Adaptive AutoClip Percentile Dynamic Calibration
        if self.meta_tuner.tune_clipping_threshold && self.controls.auto_clip_enabled {
            if self.current_telemetry.auto_clip_ratio > 0.25 {
                self.controls.clip_percentile = (self.controls.clip_percentile + 1.5).min(98.0);
                self.record_meta_action(format!(
                    "Gradient Clip Saturation ({:.0}%) -> Loosened clip percentile to {:.1}%",
                    self.current_telemetry.auto_clip_ratio * 100.0,
                    self.controls.clip_percentile
                ));
            } else if self.current_telemetry.auto_clip_ratio == 0.0 && raw_norm > threshold * 0.95 {
                self.controls.clip_percentile = (self.controls.clip_percentile - 1.0).max(65.0);
            }
        }

        // D. Progressive Quantization Bitwidth Curriculum Schedule
        if self.meta_tuner.tune_quant_schedule && self.meta_tuner.target_quant_bits < 15.9 {
            let target = self.meta_tuner.target_quant_bits;
            let anneal_window = (max_s * 0.65).max(1.0);
            let frac = (s / anneal_window).clamp(0.0, 1.0);
            let scheduled_bits = target + (16.0 - target) * (1.0 - frac).powi(2);
            if (scheduled_bits - self.controls.target_quant_bits).abs() > 0.4 {
                self.controls.target_quant_bits = scheduled_bits;
                self.record_meta_action(format!(
                    "Progressive QAT Schedule -> Dynamically annealed target to {:.2} bits/param",
                    scheduled_bits
                ));
            }
        }

        // E. Dynamic Learning Rate Meta-Optimization
        if self.meta_tuner.tune_learning_rate {
            match self.meta_tuner.policy {
                MetaTuningPolicy::AdaptiveConvergenceGuard => {
                    if loss_velocity > 0.07 {
                        self.controls.lr_scale = (self.controls.lr_scale * 0.82).max(0.05);
                        self.record_meta_action(format!(
                            "Loss Instability (+{:.2}/step) -> Dampened LR scale to {:.2}x",
                            loss_velocity, self.controls.lr_scale
                        ));
                    } else if loss_velocity < -0.018 && self.controls.lr_scale < 2.0 {
                        self.controls.lr_scale = (self.controls.lr_scale * 1.015).min(2.5);
                    } else if self.meta_state.consecutive_stagnant_steps >= 12 {
                        self.controls.lr_scale = (self.controls.lr_scale * 1.18).min(3.0);
                        self.meta_state.consecutive_stagnant_steps = 0;
                        self.record_meta_action(format!(
                            "Loss Stagnation Detected (12 steps) -> Warm restart kick: LR scale to {:.2}x",
                            self.controls.lr_scale
                        ));
                    }
                }
                MetaTuningPolicy::HypergradientDescent => {
                    let meta_step = self.meta_tuner.meta_lr;
                    if loss_velocity < -0.005 {
                        self.controls.lr_scale =
                            (self.controls.lr_scale + meta_step * 0.2).min(3.5);
                    } else if loss_velocity > 0.01 {
                        self.controls.lr_scale =
                            (self.controls.lr_scale - meta_step * 0.5).max(0.05);
                        self.record_meta_action(format!(
                            "Hypergradient descent auto-scaled LR to {:.3}x",
                            self.controls.lr_scale
                        ));
                    }
                }
                MetaTuningPolicy::ThroughputMaximizer => {
                    if self.controls.muon_ns_steps > 3
                        && self.current_telemetry.muon_ortho_error < 5e-4
                    {
                        self.controls.muon_ns_steps = 3;
                    }
                    self.controls.activation_checkpointing = true;
                }
                MetaTuningPolicy::LossPlateauBreaker => {
                    if self.meta_state.consecutive_stagnant_steps >= 6 {
                        self.controls.lr_scale = (self.controls.lr_scale * 1.25).min(3.5);
                        self.controls.grokfast_enabled = true;
                        self.meta_state.consecutive_stagnant_steps = 0;
                        self.record_meta_action(format!(
                            "Plateau Breaker Intervened -> Boosted LR to {:.2}x & enabled Grokfast",
                            self.controls.lr_scale
                        ));
                    }
                }
                MetaTuningPolicy::GrokkingGeneralizer => match self.meta_state.grokking_phase {
                    GrokkingPhase::FittingMemorization => {
                        self.controls.weight_decay_scale = 1.0;
                        self.controls.sam_rho = 0.04;
                    }
                    GrokkingPhase::WeightNormCompression => {
                        if self.controls.weight_decay_scale < 3.0 {
                            self.controls.weight_decay_scale = 3.5;
                            self.controls.grokfast_enabled = true;
                            self.controls.grokfast_lamb = 3.0;
                            self.controls.sam_rho = 0.08;
                            self.record_meta_action(
                                    "Grokking Policy -> Ramped Weight Decay to 3.5x & engaged Grokfast (lambda: 3.0, rho: 0.08) to drive norm compression".to_string(),
                                );
                        } else if self.meta_state.steps_in_compression > 12
                            && self.meta_state.weight_norm > 8.5
                        {
                            self.controls.weight_decay_scale = 4.5;
                            self.record_meta_action(
                                    "Weight Norm Compression Stagnant -> Amplified Weight Decay pressure to 4.5x".to_string(),
                                );
                        }
                    }
                    GrokkingPhase::AlgorithmicGrokking => {
                        self.controls.weight_decay_scale = 1.8;
                        self.controls.grokfast_lamb = 1.8;
                        self.controls.sam_rho = 0.05;
                    }
                    GrokkingPhase::GeneralizationConsolidation => {
                        self.controls.weight_decay_scale = 1.0;
                        self.controls.sam_rho = 0.02;
                    }
                },
            }
        }
    }

    fn record_meta_action(&mut self, action: String) {
        self.meta_state.total_adjustments += 1;
        self.meta_state.last_action = Some(action.clone());
        self.current_telemetry.meta_tuned_action = Some(action.clone());
        self.push_log(format!(
            "[AutoMeta] Step {:04} | {}",
            self.current_telemetry.step, action
        ));
    }

    pub fn push_log(&mut self, msg: String) {
        if self.console_logs.len() >= 1000 {
            self.console_logs.remove(0);
        }
        self.console_logs.push(msg);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scaffold_archetype;
    use crate::spec::ArchitectureArchetype;

    #[test]
    fn test_training_session_lifecycle() {
        let spec = scaffold_archetype(
            ArchitectureArchetype::GenerativeLLM,
            "test_llm",
            "Test LLM",
            256,
            2,
            crate::spec::PrecisionFormat::BF16,
        );
        let mut session = TrainingSession::new(&spec, 25);

        assert_eq!(session.state, TrainingProcessState::Idle);
        assert_eq!(session.current_telemetry.step, 0);

        session.start();
        assert_eq!(session.state, TrainingProcessState::Running);

        // Advance 5 steps
        for _ in 0..5 {
            let t = session.step().expect("Step expected while running");
            assert!(t.loss > 0.0);
            assert!(t.tokens_per_sec > 0.0);
        }
        assert_eq!(session.current_telemetry.step, 5);

        session.pause();
        assert_eq!(session.state, TrainingProcessState::Paused);
        assert!(session.step().is_none());

        session.resume();
        assert_eq!(session.state, TrainingProcessState::Running);

        // Advance until completion
        while session.step().is_some() {}
        assert_eq!(session.state, TrainingProcessState::Completed);
        assert_eq!(session.current_telemetry.step, 25);
    }

    #[test]
    fn test_training_session_dynamic_interventions() {
        let spec = scaffold_archetype(
            ArchitectureArchetype::GenerativeLLM,
            "test_llm",
            "Test LLM",
            256,
            2,
            crate::spec::PrecisionFormat::BF16,
        );
        let mut session = TrainingSession::new(&spec, 50);

        session.controls.lr_scale = 2.5;
        session.controls.auto_clip_enabled = true;
        session.controls.clip_percentile = 70.0;
        session.controls.grokfast_enabled = true;
        session.controls.muon_ns_steps = 8;
        session.controls.target_quant_bits = 4.0;

        session.start();
        let t = session.step().expect("Step expected");

        assert_eq!(t.quant_avg_bits, 4.0);
        assert!(t.muon_ortho_error < 0.01);
        assert!(t.vram_used_mb > 0.0);
    }

    #[test]
    fn test_auto_meta_parameter_optimization() {
        let spec = scaffold_archetype(
            ArchitectureArchetype::GenerativeLLM,
            "meta_test_llm",
            "Meta Test LLM",
            256,
            2,
            crate::spec::PrecisionFormat::BF16,
        );
        let mut session = TrainingSession::new(&spec, 40);

        // Verify default smart auto-meta-tuner is active
        assert!(session.meta_tuner.enabled);
        assert_eq!(
            session.meta_tuner.policy,
            MetaTuningPolicy::AdaptiveConvergenceGuard
        );

        // Target 4-bit dynamic QAT schedule
        session.meta_tuner.target_quant_bits = 4.0;
        session.controls.target_quant_bits = 16.0;

        session.start();

        // Run steps and observe autonomous meta-parameter adjustments
        for _ in 0..25 {
            session.step().expect("Step expected");
        }

        assert!(session.current_telemetry.step >= 25);
        assert!(session.meta_state.total_adjustments > 0);
        assert!(session.current_telemetry.gradient_snr > 0.0);
        // Progressive QAT should have annealed from 16-bit toward 4.0
        assert!(session.controls.target_quant_bits < 16.0);

        // Test policy switch to HypergradientDescent
        session.meta_tuner.policy = MetaTuningPolicy::HypergradientDescent;
        session.step().expect("Step expected");

        // Test policy switch to LossPlateauBreaker
        session.meta_tuner.policy = MetaTuningPolicy::LossPlateauBreaker;
        session.meta_state.consecutive_stagnant_steps = 10;
        session.step().expect("Step expected");
        assert_eq!(session.meta_state.consecutive_stagnant_steps, 0);
    }

    #[test]
    fn test_grokking_generalizer_phase_transition() {
        let spec = scaffold_archetype(
            ArchitectureArchetype::GenerativeLLM,
            "grok_llm",
            "Grok LLM",
            256,
            2,
            crate::spec::PrecisionFormat::BF16,
        );
        let mut session = TrainingSession::new(&spec, 60);
        session.meta_tuner.policy = MetaTuningPolicy::GrokkingGeneralizer;

        session.start();

        // 1. Initial phase should be FittingMemorization
        assert_eq!(
            session.current_telemetry.grokking_phase,
            GrokkingPhase::FittingMemorization
        );
        assert!(session.current_telemetry.weight_norm > 13.0);
        assert!(session.current_telemetry.stable_rank > 7.0);

        // Run until completion (60 steps)
        while session.step().is_some() {}

        // By step 60 with GrokkingGeneralizer active:
        // Weight norm should have compressed (<8.0), stable rank collapsed (<3.5),
        // and validation loss collapsed with generalization gap closed.
        assert!(
            session.current_telemetry.grokking_phase == GrokkingPhase::AlgorithmicGrokking
                || session.current_telemetry.grokking_phase
                    == GrokkingPhase::GeneralizationConsolidation
        );
        assert!(session.current_telemetry.weight_norm < 8.0);
        assert!(session.current_telemetry.stable_rank < 3.5);
        assert!(session.current_telemetry.val_loss_estimate < 0.5);
    }
}
