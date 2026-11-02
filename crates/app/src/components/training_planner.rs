//! Interactive Training Plan Designer & Multi-Stage Compute Budget Planner.
//!
//! Provides a dedicated canvas mode (`CanvasMode::TrainingPlanner`) for designing
//! multi-stage training pipelines (Pretraining -> Context Extension -> SFT -> DPO/GRPO -> QAT),
//! 4D distributed parallelism (FSDP2/TP/PP/CP/EP), Chinchilla token budgets, and LR schedules.

use crate::icons::bi;
use crate::studio_state::StudioState;
use eframe::egui::{
    self, Color32, CornerRadius, Pos2, Rect, RichText, ScrollArea, Stroke, Ui, Vec2,
};
use shared::spec::training::{
    LossBalancingStrategy, ParameterFreezePolicy, SchedulerSpec, ShardingStrategy,
    TrainingParadigm, TrainingPlanPreset, TrainingStageSpec,
};

pub struct TrainingPlannerView {
    pub selected_preset: TrainingPlanPreset,
    pub preset_num_gpus: usize,
    pub preset_seq_len: usize,
}

impl Default for TrainingPlannerView {
    fn default() -> Self {
        Self {
            selected_preset: TrainingPlanPreset::FrontierLlm4StagePipeline,
            preset_num_gpus: 8,
            preset_seq_len: 4096,
        }
    }
}

impl TrainingPlannerView {
    pub fn render(&mut self, ui: &mut Ui, state: &mut StudioState) {
        let mut changed = false;

        ScrollArea::vertical()
            .id_salt("training_planner_scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.add_space(6.0);

                // 1. Header & Compute Budget Telemetry Banner
                self.render_compute_summary_banner(ui, state);

                ui.add_space(10.0);

                // 2. Quick-Apply Training Plan Wizard Bar
                changed |= self.render_preset_wizard_bar(ui, state);

                ui.add_space(10.0);

                // 3. Multi-Stage Training Curriculum Pipeline Designer
                changed |= self.render_multi_stage_pipeline(ui, state);

                ui.add_space(10.0);

                // 4. Two-Column Layout: Left = LR Schedule & Optimizer Partitioning, Right = 4D Parallelism & Dataset Curriculum
                ui.columns(2, |cols| {
                    changed |= self.render_schedule_and_optimizers(&mut cols[0], state);
                    changed |= self.render_parallelism_and_curriculum(&mut cols[1], state);
                });

                ui.add_space(16.0);
            });

        if changed {
            state.refresh_analyses();
        }
    }

    fn render_compute_summary_banner(&self, ui: &mut Ui, state: &StudioState) {
        let total_params = state.partition_report.total_param_count.max(1) as f64;
        let chinchilla_optimal_tokens = (total_params * 20.0) as u64;

        let planned_tokens: u64 = if state.spec.training.stages.is_empty() {
            state.spec.training.curriculum.total_training_tokens
        } else {
            state
                .spec
                .training
                .stages
                .iter()
                .map(|s| s.token_budget)
                .sum()
        };

        let chinchilla_ratio = planned_tokens as f64 / total_params;
        // C = 6 * N * D FLOPs for forward + backward pass
        let total_training_flops = 6.0 * total_params * (planned_tokens as f64);

        let num_accel = state.spec.training.parallelism.total_accelerators().max(1) as f64;
        let peak_tflops = state
            .spec
            .metadata
            .target_hardware
            .peak_fp16_tflops
            .max(1.0);
        // Assume 42% Model FLOPs Utilization (MFU) across cluster
        let cluster_effective_flops_per_sec = num_accel * peak_tflops * 1e12 * 0.42;
        let est_hours = (total_training_flops / cluster_effective_flops_per_sec) / 3600.0;

        let global_batch_tokens = state.spec.training.parallelism.global_batch_tokens().max(1);
        let total_steps = planned_tokens / (global_batch_tokens as u64).max(1);

        egui::Frame::group(ui.style())
            .fill(Color32::from_rgb(20, 26, 38))
            .stroke(Stroke::new(1.0, Color32::from_rgb(55, 85, 130)))
            .corner_radius(CornerRadius::same(8))
            .inner_margin(egui::Margin::symmetric(14, 10))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(format!(
                            "{} Training Plan & Compute Budget Designer",
                            bi::SPEEDOMETER2
                        ))
                        .size(16.0)
                        .color(Color32::from_rgb(110, 200, 255))
                        .strong(),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            RichText::new(format!(
                                "Target Hardware: {} × {}",
                                state.spec.training.parallelism.total_accelerators(),
                                state.spec.metadata.target_hardware.profile_name
                            ))
                            .size(11.5)
                            .monospace()
                            .color(Color32::from_rgb(140, 220, 165)),
                        );
                    });
                });

                ui.add_space(6.0);
                ui.separator();
                ui.add_space(6.0);

                ui.columns(6, |cols| {
                    metric_tile(
                        &mut cols[0],
                        "Active Parameters (N)",
                        &format_large_count(state.partition_report.total_param_count as u64),
                        &format!(
                            "{:.0}% Muon 2D / {:.0}% AdamW",
                            state.partition_report.muon_ratio * 100.0,
                            (1.0 - state.partition_report.muon_ratio) * 100.0
                        ),
                        Color32::from_rgb(110, 195, 255),
                    );
                    metric_tile(
                        &mut cols[1],
                        "Chinchilla Target (20N)",
                        &format!("{} tok", format_large_count(chinchilla_optimal_tokens)),
                        "Compute-optimal baseline",
                        Color32::from_rgb(175, 190, 215),
                    );
                    metric_tile(
                        &mut cols[2],
                        "Planned Token Budget (D)",
                        &format!("{} tok", format_large_count(planned_tokens)),
                        &format!("{chinchilla_ratio:.1} tokens / param"),
                        Color32::from_rgb(120, 230, 155),
                    );
                    metric_tile(
                        &mut cols[3],
                        "Global Batch / Step",
                        &format!("{} tok", format_large_count(global_batch_tokens as u64)),
                        &format!("{} optimizer steps", format_large_count(total_steps)),
                        Color32::from_rgb(245, 195, 95),
                    );
                    metric_tile(
                        &mut cols[4],
                        "Total Compute (6ND)",
                        &format!("{total_training_flops:.2e} FLOPs"),
                        "Forward + backward total",
                        Color32::from_rgb(205, 160, 255),
                    );
                    metric_tile(
                        &mut cols[5],
                        "Est. Wall-Clock Time",
                        &format_duration_hours(est_hours),
                        "@ 42% cluster MFU",
                        Color32::from_rgb(110, 225, 210),
                    );
                });
            });
    }

    fn render_preset_wizard_bar(&mut self, ui: &mut Ui, state: &mut StudioState) -> bool {
        let mut applied = false;
        egui::Frame::group(ui.style())
            .fill(Color32::from_rgb(24, 29, 42))
            .stroke(Stroke::new(1.0, Color32::from_rgb(70, 95, 140)))
            .corner_radius(CornerRadius::same(8))
            .inner_margin(egui::Margin::symmetric(12, 8))
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.label(
                        RichText::new(format!("{} Guided Training Plan Wizard:", bi::MAGIC))
                            .size(12.5)
                            .color(Color32::from_rgb(120, 200, 255))
                            .strong(),
                    );

                    egui::ComboBox::from_id_salt("training_plan_preset_combo")
                        .selected_text(self.selected_preset.display_name())
                        .width(390.0)
                        .show_ui(ui, |ui| {
                            for &preset in TrainingPlanPreset::ALL {
                                ui.selectable_value(
                                    &mut self.selected_preset,
                                    preset,
                                    format!("{} [{}]", preset.display_name(), preset.domain_tag()),
                                );
                            }
                        });

                    ui.separator();
                    ui.label("Cluster GPUs:");
                    ui.add(
                        egui::DragValue::new(&mut self.preset_num_gpus)
                            .range(1..=4096)
                            .speed(1),
                    );

                    ui.label("Base Seq Len:");
                    ui.add(
                        egui::DragValue::new(&mut self.preset_seq_len)
                            .range(64..=131072)
                            .speed(128),
                    );

                    if ui
                        .button(
                            RichText::new(format!(
                                "{} Auto-Configure Training Plan",
                                bi::LIGHTNING_CHARGE_FILL
                            ))
                            .color(Color32::from_rgb(120, 235, 155))
                            .strong(),
                        )
                        .on_hover_text(self.selected_preset.description())
                        .clicked()
                    {
                        self.selected_preset.apply_to_spec(
                            &mut state.spec,
                            self.preset_num_gpus,
                            Some(self.preset_seq_len),
                        );
                        applied = true;
                    }
                });
                ui.add_space(2.0);
                ui.label(
                    RichText::new(self.selected_preset.description())
                        .size(11.0)
                        .color(Color32::from_rgb(165, 180, 205)),
                );
            });
        applied
    }

    fn render_multi_stage_pipeline(&mut self, ui: &mut Ui, state: &mut StudioState) -> bool {
        let mut changed = false;

        egui::Frame::group(ui.style())
            .fill(Color32::from_rgb(18, 22, 32))
            .stroke(Stroke::new(1.0, Color32::from_rgb(48, 60, 82)))
            .corner_radius(CornerRadius::same(8))
            .inner_margin(egui::Margin::symmetric(12, 10))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(format!(
                            "{} Multi-Stage Training Curriculum Pipeline ({} Stages)",
                            bi::DIAGRAM_3,
                            state.spec.training.stages.len().max(1)
                        ))
                        .size(14.0)
                        .strong(),
                    );

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .button(
                                RichText::new(format!("{} Add Training Stage", bi::PLUS_LG))
                                    .color(Color32::from_rgb(115, 225, 150)),
                            )
                            .clicked()
                        {
                            let next_idx = state.spec.training.stages.len() + 1;
                            let default_paradigm = match next_idx {
                                1 => TrainingParadigm::Pretraining,
                                2 => TrainingParadigm::ContextExtension,
                                3 => TrainingParadigm::SupervisedFineTuning,
                                4 => TrainingParadigm::ReinforcementLearningGrpo,
                                _ => TrainingParadigm::KnowledgeDistillationQat,
                            };
                            state.spec.training.stages.push(TrainingStageSpec {
                                stage_id: format!("stage_{next_idx}"),
                                stage_name: format!(
                                    "Stage {next_idx}: {}",
                                    default_paradigm.label()
                                ),
                                paradigm: default_paradigm,
                                token_budget: 2_000_000_000,
                                sequence_length: state.spec.training.parallelism.sequence_length,
                                lr_multiplier: if next_idx == 1 { 1.0 } else { 0.2 },
                                freeze_policy: ParameterFreezePolicy::FullTrainable,
                            });
                            changed = true;
                        }

                        if !state.spec.training.stages.is_empty()
                            && ui
                                .button("Reset to Single-Stage")
                                .on_hover_text(
                                    "Clear explicit multi-stage list and use single-stage curriculum budget",
                                )
                                .clicked()
                        {
                            state.spec.training.stages.clear();
                            changed = true;
                        }
                    });
                });

                ui.add_space(6.0);

                if state.spec.training.stages.is_empty() {
                    ui.horizontal(|ui| {
                        ui.label(
                            RichText::new(
                                "Single-Stage Training Mode active. Click 'Auto-Configure Training Plan' above or '+ Add Training Stage' to build a multi-stage curriculum (Pretrain -> Context Extension -> SFT -> DPO/GRPO).",
                            )
                            .size(11.5)
                            .color(Color32::from_rgb(165, 178, 200)),
                        );
                    });
                    return;
                }

                let mut stage_to_delete: Option<usize> = None;
                let mut stage_to_move_up: Option<usize> = None;
                let mut stage_to_move_down: Option<usize> = None;
                let num_stages = state.spec.training.stages.len();

                for (idx, stage) in state.spec.training.stages.iter_mut().enumerate() {
                    let badge_color = match stage.paradigm {
                        TrainingParadigm::Pretraining => Color32::from_rgb(90, 175, 255),
                        TrainingParadigm::ContextExtension => Color32::from_rgb(165, 135, 255),
                        TrainingParadigm::SupervisedFineTuning => Color32::from_rgb(100, 220, 155),
                        TrainingParadigm::PreferenceAlignmentDpo => Color32::from_rgb(245, 180, 80),
                        TrainingParadigm::ReinforcementLearningGrpo => {
                            Color32::from_rgb(255, 120, 145)
                        }
                        TrainingParadigm::PhysicsCurriculumSobolev => {
                            Color32::from_rgb(90, 225, 220)
                        }
                        TrainingParadigm::KnowledgeDistillationQat => {
                            Color32::from_rgb(220, 205, 110)
                        }
                        TrainingParadigm::SelfSupervisedContrastive
                        | TrainingParadigm::SelfSupervisedMaskedModeling
                        | TrainingParadigm::SelfSupervisedJointEmbedding
                        | TrainingParadigm::SelfSupervisedAntiCollapse
                        | TrainingParadigm::SelfSupervisedStudentTeacher => {
                            Color32::from_rgb(130, 210, 255)
                        }
                        TrainingParadigm::TabularGradientBoostingCv
                        | TrainingParadigm::TabularSelfSupervisedVime => {
                            Color32::from_rgb(125, 235, 175)
                        }
                        TrainingParadigm::FlowMatchingRectifiedFlow => {
                            Color32::from_rgb(215, 145, 255)
                        }
                        TrainingParadigm::ReinforcementLearningPpo => {
                            Color32::from_rgb(255, 135, 115)
                        }
                        TrainingParadigm::ContinualLearningEwc => {
                            Color32::from_rgb(185, 215, 130)
                        }
                        TrainingParadigm::LearnableQuantizationBitwidthQat => {
                            Color32::from_rgb(255, 195, 95)
                        }
                        TrainingParadigm::SuperSmoothSobolevLipschitz => {
                            Color32::from_rgb(105, 235, 210)
                        }
                    };

                    egui::Frame::group(ui.style())
                        .fill(Color32::from_rgb(24, 30, 44))
                        .stroke(Stroke::new(1.0, badge_color.gamma_multiply(0.55)))
                        .corner_radius(CornerRadius::same(6))
                        .inner_margin(egui::Margin::symmetric(10, 8))
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.label(
                                    RichText::new(format!("STAGE {}", idx + 1))
                                        .size(11.0)
                                        .monospace()
                                        .color(badge_color)
                                        .strong(),
                                );
                                if ui.text_edit_singleline(&mut stage.stage_name).changed() {
                                    changed = true;
                                }

                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        if ui
                                            .small_button(
                                                RichText::new(bi::TRASH3)
                                                    .color(Color32::from_rgb(240, 110, 110)),
                                            )
                                            .on_hover_text("Remove this training stage")
                                            .clicked()
                                        {
                                            stage_to_delete = Some(idx);
                                        }
                                        if idx + 1 < num_stages
                                            && ui
                                                .small_button("↓")
                                                .on_hover_text("Move stage later")
                                                .clicked()
                                        {
                                            stage_to_move_down = Some(idx);
                                        }
                                        if idx > 0
                                            && ui
                                                .small_button("↑")
                                                .on_hover_text("Move stage earlier")
                                                .clicked()
                                        {
                                            stage_to_move_up = Some(idx);
                                        }
                                    },
                                );
                            });

                            ui.add_space(4.0);
                            ui.horizontal_wrapped(|ui| {
                                ui.label("Paradigm:");
                                egui::ComboBox::from_id_salt(format!("stage_paradigm_{idx}"))
                                    .selected_text(stage.paradigm.label())
                                    .width(240.0)
                                    .show_ui(ui, |ui| {
                                        for &p in TrainingParadigm::ALL {
                                            if ui
                                                .selectable_value(&mut stage.paradigm, p, p.label())
                                                .on_hover_text(p.description())
                                                .changed()
                                            {
                                                changed = true;
                                            }
                                        }
                                    })
                                    .response
                                    .on_hover_text(stage.paradigm.description());

                                ui.separator();
                                ui.label("Token Budget:");
                                if ui
                                    .add(
                                        egui::DragValue::new(&mut stage.token_budget)
                                            .range(1_000_000..=100_000_000_000_000u64)
                                            .speed(100_000_000.0),
                                    )
                                    .changed()
                                {
                                    changed = true;
                                }
                                ui.label(
                                    RichText::new(format!(
                                        "({})",
                                        format_large_count(stage.token_budget)
                                    ))
                                    .size(11.0)
                                    .color(Color32::from_rgb(145, 215, 165)),
                                );

                                ui.separator();
                                ui.label("Seq Len:");
                                if ui
                                    .add(
                                        egui::DragValue::new(&mut stage.sequence_length)
                                            .range(64..=1_048_576)
                                            .speed(128),
                                    )
                                    .changed()
                                {
                                    changed = true;
                                }

                                ui.separator();
                                ui.label("LR Scale:");
                                if ui
                                    .add(
                                        egui::DragValue::new(&mut stage.lr_multiplier)
                                            .range(0.001..=5.0)
                                            .speed(0.01),
                                    )
                                    .changed()
                                {
                                    changed = true;
                                }

                                ui.separator();
                                ui.label("Freeze / PEFT:");
                                egui::ComboBox::from_id_salt(format!("stage_freeze_{idx}"))
                                    .selected_text(stage.freeze_policy.label())
                                    .width(210.0)
                                    .show_ui(ui, |ui| {
                                        for &fp in ParameterFreezePolicy::ALL {
                                            if ui
                                                .selectable_value(
                                                    &mut stage.freeze_policy,
                                                    fp,
                                                    fp.label(),
                                                )
                                                .changed()
                                            {
                                                changed = true;
                                            }
                                        }
                                    });
                            });
                        });

                    if idx + 1 < num_stages {
                        ui.vertical_centered(|ui| {
                            ui.label(
                                RichText::new("↓ Checkpoint Handoff ↓")
                                    .size(10.5)
                                    .color(Color32::from_rgb(130, 155, 195)),
                            );
                        });
                    }
                }

                if let Some(del_idx) = stage_to_delete {
                    state.spec.training.stages.remove(del_idx);
                    changed = true;
                } else if let Some(up_idx) = stage_to_move_up {
                    state.spec.training.stages.swap(up_idx, up_idx - 1);
                    changed = true;
                } else if let Some(down_idx) = stage_to_move_down {
                    state.spec.training.stages.swap(down_idx, down_idx + 1);
                    changed = true;
                }

                if changed && !state.spec.training.stages.is_empty() {
                    state.spec.training.curriculum.total_training_tokens = state
                        .spec
                        .training
                        .stages
                        .iter()
                        .map(|s| s.token_budget)
                        .sum();
                }
            });

        changed
    }

    fn render_schedule_and_optimizers(&mut self, ui: &mut Ui, state: &mut StudioState) -> bool {
        let mut changed = false;

        egui::Frame::group(ui.style())
            .fill(Color32::from_rgb(18, 22, 32))
            .stroke(Stroke::new(1.0, Color32::from_rgb(48, 60, 82)))
            .corner_radius(CornerRadius::same(8))
            .inner_margin(egui::Margin::symmetric(12, 10))
            .show(ui, |ui| {
                ui.label(
                    RichText::new(format!(
                        "{} LR Schedule Curve & Optimizer Partitioning",
                        bi::GRAPH_UP_ARROW
                    ))
                    .size(13.5)
                    .strong(),
                );
                ui.add_space(6.0);

                // Interactive LR Schedule Curve Preview
                self.draw_lr_schedule_curve(ui, &state.spec.training.scheduler);

                ui.add_space(6.0);
                let cur_sched_label = match &state.spec.training.scheduler {
                    SchedulerSpec::WarmupStableDecay { .. } => {
                        "Warmup-Stable-Decay (WSD Trapezoid)"
                    }
                    SchedulerSpec::CosineWithWarmup { .. } => "Cosine Annealing with Warmup",
                    SchedulerSpec::Constant { .. } => "Constant Learning Rate",
                };

                ui.horizontal(|ui| {
                    ui.label("Schedule Family:");
                    egui::ComboBox::from_id_salt("planner_sched_kind")
                        .selected_text(cur_sched_label)
                        .width(240.0)
                        .show_ui(ui, |ui| {
                            if ui
                                .selectable_label(
                                    matches!(
                                        state.spec.training.scheduler,
                                        SchedulerSpec::WarmupStableDecay { .. }
                                    ),
                                    "Warmup-Stable-Decay (WSD Trapezoid)",
                                )
                                .clicked()
                            {
                                state.spec.training.scheduler = SchedulerSpec::default();
                                changed = true;
                            }
                            if ui
                                .selectable_label(
                                    matches!(
                                        state.spec.training.scheduler,
                                        SchedulerSpec::CosineWithWarmup { .. }
                                    ),
                                    "Cosine Annealing with Warmup",
                                )
                                .clicked()
                            {
                                state.spec.training.scheduler = SchedulerSpec::CosineWithWarmup {
                                    warmup_ratio: 0.05,
                                    min_lr_ratio: 0.05,
                                    warmup_steps: Some(2000),
                                    total_steps: Some(100_000),
                                    min_lr: Some(1e-5),
                                };
                                changed = true;
                            }
                            if ui
                                .selectable_label(
                                    matches!(
                                        state.spec.training.scheduler,
                                        SchedulerSpec::Constant { .. }
                                    ),
                                    "Constant Learning Rate",
                                )
                                .clicked()
                            {
                                state.spec.training.scheduler = SchedulerSpec::Constant {
                                    lr: state.spec.training.adamw.lr,
                                };
                                changed = true;
                            }
                        });
                });

                match &mut state.spec.training.scheduler {
                    SchedulerSpec::WarmupStableDecay {
                        warmup_ratio,
                        stable_ratio,
                        decay_ratio,
                        min_lr_ratio,
                    } => {
                        changed |= ui
                            .add(
                                egui::Slider::new(warmup_ratio, 0.01..=0.25)
                                    .text("Warmup Fraction"),
                            )
                            .changed();
                        changed |= ui
                            .add(
                                egui::Slider::new(stable_ratio, 0.20..=0.90)
                                    .text("Stable Plateau Fraction"),
                            )
                            .changed();
                        changed |= ui
                            .add(
                                egui::Slider::new(decay_ratio, 0.05..=0.50)
                                    .text("Cooldown Decay Fraction"),
                            )
                            .changed();
                        changed |= ui
                            .add(
                                egui::Slider::new(min_lr_ratio, 0.0..=0.20)
                                    .text("Min LR Floor Ratio"),
                            )
                            .changed();
                    }
                    SchedulerSpec::CosineWithWarmup {
                        warmup_ratio,
                        min_lr_ratio,
                        ..
                    } => {
                        changed |= ui
                            .add(
                                egui::Slider::new(warmup_ratio, 0.01..=0.25)
                                    .text("Warmup Fraction"),
                            )
                            .changed();
                        changed |= ui
                            .add(
                                egui::Slider::new(min_lr_ratio, 0.0..=0.25)
                                    .text("Cosine Floor Ratio"),
                            )
                            .changed();
                    }
                    SchedulerSpec::Constant { lr } => {
                        changed |= ui
                            .add(egui::DragValue::new(lr).speed(0.0001).prefix("LR: "))
                            .changed();
                    }
                }

                ui.add_space(8.0);
                ui.separator();
                ui.label(
                    RichText::new("Optimizer Algorithm Family & Curvature Dynamics")
                        .size(12.5)
                        .strong(),
                );

                ui.horizontal(|ui| {
                    ui.label("Algorithm Family:");
                    egui::ComboBox::from_id_salt("planner_opt_fam_combo")
                        .selected_text(state.spec.training.optimizer_family.label())
                        .width(260.0)
                        .show_ui(ui, |ui| {
                            for &fam in shared::spec::training::OptimizerFamily::ALL {
                                if ui
                                    .selectable_label(state.spec.training.optimizer_family == fam, fam.label())
                                    .on_hover_text(fam.description())
                                    .clicked()
                                {
                                    state.spec.training.optimizer_family = fam;
                                    changed = true;
                                }
                            }
                        });
                });

                match state.spec.training.optimizer_family {
                    shared::spec::training::OptimizerFamily::MuonAdamWPartitioned => {
                        changed |= ui
                            .checkbox(
                                &mut state.spec.training.auto_partition_parameters,
                                "Automatically partition 2D internal matrices to Muon & 1D/Embeddings to AdamW",
                            )
                            .changed();

                        egui::Grid::new("planner_optimizer_grid")
                            .num_columns(2)
                            .spacing([14.0, 6.0])
                            .show(ui, |ui| {
                                ui.label(
                                    RichText::new(format!(
                                        "Muon 2D LR ({} params):",
                                        format_large_count(state.partition_report.muon_param_count as u64)
                                    ))
                                    .color(Color32::from_rgb(110, 200, 255)),
                                );
                                changed |= ui
                                    .add(
                                        egui::DragValue::new(&mut state.spec.training.muon.lr)
                                            .speed(0.0005)
                                            .range(1e-5..=0.2),
                                    )
                                    .changed();
                                ui.end_row();

                                ui.label("Muon Momentum & NS Steps:");
                                ui.horizontal(|ui| {
                                    changed |= ui
                                        .add(
                                            egui::DragValue::new(&mut state.spec.training.muon.momentum)
                                                .speed(0.005)
                                                .range(0.5..=0.999),
                                        )
                                        .changed();
                                    ui.label("Steps:");
                                    changed |= ui
                                        .add(
                                            egui::DragValue::new(
                                                &mut state.spec.training.muon.newton_schulz_steps,
                                            )
                                            .range(3..=10),
                                        )
                                        .changed();
                                });
                                ui.end_row();

                                ui.label(
                                    RichText::new(format!(
                                        "AdamW 1D/Embed LR ({} params):",
                                        format_large_count(state.partition_report.adamw_param_count as u64)
                                    ))
                                    .color(Color32::from_rgb(130, 230, 165)),
                                );
                                changed |= ui
                                    .add(
                                        egui::DragValue::new(&mut state.spec.training.adamw.lr)
                                            .speed(0.00005)
                                            .range(1e-6..=0.05),
                                    )
                                    .changed();
                                ui.end_row();

                                ui.label("AdamW Weight Decay:");
                                changed |= ui
                                    .add(
                                        egui::DragValue::new(&mut state.spec.training.adamw.weight_decay)
                                            .speed(0.001)
                                            .range(0.0..=0.5),
                                    )
                                    .changed();
                                ui.end_row();
                            });
                    }
                    shared::spec::training::OptimizerFamily::AdamWStandard => {
                        egui::Grid::new("planner_adamw_grid")
                            .num_columns(2)
                            .spacing([14.0, 6.0])
                            .show(ui, |ui| {
                                ui.label("Base Learning Rate:");
                                changed |= ui.add(egui::DragValue::new(&mut state.spec.training.adamw.lr).speed(0.0001).range(1e-6..=0.1)).changed();
                                ui.end_row();
                                ui.label("Weight Decay:");
                                changed |= ui.add(egui::DragValue::new(&mut state.spec.training.adamw.weight_decay).speed(0.001).range(0.0..=0.5)).changed();
                                ui.end_row();
                                ui.label("Betas (β1, β2):");
                                ui.horizontal(|ui| {
                                    changed |= ui.add(egui::DragValue::new(&mut state.spec.training.adamw.betas.0).speed(0.01).range(0.5..=0.999)).changed();
                                    changed |= ui.add(egui::DragValue::new(&mut state.spec.training.adamw.betas.1).speed(0.001).range(0.8..=0.9999)).changed();
                                });
                                ui.end_row();
                            });
                    }
                    shared::spec::training::OptimizerFamily::LionSignMomentum => {
                        egui::Grid::new("planner_lion_grid")
                            .num_columns(2)
                            .spacing([14.0, 6.0])
                            .show(ui, |ui| {
                                ui.label("Lion Learning Rate:")
                                    .on_hover_text("EvoLved Sign Momentum learning rate (typically 3-10x smaller than AdamW).");
                                changed |= ui.add(egui::DragValue::new(&mut state.spec.training.lion.lr).speed(0.00002).range(1e-6..=0.01)).changed();
                                ui.end_row();
                                ui.label("Weight Decay:");
                                changed |= ui.add(egui::DragValue::new(&mut state.spec.training.lion.weight_decay).speed(0.01).range(0.0..=2.0)).changed();
                                ui.end_row();
                                ui.label("Betas (β1, β2):");
                                ui.horizontal(|ui| {
                                    changed |= ui.add(egui::DragValue::new(&mut state.spec.training.lion.betas.0).speed(0.01).range(0.5..=0.99)).changed();
                                    changed |= ui.add(egui::DragValue::new(&mut state.spec.training.lion.betas.1).speed(0.005).range(0.8..=0.999)).changed();
                                });
                                ui.end_row();
                            });
                    }
                    shared::spec::training::OptimizerFamily::SophiaSecondOrder => {
                        egui::Grid::new("planner_sophia_grid")
                            .num_columns(2)
                            .spacing([14.0, 6.0])
                            .show(ui, |ui| {
                                ui.label("Sophia Learning Rate:")
                                    .on_hover_text("Second-order diagonal Hessian clipped descent learning rate.");
                                changed |= ui.add(egui::DragValue::new(&mut state.spec.training.sophia.lr).speed(0.0001).range(1e-5..=0.05)).changed();
                                ui.end_row();
                                ui.label("Hessian Clip (rho):")
                                    .on_hover_text("Curvature clipping threshold rho (typically 0.04).");
                                changed |= ui.add(egui::DragValue::new(&mut state.spec.training.sophia.rho).speed(0.005).range(0.001..=0.5)).changed();
                                ui.end_row();
                                ui.label("Weight Decay:");
                                changed |= ui.add(egui::DragValue::new(&mut state.spec.training.sophia.weight_decay).speed(0.01).range(0.0..=1.0)).changed();
                                ui.end_row();
                                ui.label("Betas (β1, β2):");
                                ui.horizontal(|ui| {
                                    changed |= ui.add(egui::DragValue::new(&mut state.spec.training.sophia.betas.0).speed(0.01).range(0.5..=0.999)).changed();
                                    changed |= ui.add(egui::DragValue::new(&mut state.spec.training.sophia.betas.1).speed(0.005).range(0.8..=0.999)).changed();
                                });
                                ui.end_row();
                            });
                    }
                    shared::spec::training::OptimizerFamily::LBFGSQuasiNewton => {
                        egui::Grid::new("planner_lbfgs_grid")
                            .num_columns(2)
                            .spacing([14.0, 6.0])
                            .show(ui, |ui| {
                                ui.label("Max Iterations:");
                                changed |= ui.add(egui::DragValue::new(&mut state.spec.training.lbfgs.max_iter).range(1..=100)).changed();
                                ui.end_row();
                                ui.label("History Size:")
                                    .on_hover_text("Number of prior gradient and displacement pairs stored for inverse Hessian representation.");
                                changed |= ui.add(egui::DragValue::new(&mut state.spec.training.lbfgs.history_size).range(5..=100)).changed();
                                ui.end_row();
                            });
                    }
                    _ => {
                        ui.horizontal(|ui| {
                            ui.label("Base Learning Rate:");
                            changed |= ui.add(egui::DragValue::new(&mut state.spec.training.adamw.lr).speed(0.0001).range(1e-6..=0.1)).changed();
                        });
                    }
                }

                ui.add_space(10.0);
                ui.separator();
                ui.label(
                    RichText::new("Learnable Quantization Level & Mixed-Bitwidth QAT")
                        .size(12.5)
                        .color(Color32::from_rgb(245, 200, 110))
                        .strong(),
                );
                ui.label(
                    RichText::new("Train per-layer step-size (LSQ+), activation clipping (PACT), or continuous bitwidths (FracBits) with BOPs penalty.")
                        .size(10.5)
                        .color(Color32::from_rgb(160, 175, 195)),
                );
                ui.add_space(4.0);

                let q = &mut state.spec.training.quantization_training;
                ui.horizontal(|ui| {
                    ui.label("Quantization Training Mode:");
                    egui::ComboBox::from_id_salt("planner_quant_train_mode")
                        .selected_text(q.mode.label())
                        .width(250.0)
                        .show_ui(ui, |ui| {
                            for &m in shared::spec::training::QuantizationTrainingMode::ALL {
                                if ui
                                    .selectable_value(&mut q.mode, m, m.label())
                                    .on_hover_text(m.description())
                                    .changed()
                                {
                                    changed = true;
                                }
                            }
                        })
                        .response
                        .on_hover_text(q.mode.description());
                });

                if q.mode != shared::spec::training::QuantizationTrainingMode::FixedPrecision {
                    ui.add_space(4.0);
                    egui::Grid::new("planner_quant_train_grid")
                        .num_columns(2)
                        .spacing([12.0, 6.0])
                        .show(ui, |ui| {
                            ui.label("Target Avg Bitwidth (b_target):")
                                .on_hover_text("Target average bitwidth across all quantized layers for the BOPs hardware complexity budget.");
                            changed |= ui
                                .add(egui::Slider::new(&mut q.target_avg_bits, 1.0..=16.0).text("bits"))
                                .changed();
                            ui.end_row();

                            ui.label("Layer Bitwidth Bounds [min, max]:");
                            ui.horizontal(|ui| {
                                changed |= ui
                                    .add(
                                        egui::DragValue::new(&mut q.min_layer_bits)
                                            .speed(0.1)
                                            .range(1.0..=8.0)
                                            .prefix("min: "),
                                    )
                                    .changed();
                                changed |= ui
                                    .add(
                                        egui::DragValue::new(&mut q.max_layer_bits)
                                            .speed(0.5)
                                            .range(2.0..=16.0)
                                            .prefix("max: "),
                                    )
                                    .changed();
                            });
                            ui.end_row();

                            ui.label("BOPs Budget Penalty (λ_BOPs):")
                                .on_hover_text("Lagrangian multiplier penalizing excess Bit-Operations (FLOPs * b_w * b_a) above target bitwidth.");
                            changed |= ui
                                .add(
                                    egui::DragValue::new(&mut q.bops_penalty_lambda)
                                        .speed(1e-5)
                                        .range(0.0..=1.0),
                                )
                                .changed();
                            ui.end_row();

                            ui.label("Soft-STE Annealing Temp (τ):")
                                .on_hover_text("Temperature for sigmoid/tanh relaxed Straight-Through Estimator rounding during QAT.");
                            changed |= ui
                                .add(egui::Slider::new(&mut q.ste_temperature, 0.05..=5.0))
                                .changed();
                            ui.end_row();
                        });

                    ui.horizontal_wrapped(|ui| {
                        changed |= ui
                            .checkbox(
                                &mut q.learn_activation_clip_alpha,
                                "Learn PACT Activation Clip (α_l)",
                            )
                            .changed();
                        changed |= ui
                            .checkbox(
                                &mut q.learn_non_uniform_grid,
                                "Learn Non-Uniform Codebook Grid (LQ-Nets)",
                            )
                            .changed();
                    });

                    let compression_vs_bf16 = (16.0 / q.target_avg_bits.max(1.0)).max(1.0);
                    let bops_reduction = (16.0 * 16.0) / (q.target_avg_bits.max(1.0).powi(2));
                    ui.label(
                        RichText::new(format!(
                            "Est. Footprint Compression vs BF16: {:.2}× smaller weights • {:.1}× BOPs reduction",
                            compression_vs_bf16, bops_reduction
                        ))
                        .size(10.5)
                        .color(Color32::from_rgb(245, 205, 115)),
                    );
                }
            });

        changed
    }

    fn render_parallelism_and_curriculum(&mut self, ui: &mut Ui, state: &mut StudioState) -> bool {
        let mut changed = false;

        egui::Frame::group(ui.style())
            .fill(Color32::from_rgb(18, 22, 32))
            .stroke(Stroke::new(1.0, Color32::from_rgb(48, 60, 82)))
            .corner_radius(CornerRadius::same(8))
            .inner_margin(egui::Margin::symmetric(12, 10))
            .show(ui, |ui| {
                ui.label(
                    RichText::new(format!(
                        "{} 4D Distributed Parallelism & Batch Planner",
                        bi::CPU
                    ))
                    .size(13.5)
                    .strong(),
                );
                ui.add_space(6.0);

                let p = &mut state.spec.training.parallelism;
                ui.horizontal(|ui| {
                    ui.label("Memory Sharding:");
                    egui::ComboBox::from_id_salt("planner_sharding_combo")
                        .selected_text(p.sharding.label())
                        .width(240.0)
                        .show_ui(ui, |ui| {
                            for &mode in ShardingStrategy::ALL {
                                if ui
                                    .selectable_value(&mut p.sharding, mode, mode.label())
                                    .changed()
                                {
                                    changed = true;
                                }
                            }
                        });
                });

                ui.add_space(4.0);
                egui::Grid::new("planner_parallelism_grid")
                    .num_columns(4)
                    .spacing([12.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Data Parallel (DP):");
                        changed |= ui
                            .add(
                                egui::DragValue::new(&mut p.data_parallel_degree).range(1..=2048),
                            )
                            .changed();
                        ui.label("Tensor Parallel (TP):");
                        changed |= ui
                            .add(egui::DragValue::new(&mut p.tensor_parallel_degree).range(1..=64))
                            .changed();
                        ui.end_row();

                        ui.label("Pipeline Parallel (PP):");
                        changed |= ui
                            .add(
                                egui::DragValue::new(&mut p.pipeline_parallel_degree).range(1..=64),
                            )
                            .changed();
                        ui.label("Context/Ring (CP):");
                        changed |= ui
                            .add(egui::DragValue::new(&mut p.context_parallel_degree).range(1..=64))
                            .changed();
                        ui.end_row();

                        ui.label("Expert Parallel (EP):");
                        changed |= ui
                            .add(egui::DragValue::new(&mut p.expert_parallel_degree).range(1..=256))
                            .changed();
                        ui.label("Micro-Batch (B_micro):");
                        changed |= ui
                            .add(egui::DragValue::new(&mut p.micro_batch_size).range(1..=512))
                            .changed();
                        ui.end_row();

                        ui.label("Grad Accum Steps:");
                        changed |= ui
                            .add(
                                egui::DragValue::new(&mut p.gradient_accumulation_steps)
                                    .range(1..=512),
                            )
                            .changed();
                        ui.label("Sequence Length:");
                        changed |= ui
                            .add(
                                egui::DragValue::new(&mut p.sequence_length)
                                    .range(64..=1_048_576)
                                    .speed(128),
                            )
                            .changed();
                        ui.end_row();
                    });

                ui.add_space(4.0);
                ui.label(
                    RichText::new(format!(
                        "Global Batch = {} micro × {} accum × {} DP = {} samples ({} tokens/step across {} GPUs)",
                        p.micro_batch_size,
                        p.gradient_accumulation_steps,
                        p.data_parallel_degree,
                        p.global_batch_samples(),
                        format_large_count(p.global_batch_tokens() as u64),
                        p.total_accelerators()
                    ))
                    .size(11.0)
                    .color(Color32::from_rgb(130, 220, 165)),
                );

                ui.add_space(8.0);
                ui.separator();
                ui.label(
                    RichText::new("Dataset Curriculum & Gradient Hygiene")
                        .size(12.5)
                        .strong(),
                );

                let c = &mut state.spec.training.curriculum;
                egui::Grid::new("planner_curriculum_grid")
                    .num_columns(2)
                    .spacing([12.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Total Token Budget:");
                        changed |= ui
                            .add(
                                egui::DragValue::new(&mut c.total_training_tokens)
                                    .range(1_000_000..=100_000_000_000_000u64)
                                    .speed(500_000_000.0),
                            )
                            .changed();
                        ui.end_row();

                        ui.label("Synthetic Data Mix Ratio:");
                        changed |= ui
                            .add(egui::Slider::new(&mut c.synthetic_data_ratio, 0.0..=1.0))
                            .changed();
                        ui.end_row();

                        ui.label("Checkpoint / Eval Interval (steps):");
                        ui.horizontal(|ui| {
                            changed |= ui
                                .add(
                                    egui::DragValue::new(&mut c.checkpoint_interval_steps)
                                        .range(10..=100_000),
                                )
                                .changed();
                            ui.label("/");
                            changed |= ui
                                .add(
                                    egui::DragValue::new(&mut c.eval_interval_steps)
                                        .range(10..=50_000),
                                )
                                .changed();
                        });
                        ui.end_row();
                    });

                ui.horizontal_wrapped(|ui| {
                    changed |= ui
                        .checkbox(&mut c.sequence_packing, "Multipack Sequence Packing")
                        .changed();
                    changed |= ui
                        .checkbox(&mut c.curriculum_length_ramp, "Curriculum Length Ramp")
                        .changed();
                });

                let h = &mut state.spec.training.hygiene;
                ui.horizontal_wrapped(|ui| {
                    changed |= ui
                        .checkbox(&mut h.auto_clip, "AutoClip (90th-p Norm)")
                        .changed();
                    changed |= ui
                        .checkbox(&mut h.grokfast, "Grokfast EMA Filter")
                        .changed();
                    changed |= ui
                        .checkbox(
                            &mut h.activation_checkpointing,
                            "Activation Checkpointing (3x VRAM Save)",
                        )
                        .changed();
                });

                ui.horizontal(|ui| {
                    ui.label("Multi-Task Loss Balancing:");
                    egui::ComboBox::from_id_salt("planner_loss_balancing")
                        .selected_text(format!("{:?}", state.spec.training.loss_balancing))
                        .show_ui(ui, |ui| {
                            for &strategy in &[
                                LossBalancingStrategy::FixedWeights,
                                LossBalancingStrategy::PCGrad,
                                LossBalancingStrategy::NTKBalancing,
                                LossBalancingStrategy::GradNorm,
                                LossBalancingStrategy::UncertaintyKendallGal,
                            ] {
                                if ui
                                    .selectable_value(
                                        &mut state.spec.training.loss_balancing,
                                        strategy,
                                        format!("{strategy:?}"),
                                    )
                                    .changed()
                                {
                                    changed = true;
                                }
                            }
                        });
                });
            });

        changed
    }

    fn draw_lr_schedule_curve(&self, ui: &mut Ui, scheduler: &SchedulerSpec) {
        let desired_size = Vec2::new(ui.available_width().max(240.0), 96.0);
        let (rect, _response) = ui.allocate_exact_size(desired_size, egui::Sense::hover());
        let painter = ui.painter_at(rect);

        painter.rect_filled(rect, 6.0, Color32::from_rgb(13, 16, 24));
        painter.rect_stroke(
            rect,
            6.0,
            Stroke::new(1.0, Color32::from_rgb(42, 54, 76)),
            egui::StrokeKind::Inside,
        );

        let pad = 12.0;
        let plot_rect = Rect::from_min_max(
            Pos2::new(rect.left() + pad, rect.top() + pad),
            Pos2::new(rect.right() - pad, rect.bottom() - pad - 8.0),
        );

        let num_samples = 80;
        let mut points = Vec::with_capacity(num_samples + 1);
        for i in 0..=num_samples {
            let t = i as f64 / num_samples as f64;
            let lr_mult = match scheduler {
                SchedulerSpec::WarmupStableDecay {
                    warmup_ratio,
                    stable_ratio,
                    decay_ratio,
                    min_lr_ratio,
                } => {
                    let w = warmup_ratio.clamp(0.01, 0.9);
                    let s = stable_ratio.clamp(0.05, 0.95);
                    let d = decay_ratio.max(0.01);
                    let total = (w + s + d).max(1e-6);
                    let wn = w / total;
                    let sn = s / total;
                    if t < wn {
                        t / wn
                    } else if t < wn + sn {
                        1.0
                    } else {
                        let decay_progress =
                            ((t - wn - sn) / (1.0 - wn - sn).max(1e-6)).clamp(0.0, 1.0);
                        1.0 - decay_progress * (1.0 - min_lr_ratio.clamp(0.0, 1.0))
                    }
                }
                SchedulerSpec::CosineWithWarmup {
                    warmup_ratio,
                    min_lr_ratio,
                    ..
                } => {
                    let w = warmup_ratio.clamp(0.01, 0.5);
                    if t < w {
                        t / w
                    } else {
                        let progress = ((t - w) / (1.0 - w).max(1e-6)).clamp(0.0, 1.0);
                        let cosine = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
                        min_lr_ratio + (1.0 - min_lr_ratio) * cosine
                    }
                }
                SchedulerSpec::Constant { .. } => 1.0,
            };

            let x = plot_rect.left() + (t as f32) * plot_rect.width();
            let y = plot_rect.bottom() - (lr_mult.clamp(0.0, 1.0) as f32) * plot_rect.height();
            points.push(Pos2::new(x, y));
        }

        painter.add(egui::Shape::line(
            points,
            Stroke::new(2.2, Color32::from_rgb(100, 210, 255)),
        ));
        painter.text(
            Pos2::new(plot_rect.left() + 4.0, rect.bottom() - 10.0),
            egui::Align2::LEFT_CENTER,
            "0% Training Progress -> 100% Token Budget",
            egui::FontId::monospace(10.0),
            Color32::from_rgb(145, 160, 185),
        );
    }
}

fn metric_tile(ui: &mut Ui, title: &str, primary: &str, subtitle: &str, accent: Color32) {
    ui.vertical(|ui| {
        ui.label(
            RichText::new(title)
                .size(10.5)
                .color(Color32::from_rgb(160, 175, 195)),
        );
        ui.label(RichText::new(primary).size(14.0).color(accent).strong());
        ui.label(
            RichText::new(subtitle)
                .size(10.0)
                .color(Color32::from_rgb(135, 150, 170)),
        );
    });
}

fn format_large_count(n: u64) -> String {
    if n >= 1_000_000_000_000 {
        format!("{:.2}T", n as f64 / 1e12)
    } else if n >= 1_000_000_000 {
        format!("{:.2}B", n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1e3)
    } else {
        format!("{n}")
    }
}

fn format_duration_hours(hours: f64) -> String {
    if hours < 1.0 {
        format!("{:.1} mins", (hours * 60.0).max(0.1))
    } else if hours < 48.0 {
        format!("{hours:.1} hrs")
    } else {
        format!("{:.1} days", hours / 24.0)
    }
}
