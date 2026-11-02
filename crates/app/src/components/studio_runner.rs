//! Studio Runner: live execution panel with turnkey environment generation,
//! real-time process mediation, dynamic optimization controls, and high-frequency telemetry.

use crate::{
    icons::bi,
    studio_state::{RunnerTab, StudioState},
};
use eframe::egui::{self, Color32, Pos2, RichText, ScrollArea, Stroke, Ui, Vec2};
use shared::spec::training::OptimizerFamily;
use shared::training::MetaTuningPolicy;
use std::path::Path;

pub struct StudioRunnerView;

impl Default for StudioRunnerView {
    fn default() -> Self {
        Self
    }
}

impl StudioRunnerView {
    pub fn render(&mut self, ui: &mut Ui, state: &mut StudioState) {
        ui.vertical(|ui| {
            ui.add_space(4.0);

            // 1. Process Lifecycle & Header Ribbon
            self.render_header_controls(ui, state);

            ui.add_space(4.0);
            ui.separator();

            // 2. Training Process Step Advancement
            if state.runner.session.state == shared::TrainingProcessState::Running {
                if let Some(telemetry) = state.runner.session.step() {
                    let step = telemetry.step as f64;
                    let loss = telemetry.loss as f64;
                    let val_loss = telemetry.val_loss_estimate as f64;
                    let grad_norm = telemetry.grad_norm as f64;
                    state.runner.current_step = telemetry.step;
                    state.runner.max_steps = telemetry.max_steps;
                    state.runner.current_loss = telemetry.loss;
                    state.runner.loss_history.push([step, loss]);
                    state.runner.val_loss_history.push([step, val_loss]);
                    state.runner.grad_norm_history.push([step, grad_norm]);
                    state.runner.console_logs = state.runner.session.console_logs.clone();
                }
                state.runner.sync_from_session();
                ui.ctx().request_repaint();
            }

            // 3. Tab Bar Selection
            self.render_tab_bar(ui, state);

            ui.add_space(4.0);

            // 4. Tab Body Content
            match state.runner.selected_tab {
                RunnerTab::LiveConvergence => self.render_live_convergence(ui, state),
                RunnerTab::LiveOptimizations => self.render_live_optimizations(ui, state),
                RunnerTab::EnvironmentSetup => self.render_environment_setup(ui, state),
                RunnerTab::ConsoleLog => self.render_console_log(ui, state),
            }
        });
    }

    fn render_header_controls(&self, ui: &mut Ui, state: &mut StudioState) {
        ui.horizontal(|ui| {
            ui.heading(
                RichText::new("Studio Execution & Training Runner")
                    .size(15.0)
                    .strong(),
            );

            // Process State Badge
            let (badge_bg, badge_fg) = match state.runner.session.state {
                shared::TrainingProcessState::Running => (
                    Color32::from_rgb(25, 75, 40),
                    Color32::from_rgb(140, 240, 160),
                ),
                shared::TrainingProcessState::Paused => (
                    Color32::from_rgb(80, 60, 20),
                    Color32::from_rgb(255, 210, 90),
                ),
                shared::TrainingProcessState::Completed => (
                    Color32::from_rgb(25, 55, 90),
                    Color32::from_rgb(130, 200, 255),
                ),
                shared::TrainingProcessState::Stopped | shared::TrainingProcessState::Failed => (
                    Color32::from_rgb(85, 25, 25),
                    Color32::from_rgb(255, 140, 140),
                ),
                _ => (
                    Color32::from_rgb(40, 45, 55),
                    Color32::from_rgb(180, 190, 205),
                ),
            };

            ui.group(|ui| {
                ui.style_mut().visuals.override_text_color = Some(badge_fg);
                ui.label(
                    RichText::new(format!(" {} ", state.runner.session.state.badge()))
                        .size(11.0)
                        .strong()
                        .background_color(badge_bg),
                );
            });

            // Grokking Phase Pill Badge
            let grok_phase = state.runner.session.current_telemetry.grokking_phase;
            let (grok_bg, grok_fg) = match grok_phase {
                shared::GrokkingPhase::FittingMemorization => (
                    Color32::from_rgb(60, 30, 80),
                    Color32::from_rgb(220, 160, 255),
                ),
                shared::GrokkingPhase::WeightNormCompression => (
                    Color32::from_rgb(80, 55, 20),
                    Color32::from_rgb(255, 205, 95),
                ),
                shared::GrokkingPhase::AlgorithmicGrokking => (
                    Color32::from_rgb(20, 70, 95),
                    Color32::from_rgb(105, 235, 255),
                ),
                shared::GrokkingPhase::GeneralizationConsolidation => (
                    Color32::from_rgb(25, 75, 45),
                    Color32::from_rgb(130, 245, 170),
                ),
            };

            ui.group(|ui| {
                ui.style_mut().visuals.override_text_color = Some(grok_fg);
                ui.label(
                    RichText::new(format!(" {} ", grok_phase.badge()))
                        .size(11.0)
                        .strong()
                        .background_color(grok_bg),
                );
            });

            // Action Buttons
            match state.runner.session.state {
                shared::TrainingProcessState::Running => {
                    if ui
                        .button(
                            RichText::new(format!("{} Pause", bi::PAUSE_FILL))
                                .color(Color32::from_rgb(255, 200, 80)),
                        )
                        .clicked()
                    {
                        state.runner.session.pause();
                        state.runner.sync_from_session();
                    }
                    if ui
                        .button(
                            RichText::new(format!("{} Stop", bi::STOP_FILL))
                                .color(Color32::from_rgb(240, 80, 80)),
                        )
                        .clicked()
                    {
                        state.runner.session.stop();
                        state.runner.sync_from_session();
                    }
                }
                shared::TrainingProcessState::Paused => {
                    if ui
                        .button(
                            RichText::new(format!("{} Resume", bi::PLAY_FILL))
                                .color(Color32::from_rgb(120, 220, 140))
                                .strong(),
                        )
                        .clicked()
                    {
                        state.runner.session.resume();
                        state.runner.sync_from_session();
                    }
                    if ui
                        .button(
                            RichText::new(format!("{} Stop", bi::STOP_FILL))
                                .color(Color32::from_rgb(240, 80, 80)),
                        )
                        .clicked()
                    {
                        state.runner.session.stop();
                        state.runner.sync_from_session();
                    }
                    if ui
                        .button(
                            RichText::new(format!("{} Reset", bi::ARROW_COUNTERCLOCKWISE))
                                .color(Color32::from_rgb(180, 190, 200)),
                        )
                        .clicked()
                    {
                        state.runner.reset_with_spec(&state.spec, 100);
                    }
                }
                _ => {
                    if ui
                        .button(
                            RichText::new(format!("{} Start Training", bi::PLAY_FILL))
                                .color(Color32::from_rgb(120, 220, 140))
                                .strong(),
                        )
                        .clicked()
                    {
                        state.runner.session.start();
                        state.runner.loss_history.clear();
                        state.runner.val_loss_history.clear();
                        state.runner.grad_norm_history.clear();
                        state.runner.sync_from_session();
                    }
                    if ui
                        .button(
                            RichText::new(format!("{} Test Parity", bi::SHIELD_CHECK))
                                .color(Color32::from_rgb(100, 190, 255)),
                        )
                        .clicked()
                    {
                        let report = shared::verify_transitive_parity(&state.spec);
                        state
                            .runner
                            .push_log(format!("=== Parity Report for '{}' ===", report.model_id));
                        state.runner.push_log(format!(
                            "Evaluated {} nodes. Status: {:?}",
                            report.nodes_evaluated, report.status
                        ));
                        state.runner.push_log(format!(
                            "Dynamic Range: [{:.2e}, {:.2e}] | NaN/Inf: {}",
                            report.min_activation,
                            report.max_activation,
                            if report.has_nan_or_inf {
                                "YES (FAIL)"
                            } else {
                                "None"
                            }
                        ));
                        for diag in report.diagnostics {
                            state.runner.push_log(format!("  [*] {}", diag));
                        }
                    }
                    if state.runner.current_step > 0
                        && ui
                            .button(
                                RichText::new(format!("{} Reset", bi::ARROW_COUNTERCLOCKWISE))
                                    .color(Color32::from_rgb(180, 190, 200)),
                            )
                            .clicked()
                    {
                        state.runner.reset_with_spec(&state.spec, 100);
                    }
                }
            }

            ui.separator();

            // Telemetry summary ribbon
            let telem = &state.runner.session.current_telemetry;
            ui.label(format!("Step: {} / {}", telem.step, telem.max_steps));
            if telem.loss > 0.0 {
                ui.label(
                    RichText::new(format!("Train: {:.4}", telem.loss))
                        .strong()
                        .color(Color32::from_rgb(120, 200, 255)),
                );
                ui.label(
                    RichText::new(format!("Val: {:.4}", telem.val_loss_estimate))
                        .strong()
                        .color(Color32::from_rgb(255, 120, 205)),
                );
                ui.label(
                    RichText::new(format!("||W||: {:.2}", telem.weight_norm))
                        .color(Color32::from_rgb(205, 215, 175)),
                );
            }
            ui.label(
                RichText::new(format!("{:.0} tok/s", telem.tokens_per_sec))
                    .color(Color32::from_rgb(170, 210, 180)),
            );
            ui.label(
                RichText::new(format!("{:.0} MB VRAM", telem.vram_used_mb))
                    .color(Color32::from_rgb(200, 180, 230)),
            );
            ui.label(
                RichText::new(format!("Muon Ortho: {:.1e}", telem.muon_ortho_error))
                    .color(Color32::from_rgb(190, 190, 190)),
            );

            if state.runner.session.meta_tuner.enabled {
                ui.label(
                    RichText::new(format!(
                        "⚡ Auto-Meta: {:?}",
                        state.runner.session.meta_tuner.policy
                    ))
                    .size(11.0)
                    .strong()
                    .color(Color32::from_rgb(130, 240, 170)),
                );
            } else {
                ui.label(
                    RichText::new("Auto-Meta: OFF")
                        .size(11.0)
                        .color(Color32::from_rgb(160, 160, 170)),
                );
            }
        });
    }

    fn render_tab_bar(&self, ui: &mut Ui, state: &mut StudioState) {
        ui.horizontal(|ui| {
            ui.selectable_value(
                &mut state.runner.selected_tab,
                RunnerTab::LiveConvergence,
                RichText::new(format!("{} Convergence & Telemetry", bi::ACTIVITY)).strong(),
            );
            ui.selectable_value(
                &mut state.runner.selected_tab,
                RunnerTab::LiveOptimizations,
                RichText::new(format!("{} Dynamic Optimizations", bi::SLIDERS)).strong(),
            );
            ui.selectable_value(
                &mut state.runner.selected_tab,
                RunnerTab::EnvironmentSetup,
                RichText::new(format!("{} Environment Setup", bi::FOLDER2_OPEN)).strong(),
            );
            ui.selectable_value(
                &mut state.runner.selected_tab,
                RunnerTab::ConsoleLog,
                RichText::new(format!(
                    "{} Console ({})",
                    bi::TERMINAL,
                    state.runner.console_logs.len()
                ))
                .strong(),
            );
        });
    }

    fn render_live_convergence(&self, ui: &mut Ui, state: &mut StudioState) {
        ui.columns(2, |columns| {
            // Left Column: Native Painter Loss & Gradient Curve
            columns[0].vertical(|ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("Convergence & Grokking Generalization Dynamics").strong(),
                    );
                    ui.label(
                        RichText::new("● Train (Cyan)  ● Val (Magenta)  ● Grad (Amber)")
                            .size(10.5)
                            .color(Color32::from_rgb(165, 180, 205)),
                    );
                    let telem = &state.runner.session.current_telemetry;
                    if telem.step > 0 {
                        ui.label(
                            RichText::new(format!(
                                "Best: {:.4} | Gap: {:.4}",
                                telem.best_loss, telem.generalization_gap
                            ))
                            .size(11.0)
                            .color(Color32::from_rgb(150, 190, 220)),
                        );
                    }
                });

                let (rect, _) = ui.allocate_exact_size(
                    Vec2::new(ui.available_width(), 175.0),
                    egui::Sense::hover(),
                );
                let painter = ui.painter_at(rect);

                // Background & border
                painter.rect_filled(rect, 4.0, Color32::from_rgb(18, 22, 30));
                painter.rect_stroke(
                    rect,
                    4.0,
                    Stroke::new(1.0, Color32::from_rgb(45, 55, 75)),
                    egui::epaint::StrokeKind::Inside,
                );

                // Grid lines
                let grid_stroke = Stroke::new(1.0, Color32::from_rgb(30, 36, 50));
                for i in 1..4 {
                    let y = rect.min.y + (rect.height() * i as f32 / 4.0);
                    painter.line_segment(
                        [Pos2::new(rect.min.x, y), Pos2::new(rect.max.x, y)],
                        grid_stroke,
                    );
                }

                if state.runner.loss_history.len() >= 2 {
                    let max_step = state.runner.max_steps.max(1) as f32;
                    let max_loss = 3.0f32;
                    let min_loss = 0.0f32;

                    let padding = 10.0;
                    let inner_rect = rect.shrink(padding);

                    let to_screen_loss = |step: f64, val: f64| -> Pos2 {
                        let x_frac = (step as f32 / max_step).clamp(0.0, 1.0);
                        let y_frac =
                            ((val as f32 - min_loss) / (max_loss - min_loss)).clamp(0.0, 1.0);
                        Pos2::new(
                            inner_rect.min.x + (x_frac * inner_rect.width()),
                            inner_rect.max.y - (y_frac * inner_rect.height()),
                        )
                    };

                    // Draw train loss history line (cyan)
                    let loss_stroke = Stroke::new(2.0, Color32::from_rgb(100, 180, 255));
                    for i in 0..state.runner.loss_history.len() - 1 {
                        let p1 = to_screen_loss(
                            state.runner.loss_history[i][0],
                            state.runner.loss_history[i][1],
                        );
                        let p2 = to_screen_loss(
                            state.runner.loss_history[i + 1][0],
                            state.runner.loss_history[i + 1][1],
                        );
                        painter.line_segment([p1, p2], loss_stroke);
                    }

                    // Draw validation loss history line (magenta)
                    if state.runner.val_loss_history.len() >= 2 {
                        let val_stroke = Stroke::new(2.0, Color32::from_rgb(255, 120, 210));
                        for i in 0..state.runner.val_loss_history.len() - 1 {
                            let p1 = to_screen_loss(
                                state.runner.val_loss_history[i][0],
                                state.runner.val_loss_history[i][1],
                            );
                            let p2 = to_screen_loss(
                                state.runner.val_loss_history[i + 1][0],
                                state.runner.val_loss_history[i + 1][1],
                            );
                            painter.line_segment([p1, p2], val_stroke);
                        }
                    }

                    // Draw gradient norm line (amber)
                    if state.runner.grad_norm_history.len() >= 2 {
                        let grad_stroke = Stroke::new(1.2, Color32::from_rgb(255, 175, 75));
                        for i in 0..state.runner.grad_norm_history.len() - 1 {
                            let p1 = to_screen_loss(
                                state.runner.grad_norm_history[i][0],
                                state.runner.grad_norm_history[i][1],
                            );
                            let p2 = to_screen_loss(
                                state.runner.grad_norm_history[i + 1][0],
                                state.runner.grad_norm_history[i + 1][1],
                            );
                            painter.line_segment([p1, p2], grad_stroke);
                        }
                    }

                    // Current point dots
                    if let Some(last_val) = state.runner.val_loss_history.last() {
                        let pt_val = to_screen_loss(last_val[0], last_val[1]);
                        painter.circle_filled(pt_val, 3.5, Color32::from_rgb(255, 120, 210));
                    }
                    if let Some(last) = state.runner.loss_history.last() {
                        let pt = to_screen_loss(last[0], last[1]);
                        painter.circle_filled(pt, 4.0, Color32::from_rgb(120, 220, 140));
                    }
                } else {
                    painter.text(
                        rect.center(),
                        egui::Align2::CENTER_CENTER,
                        "Click 'Start Training' to begin live convergence telemetry",
                        egui::FontId::proportional(12.0),
                        Color32::from_rgb(120, 130, 150),
                    );
                }
            });

            // Right Column: Telemetry Cards & Curriculum Stage Info
            columns[1].vertical(|ui| {
                ui.label(RichText::new("Live Telemetry & Curriculum Status").strong());
                let telem = &state.runner.session.current_telemetry;

                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Curriculum Stage:").strong());
                        ui.label(
                            RichText::new(format!(
                                "[Stage {}] {}",
                                telem.active_stage_idx + 1,
                                telem.active_stage_name
                            ))
                            .color(Color32::from_rgb(120, 200, 255)),
                        );
                    });

                    let progress = if telem.max_steps > 0 {
                        (telem.step as f32 / telem.max_steps as f32).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    ui.add(
                        egui::ProgressBar::new(progress)
                            .show_percentage()
                            .text(format!("ETA: {:.1}s", telem.eta_seconds)),
                    );

                    ui.add_space(4.0);

                    egui::Grid::new("telemetry_grid")
                        .num_columns(2)
                        .spacing([16.0, 4.0])
                        .show(ui, |ui| {
                            ui.label("Tokens Throughput:");
                            ui.label(
                                RichText::new(format!("{:.0} tok/s", telem.tokens_per_sec))
                                    .strong(),
                            );
                            ui.end_row();

                            ui.label("Sample Rate:");
                            ui.label(format!("{:.1} samples/s", telem.samples_per_sec));
                            ui.end_row();

                            ui.label("VRAM Usage / Peak:");
                            ui.label(format!(
                                "{:.1} MB / {:.1} MB",
                                telem.vram_used_mb, telem.vram_peak_mb
                            ));
                            ui.end_row();

                            ui.label("Muon Orthogonality Error:");
                            ui.label(format!(
                                "{:.2e} (NS Steps: {})",
                                telem.muon_ortho_error, state.runner.session.controls.muon_ns_steps
                            ));
                            ui.end_row();

                            ui.label("AutoClip Gradient Norm:");
                            ui.label(format!(
                                "{:.3} (Clip Thresh: {:.3})",
                                telem.grad_norm, telem.clip_threshold
                            ));
                            ui.end_row();

                            ui.label("Dynamic Quant Target:");
                            ui.label(format!("{:.2} avg bits", telem.quant_avg_bits));
                            ui.end_row();

                            ui.label("Loss Velocity / SNR:");
                            ui.label(format!(
                                "{:+.3}/step | SNR: {:.2}",
                                telem.loss_velocity, telem.gradient_snr
                            ));
                            ui.end_row();

                            ui.label("Grokking Dynamics:");
                            ui.label(
                                RichText::new(format!(
                                    "[{}] {}",
                                    telem.grokking_phase.badge(),
                                    telem.grokking_phase.label()
                                ))
                                .color(Color32::from_rgb(160, 230, 255))
                                .strong(),
                            );
                            ui.end_row();

                            ui.label("Weight Norm / Stable Rank:");
                            ui.label(format!(
                                "{:.2} (Stable Rank: {:.2})",
                                telem.weight_norm, telem.stable_rank
                            ));
                            ui.end_row();

                            ui.label("Generalization Gap:");
                            ui.label(format!(
                                "{:.4} (Val: {:.4} vs Train: {:.4})",
                                telem.generalization_gap, telem.val_loss_estimate, telem.loss
                            ));
                            ui.end_row();

                            if let Some(ref action) = telem.meta_tuned_action {
                                ui.label("Auto-Meta Tuning:");
                                ui.label(
                                    RichText::new(format!("⚡ {}", action))
                                        .size(11.0)
                                        .color(Color32::from_rgb(130, 240, 160)),
                                );
                                ui.end_row();
                            }
                        });
                });
            });
        });
    }

    fn render_live_optimizations(&self, ui: &mut Ui, state: &mut StudioState) {
        ScrollArea::vertical().max_height(240.0).show(ui, |ui| {
            // 1. Automated Meta-Parameter Optimization Panel
            ui.group(|ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("⚡ Automated Meta-Parameter Optimizer")
                            .strong()
                            .color(Color32::from_rgb(130, 240, 170)),
                    );
                    ui.checkbox(
                        &mut state.runner.session.meta_tuner.enabled,
                        "Enable Online Meta-Tuning",
                    );
                });

                ui.horizontal(|ui| {
                    ui.label(RichText::new("Optimization Policy:").strong());
                    egui::ComboBox::from_id_salt("meta_tuning_policy_combo")
                        .selected_text(state.runner.session.meta_tuner.policy.label())
                        .show_ui(ui, |ui| {
                            for policy in MetaTuningPolicy::ALL {
                                ui.selectable_value(
                                    &mut state.runner.session.meta_tuner.policy,
                                    *policy,
                                    policy.label(),
                                );
                            }
                        });
                });
                ui.label(
                    RichText::new(state.runner.session.meta_tuner.policy.description())
                        .size(11.0)
                        .color(Color32::from_rgb(170, 185, 205)),
                );

                ui.separator();

                ui.horizontal_wrapped(|ui| {
                    ui.checkbox(
                        &mut state.runner.session.meta_tuner.tune_learning_rate,
                        "LR Velocity Adaptation",
                    );
                    ui.checkbox(
                        &mut state.runner.session.meta_tuner.tune_clipping_threshold,
                        "AutoClip Saturation Guard",
                    );
                    ui.checkbox(
                        &mut state.runner.session.meta_tuner.tune_grokfast_filtering,
                        "Grokfast SNR Filter",
                    );
                    ui.checkbox(
                        &mut state.runner.session.meta_tuner.tune_muon_ns_iterations,
                        "Muon Polar Ortho Loop",
                    );
                    ui.checkbox(
                        &mut state.runner.session.meta_tuner.tune_quant_schedule,
                        "Progressive QAT Schedule",
                    );
                    ui.checkbox(
                        &mut state.runner.session.meta_tuner.tune_weight_decay,
                        "Weight Decay Norm Pressure",
                    );
                    ui.checkbox(
                        &mut state.runner.session.meta_tuner.tune_sam_flatness,
                        "SAM Flatness Optimization",
                    );
                    ui.checkbox(
                        &mut state.runner.session.meta_tuner.eos_guidance,
                        "Edge-of-Stability Guidance",
                    );
                });

                ui.horizontal(|ui| {
                    ui.label(format!(
                        "Loss Velocity: {:+.4}/step",
                        state.runner.session.current_telemetry.loss_velocity
                    ));
                    ui.separator();
                    ui.label(format!(
                        "Gradient SNR: {:.2}",
                        state.runner.session.current_telemetry.gradient_snr
                    ));
                    ui.separator();
                    ui.label(format!(
                        "Total Meta-Interventions: {}",
                        state.runner.session.meta_state.total_adjustments
                    ));
                    if let Some(ref action) = state.runner.session.meta_state.last_action {
                        ui.separator();
                        ui.label(
                            RichText::new(format!("Latest: {}", action))
                                .size(11.0)
                                .color(Color32::from_rgb(140, 230, 170)),
                        );
                    }
                });
            });

            ui.add_space(4.0);

            // 2. Manual Hyperparameter Sliders & Controls
            ui.group(|ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(format!("{} Live Hyperparameter Controls", bi::SLIDERS))
                            .strong()
                            .color(Color32::from_rgb(120, 200, 255)),
                    );
                    ui.label(
                        RichText::new(if state.runner.session.meta_tuner.enabled {
                            "Auto-tuner active. Sliders provide user bias and boundary limits."
                        } else {
                            "Manual mode. Modifications apply immediately on next step."
                        })
                        .size(11.0)
                        .color(Color32::from_rgb(160, 170, 180)),
                    );
                });

                ui.separator();

                egui::Grid::new("live_optimizations_grid")
                    .num_columns(3)
                    .spacing([20.0, 8.0])
                    .show(ui, |ui| {
                        // 1. Learning Rate Multiplier
                        ui.label(RichText::new("Learning Rate Multiplier:").strong());
                        ui.add(
                            egui::Slider::new(
                                &mut state.runner.session.controls.lr_scale,
                                0.01..=5.0,
                            )
                            .text("Scale")
                            .logarithmic(true),
                        );
                        if ui.button("Reset (1.0x)").clicked() {
                            state.runner.session.controls.lr_scale = 1.0;
                        }
                        ui.end_row();

                        // 2. AutoClip
                        ui.label(RichText::new("AutoClip Hygiene:").strong());
                        ui.checkbox(
                            &mut state.runner.session.controls.auto_clip_enabled,
                            "Enable Adaptive Clipping",
                        );
                        if state.runner.session.controls.auto_clip_enabled {
                            ui.add(
                                egui::Slider::new(
                                    &mut state.runner.session.controls.clip_percentile,
                                    50.0..=99.0,
                                )
                                .text("Percentile"),
                            );
                        } else {
                            ui.label("Disabled (fixed norm)");
                        }
                        ui.end_row();

                        // 3. Grokfast Fast-Weight Filtering
                        ui.label(RichText::new("Grokfast Low-Pass EMA:").strong());
                        ui.checkbox(
                            &mut state.runner.session.controls.grokfast_enabled,
                            "Filter Gradient Oscillations",
                        );
                        if state.runner.session.controls.grokfast_enabled {
                            ui.horizontal(|ui| {
                                ui.add(
                                    egui::Slider::new(
                                        &mut state.runner.session.controls.grokfast_alpha,
                                        0.80..=0.999,
                                    )
                                    .text("Alpha"),
                                );
                                ui.add(
                                    egui::Slider::new(
                                        &mut state.runner.session.controls.grokfast_lamb,
                                        0.1..=5.0,
                                    )
                                    .text("Lambda"),
                                );
                            });
                        } else {
                            ui.label("Disabled");
                        }
                        ui.end_row();

                        // 4. Muon Newton-Schulz Steps
                        ui.label(RichText::new("Muon Newton-Schulz Iterations:").strong());
                        ui.add(
                            egui::Slider::new(
                                &mut state.runner.session.controls.muon_ns_steps,
                                1..=10,
                            )
                            .text("Steps (Matrix Polar Orthogonalization)"),
                        );
                        ui.label(format!(
                            "Residual bound: {:.2e}",
                            0.5f64.powi(state.runner.session.controls.muon_ns_steps as i32)
                        ));
                        ui.end_row();

                        // 5. Dynamic Quantization Target Bitwidth
                        ui.label(RichText::new("Target Quantization Bitwidth:").strong());
                        ui.add(
                            egui::Slider::new(
                                &mut state.runner.session.controls.target_quant_bits,
                                1.58..=16.0,
                            )
                            .text("Bits per Parameter (QAT Dynamic Target)"),
                        );
                        ui.label(format!(
                            "Memory saving: {:.1}%",
                            (1.0 - (state.runner.session.controls.target_quant_bits / 16.0))
                                * 100.0
                        ));
                        ui.end_row();

                        // 6. Activation Checkpointing
                        ui.label(RichText::new("Activation Checkpointing:").strong());
                        ui.checkbox(
                            &mut state.runner.session.controls.activation_checkpointing,
                            "Trade compute for ~35% VRAM savings",
                        );
                        ui.label(if state.runner.session.controls.activation_checkpointing {
                            "Active (Reduced peak VRAM)"
                        } else {
                            "Full activations stored"
                        });
                        ui.end_row();

                        // 7. Optimizer Family
                        ui.label(RichText::new("Active Optimizer Family:").strong());
                        egui::ComboBox::from_id_salt("optimizer_family_live")
                            .selected_text(state.runner.session.controls.optimizer_family.label())
                            .show_ui(ui, |ui| {
                                for opt in OptimizerFamily::ALL {
                                    ui.selectable_value(
                                        &mut state.runner.session.controls.optimizer_family,
                                        *opt,
                                        opt.label(),
                                    );
                                }
                            });
                        ui.label("Dynamically redistributes weights across optimizers");
                        ui.end_row();

                        // 8. Weight Decay Normative Pressure (Grokking Drive)
                        ui.label(RichText::new("Weight Decay Norm Pressure:").strong());
                        ui.add(
                            egui::Slider::new(
                                &mut state.runner.session.controls.weight_decay_scale,
                                0.0..=10.0,
                            )
                            .text("Scale (L2 Compression for Grokking)"),
                        );
                        if ui.button("Reset (1.0x)").clicked() {
                            state.runner.session.controls.weight_decay_scale = 1.0;
                        }
                        ui.end_row();

                        // 9. Sharpness-Aware Minimization (SAM Flatness)
                        ui.label(RichText::new("SAM Flatness Radius (ρ):").strong());
                        ui.add(
                            egui::Slider::new(
                                &mut state.runner.session.controls.sam_rho,
                                0.01..=0.25,
                            )
                            .text("Radius (Anti-Sharp Minimum Curvature Bound)"),
                        );
                        ui.label(format!(
                            "Curvature bound: ρ = {:.3} (Hessian flatness)",
                            state.runner.session.controls.sam_rho
                        ));
                        ui.end_row();
                    });
            });
        });
    }

    fn render_environment_setup(&self, ui: &mut Ui, state: &mut StudioState) {
        ScrollArea::vertical().max_height(220.0).show(ui, |ui| {
            ui.group(|ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(format!("{} Turnkey Training Workspace & Launch Generator", bi::FOLDER2_OPEN))
                            .strong()
                            .color(Color32::from_rgb(120, 200, 255)),
                    );
                });
                ui.label("Generates complete virtual environments (uv / Conda), production launch scripts, model harnesses, loss functions, and data pipelines on disk.");

                ui.add_space(4.0);

                ui.horizontal(|ui| {
                    ui.label("Target Directory:");
                    ui.text_edit_singleline(&mut state.runner.env_setup_target_dir);

                    if ui
                        .button(
                            RichText::new(format!("{} Generate Turnkey Environment", bi::ROCKET_TAKEOFF))
                                .color(Color32::from_rgb(120, 220, 140))
                                .strong(),
                        )
                        .clicked()
                    {
                        let target_path = Path::new(&state.runner.env_setup_target_dir);
                        match shared::training::setup_training_environment(&state.spec, target_path) {
                            Ok(manifest) => {
                                state.runner.push_log(format!(
                                    "[Environment Created] Turnkey workspace generated at '{}' ({} files created).",
                                    manifest.target_dir,
                                    manifest.created_files.len()
                                ));
                                state.runner.env_setup_status_msg = Some(format!(
                                    "Successfully created {} files in '{}'",
                                    manifest.created_files.len(),
                                    manifest.target_dir
                                ));
                                state.runner.env_setup_result = Some(manifest);
                            }
                            Err(e) => {
                                state.runner.push_log(format!("[Setup Error] Failed to generate environment: {}", e));
                                state.runner.env_setup_status_msg = Some(format!("Error: {}", e));
                            }
                        }
                    }
                });

                if let Some(ref msg) = state.runner.env_setup_status_msg {
                    ui.label(
                        RichText::new(msg)
                            .color(if msg.starts_with("Error") {
                                Color32::from_rgb(255, 120, 120)
                            } else {
                                Color32::from_rgb(120, 220, 140)
                            })
                            .strong(),
                    );
                }

                if let Some(ref manifest) = state.runner.env_setup_result {
                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(format!("Model ID: {}", manifest.model_id)).strong());
                        ui.label(format!("Parameters: {}", manifest.total_parameters));
                        ui.label(format!("Est. VRAM: {:.2} GB", manifest.estimated_vram_gb));
                        ui.label(format!("Python: {}", manifest.python_version_req));
                        ui.label(format!("PyTorch: {}", manifest.torch_version_req));
                        ui.label(format!("CUDA Support: {}", if manifest.has_cuda_support { "Yes" } else { "CPU Fallback" }));
                    });

                    ui.add_space(4.0);
                    ui.label(RichText::new("Created Workspace Files:").strong());
                    ui.horizontal_wrapped(|ui| {
                        for file in &manifest.created_files {
                            ui.group(|ui| {
                                ui.label(RichText::new(file).font(egui::FontId::monospace(10.0)).color(Color32::from_rgb(180, 220, 255)));
                            });
                        }
                    });

                    ui.add_space(4.0);
                    ui.group(|ui| {
                        ui.label(RichText::new("Recommended Launch Command:").strong());
                        ui.label(
                            RichText::new(format!(
                                "cd {} && {}",
                                manifest.target_dir, manifest.recommended_launch_command
                            ))
                            .font(egui::FontId::monospace(11.0))
                            .color(Color32::from_rgb(140, 240, 160)),
                        );
                    });
                }
            });
        });
    }

    fn render_console_log(&self, ui: &mut Ui, state: &mut StudioState) {
        ui.horizontal(|ui| {
            ui.label(RichText::new("Mediated Process Stdout / Stderr").strong());
            if ui
                .button(RichText::new(format!("{} Clear", bi::TRASH3)).size(11.0))
                .clicked()
            {
                state.runner.console_logs.clear();
            }
            if ui
                .button(RichText::new(format!("{} Copy Logs", bi::CLIPBOARD_CHECK)).size(11.0))
                .clicked()
            {
                let text = state.runner.console_logs.join("\n");
                ui.ctx().copy_text(text);
            }
        });

        ScrollArea::vertical()
            .stick_to_bottom(true)
            .max_height(175.0)
            .show(ui, |ui| {
                ui.group(|ui| {
                    for log in &state.runner.console_logs {
                        let color = if log.starts_with("[Curriculum") {
                            Color32::from_rgb(255, 210, 100)
                        } else if log.starts_with("[Completed") {
                            Color32::from_rgb(120, 240, 150)
                        } else if log.starts_with("[Stopped") || log.starts_with("[Setup Error") {
                            Color32::from_rgb(255, 120, 120)
                        } else if log.starts_with("[Paused") {
                            Color32::from_rgb(255, 180, 80)
                        } else if log.starts_with("Step ") {
                            Color32::from_rgb(180, 220, 255)
                        } else {
                            Color32::from_rgb(200, 220, 200)
                        };

                        ui.label(
                            RichText::new(log)
                                .font(egui::FontId::monospace(10.0))
                                .color(color),
                        );
                    }
                });
            });
    }
}
