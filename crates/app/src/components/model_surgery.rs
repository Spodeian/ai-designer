//! Model Surgery, Mechanistic Abliteration & Model Soups Workbench View.
//! Provides interactive controls and geometric visualizers for directional weight orthogonalization,
//! LEACE concept erasure, contrastive activation addition, model soup synthesis (Uniform, Greedy,
//! Fisher, SLERP, Task Arithmetic, TIES, DARE), and Frankenmerge layer depth grafting.

use crate::icons::bi;
use crate::studio_state::{StudioState, SurgeryTab};
use eframe::egui::{self, Color32, Pos2, RichText, ScrollArea, Stroke, Ui, Vec2};
use shared::surgery::{
    AbliterationTechnique, GraftingStrategy, ModelSoupStrategy, SoupCheckpoint,
    SurgeryTargetConcept, execute_abliteration, execute_layer_graft, execute_soup_merge,
};

pub struct ModelSurgeryView;

impl Default for ModelSurgeryView {
    fn default() -> Self {
        Self
    }
}

impl ModelSurgeryView {
    pub fn render(&mut self, ui: &mut Ui, state: &mut StudioState) {
        ui.vertical(|ui| {
            ui.add_space(4.0);

            // 1. Header & Navigation Ribbon
            self.render_header_ribbon(ui, state);

            ui.add_space(4.0);
            ui.separator();

            // 2. Tab Body Content
            match state.surgery_tab {
                SurgeryTab::Abliteration => self.render_abliteration_tab(ui, state),
                SurgeryTab::ModelSoups => self.render_model_soups_tab(ui, state),
                SurgeryTab::LayerGrafting => self.render_layer_grafting_tab(ui, state),
                SurgeryTab::MechanisticProbes => self.render_mechanistic_probes_tab(ui, state),
            }
        });
    }

    fn render_header_ribbon(&self, ui: &mut Ui, state: &mut StudioState) {
        ui.horizontal(|ui| {
            ui.heading(
                RichText::new(format!(
                    "{} Model Surgery & Weight Manifold Laboratory",
                    bi::MAGIC
                ))
                .size(15.0)
                .strong(),
            );

            ui.separator();

            ui.selectable_value(
                &mut state.surgery_tab,
                SurgeryTab::Abliteration,
                RichText::new(format!(
                    "{} Refusal & Concept Abliteration",
                    bi::SHIELD_EXCLAMATION
                ))
                .strong(),
            );
            ui.selectable_value(
                &mut state.surgery_tab,
                SurgeryTab::ModelSoups,
                RichText::new(format!("{} Model Soups & Merging Lab", bi::LAYERS_FILL)).strong(),
            );
            ui.selectable_value(
                &mut state.surgery_tab,
                SurgeryTab::LayerGrafting,
                RichText::new(format!("{} Layer Grafting & Frankenmerge", bi::HDD_STACK)).strong(),
            );
            ui.selectable_value(
                &mut state.surgery_tab,
                SurgeryTab::MechanisticProbes,
                RichText::new(format!("{} Mechanistic Probes & SVD", bi::SEARCH)).strong(),
            );
        });
    }

    fn render_abliteration_tab(&self, ui: &mut Ui, state: &mut StudioState) {
        ui.columns(2, |columns| {
            // Left Column: Interactive Abliteration Controls
            columns[0].vertical(|ui| {
                ScrollArea::vertical().max_height(420.0).show(ui, |ui| {
                    ui.group(|ui| {
                        ui.label(
                            RichText::new("Abliteration & Subspace Surgery Parameters")
                                .strong()
                                .color(Color32::from_rgb(120, 200, 255)),
                        );
                        ui.add_space(4.0);

                        // Technique ComboBox
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("Surgical Technique:").strong());
                            egui::ComboBox::from_id_salt("abliteration_technique_combo")
                                .selected_text(state.abliteration_plan.technique.label())
                                .show_ui(ui, |ui| {
                                    for tech in AbliterationTechnique::ALL {
                                        ui.selectable_value(
                                            &mut state.abliteration_plan.technique,
                                            *tech,
                                            tech.label(),
                                        );
                                    }
                                });
                        });
                        ui.label(
                            RichText::new(state.abliteration_plan.technique.description())
                                .size(11.0)
                                .color(Color32::from_rgb(170, 185, 205)),
                        );

                        ui.add_space(4.0);
                        ui.separator();

                        // Target Concept
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("Target Concept:").strong());
                            let current_label = state.abliteration_plan.target_concept.label();
                            egui::ComboBox::from_id_salt("abliteration_concept_combo")
                                .selected_text(current_label)
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(
                                        &mut state.abliteration_plan.target_concept,
                                        SurgeryTargetConcept::RefusalAndSafety,
                                        "Refusal Subspace (Safety Alignment)",
                                    );
                                    ui.selectable_value(
                                        &mut state.abliteration_plan.target_concept,
                                        SurgeryTargetConcept::ToxicityAndHarm,
                                        "Toxicity & Harm Induction",
                                    );
                                    ui.selectable_value(
                                        &mut state.abliteration_plan.target_concept,
                                        SurgeryTargetConcept::SycophancyAndFlattery,
                                        "Sycophancy & Opinion Flattery",
                                    );
                                    ui.selectable_value(
                                        &mut state.abliteration_plan.target_concept,
                                        SurgeryTargetConcept::HallucinationAndOverconfidence,
                                        "Hallucination & Logit Overconfidence",
                                    );
                                });
                        });

                        ui.add_space(4.0);

                        // Sliders Grid
                        egui::Grid::new("abliteration_controls_grid")
                            .num_columns(2)
                            .spacing([16.0, 6.0])
                            .show(ui, |ui| {
                                ui.label(RichText::new("Layer Range:").strong());
                                ui.horizontal(|ui| {
                                    ui.add(
                                        egui::Slider::new(
                                            &mut state.abliteration_plan.layer_start,
                                            0..=32,
                                        )
                                        .text("Start"),
                                    );
                                    ui.add(
                                        egui::Slider::new(
                                            &mut state.abliteration_plan.layer_end,
                                            0..=32,
                                        )
                                        .text("End"),
                                    );
                                });
                                ui.end_row();

                                ui.label(RichText::new("Projection Strength:").strong());
                                ui.add(
                                    egui::Slider::new(
                                        &mut state.abliteration_plan.projection_strength,
                                        0.0..=2.0,
                                    )
                                    .text("Scalar Factor (1.0 = Pure Null Space)"),
                                );
                                ui.end_row();

                                ui.label(RichText::new("Anti-Degeneration:").strong());
                                ui.checkbox(
                                    &mut state.abliteration_plan.anti_degeneration_guard,
                                    "Preserve math/coding manifold (PCA nulling)",
                                );
                                ui.end_row();

                                if state.abliteration_plan.technique
                                    == AbliterationTechnique::LeaceConceptErasure
                                {
                                    ui.label(RichText::new("LEACE Covariance Alpha:").strong());
                                    ui.add(
                                        egui::Slider::new(
                                            &mut state.abliteration_plan.leace_alpha,
                                            1e-5..=1e-1,
                                        )
                                        .logarithmic(true)
                                        .text("Regularization"),
                                    );
                                    ui.end_row();
                                }
                            });

                        ui.add_space(6.0);

                        // Execute Button
                        if ui
                            .button(
                                RichText::new(format!(
                                    "{} Execute Abliteration Surgery",
                                    bi::SCISSORS
                                ))
                                .color(Color32::from_rgb(255, 120, 120))
                                .strong(),
                            )
                            .clicked()
                        {
                            let res = execute_abliteration(&state.spec, &state.abliteration_plan);
                            state.abliteration_result = Some(res);
                        }
                    });

                    // Result card
                    if let Some(ref res) = state.abliteration_result {
                        ui.add_space(4.0);
                        ui.group(|ui| {
                            ui.label(
                                RichText::new("Surgical Verification Postmortem")
                                    .strong()
                                    .color(Color32::from_rgb(130, 240, 160)),
                            );
                            ui.label(format!("Status: {}", res.status));

                            ui.horizontal(|ui| {
                                ui.label(format!(
                                    "Refusal Suppression: {:.1}%",
                                    res.refusal_suppression_pct
                                ));
                                ui.add(
                                    egui::ProgressBar::new(
                                        (res.refusal_suppression_pct / 100.0) as f32,
                                    )
                                    .fill(Color32::from_rgb(100, 220, 140)),
                                );
                            });

                            ui.horizontal(|ui| {
                                ui.label(format!(
                                    "Capability Retention: {:.1}%",
                                    res.benign_capability_retention_pct
                                ));
                                ui.add(
                                    egui::ProgressBar::new(
                                        (res.benign_capability_retention_pct / 100.0) as f32,
                                    )
                                    .fill(Color32::from_rgb(120, 200, 255)),
                                );
                            });

                            ui.label(format!(
                                "Subspace Angle: {:.2}° | Parameters Modified: {}",
                                res.subspace_angle_degrees, res.total_parameters_ablated
                            ));

                            ui.separator();
                            ui.label(RichText::new("Applied Actions:").strong().size(11.0));
                            for act in &res.applied_actions {
                                ui.label(
                                    RichText::new(format!("• {}", act))
                                        .size(10.5)
                                        .color(Color32::from_rgb(190, 200, 215)),
                                );
                            }
                        });
                    }
                });
            });

            // Right Column: Interactive Subspace Vector Diagram
            columns[1].vertical(|ui| {
                ui.label(RichText::new("Geometric Subspace Projection Diagram").strong());
                ui.label(
                    RichText::new("W' = W - r(r^T W) renders model blind to refusal direction r")
                        .size(11.0)
                        .color(Color32::from_rgb(170, 185, 205)),
                );

                let (rect, _) = ui.allocate_exact_size(
                    Vec2::new(ui.available_width(), 240.0),
                    egui::Sense::hover(),
                );
                let painter = ui.painter_at(rect);

                // Background
                painter.rect_filled(rect, 4.0, Color32::from_rgb(18, 22, 30));
                painter.rect_stroke(
                    rect,
                    4.0,
                    Stroke::new(1.0, Color32::from_rgb(45, 55, 75)),
                    egui::epaint::StrokeKind::Inside,
                );

                let center = Pos2::new(rect.min.x + 80.0, rect.max.y - 40.0);

                // Coordinate Axes
                let axis_stroke = Stroke::new(1.0, Color32::from_rgb(50, 60, 80));
                painter.line_segment(
                    [
                        Pos2::new(rect.min.x + 20.0, center.y),
                        Pos2::new(rect.max.x - 20.0, center.y),
                    ],
                    axis_stroke,
                );
                painter.line_segment(
                    [
                        Pos2::new(center.x, rect.max.y - 10.0),
                        Pos2::new(center.x, rect.min.y + 10.0),
                    ],
                    axis_stroke,
                );

                // Vector 1: Refusal Direction r (Amber/Yellow)
                let r_vec = Pos2::new(center.x + 160.0, center.y);
                painter.line_segment(
                    [center, r_vec],
                    Stroke::new(2.5, Color32::from_rgb(255, 200, 70)),
                );
                painter.circle_filled(r_vec, 4.0, Color32::from_rgb(255, 200, 70));
                painter.text(
                    Pos2::new(r_vec.x + 8.0, r_vec.y - 8.0),
                    egui::Align2::LEFT_CENTER,
                    "Refusal Direction r",
                    egui::FontId::proportional(11.0),
                    Color32::from_rgb(255, 200, 70),
                );

                // Vector 2: Original Weight W (Blue)
                let w_vec = Pos2::new(center.x + 120.0, center.y - 130.0);
                painter.line_segment(
                    [center, w_vec],
                    Stroke::new(2.0, Color32::from_rgb(100, 170, 255)),
                );
                painter.circle_filled(w_vec, 4.0, Color32::from_rgb(100, 170, 255));
                painter.text(
                    Pos2::new(w_vec.x + 6.0, w_vec.y - 6.0),
                    egui::Align2::LEFT_CENTER,
                    "Pre-Surgery W",
                    egui::FontId::proportional(11.0),
                    Color32::from_rgb(100, 170, 255),
                );

                // Vector 3: Surgically Ablated Weight W' = W (I - r r^T) (Cyan, orthogonal to r)
                let str_fac = state.abliteration_plan.projection_strength as f32;
                let w_proj_x = center.x + 120.0 * (1.0 - str_fac).max(0.0);
                let w_proj_y = center.y - 130.0;
                let w_proj = Pos2::new(w_proj_x, w_proj_y);

                painter.line_segment(
                    [center, w_proj],
                    Stroke::new(3.0, Color32::from_rgb(110, 240, 210)),
                );
                painter.circle_filled(w_proj, 5.0, Color32::from_rgb(110, 240, 210));
                painter.text(
                    Pos2::new(w_proj.x - 8.0, w_proj.y - 8.0),
                    egui::Align2::RIGHT_CENTER,
                    "Ablated W' (r ⊥ W')",
                    egui::FontId::proportional(11.5),
                    Color32::from_rgb(110, 240, 210),
                );

                // Orthogonal projection dashed connector
                painter.line_segment(
                    [w_vec, w_proj],
                    Stroke::new(1.0, Color32::from_rgb(180, 180, 180)),
                );

                // Right-angle marker at projection foot if strength == 1.0
                if (str_fac - 1.0).abs() < 0.15 {
                    let sz = 12.0;
                    painter.line_segment(
                        [
                            Pos2::new(center.x, center.y - sz),
                            Pos2::new(center.x + sz, center.y - sz),
                        ],
                        Stroke::new(1.0, Color32::from_rgb(110, 240, 210)),
                    );
                    painter.line_segment(
                        [
                            Pos2::new(center.x + sz, center.y - sz),
                            Pos2::new(center.x + sz, center.y),
                        ],
                        Stroke::new(1.0, Color32::from_rgb(110, 240, 210)),
                    );
                }
            });
        });
    }

    fn render_model_soups_tab(&self, ui: &mut Ui, state: &mut StudioState) {
        ui.columns(2, |columns| {
            // Left Column: Recipe Configuration
            columns[0].vertical(|ui| {
                ScrollArea::vertical().max_height(420.0).show(ui, |ui| {
                    ui.group(|ui| {
                        ui.label(
                            RichText::new("Model Soups & Merging Recipe")
                                .strong()
                                .color(Color32::from_rgb(120, 200, 255)),
                        );
                        ui.add_space(4.0);

                        // Strategy Selector
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("Merging Strategy:").strong());
                            egui::ComboBox::from_id_salt("soup_strategy_combo")
                                .selected_text(state.soup_recipe.strategy.label())
                                .show_ui(ui, |ui| {
                                    for strat in ModelSoupStrategy::ALL {
                                        ui.selectable_value(
                                            &mut state.soup_recipe.strategy,
                                            *strat,
                                            strat.label(),
                                        );
                                    }
                                });
                        });
                        ui.label(
                            RichText::new(state.soup_recipe.strategy.description())
                                .size(11.0)
                                .color(Color32::from_rgb(170, 185, 205)),
                        );

                        ui.separator();

                        // Strategy-Specific Hyperparameters
                        match state.soup_recipe.strategy {
                            ModelSoupStrategy::TiesMerging | ModelSoupStrategy::DareTies => {
                                ui.label(RichText::new("TIES Trim Density:").strong());
                                ui.add(
                                    egui::Slider::new(
                                        &mut state.soup_recipe.trim_density,
                                        0.05..=0.60,
                                    )
                                    .text("Top-p% parameters kept (rest zeroed)"),
                                );
                            }
                            ModelSoupStrategy::DareLinear => {
                                ui.label(RichText::new("DARE Drop Probability:").strong());
                                ui.add(
                                    egui::Slider::new(
                                        &mut state.soup_recipe.dare_drop_prob,
                                        0.10..=0.95,
                                    )
                                    .text("Random drop rate p"),
                                );
                            }
                            ModelSoupStrategy::Slerp => {
                                ui.label(RichText::new("SLERP Interpolation (t):").strong());
                                ui.add(
                                    egui::Slider::new(&mut state.soup_recipe.slerp_t, 0.0..=1.0)
                                        .text("0.0 = Model A, 1.0 = Model B"),
                                );
                            }
                            _ => {}
                        }

                        ui.add_space(4.0);
                        if ui
                            .button(
                                RichText::new(format!("{} Synthesize Model Soup", bi::LAYERS_FILL))
                                    .color(Color32::from_rgb(120, 220, 140))
                                    .strong(),
                            )
                            .clicked()
                        {
                            let res = execute_soup_merge(&state.spec, &state.soup_recipe);
                            state.soup_result = Some(res);
                        }
                    });

                    // Result card
                    if let Some(ref res) = state.soup_result {
                        ui.add_space(4.0);
                        ui.group(|ui| {
                            ui.label(
                                RichText::new("Soup Synthesis Report")
                                    .strong()
                                    .color(Color32::from_rgb(130, 240, 160)),
                            );
                            ui.label(format!("Status: {}", res.status));
                            ui.label(format!(
                                "Ensemble Gain: +{:.2}% | Sparsity: {:.1}% | Drift: {:.4}",
                                res.ensemble_gain_pct, res.sparsity_pct, res.weight_norm_drift
                            ));
                            ui.separator();
                            for log in &res.log_messages {
                                ui.label(
                                    RichText::new(format!("• {}", log))
                                        .size(10.5)
                                        .color(Color32::from_rgb(180, 195, 210)),
                                );
                            }
                        });
                    }
                });
            });

            // Right Column: Checkpoint Candidates Table
            columns[1].vertical(|ui| {
                ui.label(RichText::new("Candidate Checkpoints in Soup Pool").strong());
                ui.label(
                    RichText::new("Adjust task coefficients λ_k. Supports task negation (λ < 0).")
                        .size(11.0)
                        .color(Color32::from_rgb(170, 185, 205)),
                );

                ScrollArea::vertical().max_height(350.0).show(ui, |ui| {
                    let mut to_remove = None;
                    for (i, chk) in state.soup_recipe.checkpoints.iter_mut().enumerate() {
                        ui.group(|ui| {
                            ui.horizontal(|ui| {
                                ui.label(RichText::new(&chk.name).strong().color(if chk.is_base {
                                    Color32::from_rgb(120, 200, 255)
                                } else {
                                    Color32::from_rgb(220, 220, 220)
                                }));
                                ui.label(format!("({})", chk.task_label));
                                if !chk.is_base && ui.button("×").clicked() {
                                    to_remove = Some(i);
                                }
                            });
                            ui.horizontal(|ui| {
                                ui.label("Weight (λ):");
                                ui.add(
                                    egui::Slider::new(&mut chk.coefficient, -1.0..=2.0)
                                        .text("Scalar"),
                                );
                                ui.label(format!("Val Benchmark: {:.1}", chk.val_score));
                            });
                        });
                    }
                    if let Some(idx) = to_remove {
                        state.soup_recipe.checkpoints.remove(idx);
                    }

                    if ui
                        .button(format!("{} Add Fine-Tuned Checkpoint", bi::PLUS_LG))
                        .clicked()
                    {
                        let count = state.soup_recipe.checkpoints.len();
                        state.soup_recipe.checkpoints.push(SoupCheckpoint {
                            id: format!("chk_{count}"),
                            name: format!("Specialist Task #{count}"),
                            task_label: "Domain Adaptation".to_string(),
                            coefficient: 0.5,
                            val_score: 84.0,
                            is_base: false,
                        });
                    }
                });
            });
        });
    }

    fn render_layer_grafting_tab(&self, ui: &mut Ui, state: &mut StudioState) {
        ui.columns(2, |columns| {
            columns[0].vertical(|ui| {
                ui.group(|ui| {
                    ui.label(
                        RichText::new("Layer Grafting & Structural Surgery")
                            .strong()
                            .color(Color32::from_rgb(120, 200, 255)),
                    );

                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Grafting Strategy:").strong());
                        egui::ComboBox::from_id_salt("grafting_strategy_combo")
                            .selected_text(state.grafting_plan.strategy.label())
                            .show_ui(ui, |ui| {
                                for strat in GraftingStrategy::ALL {
                                    ui.selectable_value(
                                        &mut state.grafting_plan.strategy,
                                        *strat,
                                        strat.label(),
                                    );
                                }
                            });
                    });
                    ui.label(
                        RichText::new(state.grafting_plan.strategy.description())
                            .size(11.0)
                            .color(Color32::from_rgb(170, 185, 205)),
                    );

                    ui.separator();

                    match state.grafting_plan.strategy {
                        GraftingStrategy::FrankenmergeDepthStacking => {
                            ui.horizontal(|ui| {
                                ui.add(
                                    egui::Slider::new(
                                        &mut state.grafting_plan.source_layer_start,
                                        0..=32,
                                    )
                                    .text("Start Layer"),
                                );
                                ui.add(
                                    egui::Slider::new(
                                        &mut state.grafting_plan.source_layer_end,
                                        0..=32,
                                    )
                                    .text("End Layer"),
                                );
                            });
                            ui.add(
                                egui::Slider::new(
                                    &mut state.grafting_plan.target_depth_multiplier,
                                    1.0..=2.0,
                                )
                                .text("Depth Expansion Multiplier"),
                            );
                        }
                        GraftingStrategy::LoRAAdapterFusion => {
                            ui.add(
                                egui::Slider::new(&mut state.grafting_plan.lora_rank, 4..=128)
                                    .text("Adapter Rank (r)"),
                            );
                            ui.add(
                                egui::Slider::new(&mut state.grafting_plan.lora_alpha, 8.0..=256.0)
                                    .text("LoRA Alpha Scaling"),
                            );
                        }
                        GraftingStrategy::LayerPruningSurgery => {
                            ui.add(
                                egui::Slider::new(&mut state.grafting_plan.prune_count, 1..=8)
                                    .text("Layers to Prune (Identity Cosine)"),
                            );
                        }
                        GraftingStrategy::CrossArchitectureBridge => {
                            ui.label("Connects vision/speech encoder projections into autoregressive decoder backbone.");
                        }
                    }

                    ui.add_space(6.0);
                    if ui
                        .button(
                            RichText::new(format!("{} Execute Structural Graft", bi::HDD_STACK))
                                .color(Color32::from_rgb(120, 220, 140))
                                .strong(),
                        )
                        .clicked()
                    {
                        let res = execute_layer_graft(&state.spec, &state.grafting_plan);
                        state.grafting_result = Some(res);
                    }
                });

                if let Some(ref res) = state.grafting_result {
                    ui.add_space(4.0);
                    ui.group(|ui| {
                        ui.label(
                            RichText::new("Grafting Execution Result")
                                .strong()
                                .color(Color32::from_rgb(130, 240, 160)),
                        );
                        ui.label(format!("Status: {}", res.status));
                        ui.label(format!(
                            "Layer Depth: {} -> {} | Parameter Delta: {:+}",
                            res.initial_layers, res.final_layers, res.total_parameters_delta
                        ));
                        for msg in &res.log_messages {
                            ui.label(
                                RichText::new(format!("• {}", msg))
                                    .size(10.5)
                                    .color(Color32::from_rgb(180, 195, 210)),
                            );
                        }
                    });
                }
            });

            columns[1].vertical(|ui| {
                ui.label(RichText::new("Layer Depth Architecture Profile").strong());
                ui.label(
                    RichText::new("Visualizing semantic block duplication and residual bridges")
                        .size(11.0)
                        .color(Color32::from_rgb(170, 185, 205)),
                );

                let (rect, _) = ui.allocate_exact_size(
                    Vec2::new(ui.available_width(), 220.0),
                    egui::Sense::hover(),
                );
                let painter = ui.painter_at(rect);
                painter.rect_filled(rect, 4.0, Color32::from_rgb(18, 22, 30));
                painter.rect_stroke(
                    rect,
                    4.0,
                    Stroke::new(1.0, Color32::from_rgb(45, 55, 75)),
                    egui::epaint::StrokeKind::Inside,
                );

                // Draw layer stacking ladder
                let layers = 16;
                let bar_width = (rect.width() - 40.0) / layers as f32;
                for i in 0..layers {
                    let is_grafted = i >= state.grafting_plan.source_layer_start.min(10)
                        && i <= state.grafting_plan.source_layer_end.min(14);
                    let color = if is_grafted {
                        Color32::from_rgb(110, 220, 170)
                    } else {
                        Color32::from_rgb(60, 90, 130)
                    };
                    let bar_x = rect.min.x + 20.0 + (i as f32 * bar_width);
                    let bar_rect = egui::Rect::from_min_size(
                        Pos2::new(bar_x, rect.max.y - 60.0),
                        Vec2::new(bar_width - 2.0, 45.0),
                    );
                    painter.rect_filled(bar_rect, 2.0, color);
                }
                painter.text(
                    Pos2::new(rect.min.x + 20.0, rect.max.y - 75.0),
                    egui::Align2::LEFT_BOTTOM,
                    "Green blocks = duplicated semantic layers (Solar Frankenmerge)",
                    egui::FontId::proportional(11.0),
                    Color32::from_rgb(160, 210, 180),
                );
            });
        });
    }

    fn render_mechanistic_probes_tab(&self, ui: &mut Ui, _state: &mut StudioState) {
        ui.vertical(|ui| {
            ui.label(RichText::new("Mechanistic SVD Energy Spectrum & Concept Probing").strong());
            ui.label(
                RichText::new("Singular value decomposition (SVD) of weight residual updates across layers. Demonstrates that refusal and bias concentrate into top-1 singular vectors.")
                    .size(11.0)
                    .color(Color32::from_rgb(170, 185, 205)),
            );

            ui.add_space(4.0);

            let (rect, _) = ui.allocate_exact_size(
                Vec2::new(ui.available_width(), 200.0),
                egui::Sense::hover(),
            );
            let painter = ui.painter_at(rect);
            painter.rect_filled(rect, 4.0, Color32::from_rgb(18, 22, 30));
            painter.rect_stroke(
                rect,
                4.0,
                Stroke::new(1.0, Color32::from_rgb(45, 55, 75)),
                egui::epaint::StrokeKind::Inside,
            );

            // SVD Spectrum curve (Simulated singular value decay)
            let points = 24;
            let bar_w = (rect.width() - 40.0) / points as f32;
            for i in 0..points {
                let sigma = 140.0 * (-0.25 * i as f64).exp();
                let x = rect.min.x + 20.0 + (i as f32 * bar_w);
                let y = rect.max.y - 25.0 - sigma as f32;
                let color = if i == 0 {
                    Color32::from_rgb(255, 100, 100) // Top refusal direction
                } else {
                    Color32::from_rgb(100, 180, 255)
                };
                painter.rect_filled(
                    egui::Rect::from_min_max(Pos2::new(x, y), Pos2::new(x + bar_w - 2.0, rect.max.y - 25.0)),
                    2.0,
                    color,
                );
            }
            painter.text(
                Pos2::new(rect.min.x + 25.0, rect.min.y + 20.0),
                egui::Align2::LEFT_TOP,
                "σ_1 (Red) = Dominant Refusal Vector (78.4% of Concept Variance). Excising σ_1 leaves remaining reasoning subspace intact.",
                egui::FontId::proportional(11.0),
                Color32::from_rgb(255, 140, 140),
            );
        });
    }
}
