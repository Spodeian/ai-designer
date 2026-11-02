//! Architecture Design & Scaffolding Wizard: guided creation of AI models from scratch,
//! architecture archetypes (LLM, Vision DiT, FNO PINN, ACT Policy, Forecaster), or blank slate.

use crate::components::palette::{TEMPLATES, get_template_by_filename};
use crate::icons::bi;
use crate::studio_state::{CanvasMode, StudioState};
use eframe::egui::{self, Color32, CornerRadius, RichText, ScrollArea, Stroke, Ui, Vec2};
use shared::spec::training::TrainingPlanPreset;
use shared::{ArchitectureArchetype, PrecisionFormat, scaffold_archetype};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WizardTab {
    #[default]
    Archetypes,
    StarterTemplates,
    TrainingPlanner,
    BlankSlate,
}

pub struct DesignWizardModal {
    pub is_open: bool,
    pub current_tab: WizardTab,
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
            current_tab: WizardTab::Archetypes,
            selected_archetype: ArchitectureArchetype::GenerativeLLM,
            selected_training_preset: TrainingPlanPreset::FrontierLlm4StagePipeline,
            training_num_gpus: 8,
            training_seq_len: 4096,
            model_id: "my_custom_ai".to_string(),
            display_name: "My Custom AI Model".to_string(),
            d_model: 512,
            depth: 4,
            precision: PrecisionFormat::BF16,
            search_query: String::new(),
            selected_template_index: None,
        }
    }
}

impl DesignWizardModal {
    pub fn open(&mut self) {
        self.is_open = true;
        self.apply_archetype_defaults(self.selected_archetype);
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

    pub fn render(&mut self, ctx: &egui::Context, studio: &mut StudioState) {
        if !self.is_open {
            return;
        }

        let mut should_close = false;
        let mut should_create = false;

        egui::Window::new(
            RichText::new(format!("{} AI Architecture & Training Plan Wizard", bi::MAGIC))
                .size(18.0)
                .strong(),
        )
        .collapsible(false)
        .resizable(true)
        .default_size(Vec2::new(800.0, 600.0))
        .anchor(egui::Align2::CENTER_CENTER, Vec2::ZERO)
        .show(ctx, |ui| {
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new("Scaffold a model architecture, load an industry blueprint, or auto-configure a multi-stage Training Plan:")
                        .size(13.0)
                        .color(Color32::from_rgb(200, 210, 225)),
                );
            });

            ui.add_space(8.0);
            ui.horizontal_wrapped(|ui| {
                if ui
                    .selectable_label(
                        self.current_tab == WizardTab::Archetypes,
                        format!(
                            "{} Guided Archetypes ({})",
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
                            "{} Industry & Starter Blueprints ({})",
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
                        format!("{} Blank Canvas (Clean Slate)", bi::FILE_EARMARK_TEXT),
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
                WizardTab::Archetypes => self.render_archetypes_tab(ui),
                WizardTab::StarterTemplates => self.render_templates_tab(ui),
                WizardTab::TrainingPlanner => self.render_training_planner_tab(ui, studio),
                WizardTab::BlankSlate => self.render_blank_tab(ui),
            }

            ui.add_space(10.0);
            ui.separator();

            // Bottom Actions
            ui.horizontal(|ui| {
                if ui.button("Cancel").clicked() {
                    should_close = true;
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let action_label = match self.current_tab {
                        WizardTab::Archetypes => {
                            format!("{} Create & Scaffold Architecture", bi::ROCKET_TAKEOFF)
                        }
                        WizardTab::StarterTemplates => {
                            format!("{} Load Starter Template", bi::ROCKET_TAKEOFF)
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
                                .size(14.0)
                                .color(Color32::from_rgb(120, 230, 150))
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
