//! Workspace & Multi-Model Project Manager Modal.
//!
//! Provides UI controls to manage multiple research workspaces, hot-swap between
//! individual model designs (e.g. Encoder + Backbone + Decoder suites), duplicate
//! ablation variants, export/import complete workspace packages, and monitor
//! WebAssembly vs. Desktop parity status.

use crate::icons::bi;
use crate::studio_state::StudioState;
use eframe::egui::{self, Color32, RichText, ScrollArea, Stroke, Ui, Vec2};
use shared::workspace::{Workspace, WorkspaceModelDesign, WorkspaceRegistry};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WorkspaceModalTab {
    #[default]
    ModelsInWorkspace,
    AllWorkspaces,
    WorkspaceNotes,
    DeploymentParity,
}

pub struct WorkspaceManagerModal {
    pub is_open: bool,
    pub selected_tab: WorkspaceModalTab,
    pub new_workspace_name: String,
    pub new_workspace_desc: String,
    pub new_model_name: String,
    pub import_buffer: String,
    pub show_import_box: bool,
    pub status_message: Option<(String, bool)>, // (message, is_error)
}

impl Default for WorkspaceManagerModal {
    fn default() -> Self {
        Self {
            is_open: false,
            selected_tab: WorkspaceModalTab::ModelsInWorkspace,
            new_workspace_name: String::new(),
            new_workspace_desc: String::new(),
            new_model_name: String::new(),
            import_buffer: String::new(),
            show_import_box: false,
            status_message: None,
        }
    }
}

impl WorkspaceManagerModal {
    pub fn render(
        &mut self,
        ctx: &egui::Context,
        registry: &mut WorkspaceRegistry,
        studio: &mut StudioState,
    ) {
        if !self.is_open {
            return;
        }

        let mut is_open = self.is_open;
        egui::Window::new(
            RichText::new(format!(
                "{} Workspace & Multi-Model Project Manager",
                bi::FOLDER2_OPEN
            ))
            .strong(),
        )
        .open(&mut is_open)
        .resizable(true)
        .default_width(820.0)
        .default_height(580.0)
        .anchor(egui::Align2::CENTER_CENTER, Vec2::ZERO)
        .show(ctx, |ui| {
            self.render_content(ui, registry, studio);
        });

        self.is_open = is_open;
    }

    fn render_content(
        &mut self,
        ui: &mut Ui,
        registry: &mut WorkspaceRegistry,
        studio: &mut StudioState,
    ) {
        // Top tab header
        ui.horizontal(|ui| {
            let active_ws_name = registry
                .active_workspace()
                .map(|w| w.name.as_str())
                .unwrap_or("Workspace");

            ui.selectable_value(
                &mut self.selected_tab,
                WorkspaceModalTab::ModelsInWorkspace,
                RichText::new(format!(
                    "{} Models in Active Workspace ({})",
                    bi::CPU,
                    registry
                        .active_workspace()
                        .map(|w| w.designs.len())
                        .unwrap_or(0)
                ))
                .strong(),
            );

            ui.selectable_value(
                &mut self.selected_tab,
                WorkspaceModalTab::AllWorkspaces,
                RichText::new(format!(
                    "{} Manage Workspaces ({})",
                    bi::COLLECTION,
                    registry.workspaces.len()
                ))
                .strong(),
            );

            ui.selectable_value(
                &mut self.selected_tab,
                WorkspaceModalTab::WorkspaceNotes,
                RichText::new(format!("{} Research Notes & Pipeline", bi::JOURNAL_CODE)).strong(),
            );

            ui.selectable_value(
                &mut self.selected_tab,
                WorkspaceModalTab::DeploymentParity,
                RichText::new(format!("{} Web/Desktop Parity", bi::COMPASS)).strong(),
            );

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(
                    RichText::new(format!("Active: {active_ws_name}"))
                        .color(Color32::from_rgb(120, 220, 140))
                        .small(),
                );
            });
        });

        ui.separator();

        if let Some((msg, is_err)) = &self.status_message {
            let color = if *is_err {
                Color32::from_rgb(240, 120, 120)
            } else {
                Color32::from_rgb(120, 220, 140)
            };
            ui.label(RichText::new(msg).color(color));
            ui.separator();
        }

        ScrollArea::vertical().show(ui, |ui| match self.selected_tab {
            WorkspaceModalTab::ModelsInWorkspace => {
                self.render_models_tab(ui, registry, studio);
            }
            WorkspaceModalTab::AllWorkspaces => {
                self.render_workspaces_tab(ui, registry, studio);
            }
            WorkspaceModalTab::WorkspaceNotes => {
                self.render_notes_tab(ui, registry);
            }
            WorkspaceModalTab::DeploymentParity => {
                self.render_parity_tab(ui);
            }
        });
    }

    fn render_models_tab(
        &mut self,
        ui: &mut Ui,
        registry: &mut WorkspaceRegistry,
        studio: &mut StudioState,
    ) {
        let Some(active_ws) = registry.active_workspace_mut() else {
            ui.label("No active workspace found.");
            return;
        };

        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.heading(&active_ws.name);
                if !active_ws.description.is_empty() {
                    ui.label(RichText::new(&active_ws.description).weak());
                }
            });

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .button(format!("{} + Blank Model", bi::PLUS_LG))
                    .on_hover_text("Add a new blank model to this workspace")
                    .clicked()
                {
                    let count = active_ws.designs.len() + 1;
                    let id = format!("model_{count}");
                    let name = format!("Model #{count}");
                    let spec = shared::AiDesignSpec::new(&id, &name);
                    let design = WorkspaceModelDesign::new(&id, &name, spec);
                    active_ws.add_design(design, true);
                    Self::sync_active_to_studio(active_ws, studio);
                    self.status_message =
                        Some((format!("Created new model '{name}' and activated."), false));
                }

                if ui
                    .button(format!("{} Clone Active", bi::COPY))
                    .on_hover_text(
                        "Duplicate the active model into an ablation variant (keeps original intact)",
                    )
                    .clicked()
                {
                    let active_id = active_ws.active_design_id.clone();
                    if let Some(new_id) = active_ws.duplicate_design(&active_id) {
                        Self::sync_active_to_studio(active_ws, studio);
                        self.status_message = Some((
                            format!("Cloned model '{active_id}' -> variant '{new_id}'"),
                            false,
                        ));
                    }
                }
            });
        });

        ui.add_space(8.0);
        ui.label(RichText::new("Models Configured in this Workspace:").strong());

        let mut to_activate = None;
        let mut to_duplicate = None;
        let mut to_remove = None;
        let active_id = active_ws.active_design_id.clone();

        for design in &active_ws.designs {
            let is_active = design.id == active_id;
            let bg_color = if is_active {
                Color32::from_rgb(32, 48, 40)
            } else {
                Color32::from_rgb(26, 28, 34)
            };

            egui::Frame::new()
                .fill(bg_color)
                .corner_radius(6.0)
                .stroke(Stroke::new(
                    1.0,
                    if is_active {
                        Color32::from_rgb(120, 220, 140)
                    } else {
                        Color32::from_rgb(60, 65, 75)
                    },
                ))
                .inner_margin(10.0)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.vertical(|ui| {
                            ui.horizontal(|ui| {
                                ui.label(RichText::new(&design.name).strong().size(15.0));
                                if is_active {
                                    ui.label(
                                        RichText::new("● ACTIVE IN STUDIO")
                                            .color(Color32::from_rgb(120, 220, 140))
                                            .small()
                                            .strong(),
                                    );
                                }
                            });

                            ui.label(
                                RichText::new(format!(
                                    "ID: {} | Nodes: {} | Inputs: {} | Outputs: {} | Stages: {}",
                                    design.id,
                                    design.spec.graph.nodes.len(),
                                    design.spec.inputs.len(),
                                    design.spec.outputs.len(),
                                    design.spec.training.stages.len().max(1),
                                ))
                                .weak()
                                .small(),
                            );
                        });

                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if active_ws.designs.len() > 1
                                && ui
                                    .button(
                                        RichText::new(format!("{} Delete", bi::TRASH3))
                                            .color(Color32::from_rgb(240, 120, 120)),
                                    )
                                    .clicked()
                            {
                                to_remove = Some(design.id.clone());
                            }

                            if ui
                                .button(format!("{} Duplicate", bi::COPY))
                                .on_hover_text("Duplicate as an experimental variant")
                                .clicked()
                            {
                                to_duplicate = Some(design.id.clone());
                            }

                            if !is_active
                                && ui
                                    .button(
                                        RichText::new(format!("{} Activate", bi::CHECK_LG))
                                            .color(Color32::from_rgb(120, 220, 140)),
                                    )
                                    .clicked()
                            {
                                to_activate = Some(design.id.clone());
                            }
                        });
                    });
                });
            ui.add_space(4.0);
        }

        if let Some(target_id) = to_activate {
            active_ws.set_active_design(&target_id);
            Self::sync_active_to_studio(active_ws, studio);
            self.status_message =
                Some((format!("Switched active design to '{target_id}'."), false));
        }
        if let Some(target_id) = to_duplicate {
            if let Some(new_id) = active_ws.duplicate_design(&target_id) {
                Self::sync_active_to_studio(active_ws, studio);
                self.status_message = Some((format!("Duplicated variant '{new_id}'."), false));
            }
        }
        if let Some(target_id) = to_remove {
            if active_ws.remove_design(&target_id) {
                Self::sync_active_to_studio(active_ws, studio);
                self.status_message = Some((format!("Removed design '{target_id}'."), false));
            }
        }
    }

    fn render_workspaces_tab(
        &mut self,
        ui: &mut Ui,
        registry: &mut WorkspaceRegistry,
        studio: &mut StudioState,
    ) {
        ui.horizontal(|ui| {
            ui.heading("Research Workspaces");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .button(format!("{} Import Workspace", bi::UPLOAD))
                    .clicked()
                {
                    self.show_import_box = !self.show_import_box;
                }
            });
        });

        if self.show_import_box {
            egui::Frame::new()
                .fill(Color32::from_rgb(30, 32, 40))
                .inner_margin(8.0)
                .corner_radius(4.0)
                .show(ui, |ui| {
                    ui.label("Paste workspace JSON document or drop .aidws binary / .json:");
                    ui.add(
                        egui::TextEdit::multiline(&mut self.import_buffer)
                            .desired_rows(4)
                            .desired_width(f32::INFINITY),
                    );
                    ui.horizontal(|ui| {
                        if ui.button("Import Document").clicked() {
                            match registry.import_workspace_bytes(self.import_buffer.as_bytes()) {
                                Ok((new_id, is_bin)) => {
                                    if let Some(ws) = registry.active_workspace() {
                                        Self::sync_active_to_studio(ws, studio);
                                    }
                                    let fmt_label = if is_bin { "Native Binary .aidws" } else { "JSON" };
                                    self.status_message = Some((
                                        format!("Successfully imported workspace '{new_id}' ({fmt_label})!"),
                                        false,
                                    ));
                                    self.show_import_box = false;
                                    self.import_buffer.clear();
                                }
                                Err(e) => {
                                    self.status_message =
                                        Some((format!("Import failed: {e}"), true));
                                }
                            }
                        }
                        if ui.button("Cancel").clicked() {
                            self.show_import_box = false;
                        }
                    });
                });
            ui.add_space(6.0);
        }

        ui.add_space(4.0);

        // Add new workspace row
        ui.horizontal(|ui| {
            ui.label("New Workspace:");
            ui.add(
                egui::TextEdit::singleline(&mut self.new_workspace_name)
                    .hint_text("Workspace Name (e.g. LLM Reasoning Lab)"),
            );
            if ui.button(format!("{} Create", bi::PLUS_LG)).clicked()
                && !self.new_workspace_name.trim().is_empty()
            {
                let name = self.new_workspace_name.trim().to_string();
                let id = name.to_lowercase().replace([' ', '-'], "_");
                let starter = studio.spec.clone();
                registry.create_workspace(&id, &name, starter);
                self.new_workspace_name.clear();
                self.status_message = Some((format!("Created workspace '{name}'!"), false));
            }
        });

        ui.add_space(8.0);

        let mut to_switch = None;
        let mut to_duplicate = None;
        let mut to_remove = None;
        let active_ws_id = registry.active_workspace_id.clone();

        for ws in &registry.workspaces {
            let is_active = ws.id == active_ws_id;
            let bg_color = if is_active {
                Color32::from_rgb(32, 44, 48)
            } else {
                Color32::from_rgb(24, 26, 32)
            };

            egui::Frame::new()
                .fill(bg_color)
                .corner_radius(6.0)
                .stroke(Stroke::new(
                    1.0,
                    if is_active {
                        Color32::from_rgb(120, 200, 240)
                    } else {
                        Color32::from_rgb(50, 55, 65)
                    },
                ))
                .inner_margin(10.0)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.vertical(|ui| {
                            ui.horizontal(|ui| {
                                ui.label(RichText::new(&ws.name).strong().size(15.0));
                                if is_active {
                                    ui.label(
                                        RichText::new("● ACTIVE WORKSPACE")
                                            .color(Color32::from_rgb(120, 200, 240))
                                            .small()
                                            .strong(),
                                    );
                                }
                            });
                            ui.label(
                                RichText::new(format!(
                                    "ID: {} | Models: {} | Active Model: {}",
                                    ws.id,
                                    ws.designs.len(),
                                    ws.active_design_id
                                ))
                                .weak()
                                .small(),
                            );
                        });

                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if registry.workspaces.len() > 1
                                && ui
                                    .button(
                                        RichText::new(format!("{} Delete", bi::TRASH3))
                                            .color(Color32::from_rgb(240, 120, 120)),
                                    )
                                    .clicked()
                            {
                                to_remove = Some(ws.id.clone());
                            }

                            if ui
                                .button(format!("{} Export (.json)", bi::FILE_EARMARK_CODE))
                                .on_hover_text(
                                    "Download portable JSON format of all models in this workspace",
                                )
                                .clicked()
                            {
                                if let Ok(json_str) = registry.export_workspace_json(&ws.id) {
                                    crate::storage_manager::trigger_text_download(
                                        &format!("{}.json", ws.id),
                                        &json_str,
                                        "application/json",
                                    );
                                    self.status_message = Some((
                                        format!(
                                            "Exported workspace JSON package '{}.json'",
                                            ws.id
                                        ),
                                        false,
                                    ));
                                }
                            }

                            if ui
                                .button(format!("{} Export (.aidws)", bi::FILE_EARMARK_BINARY))
                                .on_hover_text(
                                    "Download high-performance compressed binary container (.aidws with CRC32 checksum)",
                                )
                                .clicked()
                            {
                                if let Ok(bin_bytes) = registry.export_workspace_aidws(&ws.id) {
                                    crate::storage_manager::trigger_binary_download(
                                        &format!("{}.aidws", ws.id),
                                        &bin_bytes,
                                        "application/octet-stream",
                                    );
                                    self.status_message = Some((
                                        format!(
                                            "Exported native binary workspace package '{}.aidws' ({} bytes)",
                                            ws.id,
                                            bin_bytes.len()
                                        ),
                                        false,
                                    ));
                                }
                            }

                            if ui
                                .button(format!("{} Clone", bi::COPY))
                                .on_hover_text("Duplicate entire workspace and all its models")
                                .clicked()
                            {
                                to_duplicate = Some(ws.id.clone());
                            }

                            if !is_active
                                && ui
                                    .button(
                                        RichText::new(format!(
                                            "{} Switch to Workspace",
                                            bi::CHECK_LG
                                        ))
                                        .color(Color32::from_rgb(120, 200, 240)),
                                    )
                                    .clicked()
                            {
                                to_switch = Some(ws.id.clone());
                            }
                        });
                    });
                });
            ui.add_space(4.0);
        }

        if let Some(target_id) = to_switch {
            registry.set_active_workspace(&target_id);
            if let Some(ws) = registry.active_workspace() {
                Self::sync_active_to_studio(ws, studio);
            }
            self.status_message = Some((format!("Switched to workspace '{target_id}'."), false));
        }
        if let Some(target_id) = to_duplicate {
            if let Some(new_id) = registry.duplicate_workspace(&target_id) {
                self.status_message = Some((format!("Duplicated workspace '{new_id}'."), false));
            }
        }
        if let Some(target_id) = to_remove {
            if registry.remove_workspace(&target_id) {
                if let Some(ws) = registry.active_workspace() {
                    Self::sync_active_to_studio(ws, studio);
                }
                self.status_message = Some((format!("Removed workspace '{target_id}'."), false));
            }
        }
    }

    fn render_notes_tab(&mut self, ui: &mut Ui, registry: &mut WorkspaceRegistry) {
        let Some(active_ws) = registry.active_workspace_mut() else {
            ui.label("No active workspace found.");
            return;
        };

        ui.heading(format!(
            "{} Research Log & Pipeline Architecture",
            bi::JOURNAL_CODE
        ));
        ui.label(RichText::new("Document hypotheses, training curricula targets, ablation results, and model pipeline relationships:").weak());

        ui.add_space(4.0);
        ui.add(
            egui::TextEdit::multiline(&mut active_ws.notes)
                .desired_rows(12)
                .desired_width(f32::INFINITY)
                .hint_text("# Experiment Notes\n- Hypothesis 1: Muon polar momentum accelerates grokking in sparse transformer\n- Pipeline: Encoder (model_1) -> DiT Backbone (model_2) -> Autoencoder (model_3)")
        );

        ui.add_space(8.0);
        ui.label(RichText::new("Multi-Model Pipeline Flow in this Workspace:").strong());
        ui.horizontal_wrapped(|ui| {
            for (idx, design) in active_ws.designs.iter().enumerate() {
                let is_active = design.id == active_ws.active_design_id;
                egui::Frame::new()
                    .fill(if is_active {
                        Color32::from_rgb(32, 48, 40)
                    } else {
                        Color32::from_rgb(26, 28, 34)
                    })
                    .corner_radius(4.0)
                    .stroke(Stroke::new(1.0, Color32::from_rgb(70, 75, 85)))
                    .inner_margin(6.0)
                    .show(ui, |ui| {
                        ui.label(RichText::new(format!("[{}] {}", idx + 1, design.name)).strong());
                        ui.label(
                            RichText::new(format!("{} nodes", design.spec.graph.nodes.len()))
                                .small()
                                .weak(),
                        );
                    });
                if idx < active_ws.designs.len() - 1 {
                    ui.label(
                        RichText::new("──►")
                            .color(Color32::from_rgb(120, 200, 240))
                            .strong(),
                    );
                }
            }
        });
    }

    fn render_parity_tab(&mut self, ui: &mut Ui) {
        let is_wasm = cfg!(target_arch = "wasm32");

        ui.heading("Web vs. Desktop Capabilities & Parity Audit");

        if is_wasm {
            egui::Frame::new()
                .fill(Color32::from_rgb(40, 36, 24))
                .stroke(Stroke::new(1.0, Color32::from_rgb(240, 180, 60)))
                .inner_margin(10.0)
                .corner_radius(6.0)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(format!("{} Active Environment: WebAssembly Sandboxed Client", bi::COMPASS)).color(Color32::from_rgb(240, 180, 60)).strong());
                    });
                    ui.label("Running client-side in browser sandbox. In accordance with Web platform security standards, direct native shell commands (python, nvidia-smi) are restricted. Full backup methods are active below.");
                });
        } else {
            egui::Frame::new()
                .fill(Color32::from_rgb(24, 40, 32))
                .stroke(Stroke::new(1.0, Color32::from_rgb(120, 220, 140)))
                .inner_margin(10.0)
                .corner_radius(6.0)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(format!("{} Active Environment: Native Desktop Station (Full Feature Tier)", bi::CPU)).color(Color32::from_rgb(120, 220, 140)).strong());
                    });
                    ui.label("Full access to local hardware GPU drivers (CUDA/Metal/Vulkan), direct process mediation, and low-latency disk storage.");
                });
        }

        ui.add_space(8.0);
        ui.label(RichText::new("Feature Parity Matrix:").strong());

        egui::Grid::new("parity_grid")
            .striped(true)
            .min_col_width(120.0)
            .spacing([12.0, 6.0])
            .show(ui, |ui| {
                ui.label(RichText::new("Feature Domain").strong());
                ui.label(RichText::new("Desktop Tier").strong());
                ui.label(RichText::new("WebAssembly Tier").strong());
                ui.label(RichText::new("Web Backup / Parity Strategy").strong());
                ui.end_row();

                ui.label("Visual Graph & Outliner Editor");
                ui.label(RichText::new("✓ 100% Native").color(Color32::from_rgb(120, 220, 140)));
                ui.label(
                    RichText::new("✓ 100% WebGPU / Canvas").color(Color32::from_rgb(120, 220, 140)),
                );
                ui.label("Identical full-featured visual canvas");
                ui.end_row();

                ui.label("55+ Rule Advisor Diagnostics");
                ui.label(RichText::new("✓ Instant").color(Color32::from_rgb(120, 220, 140)));
                ui.label(RichText::new("✓ Instant").color(Color32::from_rgb(120, 220, 140)));
                ui.label("Pure Rust in-memory AST analyzer");
                ui.end_row();

                ui.label("Roofline & Pareto Frontiers");
                ui.label(RichText::new("✓ Instant").color(Color32::from_rgb(120, 220, 140)));
                ui.label(RichText::new("✓ Instant").color(Color32::from_rgb(120, 220, 140)));
                ui.label("Analytical hardware ceiling modeling");
                ui.end_row();

                ui.label("Training Curriculum Planner");
                ui.label(RichText::new("✓ 100%").color(Color32::from_rgb(120, 220, 140)));
                ui.label(RichText::new("✓ 100%").color(Color32::from_rgb(120, 220, 140)));
                ui.label("Interactive stage & LR curve designer");
                ui.end_row();

                ui.label("Model Surgery (Abliterate/Soups/Graft)");
                ui.label(RichText::new("✓ 100%").color(Color32::from_rgb(120, 220, 140)));
                ui.label(RichText::new("✓ 100%").color(Color32::from_rgb(120, 220, 140)));
                ui.label("SLERP/TIES/DARE mathematical merge engine");
                ui.end_row();

                ui.label("Code Synthesis (PyTorch/JAX/Candle/WGSL)");
                ui.label(
                    RichText::new("✓ Full AST CodeGen").color(Color32::from_rgb(120, 220, 140)),
                );
                ui.label(
                    RichText::new("✓ Full AST CodeGen").color(Color32::from_rgb(120, 220, 140)),
                );
                ui.label("Immediate in-browser code download");
                ui.end_row();

                ui.label("Multi-Model Workspaces");
                ui.label(
                    RichText::new("✓ Native Disk CAS").color(Color32::from_rgb(120, 220, 140)),
                );
                ui.label(
                    RichText::new("✓ IndexedDB + LocalStorage")
                        .color(Color32::from_rgb(120, 220, 140)),
                );
                ui.label("1-Click .aidws.json bundle export/import");
                ui.end_row();

                ui.label("Live Training & Parity Runner");
                ui.label(
                    RichText::new("✓ Local Subprocess / GPU")
                        .color(Color32::from_rgb(120, 220, 140)),
                );
                ui.label(
                    RichText::new("⚡ In-Browser Simulator").color(Color32::from_rgb(240, 180, 60)),
                );
                ui.label("Web Simulator + Turnkey Script Zip Download");
                ui.end_row();
            });

        ui.add_space(8.0);
        ui.label(RichText::new("Turnkey Web-to-Compute Bridge Options:").strong());
        ui.horizontal_wrapped(|ui| {
            if ui.button(format!("{} Copy Colab 1-Click Bootstrap", bi::COPY)).clicked() {
                let colab_script = "# AI Designer Colab Bootstrap\n!pip install torch triton safetensors\n!git clone https://github.com/.../AI-Examples.git\n# Run training with mediated telemetry\n!aid train run model.yaml --grokfast\n";
                ui.ctx().copy_text(colab_script.to_string());
                self.status_message = Some(("Copied Colab bootstrap snippet to clipboard!".to_string(), false));
            }

            if ui.button(format!("{} Download Turnkey Training Zip", bi::DOWNLOAD)).clicked() {
                self.status_message = Some(("Generating turnkey training archive for download...".to_string(), false));
            }
        });
    }

    /// Synchronizes the workspace's active model design into the StudioState.
    pub fn sync_active_to_studio(ws: &Workspace, studio: &mut StudioState) {
        if let Some(design) = ws.active_design() {
            studio.spec = design.spec.clone();
            studio.refresh_analyses();
            studio.auto_layout_nodes();

            // Restore surgery state if saved in design
            if let Some(plan) = &design.abliteration_plan {
                studio.abliteration_plan = plan.clone();
            }
            if let Some(recipe) = &design.soup_recipe {
                studio.soup_recipe = recipe.clone();
            }
            if let Some(graft) = &design.grafting_plan {
                studio.grafting_plan = graft.clone();
            }
        }
    }

    /// Synchronizes the current studio design back into the workspace's active design.
    pub fn sync_studio_to_workspace(ws: &mut Workspace, studio: &StudioState) {
        if let Some(design) = ws.active_design_mut() {
            design.spec = studio.spec.clone();
            design.abliteration_plan = Some(studio.abliteration_plan.clone());
            design.soup_recipe = Some(studio.soup_recipe.clone());
            design.grafting_plan = Some(studio.grafting_plan.clone());
            design.updated_at = shared::workspace::current_unix_timestamp();
        }
    }
}
