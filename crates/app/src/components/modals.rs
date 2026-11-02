//! Modal dialogs, warning banners, and data transfer views.

use crate::icons::bi;
use crate::storage_manager::*;
use crate::studio_state::StudioState;
use crate::{AiDesignerApp, AppPage, BlueprintExportFormat};
use eframe::egui;

pub fn render_warning_banners(app: &mut AiDesignerApp, ctx: &egui::Context) {
    let is_ephemeral = app.storage_diag.is_persisted == Some(false);
    let is_quota = app.storage_diag.quota_exceeded;

    let show_combined = is_ephemeral && is_quota && !app.dismissed_combined_warning;
    let show_ephemeral = is_ephemeral && !app.dismissed_ephemeral_warning;
    let show_quota = is_quota && !app.dismissed_quota_warning;

    if show_combined || show_ephemeral || show_quota {
        let (msg, fill_color, stroke_color, text_color) = if show_combined {
            (
                "Storage Alert: Ephemeral Mode & Quota Limit Exceeded. Save your blueprint backup!",
                egui::Color32::from_rgba_premultiplied(35, 20, 20, 245),
                egui::Color32::from_rgb(240, 80, 80),
                egui::Color32::from_rgb(255, 120, 120),
            )
        } else if show_ephemeral {
            (
                "Ephemeral Storage: Browser may clear local state if device storage is constrained.",
                egui::Color32::from_rgba_premultiplied(35, 28, 15, 245),
                egui::Color32::from_rgb(220, 160, 30),
                egui::Color32::from_rgb(255, 200, 80),
            )
        } else {
            (
                "Storage Quota Exceeded: Model state migrated to IndexedDB fallback tier.",
                egui::Color32::from_rgba_premultiplied(35, 28, 15, 245),
                egui::Color32::from_rgb(220, 160, 30),
                egui::Color32::from_rgb(255, 200, 80),
            )
        };

        egui::Area::new(egui::Id::new("storage_warning_banner_area"))
            .order(egui::Order::Foreground)
            .anchor(egui::Align2::CENTER_BOTTOM, egui::vec2(0.0, -20.0))
            .show(ctx, |ui| {
                egui::Frame::NONE
                    .fill(fill_color)
                    .stroke(egui::Stroke::new(1.0_f32, stroke_color))
                    .corner_radius(8)
                    .inner_margin(egui::Margin::symmetric(16, 10))
                    .show(ui, |ui| {
                        ui.with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
                            ui.label(egui::RichText::new(msg).strong().color(text_color));
                            ui.add_space(6.0);
                            ui.with_layout(
                                egui::Layout::left_to_right(egui::Align::Center)
                                    .with_main_align(egui::Align::Center),
                                |ui| {
                                    if ui
                                        .button(format!(
                                            "{} Save .aid Backup",
                                            bi::FILE_EARMARK_BINARY
                                        ))
                                        .clicked()
                                    {
                                        if let Ok(bytes) = shared::format::aid_envelope::pack_aid(
                                            &app.studio.spec,
                                            shared::format::AidFormatTag::Bson,
                                        ) {
                                            let fname = format!(
                                                "{}.aid",
                                                app.studio.spec.metadata.model_id
                                            );
                                            trigger_binary_download(
                                                &fname,
                                                &bytes,
                                                "application/octet-stream",
                                            );
                                        }
                                    }
                                    if ui.button("Request Persistence").clicked() {
                                        request_persistent_storage();
                                    }
                                    if ui.button("Dismiss").clicked() {
                                        if show_combined {
                                            app.dismissed_combined_warning = true;
                                        } else if show_ephemeral {
                                            app.dismissed_ephemeral_warning = true;
                                        } else {
                                            app.dismissed_quota_warning = true;
                                        }
                                    }
                                },
                            );
                        });
                    });
            });
    }
}

pub fn render_dialogs(app: &mut AiDesignerApp, ui: &mut egui::Ui) {
    if app.show_help_dialog {
        let mut open = true;
        let win_w = (ui.available_width() - 24.0).clamp(340.0, 620.0);
        let win_h = (ui.available_height() - 32.0).clamp(440.0, 680.0);

        egui::Window::new(format!(
            "{} AI Designer Studio — Workflow & Shortcuts",
            bi::QUESTION_CIRCLE
        ))
        .open(&mut open)
        .resizable(true)
        .collapsible(true)
        .default_size(egui::vec2(win_w, win_h))
        .min_width(320.0)
        .min_height(380.0)
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
        .show(ui.ctx(), |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.heading(format!(
                        "{} AI Designer Studio & Training Suite",
                        bi::CPU_FILL
                    ));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            egui::RichText::new(format!("v{}", env!("CARGO_PKG_VERSION")))
                                .strong()
                                .color(ui.visuals().hyperlink_color),
                        );
                    });
                });
                ui.add_space(4.0);
                ui.separator();
                ui.add_space(6.0);

                ui.heading(format!("{} 5-Stage Architecture Design Workflow", bi::COMPASS));
                ui.add_space(4.0);
                ui.label("1. Scaffold: Click '1. New / Wizard' to pick from 9 Guided Archetypes, 20 Cross-Domain Starter Blueprints, or a Blank Canvas.");
                ui.label("2. Compose: Filter and add 35+ operators from the Left Palette, inspect and configure layers in the Right Inspector, and view dataflow on the Canvas.");
                ui.label("3. Diagnose & Optimize: Review 55+ mathematical/hardware checks with 1-click Quick Fixes in the Advisor dock, inspect VRAM in Roofline, and compare 5 scaling variants in Pareto Frontier.");
                ui.label("4. Verify & Train: Click 'Test Parity & Condition' for Seed-42 activation verification and 'Run Verification & Training' for live convergence monitoring.");
                ui.label("5. Export & Codegen: Click 'Generate Target Code' to emit PyTorch 2, JAX/Equinox, Rust Candle, WebGPU WGSL, OpenAI Triton, and Parity Harness files, or export native .aid, .yaml, or .aidpkg.");

                ui.add_space(8.0);
                ui.separator();
                ui.add_space(6.0);

                ui.heading(format!("{} Canvas Controls & Keyboard Shortcuts", bi::KEYBOARD));
                ui.add_space(4.0);
                ui.label("• Pan Canvas: Middle-Mouse Drag or Alt + Left-Drag.");
                ui.label("• Zoom Canvas: Mouse Scroll Wheel or HUD - / 100% / + buttons.");
                ui.label("• Auto-Layout: Press Home or click 'Auto-Layout' in the Canvas HUD.");
                ui.label("• Layer Actions: Select a node to Move Up, Move Down, Duplicate, or Delete (Del / Backspace).");
                ui.label("• Close Dialogs / Deselect: Press Escape.");

                ui.add_space(8.0);
                ui.separator();
                ui.add_space(6.0);

                ui.heading(format!("{} Multi-Tier Storage & Offline Privacy", bi::HDD_STACK));
                ui.add_space(4.0);
                ui.label("• Tier 1: Fast synchronous local storage.");
                ui.label("• Tier 2: Asynchronous IndexedDB extended quota fallback.");
                ui.label(
                    egui::RichText::new("100% Client-Side: Runs natively on Desktop or offline in Browser WebAssembly.")
                        .color(egui::Color32::from_rgb(80, 160, 90))
                        .strong(),
                );
            });
        });
        if !open {
            app.show_help_dialog = false;
        }
    }

    if app.show_reset_dialog {
        egui::Window::new("Reset Studio Workbench?")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .show(ui.ctx(), |ui| {
                ui.label("Choose how to reset your active neural architecture workbench:");
                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    if ui
                        .button(format!(
                            "{} Load Default Starter",
                            bi::ARROW_COUNTERCLOCKWISE
                        ))
                        .clicked()
                    {
                        let starter_yaml = shared::templates::TEMPLATES[0].yaml_content;
                        if let Ok(spec) = shared::format::aid_yaml::from_yaml_str(starter_yaml) {
                            app.studio = StudioState::new_with_starter(spec);
                            app.state.active_page = AppPage::Designer;
                            app.persist_state();
                        }
                        app.show_reset_dialog = false;
                    }
                    if ui
                        .button(format!("{} Blank Canvas", bi::FILE_EARMARK_TEXT))
                        .clicked()
                    {
                        let blank_spec = shared::scaffold_archetype(
                            shared::ArchitectureArchetype::Blank,
                            "blank_custom_model",
                            "Custom Neural Network",
                            256,
                            0,
                            shared::PrecisionFormat::BF16,
                        );
                        app.studio = StudioState::new_with_starter(blank_spec);
                        app.state.active_page = AppPage::Designer;
                        app.persist_state();
                        app.show_reset_dialog = false;
                    }
                    if ui.button("Cancel").clicked() {
                        app.show_reset_dialog = false;
                    }
                });
            });
    }

    if let Some(format) = app.show_export_dialog {
        let mut open = true;
        egui::Window::new(format!("Export Architecture ({})", format.label()))
            .open(&mut open)
            .default_size(egui::vec2(580.0, 400.0))
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .show(ui.ctx(), |ui| {
                ui.horizontal(|ui| {
                    if format.is_text() {
                        if ui
                            .button(format!("{} Copy to Clipboard", bi::COPY))
                            .clicked()
                        {
                            ui.ctx().copy_text(app.export_text_buffer.clone());
                            app.export_copied_notification = Some(ui.input(|i| i.time));
                        }
                        if let Some(t) = app.export_copied_notification
                            && ui.input(|i| i.time) - t < 3.0
                        {
                            ui.label(
                                egui::RichText::new("Copied to clipboard!")
                                    .color(egui::Color32::GREEN),
                            );
                        }
                        ui.separator();
                    }

                    if ui
                        .button(format!("{} Download File", bi::DOWNLOAD))
                        .clicked()
                    {
                        let model_id = &app.studio.spec.metadata.model_id;
                        match format {
                            BlueprintExportFormat::Yaml => {
                                let fname = format!("{model_id}.yaml");
                                trigger_text_download(
                                    &fname,
                                    &app.export_text_buffer,
                                    "text/yaml;charset=utf-8",
                                );
                            }
                            BlueprintExportFormat::Json => {
                                let fname = format!("{model_id}.json");
                                trigger_text_download(
                                    &fname,
                                    &app.export_text_buffer,
                                    "application/json;charset=utf-8",
                                );
                            }
                            BlueprintExportFormat::Toml => {
                                let fname = format!("{model_id}.toml");
                                trigger_text_download(
                                    &fname,
                                    &app.export_text_buffer,
                                    "text/plain;charset=utf-8",
                                );
                            }
                            BlueprintExportFormat::Ron => {
                                let fname = format!("{model_id}.ron");
                                trigger_text_download(
                                    &fname,
                                    &app.export_text_buffer,
                                    "text/plain;charset=utf-8",
                                );
                            }
                            BlueprintExportFormat::NativeAid => {
                                if let Ok(bytes) = shared::format::aid_envelope::pack_aid(
                                    &app.studio.spec,
                                    shared::format::AidFormatTag::Bson,
                                ) {
                                    let fname = format!("{model_id}.aid");
                                    trigger_binary_download(
                                        &fname,
                                        &bytes,
                                        "application/octet-stream",
                                    );
                                }
                            }
                            BlueprintExportFormat::AidPkg => {
                                if let Ok(bytes) = shared::format::aidpkg::create_aidpkg(
                                    &app.studio.spec,
                                    None,
                                    None,
                                ) {
                                    let fname = format!("{model_id}.aidpkg");
                                    trigger_binary_download(
                                        &fname,
                                        &bytes,
                                        "application/octet-stream",
                                    );
                                }
                            }
                        }
                    }
                });

                ui.add_space(8.0);
                egui::ScrollArea::both().show(ui, |ui| {
                    ui.add(
                        egui::TextEdit::multiline(&mut app.export_text_buffer)
                            .font(egui::TextStyle::Monospace)
                            .code_editor()
                            .lock_focus(true)
                            .desired_width(f32::INFINITY),
                    );
                });
            });

        if !open {
            app.show_export_dialog = None;
            app.export_text_buffer.clear();
        }
    }

    if app.show_import_dialog {
        let mut open = true;
        egui::Window::new(format!("{} Import Architecture Blueprint", bi::UPLOAD))
            .open(&mut open)
            .default_size(egui::vec2(580.0, 400.0))
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .show(ui.ctx(), |ui| {
                ui.label(
                    "Paste an AI Blueprint Specification in YAML (.yaml), JSON (.json), TOML (.toml), or RON (.ron):",
                );
                ui.add_space(8.0);

                egui::ScrollArea::both().max_height(220.0).show(ui, |ui| {
                    ui.add(
                        egui::TextEdit::multiline(&mut app.import_text_buffer)
                            .font(egui::TextStyle::Monospace)
                            .hint_text("Paste .yaml, .json, .toml, or .ron blueprint specification here...")
                            .desired_width(f32::INFINITY)
                            .desired_rows(10),
                    );
                });

                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    if ui
                        .button(
                            egui::RichText::new(format!("{} Load into Studio Workbench", bi::CHECK_LG))
                                .strong()
                                .color(egui::Color32::from_rgb(120, 220, 140)),
                        )
                        .clicked()
                    {
                        let input = app.import_text_buffer.trim();
                        if input.is_empty() {
                            app.import_result_message = Some(Err("Input buffer is empty.".to_string()));
                        } else {
                            match shared::format::load_blueprint_from_bytes(input.as_bytes(), None) {
                                Ok(spec) => {
                                    let name = spec.metadata.display_name.clone();
                                    app.studio = StudioState::new_with_starter(spec);
                                    app.state.active_page = AppPage::Designer;
                                    app.import_result_message = Some(Ok(format!(
                                        "Successfully imported architecture '{name}' with {} layers!",
                                        app.studio.spec.graph.nodes.len()
                                    )));
                                    app.persist_state();
                                }
                                Err(err) => {
                                    app.import_result_message = Some(Err(format!("Import parsing failed: {err}")));
                                }
                            }
                        }
                    }

                    if ui.button("Cancel").clicked() {
                        app.show_import_dialog = false;
                        app.import_text_buffer.clear();
                        app.import_result_message = None;
                    }
                });

                if let Some(ref result) = app.import_result_message {
                    ui.add_space(6.0);
                    match result {
                        Ok(msg) => {
                            ui.label(
                                egui::RichText::new(msg)
                                    .color(egui::Color32::from_rgb(80, 180, 90))
                                    .strong(),
                            );
                        }
                        Err(msg) => {
                            ui.label(
                                egui::RichText::new(msg)
                                    .color(egui::Color32::from_rgb(220, 70, 70))
                                    .strong(),
                            );
                        }
                    }
                }
            });

        if !open {
            app.show_import_dialog = false;
            app.import_text_buffer.clear();
            app.import_result_message = None;
        }
    }

    if app.show_storage_modal {
        render_storage_modal(app, ui);
    }
}

pub fn render_storage_modal(app: &mut AiDesignerApp, ui: &mut egui::Ui) {
    let mut open = true;
    egui::Window::new("Storage & Persistence Management")
        .open(&mut open)
        .default_size(egui::vec2(500.0, 360.0))
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
        .show(ui.ctx(), |ui| {
            ui.heading("Client-Side Storage Diagnostics");
            ui.add_space(4.0);

            let status_color = match app.storage_diag.is_persisted {
                Some(true) => egui::Color32::from_rgb(80, 200, 100),
                Some(false) => egui::Color32::from_rgb(240, 160, 50),
                None => egui::Color32::GRAY,
            };
            let status_label = match app.storage_diag.is_persisted {
                Some(true) => "Persistent (Guaranteed against browser auto-eviction)",
                Some(false) => "Ephemeral (May be purged under device storage pressure)",
                None => "Native Host Disk",
            };

            ui.horizontal(|ui| {
                ui.label("Persistence Mode:");
                ui.colored_label(status_color, egui::RichText::new(status_label).strong());
            });

            ui.horizontal(|ui| {
                ui.label("Active Storage Tier:");
                ui.label(egui::RichText::new(app.storage_diag.backend.label()).strong());
            });

            ui.horizontal(|ui| {
                ui.label("Active Model Footprint:");
                ui.label(format!(
                    "{} layers, {} params ({:.2} MiB VRAM estimate)",
                    app.studio.spec.graph.nodes.len(),
                    app.studio.partition_report.total_param_count,
                    app.studio.roofline_report.vram.total_allocated_bytes as f64
                        / (1024.0 * 1024.0)
                ));
            });

            ui.horizontal(|ui| {
                ui.label("PWA Installation:");
                if app.storage_diag.is_pwa_installed {
                    ui.colored_label(
                        egui::Color32::from_rgb(80, 200, 100),
                        "Installed (Permanent Desktop/Mobile App)",
                    );
                } else if app.storage_diag.pwa_install_available {
                    ui.label("Available to Install");
                } else {
                    ui.label("Running in Browser Tab / Native Window");
                }
            });

            ui.add_space(8.0);
            ui.separator();
            ui.add_space(6.0);

            ui.heading("Storage Actions");
            ui.add_space(4.0);

            ui.horizontal(|ui| {
                if app.storage_diag.is_persisted != Some(true) {
                    if ui
                        .button("Request Persistent Storage")
                        .on_hover_text(
                            "Ask browser permission to protect session from auto-eviction",
                        )
                        .clicked()
                    {
                        request_persistent_storage();
                    }
                }

                if app.storage_diag.pwa_install_available && !app.storage_diag.is_pwa_installed {
                    if ui
                        .button("Install Web App")
                        .on_hover_text(
                            "Install to homescreen or desktop for permanent offline durability",
                        )
                        .clicked()
                    {
                        trigger_pwa_install();
                    }
                }
            });

            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui
                    .button(format!(
                        "{} Download .aid Blueprint Backup",
                        bi::FILE_EARMARK_BINARY
                    ))
                    .clicked()
                {
                    if let Ok(bytes) = shared::format::aid_envelope::pack_aid(
                        &app.studio.spec,
                        shared::format::AidFormatTag::Bson,
                    ) {
                        let fname = format!("{}.aid", app.studio.spec.metadata.model_id);
                        trigger_binary_download(&fname, &bytes, "application/octet-stream");
                    }
                }

                if ui
                    .button(format!("{} Import Blueprint", bi::UPLOAD))
                    .clicked()
                {
                    app.show_storage_modal = false;
                    app.show_import_dialog = true;
                }
            });
        });

    if !open {
        app.show_storage_modal = false;
    }
}

pub fn render_shortcuts_modal(ctx: &egui::Context, is_open: &mut bool) {
    if !*is_open {
        return;
    }
    let mut open = true;
    let screen = ctx.viewport_rect();
    let win_w = (screen.width() - 24.0).clamp(360.0, 640.0);
    let win_h = (screen.height() - 40.0).clamp(380.0, 540.0);

    egui::Window::new(format!("{} Keyboard Shortcuts & Canvas Gestures", bi::KEYBOARD))
        .open(&mut open)
        .resizable(true)
        .collapsible(false)
        .default_size(egui::vec2(win_w, win_h))
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
        .show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading(
                    egui::RichText::new("Interactive Studio Hotkeys")
                        .size(15.0)
                        .strong()
                        .color(egui::Color32::from_rgb(110, 200, 255)),
                );
                ui.label(
                    egui::RichText::new("Professional keyboard-first shortcuts for seamless model design, training, and surgery:")
                        .size(11.5)
                        .color(egui::Color32::from_rgb(170, 185, 210)),
                );
                ui.add_space(6.0);
                ui.separator();
                ui.add_space(6.0);

                let render_group = |ui: &mut egui::Ui, title: &str, rows: &[(&str, &str)]| {
                    ui.label(
                        egui::RichText::new(title)
                            .size(12.5)
                            .strong()
                            .color(egui::Color32::from_rgb(255, 205, 110)),
                    );
                    ui.add_space(2.0);
                    egui::Grid::new(title)
                        .num_columns(2)
                        .spacing([24.0, 4.0])
                        .show(ui, |ui| {
                            for &(key, desc) in rows {
                                ui.label(
                                    egui::RichText::new(key)
                                        .monospace()
                                        .strong()
                                        .color(egui::Color32::from_rgb(130, 225, 160)),
                                );
                                ui.label(
                                    egui::RichText::new(desc)
                                        .size(11.5)
                                        .color(egui::Color32::from_rgb(220, 230, 245)),
                                );
                                ui.end_row();
                            }
                        });
                    ui.add_space(6.0);
                    ui.separator();
                    ui.add_space(4.0);
                };

                render_group(
                    ui,
                    "Global & Command Palette",
                    &[
                        ("Ctrl+K / ⌘K", "Open Global Command Palette (search layers, ops, blueprints, presets)"),
                        ("F1 / ?", "Open this Keyboard Shortcuts cheat sheet"),
                        ("Escape", "Close open modal or deselect current layer"),
                    ],
                );

                render_group(
                    ui,
                    "Discipline Focus Modes",
                    &[
                        ("Alt+M", "Focus: Model Architecture (Graph, Outliner Tree, YAML Blueprint)"),
                        ("Alt+T", "Focus: Training & Optimization (Curriculum, 4D Parallelism, Runner)"),
                        ("Alt+S", "Focus: Model Surgery & Soups Lab (Abliteration, SLERP/TIES, Grafting)"),
                    ],
                );

                render_group(
                    ui,
                    "Layout & Workspace Visibility",
                    &[
                        ("F11", "Toggle Zen Mode (maximize canvas by hiding all sidebars and docks)"),
                        ("Alt+1", "Toggle Left Palette (Layers, Topic Catalog, Starters)"),
                        ("Alt+2", "Toggle Bottom Dock (Diagnostics Advisor, Roofline, Pareto, Runner)"),
                        ("Alt+3", "Toggle Right Inspector (Node properties, Hyperparameters, Precision)"),
                    ],
                );

                render_group(
                    ui,
                    "Graph Canvas & Layer Editing",
                    &[
                        ("Ctrl+Z", "Undo last blueprint modification (50-state history)"),
                        ("Ctrl+Y / Ctrl+Shift+Z", "Redo undone blueprint modification"),
                        ("Home", "Auto-layout nodes in clean pipeline & center camera"),
                        ("Ctrl+D", "Duplicate currently selected layer"),
                        ("Delete / Backspace", "Delete currently selected layer"),
                        ("Scroll", "Zoom canvas in / out (25% - 250%)"),
                        ("Middle Drag / Alt+Drag", "Pan canvas viewport"),
                    ],
                );
            });
        });
    if !open {
        *is_open = false;
    }
}
