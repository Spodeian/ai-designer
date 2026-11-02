//! Top navigation bar and global header controls using Bootstrap Icons v1.13.2.

use crate::icons::bi;
use crate::storage_manager::*;
use crate::studio_state::StudioFocus;
use crate::{AiDesignerApp, AppPage, ScreenConstraints};
use eframe::egui::{self, Color32, RichText};

pub fn render_navbar(app: &mut AiDesignerApp, ui: &mut egui::Ui, constraints: &ScreenConstraints) {
    egui::Panel::top("top_panel").show(ui, |ui| {
        ui.add_space(3.0);
        let header_row_height = if constraints.is_mobile { 40.0 } else { 32.0 };

        ui.horizontal(|ui| {
            ui.set_height(header_row_height);

            // 1. Logo & App Title
            ui.label(
                RichText::new(format!("{} AI Designer", bi::CPU_FILL))
                    .size(15.0)
                    .color(Color32::from_rgb(110, 200, 255))
                    .strong(),
            );

            ui.add_space(4.0);
            ui.separator();

            // 2. Main Page Switcher: Studio | Gallery | Docs
            let is_designer = app.state.active_page == AppPage::Designer;
            if ui
                .selectable_label(
                    is_designer,
                    format!("{} Studio", bi::DIAGRAM_3_FILL),
                )
                .on_hover_text("Interactive Multi-Discipline Studio Workbench")
                .clicked()
            {
                app.state.active_page = AppPage::Designer;
                app.persist_state();
            }

            if ui
                .selectable_label(
                    app.state.active_page == AppPage::Gallery,
                    format!("{} Gallery", bi::COLLECTION),
                )
                .on_hover_text("Browse 32 Starter & Frontier Neural Blueprints")
                .clicked()
            {
                app.state.active_page = AppPage::Gallery;
                app.persist_state();
            }

            if ui
                .selectable_label(
                    app.state.active_page == AppPage::About,
                    format!("{} Docs", bi::INFO_CIRCLE),
                )
                .on_hover_text("Architectural Guides, Reference & System Design")
                .clicked()
            {
                app.state.active_page = AppPage::About;
                app.persist_state();
            }

            // 3. Studio Workspace & Model Identity (when on Studio page)
            if is_designer {
                ui.separator();

                let (ws_name, model_count) = app
                    .state
                    .workspace_registry
                    .active_workspace()
                    .map(|w| (w.name.clone(), w.designs.len()))
                    .unwrap_or_else(|| ("Default Workspace".to_string(), 1));

                if ui
                    .button(
                        RichText::new(format!("{} {ws_name} ({model_count})", bi::FOLDER2_OPEN))
                            .color(Color32::from_rgb(120, 200, 240))
                            .strong(),
                    )
                    .on_hover_text("Manage Workspaces & Multi-Model Projects")
                    .clicked()
                {
                    app.workspace_manager_modal.is_open = true;
                }

                ui.label(
                    RichText::new(&app.studio.spec.metadata.display_name)
                        .size(12.5)
                        .color(Color32::from_rgb(225, 235, 250))
                        .strong(),
                );

                ui.separator();

                // 4. Primary Focus Switcher (Model Architecture | Training & Optimization | Surgery Lab)
                let is_model = app.studio.focus_mode == StudioFocus::Model;
                if ui
                    .selectable_label(
                        is_model,
                        RichText::new(format!("{} Model", bi::DIAGRAM_3_FILL))
                            .color(if is_model {
                                Color32::from_rgb(120, 210, 255)
                            } else {
                                Color32::from_rgb(180, 195, 215)
                            })
                            .strong(),
                    )
                    .on_hover_text("Focus: Model Architecture (Layers, Dataflow Graph, Outliner, YAML Blueprint)")
                    .clicked()
                {
                    app.studio.set_focus(StudioFocus::Model);
                }

                let is_training = app.studio.focus_mode == StudioFocus::Training;
                if ui
                    .selectable_label(
                        is_training,
                        RichText::new(format!("{} Training", bi::SPEEDOMETER2))
                            .color(if is_training {
                                Color32::from_rgb(130, 235, 165)
                            } else {
                                Color32::from_rgb(180, 195, 215)
                            })
                            .strong(),
                    )
                    .on_hover_text("Focus: Training & Optimization (Curriculum stages, 4D Parallelism, Optimizers, Live Runner)")
                    .clicked()
                {
                    app.studio.set_focus(StudioFocus::Training);
                }

                let is_surgery = app.studio.focus_mode == StudioFocus::Surgery;
                if ui
                    .selectable_label(
                        is_surgery,
                        RichText::new(format!("{} Surgery", bi::SCISSORS))
                            .color(if is_surgery {
                                Color32::from_rgb(255, 180, 120)
                            } else {
                                Color32::from_rgb(180, 195, 215)
                            })
                            .strong(),
                    )
                    .on_hover_text("Focus: Surgery & Soups Lab (Refusal abliteration, SLERP/TIES/DARE model soups, layer grafting)")
                    .clicked()
                {
                    app.studio.set_focus(StudioFocus::Surgery);
                }
            }

            // 5. Right-Aligned Controls: Settings, Quick Search, Layout Toggles, and Metrics
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                // Unified Settings & Utilities Menu (gear dropdown)
                egui::ComboBox::from_id_salt("global_settings_menu")
                    .selected_text(format!("{} ▾", bi::GEAR))
                    .show_ui(ui, |ui| {
                        let theme_icon = if app.state.theme.is_high_contrast() {
                            bi::CIRCLE_HALF
                        } else {
                            bi::PALETTE2
                        };
                        if ui
                            .button(format!("{theme_icon} Theme: {}", app.state.theme.label()))
                            .on_hover_text("Cycle visual themes")
                            .clicked()
                        {
                            app.state.theme = app.state.theme.next();
                            app.persist_state();
                        }

                        let storage_label = match app.storage_diag.is_persisted {
                            Some(true) => format!("{} Storage: Persistent", bi::HDD_STACK),
                            Some(false) => format!("{} Storage: Ephemeral", bi::HDD_STACK),
                            None => format!("{} Storage", bi::HDD_STACK),
                        };
                        if ui
                            .button(storage_label)
                            .on_hover_text("Storage persistence diagnostics and backups")
                            .clicked()
                        {
                            app.show_storage_modal = true;
                        }

                        if ui
                            .button(format!("{} Telemetry", bi::SPEEDOMETER2))
                            .on_hover_text("View live studio subsystem performance")
                            .clicked()
                        {
                            app.show_telemetry_modal = true;
                        }

                        if ui
                            .button(format!("{} Import Blueprint", bi::UPLOAD))
                            .on_hover_text("Import .aid, .yaml, .json, or .toml specification")
                            .clicked()
                        {
                            app.show_import_dialog = true;
                            app.import_text_buffer.clear();
                            app.import_result_message = None;
                        }

                        if app.storage_diag.pwa_install_available && !app.storage_diag.is_pwa_installed {
                            if ui
                                .button(format!("{} Install App", bi::DOWNLOAD))
                                .on_hover_text("Install application for permanent offline storage")
                                .clicked()
                            {
                                trigger_pwa_install();
                            }
                        }

                        ui.separator();

                        if ui
                            .button(format!("{} Keyboard Shortcuts (F1)", bi::KEYBOARD))
                            .on_hover_text("View hotkeys cheat sheet & canvas gestures")
                            .clicked()
                        {
                            app.show_shortcuts_modal = true;
                        }

                        if ui
                            .button(format!("{} 5-Stage Workflow Guide", bi::QUESTION_CIRCLE))
                            .on_hover_text("Architecture design guide and documentation")
                            .clicked()
                        {
                            app.show_help_dialog = true;
                        }
                    });

                // Quick Command Palette Trigger (Ctrl+K)
                if ui
                    .button(
                        RichText::new(format!("{} Ctrl+K", bi::SEARCH))
                            .color(Color32::from_rgb(130, 200, 255))
                            .strong(),
                    )
                    .on_hover_text("Search layers, operators, blueprints, presets & commands (Ctrl+K / ⌘K)")
                    .clicked()
                {
                    app.command_palette.toggle();
                }

                // Studio Layout Toggles & Metrics (when on Studio page)
                if is_designer {
                    ui.separator();

                    // Zen Mode Toggle
                    let zen_active = !app.studio.left_panel_visible
                        && !app.studio.right_panel_visible
                        && !app.studio.bottom_dock_visible;
                    if ui
                        .selectable_label(
                            zen_active,
                            RichText::new(format!("{} Zen", bi::ARROWS_FULLSCREEN)).strong(),
                        )
                        .on_hover_text("Zen Mode (F11): Maximize canvas space by toggling all side panels and docks")
                        .clicked()
                    {
                        app.studio.toggle_zen_mode();
                    }

                    // Inspector Toggle
                    if ui
                        .selectable_label(
                            app.studio.right_panel_visible,
                            format!("{} Inspector", bi::SLIDERS),
                        )
                        .on_hover_text("Toggle Right Inspector panel (Alt+3): edit node properties and precision")
                        .clicked()
                    {
                        app.studio.toggle_right_panel();
                    }

                    // Bottom Dock Toggle
                    if ui
                        .selectable_label(
                            app.studio.bottom_dock_visible,
                            format!("{} Dock", bi::GRID_1X2),
                        )
                        .on_hover_text("Toggle Bottom Dock (Alt+2): Diagnostics, Roofline, Pareto, and Training Runner")
                        .clicked()
                    {
                        app.studio.toggle_bottom_dock();
                    }

                    // Left Palette Toggle
                    if ui
                        .selectable_label(
                            app.studio.left_panel_visible,
                            format!("{} Palette", bi::LAYERS),
                        )
                        .on_hover_text("Toggle Left Catalog / Palette panel (Alt+1): browse layers and topics")
                        .clicked()
                    {
                        app.studio.toggle_left_panel();
                    }

                    ui.separator();

                    // Live Architecture Metrics HUD Badge
                    let param_count = app.studio.partition_report.total_param_count;
                    let param_str = if param_count >= 1_000_000_000 {
                        format!("{:.2}B", param_count as f64 / 1e9)
                    } else if param_count >= 1_000_000 {
                        format!("{:.1}M", param_count as f64 / 1e6)
                    } else if param_count >= 1_000 {
                        format!("{:.1}K", param_count as f64 / 1e3)
                    } else {
                        format!("{param_count}")
                    };

                    let vram_mb = (param_count as f64 * 2.0) / (1024.0 * 1024.0);
                    ui.label(
                        RichText::new(format!("{} {param_str} • {:.0}MB (BF16)", bi::CPU, vram_mb))
                            .size(11.0)
                            .monospace()
                            .color(Color32::from_rgb(130, 210, 255)),
                    )
                    .on_hover_text(format!(
                        "Live Architecture Metrics:\n• Total Parameters: {}\n• Muon 2D Weights: {} ({:.1}%)\n• AdamW 1D Vectors: {}\n• Weight Memory: {:.1} MB in BF16 (or {:.1} MB in INT4)\n• Target Accelerators: {} x {}",
                        param_count,
                        app.studio.partition_report.muon_param_count,
                        app.studio.partition_report.muon_ratio * 100.0,
                        app.studio.partition_report.adamw_param_count,
                        vram_mb,
                        vram_mb / 4.0,
                        app.studio.spec.training.parallelism.total_accelerators(),
                        app.studio.spec.metadata.target_hardware.profile_name
                    ));
                }
            });
        });
        ui.add_space(2.0);
    });
}
