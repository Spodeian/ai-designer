//! About, Workflow & Architecture documentation view page with Bootstrap Icons and clickable repository links.

use crate::icons::bi;
use crate::{AiDesignerApp, ScreenConstraints};
use eframe::egui::{self, Color32, RichText};

pub fn render_about_page(
    app: &mut AiDesignerApp,
    ui: &mut egui::Ui,
    constraints: &ScreenConstraints,
) {
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.add_space(8.0);

            // Hero / Title
            ui.vertical_centered(|ui| {
                ui.heading(format!(
                    "{} AI Designer Studio & Training Suite",
                    bi::CPU_FILL
                ));
                ui.label(
                    RichText::new(
                        "Universal AI Architect Workbench for the 18-Category, 263-Topic AI-Examples Curriculum",
                    )
                    .color(ui.visuals().weak_text_color()),
                );
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(
                            "Version 0.1.0 • Rust 2024 • egui 0.36 • Bootstrap Icons 1.13.2",
                        )
                        .small()
                        .monospace(),
                    );
                    ui.hyperlink_to(
                        RichText::new(format!(
                            "{} AI-Examples Repository {}",
                            bi::BOOK,
                            bi::BOX_ARROW_UP_RIGHT
                        ))
                        .small()
                        .color(Color32::from_rgb(100, 195, 255)),
                        "https://github.com/Spodeian/AI-Examples",
                    );
                });
            });

            ui.add_space(16.0);

            // Card 0: 5-Stage Studio Workflow
            ui.group(|ui| {
                ui.set_width(ui.available_width());
                ui.heading(format!(
                    "{} 5-Stage Architecture Design Workflow",
                    bi::COMPASS
                ));
                ui.add_space(4.0);
                egui::Grid::new("workflow_steps_grid")
                    .num_columns(2)
                    .spacing([16.0, 6.0])
                    .striped(true)
                    .show(ui, |ui| {
                        ui.label(RichText::new("1. Scaffold").strong());
                        ui.label("Launch '1. New / Wizard' to pick from 9 Guided Archetypes, 46 Industry & Cross-Domain Starter Blueprints, or a Blank Canvas.");
                        ui.end_row();

                        ui.label(RichText::new("2. Compose & Inspect").strong());
                        ui.label("Filter and add operators from the Left Palette, inspect/reorder/duplicate layers in the Right Inspector (with node overview & next-layer recommendations), and view Input -> Layer -> Output dataflow.");
                        ui.end_row();

                        ui.label(RichText::new("3. Diagnose & Optimize").strong());
                        ui.label("Review 55+ mathematical/hardware checks with 1-click Quick Fixes in the Advisor tab, inspect VRAM waterfalls in Roofline, and compare 5 scaling variants in Pareto Frontier.");
                        ui.end_row();

                        ui.label(RichText::new("4. Verify & Simulate").strong());
                        ui.label("Run deterministic Seed-42 transitive numerical parity checks and monitor live Muon + AdamW training convergence in the Training Runner dock.");
                        ui.end_row();

                        ui.label(RichText::new("5. Export & Codegen").strong());
                        ui.label("Click '5. Generate Target Code' to emit PyTorch 2, JAX/Equinox, Rust Candle, WebGPU WGSL, OpenAI Triton, and Parity Harness files, or export native .aid, .yaml, or .aidpkg.");
                        ui.end_row();
                    });
            });

            ui.add_space(12.0);

            // Card 1: Architecture Overview
            ui.group(|ui| {
                ui.set_width(ui.available_width());
                ui.heading(format!("{} Four-Crate Modular Architecture", bi::BOXES));
                ui.add_space(4.0);
                ui.label("Cleanly decoupled into four single-responsibility Rust crates:");
                ui.add_space(6.0);

                egui::Grid::new("arch_grid")
                    .num_columns(2)
                    .spacing([16.0, 8.0])
                    .striped(true)
                    .show(ui, |ui| {
                        ui.label(RichText::new("crates/shared").strong().monospace());
                        ui.label("AiDesignSpec AST, 27-Precision Lattice, 55+ Advisor Rules, Roofline, Pareto Search, 5-Backend Codegen, and .aid / .yaml / .aidpkg codecs.");
                        ui.end_row();

                        ui.label(RichText::new("crates/app").strong().monospace());
                        ui.label("Universal egui 0.36 Studio UI, Bootstrap Icons v1.13.2 font integration, Dataflow Canvas, Inspector, and Multi-Tier Storage.");
                        ui.end_row();

                        ui.label(RichText::new("crates/desktop").strong().monospace());
                        ui.label("Native desktop Studio runner via eframe/winit and 12-command Headless CLI compiler with mimalloc.");
                        ui.end_row();

                        ui.label(RichText::new("crates/web").strong().monospace());
                        ui.label("WebAssembly client entrypoint, Trunk bundling, Bootstrap Icons CDN, PWA service worker, and edge headers.");
                        ui.end_row();
                    });
            });

            ui.add_space(12.0);

            // Card 2: System & Storage Status
            ui.group(|ui| {
                ui.set_width(ui.available_width());
                ui.heading(format!(
                    "{} Live Diagnostics & Storage Status",
                    bi::SPEEDOMETER2
                ));
                ui.add_space(4.0);

                let backend_label = app.storage_diag.backend.label();
                let persistence_status = match app.storage_diag.is_persisted {
                    Some(true) => "Persistent (Browser storage manager guaranteed)",
                    Some(false) => "Ephemeral (May be purged under storage pressure)",
                    None => "Native / Unknown",
                };

                egui::Grid::new("diag_grid")
                    .num_columns(2)
                    .spacing([16.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Active Storage Tier:");
                        ui.label(RichText::new(backend_label).monospace());
                        ui.end_row();

                        ui.label("Persistence Mode:");
                        ui.label(RichText::new(persistence_status).monospace());
                        ui.end_row();

                        ui.label("Active Visual Theme:");
                        ui.label(format!(
                            "{} {}",
                            bi::PALETTE2,
                            app.state.theme.label()
                        ));
                        ui.end_row();

                        ui.label("Active Architecture:");
                        ui.label(format!(
                            "{} ({} layers, {} params)",
                            app.studio.spec.metadata.display_name,
                            app.studio.spec.graph.nodes.len(),
                            app.studio.partition_report.total_param_count
                        ));
                        ui.end_row();

                        ui.label("Storage Usage:");
                        ui.label(format!("{} bytes in local state", app.storage_diag.usage_bytes));
                        ui.end_row();

                        ui.label("Viewport Layout:");
                        let mode = if constraints.is_mobile {
                            "Mobile Portrait"
                        } else {
                            "Widescreen Desktop"
                        };
                        let size = ui.available_size();
                        ui.label(format!("{} (~{}x{})", mode, size.x as i32, size.y as i32));
                        ui.end_row();
                    });

                ui.add_space(8.0);
                if ui
                    .button(format!("{} Inspect Full Storage Details", bi::HDD_STACK))
                    .clicked()
                {
                    app.show_storage_modal = true;
                }
            });

            ui.add_space(12.0);

            // Card 3: Shortcuts & User Guide
            ui.group(|ui| {
                ui.set_width(ui.available_width());
                ui.heading(format!("{} Keyboard Shortcuts & File Formats", bi::KEYBOARD));
                ui.add_space(4.0);

                egui::Grid::new("shortcuts_grid")
                    .num_columns(2)
                    .spacing([16.0, 6.0])
                    .show(ui, |ui| {
                        ui.label(RichText::new("Middle-Drag / Alt+Drag").monospace().strong());
                        ui.label("Pan around the interactive Dataflow Graph canvas.");
                        ui.end_row();

                        ui.label(RichText::new("Scroll Wheel").monospace().strong());
                        ui.label("Zoom in and out on the Dataflow Graph canvas (25% – 250%).");
                        ui.end_row();

                        ui.label(RichText::new("Home").monospace().strong());
                        ui.label("Reset canvas camera and auto-layout Input, Layer, and Output cards.");
                        ui.end_row();

                        ui.label(RichText::new("Delete / Backspace").monospace().strong());
                        ui.label("Delete the currently selected computation layer on the canvas.");
                        ui.end_row();

                        ui.label(RichText::new("Escape").monospace().strong());
                        ui.label("Close any active modal dialog or clear node selection.");
                        ui.end_row();

                        ui.label(RichText::new(".aid / .yaml / .aidpkg").monospace().strong());
                        ui.label("Native binary byte format (.aid), human-readable YAML (.yaml), and page-aligned bundle (.aidpkg).");
                        ui.end_row();
                    });
            });

            ui.add_space(20.0);
        });
}
