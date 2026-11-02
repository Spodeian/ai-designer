//! AI-Examples Starter Blueprint Gallery View.
//!
//! Provides a searchable, categorized showcase of all 20 production starter blueprints
//! spanning all 18 AI-Examples curriculum categories, with 1-click loading into the
//! Studio Workbench and multi-format download capabilities (.aid, .yaml, .aidpkg).

use crate::components::palette::TEMPLATES;
use crate::icons::bi;
use crate::storage_manager::{trigger_binary_download, trigger_text_download};
use crate::studio_state::StudioState;
use crate::{AiDesignerApp, AppPage, ScreenConstraints};
use eframe::egui::{self, Color32, RichText};
use shared::REPO_CATEGORIES;

pub fn render_gallery_page(
    app: &mut AiDesignerApp,
    ui: &mut egui::Ui,
    constraints: &ScreenConstraints,
) {
    let mut loaded_yaml: Option<&'static str> = None;

    egui::ScrollArea::vertical()
        .id_salt("blueprint_gallery_main_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            // 1. Header Banner
            egui::Frame::group(ui.style())
                .inner_margin(if constraints.is_mobile { 10.0 } else { 16.0 })
                .corner_radius(8.0)
                .show(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.vertical(|ui| {
                            ui.label(
                                RichText::new(format!(
                                    "{} AI-Examples Starter & Industry Blueprint Gallery ({} Blueprints • 18 Categories)",
                                    bi::COLLECTION,
                                    TEMPLATES.len()
                                ))
                                .strong()
                                .size(17.0),
                            );
                            ui.add_space(2.0);
                            ui.label(
                                RichText::new(
                                    "Browse pre-engineered industry & frontier neural architectures (DeepSeek-V3/R1, Flux.1 MMDiT, Llama-3/Qwen-2.5, Gemma-2, TTT/Titans, Liquid CfC, SNNs, zkML, and more). Open directly in the Studio Workbench or export to .aid, .yaml, or .aidpkg.",
                                )
                                .size(12.5)
                                .color(Color32::from_rgb(175, 185, 205)),
                            );
                        });

                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui
                                .button(
                                    RichText::new(format!(
                                        "{} Return to Studio Workbench",
                                        bi::DIAGRAM_3_FILL
                                    ))
                                    .color(Color32::from_rgb(100, 195, 255))
                                    .strong(),
                                )
                                .clicked()
                            {
                                app.state.active_page = AppPage::Designer;
                                app.persist_state();
                            }
                        });
                    });
                });

            ui.add_space(10.0);

            // 2. Search & Category Filters
            ui.horizontal(|ui| {
                ui.label(RichText::new(format!("{} Filter Blueprints:", bi::SEARCH)).strong());
                ui.add(
                    egui::TextEdit::singleline(&mut app.gallery_search_query)
                        .hint_text("Search by model name, category, or domain...")
                        .desired_width(if constraints.is_mobile { 180.0 } else { 320.0 }),
                );
                if !app.gallery_search_query.is_empty()
                    && ui.small_button(format!("{} Clear", bi::X_LG)).clicked()
                {
                    app.gallery_search_query.clear();
                }

                ui.separator();

                let cat_label = match app.gallery_selected_category {
                    None => "All 18 Categories".to_string(),
                    Some(idx) => {
                        if let Some(cat) = REPO_CATEGORIES.get(idx) {
                            format!("[{:02}] {}", cat.index, cat.name)
                        } else {
                            "All Categories".to_string()
                        }
                    }
                };

                egui::ComboBox::from_id_salt("gallery_category_filter")
                    .selected_text(cat_label)
                    .show_ui(ui, |ui| {
                        if ui
                            .selectable_label(
                                app.gallery_selected_category.is_none(),
                                "All 18 Categories",
                            )
                            .clicked()
                        {
                            app.gallery_selected_category = None;
                        }
                        for (idx, cat) in REPO_CATEGORIES.iter().enumerate() {
                            let label = format!("[{:02}] {}", cat.index, cat.name);
                            if ui
                                .selectable_label(
                                    app.gallery_selected_category == Some(idx),
                                    label,
                                )
                                .clicked()
                            {
                                app.gallery_selected_category = Some(idx);
                            }
                        }
                    });
            });

            ui.add_space(10.0);

            let query = app.gallery_search_query.trim().to_lowercase();
            let selected_cat_filter = app.gallery_selected_category;

            // 3. Blueprint Cards Grid
            let mut matching_count = 0usize;

            for (idx, tmpl) in TEMPLATES.iter().enumerate() {
                let clean_id = tmpl
                    .file_name
                    .trim_end_matches(".yaml")
                    .trim_end_matches(".yaml");
                let target_cat_id = tmpl.category_id();
                let linked_cat = REPO_CATEGORIES
                    .iter()
                    .find(|c| {
                        c.id == target_cat_id || c.recommended_starter == Some(tmpl.file_name)
                    })
                    .unwrap_or(&REPO_CATEGORIES[idx % REPO_CATEGORIES.len()]);
                let tmpl_summary = tmpl.summary();

                if let Some(cat_idx) = selected_cat_filter {
                    if let Some(target_cat) = REPO_CATEGORIES.get(cat_idx) {
                        if linked_cat.id != target_cat.id {
                            continue;
                        }
                    }
                }

                if !query.is_empty()
                    && !tmpl.display_name.to_lowercase().contains(&query)
                    && !clean_id.to_lowercase().contains(&query)
                    && !tmpl_summary.to_lowercase().contains(&query)
                    && !linked_cat.name.to_lowercase().contains(&query)
                    && !linked_cat.description.to_lowercase().contains(&query)
                {
                    continue;
                }

                matching_count += 1;

                egui::Frame::group(ui.style())
                    .inner_margin(12.0)
                    .corner_radius(6.0)
                    .show(ui, |ui| {
                        ui.horizontal_wrapped(|ui| {
                            ui.vertical(|ui| {
                                ui.horizontal(|ui| {
                                    ui.label(
                                        RichText::new(format!(
                                            "{} {}",
                                            bi::DIAGRAM_3,
                                            tmpl.display_name
                                        ))
                                        .size(14.5)
                                        .strong(),
                                    );
                                    let cat_url = linked_cat.category_url();
                                    ui.hyperlink_to(
                                        RichText::new(format!(
                                            "{} [{}] {} {}",
                                            bi::BOOK,
                                            linked_cat.id,
                                            linked_cat.name,
                                            bi::BOX_ARROW_UP_RIGHT
                                        ))
                                        .size(11.0)
                                        .monospace()
                                        .color(Color32::from_rgb(110, 195, 250)),
                                        &cat_url,
                                    )
                                    .on_hover_text(format!("Open curriculum category: {cat_url}"));
                                });

                                ui.add_space(2.0);
                                ui.label(
                                    RichText::new(tmpl_summary)
                                        .size(12.0)
                                        .color(Color32::from_rgb(175, 185, 200)),
                                );
                            });

                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if ui
                                        .button(
                                            RichText::new(format!(
                                                "{} Open in Workbench",
                                                bi::ROCKET_TAKEOFF
                                            ))
                                            .color(Color32::from_rgb(120, 225, 150))
                                            .strong(),
                                        )
                                        .on_hover_text("Load this architecture into the Studio Workbench")
                                        .clicked()
                                    {
                                        loaded_yaml = Some(tmpl.yaml_content);
                                    }

                                    if ui
                                        .button(format!("{} .aidpkg", bi::BOX_SEAM))
                                        .on_hover_text("Download page-aligned .aidpkg container")
                                        .clicked()
                                        && let Ok(spec) =
                                            shared::format::aid_yaml::from_yaml_str(tmpl.yaml_content)
                                        && let Ok(bytes) =
                                            shared::format::aidpkg::create_aidpkg(&spec, None, None)
                                    {
                                        let fname = format!("{clean_id}.aidpkg");
                                        trigger_binary_download(
                                            &fname,
                                            &bytes,
                                            "application/octet-stream",
                                        );
                                    }

                                    if ui
                                        .button(format!("{} .aid", bi::FILE_EARMARK_BINARY))
                                        .on_hover_text(
                                            "Download native binary .aid blueprint container",
                                        )
                                        .clicked()
                                        && let Ok(spec) =
                                            shared::format::aid_yaml::from_yaml_str(
                                                tmpl.yaml_content,
                                            )
                                        && let Ok(aid_bytes) =
                                            shared::format::aid_envelope::pack_aid(
                                                &spec,
                                                shared::format::AidFormatTag::Bson,
                                            )
                                    {
                                        let fname = format!("{clean_id}.aid");
                                        trigger_binary_download(
                                            &fname,
                                            &aid_bytes,
                                            "application/octet-stream",
                                        );
                                    }

                                    if ui
                                        .button(format!("{} .yaml", bi::FILE_EARMARK_CODE))
                                        .on_hover_text("Download human-readable .yaml blueprint")
                                        .clicked()
                                    {
                                        let fname = format!("{clean_id}.yaml");
                                        trigger_text_download(
                                            &fname,
                                            tmpl.yaml_content,
                                            "text/yaml;charset=utf-8",
                                        );
                                    }
                                },
                            );
                        });
                    });
                ui.add_space(4.0);
            }

            if matching_count == 0 {
                ui.add_space(20.0);
                ui.centered_and_justified(|ui| {
                    ui.label(
                        RichText::new("No starter blueprints match your filter criteria.")
                            .size(13.0)
                            .weak(),
                    );
                });
            }

            ui.add_space(20.0);
        });

    if let Some(yaml) = loaded_yaml
        && let Ok(spec) = shared::format::aid_yaml::from_yaml_str(yaml)
    {
        app.studio = StudioState::new_with_starter(spec);
        app.state.active_page = AppPage::Designer;
        app.persist_state();
    }
}
