//! Main egui application view controller, shell layout, and cross-platform runners.
#![allow(clippy::collapsible_if)]
#![allow(clippy::if_same_then_else)]

pub mod components;
pub mod icons;
pub mod storage_manager;
pub mod studio_state;

pub use components::*;
pub use icons::{bi, install_bootstrap_icons};
pub use storage_manager::*;
pub use studio_state::*;

use eframe::egui;
use serde::{Deserialize, Serialize};
use shared::ThemeMode;
pub use spodeian_ui::ScreenConstraints;
#[allow(unused_imports)]
use tracing::{error, info, warn};

/// Active page in AI Designer Studio.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AppPage {
    #[default]
    Designer,
    Gallery,
    About,
}

impl AppPage {
    pub fn label(self) -> &'static str {
        match self {
            Self::Designer => "Studio Workbench",
            Self::Gallery => "Blueprint Gallery",
            Self::About => "Docs & Architecture",
        }
    }
}

/// Supported export formats for active neural architecture blueprints.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum BlueprintExportFormat {
    #[default]
    Yaml,
    Json,
    Toml,
    Ron,
    NativeAid,
    AidPkg,
}

impl BlueprintExportFormat {
    pub fn label(self) -> &'static str {
        match self {
            Self::Yaml => "YAML Blueprint (.yaml)",
            Self::Json => "JSON Blueprint (.json)",
            Self::Toml => "TOML Blueprint (.toml)",
            Self::Ron => "RON Blueprint (.ron)",
            Self::NativeAid => "Native Binary Container (.aid)",
            Self::AidPkg => "Page-Aligned Package (.aidpkg)",
        }
    }

    pub fn is_text(self) -> bool {
        matches!(self, Self::Yaml | Self::Json | Self::Toml | Self::Ron)
    }
}

pub type ExportFormat = BlueprintExportFormat;

/// Persisted user session state for AI Designer Studio.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AppState {
    #[serde(default)]
    pub theme: ThemeMode,
    #[serde(default)]
    pub active_page: AppPage,
    #[serde(default)]
    pub saved_spec: Option<shared::AiDesignSpec>,
    #[serde(default)]
    pub canvas_mode: CanvasMode,
    #[serde(default)]
    pub bottom_dock_tab: BottomDockTab,
    #[serde(default)]
    pub workspace_registry: shared::WorkspaceRegistry,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            theme: ThemeMode::Dark,
            active_page: AppPage::Designer,
            saved_spec: None,
            canvas_mode: CanvasMode::NodeGraph,
            bottom_dock_tab: BottomDockTab::Advisor,
            workspace_registry: shared::WorkspaceRegistry::default(),
        }
    }
}

/// AI Designer Studio root application struct.
pub struct AiDesignerApp {
    pub state: AppState,
    pub current_theme: Option<ThemeMode>,
    pub show_reset_dialog: bool,
    pub show_help_dialog: bool,
    pub show_import_dialog: bool,
    pub import_text_buffer: String,
    pub import_result_message: Option<Result<String, String>>,
    pub show_export_dialog: Option<BlueprintExportFormat>,
    pub export_text_buffer: String,
    pub export_copied_notification: Option<f64>,
    pub selected_export_format: BlueprintExportFormat,
    pub gallery_search_query: String,
    pub gallery_selected_category: Option<usize>,
    pub storage_diag: StorageDiagnostics,
    pub show_storage_modal: bool,
    pub show_telemetry_modal: bool,
    pub dismissed_ephemeral_warning: bool,
    pub dismissed_quota_warning: bool,
    pub dismissed_combined_warning: bool,
    pub last_diag_poll_time: f64,

    // AI Designer Studio Workbench Views
    pub studio: StudioState,
    pub palette_view: components::PaletteView,
    pub inspector_view: components::InspectorView,
    pub canvas_node_view: components::CanvasNodeView,
    pub canvas_tree_view: components::CanvasTreeView,
    pub training_planner_view: components::TrainingPlannerView,
    pub canvas_yaml_view: components::CanvasYamlView,
    pub advisor_dock_view: components::AdvisorDockView,
    pub roofline_dock_view: components::RooflineDockView,
    pub code_viewer_view: components::CodeViewerView,
    pub studio_runner_view: components::StudioRunnerView,
    pub model_surgery_view: components::ModelSurgeryView,
    pub design_wizard_modal: components::DesignWizardModal,
    pub workspace_manager_modal: components::WorkspaceManagerModal,
}

pub type TemplateApp = AiDesignerApp;

impl Default for AiDesignerApp {
    fn default() -> Self {
        let starter_yaml = shared::templates::TEMPLATES[0].yaml_content;
        let starter_spec =
            shared::format::aid_yaml::from_yaml_str(starter_yaml).unwrap_or_else(|_| {
                shared::AiDesignSpec::new("spatiotemporal_dit_flow", "Spatiotemporal DiT Flow")
            });

        Self {
            state: AppState::default(),
            current_theme: None,
            show_reset_dialog: false,
            show_help_dialog: false,
            show_import_dialog: false,
            import_text_buffer: String::new(),
            import_result_message: None,
            show_export_dialog: None,
            export_text_buffer: String::new(),
            export_copied_notification: None,
            selected_export_format: BlueprintExportFormat::default(),
            gallery_search_query: String::new(),
            gallery_selected_category: None,
            storage_diag: query_storage_diagnostics(),
            show_storage_modal: false,
            show_telemetry_modal: false,
            dismissed_ephemeral_warning: false,
            dismissed_quota_warning: false,
            dismissed_combined_warning: false,
            last_diag_poll_time: 0.0,
            studio: StudioState::new_with_starter(starter_spec),
            palette_view: components::PaletteView::default(),
            inspector_view: components::InspectorView,
            canvas_node_view: components::CanvasNodeView,
            canvas_tree_view: components::CanvasTreeView,
            training_planner_view: components::TrainingPlannerView::default(),
            canvas_yaml_view: components::CanvasYamlView,
            advisor_dock_view: components::AdvisorDockView,
            roofline_dock_view: components::RooflineDockView,
            code_viewer_view: components::CodeViewerView,
            studio_runner_view: components::StudioRunnerView,
            model_surgery_view: components::ModelSurgeryView,
            design_wizard_modal: components::DesignWizardModal::default(),
            workspace_manager_modal: components::WorkspaceManagerModal::default(),
        }
    }
}

impl AiDesignerApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        info!("Initializing AI Designer Studio & Training Suite...");

        let loaded_state = load_state_multi_tier(cc.storage).unwrap_or_default();
        let initial_spec = if let Some(ws) = loaded_state.workspace_registry.active_workspace() {
            if let Some(d) = ws.active_design() {
                d.spec.clone()
            } else {
                loaded_state.saved_spec.clone().unwrap_or_else(|| {
                    let starter_yaml = shared::templates::TEMPLATES[0].yaml_content;
                    shared::format::aid_yaml::from_yaml_str(starter_yaml).unwrap_or_else(|_| {
                        shared::AiDesignSpec::new(
                            "spatiotemporal_dit_flow",
                            "Spatiotemporal DiT Flow",
                        )
                    })
                })
            }
        } else {
            loaded_state.saved_spec.clone().unwrap_or_else(|| {
                let starter_yaml = shared::templates::TEMPLATES[0].yaml_content;
                shared::format::aid_yaml::from_yaml_str(starter_yaml).unwrap_or_else(|_| {
                    shared::AiDesignSpec::new("spatiotemporal_dit_flow", "Spatiotemporal DiT Flow")
                })
            })
        };

        let mut studio = StudioState::new_with_starter(initial_spec);
        studio.canvas_mode = loaded_state.canvas_mode;
        studio.bottom_dock_tab = loaded_state.bottom_dock_tab;

        Self {
            state: loaded_state,
            studio,
            ..Default::default()
        }
    }

    /// Persists the active session state and architecture blueprint to client storage.
    pub fn persist_state(&mut self) {
        // Sync active studio spec & surgery into active workspace design
        if let Some(ws) = self.state.workspace_registry.active_workspace_mut() {
            components::WorkspaceManagerModal::sync_studio_to_workspace(ws, &self.studio);
        }

        self.state.saved_spec = Some(self.studio.spec.clone());
        self.state.canvas_mode = self.studio.canvas_mode;
        self.state.bottom_dock_tab = self.studio.bottom_dock_tab;

        if let Ok(json_str) = serde_json::to_string(&self.state) {
            let size = json_str.len();
            self.storage_diag.usage_bytes = size as u64;
            match save_state_multi_tier(DEDICATED_STORAGE_KEY, &json_str) {
                Ok(backend) => {
                    self.storage_diag.backend = backend;
                    self.storage_diag.quota_exceeded = backend == StorageBackend::IndexedDb;
                }
                Err(_) => {
                    self.storage_diag.quota_exceeded = true;
                }
            }
        }
    }

    pub fn open_export_dialog(&mut self, format: BlueprintExportFormat) {
        match format {
            BlueprintExportFormat::Yaml => {
                self.export_text_buffer =
                    shared::format::aid_yaml::to_yaml_string(&self.studio.spec).unwrap_or_default();
            }
            BlueprintExportFormat::Json => {
                self.export_text_buffer =
                    shared::format::aid_json::to_json_string(&self.studio.spec).unwrap_or_default();
            }
            BlueprintExportFormat::Toml => {
                self.export_text_buffer =
                    shared::format::aid_toml::to_toml_string(&self.studio.spec).unwrap_or_default();
            }
            BlueprintExportFormat::Ron => {
                self.export_text_buffer =
                    shared::format::aid_ron::to_ron_string(&self.studio.spec).unwrap_or_default();
            }
            BlueprintExportFormat::NativeAid => {
                if let Ok(bytes) = shared::format::aid_envelope::pack_aid(
                    &self.studio.spec,
                    shared::format::AidFormatTag::Bson,
                ) {
                    let filename = format!("{}.aid", self.studio.spec.metadata.model_id);
                    trigger_binary_download(&filename, &bytes, "application/octet-stream");
                }
                self.export_text_buffer = format!(
                    "# Native binary .aid format triggered for download: {}.aid",
                    self.studio.spec.metadata.model_id
                );
            }
            BlueprintExportFormat::AidPkg => {
                if let Ok(bytes) =
                    shared::format::aidpkg::create_aidpkg(&self.studio.spec, None, None)
                {
                    let filename = format!("{}.aidpkg", self.studio.spec.metadata.model_id);
                    trigger_binary_download(&filename, &bytes, "application/octet-stream");
                }
                self.export_text_buffer = format!(
                    "# Page-aligned .aidpkg container triggered for download: {}.aidpkg",
                    self.studio.spec.metadata.model_id
                );
            }
        }
        self.show_export_dialog = Some(format);
        self.export_copied_notification = None;
    }

    pub fn render_designer_studio(&mut self, ui: &mut egui::Ui) {
        // 1. Studio Sub-header Toolbar & 5-Stage Workflow Strip
        ui.horizontal_wrapped(|ui| {
            let (ws_name, model_count) = self
                .state
                .workspace_registry
                .active_workspace()
                .map(|w| (w.name.clone(), w.designs.len()))
                .unwrap_or_else(|| ("Default Workspace".to_string(), 1));

            if ui
                .button(
                    egui::RichText::new(format!(
                        "{} Workspace: {ws_name} ({model_count})",
                        bi::FOLDER2_OPEN
                    ))
                    .color(egui::Color32::from_rgb(120, 200, 240))
                    .strong(),
                )
                .on_hover_text(
                    "Open Workspace & Multi-Model Project Manager: switch workspaces, manage multiple models, ablate variants, view parity matrix",
                )
                .clicked()
            {
                self.workspace_manager_modal.is_open = true;
            }

            ui.separator();

            ui.heading(
                egui::RichText::new(&self.studio.spec.metadata.display_name)
                    .size(15.5)
                    .strong(),
            );
            ui.label(
                egui::RichText::new(format!("({})", self.studio.spec.metadata.model_id))
                    .size(11.5)
                    .monospace()
                    .color(egui::Color32::from_rgb(155, 170, 195)),
            );

            ui.separator();

            // Stage 1: Scaffold & Wizards
            if ui
                .button(
                    egui::RichText::new(format!("1. {} New / Wizard", bi::MAGIC))
                        .color(egui::Color32::from_rgb(100, 190, 255))
                        .strong(),
                )
                .on_hover_text(
                    "Stage 1 — Scaffold: Open the guided AI Architecture Wizard (16 Archetypes, 59 Industry Blueprints, or Blank Canvas)",
                )
                .clicked()
            {
                self.design_wizard_modal.open();
            }
            if ui
                .button(
                    egui::RichText::new(format!("{} Plan Training Wizard", bi::SPEEDOMETER2))
                        .color(egui::Color32::from_rgb(130, 225, 165)),
                )
                .on_hover_text(
                    "Open the Guided Training Plan Wizard (12 multi-stage presets: Frontier LLM, Muon+WSD, Flow Matching, SciML Sobolev, Super-Smooth Lipschitz, Learnable QAT, Tabular SSL, etc.)",
                )
                .clicked()
            {
                self.design_wizard_modal.open_training_wizard();
            }

            ui.separator();

            // Stage 2: Compose & Plan Training
            ui.label(
                egui::RichText::new("2. Compose:")
                    .size(11.5)
                    .color(egui::Color32::from_rgb(170, 180, 200)),
            );
            if ui
                .selectable_label(
                    self.studio.canvas_mode == CanvasMode::NodeGraph,
                    format!(
                        "{} Graph ({})",
                        bi::DIAGRAM_3,
                        self.studio.spec.graph.nodes.len()
                    ),
                )
                .on_hover_text("Interactive Input -> Layer -> Output dataflow graph canvas")
                .clicked()
            {
                self.studio.canvas_mode = CanvasMode::NodeGraph;
            }
            if ui
                .selectable_label(
                    self.studio.canvas_mode == CanvasMode::OutlinerTree,
                    format!("{} Outliner", bi::LIST_NESTED),
                )
                .on_hover_text("Hierarchical layer & port tree editor")
                .clicked()
            {
                self.studio.canvas_mode = CanvasMode::OutlinerTree;
            }
            if ui
                .selectable_label(
                    self.studio.canvas_mode == CanvasMode::TrainingPlanner,
                    format!(
                        "{} Training Plan ({})",
                        bi::SPEEDOMETER2,
                        self.studio.spec.training.stages.len().max(1)
                    ),
                )
                .on_hover_text(
                    "Interactive Training Plan Designer: Multi-stage curriculum, LR schedule curve, 4D parallelism & Chinchilla compute budget",
                )
                .clicked()
            {
                self.studio.canvas_mode = CanvasMode::TrainingPlanner;
            }
            if ui
                .selectable_label(
                    self.studio.canvas_mode == CanvasMode::YamlSource,
                    format!("{} YAML (.yaml)", bi::BRACES),
                )
                .on_hover_text("Live .yaml specification editor")
                .clicked()
            {
                self.studio.canvas_mode = CanvasMode::YamlSource;
            }
            if ui
                .selectable_label(
                    self.studio.canvas_mode == CanvasMode::ModelSurgery,
                    format!("{} Surgery & Soups", bi::SCISSORS),
                )
                .on_hover_text("Model Surgery & Weight Manifold Lab: Refusal abliteration, model soups (SLERP/TIES/DARE), and Frankenmerging")
                .clicked()
            {
                self.studio.canvas_mode = CanvasMode::ModelSurgery;
            }

            ui.separator();

            // Stage 3: Diagnose & Pareto
            let finding_count = self.studio.advisor_findings.len();
            let (adv_pill_text, adv_pill_color) = if finding_count == 0 {
                (
                    format!("3. {} 0 Diagnostics", bi::CHECK_CIRCLE_FILL),
                    egui::Color32::from_rgb(120, 220, 140),
                )
            } else {
                (
                    format!("3. {} {finding_count} Diagnostics", bi::EXCLAMATION_TRIANGLE_FILL),
                    egui::Color32::from_rgb(240, 180, 60),
                )
            };
            if ui
                .selectable_label(
                    self.studio.bottom_dock_tab == BottomDockTab::Advisor,
                    egui::RichText::new(adv_pill_text).color(adv_pill_color),
                )
                .on_hover_text(
                    "Stage 3 — Diagnose: Inspect 55+ mathematical & hardware compatibility checks with 1-click quick-fixes",
                )
                .clicked()
            {
                self.studio.bottom_dock_tab = BottomDockTab::Advisor;
            }

            if ui
                .selectable_label(
                    self.studio.bottom_dock_tab == BottomDockTab::ParetoExplorer,
                    format!("{} Pareto (5)", bi::GRAPH_UP_ARROW),
                )
                .on_hover_text(
                    "Stage 3 — Optimize: Compare 5 hardware-aware Pareto scaling candidates",
                )
                .clicked()
            {
                self.studio.bottom_dock_tab = BottomDockTab::ParetoExplorer;
            }

            ui.separator();

            // Stage 4: Verify & Train
            if ui
                .selectable_label(
                    self.studio.bottom_dock_tab == BottomDockTab::StudioRunner,
                    format!("4. {} Verify & Train", bi::PLAY_FILL),
                )
                .on_hover_text(
                    "Stage 4 — Verify: Run deterministic Seed-42 parity checks and live training convergence monitor",
                )
                .clicked()
            {
                self.studio.bottom_dock_tab = BottomDockTab::StudioRunner;
            }

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                // Stage 5: Export & Codegen
                if ui
                    .button(
                        egui::RichText::new(format!(
                            "5. {} Generate Target Code",
                            bi::LIGHTNING_CHARGE_FILL
                        ))
                        .color(egui::Color32::from_rgb(120, 220, 140))
                        .strong(),
                    )
                    .on_hover_text(
                        "Stage 5 — Synthesize: Generate PyTorch 2, JAX/Equinox, Rust Candle, WebGPU WGSL, Triton & Parity Harness",
                    )
                    .clicked()
                {
                    self.studio.trigger_code_generation();
                }

                if ui
                    .button(format!("{} .aidpkg", bi::BOX_SEAM))
                    .on_hover_text("Download page-aligned binary .aidpkg package (blueprint + weights container)")
                    .clicked()
                {
                    if let Ok(bytes) =
                        shared::format::aidpkg::create_aidpkg(&self.studio.spec, None, None)
                    {
                        let filename = format!("{}.aidpkg", self.studio.spec.metadata.model_id);
                        trigger_binary_download(&filename, &bytes, "application/octet-stream");
                    }
                }

                if ui
                    .button(format!("{} .aid", bi::FILE_EARMARK_BINARY))
                    .on_hover_text("Download native binary .aid byte format (AID\\x01 header + CRC32 + BSON payload)")
                    .clicked()
                {
                    if let Ok(bytes) = shared::format::aid_envelope::pack_aid(
                        &self.studio.spec,
                        shared::format::AidFormatTag::Bson,
                    ) {
                        let filename = format!("{}.aid", self.studio.spec.metadata.model_id);
                        trigger_binary_download(&filename, &bytes, "application/octet-stream");
                    }
                }

                if ui
                    .button(format!("{} .yaml", bi::FILE_EARMARK_CODE))
                    .on_hover_text("Download human-readable .yaml blueprint specification")
                    .clicked()
                {
                    if let Ok(yaml_str) =
                        shared::format::aid_yaml::to_yaml_string(&self.studio.spec)
                    {
                        let filename = format!("{}.yaml", self.studio.spec.metadata.model_id);
                        trigger_text_download(&filename, &yaml_str, "text/yaml;charset=utf-8");
                    }
                }
            });
        });

        ui.add_space(4.0);
        ui.separator();

        // 2. Left Sidepanel: Palette
        egui::Panel::left("studio_palette_panel")
            .resizable(true)
            .default_size(260.0)
            .show(ui, |ui| {
                self.palette_view.render(ui, &mut self.studio);
            });

        // 3. Right Sidepanel: Inspector
        egui::Panel::right("studio_inspector_panel")
            .resizable(true)
            .default_size(280.0)
            .show(ui, |ui| {
                self.inspector_view.render(ui, &mut self.studio);
            });

        // 4. Bottom Panel: Collapsible Advisor, Roofline, Pareto & Runner Dock
        egui::Panel::bottom("studio_bottom_dock")
            .resizable(true)
            .default_size(200.0)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    let adv_badge = format!(
                        "{} Advisor ({})",
                        bi::SHIELD_CHECK,
                        self.studio.advisor_findings.len()
                    );
                    if ui
                        .selectable_label(
                            self.studio.bottom_dock_tab == BottomDockTab::Advisor,
                            adv_badge,
                        )
                        .clicked()
                    {
                        self.studio.bottom_dock_tab = BottomDockTab::Advisor;
                    }
                    if ui
                        .selectable_label(
                            self.studio.bottom_dock_tab == BottomDockTab::Roofline,
                            format!("{} Hardware Roofline & VRAM", bi::BAR_CHART_LINE),
                        )
                        .clicked()
                    {
                        self.studio.bottom_dock_tab = BottomDockTab::Roofline;
                    }
                    let pareto_badge = format!(
                        "{} Pareto Frontier ({})",
                        bi::GRAPH_UP_ARROW,
                        self.studio.pareto_candidates.len()
                    );
                    if ui
                        .selectable_label(
                            self.studio.bottom_dock_tab == BottomDockTab::ParetoExplorer,
                            pareto_badge,
                        )
                        .clicked()
                    {
                        self.studio.bottom_dock_tab = BottomDockTab::ParetoExplorer;
                    }
                    if ui
                        .selectable_label(
                            self.studio.bottom_dock_tab == BottomDockTab::StudioRunner,
                            format!("{} Training & Parity Runner", bi::PLAY_FILL),
                        )
                        .clicked()
                    {
                        self.studio.bottom_dock_tab = BottomDockTab::StudioRunner;
                    }
                    if ui
                        .selectable_label(
                            self.studio.bottom_dock_tab == BottomDockTab::ModelSurgery,
                            format!("{} Surgery Lab", bi::SCISSORS),
                        )
                        .clicked()
                    {
                        self.studio.bottom_dock_tab = BottomDockTab::ModelSurgery;
                    }
                });
                ui.separator();

                match self.studio.bottom_dock_tab {
                    BottomDockTab::Advisor => self.advisor_dock_view.render(ui, &mut self.studio),
                    BottomDockTab::Roofline => self.roofline_dock_view.render(ui, &mut self.studio),
                    BottomDockTab::ParetoExplorer => self
                        .roofline_dock_view
                        .render_pareto_explorer(ui, &mut self.studio),
                    BottomDockTab::StudioRunner => {
                        self.studio_runner_view.render(ui, &mut self.studio)
                    }
                    BottomDockTab::ModelSurgery => {
                        self.model_surgery_view.render(ui, &mut self.studio)
                    }
                }
            });

        // 5. Central Canvas Panel
        egui::CentralPanel::default().show(ui, |ui| match self.studio.canvas_mode {
            CanvasMode::NodeGraph => self.canvas_node_view.render(ui, &mut self.studio),
            CanvasMode::OutlinerTree => self.canvas_tree_view.render(ui, &mut self.studio),
            CanvasMode::TrainingPlanner => self.training_planner_view.render(ui, &mut self.studio),
            CanvasMode::YamlSource => self.canvas_yaml_view.render(ui, &mut self.studio),
            CanvasMode::ModelSurgery => self.model_surgery_view.render(ui, &mut self.studio),
        });

        // Modals
        self.code_viewer_view.render(ui.ctx(), &mut self.studio);
        self.design_wizard_modal.render(ui.ctx(), &mut self.studio);
        self.workspace_manager_modal.render(
            ui.ctx(),
            &mut self.state.workspace_registry,
            &mut self.studio,
        );
    }

    fn apply_theme(&mut self, ctx: &egui::Context) {
        if self.current_theme == Some(self.state.theme) {
            return;
        }
        if self.current_theme.is_none() {
            install_bootstrap_icons(ctx);
        }
        self.current_theme = Some(self.state.theme);
        spodeian_ui::apply_theme(ctx, self.state.theme);
    }

    fn handle_keyboard_shortcuts(&mut self, ctx: &egui::Context) {
        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            if self.design_wizard_modal.is_open {
                self.design_wizard_modal.is_open = false;
            } else if self.studio.show_code_modal {
                self.studio.show_code_modal = false;
            } else if self.show_help_dialog {
                self.show_help_dialog = false;
            } else if self.show_reset_dialog {
                self.show_reset_dialog = false;
            } else if self.show_storage_modal {
                self.show_storage_modal = false;
            } else if self.show_telemetry_modal {
                self.show_telemetry_modal = false;
            } else if self.show_export_dialog.is_some() {
                self.show_export_dialog = None;
                self.export_text_buffer.clear();
            } else if self.show_import_dialog {
                self.show_import_dialog = false;
                self.import_text_buffer.clear();
                self.import_result_message = None;
            } else if self.studio.selected_node_id.is_some() {
                self.studio.selected_node_id = None;
            }
        }

        if self.state.active_page == AppPage::Designer && !ctx.egui_wants_keyboard_input() {
            if ctx
                .input(|i| i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace))
            {
                self.studio.delete_selected_node();
            }
            if ctx.input(|i| i.key_pressed(egui::Key::Home)) {
                self.studio.auto_layout_nodes();
            }
        }
    }
}

impl eframe::App for AiDesignerApp {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        // 1. Synchronize in-memory workbench state to persisted state struct
        self.state.saved_spec = Some(self.studio.spec.clone());
        self.state.canvas_mode = self.studio.canvas_mode;
        self.state.bottom_dock_tab = self.studio.bottom_dock_tab;

        // 2. Write standard RON state to eframe::APP_KEY
        eframe::set_value(storage, eframe::APP_KEY, &self.state);

        // 3. Write JSON to dedicated key in storage
        if let Ok(json_str) = serde_json::to_string(&self.state) {
            storage.set_string(DEDICATED_STORAGE_KEY, json_str);
        }
        storage.flush();

        // 4. Persist to multi-tier engine
        self.persist_state();
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.apply_theme(ui.ctx());
        if self.state.theme.is_high_contrast() {
            ui.spacing_mut().interact_size = egui::vec2(44.0, 44.0);
            ui.spacing_mut().button_padding = egui::vec2(14.0, 10.0);
        } else {
            ui.spacing_mut().interact_size.y = ui.spacing_mut().interact_size.y.max(32.0);
            ui.spacing_mut().button_padding = egui::vec2(12.0, 8.0);
        }
        self.handle_keyboard_shortcuts(ui.ctx());

        // Periodic diagnostics poll (every 2 seconds)
        let cur_time = ui.input(|i| i.time);
        if cur_time - self.last_diag_poll_time > 2.0 {
            self.last_diag_poll_time = cur_time;
            let queried = query_storage_diagnostics();
            self.storage_diag.is_persisted = queried.is_persisted;
            self.storage_diag.pwa_install_available = queried.pwa_install_available;
            self.storage_diag.is_pwa_installed = queried.is_pwa_installed;
        }

        let constraints = ScreenConstraints::compute(ui);
        components::navbar::render_navbar(self, ui, &constraints);

        egui::Panel::bottom("bottom_footer").show(ui, |ui| {
            ui.add_space(3.0);
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(
                        "Built with Rust & egui 0.36 • Serverless WASM & Native Desktop",
                    )
                    .small()
                    .color(ui.visuals().weak_text_color()),
                );

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let tier_text = format!("Storage: {}", self.storage_diag.backend.label());
                    ui.label(
                        egui::RichText::new(tier_text)
                            .small()
                            .monospace()
                            .color(ui.visuals().weak_text_color()),
                    );

                    ui.label(
                        egui::RichText::new("•")
                            .small()
                            .color(ui.visuals().weak_text_color()),
                    );

                    if ui
                        .link(format!("{} Studio Workbench", bi::DIAGRAM_3_FILL))
                        .clicked()
                    {
                        self.state.active_page = AppPage::Designer;
                        self.persist_state();
                    }

                    if ui
                        .link(format!("{} Docs & Architecture", bi::INFO_CIRCLE))
                        .clicked()
                    {
                        self.state.active_page = AppPage::About;
                        self.persist_state();
                    }

                    if ui
                        .link(format!("{} Blueprint Gallery", bi::COLLECTION))
                        .clicked()
                    {
                        self.state.active_page = AppPage::Gallery;
                        self.persist_state();
                    }
                });
            });
            ui.add_space(3.0);
        });

        egui::CentralPanel::default().show(ui, |ui| match self.state.active_page {
            AppPage::Designer => {
                self.render_designer_studio(ui);
            }
            AppPage::Gallery => {
                components::gallery::render_gallery_page(self, ui, &constraints);
            }
            AppPage::About => {
                components::about::render_about_page(self, ui, &constraints);
            }
        });

        components::modals::render_dialogs(self, ui);
        components::modals::render_warning_banners(self, ui.ctx());
        components::telemetry_viewer::render_telemetry_modal(self, ui.ctx());
    }
}

#[cfg(target_os = "android")]
#[unsafe(no_mangle)]
fn android_main(app: winit::platform::android::activity::AndroidApp) {
    use eframe::NativeOptions;
    let mut options = NativeOptions::default();
    options.android_app = Some(app);
    if let Err(e) = eframe::run_native(
        "AI Designer Studio",
        options,
        Box::new(|cc| Ok(Box::new(AiDesignerApp::new(cc)))),
    ) {
        tracing::error!("Failed to run egui application on Android: {e}");
    }
}

#[cfg(target_os = "ios")]
#[unsafe(no_mangle)]
pub extern "C" fn ios_main() {
    use eframe::NativeOptions;
    let options = NativeOptions::default();
    let _ = eframe::run_native(
        "AI Designer Studio",
        options,
        Box::new(|cc| Ok(Box::new(AiDesignerApp::new(cc)))),
    );
}
