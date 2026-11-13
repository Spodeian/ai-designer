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
    #[serde(default)]
    pub focus_mode: StudioFocus,
    #[serde(default = "default_panel_visible")]
    pub left_panel_visible: bool,
    #[serde(default = "default_panel_visible")]
    pub right_panel_visible: bool,
    #[serde(default = "default_panel_visible")]
    pub bottom_dock_visible: bool,
}

fn default_panel_visible() -> bool {
    true
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
            focus_mode: StudioFocus::Model,
            left_panel_visible: true,
            right_panel_visible: true,
            bottom_dock_visible: true,
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
    pub command_palette: components::CommandPalette,
    pub show_shortcuts_modal: bool,
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
            command_palette: components::CommandPalette::default(),
            show_shortcuts_modal: false,
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
        studio.focus_mode = loaded_state.focus_mode;
        studio.left_panel_visible = loaded_state.left_panel_visible;
        studio.right_panel_visible = loaded_state.right_panel_visible;
        studio.bottom_dock_visible = loaded_state.bottom_dock_visible;

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
        self.state.focus_mode = self.studio.focus_mode;
        self.state.left_panel_visible = self.studio.left_panel_visible;
        self.state.right_panel_visible = self.studio.right_panel_visible;
        self.state.bottom_dock_visible = self.studio.bottom_dock_visible;

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
        // 1. Contextual Action Ribbon (Top panel for active Focus Mode)
        egui::Panel::top("designer_context_ribbon").show(ui, |ui| {
            ui.add_space(2.0);
            ui.horizontal(|ui| {
            match self.studio.focus_mode {
                StudioFocus::Model => {
                    // Sub-view selectors: Graph, Outliner, YAML
                    if ui
                        .selectable_label(
                            self.studio.canvas_mode == CanvasMode::NodeGraph,
                            format!(
                                "{} Graph Canvas ({})",
                                bi::DIAGRAM_3,
                                self.studio.spec.graph.nodes.len()
                            ),
                        )
                        .on_hover_text("Interactive Input -> Layer -> Output dataflow graph canvas: drag layers, inspect wires, edit tensors")
                        .clicked()
                    {
                        self.studio.canvas_mode = CanvasMode::NodeGraph;
                    }

                    if ui
                        .selectable_label(
                            self.studio.canvas_mode == CanvasMode::OutlinerTree,
                            format!("{} Outliner Tree", bi::LIST_NESTED),
                        )
                        .on_hover_text("Hierarchical layer & port tree editor: reorder execution sequence, duplicate, and nest blocks")
                        .clicked()
                    {
                        self.studio.canvas_mode = CanvasMode::OutlinerTree;
                    }

                    if ui
                        .selectable_label(
                            self.studio.canvas_mode == CanvasMode::YamlSource,
                            format!("{} YAML Blueprint (.yaml)", bi::BRACES),
                        )
                        .on_hover_text("Direct live .yaml specification editor with bidirectional schema synchronization")
                        .clicked()
                    {
                        self.studio.canvas_mode = CanvasMode::YamlSource;
                    }

                    ui.separator();

                    // Model Actions: Wizard, Diagnostics, Roofline/Pareto
                    if ui
                        .button(
                            egui::RichText::new(format!("{} New / Wizard", bi::MAGIC))
                                .color(egui::Color32::from_rgb(100, 190, 255))
                                .strong(),
                        )
                        .on_hover_text("Open the 7-step interactive model architecture builder, 16 research archetypes, or starter templates")
                        .clicked()
                    {
                        self.design_wizard_modal.open();
                    }

                    let finding_count = self.studio.advisor_findings.len();
                    let (adv_pill_text, adv_pill_color) = if finding_count == 0 {
                        (
                            format!("{} 0 Diagnostics", bi::CHECK_CIRCLE_FILL),
                            egui::Color32::from_rgb(120, 220, 140),
                        )
                    } else {
                        (
                            format!("{} {finding_count} Diagnostics", bi::EXCLAMATION_TRIANGLE_FILL),
                            egui::Color32::from_rgb(240, 180, 60),
                        )
                    };
                    if ui
                        .selectable_label(
                            self.studio.bottom_dock_tab == BottomDockTab::Advisor && self.studio.bottom_dock_visible,
                            egui::RichText::new(adv_pill_text).color(adv_pill_color),
                        )
                        .on_hover_text("Inspect 55+ mathematical, topological, and hardware compatibility checks with 1-click automated repairs (click to toggle)")
                        .clicked()
                    {
                        if self.studio.bottom_dock_visible && self.studio.bottom_dock_tab == BottomDockTab::Advisor {
                            self.studio.bottom_dock_visible = false;
                        } else {
                            self.studio.bottom_dock_visible = true;
                            self.studio.bottom_dock_tab = BottomDockTab::Advisor;
                        }
                    }

                    if ui
                        .selectable_label(
                            self.studio.bottom_dock_tab == BottomDockTab::Roofline && self.studio.bottom_dock_visible,
                            format!("{} Roofline & VRAM", bi::BAR_CHART_LINE),
                        )
                        .on_hover_text("Analyze arithmetic intensity, memory bandwidth ceiling, and activation memory footprint (click to toggle)")
                        .clicked()
                    {
                        if self.studio.bottom_dock_visible && self.studio.bottom_dock_tab == BottomDockTab::Roofline {
                            self.studio.bottom_dock_visible = false;
                        } else {
                            self.studio.bottom_dock_visible = true;
                            self.studio.bottom_dock_tab = BottomDockTab::Roofline;
                        }
                    }

                    if ui
                        .selectable_label(
                            self.studio.bottom_dock_tab == BottomDockTab::ParetoExplorer && self.studio.bottom_dock_visible,
                            format!(
                                "{} Pareto Frontier ({})",
                                bi::GRAPH_UP_ARROW,
                                self.studio.pareto_candidates.len()
                            ),
                        )
                        .on_hover_text("Compare 5 hardware-aware Pareto scaling candidates across latency, parameter count, and loss (click to toggle)")
                        .clicked()
                    {
                        if self.studio.bottom_dock_visible && self.studio.bottom_dock_tab == BottomDockTab::ParetoExplorer {
                            self.studio.bottom_dock_visible = false;
                        } else {
                            self.studio.bottom_dock_visible = true;
                            self.studio.bottom_dock_tab = BottomDockTab::ParetoExplorer;
                        }
                    }
                }
                StudioFocus::Training => {
                    // Actions for Training Focus
                    if ui
                        .button(
                            egui::RichText::new(format!("{} Plan Training Wizard", bi::SPEEDOMETER2))
                                .color(egui::Color32::from_rgb(130, 225, 165))
                                .strong(),
                        )
                        .on_hover_text("Open Guided Training Plan Wizard (12 presets: Frontier LLM, Muon+WSD, SciML Sobolev, Learnable QAT, etc.)")
                        .clicked()
                    {
                        self.design_wizard_modal.open_training_wizard();
                    }

                    ui.separator();

                    let stage_count = self.studio.spec.training.stages.len().max(1);
                    let total_tok: u64 = if self.studio.spec.training.stages.is_empty() {
                        self.studio.spec.training.curriculum.total_training_tokens
                    } else {
                        self.studio.spec.training.stages.iter().map(|s| s.token_budget).sum()
                    };
                    let total_tok_str = if total_tok >= 1_000_000_000 {
                        format!("{:.2}B", total_tok as f64 / 1e9)
                    } else {
                        format!("{:.1}M", total_tok as f64 / 1e6)
                    };

                    ui.label(
                        egui::RichText::new(format!(
                            "Curriculum: {stage_count} Stage(s) • Total Budget: {total_tok_str} Tokens"
                        ))
                        .size(11.5)
                        .color(egui::Color32::from_rgb(170, 210, 185)),
                    );

                    ui.separator();

                    if ui
                        .selectable_label(
                            self.studio.bottom_dock_tab == BottomDockTab::StudioRunner && self.studio.bottom_dock_visible,
                            format!("{} Live Runner & Convergence", bi::PLAY_FILL),
                        )
                        .on_hover_text("Open the live training convergence monitor, loss curves, and Seed-42 parity verifier dock (click to toggle)")
                        .clicked()
                    {
                        if self.studio.bottom_dock_visible && self.studio.bottom_dock_tab == BottomDockTab::StudioRunner {
                            self.studio.bottom_dock_visible = false;
                        } else {
                            self.studio.bottom_dock_visible = true;
                            self.studio.bottom_dock_tab = BottomDockTab::StudioRunner;
                        }
                    }
                }
                StudioFocus::Surgery => {
                    // Actions for Surgery Focus
                    ui.selectable_value(
                        &mut self.studio.surgery_tab,
                        SurgeryTab::Abliteration,
                        egui::RichText::new(format!("{} Refusal Abliteration", bi::SHIELD_EXCLAMATION)).strong(),
                    );
                    ui.selectable_value(
                        &mut self.studio.surgery_tab,
                        SurgeryTab::ModelSoups,
                        egui::RichText::new(format!("{} Model Soups (SLERP/TIES/DARE)", bi::LAYERS_FILL)).strong(),
                    );
                    ui.selectable_value(
                        &mut self.studio.surgery_tab,
                        SurgeryTab::LayerGrafting,
                        egui::RichText::new(format!("{} Layer Grafting & Frankenmerge", bi::HDD_STACK)).strong(),
                    );
                    ui.selectable_value(
                        &mut self.studio.surgery_tab,
                        SurgeryTab::MechanisticProbes,
                        egui::RichText::new(format!("{} Mechanistic Probes & SVD", bi::SEARCH)).strong(),
                    );
                }
            }

            // Right-aligned Codegen & Export controls
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                // Unified Export Dropdown
                egui::ComboBox::from_id_salt("global_export_combo")
                    .selected_text(format!("{} Export Blueprint ▾", bi::DOWNLOAD))
                    .show_ui(ui, |ui| {
                        if ui
                            .selectable_label(false, format!("{} Page-Aligned Package (.aidpkg)", bi::BOX_SEAM))
                            .on_hover_text("Page-aligned container packing YAML blueprint and binary weight buffers (ideal for production)")
                            .clicked()
                        {
                            if let Ok(bytes) = shared::format::aidpkg::create_aidpkg(&self.studio.spec, None, None) {
                                let filename = format!("{}.aidpkg", self.studio.spec.metadata.model_id);
                                trigger_binary_download(&filename, &bytes, "application/octet-stream");
                            }
                        }

                        if ui
                            .selectable_label(false, format!("{} Native Binary Container (.aid)", bi::FILE_EARMARK_BINARY))
                            .on_hover_text("High-performance binary serialization with AID\\x01 magic, CRC32, and BSON payload")
                            .clicked()
                        {
                            if let Ok(bytes) = shared::format::aid_envelope::pack_aid(&self.studio.spec, shared::format::AidFormatTag::Bson) {
                                let filename = format!("{}.aid", self.studio.spec.metadata.model_id);
                                trigger_binary_download(&filename, &bytes, "application/octet-stream");
                            }
                        }

                        if ui
                            .selectable_label(false, format!("{} YAML Blueprint (.yaml)", bi::FILE_EARMARK_CODE))
                            .on_hover_text("Human-readable YAML blueprint specification with complete metadata, nodes, and training plan")
                            .clicked()
                        {
                            if let Ok(yaml_str) = shared::format::aid_yaml::to_yaml_string(&self.studio.spec) {
                                let filename = format!("{}.yaml", self.studio.spec.metadata.model_id);
                                trigger_text_download(&filename, &yaml_str, "text/yaml;charset=utf-8");
                            }
                        }

                        if ui
                            .selectable_label(false, format!("{} JSON Specification (.json)", bi::FILE_EARMARK_TEXT))
                            .on_hover_text("Strict JSON format for programmatic web tools and automated API pipelines")
                            .clicked()
                        {
                            self.open_export_dialog(BlueprintExportFormat::Json);
                        }

                        if ui
                            .selectable_label(false, format!("{} TOML Document (.toml)", bi::FILE_EARMARK_TEXT))
                            .on_hover_text("TOML document format matching standard Cargo/Python configuration files")
                            .clicked()
                        {
                            self.open_export_dialog(BlueprintExportFormat::Toml);
                        }
                    });

                // Primary Target Code Generator Button
                if ui
                    .button(
                        egui::RichText::new(format!("{} Generate Target Code", bi::LIGHTNING_CHARGE_FILL))
                            .color(egui::Color32::from_rgb(120, 220, 140))
                            .strong(),
                    )
                    .on_hover_text("Generate verified target code: PyTorch 2, JAX/Equinox, Rust Candle, WebGPU WGSL, Triton Ops & Parity Harness")
                    .clicked()
                {
                    self.studio.trigger_code_generation();
                }
            });
            });
            ui.add_space(2.0);
        });

        // 2. Left Sidepanel: Palette / Catalog (only when visible)
        if self.studio.left_panel_visible {
            egui::Panel::left("studio_palette_panel")
                .resizable(true)
                .default_size(260.0)
                .show(ui, |ui| {
                    self.palette_view.render(ui, &mut self.studio);
                });
        }

        // 3. Right Sidepanel: Inspector (only when visible)
        if self.studio.right_panel_visible {
            egui::Panel::right("studio_inspector_panel")
                .resizable(true)
                .default_size(280.0)
                .show(ui, |ui| {
                    self.inspector_view.render(ui, &mut self.studio);
                });
        }

        // 4. Bottom Panel: Collapsible Advisor, Roofline, Pareto & Runner Dock (only when visible)
        if self.studio.bottom_dock_visible {
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
                            .on_hover_text("Inspect 55+ mathematical, shape, and hardware compatibility checks with auto-fixes")
                            .clicked()
                        {
                            self.studio.bottom_dock_tab = BottomDockTab::Advisor;
                        }
                        if ui
                            .selectable_label(
                                self.studio.bottom_dock_tab == BottomDockTab::Roofline,
                                format!("{} Hardware Roofline & VRAM", bi::BAR_CHART_LINE),
                            )
                            .on_hover_text("Hardware arithmetic intensity and activation memory analysis")
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
                            .on_hover_text("5 hardware-aware Pareto scaling candidates")
                            .clicked()
                        {
                            self.studio.bottom_dock_tab = BottomDockTab::ParetoExplorer;
                        }
                        if ui
                            .selectable_label(
                                self.studio.bottom_dock_tab == BottomDockTab::StudioRunner,
                                format!("{} Training & Parity Runner", bi::PLAY_FILL),
                            )
                            .on_hover_text("Live loss curve, gradient norms, Seed-42 parity checks, and local environment setup")
                            .clicked()
                        {
                            self.studio.bottom_dock_tab = BottomDockTab::StudioRunner;
                        }

                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui
                                .small_button(bi::X_LG)
                                .on_hover_text("Hide bottom dock (Alt+2)")
                                .clicked()
                            {
                                self.studio.bottom_dock_visible = false;
                            }
                        });
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
                        BottomDockTab::ModelSurgery => {}
                    }
                });
        }

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
        self.command_palette
            .render(ui.ctx(), &mut self.studio, &mut self.show_shortcuts_modal);
        components::modals::render_shortcuts_modal(ui.ctx(), &mut self.show_shortcuts_modal);
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
        // Global Command Palette & Shortcuts Modal
        if ctx.input(|i| (i.modifiers.command || i.modifiers.ctrl) && i.key_pressed(egui::Key::K)) {
            self.command_palette.toggle();
        }
        if ctx.input(|i| i.key_pressed(egui::Key::F1)) {
            self.show_shortcuts_modal = !self.show_shortcuts_modal;
        }

        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            if self.command_palette.is_open {
                self.command_palette.close();
            } else if self.show_shortcuts_modal {
                self.show_shortcuts_modal = false;
            } else if self.design_wizard_modal.is_open {
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
            // Undo / Redo Shortcuts
            if ctx.input(|i| {
                (i.modifiers.command || i.modifiers.ctrl)
                    && !i.modifiers.shift
                    && i.key_pressed(egui::Key::Z)
            }) {
                self.studio.undo();
            }
            if ctx.input(|i| {
                (i.modifiers.command || i.modifiers.ctrl)
                    && (i.key_pressed(egui::Key::Y)
                        || (i.modifiers.shift && i.key_pressed(egui::Key::Z)))
            }) {
                self.studio.redo();
            }
            if ctx
                .input(|i| (i.modifiers.command || i.modifiers.ctrl) && i.key_pressed(egui::Key::D))
            {
                self.studio.duplicate_selected_node();
            }

            // Layout & Focus Shortcuts
            if ctx.input(|i| i.key_pressed(egui::Key::F11)) {
                self.studio.toggle_zen_mode();
            }
            if ctx.input(|i| i.modifiers.alt && i.key_pressed(egui::Key::Num1)) {
                self.studio.toggle_left_panel();
            }
            if ctx.input(|i| i.modifiers.alt && i.key_pressed(egui::Key::Num2)) {
                self.studio.toggle_bottom_dock();
            }
            if ctx.input(|i| i.modifiers.alt && i.key_pressed(egui::Key::Num3)) {
                self.studio.toggle_right_panel();
            }
            if ctx.input(|i| i.modifiers.alt && i.key_pressed(egui::Key::M)) {
                self.studio.set_focus(StudioFocus::Model);
            }
            if ctx.input(|i| i.modifiers.alt && i.key_pressed(egui::Key::T)) {
                self.studio.set_focus(StudioFocus::Training);
            }
            if ctx.input(|i| i.modifiers.alt && i.key_pressed(egui::Key::S)) {
                self.studio.set_focus(StudioFocus::Surgery);
            }

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
        self.state.focus_mode = self.studio.focus_mode;
        self.state.left_panel_visible = self.studio.left_panel_visible;
        self.state.right_panel_visible = self.studio.right_panel_visible;
        self.state.bottom_dock_visible = self.studio.bottom_dock_visible;

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
                if self.state.active_page == AppPage::Designer {
                    let sel_info = if let Some(ref sel) = self.studio.selected_node_id {
                        format!("Layer: {sel}")
                    } else {
                        format!("Blueprint: {}", self.studio.spec.metadata.display_name)
                    };
                    let zoom_pct = format!("{:.0}%", self.studio.zoom_level * 100.0);
                    ui.label(
                        egui::RichText::new(format!(
                            "{} {} • Zoom: {} • F: Fit | Z: Focus | M: Map | Del: Delete | Ctrl+K: Cmds",
                            bi::DIAGRAM_3,
                            sel_info,
                            zoom_pct
                        ))
                        .small()
                        .color(ui.visuals().weak_text_color()),
                    );
                } else {
                    ui.label(
                        egui::RichText::new(
                            "Built with Rust & egui 0.36 • Serverless WASM & Native Desktop",
                        )
                        .small()
                        .color(ui.visuals().weak_text_color()),
                    );
                }

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

        match self.state.active_page {
            AppPage::Designer => {
                self.render_designer_studio(ui);
            }
            AppPage::Gallery => {
                egui::CentralPanel::default().show(ui, |ui| {
                    components::gallery::render_gallery_page(self, ui, &constraints);
                });
            }
            AppPage::About => {
                egui::CentralPanel::default().show(ui, |ui| {
                    components::about::render_about_page(self, ui, &constraints);
                });
            }
        }

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
