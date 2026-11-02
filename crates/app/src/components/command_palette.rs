//! Global Command Palette & Quick Search (`Ctrl+K` / `⌘K`).
//!
//! Provides instant keyboard-first search and action dispatch across:
//! 1. Layers in the active neural architecture (jump, inspect, and center camera)
//! 2. Complete Operator Catalog (insert GQA, MLA, Mamba, FNO, ConvNeXt, MoE, etc.)
//! 3. Starter Blueprints & Industry Archetypes (instant load with undo snapshot)
//! 4. Multi-Stage Training Presets (Frontier LLM, Muon+WSD, SciML, QAT, etc.)
//! 5. Studio Commands (Zen mode, layout panel toggles, auto-layout, codegen, undo/redo)

use crate::icons::bi;
use crate::studio_state::{CanvasMode, StudioFocus, StudioState};
use eframe::egui::{self, Color32, CornerRadius, Pos2, RichText, Stroke, Vec2};
use shared::spec::training::TrainingPlanPreset;
use shared::templates::TEMPLATES;
use shared::{NodeSpec, OperatorKind, PrecisionFormat};

#[derive(Debug, Clone, PartialEq)]
pub enum PaletteAction {
    SelectNode(String),
    InsertOperator(&'static str, OperatorKind),
    LoadTemplate(usize),
    ApplyTrainingPreset(TrainingPlanPreset),
    ExecuteCommand(StudioCommand),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StudioCommand {
    FocusModel,
    FocusTraining,
    FocusSurgery,
    ToggleZen,
    TogglePalette,
    ToggleInspector,
    ToggleDock,
    AutoLayout,
    GenerateCode,
    Undo,
    Redo,
    OpenShortcuts,
}

#[derive(Debug, Clone)]
pub struct PaletteItem {
    pub category: &'static str,
    pub title: String,
    pub subtitle: String,
    pub icon: &'static str,
    pub badge_color: Color32,
    pub action: PaletteAction,
}

pub struct CommandPalette {
    pub is_open: bool,
    pub query: String,
    pub selected_index: usize,
    pub request_focus: bool,
}

impl Default for CommandPalette {
    fn default() -> Self {
        Self {
            is_open: false,
            query: String::new(),
            selected_index: 0,
            request_focus: false,
        }
    }
}

impl CommandPalette {
    pub fn open(&mut self) {
        self.is_open = true;
        self.query.clear();
        self.selected_index = 0;
        self.request_focus = true;
    }

    pub fn close(&mut self) {
        self.is_open = false;
        self.query.clear();
        self.selected_index = 0;
    }

    pub fn toggle(&mut self) {
        if self.is_open {
            self.close();
        } else {
            self.open();
        }
    }

    pub fn render(
        &mut self,
        ctx: &egui::Context,
        state: &mut StudioState,
        show_shortcuts_modal: &mut bool,
    ) {
        if !self.is_open {
            return;
        }

        let mut should_close = false;
        let mut executed_action: Option<PaletteAction> = None;

        let screen_rect = ctx.viewport_rect();
        let modal_width = (screen_rect.width() - 32.0).clamp(320.0, 680.0);
        let modal_height = (screen_rect.height() - 100.0).clamp(300.0, 480.0);

        egui::Window::new("command_palette_window")
            .title_bar(false)
            .resizable(false)
            .collapsible(false)
            .fixed_size(Vec2::new(modal_width, modal_height))
            .anchor(egui::Align2::CENTER_TOP, Vec2::new(0.0, 70.0))
            .frame(
                egui::Frame::NONE
                    .fill(Color32::from_rgb(18, 22, 32))
                    .stroke(Stroke::new(1.5, Color32::from_rgb(70, 110, 180)))
                    .corner_radius(CornerRadius::same(10))
                    .inner_margin(egui::Margin::symmetric(14, 12)),
            )
            .show(ctx, |ui| {
                // Top Search Bar with Icon
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(bi::SEARCH)
                            .size(16.0)
                            .color(Color32::from_rgb(120, 180, 255)),
                    );

                    let search_input = egui::TextEdit::singleline(&mut self.query)
                        .hint_text("Search layers, ops, blueprints, presets, or commands... (Esc to close)")
                        .desired_width(ui.available_width() - 40.0)
                        .margin(egui::Margin::symmetric(8, 6));

                    let output = ui.add(search_input);
                    if self.request_focus {
                        output.request_focus();
                        self.request_focus = false;
                    }

                    if ui
                        .small_button(bi::X_LG)
                        .on_hover_text("Close Command Palette (Esc)")
                        .clicked()
                    {
                        should_close = true;
                    }
                });

                ui.add_space(4.0);
                ui.separator();
                ui.add_space(4.0);

                // Collect and filter items based on query
                let items = self.collect_matching_items(state);
                let total_items = items.len();

                if total_items == 0 {
                    ui.add_space(20.0);
                    ui.vertical_centered(|ui| {
                        ui.label(
                            RichText::new("No matching layers, operators, blueprints, or commands found.")
                                .color(Color32::from_rgb(150, 165, 185)),
                        );
                        ui.label(
                            RichText::new("Try typing 'attention', 'norm', 'llama', 'muon', or 'zen'.")
                                .size(11.0)
                                .italics()
                                .color(Color32::from_rgb(120, 135, 160)),
                        );
                    });
                    return;
                }

                // Keyboard Navigation (Up / Down / Enter / Esc)
                if ctx.input(|i| i.key_pressed(egui::Key::ArrowDown)) {
                    self.selected_index = (self.selected_index + 1) % total_items;
                }
                if ctx.input(|i| i.key_pressed(egui::Key::ArrowUp)) {
                    self.selected_index = if self.selected_index == 0 {
                        total_items - 1
                    } else {
                        self.selected_index - 1
                    };
                }
                if ctx.input(|i| i.key_pressed(egui::Key::Enter)) {
                    if let Some(item) = items.get(self.selected_index) {
                        executed_action = Some(item.action.clone());
                        should_close = true;
                    }
                }
                if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
                    should_close = true;
                }

                // Results List
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        for (idx, item) in items.iter().enumerate() {
                            let is_selected = idx == self.selected_index;
                            let bg_color = if is_selected {
                                Color32::from_rgb(32, 48, 75)
                            } else {
                                Color32::from_rgba_premultiplied(0, 0, 0, 0)
                            };

                            let border_stroke = if is_selected {
                                Stroke::new(1.0, Color32::from_rgb(90, 150, 230))
                            } else {
                                Stroke::NONE
                            };

                            egui::Frame::NONE
                                .fill(bg_color)
                                .stroke(border_stroke)
                                .corner_radius(CornerRadius::same(6))
                                .inner_margin(egui::Margin::symmetric(10, 6))
                                .show(ui, |ui| {
                                    let row_resp = ui.horizontal(|ui| {
                                        // Category badge
                                        ui.label(
                                            RichText::new(format!("{} {}", item.icon, item.category))
                                                .size(10.5)
                                                .color(item.badge_color)
                                                .monospace()
                                                .strong(),
                                        );

                                        ui.separator();

                                        // Item Title & Subtitle
                                        ui.label(
                                            RichText::new(&item.title)
                                                .size(12.5)
                                                .color(Color32::from_rgb(230, 240, 255))
                                                .strong(),
                                        );

                                        ui.with_layout(
                                            egui::Layout::right_to_left(egui::Align::Center),
                                            |ui| {
                                                ui.label(
                                                    RichText::new(&item.subtitle)
                                                        .size(10.5)
                                                        .color(Color32::from_rgb(140, 160, 190)),
                                                );
                                            },
                                        );
                                    });

                                    if row_resp.response.hovered() {
                                        self.selected_index = idx;
                                    }
                                    if row_resp.response.clicked() {
                                        executed_action = Some(item.action.clone());
                                        should_close = true;
                                    }
                                });
                            ui.add_space(2.0);
                        }
                    });
            });

        // Execute dispatched action
        if let Some(action) = executed_action {
            self.dispatch_action(action, state, show_shortcuts_modal);
        }

        if should_close {
            self.close();
        }
    }

    fn collect_matching_items(&self, state: &StudioState) -> Vec<PaletteItem> {
        let q = self.query.trim().to_lowercase();
        let mut results = Vec::new();

        // 1. Studio Commands
        let commands = [
            (
                StudioCommand::FocusModel,
                "Focus: Model Architecture",
                "Switch to layers, wiring graph, and target code generator (Alt+M)",
                bi::DIAGRAM_3_FILL,
                Color32::from_rgb(110, 200, 255),
            ),
            (
                StudioCommand::FocusTraining,
                "Focus: Training & Optimization",
                "Switch to curriculum stages, 4D parallelism, and runner (Alt+T)",
                bi::SPEEDOMETER2,
                Color32::from_rgb(120, 230, 160),
            ),
            (
                StudioCommand::FocusSurgery,
                "Focus: Surgery & Soups Lab",
                "Switch to refusal abliteration, model soups, and grafting (Alt+S)",
                bi::SCISSORS,
                Color32::from_rgb(255, 175, 110),
            ),
            (
                StudioCommand::ToggleZen,
                "Toggle Zen Mode",
                "Maximize canvas by hiding all sidebars and bottom dock (F11)",
                bi::ARROWS_FULLSCREEN,
                Color32::from_rgb(240, 215, 90),
            ),
            (
                StudioCommand::AutoLayout,
                "Auto-Layout & Center Nodes",
                "Organize Input, Layer, and Output cards cleanly on canvas (Home)",
                bi::CROSSHAIR,
                Color32::from_rgb(140, 210, 255),
            ),
            (
                StudioCommand::GenerateCode,
                "Generate Target Code",
                "Emit PyTorch 2, JAX/Equinox, Rust Candle, WebGPU & Triton projects",
                bi::LIGHTNING_CHARGE_FILL,
                Color32::from_rgb(120, 230, 150),
            ),
            (
                StudioCommand::Undo,
                "Undo Last Blueprint Change",
                "Revert to prior model architecture state (Ctrl+Z)",
                bi::ARROW_COUNTERCLOCKWISE,
                Color32::from_rgb(245, 160, 140),
            ),
            (
                StudioCommand::Redo,
                "Redo Blueprint Change",
                "Re-apply undone model architecture state (Ctrl+Y)",
                bi::ARROW_RIGHT,
                Color32::from_rgb(160, 240, 170),
            ),
            (
                StudioCommand::TogglePalette,
                "Toggle Left Palette",
                "Show/hide operator catalog & starters panel (Alt+1)",
                bi::LAYERS,
                Color32::from_rgb(175, 195, 220),
            ),
            (
                StudioCommand::ToggleDock,
                "Toggle Bottom Dock",
                "Show/hide diagnostics, roofline, pareto & runner dock (Alt+2)",
                bi::GRID_1X2,
                Color32::from_rgb(175, 195, 220),
            ),
            (
                StudioCommand::ToggleInspector,
                "Toggle Right Inspector",
                "Show/hide layer parameters & precision panel (Alt+3)",
                bi::SLIDERS,
                Color32::from_rgb(175, 195, 220),
            ),
            (
                StudioCommand::OpenShortcuts,
                "Keyboard Shortcuts Cheat Sheet",
                "View complete list of hotkeys and canvas gestures (?)",
                bi::KEYBOARD,
                Color32::from_rgb(195, 175, 245),
            ),
        ];

        for (cmd, title, subtitle, icon, color) in commands {
            if q.is_empty()
                || title.to_lowercase().contains(&q)
                || subtitle.to_lowercase().contains(&q)
            {
                results.push(PaletteItem {
                    category: "COMMAND",
                    title: title.to_string(),
                    subtitle: subtitle.to_string(),
                    icon,
                    badge_color: color,
                    action: PaletteAction::ExecuteCommand(cmd),
                });
            }
        }

        // 2. Active Model Layers
        for (i, node) in state.spec.graph.nodes.iter().enumerate() {
            let matches_title = node.name.to_lowercase().contains(&q);
            let matches_id = node.id.to_lowercase().contains(&q);
            let matches_op = node.operator.category_tag().to_lowercase().contains(&q);

            if q.is_empty() || matches_title || matches_id || matches_op {
                results.push(PaletteItem {
                    category: "LAYER",
                    title: format!("{}. {}", i + 1, node.name),
                    subtitle: format!(
                        "ID: `{}` • {} • {}",
                        node.id,
                        node.operator.category_tag(),
                        node.precision.unwrap_or(PrecisionFormat::BF16).label()
                    ),
                    icon: bi::LAYERS_FILL,
                    badge_color: Color32::from_rgb(110, 205, 255),
                    action: PaletteAction::SelectNode(node.id.clone()),
                });
            }
        }

        // 3. Insert Operators Catalog
        let catalog_ops = vec![
            (
                "Grouped Query Attention (GQA)",
                OperatorKind::GQA {
                    num_q_heads: 16,
                    num_kv_heads: 4,
                    d_k: 128,
                },
                "Transformer multi-head attention with shared KV groups (LLaMA-3/Gemma)",
            ),
            (
                "Multi-Head Attention (MHA)",
                OperatorKind::MHA {
                    num_heads: 16,
                    d_k: 128,
                    d_v: 128,
                    dropout: 0.0,
                },
                "Standard uncompressed multi-head attention",
            ),
            (
                "Multi-Head Latent Attention (MLA)",
                OperatorKind::MLA {
                    num_heads: 16,
                    d_latent_kv: 512,
                    d_rope: 64,
                    d_v: 128,
                },
                "DeepSeek-V3 low-rank compressed KV attention mechanism",
            ),
            (
                "Mamba-2 Selective State Space (SSD)",
                OperatorKind::Mamba2SSD {
                    d_model: 2048,
                    d_state: 128,
                    d_conv: 4,
                    expand: 2,
                },
                "Linear-time state space operator with structured semi-separable matrices",
            ),
            (
                "SwiGLU Feed-Forward Network",
                OperatorKind::SwiGLU {
                    in_features: 2048,
                    hidden_features: 5632,
                },
                "Gated Linear Unit with SiLU activation function",
            ),
            (
                "DeepSeek Mixture-of-Experts (MoE)",
                OperatorKind::DeepSeekMoE {
                    total_experts: 64,
                    routed_k: 8,
                    shared_experts: 2,
                    d_model: 2048,
                    d_ffn: 1408,
                    aux_loss_free_bias: true,
                },
                "Fine-grained multi-expert routing with shared isolated expert pathways",
            ),
            (
                "RMSNorm (Root Mean Square Normalization)",
                OperatorKind::RMSNorm {
                    dim: 2048,
                    eps: 1e-6,
                },
                "Zero-mean scaling layer normalization used in modern LLMs",
            ),
            (
                "LayerNorm (Affine Layer Normalization)",
                OperatorKind::LayerNorm {
                    dim: 2048,
                    eps: 1e-5,
                },
                "Standard mean and variance normalization",
            ),
            (
                "ConvNeXt V2 Depthwise 2D Block",
                OperatorKind::ConvNeXtV2Block {
                    dim: 256,
                    kernel_size: 7,
                    mlp_ratio: 4.0,
                },
                "Modern pure-convolutional vision and physical spatial block",
            ),
            (
                "Fourier Neural Spectral Operator (FNO)",
                OperatorKind::FourierNeuralOperator {
                    modes: vec![16, 16],
                    width: 64,
                },
                "Spectral frequency domain convolution for scientific PDE simulation",
            ),
            (
                "Linear Dense Projection",
                OperatorKind::Linear {
                    in_features: 2048,
                    out_features: 2048,
                    bias: false,
                },
                "Fully connected matrix multiplication projection",
            ),
        ];

        for (name, op, desc) in catalog_ops {
            if q.is_empty() || name.to_lowercase().contains(&q) || desc.to_lowercase().contains(&q)
            {
                results.push(PaletteItem {
                    category: "INSERT OP",
                    title: format!("Insert {name}"),
                    subtitle: desc.to_string(),
                    icon: bi::PLUS_CIRCLE,
                    badge_color: Color32::from_rgb(130, 230, 160),
                    action: PaletteAction::InsertOperator(name, op),
                });
            }
        }

        // 4. Starter Blueprints
        for (idx, tmpl) in TEMPLATES.iter().enumerate() {
            let cat = tmpl.category_id();
            let matches_title = tmpl.display_name.to_lowercase().contains(&q);
            let matches_cat = cat.to_lowercase().contains(&q);

            if q.is_empty() || matches_title || matches_cat {
                results.push(PaletteItem {
                    category: "BLUEPRINT",
                    title: format!("Load {}", tmpl.display_name),
                    subtitle: format!("Category: {} • Bundled Architecture", cat),
                    icon: bi::COLLECTION,
                    badge_color: Color32::from_rgb(255, 190, 90),
                    action: PaletteAction::LoadTemplate(idx),
                });
            }
        }

        // 5. Training Presets
        for &preset in TrainingPlanPreset::ALL {
            let matches_title = preset.display_name().to_lowercase().contains(&q);
            let matches_desc = preset.description().to_lowercase().contains(&q);

            if q.is_empty() || matches_title || matches_desc {
                results.push(PaletteItem {
                    category: "PRESET",
                    title: format!("Apply {}", preset.display_name()),
                    subtitle: format!("[{}] {}", preset.domain_tag(), preset.description()),
                    icon: bi::SPEEDOMETER2,
                    badge_color: Color32::from_rgb(220, 160, 255),
                    action: PaletteAction::ApplyTrainingPreset(preset),
                });
            }
        }

        results
    }

    fn dispatch_action(
        &mut self,
        action: PaletteAction,
        state: &mut StudioState,
        show_shortcuts_modal: &mut bool,
    ) {
        match action {
            PaletteAction::SelectNode(node_id) => {
                state.focus_mode = StudioFocus::Model;
                state.canvas_mode = CanvasMode::NodeGraph;
                state.selected_node_id = Some(node_id.clone());
                // Center camera on node
                let node_pos = state.get_node_pos(&node_id);
                state.pan_offset = Vec2::new(300.0 - node_pos.x, 200.0 - node_pos.y);
            }
            PaletteAction::InsertOperator(name, operator) => {
                state.snapshot();
                let suffix = state.spec.graph.nodes.len() + 1;
                let id = format!("layer_{suffix}");
                let node = NodeSpec {
                    id: id.clone(),
                    name: format!("{name} #{suffix}"),
                    operator,
                    precision: Some(PrecisionFormat::BF16),
                    ..Default::default()
                };
                let insert_pos = Pos2::new(
                    400.0 + (suffix as f32 % 3.0 * 290.0),
                    120.0 + (suffix as f32 / 3.0 * 165.0),
                );
                state.node_positions.insert(id.clone(), insert_pos);
                state.spec.graph.nodes.push(node);
                state.selected_node_id = Some(id);
                state.refresh_analyses();
            }
            PaletteAction::LoadTemplate(idx) => {
                if let Some(tmpl) = TEMPLATES.get(idx) {
                    state.snapshot();
                    if let Ok(spec) = shared::format::aid_yaml::from_yaml_str(tmpl.yaml_content) {
                        *state = StudioState::new_with_starter(spec);
                    }
                }
            }
            PaletteAction::ApplyTrainingPreset(preset) => {
                state.snapshot();
                preset.apply_to_spec(&mut state.spec, 8, Some(4096));
                state.focus_mode = StudioFocus::Training;
                state.canvas_mode = CanvasMode::TrainingPlanner;
                state.refresh_analyses();
            }
            PaletteAction::ExecuteCommand(cmd) => match cmd {
                StudioCommand::FocusModel => state.set_focus(StudioFocus::Model),
                StudioCommand::FocusTraining => state.set_focus(StudioFocus::Training),
                StudioCommand::FocusSurgery => state.set_focus(StudioFocus::Surgery),
                StudioCommand::ToggleZen => state.toggle_zen_mode(),
                StudioCommand::TogglePalette => state.toggle_left_panel(),
                StudioCommand::ToggleInspector => state.toggle_right_panel(),
                StudioCommand::ToggleDock => state.toggle_bottom_dock(),
                StudioCommand::AutoLayout => state.auto_layout_nodes(),
                StudioCommand::GenerateCode => state.trigger_code_generation(),
                StudioCommand::Undo => {
                    state.undo();
                }
                StudioCommand::Redo => {
                    state.redo();
                }
                StudioCommand::OpenShortcuts => {
                    *show_shortcuts_modal = true;
                }
            },
        }
    }
}
