//! Studio State: manages the active AI blueprint design, node selections, canvas panning, runner processes, and analysis caches.

use eframe::egui::{Pos2, Vec2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use shared::{
    AdvisorFinding, AiDesignSpec, ParetoCandidate, PartitionReport, ProjectBundle, RooflineReport,
    ShapeReport, analyze_roofline, generate_all_backends, generate_pareto_frontier, infer_shapes,
    partition_parameters, run_advisor_checks,
};

/// Canvas visualization and interaction mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CanvasMode {
    #[default]
    NodeGraph,
    OutlinerTree,
    TrainingPlanner,
    ModelSurgery,
    YamlSource,
}

/// Primary studio focus mode: isolates the workspace to a single discipline without clutter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum StudioFocus {
    #[default]
    Model,     // Model architecture, layers, dataflow graph, outliner, YAML
    Training,  // Training plan, stages, compute budget, curriculum, live runner
    Surgery,   // Post-training surgery, refusal abliteration, model soups, grafting
}

impl StudioFocus {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Model => "Model Architecture",
            Self::Training => "Training & Optimization",
            Self::Surgery => "Model Surgery & Soups",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::Model => "Design neural network layers, tensor ports, residual wiring, and export code (Graph, Outliner, YAML)",
            Self::Training => "Configure multi-stage curriculum, optimizers (Muon/AdamW), 4D parallelism, compute budgets, and live runner",
            Self::Surgery => "Perform post-training weight surgery: refusal abliteration, SLERP/TIES model soups, and Frankenmerging",
        }
    }
}

/// Active bottom dock tab.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum BottomDockTab {
    #[default]
    Advisor,
    Roofline,
    ParetoExplorer,
    StudioRunner,
    ModelSurgery,
}

/// Active tab in the Model Surgery & Analysis workbench.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SurgeryTab {
    #[default]
    Abliteration,
    ModelSoups,
    LayerGrafting,
    MechanisticProbes,
}

/// Target language tab for the Code Viewer modal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CodeTab {
    #[default]
    PyTorch2,
    JAXEquinox,
    RustCandle,
    WebGPUWGSL,
    TritonOps,
    CrossBackendHarness,
}

/// Active tab within the Training Runner dock.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RunnerTab {
    #[default]
    LiveConvergence,
    LiveOptimizations,
    EnvironmentSetup,
    ConsoleLog,
}

/// Live execution state of the local training runner and process mediator.
#[derive(Debug, Clone)]
pub struct RunnerState {
    pub is_running: bool,
    pub current_step: usize,
    pub max_steps: usize,
    pub current_loss: f32,
    pub loss_history: Vec<[f64; 2]>,      // [step, loss]
    pub val_loss_history: Vec<[f64; 2]>,  // [step, val_loss_estimate]
    pub grad_norm_history: Vec<[f64; 2]>, // [step, grad_norm]
    pub console_logs: Vec<String>,
    pub session: shared::TrainingSession,
    pub selected_tab: RunnerTab,
    pub env_setup_target_dir: String,
    pub env_setup_result: Option<shared::TrainingEnvironmentManifest>,
    pub env_setup_status_msg: Option<String>,
}

impl Default for RunnerState {
    fn default() -> Self {
        let dummy_spec = shared::AiDesignSpec::default();
        Self {
            is_running: false,
            current_step: 0,
            max_steps: 100,
            current_loss: 0.0,
            loss_history: Vec::new(),
            val_loss_history: Vec::new(),
            grad_norm_history: Vec::new(),
            console_logs: Vec::new(),
            session: shared::TrainingSession::new(&dummy_spec, 100),
            selected_tab: RunnerTab::LiveConvergence,
            env_setup_target_dir: "training_workspace".to_string(),
            env_setup_result: None,
            env_setup_status_msg: None,
        }
    }
}

impl RunnerState {
    pub fn new_for_spec(spec: &shared::AiDesignSpec) -> Self {
        Self {
            is_running: false,
            current_step: 0,
            max_steps: 100,
            current_loss: 0.0,
            loss_history: Vec::new(),
            val_loss_history: Vec::new(),
            grad_norm_history: Vec::new(),
            console_logs: Vec::new(),
            session: shared::TrainingSession::new(spec, 100),
            selected_tab: RunnerTab::LiveConvergence,
            env_setup_target_dir: "training_workspace".to_string(),
            env_setup_result: None,
            env_setup_status_msg: None,
        }
    }

    pub fn reset_with_spec(&mut self, spec: &shared::AiDesignSpec, max_steps: usize) {
        self.session = shared::TrainingSession::new(spec, max_steps);
        self.is_running = false;
        self.current_step = 0;
        self.max_steps = max_steps;
        self.current_loss = 0.0;
        self.loss_history.clear();
        self.val_loss_history.clear();
        self.grad_norm_history.clear();
    }

    pub fn push_log(&mut self, line: String) {
        if self.console_logs.len() >= 1000 {
            self.console_logs.remove(0);
        }
        self.console_logs.push(line.clone());
        self.session.push_log(line);
    }

    pub fn sync_from_session(&mut self) {
        self.is_running = self.session.state == shared::TrainingProcessState::Running;
        self.current_step = self.session.current_telemetry.step;
        self.max_steps = self.session.current_telemetry.max_steps;
        self.current_loss = self.session.current_telemetry.loss;
    }
}

/// Complete in-memory state of the AI Designer Studio UI session.
pub struct StudioState {
    pub spec: AiDesignSpec,
    pub canvas_mode: CanvasMode,
    pub bottom_dock_tab: BottomDockTab,
    pub selected_node_id: Option<String>,
    pub node_positions: HashMap<String, Pos2>,
    pub pan_offset: Vec2,
    pub zoom_level: f32,
    pub is_dragging_canvas: bool,
    pub dragging_node_id: Option<String>,
    pub drag_start_pos: Option<Pos2>,

    // Text editor buffer for YAML mode
    pub yaml_buffer: String,
    pub yaml_error: Option<String>,

    // Cached analytical reports
    pub shape_report: ShapeReport,
    pub advisor_findings: Vec<AdvisorFinding>,
    pub partition_report: PartitionReport,
    pub roofline_report: RooflineReport,
    pub pareto_candidates: Vec<ParetoCandidate>,

    // Code Generation & Viewer
    pub show_code_modal: bool,
    pub active_code_tab: CodeTab,
    pub generated_bundle: Option<ProjectBundle>,
    pub code_status_message: Option<String>,

    // Runner state
    pub runner: RunnerState,

    // Model Surgery & Analysis State
    pub surgery_tab: SurgeryTab,
    pub abliteration_plan: shared::AbliterationPlan,
    pub abliteration_result: Option<shared::AbliterationExecutionResult>,
    pub soup_recipe: shared::SoupMergeRecipe,
    pub soup_result: Option<shared::SoupMergeResult>,
    pub grafting_plan: shared::GraftingPlan,
    pub grafting_result: Option<shared::GraftingExecutionResult>,

    // Workflow Discipline Focus & Layout Panel Visibility
    pub focus_mode: StudioFocus,
    pub left_panel_visible: bool,
    pub right_panel_visible: bool,
    pub bottom_dock_visible: bool,
}

impl StudioState {
    pub fn new_with_starter(spec: AiDesignSpec) -> Self {
        let shapes = infer_shapes(&spec);
        let findings = run_advisor_checks(&spec);
        let partition = partition_parameters(&spec);
        let roofline = analyze_roofline(&spec, 8, 256);
        let pareto = generate_pareto_frontier(&spec, None);

        let yaml_str = shared::format::aid_yaml::to_yaml_string(&spec).unwrap_or_default();
        let runner = RunnerState::new_for_spec(&spec);

        let mut studio = Self {
            spec,
            canvas_mode: CanvasMode::NodeGraph,
            bottom_dock_tab: BottomDockTab::Advisor,
            selected_node_id: None,
            node_positions: HashMap::new(),
            pan_offset: Vec2::ZERO,
            zoom_level: 1.0,
            is_dragging_canvas: false,
            dragging_node_id: None,
            drag_start_pos: None,
            yaml_buffer: yaml_str,
            yaml_error: None,
            shape_report: shapes,
            advisor_findings: findings,
            partition_report: partition,
            roofline_report: roofline,
            pareto_candidates: pareto,
            show_code_modal: false,
            active_code_tab: CodeTab::PyTorch2,
            generated_bundle: None,
            code_status_message: None,
            runner,
            surgery_tab: SurgeryTab::default(),
            abliteration_plan: shared::AbliterationPlan::default(),
            abliteration_result: None,
            soup_recipe: shared::SoupMergeRecipe::default(),
            soup_result: None,
            grafting_plan: shared::GraftingPlan::default(),
            grafting_result: None,
            focus_mode: StudioFocus::Model,
            left_panel_visible: true,
            right_panel_visible: true,
            bottom_dock_visible: true,
        };
        studio.auto_layout_nodes();
        studio
    }

    /// Toggles the left palette / stage catalog panel.
    pub fn toggle_left_panel(&mut self) {
        self.left_panel_visible = !self.left_panel_visible;
    }

    /// Toggles the right property inspector panel.
    pub fn toggle_right_panel(&mut self) {
        self.right_panel_visible = !self.right_panel_visible;
    }

    /// Toggles the bottom diagnostics / runner dock.
    pub fn toggle_bottom_dock(&mut self) {
        self.bottom_dock_visible = !self.bottom_dock_visible;
    }

    /// Toggles Zen Mode: hides all side panels and docks to maximize canvas space, or restores them.
    pub fn toggle_zen_mode(&mut self) {
        if self.left_panel_visible || self.right_panel_visible || self.bottom_dock_visible {
            self.left_panel_visible = false;
            self.right_panel_visible = false;
            self.bottom_dock_visible = false;
        } else {
            self.left_panel_visible = true;
            self.right_panel_visible = true;
            self.bottom_dock_visible = true;
        }
    }

    /// Switches the primary studio focus mode, cleanly configuring views without clutter.
    pub fn set_focus(&mut self, focus: StudioFocus) {
        self.focus_mode = focus;
        match focus {
            StudioFocus::Model => {
                if matches!(
                    self.canvas_mode,
                    CanvasMode::TrainingPlanner | CanvasMode::ModelSurgery
                ) {
                    self.canvas_mode = CanvasMode::NodeGraph;
                }
                if matches!(
                    self.bottom_dock_tab,
                    BottomDockTab::StudioRunner | BottomDockTab::ModelSurgery
                ) {
                    self.bottom_dock_tab = BottomDockTab::Advisor;
                }
            }
            StudioFocus::Training => {
                self.canvas_mode = CanvasMode::TrainingPlanner;
                self.bottom_dock_tab = BottomDockTab::StudioRunner;
            }
            StudioFocus::Surgery => {
                self.canvas_mode = CanvasMode::ModelSurgery;
                self.bottom_dock_tab = BottomDockTab::ModelSurgery;
            }
        }
    }

    /// Re-evaluates all analytical and advisory engines against current spec.
    pub fn refresh_analyses(&mut self) {
        self.shape_report = infer_shapes(&self.spec);
        self.advisor_findings = run_advisor_checks(&self.spec);
        self.partition_report = partition_parameters(&self.spec);
        self.roofline_report = analyze_roofline(&self.spec, 8, 256);
        self.pareto_candidates = generate_pareto_frontier(&self.spec, None);
        self.yaml_buffer = shared::format::aid_yaml::to_yaml_string(&self.spec).unwrap_or_default();
    }

    /// Re-generates all multi-backend projects.
    pub fn trigger_code_generation(&mut self) {
        self.generated_bundle = Some(generate_all_backends(&self.spec));
        self.code_status_message = None;
        self.show_code_modal = true;
    }

    /// Resets canvas pan/zoom and arranges Input ports, Computation layers, and Output ports in a clean left-to-right dataflow pipeline.
    pub fn auto_layout_nodes(&mut self) {
        self.pan_offset = Vec2::ZERO;
        self.zoom_level = 1.0;
        self.node_positions.clear();

        // 1. Input port cards on the left (Column 0)
        for (i, input) in self.spec.inputs.iter().enumerate() {
            self.node_positions.insert(
                format!("__input__:{}", input.id),
                Pos2::new(40.0, 110.0 + (i as f32 * 140.0)),
            );
        }

        // 2. Computation layer nodes in the middle columns
        let cols_per_row = 3usize;
        for (i, node) in self.spec.graph.nodes.iter().enumerate() {
            let col = i % cols_per_row;
            let row = i / cols_per_row;
            self.node_positions.insert(
                node.id.clone(),
                Pos2::new(340.0 + (col as f32 * 290.0), 110.0 + (row as f32 * 165.0)),
            );
        }

        // 3. Output port cards on the right of the last computation node
        let node_count = self.spec.graph.nodes.len();
        let (out_x, out_base_y) = if node_count == 0 {
            (660.0, 110.0)
        } else {
            let last_idx = node_count - 1;
            let last_col = (last_idx % cols_per_row) + 1;
            let last_row = last_idx / cols_per_row;
            (
                340.0 + (last_col as f32 * 290.0),
                110.0 + (last_row as f32 * 165.0),
            )
        };
        for (i, output) in self.spec.outputs.iter().enumerate() {
            self.node_positions.insert(
                format!("__output__:{}", output.id),
                Pos2::new(out_x, out_base_y + (i as f32 * 140.0)),
            );
        }
    }

    /// Moves the currently selected node earlier (`delta < 0`) or later (`delta > 0`) in execution order.
    pub fn move_selected_node(&mut self, delta: isize) -> bool {
        let Some(ref selected_id) = self.selected_node_id else {
            return false;
        };
        let Some(idx) = self
            .spec
            .graph
            .nodes
            .iter()
            .position(|n| &n.id == selected_id)
        else {
            return false;
        };
        let new_idx = idx as isize + delta;
        if new_idx < 0 || new_idx >= self.spec.graph.nodes.len() as isize {
            return false;
        }
        let target_idx = new_idx as usize;
        let id_a = self.spec.graph.nodes[idx].id.clone();
        let id_b = self.spec.graph.nodes[target_idx].id.clone();
        self.spec.graph.nodes.swap(idx, target_idx);
        if let (Some(pos_a), Some(pos_b)) = (
            self.node_positions.get(&id_a).copied(),
            self.node_positions.get(&id_b).copied(),
        ) {
            self.node_positions.insert(id_a, pos_b);
            self.node_positions.insert(id_b, pos_a);
        }
        self.refresh_analyses();
        true
    }

    /// Duplicates the currently selected node immediately after it in the graph.
    pub fn duplicate_selected_node(&mut self) -> Option<String> {
        let selected_id = self.selected_node_id.clone()?;
        let idx = self
            .spec
            .graph
            .nodes
            .iter()
            .position(|n| n.id == selected_id)?;
        let orig = self.spec.graph.nodes[idx].clone();
        let mut suffix = self.spec.graph.nodes.len() + 1;
        let mut new_id = format!("{}_copy_{}", orig.id, suffix);
        while self.spec.graph.nodes.iter().any(|n| n.id == new_id) {
            suffix += 1;
            new_id = format!("{}_copy_{}", orig.id, suffix);
        }
        let new_node = shared::NodeSpec {
            id: new_id.clone(),
            name: format!("{} (Copy)", orig.name),
            operator: orig.operator,
            precision: orig.precision,
            modular_role: orig.modular_role,
            execution_condition: orig.execution_condition,
            metadata: orig.metadata,
        };
        let orig_pos = self.get_node_pos(&selected_id);
        self.spec.graph.nodes.insert(idx + 1, new_node);
        self.node_positions
            .insert(new_id.clone(), orig_pos + Vec2::new(40.0, 40.0));
        self.selected_node_id = Some(new_id.clone());
        self.refresh_analyses();
        Some(new_id)
    }

    /// Deletes the currently selected node from the graph.
    pub fn delete_selected_node(&mut self) -> bool {
        let Some(selected_id) = self.selected_node_id.take() else {
            return false;
        };
        let before = self.spec.graph.nodes.len();
        self.spec.graph.nodes.retain(|n| n.id != selected_id);
        self.node_positions.remove(&selected_id);
        if self.spec.graph.nodes.len() != before {
            self.refresh_analyses();
            true
        } else {
            false
        }
    }

    /// Selects a node by ID.
    pub fn select_node(&mut self, node_id: Option<String>) {
        self.selected_node_id = node_id;
    }

    /// Retrieves position of a node on canvas.
    pub fn get_node_pos(&self, id: &str) -> Pos2 {
        self.node_positions
            .get(id)
            .cloned()
            .unwrap_or(Pos2::new(150.0, 150.0))
    }
}
