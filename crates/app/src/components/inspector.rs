//! Node and Model Inspector sidebar: editing layer parameters, 27-precision lattice, and optimizer rules.

use crate::{icons::bi, studio_state::StudioState};
use eframe::egui::{self, Color32, RichText, ScrollArea, Ui};
use shared::{
    CoordinateEncoding, HaltingGate, LossPairing, NormPlacement, NormalizerSpec, OperatorKind,
    PrecisionFormat, StructuralConstraint, WiringPattern,
};

const ALL_PRECISION_FORMATS: &[PrecisionFormat] = &[
    PrecisionFormat::BF16,
    PrecisionFormat::FP16,
    PrecisionFormat::FP32,
    PrecisionFormat::TF32,
    PrecisionFormat::FP64,
    PrecisionFormat::FP8E4M3FN,
    PrecisionFormat::FP8E5M2,
    PrecisionFormat::MXFP8,
    PrecisionFormat::MXFP6E3M2,
    PrecisionFormat::MXFP4E2M1,
    PrecisionFormat::FP4E2M1,
    PrecisionFormat::INT8Sym,
    PrecisionFormat::INT8Asym,
    PrecisionFormat::INT4Sym,
    PrecisionFormat::NF4,
    PrecisionFormat::SF4,
    PrecisionFormat::INT2,
    PrecisionFormat::BitNet158b,
    PrecisionFormat::BinaryXnor,
    PrecisionFormat::Posit8_0,
    PrecisionFormat::Posit8_1,
    PrecisionFormat::Posit16_1,
    PrecisionFormat::LNS8,
    PrecisionFormat::E8LatticeQTIP,
    PrecisionFormat::ZKMLFiniteFieldFp,
    PrecisionFormat::CKKSRingLWE,
    PrecisionFormat::IBPCROWNIntervalBounds,
];

pub struct InspectorView;

impl Default for InspectorView {
    fn default() -> Self {
        Self
    }
}

impl InspectorView {
    pub fn render(&mut self, ui: &mut Ui, state: &mut StudioState) {
        ScrollArea::vertical().show(ui, |ui| {
            ui.add_space(8.0);
            if let Some(ref selected_id) = state.selected_node_id.clone() {
                self.render_node_inspector(ui, state, selected_id);
            } else {
                self.render_model_inspector(ui, state);
            }
        });
    }

    fn render_node_inspector(&mut self, ui: &mut Ui, state: &mut StudioState, node_id: &str) {
        ui.horizontal(|ui| {
            ui.heading(
                RichText::new(format!("{} Layer Inspector", bi::SLIDERS))
                    .size(16.0)
                    .strong(),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .button(format!("{} Deselect", bi::X_LG))
                    .on_hover_text("Deselect layer and return to Blueprint & I/O Configuration")
                    .clicked()
                {
                    state.selected_node_id = None;
                }
            });
        });

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            if ui
                .small_button(format!("{} Move Up", bi::ARROW_UP))
                .on_hover_text("Move layer earlier in execution sequence")
                .clicked()
            {
                state.move_selected_node(-1);
            }
            if ui
                .small_button(format!("{} Move Down", bi::ARROW_DOWN))
                .on_hover_text("Move layer later in execution sequence")
                .clicked()
            {
                state.move_selected_node(1);
            }
            if ui
                .small_button(format!("{} Duplicate", bi::COPY))
                .on_hover_text("Clone this layer immediately after current position")
                .clicked()
            {
                state.duplicate_selected_node();
            }
        });

        ui.add_space(4.0);
        ui.separator();

        let mut should_delete = false;
        let mut should_refresh = false;
        let mut next_layer_to_append: Option<(&'static str, OperatorKind)> = None;

        if let Some(node) = state.spec.graph.find_node_mut(node_id) {
            let slug = node.operator.curriculum_slug();
            let url = node.operator.curriculum_url();
            let overview = node.operator.overview();
            let recommended_next = node.operator.recommended_next_layers();

            ui.horizontal_wrapped(|ui| {
                ui.label(
                    RichText::new(format!("{} Reference:", bi::BOOK))
                        .size(11.0)
                        .color(Color32::from_rgb(160, 185, 210)),
                );
                ui.hyperlink_to(
                    RichText::new(format!("{slug} {}", bi::BOX_ARROW_UP_RIGHT))
                        .size(10.5)
                        .monospace()
                        .color(Color32::from_rgb(110, 205, 255)),
                    &url,
                )
                .on_hover_text(format!("Open AI-Examples reference: {url}"));
            });
            ui.add_space(4.0);

            ui.label(
                RichText::new("Node Identifier")
                    .size(11.0)
                    .color(Color32::from_rgb(180, 180, 180)),
            );
            if ui.text_edit_singleline(&mut node.id).changed() {
                should_refresh = true;
            }

            ui.add_space(4.0);
            ui.label(
                RichText::new("Display Label")
                    .size(11.0)
                    .color(Color32::from_rgb(180, 180, 180)),
            );
            if ui.text_edit_singleline(&mut node.name).changed() {
                should_refresh = true;
            }

            ui.add_space(8.0);
            ui.separator();

            // Precision selector (Full 27-precision lattice)
            ui.label(
                RichText::new("Execution Precision (27-Format Lattice)")
                    .size(11.0)
                    .color(Color32::from_rgb(180, 180, 180)),
            );
            let current_prec = node.precision.unwrap_or(PrecisionFormat::BF16);
            egui::ComboBox::from_id_salt("node_prec_select")
                .selected_text(format!(
                    "{} ({}b)",
                    current_prec.label(),
                    current_prec.bits()
                ))
                .width(230.0)
                .show_ui(ui, |ui| {
                    for &fmt in ALL_PRECISION_FORMATS {
                        if ui
                            .selectable_label(
                                current_prec == fmt,
                                format!("{} ({} bits)", fmt.label(), fmt.bits()),
                            )
                            .clicked()
                        {
                            node.precision = Some(fmt);
                            should_refresh = true;
                        }
                    }
                });

            ui.add_space(10.0);
            ui.separator();

            // Modular AI Role & Conditional Execution
            ui.label(
                RichText::new("Modular AI Role & Conditional Execution")
                    .size(12.0)
                    .color(Color32::from_rgb(140, 210, 255))
                    .strong(),
            );
            ui.label(
                RichText::new("Configure compound system role and runtime dispatch (e.g. alternate conformers, speculative draft heads, or dynamic gates).")
                    .size(10.5)
                    .color(Color32::from_rgb(160, 175, 195)),
            );
            ui.add_space(2.0);

            // Modular role selector
            egui::ComboBox::from_id_salt(format!("{}_modular_role", node.id))
                .selected_text(format!("Role: {}", node.modular_role.label()))
                .width(230.0)
                .show_ui(ui, |ui| {
                    for &role in shared::spec::graph::ModularRole::ALL {
                        if ui
                            .selectable_label(node.modular_role == role, role.label())
                            .on_hover_text(role.description())
                            .clicked()
                        {
                            node.modular_role = role;
                            should_refresh = true;
                        }
                    }
                });

            // Execution condition selector
            let cond_label = match &node.execution_condition {
                shared::spec::graph::ExecutionCondition::AlwaysActive => {
                    "Always Active (100% of tokens/inputs)"
                }
                shared::spec::graph::ExecutionCondition::Conditional { .. } => {
                    "Conditional (Dynamic Gate / Modality)"
                }
                shared::spec::graph::ExecutionCondition::AlternateChoice { .. } => {
                    "Alternate Choice (Multi-Option Group)"
                }
                shared::spec::graph::ExecutionCondition::OptionalAdapter { .. } => {
                    "Optional Adapter (Toggleable Branch)"
                }
            };
            egui::ComboBox::from_id_salt(format!("{}_exec_cond", node.id))
                .selected_text(cond_label)
                .width(230.0)
                .show_ui(ui, |ui| {
                    if ui
                        .selectable_label(
                            matches!(
                                node.execution_condition,
                                shared::spec::graph::ExecutionCondition::AlwaysActive
                            ),
                            "Always Active",
                        )
                        .clicked()
                    {
                        node.execution_condition =
                            shared::spec::graph::ExecutionCondition::AlwaysActive;
                        should_refresh = true;
                    }
                    if ui
                        .selectable_label(
                            matches!(
                                node.execution_condition,
                                shared::spec::graph::ExecutionCondition::Conditional { .. }
                            ),
                            "Conditional (Dynamic Gate)",
                        )
                        .clicked()
                    {
                        node.execution_condition =
                            shared::spec::graph::ExecutionCondition::Conditional {
                                condition_tag: "audio_present".to_string(),
                                bypass_residual: true,
                            };
                        should_refresh = true;
                    }
                    if ui
                        .selectable_label(
                            matches!(
                                node.execution_condition,
                                shared::spec::graph::ExecutionCondition::AlternateChoice { .. }
                            ),
                            "Alternate Choice (e.g. Option A vs B)",
                        )
                        .clicked()
                    {
                        node.execution_condition =
                            shared::spec::graph::ExecutionCondition::AlternateChoice {
                                group_id: "input_conformer_group".to_string(),
                                option_name: "conformer_wav2vec".to_string(),
                                is_active: true,
                            };
                        should_refresh = true;
                    }
                    if ui
                        .selectable_label(
                            matches!(
                                node.execution_condition,
                                shared::spec::graph::ExecutionCondition::OptionalAdapter { .. }
                            ),
                            "Optional Adapter (e.g. Speculative Head)",
                        )
                        .clicked()
                    {
                        node.execution_condition =
                            shared::spec::graph::ExecutionCondition::OptionalAdapter {
                                adapter_tag: "speculative_draft_head".to_string(),
                                enabled_by_default: false,
                            };
                        should_refresh = true;
                    }
                });

            // Specific controls based on chosen condition
            match &mut node.execution_condition {
                shared::spec::graph::ExecutionCondition::AlwaysActive => {}
                shared::spec::graph::ExecutionCondition::Conditional {
                    condition_tag,
                    bypass_residual,
                } => {
                    ui.horizontal(|ui| {
                        ui.label("Condition Tag:");
                        if ui.text_edit_singleline(condition_tag).changed() {
                            should_refresh = true;
                        }
                    });
                    if ui
                        .checkbox(bypass_residual, "Bypass via Residual Connection")
                        .changed()
                    {
                        should_refresh = true;
                    }
                }
                shared::spec::graph::ExecutionCondition::AlternateChoice {
                    group_id,
                    option_name,
                    is_active,
                } => {
                    ui.horizontal(|ui| {
                        ui.label("Group ID:");
                        if ui.text_edit_singleline(group_id).changed() {
                            should_refresh = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Option Name:");
                        if ui.text_edit_singleline(option_name).changed() {
                            should_refresh = true;
                        }
                    });
                    if ui.checkbox(is_active, "Active Option in Graph").changed() {
                        should_refresh = true;
                    }
                }
                shared::spec::graph::ExecutionCondition::OptionalAdapter {
                    adapter_tag,
                    enabled_by_default,
                } => {
                    ui.horizontal(|ui| {
                        ui.label("Adapter Tag:");
                        if ui.text_edit_singleline(adapter_tag).changed() {
                            should_refresh = true;
                        }
                    });
                    if ui
                        .checkbox(enabled_by_default, "Enabled by Default")
                        .changed()
                    {
                        should_refresh = true;
                    }
                }
            }

            ui.add_space(10.0);
            ui.separator();
            ui.label(
                RichText::new("Operator Hyperparameters")
                    .size(12.0)
                    .strong(),
            );

            match &mut node.operator {
                OperatorKind::Linear {
                    in_features,
                    out_features,
                    bias,
                } => {
                    should_refresh |= drag_usize(ui, "In Features:", in_features, 1..=65536);
                    should_refresh |= drag_usize(ui, "Out Features:", out_features, 1..=65536);
                    if ui.checkbox(bias, "Enable Bias").changed() {
                        should_refresh = true;
                    }
                }

                OperatorKind::MHA {
                    num_heads,
                    d_k,
                    d_v,
                    dropout,
                } => {
                    should_refresh |= drag_usize(ui, "Num Heads:", num_heads, 1..=256);
                    should_refresh |= drag_usize(ui, "Key Dim (d_k):", d_k, 1..=1024);
                    should_refresh |= drag_usize(ui, "Value Dim (d_v):", d_v, 1..=1024);
                    if ui
                        .add(egui::Slider::new(dropout, 0.0..=0.5).text("Dropout"))
                        .changed()
                    {
                        should_refresh = true;
                    }
                }

                OperatorKind::GQA {
                    num_q_heads,
                    num_kv_heads,
                    d_k,
                } => {
                    should_refresh |= drag_usize(ui, "Query Heads:", num_q_heads, 1..=256);
                    should_refresh |= drag_usize(ui, "KV Heads:", num_kv_heads, 1..=256);
                    should_refresh |= drag_usize(ui, "Head Dim (d_k):", d_k, 8..=512);
                }

                OperatorKind::MLA {
                    num_heads,
                    d_latent_kv,
                    d_rope,
                    d_v,
                } => {
                    should_refresh |= drag_usize(ui, "Num Heads:", num_heads, 1..=256);
                    should_refresh |= drag_usize(ui, "KV Latent Dim:", d_latent_kv, 16..=4096);
                    should_refresh |= drag_usize(ui, "Decoupled RoPE Dim:", d_rope, 8..=512);
                    should_refresh |= drag_usize(ui, "Value Dim (d_v):", d_v, 8..=1024);
                }

                OperatorKind::DiffAttn { num_heads, d_k } => {
                    should_refresh |= drag_usize(ui, "Num Heads:", num_heads, 1..=256);
                    should_refresh |= drag_usize(ui, "Head Dim (d_k):", d_k, 8..=512);
                }

                OperatorKind::LinearAttention {
                    num_heads,
                    d_k,
                    feature_map,
                } => {
                    should_refresh |= drag_usize(ui, "Num Heads:", num_heads, 1..=256);
                    should_refresh |= drag_usize(ui, "Head Dim (d_k):", d_k, 8..=512);
                    ui.horizontal(|ui| {
                        ui.label("Kernel Map:");
                        if ui.text_edit_singleline(feature_map).changed() {
                            should_refresh = true;
                        }
                    });
                }

                OperatorKind::SlidingWindowSink {
                    window_size,
                    num_sink_tokens,
                } => {
                    should_refresh |= drag_usize(ui, "Window Size:", window_size, 16..=32768);
                    should_refresh |= drag_usize(ui, "Sink Tokens:", num_sink_tokens, 0..=64);
                }

                OperatorKind::EvoformerTriangle { d_pair } => {
                    should_refresh |= drag_usize(ui, "Pair Rep Dim:", d_pair, 16..=1024);
                }

                OperatorKind::InvariantPointAttention {
                    num_heads,
                    num_points,
                } => {
                    should_refresh |= drag_usize(ui, "Num Heads:", num_heads, 1..=64);
                    should_refresh |= drag_usize(ui, "Query/Value Points:", num_points, 1..=64);
                }

                OperatorKind::Mamba2SSD {
                    d_model,
                    d_state,
                    d_conv,
                    expand,
                }
                | OperatorKind::MambaS6 {
                    d_model,
                    d_state,
                    d_conv,
                    expand,
                } => {
                    should_refresh |= drag_usize(ui, "Model Dim (d_model):", d_model, 16..=16384);
                    should_refresh |= drag_usize(ui, "State Dim (d_state):", d_state, 4..=1024);
                    should_refresh |= drag_usize(ui, "Conv Kernel (d_conv):", d_conv, 1..=16);
                    should_refresh |= drag_usize(ui, "Expand Factor:", expand, 1..=8);
                }

                OperatorKind::S4D { d_model, d_state } | OperatorKind::LRU { d_model, d_state } => {
                    should_refresh |= drag_usize(ui, "Model Dim (d_model):", d_model, 16..=16384);
                    should_refresh |= drag_usize(ui, "State Dim (d_state):", d_state, 4..=1024);
                }

                OperatorKind::xLSTM {
                    d_model,
                    num_heads,
                    variant,
                } => {
                    should_refresh |= drag_usize(ui, "Model Dim (d_model):", d_model, 16..=16384);
                    should_refresh |= drag_usize(ui, "Memory Heads:", num_heads, 1..=64);
                    ui.horizontal(|ui| {
                        ui.label("Cell Variant:");
                        egui::ComboBox::from_id_salt("xlstm_variant")
                            .selected_text(variant.as_str())
                            .show_ui(ui, |ui| {
                                if ui
                                    .selectable_label(variant == "mLSTM", "mLSTM (Matrix Memory)")
                                    .clicked()
                                {
                                    *variant = "mLSTM".to_string();
                                    should_refresh = true;
                                }
                                if ui
                                    .selectable_label(
                                        variant == "sLSTM",
                                        "sLSTM (Scalar Stabilized)",
                                    )
                                    .clicked()
                                {
                                    *variant = "sLSTM".to_string();
                                    should_refresh = true;
                                }
                            });
                    });
                }

                OperatorKind::TitansMemory {
                    d_model,
                    memory_depth,
                } => {
                    should_refresh |= drag_usize(ui, "Model Dim:", d_model, 16..=16384);
                    should_refresh |= drag_usize(ui, "Memory MLP Depth:", memory_depth, 1..=16);
                }

                OperatorKind::EchoStateNetwork {
                    reservoir_size,
                    spectral_radius,
                } => {
                    should_refresh |= drag_usize(ui, "Reservoir Size:", reservoir_size, 32..=16384);
                    ui.horizontal(|ui| {
                        ui.label("Spectral Radius:");
                        if ui
                            .add(
                                egui::DragValue::new(spectral_radius)
                                    .speed(0.01)
                                    .range(0.1..=1.5),
                            )
                            .changed()
                        {
                            should_refresh = true;
                        }
                    });
                }

                OperatorKind::NextGenReservoir {
                    delay_taps,
                    poly_degree,
                } => {
                    should_refresh |= drag_usize(ui, "Delay Taps (k):", delay_taps, 1..=32);
                    should_refresh |= drag_usize(ui, "Polynomial Degree:", poly_degree, 1..=5);
                }

                OperatorKind::LISTA {
                    num_unrolled_steps,
                    lambda,
                } => {
                    should_refresh |= drag_usize(ui, "Unrolled Steps:", num_unrolled_steps, 1..=64);
                    ui.horizontal(|ui| {
                        ui.label("Sparsity Lambda:");
                        if ui.add(egui::DragValue::new(lambda).speed(0.005)).changed() {
                            should_refresh = true;
                        }
                    });
                }

                OperatorKind::ADMMNet { num_stages, rho } => {
                    should_refresh |= drag_usize(ui, "ADMM Stages:", num_stages, 1..=64);
                    ui.horizontal(|ui| {
                        ui.label("Penalty Rho:");
                        if ui.add(egui::DragValue::new(rho).speed(0.05)).changed() {
                            should_refresh = true;
                        }
                    });
                }

                OperatorKind::CRATE {
                    d_model,
                    num_heads,
                    lambda_step,
                } => {
                    should_refresh |= drag_usize(ui, "Model Dim:", d_model, 16..=8192);
                    should_refresh |= drag_usize(ui, "Subspace Heads:", num_heads, 1..=128);
                    ui.horizontal(|ui| {
                        ui.label("ISTA Step Lambda:");
                        if ui
                            .add(egui::DragValue::new(lambda_step).speed(0.01))
                            .changed()
                        {
                            should_refresh = true;
                        }
                    });
                }

                OperatorKind::DEQ {
                    d_model,
                    max_iters,
                    tolerance,
                } => {
                    should_refresh |= drag_usize(ui, "Model Dim:", d_model, 16..=8192);
                    should_refresh |= drag_usize(ui, "Anderson Max Iters:", max_iters, 2..=100);
                    ui.horizontal(|ui| {
                        ui.label("Fixed-Point Tol:");
                        if ui
                            .add(egui::DragValue::new(tolerance).speed(1e-5))
                            .changed()
                        {
                            should_refresh = true;
                        }
                    });
                }

                OperatorKind::TTT {
                    d_model,
                    mini_batch_size,
                } => {
                    should_refresh |= drag_usize(ui, "Model Dim:", d_model, 16..=8192);
                    should_refresh |= drag_usize(ui, "TTT Mini-Batch:", mini_batch_size, 1..=512);
                }

                OperatorKind::ModernHopfield { num_heads, beta } => {
                    should_refresh |= drag_usize(ui, "Num Heads:", num_heads, 1..=64);
                    ui.horizontal(|ui| {
                        ui.label("Inverse Temp Beta:");
                        if ui.add(egui::DragValue::new(beta).speed(0.1)).changed() {
                            should_refresh = true;
                        }
                    });
                }

                OperatorKind::ConvNeXtV2Block {
                    dim,
                    kernel_size,
                    mlp_ratio,
                } => {
                    should_refresh |= drag_usize(ui, "Channels (dim):", dim, 8..=4096);
                    should_refresh |= drag_usize(ui, "Kernel Size:", kernel_size, 3..=15);
                    ui.horizontal(|ui| {
                        ui.label("MLP Ratio:");
                        if ui
                            .add(egui::DragValue::new(mlp_ratio).speed(0.25).range(1.0..=8.0))
                            .changed()
                        {
                            should_refresh = true;
                        }
                    });
                }

                OperatorKind::WTConv {
                    in_channels,
                    out_channels,
                    wavelet_levels,
                } => {
                    should_refresh |= drag_usize(ui, "In Channels:", in_channels, 1..=4096);
                    should_refresh |= drag_usize(ui, "Out Channels:", out_channels, 1..=4096);
                    should_refresh |= drag_usize(ui, "Wavelet Levels:", wavelet_levels, 1..=6);
                }

                OperatorKind::FourierNeuralOperator { modes, width } => {
                    should_refresh |= drag_usize(ui, "Spectral Width:", width, 8..=1024);
                    if let Some(m0) = modes.first_mut() {
                        should_refresh |= drag_usize(ui, "Fourier Modes:", m0, 2..=128);
                    }
                }

                OperatorKind::DeepONet {
                    branch_dim,
                    trunk_dim,
                    basis_dim,
                } => {
                    should_refresh |= drag_usize(ui, "Branch Dim:", branch_dim, 8..=4096);
                    should_refresh |= drag_usize(ui, "Trunk Dim:", trunk_dim, 8..=4096);
                    should_refresh |= drag_usize(ui, "Basis Dim (p):", basis_dim, 8..=1024);
                }

                OperatorKind::EGNN { node_dim, edge_dim } => {
                    should_refresh |= drag_usize(ui, "Node Feature Dim:", node_dim, 8..=2048);
                    should_refresh |= drag_usize(ui, "Edge Feature Dim:", edge_dim, 4..=1024);
                }

                OperatorKind::CellularSheafConv {
                    node_dim,
                    stalk_dim,
                } => {
                    should_refresh |= drag_usize(ui, "Node Dim:", node_dim, 8..=2048);
                    should_refresh |= drag_usize(ui, "Stalk Dim (d):", stalk_dim, 2..=64);
                }

                OperatorKind::CliffordGeometricProduct { p, q } => {
                    should_refresh |= drag_usize(ui, "Positive Metric (p):", p, 0..=8);
                    should_refresh |= drag_usize(ui, "Negative Metric (q):", q, 0..=8);
                }

                OperatorKind::GaussianSplatting3D { num_gaussians } => {
                    should_refresh |=
                        drag_usize(ui, "3D Gaussians:", num_gaussians, 128..=1_000_000);
                }

                OperatorKind::Embedding {
                    num_embeddings,
                    embedding_dim,
                } => {
                    should_refresh |= drag_usize(ui, "Vocab Size:", num_embeddings, 16..=256_000);
                    should_refresh |= drag_usize(ui, "Embedding Dim:", embedding_dim, 16..=16384);
                }

                OperatorKind::KANLayer {
                    in_features,
                    out_features,
                    grid_size,
                    spline_order,
                } => {
                    should_refresh |= drag_usize(ui, "In Features:", in_features, 1..=4096);
                    should_refresh |= drag_usize(ui, "Out Features:", out_features, 1..=4096);
                    should_refresh |= drag_usize(ui, "Spline Grid (G):", grid_size, 2..=64);
                    should_refresh |= drag_usize(ui, "Spline Order (k):", spline_order, 1..=5);
                }

                OperatorKind::SwiGLU {
                    in_features,
                    hidden_features,
                }
                | OperatorKind::GeGLU {
                    in_features,
                    hidden_features,
                } => {
                    should_refresh |= drag_usize(ui, "In Features:", in_features, 8..=32768);
                    should_refresh |=
                        drag_usize(ui, "Hidden Features:", hidden_features, 8..=65536);
                }

                OperatorKind::Snake { frequency } => {
                    ui.horizontal(|ui| {
                        ui.label("Periodic Freq (a):");
                        if ui
                            .add(egui::DragValue::new(frequency).speed(0.1).range(0.1..=50.0))
                            .changed()
                        {
                            should_refresh = true;
                        }
                    });
                }

                OperatorKind::RMSNorm { dim, eps } | OperatorKind::LayerNorm { dim, eps } => {
                    should_refresh |= drag_usize(ui, "Dimension:", dim, 1..=32768);
                    ui.horizontal(|ui| {
                        ui.label("Epsilon:");
                        if ui.add(egui::DragValue::new(eps).speed(1e-6)).changed() {
                            should_refresh = true;
                        }
                    });
                }

                OperatorKind::QKNorm { d_head } => {
                    should_refresh |= drag_usize(ui, "Head Dim:", d_head, 8..=512);
                }

                OperatorKind::LogitSoftCap { cap_val } => {
                    ui.horizontal(|ui| {
                        ui.label("Soft-Cap Ceiling:");
                        if ui
                            .add(egui::DragValue::new(cap_val).speed(1.0).range(5.0..=100.0))
                            .changed()
                        {
                            should_refresh = true;
                        }
                    });
                }

                OperatorKind::RevINLayer { channels } => {
                    should_refresh |= drag_usize(ui, "Channels:", channels, 1..=4096);
                }

                OperatorKind::DeepSeekMoE {
                    total_experts,
                    routed_k,
                    shared_experts,
                    d_model,
                    d_ffn,
                    aux_loss_free_bias,
                } => {
                    should_refresh |= drag_usize(ui, "Total Experts:", total_experts, 2..=512);
                    should_refresh |= drag_usize(ui, "Routed Top-k:", routed_k, 1..=32);
                    should_refresh |= drag_usize(ui, "Shared Experts:", shared_experts, 0..=16);
                    should_refresh |= drag_usize(ui, "Model Dim (d_model):", d_model, 16..=16384);
                    should_refresh |= drag_usize(ui, "Expert FFN Dim:", d_ffn, 16..=32768);
                    if ui
                        .checkbox(
                            aux_loss_free_bias,
                            "Aux-Loss-Free Dynamic Bias Load Balancing",
                        )
                        .changed()
                    {
                        should_refresh = true;
                    }
                }

                OperatorKind::AdaLNZeroDiT {
                    d_model,
                    num_heads,
                    cond_dim,
                    dual_stream_mmdit,
                } => {
                    should_refresh |= drag_usize(ui, "Model Dim (d_model):", d_model, 16..=16384);
                    should_refresh |= drag_usize(ui, "Attention Heads:", num_heads, 1..=128);
                    should_refresh |= drag_usize(ui, "Conditioning Dim:", cond_dim, 16..=8192);
                    if ui
                        .checkbox(dual_stream_mmdit, "Dual-Stream MMDiT Joint Text-Image Attn")
                        .changed()
                    {
                        should_refresh = true;
                    }
                }

                OperatorKind::MultiTokenPrediction {
                    d_model,
                    vocab_size,
                    num_predict_tokens,
                } => {
                    should_refresh |= drag_usize(ui, "Model Dim (d_model):", d_model, 16..=16384);
                    should_refresh |= drag_usize(ui, "Vocab Size:", vocab_size, 16..=256_000);
                    should_refresh |=
                        drag_usize(ui, "MTP Predict Depth (k):", num_predict_tokens, 1..=8);
                }

                OperatorKind::SpikingLIF {
                    d_model,
                    tau_mem,
                    v_threshold,
                    surrogate_slope,
                } => {
                    should_refresh |= drag_usize(ui, "Membrane Neurons:", d_model, 8..=16384);
                    ui.horizontal(|ui| {
                        ui.label("Membrane Tau (ms):");
                        if ui
                            .add(egui::DragValue::new(tau_mem).speed(0.5).range(1.0..=200.0))
                            .changed()
                        {
                            should_refresh = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Firing Threshold V_th:");
                        if ui
                            .add(
                                egui::DragValue::new(v_threshold)
                                    .speed(0.05)
                                    .range(0.1..=5.0),
                            )
                            .changed()
                        {
                            should_refresh = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Surrogate Slope:");
                        if ui
                            .add(
                                egui::DragValue::new(surrogate_slope)
                                    .speed(1.0)
                                    .range(1.0..=100.0),
                            )
                            .changed()
                        {
                            should_refresh = true;
                        }
                    });
                }

                OperatorKind::LiquidNeuralCfC {
                    d_model,
                    backbone_units,
                } => {
                    should_refresh |= drag_usize(ui, "State Dim (d_model):", d_model, 8..=8192);
                    should_refresh |=
                        drag_usize(ui, "Liquid Backbone Units:", backbone_units, 8..=8192);
                }

                OperatorKind::HyperbolicPoincareLayer { dim, curvature_c } => {
                    should_refresh |= drag_usize(ui, "Manifold Dim:", dim, 4..=8192);
                    ui.horizontal(|ui| {
                        ui.label("Curvature (-c):");
                        if ui
                            .add(
                                egui::DragValue::new(curvature_c)
                                    .speed(0.05)
                                    .range(0.01..=10.0),
                            )
                            .changed()
                        {
                            should_refresh = true;
                        }
                    });
                }

                OperatorKind::JEPAPredictor {
                    d_model,
                    predictor_dim,
                    num_heads,
                    ema_momentum,
                } => {
                    should_refresh |= drag_usize(ui, "Encoder Dim (d_model):", d_model, 32..=16384);
                    should_refresh |= drag_usize(ui, "Predictor Dim:", predictor_dim, 16..=8192);
                    should_refresh |= drag_usize(ui, "Attention Heads:", num_heads, 1..=128);
                    ui.horizontal(|ui| {
                        ui.label("EMA Target Momentum:");
                        if ui
                            .add(
                                egui::DragValue::new(ema_momentum)
                                    .speed(0.0005)
                                    .range(0.90..=0.9999),
                            )
                            .changed()
                        {
                            should_refresh = true;
                        }
                    });
                }

                OperatorKind::PairformerBlock {
                    d_single,
                    d_pair,
                    num_heads,
                } => {
                    should_refresh |= drag_usize(ui, "Single Rep Dim (s_i):", d_single, 16..=4096);
                    should_refresh |= drag_usize(ui, "Pair Rep Dim (z_ij):", d_pair, 16..=1024);
                    should_refresh |= drag_usize(ui, "Attention Heads:", num_heads, 1..=64);
                }

                OperatorKind::WorldModelRSSM {
                    d_model,
                    stochastic_classes,
                    stochastic_dim,
                } => {
                    should_refresh |=
                        drag_usize(ui, "Deterministic Dim (h_t):", d_model, 32..=8192);
                    should_refresh |=
                        drag_usize(ui, "Categorical Variables:", stochastic_classes, 4..=128);
                    should_refresh |=
                        drag_usize(ui, "Classes per Variable:", stochastic_dim, 4..=128);
                }

                OperatorKind::RWKV7DeltaRule { d_model, num_heads } => {
                    should_refresh |= drag_usize(ui, "State Dim (d_model):", d_model, 32..=16384);
                    should_refresh |= drag_usize(ui, "State Heads:", num_heads, 1..=128);
                }

                OperatorKind::EntropyBytePatcher {
                    d_byte,
                    d_latent,
                    entropy_threshold,
                } => {
                    should_refresh |= drag_usize(ui, "Local Byte Dim:", d_byte, 16..=2048);
                    should_refresh |= drag_usize(ui, "Latent Patch Dim:", d_latent, 64..=16384);
                    ui.horizontal(|ui| {
                        ui.label("Entropy Threshold (bits):");
                        if ui
                            .add(
                                egui::DragValue::new(entropy_threshold)
                                    .speed(0.05)
                                    .range(0.1..=8.0),
                            )
                            .changed()
                        {
                            should_refresh = true;
                        }
                    });
                }

                OperatorKind::ColPaliMaxSim { d_model, proj_dim } => {
                    should_refresh |= drag_usize(ui, "VLM Backbone Dim:", d_model, 64..=16384);
                    should_refresh |= drag_usize(ui, "Late-Interaction Dim:", proj_dim, 16..=1024);
                }

                OperatorKind::SphericalMeshGNN {
                    d_node,
                    d_edge,
                    mesh_refinement_level,
                } => {
                    should_refresh |= drag_usize(ui, "Mesh Node Dim:", d_node, 16..=4096);
                    should_refresh |= drag_usize(ui, "Mesh Edge Dim:", d_edge, 16..=4096);
                    should_refresh |=
                        drag_usize(ui, "Icosahedral Level (R):", mesh_refinement_level, 1..=8);
                }

                OperatorKind::QuantumVariationalCircuit {
                    num_qubits,
                    circuit_depth,
                    entanglement,
                } => {
                    should_refresh |= drag_usize(ui, "Qubit Register (N):", num_qubits, 2..=64);
                    should_refresh |= drag_usize(ui, "Ansatz Depth (L):", circuit_depth, 1..=64);
                    ui.horizontal(|ui| {
                        ui.label("Entanglement Topology:");
                        if ui.text_edit_singleline(entanglement).changed() {
                            should_refresh = true;
                        }
                    });
                }

                OperatorKind::PhotonicMZIMesh {
                    num_modes,
                    adc_bits,
                } => {
                    should_refresh |=
                        drag_usize(ui, "Optical Waveguide Modes:", num_modes, 4..=4096);
                    should_refresh |= drag_usize(ui, "ADC/DAC Quant Bits:", adc_bits, 2..=16);
                }

                OperatorKind::SimplicialHodgeConv {
                    d_node,
                    d_edge,
                    d_face,
                } => {
                    should_refresh |= drag_usize(ui, "0-Simplex Node Dim:", d_node, 8..=4096);
                    should_refresh |= drag_usize(ui, "1-Simplex Edge Dim:", d_edge, 8..=4096);
                    should_refresh |= drag_usize(ui, "2-Simplex Face Dim:", d_face, 8..=4096);
                }

                OperatorKind::ContinuousNormalizingFlow {
                    dim,
                    hutchinson_samples,
                } => {
                    should_refresh |= drag_usize(ui, "Flow Manifold Dim:", dim, 4..=8192);
                    should_refresh |=
                        drag_usize(ui, "Hutchinson Trace Samples:", hutchinson_samples, 1..=64);
                }

                OperatorKind::EnergyLangevinMCMC {
                    dim,
                    langevin_steps,
                    step_size,
                } => {
                    should_refresh |= drag_usize(ui, "Energy Latent Dim:", dim, 8..=8192);
                    should_refresh |=
                        drag_usize(ui, "Langevin MCMC Steps:", langevin_steps, 1..=64);
                    ui.horizontal(|ui| {
                        ui.label("Langevin Step Size:");
                        if ui
                            .add(
                                egui::DragValue::new(step_size)
                                    .speed(0.005)
                                    .range(0.001..=1.0),
                            )
                            .changed()
                        {
                            should_refresh = true;
                        }
                    });
                }

                OperatorKind::NeurosymbolicLogicDNC {
                    d_model,
                    memory_slots,
                    t_norm,
                } => {
                    should_refresh |= drag_usize(ui, "Controller Dim:", d_model, 16..=8192);
                    should_refresh |=
                        drag_usize(ui, "DNC Memory Slots (N):", memory_slots, 4..=4096);
                    ui.horizontal(|ui| {
                        ui.label("Fuzzy Logic t-Norm:");
                        if ui.text_edit_singleline(t_norm).changed() {
                            should_refresh = true;
                        }
                    });
                }

                OperatorKind::MultiAgentQMIX {
                    num_agents,
                    state_dim,
                    mixing_embed_dim,
                } => {
                    should_refresh |= drag_usize(ui, "Cooperative Agents:", num_agents, 2..=256);
                    should_refresh |= drag_usize(ui, "Global State Dim:", state_dim, 8..=4096);
                    should_refresh |=
                        drag_usize(ui, "Monotonic Mixer Dim:", mixing_embed_dim, 8..=2048);
                }

                OperatorKind::BioOrganoidActiveInference {
                    num_channels,
                    free_energy_iters,
                } => {
                    should_refresh |=
                        drag_usize(ui, "HD-MEA / Skyrmion Channels:", num_channels, 8..=16384);
                    should_refresh |= drag_usize(
                        ui,
                        "Variational Free-Energy Iters:",
                        free_energy_iters,
                        1..=64,
                    );
                }

                OperatorKind::AttentionFreeGatedMLP {
                    d_model,
                    seq_len,
                    filter_order,
                } => {
                    should_refresh |= drag_usize(ui, "Model Dim (d_model):", d_model, 16..=16384);
                    should_refresh |= drag_usize(ui, "Spatial/Seq Length:", seq_len, 16..=131072);
                    should_refresh |=
                        drag_usize(ui, "Hyena / SGU Filter Order:", filter_order, 1..=16);
                }

                OperatorKind::RawBitByteMultiScale {
                    d_global,
                    d_local,
                    patch_bytes,
                    bit_plane_decomposition,
                } => {
                    should_refresh |=
                        drag_usize(ui, "Global Patch Dim (d_global):", d_global, 64..=16384);
                    should_refresh |=
                        drag_usize(ui, "Local Byte/Bit Dim (d_local):", d_local, 32..=8192);
                    should_refresh |= drag_usize(ui, "Bytes per Patch (P):", patch_bytes, 1..=64);
                    if ui
                        .checkbox(
                            bit_plane_decomposition,
                            "8-Bit Binary Bit-Plane Decomposition",
                        )
                        .on_hover_text(
                            "Decomposes each byte into its 8 raw binary bits {0, 1}^8 prior to local/global projection.",
                        )
                        .changed()
                    {
                        should_refresh = true;
                    }
                }

                OperatorKind::VoiceCloningCodecTTS {
                    d_model,
                    num_rvq_codebooks,
                    speaker_embed_dim,
                } => {
                    should_refresh |=
                        drag_usize(ui, "Acoustic Flow Dim (d_model):", d_model, 64..=8192);
                    should_refresh |=
                        drag_usize(ui, "RVQ Quantizer Codebooks:", num_rvq_codebooks, 1..=32);
                    should_refresh |= drag_usize(
                        ui,
                        "Speaker Timbre Prompt Dim:",
                        speaker_embed_dim,
                        32..=2048,
                    );
                }

                OperatorKind::TabularGBDTVimeSSL {
                    num_trees,
                    tree_depth,
                    vime_corruption_rate,
                } => {
                    should_refresh |= drag_usize(ui, "Oblivious Trees (T):", num_trees, 4..=4096);
                    should_refresh |= drag_usize(ui, "Tree Split Depth (D):", tree_depth, 2..=12);
                    ui.horizontal(|ui| {
                        ui.label("VIME Mask Corruption (p):")
                            .on_hover_text("Probability of masking tabular features and replacing with marginal draws during VIME self-supervised training.");
                        if ui
                            .add(egui::Slider::new(vime_corruption_rate, 0.05..=0.80))
                            .changed()
                        {
                            should_refresh = true;
                        }
                    });
                }

                OperatorKind::SuperSmoothLipschitzLayer {
                    dim,
                    bjorck_iters,
                    sobolev_order,
                    spectral_bound_k,
                } => {
                    should_refresh |= drag_usize(ui, "Manifold Dim (dim):", dim, 8..=16384);
                    should_refresh |= drag_usize(ui, "Björck Ortho Iters:", bjorck_iters, 1..=25);
                    should_refresh |=
                        drag_usize(ui, "Sobolev Order (W^{k,2}):", sobolev_order, 1..=6);
                    ui.horizontal(|ui| {
                        ui.label("Lipschitz Bound (K):")
                            .on_hover_text("Certified upper bound K on the layer's spectral norm ||W||_2 <= K via Björck-Cayley projection.");
                        if ui
                            .add(
                                egui::DragValue::new(spectral_bound_k)
                                    .speed(0.05)
                                    .range(0.05..=20.0),
                            )
                            .changed()
                        {
                            should_refresh = true;
                        }
                    });
                }

                OperatorKind::LearnableQuantizerQAT {
                    dim,
                    init_bits,
                    min_bits,
                    max_bits,
                    learn_step_size_and_clip,
                } => {
                    should_refresh |= drag_usize(ui, "Channel Dim:", dim, 8..=32768);
                    ui.horizontal(|ui| {
                        ui.label("Learnable Bitwidth (b_l):")
                            .on_hover_text("Continuous differentiable bitwidth param trained via FracBits / Mixed-Bitwidth QAT.");
                        if ui
                            .add(egui::Slider::new(init_bits, 1.0..=16.0).text("bits"))
                            .changed()
                        {
                            should_refresh = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Min Bitwidth Bound:");
                        if ui
                            .add(egui::DragValue::new(min_bits).speed(0.1).range(1.0..=8.0))
                            .changed()
                        {
                            should_refresh = true;
                        }
                        ui.label("Max:");
                        if ui
                            .add(egui::DragValue::new(max_bits).speed(0.5).range(2.0..=16.0))
                            .changed()
                        {
                            should_refresh = true;
                        }
                    });
                    if ui
                        .checkbox(
                            learn_step_size_and_clip,
                            "Learn Step-Size (LSQ+ s_l) & Activation Clipping (PACT alpha_l)",
                        )
                        .on_hover_text("Jointly trains quantization step-size s_l with 1/sqrt(N*Q_p) gradient scaling and PACT activation upper bound alpha_l.")
                        .changed()
                    {
                        should_refresh = true;
                    }
                }

                OperatorKind::SlotAttention {
                    num_slots,
                    slot_dim,
                    iters,
                } => {
                    should_refresh |= drag_usize(ui, "Object Slots (K):", num_slots, 2..=64);
                    should_refresh |= drag_usize(ui, "Slot Dim:", slot_dim, 16..=1024);
                    should_refresh |= drag_usize(ui, "Routing Iters:", iters, 1..=10);
                }

                OperatorKind::HardConcreteL0 { dim } => {
                    should_refresh |= drag_usize(ui, "Gate Dim:", dim, 1..=16384);
                }

                OperatorKind::SoftSortPermutahedron { num_items } => {
                    should_refresh |= drag_usize(ui, "Ranked Items:", num_items, 2..=4096);
                }

                OperatorKind::MixtureOfDepths { capacity_factor } => {
                    ui.horizontal(|ui| {
                        ui.label("Token Capacity:");
                        if ui
                            .add(egui::Slider::new(capacity_factor, 0.05..=1.0))
                            .changed()
                        {
                            should_refresh = true;
                        }
                    });
                }

                OperatorKind::Activation { kind } => {
                    ui.label(
                        RichText::new("Activation Function:")
                            .size(11.0)
                            .color(Color32::from_rgb(180, 180, 180)),
                    );
                    egui::ComboBox::from_id_salt(format!("{}_act_kind", node.id))
                        .selected_text(kind.label())
                        .width(220.0)
                        .show_ui(ui, |ui| {
                            for &k in shared::spec::operators::ActivationKind::ALL {
                                if ui
                                    .selectable_label(*kind == k, k.label())
                                    .on_hover_text(format!(
                                        "Formula: {}\nUsage: {}",
                                        k.formula(),
                                        k.typical_usage()
                                    ))
                                    .clicked()
                                {
                                    *kind = k;
                                    should_refresh = true;
                                }
                            }
                        });
                    ui.add_space(4.0);
                    egui::Frame::group(ui.style())
                        .fill(Color32::from_rgb(18, 22, 32))
                        .stroke(egui::Stroke::new(1.0, Color32::from_rgb(45, 60, 90)))
                        .corner_radius(4.0)
                        .inner_margin(egui::Margin::symmetric(8, 6))
                        .show(ui, |ui| {
                            ui.label(
                                RichText::new(format!("Formula: {}", kind.formula()))
                                    .monospace()
                                    .size(11.0)
                                    .color(Color32::from_rgb(140, 205, 255)),
                            );
                            ui.add_space(2.0);
                            ui.label(
                                RichText::new(kind.typical_usage())
                                    .size(10.5)
                                    .color(Color32::from_rgb(175, 195, 220)),
                            );
                        });
                }

                OperatorKind::SparsemaxGate | OperatorKind::Entmax15Gate => {
                    ui.label("Exact-zero Fenchel-Young simplex projection (parameter-free).");
                }
            }

            ui.add_space(12.0);
            ui.separator();
            ui.add_space(4.0);

            // Node Architectural Overview & Typical Composition Guide at the bottom of Layer Inspector
            egui::Frame::group(ui.style())
                .fill(Color32::from_rgb(22, 27, 38))
                .stroke(egui::Stroke::new(1.0, Color32::from_rgb(55, 75, 110)))
                .corner_radius(6.0)
                .inner_margin(egui::Margin::symmetric(10, 8))
                .show(ui, |ui| {
                    ui.label(
                        RichText::new(format!(
                            "{} Node Overview & Architecture Guide",
                            bi::INFO_CIRCLE_FILL
                        ))
                        .size(12.5)
                        .color(Color32::from_rgb(110, 200, 255))
                        .strong(),
                    );
                    ui.add_space(2.0);
                    ui.label(
                        RichText::new(format!(
                            "Category: {} • {}",
                            overview.category, overview.complexity_tag
                        ))
                        .size(10.5)
                        .color(Color32::from_rgb(150, 185, 220)),
                    );

                    ui.add_space(6.0);
                    ui.label(
                        RichText::new("What it does:")
                            .size(11.0)
                            .color(Color32::from_rgb(210, 220, 235))
                            .strong(),
                    );
                    ui.label(
                        RichText::new(overview.what_it_does)
                            .size(11.0)
                            .color(Color32::from_rgb(180, 190, 205)),
                    );

                    ui.add_space(6.0);
                    ui.label(
                        RichText::new("What it's usually used for:")
                            .size(11.0)
                            .color(Color32::from_rgb(210, 220, 235))
                            .strong(),
                    );
                    ui.label(
                        RichText::new(overview.typical_usage)
                            .size(11.0)
                            .color(Color32::from_rgb(180, 190, 205)),
                    );

                    ui.add_space(6.0);
                    ui.label(
                        RichText::new("What usually follows it:")
                            .size(11.0)
                            .color(Color32::from_rgb(130, 230, 165))
                            .strong(),
                    );
                    ui.label(
                        RichText::new(overview.typically_followed_by)
                            .size(11.0)
                            .color(Color32::from_rgb(180, 205, 190)),
                    );

                    if !recommended_next.is_empty() {
                        ui.add_space(6.0);
                        ui.label(
                            RichText::new("Quick-Append Next Layer:")
                                .size(10.5)
                                .color(Color32::from_rgb(160, 175, 200)),
                        );
                        ui.horizontal_wrapped(|ui| {
                            for (next_name, next_op) in recommended_next {
                                if ui
                                    .small_button(
                                        RichText::new(format!("{} {next_name}", bi::PLUS_LG))
                                            .color(Color32::from_rgb(120, 225, 155)),
                                    )
                                    .on_hover_text(format!(
                                        "Insert '{next_name}' immediately after this layer"
                                    ))
                                    .clicked()
                                {
                                    next_layer_to_append = Some((next_name, next_op));
                                }
                            }
                        });
                    }

                    ui.add_space(6.0);
                    ui.hyperlink_to(
                        RichText::new(format!(
                            "{} Open Curriculum Topic ({}) {}",
                            bi::JOURNAL_CODE,
                            slug,
                            bi::BOX_ARROW_UP_RIGHT
                        ))
                        .size(10.5)
                        .color(Color32::from_rgb(110, 200, 255)),
                        &url,
                    );
                });

            ui.add_space(12.0);
            ui.separator();
            if ui
                .button(
                    RichText::new(format!("{} Delete Layer", bi::TRASH3))
                        .color(Color32::from_rgb(240, 100, 100)),
                )
                .clicked()
            {
                should_delete = true;
            }
        }

        if let Some((next_name, next_op)) = next_layer_to_append {
            if let Some(pos_idx) = state.spec.graph.nodes.iter().position(|n| n.id == node_id) {
                let mut suffix = state.spec.graph.nodes.len() + 1;
                let base_slug = next_name
                    .to_lowercase()
                    .replace([' ', '-'], "_")
                    .replace(['(', ')'], "");
                let mut new_id = format!("{base_slug}_{suffix}");
                while state.spec.graph.nodes.iter().any(|n| n.id == new_id) {
                    suffix += 1;
                    new_id = format!("{base_slug}_{suffix}");
                }
                state.spec.graph.nodes.insert(
                    pos_idx + 1,
                    shared::NodeSpec {
                        id: new_id.clone(),
                        name: next_name.to_string(),
                        operator: next_op,
                        precision: Some(PrecisionFormat::BF16),
                        ..Default::default()
                    },
                );
                state.auto_layout_nodes();
                state.select_node(Some(new_id));
                state.refresh_analyses();
            }
        } else if should_delete {
            state.spec.graph.nodes.retain(|n| n.id != node_id);
            state.node_positions.remove(node_id);
            state.selected_node_id = None;
            state.auto_layout_nodes();
            state.refresh_analyses();
        } else if should_refresh {
            state.refresh_analyses();
        }
    }

    fn render_model_inspector(&mut self, ui: &mut Ui, state: &mut StudioState) {
        ui.heading(
            RichText::new(format!("{} Blueprint & I/O Inspector", bi::SLIDERS))
                .size(16.0)
                .strong(),
        );
        ui.add_space(8.0);
        ui.separator();

        ui.label(
            RichText::new("Model Identifier")
                .size(11.0)
                .color(Color32::from_rgb(180, 180, 180)),
        );
        ui.text_edit_singleline(&mut state.spec.metadata.model_id);

        ui.add_space(4.0);
        ui.label(
            RichText::new("Display Title")
                .size(11.0)
                .color(Color32::from_rgb(180, 180, 180)),
        );
        ui.text_edit_singleline(&mut state.spec.metadata.display_name);

        ui.add_space(4.0);
        ui.label(
            RichText::new("Application Domain")
                .size(11.0)
                .color(Color32::from_rgb(180, 180, 180)),
        );
        ui.text_edit_singleline(&mut state.spec.metadata.domain);

        ui.add_space(8.0);
        ui.separator();
        ui.horizontal(|ui| {
            ui.label(RichText::new("Target Hardware Profile").size(12.0).strong());
            if ui.button(format!("{} Auto-Detect Host", bi::CPU)).clicked() {
                let report = shared::detect_host_hardware();
                state.spec.metadata.target_hardware = report.recommended_profile;
                state.refresh_analyses();
            }
        });

        let cur_hw = state.spec.metadata.target_hardware.profile_name.clone();
        let mut new_hw = None;
        egui::ComboBox::from_id_salt("hw_profile_select")
            .selected_text(&cur_hw)
            .show_ui(ui, |ui| {
                if ui
                    .selectable_label(cur_hw.starts_with("Host_"), "Host Hardware (Auto-Detected)")
                    .clicked()
                {
                    let report = shared::detect_host_hardware();
                    new_hw = Some(report.recommended_profile);
                }
                if ui
                    .selectable_label(cur_hw == "ConsumerGPU_RTX4090", "NVIDIA RTX 4090 (24 GB)")
                    .clicked()
                {
                    new_hw = Some(shared::spec::HardwareTargetProfile::rtx_4090());
                }
                if ui
                    .selectable_label(cur_hw == "DataCenter_H100_SXM5", "NVIDIA H100 SXM (80 GB)")
                    .clicked()
                {
                    new_hw = Some(shared::spec::HardwareTargetProfile::h100());
                }
                if ui
                    .selectable_label(
                        cur_hw == "EdgeRobotics_JetsonOrin",
                        "NVIDIA Jetson Orin (64 GB)",
                    )
                    .clicked()
                {
                    new_hw = Some(shared::spec::HardwareTargetProfile::jetson_orin());
                }
                if ui
                    .selectable_label(cur_hw == "EdgeMCU_CortexM55", "ARM Cortex-M55 (Edge MCU)")
                    .clicked()
                {
                    new_hw = Some(shared::spec::HardwareTargetProfile::cortex_m55());
                }
            });
        if let Some(hw) = new_hw {
            state.spec.metadata.target_hardware = hw;
            state.refresh_analyses();
        }

        ui.add_space(8.0);
        ui.separator();
        ui.horizontal(|ui| {
            ui.label(
                RichText::new("Optimization & Partitioning")
                    .size(12.0)
                    .strong(),
            );
            if ui
                .small_button(
                    RichText::new(format!("{} Plan Training", bi::SPEEDOMETER2))
                        .color(Color32::from_rgb(120, 220, 160)),
                )
                .on_hover_text(
                    "Open the full interactive Training Plan Designer (Multi-Stage Curriculum, LR Schedule, 4D Parallelism & Chinchilla Budget)",
                )
                .clicked()
            {
                state.canvas_mode = crate::studio_state::CanvasMode::TrainingPlanner;
            }
        });
        // Optimizer Family selection
        ui.horizontal(|ui| {
            ui.label("Optimizer Family:");
            egui::ComboBox::from_id_salt("inspector_opt_family_select")
                .selected_text(state.spec.training.optimizer_family.label())
                .width(220.0)
                .show_ui(ui, |ui| {
                    for &fam in shared::spec::training::OptimizerFamily::ALL {
                        if ui
                            .selectable_label(
                                state.spec.training.optimizer_family == fam,
                                fam.label(),
                            )
                            .on_hover_text(fam.description())
                            .clicked()
                        {
                            state.spec.training.optimizer_family = fam;
                            state.refresh_analyses();
                        }
                    }
                });
        });

        match state.spec.training.optimizer_family {
            shared::spec::training::OptimizerFamily::MuonAdamWPartitioned => {
                if ui
                    .checkbox(
                        &mut state.spec.training.auto_partition_parameters,
                        "Auto-Partition Muon 2D vs AdamW",
                    )
                    .changed()
                {
                    state.refresh_analyses();
                }
                ui.horizontal(|ui| {
                    ui.label("Muon LR:")
                        .on_hover_text("Learning rate for Muon 5-step Newton-Schulz orthogonalized 2D GEMMs (typically 0.02).");
                    ui.add(egui::DragValue::new(&mut state.spec.training.muon.lr).speed(0.001));
                });
                ui.horizontal(|ui| {
                    ui.label("AdamW LR:").on_hover_text(
                        "Learning rate for embeddings, 1D norms/biases, and scalar parameters.",
                    );
                    ui.add(egui::DragValue::new(&mut state.spec.training.adamw.lr).speed(0.0001));
                });
            }
            shared::spec::training::OptimizerFamily::AdamWStandard => {
                ui.horizontal(|ui| {
                    ui.label("Learning Rate:");
                    ui.add(egui::DragValue::new(&mut state.spec.training.adamw.lr).speed(0.0001));
                });
                ui.horizontal(|ui| {
                    ui.label("Weight Decay:");
                    ui.add(
                        egui::DragValue::new(&mut state.spec.training.adamw.weight_decay)
                            .speed(0.001),
                    );
                });
            }
            shared::spec::training::OptimizerFamily::LionSignMomentum => {
                ui.horizontal(|ui| {
                    ui.label("Lion LR:")
                        .on_hover_text("EvoLved Sign Momentum learning rate (typically 3-10x smaller than AdamW, e.g. 1e-4).");
                    ui.add(egui::DragValue::new(&mut state.spec.training.lion.lr).speed(0.00005));
                });
                ui.horizontal(|ui| {
                    ui.label("Weight Decay:");
                    ui.add(
                        egui::DragValue::new(&mut state.spec.training.lion.weight_decay)
                            .speed(0.01),
                    );
                });
            }
            shared::spec::training::OptimizerFamily::SophiaSecondOrder => {
                ui.horizontal(|ui| {
                    ui.label("Sophia LR:").on_hover_text(
                        "Second-order diagonal Hessian clipped descent learning rate.",
                    );
                    ui.add(egui::DragValue::new(&mut state.spec.training.sophia.lr).speed(0.0001));
                });
                ui.horizontal(|ui| {
                    ui.label("Hessian Clip (rho):").on_hover_text(
                        "Hessian curvature clipping threshold rho (typically 0.04).",
                    );
                    ui.add(egui::DragValue::new(&mut state.spec.training.sophia.rho).speed(0.005));
                });
            }
            shared::spec::training::OptimizerFamily::LBFGSQuasiNewton => {
                ui.horizontal(|ui| {
                    ui.label("Max Iterations:").on_hover_text(
                        "Number of quasi-Newton curvature steps per optimizer step.",
                    );
                    ui.add(
                        egui::DragValue::new(&mut state.spec.training.lbfgs.max_iter)
                            .range(1..=100),
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("History Size:").on_hover_text(
                        "Number of previous vector pairs stored for inverse Hessian approximation.",
                    );
                    ui.add(
                        egui::DragValue::new(&mut state.spec.training.lbfgs.history_size)
                            .range(5..=100),
                    );
                });
            }
            shared::spec::training::OptimizerFamily::ScheduleFreeAdamW => {
                ui.horizontal(|ui| {
                    ui.label("Base LR:");
                    ui.add(egui::DragValue::new(&mut state.spec.training.adamw.lr).speed(0.0001));
                });
                ui.horizontal(|ui| {
                    ui.label("Weight Decay:");
                    ui.add(
                        egui::DragValue::new(&mut state.spec.training.adamw.weight_decay)
                            .speed(0.001),
                    );
                });
            }
            shared::spec::training::OptimizerFamily::PagedAdamW8Bit => {
                ui.label(
                    RichText::new("Paged 8-bit AdamW (CUDA Unified Memory / bitsandbytes)")
                        .size(11.0)
                        .color(Color32::from_rgb(120, 220, 160)),
                );
                ui.horizontal(|ui| {
                    ui.label("Base LR:");
                    ui.add(egui::DragValue::new(&mut state.spec.training.adamw.lr).speed(0.0001));
                });
            }
            _ => {
                ui.horizontal(|ui| {
                    ui.label("Base LR:");
                    ui.add(egui::DragValue::new(&mut state.spec.training.adamw.lr).speed(0.0001));
                });
            }
        }

        ui.add_space(8.0);
        ui.label(RichText::new("Gradient Hygiene").size(12.0).strong());
        ui.checkbox(
            &mut state.spec.training.hygiene.auto_clip,
            "AutoClip (Adaptive 90th-p norm)",
        );
        ui.checkbox(
            &mut state.spec.training.hygiene.grokfast,
            "Grokfast (Low-Frequency EMA Amplification)",
        );
        if ui
            .checkbox(
                &mut state.spec.training.hygiene.activation_checkpointing,
                "Activation Checkpointing (3x VRAM Save)",
            )
            .changed()
        {
            state.refresh_analyses();
        }

        ui.add_space(8.0);
        ui.separator();
        ui.label(
            RichText::new("Macro Wiring & Residual Topology")
                .size(12.0)
                .strong(),
        );
        let mut topo_changed = false;
        let current_topo_label = match &state.spec.graph.topology {
            WiringPattern::SerialStack => "Serial Stack (Feedforward)",
            WiringPattern::ResidualStream { norm_placement, .. } => match norm_placement {
                NormPlacement::PreLN => "Residual Stream (Pre-LN)",
                NormPlacement::PostLN => "Residual Stream (Post-LN)",
                NormPlacement::SandwichLN => "Residual Stream (Sandwich-LN)",
                NormPlacement::PeriLN => "Residual Stream (Peri-LN)",
            },
            WiringPattern::ReversibleSplitStream => "Reversible Split-Stream O(1) VRAM",
            WiringPattern::HyperConnections { .. } => "Multi-Lane Hyper-Connections (HC)",
            WiringPattern::RecurrentDepthLoop { .. } => "Recurrent-Depth Deliberation Loop",
            WiringPattern::ConditionalMoE { .. } => "Conditional Mixture-of-Experts",
        };
        egui::ComboBox::from_id_salt("macro_topology_select")
            .selected_text(current_topo_label)
            .width(230.0)
            .show_ui(ui, |ui| {
                if ui
                    .selectable_label(
                        matches!(state.spec.graph.topology, WiringPattern::SerialStack),
                        "Serial Stack (Feedforward)",
                    )
                    .clicked()
                {
                    state.spec.graph.topology = WiringPattern::SerialStack;
                    topo_changed = true;
                }
                if ui
                    .selectable_label(
                        matches!(
                            state.spec.graph.topology,
                            WiringPattern::ResidualStream {
                                norm_placement: NormPlacement::PreLN,
                                ..
                            }
                        ),
                        "Residual Stream (Pre-LN)",
                    )
                    .clicked()
                {
                    state.spec.graph.topology = WiringPattern::ResidualStream {
                        norm_placement: NormPlacement::PreLN,
                        layer_scale_init: Some(1e-2),
                    };
                    topo_changed = true;
                }
                if ui
                    .selectable_label(
                        matches!(
                            state.spec.graph.topology,
                            WiringPattern::ResidualStream {
                                norm_placement: NormPlacement::SandwichLN,
                                ..
                            }
                        ),
                        "Residual Stream (Sandwich-LN)",
                    )
                    .clicked()
                {
                    state.spec.graph.topology = WiringPattern::ResidualStream {
                        norm_placement: NormPlacement::SandwichLN,
                        layer_scale_init: Some(1e-2),
                    };
                    topo_changed = true;
                }
                if ui
                    .selectable_label(
                        matches!(
                            state.spec.graph.topology,
                            WiringPattern::ReversibleSplitStream
                        ),
                        "Reversible Split-Stream O(1) VRAM",
                    )
                    .clicked()
                {
                    state.spec.graph.topology = WiringPattern::ReversibleSplitStream;
                    topo_changed = true;
                }
                if ui
                    .selectable_label(
                        matches!(
                            state.spec.graph.topology,
                            WiringPattern::HyperConnections { .. }
                        ),
                        "Multi-Lane Hyper-Connections (HC)",
                    )
                    .clicked()
                {
                    state.spec.graph.topology = WiringPattern::HyperConnections { num_lanes: 4 };
                    topo_changed = true;
                }
                if ui
                    .selectable_label(
                        matches!(
                            state.spec.graph.topology,
                            WiringPattern::RecurrentDepthLoop { .. }
                        ),
                        "Recurrent-Depth Deliberation Loop",
                    )
                    .clicked()
                {
                    state.spec.graph.topology = WiringPattern::RecurrentDepthLoop {
                        max_steps: 8,
                        input_anchored: true,
                        halting: HaltingGate::PonderNetKL { lambda_prior: 0.1 },
                    };
                    topo_changed = true;
                }
            });

        if let WiringPattern::RecurrentDepthLoop {
            max_steps,
            input_anchored,
            ..
        } = &mut state.spec.graph.topology
        {
            topo_changed |= drag_usize(ui, "Loop Depth (K):", max_steps, 1..=64);
            if ui
                .checkbox(input_anchored, "Input-Anchored (e_x Injection)")
                .changed()
            {
                topo_changed = true;
            }
        } else if let WiringPattern::HyperConnections { num_lanes } = &mut state.spec.graph.topology
        {
            topo_changed |= drag_usize(ui, "HC Mixing Lanes:", num_lanes, 2..=16);
        }

        if let Some(in_port) = state.spec.inputs.first_mut() {
            ui.add_space(8.0);
            ui.separator();
            ui.label(
                RichText::new(format!(
                    "{} Input Port ('{}') & Encoding",
                    bi::BOX_ARROW_IN_RIGHT,
                    in_port.id
                ))
                .size(12.0)
                .strong(),
            );
            let cur_norm_label = match &in_port.normalizer {
                NormalizerSpec::None => "None (Raw Tensor)",
                NormalizerSpec::Standardize { .. } => "Standardize (Zero-Mean Unit-Var)",
                NormalizerSpec::MinMax { .. } => "MinMax [0, 1] Scaling",
                NormalizerSpec::RevIN { .. } => "RevIN (Reversible Instance Norm)",
                NormalizerSpec::Symlog => "Symlog sign(x)*ln(|x|+1)",
                NormalizerSpec::PLE { .. } => "PLE (Piecewise Linear Embeddings)",
                NormalizerSpec::ImageNetNormalize => "ImageNet RGB Normalization",
            };
            egui::ComboBox::from_id_salt("input_norm_select")
                .selected_text(cur_norm_label)
                .width(230.0)
                .show_ui(ui, |ui| {
                    for (label, val) in [
                        ("None (Raw Tensor)", NormalizerSpec::None),
                        (
                            "Standardize (Zero-Mean Unit-Var)",
                            NormalizerSpec::Standardize { mean_fill: 0.0 },
                        ),
                        (
                            "RevIN (Reversible Instance Norm)",
                            NormalizerSpec::RevIN {
                                eps: 1e-5,
                                affine: true,
                            },
                        ),
                        ("Symlog sign(x)*ln(|x|+1)", NormalizerSpec::Symlog),
                        (
                            "PLE (Piecewise Linear Embeddings)",
                            NormalizerSpec::PLE { num_bins: 16 },
                        ),
                        (
                            "ImageNet RGB Normalization",
                            NormalizerSpec::ImageNetNormalize,
                        ),
                    ] {
                        if ui
                            .selectable_label(cur_norm_label == label, label)
                            .clicked()
                        {
                            in_port.normalizer = val;
                            topo_changed = true;
                        }
                    }
                });

            let cur_enc_label = match &in_port.coordinate_encoding {
                CoordinateEncoding::None => "No Positional Encoding",
                CoordinateEncoding::RoPE1D { .. } => "1D RoPE (Rotary Embeddings)",
                CoordinateEncoding::RoPE2D_Dynamic { .. } => "2D Dynamic Aspect RoPE",
                CoordinateEncoding::RoPE3D { .. } => "3D Spatiotemporal RoPE",
                CoordinateEncoding::DecoupledRoPE => "Decoupled RoPE (MLA Compatible)",
                CoordinateEncoding::ALiBi => "ALiBi Linear Biases",
                CoordinateEncoding::PluckerRays => "6D Plücker Camera Rays",
                CoordinateEncoding::RWPE_Graph { .. } => "Graph Random-Walk (RWPE)",
                CoordinateEncoding::RandomFourierFeatures { .. } => "Random Fourier Features",
                CoordinateEncoding::MultiresHashGrid { .. } => "Multires Hash Grid (Instant-NGP)",
                CoordinateEncoding::PeriodicCrystalLattice => {
                    "Periodic Crystal Lattice (SO(3)xZ^3)"
                }
            };
            egui::ComboBox::from_id_salt("input_coord_select")
                .selected_text(cur_enc_label)
                .width(230.0)
                .show_ui(ui, |ui| {
                    for (label, val) in [
                        ("No Positional Encoding", CoordinateEncoding::None),
                        (
                            "1D RoPE (Rotary Embeddings)",
                            CoordinateEncoding::RoPE1D { base: 10000.0 },
                        ),
                        (
                            "2D Dynamic Aspect RoPE",
                            CoordinateEncoding::RoPE2D_Dynamic { base: 10000.0 },
                        ),
                        (
                            "3D Spatiotemporal RoPE",
                            CoordinateEncoding::RoPE3D { base: 10000.0 },
                        ),
                        (
                            "Decoupled RoPE (MLA Compatible)",
                            CoordinateEncoding::DecoupledRoPE,
                        ),
                        ("ALiBi Linear Biases", CoordinateEncoding::ALiBi),
                        ("6D Plücker Camera Rays", CoordinateEncoding::PluckerRays),
                        (
                            "Graph Random-Walk (RWPE)",
                            CoordinateEncoding::RWPE_Graph { walk_steps: 16 },
                        ),
                        (
                            "Multires Hash Grid (Instant-NGP)",
                            CoordinateEncoding::MultiresHashGrid {
                                num_levels: 16,
                                base_resolution: 16,
                            },
                        ),
                        (
                            "Periodic Crystal Lattice (SO(3)xZ^3)",
                            CoordinateEncoding::PeriodicCrystalLattice,
                        ),
                    ] {
                        if ui.selectable_label(cur_enc_label == label, label).clicked() {
                            in_port.coordinate_encoding = val;
                            topo_changed = true;
                        }
                    }
                });
        }

        if let Some(out_port) = state.spec.outputs.first_mut() {
            ui.add_space(8.0);
            ui.separator();
            ui.label(
                RichText::new(format!(
                    "{} Output Port ('{}') & Loss",
                    bi::BOX_ARROW_RIGHT,
                    out_port.id
                ))
                .size(12.0)
                .strong(),
            );
            let current_loss = out_port.loss;
            egui::ComboBox::from_id_salt("output_loss_select")
                .selected_text(format!("{:?}", current_loss))
                .width(230.0)
                .show_ui(ui, |ui| {
                    for &loss in &[
                        LossPairing::MSE,
                        LossPairing::CrossEntropy,
                        LossPairing::SmoothL1,
                        LossPairing::Focal,
                        LossPairing::FenchelYoung,
                        LossPairing::OptimalTransportConditionalFlowMatching,
                        LossPairing::ScoreMatchingSDE,
                        LossPairing::SobolevH1PhysicsResidual,
                        LossPairing::SymplecticEnergyConservation,
                        LossPairing::FrameAlignedPointErrorFAPE,
                        LossPairing::QuantileCRPS,
                        LossPairing::VICRegAntiCollapse,
                        LossPairing::GFlowNetTrajectoryBalance,
                        LossPairing::GRPOOutcomeReward,
                        LossPairing::DPOContrastiveLogRatio,
                        LossPairing::InfoNCEContrastive,
                        LossPairing::SobolevH2CurvaturePenalty,
                        LossPairing::LearnableBitwidthBopsPenalty,
                    ] {
                        if ui
                            .selectable_label(current_loss == loss, format!("{:?}", loss))
                            .clicked()
                        {
                            out_port.loss = loss;
                            topo_changed = true;
                        }
                    }
                });

            let cur_constraint_label = match &out_port.constraint {
                StructuralConstraint::DirectLinear => "Direct Linear Projection",
                StructuralConstraint::ContinuousRotation6D => "SO(3) 6D Gram-Schmidt Rotation",
                StructuralConstraint::CurlFreeVectorField => "Divergence-Free Curl(A) Field",
                StructuralConstraint::SymplecticHamiltonianField => "Symplectic Hamiltonian Flow",
                StructuralConstraint::Sparsemax => "Sparsemax Simplex Projection",
                StructuralConstraint::Entmax15 => "Entmax-1.5 Sparse Projection",
                StructuralConstraint::Monotonic { .. } => "Certified Monotonic Weights",
                StructuralConstraint::InputConvexICNN => "Input-Convex Neural Net (ICNN)",
                StructuralConstraint::OptNetKKT => "OptNet Differentiable KKT QP",
                StructuralConstraint::CBF_QP { .. } => "Control Barrier Function (CBF-QP)",
                StructuralConstraint::EvidentialDirichlet { .. } => "Evidential Dirichlet",
                StructuralConstraint::QuantileCRPS_GPD { .. } => "Quantile CRPS + GPD Tail",
                StructuralConstraint::LipschitzBounded { .. } => {
                    "Certified K-Lipschitz Spectral Bound"
                }
                StructuralConstraint::CInfinitySmoothSobolev { .. } => {
                    "Super-Smooth C^∞ Sobolev W^{k,2} Manifold"
                }
            };
            egui::ComboBox::from_id_salt("output_constraint_select")
                .selected_text(cur_constraint_label)
                .width(230.0)
                .show_ui(ui, |ui| {
                    for (label, val) in [
                        (
                            "Direct Linear Projection",
                            StructuralConstraint::DirectLinear,
                        ),
                        (
                            "SO(3) 6D Gram-Schmidt Rotation",
                            StructuralConstraint::ContinuousRotation6D,
                        ),
                        (
                            "Divergence-Free Curl(A) Field",
                            StructuralConstraint::CurlFreeVectorField,
                        ),
                        (
                            "Symplectic Hamiltonian Flow",
                            StructuralConstraint::SymplecticHamiltonianField,
                        ),
                        (
                            "Sparsemax Simplex Projection",
                            StructuralConstraint::Sparsemax,
                        ),
                        (
                            "Entmax-1.5 Sparse Projection",
                            StructuralConstraint::Entmax15,
                        ),
                        (
                            "Certified Monotonic Weights",
                            StructuralConstraint::Monotonic {
                                positive_weights: true,
                            },
                        ),
                        (
                            "Input-Convex Neural Net (ICNN)",
                            StructuralConstraint::InputConvexICNN,
                        ),
                        (
                            "OptNet Differentiable KKT QP",
                            StructuralConstraint::OptNetKKT,
                        ),
                        (
                            "Control Barrier Function (CBF-QP)",
                            StructuralConstraint::CBF_QP {
                                safe_distance: 0.25,
                                max_velocity: 2.0,
                            },
                        ),
                        (
                            "Certified K-Lipschitz Spectral Bound",
                            StructuralConstraint::LipschitzBounded {
                                max_lipschitz_k: 1.0,
                            },
                        ),
                        (
                            "Super-Smooth C^∞ Sobolev W^{k,2} Manifold",
                            StructuralConstraint::CInfinitySmoothSobolev {
                                derivative_order: 2,
                            },
                        ),
                    ] {
                        if ui
                            .selectable_label(cur_constraint_label == label, label)
                            .clicked()
                        {
                            out_port.constraint = val;
                            topo_changed = true;
                        }
                    }
                });
        }

        if topo_changed {
            state.refresh_analyses();
        }
    }
}

fn drag_usize(
    ui: &mut Ui,
    label: &str,
    target: &mut usize,
    range: std::ops::RangeInclusive<i32>,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label);
        let mut val = *target as i32;
        if ui
            .add(egui::DragValue::new(&mut val).range(range))
            .changed()
        {
            *target = val.max(0) as usize;
            changed = true;
        }
    });
    changed
}
