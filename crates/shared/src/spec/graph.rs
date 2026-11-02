//! Morphological graph, wiring patterns, residual topologies, and node specifications.

use serde::{Deserialize, Serialize};

use super::operators::OperatorKind;
use super::precision::PrecisionFormat;

/// Normalization layer placement relative to attention and FFN blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum NormPlacement {
    #[default]
    #[serde(rename = "pre_ln", alias = "pre_l_n")]
    PreLN,
    #[serde(rename = "post_ln", alias = "post_l_n")]
    PostLN,
    #[serde(rename = "sandwich_ln", alias = "sandwich_l_n")]
    SandwichLN,
    #[serde(rename = "peri_ln", alias = "peri_l_n")]
    PeriLN,
}

/// Adaptive halting gate mechanism for recurrent depth deliberation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum HaltingGate {
    #[default]
    None,
    /// PonderNet geometric KL halting hazard regularization.
    #[serde(rename = "ponder_net_kl", alias = "ponder_net_k_l")]
    PonderNetKL { lambda_prior: f64 },
    /// Adaptive Computation Time (ACT) cumulative threshold halting.
    #[serde(rename = "act_threshold", alias = "a_c_t_threshold")]
    ACTThreshold { threshold: f64 },
}

/// Macro wiring topology across layers and residual streams.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WiringPattern {
    SerialStack,
    ResidualStream {
        #[serde(default)]
        norm_placement: NormPlacement,
        #[serde(default)]
        layer_scale_init: Option<f64>,
    },
    /// Reversible Split-Stream bijection y1 = x1 + F(x2), y2 = x2 + G(y1) achieving O(1) activation memory.
    ReversibleSplitStream,
    /// Multi-Lane Hyper-Connections with learnable depth/width mixing matrices.
    HyperConnections {
        num_lanes: usize,
    },
    /// Input-anchored recurrent depth deliberation s_{k+1} = T([s_k; e_x]) with dynamic halting.
    RecurrentDepthLoop {
        max_steps: usize,
        #[serde(default = "default_true")]
        input_anchored: bool,
        #[serde(default)]
        halting: HaltingGate,
    },
    /// Conditional sparse Mixture-of-Experts with routed and shared experts.
    ConditionalMoE {
        total_experts: usize,
        routed_k: usize,
        shared_experts: usize,
    },
}

impl Default for WiringPattern {
    fn default() -> Self {
        Self::ResidualStream {
            norm_placement: NormPlacement::PreLN,
            layer_scale_init: Some(1e-2),
        }
    }
}

fn default_true() -> bool {
    true
}

/// Functional role of a node in a modular compound AI system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ModularRole {
    #[default]
    StandardBackbone,
    /// Input Conformer: converts raw heterogeneous inputs (audio spectrograms, vision patches, tabular tokens, bytes) into shared latent representations.
    InputConformer,
    /// Backbone Processor: core reasoning, attention, recurrence, or state space transformation.
    BackboneProcessor,
    /// Routing Gate: dynamic router dispatching tokens or batches conditionally across modular branches.
    RoutingGate,
    /// Speculative Draft Head: speculative proposal unit only active during accelerated speculative decoding.
    SpeculativeDraftHead,
    /// Task-specific Output Adapter (e.g. action policy, text generation, segmentation mask).
    OutputAdapter,
    /// Auxiliary Diagnostic / Interpretability Probe: sidecar probe or auxiliary loss head.
    AuxiliaryProbe,
}

impl ModularRole {
    pub const ALL: &'static [ModularRole] = &[
        Self::StandardBackbone,
        Self::InputConformer,
        Self::BackboneProcessor,
        Self::RoutingGate,
        Self::SpeculativeDraftHead,
        Self::OutputAdapter,
        Self::AuxiliaryProbe,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            Self::StandardBackbone => "Standard Backbone Layer",
            Self::InputConformer => "Input Conformer / Modality Adapter",
            Self::BackboneProcessor => "Modular Backbone Processor",
            Self::RoutingGate => "Dynamic Routing Gate / Dispatcher",
            Self::SpeculativeDraftHead => "Speculative Draft Head (Inference Fastpath)",
            Self::OutputAdapter => "Task-Specific Output Adapter",
            Self::AuxiliaryProbe => "Auxiliary Diagnostic / Probe Head",
        }
    }

    pub fn badge(&self) -> &'static str {
        match self {
            Self::StandardBackbone => "Backbone",
            Self::InputConformer => "Conformer",
            Self::BackboneProcessor => "Processor",
            Self::RoutingGate => "Router",
            Self::SpeculativeDraftHead => "Draft Head",
            Self::OutputAdapter => "Adapter",
            Self::AuxiliaryProbe => "Probe",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::StandardBackbone => {
                "Primary core feedforward or sequence transformation layer active on all tokens."
            }
            Self::InputConformer => {
                "Modality-specific front-end conformer (e.g. patch embedding, audio convolution, RevIN, tokenizer) adapting heterogeneous inputs into a unified latent space."
            }
            Self::BackboneProcessor => {
                "Modular core block in a multi-model architecture that can be swapped or combined with alternative processors."
            }
            Self::RoutingGate => {
                "Dynamic gating mechanism that selectively activates downstream sub-modules (e.g. MoE router, Mixture of Depths, modal gate)."
            }
            Self::SpeculativeDraftHead => {
                "Fast draft head generating candidate token speculations, activated only during speculative decoding verification."
            }
            Self::OutputAdapter => {
                "Specialized readout head mapping shared latent states into task-specific formats (e.g. text logits, continuous 6D action, probability quantiles)."
            }
            Self::AuxiliaryProbe => {
                "Sidecar monitoring head computing intermediate supervision, interpretability metrics, or auxiliary regularization."
            }
        }
    }
}

/// Execution condition defining when a component is active or skipped/bypassed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExecutionCondition {
    #[default]
    AlwaysActive,
    /// Conditional execution based on a dynamic gate or input modality (e.g. Mixture of Depths, audio present).
    Conditional {
        condition_tag: String,
        #[serde(default = "default_true")]
        bypass_residual: bool,
    },
    /// Alternate modular option in a multi-choice group (e.g., Conformer Option A vs B, or Backbone Option A vs B).
    AlternateChoice {
        group_id: String,
        option_name: String,
        #[serde(default = "default_true")]
        is_active: bool,
    },
    /// Optional adapter branch (e.g. LoRA/DoRA adapter, speculative verification head) that can be enabled/disabled.
    OptionalAdapter {
        adapter_tag: String,
        #[serde(default)]
        enabled_by_default: bool,
    },
}

impl ExecutionCondition {
    pub fn label(&self) -> String {
        match self {
            Self::AlwaysActive => "Always Active (100% of inputs)".to_string(),
            Self::Conditional {
                condition_tag,
                bypass_residual,
            } => {
                let bypass = if *bypass_residual {
                    "residual bypass"
                } else {
                    "drop"
                };
                format!("Conditional ('{condition_tag}', {bypass})")
            }
            Self::AlternateChoice {
                group_id,
                option_name,
                is_active,
            } => {
                let status = if *is_active {
                    "Active Option"
                } else {
                    "Inactive Option"
                };
                format!("Choice in '{group_id}': {option_name} [{status}]")
            }
            Self::OptionalAdapter {
                adapter_tag,
                enabled_by_default,
            } => {
                let def = if *enabled_by_default {
                    "Default ON"
                } else {
                    "Default OFF"
                };
                format!("Optional Adapter ('{adapter_tag}', {def})")
            }
        }
    }

    pub fn is_currently_active(&self) -> bool {
        match self {
            Self::AlwaysActive => true,
            Self::Conditional { .. } => true,
            Self::AlternateChoice { is_active, .. } => *is_active,
            Self::OptionalAdapter {
                enabled_by_default, ..
            } => *enabled_by_default,
        }
    }
}

/// Individual computation node within the graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeSpec {
    pub id: String,
    pub name: String,
    pub operator: OperatorKind,
    #[serde(default)]
    pub precision: Option<PrecisionFormat>,
    #[serde(default)]
    pub modular_role: ModularRole,
    #[serde(default)]
    pub execution_condition: ExecutionCondition,
    #[serde(default)]
    pub metadata: std::collections::HashMap<String, String>,
}

impl Default for NodeSpec {
    fn default() -> Self {
        Self {
            id: String::new(),
            name: String::new(),
            operator: OperatorKind::Activation {
                kind: super::operators::ActivationKind::SiLU,
            },
            precision: None,
            modular_role: ModularRole::default(),
            execution_condition: ExecutionCondition::default(),
            metadata: std::collections::HashMap::new(),
        }
    }
}

impl NodeSpec {
    pub fn new(id: impl Into<String>, name: impl Into<String>, operator: OperatorKind) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            operator,
            precision: None,
            modular_role: ModularRole::default(),
            execution_condition: ExecutionCondition::default(),
            metadata: std::collections::HashMap::new(),
        }
    }

    pub fn with_modular(mut self, role: ModularRole, condition: ExecutionCondition) -> Self {
        self.modular_role = role;
        self.execution_condition = condition;
        self
    }
}

/// Directed connection channel between two node ports.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConnectionSpec {
    pub from_node: String,
    pub from_port: usize,
    pub to_node: String,
    pub to_port: usize,
}

/// Hierarchical computation graph combining topology, nodes, and interconnections.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ComputeGraph {
    pub topology: WiringPattern,
    pub nodes: Vec<NodeSpec>,
    #[serde(default)]
    pub connections: Vec<ConnectionSpec>,
}

impl ComputeGraph {
    pub fn find_node(&self, id: &str) -> Option<&NodeSpec> {
        self.nodes.iter().find(|n| n.id == id)
    }

    pub fn find_node_mut(&mut self, id: &str) -> Option<&mut NodeSpec> {
        self.nodes.iter_mut().find(|n| n.id == id)
    }

    pub fn total_flops(&self, seq_len: usize) -> usize {
        self.nodes
            .iter()
            .map(|n| n.operator.compute_flops(seq_len))
            .sum()
    }
}
