//! LoRA / DoRA parameter-efficient adapter planning and structured pruning surgery.

use crate::spec::{AiDesignSpec, OperatorKind};
use serde::{Deserialize, Serialize};

/// Configuration for LoRA / DoRA low-rank adapter injection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoraAdapterConfig {
    pub rank: usize,
    pub alpha: f32,
    pub use_dora: bool,
    pub target_modules: Vec<String>,
}

impl Default for LoraAdapterConfig {
    fn default() -> Self {
        Self {
            rank: 16,
            alpha: 32.0,
            use_dora: true,
            target_modules: vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
                "o_proj".to_string(),
            ],
        }
    }
}

/// Summary of an adapter surgery plan on an `AiDesignSpec`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdapterSurgeryPlan {
    pub model_id: String,
    pub is_dora: bool,
    pub adapter_params: usize,
    pub base_params: usize,
    pub trainable_fraction: f64,
    pub attached_layers: Vec<String>,
}

fn estimate_operator_width(op: &OperatorKind) -> (usize, bool) {
    match op {
        OperatorKind::MHA { num_heads, d_k, .. } => {
            ((*num_heads).saturating_mul(*d_k).max(64), true)
        }
        OperatorKind::GQA {
            num_q_heads, d_k, ..
        } => ((*num_q_heads).saturating_mul(*d_k).max(64), true),
        OperatorKind::MLA { num_heads, d_v, .. } => {
            ((*num_heads).saturating_mul(*d_v).max(64), true)
        }
        OperatorKind::DiffAttn { num_heads, d_k } => {
            ((*num_heads).saturating_mul(*d_k).max(64), true)
        }
        OperatorKind::LinearAttention { num_heads, d_k, .. } => {
            ((*num_heads).saturating_mul(*d_k).max(64), true)
        }
        OperatorKind::MambaS6 { d_model, .. }
        | OperatorKind::Mamba2SSD { d_model, .. }
        | OperatorKind::S4D { d_model, .. }
        | OperatorKind::LRU { d_model, .. } => (*d_model, true),
        _ => (256, false),
    }
}

/// Plans LoRA/DoRA adapter injection across eligible projection/attention operators of `spec`.
pub fn plan_lora_dora_surgery(
    spec: &AiDesignSpec,
    config: &LoraAdapterConfig,
) -> AdapterSurgeryPlan {
    let r = config.rank.max(1);
    let mut base_params = 0usize;
    let mut adapter_params = 0usize;
    let mut attached_layers = Vec::new();

    for node in &spec.graph.nodes {
        let (w, is_eligible) = estimate_operator_width(&node.operator);
        let layer_params = w.saturating_mul(w);
        base_params = base_params.saturating_add(layer_params);

        if is_eligible {
            // A (w x r) + B (r x w) + optional DoRA magnitude vector (w)
            let per_proj = 2 * w * r + if config.use_dora { w } else { 0 };
            let n_targets = config.target_modules.len().max(1);
            adapter_params = adapter_params.saturating_add(per_proj * n_targets);
            attached_layers.push(node.id.clone());
        }
    }

    let total = base_params.saturating_add(adapter_params).max(1);
    AdapterSurgeryPlan {
        model_id: spec.metadata.model_id.clone(),
        is_dora: config.use_dora,
        adapter_params,
        base_params,
        trainable_fraction: adapter_params as f64 / total as f64,
        attached_layers,
    }
}

/// Structured pruning granularity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PruningGranularity {
    Channel,
    AttentionHead,
    LayerDrop,
}

/// Configuration for structured pruning surgery.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuredPruningConfig {
    pub sparsity_ratio: f64,
    pub granularity: PruningGranularity,
}

/// Summary of structured pruning impact on an `AiDesignSpec`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PruningPlan {
    pub granularity: PruningGranularity,
    pub target_sparsity: f64,
    pub remaining_nodes: usize,
    pub estimated_speedup: f64,
}

/// Estimates structured pruning impact for a model architecture.
pub fn plan_structured_pruning(
    spec: &AiDesignSpec,
    config: &StructuredPruningConfig,
) -> PruningPlan {
    let s = config.sparsity_ratio.clamp(0.0, 0.9);
    let total_nodes = spec.graph.nodes.len();
    let remaining_nodes = match config.granularity {
        PruningGranularity::LayerDrop => ((total_nodes as f64) * (1.0 - s)).ceil() as usize,
        PruningGranularity::Channel | PruningGranularity::AttentionHead => total_nodes,
    };
    let estimated_speedup = 1.0 / (1.0 - s * 0.85).max(0.1);
    PruningPlan {
        granularity: config.granularity,
        target_sparsity: s,
        remaining_nodes,
        estimated_speedup,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::NodeSpec;

    #[test]
    fn test_lora_dora_and_pruning_plans() {
        let mut spec = AiDesignSpec::new("surgery_test", "Surgery Test");
        spec.graph.nodes.push(NodeSpec::new(
            "attn0",
            "Attention 0",
            OperatorKind::MHA {
                num_heads: 8,
                d_k: 64,
                d_v: 64,
                dropout: 0.0,
            },
        ));

        let dora = plan_lora_dora_surgery(&spec, &LoraAdapterConfig::default());
        let lora = plan_lora_dora_surgery(
            &spec,
            &LoraAdapterConfig {
                use_dora: false,
                ..Default::default()
            },
        );
        assert!(dora.is_dora);
        assert!(dora.adapter_params > lora.adapter_params);
        assert!(dora.trainable_fraction > 0.0 && dora.trainable_fraction < 1.0);

        let prune = plan_structured_pruning(
            &spec,
            &StructuredPruningConfig {
                sparsity_ratio: 0.5,
                granularity: PruningGranularity::Channel,
            },
        );
        assert!(prune.estimated_speedup > 1.5);
    }
}
