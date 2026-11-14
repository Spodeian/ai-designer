//! ONNX & PyTorch `torch.export` Graph Importer (`onnx_export_import`).
//!
//! Ingests JSON-serialized ONNX graphs (`{"graph": {"name": ..., "node": [...]}}`) and
//! PyTorch 2 `torch.export` ExportedProgram JSON (`{"graph_module": {"nodes": [...]}}`)
//! into strongly-typed `AiDesignSpec` blueprints.

use crate::spec::{
    ActivationKind, AiDesignSpec, ConnectionSpec, CoordinateEncoding, InputPortSpec, LossPairing,
    NodeSpec, NormalizerSpec, OperatorKind, OutputPortSpec, PrecisionFormat, StructuralConstraint,
    SymbolicDim, TensorSpace, WiringPattern,
};
use serde_json::Value;

/// Imports an ONNX JSON or `torch.export` ExportedProgram JSON graph into an `AiDesignSpec`.
pub fn import_from_onnx_or_torch_export_json(json_str: &str) -> Result<AiDesignSpec, String> {
    let root: Value =
        serde_json::from_str(json_str).map_err(|e| format!("Invalid graph JSON: {e}"))?;

    if let Some(graph) = root.get("graph") {
        import_onnx_json(&root, graph)
    } else if let Some(gm) = root.get("graph_module") {
        import_torch_export_json(&root, gm)
    } else if root.get("nodes").and_then(|v| v.as_array()).is_some() {
        import_torch_export_json(&root, &root)
    } else {
        Err(
            "Unrecognized graph JSON: expected ONNX 'graph.node' or torch.export 'graph_module.nodes'"
                .to_string(),
        )
    }
}

fn import_onnx_json(root: &Value, graph: &Value) -> Result<AiDesignSpec, String> {
    let model_name = graph
        .get("name")
        .or_else(|| root.get("model_name"))
        .and_then(|v| v.as_str())
        .unwrap_or("onnx_imported_model");
    let raw_nodes = graph
        .get("node")
        .or_else(|| graph.get("nodes"))
        .and_then(|v| v.as_array())
        .ok_or_else(|| "ONNX JSON 'graph' object must contain a 'node' array".to_string())?;

    let default_dim = graph
        .get("d_model")
        .or_else(|| root.get("d_model"))
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(512);

    build_spec_from_raw_nodes(model_name, "onnx_import", raw_nodes, default_dim, |node| {
        let op_type = node
            .get("op_type")
            .or_else(|| node.get("op"))
            .and_then(|v| v.as_str())
            .unwrap_or("Linear");
        let name = node.get("name").and_then(|v| v.as_str()).unwrap_or(op_type);
        let in_dim = extract_dim_hint(node, &["in_features", "in_channels", "dim"], default_dim);
        let out_dim = extract_dim_hint(
            node,
            &["out_features", "out_channels", "hidden_size", "dim"],
            in_dim,
        );
        let heads = extract_dim_hint(node, &["num_heads", "heads"], 8);
        (
            name.to_string(),
            op_type.to_string(),
            in_dim,
            out_dim,
            heads,
        )
    })
}

fn import_torch_export_json(root: &Value, gm: &Value) -> Result<AiDesignSpec, String> {
    let model_name = gm
        .get("name")
        .or_else(|| root.get("name"))
        .or_else(|| root.get("model_id"))
        .and_then(|v| v.as_str())
        .unwrap_or("torch_export_model");
    let raw_nodes = gm
        .get("nodes")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "torch.export JSON must contain a 'nodes' array".to_string())?;

    let default_dim = gm
        .get("d_model")
        .or_else(|| root.get("d_model"))
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(512);

    build_spec_from_raw_nodes(
        model_name,
        "torch_export_import",
        raw_nodes,
        default_dim,
        |node| {
            let target = node
                .get("target")
                .or_else(|| node.get("op_type"))
                .or_else(|| node.get("op"))
                .and_then(|v| v.as_str())
                .unwrap_or("Linear");
            let name = node.get("name").and_then(|v| v.as_str()).unwrap_or(target);
            let in_dim =
                extract_dim_hint(node, &["in_features", "in_channels", "dim"], default_dim);
            let out_dim = extract_dim_hint(
                node,
                &["out_features", "out_channels", "hidden_size", "dim"],
                in_dim,
            );
            let heads = extract_dim_hint(node, &["num_heads", "heads"], 8);
            (name.to_string(), target.to_string(), in_dim, out_dim, heads)
        },
    )
}

fn extract_dim_hint(node: &Value, keys: &[&str], fallback: usize) -> usize {
    for container in [
        Some(node),
        node.get("attrs"),
        node.get("attributes"),
        node.get("kwargs"),
    ] {
        let Some(obj) = container.and_then(|v| v.as_object()) else {
            continue;
        };
        for &k in keys {
            if let Some(val) = obj.get(k).and_then(|v| v.as_u64())
                && val > 0
            {
                return val as usize;
            }
        }
    }
    fallback
}

fn build_spec_from_raw_nodes<F>(
    model_name: &str,
    tag: &str,
    raw_nodes: &[Value],
    default_dim: usize,
    extract: F,
) -> Result<AiDesignSpec, String>
where
    F: Fn(&Value) -> (String, String, usize, usize, usize),
{
    let clean_id = sanitize_id(model_name);
    let mut spec = AiDesignSpec::new(&clean_id, &format!("{model_name} (Imported)"));
    spec.metadata.tags.push(tag.to_string());

    let mut saw_residual = false;
    let mut first_in_dim = None;
    let mut last_out_dim = default_dim;

    for (idx, raw) in raw_nodes.iter().enumerate() {
        let (raw_name, raw_op, in_dim, out_dim, heads) = extract(raw);
        let lower = raw_op.to_ascii_lowercase();
        if matches!(lower.as_str(), "placeholder" | "output" | "get_attr") {
            continue;
        }
        if first_in_dim.is_none() {
            first_in_dim = Some(in_dim);
        }
        last_out_dim = out_dim;

        if lower.contains("residual") || lower == "add" || lower.ends_with(".add") {
            saw_residual = true;
        }

        let operator = map_external_op(&lower, in_dim, out_dim, heads);
        let mut node = NodeSpec::new(
            format!("{}_{}", sanitize_id(&raw_name), idx),
            format!("{} ({})", raw_name, raw_op),
            operator,
        );
        node.precision = Some(PrecisionFormat::BF16);
        node.metadata.insert("source_op".to_string(), raw_op);
        spec.graph.nodes.push(node);
    }

    if spec.graph.nodes.is_empty() {
        spec.graph.nodes.push(NodeSpec::new(
            "linear_0",
            "Fallback Linear Projection",
            OperatorKind::Linear {
                in_features: default_dim,
                out_features: default_dim,
                bias: true,
            },
        ));
    }

    if saw_residual {
        spec.graph.topology = WiringPattern::ResidualStream {
            norm_placement: crate::spec::NormPlacement::PreLN,
            layer_scale_init: None,
        };
    }

    for i in 0..spec.graph.nodes.len().saturating_sub(1) {
        spec.graph.connections.push(ConnectionSpec {
            from_node: spec.graph.nodes[i].id.clone(),
            from_port: 0,
            to_node: spec.graph.nodes[i + 1].id.clone(),
            to_port: 0,
        });
    }

    let in_dim = first_in_dim.unwrap_or(default_dim);
    spec.inputs.push(InputPortSpec {
        id: "x".to_string(),
        name: "Input Tensor".to_string(),
        space: TensorSpace::ContinuousVector {
            dimension: in_dim,
            range: None,
        },
        symbolic_shape: vec![
            SymbolicDim::from("B"),
            SymbolicDim::from("S"),
            SymbolicDim::from(in_dim),
        ],
        normalizer: NormalizerSpec::None,
        coordinate_encoding: CoordinateEncoding::None,
        patching: None,
    });

    spec.outputs.push(OutputPortSpec {
        id: "y".to_string(),
        name: "Output Tensor".to_string(),
        space: TensorSpace::ContinuousVector {
            dimension: last_out_dim,
            range: None,
        },
        symbolic_shape: vec![
            SymbolicDim::from("B"),
            SymbolicDim::from("S"),
            SymbolicDim::from(last_out_dim),
        ],
        constraint: StructuralConstraint::DirectLinear,
        loss: LossPairing::MSE,
    });

    spec.validate()?;
    Ok(spec)
}

fn map_external_op(op_lower: &str, in_dim: usize, out_dim: usize, heads: usize) -> OperatorKind {
    if op_lower.contains("conv") {
        OperatorKind::ConvNeXtV2Block {
            dim: out_dim.max(1),
            kernel_size: 3,
            mlp_ratio: 4.0,
        }
    } else if op_lower.contains("rmsnorm") || op_lower.contains("rms_norm") {
        OperatorKind::RMSNorm {
            dim: out_dim.max(1),
            eps: 1e-5,
        }
    } else if op_lower.contains("batchnorm") || op_lower.contains("layernorm") {
        OperatorKind::LayerNorm {
            dim: out_dim.max(1),
            eps: 1e-5,
        }
    } else if op_lower.contains("rope") || op_lower.contains("rotary") {
        let h = heads.max(1);
        OperatorKind::MLA {
            num_heads: h,
            d_latent_kv: (out_dim / 4).max(16),
            d_rope: (out_dim / h).max(8),
            d_v: (out_dim / h).max(8),
        }
    } else if op_lower.contains("attention") || op_lower == "mha" {
        let h = heads.max(1);
        let dk = (out_dim / h).max(8);
        OperatorKind::MHA {
            num_heads: h,
            d_k: dk,
            d_v: dk,
            dropout: 0.0,
        }
    } else if op_lower.contains("swiglu") {
        OperatorKind::SwiGLU {
            in_features: in_dim.max(1),
            hidden_features: out_dim.max(1),
        }
    } else if op_lower.contains("moe") || op_lower.contains("mixture_of_experts") {
        OperatorKind::DeepSeekMoE {
            total_experts: 8,
            routed_k: 2,
            shared_experts: 1,
            d_model: in_dim.max(1),
            d_ffn: out_dim.max(in_dim * 4),
            aux_loss_free_bias: true,
        }
    } else if op_lower.contains("residual")
        || op_lower == "add"
        || op_lower.ends_with(".add")
        || op_lower.contains("flatten")
        || op_lower.contains("concat")
        || op_lower.contains("softmax")
        || op_lower.contains("upsample")
        || op_lower.contains("resize")
    {
        OperatorKind::Activation {
            kind: ActivationKind::SiLU,
        }
    } else {
        OperatorKind::Linear {
            in_features: in_dim.max(1),
            out_features: out_dim.max(1),
            bias: true,
        }
    }
}

fn sanitize_id(raw: &str) -> String {
    let mut s: String = raw
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect();
    while s.contains("__") {
        s = s.replace("__", "_");
    }
    let trimmed = s.trim_matches('_').to_string();
    if trimmed.is_empty() {
        "imported_node".to_string()
    } else {
        trimmed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_import_onnx_graph_operators() {
        let onnx_json = serde_json::json!({
            "graph": {
                "name": "VisionTransformerOnnx",
                "d_model": 256,
                "node": [
                    { "name": "stem", "op_type": "Conv2d", "attrs": { "in_channels": 3, "out_channels": 256 } },
                    { "name": "norm1", "op_type": "BatchNormalization", "attrs": { "dim": 256 } },
                    { "name": "norm2", "op_type": "LayerNormalization", "attrs": { "dim": 256 } },
                    { "name": "attn", "op_type": "MultiHeadAttention", "attrs": { "dim": 256, "num_heads": 8 } },
                    { "name": "res1", "op_type": "Residual", "attrs": { "dim": 256 } },
                    { "name": "flat", "op_type": "Flatten", "attrs": { "dim": 256 } },
                    { "name": "cat", "op_type": "Concat", "attrs": { "dim": 256 } },
                    { "name": "probs", "op_type": "Softmax", "attrs": { "dim": 256 } },
                    { "name": "up", "op_type": "Upsample", "attrs": { "dim": 256 } }
                ]
            }
        });

        let spec = import_from_onnx_or_torch_export_json(&onnx_json.to_string())
            .expect("ONNX import should succeed");
        assert_eq!(spec.metadata.model_id, "visiontransformeronnx");
        assert_eq!(spec.graph.nodes.len(), 9);
        assert!(matches!(
            spec.graph.topology,
            WiringPattern::ResidualStream { .. }
        ));
        assert!(matches!(
            spec.graph.nodes[0].operator,
            OperatorKind::ConvNeXtV2Block { .. }
        ));
        assert!(matches!(
            spec.graph.nodes[3].operator,
            OperatorKind::MHA { num_heads: 8, .. }
        ));
    }

    #[test]
    fn test_import_torch_export_graph_operators() {
        let torch_export_json = serde_json::json!({
            "graph_module": {
                "name": "FrontierBlockExport",
                "d_model": 512,
                "nodes": [
                    { "name": "x", "target": "placeholder" },
                    { "name": "rope", "target": "aten.RotaryEmbedding", "kwargs": { "dim": 512, "num_heads": 8 } },
                    { "name": "swiglu", "target": "custom.SwiGLU", "kwargs": { "in_features": 512, "hidden_features": 2048, "out_features": 2048 } },
                    { "name": "moe", "target": "custom.MoE", "kwargs": { "dim": 512 } },
                    { "name": "rms", "target": "aten.RMSNorm", "kwargs": { "dim": 512 } },
                    { "name": "out", "target": "output" }
                ]
            }
        });

        let spec = import_from_onnx_or_torch_export_json(&torch_export_json.to_string())
            .expect("torch.export import should succeed");
        assert_eq!(spec.graph.nodes.len(), 4);
        assert!(matches!(
            spec.graph.nodes[0].operator,
            OperatorKind::MLA { .. }
        ));
        assert!(matches!(
            spec.graph.nodes[1].operator,
            OperatorKind::SwiGLU { .. }
        ));
        assert!(matches!(
            spec.graph.nodes[2].operator,
            OperatorKind::DeepSeekMoE { .. }
        ));
        assert!(matches!(
            spec.graph.nodes[3].operator,
            OperatorKind::RMSNorm { .. }
        ));
    }
}
