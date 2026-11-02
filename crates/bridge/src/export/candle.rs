//! Rust Candle Exporter: emits a complete, standalone Cargo crate that constructs
//! and runs the model using Candle + Candle Bridge layers.

use shared::spec::{AiDesignSpec, OperatorKind};
use std::collections::HashMap;

/// Generates a complete native Rust Candle project file map.
pub fn generate_candle_project(spec: &AiDesignSpec) -> HashMap<String, String> {
    let mut files = HashMap::new();
    files.insert("Cargo.toml".to_string(), generate_candle_cargo_toml(spec));
    files.insert("src/lib.rs".to_string(), generate_candle_lib_rs(spec));
    files.insert("src/model.rs".to_string(), generate_candle_model_rs(spec));
    files.insert(
        "examples/infer.rs".to_string(),
        generate_candle_infer_rs(spec),
    );
    files
}

pub fn generate_candle_cargo_toml(spec: &AiDesignSpec) -> String {
    format!(
        r#"[package]
name = "{}"
version = "0.1.0"
edition = "2024"

[dependencies]
candle-core = "0.8"
candle-nn = "0.8"
safetensors = "0.4"
anyhow = "1.0"
ai-designer-bridge = {{ path = "../../../crates/bridge" }}
"#,
        spec.metadata.model_id
    )
}

pub fn generate_candle_lib_rs(_spec: &AiDesignSpec) -> String {
    r#"pub mod model;
pub use model::*;
"#
    .to_string()
}

pub fn generate_candle_model_rs(spec: &AiDesignSpec) -> String {
    let mut s = String::new();
    s.push_str("//! Auto-generated Candle Model leveraging the AI Designer Candle Bridge.\n");
    s.push_str("use candle_core::{Result, Tensor};\n");
    s.push_str("use candle_nn::{Linear, Module, VarBuilder, linear, linear_no_bias};\n");
    s.push_str("use ai_designer_bridge::candle_bridge::*;\n\n");

    let class_name = sanitize_class_name(&spec.metadata.model_id);
    s.push_str(&format!("pub struct {class_name} {{\n"));

    for node in &spec.graph.nodes {
        match &node.operator {
            OperatorKind::Linear { .. } => {
                s.push_str(&format!("    pub {}: Linear,\n", node.id));
            }
            OperatorKind::MHA { .. } => {
                s.push_str(&format!("    pub {}: CandleMHA,\n", node.id));
            }
            OperatorKind::GQA { .. } => {
                s.push_str(&format!("    pub {}: CandleGQA,\n", node.id));
            }
            OperatorKind::MLA { .. } => {
                s.push_str(&format!("    pub {}: CandleMLA,\n", node.id));
            }
            OperatorKind::DiffAttn { .. } => {
                s.push_str(&format!("    pub {}: CandleDiffAttn,\n", node.id));
            }
            OperatorKind::SwiGLU { .. } => {
                s.push_str(&format!("    pub {}: CandleSwiGlu,\n", node.id));
            }
            OperatorKind::GeGLU { .. } => {
                s.push_str(&format!("    pub {}: CandleGeGlu,\n", node.id));
            }
            OperatorKind::DeepSeekMoE { .. } => {
                s.push_str(&format!("    pub {}: CandleMoE,\n", node.id));
            }
            OperatorKind::RMSNorm { .. } => {
                s.push_str(&format!("    pub {}: CandleRmsNorm,\n", node.id));
            }
            OperatorKind::RevINLayer { .. } => {
                s.push_str(&format!("    pub {}: CandleRevIn,\n", node.id));
            }
            OperatorKind::MambaS6 { .. } => {
                s.push_str(&format!("    pub {}: CandleMamba,\n", node.id));
            }
            OperatorKind::LISTA { .. } => {
                s.push_str(&format!("    pub {}: CandleLista,\n", node.id));
            }
            OperatorKind::ModernHopfield { .. } => {
                s.push_str(&format!("    pub {}: CandleHopfield,\n", node.id));
            }
            _ => {}
        }
    }
    s.push_str("}\n\n");

    s.push_str(&format!("impl {class_name} {{\n"));
    s.push_str("    pub fn load(vb: VarBuilder) -> Result<Self> {\n");
    s.push_str("        Ok(Self {\n");

    for node in &spec.graph.nodes {
        match &node.operator {
            OperatorKind::Linear {
                in_features,
                out_features,
                bias,
            } => {
                if *bias {
                    s.push_str(&format!(
                        "            {}: linear({}, {}, vb.pp(\"{}\"))?,\n",
                        node.id, in_features, out_features, node.id
                    ));
                } else {
                    s.push_str(&format!(
                        "            {}: linear_no_bias({}, {}, vb.pp(\"{}\"))?,\n",
                        node.id, in_features, out_features, node.id
                    ));
                }
            }
            OperatorKind::MHA {
                num_heads,
                d_k,
                d_v,
                ..
            } => {
                let d_model = num_heads * d_k;
                s.push_str(&format!(
                    "            {}: CandleMHA::new({}, {}, {}, {}, vb.pp(\"{}\"))?,\n",
                    node.id, d_model, num_heads, d_k, d_v, node.id
                ));
            }
            OperatorKind::GQA {
                num_q_heads,
                num_kv_heads,
                d_k,
            } => {
                let d_model = num_q_heads * d_k;
                s.push_str(&format!(
                    "            {}: CandleGQA::new({}, {}, {}, {}, vb.pp(\"{}\"))?,\n",
                    node.id, d_model, num_q_heads, num_kv_heads, d_k, node.id
                ));
            }
            OperatorKind::MLA {
                num_heads,
                d_latent_kv,
                d_v,
                ..
            } => {
                let d_model = num_heads * d_v;
                s.push_str(&format!(
                    "            {}: CandleMLA::new({}, {}, {}, {}, vb.pp(\"{}\"))?,\n",
                    node.id, d_model, num_heads, d_latent_kv, d_v, node.id
                ));
            }
            OperatorKind::DiffAttn { num_heads, d_k } => {
                let d_model = num_heads * d_k;
                s.push_str(&format!(
                    "            {}: CandleDiffAttn::new({}, {}, {}, vb.pp(\"{}\"))?,\n",
                    node.id, d_model, num_heads, d_k, node.id
                ));
            }
            OperatorKind::SwiGLU {
                in_features,
                hidden_features,
            } => {
                s.push_str(&format!(
                    "            {}: CandleSwiGlu::new({}, {}, vb.pp(\"{}\"))?,\n",
                    node.id, in_features, hidden_features, node.id
                ));
            }
            OperatorKind::GeGLU {
                in_features,
                hidden_features,
            } => {
                s.push_str(&format!(
                    "            {}: CandleGeGlu::new({}, {}, vb.pp(\"{}\"))?,\n",
                    node.id, in_features, hidden_features, node.id
                ));
            }
            OperatorKind::DeepSeekMoE {
                d_model,
                d_ffn,
                total_experts,
                routed_k,
                ..
            } => {
                s.push_str(&format!(
                    "            {}: CandleMoE::new({}, {}, {}, {}, vb.pp(\"{}\"))?,\n",
                    node.id, d_model, d_ffn, total_experts, routed_k, node.id
                ));
            }
            OperatorKind::RMSNorm { dim, eps } => {
                s.push_str(&format!(
                    "            {}: CandleRmsNorm::new({}, {}, vb.pp(\"{}\"))?,\n",
                    node.id, dim, eps, node.id
                ));
            }
            OperatorKind::RevINLayer { channels } => {
                s.push_str(&format!(
                    "            {}: CandleRevIn::new({}, 1e-5, vb.pp(\"{}\"))?,\n",
                    node.id, channels, node.id
                ));
            }
            OperatorKind::MambaS6 {
                d_model,
                d_state,
                expand,
                ..
            } => {
                s.push_str(&format!(
                    "            {}: CandleMamba::new({}, {}, {}, vb.pp(\"{}\"))?,\n",
                    node.id, d_model, d_state, expand, node.id
                ));
            }
            OperatorKind::LISTA {
                num_unrolled_steps,
                lambda,
            } => {
                s.push_str(&format!(
                    "            {}: CandleLista::new(128, {}, {}, vb.pp(\"{}\"))?,\n",
                    node.id, num_unrolled_steps, lambda, node.id
                ));
            }
            OperatorKind::ModernHopfield { num_heads, beta } => {
                let dim = num_heads * 64;
                s.push_str(&format!(
                    "            {}: CandleHopfield::new({}, {}, vb.pp(\"{}\"))?,\n",
                    node.id, dim, beta, node.id
                ));
            }
            _ => {}
        }
    }
    s.push_str("        })\n");
    s.push_str("    }\n\n");

    s.push_str("    pub fn forward(&self, mut x: Tensor) -> Result<Tensor> {\n");
    for node in &spec.graph.nodes {
        match &node.operator {
            OperatorKind::Linear { .. }
            | OperatorKind::MHA { .. }
            | OperatorKind::GQA { .. }
            | OperatorKind::MLA { .. }
            | OperatorKind::DiffAttn { .. }
            | OperatorKind::SwiGLU { .. }
            | OperatorKind::GeGLU { .. }
            | OperatorKind::DeepSeekMoE { .. }
            | OperatorKind::RMSNorm { .. }
            | OperatorKind::MambaS6 { .. }
            | OperatorKind::LISTA { .. } => {
                s.push_str(&format!("        x = self.{}.forward(&x)?;\n", node.id));
            }
            OperatorKind::RevINLayer { .. } => {
                s.push_str(&format!(
                    "        let (norm, _, _) = self.{}.normalize(&x)?;\n        x = norm;\n",
                    node.id
                ));
            }
            OperatorKind::ModernHopfield { .. } => {
                s.push_str(&format!("        x = self.{}.forward(&x, &x)?;\n", node.id));
            }
            OperatorKind::Activation { kind } => {
                let act_fn = match kind {
                    shared::spec::ActivationKind::GELU
                    | shared::spec::ActivationKind::QuickGELU => "x.gelu_erf()?",
                    shared::spec::ActivationKind::SiLU => "candle_nn::ops::silu(&x)?",
                    shared::spec::ActivationKind::ReLU => "x.relu()?",
                    _ => "candle_nn::ops::silu(&x)?",
                };
                s.push_str(&format!("        x = {act_fn};\n"));
            }
            _ => {}
        }
    }
    s.push_str("        Ok(x)\n");
    s.push_str("    }\n");
    s.push_str("}\n");

    s
}

pub fn generate_candle_infer_rs(spec: &AiDesignSpec) -> String {
    format!(
        r#"use candle_core::{{Device, Tensor}};
use candle_nn::VarBuilder;
use {}::model::{};

fn main() -> anyhow::Result<()> {{
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("Running Candle inference on: {{:?}}", device);

    let vb = VarBuilder::zeros(candle_core::DType::F32, &device);
    let model = {}::load(vb)?;

    let input = Tensor::randn(0f32, 1f32, (1, 16, 64), &device)?;
    let output = model.forward(input)?;
    println!("Forward pass success! Output shape: {{:?}}", output.shape());
    Ok(())
}}
"#,
        spec.metadata.model_id,
        sanitize_class_name(&spec.metadata.model_id),
        sanitize_class_name(&spec.metadata.model_id)
    )
}

fn sanitize_class_name(name: &str) -> String {
    let mut s = String::new();
    let mut capitalize = true;
    for c in name.chars() {
        if c.is_alphanumeric() {
            if capitalize {
                s.extend(c.to_uppercase());
                capitalize = false;
            } else {
                s.push(c);
            }
        } else {
            capitalize = true;
        }
    }
    if s.is_empty() {
        "CustomModel".to_string()
    } else {
        s
    }
}
