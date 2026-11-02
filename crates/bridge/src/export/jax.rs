//! JAX / Equinox Generator: emits modern pure-functional JAX modules supporting
//! jax.jit, jax.vmap, and shard_map array partitioning.

use shared::spec::{AiDesignSpec, OperatorKind};
use std::collections::HashMap;

/// Generates a complete JAX / Equinox project directory.
pub fn generate_jax_project(spec: &AiDesignSpec) -> HashMap<String, String> {
    let mut files = HashMap::new();
    files.insert("model.py".to_string(), generate_jax_model(spec));
    files.insert("train.py".to_string(), generate_jax_train(spec));
    files.insert(
        "requirements.txt".to_string(),
        "jax>=0.4.25\njaxlib>=0.4.25\nequinox>=0.11.3\noptax>=0.1.9\n".to_string(),
    );
    files
}

pub fn generate_jax_model(spec: &AiDesignSpec) -> String {
    let mut s = String::new();
    s.push_str(
        r#"# Auto-generated JAX / Equinox Pure-Functional Model
import equinox as eqx
import jax
import jax.numpy as jnp

"#,
    );

    s.push_str(&format!(
        "class {}(eqx.Module):\n",
        sanitize_class_name(&spec.metadata.model_id)
    ));

    for node in &spec.graph.nodes {
        match &node.operator {
            OperatorKind::Linear { .. } => {
                s.push_str(&format!("    {}: eqx.nn.Linear\n", node.id));
            }
            OperatorKind::MHA { .. } | OperatorKind::GQA { .. } => {
                s.push_str(&format!("    {}: eqx.nn.MultiheadAttention\n", node.id));
            }
            OperatorKind::SwiGLU { .. } => {
                s.push_str(&format!("    {}_gate: eqx.nn.Linear\n", node.id));
                s.push_str(&format!("    {}_up: eqx.nn.Linear\n", node.id));
                s.push_str(&format!("    {}_down: eqx.nn.Linear\n", node.id));
            }
            OperatorKind::Embedding { .. } => {
                s.push_str(&format!("    {}: eqx.nn.Embedding\n", node.id));
            }
            OperatorKind::RMSNorm { .. } => {
                s.push_str(&format!("    {}: eqx.nn.RMSNorm\n", node.id));
            }
            OperatorKind::LayerNorm { .. } => {
                s.push_str(&format!("    {}: eqx.nn.LayerNorm\n", node.id));
            }
            _ => {}
        }
    }

    s.push_str("\n    def __init__(self, key: jax.Array):\n");
    s.push_str("        keys = jax.random.split(key, 64)\n");
    let mut key_idx = 0;
    for node in &spec.graph.nodes {
        match &node.operator {
            OperatorKind::Linear {
                in_features,
                out_features,
                bias,
            } => {
                s.push_str(&format!(
                    "        self.{} = eqx.nn.Linear({}, {}, use_bias={}, key=keys[{}])\n",
                    node.id, in_features, out_features, bias, key_idx
                ));
                key_idx += 1;
            }
            OperatorKind::MHA { num_heads, d_k, .. } => {
                let d_model = num_heads * d_k;
                s.push_str(&format!(
                    "        self.{} = eqx.nn.MultiheadAttention(num_heads={}, query_size={}, key=keys[{}])\n",
                    node.id, num_heads, d_model, key_idx
                ));
                key_idx += 1;
            }
            OperatorKind::GQA {
                num_q_heads, d_k, ..
            } => {
                let d_model = num_q_heads * d_k;
                s.push_str(&format!(
                    "        self.{} = eqx.nn.MultiheadAttention(num_heads={}, query_size={}, key=keys[{}])\n",
                    node.id, num_q_heads, d_model, key_idx
                ));
                key_idx += 1;
            }
            OperatorKind::SwiGLU {
                in_features,
                hidden_features,
            } => {
                s.push_str(&format!(
                    "        self.{}_gate = eqx.nn.Linear({}, {}, use_bias=False, key=keys[{}])\n",
                    node.id, in_features, hidden_features, key_idx
                ));
                key_idx += 1;
                s.push_str(&format!(
                    "        self.{}_up = eqx.nn.Linear({}, {}, use_bias=False, key=keys[{}])\n",
                    node.id, in_features, hidden_features, key_idx
                ));
                key_idx += 1;
                s.push_str(&format!(
                    "        self.{}_down = eqx.nn.Linear({}, {}, use_bias=False, key=keys[{}])\n",
                    node.id, hidden_features, in_features, key_idx
                ));
                key_idx += 1;
            }
            OperatorKind::Embedding {
                num_embeddings,
                embedding_dim,
            } => {
                s.push_str(&format!(
                    "        self.{} = eqx.nn.Embedding({}, {}, key=keys[{}])\n",
                    node.id, num_embeddings, embedding_dim, key_idx
                ));
                key_idx += 1;
            }
            OperatorKind::RMSNorm { dim, .. } => {
                s.push_str(&format!(
                    "        self.{} = eqx.nn.RMSNorm({})\n",
                    node.id, dim
                ));
            }
            OperatorKind::LayerNorm { dim, .. } => {
                s.push_str(&format!(
                    "        self.{} = eqx.nn.LayerNorm({})\n",
                    node.id, dim
                ));
            }
            _ => {}
        }
    }

    s.push_str("\n    def __call__(self, x: jax.Array) -> jax.Array:\n");
    for node in &spec.graph.nodes {
        match &node.operator {
            OperatorKind::Linear { .. } => {
                s.push_str(&format!("        x = jax.vmap(self.{})(x)\n", node.id));
            }
            OperatorKind::MHA { .. } | OperatorKind::GQA { .. } => {
                s.push_str(&format!("        x = self.{}(x, x, x)\n", node.id));
            }
            OperatorKind::SwiGLU { .. } => {
                s.push_str(&format!(
                    "        x = jax.vmap(self.{0}_down)(jax.nn.silu(jax.vmap(self.{0}_gate)(x)) * jax.vmap(self.{0}_up)(x))\n",
                    node.id
                ));
            }
            OperatorKind::Embedding { .. } => {
                s.push_str(&format!("        x = jax.vmap(self.{})(x)\n", node.id));
            }
            OperatorKind::RMSNorm { .. } | OperatorKind::LayerNorm { .. } => {
                s.push_str(&format!("        x = jax.vmap(self.{})(x)\n", node.id));
            }
            OperatorKind::Activation { kind } => {
                let jax_act = match kind {
                    shared::spec::ActivationKind::GELU => "jax.nn.gelu",
                    shared::spec::ActivationKind::SiLU => "jax.nn.silu",
                    shared::spec::ActivationKind::ReLU => "jax.nn.relu",
                    _ => "jax.nn.silu",
                };
                s.push_str(&format!("        x = {jax_act}(x)\n"));
            }
            _ => {}
        }
    }
    s.push_str("        return x\n");
    s
}

pub fn generate_jax_train(_spec: &AiDesignSpec) -> String {
    r#"import jax
import jax.numpy as jnp
import optax
from model import CustomModel

def train():
    key = jax.random.PRNGKey(42)
    model = CustomModel(key)
    optimizer = optax.adamw(learning_rate=1e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    print("[+] JAX model and Optax optimizer initialized.")

if __name__ == "__main__":
    train()
"#
    .to_string()
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
