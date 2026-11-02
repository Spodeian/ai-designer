//! Dynamic Candle Model Graph Builder.
//! Instantiates an executable Candle neural network from an `AiDesignSpec`.

use candle_core::{Result, Tensor};
use candle_nn::{
    Embedding, LayerNorm, LayerNormConfig, Linear, Module, VarBuilder, linear, linear_no_bias,
};
use shared::spec::{AiDesignSpec, OperatorKind};

use super::attention::{CandleDiffAttn, CandleGQA, CandleMHA, CandleMLA};
use super::feedforward::{CandleGeGlu, CandleMoE, CandleSwiGlu};
use super::norm::{CandleRevIn, CandleRmsNorm};
use super::quantized::{CandleBitNetLinear, CandleQuantizedLinear};
use super::state_space::CandleMamba;
use super::unfolded::{CandleHopfield, CandleLista};

/// An instantiated Candle operator node ready for execution.
#[derive(Debug, Clone)]
pub enum CandleModuleNode {
    Linear(Linear),
    MHA(CandleMHA),
    GQA(CandleGQA),
    MLA(CandleMLA),
    DiffAttn(CandleDiffAttn),
    SwiGlu(CandleSwiGlu),
    GeGlu(CandleGeGlu),
    MoE(CandleMoE),
    BitNet(CandleBitNetLinear),
    Quantized(CandleQuantizedLinear),
    RmsNorm(CandleRmsNorm),
    LayerNorm(LayerNorm),
    Embedding(Embedding),
    RevIn(CandleRevIn),
    Mamba(CandleMamba),
    Lista(CandleLista),
    Hopfield(CandleHopfield),
    Gelu,
    Silu,
    Relu,
    Tanh,
    Sigmoid,
    Identity,
}

impl CandleModuleNode {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Linear(l) => l.forward(x),
            Self::MHA(m) => m.forward(x),
            Self::GQA(g) => g.forward(x),
            Self::MLA(mla) => mla.forward(x),
            Self::DiffAttn(d) => d.forward(x),
            Self::SwiGlu(s) => s.forward(x),
            Self::GeGlu(g) => g.forward(x),
            Self::MoE(m) => m.forward(x),
            Self::BitNet(b) => b.forward(x),
            Self::Quantized(q) => q.forward(x),
            Self::RmsNorm(r) => r.forward(x),
            Self::LayerNorm(l) => l.forward(x),
            Self::Embedding(e) => e.forward(x),
            Self::RevIn(r) => {
                let (norm, _, _) = r.normalize(x)?;
                Ok(norm)
            }
            Self::Mamba(m) => m.forward(x),
            Self::Lista(l) => l.forward(x),
            Self::Hopfield(h) => h.forward(x, x),
            Self::Gelu => x.gelu_erf(),
            Self::Silu => candle_nn::ops::silu(x),
            Self::Relu => x.relu(),
            Self::Tanh => x.tanh(),
            Self::Sigmoid => candle_nn::ops::sigmoid(x),
            Self::Identity => Ok(x.clone()),
        }
    }
}

/// An executable Candle neural graph constructed directly from an `AiDesignSpec`.
#[derive(Debug, Clone)]
pub struct CandleModelGraph {
    pub model_id: String,
    pub nodes: Vec<(String, CandleModuleNode)>,
}

impl CandleModelGraph {
    /// Builds an executable Candle module graph from a spec and VarBuilder weights.
    pub fn from_spec(spec: &AiDesignSpec, vb: VarBuilder) -> Result<Self> {
        let mut nodes = Vec::new();

        for node in &spec.graph.nodes {
            let node_vb = vb.pp(&node.id);
            let module = match &node.operator {
                OperatorKind::Linear {
                    in_features,
                    out_features,
                    bias,
                } => {
                    if node.precision == Some(shared::spec::PrecisionFormat::BitNet158b) {
                        CandleModuleNode::BitNet(CandleBitNetLinear::new(
                            *in_features,
                            *out_features,
                            node_vb,
                        )?)
                    } else if node.precision.map(|p| p.is_quantized()).unwrap_or(false) {
                        CandleModuleNode::Quantized(CandleQuantizedLinear::new(
                            *in_features,
                            *out_features,
                            node_vb,
                        )?)
                    } else {
                        let l = if *bias {
                            linear(*in_features, *out_features, node_vb)?
                        } else {
                            linear_no_bias(*in_features, *out_features, node_vb)?
                        };
                        CandleModuleNode::Linear(l)
                    }
                }
                OperatorKind::MHA {
                    num_heads,
                    d_k,
                    d_v,
                    ..
                } => {
                    let d_model = num_heads * d_k;
                    CandleModuleNode::MHA(CandleMHA::new(d_model, *num_heads, *d_k, *d_v, node_vb)?)
                }
                OperatorKind::GQA {
                    num_q_heads,
                    num_kv_heads,
                    d_k,
                } => {
                    let d_model = num_q_heads * d_k;
                    CandleModuleNode::GQA(CandleGQA::new(
                        d_model,
                        *num_q_heads,
                        *num_kv_heads,
                        *d_k,
                        node_vb,
                    )?)
                }
                OperatorKind::MLA {
                    num_heads,
                    d_latent_kv,
                    d_v,
                    ..
                } => {
                    let d_model = num_heads * d_v;
                    CandleModuleNode::MLA(CandleMLA::new(
                        d_model,
                        *num_heads,
                        *d_latent_kv,
                        *d_v,
                        node_vb,
                    )?)
                }
                OperatorKind::DiffAttn { num_heads, d_k } => {
                    let d_model = num_heads * d_k;
                    CandleModuleNode::DiffAttn(CandleDiffAttn::new(
                        d_model, *num_heads, *d_k, node_vb,
                    )?)
                }
                OperatorKind::SwiGLU {
                    in_features,
                    hidden_features,
                } => CandleModuleNode::SwiGlu(CandleSwiGlu::new(
                    *in_features,
                    *hidden_features,
                    node_vb,
                )?),
                OperatorKind::GeGLU {
                    in_features,
                    hidden_features,
                } => CandleModuleNode::GeGlu(CandleGeGlu::new(
                    *in_features,
                    *hidden_features,
                    node_vb,
                )?),
                OperatorKind::DeepSeekMoE {
                    total_experts,
                    routed_k,
                    d_model,
                    d_ffn,
                    ..
                } => CandleModuleNode::MoE(CandleMoE::new(
                    *d_model,
                    *d_ffn,
                    *total_experts,
                    *routed_k,
                    node_vb,
                )?),
                OperatorKind::RMSNorm { dim, eps } => {
                    CandleModuleNode::RmsNorm(CandleRmsNorm::new(*dim, *eps, node_vb)?)
                }
                OperatorKind::LayerNorm { dim, eps } => {
                    let cfg = LayerNormConfig {
                        eps: *eps,
                        ..Default::default()
                    };
                    CandleModuleNode::LayerNorm(candle_nn::layer_norm(*dim, cfg, node_vb)?)
                }
                OperatorKind::Embedding {
                    num_embeddings,
                    embedding_dim,
                } => CandleModuleNode::Embedding(candle_nn::embedding(
                    *num_embeddings,
                    *embedding_dim,
                    node_vb,
                )?),
                OperatorKind::RevINLayer { channels } => {
                    CandleModuleNode::RevIn(CandleRevIn::new(*channels, 1e-5, node_vb)?)
                }
                OperatorKind::Activation { kind } => match kind {
                    shared::spec::ActivationKind::GELU
                    | shared::spec::ActivationKind::QuickGELU => CandleModuleNode::Gelu,
                    shared::spec::ActivationKind::SiLU => CandleModuleNode::Silu,
                    shared::spec::ActivationKind::ReLU => CandleModuleNode::Relu,
                    shared::spec::ActivationKind::Tanh => CandleModuleNode::Tanh,
                    shared::spec::ActivationKind::Sigmoid => CandleModuleNode::Sigmoid,
                    _ => CandleModuleNode::Silu,
                },
                OperatorKind::MambaS6 {
                    d_model,
                    d_state,
                    expand,
                    ..
                } => {
                    CandleModuleNode::Mamba(CandleMamba::new(*d_model, *d_state, *expand, node_vb)?)
                }
                OperatorKind::LISTA {
                    num_unrolled_steps,
                    lambda,
                } => CandleModuleNode::Lista(CandleLista::new(
                    128,
                    *num_unrolled_steps,
                    *lambda,
                    node_vb,
                )?),
                OperatorKind::ModernHopfield { num_heads, beta } => {
                    let dim = num_heads * 64;
                    CandleModuleNode::Hopfield(CandleHopfield::new(dim, *beta, node_vb)?)
                }
                _ => CandleModuleNode::Identity,
            };

            nodes.push((node.id.clone(), module));
        }

        Ok(Self {
            model_id: spec.metadata.model_id.clone(),
            nodes,
        })
    }

    /// Executes a sequential forward pass through the graph.
    pub fn forward(&self, mut x: Tensor) -> Result<Tensor> {
        for (_id, node) in &self.nodes {
            x = node.forward(&x)?;
        }
        Ok(x)
    }
}
