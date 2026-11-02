//! Automated partitioning of model parameters into Muon 2D GEMMs, AdamW 1D/heads, and 0-WD groups.

use crate::spec::AiDesignSpec;
use crate::spec::operators::OperatorKind;
use serde::{Deserialize, Serialize};

/// Categorical optimizer group assignment based on tensor dimensionality and inductive bias.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OptimizerGroup {
    /// Internal 2D weight matrices (d_out >= 64, d_in >= 64) updated via Newton-Schulz orthogonalization.
    Muon2DGEMM,
    /// Token embeddings, small 1D vectors, and terminal readout projection heads.
    AdamW1D,
    /// Scale vectors (RMSNorm/LayerNorm), biases, router load-balancing biases, and KAN spline knots.
    ZeroWeightDecay,
}

/// Metadata and parameter configuration for an individual learnable tensor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParamDescriptor {
    pub name: String,
    pub node_id: String,
    pub shape: Vec<usize>,
    pub param_count: usize,
    pub group: OptimizerGroup,
    pub suggested_lr: f64,
    pub suggested_weight_decay: f64,
}

/// Consolidated parameter partitioning report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PartitionReport {
    pub muon_params: Vec<ParamDescriptor>,
    pub adamw_params: Vec<ParamDescriptor>,
    pub zero_wd_params: Vec<ParamDescriptor>,
    pub total_param_count: usize,
    pub muon_param_count: usize,
    pub adamw_param_count: usize,
    pub zero_wd_param_count: usize,
    pub muon_ratio: f32,
}

impl PartitionReport {
    pub fn new() -> Self {
        Self {
            muon_params: Vec::new(),
            adamw_params: Vec::new(),
            zero_wd_params: Vec::new(),
            total_param_count: 0,
            muon_param_count: 0,
            adamw_param_count: 0,
            zero_wd_param_count: 0,
            muon_ratio: 0.0,
        }
    }
}

impl Default for PartitionReport {
    fn default() -> Self {
        Self::new()
    }
}

/// Inspects an AiDesignSpec and computes the three-way optimizer parameter partitioning.
pub fn partition_parameters(spec: &AiDesignSpec) -> PartitionReport {
    let mut report = PartitionReport::new();

    let muon_base_lr = spec.training.muon.lr;
    let adamw_base_lr = spec.training.adamw.lr;
    let adamw_wd = spec.training.adamw.weight_decay;

    let node_count = spec.graph.nodes.len();

    for (node_idx, node) in spec.graph.nodes.iter().enumerate() {
        let is_last_node = node_idx == node_count - 1;

        match &node.operator {
            OperatorKind::Linear {
                in_features,
                out_features,
                bias,
            } => {
                let is_gemm_large = *in_features >= 64 && *out_features >= 64;
                let weight_count = in_features * out_features;

                // Terminal readout head goes to AdamW to preserve calibrated loss scale
                if is_last_node || !is_gemm_large {
                    report.adamw_params.push(ParamDescriptor {
                        name: format!("{}.weight", node.id),
                        node_id: node.id.clone(),
                        shape: vec![*out_features, *in_features],
                        param_count: weight_count,
                        group: OptimizerGroup::AdamW1D,
                        suggested_lr: adamw_base_lr,
                        suggested_weight_decay: adamw_wd,
                    });
                } else {
                    report.muon_params.push(ParamDescriptor {
                        name: format!("{}.weight", node.id),
                        node_id: node.id.clone(),
                        shape: vec![*out_features, *in_features],
                        param_count: weight_count,
                        group: OptimizerGroup::Muon2DGEMM,
                        suggested_lr: muon_base_lr,
                        suggested_weight_decay: 0.0,
                    });
                }

                if *bias {
                    report.zero_wd_params.push(ParamDescriptor {
                        name: format!("{}.bias", node.id),
                        node_id: node.id.clone(),
                        shape: vec![*out_features],
                        param_count: *out_features,
                        group: OptimizerGroup::ZeroWeightDecay,
                        suggested_lr: adamw_base_lr,
                        suggested_weight_decay: 0.0,
                    });
                }
            }

            OperatorKind::Embedding {
                num_embeddings,
                embedding_dim,
            } => {
                let count = num_embeddings * embedding_dim;
                report.adamw_params.push(ParamDescriptor {
                    name: format!("{}.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![*num_embeddings, *embedding_dim],
                    param_count: count,
                    group: OptimizerGroup::AdamW1D,
                    suggested_lr: adamw_base_lr,
                    suggested_weight_decay: adamw_wd,
                });
            }

            OperatorKind::MHA {
                num_heads,
                d_k,
                d_v,
                ..
            } => {
                let d_model = num_heads * d_k;
                let qkv_weights = 3 * d_model * d_model;
                let out_weights = (num_heads * d_v) * d_model;

                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.qkv_proj.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![3 * d_model, d_model],
                    param_count: qkv_weights,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });

                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.out_proj.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![d_model, num_heads * d_v],
                    param_count: out_weights,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::MLA {
                num_heads,
                d_latent_kv,
                d_rope,
                d_v,
            } => {
                let d_model = num_heads * d_v;
                let q_down_count = d_model * d_latent_kv;
                let kv_down_count = d_model * (d_latent_kv + d_rope);
                let out_count = d_model * (num_heads * d_v);

                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.q_down.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![*d_latent_kv, d_model],
                    param_count: q_down_count,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });

                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.kv_down.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![d_latent_kv + d_rope, d_model],
                    param_count: kv_down_count,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });

                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.out_proj.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![d_model, num_heads * d_v],
                    param_count: out_count,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::Mamba2SSD {
                d_model,
                d_state: _,
                d_conv,
                expand,
            } => {
                let d_inner = expand * d_model;
                let in_proj_count = 2 * d_inner * d_model;
                let out_proj_count = d_model * d_inner;
                let conv_count = d_inner * d_conv;
                let dt_count = d_inner;

                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.in_proj.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![2 * d_inner, *d_model],
                    param_count: in_proj_count,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });

                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.out_proj.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![*d_model, d_inner],
                    param_count: out_proj_count,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });

                report.zero_wd_params.push(ParamDescriptor {
                    name: format!("{}.conv1d.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![d_inner, 1, *d_conv],
                    param_count: conv_count,
                    group: OptimizerGroup::ZeroWeightDecay,
                    suggested_lr: adamw_base_lr,
                    suggested_weight_decay: 0.0,
                });

                report.zero_wd_params.push(ParamDescriptor {
                    name: format!("{}.dt_bias", node.id),
                    node_id: node.id.clone(),
                    shape: vec![dt_count],
                    param_count: dt_count,
                    group: OptimizerGroup::ZeroWeightDecay,
                    suggested_lr: adamw_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::RMSNorm { dim, .. } | OperatorKind::LayerNorm { dim, .. } => {
                report.zero_wd_params.push(ParamDescriptor {
                    name: format!("{}.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![*dim],
                    param_count: *dim,
                    group: OptimizerGroup::ZeroWeightDecay,
                    suggested_lr: adamw_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::KANLayer {
                in_features,
                out_features,
                grid_size,
                spline_order,
            } => {
                let base_weights = in_features * out_features;
                let spline_weights = in_features * out_features * (grid_size + spline_order);

                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.base_weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![*out_features, *in_features],
                    param_count: base_weights,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });

                // Spline knot coefficients receive zero weight decay to prevent collapsing grid basis
                report.zero_wd_params.push(ParamDescriptor {
                    name: format!("{}.spline_weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![*out_features, *in_features, grid_size + spline_order],
                    param_count: spline_weights,
                    group: OptimizerGroup::ZeroWeightDecay,
                    suggested_lr: adamw_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::DeepSeekMoE {
                total_experts,
                shared_experts,
                d_model,
                d_ffn,
                ..
            } => {
                let expert_gate_count = total_experts * d_model;
                let per_expert_ffn = 2 * d_model * d_ffn; // gate + up + down
                let total_routed_ffn = (total_experts + shared_experts) * per_expert_ffn;

                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.expert_weights", node.id),
                    node_id: node.id.clone(),
                    shape: vec![total_experts + shared_experts, 2 * d_ffn, *d_model],
                    param_count: total_routed_ffn,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });

                // Router gating logits
                report.zero_wd_params.push(ParamDescriptor {
                    name: format!("{}.router_gate.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![*total_experts, *d_model],
                    param_count: expert_gate_count,
                    group: OptimizerGroup::ZeroWeightDecay,
                    suggested_lr: adamw_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::GQA {
                num_q_heads,
                num_kv_heads,
                d_k,
            } => {
                let d_model = num_q_heads * d_k;
                let kv_dim = num_kv_heads * d_k;
                let qkv_weights = d_model * (d_model + 2 * kv_dim);
                let out_weights = d_model * d_model;

                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.qkv_proj.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![d_model + 2 * kv_dim, d_model],
                    param_count: qkv_weights,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });
                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.out_proj.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![d_model, d_model],
                    param_count: out_weights,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::DiffAttn { num_heads, d_k }
            | OperatorKind::LinearAttention { num_heads, d_k, .. } => {
                let d_model = num_heads * d_k;
                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.qkv_proj.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![3 * d_model, d_model],
                    param_count: 3 * d_model * d_model,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });
                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.out_proj.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![d_model, d_model],
                    param_count: d_model * d_model,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::SwiGLU {
                in_features,
                hidden_features,
            }
            | OperatorKind::GeGLU {
                in_features,
                hidden_features,
            } => {
                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.gate_up_proj.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![2 * hidden_features, *in_features],
                    param_count: 2 * in_features * hidden_features,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });
                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.down_proj.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![*in_features, *hidden_features],
                    param_count: in_features * hidden_features,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::AdaLNZeroDiT {
                d_model,
                cond_dim,
                dual_stream_mmdit,
                ..
            } => {
                let mult = if *dual_stream_mmdit { 2 } else { 1 };
                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.mmdit_qkv_out.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![mult * 4 * d_model, *d_model],
                    param_count: mult * 4 * d_model * d_model,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });
                report.zero_wd_params.push(ParamDescriptor {
                    name: format!("{}.adaln_modulation.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![mult * 6 * d_model, *cond_dim],
                    param_count: mult * 6 * d_model * cond_dim,
                    group: OptimizerGroup::ZeroWeightDecay,
                    suggested_lr: adamw_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::MultiTokenPrediction {
                d_model,
                num_predict_tokens,
                ..
            } => {
                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.mtp_chain_proj.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![*num_predict_tokens, 2 * d_model, *d_model],
                    param_count: num_predict_tokens * 2 * d_model * d_model,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::xLSTM { d_model, .. } => {
                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.qkv_out.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![4 * d_model, *d_model],
                    param_count: 4 * d_model * d_model,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::MambaS6 {
                d_model,
                d_conv,
                expand,
                ..
            } => {
                let d_inner = expand * d_model;
                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.in_out_proj.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![3 * d_inner, *d_model],
                    param_count: 3 * d_inner * d_model,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });
                report.zero_wd_params.push(ParamDescriptor {
                    name: format!("{}.conv1d.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![d_inner, *d_conv],
                    param_count: d_inner * d_conv,
                    group: OptimizerGroup::ZeroWeightDecay,
                    suggested_lr: adamw_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::S4D { d_model, d_state } | OperatorKind::LRU { d_model, d_state } => {
                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.ssm_proj.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![*d_model, *d_model],
                    param_count: d_model * d_model,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });
                report.zero_wd_params.push(ParamDescriptor {
                    name: format!("{}.eigen_state", node.id),
                    node_id: node.id.clone(),
                    shape: vec![*d_model, *d_state],
                    param_count: d_model * d_state,
                    group: OptimizerGroup::ZeroWeightDecay,
                    suggested_lr: adamw_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::TitansMemory {
                d_model,
                memory_depth,
            } => {
                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.memory_mlp.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![*memory_depth, *d_model, *d_model],
                    param_count: memory_depth * d_model * d_model,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::LiquidNeuralCfC {
                d_model,
                backbone_units,
            } => {
                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.cfc_synapse.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![3 * backbone_units, *d_model],
                    param_count: 3 * backbone_units * d_model,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });
                report.zero_wd_params.push(ParamDescriptor {
                    name: format!("{}.time_constant_tau", node.id),
                    node_id: node.id.clone(),
                    shape: vec![*d_model],
                    param_count: *d_model,
                    group: OptimizerGroup::ZeroWeightDecay,
                    suggested_lr: adamw_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::SpikingLIF { d_model, .. } => {
                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.synaptic_current.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![*d_model, *d_model],
                    param_count: d_model * d_model,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::HyperbolicPoincareLayer { dim, .. } => {
                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.mobius_weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![*dim, *dim],
                    param_count: dim * dim,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });
                report.zero_wd_params.push(ParamDescriptor {
                    name: format!("{}.poincare_bias", node.id),
                    node_id: node.id.clone(),
                    shape: vec![*dim],
                    param_count: *dim,
                    group: OptimizerGroup::ZeroWeightDecay,
                    suggested_lr: adamw_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::EchoStateNetwork { reservoir_size, .. } => {
                report.zero_wd_params.push(ParamDescriptor {
                    name: format!("{}.reservoir_readout", node.id),
                    node_id: node.id.clone(),
                    shape: vec![*reservoir_size],
                    param_count: *reservoir_size,
                    group: OptimizerGroup::ZeroWeightDecay,
                    suggested_lr: adamw_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::ConvNeXtV2Block {
                dim,
                kernel_size,
                mlp_ratio,
            } => {
                let hidden = (*dim as f32 * mlp_ratio) as usize;
                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.pwconv.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![2 * hidden, *dim],
                    param_count: 2 * hidden * dim,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });
                report.zero_wd_params.push(ParamDescriptor {
                    name: format!("{}.dwconv.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![*dim, *kernel_size, *kernel_size],
                    param_count: dim * kernel_size * kernel_size,
                    group: OptimizerGroup::ZeroWeightDecay,
                    suggested_lr: adamw_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::WTConv {
                in_channels,
                out_channels,
                wavelet_levels,
            } => {
                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.wavelet_proj.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![*out_channels, *in_channels],
                    param_count: out_channels * in_channels * wavelet_levels,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::FourierNeuralOperator { modes, width } => {
                let m = modes.first().copied().unwrap_or(16);
                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.spectral_weights", node.id),
                    node_id: node.id.clone(),
                    shape: vec![m, *width, *width],
                    param_count: m * width * width,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::DeepONet {
                branch_dim,
                trunk_dim,
                basis_dim,
            } => {
                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.branch_trunk.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![*basis_dim, branch_dim + trunk_dim],
                    param_count: (branch_dim + trunk_dim) * basis_dim,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::EGNN { node_dim, edge_dim } => {
                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.equivariant_mlp.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![2 * node_dim, node_dim + edge_dim],
                    param_count: 2 * node_dim * (node_dim + edge_dim),
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::CellularSheafConv {
                node_dim,
                stalk_dim,
            } => {
                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.sheaf_laplacian.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![*node_dim, stalk_dim * stalk_dim],
                    param_count: node_dim * stalk_dim * stalk_dim,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::EvoformerTriangle { d_pair } => {
                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.triangle_mult.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![6 * d_pair, *d_pair],
                    param_count: 6 * d_pair * d_pair,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::InvariantPointAttention {
                num_heads,
                num_points,
            } => {
                let d_ipa = num_heads * num_points * 16;
                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.ipa_proj.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![d_ipa, d_ipa],
                    param_count: d_ipa * d_ipa,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::CRATE { d_model, .. }
            | OperatorKind::DEQ { d_model, .. }
            | OperatorKind::TTT { d_model, .. } => {
                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.implicit_op.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![2 * d_model, *d_model],
                    param_count: 2 * d_model * d_model,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::SlotAttention {
                num_slots,
                slot_dim,
                ..
            } => {
                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.slot_qkv.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![3 * slot_dim, *slot_dim],
                    param_count: 3 * slot_dim * slot_dim + num_slots * slot_dim,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::QKNorm { d_head } => {
                report.zero_wd_params.push(ParamDescriptor {
                    name: format!("{}.qk_scale", node.id),
                    node_id: node.id.clone(),
                    shape: vec![2 * d_head],
                    param_count: 2 * d_head,
                    group: OptimizerGroup::ZeroWeightDecay,
                    suggested_lr: adamw_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::JEPAPredictor {
                d_model,
                predictor_dim,
                ..
            } => {
                let param_count = 2 * d_model * predictor_dim + 4 * predictor_dim * predictor_dim;
                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.jepa_predictor.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![4 * predictor_dim, *predictor_dim],
                    param_count,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::PairformerBlock {
                d_single, d_pair, ..
            } => {
                let param_count = 4 * d_single * d_single + 12 * d_pair * d_pair;
                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.pairformer_trunk.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![4 * d_single, *d_single],
                    param_count,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::WorldModelRSSM {
                d_model,
                stochastic_classes,
                stochastic_dim,
            } => {
                let z_dim = stochastic_classes * stochastic_dim;
                let param_count = 3 * d_model * (d_model + z_dim);
                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.rssm_transition.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![3 * d_model, d_model + z_dim],
                    param_count,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::RWKV7DeltaRule { d_model, .. } => {
                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.rwkv7_delta_proj.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![5 * d_model, *d_model],
                    param_count: 5 * d_model * d_model,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::EntropyBytePatcher {
                d_byte, d_latent, ..
            } => {
                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.byte_cross_attn.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![4 * d_latent, *d_byte],
                    param_count: 4 * d_latent * d_byte,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::ColPaliMaxSim { d_model, proj_dim } => {
                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.late_interaction_proj.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![*proj_dim, *d_model],
                    param_count: d_model * proj_dim,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::SphericalMeshGNN { d_node, d_edge, .. } => {
                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.icosahedral_mesh_mlp.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![3 * d_node, d_node + d_edge],
                    param_count: 3 * d_node * (d_node + d_edge),
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::QuantumVariationalCircuit {
                num_qubits,
                circuit_depth,
                ..
            } => {
                report.zero_wd_params.push(ParamDescriptor {
                    name: format!("{}.pqc_pauli_angles", node.id),
                    node_id: node.id.clone(),
                    shape: vec![*circuit_depth, *num_qubits, 3],
                    param_count: 3 * num_qubits * circuit_depth,
                    group: OptimizerGroup::ZeroWeightDecay,
                    suggested_lr: adamw_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::PhotonicMZIMesh { num_modes, .. } => {
                report.zero_wd_params.push(ParamDescriptor {
                    name: format!("{}.mzi_phase_shifters", node.id),
                    node_id: node.id.clone(),
                    shape: vec![*num_modes, *num_modes],
                    param_count: num_modes * num_modes,
                    group: OptimizerGroup::ZeroWeightDecay,
                    suggested_lr: adamw_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::SimplicialHodgeConv {
                d_node,
                d_edge,
                d_face,
            } => {
                let param_count = 3 * (d_node * d_node + d_edge * d_edge + d_face * d_face);
                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.hodge_laplacian_filters.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![3 * d_node, d_node + d_edge + d_face],
                    param_count,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::ContinuousNormalizingFlow { dim, .. } => {
                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.cnf_vector_field.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![4 * dim, *dim],
                    param_count: 4 * dim * dim,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::EnergyLangevinMCMC { dim, .. } => {
                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.ebm_pcn_synapse.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![2 * dim, *dim],
                    param_count: 2 * dim * dim,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::NeurosymbolicLogicDNC {
                d_model,
                memory_slots,
                ..
            } => {
                let param_count = 4 * d_model * d_model + 2 * d_model * memory_slots;
                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.dnc_controller_ltn.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![4 * d_model, *d_model],
                    param_count,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::MultiAgentQMIX {
                num_agents,
                state_dim,
                mixing_embed_dim,
            } => {
                let param_count = 2 * num_agents * mixing_embed_dim * state_dim;
                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.qmix_hypernet.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![num_agents * mixing_embed_dim, *state_dim],
                    param_count,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::BioOrganoidActiveInference { num_channels, .. } => {
                report.zero_wd_params.push(ParamDescriptor {
                    name: format!("{}.organoid_fep_coupling", node.id),
                    node_id: node.id.clone(),
                    shape: vec![*num_channels, *num_channels],
                    param_count: 2 * num_channels * num_channels,
                    group: OptimizerGroup::ZeroWeightDecay,
                    suggested_lr: adamw_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::AttentionFreeGatedMLP {
                d_model,
                seq_len,
                filter_order,
            } => {
                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.gmlp_channel_proj.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![*d_model * 2, *d_model],
                    param_count: 3 * d_model * d_model * (*filter_order).max(1),
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });
                report.zero_wd_params.push(ParamDescriptor {
                    name: format!("{}.sgu_spatial_filter", node.id),
                    node_id: node.id.clone(),
                    shape: vec![*seq_len, *seq_len],
                    param_count: seq_len * (*filter_order).max(1) * 4,
                    group: OptimizerGroup::ZeroWeightDecay,
                    suggested_lr: adamw_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::RawBitByteMultiScale {
                d_global,
                d_local,
                patch_bytes,
                ..
            } => {
                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.megabyte_global_local.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![*d_global, *d_local],
                    param_count: 4 * d_global * d_global
                        + 2 * d_local * d_local * (*patch_bytes).max(1),
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });
                report.adamw_params.push(ParamDescriptor {
                    name: format!("{}.bit_byte_embed.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![256, *d_local],
                    param_count: 256 * d_local,
                    group: OptimizerGroup::AdamW1D,
                    suggested_lr: adamw_base_lr,
                    suggested_weight_decay: adamw_wd,
                });
            }

            OperatorKind::VoiceCloningCodecTTS {
                d_model,
                num_rvq_codebooks,
                speaker_embed_dim,
            } => {
                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.flow_tts_backbone.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![*d_model, *d_model],
                    param_count: 6 * d_model * d_model + 2 * d_model * speaker_embed_dim,
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });
                report.adamw_params.push(ParamDescriptor {
                    name: format!("{}.rvq_codebooks.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![*num_rvq_codebooks * 1024, *d_model],
                    param_count: num_rvq_codebooks * 1024 * d_model,
                    group: OptimizerGroup::AdamW1D,
                    suggested_lr: adamw_base_lr,
                    suggested_weight_decay: adamw_wd,
                });
            }

            OperatorKind::TabularGBDTVimeSSL {
                num_trees,
                tree_depth,
                ..
            } => {
                let leaves = 1usize << (*tree_depth).min(12);
                report.zero_wd_params.push(ParamDescriptor {
                    name: format!("{}.gbdt_oblivious_thresholds_and_leaves", node.id),
                    node_id: node.id.clone(),
                    shape: vec![*num_trees, leaves],
                    param_count: num_trees * (leaves + *tree_depth * 2),
                    group: OptimizerGroup::ZeroWeightDecay,
                    suggested_lr: adamw_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::SuperSmoothLipschitzLayer {
                dim,
                bjorck_iters,
                sobolev_order,
                ..
            } => {
                report.muon_params.push(ParamDescriptor {
                    name: format!("{}.bjorck_cayley_lipschitz.weight", node.id),
                    node_id: node.id.clone(),
                    shape: vec![*dim, *dim],
                    param_count: 2 * dim * dim + dim * (*sobolev_order + *bjorck_iters),
                    group: OptimizerGroup::Muon2DGEMM,
                    suggested_lr: muon_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            OperatorKind::LearnableQuantizerQAT { dim, .. } => {
                report.zero_wd_params.push(ParamDescriptor {
                    name: format!("{}.lsq_fracbits_step_and_bitwidth", node.id),
                    node_id: node.id.clone(),
                    shape: vec![*dim, 4],
                    param_count: dim * 4 + 16,
                    group: OptimizerGroup::ZeroWeightDecay,
                    suggested_lr: adamw_base_lr,
                    suggested_weight_decay: 0.0,
                });
            }

            _ => {}
        }
    }

    report.muon_param_count = report.muon_params.iter().map(|p| p.param_count).sum();
    report.adamw_param_count = report.adamw_params.iter().map(|p| p.param_count).sum();
    report.zero_wd_param_count = report.zero_wd_params.iter().map(|p| p.param_count).sum();

    report.total_param_count =
        report.muon_param_count + report.adamw_param_count + report.zero_wd_param_count;
    if report.total_param_count > 0 {
        report.muon_ratio = report.muon_param_count as f32 / report.total_param_count as f32;
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::graph::NodeSpec;

    #[test]
    fn test_optimizer_partitioning_rules() {
        let mut spec = AiDesignSpec::new("test_opt", "Test Optimizer Model");
        spec.graph.nodes.push(NodeSpec {
            id: "embed".to_string(),
            name: "Embedding".to_string(),
            operator: OperatorKind::Embedding {
                num_embeddings: 1000,
                embedding_dim: 128,
            },
            precision: None,
            ..Default::default()
        });
        spec.graph.nodes.push(NodeSpec {
            id: "internal_gemm".to_string(),
            name: "Internal GEMM".to_string(),
            operator: OperatorKind::Linear {
                in_features: 128,
                out_features: 256,
                bias: true,
            },
            precision: None,
            ..Default::default()
        });
        spec.graph.nodes.push(NodeSpec {
            id: "norm".to_string(),
            name: "RMS Norm".to_string(),
            operator: OperatorKind::RMSNorm {
                dim: 256,
                eps: 1e-5,
            },
            precision: None,
            ..Default::default()
        });
        spec.graph.nodes.push(NodeSpec {
            id: "head".to_string(),
            name: "Final Readout Head".to_string(),
            operator: OperatorKind::Linear {
                in_features: 256,
                out_features: 10,
                bias: true,
            },
            precision: None,
            ..Default::default()
        });

        let report = partition_parameters(&spec);
        assert_eq!(report.muon_params.len(), 1);
        assert_eq!(report.muon_params[0].name, "internal_gemm.weight");

        assert_eq!(report.adamw_params.len(), 2);
        assert_eq!(report.adamw_params[0].name, "embed.weight");
        assert_eq!(report.adamw_params[1].name, "head.weight");

        assert_eq!(report.zero_wd_params.len(), 3); // internal_gemm.bias, norm.weight, head.bias
    }
}
