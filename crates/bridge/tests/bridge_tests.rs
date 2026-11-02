//! Comprehensive Integration Tests for `ai-designer-bridge`.
//! Verifies exporting to PyTorch 2, JAX, Candle, wGPU, SIMD CPU,
//! and tests Candle Bridge layer forward passes.

use ai_designer_bridge::candle_bridge::*;
use ai_designer_bridge::export::*;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use shared::spec::*;

fn create_test_spec() -> AiDesignSpec {
    let mut spec = AiDesignSpec::new("frontier_hybrid", "Frontier Hybrid Model");
    spec.graph.nodes.push(NodeSpec {
        id: "input_linear".to_string(),
        name: "Input Projection".to_string(),
        operator: OperatorKind::Linear {
            in_features: 64,
            out_features: 64,
            bias: false,
        },
        precision: Some(PrecisionFormat::BF16),
        ..Default::default()
    });
    spec.graph.nodes.push(NodeSpec {
        id: "norm_1".to_string(),
        name: "RMS Normalization".to_string(),
        operator: OperatorKind::RMSNorm { dim: 64, eps: 1e-5 },
        ..Default::default()
    });
    spec.graph.nodes.push(NodeSpec {
        id: "norm_2".to_string(),
        name: "Layer Normalization".to_string(),
        operator: OperatorKind::LayerNorm { dim: 64, eps: 1e-5 },
        ..Default::default()
    });
    spec.graph.nodes.push(NodeSpec {
        id: "attn_block".to_string(),
        name: "GQA Attention".to_string(),
        operator: OperatorKind::GQA {
            num_q_heads: 4,
            num_kv_heads: 2,
            d_k: 16,
        },
        ..Default::default()
    });
    spec.graph.nodes.push(NodeSpec {
        id: "ffn_swiglu".to_string(),
        name: "SwiGLU FFN".to_string(),
        operator: OperatorKind::SwiGLU {
            in_features: 64,
            hidden_features: 128,
        },
        ..Default::default()
    });
    spec.graph.nodes.push(NodeSpec {
        id: "act_tanh".to_string(),
        name: "Tanh Activation".to_string(),
        operator: OperatorKind::Activation {
            kind: ActivationKind::Tanh,
        },
        ..Default::default()
    });
    spec
}

#[test]
fn test_export_all_targets() {
    let spec = create_test_spec();
    let artifacts = export_all_targets(&spec);

    // Verify PyTorch 2 files
    assert!(artifacts.files.contains_key("pytorch/model.py"));
    assert!(artifacts.files.contains_key("pytorch/train.py"));
    assert!(
        artifacts
            .files
            .contains_key("pytorch/export_torchscript.py")
    );

    // Verify JAX files
    assert!(artifacts.files.contains_key("jax/model.py"));
    assert!(artifacts.files.contains_key("jax/train.py"));

    // Verify Candle files
    assert!(artifacts.files.contains_key("candle/Cargo.toml"));
    assert!(artifacts.files.contains_key("candle/src/model.rs"));
    assert!(artifacts.files.contains_key("candle/examples/infer.rs"));

    // Verify wGPU files
    assert!(artifacts.files.contains_key("wgpu/shaders.wgsl"));
    assert!(artifacts.files.contains_key("wgpu/pipeline.rs"));

    // Verify SIMD CPU files
    assert!(artifacts.files.contains_key("simd_cpu/cpu_simd.rs"));
}

#[test]
fn test_candle_attention_bridge_layers() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let vb = VarBuilder::zeros(candle_core::DType::F32, &dev);

    let input = Tensor::randn(0.0f32, 1.0f32, (2, 8, 64), &dev)?;

    // 1. MHA
    let mha = CandleMHA::new(64, 4, 16, 16, vb.pp("mha"))?;
    let out_mha = mha.forward(&input)?;
    assert_eq!(out_mha.dims(), &[2, 8, 64]);

    // 2. GQA
    let gqa = CandleGQA::new(64, 4, 2, 16, vb.pp("gqa"))?;
    let out_gqa = gqa.forward(&input)?;
    assert_eq!(out_gqa.dims(), &[2, 8, 64]);

    // 3. MLA
    let mla = CandleMLA::new(64, 4, 32, 16, vb.pp("mla"))?;
    let out_mla = mla.forward(&input)?;
    assert_eq!(out_mla.dims(), &[2, 8, 64]);

    // 4. DiffAttn
    let diff = CandleDiffAttn::new(64, 4, 16, vb.pp("diff"))?;
    let out_diff = diff.forward(&input)?;
    assert_eq!(out_diff.dims(), &[2, 8, 64]);

    // 5. RoPE
    let rope = CandleRoPE::new(16, 128, 10000.0, &dev)?;
    let q_heads = Tensor::randn(0.0f32, 1.0f32, (2, 8, 4, 16), &dev)?;
    let out_rope = rope.apply(&q_heads, 8)?;
    assert_eq!(out_rope.dims(), &[2, 8, 4, 16]);

    Ok(())
}

#[test]
fn test_candle_feedforward_and_moe_bridge_layers() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let vb = VarBuilder::zeros(candle_core::DType::F32, &dev);
    let input = Tensor::randn(0.0f32, 1.0f32, (2, 8, 64), &dev)?;

    // 1. SwiGLU
    let swiglu = CandleSwiGlu::new(64, 128, vb.pp("swiglu"))?;
    let out_swi = swiglu.forward(&input)?;
    assert_eq!(out_swi.dims(), &[2, 8, 64]);

    // 2. GeGLU
    let geglu = CandleGeGlu::new(64, 128, vb.pp("geglu"))?;
    let out_ge = geglu.forward(&input)?;
    assert_eq!(out_ge.dims(), &[2, 8, 64]);

    // 3. Sparse MoE
    let moe = CandleMoE::new(64, 128, 4, 2, vb.pp("moe"))?;
    let out_moe = moe.forward(&input)?;
    assert_eq!(out_moe.dims(), &[2, 8, 64]);

    Ok(())
}

#[test]
fn test_candle_quantized_and_norm_bridge_layers() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let vb = VarBuilder::zeros(candle_core::DType::F32, &dev);
    let input = Tensor::randn(0.0f32, 1.0f32, (2, 8, 64), &dev)?;

    // 1. BitNet 1.58b Linear
    let bitnet = CandleBitNetLinear::new(64, 32, vb.pp("bitnet"))?;
    let out_bitnet = bitnet.forward(&input)?;
    assert_eq!(out_bitnet.dims(), &[2, 8, 32]);

    // 2. Quantized Linear
    let quant = CandleQuantizedLinear::new(64, 32, vb.pp("quant"))?;
    let out_quant = quant.forward(&input)?;
    assert_eq!(out_quant.dims(), &[2, 8, 32]);

    // 3. RMSNorm
    let rms = CandleRmsNorm::new(64, 1e-5, vb.pp("rmsnorm"))?;
    let out_rms = rms.forward(&input)?;
    assert_eq!(out_rms.dims(), &[2, 8, 64]);

    // 4. RevIN
    let revin = CandleRevIn::new(64, 1e-5, vb.pp("revin"))?;
    let (norm_x, mean, std) = revin.normalize(&input)?;
    assert_eq!(norm_x.dims(), &[2, 8, 64]);
    let denorm = revin.denormalize(&norm_x, &mean, &std)?;
    assert_eq!(denorm.dims(), &[2, 8, 64]);

    Ok(())
}

#[test]
fn test_candle_state_space_and_unfolded_bridge_layers() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let vb = VarBuilder::zeros(candle_core::DType::F32, &dev);
    let input = Tensor::randn(0.0f32, 1.0f32, (2, 4, 32), &dev)?;

    // 1. Mamba S6
    let mamba = CandleMamba::new(32, 8, 2, vb.pp("mamba"))?;
    let out_mamba = mamba.forward(&input)?;
    assert_eq!(out_mamba.dims(), &[2, 4, 32]);

    // 2. LISTA
    let lista = CandleLista::new(32, 3, 0.1, vb.pp("lista"))?;
    let out_lista = lista.forward(&input)?;
    assert_eq!(out_lista.dims(), &[2, 4, 32]);

    // 3. Modern Hopfield
    let hopfield = CandleHopfield::new(32, 0.5, vb.pp("hopfield"))?;
    let out_hopfield = hopfield.forward(&input, &input)?;
    assert_eq!(out_hopfield.dims(), &[2, 4, 32]);

    Ok(())
}

#[test]
fn test_candle_dynamic_model_graph_from_spec() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let vb = VarBuilder::zeros(candle_core::DType::F32, &dev);

    let spec = create_test_spec();
    let graph = CandleModelGraph::from_spec(&spec, vb)?;
    assert_eq!(graph.nodes.len(), 6);

    let input = Tensor::randn(0.0f32, 1.0f32, (2, 8, 64), &dev)?;
    let output = graph.forward(input)?;
    assert_eq!(output.dims(), &[2, 8, 64]);

    Ok(())
}
