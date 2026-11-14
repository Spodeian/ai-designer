//! GGUF v3 Binary Serializer & Conversion Bundle Exporter: generates binary `.gguf` headers/
//! descriptors via `sml_runtime::format::gguf`, metadata manifests, Python conversion scripts
//! (`convert.py`), and KoboldCpp/llama.cpp inference launching harnesses from `AiDesignSpec`.

use shared::spec::{AiDesignSpec, OperatorKind};
use sml_runtime::format::{
    converter::ModelArchMetadata,
    gguf::{GGUF_DEFAULT_ALIGNMENT, GGUF_MAGIC, GGUF_VERSION, GgufModel, write_gguf_string},
};
use std::collections::HashMap;
use std::io::Cursor;

/// Generates a GGUF conversion bundle including metadata manifest, Python conversion script, and KoboldCpp runner.
pub fn generate_gguf_bundle(spec: &AiDesignSpec) -> HashMap<String, String> {
    let mut files = HashMap::new();
    files.insert(
        "gguf/manifest.json".to_string(),
        generate_gguf_manifest(spec),
    );
    files.insert(
        "gguf/convert.py".to_string(),
        generate_gguf_convert_script(spec),
    );
    files.insert(
        "gguf/kobold_launch.bat".to_string(),
        generate_kobold_launcher(spec),
    );
    files.insert("gguf/README.md".to_string(), generate_gguf_readme(spec));
    files
}

/// Builds a `ModelArchMetadata` descriptor for `sml_runtime::format::converter`.
pub fn build_model_arch_metadata(spec: &AiDesignSpec) -> ModelArchMetadata {
    let (arch, ctx, emb, blk, hd, hdkv) = extract_arch_params(spec);
    ModelArchMetadata {
        architecture: arch,
        context_length: ctx,
        embedding_length: emb,
        block_count: blk,
        head_count: hd,
        head_count_kv: hdkv,
        extra_metadata: HashMap::new(),
    }
}

/// Serializes an `AiDesignSpec` into a valid binary GGUF v3 header + metadata + descriptor tensor buffer.
pub fn serialize_spec_to_gguf_bytes(spec: &AiDesignSpec) -> Result<Vec<u8>, String> {
    let meta = build_model_arch_metadata(spec);
    let arch = &meta.architecture;

    let mut out = Vec::with_capacity(1024);
    out.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    out.extend_from_slice(&GGUF_VERSION.to_le_bytes());

    let tensor_count: u64 = 1;
    let metadata_kv_count: u64 = 8;
    out.extend_from_slice(&tensor_count.to_le_bytes());
    out.extend_from_slice(&metadata_kv_count.to_le_bytes());

    push_kv_str(&mut out, "general.architecture", arch)?;
    push_kv_str(&mut out, "general.name", &spec.metadata.model_id)?;
    push_kv_u32(&mut out, "general.alignment", GGUF_DEFAULT_ALIGNMENT as u32)?;
    push_kv_u32(
        &mut out,
        &format!("{arch}.context_length"),
        meta.context_length as u32,
    )?;
    push_kv_u32(
        &mut out,
        &format!("{arch}.embedding_length"),
        meta.embedding_length as u32,
    )?;
    push_kv_u32(
        &mut out,
        &format!("{arch}.block_count"),
        meta.block_count as u32,
    )?;
    push_kv_u32(
        &mut out,
        &format!("{arch}.attention.head_count"),
        meta.head_count as u32,
    )?;
    push_kv_u32(
        &mut out,
        &format!("{arch}.attention.head_count_kv"),
        meta.head_count_kv as u32,
    )?;

    // Descriptor tensor carrying per-node FLOP/dimension signature in FP32.
    let node_count = spec.graph.nodes.len().max(1);
    write_gguf_string(&mut out, "ai_designer.arch_signature.weight")
        .map_err(|e| format!("GGUF string write error: {e}"))?;
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&(node_count as u64).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // GGML_TYPE_F32
    out.extend_from_slice(&0u64.to_le_bytes()); // Offset relative to aligned data section

    let rem = out.len() % GGUF_DEFAULT_ALIGNMENT;
    if rem != 0 {
        out.resize(out.len() + (GGUF_DEFAULT_ALIGNMENT - rem), 0u8);
    }

    if spec.graph.nodes.is_empty() {
        out.extend_from_slice(&0.0f32.to_le_bytes());
    } else {
        for (idx, node) in spec.graph.nodes.iter().enumerate() {
            let val = (idx + 1) as f32 + (node.operator.compute_flops(1) as f32).ln_1p();
            out.extend_from_slice(&val.to_le_bytes());
        }
    }

    Ok(out)
}

/// Parses binary GGUF v2/v3 bytes using `sml_runtime::format::gguf::GgufModel`.
pub fn parse_gguf_bytes_to_spec_metadata(bytes: &[u8]) -> Result<GgufModel, String> {
    let mut cursor = Cursor::new(bytes);
    GgufModel::read(&mut cursor).map_err(|e| format!("Failed to parse GGUF binary: {e}"))
}

fn push_kv_str(out: &mut Vec<u8>, key: &str, val: &str) -> Result<(), String> {
    write_gguf_string(out, key).map_err(|e| format!("GGUF KV key write error: {e}"))?;
    out.extend_from_slice(&8u32.to_le_bytes()); // GgufType::String
    write_gguf_string(out, val).map_err(|e| format!("GGUF KV val write error: {e}"))?;
    Ok(())
}

fn push_kv_u32(out: &mut Vec<u8>, key: &str, val: u32) -> Result<(), String> {
    write_gguf_string(out, key).map_err(|e| format!("GGUF KV key write error: {e}"))?;
    out.extend_from_slice(&4u32.to_le_bytes()); // GgufType::Uint32
    out.extend_from_slice(&val.to_le_bytes());
    Ok(())
}

/// Extracts architecture dimensions (context length, embedding length, layer count, heads).
fn extract_arch_params(spec: &AiDesignSpec) -> (String, usize, usize, usize, usize, usize) {
    let mut arch_name = "llama".to_string();
    let mut context_len = 2048;
    let mut embed_len = 512;
    let mut num_blocks = 0;
    let mut num_heads = 8;
    let mut num_kv_heads = 4;

    for tag in &spec.metadata.tags {
        if tag.contains("jepa") {
            arch_name = "jepa".to_string();
        } else if tag.contains("dit") || tag.contains("diffusion") {
            arch_name = "dit".to_string();
        }
    }

    for node in &spec.graph.nodes {
        match &node.operator {
            OperatorKind::MHA { num_heads: h, .. } => {
                num_heads = *h;
                num_kv_heads = *h;
                num_blocks += 1;
            }
            OperatorKind::GQA {
                num_q_heads,
                num_kv_heads: kv,
                ..
            } => {
                num_heads = *num_q_heads;
                num_kv_heads = *kv;
                num_blocks += 1;
            }
            OperatorKind::Linear { out_features, .. }
                if *out_features > embed_len && *out_features % 64 == 0 =>
            {
                embed_len = *out_features;
            }
            _ => {}
        }
    }

    if num_blocks == 0 {
        num_blocks = 4;
    }

    for input in &spec.inputs {
        match &input.space {
            shared::spec::TensorSpace::TimeSeries { horizon, .. } => {
                context_len = *horizon;
            }
            shared::spec::TensorSpace::ContinuousVector { dimension, .. }
                if *dimension > 1 && context_len == 2048 =>
            {
                context_len = *dimension;
            }
            _ => {}
        }
    }

    (
        arch_name,
        context_len,
        embed_len,
        num_blocks,
        num_heads,
        num_kv_heads,
    )
}

pub fn generate_gguf_manifest(spec: &AiDesignSpec) -> String {
    let (arch, ctx, emb, blk, hd, hdkv) = extract_arch_params(spec);

    format!(
        r#"{{
  "general.architecture": "{arch}",
  "general.name": "{}",
  "general.file_type": 0,
  "{arch}.context_length": {ctx},
  "{arch}.embedding_length": {emb},
  "{arch}.block_count": {blk},
  "{arch}.attention.head_count": {hd},
  "{arch}.attention.head_count_kv": {hdkv},
  "alignment": 32,
  "description": "Exported from AI Designer Studio for sml-runtime, KoboldCpp, and llama.cpp"
}}
"#,
        spec.metadata.model_id
    )
}

pub fn generate_gguf_convert_script(spec: &AiDesignSpec) -> String {
    let (arch, ctx, emb, blk, hd, hdkv) = extract_arch_params(spec);
    format!(
        r#"#!/usr/bin/env python3
"""
Convert SafeTensors weights to GGUF v3 format for {}
"""

import sys
from pathlib import Path
from safetensors.torch import load_file
import gguf

def convert(safetensors_path: str, gguf_path: str):
    print(f"Loading weights from {{safetensors_path}}...")
    state_dict = load_file(safetensors_path)

    writer = gguf.GGUFWriter(gguf_path, "{arch}")
    writer.add_name("{}")
    writer.add_context_length({ctx})
    writer.add_embedding_length({emb})
    writer.add_block_count({blk})
    writer.add_head_count({hd})
    writer.add_head_count_kv({hdkv})

    for name, tensor in state_dict.items():
        data = tensor.detach().cpu().numpy()
        writer.add_tensor(name, data)

    print(f"Writing GGUF model to {{gguf_path}}...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print("Done!")

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "model.safetensors"
    dst = sys.argv[2] if len(sys.argv) > 2 else "model.gguf"
    convert(src, dst)
"#,
        spec.metadata.model_id, spec.metadata.model_id
    )
}

pub fn generate_kobold_launcher(spec: &AiDesignSpec) -> String {
    let (_, ctx, _, _, _, _) = extract_arch_params(spec);
    format!(
        r#"@echo off
REM KoboldCpp launcher for {}
koboldcpp.exe --model model.gguf --contextsize {} --smart-context --usecublas --skiplauncher
"#,
        spec.metadata.model_id, ctx
    )
}

pub fn generate_gguf_readme(spec: &AiDesignSpec) -> String {
    let (arch, ctx, emb, blk, hd, hdkv) = extract_arch_params(spec);
    format!(
        r#"# GGUF Conversion Bundle for `{}`

This folder contains a GGUF v3 conversion bundle (`manifest.json`, `convert.py`, launcher, and docs) generated by AI Designer Studio to convert SafeTensors weights into binary `.gguf` models.

## Architecture Parameters
- Architecture: `{arch}`
- Context Length: `{ctx}`
- Embedding Dimension: `{emb}`
- Block Count: `{blk}`
- Attention Heads: `{hd}` (KV: `{hdkv}`)

## Usage with Python (`gguf`) & KoboldCpp
1. **Convert SafeTensors weights to `.gguf`**:
   ```bash
   python convert.py model.safetensors model.gguf
   ```

2. **Inference & Native Binary Export**:
   - Use `serialize_spec_to_gguf_bytes` / `parse_gguf_bytes_to_spec_metadata` for native GGUF v3 header and tensor serialization via `sml-runtime`, or run `kobold_launch.bat` / `llama.cpp` with `model.gguf`.
"#,
        spec.metadata.model_id
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use shared::spec::archetypes::{ArchitectureArchetype, scaffold_archetype};
    use shared::spec::precision::PrecisionFormat;

    #[test]
    fn test_binary_gguf_serialization_and_sml_runtime_roundtrip() {
        let spec = scaffold_archetype(
            ArchitectureArchetype::GenerativeLLM,
            "gguf_roundtrip_llm",
            "GGUF Roundtrip LLM",
            512,
            4,
            PrecisionFormat::BF16,
        );
        let bytes = serialize_spec_to_gguf_bytes(&spec).expect("GGUF binary export should succeed");
        let parsed = parse_gguf_bytes_to_spec_metadata(&bytes)
            .expect("sml-runtime GgufModel::read should parse generated binary");

        assert_eq!(parsed.version, GGUF_VERSION);
        assert_eq!(parsed.architecture(), Some("llama"));
        assert!(
            parsed
                .tensors
                .contains_key("ai_designer.arch_signature.weight")
        );
    }
}
