//! GGUF Conversion Bundle Exporter: generates GGUF metadata manifests, Python conversion scripts
//! (`convert.py`), and KoboldCpp/llama.cpp inference launching harnesses from `AiDesignSpec`.
//!
//! Note: This currently emits a conversion bundle (`manifest.json`, `convert.py`, `kobold_launch.bat`,
//! `README.md`) rather than a raw binary `.gguf` tensor file; native binary `.gguf` serialization
//! via `sml-runtime` is scheduled for Wave 2.

use shared::spec::{AiDesignSpec, OperatorKind};
use std::collections::HashMap;

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

2. **Inference**:
   - Run `kobold_launch.bat` with KoboldCpp or pass `model.gguf` to `llama.cpp`.
   - Native binary `.gguf` emission and execution via `sml-runtime` is scheduled for Wave 2.
"#,
        spec.metadata.model_id
    )
}
