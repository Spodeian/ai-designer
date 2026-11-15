//! Edge & Apple Silicon codegen targets: MLX, Core ML, TFLite/LiteRT, and ExecuTorch.

use crate::spec::AiDesignSpec;
use std::collections::HashMap;

/// Generates edge deployment scripts/modules (`edge/mlx/model.py`, `edge/coreml/convert_coreml.py`,
/// `edge/tflite/convert_tflite.py`, `edge/executorch/export_pte.py`) for `spec`.
pub fn generate_edge_targets(spec: &AiDesignSpec) -> HashMap<String, String> {
    let mut files = HashMap::new();
    let id = &spec.metadata.model_id;
    let nodes = spec.graph.nodes.len();

    files.insert(
        "edge/mlx/model.py".to_string(),
        format!(
            "\"\"\"Apple Silicon MLX target for {id} ({nodes} layers).\"\"\"\nimport mlx.core as mx\nimport mlx.nn as nn\n\n\nclass {id}_MLX(nn.Module):\n    def __init__(self, d_model: int = 256):\n        super().__init__()\n        self.proj = nn.Linear(d_model, d_model)\n\n    def __call__(self, x: mx.array) -> mx.array:\n        return self.proj(x)\n"
        ),
    );

    files.insert(
        "edge/coreml/convert_coreml.py".to_string(),
        format!(
            "\"\"\"Core ML (.mlpackage) export script for {id}.\"\"\"\nimport coremltools as ct\nimport torch\n\ndef export_coreml(traced_model, example_input, out_path=\"{id}.mlpackage\"):\n    mlmodel = ct.convert(traced_model, inputs=[ct.TensorType(shape=example_input.shape)])\n    mlmodel.save(out_path)\n    return out_path\n"
        ),
    );

    files.insert(
        "edge/tflite/convert_tflite.py".to_string(),
        format!(
            "\"\"\"LiteRT / TFLite INT8-ready export script for {id}.\"\"\"\nimport tensorflow as tf\n\ndef export_tflite(saved_model_dir: str, out_path: str = \"{id}.tflite\"):\n    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)\n    converter.optimizations = [tf.lite.Optimize.DEFAULT]\n    tflite_bytes = converter.convert()\n    with open(out_path, \"wb\") as f:\n        f.write(tflite_bytes)\n    return out_path\n"
        ),
    );

    files.insert(
        "edge/executorch/export_pte.py".to_string(),
        format!(
            "\"\"\"ExecuTorch (.pte) AOT export script for {id}.\"\"\"\nimport torch\nfrom executorch.exir import to_edge\n\ndef export_executorch(model, example_args, out_path: str = \"{id}.pte\"):\n    aten_prog = torch.export.export(model, example_args)\n    et_prog = to_edge(aten_prog).to_executorch()\n    with open(out_path, \"wb\") as f:\n        f.write(et_prog.buffer)\n    return out_path\n"
        ),
    );

    files
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_edge_targets_contains_all_four_backends() {
        let spec = AiDesignSpec::new("edge_model", "Edge Model");
        let files = generate_edge_targets(&spec);
        assert!(files.contains_key("edge/mlx/model.py"));
        assert!(files.contains_key("edge/coreml/convert_coreml.py"));
        assert!(files.contains_key("edge/tflite/convert_tflite.py"));
        assert!(files.contains_key("edge/executorch/export_pte.py"));
    }
}
