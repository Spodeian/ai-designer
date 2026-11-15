use shared::format::aid_ron::to_ron_string;
use shared::format::aid_yaml::{from_yaml_str, to_yaml_string};
use shared::format::aidpkg::{create_aidpkg, read_aidpkg};
use std::fs;
use std::path::Path;

#[test]
fn test_all_starter_templates_parse_and_roundtrip() {
    let templates_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("templates");

    assert!(
        templates_dir.exists(),
        "Templates dir must exist: {:?}",
        templates_dir
    );

    let entries = fs::read_dir(&templates_dir).expect("Failed to read templates directory");
    let mut template_count = 0;

    for entry in entries {
        let entry = entry.expect("Valid entry");
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("yaml") {
            println!("Testing template: {:?}", path.file_name().unwrap());
            let yaml_content = fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("Failed to read {:?}: {}", path, e));

            // 1. YAML parse
            let spec = from_yaml_str(&yaml_content)
                .unwrap_or_else(|e| panic!("YAML parsing failed for {:?}: {}", path, e));

            // 2. YAML re-serialize and re-parse
            let re_yaml = to_yaml_string(&spec).expect("YAML serialization failed");
            let re_spec = from_yaml_str(&re_yaml).expect("YAML re-parse failed");
            assert_eq!(spec, re_spec, "YAML round-trip mismatch for {:?}", path);

            // 3. RON serialization & re-parse
            let ron_str = to_ron_string(&spec).expect("RON serialization failed");
            let ron_spec =
                shared::format::aid_ron::from_ron_str(&ron_str).expect("RON re-parse failed");
            assert_eq!(spec, ron_spec, "RON round-trip mismatch for {:?}", path);

            // 4. .aidpkg packing with dummy weights
            let dummy_weights = vec![0xABu8; 16384]; // 16 KB dummy weights
            let pkg_bytes = create_aidpkg(&spec, Some(&dummy_weights), None)
                .unwrap_or_else(|e| panic!("Failed to write aidpkg for {:?}: {}", path, e));

            // 5. .aidpkg unpacking and verification
            let unpacked = read_aidpkg(&pkg_bytes)
                .unwrap_or_else(|e| panic!("Failed to unpack aidpkg for {:?}: {}", path, e));

            assert_eq!(spec, unpacked.spec, "aidpkg spec mismatch for {:?}", path);
            assert_eq!(
                unpacked.weights_bytes.as_deref(),
                Some(dummy_weights.as_slice()),
                "aidpkg weights payload mismatch for {:?}",
                path
            );

            // 6. Shape inference & symbolic dimension propagation
            let shapes = shared::analysis::infer_shapes(&spec);
            assert!(
                shapes.is_valid,
                "Shape inference failed for {:?}: {:?}",
                path, shapes.diagnostics
            );

            // 7. Optimizer partitioning & parameter count (> 0)
            let partition = shared::analysis::partition_parameters(&spec);
            assert!(
                partition.total_param_count > 0,
                "Expected non-zero parameter count for {:?}, got 0",
                path
            );

            // 8. Seed-42 transitive numerical parity & stability verification
            let parity = shared::analysis::verify_transitive_parity(&spec);
            assert!(
                !parity.has_nan_or_inf,
                "NaN/Inf detected during Seed-42 forward pass for {:?}",
                path
            );

            template_count += 1;
        }
    }

    assert_eq!(
        template_count,
        shared::templates::TEMPLATES.len(),
        "Expected template directory count to match TEMPLATES registry"
    );
    assert_eq!(
        template_count, 59,
        "Expected exactly 59 starter & industry templates, found {}",
        template_count
    );
}

struct TempPyDir(std::path::PathBuf);

impl TempPyDir {
    fn new(prefix: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "ai_designer_{prefix}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).expect("create temp py dir");
        Self(dir)
    }
}

impl Drop for TempPyDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn test_topic_metadata_json_roundtrip() {
    let meta = shared::importer::TopicMetadata {
        category: "00_foundations".to_string(),
        slug: "01_linear_neuron".to_string(),
        title: "Linear Neuron and Invertible Transforms".to_string(),
        d_model: 64,
        seq_len: 16,
        parameter_count: 65,
        memory_bound: true,
        parity_tolerance: 1e-5,
    };

    let json = serde_json::to_string(&meta).expect("serialize TopicMetadata");
    let decoded: shared::importer::TopicMetadata =
        serde_json::from_str(&json).expect("deserialize TopicMetadata");
    assert_eq!(meta, decoded);
}

#[test]
fn test_import_from_python_directory_skips_ignored_dirs() {
    let root = TempPyDir::new("py_dir_import");
    let pkg = root.0.join("model");
    let ignored_cache = pkg.join("__pycache__");
    let ignored_hidden = root.0.join(".hidden");
    fs::create_dir_all(&ignored_cache).unwrap();
    fs::create_dir_all(&ignored_hidden).unwrap();

    fs::write(
        pkg.join("01_block.py"),
        "import torch.nn as nn\nclass SubBlock(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(32, 32)\n    def forward(self, x):\n        return self.fc(x)\n",
    )
    .unwrap();
    fs::write(
        pkg.join("02_net.py"),
        "class MainNet(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.norm = nn.RMSNorm(64)\n        self.proj = nn.Linear(64, 128, bias=False)\n    def forward(self, x):\n        return self.proj(self.norm(x))\n",
    )
    .unwrap();
    fs::write(
        ignored_cache.join("cached.py"),
        "class IgnoredCacheNet(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.a = nn.Linear(8, 8)\n        self.b = nn.Linear(8, 8)\n        self.c = nn.Linear(8, 8)\n",
    )
    .unwrap();
    fs::write(
        ignored_hidden.join("secret.py"),
        "class IgnoredHiddenNet(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.a = nn.Linear(8, 8)\n",
    )
    .unwrap();

    let spec = shared::importer::import_from_python_directory(&root.0, None)
        .expect("import multi-file python directory");
    assert_eq!(spec.metadata.model_id, "main_net");
    assert_eq!(spec.graph.nodes.len(), 2);

    let err = shared::importer::import_from_python_directory(&root.0, Some("IgnoredCacheNet"))
        .expect_err("ignored __pycache__ directory must not be scanned");
    assert!(err.contains("IgnoredCacheNet"));
}
