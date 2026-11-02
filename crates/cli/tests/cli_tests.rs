//! Integration tests for AI Designer CLI (`aid`).

use ai_designer_cli::run_cli;
use std::fs;
use std::path::Path;

#[test]
fn test_cli_help_and_version() {
    assert_eq!(run_cli(&["aid".into(), "--help".into()]), Ok(true));
    assert_eq!(run_cli(&["aid".into(), "-h".into()]), Ok(true));
    assert_eq!(run_cli(&["aid".into(), "help".into()]), Ok(true));
    assert_eq!(run_cli(&["aid".into(), "--version".into()]), Ok(true));
    assert_eq!(run_cli(&["aid".into(), "-v".into()]), Ok(true));
}

#[test]
fn test_cli_empty_args_returns_false() {
    assert_eq!(run_cli(&["aid".into()]), Ok(false));
    assert_eq!(run_cli(&["aid".into(), "--gui".into()]), Ok(false));
}

#[test]
fn test_cli_hardware_and_catalog() {
    assert_eq!(run_cli(&["aid".into(), "hardware".into()]), Ok(true));
    assert_eq!(run_cli(&["aid".into(), "catalog".into()]), Ok(true));
}

#[test]
fn test_cli_plan_list() {
    assert_eq!(
        run_cli(&["aid".into(), "plan".into(), "list".into()]),
        Ok(true)
    );
    assert_eq!(run_cli(&["aid".into(), "plan".into()]), Ok(true));
}

#[test]
fn test_cli_new_scaffolding_and_validation() {
    let tmp_dir = std::env::temp_dir().join("aid_cli_test_new");
    fs::create_dir_all(&tmp_dir).unwrap();
    let out_file = tmp_dir.join("test_transformer.yaml");

    let res = run_cli(&[
        "aid".into(),
        "new".into(),
        "cli_test_net".into(),
        "--preset".into(),
        "transformer".into(),
        "--dim".into(),
        "256".into(),
        "--depth".into(),
        "3".into(),
        "--precision".into(),
        "bf16".into(),
        "-o".into(),
        out_file.to_string_lossy().to_string(),
    ]);
    assert_eq!(res, Ok(true));
    assert!(out_file.exists());

    // Lint
    assert_eq!(
        run_cli(&[
            "aid".into(),
            "lint".into(),
            out_file.to_string_lossy().to_string(),
        ]),
        Ok(true)
    );

    // Verify
    assert_eq!(
        run_cli(&[
            "aid".into(),
            "verify".into(),
            out_file.to_string_lossy().to_string(),
        ]),
        Ok(true)
    );

    // Pareto
    assert_eq!(
        run_cli(&[
            "aid".into(),
            "pareto".into(),
            out_file.to_string_lossy().to_string(),
        ]),
        Ok(true)
    );

    // Plan inspect
    assert_eq!(
        run_cli(&[
            "aid".into(),
            "plan".into(),
            out_file.to_string_lossy().to_string(),
        ]),
        Ok(true)
    );

    // Plan preset apply
    let plan_out_file = tmp_dir.join("test_transformer_planned.yaml");
    assert_eq!(
        run_cli(&[
            "aid".into(),
            "plan".into(),
            out_file.to_string_lossy().to_string(),
            "--preset".into(),
            "frontier_llm".into(),
            "--gpus".into(),
            "16".into(),
            "-o".into(),
            plan_out_file.to_string_lossy().to_string(),
        ]),
        Ok(true)
    );
    assert!(plan_out_file.exists());

    // Export to JSON
    let json_out = tmp_dir.join("test_transformer.json");
    assert_eq!(
        run_cli(&[
            "aid".into(),
            "export".into(),
            out_file.to_string_lossy().to_string(),
            "-o".into(),
            json_out.to_string_lossy().to_string(),
        ]),
        Ok(true)
    );
    assert!(json_out.exists());

    // Clean up
    fs::remove_dir_all(&tmp_dir).ok();
}

#[test]
fn test_cli_template_scaffolding() {
    let tmp_dir = std::env::temp_dir().join("aid_cli_test_template");
    fs::create_dir_all(&tmp_dir).unwrap();
    let out_file = tmp_dir.join("scaffolded_deepseek.yaml");

    // Scaffold template by name substring
    let res = run_cli(&[
        "aid".into(),
        "template".into(),
        "deepseek".into(),
        "-o".into(),
        out_file.to_string_lossy().to_string(),
    ]);

    // May succeed if run from repo root or tools/ai_designer
    if Path::new("templates").exists() || Path::new("tools/ai_designer/templates").exists() {
        assert_eq!(res, Ok(true));
        assert!(out_file.exists());
    }

    fs::remove_dir_all(&tmp_dir).ok();
}

#[test]
fn test_cli_train_help_and_status() {
    assert_eq!(
        run_cli(&["aid".into(), "train".into(), "help".into()]),
        Ok(true)
    );
    assert_eq!(
        run_cli(&["aid".into(), "train".into(), "--help".into()]),
        Ok(true)
    );
    assert_eq!(
        run_cli(&["aid".into(), "train".into(), "status".into()]),
        Ok(true)
    );
}

#[test]
fn test_cli_train_setup_and_run() {
    let tmp_dir = std::env::temp_dir().join("aid_cli_test_train");
    fs::create_dir_all(&tmp_dir).unwrap();
    let blueprint_file = tmp_dir.join("mini_model.yaml");
    let workspace_dir = tmp_dir.join("training_workspace");

    // 1. Scaffold blueprint
    let new_res = run_cli(&[
        "aid".into(),
        "new".into(),
        "mini_train_model".into(),
        "--preset".into(),
        "transformer".into(),
        "--dim".into(),
        "128".into(),
        "--depth".into(),
        "2".into(),
        "-o".into(),
        blueprint_file.to_string_lossy().to_string(),
    ]);
    assert_eq!(new_res, Ok(true));
    assert!(blueprint_file.exists());

    // 2. Setup training environment
    let setup_res = run_cli(&[
        "aid".into(),
        "train".into(),
        "setup".into(),
        blueprint_file.to_string_lossy().to_string(),
        "-o".into(),
        workspace_dir.to_string_lossy().to_string(),
    ]);
    assert_eq!(setup_res, Ok(true));
    assert!(workspace_dir.exists());
    assert!(workspace_dir.join("train.py").exists());
    assert!(workspace_dir.join("model.py").exists());
    assert!(workspace_dir.join("setup_env.sh").exists());
    assert!(workspace_dir.join("setup_env.ps1").exists());
    assert!(workspace_dir.join("run_training.sh").exists());
    assert!(workspace_dir.join("run_training.ps1").exists());
    assert!(workspace_dir.join("environment.yml").exists());
    assert!(workspace_dir.join("pyproject.toml").exists());
    assert!(workspace_dir.join("training_config.json").exists());

    // 3. Run mediated training process
    let run_res = run_cli(&[
        "aid".into(),
        "train".into(),
        "run".into(),
        blueprint_file.to_string_lossy().to_string(),
        "--steps".into(),
        "30".into(),
        "--lr-scale".into(),
        "1.25".into(),
        "--clip-pct".into(),
        "88.0".into(),
        "--grokfast".into(),
        "--muon-ns".into(),
        "6".into(),
        "--quant-bits".into(),
        "4.0".into(),
    ]);
    assert_eq!(run_res, Ok(true));

    // Cleanup
    fs::remove_dir_all(&tmp_dir).ok();
}

#[test]
fn test_cli_surgery_commands() {
    let tmp_dir = std::env::temp_dir().join("aid_cli_test_surgery");
    fs::create_dir_all(&tmp_dir).unwrap();
    let blueprint_file = tmp_dir.join("base_model.yaml");

    // Scaffold base model
    assert_eq!(
        run_cli(&[
            "aid".into(),
            "new".into(),
            "surgery_test_net".into(),
            "--preset".into(),
            "transformer".into(),
            "--dim".into(),
            "128".into(),
            "--depth".into(),
            "4".into(),
            "-o".into(),
            blueprint_file.to_string_lossy().to_string(),
        ]),
        Ok(true)
    );
    assert!(blueprint_file.exists());

    // Help
    assert_eq!(
        run_cli(&["aid".into(), "surgery".into(), "--help".into()]),
        Ok(true)
    );
    assert_eq!(
        run_cli(&["aid".into(), "surgery".into(), "help".into()]),
        Ok(true)
    );

    // 1. Abliteration with directional orthogonalization
    let abliterated_ortho = tmp_dir.join("abliterated_ortho.yaml");
    assert_eq!(
        run_cli(&[
            "aid".into(),
            "surgery".into(),
            "abliterate".into(),
            blueprint_file.to_string_lossy().to_string(),
            "--technique".into(),
            "orthogonal".into(),
            "--concept".into(),
            "refusal".into(),
            "--strength".into(),
            "1.2".into(),
            "--guard".into(),
            "-o".into(),
            abliterated_ortho.to_string_lossy().to_string(),
        ]),
        Ok(true)
    );
    assert!(abliterated_ortho.exists());

    // 2. Abliteration with LEACE
    let abliterated_leace = tmp_dir.join("abliterated_leace.yaml");
    assert_eq!(
        run_cli(&[
            "aid".into(),
            "surgery".into(),
            "abliterate".into(),
            blueprint_file.to_string_lossy().to_string(),
            "--technique".into(),
            "leace".into(),
            "--concept".into(),
            "hallucination".into(),
            "-o".into(),
            abliterated_leace.to_string_lossy().to_string(),
        ]),
        Ok(true)
    );
    assert!(abliterated_leace.exists());

    // 3. Model Soup with TIES
    let soup_ties = tmp_dir.join("soup_ties.yaml");
    assert_eq!(
        run_cli(&[
            "aid".into(),
            "surgery".into(),
            "soup".into(),
            blueprint_file.to_string_lossy().to_string(),
            "--strategy".into(),
            "ties".into(),
            "--checkpoints".into(),
            "4".into(),
            "-o".into(),
            soup_ties.to_string_lossy().to_string(),
        ]),
        Ok(true)
    );
    assert!(soup_ties.exists());

    // 4. Model Soup with DARE
    let soup_dare = tmp_dir.join("soup_dare.yaml");
    assert_eq!(
        run_cli(&[
            "aid".into(),
            "surgery".into(),
            "soup".into(),
            blueprint_file.to_string_lossy().to_string(),
            "--strategy".into(),
            "dare".into(),
            "--checkpoints".into(),
            "2".into(),
            "-o".into(),
            soup_dare.to_string_lossy().to_string(),
        ]),
        Ok(true)
    );
    assert!(soup_dare.exists());

    // 5. Model Grafting Frankenmerge depth scaling
    let grafted_frankenmerge = tmp_dir.join("grafted_frankenmerge.yaml");
    assert_eq!(
        run_cli(&[
            "aid".into(),
            "surgery".into(),
            "graft".into(),
            blueprint_file.to_string_lossy().to_string(),
            "--strategy".into(),
            "frankenmerge".into(),
            "--multiplier".into(),
            "1.5".into(),
            "-o".into(),
            grafted_frankenmerge.to_string_lossy().to_string(),
        ]),
        Ok(true)
    );
    assert!(grafted_frankenmerge.exists());

    // 6. Model Grafting Layer Pruning
    let grafted_prune = tmp_dir.join("grafted_prune.yaml");
    assert_eq!(
        run_cli(&[
            "aid".into(),
            "surgery".into(),
            "graft".into(),
            blueprint_file.to_string_lossy().to_string(),
            "--strategy".into(),
            "prune".into(),
            "--multiplier".into(),
            "0.75".into(),
            "-o".into(),
            grafted_prune.to_string_lossy().to_string(),
        ]),
        Ok(true)
    );
    assert!(grafted_prune.exists());

    fs::remove_dir_all(&tmp_dir).ok();
}

#[test]
fn test_cli_workspace_binary_aidws_and_json_lifecycle() {
    let tmp_dir = std::env::temp_dir().join("aid_cli_test_workspace");
    fs::create_dir_all(&tmp_dir).unwrap();

    // 1. List workspaces
    assert_eq!(
        run_cli(&["aid".into(), "workspace".into(), "list".into()]),
        Ok(true)
    );

    // 2. Export default workspace to native binary .aidws
    let bin_out = tmp_dir.join("exported_ws.aidws");
    assert_eq!(
        run_cli(&[
            "aid".into(),
            "workspace".into(),
            "export".into(),
            "default_workspace".into(),
            "-o".into(),
            bin_out.to_string_lossy().to_string(),
        ]),
        Ok(true)
    );
    assert!(bin_out.exists());

    // Validate binary container format
    let bin_bytes = fs::read(&bin_out).unwrap();
    assert!(
        bin_bytes.len() >= 15,
        "Binary .aidws must contain at least 15-byte envelope header"
    );
    assert_eq!(
        &bin_bytes[0..6],
        b"AIDWS\x01",
        "Binary .aidws must start with AIDWS\\x01 magic bytes"
    );
    assert_eq!(
        bin_bytes[6], 0x01,
        "Default aidws format tag must be JsonCompressed (0x01)"
    );

    // 3. Export workspace to explicit JSON format
    let json_out = tmp_dir.join("exported_ws.json");
    assert_eq!(
        run_cli(&[
            "aid".into(),
            "workspace".into(),
            "export".into(),
            "default_workspace".into(),
            "--json".into(),
            "-o".into(),
            json_out.to_string_lossy().to_string(),
        ]),
        Ok(true)
    );
    assert!(json_out.exists());

    // Validate JSON format
    let json_text = fs::read_to_string(&json_out).unwrap();
    let parsed: serde_json::Value =
        serde_json::from_str(&json_text).expect("Exported JSON must be valid JSON");
    assert!(
        parsed.get("id").is_some() || parsed.get("workspaces").is_some(),
        "Exported JSON must represent valid workspace data"
    );

    // 4. Import binary .aidws file
    assert_eq!(
        run_cli(&[
            "aid".into(),
            "workspace".into(),
            "import".into(),
            bin_out.to_string_lossy().to_string(),
        ]),
        Ok(true)
    );

    // 5. Import JSON file
    assert_eq!(
        run_cli(&[
            "aid".into(),
            "workspace".into(),
            "import".into(),
            json_out.to_string_lossy().to_string(),
        ]),
        Ok(true)
    );

    // 5b. Export and import BSON workspace
    let bson_out = tmp_dir.join("exported_ws.bson");
    assert_eq!(
        run_cli(&[
            "aid".into(),
            "workspace".into(),
            "export".into(),
            "default_workspace".into(),
            "--bson".into(),
            "-o".into(),
            bson_out.to_string_lossy().to_string(),
        ]),
        Ok(true)
    );
    assert!(bson_out.exists());
    assert_eq!(
        run_cli(&[
            "aid".into(),
            "workspace".into(),
            "import".into(),
            bson_out.to_string_lossy().to_string(),
        ]),
        Ok(true)
    );

    // 6. Test corrupted binary container handling (bad CRC32)
    let mut corrupt_bytes = bin_bytes.clone();
    let len = corrupt_bytes.len();
    corrupt_bytes[len - 1] ^= 0xFF; // Invalidate payload/checksum
    let corrupt_file = tmp_dir.join("corrupt.aidws");
    fs::write(&corrupt_file, &corrupt_bytes).unwrap();
    let corrupt_res = run_cli(&[
        "aid".into(),
        "workspace".into(),
        "import".into(),
        corrupt_file.to_string_lossy().to_string(),
    ]);
    assert!(
        corrupt_res.is_err(),
        "Corrupted binary container must be rejected"
    );

    fs::remove_dir_all(&tmp_dir).ok();
}
