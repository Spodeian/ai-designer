//! Headless CLI Interface for AI Designer (`aid` / `ai-designer`):
//! Batch compilation, packaging, linting, importing, Pareto search, training plan synthesis,
//! and multi-format conversion matrix (.json, .toml, .yaml, .bson, .ron, .aid, .pkg, .aidpkg, .tar.gz).

use shared::surgery::{
    AbliterationPlan, AbliterationTechnique, GraftingPlan, GraftingStrategy, ModelSoupStrategy,
    SoupCheckpoint, SoupMergeRecipe, SurgeryTargetConcept, execute_abliteration,
    execute_layer_graft, execute_soup_merge,
};
use shared::workspace::WorkspaceRegistry;
use shared::{
    AdvisorySeverity, ArchitectureArchetype, BlueprintFileFormat, PkgWeightsFormat,
    PrecisionFormat, REPO_CATEGORIES, analyze_roofline, create_aidpkg, export_project_directory,
    generate_pareto_frontier, import_from_huggingface_config, import_from_python_source,
    import_project_directory, import_spec_from_safetensors, infer_shapes, load_blueprint_auto,
    pack_pkg, parse_safetensors_header, read_aidpkg, read_tar_gz, run_advisor_checks,
    save_blueprint_auto, scaffold_archetype,
    spec::training::{QuantizationTrainingMode, TrainingPlanPreset},
    to_bson_bytes, to_json_string, to_yaml_string, unpack_pkg,
};
use std::fs;
use std::path::Path;

/// Runs the CLI parser and executes subcommands.
/// Returns `Ok(true)` if a CLI command was handled, `Ok(false)` if no command was passed.
pub fn run_cli(args: &[String]) -> Result<bool, String> {
    if args.len() <= 1 {
        return Ok(false);
    }

    let cmd = args[1].as_str();
    if cmd == "--gui" {
        return Ok(false);
    }

    match cmd {
        "help" | "--help" | "-h" => {
            print_help();
            Ok(true)
        }
        "version" | "--version" | "-v" => {
            println!("aid (AI Designer CLI) 0.1.0");
            Ok(true)
        }
        "new" | "init" => {
            if args.len() < 3 {
                return Err("Usage: aid new <model_id> [--preset <blank|transformer|vision|physics|robotics|timeseries|tabular|residual|moe|supersmooth|quantization|tts|rawbyte|gatedmixer|ssl|gbdt>] [--name <display_name>] [--dim <d_model>] [--depth <depth>] [--precision <bf16|fp16|fp8|bitnet|fp32>] [-o <out_file>]".to_string());
            }
            let model_id = &args[2];
            let preset = get_arg_val(args, "--preset").unwrap_or_else(|| "transformer".to_string());
            let display_name = get_arg_val(args, "--name");
            let dim: usize = get_arg_val(args, "--dim")
                .and_then(|s| s.parse().ok())
                .unwrap_or(512);
            let depth: usize = get_arg_val(args, "--depth")
                .and_then(|s| s.parse().ok())
                .unwrap_or(4);
            let prec_str = get_arg_val(args, "--precision").unwrap_or_else(|| "bf16".to_string());
            let out_target = get_arg_val(args, "-o").unwrap_or_else(|| format!("{model_id}.yaml"));
            cmd_new(
                model_id,
                &preset,
                display_name.as_deref(),
                dim,
                depth,
                &prec_str,
                &out_target,
            )?;
            Ok(true)
        }
        "template" | "templates" | "list" => {
            let sub = if cmd == "list" {
                Some("list")
            } else {
                args.get(2).map(|s| s.as_str())
            };
            let out_target = get_arg_val(args, "-o");
            cmd_template(sub, out_target.as_deref())?;
            Ok(true)
        }
        "catalog" => {
            print_catalog();
            Ok(true)
        }
        "hardware" | "info" => {
            print_hardware();
            Ok(true)
        }
        "verify" => {
            if args.len() < 3 {
                return Err("Usage: aid verify <blueprint.yaml | blueprint.aid>".to_string());
            }
            let blueprint_path = &args[2];
            cmd_verify(blueprint_path)?;
            Ok(true)
        }
        "pareto" => {
            if args.len() < 3 {
                return Err("Usage: aid pareto <blueprint.yaml | blueprint.aid>".to_string());
            }
            let blueprint_path = &args[2];
            cmd_pareto(blueprint_path)?;
            Ok(true)
        }
        "export" | "convert" => {
            if args.len() < 3 {
                return Err("Usage: aid export <blueprint.yaml | .aid | .json | .toml | .bson> [-o <out_dir_or_file>]".to_string());
            }
            let blueprint_path = &args[2];
            let out_target = get_arg_val(args, "-o").unwrap_or_else(|| "build_output".to_string());
            cmd_export(blueprint_path, &out_target)?;
            Ok(true)
        }
        "lint" => {
            if args.len() < 3 {
                return Err("Usage: aid lint <blueprint.yaml | blueprint.aid> [--fix]".to_string());
            }
            let blueprint_path = &args[2];
            let fix = args.iter().any(|a| a == "--fix");
            cmd_lint(blueprint_path, fix)?;
            Ok(true)
        }
        "plan" | "training-plan" => {
            if args.len() < 3 || args[2] == "list" {
                cmd_plan_list();
                return Ok(true);
            }
            let blueprint_path = &args[2];
            let preset = get_arg_val(args, "--preset");
            let out_target = get_arg_val(args, "-o");
            let gpus: usize = get_arg_val(args, "--gpus")
                .and_then(|s| s.parse().ok())
                .unwrap_or(8);
            let seq_len: usize = get_arg_val(args, "--seq-len")
                .and_then(|s| s.parse().ok())
                .unwrap_or(4096);
            cmd_plan(
                blueprint_path,
                preset.as_deref(),
                out_target.as_deref(),
                gpus,
                seq_len,
            )?;
            Ok(true)
        }
        "package" => {
            if args.len() < 3 {
                return Err("Usage: aid package <blueprint.yaml | blueprint.aid> [--weights <weights.safetensors>] -o <out.aidpkg | out.pkg | out.tar.gz>".to_string());
            }
            let blueprint_path = &args[2];
            let weights_path = get_arg_val(args, "--weights");
            let out_pkg = get_arg_val(args, "-o").unwrap_or_else(|| "model.aidpkg".to_string());
            cmd_package(blueprint_path, weights_path.as_deref(), &out_pkg)?;
            Ok(true)
        }
        "unpack" => {
            if args.len() < 3 {
                return Err(
                    "Usage: aid unpack <model.aidpkg | model.pkg | archive.tar.gz> [-o <out_dir>]"
                        .to_string(),
                );
            }
            let pkg_path = &args[2];
            let out_dir = get_arg_val(args, "-o").unwrap_or_else(|| "unpacked".to_string());
            cmd_unpack(pkg_path, &out_dir)?;
            Ok(true)
        }
        "import" => {
            if args.len() < 3 {
                return Err("Usage: aid import <src_file_or_dir> -o <out_file_or_dir>".to_string());
            }
            let src_path = &args[2];
            let out_target = get_arg_val(args, "-o").unwrap_or_else(|| "imported.yaml".to_string());
            cmd_import(src_path, &out_target)?;
            Ok(true)
        }
        "audit" => {
            let root_dir = if args.len() >= 3 {
                &args[2]
            } else {
                "../../pytorch2"
            };
            cmd_audit(root_dir)?;
            Ok(true)
        }
        "train" | "training" => {
            if args.len() < 3 || args[2] == "help" || args[2] == "--help" {
                print_train_help();
                return Ok(true);
            }
            let sub = args[2].as_str();
            match sub {
                "setup" | "env" => {
                    if args.len() < 4 {
                        return Err(
                            "Usage: aid train setup <blueprint.yaml | .aid> [-o <out_dir>]"
                                .to_string(),
                        );
                    }
                    let blueprint_path = &args[3];
                    let out_dir =
                        get_arg_val(args, "-o").unwrap_or_else(|| "training_workspace".to_string());
                    cmd_train_setup(blueprint_path, &out_dir)?;
                    Ok(true)
                }
                "run" | "start" => {
                    if args.len() < 4 {
                        return Err("Usage: aid train run <blueprint.yaml | .aid> [--steps <n>] [--lr-scale <f>] [--clip-pct <f>] [--grokfast] [--muon-ns <n>] [--quant-bits <f>] [--optimizer <family>]".to_string());
                    }
                    let blueprint_path = &args[3];
                    let steps: usize = get_arg_val(args, "--steps")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(100);
                    let lr_scale: f64 = get_arg_val(args, "--lr-scale")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1.0);
                    let clip_pct: f64 = get_arg_val(args, "--clip-pct")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(90.0);
                    let grokfast = args.iter().any(|a| a == "--grokfast");
                    let muon_ns: usize = get_arg_val(args, "--muon-ns")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(5);
                    let quant_bits: Option<f64> =
                        get_arg_val(args, "--quant-bits").and_then(|s| s.parse().ok());
                    let opt_family = get_arg_val(args, "--optimizer");
                    let no_auto_tune = args
                        .iter()
                        .any(|a| a == "--no-auto-tune" || a == "--manual");
                    let meta_policy = get_arg_val(args, "--policy")
                        .or_else(|| get_arg_val(args, "--meta-policy"));
                    let weight_decay: Option<f64> =
                        get_arg_val(args, "--weight-decay").and_then(|s| s.parse().ok());
                    let sam_rho: Option<f64> =
                        get_arg_val(args, "--sam-rho").and_then(|s| s.parse().ok());
                    let opts = TrainRunOptions {
                        steps,
                        lr_scale,
                        clip_pct,
                        grokfast,
                        muon_ns,
                        quant_bits,
                        opt_family: opt_family.as_deref(),
                        auto_tune: !no_auto_tune,
                        meta_policy: meta_policy.as_deref(),
                        weight_decay,
                        sam_rho,
                    };
                    cmd_train_run(blueprint_path, opts)?;
                    Ok(true)
                }
                "status" | "info" => {
                    cmd_train_status()?;
                    Ok(true)
                }
                other => Err(format!(
                    "Unknown train subcommand: '{other}'. Pass 'aid train --help' for usage."
                )),
            }
        }
        "surgery" => {
            if args.len() < 3 || args[2] == "help" || args[2] == "--help" {
                print_surgery_help();
                return Ok(true);
            }
            let sub = args[2].as_str();
            match sub {
                "abliterate" => {
                    if args.len() < 4 {
                        return Err("Usage: aid surgery abliterate <blueprint.yaml | .aid> [--technique <orthogonal|leace|repe|circuit_breaker|svd>] [--concept <refusal|hallucination|sycophancy|toxicity>] [--strength <f>] [--guard] [-o <out_yaml>]".to_string());
                    }
                    let blueprint_path = &args[3];
                    let technique = get_arg_val(args, "--technique");
                    let concept = get_arg_val(args, "--concept");
                    let strength: f64 = get_arg_val(args, "--strength")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1.0);
                    let guard = args
                        .iter()
                        .any(|a| a == "--guard" || a == "--retain-capabilities");
                    let out_target = get_arg_val(args, "-o");
                    cmd_surgery_abliterate(
                        blueprint_path,
                        technique.as_deref(),
                        concept.as_deref(),
                        strength,
                        guard,
                        out_target.as_deref(),
                    )?;
                    Ok(true)
                }
                "soup" | "merge" => {
                    if args.len() < 4 {
                        return Err("Usage: aid surgery soup <blueprint.yaml | .aid> [--strategy <uniform|greedy|fisher|slerp|task|ties|dare>] [--checkpoints <n>] [-o <out_yaml>]".to_string());
                    }
                    let blueprint_path = &args[3];
                    let strategy = get_arg_val(args, "--strategy");
                    let checkpoints: usize = get_arg_val(args, "--checkpoints")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(2);
                    let out_target = get_arg_val(args, "-o");
                    cmd_surgery_soup(
                        blueprint_path,
                        strategy.as_deref(),
                        checkpoints,
                        out_target.as_deref(),
                    )?;
                    Ok(true)
                }
                "graft" | "frankenmerge" => {
                    if args.len() < 4 {
                        return Err("Usage: aid surgery graft <blueprint.yaml | .aid> [--strategy <frankenmerge|prune|lora>] [--multiplier <f>] [-o <out_yaml>]".to_string());
                    }
                    let blueprint_path = &args[3];
                    let strategy = get_arg_val(args, "--strategy");
                    let multiplier: f64 = get_arg_val(args, "--multiplier")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1.5);
                    let out_target = get_arg_val(args, "-o");
                    cmd_surgery_graft(
                        blueprint_path,
                        strategy.as_deref(),
                        multiplier,
                        out_target.as_deref(),
                    )?;
                    Ok(true)
                }
                other => Err(format!(
                    "Unknown surgery subcommand: '{other}'. Pass 'aid surgery --help' for usage."
                )),
            }
        }
        "workspace" | "workspaces" | "ws" => {
            let sub = args.get(2).map(|s| s.as_str()).unwrap_or("list");
            match sub {
                "list" => {
                    cmd_workspace_list()?;
                    Ok(true)
                }
                "export" => {
                    let force_json = args.iter().any(|a| a == "--json");
                    let force_bson = args.iter().any(|a| a == "--bson");
                    let ws_id = args
                        .get(3)
                        .filter(|s| !s.starts_with('-'))
                        .map(|s| s.as_str())
                        .unwrap_or("default_workspace");
                    let default_ext = if force_json {
                        "json"
                    } else if force_bson {
                        "bson"
                    } else {
                        "aidws"
                    };
                    let out_file =
                        get_arg_val(args, "-o").unwrap_or_else(|| format!("{ws_id}.{default_ext}"));
                    cmd_workspace_export(ws_id, &out_file, force_json, force_bson)?;
                    Ok(true)
                }
                "import" => {
                    if args.len() < 4 {
                        return Err(
                            "Usage: aid workspace import <bundle.aidws | bundle.json | bundle.bson>".to_string()
                        );
                    }
                    let in_file = &args[3];
                    cmd_workspace_import(in_file)?;
                    Ok(true)
                }
                other => Err(format!(
                    "Unknown workspace subcommand: '{other}'. Options: list, export, import"
                )),
            }
        }
        other => Err(format!(
            "Unknown subcommand: '{other}'. Run 'aid --help' for usage."
        )),
    }
}

pub fn get_arg_val(args: &[String], flag: &str) -> Option<String> {
    for i in 0..args.len().saturating_sub(1) {
        if args[i] == flag {
            return Some(args[i + 1].clone());
        }
    }
    None
}

pub fn print_help() {
    println!(
        r#"
=============================================================================
  AI DESIGNER STUDIO & TRAINING SUITE — STANDALONE CLI ENGINE (aid)
=============================================================================
Usage:
  aid [SUBCOMMAND] [OPTIONS]
  ai-designer [SUBCOMMAND] [OPTIONS]

Subcommands:
  new <model_id> [OPTIONS]
      Scaffolds a new AI architecture from scratch or guided archetype:
      Options:
        --preset <blank|transformer|vision|physics|robotics|timeseries|tabular|residual|moe|supersmooth|quantization|tts|rawbyte|gatedmixer|ssl|gbdt>
        --name <display_name>
        --dim <d_model> (e.g. 512)
        --depth <depth> (e.g. 4)
        --precision <bf16|fp16|fp8|bitnet|fp32>
        -o <out_file> (e.g. model.aid, model.yaml, model.json, model.aidpkg)

  template [list | <name_or_index>] [-o <out_file>]
      Inspects or scaffolds any of the 59 pre-engineered starter blueprints.

  plan <blueprint> [OPTIONS]
      Inspects training plan stages, Chinchilla token budget, parameter
      partitioning (Muon vs AdamW), 4D parallelism, and QAT bitwidth schedule.
      Pass 'aid plan list' to see all 12 training plan presets.
      Options:
        --preset <preset_key>    Apply a pre-configured training curriculum
        --gpus <num_gpus>        Target accelerator count (default: 8)
        --seq-len <tokens>       Context window length (default: 4096)
        -o <out_file>            Write updated blueprint to disk

  train <setup|run|status> [OPTIONS]
      Mediates the training process, setting up turnkey training environments,
      and streaming real-time convergence telemetry with dynamic optimizations:
        aid train setup <blueprint> [-o <out_dir>]  Turnkey workspace generator
        aid train run <blueprint> [OPTIONS]         Mediated execution runner
        aid train status                            Accelerator hardware probe

  surgery <abliterate|soup|graft> <blueprint> [OPTIONS]
      Mechanistic model surgery, refusal abliteration, and manifold weight merging:
        aid surgery abliterate <blueprint> [--technique <orthogonal|leace|repe|circuit_breaker|svd>]
        aid surgery soup <blueprint> [--strategy <uniform|greedy|fisher|slerp|task|ties|dare>]
        aid surgery graft <blueprint> [--strategy <frankenmerge|prune|lora>]

  workspace <list|export|import> [OPTIONS]
      Manages multi-model research workspaces, setups, and experiment portfolios:
        aid workspace list
        aid workspace export <ws_id> [-o <out.aidws | out.json | out.bson>] [--json] [--bson]
        aid workspace import <bundle.aidws | bundle.json | bundle.bson>

  export <blueprint> [-o <out_dir_or_file>]
      Exports to a complete multi-backend project directory (with PyTorch, JAX,
      Candle, WGSL, Triton code) OR directly to a single-file format based on
      extension (.yaml, .json, .toml, .bson, .ron, .aid, .aidpkg, .tar.gz).

  import <src_path> -o <out_target>
      Ingests any supported format (.safetensors, config.json, .yaml, .json,
      .toml, .bson, .ron, .aid, .pkg, .aidpkg, .tar.gz, or project directory)
      and converts it seamlessly to the requested output format.

  package <blueprint> [--weights <weights>] -o <out.aidpkg | out.pkg | out.tar.gz>
      Bundles blueprint and weights into a page-aligned .aidpkg, weights-only
      .pkg container, or compressed .tar.gz archive.

  unpack <pkg_or_archive> [-o <out_dir>]
      Extracts all artifacts (blueprints in YAML/JSON/BSON, manifest, weights)
      from .aidpkg, .pkg, or .tar.gz into a target folder.

  verify <blueprint>
      Executes deterministic Seed-42 forward pass, checking activation stability,
      vanishing/exploding magnitudes, and verifying absence of NaN/Inf traps.

  pareto <blueprint>
      Synthesizes and evaluates 5 hardware-aware Pareto architecture candidates
      (Edge-Minimal, Latency-Optimal, Workstation-Balanced, High-Capacity, Frontier).

  lint <blueprint> [--fix]
      Inspects blueprint against 55+ domain rules with optional AST auto-fix.

  audit [pytorch2_dir]
      Audits repository PyTorch 2 modules and extracts architecture AST statistics.

  hardware, info
      Probes host CPU parallelism, OS, architecture, and live GPU specs via nvidia-smi.

  catalog
      Lists the 18 architectural categories and representative topics in AI-Examples.

  help, --help, -h
      Displays this manual.

  version, -v
      Prints version information.
"#
    );
}

pub fn cmd_new(
    model_id: &str,
    preset: &str,
    display_name: Option<&str>,
    dim: usize,
    depth: usize,
    prec_str: &str,
    out_target: &str,
) -> Result<(), String> {
    let archetype = match preset.to_lowercase().as_str() {
        "blank" | "scratch" | "custom" => ArchitectureArchetype::Blank,
        "transformer" | "llm" | "lm" | "gpt" => ArchitectureArchetype::GenerativeLLM,
        "vision" | "dit" | "diffusion" => ArchitectureArchetype::VisionDiffusion,
        "physics" | "pinn" | "fno" => ArchitectureArchetype::PhysicsOperator,
        "robotics" | "act" | "policy" => ArchitectureArchetype::RoboticsPolicy,
        "timeseries" | "xlstm" | "ts" => ArchitectureArchetype::TimeSeriesForecaster,
        "tabular" | "tabpfn" => ArchitectureArchetype::TabularPriorNetwork,
        "residual" | "deep" | "mlp" => ArchitectureArchetype::DeepResidual,
        "moe" | "sparse" => ArchitectureArchetype::DenseMoE,
        "supersmooth" | "lipschitz" | "sobolev" => ArchitectureArchetype::SuperSmoothLipschitz,
        "quantization" | "qat" | "fracbits" | "lsq" => {
            ArchitectureArchetype::LearnableQuantizationQAT
        }
        "tts" | "voice" | "voicecloning" => ArchitectureArchetype::ZeroShotVoiceCloningTTS,
        "rawbyte" | "byte" | "megabyte" | "mambabyte" => {
            ArchitectureArchetype::RawBitByteMultiScaleLLM
        }
        "gatedmixer" | "gmlp" | "hyena" => ArchitectureArchetype::AttentionFreeGatedMixer,
        "ssl" | "dinov2" | "mae" | "vicreg" => ArchitectureArchetype::SelfSupervisedFoundation,
        "gbdt" | "vime" | "tree" | "trees" => ArchitectureArchetype::TabularGBDTVimeSSL,
        other => {
            return Err(format!(
                "Unknown preset '{other}'. Available: blank, transformer, vision, physics, robotics, timeseries, tabular, residual, moe, supersmooth, quantization, tts, rawbyte, gatedmixer, ssl, gbdt"
            ));
        }
    };

    let precision = match prec_str.to_lowercase().as_str() {
        "bf16" => PrecisionFormat::BF16,
        "fp16" => PrecisionFormat::FP16,
        "fp8" | "fp8_e4m3" => PrecisionFormat::FP8E4M3FN,
        "bitnet" | "bitnet158" => PrecisionFormat::BitNet158b,
        "fp32" => PrecisionFormat::FP32,
        other => {
            return Err(format!(
                "Unknown precision '{other}'. Choose: bf16, fp16, fp8, bitnet, fp32"
            ));
        }
    };

    let name = display_name.unwrap_or_else(|| archetype.display_name());
    let spec = scaffold_archetype(archetype, model_id, name, dim, depth, precision);

    let out_path = Path::new(out_target);
    save_blueprint_auto(&spec, out_path, None)?;

    let _shapes = infer_shapes(&spec);
    let roofline = analyze_roofline(&spec, 8, 256);
    let findings = run_advisor_checks(&spec);

    println!("\n=============================================================================");
    println!("  AI ARCHITECTURE CREATED SUCCESSFULLY");
    println!("=============================================================================");
    println!("  Model ID           : {}", spec.metadata.model_id);
    println!("  Display Title      : {}", spec.metadata.display_name);
    println!(
        "  Archetype Preset   : {:?} ({})",
        archetype,
        archetype.domain()
    );
    println!("  Hidden Dimension   : {}", dim);
    println!("  Layers / Depth     : {}", spec.graph.nodes.len());
    let partition = shared::partition_parameters(&spec);
    let vram_mb = roofline.vram.total_allocated_bytes as f64 / (1024.0 * 1024.0);
    println!("  Total Parameters   : {}", partition.total_param_count);
    println!(
        "  FLOPs / Token      : {:.2} MFLOPs",
        roofline.total_flops_per_token as f64 / 1e6
    );
    println!("  Est. VRAM Footprint: {:.2} MiB", vram_mb);
    println!("  Roofline Regime    : {:?}", roofline.regime);
    println!("  Advisor Checks     : {} findings", findings.len());
    println!("  Output File Saved  : {}", out_path.display());
    println!("=============================================================================\n");

    Ok(())
}

pub fn cmd_template(sub: Option<&str>, out_target: Option<&str>) -> Result<(), String> {
    let mut templates_dir = Path::new("templates");
    let fallback_dir = Path::new("tools/ai_designer/templates");
    if !templates_dir.exists() && fallback_dir.exists() {
        templates_dir = fallback_dir;
    }
    if !templates_dir.exists() {
        return Err(
            "Templates directory not found. Please run from tools/ai_designer directory."
                .to_string(),
        );
    }

    let mut entries = Vec::new();
    if let Ok(rd) = fs::read_dir(templates_dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.extension().and_then(|x| x.to_str()) == Some("yaml") {
                entries.push(p);
            }
        }
    }
    entries.sort();

    match sub {
        None | Some("list") => {
            println!(
                "\n=== Available Starter Blueprints ({} Templates) ===",
                entries.len()
            );
            for (idx, p) in entries.iter().enumerate() {
                let filename = p.file_name().unwrap_or_default().to_string_lossy();
                println!("  [{:02}] {}", idx + 1, filename);
            }
            println!(
                "\nTo scaffold a template, run: aid template <name_or_index> -o <target_file>\n"
            );
            Ok(())
        }
        Some(target) => {
            let found = if let Ok(idx) = target.parse::<usize>() {
                if idx > 0 && idx <= entries.len() {
                    Some(&entries[idx - 1])
                } else {
                    None
                }
            } else {
                entries.iter().find(|p| {
                    let name = p
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_lowercase();
                    name.contains(&target.to_lowercase())
                })
            };

            match found {
                Some(path) => {
                    let content = fs::read_to_string(path).map_err(|e| e.to_string())?;
                    if let Some(out) = out_target {
                        fs::write(out, &content).map_err(|e| e.to_string())?;
                        println!(
                            ">>> Scaffolding template '{}' to '{}' completed.",
                            path.display(),
                            out
                        );
                    } else {
                        print!("{content}");
                    }
                    Ok(())
                }
                None => Err(format!(
                    "Template '{target}' not found. Run 'aid template list' to view available templates."
                )),
            }
        }
    }
}

pub fn cmd_plan_list() {
    println!("\n=== Available Training Plan & Curriculum Presets (12 Presets) ===");
    for (i, p) in TrainingPlanPreset::ALL.iter().enumerate() {
        println!(
            "  [{:02}] {:<35} - {}",
            i + 1,
            format!("{:?}", p),
            p.display_name()
        );
        println!("       Curriculum Domain: {}", p.domain_tag());
    }
    println!("\nUsage:");
    println!("  aid plan <blueprint.yaml>                         # Inspect active training plan");
    println!("  aid plan <blueprint.yaml> --preset <preset_name>  # Apply training preset\n");
}

pub fn cmd_plan(
    blueprint_path: &str,
    preset_name: Option<&str>,
    out_target: Option<&str>,
    gpus: usize,
    seq_len: usize,
) -> Result<(), String> {
    println!(
        ">>> Loading blueprint for training analysis: {}",
        blueprint_path
    );
    let mut spec = load_blueprint_auto(Path::new(blueprint_path))?;

    if let Some(preset_str) = preset_name {
        let preset = match preset_str.to_lowercase().as_str() {
            "frontier_llm" | "llm_4stage" | "frontierllm4stagepipeline" => {
                TrainingPlanPreset::FrontierLlm4StagePipeline
            }
            "chinchilla" | "muon_wsd" | "chinchillamuonwsdfastpath" => {
                TrainingPlanPreset::ChinchillaMuonWsdFastpath
            }
            "flow" | "diffusion" | "generativeflowprogressivecurriculum" => {
                TrainingPlanPreset::GenerativeFlowProgressiveCurriculum
            }
            "sciml" | "physics" | "pinn" | "scimlphysicsntktwostage" => {
                TrainingPlanPreset::SciMlPhysicsNtkTwoStage
            }
            "robotics" | "vla" | "embodiedroboticsvlasaferl" => {
                TrainingPlanPreset::EmbodiedRoboticsVlaSafeRl
            }
            "qlora" | "dpo" | "consumer_gpu" | "consumergpuqloradpo" => {
                TrainingPlanPreset::ConsumerGpuQloraDpo
            }
            "ssl" | "dino" | "mae" | "selfsupervisedvisiondinomae" => {
                TrainingPlanPreset::SelfSupervisedVisionDinoMae
            }
            "tabular" | "gbdt" | "vime" | "tabulargbdtvimeselfsupervised" => {
                TrainingPlanPreset::TabularGbdtVimeSelfSupervised
            }
            "voice" | "tts" | "zeroshotvoicecloningflowtts" => {
                TrainingPlanPreset::ZeroShotVoiceCloningFlowTts
            }
            "rawbyte" | "byte" | "rawbitbytemultiscalellm" => {
                TrainingPlanPreset::RawBitByteMultiScaleLlm
            }
            "supersmooth" | "sobolev" | "lipschitz" | "supersmoothsobolevlipschitzcertified" => {
                TrainingPlanPreset::SuperSmoothSobolevLipschitzCertified
            }
            "quantization" | "qat" | "fracbits" | "learnablequantizationmixedbitwidthqat" => {
                TrainingPlanPreset::LearnableQuantizationMixedBitwidthQat
            }
            other => {
                return Err(format!(
                    "Unknown training preset '{other}'. Run 'aid plan list' to see all 12 options."
                ));
            }
        };

        println!(
            ">>> Applying Training Preset: {:?} (GPUs={}, SeqLen={})",
            preset, gpus, seq_len
        );
        preset.apply_to_spec(&mut spec, gpus, Some(seq_len));

        if let Some(out) = out_target {
            save_blueprint_auto(&spec, Path::new(out), None)?;
            println!(">>> Saved updated training blueprint to: {}", out);
        }
    }

    let partition = shared::partition_parameters(&spec);
    let total_params = partition.total_param_count.max(1) as f64;
    let planned_tokens: u64 = if spec.training.stages.is_empty() {
        spec.training.curriculum.total_training_tokens
    } else {
        spec.training.stages.iter().map(|s| s.token_budget).sum()
    };
    let chinchilla_optimal = (total_params * 20.0) as u64;
    let ratio = planned_tokens as f64 / total_params;
    let total_flops = 6.0 * total_params * (planned_tokens as f64);
    let num_accel = spec.training.parallelism.total_accelerators().max(1) as f64;
    let peak_tflops = spec.metadata.target_hardware.peak_fp16_tflops.max(1.0);
    let cluster_effective_flops = num_accel * peak_tflops * 1e12 * 0.42;
    let est_hours = total_flops / (cluster_effective_flops * 3600.0);

    println!("\n=============================================================================");
    println!("  TRAINING PLAN & MULTI-STAGE COMPUTE BUDGET ANALYSIS");
    println!("=============================================================================");
    println!("  Model ID                   : {}", spec.metadata.model_id);
    println!(
        "  Total Trainable Parameters : {}",
        partition.total_param_count
    );
    println!(
        "  Planned Training Tokens    : {} ({:.2}B)",
        planned_tokens,
        planned_tokens as f64 / 1e9
    );
    println!(
        "  Chinchilla Optimal Tokens  : {} ({:.2}B)",
        chinchilla_optimal,
        chinchilla_optimal as f64 / 1e9
    );
    println!("  Tokens-to-Param Ratio (D/N): {:.1} tokens/param", ratio);
    println!(
        "  Estimated Compute (6ND)    : {:.2e} FLOPs ({:.1} ZettaFLOPs)",
        total_flops,
        total_flops / 1e21
    );
    println!(
        "  Cluster Hardware (42% MFU) : {}x accelerators @ {:.1} TFLOPs",
        num_accel, peak_tflops
    );
    println!(
        "  Estimated Wall-Clock Time  : {:.1} hours ({:.1} days)",
        est_hours,
        est_hours / 24.0
    );

    println!("\n--- Optimizer Parameter Partitioning ---");
    println!(
        "  Optimizer Family           : {:?}",
        spec.training.optimizer_family
    );
    println!(
        "  Muon 2D GEMM Parameters    : {} ({:.1}%)",
        partition.muon_param_count,
        (partition.muon_param_count as f64 / total_params) * 100.0
    );
    println!(
        "  AdamW 1D / Embed Parameters: {} ({:.1}%)",
        partition.adamw_param_count,
        (partition.adamw_param_count as f64 / total_params) * 100.0
    );
    println!(
        "  Zero Weight Decay Params   : {} ({:.1}%)",
        partition.zero_wd_param_count,
        (partition.zero_wd_param_count as f64 / total_params) * 100.0
    );

    println!("\n--- 4D Distributed Parallelism ---");
    println!(
        "  Sharding Strategy          : {:?}",
        spec.training.parallelism.sharding
    );
    println!(
        "  Degrees (DP x TP x PP x CP): {} x {} x {} x {}",
        spec.training.parallelism.data_parallel_degree,
        spec.training.parallelism.tensor_parallel_degree,
        spec.training.parallelism.pipeline_parallel_degree,
        spec.training.parallelism.context_parallel_degree,
    );

    println!("\n--- Differentiable Quantization & QAT ---");
    let q = &spec.training.quantization_training;
    println!("  Quantization Training Mode : {:?}", q.mode);
    if q.mode != QuantizationTrainingMode::FixedPrecision {
        println!(
            "  Target Avg Bitwidth (b_tgt): {:.2} bits",
            q.target_avg_bits
        );
        println!(
            "  Layer Bounds [min, max]    : [{:.1}, {:.1}] bits",
            q.min_layer_bits, q.max_layer_bits
        );
        println!(
            "  BOPs Budget Penalty (λ)    : {:.2e}",
            q.bops_penalty_lambda
        );
        println!("  Soft-STE Annealing Temp (τ): {:.2}", q.ste_temperature);
        println!(
            "  Learn PACT Alpha Clip      : {}",
            q.learn_activation_clip_alpha
        );
    }

    if !spec.training.stages.is_empty() {
        println!(
            "\n--- Multi-Stage Training Pipeline ({} Stages) ---",
            spec.training.stages.len()
        );
        for (idx, stage) in spec.training.stages.iter().enumerate() {
            println!("  [{}] {} ({})", idx + 1, stage.stage_name, stage.stage_id);
            println!("       Paradigm       : {:?}", stage.paradigm);
            println!("       Token Budget   : {} tokens", stage.token_budget);
            println!("       Sequence Length: {} tokens", stage.sequence_length);
            println!("       LR Multiplier  : {:.2}x", stage.lr_multiplier);
            println!("       Freeze Policy  : {:?}", stage.freeze_policy);
        }
    }
    println!("=============================================================================\n");

    Ok(())
}

pub fn print_hardware() {
    let report = shared::analysis::detect_host_hardware();
    println!("\n=== Host Hardware & System Introspection ===");
    println!("  Operating System   : {} ({})", report.os, report.arch);
    println!("  CPU Logical Cores  : {}", report.cpu_logical_cores);
    if report.gpu_detected {
        println!("  GPU Detected       : YES");
        println!(
            "  GPU Model          : {}",
            report.gpu_name.as_deref().unwrap_or("Unknown")
        );
        println!(
            "  GPU VRAM           : {} MiB ({:.2} GiB)",
            report.gpu_vram_mib.unwrap_or(0),
            report.gpu_vram_mib.unwrap_or(0) as f64 / 1024.0
        );
        println!(
            "  Driver Version     : {}",
            report.driver_version.as_deref().unwrap_or("Unknown")
        );
    } else {
        println!("  GPU Detected       : NO (Falling back to Host CPU SIMD)");
    }
    println!("\n--- Recommended Target Roofline Profile ---");
    println!(
        "  Profile Name       : {}",
        report.recommended_profile.profile_name
    );
    println!(
        "  Max VRAM Allocation: {:.2} GiB",
        report.recommended_profile.max_vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    println!(
        "  Peak FP16 Compute  : {:.1} TFLOPs",
        report.recommended_profile.peak_fp16_tflops
    );
    println!(
        "  Peak FP8 Compute   : {:.1} TFLOPs",
        report.recommended_profile.peak_fp8_tflops
    );
    println!(
        "  Memory Bandwidth   : {:.1} GB/s",
        report.recommended_profile.memory_bandwidth_gbs
    );
    println!(
        "  Ridge Point (Ops/B): {:.1} FLOPs/Byte\n",
        report.recommended_profile.ridge_point()
    );
}

pub fn print_catalog() {
    println!("\n=== AI-Examples Architecture & Category Catalog (18 Categories) ===\n");
    for cat in REPO_CATEGORIES {
        println!("[{}] {}", cat.id, cat.name);
        println!("    Description : {}", cat.description);
        println!("    Topics      : {}", cat.representative_topics.join(", "));
        if let Some(starter) = cat.recommended_starter {
            println!("    Starter     : templates/{}", starter);
        }
        println!();
    }
}

pub fn cmd_export(blueprint_path: &str, out_target: &str) -> Result<(), String> {
    println!(">>> Loading blueprint: {}", blueprint_path);
    let spec = load_blueprint_auto(Path::new(blueprint_path))?;

    let out_path = Path::new(out_target);
    let filename = out_path.file_name().and_then(|f| f.to_str()).unwrap_or("");
    let is_single_file = filename.contains('.');

    if is_single_file {
        println!(
            ">>> Exporting blueprint to format based on extension: {}",
            out_path.display()
        );
        save_blueprint_auto(&spec, out_path, None)?;
        println!(">>> Single-file export successful: {}", out_path.display());
    } else {
        println!(
            ">>> Generating complete Multi-Backend Project Directory for '{}'...",
            spec.metadata.model_id
        );
        export_project_directory(&spec, out_path, BlueprintFileFormat::Yaml, None)?;
        println!(
            "\n>>> Export successful! Turnkey multi-backend project written to: {}",
            out_path.display()
        );
    }
    Ok(())
}

pub fn cmd_lint(blueprint_path: &str, fix: bool) -> Result<(), String> {
    println!(">>> Linting blueprint: {}", blueprint_path);
    let mut spec = load_blueprint_auto(Path::new(blueprint_path))?;

    let findings = run_advisor_checks(&spec);
    if findings.is_empty() {
        println!("  [OK] No mathematical or architectural issues detected. Blueprint is clean!");
        return Ok(());
    }

    println!(">>> Found {} diagnostic advisories:", findings.len());
    for (i, finding) in findings.iter().enumerate() {
        let tag = match finding.severity {
            AdvisorySeverity::CriticalMathematicalDefect => "[CRITICAL DEFECT]",
            AdvisorySeverity::Warning => "[WARNING]",
            AdvisorySeverity::BestPractice => "[BEST PRACTICE]",
            AdvisorySeverity::Info => "[INFO]",
        };
        println!("  {} {}. {}", tag, i + 1, finding.title);
        println!("       Rationale: {}", finding.mathematical_rationale);
        if let Some(ref fix_op) = finding.quick_fix {
            println!("       Quick-Fix Available: {:?}", fix_op);
        }
    }

    if fix {
        println!("\n>>> Applying automated AST quick-fixes...");
        let mut fix_count = 0;
        for finding in &findings {
            if let Some(ref mutation) = finding.quick_fix {
                mutation.apply(&mut spec);
                fix_count += 1;
            }
        }
        save_blueprint_auto(&spec, Path::new(blueprint_path), None)?;
        println!(
            "  [OK] Applied {} quick-fixes and updated {}",
            fix_count, blueprint_path
        );
    } else {
        println!("\n>>> Tip: Pass '--fix' to automatically apply recommended mutations.");
    }

    Ok(())
}

pub fn cmd_package(
    blueprint_path: &str,
    weights_path: Option<&str>,
    out_pkg: &str,
) -> Result<(), String> {
    println!(">>> Packaging: {}", blueprint_path);
    let spec = load_blueprint_auto(Path::new(blueprint_path))?;

    let weights_bytes = if let Some(wp) = weights_path {
        println!(">>> Reading weights from: {}", wp);
        Some(fs::read(wp).map_err(|e| format!("Failed to read weights file: {}", e))?)
    } else {
        None
    };

    let out = Path::new(out_pkg);
    if out_pkg.ends_with(".pkg") {
        if let Some(ref w) = weights_bytes {
            let pkg = pack_pkg(w, PkgWeightsFormat::Safetensors, spec.graph.nodes.len());
            fs::write(out, pkg).map_err(|e| format!("Failed to write .pkg: {}", e))?;
            println!(
                ">>> Successfully created page-aligned weights package: {}",
                out_pkg
            );
        } else {
            return Err("Cannot create .pkg weights container without --weights file".to_string());
        }
    } else if out_pkg.ends_with(".tar.gz") || out_pkg.ends_with(".tgz") {
        save_blueprint_auto(&spec, out, weights_bytes.as_deref())?;
        println!(
            ">>> Successfully created compressed TAR.GZ archive: {}",
            out_pkg
        );
    } else {
        let pkg_bytes = create_aidpkg(&spec, weights_bytes.as_deref(), None)
            .map_err(|e| format!("Package creation failed: {}", e))?;
        fs::write(out, &pkg_bytes)
            .map_err(|e| format!("Failed to write package '{}': {}", out_pkg, e))?;
        println!(
            ">>> Successfully created page-aligned binary package: {} ({} bytes)",
            out_pkg,
            pkg_bytes.len()
        );
    }

    Ok(())
}

pub fn cmd_unpack(pkg_path: &str, out_dir: &str) -> Result<(), String> {
    println!(">>> Unpacking: {}", pkg_path);
    let bytes = fs::read(pkg_path).map_err(|e| format!("Failed to read package: {}", e))?;
    let out = Path::new(out_dir);
    fs::create_dir_all(out).map_err(|e| format!("Failed to create out dir: {}", e))?;

    if pkg_path.ends_with(".pkg") {
        let (manifest, weights) = unpack_pkg(&bytes)?;
        let ext = manifest.format.extension();
        let weights_name = format!("weights.{}", ext);
        fs::write(out.join(&weights_name), &weights)
            .map_err(|e| format!("Failed to write weights: {}", e))?;
        println!(
            "  [+] Extracted weights container: {} ({} bytes)",
            weights_name,
            weights.len()
        );
    } else if pkg_path.ends_with(".tar.gz") || pkg_path.ends_with(".tgz") {
        let entries = read_tar_gz(&bytes)?;
        for entry in entries {
            let dest = out.join(&entry.path);
            if let Some(p) = dest.parent() {
                fs::create_dir_all(p).ok();
            }
            fs::write(&dest, &entry.data).map_err(|e| format!("Failed to write: {}", e))?;
            println!("  [+] Extracted from archive: {}", entry.path);
        }
    } else {
        let contents =
            read_aidpkg(&bytes).map_err(|e| format!("Failed to parse package: {}", e))?;

        fs::write(out.join("manifest.json"), &contents.manifest_json)
            .map_err(|e| format!("Failed to write manifest: {}", e))?;
        println!("  [+] Extracted: manifest.json");

        let yaml = to_yaml_string(&contents.spec)
            .map_err(|e| format!("Failed to serialize YAML: {}", e))?;
        fs::write(out.join("blueprint.yaml"), &yaml)
            .map_err(|e| format!("Failed to write YAML blueprint: {}", e))?;
        println!("  [+] Extracted: blueprint.yaml");

        let json = to_json_string(&contents.spec)?;
        fs::write(out.join("blueprint.json"), &json)
            .map_err(|e| format!("Failed to write JSON blueprint: {}", e))?;
        println!("  [+] Extracted: blueprint.json");

        let bson = to_bson_bytes(&contents.spec)?;
        fs::write(out.join("blueprint.bson"), &bson)
            .map_err(|e| format!("Failed to write BSON blueprint: {}", e))?;
        println!("  [+] Extracted: blueprint.bson");

        if let Some(w) = contents.weights_bytes {
            fs::write(out.join("weights.safetensors"), &w)
                .map_err(|e| format!("Failed to write weights: {}", e))?;
            println!(
                "  [+] Extracted: weights.safetensors ({} bytes, 4096-byte page aligned)",
                w.len()
            );
        }
    }

    println!(">>> Unpack complete to directory: {}", out.display());
    Ok(())
}

pub fn cmd_import(src_path: &str, out_target: &str) -> Result<(), String> {
    println!(">>> Ingesting model from: {}", src_path);
    let src = Path::new(src_path);

    let (spec, weights) = if src.is_dir() {
        import_project_directory(src)?
    } else if src_path.ends_with(".safetensors") {
        let bytes = fs::read(src).map_err(|e| format!("Failed to read source file: {}", e))?;
        let parsed_headers = parse_safetensors_header(&bytes)?;
        let model_id = src
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("imported_model");
        (
            import_spec_from_safetensors(model_id, &parsed_headers)?,
            Some(bytes),
        )
    } else if src_path.ends_with("config.json") {
        let text =
            fs::read_to_string(src).map_err(|e| format!("File is not valid UTF-8 text: {}", e))?;
        (import_from_huggingface_config(&text)?, None)
    } else if src_path.ends_with(".py") {
        println!(">>> Parsing Python AST and extracting PyTorch nn.Module architecture...");
        let text =
            fs::read_to_string(src).map_err(|e| format!("File is not valid UTF-8 text: {}", e))?;
        (import_from_python_source(&text, None)?, None)
    } else {
        (load_blueprint_auto(src)?, None)
    };

    let out = Path::new(out_target);
    let filename = out.file_name().and_then(|f| f.to_str()).unwrap_or("");
    if filename.contains('.') {
        save_blueprint_auto(&spec, out, weights.as_deref())?;
    } else {
        export_project_directory(&spec, out, BlueprintFileFormat::Yaml, weights.as_deref())?;
    }

    println!(">>> Ingestion complete! Output written to: {}", out_target);
    Ok(())
}

pub fn cmd_verify(blueprint_path: &str) -> Result<(), String> {
    println!(
        ">>> Verifying Transitive Parity & Numerical Stability: {}",
        blueprint_path
    );
    let spec = load_blueprint_auto(Path::new(blueprint_path))?;
    let report = shared::analysis::verify_transitive_parity(&spec);

    println!("\n=== Parity & Numerical Condition Report ===");
    println!("  Model ID           : {}", report.model_id);
    println!("  Nodes Evaluated    : {}", report.nodes_evaluated);
    println!("  Max Activation     : {:.4e}", report.max_activation);
    println!("  Min Activation     : {:.4e}", report.min_activation);
    println!(
        "  NaN / Inf Traps    : {}",
        if report.has_nan_or_inf {
            "DETECTED (FAIL)"
        } else {
            "None (Clean)"
        }
    );
    println!(
        "  Condition Health   : {}",
        if report.condition_stable {
            "Stable"
        } else {
            "Ill-conditioned"
        }
    );
    println!("  Verification Status: {:?}", report.status);

    println!("\n--- Activation Health Diagnostics ---");
    for diag in &report.diagnostics {
        println!("  [*] {}", diag);
    }
    println!();

    if report.status == shared::analysis::ParityStatus::Failed {
        return Err("Numerical verification failed: model graph contains NaN/Inf or catastrophic instability.".to_string());
    }

    Ok(())
}

pub fn cmd_pareto(blueprint_path: &str) -> Result<(), String> {
    println!(
        ">>> Generating Multi-Objective Pareto Frontier: {}",
        blueprint_path
    );
    let spec = load_blueprint_auto(Path::new(blueprint_path))?;
    let candidates = generate_pareto_frontier(&spec, None);

    println!("\n=== Multi-Objective Pareto Architecture Frontier (5 Candidates) ===");
    println!(
        "{:<24} | {:<12} | {:<12} | {:<12} | {:<12} | {:<8}",
        "Candidate", "Parameters", "FLOPs/Token", "Latency (ms)", "VRAM (MiB)", "OOM Risk"
    );
    println!(
        "{:-<24}-+-{:-<12}-+-{:-<12}-+-{:-<12}-+-{:-<12}-+---------",
        "", "", "", "", ""
    );

    for c in &candidates {
        let vram_mib = c.vram_bytes as f64 / (1024.0 * 1024.0);
        let oom_str = if c.is_oom_risk { "YES (FAIL)" } else { "Safe" };
        println!(
            "{:<24} | {:<12} | {:<12} | {:<12.3} | {:<12.1} | {:<8}",
            c.label, c.total_param_count, c.flops_per_token, c.latency_ms, vram_mib, oom_str
        );
    }
    println!("\nCandidate Profiles:");
    for c in &candidates {
        println!("  [*] {}: {}", c.label, c.description);
    }
    println!();
    Ok(())
}

pub fn cmd_audit(search_dir: &str) -> Result<(), String> {
    println!(">>> Sweeping and auditing models in: {}", search_dir);
    let root = Path::new(search_dir);
    if !root.exists() {
        return Err(format!("Directory '{}' does not exist.", search_dir));
    }

    let mut py_files = Vec::new();
    find_files_recursive(root, &mut py_files);

    if py_files.is_empty() {
        println!(">>> No models found in '{}'.", search_dir);
        return Ok(());
    }

    println!(
        ">>> Discovered {} model candidate files. Running 55+ advisory rules & numerical parity audits...",
        py_files.len()
    );

    let mut audited_count = 0;
    let mut clean_count = 0;
    let mut total_advisories = 0;
    let mut critical_defects = 0;
    let mut parity_passed = 0;

    for path in &py_files {
        let text = match fs::read_to_string(path) {
            Ok(t) => t,
            Err(_) => continue,
        };

        let spec = if path.extension().and_then(|e| e.to_str()) == Some("py") {
            match import_from_python_source(&text, None) {
                Ok(s) => s,
                Err(_) => continue,
            }
        } else {
            match load_blueprint_auto(path) {
                Ok(s) => s,
                Err(_) => continue,
            }
        };

        audited_count += 1;
        let findings = run_advisor_checks(&spec);
        let parity = shared::analysis::verify_transitive_parity(&spec);

        if parity.status == shared::analysis::ParityStatus::Passed {
            parity_passed += 1;
        }

        if findings.is_empty() {
            clean_count += 1;
        } else {
            total_advisories += findings.len();
            for f in &findings {
                if matches!(f.severity, AdvisorySeverity::CriticalMathematicalDefect) {
                    critical_defects += 1;
                }
            }
        }
    }

    println!("\n=============================================================================");
    println!("  AI-EXAMPLES ARCHITECTURE COMPLIANCE & REPOSITORY AUDIT REPORT");
    println!("=============================================================================");
    println!("  Search Directory       : {}", search_dir);
    println!("  Models Audited         : {}", audited_count);
    println!(
        "  Parity Pass Rate       : {} / {} ({:.1}%)",
        parity_passed,
        audited_count,
        (parity_passed as f64 / audited_count.max(1) as f64) * 100.0
    );
    println!(
        "  Clean (0 Advisories)   : {} / {} ({:.1}%)",
        clean_count,
        audited_count,
        (clean_count as f64 / audited_count.max(1) as f64) * 100.0
    );
    println!("  Total Advisories Raised: {}", total_advisories);
    println!("  Critical Defect Alerts : {}", critical_defects);
    println!("=============================================================================\n");

    Ok(())
}

fn find_files_recursive(dir: &Path, list: &mut Vec<std::path::PathBuf>) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                find_files_recursive(&path, list);
            } else if let Some(ext) = path.extension().and_then(|e| e.to_str())
                && (ext == "py" || ext == "yaml")
            {
                list.push(path);
            }
        }
    }
}

pub fn print_train_help() {
    println!(
        r#"
=============================================================================
  AI DESIGNER TRAINING ORCHESTRATION & MEDIATION CLI (aid train)
=============================================================================
Usage:
  aid train [SUBCOMMAND] [OPTIONS]

Subcommands:
  setup <blueprint> [-o <out_dir>]
      Generates a turnkey, production-ready training environment workspace on disk:
        - Virtualenv setup: setup_env.sh (POSIX), setup_env.ps1 (PowerShell)
        - Execution runners: run_training.sh, run_training.ps1
        - Environment dependencies: environment.yml (Conda), pyproject.toml (uv), requirements.txt
        - Generated model code: model.py, train.py, losses.py, data.py, training_config.json

  run <blueprint> [OPTIONS]
      Executes a mediated training process with high-frequency telemetry, dynamic
      convergence modeling, and real-time hyperparameter interventions:
        --steps <n>             Maximum optimization steps (default: 100)
        --lr-scale <factor>     Dynamic learning rate scaling factor (default: 1.0)
        --clip-pct <percentile> AutoClip gradient clipping percentile (default: 90.0)
        --grokfast              Enable Grokfast low-pass EMA gradient filtering
        --muon-ns <steps>       Muon Newton-Schulz matrix polar iterations (default: 5)
        --quant-bits <bits>     Dynamic Quantization-Aware bitwidth target (1.58 - 16.0)
        --optimizer <family>    Optimizer family: muon, adamw, lion, adafactor, shampoo
        --no-auto-tune          Disable automatic meta-parameter optimization (manual mode)
        --policy <policy>       Auto-tuning policy: guard, hypergrad, throughput, plateau, grokking
        --weight-decay <scale>  Dynamic weight decay normative pressure scaling factor (default: 1.0)
        --sam-rho <rho>         Sharpness-Aware Minimization perturbation radius (default: 0.05)

  status
      Reports host acceleration capabilities, memory rooflines, supported optimizers,
      and telemetry channels.
"#
    );
}

pub fn cmd_train_setup(blueprint_path: &str, out_dir: &str) -> Result<(), String> {
    println!("\n[+] Loading blueprint: {blueprint_path}...");
    let spec = load_blueprint_auto(Path::new(blueprint_path))?;
    let target = Path::new(out_dir);

    println!(
        "[+] Generating turnkey training environment in '{}'...",
        target.display()
    );
    let manifest = shared::training::setup_training_environment(&spec, target)?;

    println!("\n=============================================================================");
    println!("  TURNKEY TRAINING WORKSPACE GENERATED");
    println!("=============================================================================");
    println!("  Model ID                   : {}", manifest.model_id);
    println!("  Target Directory           : {}", manifest.target_dir);
    println!(
        "  Total Parameters           : {}",
        manifest.total_parameters
    );
    println!(
        "  Estimated VRAM Allocation  : {:.2} GB",
        manifest.estimated_vram_gb
    );
    println!(
        "  Python Version Req         : {}",
        manifest.python_version_req
    );
    println!(
        "  PyTorch Version Req        : {}",
        manifest.torch_version_req
    );
    println!(
        "  CUDA Hardware Support      : {}",
        if manifest.has_cuda_support {
            "Detected"
        } else {
            "CPU Fallback"
        }
    );
    println!("-----------------------------------------------------------------------------");
    println!(
        "  Created Workspace Files ({} files):",
        manifest.created_files.len()
    );
    for f in &manifest.created_files {
        println!("    • {}", f);
    }
    println!("-----------------------------------------------------------------------------");
    println!("  To initialize and run training:");
    if cfg!(windows) {
        println!("    cd {}", manifest.target_dir);
        println!("    powershell -ExecutionPolicy Bypass -File .\\setup_env.ps1");
        println!("    powershell -ExecutionPolicy Bypass -File .\\run_training.ps1");
    } else {
        println!("    cd {}", manifest.target_dir);
        println!("    chmod +x setup_env.sh run_training.sh");
        println!("    ./setup_env.sh");
        println!("    ./run_training.sh");
    }
    println!("=============================================================================\n");

    Ok(())
}

/// Execution options for mediated live training runs (`aid train run`).
#[derive(Debug, Clone)]
pub struct TrainRunOptions<'a> {
    pub steps: usize,
    pub lr_scale: f64,
    pub clip_pct: f64,
    pub grokfast: bool,
    pub muon_ns: usize,
    pub quant_bits: Option<f64>,
    pub opt_family: Option<&'a str>,
    pub auto_tune: bool,
    pub meta_policy: Option<&'a str>,
    pub weight_decay: Option<f64>,
    pub sam_rho: Option<f64>,
}

impl Default for TrainRunOptions<'_> {
    fn default() -> Self {
        Self {
            steps: 100,
            lr_scale: 1.0,
            clip_pct: 90.0,
            grokfast: false,
            muon_ns: 5,
            quant_bits: None,
            opt_family: None,
            auto_tune: true,
            meta_policy: None,
            weight_decay: None,
            sam_rho: None,
        }
    }
}

pub fn cmd_train_run(blueprint_path: &str, opts: TrainRunOptions<'_>) -> Result<(), String> {
    println!("\n[+] Loading blueprint: {blueprint_path}...");
    let spec = load_blueprint_auto(Path::new(blueprint_path))?;

    let mut session = shared::TrainingSession::new(&spec, opts.steps);

    // Apply live optimization parameters
    session.controls.lr_scale = opts.lr_scale;
    session.controls.clip_percentile = opts.clip_pct;
    session.controls.grokfast_enabled = opts.grokfast;
    session.controls.muon_ns_steps = opts.muon_ns;
    session.meta_tuner.enabled = opts.auto_tune;

    if let Some(wd) = opts.weight_decay {
        session.controls.weight_decay_scale = wd;
    }
    if let Some(rho) = opts.sam_rho {
        session.controls.sam_rho = rho;
    }

    if let Some(p) = opts.meta_policy {
        session.meta_tuner.policy = match p.to_lowercase().as_str() {
            "guard" | "adaptive" => shared::training::MetaTuningPolicy::AdaptiveConvergenceGuard,
            "hd" | "hypergrad" => shared::training::MetaTuningPolicy::HypergradientDescent,
            "throughput" => shared::training::MetaTuningPolicy::ThroughputMaximizer,
            "plateau" => shared::training::MetaTuningPolicy::LossPlateauBreaker,
            "grokking" | "grok" | "omnigrok" => {
                shared::training::MetaTuningPolicy::GrokkingGeneralizer
            }
            _ => {
                return Err(format!(
                    "Unknown meta policy '{p}'. Options: guard, hypergrad, throughput, plateau, grokking"
                ));
            }
        };
    }

    if let Some(qb) = opts.quant_bits {
        session.controls.target_quant_bits = qb;
        session.meta_tuner.target_quant_bits = qb;
    }
    if let Some(opt) = opts.opt_family {
        session.controls.optimizer_family = match opt.to_lowercase().as_str() {
            "muon" | "muonadamw" | "muon-adamw" => {
                shared::spec::training::OptimizerFamily::MuonAdamWPartitioned
            }
            "adamw" | "adam" => shared::spec::training::OptimizerFamily::AdamWStandard,
            "lion" => shared::spec::training::OptimizerFamily::LionSignMomentum,
            "sophia" => shared::spec::training::OptimizerFamily::SophiaSecondOrder,
            "adan" => shared::spec::training::OptimizerFamily::AdanAdaptiveNesterov,
            "lbfgs" => shared::spec::training::OptimizerFamily::LBFGSQuasiNewton,
            "schedule-free" | "schedulefree" => {
                shared::spec::training::OptimizerFamily::ScheduleFreeAdamW
            }
            "paged" | "8bit" | "paged8bit" => {
                shared::spec::training::OptimizerFamily::PagedAdamW8Bit
            }
            "sgd" => shared::spec::training::OptimizerFamily::SGDWithNesterov,
            _ => {
                return Err(format!(
                    "Unknown optimizer family '{opt}'. Options: muon, adamw, lion, sophia, adan, lbfgs, schedule-free, paged8bit, sgd"
                ));
            }
        };
    }

    let partition = shared::analysis::partition_parameters(&spec);

    println!("\n=============================================================================");
    println!("  AI DESIGNER MEDIATED TRAINING SESSION — RUNNING");
    println!("=============================================================================");
    println!("  Model ID                   : {}", session.model_id);
    println!(
        "  Total Parameters           : {}",
        session.total_parameters
    );
    println!(
        "  Parameter Partitioning     : Muon: {} params ({:.1}%) | AdamW: {} params ({:.1}%)",
        partition.muon_param_count,
        (partition.muon_param_count as f64 / session.total_parameters.max(1) as f64) * 100.0,
        partition.adamw_param_count,
        (partition.adamw_param_count as f64 / session.total_parameters.max(1) as f64) * 100.0,
    );
    println!(
        "  Active Optimizer           : {:?}",
        session.controls.optimizer_family
    );
    println!(
        "  Automated Meta-Tuning      : {} (Policy: {:?})",
        if session.meta_tuner.enabled {
            "ENABLED (Auto-optimizing meta parameters)"
        } else {
            "DISABLED (Manual static control)"
        },
        session.meta_tuner.policy
    );
    println!(
        "  AutoClip Gradient Hygiene  : {} (percentile: {:.1}%)",
        if session.controls.auto_clip_enabled {
            "ENABLED"
        } else {
            "DISABLED"
        },
        session.controls.clip_percentile
    );
    println!(
        "  Grokfast Filtering         : {}",
        if session.controls.grokfast_enabled {
            "ENABLED (alpha: 0.98, lamb: 2.0)"
        } else {
            "DISABLED"
        }
    );
    println!(
        "  Muon Newton-Schulz Steps   : {}",
        session.controls.muon_ns_steps
    );
    println!(
        "  Quantization Target Bits   : {:.2} bits/param",
        session.controls.target_quant_bits
    );
    println!(
        "  LR Scaling Multiplier      : {:.2}x",
        session.controls.lr_scale
    );
    println!(
        "  Weight Decay Norm Pressure : {:.2}x (Normative grokking compression)",
        session.controls.weight_decay_scale
    );
    println!(
        "  SAM Flatness Curvature (ρ) : {:.3} (Sharpness-aware minimization)",
        session.controls.sam_rho
    );
    println!(
        "  Grokking Anti-Memorization : {}",
        if session.meta_tuner.policy == shared::training::MetaTuningPolicy::GrokkingGeneralizer {
            "ACTIVE (Auto-driving delayed generalization phase transitions)"
        } else {
            "STANDARD MONITORING"
        }
    );
    println!("  Max Steps Scheduled        : {}", opts.steps);
    println!(
        "-----------------------------------------------------------------------------------------------------------------"
    );
    println!(
        "  Step  | TrainLoss | ValLoss  | GradNorm | WNorm  | SRank | MuonOrtho | Phase       | Throughput  | Stage"
    );
    println!(
        "--------+-----------+----------+----------+--------+-------+-----------+-------------+-------------+-------------------"
    );

    session.start();

    while let Some(telem) = session.step() {
        let print_step = telem.step == 1
            || telem.step.is_multiple_of(10)
            || telem.step == telem.max_steps
            || telem.step == opts.steps;

        if print_step {
            println!(
                "  {:05} | {:9.4} | {:8.4} | {:8.3} | {:6.2} | {:5.2} | {:9.2e} | {:11} | {:6.0} tok/s | [S{}] {}",
                telem.step,
                telem.loss,
                telem.val_loss_estimate,
                telem.grad_norm,
                telem.weight_norm,
                telem.stable_rank,
                telem.muon_ortho_error,
                telem.grokking_phase.badge(),
                telem.tokens_per_sec,
                telem.active_stage_idx + 1,
                telem.active_stage_name
            );
        }

        if let Some(ref action) = telem.meta_tuned_action {
            println!("         ↳ [⚡ AutoMeta] {}", action);
        }
    }

    println!("-----------------------------------------------------------------------------");
    println!("  Validating Transitive Numerical Parity...");
    let parity = shared::analysis::verify_transitive_parity(&spec);

    println!("\n=============================================================================");
    println!("  TRAINING SESSION COMPLETE");
    println!("=============================================================================");
    println!(
        "  Final Train Loss           : {:.5}",
        session.current_telemetry.loss
    );
    println!(
        "  Final Val Loss Estimate    : {:.5}",
        session.current_telemetry.val_loss_estimate
    );
    println!(
        "  Generalization Gap         : {:.5}",
        session.current_telemetry.generalization_gap
    );
    println!(
        "  Grokking Phase Status      : [{}] {}",
        session.current_telemetry.grokking_phase.badge(),
        session.current_telemetry.grokking_phase.label()
    );
    println!(
        "  Weight L2 Norm / Rank      : {:.2} (Stable Rank: {:.2})",
        session.current_telemetry.weight_norm, session.current_telemetry.stable_rank
    );
    println!(
        "  Best Loss Reached          : {:.5}",
        session.current_telemetry.best_loss
    );
    println!(
        "  Peak VRAM Allocated        : {:.1} MB",
        session.current_telemetry.vram_peak_mb
    );
    println!(
        "  Final Muon Ortho Residual  : {:.2e}",
        session.current_telemetry.muon_ortho_error
    );
    println!(
        "  Final AutoClip Ratio       : {:.1}%",
        session.current_telemetry.auto_clip_ratio * 100.0
    );
    println!(
        "  Meta-Parameter Adaptations : {} automated interventions",
        session.meta_state.total_adjustments
    );
    println!("  Transitive Parity Status   : {:?}", parity.status);
    println!(
        "  Dynamic Activation Range   : [{:.2e}, {:.2e}]",
        parity.min_activation, parity.max_activation
    );
    println!(
        "  NaN / Inf Trap Check       : {}",
        if parity.has_nan_or_inf {
            "FAILED (Traps Detected)"
        } else {
            "PASSED (Clean)"
        }
    );
    println!("=============================================================================\n");

    Ok(())
}

pub fn cmd_train_status() -> Result<(), String> {
    println!("\n=============================================================================");
    println!("  AI DESIGNER TRAINING & ACCELERATOR ENVIRONMENT STATUS");
    println!("=============================================================================");
    println!(
        "  OS Platform                : {} {}",
        std::env::consts::OS,
        std::env::consts::ARCH
    );
    println!(
        "  Host Cores Available       : {}",
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    );

    // Probe python / torch if accessible
    let py_check = std::process::Command::new("python")
        .arg("-c")
        .arg("import torch; print(f'{torch.__version__}|{torch.cuda.is_available()}|{torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')")
        .output();

    match py_check {
        Ok(out) if out.status.success() => {
            let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
            let parts: Vec<&str> = s.split('|').collect();
            if parts.len() >= 3 {
                println!("  PyTorch Runtime Detected   : v{}", parts[0]);
                println!(
                    "  CUDA Hardware Acceleration : {}",
                    if parts[1] == "True" {
                        "Active"
                    } else {
                        "Inactive (CPU mode)"
                    }
                );
                println!("  Primary Compute Device     : {}", parts[2]);
            } else {
                println!("  PyTorch Output             : {}", s);
            }
        }
        _ => {
            println!(
                "  PyTorch Runtime Detected   : Not found on system PATH (run 'aid train setup' to create turnkey venv)"
            );
        }
    }

    println!("-----------------------------------------------------------------------------");
    println!(
        "  Supported Optimizers       : MuonAdamWPartitioned, AdamW, Lion, Adafactor, Shampoo"
    );
    println!("  Supported Precision Formats: BF16, FP16, FP8, BitNet (1.58-bit QAT), FP32");
    println!("  Supported Gradient Hygiene : AutoClip (Percentile-based), Grokfast (Low-pass EMA)");
    println!(
        "  Live Telemetry Channels    : Step Loss, Smoothed Loss, Gradient Norm, Muon Orthogonality,"
    );
    println!(
        "                               Token Throughput, VRAM Usage, Multi-Stage Curriculums"
    );
    println!("=============================================================================\n");

    Ok(())
}

pub fn print_surgery_help() {
    println!(
        r#"
=============================================================================
  AI DESIGNER MODEL SURGERY & WEIGHT MANIFOLD CLI (aid surgery)
=============================================================================
Usage:
  aid surgery [SUBCOMMAND] [OPTIONS]

Subcommands:
  abliterate <blueprint> [OPTIONS]
      Erases refusal vectors or behavioral concepts via orthogonal projection,
      LEACE, contrastive activation addition (RepE), or rank-truncated SVD:
        --technique <technique>  orthogonal, leace, repe, circuit_breaker, svd
        --concept <concept>      refusal, hallucination, sycophancy, toxicity
        --strength <factor>      Projection erasure strength (default: 1.0)
        --guard                  Enforce general capability retention guard
        -o <out_blueprint>       Save modified architecture blueprint

  soup <blueprint> [OPTIONS]
      Synthesizes model soups across checkpoints on flat loss plateaus:
        --strategy <strategy>    uniform, greedy, fisher, slerp, task, ties, dare
        --checkpoints <count>    Number of synthetic checkpoint vectors (default: 2)
        -o <out_blueprint>       Save merged architecture blueprint

  graft <blueprint> [OPTIONS]
      Performs Frankenstein depth stacking, redundant layer pruning, or LoRA fusion:
        --strategy <strategy>    frankenmerge, prune, lora
        --multiplier <factor>    Layer depth scaling multiplier (default: 1.5)
        -o <out_blueprint>       Save grafted architecture blueprint
"#
    );
}

pub fn cmd_surgery_abliterate(
    blueprint_path: &str,
    technique_str: Option<&str>,
    concept_str: Option<&str>,
    strength: f64,
    guard: bool,
    out_target: Option<&str>,
) -> Result<(), String> {
    println!("\n=============================================================================");
    println!("  AI DESIGNER MODEL SURGERY: CONCEPT & REFUSAL ABLITERATION");
    println!("=============================================================================");
    let spec = load_blueprint_auto(Path::new(blueprint_path))?;
    println!(
        "  Target Model: {} ({})",
        spec.metadata.display_name, spec.metadata.model_id
    );

    let technique = match technique_str
        .unwrap_or("orthogonal")
        .to_lowercase()
        .as_str()
    {
        "orthogonal" | "ortho" => AbliterationTechnique::DirectionalWeightOrthogonalization,
        "leace" => AbliterationTechnique::LeaceConceptErasure,
        "repe" | "contrastive" | "actadd" => AbliterationTechnique::ContrastiveActivationAddition,
        "circuit_breaker" | "circuitbreaker" => AbliterationTechnique::SafetyCircuitBreakers,
        "knockout" | "head" => AbliterationTechnique::AttentionHeadKnockout,
        "svd" | "svd_truncate" => AbliterationTechnique::RankTruncatedSvd,
        other => {
            return Err(format!(
                "Unknown abliteration technique '{other}'. Options: orthogonal, leace, repe, circuit_breaker, knockout, svd"
            ));
        }
    };

    let concept = match concept_str.unwrap_or("refusal").to_lowercase().as_str() {
        "refusal" => SurgeryTargetConcept::RefusalAndSafety,
        "hallucination" => SurgeryTargetConcept::HallucinationAndOverconfidence,
        "sycophancy" => SurgeryTargetConcept::SycophancyAndFlattery,
        "toxicity" | "harm" => SurgeryTargetConcept::ToxicityAndHarm,
        other => SurgeryTargetConcept::CustomDirection {
            name: other.to_string(),
            dimension: 2048,
        },
    };

    let plan = AbliterationPlan {
        technique,
        target_concept: concept,
        projection_strength: strength,
        anti_degeneration_guard: guard,
        ..Default::default()
    };

    let res = execute_abliteration(&spec, &plan);
    println!("  Technique                  : {:?}", res.technique_applied);
    println!("  Target Concept             : {}", res.target_concept);
    println!(
        "  Target Layers Modified     : {}",
        res.modified_layers_count
    );
    println!(
        "  Total Parameters Ablated   : {}",
        res.total_parameters_ablated
    );
    println!(
        "  Refusal Suppression        : {:.1}%",
        res.refusal_suppression_pct
    );
    println!(
        "  Capability Retention Score : {:.1}%",
        res.benign_capability_retention_pct
    );
    println!(
        "  Subspace Angle             : {:.2}°",
        res.subspace_angle_degrees
    );
    println!(
        "  Frobenius Norm Delta       : {:.4}",
        res.frobenius_norm_delta
    );
    println!("  Verification Status        : {}", res.status);

    if let Some(out) = out_target {
        save_blueprint_auto(&spec, Path::new(out), None)?;
        println!("  Saved modified blueprint to: {}", out);
    }
    println!("=============================================================================\n");
    Ok(())
}

pub fn cmd_surgery_soup(
    blueprint_path: &str,
    strategy_str: Option<&str>,
    checkpoints: usize,
    out_target: Option<&str>,
) -> Result<(), String> {
    println!("\n=============================================================================");
    println!("  AI DESIGNER MODEL SURGERY: MODEL SOUPS & MANIFOLD MERGING");
    println!("=============================================================================");
    let spec = load_blueprint_auto(Path::new(blueprint_path))?;
    println!(
        "  Base Architecture: {} ({})",
        spec.metadata.display_name, spec.metadata.model_id
    );

    let strategy = match strategy_str.unwrap_or("ties").to_lowercase().as_str() {
        "uniform" => ModelSoupStrategy::UniformSoup,
        "greedy" => ModelSoupStrategy::GreedySoup,
        "fisher" => ModelSoupStrategy::FisherWeightedSoup,
        "slerp" => ModelSoupStrategy::Slerp,
        "task" | "task_arithmetic" => ModelSoupStrategy::TaskArithmetic,
        "ties" => ModelSoupStrategy::TiesMerging,
        "dare_linear" => ModelSoupStrategy::DareLinear,
        "dare_ties" | "dare" => ModelSoupStrategy::DareTies,
        other => {
            return Err(format!(
                "Unknown soup strategy '{other}'. Options: uniform, greedy, fisher, slerp, task, ties, dare_linear, dare_ties"
            ));
        }
    };

    let mut recipe = SoupMergeRecipe {
        strategy,
        ..Default::default()
    };
    if checkpoints > 2 {
        for i in 2..checkpoints {
            recipe.checkpoints.push(SoupCheckpoint {
                id: format!("chk_{i}"),
                name: format!("Fine-tuned Checkpoint #{i}"),
                task_label: "Downstream Specialist".to_string(),
                coefficient: 0.5,
                val_score: 83.0 + (i as f64 * 0.5),
                is_base: false,
            });
        }
    }

    let res = execute_soup_merge(&spec, &recipe);
    println!("  Merge Strategy             : {:?}", res.strategy);
    println!("  Included Checkpoints       : {}", res.checkpoints_used);
    println!(
        "  Total Parameters Merged    : {}",
        res.total_parameters_merged
    );
    println!(
        "  Simulated Validation Metric: {:.2}",
        res.simulated_val_score
    );
    println!(
        "  Ensemble Improvement Gain  : {:+.2}%",
        res.ensemble_gain_pct
    );
    println!(
        "  Weight Norm Drift          : {:.4}",
        res.weight_norm_drift
    );
    println!("  Sparsity Level             : {:.1}%", res.sparsity_pct);
    println!("  Verification Status        : {}", res.status);

    if let Some(out) = out_target {
        save_blueprint_auto(&spec, Path::new(out), None)?;
        println!("  Saved soup blueprint to    : {}", out);
    }
    println!("=============================================================================\n");
    Ok(())
}

pub fn cmd_surgery_graft(
    blueprint_path: &str,
    strategy_str: Option<&str>,
    depth_multiplier: f64,
    out_target: Option<&str>,
) -> Result<(), String> {
    println!("\n=============================================================================");
    println!("  AI DESIGNER MODEL SURGERY: FRANKENMERGE & LAYER GRAFTING");
    println!("=============================================================================");
    let spec = load_blueprint_auto(Path::new(blueprint_path))?;
    println!(
        "  Source Blueprint: {} ({})",
        spec.metadata.display_name, spec.metadata.model_id
    );

    let strategy = match strategy_str
        .unwrap_or("frankenmerge")
        .to_lowercase()
        .as_str()
    {
        "frankenmerge" | "solar" | "stack" => GraftingStrategy::FrankenmergeDepthStacking,
        "prune" | "pruning" => GraftingStrategy::LayerPruningSurgery,
        "lora" | "lora_fuse" => GraftingStrategy::LoRAAdapterFusion,
        "bridge" => GraftingStrategy::CrossArchitectureBridge,
        other => {
            return Err(format!(
                "Unknown grafting strategy '{other}'. Options: frankenmerge, prune, lora, bridge"
            ));
        }
    };

    let plan = GraftingPlan {
        strategy,
        target_depth_multiplier: depth_multiplier,
        ..Default::default()
    };

    let res = execute_layer_graft(&spec, &plan);
    println!("  Grafting Strategy          : {:?}", res.strategy);
    println!("  Initial Layer Depth        : {}", res.initial_layers);
    println!("  Synthesized Layer Depth    : {}", res.final_layers);
    println!(
        "  Parameter Delta Count      : {:+} params",
        res.total_parameters_delta
    );
    println!(
        "  Estimated Speedup/Slowdown : {:+.1}%",
        res.estimated_speedup_or_slowdown_pct
    );
    println!("  Structural Status          : {}", res.status);

    if let Some(out) = out_target {
        save_blueprint_auto(&spec, Path::new(out), None)?;
        println!("  Saved grafted blueprint to : {}", out);
    }
    println!("=============================================================================\n");
    Ok(())
}

pub fn cmd_workspace_list() -> Result<(), String> {
    println!("\n=============================================================================");
    println!("  AI DESIGNER RESEARCH WORKSPACES & MULTI-MODEL SUITES");
    println!("=============================================================================");
    let reg = WorkspaceRegistry::default();
    println!("  Active Workspace: {}", reg.active_workspace_id);
    println!("  Total Workspaces: {}", reg.workspaces.len());
    println!("-----------------------------------------------------------------------------");
    for ws in &reg.workspaces {
        println!("  * Workspace ID   : {}", ws.id);
        println!("    Name           : {}", ws.name);
        println!("    Active Model   : {}", ws.active_design_id);
        println!("    Configured Models: {}", ws.designs.len());
        for (i, d) in ws.designs.iter().enumerate() {
            println!(
                "      [{}] {} (nodes: {})",
                i + 1,
                d.name,
                d.spec.graph.nodes.len()
            );
        }
    }
    println!("=============================================================================\n");
    Ok(())
}

pub fn cmd_workspace_export(
    ws_id: &str,
    out_file: &str,
    force_json: bool,
    force_bson: bool,
) -> Result<(), String> {
    let reg = WorkspaceRegistry::default();
    let is_json = force_json || out_file.ends_with(".json");
    let is_bson = force_bson || out_file.ends_with(".bson");
    if is_json {
        let json_str = reg.export_workspace_json(ws_id)?;
        fs::write(out_file, json_str)
            .map_err(|e| format!("Failed to write JSON workspace '{out_file}': {e}"))?;
        println!("[+] Successfully exported workspace '{ws_id}' JSON document to '{out_file}'");
    } else if is_bson {
        let bson_bytes = reg.export_workspace_bson(ws_id)?;
        fs::write(out_file, &bson_bytes)
            .map_err(|e| format!("Failed to write BSON workspace '{out_file}': {e}"))?;
        println!(
            "[+] Successfully exported workspace '{ws_id}' BSON document ({} bytes) to '{out_file}'",
            bson_bytes.len()
        );
    } else {
        let bin_bytes = reg.export_workspace_aidws(ws_id)?;
        fs::write(out_file, &bin_bytes)
            .map_err(|e| format!("Failed to write binary workspace '{out_file}': {e}"))?;
        println!(
            "[+] Successfully exported workspace '{ws_id}' binary container (.aidws, {} bytes) to '{out_file}'",
            bin_bytes.len()
        );
    }
    Ok(())
}

pub fn cmd_workspace_import(in_file: &str) -> Result<(), String> {
    let bytes = fs::read(in_file).map_err(|e| format!("Failed to read '{in_file}': {e}"))?;
    let mut reg = WorkspaceRegistry::default();
    let (imported_id, is_bin) = reg.import_workspace_bytes(&bytes)?;
    let format_str = if is_bin {
        "native binary container (.aidws, deflate + crc32)"
    } else {
        "JSON document"
    };
    println!(
        "[+] Successfully imported workspace from '{in_file}' ({format_str}) -> ID: '{imported_id}'"
    );
    Ok(())
}
