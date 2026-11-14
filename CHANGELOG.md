# Changelog

All notable changes to `ai-designer` are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and Semantic Versioning.

## [0.2.0] - 2026-11-14

### Added
- Self-hosted `TopicMetadata` catalog, `ingest_directory_tree` Python AST importer, and `import_from_onnx_or_torch_export_json`.
- Versioned `AiDesignSpec::SCHEMA_VERSION` (`2.0.0`) with JSON Schema generation and v1 migration.
- Binary GGUF v3 serialization and parsing via `sml_runtime::format::gguf`.
- `SelfOptimizingMeta` & `SelfOptimizingMetaIntegrator` training planner, PyTorch/JAX/Candle codegen, and `meta` CLI subcommand.
- `surgery/adapters.rs`: LoRA/DoRA adapter and structured pruning planning.
- `codegen/edge_gen.rs`: Apple Silicon MLX, Core ML, TFLite/LiteRT, and ExecuTorch export scripts.
- SVG and TikZ diagram export (`export_spec_to_svg`, `export_spec_to_tikz`) with 1,000-node scale test (`95` workspace tests).

### Changed
- Replaced all external `Spodeian/AI-Examples` URLs with self-hosted assets and wired local `spodeian-core v3.0` crates.
