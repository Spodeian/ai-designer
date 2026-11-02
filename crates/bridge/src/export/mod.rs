//! Unified Multi-Backend Exporter: PyTorch 2, JAX, Candle, wGPU, and SIMD CPU.

pub mod candle;
pub mod jax;
pub mod pytorch2;
pub mod simd_cpu;
pub mod wgpu;

pub use candle::*;
pub use jax::*;
pub use pytorch2::*;
pub use simd_cpu::*;
pub use wgpu::*;

use serde::{Deserialize, Serialize};
use shared::spec::AiDesignSpec;
use std::collections::HashMap;
use std::path::Path;

/// Supported multi-framework export targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExportBackend {
    PyTorch2,
    Jax,
    Candle,
    WGpu,
    SimdCpu,
}

impl ExportBackend {
    pub fn label(&self) -> &'static str {
        match self {
            Self::PyTorch2 => "PyTorch 2.x (Inductor / Dynamo)",
            Self::Jax => "JAX / Equinox (Functional / Pjit)",
            Self::Candle => "Rust Candle (Candle Bridge)",
            Self::WGpu => "WebGPU (WGSL Compute Shaders)",
            Self::SimdCpu => "SIMD CPU (AVX2 / NEON Vectorized)",
        }
    }
}

/// Complete collection of exported artifacts and code files.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ExportArtifacts {
    pub model_id: String,
    pub files: HashMap<String, String>,
}

impl ExportArtifacts {
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            files: HashMap::new(),
        }
    }

    pub fn insert(&mut self, path: impl Into<String>, content: impl Into<String>) {
        self.files.insert(path.into(), content.into());
    }

    /// Writes all bundled source files to a target directory on disk.
    pub fn save_to_disk(&self, base_dir: &Path) -> std::io::Result<()> {
        for (rel_path, content) in &self.files {
            let full_path = base_dir.join(rel_path);
            if let Some(parent) = full_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&full_path, content)?;
        }
        Ok(())
    }
}

/// Generates export code for a specific backend target.
pub fn export_backend(spec: &AiDesignSpec, backend: ExportBackend) -> ExportArtifacts {
    let mut artifacts = ExportArtifacts::new(&spec.metadata.model_id);

    match backend {
        ExportBackend::PyTorch2 => {
            for (path, content) in generate_pytorch2_project(spec) {
                artifacts.insert(format!("pytorch/{}", path), content);
            }
        }
        ExportBackend::Jax => {
            for (path, content) in generate_jax_project(spec) {
                artifacts.insert(format!("jax/{}", path), content);
            }
        }
        ExportBackend::Candle => {
            for (path, content) in generate_candle_project(spec) {
                artifacts.insert(format!("candle/{}", path), content);
            }
        }
        ExportBackend::WGpu => {
            for (path, content) in generate_wgpu_bundle(spec) {
                artifacts.insert(path, content);
            }
        }
        ExportBackend::SimdCpu => {
            for (path, content) in generate_simd_cpu_bundle(spec) {
                artifacts.insert(path, content);
            }
        }
    }

    artifacts
}

/// Exports all supported backend targets in a single unified bundle.
pub fn export_all_targets(spec: &AiDesignSpec) -> ExportArtifacts {
    let mut all = ExportArtifacts::new(&spec.metadata.model_id);

    for backend in [
        ExportBackend::PyTorch2,
        ExportBackend::Jax,
        ExportBackend::Candle,
        ExportBackend::WGpu,
        ExportBackend::SimdCpu,
    ] {
        let sub = export_backend(spec, backend);
        for (k, v) in sub.files {
            all.insert(k, v);
        }
    }

    all
}
