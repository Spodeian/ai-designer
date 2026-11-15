//! Complete strongly-typed AI Design Specification (AiDesignSpec) AST.

pub mod archetypes;
pub mod graph;
pub mod io;
pub mod operators;
pub mod precision;
pub mod tradeoff;
pub mod training;

pub use archetypes::*;
pub use graph::*;
pub use io::*;
pub use operators::*;
pub use precision::*;
pub use tradeoff::*;
pub use training::*;

use serde::{Deserialize, Serialize};

/// Target hardware execution profile for analytical roofline and VRAM budgeting.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HardwareTargetProfile {
    pub profile_name: String,
    pub max_vram_bytes: u64,
    pub peak_fp16_tflops: f64,
    pub peak_fp8_tflops: f64,
    pub memory_bandwidth_gbs: f64,
}

impl Default for HardwareTargetProfile {
    fn default() -> Self {
        // Default: Consumer Workstation GPU (RTX 4090 24GB)
        Self {
            profile_name: "ConsumerGPU_RTX4090".to_string(),
            max_vram_bytes: 24 * 1024 * 1024 * 1024,
            peak_fp16_tflops: 330.0,
            peak_fp8_tflops: 660.0,
            memory_bandwidth_gbs: 1008.0,
        }
    }
}

impl HardwareTargetProfile {
    pub fn rtx_4090() -> Self {
        Self::default()
    }

    pub fn jetson_orin() -> Self {
        Self {
            profile_name: "EdgeRobotics_JetsonOrin".to_string(),
            max_vram_bytes: 8 * 1024 * 1024 * 1024,
            peak_fp16_tflops: 275.0,
            peak_fp8_tflops: 550.0,
            memory_bandwidth_gbs: 204.8,
        }
    }

    pub fn datacenter_h100() -> Self {
        Self {
            profile_name: "DataCenter_H100_SXM5".to_string(),
            max_vram_bytes: 80 * 1024 * 1024 * 1024,
            peak_fp16_tflops: 989.0,
            peak_fp8_tflops: 1978.0,
            memory_bandwidth_gbs: 3350.0,
        }
    }

    pub fn h100() -> Self {
        Self::datacenter_h100()
    }

    pub fn edge_microcontroller() -> Self {
        Self {
            profile_name: "EdgeMCU_CortexM55".to_string(),
            max_vram_bytes: 16 * 1024 * 1024, // 16 MB
            peak_fp16_tflops: 0.05,
            peak_fp8_tflops: 0.10,
            memory_bandwidth_gbs: 2.0,
        }
    }

    pub fn cortex_m55() -> Self {
        Self::edge_microcontroller()
    }

    /// Operational intensity ridge point I_ridge = PeakCompute / MemoryBandwidth (FLOPs / Byte).
    pub fn ridge_point(&self) -> f64 {
        (self.peak_fp16_tflops * 1e12) / (self.memory_bandwidth_gbs * 1e9)
    }
}

/// Metadata describing author, license, and model identity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlueprintMetadata {
    pub model_id: String,
    pub display_name: String,
    #[serde(default = "default_domain")]
    pub domain: String,
    #[serde(default = "default_author")]
    pub author: String,
    #[serde(default = "default_license")]
    pub license: String,
    #[serde(default)]
    pub target_hardware: HardwareTargetProfile,
    #[serde(default)]
    pub tags: Vec<String>,
}

fn default_domain() -> String {
    "GeneralAI".to_string()
}
fn default_author() -> String {
    "AI Designer Studio".to_string()
}
fn default_license() -> String {
    "Apache-2.0".to_string()
}

impl Default for BlueprintMetadata {
    fn default() -> Self {
        Self {
            model_id: "custom_model".to_string(),
            display_name: "Custom Model Architecture".to_string(),
            domain: default_domain(),
            author: default_author(),
            license: default_license(),
            target_hardware: HardwareTargetProfile::default(),
            tags: Vec::new(),
        }
    }
}

/// Root AI Design Specification document (.aid / .aidpkg).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct AiDesignSpec {
    #[serde(default = "default_version")]
    pub aid_version: String,
    pub metadata: BlueprintMetadata,
    pub inputs: Vec<InputPortSpec>,
    pub outputs: Vec<OutputPortSpec>,
    pub graph: ComputeGraph,
    #[serde(default)]
    pub training: TrainingSpec,
    #[serde(default)]
    pub precision: NumericLatticeConfig,
}

fn default_version() -> String {
    "1.0.0".to_string()
}

impl AiDesignSpec {
    pub const SCHEMA_VERSION: &'static str = "2.0.0";

    pub fn new(model_id: &str, display_name: &str) -> Self {
        Self {
            aid_version: default_version(),
            metadata: BlueprintMetadata {
                model_id: model_id.to_string(),
                display_name: display_name.to_string(),
                ..Default::default()
            },
            inputs: Vec::new(),
            outputs: Vec::new(),
            graph: ComputeGraph::default(),
            training: TrainingSpec::default(),
            precision: NumericLatticeConfig::default(),
        }
    }

    /// Basic structural consistency validation.
    pub fn validate(&self) -> Result<(), String> {
        if self.inputs.is_empty() {
            return Err("Model specification must declare at least one input port".to_string());
        }
        if self.outputs.is_empty() {
            return Err("Model specification must declare at least one output port".to_string());
        }
        if self.graph.nodes.is_empty() {
            return Err("Model computation graph must contain at least one node".to_string());
        }
        Ok(())
    }

    /// Returns the versioned JSON Schema (Draft 2020-12) for `AiDesignSpec`.
    pub fn json_schema() -> serde_json::Value {
        serde_json::json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": format!("https://spodeian.com/schemas/ai-designer/{}/aid-spec.schema.json", Self::SCHEMA_VERSION),
            "title": "AiDesignSpec",
            "description": "Typed Neural & Scientific ML Architecture Specification for AI Designer Studio",
            "schema_version": Self::SCHEMA_VERSION,
            "type": "object",
            "required": ["metadata", "inputs", "outputs", "graph"],
            "properties": {
                "aid_version": { "type": "string", "default": Self::SCHEMA_VERSION },
                "metadata": {
                    "type": "object",
                    "required": ["model_id", "display_name"],
                    "properties": {
                        "model_id": { "type": "string" },
                        "display_name": { "type": "string" },
                        "domain": { "type": "string" },
                        "author": { "type": "string" },
                        "license": { "type": "string" },
                        "target_hardware": { "type": "object" },
                        "tags": { "type": "array", "items": { "type": "string" } }
                    }
                },
                "inputs": { "type": "array", "minItems": 1 },
                "outputs": { "type": "array", "minItems": 1 },
                "graph": {
                    "type": "object",
                    "required": ["topology", "nodes"],
                    "properties": {
                        "topology": { "type": "object" },
                        "nodes": { "type": "array", "minItems": 1 },
                        "connections": { "type": "array" }
                    }
                },
                "training": { "type": "object" },
                "precision": { "type": "object" }
            }
        })
    }

    /// Parses JSON (including legacy v1 blueprints missing newer optional fields),
    /// upgrades `aid_version` to `Self::SCHEMA_VERSION`, and validates the specification.
    pub fn validate_and_migrate_json(raw: &str) -> Result<Self, String> {
        let mut val: serde_json::Value =
            serde_json::from_str(raw).map_err(|e| format!("Invalid JSON spec: {e}"))?;
        let obj = val
            .as_object_mut()
            .ok_or_else(|| "AiDesignSpec JSON root must be an object".to_string())?;

        // Migrate legacy `"version"` field if `"aid_version"` is omitted.
        if !obj.contains_key("aid_version")
            && let Some(v) = obj.remove("version")
        {
            obj.insert("aid_version".to_string(), v);
        }
        obj.entry("training".to_string()).or_insert_with(|| {
            serde_json::to_value(TrainingSpec::default()).unwrap_or_else(|_| serde_json::json!({}))
        });
        obj.entry("precision".to_string()).or_insert_with(|| {
            serde_json::to_value(NumericLatticeConfig::default())
                .unwrap_or_else(|_| serde_json::json!({}))
        });

        let mut spec: Self = serde_json::from_value(val)
            .map_err(|e| format!("AiDesignSpec deserialization failed: {e}"))?;
        if spec.aid_version.is_empty() || spec.aid_version.starts_with("1.") {
            spec.aid_version = Self::SCHEMA_VERSION.to_string();
        }
        spec.validate()?;
        Ok(spec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_schema_and_v1_migration() {
        let schema = AiDesignSpec::json_schema();
        assert_eq!(schema["title"], "AiDesignSpec");
        assert_eq!(schema["schema_version"], AiDesignSpec::SCHEMA_VERSION);
        assert_eq!(
            schema["required"],
            serde_json::json!(["metadata", "inputs", "outputs", "graph"])
        );

        let v1_json = serde_json::json!({
            "version": "1.0.0",
            "metadata": {
                "model_id": "legacy_mlp",
                "display_name": "Legacy MLP"
            },
            "inputs": [{
                "id": "x",
                "name": "Input",
                "space": { "kind": "continuous_vector", "dimension": 64 },
                "symbolic_shape": ["B", 64]
            }],
            "outputs": [{
                "id": "y",
                "name": "Output",
                "space": { "kind": "continuous_vector", "dimension": 32 },
                "symbolic_shape": ["B", 32]
            }],
            "graph": {
                "topology": { "kind": "serial_stack" },
                "nodes": [{
                    "id": "proj",
                    "name": "Projection",
                    "operator": { "type": "linear", "in_features": 64, "out_features": 32, "bias": true }
                }]
            }
        });

        let migrated = AiDesignSpec::validate_and_migrate_json(&v1_json.to_string())
            .expect("v1 migration should succeed");
        assert_eq!(migrated.aid_version, AiDesignSpec::SCHEMA_VERSION);
        assert_eq!(migrated.metadata.model_id, "legacy_mlp");
        assert_eq!(migrated.graph.nodes.len(), 1);
    }
}
