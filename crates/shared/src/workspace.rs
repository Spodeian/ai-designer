//! Workspace and Multi-Model Project Management System.
//!
//! Provides structures and serialization for managing multiple model architectures,
//! associated training curriculums, analysis states, and surgery configurations
//! within unified project workspaces across both WebAssembly (browser) and Native Desktop.
//!
//! Supports native binary `.aidws` envelope packaging (with magic header `AIDWS\x01`,
//! format tag, uncompressed length, and CRC32 integrity check) as well as human-readable `.json`.

use crate::spec::AiDesignSpec;
use crate::surgery::{AbliterationPlan, GraftingPlan, SoupMergeRecipe};
use crc32fast::Hasher;
use miniz_oxide::deflate::compress_to_vec_zlib;
use miniz_oxide::inflate::decompress_to_vec_zlib;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Magic 6-byte header identifying a native binary .aidws file: b"AIDWS\x01"
pub const AIDWS_MAGIC: &[u8; 6] = b"AIDWS\x01";

/// Format descriptor tag byte within the .aidws binary header.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum AidwsFormatTag {
    /// JSON payload compressed with zlib/deflate (default).
    #[default]
    JsonCompressed = 0x01,
    /// BSON binary payload compressed with zlib/deflate.
    BsonCompressed = 0x02,
    /// Uncompressed raw JSON text payload.
    RawJson = 0x03,
    /// Uncompressed raw BSON binary payload.
    RawBson = 0x04,
}

impl AidwsFormatTag {
    pub fn from_u8(b: u8) -> Option<Self> {
        match b {
            0x01 => Some(Self::JsonCompressed),
            0x02 => Some(Self::BsonCompressed),
            0x03 => Some(Self::RawJson),
            0x04 => Some(Self::RawBson),
            _ => None,
        }
    }
}

/// An individual model design stored within a workspace.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkspaceModelDesign {
    /// Unique identifier for this model design within the workspace.
    pub id: String,
    /// Human-readable display name (e.g. "Llama 3.2 1B MoE Specialist").
    pub name: String,
    /// Architectural specification and layer graph.
    pub spec: AiDesignSpec,
    /// 2D Canvas positions for node graph rendering [x, y].
    #[serde(default)]
    pub node_positions: HashMap<String, [f32; 2]>,
    /// Research notes, experiment logs, or architectural decisions.
    #[serde(default)]
    pub notes: String,
    /// Unix timestamp when created.
    pub created_at: u64,
    /// Unix timestamp when last updated.
    pub updated_at: u64,
    /// Optional saved abliteration plan for refusal/concept surgery.
    pub abliteration_plan: Option<AbliterationPlan>,
    /// Optional saved model soup merge recipe.
    pub soup_recipe: Option<SoupMergeRecipe>,
    /// Optional saved layer grafting / Frankenmerge plan.
    pub grafting_plan: Option<GraftingPlan>,
}

impl WorkspaceModelDesign {
    pub fn new(id: impl Into<String>, name: impl Into<String>, spec: AiDesignSpec) -> Self {
        let now = current_unix_timestamp();
        Self {
            id: id.into(),
            name: name.into(),
            spec,
            node_positions: HashMap::new(),
            notes: String::new(),
            created_at: now,
            updated_at: now,
            abliteration_plan: None,
            soup_recipe: None,
            grafting_plan: None,
        }
    }

    /// Creates a cloned variant of this design with a new ID and suffix.
    pub fn clone_variant(&self, new_id: impl Into<String>, new_name: impl Into<String>) -> Self {
        let now = current_unix_timestamp();
        let mut cloned = self.clone();
        cloned.id = new_id.into();
        cloned.name = new_name.into();
        cloned.created_at = now;
        cloned.updated_at = now;
        cloned
    }
}

/// A workspace containing multiple related model designs, setups, and notes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Workspace {
    /// Unique identifier for the workspace (slug format).
    pub id: String,
    /// Human-readable workspace name (e.g. "Multimodal Vision-Language Suite").
    pub name: String,
    /// Detailed description or research goal.
    #[serde(default)]
    pub description: String,
    /// ID of the currently active model design in this workspace.
    pub active_design_id: String,
    /// List of model designs belonging to this workspace.
    pub designs: Vec<WorkspaceModelDesign>,
    /// Categorization tags (e.g. ["diffusion", "transformer", "fp8"]).
    #[serde(default)]
    pub tags: Vec<String>,
    /// Research notes and hypotheses.
    #[serde(default)]
    pub notes: String,
    /// Unix timestamp of creation.
    pub created_at: u64,
    /// Unix timestamp of last update.
    pub updated_at: u64,
}

impl Workspace {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        starter_design: WorkspaceModelDesign,
    ) -> Self {
        let now = current_unix_timestamp();
        let active_id = starter_design.id.clone();
        Self {
            id: id.into(),
            name: name.into(),
            description: String::new(),
            active_design_id: active_id,
            designs: vec![starter_design],
            tags: Vec::new(),
            notes: String::new(),
            created_at: now,
            updated_at: now,
        }
    }

    /// Finds a model design by ID.
    pub fn get_design(&self, design_id: &str) -> Option<&WorkspaceModelDesign> {
        self.designs.iter().find(|d| d.id == design_id)
    }

    /// Finds a model design by ID mutably.
    pub fn get_design_mut(&mut self, design_id: &str) -> Option<&mut WorkspaceModelDesign> {
        self.designs.iter_mut().find(|d| d.id == design_id)
    }

    /// Returns the currently active model design.
    pub fn active_design(&self) -> Option<&WorkspaceModelDesign> {
        self.get_design(&self.active_design_id)
            .or_else(|| self.designs.first())
    }

    /// Returns the currently active model design mutably.
    pub fn active_design_mut(&mut self) -> Option<&mut WorkspaceModelDesign> {
        let active_id = self.active_design_id.clone();
        if let Some(pos) = self.designs.iter().position(|d| d.id == active_id) {
            Some(&mut self.designs[pos])
        } else {
            self.designs.first_mut()
        }
    }

    /// Switches the active design to the specified ID.
    pub fn set_active_design(&mut self, design_id: &str) -> bool {
        if self.designs.iter().any(|d| d.id == design_id) {
            self.active_design_id = design_id.to_string();
            self.updated_at = current_unix_timestamp();
            true
        } else {
            false
        }
    }

    /// Adds a new design to the workspace and optionally activates it.
    pub fn add_design(&mut self, design: WorkspaceModelDesign, activate: bool) {
        if activate {
            self.active_design_id = design.id.clone();
        }
        self.designs.push(design);
        self.updated_at = current_unix_timestamp();
    }

    /// Duplicates an existing design, creating a new variant with a unique ID.
    pub fn duplicate_design(&mut self, design_id: &str) -> Option<String> {
        let original = self.get_design(design_id)?.clone();
        let mut count = self.designs.len() + 1;
        let mut new_id = format!("{}_copy_{count}", original.id);
        while self.designs.iter().any(|d| d.id == new_id) {
            count += 1;
            new_id = format!("{}_copy_{count}", original.id);
        }
        let new_name = format!("{} (Variant)", original.name);
        let variant = original.clone_variant(&new_id, new_name);
        self.designs.push(variant);
        self.active_design_id = new_id.clone();
        self.updated_at = current_unix_timestamp();
        Some(new_id)
    }

    /// Removes a design by ID, ensuring at least one design remains.
    pub fn remove_design(&mut self, design_id: &str) -> bool {
        if self.designs.len() <= 1 {
            return false;
        }
        let initial_len = self.designs.len();
        self.designs.retain(|d| d.id != design_id);
        if self.designs.len() < initial_len {
            if self.active_design_id == design_id
                && let Some(first) = self.designs.first()
            {
                self.active_design_id = first.id.clone();
            }
            self.updated_at = current_unix_timestamp();
            true
        } else {
            false
        }
    }

    /// Serializes this workspace into a native binary .aidws package.
    pub fn to_aidws_bytes(&self) -> Result<Vec<u8>, String> {
        pack_aidws(self, AidwsFormatTag::JsonCompressed)
    }

    /// Deserializes a workspace from native binary .aidws bytes.
    pub fn from_aidws_bytes(bytes: &[u8]) -> Result<Self, String> {
        let (ws, _) = unpack_aidws(bytes)?;
        Ok(ws)
    }

    /// Serializes this workspace into a formatted JSON string.
    pub fn to_json_string(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserializes a workspace from a JSON string.
    pub fn from_json_str(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }

    /// Serializes this workspace into raw BSON binary bytes.
    pub fn to_bson_bytes(&self) -> Result<Vec<u8>, String> {
        let doc = bson::to_document(self).map_err(|e| format!("BSON serialization error: {e}"))?;
        let mut buf = Vec::new();
        doc.to_writer(&mut buf)
            .map_err(|e| format!("BSON writer error: {e}"))?;
        Ok(buf)
    }

    /// Deserializes a workspace from raw BSON binary bytes.
    pub fn from_bson_bytes(bytes: &[u8]) -> Result<Self, String> {
        let mut cursor = std::io::Cursor::new(bytes);
        let doc = bson::Document::from_reader(&mut cursor)
            .map_err(|e| format!("BSON read error: {e}"))?;
        bson::from_document(doc).map_err(|e| format!("BSON deserialization error: {e}"))
    }
}

/// Packs a Workspace into a native binary .aidws envelope.
pub fn pack_aidws(ws: &Workspace, tag: AidwsFormatTag) -> Result<Vec<u8>, String> {
    let uncompressed_payload = match tag {
        AidwsFormatTag::JsonCompressed | AidwsFormatTag::RawJson => {
            serde_json::to_vec(ws).map_err(|e| format!("Serialization error: {e}"))?
        }
        AidwsFormatTag::BsonCompressed | AidwsFormatTag::RawBson => {
            let doc =
                bson::to_document(ws).map_err(|e| format!("BSON serialization error: {e}"))?;
            let mut buf = Vec::new();
            doc.to_writer(&mut buf)
                .map_err(|e| format!("BSON writer error: {e}"))?;
            buf
        }
    };

    let mut hasher = Hasher::new();
    hasher.update(&uncompressed_payload);
    let crc = hasher.finalize();
    let uncompressed_len = uncompressed_payload.len() as u32;

    let payload = match tag {
        AidwsFormatTag::JsonCompressed | AidwsFormatTag::BsonCompressed => {
            compress_to_vec_zlib(&uncompressed_payload, 9)
        }
        AidwsFormatTag::RawJson | AidwsFormatTag::RawBson => uncompressed_payload,
    };

    let mut out = Vec::with_capacity(6 + 1 + 4 + 4 + payload.len());
    out.extend_from_slice(AIDWS_MAGIC);
    out.push(tag as u8);
    out.extend_from_slice(&uncompressed_len.to_le_bytes());
    out.extend_from_slice(&crc.to_le_bytes());
    out.extend_from_slice(&payload);
    Ok(out)
}

/// Unpacks a Workspace from a native binary .aidws envelope.
pub fn unpack_aidws(bytes: &[u8]) -> Result<(Workspace, AidwsFormatTag), String> {
    if bytes.len() < 15 {
        return Err(format!(
            "Binary .aidws package is truncated: length is {} bytes, minimum header is 15 bytes",
            bytes.len()
        ));
    }
    if &bytes[0..6] != AIDWS_MAGIC {
        return Err(format!(
            "Invalid .aidws binary magic header: expected {:?}, got {:?}",
            AIDWS_MAGIC,
            &bytes[0..6]
        ));
    }
    let tag = AidwsFormatTag::from_u8(bytes[6])
        .ok_or_else(|| format!("Unknown .aidws inner format tag: 0x{:02x}", bytes[6]))?;
    let uncompressed_len = u32::from_le_bytes(bytes[7..11].try_into().unwrap());
    let expected_crc = u32::from_le_bytes(bytes[11..15].try_into().unwrap());
    let payload = &bytes[15..];

    let uncompressed = match tag {
        AidwsFormatTag::JsonCompressed | AidwsFormatTag::BsonCompressed => {
            decompress_to_vec_zlib(payload)
                .map_err(|e| format!("Zlib decompression failed for .aidws: {:?}", e))?
        }
        AidwsFormatTag::RawJson | AidwsFormatTag::RawBson => payload.to_vec(),
    };

    if uncompressed.len() != uncompressed_len as usize {
        return Err(format!(
            "Decompressed length mismatch in .aidws: header recorded {} bytes, decompressed {} bytes",
            uncompressed_len,
            uncompressed.len()
        ));
    }

    let mut hasher = Hasher::new();
    hasher.update(&uncompressed);
    let actual_crc = hasher.finalize();
    if actual_crc != expected_crc {
        return Err(format!(
            "CRC32 checksum mismatch in .aidws package: expected {:08x}, got {:08x}",
            expected_crc, actual_crc
        ));
    }

    let ws: Workspace = match tag {
        AidwsFormatTag::JsonCompressed | AidwsFormatTag::RawJson => {
            serde_json::from_slice(&uncompressed)
                .map_err(|e| format!("JSON deserialization error in .aidws: {e}"))?
        }
        AidwsFormatTag::BsonCompressed | AidwsFormatTag::RawBson => {
            let mut cursor = std::io::Cursor::new(&uncompressed);
            let doc = bson::Document::from_reader(&mut cursor)
                .map_err(|e| format!("BSON read error in .aidws: {e}"))?;
            bson::from_document(doc)
                .map_err(|e| format!("BSON deserialization error in .aidws: {e}"))?
        }
    };

    Ok((ws, tag))
}

/// Automatically unpacks a Workspace from either native binary .aidws bytes, BSON bytes, OR JSON text bytes.
/// Returns (workspace, is_binary).
pub fn unpack_workspace_auto(bytes: &[u8]) -> Result<(Workspace, bool), String> {
    if bytes.len() >= 6 && &bytes[0..6] == AIDWS_MAGIC {
        let (ws, _) = unpack_aidws(bytes)?;
        Ok((ws, true))
    } else if let Ok(ws) = Workspace::from_bson_bytes(bytes) {
        Ok((ws, true))
    } else {
        match serde_json::from_slice::<Workspace>(bytes) {
            Ok(ws) => Ok((ws, false)),
            Err(e1) => match serde_json::from_slice::<WorkspaceRegistry>(bytes) {
                Ok(reg) => {
                    let active = reg.active_workspace().cloned().ok_or_else(|| {
                        "Imported workspace registry has no active workspace".to_string()
                    })?;
                    Ok((active, false))
                }
                Err(_) => Err(format!(
                    "Failed to auto-detect workspace format: not valid .aidws container, BSON, or JSON: {e1}"
                )),
            },
        }
    }
}

/// Registry managing all user workspaces and tracking the active workspace.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkspaceRegistry {
    /// ID of the currently active workspace.
    pub active_workspace_id: String,
    /// List of all managed workspaces.
    pub workspaces: Vec<Workspace>,
}

impl Default for WorkspaceRegistry {
    fn default() -> Self {
        let starter_yaml = crate::templates::TEMPLATES[0].yaml_content;
        let starter_spec =
            crate::format::aid_yaml::from_yaml_str(starter_yaml).unwrap_or_else(|_| {
                AiDesignSpec::new("spatiotemporal_dit_flow", "Spatiotemporal DiT Flow")
            });

        let starter_design = WorkspaceModelDesign::new(
            "dit_baseline",
            "Spatiotemporal DiT Flow (Baseline)",
            starter_spec,
        );

        let default_workspace = Workspace::new(
            "default_workspace",
            "Default Research Workspace",
            starter_design,
        );

        let active_id = default_workspace.id.clone();
        Self {
            active_workspace_id: active_id,
            workspaces: vec![default_workspace],
        }
    }
}

impl WorkspaceRegistry {
    /// Returns the currently active workspace.
    pub fn active_workspace(&self) -> Option<&Workspace> {
        self.workspaces
            .iter()
            .find(|w| w.id == self.active_workspace_id)
            .or_else(|| self.workspaces.first())
    }

    /// Returns the currently active workspace mutably.
    pub fn active_workspace_mut(&mut self) -> Option<&mut Workspace> {
        let active_id = self.active_workspace_id.clone();
        if let Some(pos) = self.workspaces.iter().position(|w| w.id == active_id) {
            Some(&mut self.workspaces[pos])
        } else {
            self.workspaces.first_mut()
        }
    }

    /// Switches the active workspace.
    pub fn set_active_workspace(&mut self, workspace_id: &str) -> bool {
        if self.workspaces.iter().any(|w| w.id == workspace_id) {
            self.active_workspace_id = workspace_id.to_string();
            true
        } else {
            false
        }
    }

    /// Creates and adds a new workspace with a starter design.
    pub fn create_workspace(
        &mut self,
        id: impl Into<String>,
        name: impl Into<String>,
        starter_spec: AiDesignSpec,
    ) -> String {
        let ws_id = id.into();
        let ws_name = name.into();
        let design_id = format!("{}_base", ws_id);
        let starter_design =
            WorkspaceModelDesign::new(&design_id, format!("{ws_name} Model"), starter_spec);
        let workspace = Workspace::new(&ws_id, &ws_name, starter_design);
        self.workspaces.push(workspace);
        self.active_workspace_id = ws_id.clone();
        ws_id
    }

    /// Duplicates an existing workspace with all contained models.
    pub fn duplicate_workspace(&mut self, workspace_id: &str) -> Option<String> {
        let original = self
            .workspaces
            .iter()
            .find(|w| w.id == workspace_id)?
            .clone();
        let mut count = self.workspaces.len() + 1;
        let mut new_id = format!("{}_copy_{count}", original.id);
        while self.workspaces.iter().any(|w| w.id == new_id) {
            count += 1;
            new_id = format!("{}_copy_{count}", original.id);
        }
        let now = current_unix_timestamp();
        let mut duplicated = original;
        duplicated.id = new_id.clone();
        duplicated.name = format!("{} (Copy)", duplicated.name);
        duplicated.created_at = now;
        duplicated.updated_at = now;
        self.workspaces.push(duplicated);
        self.active_workspace_id = new_id.clone();
        Some(new_id)
    }

    /// Removes a workspace by ID. Retains at least one workspace.
    pub fn remove_workspace(&mut self, workspace_id: &str) -> bool {
        if self.workspaces.len() <= 1 {
            return false;
        }
        let initial_len = self.workspaces.len();
        self.workspaces.retain(|w| w.id != workspace_id);
        if self.workspaces.len() < initial_len {
            if self.active_workspace_id == workspace_id
                && let Some(first) = self.workspaces.first()
            {
                self.active_workspace_id = first.id.clone();
            }
            true
        } else {
            false
        }
    }

    /// Exports a workspace as a native binary .aidws package.
    pub fn export_workspace_aidws(&self, workspace_id: &str) -> Result<Vec<u8>, String> {
        let ws = self
            .workspaces
            .iter()
            .find(|w| w.id == workspace_id)
            .ok_or_else(|| format!("Workspace '{workspace_id}' not found"))?;
        ws.to_aidws_bytes()
    }

    /// Exports a workspace as a human-readable JSON string.
    pub fn export_workspace_json(&self, workspace_id: &str) -> Result<String, String> {
        let ws = self
            .workspaces
            .iter()
            .find(|w| w.id == workspace_id)
            .ok_or_else(|| format!("Workspace '{workspace_id}' not found"))?;
        ws.to_json_string().map_err(|e| e.to_string())
    }

    /// Exports a workspace as raw binary BSON bytes.
    pub fn export_workspace_bson(&self, workspace_id: &str) -> Result<Vec<u8>, String> {
        let ws = self
            .workspaces
            .iter()
            .find(|w| w.id == workspace_id)
            .ok_or_else(|| format!("Workspace '{workspace_id}' not found"))?;
        ws.to_bson_bytes()
    }

    /// Imports a workspace from raw bytes (supporting both binary .aidws and text .json).
    /// Returns (new_workspace_id, is_binary).
    pub fn import_workspace_bytes(&mut self, bytes: &[u8]) -> Result<(String, bool), String> {
        let (mut ws, is_binary) = unpack_workspace_auto(bytes)?;

        // Ensure unique workspace ID
        if self.workspaces.iter().any(|w| w.id == ws.id) {
            let mut count = 2;
            let base_id = ws.id.clone();
            while self
                .workspaces
                .iter()
                .any(|w| w.id == format!("{base_id}_{count}"))
            {
                count += 1;
            }
            ws.id = format!("{base_id}_{count}");
            ws.name = format!("{} (Imported #{count})", ws.name);
        }

        let new_id = ws.id.clone();
        self.workspaces.push(ws);
        self.active_workspace_id = new_id.clone();
        Ok((new_id, is_binary))
    }

    /// Legacy compatibility wrapper: imports from JSON string.
    pub fn import_workspace_bundle(&mut self, json_str: &str) -> Result<String, String> {
        let (id, _) = self.import_workspace_bytes(json_str.as_bytes())?;
        Ok(id)
    }

    /// Legacy compatibility wrapper: exports to JSON string.
    pub fn export_workspace_bundle(&self, workspace_id: &str) -> Result<String, String> {
        self.export_workspace_json(workspace_id)
    }

    /// Serializes entire workspace registry to formatted JSON string.
    pub fn to_json_string(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserializes a workspace registry from JSON string.
    pub fn from_json_str(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }
}

/// Helper returning current unix timestamp in seconds.
pub fn current_unix_timestamp() -> u64 {
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }
    #[cfg(target_arch = "wasm32")]
    {
        (js_sys::Date::now() / 1000.0) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_test_workspace() -> Workspace {
        let starter_spec = AiDesignSpec::new("test_model_a", "Test Model Alpha");
        let design_a = WorkspaceModelDesign::new("model_a", "Model Alpha", starter_spec);
        let mut ws = Workspace::new("test_suite", "Multi-Model Test Suite", design_a);
        ws.notes = "Suite for testing multi-model pipelining and surgery persistence.".to_string();

        let mut design_b = WorkspaceModelDesign::new(
            "model_b",
            "Model Beta",
            AiDesignSpec::new("test_model_b", "Test Model Beta"),
        );
        design_b
            .node_positions
            .insert("node_1".to_string(), [120.0, 340.0]);
        design_b.abliteration_plan = Some(AbliterationPlan::default());
        design_b.soup_recipe = Some(SoupMergeRecipe::default());
        design_b.grafting_plan = Some(GraftingPlan::default());
        ws.add_design(design_b, false);
        ws
    }

    #[test]
    fn test_aidws_binary_roundtrip_integrity() {
        let original_ws = sample_test_workspace();
        assert_eq!(original_ws.designs.len(), 2);

        // Pack into binary .aidws
        let binary_bytes = original_ws.to_aidws_bytes().expect("Binary packing failed");
        assert!(binary_bytes.len() >= 15);
        assert_eq!(&binary_bytes[0..6], AIDWS_MAGIC);
        assert_eq!(binary_bytes[6], AidwsFormatTag::JsonCompressed as u8);

        // Unpack from binary .aidws
        let (unpacked_ws, format_tag) =
            unpack_aidws(&binary_bytes).expect("Binary unpacking failed");
        assert_eq!(format_tag, AidwsFormatTag::JsonCompressed);
        assert_eq!(unpacked_ws.id, original_ws.id);
        assert_eq!(unpacked_ws.name, original_ws.name);
        assert_eq!(unpacked_ws.notes, original_ws.notes);
        assert_eq!(unpacked_ws.designs.len(), 2);

        // Verify preserved surgery state in design_b
        let design_b = unpacked_ws.get_design("model_b").expect("model_b missing");
        assert!(design_b.abliteration_plan.is_some());
        assert!(design_b.soup_recipe.is_some());
        assert!(design_b.grafting_plan.is_some());
        assert_eq!(design_b.node_positions.get("node_1"), Some(&[120.0, 340.0]));
    }

    #[test]
    fn test_aidws_binary_corrupted_magic_rejected() {
        let ws = sample_test_workspace();
        let mut bytes = ws.to_aidws_bytes().unwrap();

        // Corrupt magic header byte
        bytes[0] = b'X';
        let err = unpack_aidws(&bytes).unwrap_err();
        assert!(err.contains("Invalid .aidws binary magic header"));
    }

    #[test]
    fn test_aidws_binary_crc_mismatch_rejected() {
        let ws = sample_test_workspace();
        let mut bytes = ws.to_aidws_bytes().unwrap();

        // Corrupt CRC checksum bytes (offsets 11..15)
        bytes[11] ^= 0xFF;
        let err = unpack_aidws(&bytes).unwrap_err();
        assert!(err.contains("CRC32 checksum mismatch"));
    }

    #[test]
    fn test_aidws_binary_truncated_header_rejected() {
        let short_bytes = b"AIDWS\x01\x01";
        let err = unpack_aidws(short_bytes).unwrap_err();
        assert!(err.contains("truncated"));
    }

    #[test]
    fn test_workspace_json_roundtrip() {
        let ws = sample_test_workspace();
        let json_str = ws.to_json_string().expect("JSON serialization failed");
        assert!(json_str.contains("test_suite"));
        assert!(json_str.contains("Multi-Model Test Suite"));

        let deserialized = Workspace::from_json_str(&json_str).expect("JSON parsing failed");
        assert_eq!(deserialized.id, ws.id);
        assert_eq!(deserialized.designs.len(), 2);
    }

    #[test]
    fn test_workspace_bson_roundtrip() {
        let ws = sample_test_workspace();
        let bson_bytes = ws.to_bson_bytes().expect("BSON serialization failed");
        assert!(!bson_bytes.is_empty());

        let deserialized =
            Workspace::from_bson_bytes(&bson_bytes).expect("BSON deserialization failed");
        assert_eq!(deserialized.id, ws.id);
        assert_eq!(deserialized.designs.len(), 2);

        // Test auto-detect with raw BSON
        let (unpacked_bson, is_bin) =
            unpack_workspace_auto(&bson_bytes).expect("Auto detection failed on BSON");
        assert!(is_bin);
        assert_eq!(unpacked_bson.id, "test_suite");
    }

    #[test]
    fn test_workspace_auto_detection_binary_and_json() {
        let ws = sample_test_workspace();

        // Binary .aidws auto-detection
        let bin_bytes = ws.to_aidws_bytes().unwrap();
        let (unpacked_bin, is_binary_1) =
            unpack_workspace_auto(&bin_bytes).expect("Auto detection failed on binary");
        assert!(is_binary_1);
        assert_eq!(unpacked_bin.id, "test_suite");

        // JSON text auto-detection
        let json_str = ws.to_json_string().unwrap();
        let (unpacked_json, is_binary_2) =
            unpack_workspace_auto(json_str.as_bytes()).expect("Auto detection failed on JSON");
        assert!(!is_binary_2);
        assert_eq!(unpacked_json.id, "test_suite");
    }

    #[test]
    fn test_multi_model_pipeline_crud_and_safety() {
        let mut ws = sample_test_workspace();
        assert_eq!(ws.designs.len(), 2);
        assert_eq!(ws.active_design_id, "model_a");

        // Duplicate active model
        let clone_id = ws.duplicate_design("model_a").expect("Duplication failed");
        assert_eq!(ws.designs.len(), 3);
        assert_eq!(ws.active_design_id, clone_id);

        // Switch to model_b
        assert!(ws.set_active_design("model_b"));
        assert_eq!(ws.active_design().unwrap().id, "model_b");

        // Remove clone
        assert!(ws.remove_design(&clone_id));
        assert_eq!(ws.designs.len(), 2);

        // Cannot remove down to 0 models
        assert!(ws.remove_design("model_a"));
        assert_eq!(ws.designs.len(), 1);
        assert!(!ws.remove_design("model_b")); // Refuse deleting the only remaining model
        assert_eq!(ws.designs.len(), 1);
    }

    #[test]
    fn test_workspace_registry_lifecycle() {
        let mut reg = WorkspaceRegistry::default();
        assert_eq!(reg.workspaces.len(), 1);

        // Create new workspace
        let spec2 = AiDesignSpec::new("vision_net", "Vision Net");
        let ws2_id = reg.create_workspace("vision_lab", "Vision Lab Suite", spec2);
        assert_eq!(reg.workspaces.len(), 2);
        assert_eq!(reg.active_workspace_id, ws2_id);

        // Export binary .aidws from registry
        let aidws_bytes = reg
            .export_workspace_aidws("vision_lab")
            .expect("Failed to export .aidws");
        assert_eq!(&aidws_bytes[0..6], AIDWS_MAGIC);

        // Export JSON from registry
        let json_text = reg
            .export_workspace_json("vision_lab")
            .expect("Failed to export JSON");
        assert!(json_text.contains("Vision Lab Suite"));

        // Import binary package as new workspace
        let (imported_id, is_bin) = reg
            .import_workspace_bytes(&aidws_bytes)
            .expect("Import failed");
        assert!(is_bin);
        assert_eq!(reg.workspaces.len(), 3);
        assert_eq!(reg.active_workspace_id, imported_id);

        // Duplicate workspace
        let dup_id = reg
            .duplicate_workspace(&imported_id)
            .expect("Duplication failed");
        assert_eq!(reg.workspaces.len(), 4);
        assert_eq!(reg.active_workspace_id, dup_id);

        // Delete workspaces safely
        assert!(reg.remove_workspace(&dup_id));
        assert_eq!(reg.workspaces.len(), 3);
    }
}
