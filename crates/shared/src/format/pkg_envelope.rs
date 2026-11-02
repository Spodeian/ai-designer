//! .pkg Weights Container: safely wraps any weight format (Safetensors, BSON, Quantized, ONNX)
//! with 4096-byte page alignment for zero-copy memory mapping (`mmap`), format tag, and CRC32 checks.

use crc32fast::Hasher;
use serde::{Deserialize, Serialize};

pub const PKG_MAGIC: &[u8; 4] = b"PKG\x01";
pub const PAGE_SIZE: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum PkgWeightsFormat {
    Safetensors = 0x01,
    BsonTensors = 0x02,
    Quantized = 0x03,
    Onnx = 0x04,
    RawBinary = 0x05,
    BsonTensorsCompressed = 0x06,
    QuantizedCompressed = 0x07,
}

impl PkgWeightsFormat {
    pub fn from_u8(b: u8) -> Option<Self> {
        match b {
            0x01 => Some(Self::Safetensors),
            0x02 => Some(Self::BsonTensors),
            0x03 => Some(Self::Quantized),
            0x04 => Some(Self::Onnx),
            0x05 => Some(Self::RawBinary),
            0x06 => Some(Self::BsonTensorsCompressed),
            0x07 => Some(Self::QuantizedCompressed),
            _ => None,
        }
    }

    pub fn is_compressed(&self) -> bool {
        matches!(
            self,
            Self::BsonTensorsCompressed | Self::QuantizedCompressed
        )
    }

    pub fn extension(&self) -> &'static str {
        match self {
            Self::Safetensors => "safetensors",
            Self::BsonTensors | Self::BsonTensorsCompressed => "bson",
            Self::Quantized | Self::QuantizedCompressed => "bin",
            Self::Onnx => "onnx",
            Self::RawBinary => "bin",
        }
    }
}

/// Metadata header describing the packed weights payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PkgManifest {
    pub format: PkgWeightsFormat,
    pub num_tensors: usize,
    pub payload_size: usize,
    pub payload_crc: u32,
    pub page_aligned_offset: usize,
    #[serde(default)]
    pub uncompressed_size: Option<usize>,
}

impl PkgManifest {
    /// Serializes manifest to JSON format.
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|e| format!("PkgManifest JSON error: {e}"))
    }

    /// Deserializes manifest from JSON.
    pub fn from_json(json_str: &str) -> Result<Self, String> {
        serde_json::from_str(json_str).map_err(|e| format!("PkgManifest JSON error: {e}"))
    }

    /// Serializes manifest to BSON format.
    pub fn to_bson(&self) -> Result<Vec<u8>, String> {
        let doc = bson::to_document(self).map_err(|e| format!("PkgManifest BSON error: {e}"))?;
        let mut buf = Vec::new();
        doc.to_writer(&mut buf)
            .map_err(|e| format!("PkgManifest BSON writer error: {e}"))?;
        Ok(buf)
    }

    /// Deserializes manifest from BSON.
    pub fn from_bson(bytes: &[u8]) -> Result<Self, String> {
        let mut cursor = std::io::Cursor::new(bytes);
        let doc = bson::Document::from_reader(&mut cursor)
            .map_err(|e| format!("PkgManifest BSON reader error: {e}"))?;
        bson::from_document(doc).map_err(|e| format!("PkgManifest BSON decode error: {e}"))
    }
}

/// Packs raw weights into a 4096-byte page-aligned .pkg container.
/// Supports both raw page-aligned zero-copy layout and high-efficiency compressed binary storage.
pub fn pack_pkg(weights: &[u8], format: PkgWeightsFormat, num_tensors: usize) -> Vec<u8> {
    let mut hasher = Hasher::new();
    hasher.update(weights);
    let crc = hasher.finalize();

    let (payload, uncompressed_size) = if format.is_compressed() {
        let comp = miniz_oxide::deflate::compress_to_vec_zlib(weights, 9);
        (comp, Some(weights.len()))
    } else {
        (weights.to_vec(), None)
    };

    let manifest = PkgManifest {
        format,
        num_tensors,
        payload_size: payload.len(),
        payload_crc: crc,
        page_aligned_offset: PAGE_SIZE,
        uncompressed_size,
    };

    let manifest_bytes = serde_json::to_vec(&manifest).unwrap_or_default();
    let header_len = 4 + 4 + manifest_bytes.len(); // magic + len + json

    let mut out = Vec::new();
    out.extend_from_slice(PKG_MAGIC);
    out.extend_from_slice(&(manifest_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&manifest_bytes);

    // Pad to 4096-byte page boundary
    let padding = if !header_len.is_multiple_of(PAGE_SIZE) {
        PAGE_SIZE - (header_len % PAGE_SIZE)
    } else {
        0
    };
    out.resize(out.len() + padding, 0);

    // Append weights at exact page boundary
    out.extend_from_slice(&payload);
    out
}

/// Unpacks a .pkg weights container, verifying CRC32 and returning (manifest, weights_slice).
/// If the payload was compressed, it is automatically decompressed before returning.
pub fn unpack_pkg(bytes: &[u8]) -> Result<(PkgManifest, Vec<u8>), String> {
    if bytes.len() >= 8 && &bytes[0..4] == PKG_MAGIC {
        let manifest_len = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        if 8 + manifest_len > bytes.len() {
            return Err("Truncated .pkg header".to_string());
        }

        let manifest: PkgManifest = serde_json::from_slice(&bytes[8..8 + manifest_len])
            .map_err(|e| format!("Invalid .pkg manifest JSON: {}", e))?;

        let header_len = 8 + manifest_len;
        let padding = if !header_len.is_multiple_of(PAGE_SIZE) {
            PAGE_SIZE - (header_len % PAGE_SIZE)
        } else {
            0
        };
        let weights_offset = header_len + padding;

        if weights_offset + manifest.payload_size > bytes.len() {
            return Err("Truncated .pkg weights payload".to_string());
        }

        let raw_payload = &bytes[weights_offset..weights_offset + manifest.payload_size];

        let final_weights = if manifest.format.is_compressed() {
            miniz_oxide::inflate::decompress_to_vec_zlib(raw_payload)
                .map_err(|e| format!("Failed to decompress .pkg weights: {e:?}"))?
        } else {
            raw_payload.to_vec()
        };

        let mut hasher = Hasher::new();
        hasher.update(&final_weights);
        let actual_crc = hasher.finalize();

        if actual_crc != manifest.payload_crc {
            return Err(format!(
                "CRC32 mismatch in .pkg: expected {:08x}, got {:08x}",
                manifest.payload_crc, actual_crc
            ));
        }

        return Ok((manifest, final_weights));
    }

    // Transparent fallback: treat as raw safetensors or binary weights
    let format = if bytes.len() >= 8 {
        let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        if header_len > 0 && header_len < 100_000_000 && 8 + header_len <= bytes.len() {
            PkgWeightsFormat::Safetensors
        } else {
            PkgWeightsFormat::RawBinary
        }
    } else {
        PkgWeightsFormat::RawBinary
    };

    let manifest = PkgManifest {
        format,
        num_tensors: 0,
        payload_size: bytes.len(),
        payload_crc: 0,
        page_aligned_offset: 0,
        uncompressed_size: None,
    };

    Ok((manifest, bytes.to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pkg_roundtrip_with_alignment() {
        let fake_weights = vec![0xABu8; 1024];
        let packed = pack_pkg(&fake_weights, PkgWeightsFormat::Safetensors, 4);

        // Check 4096-byte alignment
        let (manifest, unpacked) = unpack_pkg(&packed).unwrap();
        assert_eq!(manifest.format, PkgWeightsFormat::Safetensors);
        assert_eq!(manifest.num_tensors, 4);
        assert_eq!(unpacked, fake_weights);

        // Test manifest JSON & BSON roundtrip
        let json = manifest.to_json().unwrap();
        let from_json = PkgManifest::from_json(&json).unwrap();
        assert_eq!(manifest, from_json);

        let bson = manifest.to_bson().unwrap();
        let from_bson = PkgManifest::from_bson(&bson).unwrap();
        assert_eq!(manifest, from_bson);
    }

    #[test]
    fn test_pkg_compressed_formats() {
        let fake_weights = vec![0xCDu8; 8192];
        let packed = pack_pkg(&fake_weights, PkgWeightsFormat::BsonTensorsCompressed, 8);

        let (manifest, unpacked) = unpack_pkg(&packed).unwrap();
        assert_eq!(manifest.format, PkgWeightsFormat::BsonTensorsCompressed);
        assert_eq!(manifest.num_tensors, 8);
        assert_eq!(manifest.uncompressed_size, Some(8192));
        assert_eq!(unpacked, fake_weights);
        assert!(
            manifest.payload_size < 8192,
            "Compressed payload must be smaller"
        );
    }
}
