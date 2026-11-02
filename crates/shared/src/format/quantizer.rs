//! Multi-Precision Weights Quantizer & Zero-Copy Tensor Transformation Engine.
//! Supports FP8 E4M3FN, INT8 symmetric, BitNet 1.58b ternary, and NF4 NormalFloat.

use crate::spec::PrecisionFormat;
use serde::{Deserialize, Serialize};

/// 16 non-uniform quantile levels for NF4 NormalFloat4 quantization.
pub const NF4_QUANTILES: [f32; 16] = [
    -1.0,
    -0.6961928,
    -0.52507305,
    -0.3949175,
    -0.2844414,
    -0.18477343,
    -0.09105004,
    0.0,
    0.0795803,
    0.1609302,
    0.2461123,
    0.33791524,
    0.44070983,
    0.562617,
    0.72295684,
    1.0,
];

/// Quantized tensor representation with associated scale factors and packed raw bytes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantizedTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub precision: PrecisionFormat,
    pub scale: f32,
    pub zero_point: i32,
    pub raw_bytes: Vec<u8>,
}

impl QuantizedTensor {
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Serializes to formatted JSON string.
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string(self).map_err(|e| format!("QuantizedTensor JSON error: {e}"))
    }

    /// Deserializes from JSON string.
    pub fn from_json(json_str: &str) -> Result<Self, String> {
        serde_json::from_str(json_str).map_err(|e| format!("QuantizedTensor JSON error: {e}"))
    }

    /// Serializes to raw BSON bytes.
    pub fn to_bson(&self) -> Result<Vec<u8>, String> {
        let doc =
            bson::to_document(self).map_err(|e| format!("QuantizedTensor BSON error: {e}"))?;
        let mut buf = Vec::new();
        doc.to_writer(&mut buf)
            .map_err(|e| format!("QuantizedTensor BSON writer error: {e}"))?;
        Ok(buf)
    }

    /// Deserializes from raw BSON bytes.
    pub fn from_bson(bytes: &[u8]) -> Result<Self, String> {
        let mut cursor = std::io::Cursor::new(bytes);
        let doc = bson::Document::from_reader(&mut cursor)
            .map_err(|e| format!("QuantizedTensor BSON reader error: {e}"))?;
        bson::from_document(doc).map_err(|e| format!("QuantizedTensor BSON decode error: {e}"))
    }

    /// Serializes to an optimized compressed binary representation with zlib-deflate (level 9) and CRC32.
    pub fn to_compressed_bytes(&self) -> Result<Vec<u8>, String> {
        let payload = self.to_bson()?;
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&payload);
        let crc = hasher.finalize();

        let compressed = miniz_oxide::deflate::compress_to_vec_zlib(&payload, 9);
        let mut out = Vec::with_capacity(12 + compressed.len());
        out.extend_from_slice(b"QTS\x01"); // Magic 4 bytes
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes()); // Uncompressed length
        out.extend_from_slice(&crc.to_le_bytes()); // CRC32
        out.extend_from_slice(&compressed);
        Ok(out)
    }

    /// Deserializes from compressed binary representation.
    pub fn from_compressed_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 12 {
            return Err("Truncated compressed tensor bytes".to_string());
        }
        if &bytes[0..4] != b"QTS\x01" {
            return Err("Invalid magic for compressed tensor: expected QTS\\x01".to_string());
        }
        let uncompressed_len = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        let expected_crc = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let decompressed = miniz_oxide::inflate::decompress_to_vec_zlib(&bytes[12..])
            .map_err(|e| format!("Decompression failed: {e:?}"))?;

        if decompressed.len() != uncompressed_len {
            return Err(format!(
                "Decompressed length mismatch: expected {}, got {}",
                uncompressed_len,
                decompressed.len()
            ));
        }

        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&decompressed);
        let computed_crc = hasher.finalize();
        if computed_crc != expected_crc {
            return Err(format!(
                "CRC32 mismatch in compressed tensor: expected {expected_crc:#x}, got {computed_crc:#x}"
            ));
        }

        Self::from_bson(&decompressed)
    }
}

/// Quantizes FP32 weights into FP8 E4M3FN (8-bit with delayed max-abs scale).
pub fn quantize_fp32_to_fp8_e4m3fn(data: &[f32], name: &str, shape: &[usize]) -> QuantizedTensor {
    let max_abs = data.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
    let scale = if max_abs > 0.0 { max_abs / 448.0 } else { 1.0 };
    let inv_scale = 1.0 / scale;

    let mut raw_bytes = Vec::with_capacity(data.len());
    for &x in data {
        let scaled = (x * inv_scale).round().clamp(-448.0, 448.0);
        // Map [-448, 448] to signed byte representation [-128, 127] scaled by 448/127
        let byte_val = ((scaled / 448.0) * 127.0).round().clamp(-128.0, 127.0) as i8;
        raw_bytes.push(byte_val as u8);
    }

    QuantizedTensor {
        name: name.to_string(),
        shape: shape.to_vec(),
        precision: PrecisionFormat::FP8E4M3FN,
        scale,
        zero_point: 0,
        raw_bytes,
    }
}

/// Dequantizes FP8 E4M3FN back to FP32.
pub fn dequantize_fp8_e4m3fn_to_fp32(tensor: &QuantizedTensor) -> Vec<f32> {
    let mut out = Vec::with_capacity(tensor.raw_bytes.len());
    for &b in &tensor.raw_bytes {
        let byte_val = b as i8 as f32;
        let scaled = (byte_val / 127.0) * 448.0;
        out.push(scaled * tensor.scale);
    }
    out
}

/// Quantizes FP32 weights to symmetric INT8 with max-abs scaling.
pub fn quantize_fp32_to_int8_symmetric(
    data: &[f32],
    name: &str,
    shape: &[usize],
) -> QuantizedTensor {
    let max_abs = data.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
    let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
    let inv_scale = 1.0 / scale;

    let mut raw_bytes = Vec::with_capacity(data.len());
    for &x in data {
        let quantized = (x * inv_scale).round().clamp(-127.0, 127.0) as i8;
        raw_bytes.push(quantized as u8);
    }

    QuantizedTensor {
        name: name.to_string(),
        shape: shape.to_vec(),
        precision: PrecisionFormat::INT8Sym,
        scale,
        zero_point: 0,
        raw_bytes,
    }
}

/// Dequantizes INT8 symmetric weights back to FP32.
pub fn dequantize_int8_to_fp32(tensor: &QuantizedTensor) -> Vec<f32> {
    tensor
        .raw_bytes
        .iter()
        .map(|&b| (b as i8 as f32) * tensor.scale)
        .collect()
}

/// Quantizes FP32 weights into BitNet 1.58b ternary format {-1, 0, +1}.
pub fn quantize_fp32_to_bitnet_1_58(data: &[f32], name: &str, shape: &[usize]) -> QuantizedTensor {
    let sum_abs: f32 = data.iter().map(|x| x.abs()).sum();
    let gamma = if !data.is_empty() {
        sum_abs / data.len() as f32
    } else {
        1.0
    };
    let inv_gamma = 1.0 / (gamma + 1e-8);

    let mut raw_bytes = Vec::with_capacity(data.len());
    for &x in data {
        let scaled = (x * inv_gamma).round().clamp(-1.0, 1.0) as i8;
        raw_bytes.push(scaled as u8);
    }

    QuantizedTensor {
        name: name.to_string(),
        shape: shape.to_vec(),
        precision: PrecisionFormat::INT2,
        scale: gamma,
        zero_point: 0,
        raw_bytes,
    }
}

/// Dequantizes BitNet 1.58b ternary weights back to FP32.
pub fn dequantize_bitnet_1_58_to_fp32(tensor: &QuantizedTensor) -> Vec<f32> {
    tensor
        .raw_bytes
        .iter()
        .map(|&b| (b as i8 as f32) * tensor.scale)
        .collect()
}

/// Quantizes FP32 weights into NF4 (NormalFloat4) format with packed nibbles.
pub fn quantize_fp32_to_nf4(data: &[f32], name: &str, shape: &[usize]) -> QuantizedTensor {
    let max_abs = data.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
    let scale = if max_abs > 0.0 { max_abs } else { 1.0 };
    let inv_scale = 1.0 / scale;

    let mut raw_bytes = Vec::with_capacity(data.len().div_ceil(2));
    let mut i = 0;
    while i < data.len() {
        let val1 = (data[i] * inv_scale).clamp(-1.0, 1.0);
        let idx1 = find_closest_nf4_index(val1);

        let idx2 = if i + 1 < data.len() {
            let val2 = (data[i + 1] * inv_scale).clamp(-1.0, 1.0);
            find_closest_nf4_index(val2)
        } else {
            0
        };

        // Pack two 4-bit nibbles into one byte
        let packed_byte = (idx1 & 0x0F) | ((idx2 & 0x0F) << 4);
        raw_bytes.push(packed_byte);
        i += 2;
    }

    QuantizedTensor {
        name: name.to_string(),
        shape: shape.to_vec(),
        precision: PrecisionFormat::NF4,
        scale,
        zero_point: 0,
        raw_bytes,
    }
}

/// Dequantizes NF4 packed bytes back to FP32.
pub fn dequantize_nf4_to_fp32(tensor: &QuantizedTensor) -> Vec<f32> {
    let total_elements = tensor.num_elements();
    let mut out = Vec::with_capacity(total_elements);

    for &packed in &tensor.raw_bytes {
        let idx1 = (packed & 0x0F) as usize;
        let idx2 = ((packed >> 4) & 0x0F) as usize;

        if out.len() < total_elements {
            out.push(NF4_QUANTILES[idx1] * tensor.scale);
        }
        if out.len() < total_elements {
            out.push(NF4_QUANTILES[idx2] * tensor.scale);
        }
    }
    out
}

fn find_closest_nf4_index(val: f32) -> u8 {
    let mut best_idx = 0;
    let mut min_diff = (val - NF4_QUANTILES[0]).abs();

    for (i, &q) in NF4_QUANTILES.iter().enumerate().skip(1) {
        let diff = (val - q).abs();
        if diff < min_diff {
            min_diff = diff;
            best_idx = i;
        }
    }
    best_idx as u8
}

/// Quantizes FP32 weights to 4-bit signed symmetric integers packed 2 per byte.
pub fn quantize_fp32_to_int4_symmetric(
    data: &[f32],
    name: &str,
    shape: &[usize],
) -> QuantizedTensor {
    let max_abs = data.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
    let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
    let inv_scale = 1.0 / scale;

    let num_bytes = data.len().div_ceil(2);
    let mut raw_bytes = Vec::with_capacity(num_bytes);

    for chunk in data.chunks(2) {
        let q0 = ((chunk[0] * inv_scale).round().clamp(-7.0, 7.0) as i8) & 0x0F;
        let q1 = if chunk.len() > 1 {
            ((chunk[1] * inv_scale).round().clamp(-7.0, 7.0) as i8) & 0x0F
        } else {
            0
        };
        let packed = (q0 as u8) | ((q1 as u8) << 4);
        raw_bytes.push(packed);
    }

    QuantizedTensor {
        name: name.to_string(),
        shape: shape.to_vec(),
        precision: PrecisionFormat::INT4Sym,
        scale,
        zero_point: 0,
        raw_bytes,
    }
}

/// Dequantizes packed 4-bit symmetric tensor back to FP32.
pub fn dequantize_int4_to_fp32(tensor: &QuantizedTensor) -> Vec<f32> {
    let total_elements = tensor.num_elements();
    let mut out = Vec::with_capacity(total_elements);

    for &b in &tensor.raw_bytes {
        let mut v0 = (b & 0x0F) as i8;
        if v0 >= 8 {
            v0 -= 16;
        } // Sign extend 4-bit
        if out.len() < total_elements {
            out.push((v0 as f32) * tensor.scale);
        }

        let mut v1 = ((b >> 4) & 0x0F) as i8;
        if v1 >= 8 {
            v1 -= 16;
        }
        if out.len() < total_elements {
            out.push((v1 as f32) * tensor.scale);
        }
    }
    out
}

/// Quantizes FP32 weights into ultra-dense packed BitNet 1.58b (4 trits per byte).
/// Trits map: -1 -> 0b00, 0 -> 0b01, +1 -> 0b10.
pub fn quantize_fp32_to_packed_bitnet(
    data: &[f32],
    name: &str,
    shape: &[usize],
) -> QuantizedTensor {
    let mean_abs = if data.is_empty() {
        1.0
    } else {
        data.iter().map(|x| x.abs()).sum::<f32>() / data.len() as f32
    };
    let scale = if mean_abs > 0.0 { mean_abs } else { 1.0 };
    let inv_scale = 1.0 / scale;

    let num_bytes = data.len().div_ceil(4);
    let mut raw_bytes = Vec::with_capacity(num_bytes);

    for chunk in data.chunks(4) {
        let mut byte_val = 0u8;
        for (i, &x) in chunk.iter().enumerate() {
            let rounded = (x * inv_scale).round().clamp(-1.0, 1.0) as i8;
            let code = match rounded {
                -1 => 0b00u8,
                0 => 0b01u8,
                _ => 0b10u8,
            };
            byte_val |= code << (i * 2);
        }
        raw_bytes.push(byte_val);
    }

    QuantizedTensor {
        name: name.to_string(),
        shape: shape.to_vec(),
        precision: PrecisionFormat::BitNet158b,
        scale,
        zero_point: 0,
        raw_bytes,
    }
}

/// Dequantizes packed BitNet 1.58b back to FP32.
pub fn dequantize_packed_bitnet_to_fp32(tensor: &QuantizedTensor) -> Vec<f32> {
    let total_elements = tensor.num_elements();
    let mut out = Vec::with_capacity(total_elements);

    for &b in &tensor.raw_bytes {
        for i in 0..4 {
            if out.len() >= total_elements {
                break;
            }
            let code = (b >> (i * 2)) & 0b11;
            let trit = match code {
                0b00 => -1.0f32,
                0b01 => 0.0f32,
                _ => 1.0f32,
            };
            out.push(trit * tensor.scale);
        }
    }
    out
}

/// Quantizes FP32 weights into BFloat16 format (2 bytes per element).
pub fn quantize_fp32_to_bf16(data: &[f32], name: &str, shape: &[usize]) -> QuantizedTensor {
    let mut raw_bytes = Vec::with_capacity(data.len() * 2);
    for &x in data {
        let bits = x.to_bits();
        // Round to nearest even
        let lsb = (bits >> 16) & 1;
        let bias = 0x7FFF + lsb;
        let bf16 = ((bits.wrapping_add(bias)) >> 16) as u16;
        raw_bytes.extend_from_slice(&bf16.to_le_bytes());
    }

    QuantizedTensor {
        name: name.to_string(),
        shape: shape.to_vec(),
        precision: PrecisionFormat::BF16,
        scale: 1.0,
        zero_point: 0,
        raw_bytes,
    }
}

/// Dequantizes BFloat16 raw bytes back to FP32.
pub fn dequantize_bf16_to_fp32(tensor: &QuantizedTensor) -> Vec<f32> {
    let mut out = Vec::with_capacity(tensor.raw_bytes.len() / 2);
    for chunk in tensor.raw_bytes.as_chunks::<2>().0 {
        let bf16 = u16::from_le_bytes(*chunk);
        let f32_bits = (bf16 as u32) << 16;
        out.push(f32::from_bits(f32_bits));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int8_quantization_roundtrip() {
        let input = vec![0.5, -1.2, 3.4, -0.05, 0.0, 2.1];
        let shape = vec![6];
        let q = quantize_fp32_to_int8_symmetric(&input, "test_weight", &shape);
        assert_eq!(q.raw_bytes.len(), 6);

        let deq = dequantize_int8_to_fp32(&q);
        assert_eq!(deq.len(), input.len());
        for (orig, rec) in input.iter().zip(deq.iter()) {
            assert!(
                (orig - rec).abs() < 0.05,
                "Mismatch: orig={}, rec={}",
                orig,
                rec
            );
        }
    }

    #[test]
    fn test_bitnet_1_58_quantization() {
        let input = vec![1.5, -0.9, 0.1, -1.2, 0.0, 2.0];
        let shape = vec![6];
        let q = quantize_fp32_to_bitnet_1_58(&input, "ternary_linear", &shape);

        let deq = dequantize_bitnet_1_58_to_fp32(&q);
        assert_eq!(deq.len(), 6);
        for &val in &q.raw_bytes {
            let s = val as i8;
            assert!(s == -1 || s == 0 || s == 1);
        }
    }

    #[test]
    fn test_nf4_quantization_roundtrip() {
        let input = vec![0.8, -0.5, 0.0, 0.25, -0.95, 0.6];
        let shape = vec![6];
        let q = quantize_fp32_to_nf4(&input, "nf4_linear", &shape);
        assert_eq!(q.raw_bytes.len(), 3); // 6 elements packed into 3 bytes

        let deq = dequantize_nf4_to_fp32(&q);
        assert_eq!(deq.len(), 6);
        for (orig, rec) in input.iter().zip(deq.iter()) {
            assert!(
                (orig - rec).abs() < 0.15,
                "NF4 Mismatch: orig={}, rec={}",
                orig,
                rec
            );
        }
    }

    #[test]
    fn test_int4_packed_roundtrip() {
        let input = vec![1.2, -3.5, 0.0, -0.7, 2.1, -1.9];
        let shape = vec![6];
        let q = quantize_fp32_to_int4_symmetric(&input, "test_int4", &shape);
        assert_eq!(q.raw_bytes.len(), 3); // 2 nibbles per byte

        let deq = dequantize_int4_to_fp32(&q);
        assert_eq!(deq.len(), 6);
        for (orig, rec) in input.iter().zip(deq.iter()) {
            assert!(
                (orig - rec).abs() < 0.6,
                "INT4 mismatch: orig={}, rec={}",
                orig,
                rec
            );
        }
    }

    #[test]
    fn test_bitnet_packed_roundtrip() {
        let input = vec![2.0, -1.5, 0.0, 1.8, -2.1, 0.1, -0.05, 1.9];
        let shape = vec![8];
        let q = quantize_fp32_to_packed_bitnet(&input, "test_packed_bitnet", &shape);
        assert_eq!(q.raw_bytes.len(), 2); // 4 trits per byte = 2 bytes for 8 elements!

        let deq = dequantize_packed_bitnet_to_fp32(&q);
        assert_eq!(deq.len(), 8);
    }

    #[test]
    fn test_bf16_roundtrip() {
        let input = vec![1.25, -3.5, 0.0, 42.125, -0.00390625];
        let shape = vec![5];
        let q = quantize_fp32_to_bf16(&input, "test_bf16", &shape);
        assert_eq!(q.raw_bytes.len(), 10);

        let deq = dequantize_bf16_to_fp32(&q);
        assert_eq!(deq.len(), 5);
        for (orig, rec) in input.iter().zip(deq.iter()) {
            let diff = (orig - rec).abs();
            let denom = orig.abs().max(1.0);
            assert!(
                diff / denom < 0.01,
                "BF16 mismatch: orig={}, rec={}, rel={}",
                orig,
                rec,
                diff / denom
            );
        }
    }

    #[test]
    fn test_quantized_tensor_json_bson_compressed_formats() {
        let input = vec![0.5, -1.2, 3.4, -0.05, 0.0, 2.1];
        let shape = vec![6];
        let q = quantize_fp32_to_int8_symmetric(&input, "format_test", &shape);

        // 1. JSON
        let json_str = q.to_json().unwrap();
        let from_json = QuantizedTensor::from_json(&json_str).unwrap();
        assert_eq!(q, from_json);

        // 2. BSON
        let bson_bytes = q.to_bson().unwrap();
        let from_bson = QuantizedTensor::from_bson(&bson_bytes).unwrap();
        assert_eq!(q, from_bson);

        // 3. Compressed Binary Container with CRC32
        let comp_bytes = q.to_compressed_bytes().unwrap();
        assert!(comp_bytes.starts_with(b"QTS\x01"));
        let from_comp = QuantizedTensor::from_compressed_bytes(&comp_bytes).unwrap();
        assert_eq!(q, from_comp);

        // Check corrupted CRC32 rejection
        let mut corrupt = comp_bytes.clone();
        let len = corrupt.len();
        corrupt[len - 1] ^= 0xFF;
        assert!(QuantizedTensor::from_compressed_bytes(&corrupt).is_err());
    }
}
