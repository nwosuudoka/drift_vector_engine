use crate::unified_format::{
    UNIFIED_FLAG_HAS_EXACT_INDEX, UNIFIED_FLAG_HAS_PAYLOAD_COLUMNS,
    UNIFIED_FLAG_HAS_PAYLOAD_SCHEMA, UNIFIED_FOOTER_SIZE, UNIFIED_HEADER_SIZE, UnifiedBlockDesc,
    UnifiedBlockType, UnifiedCodec, UnifiedExactIndex, UnifiedFieldSchema, UnifiedFooter,
    UnifiedHeader, UnifiedLogicalType, UnifiedPayloadColumnChunk, UnifiedPayloadRow,
    UnifiedPayloadSchema, UnifiedPayloadValue, decode_block_directory, encode_block_directory,
    encode_exact_key,
};
use byteorder::{LittleEndian, WriteBytesExt};
use crc32fast::Hasher;
use drift_core::quantizer::Quantizer;
use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io::{self, Cursor, Seek, SeekFrom, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnifiedWriteStats {
    pub row_count: u64,
    pub dim: u32,
    pub file_len: u64,
}

pub struct UnifiedRemoteWriter;
pub struct UnifiedLocalWriter;

impl UnifiedRemoteWriter {
    pub fn write_vector_only<W: Write + Seek>(
        writer: &mut W,
        ids: &[u64],
        vectors: &[Vec<f32>],
        dim: usize,
    ) -> io::Result<UnifiedWriteStats> {
        let flat = flatten_vectors(vectors, dim)?;
        Self::write_vector_only_flat(writer, ids, &flat, dim)
    }

    pub fn write_vector_only_flat<W: Write + Seek>(
        writer: &mut W,
        ids: &[u64],
        flat_vectors: &[f32],
        dim: usize,
    ) -> io::Result<UnifiedWriteStats> {
        Self::write_vector_with_schema_flat(writer, ids, flat_vectors, dim, None)
    }

    pub fn write_vector_with_schema_flat<W: Write + Seek>(
        writer: &mut W,
        ids: &[u64],
        flat_vectors: &[f32],
        dim: usize,
        payload_schema: Option<&UnifiedPayloadSchema>,
    ) -> io::Result<UnifiedWriteStats> {
        Self::write_vector_with_payload_flat(writer, ids, flat_vectors, dim, payload_schema, None)
    }

    pub fn write_vector_with_payload_flat<W: Write + Seek>(
        writer: &mut W,
        ids: &[u64],
        flat_vectors: &[f32],
        dim: usize,
        payload_schema: Option<&UnifiedPayloadSchema>,
        payload_rows: Option<&[UnifiedPayloadRow]>,
    ) -> io::Result<UnifiedWriteStats> {
        validate_input_flat(ids, flat_vectors, dim)?;

        let row_count = ids.len() as u64;
        let blocks_and_payload =
            build_chunk_blocks(0, ids, flat_vectors, dim, payload_schema, payload_rows)?;
        let (blocks, mut payload) = blocks_and_payload;
        let mut flags = 0u32;
        if blocks
            .iter()
            .any(|b| b.block_type == UnifiedBlockType::PayloadSchema)
        {
            flags |= UNIFIED_FLAG_HAS_PAYLOAD_SCHEMA;
        }
        if blocks
            .iter()
            .any(|b| b.block_type == UnifiedBlockType::PayloadColumn)
        {
            flags |= UNIFIED_FLAG_HAS_PAYLOAD_COLUMNS;
        }
        if blocks
            .iter()
            .any(|b| b.block_type == UnifiedBlockType::PayloadExactIndex)
        {
            flags |= UNIFIED_FLAG_HAS_EXACT_INDEX;
        }

        writer.seek(SeekFrom::Start(0))?;
        writer.write_all(&[0u8; UNIFIED_HEADER_SIZE])?;
        writer.write_all(&payload)?;

        let block_dir_offset = UNIFIED_HEADER_SIZE as u64 + payload.len() as u64;
        let block_dir_bytes = encode_block_directory(&blocks)?;
        writer.write_all(&block_dir_bytes)?;

        let footer_offset = block_dir_offset + block_dir_bytes.len() as u64;
        let footer = UnifiedFooter {
            flags,
            row_count,
            block_dir_offset,
            block_count: blocks.len() as u32,
            directory_crc32: checksum32(&block_dir_bytes),
        };
        let footer_bytes = footer.encode()?;
        writer.write_all(&footer_bytes)?;

        let file_len = footer_offset + footer_bytes.len() as u64;
        let header = UnifiedHeader {
            flags,
            dim: dim as u32,
            row_count,
            quantizer_offset: blocks[0].offset,
            quantizer_length: blocks[0].compressed_len,
            block_dir_offset,
            block_count: blocks.len() as u32,
            footer_offset,
            footer_length: UNIFIED_FOOTER_SIZE as u32,
            created_at_unix_secs: now_unix_secs(),
        };

        writer.seek(SeekFrom::Start(0))?;
        writer.write_all(&header.encode()?)?;
        writer.seek(SeekFrom::Start(file_len))?;
        writer.flush()?;

        // Keep payload alive until after writes.
        payload.clear();

        Ok(UnifiedWriteStats {
            row_count,
            dim: dim as u32,
            file_len,
        })
    }

    pub fn write_vector_only_to_bytes(
        ids: &[u64],
        vectors: &[Vec<f32>],
        dim: usize,
    ) -> io::Result<Vec<u8>> {
        let flat = flatten_vectors(vectors, dim)?;
        Self::write_vector_only_flat_to_bytes(ids, &flat, dim)
    }

    pub fn write_vector_only_flat_to_bytes(
        ids: &[u64],
        flat_vectors: &[f32],
        dim: usize,
    ) -> io::Result<Vec<u8>> {
        Self::write_vector_with_schema_flat_to_bytes(ids, flat_vectors, dim, None)
    }

    pub fn write_vector_with_schema_flat_to_bytes(
        ids: &[u64],
        flat_vectors: &[f32],
        dim: usize,
        payload_schema: Option<&UnifiedPayloadSchema>,
    ) -> io::Result<Vec<u8>> {
        Self::write_vector_with_payload_flat_to_bytes(ids, flat_vectors, dim, payload_schema, None)
    }

    pub fn write_vector_with_payload_flat_to_bytes(
        ids: &[u64],
        flat_vectors: &[f32],
        dim: usize,
        payload_schema: Option<&UnifiedPayloadSchema>,
        payload_rows: Option<&[UnifiedPayloadRow]>,
    ) -> io::Result<Vec<u8>> {
        let mut cursor = Cursor::new(Vec::new());
        let _ = Self::write_vector_with_payload_flat(
            &mut cursor,
            ids,
            flat_vectors,
            dim,
            payload_schema,
            payload_rows,
        )?;
        Ok(cursor.into_inner())
    }
}

impl UnifiedLocalWriter {
    pub fn write_vector_only_to_path(
        path: impl AsRef<Path>,
        ids: &[u64],
        vectors: &[Vec<f32>],
        dim: usize,
    ) -> io::Result<UnifiedWriteStats> {
        let flat = flatten_vectors(vectors, dim)?;
        Self::write_vector_only_flat_to_path(path, ids, &flat, dim)
    }

    pub fn write_vector_only_flat_to_path(
        path: impl AsRef<Path>,
        ids: &[u64],
        flat_vectors: &[f32],
        dim: usize,
    ) -> io::Result<UnifiedWriteStats> {
        Self::write_vector_with_schema_flat_to_path(path, ids, flat_vectors, dim, None)
    }

    pub fn write_vector_with_schema_flat_to_path(
        path: impl AsRef<Path>,
        ids: &[u64],
        flat_vectors: &[f32],
        dim: usize,
        payload_schema: Option<&UnifiedPayloadSchema>,
    ) -> io::Result<UnifiedWriteStats> {
        Self::write_vector_with_payload_flat_to_path(
            path,
            ids,
            flat_vectors,
            dim,
            payload_schema,
            None,
        )
    }

    pub fn write_vector_with_payload_flat_to_path(
        path: impl AsRef<Path>,
        ids: &[u64],
        flat_vectors: &[f32],
        dim: usize,
        payload_schema: Option<&UnifiedPayloadSchema>,
        payload_rows: Option<&[UnifiedPayloadRow]>,
    ) -> io::Result<UnifiedWriteStats> {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        let stats = UnifiedRemoteWriter::write_vector_with_payload_flat(
            &mut file,
            ids,
            flat_vectors,
            dim,
            payload_schema,
            payload_rows,
        )?;
        file.sync_all()?;
        Ok(stats)
    }

    pub fn append_vector_chunk_to_path(
        path: impl AsRef<Path>,
        ids: &[u64],
        flat_vectors: &[f32],
        dim: usize,
    ) -> io::Result<UnifiedWriteStats> {
        Self::append_vector_chunk_with_schema_to_path(path, ids, flat_vectors, dim, None)
    }

    pub fn append_vector_chunk_with_schema_to_path(
        path: impl AsRef<Path>,
        ids: &[u64],
        flat_vectors: &[f32],
        dim: usize,
        payload_schema: Option<&UnifiedPayloadSchema>,
    ) -> io::Result<UnifiedWriteStats> {
        validate_input_flat(ids, flat_vectors, dim)?;
        let path = path.as_ref();

        if !path.exists() || std::fs::metadata(path)?.len() == 0 {
            return Self::write_vector_with_schema_flat_to_path(
                path,
                ids,
                flat_vectors,
                dim,
                payload_schema,
            );
        }

        let mut bytes = std::fs::read(path)?;
        if bytes.len() < UNIFIED_HEADER_SIZE + UNIFIED_FOOTER_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unified local file too small for append",
            ));
        }

        let mut header = UnifiedHeader::decode(&bytes[..UNIFIED_HEADER_SIZE])?;
        if header.dim as usize != dim {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "append dim mismatch: existing={}, requested={}",
                    header.dim, dim
                ),
            ));
        }

        let footer_end = header
            .footer_offset
            .checked_add(header.footer_length as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "append footer overflow"))?;
        if footer_end as usize > bytes.len() || header.footer_length as usize != UNIFIED_FOOTER_SIZE
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "append invalid footer position",
            ));
        }

        let footer = UnifiedFooter::decode(
            &bytes[header.footer_offset as usize
                ..(header.footer_offset as usize + UNIFIED_FOOTER_SIZE)],
        )?;
        if footer.flags != header.flags {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "append header/footer flags mismatch",
            ));
        }
        if footer.block_dir_offset != header.block_dir_offset
            || footer.block_count != header.block_count
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "append header/footer directory mismatch",
            ));
        }

        let dir_len = 4usize
            .checked_add(
                (header.block_count as usize)
                    .saturating_mul(crate::unified_format::UNIFIED_BLOCK_DESC_SIZE),
            )
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "append dir size overflow")
            })?;
        let dir_end = header
            .block_dir_offset
            .checked_add(dir_len as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "append dir overflow"))?;
        if dir_end != header.footer_offset {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "append invalid directory/footer boundary",
            ));
        }

        let dir_start = header.block_dir_offset as usize;
        let dir_bytes = &bytes[dir_start..(dir_start + dir_len)];
        let mut blocks = decode_block_directory(dir_bytes)?;
        let has_schema_block = blocks
            .iter()
            .any(|b| b.block_type == UnifiedBlockType::PayloadSchema);
        let flag_has_schema = (header.flags & UNIFIED_FLAG_HAS_PAYLOAD_SCHEMA) != 0;
        if flag_has_schema != has_schema_block {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "append payload schema flag mismatch",
            ));
        }
        if matches!(payload_schema, Some(schema) if !schema.is_empty()) && !flag_has_schema {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "append schema supplied but file has no schema",
            ));
        }
        if let Some(expected_schema) = payload_schema.filter(|schema| !schema.is_empty())
            && flag_has_schema
        {
            let schema_block = blocks
                .iter()
                .find(|b| b.block_type == UnifiedBlockType::PayloadSchema)
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "append schema flag set but schema block missing",
                    )
                })?;
            let schema_start = schema_block.offset as usize;
            let schema_end = schema_start
                .checked_add(schema_block.compressed_len as usize)
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "append schema overflow")
                })?;
            if schema_end > bytes.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "append schema block out of bounds",
                ));
            }
            let (existing_schema, _): (UnifiedPayloadSchema, usize) = bincode::decode_from_slice(
                &bytes[schema_start..schema_end],
                bincode::config::standard(),
            )
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            if &existing_schema != expected_schema {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "append payload schema mismatch",
                ));
            }
        }
        if blocks.iter().any(|b| {
            matches!(
                b.block_type,
                UnifiedBlockType::PayloadColumn | UnifiedBlockType::PayloadExactIndex
            )
        }) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "append with payload columns/indexes is not yet supported",
            ));
        }

        // Truncate old directory + footer and append new chunk data.
        bytes.truncate(header.block_dir_offset as usize);
        let row_start = header.row_count;
        let current_offset = bytes.len() as u64;
        let (mut new_blocks, new_payload) =
            build_chunk_blocks(row_start, ids, flat_vectors, dim, None, None)?;

        // Shift the new chunk block offsets to where we append in this file.
        for block in &mut new_blocks {
            block.offset = block
                .offset
                .checked_add(current_offset - UNIFIED_HEADER_SIZE as u64)
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "append block offset overflow")
                })?;
        }

        bytes.extend_from_slice(&new_payload);
        blocks.extend(new_blocks);

        let block_dir_offset = bytes.len() as u64;
        let block_dir_bytes = encode_block_directory(&blocks)?;
        bytes.extend_from_slice(&block_dir_bytes);

        let new_row_count = row_start + ids.len() as u64;
        let footer = UnifiedFooter {
            flags: header.flags,
            row_count: new_row_count,
            block_dir_offset,
            block_count: blocks.len() as u32,
            directory_crc32: checksum32(&block_dir_bytes),
        };
        bytes.extend_from_slice(&footer.encode()?);

        let first_q = blocks
            .iter()
            .find(|b| b.block_type == UnifiedBlockType::Quantizer)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing quantizer block"))?;

        header.row_count = new_row_count;
        header.block_dir_offset = block_dir_offset;
        header.block_count = blocks.len() as u32;
        header.footer_offset = block_dir_offset + block_dir_bytes.len() as u64;
        header.footer_length = UNIFIED_FOOTER_SIZE as u32;
        header.quantizer_offset = first_q.offset;
        header.quantizer_length = first_q.compressed_len;
        header.created_at_unix_secs = now_unix_secs();

        let header_bytes = header.encode()?;
        bytes[..UNIFIED_HEADER_SIZE].copy_from_slice(&header_bytes);
        std::fs::write(path, &bytes)?;

        let file = OpenOptions::new().read(true).write(true).open(path)?;
        file.sync_all()?;

        Ok(UnifiedWriteStats {
            row_count: new_row_count,
            dim: dim as u32,
            file_len: bytes.len() as u64,
        })
    }
}

fn flatten_vectors(vectors: &[Vec<f32>], dim: usize) -> io::Result<Vec<f32>> {
    if dim == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "unified writer dim must be > 0",
        ));
    }
    let mut out = Vec::with_capacity(vectors.len().saturating_mul(dim));
    for (idx, vector) in vectors.iter().enumerate() {
        if vector.len() != dim {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "unified writer vector dim mismatch at row {}: expected {}, got {}",
                    idx,
                    dim,
                    vector.len()
                ),
            ));
        }
        out.extend_from_slice(vector);
    }
    Ok(out)
}

fn validate_input_flat(ids: &[u64], flat_vectors: &[f32], dim: usize) -> io::Result<()> {
    if ids.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "unified writer input is empty",
        ));
    }
    if dim == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "unified writer dim must be > 0",
        ));
    }
    let expected = ids
        .len()
        .checked_mul(dim)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "input length overflow"))?;
    if flat_vectors.len() != expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "unified writer id/vector mismatch: ids={}, flat_len={}, dim={}",
                ids.len(),
                flat_vectors.len(),
                dim
            ),
        ));
    }
    Ok(())
}

fn build_chunk_blocks(
    row_start: u64,
    ids: &[u64],
    flat_vectors: &[f32],
    dim: usize,
    payload_schema: Option<&UnifiedPayloadSchema>,
    payload_rows: Option<&[UnifiedPayloadRow]>,
) -> io::Result<(Vec<UnifiedBlockDesc>, Vec<u8>)> {
    if let Some(rows) = payload_rows
        && rows.len() != ids.len()
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "payload row count mismatch: rows={}, vectors={}",
                rows.len(),
                ids.len()
            ),
        ));
    }

    let normalized_schema = payload_schema.filter(|s| !s.is_empty());
    if payload_rows.is_some() && normalized_schema.is_none() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "payload rows supplied without non-empty payload schema",
        ));
    }

    let quantizer = Quantizer::train(flat_vectors, dim);
    let quantizer_bytes = bincode::encode_to_vec(&quantizer, bincode::config::standard())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let ids_bytes = encode_ids(ids)?;
    let codes_bytes = encode_codes_flat(flat_vectors, dim, &quantizer)?;
    let schema_bytes = match normalized_schema {
        Some(schema) if !schema.is_empty() => Some(
            bincode::encode_to_vec(schema, bincode::config::standard())
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?,
        ),
        _ => None,
    };

    let mut payload = Vec::new();
    let mut offset = UNIFIED_HEADER_SIZE as u64;

    payload.extend_from_slice(&quantizer_bytes);
    let quantizer_block = UnifiedBlockDesc {
        block_type: UnifiedBlockType::Quantizer,
        codec: UnifiedCodec::Bincode,
        row_start,
        row_count: ids.len() as u32,
        offset,
        compressed_len: quantizer_bytes.len() as u64,
        raw_len: quantizer_bytes.len() as u64,
        crc32: checksum32(&quantizer_bytes),
    };
    offset += quantizer_bytes.len() as u64;

    payload.extend_from_slice(&ids_bytes);
    let ids_block = UnifiedBlockDesc {
        block_type: UnifiedBlockType::Ids,
        codec: UnifiedCodec::PlainLe,
        row_start,
        row_count: ids.len() as u32,
        offset,
        compressed_len: ids_bytes.len() as u64,
        raw_len: ids_bytes.len() as u64,
        crc32: checksum32(&ids_bytes),
    };
    offset += ids_bytes.len() as u64;

    payload.extend_from_slice(&codes_bytes);
    let codes_block = UnifiedBlockDesc {
        block_type: UnifiedBlockType::VectorCodes,
        codec: UnifiedCodec::Sq8,
        row_start,
        row_count: ids.len() as u32,
        offset,
        compressed_len: codes_bytes.len() as u64,
        raw_len: codes_bytes.len() as u64,
        crc32: checksum32(&codes_bytes),
    };
    offset += codes_bytes.len() as u64;

    let mut blocks = vec![quantizer_block, ids_block, codes_block];
    if let Some(schema_bytes) = schema_bytes {
        payload.extend_from_slice(&schema_bytes);
        blocks.push(UnifiedBlockDesc {
            block_type: UnifiedBlockType::PayloadSchema,
            codec: UnifiedCodec::Bincode,
            row_start,
            row_count: ids.len() as u32,
            offset,
            compressed_len: schema_bytes.len() as u64,
            raw_len: schema_bytes.len() as u64,
            crc32: checksum32(&schema_bytes),
        });
        offset += schema_bytes.len() as u64;
    }

    if let Some(rows) = payload_rows {
        let schema = normalized_schema.expect("validated above");
        for field in &schema.fields {
            let mut column_values = Vec::with_capacity(rows.len());
            for row in rows {
                let value = row
                    .get(&field.field_id)
                    .cloned()
                    .unwrap_or(UnifiedPayloadValue::Null);
                if matches!(value, UnifiedPayloadValue::Null) && !field.nullable {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("non-nullable field '{}' contains null", field.name),
                    ));
                }
                column_values.push(value);
            }

            let (codec, data, validity) =
                encode_payload_column_data(&field.logical_type, &column_values)?;
            let chunk = UnifiedPayloadColumnChunk {
                field_id: field.field_id,
                logical_type: field.logical_type.clone(),
                codec,
                row_start,
                row_count: rows.len() as u32,
                validity,
                data,
            };
            let chunk_bytes = bincode::encode_to_vec(&chunk, bincode::config::standard())
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            payload.extend_from_slice(&chunk_bytes);
            blocks.push(UnifiedBlockDesc {
                block_type: UnifiedBlockType::PayloadColumn,
                codec,
                row_start,
                row_count: rows.len() as u32,
                offset,
                compressed_len: chunk_bytes.len() as u64,
                raw_len: chunk_bytes.len() as u64,
                crc32: checksum32(&chunk_bytes),
            });
            offset += chunk_bytes.len() as u64;

            if field.indexed
                && exact_index_supported_type(&field.logical_type)
                && let Some((index_bytes, index_desc)) =
                    build_exact_index_block(field, ids, &column_values, row_start, offset)?
            {
                payload.extend_from_slice(&index_bytes);
                blocks.push(index_desc);
                offset += index_bytes.len() as u64;
            }
        }
    }

    Ok((blocks, payload))
}

fn encode_payload_column_data(
    logical_type: &UnifiedLogicalType,
    values: &[UnifiedPayloadValue],
) -> io::Result<(UnifiedCodec, Vec<u8>, Option<Vec<u8>>)> {
    let mut validity = vec![0u8; values.len().div_ceil(8)];
    let mut non_null = Vec::with_capacity(values.len());
    let mut null_count = 0usize;

    for (i, value) in values.iter().enumerate() {
        if matches!(value, UnifiedPayloadValue::Null) {
            null_count += 1;
            continue;
        }
        validity[i / 8] |= 1u8 << (i % 8);
        non_null.push(value.clone());
    }

    let codec = payload_codec_for(logical_type);
    let data = encode_non_null_payload_values(logical_type, &non_null)?;
    let validity = if null_count > 0 { Some(validity) } else { None };
    Ok((codec, data, validity))
}

fn payload_codec_for(logical_type: &UnifiedLogicalType) -> UnifiedCodec {
    match logical_type {
        UnifiedLogicalType::Bool => UnifiedCodec::Bitset,
        UnifiedLogicalType::Int64
        | UnifiedLogicalType::Float32
        | UnifiedLogicalType::Float64
        | UnifiedLogicalType::TimestampMicros => UnifiedCodec::PlainLe,
        UnifiedLogicalType::Keyword
        | UnifiedLogicalType::Text
        | UnifiedLogicalType::Bytes
        | UnifiedLogicalType::LobRef => UnifiedCodec::VarLen,
    }
}

fn encode_non_null_payload_values(
    logical_type: &UnifiedLogicalType,
    values: &[UnifiedPayloadValue],
) -> io::Result<Vec<u8>> {
    let mut out = Vec::new();
    match logical_type {
        UnifiedLogicalType::Bool => {
            out.resize(values.len().div_ceil(8), 0);
            for (i, value) in values.iter().enumerate() {
                let UnifiedPayloadValue::Bool(v) = value else {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "payload bool field received non-bool value",
                    ));
                };
                if *v {
                    out[i / 8] |= 1u8 << (i % 8);
                }
            }
        }
        UnifiedLogicalType::Int64 => {
            for value in values {
                let UnifiedPayloadValue::Int64(v) = value else {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "payload int64 field received wrong value type",
                    ));
                };
                out.write_i64::<LittleEndian>(*v)?;
            }
        }
        UnifiedLogicalType::Float32 => {
            for value in values {
                let UnifiedPayloadValue::Float32(v) = value else {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "payload float32 field received wrong value type",
                    ));
                };
                out.write_f32::<LittleEndian>(*v)?;
            }
        }
        UnifiedLogicalType::Float64 => {
            for value in values {
                let UnifiedPayloadValue::Float64(v) = value else {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "payload float64 field received wrong value type",
                    ));
                };
                out.write_f64::<LittleEndian>(*v)?;
            }
        }
        UnifiedLogicalType::TimestampMicros => {
            for value in values {
                let UnifiedPayloadValue::TimestampMicros(v) = value else {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "payload timestamp field received wrong value type",
                    ));
                };
                out.write_i64::<LittleEndian>(*v)?;
            }
        }
        UnifiedLogicalType::Keyword | UnifiedLogicalType::Text => {
            for value in values {
                let raw = match value {
                    UnifiedPayloadValue::Keyword(v) | UnifiedPayloadValue::Text(v) => v.as_bytes(),
                    _ => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            "payload string field received wrong value type",
                        ));
                    }
                };
                out.write_u32::<LittleEndian>(raw.len() as u32)?;
                out.extend_from_slice(raw);
            }
        }
        UnifiedLogicalType::Bytes => {
            for value in values {
                let UnifiedPayloadValue::Bytes(raw) = value else {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "payload bytes field received wrong value type",
                    ));
                };
                out.write_u32::<LittleEndian>(raw.len() as u32)?;
                out.extend_from_slice(raw);
            }
        }
        UnifiedLogicalType::LobRef => {
            for value in values {
                let UnifiedPayloadValue::LobRef(v) = value else {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "payload lob_ref field received wrong value type",
                    ));
                };
                let encoded = bincode::encode_to_vec(v, bincode::config::standard())
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                out.write_u32::<LittleEndian>(encoded.len() as u32)?;
                out.extend_from_slice(&encoded);
            }
        }
    }
    Ok(out)
}

fn exact_index_supported_type(logical_type: &UnifiedLogicalType) -> bool {
    matches!(
        logical_type,
        UnifiedLogicalType::Bool
            | UnifiedLogicalType::Int64
            | UnifiedLogicalType::Float32
            | UnifiedLogicalType::Float64
            | UnifiedLogicalType::TimestampMicros
            | UnifiedLogicalType::Keyword
            | UnifiedLogicalType::Text
            | UnifiedLogicalType::Bytes
            | UnifiedLogicalType::LobRef
    )
}

fn build_exact_index_block(
    field: &UnifiedFieldSchema,
    ids: &[u64],
    values: &[UnifiedPayloadValue],
    row_start: u64,
    offset: u64,
) -> io::Result<Option<(Vec<u8>, UnifiedBlockDesc)>> {
    let mut index_map: BTreeMap<Vec<u8>, Vec<u64>> = BTreeMap::new();
    for (i, value) in values.iter().enumerate() {
        let Some(key) = encode_exact_key(&field.logical_type, value)? else {
            continue;
        };
        index_map.entry(key).or_default().push(ids[i]);
    }

    if index_map.is_empty() {
        return Ok(None);
    }

    let mut dictionary = Vec::with_capacity(index_map.len());
    let mut postings = Vec::with_capacity(index_map.len());
    for (key, mut rows) in index_map {
        rows.sort_unstable();
        rows.dedup();
        dictionary.push(key);
        postings.push(rows);
    }

    let exact_index = UnifiedExactIndex {
        field_id: field.field_id,
        logical_type: field.logical_type.clone(),
        dictionary,
        postings,
    };
    let bytes = bincode::encode_to_vec(&exact_index, bincode::config::standard())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let desc = UnifiedBlockDesc {
        block_type: UnifiedBlockType::PayloadExactIndex,
        codec: UnifiedCodec::DictPostings,
        row_start,
        row_count: ids.len() as u32,
        offset,
        compressed_len: bytes.len() as u64,
        raw_len: bytes.len() as u64,
        crc32: checksum32(&bytes),
    };
    Ok(Some((bytes, desc)))
}

fn encode_ids(ids: &[u64]) -> io::Result<Vec<u8>> {
    let mut out = Vec::with_capacity(ids.len().saturating_mul(8));
    for &id in ids {
        out.write_u64::<LittleEndian>(id)?;
    }
    Ok(out)
}

fn encode_codes_flat(
    flat_vectors: &[f32],
    dim: usize,
    quantizer: &Quantizer,
) -> io::Result<Vec<u8>> {
    let mut out = Vec::with_capacity(flat_vectors.len());
    for row in flat_vectors.chunks_exact(dim) {
        let code = quantizer.encode(row);
        if code.len() != dim {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "quantizer encoded unexpected code width",
            ));
        }
        out.extend_from_slice(&code);
    }
    Ok(out)
}

fn checksum32(bytes: &[u8]) -> u32 {
    let mut hasher = Hasher::new();
    hasher.update(bytes);
    hasher.finalize()
}

fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unified_reader::UnifiedReader;
    use opendal::{Operator, services};
    use std::collections::BTreeMap;
    use tempfile::tempdir;

    #[test]
    fn test_unified_writer_produces_bytes() {
        let ids = vec![1, 2, 3];
        let vectors = vec![
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.6],
            vec![0.7, 0.8, 0.9],
        ];

        let bytes = UnifiedRemoteWriter::write_vector_only_to_bytes(&ids, &vectors, 3).unwrap();
        assert!(bytes.len() > UNIFIED_HEADER_SIZE + UNIFIED_FOOTER_SIZE);
    }

    #[test]
    fn test_unified_writer_with_payload_schema() {
        let ids = vec![1, 2];
        let flat = vec![0.1, 0.2, 0.3, 0.4];
        let schema = UnifiedPayloadSchema::new(vec![crate::unified_format::UnifiedFieldSchema {
            field_id: 1,
            name: "tag".to_string(),
            logical_type: crate::unified_format::UnifiedLogicalType::Keyword,
            nullable: false,
            indexed: true,
        }]);

        let bytes = UnifiedRemoteWriter::write_vector_with_schema_flat_to_bytes(
            &ids,
            &flat,
            2,
            Some(&schema),
        )
        .unwrap();
        assert!(bytes.len() > UNIFIED_HEADER_SIZE + UNIFIED_FOOTER_SIZE);
    }

    #[test]
    fn test_unified_local_writer_writes_file() {
        let ids = vec![1, 2];
        let vectors = vec![vec![0.1, 0.2], vec![0.3, 0.4]];
        let dir = tempdir().unwrap();
        let path = dir.path().join("bucket_1_local.driftu");

        let stats =
            UnifiedLocalWriter::write_vector_only_to_path(&path, &ids, &vectors, 2).unwrap();

        assert_eq!(stats.row_count, 2);
        assert!(path.exists());
    }

    #[tokio::test]
    async fn test_unified_local_append_chunk() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("bucket_1_local.driftu");

        let ids_a = vec![1, 2];
        let flat_a = vec![0.1, 0.2, 0.3, 0.4];
        UnifiedLocalWriter::append_vector_chunk_to_path(&path, &ids_a, &flat_a, 2).unwrap();

        let ids_b = vec![3];
        let flat_b = vec![0.5, 0.6];
        UnifiedLocalWriter::append_vector_chunk_to_path(&path, &ids_b, &flat_b, 2).unwrap();

        let root = dir.path().to_str().unwrap().to_string();
        let op = Operator::new(services::Fs::default().root(&root))
            .unwrap()
            .finish();
        let mut reader = UnifiedReader::open(op, "bucket_1_local.driftu")
            .await
            .unwrap();
        let (ids, vecs_flat) = reader.read_all_vectors_flat().await.unwrap();
        assert_eq!(ids, vec![1, 2, 3]);
        assert_eq!(vecs_flat.len(), 6);
    }

    #[test]
    fn test_unified_local_append_rejects_schema_flag_mismatch() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("bucket_flag_mismatch.driftu");

        let ids = vec![1, 2];
        let flat = vec![0.1, 0.2, 0.3, 0.4];
        UnifiedLocalWriter::write_vector_only_flat_to_path(&path, &ids, &flat, 2).unwrap();

        let mut bytes = std::fs::read(&path).unwrap();
        let mut header = UnifiedHeader::decode(&bytes[..UNIFIED_HEADER_SIZE]).unwrap();
        header.flags |= UNIFIED_FLAG_HAS_PAYLOAD_SCHEMA;
        bytes[..UNIFIED_HEADER_SIZE].copy_from_slice(&header.encode().unwrap());

        let footer_start = header.footer_offset as usize;
        let footer_end = footer_start + UNIFIED_FOOTER_SIZE;
        let mut footer = UnifiedFooter::decode(&bytes[footer_start..footer_end]).unwrap();
        footer.flags = header.flags;
        bytes[footer_start..footer_end].copy_from_slice(&footer.encode().unwrap());

        std::fs::write(&path, bytes).unwrap();

        let err = UnifiedLocalWriter::append_vector_chunk_to_path(&path, &[3], &[0.5, 0.6], 2)
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("append payload schema flag mismatch")
        );
    }

    #[test]
    fn test_unified_local_append_rejects_schema_mismatch() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("bucket_schema_mismatch.driftu");

        let schema_a = UnifiedPayloadSchema::new(vec![crate::unified_format::UnifiedFieldSchema {
            field_id: 1,
            name: "tenant".to_string(),
            logical_type: crate::unified_format::UnifiedLogicalType::Keyword,
            nullable: false,
            indexed: true,
        }]);
        let schema_b = UnifiedPayloadSchema::new(vec![crate::unified_format::UnifiedFieldSchema {
            field_id: 2,
            name: "region".to_string(),
            logical_type: crate::unified_format::UnifiedLogicalType::Keyword,
            nullable: false,
            indexed: true,
        }]);

        UnifiedLocalWriter::write_vector_with_schema_flat_to_path(
            &path,
            &[1, 2],
            &[0.1, 0.2, 0.3, 0.4],
            2,
            Some(&schema_a),
        )
        .unwrap();

        let err = UnifiedLocalWriter::append_vector_chunk_with_schema_to_path(
            &path,
            &[3],
            &[0.5, 0.6],
            2,
            Some(&schema_b),
        )
        .unwrap_err();
        assert!(err.to_string().contains("append payload schema mismatch"));
    }

    #[test]
    fn test_unified_local_append_rejects_when_payload_columns_exist() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("bucket_payload_append_unsupported.driftu");
        let schema = UnifiedPayloadSchema::new(vec![crate::unified_format::UnifiedFieldSchema {
            field_id: 1,
            name: "tenant".to_string(),
            logical_type: crate::unified_format::UnifiedLogicalType::Keyword,
            nullable: false,
            indexed: true,
        }]);
        let rows: Vec<UnifiedPayloadRow> = vec![
            BTreeMap::from([(
                1,
                crate::unified_format::UnifiedPayloadValue::Keyword("a".to_string()),
            )]),
            BTreeMap::from([(
                1,
                crate::unified_format::UnifiedPayloadValue::Keyword("b".to_string()),
            )]),
        ];

        UnifiedLocalWriter::write_vector_with_payload_flat_to_path(
            &path,
            &[1, 2],
            &[0.1, 0.2, 0.3, 0.4],
            2,
            Some(&schema),
            Some(&rows),
        )
        .unwrap();

        let err = UnifiedLocalWriter::append_vector_chunk_to_path(&path, &[3], &[0.5, 0.6], 2)
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("append with payload columns/indexes is not yet supported")
        );
    }
}
