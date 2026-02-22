use crate::disk_manager::DiskManager;
use crate::unified_format::{
    UNIFIED_BLOCK_DESC_SIZE, UNIFIED_FLAG_HAS_EXACT_INDEX, UNIFIED_FLAG_HAS_PAYLOAD_COLUMNS,
    UNIFIED_FLAG_HAS_PAYLOAD_SCHEMA, UNIFIED_FOOTER_SIZE, UNIFIED_HEADER_SIZE, UnifiedBlockDesc,
    UnifiedBlockType, UnifiedCodec, UnifiedExactIndex, UnifiedFooter, UnifiedHeader,
    UnifiedLogicalType, UnifiedPayloadColumnChunk, UnifiedPayloadRow, UnifiedPayloadSchema,
    UnifiedPayloadValue, decode_block_directory, encode_exact_key,
};
use byteorder::{LittleEndian, ReadBytesExt};
use crc32fast::Hasher;
use drift_core::quantizer::Quantizer;
use opendal::Operator;
use std::collections::{BTreeMap, HashSet};
use std::io::{self, Cursor};

#[derive(Clone)]
struct ChunkRefs {
    row_start: u64,
    row_count: u32,
    quantizer: UnifiedBlockDesc,
    ids: UnifiedBlockDesc,
    codes: UnifiedBlockDesc,
}

pub struct UnifiedReader {
    manager: DiskManager,
    pub header: UnifiedHeader,
    pub footer: UnifiedFooter,
    pub blocks: Vec<UnifiedBlockDesc>,
}

impl UnifiedReader {
    pub async fn open(op: Operator, path: &str) -> io::Result<Self> {
        let manager = DiskManager::new(op, path.to_string());
        let file_len = manager.len().await?;
        if file_len < (UNIFIED_HEADER_SIZE + UNIFIED_FOOTER_SIZE) as u64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unified file too small",
            ));
        }

        let header_bytes = manager.read_at(0, UNIFIED_HEADER_SIZE).await?;
        let header = UnifiedHeader::decode(&header_bytes)?;

        let footer_offset = header.footer_offset;
        let footer_end = footer_offset
            .checked_add(header.footer_length as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "footer range overflow"))?;
        if footer_end > file_len || header.footer_length as usize != UNIFIED_FOOTER_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid unified footer location",
            ));
        }

        let footer_bytes = manager.read_at(footer_offset, UNIFIED_FOOTER_SIZE).await?;
        let footer = UnifiedFooter::decode(&footer_bytes)?;

        if footer.row_count != header.row_count
            || footer.block_count != header.block_count
            || footer.block_dir_offset != header.block_dir_offset
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "header/footer metadata mismatch",
            ));
        }
        if footer.flags != header.flags {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "header/footer flags mismatch",
            ));
        }

        let block_dir_len = 4usize
            .checked_add((header.block_count as usize).saturating_mul(UNIFIED_BLOCK_DESC_SIZE))
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "block dir size overflow"))?;

        let block_dir_end = header
            .block_dir_offset
            .checked_add(block_dir_len as u64)
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "block dir range overflow")
            })?;

        if header.block_dir_offset < UNIFIED_HEADER_SIZE as u64 || block_dir_end > footer_offset {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid unified block directory range",
            ));
        }

        let block_dir_bytes = manager
            .read_at(header.block_dir_offset, block_dir_len)
            .await?;
        if checksum32(&block_dir_bytes) != footer.directory_crc32 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "block directory checksum mismatch",
            ));
        }
        let blocks = decode_block_directory(&block_dir_bytes)?;
        let has_schema_block = blocks
            .iter()
            .any(|b| b.block_type == UnifiedBlockType::PayloadSchema);
        let has_payload_columns = blocks
            .iter()
            .any(|b| b.block_type == UnifiedBlockType::PayloadColumn);
        let has_exact_index = blocks
            .iter()
            .any(|b| b.block_type == UnifiedBlockType::PayloadExactIndex);
        let flag_has_schema = (header.flags & UNIFIED_FLAG_HAS_PAYLOAD_SCHEMA) != 0;
        if flag_has_schema != has_schema_block {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "payload schema flag mismatch",
            ));
        }
        let flag_has_payload_columns = (header.flags & UNIFIED_FLAG_HAS_PAYLOAD_COLUMNS) != 0;
        if flag_has_payload_columns != has_payload_columns {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "payload columns flag mismatch",
            ));
        }
        let flag_has_exact_index = (header.flags & UNIFIED_FLAG_HAS_EXACT_INDEX) != 0;
        if flag_has_exact_index != has_exact_index {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "exact index flag mismatch",
            ));
        }

        let chunks = build_chunk_refs(&blocks, &header, footer_offset)?;
        let rows_sum: u64 = chunks.iter().map(|c| c.row_count as u64).sum();
        if rows_sum != header.row_count {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "chunk row sum mismatch: expected={}, got={}",
                    header.row_count, rows_sum
                ),
            ));
        }

        Ok(Self {
            manager,
            header,
            footer,
            blocks,
        })
    }

    pub async fn read_ids(&self) -> io::Result<Vec<u64>> {
        let chunks = self.chunk_refs()?;
        let mut out = Vec::with_capacity(self.header.row_count as usize);
        for chunk in chunks {
            let bytes = self.read_block_bytes(&chunk.ids).await?;
            if bytes.len() != chunk.row_count as usize * 8 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "ids block length mismatch",
                ));
            }
            let mut cursor = Cursor::new(bytes);
            for _ in 0..chunk.row_count {
                out.push(cursor.read_u64::<LittleEndian>()?);
            }
        }
        Ok(out)
    }

    pub async fn read_vector_codes(&self) -> io::Result<Vec<u8>> {
        let chunks = self.chunk_refs()?;
        let dim = self.header.dim as usize;
        let mut out = Vec::with_capacity(self.header.row_count as usize * dim);
        for chunk in chunks {
            let bytes = self.read_block_bytes(&chunk.codes).await?;
            let expected = chunk.row_count as usize * dim;
            if bytes.len() != expected {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "vector code block length mismatch",
                ));
            }
            out.extend_from_slice(&bytes);
        }
        Ok(out)
    }

    pub async fn read_all_vectors_flat(&mut self) -> io::Result<(Vec<u64>, Vec<f32>)> {
        let dim = self.header.dim as usize;
        let chunks = self.chunk_refs()?;
        let mut all_ids = Vec::with_capacity(self.header.row_count as usize);
        let mut all_flat = Vec::with_capacity(self.header.row_count as usize * dim);

        for chunk in chunks {
            let q_bytes = self.read_block_bytes(&chunk.quantizer).await?;
            let (q, _): (Quantizer, usize) =
                bincode::decode_from_slice(&q_bytes, bincode::config::standard())
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

            let ids_bytes = self.read_block_bytes(&chunk.ids).await?;
            let codes_bytes = self.read_block_bytes(&chunk.codes).await?;

            if ids_bytes.len() != chunk.row_count as usize * 8 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "ids block length mismatch",
                ));
            }
            if codes_bytes.len() != chunk.row_count as usize * dim {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "codes block length mismatch",
                ));
            }

            let mut id_cursor = Cursor::new(ids_bytes);
            for _ in 0..chunk.row_count {
                all_ids.push(id_cursor.read_u64::<LittleEndian>()?);
            }

            for code in codes_bytes.chunks_exact(dim) {
                let vec = q.reconstruct(code);
                all_flat.extend_from_slice(&vec);
            }
        }

        Ok((all_ids, all_flat))
    }

    pub async fn read_all_vectors(&mut self) -> io::Result<(Vec<u64>, Vec<Vec<f32>>)> {
        let (ids, flat) = self.read_all_vectors_flat().await?;
        let dim = self.header.dim as usize;
        let mut out = Vec::with_capacity(ids.len());
        for row in flat.chunks_exact(dim) {
            out.push(row.to_vec());
        }
        Ok((ids, out))
    }

    pub async fn read_payload_schema(&self) -> io::Result<Option<UnifiedPayloadSchema>> {
        let mut schema_blocks = self
            .blocks
            .iter()
            .filter(|b| b.block_type == UnifiedBlockType::PayloadSchema);
        let Some(first) = schema_blocks.next() else {
            return Ok(None);
        };

        let first_bytes = self.read_block_bytes(first).await?;
        let (first_schema, _): (UnifiedPayloadSchema, usize) =
            bincode::decode_from_slice(&first_bytes, bincode::config::standard())
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        for block in schema_blocks {
            let bytes = self.read_block_bytes(block).await?;
            let (schema, _): (UnifiedPayloadSchema, usize) =
                bincode::decode_from_slice(&bytes, bincode::config::standard())
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            if schema != first_schema {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "multiple payload schema blocks disagree",
                ));
            }
        }

        Ok(Some(first_schema))
    }

    pub async fn read_payload_columns(
        &self,
    ) -> io::Result<BTreeMap<u32, Vec<UnifiedPayloadValue>>> {
        if !self
            .blocks
            .iter()
            .any(|b| b.block_type == UnifiedBlockType::PayloadColumn)
        {
            return Ok(BTreeMap::new());
        }
        let Some(schema) = self.read_payload_schema().await? else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "payload columns present without payload schema",
            ));
        };
        let row_count = self.header.row_count as usize;
        if row_count == 0 {
            return Ok(BTreeMap::new());
        }

        let mut by_field: BTreeMap<u32, Vec<Option<UnifiedPayloadValue>>> = BTreeMap::new();
        let mut field_seen_chunks: HashSet<u32> = HashSet::new();

        for block in self
            .blocks
            .iter()
            .filter(|b| b.block_type == UnifiedBlockType::PayloadColumn)
        {
            let bytes = self.read_block_bytes(block).await?;
            let (chunk, _): (UnifiedPayloadColumnChunk, usize) =
                bincode::decode_from_slice(&bytes, bincode::config::standard())
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

            if chunk.codec != block.codec {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "payload column codec mismatch",
                ));
            }
            if chunk.row_count != block.row_count || chunk.row_start != block.row_start {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "payload column row range mismatch",
                ));
            }

            let field = schema
                .fields
                .iter()
                .find(|f| f.field_id == chunk.field_id)
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "payload chunk references unknown field_id={}",
                            chunk.field_id
                        ),
                    )
                })?;
            if field.logical_type != chunk.logical_type {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "payload column logical type mismatch",
                ));
            }

            field_seen_chunks.insert(field.field_id);
            let decoded_values = decode_payload_column_chunk_values(
                &chunk.logical_type,
                chunk.codec,
                chunk.row_count as usize,
                chunk.validity.as_deref(),
                &chunk.data,
            )?;

            let start = chunk.row_start as usize;
            let end = start.checked_add(chunk.row_count as usize).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "payload chunk row overflow")
            })?;
            if end > row_count {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "payload chunk out of row bounds",
                ));
            }

            let target = by_field
                .entry(chunk.field_id)
                .or_insert_with(|| vec![None; row_count]);
            for (i, value) in decoded_values.into_iter().enumerate() {
                let slot = &mut target[start + i];
                if slot.is_some() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "overlapping payload column chunks",
                    ));
                }
                *slot = Some(value);
            }
        }

        let mut out = BTreeMap::new();
        for field in &schema.fields {
            let mut materialized = Vec::with_capacity(row_count);
            if let Some(values) = by_field.remove(&field.field_id) {
                for maybe in values {
                    let value = maybe.unwrap_or(UnifiedPayloadValue::Null);
                    if matches!(value, UnifiedPayloadValue::Null) && !field.nullable {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("non-nullable field '{}' has null rows", field.name),
                        ));
                    }
                    materialized.push(value);
                }
            } else {
                if !field.nullable && field_seen_chunks.contains(&field.field_id) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "payload field chunk tracking mismatch",
                    ));
                }
                if !field.nullable {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("non-nullable field '{}' has no payload column", field.name),
                    ));
                }
                materialized.resize(row_count, UnifiedPayloadValue::Null);
            }
            out.insert(field.field_id, materialized);
        }

        Ok(out)
    }

    pub async fn read_payload_rows(&self) -> io::Result<Vec<UnifiedPayloadRow>> {
        let columns = self.read_payload_columns().await?;
        let row_count = self.header.row_count as usize;
        if row_count == 0 {
            return Ok(Vec::new());
        }
        if columns.is_empty() {
            return Ok(vec![BTreeMap::new(); row_count]);
        }

        let mut rows = vec![BTreeMap::new(); row_count];
        for (&field_id, values) in &columns {
            if values.len() != row_count {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "payload column length mismatch",
                ));
            }
            for (row_idx, value) in values.iter().enumerate() {
                if !matches!(value, UnifiedPayloadValue::Null) {
                    rows[row_idx].insert(field_id, value.clone());
                }
            }
        }
        Ok(rows)
    }

    pub async fn read_exact_index(&self, field_id: u32) -> io::Result<Option<UnifiedExactIndex>> {
        let mut found: Option<UnifiedExactIndex> = None;
        for block in self
            .blocks
            .iter()
            .filter(|b| b.block_type == UnifiedBlockType::PayloadExactIndex)
        {
            let bytes = self.read_block_bytes(block).await?;
            let (index, _): (UnifiedExactIndex, usize) =
                bincode::decode_from_slice(&bytes, bincode::config::standard())
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            if index.field_id != field_id {
                continue;
            }
            if block.codec != UnifiedCodec::DictPostings {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "exact index block has invalid codec",
                ));
            }
            if let Some(prev) = &found {
                if prev != &index {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "multiple exact indexes for field disagree",
                    ));
                }
            } else {
                found = Some(index);
            }
        }
        Ok(found)
    }

    pub async fn filter_ids_exact(
        &self,
        field_id: u32,
        value: &UnifiedPayloadValue,
    ) -> io::Result<Vec<u64>> {
        let schema = self.read_payload_schema().await?;
        let logical_type = schema
            .as_ref()
            .and_then(|s| s.fields.iter().find(|f| f.field_id == field_id))
            .map(|f| f.logical_type.clone())
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("unknown field_id={} for exact filter", field_id),
                )
            })?;

        let Some(key) = encode_exact_key(&logical_type, value)? else {
            return Ok(Vec::new());
        };
        let Some(index) = self.read_exact_index(field_id).await? else {
            return Ok(Vec::new());
        };

        match index
            .dictionary
            .binary_search_by(|candidate| candidate.as_slice().cmp(key.as_slice()))
        {
            Ok(pos) => Ok(index.postings.get(pos).cloned().unwrap_or_default()),
            Err(_) => Ok(Vec::new()),
        }
    }

    fn chunk_refs(&self) -> io::Result<Vec<ChunkRefs>> {
        build_chunk_refs(&self.blocks, &self.header, self.header.footer_offset)
    }

    async fn read_block_bytes(&self, block: &UnifiedBlockDesc) -> io::Result<Vec<u8>> {
        let bytes = self
            .manager
            .read_at(block.offset, block.compressed_len as usize)
            .await?;
        if checksum32(&bytes) != block.crc32 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("checksum mismatch for block {:?}", block.block_type),
            ));
        }
        Ok(bytes)
    }
}

fn build_chunk_refs(
    blocks: &[UnifiedBlockDesc],
    header: &UnifiedHeader,
    footer_offset: u64,
) -> io::Result<Vec<ChunkRefs>> {
    if blocks.len() != header.block_count as usize {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "block count mismatch",
        ));
    }
    if header.footer_offset != footer_offset {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "footer offset mismatch",
        ));
    }

    let mut groups: BTreeMap<
        (u64, u32),
        (
            Option<UnifiedBlockDesc>,
            Option<UnifiedBlockDesc>,
            Option<UnifiedBlockDesc>,
        ),
    > = BTreeMap::new();

    for block in blocks {
        let block_end = block
            .offset
            .checked_add(block.compressed_len)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "block range overflow"))?;
        if block.offset < UNIFIED_HEADER_SIZE as u64 || block_end > header.block_dir_offset {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("block outside payload region: {:?}", block.block_type),
            ));
        }
        if block.row_count == 0 || block.row_count as u64 > header.row_count {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid block row_count",
            ));
        }

        let entry = groups
            .entry((block.row_start, block.row_count))
            .or_insert((None, None, None));
        match block.block_type {
            UnifiedBlockType::Quantizer => {
                if entry.0.is_some() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "duplicate quantizer block in chunk",
                    ));
                }
                entry.0 = Some(block.clone());
            }
            UnifiedBlockType::Ids => {
                if entry.1.is_some() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "duplicate ids block in chunk",
                    ));
                }
                entry.1 = Some(block.clone());
            }
            UnifiedBlockType::VectorCodes => {
                if entry.2.is_some() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "duplicate vector code block in chunk",
                    ));
                }
                entry.2 = Some(block.clone());
            }
            UnifiedBlockType::PayloadSchema
            | UnifiedBlockType::PayloadColumn
            | UnifiedBlockType::PayloadExactIndex => {}
        }
    }

    if groups.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "no vector chunks present",
        ));
    }

    let mut chunks = Vec::with_capacity(groups.len());
    for ((row_start, row_count), (q, ids, codes)) in groups {
        let quantizer = q.ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "chunk missing quantizer block")
        })?;
        let ids = ids
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "chunk missing ids block"))?;
        let codes = codes.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "chunk missing vector code block",
            )
        })?;
        chunks.push(ChunkRefs {
            row_start,
            row_count,
            quantizer,
            ids,
            codes,
        });
    }

    chunks.sort_by_key(|c| c.row_start);
    let mut expected_start = 0u64;
    for chunk in &chunks {
        if chunk.row_start != expected_start {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "chunk row ranges are not contiguous",
            ));
        }
        expected_start = expected_start.saturating_add(chunk.row_count as u64);
    }

    Ok(chunks)
}

fn checksum32(bytes: &[u8]) -> u32 {
    let mut hasher = Hasher::new();
    hasher.update(bytes);
    hasher.finalize()
}

fn decode_payload_column_chunk_values(
    logical_type: &UnifiedLogicalType,
    codec: UnifiedCodec,
    row_count: usize,
    validity: Option<&[u8]>,
    data: &[u8],
) -> io::Result<Vec<UnifiedPayloadValue>> {
    let present_count = if let Some(bits) = validity {
        let min_len = row_count.div_ceil(8);
        if bits.len() < min_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "payload validity bitmap truncated",
            ));
        }
        (0..row_count)
            .filter(|i| ((bits[*i / 8] >> (*i % 8)) & 1) == 1)
            .count()
    } else {
        row_count
    };

    let non_null = decode_non_null_payload_values(logical_type, codec, data, present_count)?;
    if validity.is_none() {
        return Ok(non_null);
    }

    let mut out = Vec::with_capacity(row_count);
    let mut iter = non_null.into_iter();
    let bits = validity.expect("checked above");
    for row in 0..row_count {
        let present = ((bits[row / 8] >> (row % 8)) & 1) == 1;
        if present {
            let value = iter.next().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "payload non-null decode underflow",
                )
            })?;
            out.push(value);
        } else {
            out.push(UnifiedPayloadValue::Null);
        }
    }
    if iter.next().is_some() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "payload non-null decode overflow",
        ));
    }
    Ok(out)
}

fn decode_non_null_payload_values(
    logical_type: &UnifiedLogicalType,
    codec: UnifiedCodec,
    data: &[u8],
    value_count: usize,
) -> io::Result<Vec<UnifiedPayloadValue>> {
    let mut out = Vec::with_capacity(value_count);
    match logical_type {
        UnifiedLogicalType::Bool => {
            if codec != UnifiedCodec::Bitset {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "payload bool column codec mismatch",
                ));
            }
            let min_len = value_count.div_ceil(8);
            if data.len() < min_len {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "payload bool data truncated",
                ));
            }
            for i in 0..value_count {
                let v = ((data[i / 8] >> (i % 8)) & 1) == 1;
                out.push(UnifiedPayloadValue::Bool(v));
            }
        }
        UnifiedLogicalType::Int64 => {
            if codec != UnifiedCodec::PlainLe {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "payload int64 column codec mismatch",
                ));
            }
            let expected = value_count * 8;
            if data.len() != expected {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "payload int64 data length mismatch",
                ));
            }
            let mut cursor = Cursor::new(data);
            for _ in 0..value_count {
                out.push(UnifiedPayloadValue::Int64(
                    cursor.read_i64::<LittleEndian>()?,
                ));
            }
        }
        UnifiedLogicalType::Float32 => {
            if codec != UnifiedCodec::PlainLe {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "payload float32 column codec mismatch",
                ));
            }
            let expected = value_count * 4;
            if data.len() != expected {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "payload float32 data length mismatch",
                ));
            }
            let mut cursor = Cursor::new(data);
            for _ in 0..value_count {
                out.push(UnifiedPayloadValue::Float32(
                    cursor.read_f32::<LittleEndian>()?,
                ));
            }
        }
        UnifiedLogicalType::Float64 => {
            if codec != UnifiedCodec::PlainLe {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "payload float64 column codec mismatch",
                ));
            }
            let expected = value_count * 8;
            if data.len() != expected {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "payload float64 data length mismatch",
                ));
            }
            let mut cursor = Cursor::new(data);
            for _ in 0..value_count {
                out.push(UnifiedPayloadValue::Float64(
                    cursor.read_f64::<LittleEndian>()?,
                ));
            }
        }
        UnifiedLogicalType::TimestampMicros => {
            if codec != UnifiedCodec::PlainLe {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "payload timestamp column codec mismatch",
                ));
            }
            let expected = value_count * 8;
            if data.len() != expected {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "payload timestamp data length mismatch",
                ));
            }
            let mut cursor = Cursor::new(data);
            for _ in 0..value_count {
                out.push(UnifiedPayloadValue::TimestampMicros(
                    cursor.read_i64::<LittleEndian>()?,
                ));
            }
        }
        UnifiedLogicalType::Keyword | UnifiedLogicalType::Text => {
            if codec != UnifiedCodec::VarLen {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "payload string column codec mismatch",
                ));
            }
            let mut cursor = Cursor::new(data);
            for _ in 0..value_count {
                let len = cursor.read_u32::<LittleEndian>()? as usize;
                let start = cursor.position() as usize;
                let end = start.checked_add(len).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "payload string length overflow")
                })?;
                if end > data.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "payload string data truncated",
                    ));
                }
                let s = String::from_utf8(data[start..end].to_vec())
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                if matches!(logical_type, UnifiedLogicalType::Keyword) {
                    out.push(UnifiedPayloadValue::Keyword(s));
                } else {
                    out.push(UnifiedPayloadValue::Text(s));
                }
                cursor.set_position(end as u64);
            }
            if cursor.position() as usize != data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "payload string column has trailing bytes",
                ));
            }
        }
        UnifiedLogicalType::Bytes => {
            if codec != UnifiedCodec::VarLen {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "payload bytes column codec mismatch",
                ));
            }
            let mut cursor = Cursor::new(data);
            for _ in 0..value_count {
                let len = cursor.read_u32::<LittleEndian>()? as usize;
                let start = cursor.position() as usize;
                let end = start.checked_add(len).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "payload bytes length overflow")
                })?;
                if end > data.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "payload bytes data truncated",
                    ));
                }
                out.push(UnifiedPayloadValue::Bytes(data[start..end].to_vec()));
                cursor.set_position(end as u64);
            }
            if cursor.position() as usize != data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "payload bytes column has trailing bytes",
                ));
            }
        }
        UnifiedLogicalType::LobRef => {
            if codec != UnifiedCodec::VarLen {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "payload lob_ref column codec mismatch",
                ));
            }
            let mut cursor = Cursor::new(data);
            for _ in 0..value_count {
                let len = cursor.read_u32::<LittleEndian>()? as usize;
                let start = cursor.position() as usize;
                let end = start.checked_add(len).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "payload lob_ref length overflow",
                    )
                })?;
                if end > data.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "payload lob_ref data truncated",
                    ));
                }
                let (lob, _): (crate::unified_format::UnifiedLobRef, usize) =
                    bincode::decode_from_slice(&data[start..end], bincode::config::standard())
                        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                out.push(UnifiedPayloadValue::LobRef(lob));
                cursor.set_position(end as u64);
            }
            if cursor.position() as usize != data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "payload lob_ref column has trailing bytes",
                ));
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unified_writer::UnifiedRemoteWriter;
    use opendal::{Operator, services};
    use std::collections::BTreeMap;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_unified_writer_reader_roundtrip() {
        let ids = vec![11, 22, 33, 44];
        let vectors = vec![
            vec![0.1, 0.2, 0.3, 0.4],
            vec![0.5, 0.6, 0.7, 0.8],
            vec![0.9, 1.0, 1.1, 1.2],
            vec![1.3, 1.4, 1.5, 1.6],
        ];

        let bytes = UnifiedRemoteWriter::write_vector_only_to_bytes(&ids, &vectors, 4).unwrap();

        let dir = tempdir().unwrap();
        let root = dir.path().to_str().unwrap().to_string();
        let op = Operator::new(services::Fs::default().root(&root))
            .unwrap()
            .finish();

        let name = "bucket_1_test.driftu";
        op.write(name, bytes).await.unwrap();

        let mut reader = UnifiedReader::open(op.clone(), name).await.unwrap();
        let (decoded_ids, decoded_flat) = reader.read_all_vectors_flat().await.unwrap();

        assert_eq!(decoded_ids, ids);
        assert_eq!(decoded_flat.len(), vectors.len() * 4);
    }

    #[tokio::test]
    async fn test_unified_reader_reads_payload_schema() {
        let ids = vec![7, 8];
        let flat = vec![0.1, 0.2, 0.3, 0.4];
        let schema = crate::unified_format::UnifiedPayloadSchema::new(vec![
            crate::unified_format::UnifiedFieldSchema {
                field_id: 1,
                name: "tenant".to_string(),
                logical_type: crate::unified_format::UnifiedLogicalType::Keyword,
                nullable: false,
                indexed: true,
            },
        ]);

        let bytes = UnifiedRemoteWriter::write_vector_with_schema_flat_to_bytes(
            &ids,
            &flat,
            2,
            Some(&schema),
        )
        .unwrap();

        let dir = tempdir().unwrap();
        let root = dir.path().to_str().unwrap().to_string();
        let op = Operator::new(services::Fs::default().root(&root))
            .unwrap()
            .finish();

        let name = "bucket_1_schema.driftu";
        op.write(name, bytes).await.unwrap();

        let reader = UnifiedReader::open(op.clone(), name).await.unwrap();
        let decoded = reader.read_payload_schema().await.unwrap();
        assert_eq!(decoded, Some(schema));
    }

    #[tokio::test]
    async fn test_unified_reader_rejects_schema_flag_without_schema_block() {
        let ids = vec![1, 2];
        let flat = vec![0.1, 0.2, 0.3, 0.4];
        let mut bytes =
            UnifiedRemoteWriter::write_vector_only_flat_to_bytes(&ids, &flat, 2).unwrap();

        let mut header = UnifiedHeader::decode(&bytes[..UNIFIED_HEADER_SIZE]).unwrap();
        header.flags |= UNIFIED_FLAG_HAS_PAYLOAD_SCHEMA;
        bytes[..UNIFIED_HEADER_SIZE].copy_from_slice(&header.encode().unwrap());

        let footer_start = header.footer_offset as usize;
        let footer_end = footer_start + UNIFIED_FOOTER_SIZE;
        let mut footer = UnifiedFooter::decode(&bytes[footer_start..footer_end]).unwrap();
        footer.flags = header.flags;
        bytes[footer_start..footer_end].copy_from_slice(&footer.encode().unwrap());

        let dir = tempdir().unwrap();
        let root = dir.path().to_str().unwrap().to_string();
        let op = Operator::new(services::Fs::default().root(&root))
            .unwrap()
            .finish();

        let name = "bucket_1_schema_flag_mismatch.driftu";
        op.write(name, bytes).await.unwrap();

        match UnifiedReader::open(op.clone(), name).await {
            Ok(_) => panic!("expected schema flag mismatch to fail open"),
            Err(err) => assert!(err.to_string().contains("payload schema flag mismatch")),
        }
    }

    #[tokio::test]
    async fn test_unified_reader_rejects_header_footer_flag_mismatch() {
        let ids = vec![1, 2];
        let flat = vec![0.1, 0.2, 0.3, 0.4];
        let mut bytes =
            UnifiedRemoteWriter::write_vector_only_flat_to_bytes(&ids, &flat, 2).unwrap();

        let header = UnifiedHeader::decode(&bytes[..UNIFIED_HEADER_SIZE]).unwrap();
        let footer_start = header.footer_offset as usize;
        let footer_end = footer_start + UNIFIED_FOOTER_SIZE;
        let mut footer = UnifiedFooter::decode(&bytes[footer_start..footer_end]).unwrap();
        footer.flags ^= UNIFIED_FLAG_HAS_PAYLOAD_SCHEMA;
        bytes[footer_start..footer_end].copy_from_slice(&footer.encode().unwrap());

        let dir = tempdir().unwrap();
        let root = dir.path().to_str().unwrap().to_string();
        let op = Operator::new(services::Fs::default().root(&root))
            .unwrap()
            .finish();

        let name = "bucket_1_header_footer_flag_mismatch.driftu";
        op.write(name, bytes).await.unwrap();

        match UnifiedReader::open(op.clone(), name).await {
            Ok(_) => panic!("expected header/footer flag mismatch to fail open"),
            Err(err) => assert!(err.to_string().contains("header/footer flags mismatch")),
        }
    }

    #[tokio::test]
    async fn test_unified_reader_payload_rows_and_exact_index_roundtrip() {
        let ids = vec![101, 102, 103];
        let flat = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let schema = UnifiedPayloadSchema::new(vec![
            crate::unified_format::UnifiedFieldSchema {
                field_id: 1,
                name: "tenant".to_string(),
                logical_type: crate::unified_format::UnifiedLogicalType::Keyword,
                nullable: false,
                indexed: true,
            },
            crate::unified_format::UnifiedFieldSchema {
                field_id: 2,
                name: "score".to_string(),
                logical_type: crate::unified_format::UnifiedLogicalType::Int64,
                nullable: false,
                indexed: false,
            },
            crate::unified_format::UnifiedFieldSchema {
                field_id: 3,
                name: "active".to_string(),
                logical_type: crate::unified_format::UnifiedLogicalType::Bool,
                nullable: false,
                indexed: true,
            },
        ]);

        let rows: Vec<crate::unified_format::UnifiedPayloadRow> = vec![
            BTreeMap::from([
                (
                    1,
                    crate::unified_format::UnifiedPayloadValue::Keyword("acme".to_string()),
                ),
                (2, crate::unified_format::UnifiedPayloadValue::Int64(10)),
                (3, crate::unified_format::UnifiedPayloadValue::Bool(true)),
            ]),
            BTreeMap::from([
                (
                    1,
                    crate::unified_format::UnifiedPayloadValue::Keyword("globex".to_string()),
                ),
                (2, crate::unified_format::UnifiedPayloadValue::Int64(20)),
                (3, crate::unified_format::UnifiedPayloadValue::Bool(false)),
            ]),
            BTreeMap::from([
                (
                    1,
                    crate::unified_format::UnifiedPayloadValue::Keyword("acme".to_string()),
                ),
                (2, crate::unified_format::UnifiedPayloadValue::Int64(30)),
                (3, crate::unified_format::UnifiedPayloadValue::Bool(true)),
            ]),
        ];

        let bytes = UnifiedRemoteWriter::write_vector_with_payload_flat_to_bytes(
            &ids,
            &flat,
            2,
            Some(&schema),
            Some(&rows),
        )
        .unwrap();

        let dir = tempdir().unwrap();
        let root = dir.path().to_str().unwrap().to_string();
        let op = Operator::new(services::Fs::default().root(&root))
            .unwrap()
            .finish();

        let name = "bucket_payload_index.driftu";
        op.write(name, bytes).await.unwrap();

        let reader = UnifiedReader::open(op.clone(), name).await.unwrap();
        let decoded_rows = reader.read_payload_rows().await.unwrap();
        assert_eq!(decoded_rows, rows);

        let acme = reader
            .filter_ids_exact(
                1,
                &crate::unified_format::UnifiedPayloadValue::Keyword("acme".to_string()),
            )
            .await
            .unwrap();
        assert_eq!(acme, vec![101, 103]);

        let active_true = reader
            .filter_ids_exact(3, &crate::unified_format::UnifiedPayloadValue::Bool(true))
            .await
            .unwrap();
        assert_eq!(active_true, vec![101, 103]);
    }
}
