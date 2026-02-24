use bincode::{Decode, Encode};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::collections::BTreeMap;
use std::io::{self, Cursor, Read, Write};

pub const UNIFIED_MAGIC: [u8; 8] = *b"DRIFTU01";
pub const UNIFIED_FOOTER_MAGIC: [u8; 8] = *b"DRIFTUF1";
pub const UNIFIED_VERSION: u16 = 1;
pub const UNIFIED_HEADER_SIZE: usize = 128;
pub const UNIFIED_FOOTER_SIZE: usize = 64;
pub const UNIFIED_BLOCK_DESC_SIZE: usize = 56;

const HEADER_RESERVED_BYTES: usize = 32;
const FOOTER_RESERVED_BYTES: usize = 12;

pub const UNIFIED_FLAG_HAS_PAYLOAD_SCHEMA: u32 = 1 << 0;
pub const UNIFIED_FLAG_HAS_PAYLOAD_COLUMNS: u32 = 1 << 1;
pub const UNIFIED_FLAG_HAS_EXACT_INDEX: u32 = 1 << 2;
pub const UNIFIED_FLAG_HAS_PAYLOAD_STATS: u32 = 1 << 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum UnifiedBlockType {
    Quantizer = 1,
    Ids = 2,
    VectorCodes = 3,
    PayloadSchema = 10,
    PayloadColumn = 11,
    PayloadExactIndex = 12,
    PayloadStats = 13,
}

impl UnifiedBlockType {
    fn from_u16(v: u16) -> io::Result<Self> {
        match v {
            1 => Ok(Self::Quantizer),
            2 => Ok(Self::Ids),
            3 => Ok(Self::VectorCodes),
            10 => Ok(Self::PayloadSchema),
            11 => Ok(Self::PayloadColumn),
            12 => Ok(Self::PayloadExactIndex),
            13 => Ok(Self::PayloadStats),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported block type: {v}"),
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Encode, Decode)]
#[repr(u16)]
pub enum UnifiedCodec {
    Bincode = 1,
    PlainLe = 2,
    Sq8 = 3,
    Bitset = 4,
    VarLen = 5,
    DictPostings = 6,
    ForBitpack = 7,
    AlpRd = 8,
    DictBitpack = 9,
    DictPostingsBitpack = 10,
}

impl UnifiedCodec {
    fn from_u16(v: u16) -> io::Result<Self> {
        match v {
            1 => Ok(Self::Bincode),
            2 => Ok(Self::PlainLe),
            3 => Ok(Self::Sq8),
            4 => Ok(Self::Bitset),
            5 => Ok(Self::VarLen),
            6 => Ok(Self::DictPostings),
            7 => Ok(Self::ForBitpack),
            8 => Ok(Self::AlpRd),
            9 => Ok(Self::DictBitpack),
            10 => Ok(Self::DictPostingsBitpack),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported codec: {v}"),
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Encode, Decode, Default)]
#[repr(u16)]
pub enum UnifiedMetric {
    #[default]
    L2 = 0,
    Cosine = 1,
}

impl UnifiedMetric {
    fn from_u16(v: u16) -> io::Result<Self> {
        match v {
            0 => Ok(Self::L2),
            1 => Ok(Self::Cosine),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported unified metric: {v}"),
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Encode, Decode)]
pub enum UnifiedLogicalType {
    Bool,
    Int64,
    Float32,
    Float64,
    TimestampMicros,
    Keyword,
    Text,
    Bytes,
    LobRef,
}

#[derive(Debug, Clone, PartialEq, Eq, Encode, Decode)]
pub struct UnifiedLobRef {
    pub blob_key: String,
    pub offset: u64,
    pub length: u64,
    pub fingerprint: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Encode, Decode)]
pub enum UnifiedPayloadValue {
    Null,
    Bool(bool),
    Int64(i64),
    Float32(f32),
    Float64(f64),
    TimestampMicros(i64),
    Keyword(String),
    Text(String),
    Bytes(Vec<u8>),
    LobRef(UnifiedLobRef),
}

pub type UnifiedPayloadRow = BTreeMap<u32, UnifiedPayloadValue>;

#[derive(Debug, Clone, PartialEq, Encode, Decode)]
pub struct UnifiedPayloadColumnChunk {
    pub field_id: u32,
    pub logical_type: UnifiedLogicalType,
    pub codec: UnifiedCodec,
    pub row_start: u64,
    pub row_count: u32,
    pub validity: Option<Vec<u8>>,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Encode, Decode)]
pub struct UnifiedPayloadFieldStats {
    pub field_id: u32,
    pub logical_type: UnifiedLogicalType,
    pub null_count: u32,
    pub min: Option<UnifiedPayloadValue>,
    pub max: Option<UnifiedPayloadValue>,
    pub cardinality_hint: u32,
}

#[derive(Debug, Clone, PartialEq, Encode, Decode)]
pub struct UnifiedPayloadStatsChunk {
    pub row_start: u64,
    pub row_count: u32,
    pub fields: Vec<UnifiedPayloadFieldStats>,
}

#[derive(Debug, Clone, PartialEq, Encode, Decode)]
pub struct UnifiedExactIndex {
    pub field_id: u32,
    pub logical_type: UnifiedLogicalType,
    pub dictionary: Vec<Vec<u8>>,
    pub postings: Vec<Vec<u64>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Encode, Decode)]
struct UnifiedExactIndexBitpacked {
    field_id: u32,
    logical_type: UnifiedLogicalType,
    dictionary: Vec<Vec<u8>>,
    postings: Vec<UnifiedBitpackedPostingList>,
}

#[derive(Debug, Clone, PartialEq, Eq, Encode, Decode)]
struct UnifiedBitpackedPostingList {
    row_count: u32,
    base: u64,
    bit_width: u8,
    deltas: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Encode, Decode)]
pub struct UnifiedFieldSchema {
    pub field_id: u32,
    pub name: String,
    pub logical_type: UnifiedLogicalType,
    pub nullable: bool,
    pub indexed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Encode, Decode, Default)]
pub struct UnifiedPayloadSchema {
    pub version: u16,
    pub fields: Vec<UnifiedFieldSchema>,
}

impl UnifiedPayloadSchema {
    pub fn new(fields: Vec<UnifiedFieldSchema>) -> Self {
        Self { version: 1, fields }
    }

    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnifiedHeader {
    pub flags: u32,
    pub dim: u32,
    pub metric: UnifiedMetric,
    pub row_count: u64,
    pub payload_schema_hash: u64,
    pub quantizer_offset: u64,
    pub quantizer_length: u64,
    pub block_dir_offset: u64,
    pub block_count: u32,
    pub footer_offset: u64,
    pub footer_length: u32,
    pub created_at_unix_secs: u64,
}

impl UnifiedHeader {
    pub fn encode(&self) -> io::Result<[u8; UNIFIED_HEADER_SIZE]> {
        let mut out = [0u8; UNIFIED_HEADER_SIZE];
        let mut cursor = Cursor::new(&mut out[..]);

        cursor.write_all(&UNIFIED_MAGIC)?;
        cursor.write_u16::<LittleEndian>(UNIFIED_VERSION)?;
        cursor.write_u16::<LittleEndian>(UNIFIED_HEADER_SIZE as u16)?;
        cursor.write_u32::<LittleEndian>(self.flags)?;
        cursor.write_u32::<LittleEndian>(self.dim)?;
        cursor.write_u32::<LittleEndian>(0)?; // reserved align
        cursor.write_u64::<LittleEndian>(self.row_count)?;
        cursor.write_u64::<LittleEndian>(self.quantizer_offset)?;
        cursor.write_u64::<LittleEndian>(self.quantizer_length)?;
        cursor.write_u64::<LittleEndian>(self.block_dir_offset)?;
        cursor.write_u32::<LittleEndian>(self.block_count)?;
        cursor.write_u32::<LittleEndian>(0)?; // reserved align
        cursor.write_u64::<LittleEndian>(self.footer_offset)?;
        cursor.write_u32::<LittleEndian>(self.footer_length)?;
        cursor.write_u16::<LittleEndian>(self.metric as u16)?;
        cursor.write_u16::<LittleEndian>(0)?; // reserved align
        cursor.write_u64::<LittleEndian>(self.created_at_unix_secs)?;
        cursor.write_u64::<LittleEndian>(self.payload_schema_hash)?;
        cursor.write_all(&[0u8; HEADER_RESERVED_BYTES])?;

        if cursor.position() as usize != UNIFIED_HEADER_SIZE {
            return Err(io::Error::other("unified header encoding size mismatch"));
        }
        Ok(out)
    }

    pub fn decode(bytes: &[u8]) -> io::Result<Self> {
        if bytes.len() < UNIFIED_HEADER_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "unified header truncated",
            ));
        }

        let mut cursor = Cursor::new(&bytes[..UNIFIED_HEADER_SIZE]);
        let mut magic = [0u8; 8];
        cursor.read_exact(&mut magic)?;
        if magic != UNIFIED_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid unified magic",
            ));
        }

        let version = cursor.read_u16::<LittleEndian>()?;
        if version != UNIFIED_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported unified version: {version}"),
            ));
        }

        let header_len = cursor.read_u16::<LittleEndian>()?;
        if header_len as usize != UNIFIED_HEADER_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid unified header size: {header_len}"),
            ));
        }

        let flags = cursor.read_u32::<LittleEndian>()?;
        let dim = cursor.read_u32::<LittleEndian>()?;
        let _reserved = cursor.read_u32::<LittleEndian>()?;
        let row_count = cursor.read_u64::<LittleEndian>()?;
        let quantizer_offset = cursor.read_u64::<LittleEndian>()?;
        let quantizer_length = cursor.read_u64::<LittleEndian>()?;
        let block_dir_offset = cursor.read_u64::<LittleEndian>()?;
        let block_count = cursor.read_u32::<LittleEndian>()?;
        let _reserved2 = cursor.read_u32::<LittleEndian>()?;
        let footer_offset = cursor.read_u64::<LittleEndian>()?;
        let footer_length = cursor.read_u32::<LittleEndian>()?;
        let metric = UnifiedMetric::from_u16(cursor.read_u16::<LittleEndian>()?)?;
        let _reserved3 = cursor.read_u16::<LittleEndian>()?;
        let created_at_unix_secs = cursor.read_u64::<LittleEndian>()?;
        let payload_schema_hash = cursor.read_u64::<LittleEndian>()?;
        let mut reserved = [0u8; HEADER_RESERVED_BYTES];
        cursor.read_exact(&mut reserved)?;

        Ok(Self {
            flags,
            dim,
            metric,
            row_count,
            payload_schema_hash,
            quantizer_offset,
            quantizer_length,
            block_dir_offset,
            block_count,
            footer_offset,
            footer_length,
            created_at_unix_secs,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnifiedFooter {
    pub flags: u32,
    pub row_count: u64,
    pub metric: UnifiedMetric,
    pub payload_schema_hash: u64,
    pub block_dir_offset: u64,
    pub block_count: u32,
    pub directory_crc32: u32,
}

impl UnifiedFooter {
    pub fn encode(&self) -> io::Result<[u8; UNIFIED_FOOTER_SIZE]> {
        let mut out = [0u8; UNIFIED_FOOTER_SIZE];
        let mut cursor = Cursor::new(&mut out[..]);

        cursor.write_all(&UNIFIED_FOOTER_MAGIC)?;
        cursor.write_u16::<LittleEndian>(UNIFIED_VERSION)?;
        cursor.write_u16::<LittleEndian>(UNIFIED_FOOTER_SIZE as u16)?;
        cursor.write_u32::<LittleEndian>(self.flags)?;
        cursor.write_u64::<LittleEndian>(self.row_count)?;
        cursor.write_u64::<LittleEndian>(self.block_dir_offset)?;
        cursor.write_u32::<LittleEndian>(self.block_count)?;
        cursor.write_u32::<LittleEndian>(self.directory_crc32)?;
        cursor.write_u16::<LittleEndian>(self.metric as u16)?;
        cursor.write_u16::<LittleEndian>(0)?; // reserved align
        cursor.write_u64::<LittleEndian>(self.payload_schema_hash)?;
        cursor.write_all(&[0u8; FOOTER_RESERVED_BYTES])?;

        if cursor.position() as usize != UNIFIED_FOOTER_SIZE {
            return Err(io::Error::other("unified footer encoding size mismatch"));
        }
        Ok(out)
    }

    pub fn decode(bytes: &[u8]) -> io::Result<Self> {
        if bytes.len() < UNIFIED_FOOTER_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "unified footer truncated",
            ));
        }

        let mut cursor = Cursor::new(&bytes[..UNIFIED_FOOTER_SIZE]);
        let mut magic = [0u8; 8];
        cursor.read_exact(&mut magic)?;
        if magic != UNIFIED_FOOTER_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid unified footer magic",
            ));
        }

        let version = cursor.read_u16::<LittleEndian>()?;
        if version != UNIFIED_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported unified footer version: {version}"),
            ));
        }

        let footer_len = cursor.read_u16::<LittleEndian>()?;
        if footer_len as usize != UNIFIED_FOOTER_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid unified footer size: {footer_len}"),
            ));
        }

        let flags = cursor.read_u32::<LittleEndian>()?;
        let row_count = cursor.read_u64::<LittleEndian>()?;
        let block_dir_offset = cursor.read_u64::<LittleEndian>()?;
        let block_count = cursor.read_u32::<LittleEndian>()?;
        let directory_crc32 = cursor.read_u32::<LittleEndian>()?;
        let metric = UnifiedMetric::from_u16(cursor.read_u16::<LittleEndian>()?)?;
        let _reserved2 = cursor.read_u16::<LittleEndian>()?;
        let payload_schema_hash = cursor.read_u64::<LittleEndian>()?;
        let mut reserved = [0u8; FOOTER_RESERVED_BYTES];
        cursor.read_exact(&mut reserved)?;

        Ok(Self {
            flags,
            row_count,
            metric,
            payload_schema_hash,
            block_dir_offset,
            block_count,
            directory_crc32,
        })
    }
}

pub fn compute_payload_schema_hash(schema: &UnifiedPayloadSchema) -> io::Result<u64> {
    let bytes = bincode::encode_to_vec(schema, bincode::config::standard())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    Ok(fnv1a64(&bytes))
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;

    let mut hash = OFFSET;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnifiedBlockDesc {
    pub block_type: UnifiedBlockType,
    pub codec: UnifiedCodec,
    pub row_start: u64,
    pub row_count: u32,
    pub offset: u64,
    pub compressed_len: u64,
    pub raw_len: u64,
    pub crc32: u32,
}

impl UnifiedBlockDesc {
    pub fn encode(&self) -> io::Result<[u8; UNIFIED_BLOCK_DESC_SIZE]> {
        let mut out = [0u8; UNIFIED_BLOCK_DESC_SIZE];
        let mut cursor = Cursor::new(&mut out[..]);

        cursor.write_u16::<LittleEndian>(self.block_type as u16)?;
        cursor.write_u16::<LittleEndian>(self.codec as u16)?;
        cursor.write_u64::<LittleEndian>(self.row_start)?;
        cursor.write_u32::<LittleEndian>(self.row_count)?;
        cursor.write_u32::<LittleEndian>(0)?;
        cursor.write_u64::<LittleEndian>(self.offset)?;
        cursor.write_u64::<LittleEndian>(self.compressed_len)?;
        cursor.write_u64::<LittleEndian>(self.raw_len)?;
        cursor.write_u32::<LittleEndian>(self.crc32)?;
        cursor.write_u32::<LittleEndian>(0)?;
        cursor.write_u32::<LittleEndian>(0)?;

        if cursor.position() as usize != UNIFIED_BLOCK_DESC_SIZE {
            return Err(io::Error::other(
                "unified block desc encoding size mismatch",
            ));
        }
        Ok(out)
    }

    pub fn decode(bytes: &[u8]) -> io::Result<Self> {
        if bytes.len() < UNIFIED_BLOCK_DESC_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "unified block descriptor truncated",
            ));
        }

        let mut cursor = Cursor::new(&bytes[..UNIFIED_BLOCK_DESC_SIZE]);
        let block_type = UnifiedBlockType::from_u16(cursor.read_u16::<LittleEndian>()?)?;
        let codec = UnifiedCodec::from_u16(cursor.read_u16::<LittleEndian>()?)?;
        let row_start = cursor.read_u64::<LittleEndian>()?;
        let row_count = cursor.read_u32::<LittleEndian>()?;
        let _reserved = cursor.read_u32::<LittleEndian>()?;
        let offset = cursor.read_u64::<LittleEndian>()?;
        let compressed_len = cursor.read_u64::<LittleEndian>()?;
        let raw_len = cursor.read_u64::<LittleEndian>()?;
        let crc32 = cursor.read_u32::<LittleEndian>()?;
        let _reserved2 = cursor.read_u32::<LittleEndian>()?;
        let _reserved3 = cursor.read_u32::<LittleEndian>()?;

        Ok(Self {
            block_type,
            codec,
            row_start,
            row_count,
            offset,
            compressed_len,
            raw_len,
            crc32,
        })
    }
}

pub fn encode_block_directory(blocks: &[UnifiedBlockDesc]) -> io::Result<Vec<u8>> {
    let mut out = Vec::with_capacity(4 + blocks.len().saturating_mul(UNIFIED_BLOCK_DESC_SIZE));
    out.write_u32::<LittleEndian>(blocks.len() as u32)?;
    for block in blocks {
        out.extend_from_slice(&block.encode()?);
    }
    Ok(out)
}

pub fn decode_block_directory(bytes: &[u8]) -> io::Result<Vec<UnifiedBlockDesc>> {
    if bytes.len() < 4 {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "unified block directory truncated",
        ));
    }

    let mut cursor = Cursor::new(bytes);
    let count = cursor.read_u32::<LittleEndian>()? as usize;
    let expected = 4usize
        .checked_add(count.saturating_mul(UNIFIED_BLOCK_DESC_SIZE))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "block directory overflow"))?;

    if bytes.len() != expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "block directory length mismatch: expected={expected}, actual={}",
                bytes.len()
            ),
        ));
    }

    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let start = cursor.position() as usize;
        let end = start + UNIFIED_BLOCK_DESC_SIZE;
        out.push(UnifiedBlockDesc::decode(&bytes[start..end])?);
        cursor.set_position(end as u64);
    }
    Ok(out)
}

pub fn encode_exact_key(
    logical_type: &UnifiedLogicalType,
    value: &UnifiedPayloadValue,
) -> io::Result<Option<Vec<u8>>> {
    if matches!(value, UnifiedPayloadValue::Null) {
        return Ok(None);
    }

    let mut out = Vec::new();
    match (logical_type, value) {
        (UnifiedLogicalType::Bool, UnifiedPayloadValue::Bool(v)) => out.push(u8::from(*v)),
        (UnifiedLogicalType::Int64, UnifiedPayloadValue::Int64(v))
        | (UnifiedLogicalType::TimestampMicros, UnifiedPayloadValue::TimestampMicros(v)) => {
            out.write_i64::<LittleEndian>(*v)?;
        }
        (UnifiedLogicalType::Float32, UnifiedPayloadValue::Float32(v)) => {
            out.write_u32::<LittleEndian>(v.to_bits())?;
        }
        (UnifiedLogicalType::Float64, UnifiedPayloadValue::Float64(v)) => {
            out.write_u64::<LittleEndian>(v.to_bits())?;
        }
        (UnifiedLogicalType::Keyword, UnifiedPayloadValue::Keyword(v))
        | (UnifiedLogicalType::Text, UnifiedPayloadValue::Text(v)) => {
            out.extend_from_slice(v.as_bytes());
        }
        (UnifiedLogicalType::Bytes, UnifiedPayloadValue::Bytes(v)) => {
            out.extend_from_slice(v);
        }
        (UnifiedLogicalType::LobRef, UnifiedPayloadValue::LobRef(v)) => {
            out = bincode::encode_to_vec(v, bincode::config::standard())
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        }
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "payload value type mismatch for exact key encoding",
            ));
        }
    }
    Ok(Some(out))
}

pub fn encode_exact_index_bitpacked(index: &UnifiedExactIndex) -> io::Result<Vec<u8>> {
    if index.dictionary.len() != index.postings.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "exact index dictionary/postings length mismatch",
        ));
    }

    let mut packed_postings = Vec::with_capacity(index.postings.len());
    for rows in &index.postings {
        let row_count = u32::try_from(rows.len()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "exact index postings length exceeds u32",
            )
        })?;
        if rows.is_empty() {
            packed_postings.push(UnifiedBitpackedPostingList {
                row_count,
                base: 0,
                bit_width: 0,
                deltas: Vec::new(),
            });
            continue;
        }

        let base = rows[0];
        let mut prev = base;
        let mut deltas = Vec::with_capacity(rows.len());
        deltas.push(0);
        for &row in rows.iter().skip(1) {
            if row <= prev {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "exact index postings must be strictly increasing",
                ));
            }
            let delta = row.checked_sub(prev).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "exact index posting delta underflow",
                )
            })?;
            deltas.push(delta);
            prev = row;
        }

        let max_delta = deltas.iter().copied().max().unwrap_or(0);
        let bit_width = bit_width_u64(max_delta);
        let packed = pack_u64_values(&deltas, bit_width)?;
        packed_postings.push(UnifiedBitpackedPostingList {
            row_count,
            base,
            bit_width,
            deltas: packed,
        });
    }

    let encoded = UnifiedExactIndexBitpacked {
        field_id: index.field_id,
        logical_type: index.logical_type.clone(),
        dictionary: index.dictionary.clone(),
        postings: packed_postings,
    };
    bincode::encode_to_vec(&encoded, bincode::config::standard())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

pub fn decode_exact_index_bitpacked(bytes: &[u8]) -> io::Result<UnifiedExactIndex> {
    let (encoded, consumed): (UnifiedExactIndexBitpacked, usize) =
        bincode::decode_from_slice(bytes, bincode::config::standard())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    if consumed != bytes.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "exact index bitpacked payload has trailing bytes",
        ));
    }
    if encoded.dictionary.len() != encoded.postings.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "exact index dictionary/postings length mismatch",
        ));
    }

    let mut postings = Vec::with_capacity(encoded.postings.len());
    for entry in &encoded.postings {
        let row_count = entry.row_count as usize;
        let deltas = unpack_u64_values(&entry.deltas, row_count, entry.bit_width)?;
        let mut rows = Vec::with_capacity(row_count);
        let mut prev = entry.base;
        for (i, delta) in deltas.into_iter().enumerate() {
            let row = if i == 0 {
                entry.base.checked_add(delta).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "exact index posting overflow on first row",
                    )
                })?
            } else {
                prev.checked_add(delta).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "exact index posting overflow on delta application",
                    )
                })?
            };
            if i > 0 && row <= prev {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "exact index postings are not strictly increasing",
                ));
            }
            rows.push(row);
            prev = row;
        }
        postings.push(rows);
    }

    Ok(UnifiedExactIndex {
        field_id: encoded.field_id,
        logical_type: encoded.logical_type,
        dictionary: encoded.dictionary,
        postings,
    })
}

fn bit_width_u64(max_value: u64) -> u8 {
    if max_value == 0 {
        0
    } else {
        (64 - max_value.leading_zeros()) as u8
    }
}

fn pack_u64_values(values: &[u64], bit_width: u8) -> io::Result<Vec<u8>> {
    if bit_width > 64 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "bit width exceeds 64",
        ));
    }
    if bit_width == 0 {
        if values.iter().any(|&v| v != 0) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "zero-width bitpack cannot encode non-zero values",
            ));
        }
        return Ok(Vec::new());
    }

    let total_bits = values
        .len()
        .checked_mul(bit_width as usize)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "bitpack length overflow"))?;
    let mut out = vec![0u8; total_bits.div_ceil(8)];
    let mut bit_cursor = 0usize;
    for &value in values {
        let masked = if bit_width == 64 {
            value
        } else {
            value & ((1u64 << bit_width) - 1)
        };
        let mut current = masked;
        for _ in 0..bit_width {
            if (current & 1) != 0 {
                out[bit_cursor / 8] |= 1u8 << (bit_cursor % 8);
            }
            current >>= 1;
            bit_cursor += 1;
        }
    }
    Ok(out)
}

fn unpack_u64_values(data: &[u8], count: usize, bit_width: u8) -> io::Result<Vec<u64>> {
    if bit_width > 64 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "bit width exceeds 64",
        ));
    }
    if bit_width == 0 {
        if !data.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "zero-width bitpack has unexpected payload",
            ));
        }
        return Ok(vec![0; count]);
    }

    let total_bits = count
        .checked_mul(bit_width as usize)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "bit unpack length overflow"))?;
    let expected_bytes = total_bits.div_ceil(8);
    if data.len() != expected_bytes {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "bitpacked postings length mismatch",
        ));
    }

    let mut out = Vec::with_capacity(count);
    let mut bit_cursor = 0usize;
    for _ in 0..count {
        let mut value = 0u64;
        for bit in 0..bit_width {
            let raw = (data[bit_cursor / 8] >> (bit_cursor % 8)) & 1;
            value |= (raw as u64) << bit;
            bit_cursor += 1;
        }
        out.push(value);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unified_header_roundtrip() {
        let header = UnifiedHeader {
            flags: 7,
            dim: 768,
            metric: UnifiedMetric::Cosine,
            row_count: 1000,
            payload_schema_hash: 0xdead_beef_cafe_f00d,
            quantizer_offset: 128,
            quantizer_length: 4096,
            block_dir_offset: 9000,
            block_count: 3,
            footer_offset: 9800,
            footer_length: UNIFIED_FOOTER_SIZE as u32,
            created_at_unix_secs: 1234,
        };

        let bytes = header.encode().unwrap();
        let decoded = UnifiedHeader::decode(&bytes).unwrap();
        assert_eq!(decoded, header);
    }

    #[test]
    fn unified_footer_roundtrip() {
        let footer = UnifiedFooter {
            flags: 1,
            row_count: 55,
            metric: UnifiedMetric::L2,
            payload_schema_hash: 0x0102_0304_0506_0708,
            block_dir_offset: 1024,
            block_count: 3,
            directory_crc32: 99,
        };

        let bytes = footer.encode().unwrap();
        let decoded = UnifiedFooter::decode(&bytes).unwrap();
        assert_eq!(decoded, footer);
    }

    #[test]
    fn unified_block_directory_roundtrip() {
        let input = vec![
            UnifiedBlockDesc {
                block_type: UnifiedBlockType::Quantizer,
                codec: UnifiedCodec::Bincode,
                row_start: 0,
                row_count: 0,
                offset: 128,
                compressed_len: 512,
                raw_len: 512,
                crc32: 12,
            },
            UnifiedBlockDesc {
                block_type: UnifiedBlockType::Ids,
                codec: UnifiedCodec::PlainLe,
                row_start: 0,
                row_count: 10,
                offset: 640,
                compressed_len: 80,
                raw_len: 80,
                crc32: 34,
            },
        ];

        let bytes = encode_block_directory(&input).unwrap();
        let decoded = decode_block_directory(&bytes).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn unified_payload_schema_roundtrip() {
        let schema = UnifiedPayloadSchema::new(vec![
            UnifiedFieldSchema {
                field_id: 1,
                name: "tenant_id".to_string(),
                logical_type: UnifiedLogicalType::Int64,
                nullable: false,
                indexed: true,
            },
            UnifiedFieldSchema {
                field_id: 2,
                name: "body".to_string(),
                logical_type: UnifiedLogicalType::Text,
                nullable: true,
                indexed: false,
            },
        ]);

        let bytes = bincode::encode_to_vec(&schema, bincode::config::standard()).unwrap();
        let (decoded, _): (UnifiedPayloadSchema, usize) =
            bincode::decode_from_slice(&bytes, bincode::config::standard()).unwrap();

        assert_eq!(decoded, schema);
    }

    #[test]
    fn unified_exact_key_roundtrip_stability() {
        let encoded =
            encode_exact_key(&UnifiedLogicalType::Int64, &UnifiedPayloadValue::Int64(42)).unwrap();
        assert_eq!(encoded.unwrap().len(), 8);
    }

    #[test]
    fn unified_exact_index_bitpacked_roundtrip() {
        let index = UnifiedExactIndex {
            field_id: 7,
            logical_type: UnifiedLogicalType::Keyword,
            dictionary: vec![b"acme".to_vec(), b"globex".to_vec()],
            postings: vec![vec![10, 20, 40], vec![30]],
        };
        let bytes = encode_exact_index_bitpacked(&index).unwrap();
        let decoded = decode_exact_index_bitpacked(&bytes).unwrap();
        assert_eq!(decoded, index);
    }

    #[test]
    fn unified_exact_index_bitpacked_rejects_trailing_bytes() {
        let index = UnifiedExactIndex {
            field_id: 1,
            logical_type: UnifiedLogicalType::Int64,
            dictionary: vec![vec![1]],
            postings: vec![vec![42]],
        };
        let mut bytes = encode_exact_index_bitpacked(&index).unwrap();
        bytes.push(0xff);
        let err = decode_exact_index_bitpacked(&bytes).unwrap_err();
        assert!(err.to_string().contains("trailing bytes"));
    }

    #[test]
    fn unified_exact_index_bitpacked_rejects_invalid_bit_width() {
        let malformed = UnifiedExactIndexBitpacked {
            field_id: 2,
            logical_type: UnifiedLogicalType::Keyword,
            dictionary: vec![b"acme".to_vec()],
            postings: vec![UnifiedBitpackedPostingList {
                row_count: 1,
                base: 10,
                bit_width: 65,
                deltas: Vec::new(),
            }],
        };
        let bytes = bincode::encode_to_vec(&malformed, bincode::config::standard()).unwrap();
        let err = decode_exact_index_bitpacked(&bytes).unwrap_err();
        assert!(err.to_string().contains("bit width exceeds 64"));
    }

    #[test]
    fn unified_exact_index_bitpacked_rejects_truncated_deltas() {
        let malformed = UnifiedExactIndexBitpacked {
            field_id: 3,
            logical_type: UnifiedLogicalType::Keyword,
            dictionary: vec![b"acme".to_vec()],
            postings: vec![UnifiedBitpackedPostingList {
                row_count: 16,
                base: 7,
                bit_width: 1,
                deltas: vec![0u8], // should be 2 bytes for 16 one-bit deltas
            }],
        };
        let bytes = bincode::encode_to_vec(&malformed, bincode::config::standard()).unwrap();
        let err = decode_exact_index_bitpacked(&bytes).unwrap_err();
        assert!(
            err.to_string()
                .contains("bitpacked postings length mismatch")
        );
    }
}
