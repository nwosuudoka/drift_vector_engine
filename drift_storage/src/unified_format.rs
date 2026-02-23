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

const HEADER_RESERVED_BYTES: usize = 40;
const FOOTER_RESERVED_BYTES: usize = 24;

pub const UNIFIED_FLAG_HAS_PAYLOAD_SCHEMA: u32 = 1 << 0;
pub const UNIFIED_FLAG_HAS_PAYLOAD_COLUMNS: u32 = 1 << 1;
pub const UNIFIED_FLAG_HAS_EXACT_INDEX: u32 = 1 << 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum UnifiedBlockType {
    Quantizer = 1,
    Ids = 2,
    VectorCodes = 3,
    PayloadSchema = 10,
    PayloadColumn = 11,
    PayloadExactIndex = 12,
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
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported codec: {v}"),
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
pub struct UnifiedExactIndex {
    pub field_id: u32,
    pub logical_type: UnifiedLogicalType,
    pub dictionary: Vec<Vec<u8>>,
    pub postings: Vec<Vec<u64>>,
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
    pub row_count: u64,
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
        cursor.write_u32::<LittleEndian>(0)?; // reserved align
        cursor.write_u64::<LittleEndian>(self.created_at_unix_secs)?;
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
        let _reserved3 = cursor.read_u32::<LittleEndian>()?;
        let created_at_unix_secs = cursor.read_u64::<LittleEndian>()?;
        let mut reserved = [0u8; HEADER_RESERVED_BYTES];
        cursor.read_exact(&mut reserved)?;

        Ok(Self {
            flags,
            dim,
            row_count,
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
        let mut reserved = [0u8; FOOTER_RESERVED_BYTES];
        cursor.read_exact(&mut reserved)?;

        Ok(Self {
            flags,
            row_count,
            block_dir_offset,
            block_count,
            directory_crc32,
        })
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unified_header_roundtrip() {
        let header = UnifiedHeader {
            flags: 7,
            dim: 768,
            row_count: 1000,
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
}
