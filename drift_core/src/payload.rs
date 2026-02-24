use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq)]
pub enum PayloadValue {
    Bool(bool),
    Int64(i64),
    Float32(f32),
    Float64(f64),
    Keyword(String),
    Text(String),
    Bytes(Vec<u8>),
    TimestampMicros(i64),
    LobRef(PayloadLobRef),
    Null,
}

pub type PayloadRow = BTreeMap<u32, PayloadValue>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PayloadLogicalType {
    Bool,
    Int64,
    Float32,
    Float64,
    Keyword,
    Text,
    Bytes,
    TimestampMicros,
    LobRef,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PayloadLobRef {
    pub blob_key: String,
    pub offset: u64,
    pub length: u64,
    pub fingerprint: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PayloadFieldSchema {
    pub field_id: u32,
    pub name: String,
    pub logical_type: PayloadLogicalType,
    pub nullable: bool,
    pub indexed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PayloadSchema {
    pub fields: Vec<PayloadFieldSchema>,
}

impl PayloadSchema {
    pub fn new(fields: Vec<PayloadFieldSchema>) -> Self {
        Self { fields }
    }

    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }
}
