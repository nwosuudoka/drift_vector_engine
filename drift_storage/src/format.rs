// drift_storage/src/format.rs

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

// --- CONSTANTS ---
pub const MAGIC_V2: u64 = 0x32565F5446495244;
pub const VERSION_2: u16 = 2;
pub const HEADER_SIZE: usize = 128;
pub const ROW_GROUP_HEADER_SIZE: usize = 64;
pub const FOOTER_SIZE: usize = 128;

/// 1. File Header (Fixed 128 Bytes)
/// Zero-Copy compatible. Layout matches C-struct.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, IntoBytes, FromBytes, Immutable, KnownLayout)]
pub struct DriftHeader {
    pub magic: u64,         // 0..8
    pub version: u16,       // 8..10
    pub flags: u16,         // 10..12
    pub _reserved: u32,     // 12..16 (Added to fix alignment gap)
    pub total_vectors: u64, // 16..24 (Now naturally aligned)
    pub created_at: u64,    // 24..32
    pub run_id: [u8; 16],   // 32..48
    pub padding: [u8; 80],  // 48..128 (Reduced from 84 to 80 to account for _reserved)
}

impl DriftHeader {
    pub fn new(total_vectors: u64, run_id: [u8; 16]) -> Self {
        Self {
            magic: MAGIC_V2,
            version: VERSION_2,
            flags: 0,
            _reserved: 0, // Must initialize explicit padding
            total_vectors,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            run_id,
            padding: [0u8; 80],
        }
    }

    pub fn validate(&self) -> bool {
        self.magic == MAGIC_V2 && self.version == VERSION_2
    }
}

/// 2. Row Group Header (Fixed 64 Bytes)
/// Note: This struct was already naturally aligned, so no changes needed.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, IntoBytes, FromBytes, Immutable, KnownLayout)]
pub struct RowGroupHeader {
    pub vector_count: u32,
    pub checksum: u32,
    pub hot_offset: u64,
    pub hot_length: u32,
    pub _pad_1: u32,
    pub cold_offset: u64,
    pub cold_length: u32,
    pub padding: [u8; 28],
}

impl RowGroupHeader {
    pub fn new(
        vector_count: u32,
        checksum: u32,
        hot_offset: u64,
        hot_length: u32,
        cold_offset: u64,
        cold_length: u32,
    ) -> Self {
        Self {
            vector_count,
            checksum,
            hot_offset,
            hot_length,
            _pad_1: 0,
            cold_offset,
            cold_length,
            padding: [0u8; 28],
        }
    }
}

/// 3. File Footer (Fixed 128 Bytes)
/// Note: This struct was already naturally aligned.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, IntoBytes, FromBytes, Immutable, KnownLayout)]
pub struct DriftFooter {
    pub row_group_count: u32,
    pub _pad_1: u32,
    pub index_start_offset: u64,
    pub bloom_filter_offset: u64,
    pub bloom_filter_length: u32,
    pub padding: [u8; 92],
    pub magic: u64,
}

impl DriftFooter {
    pub fn new(
        row_group_count: u32,
        index_start_offset: u64,
        bloom_offset: u64,
        bloom_length: u32,
    ) -> Self {
        Self {
            row_group_count,
            _pad_1: 0,
            index_start_offset,
            bloom_filter_offset: bloom_offset,
            bloom_filter_length: bloom_length,
            padding: [0u8; 92],
            magic: MAGIC_V2,
        }
    }
}
