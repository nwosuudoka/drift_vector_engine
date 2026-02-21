// drift_storage/src/format.rs

use zerocopy::{FromBytes, FromZeros, Immutable, IntoBytes, KnownLayout};

// --- CONSTANTS ---
pub const MAGIC_V3: u64 = 0x33565F5446495244;
pub const MAGIC_CURRENT: u64 = MAGIC_V3;
pub const VERSION_3: u16 = 3;
pub const VERSION_CURRENT: u16 = VERSION_3;
pub const HEADER_SIZE: usize = 128;
pub const ROW_GROUP_HEADER_SIZE: usize = 64;
pub const FOOTER_SIZE: usize = 128;

/// 1. File Header (Fixed 128 Bytes)
///    Zero-Copy compatible. Layout matches C-struct.
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

    pub quantizer_offset: u64, // 48..56
    pub quantizer_length: u32, // 56..60
    pub _pad_1: u32,           // 60..64
    pub padding: [u8; 64],     // 64..128
}

impl DriftHeader {
    pub fn new(
        total_vectors: u64,
        run_id: [u8; 16],
        quantizer_offset: u64,
        quantizer_length: u32,
    ) -> Self {
        Self {
            magic: MAGIC_CURRENT,
            version: VERSION_CURRENT,
            flags: 0,
            _reserved: 0, // Must initialize explicit padding
            total_vectors,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            run_id,
            quantizer_offset,
            quantizer_length,
            _pad_1: 0,
            padding: [0u8; 64],
        }
    }

    pub fn force_copy(buf: &[u8]) -> Self {
        let mut header = DriftHeader::new_zeroed();
        let copy_len = buf.len().min(HEADER_SIZE);
        unsafe {
            std::ptr::copy_nonoverlapping(buf.as_ptr(), &mut header as *mut _ as *mut u8, copy_len);
        }
        header // Return the OWNED struct
    }

    pub fn is_supported_magic(magic: u64) -> bool {
        magic == MAGIC_CURRENT
    }

    pub fn is_supported_version(version: u16) -> bool {
        version == VERSION_CURRENT
    }

    pub fn validate(&self) -> bool {
        Self::is_supported_magic(self.magic) && Self::is_supported_version(self.version)
    }
}

/// 2. Row Group Header (Fixed 64 Bytes)
///    Note: This struct was already naturally aligned, so no changes needed.
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
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, IntoBytes, FromBytes, Immutable, KnownLayout)]
pub struct DriftFooter {
    pub total_vector_count: u64,  // 0..8
    pub index_start_offset: u64,  // 8..16
    pub bloom_filter_offset: u64, // 16..24
    pub bloom_filter_length: u32, // 24..28
    pub row_group_count: u32,     // 28..32
    pub padding: [u8; 88],        // 32..120 (Reduced from 92 to 80)
    pub magic: u64,               // 120..128
}

impl DriftFooter {
    pub fn new(
        total_vector_count: u64,
        row_group_count: u32,
        index_start_offset: u64,
        bloom_offset: u64,
        bloom_length: u32,
    ) -> Self {
        Self {
            total_vector_count,
            row_group_count,
            index_start_offset,
            bloom_filter_offset: bloom_offset,
            bloom_filter_length: bloom_length,
            padding: [0u8; 88],
            magic: MAGIC_CURRENT,
        }
    }

    pub fn is_supported_magic(magic: u64) -> bool {
        magic == MAGIC_CURRENT
    }
}
