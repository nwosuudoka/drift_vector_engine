use serde::{Deserialize, Serialize};

pub mod block;
pub mod compression;
pub mod disk_manager;
pub mod segment_reader;
pub mod segment_writer;

#[cfg(test)]
mod open_dal_integration;

#[cfg(test)]
mod segment_test;

// SHARED CONSTANTS (Single Source of Truth)
pub const MAGIC_BYTES: &[u8; 8] = b"DRIFT_V1";
pub const FOOTER_SIZE: usize = 64; // 8*7 fields + 1 version + 7 padding = 64

#[repr(C)]
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct DriftFooter {
    pub version: u8,
    pub index_offset: u64,
    pub index_length: u64,
    pub bloom_offset: u64,
    pub bloom_length: u64,
    pub quantizer_offset: u64,
    pub quantizer_length: u64,
    pub magic: [u8; 8],
}

impl DriftFooter {
    pub fn new(
        index_offset: u64,
        index_length: u64,
        bloom_offset: u64,
        bloom_length: u64,
        quantizer_offset: u64,
        quantizer_length: u64,
    ) -> Self {
        Self {
            version: 1,
            index_offset,
            index_length,
            bloom_offset,
            bloom_length,
            quantizer_offset,
            quantizer_length,
            magic: *MAGIC_BYTES,
        }
    }

    pub fn to_bytes(&self) -> [u8; FOOTER_SIZE] {
        let mut buf = [0u8; FOOTER_SIZE];
        let mut cursor = std::io::Cursor::new(&mut buf[..]);
        use byteorder::{LittleEndian, WriteBytesExt};
        use std::io::Write;

        cursor.write_u8(self.version).unwrap();
        cursor.write_u64::<LittleEndian>(self.index_offset).unwrap();
        cursor.write_u64::<LittleEndian>(self.index_length).unwrap();
        cursor.write_u64::<LittleEndian>(self.bloom_offset).unwrap();
        cursor.write_u64::<LittleEndian>(self.bloom_length).unwrap();
        cursor
            .write_u64::<LittleEndian>(self.quantizer_offset)
            .unwrap();
        cursor
            .write_u64::<LittleEndian>(self.quantizer_length)
            .unwrap();

        // Padding (7 bytes) to align to 64
        cursor.write_all(&[0u8; 7]).unwrap();

        cursor.write_all(&self.magic).unwrap();
        buf
    }
}
