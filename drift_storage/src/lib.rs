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

#[cfg(test)]
mod footer_test;

// SHARED CONSTANTS (Single Source of Truth)
pub const MAGIC_BYTES: &[u8; 8] = b"DRIFT_V1";
pub const FOOTER_SIZE: usize = 64; // 8*7 fields + 1 version + 7 padding = 64

#[repr(C)]
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
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

        // 1 byte Version
        cursor.write_u8(self.version).unwrap();

        // 6 * 8 = 48 bytes Offsets/Lengths
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

        // Padding (7 bytes) to align to 56 bytes so Magic starts at 56
        // 1 + 48 + 7 = 56
        cursor.write_all(&[0u8; 7]).unwrap();

        // 8 bytes Magic
        cursor.write_all(&self.magic).unwrap();

        // Total: 56 + 8 = 64 bytes
        buf
    }

    // Helper to read back (useful for tests and Reader)
    pub fn from_bytes(bytes: &[u8; FOOTER_SIZE]) -> std::io::Result<Self> {
        let mut cursor = std::io::Cursor::new(bytes);
        use byteorder::{LittleEndian, ReadBytesExt};
        use std::io::Read;

        let version = cursor.read_u8()?;
        let index_offset = cursor.read_u64::<LittleEndian>()?;
        let index_length = cursor.read_u64::<LittleEndian>()?;
        let bloom_offset = cursor.read_u64::<LittleEndian>()?;
        let bloom_length = cursor.read_u64::<LittleEndian>()?;
        let quantizer_offset = cursor.read_u64::<LittleEndian>()?;
        let quantizer_length = cursor.read_u64::<LittleEndian>()?;

        // Skip Padding
        cursor.set_position(cursor.position() + 7);

        let mut magic = [0u8; 8];
        cursor.read_exact(&mut magic)?;

        Ok(Self {
            version,
            index_offset,
            index_length,
            bloom_offset,
            bloom_length,
            quantizer_offset,
            quantizer_length,
            magic,
        })
    }
}
