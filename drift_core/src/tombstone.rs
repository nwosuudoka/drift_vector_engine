use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use crc32fast::Hasher;
use std::io::{self, Cursor};

pub const TOMBSTONE_MAGIC: u32 = 0xDEADBEEF;

pub struct TombstoneFile {
    pub deleted_ids: Vec<u64>,
}

impl TombstoneFile {
    pub fn new(ids: Vec<u64>) -> Self {
        Self { deleted_ids: ids }
    }

    pub fn to_bytes(&self) -> io::Result<Vec<u8>> {
        let mut buf = Vec::new();
        // 1. Header
        buf.write_u32::<LittleEndian>(TOMBSTONE_MAGIC)?;
        buf.write_u64::<LittleEndian>(self.deleted_ids.len() as u64)?;

        // 2. Payload & Checksum
        let mut hasher = Hasher::new();
        for &id in &self.deleted_ids {
            buf.write_u64::<LittleEndian>(id)?;
            hasher.update(&id.to_le_bytes());
        }

        // 3. Footer
        buf.write_u32::<LittleEndian>(hasher.finalize())?;

        Ok(buf)
    }

    pub fn from_bytes(data: &[u8]) -> io::Result<Self> {
        let mut cursor = Cursor::new(data);

        // 1. Header Check
        let magic = cursor.read_u32::<LittleEndian>()?;
        if magic != TOMBSTONE_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid Tombstone Magic",
            ));
        }

        let count = cursor.read_u64::<LittleEndian>()? as usize;

        // Safety check: Prevent massive allocation on corrupt file
        if count > data.len() / 8 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Corrupt count in tombstone file",
            ));
        }

        let mut ids = Vec::with_capacity(count);
        let mut hasher = Hasher::new();

        // 2. Read IDs
        for _ in 0..count {
            let id = cursor.read_u64::<LittleEndian>()?;
            ids.push(id);
            hasher.update(&id.to_le_bytes());
        }

        // 3. Verify CRC
        let stored_crc = cursor.read_u32::<LittleEndian>()?;
        if stored_crc != hasher.finalize() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Tombstone CRC Mismatch",
            ));
        }

        Ok(Self { deleted_ids: ids })
    }
}
