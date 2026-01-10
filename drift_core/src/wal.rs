// drift_core/src/wal.rs

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use crc32fast::Hasher;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

// OpCodes
const OP_INSERT: u8 = 0x01;
const OP_DELETE: u8 = 0x02; // Future

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub enum WalEntry {
    Insert { id: u64, vector: Vec<f32> },
    Delete { id: u64 },
}

pub struct WalWriter {
    writer: BufWriter<File>,
}

impl WalWriter {
    pub fn new(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .append(true)
            .open(path)?;

        Ok(Self {
            writer: BufWriter::new(file),
        })
    }

    pub fn write_insert(&mut self, id: u64, vector: &[f32]) -> io::Result<()> {
        // 1. Serialize Payload
        // Layout: [OpCode (1)] [ID (8)] [VecLen (4)] [VectorData (N*4)]
        let payload_len = 1 + 8 + 4 + (vector.len() * 4);
        let mut payload = Vec::with_capacity(payload_len);

        payload.write_u8(OP_INSERT)?;
        payload.write_u64::<LittleEndian>(id)?;
        payload.write_u32::<LittleEndian>(vector.len() as u32)?;

        for &val in vector {
            payload.write_f32::<LittleEndian>(val)?;
        }

        // 2. Calculate Checksum
        let mut hasher = Hasher::new();
        hasher.update(&payload);
        let checksum = hasher.finalize();

        // 3. Write Frame: [CRC (4)] [Len (4)] [Payload]
        self.writer.write_u32::<LittleEndian>(checksum)?;
        self.writer.write_u32::<LittleEndian>(payload_len as u32)?;
        self.writer.write_all(&payload)?;

        // Note: Removed the broken `bincode` block that referenced `self.file`.
        // The manual serialization above matches WalReader logic.

        Ok(())
    }

    pub fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }

    pub fn sync(&mut self) -> io::Result<()> {
        self.writer.flush()?;
        self.writer.get_ref().sync_all()
    }

    /// NEW: Truncate the WAL to 0 bytes.
    /// Used by the Janitor after successfully flushing MemTable to disk.
    pub fn truncate(&mut self) -> io::Result<()> {
        // 1. Ensure buffer is flushed
        self.writer.flush()?;

        // 2. Access the underlying File
        let file = self.writer.get_mut();

        // 3. Reset size and position
        file.set_len(0)?;
        file.seek(SeekFrom::Start(0))?;

        // 4. Sync to ensure FS metadata update is durable
        file.sync_all()?;

        Ok(())
    }

    pub fn write_delete(&mut self, id: u64) -> io::Result<()> {
        let payload_len = 1 + 8; // OpCode (1) + ID (8)
        let mut payload = Vec::with_capacity(payload_len);

        payload.write_u8(OP_DELETE)?;
        payload.write_u64::<LittleEndian>(id)?;

        let mut hasher = Hasher::new();
        hasher.update(&payload);
        let checksum = hasher.finalize();

        self.writer.write_u32::<LittleEndian>(checksum)?;
        self.writer.write_u32::<LittleEndian>(payload_len as u32)?;
        self.writer.write_all(&payload)?;

        Ok(())
    }
}

pub struct WalReader {
    reader: BufReader<File>,
}

impl WalReader {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = File::open(path)?;
        Ok(Self {
            reader: BufReader::new(file),
        })
    }

    pub fn read_all(mut self) -> Vec<WalEntry> {
        let mut entries = Vec::new();

        #[allow(clippy::while_let_loop)]
        loop {
            // 1. Read Frame Header (CRC + Len)
            let crc_expected = match self.reader.read_u32::<LittleEndian>() {
                Ok(c) => c,
                Err(_) => break, // EOF
            };

            let len = match self.reader.read_u32::<LittleEndian>() {
                Ok(l) => l as usize,
                Err(_) => break, // Truncated
            };

            // 2. Read Payload
            let mut payload = vec![0u8; len];
            if self.reader.read_exact(&mut payload).is_err() {
                break; // Truncated payload
            }

            // 3. Verify CRC
            let mut hasher = Hasher::new();
            hasher.update(&payload);
            if hasher.finalize() != crc_expected {
                break; // Corrupt
            }

            // 4. Deserialize
            if let Ok(entry) = Self::deserialize_entry(&payload) {
                entries.push(entry);
            } else {
                break;
            }
        }

        entries
    }

    fn deserialize_entry(data: &[u8]) -> io::Result<WalEntry> {
        let mut cursor = std::io::Cursor::new(data);
        let opcode = cursor.read_u8()?;

        match opcode {
            OP_INSERT => {
                let id = cursor.read_u64::<LittleEndian>()?;
                let vec_len = cursor.read_u32::<LittleEndian>()? as usize;
                let mut vector = Vec::with_capacity(vec_len);
                for _ in 0..vec_len {
                    vector.push(cursor.read_f32::<LittleEndian>()?);
                }
                Ok(WalEntry::Insert { id, vector })
            }
            OP_DELETE => {
                let id = cursor.read_u64::<LittleEndian>()?;
                Ok(WalEntry::Delete { id })
            }
            _ => Err(io::Error::new(io::ErrorKind::InvalidData, "Unknown OpCode")),
        }
    }
}
