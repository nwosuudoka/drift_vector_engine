use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use crc32fast::Hasher;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

// OpCodes
const OP_INSERT: u8 = 0x01;
const OP_DELETE: u8 = 0x02;
const OP_BEGIN: u8 = 0xF0; // New
const OP_COMMIT: u8 = 0xF1; // New

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub enum WalEntry {
    Begin { tx_id: u64 },
    Commit { tx_id: u64 },
    Insert { id: u64, vector: Vec<f32> },
    Delete { id: u64 },
}

pub struct WalWriter {
    writer: BufWriter<File>,
    current_tx_id: u64,
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
            current_tx_id: 0, // In real system, recover this from file end
        })
    }

    /// Starts a new transaction block.
    pub fn begin_transaction(&mut self) -> io::Result<u64> {
        self.current_tx_id += 1;
        let tx_id = self.current_tx_id;

        let payload_len = 1 + 8; // OpCode + TxID
        let mut payload = Vec::with_capacity(payload_len);
        payload.write_u8(OP_BEGIN)?;
        payload.write_u64::<LittleEndian>(tx_id)?;

        self.write_frame(&payload)?;
        Ok(tx_id)
    }

    /// Commits the current transaction block.
    pub fn commit_transaction(&mut self, tx_id: u64) -> io::Result<()> {
        let payload_len = 1 + 8;
        let mut payload = Vec::with_capacity(payload_len);
        payload.write_u8(OP_COMMIT)?;
        payload.write_u64::<LittleEndian>(tx_id)?;

        self.write_frame(&payload)?;
        // Force flush to OS
        self.sync()?;
        Ok(())
    }

    pub fn write_insert(&mut self, id: u64, vector: &[f32]) -> io::Result<()> {
        let payload_len = 1 + 8 + 4 + (vector.len() * 4);
        let mut payload = Vec::with_capacity(payload_len);
        payload.write_u8(OP_INSERT)?;
        payload.write_u64::<LittleEndian>(id)?;
        payload.write_u32::<LittleEndian>(vector.len() as u32)?;

        for &val in vector {
            payload.write_f32::<LittleEndian>(val)?;
        }
        self.write_frame(&payload)
    }

    pub fn write_delete(&mut self, id: u64) -> io::Result<()> {
        let payload_len = 1 + 8;
        let mut payload = Vec::with_capacity(payload_len);
        payload.write_u8(OP_DELETE)?;
        payload.write_u64::<LittleEndian>(id)?;
        self.write_frame(&payload)
    }

    /// Helper to write [CRC][Len][Payload]
    fn write_frame(&mut self, payload: &[u8]) -> io::Result<()> {
        let mut hasher = Hasher::new();
        hasher.update(payload);
        let checksum = hasher.finalize();

        self.writer.write_u32::<LittleEndian>(checksum)?;
        self.writer
            .write_u32::<LittleEndian>(payload.len() as u32)?;
        self.writer.write_all(payload)?;
        Ok(())
    }

    pub fn sync(&mut self) -> io::Result<()> {
        self.writer.flush()?;
        self.writer.get_ref().sync_all()
    }

    pub fn truncate(&mut self) -> io::Result<()> {
        self.writer.flush()?;
        let file = self.writer.get_mut();
        file.set_len(0)?;
        file.seek(SeekFrom::Start(0))?;
        file.sync_all()?;
        self.current_tx_id = 0;
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

    /// Reads all VALID (Committed) entries.
    /// Discards any partial transaction at the end.
    pub fn read_committed(mut self) -> Vec<WalEntry> {
        let mut committed_entries = Vec::new();
        let mut pending_entries = Vec::new();
        let mut active_tx_id: Option<u64> = None;

        loop {
            // 1. Read Frame
            let crc_expected = match self.reader.read_u32::<LittleEndian>() {
                Ok(c) => c,
                Err(_) => break, // EOF
            };
            let len = match self.reader.read_u32::<LittleEndian>() {
                Ok(l) => l as usize,
                Err(_) => break, // Truncated header
            };
            let mut payload = vec![0u8; len];
            if self.reader.read_exact(&mut payload).is_err() {
                break; // Truncated body
            }

            // 2. Checksum
            let mut hasher = Hasher::new();
            hasher.update(&payload);
            if hasher.finalize() != crc_expected {
                break; // Corruption
            }

            // 3. Parse & State Machine
            if let Ok(entry) = Self::deserialize_entry(&payload) {
                match entry {
                    WalEntry::Begin { tx_id } => {
                        // If we were already in a transaction that didn't commit, discard it
                        if active_tx_id.is_some() {
                            // Warn: Discarding previous incomplete transaction
                            pending_entries.clear();
                        }
                        active_tx_id = Some(tx_id);
                    }
                    WalEntry::Commit { tx_id } => {
                        if let Some(active) = active_tx_id {
                            if active == tx_id {
                                // ⚡ COMMIT: Move pending to finalized
                                committed_entries.append(&mut pending_entries);
                                active_tx_id = None;
                            }
                        }
                        // Ignore commit if no matching begin (or mismatch ID)
                    }
                    op => {
                        if active_tx_id.is_some() {
                            pending_entries.push(op);
                        } else {
                            // If it's a standalone op (legacy or auto-commit), treat as committed immediately
                            // OR treat as error. For safety, let's treat non-transactional ops as committed
                            // to support backward compatibility or simple inserts.
                            committed_entries.push(op);
                        }
                    }
                }
            }
        }

        // Note: Any items left in `pending_entries` here are discarded (incomplete tx).
        committed_entries
    }

    fn deserialize_entry(data: &[u8]) -> io::Result<WalEntry> {
        let mut cursor = std::io::Cursor::new(data);
        let opcode = cursor.read_u8()?;

        match opcode {
            OP_BEGIN => {
                let tx_id = cursor.read_u64::<LittleEndian>()?;
                Ok(WalEntry::Begin { tx_id })
            }
            OP_COMMIT => {
                let tx_id = cursor.read_u64::<LittleEndian>()?;
                Ok(WalEntry::Commit { tx_id })
            }
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
