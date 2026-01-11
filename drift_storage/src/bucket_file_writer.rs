use crate::format::{
    DriftFooter, DriftHeader, FOOTER_SIZE, HEADER_SIZE, MAGIC_V2, ROW_GROUP_HEADER_SIZE,
    RowGroupHeader,
};
use crate::row_group_writer::RowGroupWriter;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use drift_core::quantizer::Quantizer;
use fastbloom::BloomFilter;
use std::io::{self, Read, Seek, SeekFrom, Write};
use zerocopy::{FromBytes, IntoBytes};

/// Capability trait for writers that support truncation (Local Files).
/// S3 Streams will NOT implement this.
pub trait Truncatable {
    fn set_len(&mut self, size: u64) -> io::Result<()>;
}

// Implement for File
impl Truncatable for std::fs::File {
    fn set_len(&mut self, size: u64) -> io::Result<()> {
        std::fs::File::set_len(self, size)
    }
}

// Implement for Cursor (Test)
impl Truncatable for io::Cursor<Vec<u8>> {
    fn set_len(&mut self, size: u64) -> io::Result<()> {
        self.get_mut().resize(size as usize, 0);
        Ok(())
    }
}

// This tells Rust: "If T implements Truncatable,
// then a mutable reference (&mut T) also implements it."
impl<T: Truncatable + ?Sized> Truncatable for &mut T {
    fn set_len(&mut self, size: u64) -> io::Result<()> {
        // Dereference twice:
        // 1. &mut T -> T (to call the method on the object)
        (**self).set_len(size)
    }
}

/// A wrapper to allow purely streaming writers (No Seek/Read/Truncate)
/// This is what we will use for S3.
pub struct StreamWriter<W: Write>(pub W);

impl<W: Write> Write for StreamWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.write(buf)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.0.flush()
    }
}

// -----------------------------------------------------------------------------

pub struct BucketFileWriter<W> {
    writer: W,
    current_offset: u64,
    #[allow(dead_code)]
    run_id: [u8; 16],
    quantizer: Quantizer,
    dim: usize,
    row_group_meta: Vec<RowGroupHeader>,
    bloom: BloomFilter,

    // State
    append_mode: bool,
}

// -----------------------------------------------------------------------------
// BLOCK 1: Generic Implementation (Works for S3 Stream & Local File)
// -----------------------------------------------------------------------------
impl<W: Write> BucketFileWriter<W> {
    /// Creates a writer for a NEW file (Streaming Mode).
    /// Does NOT require Seek/Read. Suitable for S3.
    pub fn new_streaming(
        mut writer: W,
        run_id: [u8; 16],
        quantizer: Quantizer,
        dim: usize,
    ) -> io::Result<Self> {
        // 1. Serialize Quantizer
        let q_bytes = bincode::encode_to_vec(&quantizer, bincode::config::standard())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        // 2. Write Header (Includes Quantizer Length)
        let header = DriftHeader::new(0, run_id, HEADER_SIZE as u64, q_bytes.len() as u32);
        writer.write_all(header.as_bytes())?;

        // 3. Write Quantizer Blob
        writer.write_all(&q_bytes)?;

        let current_offset = HEADER_SIZE as u64 + q_bytes.len() as u64;

        Ok(Self {
            writer,
            current_offset,
            run_id,
            quantizer,
            dim,
            row_group_meta: Vec::new(),
            bloom: BloomFilter::with_num_bits(8_000_000).expected_items(100_000),
            append_mode: false,
        })
    }

    pub fn write_batch(&mut self, ids: &[u64], flat_vectors: &[f32]) -> io::Result<()> {
        let mut rg_writer = RowGroupWriter::new(&mut self.writer, self.current_offset);
        let rg_header =
            rg_writer.write_group(ids, flat_vectors, None, &self.quantizer, self.dim)?;

        let next_offset = rg_header.cold_offset + rg_header.cold_length as u64;
        self.current_offset = next_offset;
        self.row_group_meta.push(rg_header);

        for &id in ids {
            self.bloom.insert(&id.to_le_bytes());
        }
        Ok(())
    }

    /// Finalizes the file.
    /// If W implements `Truncatable`, it will truncate (safe for Append).
    /// If not, it just flushes (safe for Stream).
    pub fn finish(&mut self) -> io::Result<u64> {
        // 1. Write Index
        let index_start = self.current_offset;
        self.writer
            .write_u32::<LittleEndian>(self.row_group_meta.len() as u32)?;
        self.current_offset += 4;
        for meta in &self.row_group_meta {
            self.writer.write_all(meta.as_bytes())?;
            self.current_offset += ROW_GROUP_HEADER_SIZE as u64;
        }

        // 2. Write Bloom
        let bloom_start = self.current_offset;
        let bloom_bytes = bincode::serde::encode_to_vec(&self.bloom, bincode::config::standard())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        self.writer.write_all(&bloom_bytes)?;
        self.current_offset += bloom_bytes.len() as u64;

        // 3. Write Footer
        let footer = DriftFooter::new(
            self.row_group_meta.len() as u32,
            index_start,
            bloom_start,
            bloom_bytes.len() as u32,
        );
        self.writer.write_all(footer.as_bytes())?;
        self.current_offset += FOOTER_SIZE as u64;

        self.writer.flush()?;

        Ok(self.current_offset)
    }

    /// Finalizes the file.
    /// If W implements `Truncatable`, it will truncate (safe for Append).
    /// If not, it just flushes (safe for Stream).
    pub fn finalize(mut self) -> io::Result<u64> {
        self.finish()
    }
}

// -----------------------------------------------------------------------------
// BLOCK 2: Append Implementation (Requires Seek + Read + Truncatable)
// -----------------------------------------------------------------------------
impl<W: Write + Seek + Read + Truncatable> BucketFileWriter<W> {
    /// Creates a writer for EXISTING files (Append Mode).
    /// Requires Seek/Read to recover state and Truncatable to ensure safety.
    pub fn new_append(
        mut writer: W,
        run_id: [u8; 16],
        quantizer: Quantizer,
        dim: usize,
        initial_file_len: u64,
    ) -> io::Result<Self> {
        if initial_file_len < FOOTER_SIZE as u64 {
            // Fallback to new file logic if empty
            // But 'new_streaming' requires passing ownership, tricky here.
            // Just treat as error for explicit append mode.
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "File too small to append",
            ));
        }

        // 1. Seek & Recover State (Same logic as before)
        let footer_start = initial_file_len - FOOTER_SIZE as u64;
        writer.seek(SeekFrom::Start(footer_start))?;

        let mut footer_bytes = [0u8; FOOTER_SIZE];
        writer.read_exact(&mut footer_bytes)?;
        let footer = DriftFooter::read_from_bytes(&footer_bytes)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid Footer"))?;

        if footer.magic != MAGIC_V2 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Bad Magic"));
        }

        // Recover Index
        writer.seek(SeekFrom::Start(footer.index_start_offset))?;
        let rg_count = writer.read_u32::<LittleEndian>()? as usize;

        let mut row_group_meta = Vec::with_capacity(rg_count);
        let mut rg_buf = [0u8; ROW_GROUP_HEADER_SIZE];
        for _ in 0..rg_count {
            writer.read_exact(&mut rg_buf)?;
            let rg = RowGroupHeader::read_from_bytes(&rg_buf)
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Bad RG"))?;
            row_group_meta.push(rg);
        }

        // Recover Bloom
        writer.seek(SeekFrom::Start(footer.bloom_filter_offset))?;
        let mut bloom_bytes = vec![0u8; footer.bloom_filter_length as usize];
        writer.read_exact(&mut bloom_bytes)?;
        let (bloom, _): (BloomFilter, usize) =
            bincode::serde::decode_from_slice(&bloom_bytes, bincode::config::standard())
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        // 2. Prepare for Overwrite
        let current_offset = footer.index_start_offset;
        writer.seek(SeekFrom::Start(current_offset))?;

        Ok(Self {
            writer,
            current_offset,
            run_id,
            quantizer,
            dim,
            row_group_meta,
            bloom,
            append_mode: true,
        })
    }

    /// Helper to finalize with truncation safely
    pub fn finalize_and_truncate(mut self) -> io::Result<u64> {
        let final_len = self.finish()?; // Calls the generic finalize

        // Safety: Only perform truncation if we were explicitly appending
        if self.append_mode {
            self.writer.set_len(final_len)?;
        }

        Ok(final_len)
    }
}
