use byteorder::{LittleEndian, WriteBytesExt};
use drift_core::quantizer::Quantizer;
use fastbloom::BloomFilter;
use std::io::{self, Write};
use zerocopy::IntoBytes;

use crate::format::DriftFooter;
use crate::format::DriftHeader;
use crate::format::{HEADER_SIZE, ROW_GROUP_HEADER_SIZE, RowGroupHeader};
use crate::row_group_writer::RowGroupWriter;

pub struct BucketFileWriter<W: Write> {
    writer: W,
    current_offset: u64,

    // Global State
    total_vectors: u64,
    run_id: [u8; 16],
    quantizer: Quantizer,
    dim: usize,

    row_group_meta: Vec<RowGroupHeader>,
    bloom: BloomFilter,
}

impl<W: Write> BucketFileWriter<W> {
    /// Starts a new Bucket File.
    /// Immediately writes the File Header.
    pub fn new(
        mut writer: W,
        run_id: [u8; 16],
        quantizer: Quantizer,
        dim: usize,
    ) -> io::Result<Self> {
        // 1. Write File Header
        // Note: total_vectors is 0 initially.
        // In Stream Mode (S3), the reader must rely on the Footer's count.
        // In Local Mode (Append), we might seek back to update this, but we design for No-Seek.
        let header = DriftHeader::new(0, run_id);
        writer.write_all(header.as_bytes())?;

        // Standard Bloom: 1M items, 1% false positive rate (approx 1MB)
        // We can tune this or make it config later.
        let bloom = BloomFilter::with_num_bits(8_000_000).expected_items(100_000);
        Ok(Self {
            writer,
            current_offset: HEADER_SIZE as u64,
            total_vectors: 0,
            run_id,
            quantizer,
            dim,
            row_group_meta: Vec::new(),
            bloom,
        })
    }

    /// Appends a batch of vectors as a new Row Group.
    pub fn write_batch(&mut self, ids: &[u64], flat_vectors: &[f32]) -> io::Result<()> {
        // 1. Delegate to RowGroupWriter
        // We pass a mutable reference to our writer.
        let mut rg_writer = RowGroupWriter::new(&mut self.writer, self.current_offset);

        let rg_header = rg_writer.write_group(
            ids,
            flat_vectors,
            None, // Tombstones (Future)
            &self.quantizer,
            self.dim,
        )?;

        // 2. Update State
        // Calculate size written: (Hot + Cold + Padding + Header)
        // We can derive it from the returned header offsets
        let next_offset = rg_header.cold_offset + rg_header.cold_length as u64;
        let _written_bytes = next_offset - self.current_offset;

        self.current_offset = next_offset;
        self.total_vectors += ids.len() as u64;
        self.row_group_meta.push(rg_header);

        // 3. Update Bloom Filter
        for &id in ids {
            self.bloom.insert(&id.to_le_bytes());
        }

        Ok(())
    }

    /// Finalizes the file by writing the Index, Bloom Filter, and Footer.
    /// Returns the total bytes written.
    pub fn finalize(mut self) -> io::Result<u64> {
        // --- 1. Write Row Group Index (The Directory) ---
        let index_start_offset = self.current_offset;

        // Format: [Count: u32] [RGHeader 0] [RGHeader 1] ...
        self.writer
            .write_u32::<LittleEndian>(self.row_group_meta.len() as u32)?;
        self.current_offset += 4;

        for meta in &self.row_group_meta {
            self.writer.write_all(meta.as_bytes())?;
            self.current_offset += ROW_GROUP_HEADER_SIZE as u64;
        }

        let _index_len = self.current_offset - index_start_offset;

        // --- 2. Write Bloom Filter ---
        let bloom_start_offset = self.current_offset;
        let bloom_bytes = bincode::serde::encode_to_vec(&self.bloom, bincode::config::standard())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        self.writer.write_all(&bloom_bytes)?;
        self.current_offset += bloom_bytes.len() as u64;

        // --- 3. Write Footer ---
        let footer = DriftFooter::new(
            self.row_group_meta.len() as u32,
            index_start_offset,
            bloom_start_offset,
            bloom_bytes.len() as u32,
        );

        self.writer.write_all(footer.as_bytes())?;
        self.current_offset += crate::format::FOOTER_SIZE as u64;

        // Ensure everything hits the disk/network
        self.writer.flush()?;

        Ok(self.current_offset)
    }
}
