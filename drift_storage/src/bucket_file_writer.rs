use crate::format::{DriftFooter, DriftHeader, HEADER_SIZE, ROW_GROUP_HEADER_SIZE};
use crate::row_group_writer::RowGroupWriter;
use byteorder::{LittleEndian, WriteBytesExt};
use drift_core::quantizer::Quantizer;
use fastbloom::BloomFilter;
use std::io::{self, Write};
use zerocopy::IntoBytes;

pub struct BucketFileWriter<W: Write> {
    writer: W,
    current_offset: u64,
    total_vectors: u64,
    #[allow(dead_code)]
    run_id: [u8; 16],
    quantizer: Quantizer,
    dim: usize,
    row_group_meta: Vec<crate::format::RowGroupHeader>,
    bloom: BloomFilter,
}

impl<W: Write> BucketFileWriter<W> {
    pub fn new(
        mut writer: W,
        run_id: [u8; 16],
        quantizer: Quantizer,
        dim: usize,
    ) -> io::Result<Self> {
        // Write Header
        let header = DriftHeader::new(0, run_id);
        writer.write_all(header.as_bytes())?;

        // Initialize Bloom (same as before)
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

    pub fn write_batch(&mut self, ids: &[u64], flat_vectors: &[f32]) -> io::Result<()> {
        let mut rg_writer = RowGroupWriter::new(&mut self.writer, self.current_offset);
        let rg_header =
            rg_writer.write_group(ids, flat_vectors, None, &self.quantizer, self.dim)?;

        let next_offset = rg_header.cold_offset + rg_header.cold_length as u64;
        self.current_offset = next_offset;
        self.total_vectors += ids.len() as u64;
        self.row_group_meta.push(rg_header);

        for &id in ids {
            self.bloom.insert(&id.to_le_bytes());
        }
        Ok(())
    }

    pub fn finalize(mut self) -> io::Result<u64> {
        // 1. Write Row Group Index
        let index_start_offset = self.current_offset;
        self.writer
            .write_u32::<LittleEndian>(self.row_group_meta.len() as u32)?;
        self.current_offset += 4;

        for meta in &self.row_group_meta {
            self.writer.write_all(meta.as_bytes())?;
            self.current_offset += ROW_GROUP_HEADER_SIZE as u64;
        }

        // 2. Write Bloom Filter
        let bloom_start_offset = self.current_offset;
        let bloom_bytes = bincode::serde::encode_to_vec(&self.bloom, bincode::config::standard())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        self.writer.write_all(&bloom_bytes)?;
        self.current_offset += bloom_bytes.len() as u64;

        // 3. Write Quantizer (NEW)
        let quant_start_offset = self.current_offset;
        let quant_bytes = bincode::encode_to_vec(&self.quantizer, bincode::config::standard())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        self.writer.write_all(&quant_bytes)?;
        self.current_offset += quant_bytes.len() as u64;

        // 4. Write Footer (Updated with Quantizer offsets)
        let footer = DriftFooter::new(
            self.row_group_meta.len() as u32,
            index_start_offset,
            bloom_start_offset,
            bloom_bytes.len() as u32,
            quant_start_offset,       // NEW
            quant_bytes.len() as u32, // NEW
        );

        self.writer.write_all(footer.as_bytes())?;
        self.current_offset += crate::format::FOOTER_SIZE as u64;

        self.writer.flush()?;
        Ok(self.current_offset)
    }
}
