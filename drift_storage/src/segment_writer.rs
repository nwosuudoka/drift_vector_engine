use crate::DriftFooter;
use crate::compression::wrapper::{CompressedColumn, CompressionStrategy, transpose};
use crate::disk_manager::DiskManager;
use crc32fast::Hasher;
use drift_core::bucket::BucketData;
use fastbloom::BloomFilter;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{self, SeekFrom};
use tempfile::NamedTempFile;
use tokio::fs::{File, OpenOptions};
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};

#[derive(Serialize, Deserialize, Debug)]
pub struct SegmentIndex {
    pub buckets: HashMap<u32, BucketLocation>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct BucketLocation {
    // ⚡ HOT PATH: Start of the BucketData blob.
    pub index_offset: u64,
    pub index_length: u64,

    // 💿 COLD PATH: ALP Data
    pub data_offset: u64,
    pub data_length: u64,

    // Metadata
    pub vector_count: usize,
    pub compression_type: u8,
}

pub struct SegmentWriter {
    manager: DiskManager,
    scratch_file: File,
    current_offset: u64,
    index: SegmentIndex,
    quantizer_bytes: Vec<u8>,
    bloom: BloomFilter,
    _temp_handle: NamedTempFile,
}

impl SegmentWriter {
    pub async fn new(manager: DiskManager, quantizer_bytes: Vec<u8>) -> io::Result<Self> {
        let temp = NamedTempFile::new()?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(temp.path())
            .await?;

        // Standard size: 10M bits for ~1M items
        let bloom = BloomFilter::with_num_bits(10_000_000).expected_items(1_000_000);

        Ok(Self {
            manager,
            scratch_file: file,
            current_offset: 0,
            index: SegmentIndex {
                buckets: HashMap::new(),
            },
            quantizer_bytes,
            bloom,
            _temp_handle: temp,
        })
    }

    /// ⚡ Writes a single partition (Bucket) to the segment.
    pub async fn write_partition(
        &mut self,
        bucket_id: u32,
        bucket: &BucketData,
        raw_vectors: &[Vec<f32>],
        dim: usize,
    ) -> io::Result<()> {
        // --- 1. Write Hot Index (The Fat Blob) ---
        let idx_start = self.current_offset;
        let mut idx_hasher = Hasher::new();

        // Serialize the FULL bucket structure (Magic + Metadata + Codes + IDs + Tombstones)
        let bucket_bytes = bucket.to_bytes(dim)?;

        self.write_blob(&bucket_bytes, &mut idx_hasher).await?;
        let idx_len = self.current_offset - idx_start;

        // Update Bloom
        for &id in &bucket.vids {
            self.bloom.insert(&id.to_le_bytes());
        }

        // --- 2. Write Cold Data (ALP) ---
        let data_start = self.current_offset;
        let columns = transpose(raw_vectors, dim);
        for col_floats in columns {
            let compressed = CompressedColumn::compress(&col_floats, CompressionStrategy::AlpRd);

            // Write Length Prefix (4 bytes)
            let len = compressed.data.len() as u32;
            self.scratch_file.write_all(&len.to_le_bytes()).await?;
            self.current_offset += 4;

            // Write Payload
            self.scratch_file.write_all(&compressed.data).await?;
            self.current_offset += len as u64;
        }
        let data_len = self.current_offset - data_start;

        // --- 3. Record Location ---
        self.index.buckets.insert(
            bucket_id,
            BucketLocation {
                index_offset: idx_start,
                index_length: idx_len,
                data_offset: data_start,
                data_length: data_len,
                vector_count: bucket.vids.len(),
                compression_type: 1, // ALP
            },
        );

        Ok(())
    }

    /// Finalizes the segment: Index -> Metadata -> Footer -> Upload.
    pub async fn finalize(mut self) -> io::Result<HashMap<u32, BucketLocation>> {
        let config = bincode::config::standard();

        // 1. Write Segment Index
        let index_off = self.current_offset;
        let index_bytes = bincode::serde::encode_to_vec(&self.index, config)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        self.write_blob(&index_bytes, &mut Hasher::new()).await?;
        let index_len = self.current_offset - index_off;

        // 2. Write Quantizer
        let q_off = self.current_offset;
        self.scratch_file.write_all(&self.quantizer_bytes).await?;
        self.current_offset += self.quantizer_bytes.len() as u64;
        let q_len = self.current_offset - q_off;

        // 3. Write Bloom Filter
        let b_off = self.current_offset;
        let bloom_bytes = bincode::serde::encode_to_vec(&self.bloom, config)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        self.write_blob(&bloom_bytes, &mut Hasher::new()).await?;
        let b_len = self.current_offset - b_off;

        // 4. Write Footer (Using Canonical Struct)
        let footer = DriftFooter::new(index_off, index_len, b_off, b_len, q_off, q_len);
        self.scratch_file.write_all(&footer.to_bytes()).await?;

        // 5. Upload
        self.scratch_file.flush().await?;
        self.scratch_file.seek(SeekFrom::Start(0)).await?;

        let mut buffer = Vec::new();
        self.scratch_file.read_to_end(&mut buffer).await?;
        self.manager.upload(buffer).await?;

        Ok(self.index.buckets)
    }

    async fn write_blob(&mut self, data: &[u8], hasher: &mut Hasher) -> io::Result<()> {
        hasher.update(data);
        self.scratch_file.write_all(data).await?;
        self.current_offset += data.len() as u64;
        Ok(())
    }

    pub fn prepare_bucket_data(
        _bucket_id: u32,
        bucket: &BucketData,
        raw_vectors: &[Vec<f32>],
        dim: usize,
    ) -> (Vec<u8>, Vec<u8>, usize) {
        // 1. Hot Index Bytes (SQ8) [cite: 3765, 3766]
        let index_bytes = bucket.to_bytes(dim).unwrap();

        // 2. Cold Data Bytes (ALP) - This is the CPU intensive part
        let mut data_bytes = Vec::new();
        let columns = crate::compression::wrapper::transpose(raw_vectors, dim);
        for col_floats in columns {
            let compressed = CompressedColumn::compress(
                &col_floats,
                crate::compression::wrapper::CompressionStrategy::AlpRd,
            );
            data_bytes.extend_from_slice(&(compressed.data.len() as u32).to_le_bytes());
            data_bytes.extend_from_slice(&compressed.data);
        }

        (index_bytes, data_bytes, bucket.vids.len())
    }

    pub async fn write_pre_compressed_partition(
        &mut self,
        bucket_id: u32,
        index_bytes: Vec<u8>,
        data_bytes: Vec<u8>,
        vector_count: usize,
    ) -> io::Result<()> {
        let idx_start = self.current_offset;
        self.scratch_file.write_all(&index_bytes).await?;
        self.current_offset += index_bytes.len() as u64;
        let idx_len = self.current_offset - idx_start;

        let dat_start = self.current_offset;
        self.scratch_file.write_all(&data_bytes).await?;
        self.current_offset += data_bytes.len() as u64;
        let dat_len = self.current_offset - dat_start;

        self.index.buckets.insert(
            bucket_id,
            BucketLocation {
                index_offset: idx_start,
                index_length: idx_len,
                data_offset: dat_start,
                data_length: dat_len,
                vector_count,
                compression_type: 1,
            },
        );
        Ok(())
    }
}
