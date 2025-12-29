use crate::DriftFooter;
use crate::compression::wrapper::{CompressedColumn, CompressionStrategy, transpose};
use crate::disk_manager::DiskManager;
use crc32fast::Hasher;
use fastbloom::BloomFilter;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{self, SeekFrom};
use tempfile::NamedTempFile;
use tokio::fs::File;
use tokio::fs::OpenOptions;
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};

// --- Structs ---

#[derive(Serialize, Deserialize, Debug)]
pub struct SegmentIndex {
    pub buckets: HashMap<u32, BucketLocation>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct BucketLocation {
    // ⚡ FAST PATH (SQ8 Index)
    pub index_offset: u64,
    pub index_length: u64, // Raw SQ8 bytes [N * Dim]

    // 💿 COLD PATH (ALP Data)
    pub data_offset: u64,
    pub data_length: u64, // ALP Compressed Columns

    // METADATA
    pub ids_offset: u64,
    pub ids_length: u64,
    pub vector_count: usize,
    pub checksum: u32,

    // Flags
    pub compression_type: u8, // 0=None, 1=ALP (Reserved for data blob context)
}

#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum CompressionType {
    RawSQ8 = 0,
    Compressed = 1,
}

// --- The Writer ---

pub struct SegmentWriter {
    manager: DiskManager, // Destination (S3/File)
    scratch_file: File,   // Local NVMe/RAM Buffer
    current_offset: u64,
    index: SegmentIndex,
    quantizer_bytes: Vec<u8>,
    bloom: BloomFilter,
    // We hold the temp_handle to prevent early deletion
    _temp_handle: NamedTempFile,
}

impl SegmentWriter {
    pub async fn new(manager: DiskManager, quantizer_bytes: Vec<u8>) -> io::Result<Self> {
        // 1. Create Local Scratch File
        // This is synchronous IO, but creating a tempfile is fast.
        let temp = NamedTempFile::new()?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(temp.path())
            .await?;

        // 10M bits ~ 1.2MB ram for 1M items with low false positive rate
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

    /// ⚡ PRIMARY WRITE PATH: Writes both Index (SQ8) and Data (ALP).
    /// Used during L0 -> L1 Flush where we have high-fidelity floats.
    pub async fn write_bucket_dual(
        &mut self,
        bucket_id: u32,
        ids: &[u64],
        raw_vectors: &[Vec<f32>], // Raw floats (for ALP)
        sq8_codes: &[u8],         // Pre-quantized SQ8 (for Index)
        dim: usize,
    ) -> io::Result<()> {
        if raw_vectors.is_empty() {
            return Ok(());
        }
        if ids.len() != raw_vectors.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "IDs/Vectors mismatch",
            ));
        }
        // Basic collision check
        if self.index.buckets.contains_key(&bucket_id) {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "Bucket ID collision",
            ));
        }

        // --- 1. Write IDs ---
        let ids_start = self.current_offset;
        let mut ids_hasher = Hasher::new();
        let id_bytes = bincode::encode_to_vec(ids, bincode::config::standard())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        self.write_blob(&id_bytes, &mut ids_hasher).await?;
        let ids_len = self.current_offset - ids_start;

        // Update Bloom Filter
        for &id in ids {
            self.bloom.insert(&id.to_le_bytes());
        }

        // --- 2. Write SQ8 Blob (The Fast Index) ---
        let idx_start = self.current_offset;
        // We use a hasher to ensure integrity of the index blob
        let mut idx_hasher = Hasher::new();

        self.write_blob(sq8_codes, &mut idx_hasher).await?;
        let idx_len = self.current_offset - idx_start;

        // --- 3. Write ALP Blob (The High-Fidelity Data) ---
        let data_start = self.current_offset;
        // Transpose Row-Major Vectors -> Column-Major for compression [cite: 2623]
        let columns = transpose(raw_vectors, dim);

        for col_floats in columns {
            // Compress using ALP_RD (Real Doubles) strategy [cite: 2618]
            let compressed = CompressedColumn::compress(&col_floats, CompressionStrategy::AlpRd);

            // Format: [Length (4b)] [Data (...)]
            let len = compressed.data.len() as u32;
            let len_bytes = len.to_le_bytes();

            // Note: We don't hash the data blob in the primary checksum to save CPU,
            // or we can use a separate hasher. For now, we rely on ALP internal integrity.
            self.scratch_file.write_all(&len_bytes).await?;
            self.current_offset += 4;

            // We reuse idx_hasher just to write_blob, but we don't use its result for data
            // (Function writes to disk + updates offset)
            self.scratch_file.write_all(&compressed.data).await?;
            self.current_offset += compressed.data.len() as u64;
        }
        let data_len = self.current_offset - data_start;

        // --- 4. Register Location ---
        self.index.buckets.insert(
            bucket_id,
            BucketLocation {
                // SQ8 Index
                index_offset: idx_start,
                index_length: idx_len,

                // ALP Data
                data_offset: data_start,
                data_length: data_len,

                // Metadata
                ids_offset: ids_start,
                ids_length: ids_len,
                vector_count: raw_vectors.len(),
                checksum: idx_hasher.finalize(), // Checksum of the SQ8 index primarily

                // Flags
                compression_type: CompressionType::Compressed as u8,
            },
        );

        Ok(())
    }

    /// ⚡ FAST MAINTENANCE PATH: Write ONLY SQ8 bytes.
    /// Used when merging/splitting L1 buckets where we don't want to re-compress ALP data immediately,
    /// or if we are dropping high-fidelity data to save space (tiering).
    pub async fn write_bucket_sq8(
        &mut self,
        bucket_id: u32,
        ids: &[u64],
        codes: &[u8],
        dim: usize,
    ) -> io::Result<()> {
        let count = ids.len();
        if codes.len() != count * dim {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Dimension mismatch",
            ));
        }
        if self.index.buckets.contains_key(&bucket_id) {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "Bucket ID collision",
            ));
        }

        // 1. Write IDs
        let ids_start = self.current_offset;
        let mut ids_hasher = Hasher::new();
        let id_bytes = bincode::encode_to_vec(ids, bincode::config::standard())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        self.write_blob(&id_bytes, &mut ids_hasher).await?;
        let ids_len = self.current_offset - ids_start;

        // 2. Update Bloom
        for &id in ids {
            self.bloom.insert(&id.to_le_bytes());
        }

        // 3. Write SQ8 Blob
        let idx_start = self.current_offset;
        let mut idx_hasher = Hasher::new();
        self.write_blob(codes, &mut idx_hasher).await?;
        let idx_len = self.current_offset - idx_start;

        // 4. Register (No Data Blob)
        self.index.buckets.insert(
            bucket_id,
            BucketLocation {
                index_offset: idx_start,
                index_length: idx_len,

                // No High Fidelity Data
                data_offset: 0,
                data_length: 0,

                ids_offset: ids_start,
                ids_length: ids_len,
                vector_count: count,
                checksum: idx_hasher.finalize(),
                compression_type: CompressionType::RawSQ8 as u8,
            },
        );

        Ok(())
    }

    pub async fn finalize(mut self) -> io::Result<()> {
        let config = bincode::config::standard();

        // 1. Write Index Metadata
        let index_offset = self.current_offset;
        let index_bytes = bincode::serde::encode_to_vec(&self.index, config)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let mut hasher = Hasher::new();
        self.write_blob(&index_bytes, &mut hasher).await?;
        let index_len = self.current_offset - index_offset;

        // 2. Write Quantizer
        let q_offset = self.current_offset;
        let q_bytes_clone = self.quantizer_bytes.clone();
        self.scratch_file.write_all(&q_bytes_clone).await?;
        self.current_offset += q_bytes_clone.len() as u64;
        let q_len = self.current_offset - q_offset;

        // 3. Write Bloom Filter
        let b_offset = self.current_offset;
        let bloom_bytes = bincode::serde::encode_to_vec(&self.bloom, config)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let mut b_hasher = Hasher::new();
        self.write_blob(&bloom_bytes, &mut b_hasher).await?;
        let b_len = self.current_offset - b_offset;

        // 4. Write Footer
        let footer = DriftFooter::new(index_offset, index_len, b_offset, b_len, q_offset, q_len);
        let footer_bytes = footer.to_bytes();
        self.scratch_file.write_all(&footer_bytes).await?;

        // 5. UPLOAD PHASE
        self.scratch_file.sync_all().await?;

        // Rewind to start
        self.scratch_file.seek(SeekFrom::Start(0)).await?;

        // Read fully into memory
        // (Note: For >200MB segments, we should implement streaming upload via OpenDAL Writer)
        let mut buffer = Vec::new();
        self.scratch_file.read_to_end(&mut buffer).await?;

        // Atomic Upload via OpenDAL
        self.manager.upload(buffer).await?;

        Ok(())
    }

    async fn write_blob(&mut self, data: &[u8], hasher: &mut Hasher) -> io::Result<()> {
        hasher.update(data);
        self.scratch_file.write_all(data).await?;
        self.current_offset += data.len() as u64;
        Ok(())
    }
}
