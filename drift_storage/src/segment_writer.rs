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
    pub offset: u64,     // Start of Vector data
    pub length: u64,     // Length of Vector data
    pub ids_offset: u64, // NEW: Start of ID column
    pub ids_length: u64, // NEW: Length of ID column
    pub vector_count: usize,
    pub checksum: u32,
    pub payload_offset: u64,
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

    pub async fn write_bucket(
        &mut self,
        bucket_id: u32,
        ids: &[u64],
        vectors: &[Vec<f32>],
    ) -> io::Result<()> {
        if vectors.is_empty() {
            return Ok(());
        }
        if ids.len() != vectors.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "IDs/Vectors mismatch",
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

        // 2. Bloom Filter
        for &id in ids {
            self.bloom.insert(&id.to_le_bytes());
        }

        // 3. Write Vectors (Compressed)
        let dim = vectors[0].len();
        let count = vectors.len();
        let vectors_start = self.current_offset;
        let mut vec_hasher = Hasher::new();

        let columns = transpose(vectors, dim);
        for col_floats in columns {
            let compressed = CompressedColumn::compress(&col_floats, CompressionStrategy::AlpRd);
            let len = compressed.data.len() as u32;
            let len_bytes = len.to_le_bytes();

            vec_hasher.update(&len_bytes);
            self.scratch_file.write_all(&len_bytes).await?;
            self.current_offset += 4;

            self.write_blob(&compressed.data, &mut vec_hasher).await?;
        }
        let vectors_len = self.current_offset - vectors_start;

        // 4. Index Entry
        self.index.buckets.insert(
            bucket_id,
            BucketLocation {
                offset: vectors_start,
                length: vectors_len,
                ids_offset: ids_start,
                ids_length: ids_len,
                vector_count: count,
                checksum: vec_hasher.finalize(),
                payload_offset: 0,
            },
        );

        Ok(())
    }

    pub async fn finalize(mut self) -> io::Result<()> {
        let config = bincode::config::standard();

        // 1. Write Index
        let index_offset = self.current_offset;
        let index_bytes = bincode::serde::encode_to_vec(&self.index, config)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let mut hasher = Hasher::new(); // unused for index but keeps pattern
        self.write_blob(&index_bytes, &mut hasher).await?;
        let index_len = self.current_offset - index_offset;

        // 2. Write Quantizer
        let q_offset = self.current_offset;
        let q_bytes_clone = self.quantizer_bytes.clone();
        self.scratch_file.write_all(&q_bytes_clone).await?;
        self.current_offset += q_bytes_clone.len() as u64;
        let q_len = self.current_offset - q_offset;

        // 3. Write Bloom
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

        // Rewind
        self.scratch_file.seek(SeekFrom::Start(0)).await?;

        // Read fully into memory (Note: Fine for segments < 200MB. For larger, we need stream upload)
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
