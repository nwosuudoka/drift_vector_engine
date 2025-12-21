use crate::DriftFooter;
use crate::compression::wrapper::{CompressedColumn, CompressionStrategy, transpose};
use crate::disk_manager::DiskManager;
use crc32fast::Hasher;
use fastbloom::BloomFilter;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io;

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
    manager: DiskManager,
    current_offset: u64,
    index: SegmentIndex,
    quantizer_bytes: Vec<u8>,
    bloom: BloomFilter,
}

impl SegmentWriter {
    pub fn new(manager: DiskManager, quantizer_bytes: Vec<u8>) -> Self {
        let bloom = BloomFilter::with_num_bits(10_000_000).expected_items(1_000_000);

        Self {
            manager,
            current_offset: 0,
            index: SegmentIndex {
                buckets: HashMap::new(),
            },
            quantizer_bytes,
            bloom,
        }
    }

    pub async fn write_bucket(
        &mut self,
        bucket_id: u32,
        ids: &[u64],
        vectors: &[Vec<f32>],
    ) -> std::io::Result<()> {
        if vectors.is_empty() {
            return Ok(());
        }
        if ids.len() != vectors.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "IDs and Vectors length mismatch",
            ));
        }

        if self.index.buckets.contains_key(&bucket_id) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                format!("Bucket {} already exists in this segment", bucket_id),
            ));
        }

        // 1. Write the ID Column (NEW)
        // We serialize the IDs as a block. This ensures O(1) identity retrieval.
        let ids_start = self.current_offset;
        let mut ids_hasher = Hasher::new();

        let id_bytes = bincode::encode_to_vec(ids, bincode::config::standard())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        self.write_blob(&id_bytes, &mut ids_hasher).await?;
        let ids_len = self.current_offset - ids_start;

        // 2. Add IDs to Bloom Filter
        for &id in ids {
            self.bloom.insert(&id.to_le_bytes());
        }

        // 3. Write Vector Data
        let dim = vectors[0].len();
        let count = vectors.len();
        let vectors_start = self.current_offset;
        let mut vec_hasher = Hasher::new();

        // Transpose & Compress
        let columns = transpose(vectors, dim);
        for col_floats in columns {
            let compressed = CompressedColumn::compress(&col_floats, CompressionStrategy::AlpRd);

            let len = compressed.data.len() as u32;
            let len_bytes = len.to_le_bytes();

            vec_hasher.update(&len_bytes);
            self.manager.write_raw(&len_bytes).await?;
            self.current_offset += 4;

            self.write_blob(&compressed.data, &mut vec_hasher).await?;
        }
        let vectors_len = self.current_offset - vectors_start;

        // 4. Record Index with ID column tracking
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

    pub async fn finalize(&mut self) -> std::io::Result<()> {
        let config = bincode::config::standard();

        let index_bytes = bincode::serde::encode_to_vec(&self.index, config)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let index_offset = self.current_offset;
        let mut hasher = Hasher::new();
        self.write_blob(&index_bytes, &mut hasher).await?;
        let index_len = self.current_offset - index_offset;

        let q_offset = self.current_offset;
        let q_bytes: &[u8] = &self.quantizer_bytes;
        self.manager.write_raw(q_bytes).await?;
        self.current_offset += q_bytes.len() as u64;
        let q_len = self.current_offset - q_offset;

        let b_offset = self.current_offset;
        let bloom_bytes = bincode::serde::encode_to_vec(&self.bloom, config)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let mut b_hasher = Hasher::new();
        self.write_blob(&bloom_bytes, &mut b_hasher).await?;
        let b_len = self.current_offset - b_offset;

        let footer = DriftFooter::new(index_offset, index_len, b_offset, b_len, q_offset, q_len);
        let footer_bytes = footer.to_bytes();
        self.manager.write_raw(&footer_bytes).await?;
        self.manager.sync().await?;

        Ok(())
    }

    async fn write_blob(&mut self, data: &[u8], hasher: &mut Hasher) -> std::io::Result<()> {
        hasher.update(data);
        self.manager.write_raw(data).await?;
        self.current_offset += data.len() as u64;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::FOOTER_SIZE;

    use super::*;
    use byteorder::{LittleEndian, ReadBytesExt};
    use tempfile::tempdir;
    use tokio::fs;
    use tokio::io::AsyncSeekExt;

    #[tokio::test]
    async fn test_segment_v2_bloom_integration() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("segment_v2.drift");

        // 1. Write Data
        let manager = DiskManager::open(&file_path).await.unwrap();
        let quantizer_mock = vec![1, 2, 3, 4];
        let mut writer = SegmentWriter::new(manager, quantizer_mock);

        let ids = vec![101, 102];
        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![4.0, 5.0, 6.0];

        writer.write_bucket(10, &ids, &[vec1, vec2]).await.unwrap();
        writer.finalize().await.unwrap();

        // 2. Verify Footer Structure
        let mut file = fs::File::open(&file_path).await.unwrap();
        let file_len = file.metadata().await.unwrap().len();

        // Correctly use the shared FOOTER_SIZE (64)
        let footer_pos = file_len - FOOTER_SIZE as u64;
        file.seek(tokio::io::SeekFrom::Start(footer_pos))
            .await
            .unwrap();

        let mut footer_buf = [0u8; FOOTER_SIZE];

        {
            use tokio::io::AsyncReadExt;
            file.read_exact(&mut footer_buf).await.unwrap();
        }

        // Check Magic (Last 8 bytes)
        let magic = &footer_buf[FOOTER_SIZE - 8..];
        assert_eq!(magic, crate::MAGIC_BYTES);

        // Check Version (Byte 0)
        assert_eq!(
            footer_buf[0], 1,
            "Version check failed. Did we read the correct footer size?"
        );

        // 3. Verify Bloom Filter Existence
        use std::io::Cursor;
        let mut cursor = Cursor::new(&footer_buf);
        let _ver = cursor.read_u8().unwrap();
        let _idx_off = cursor.read_u64::<LittleEndian>().unwrap();
        let _idx_len = cursor.read_u64::<LittleEndian>().unwrap();
        let bloom_off = cursor.read_u64::<LittleEndian>().unwrap();
        let bloom_len = cursor.read_u64::<LittleEndian>().unwrap();

        assert!(bloom_off > 0);
        assert!(bloom_len > 0);
    }
}
