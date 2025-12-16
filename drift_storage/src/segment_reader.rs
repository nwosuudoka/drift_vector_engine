use crate::compression::wrapper::{
    CompressionStrategy, decompress_vectors, transpose_from_columns,
};
use crate::disk_manager::DiskManager;
use crate::segment_writer::SegmentIndex;
use crate::{FOOTER_SIZE, MAGIC_BYTES};
use crc32fast::Hasher;
use fastbloom::BloomFilter;
use std::io::{self, Cursor, Read};
use std::path::Path;

pub struct SegmentReader {
    manager: DiskManager,
    pub index: SegmentIndex,
    pub bloom: BloomFilter,
    pub quantizer: Vec<u8>,
}

impl SegmentReader {
    pub async fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let mut manager = DiskManager::open(path).await?;
        let file_len = manager.file().metadata().await?.len();

        if file_len < FOOTER_SIZE as u64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "File too small/truncated",
            ));
        }

        // 1. Read Footer (Fixed 64 bytes)
        let footer_pos = file_len - FOOTER_SIZE as u64;
        manager.seek(footer_pos).await?;
        let footer_bytes = manager.read_exact(FOOTER_SIZE).await?;

        // Parse Footer
        let mut cursor = Cursor::new(&footer_bytes);
        use byteorder::{LittleEndian, ReadBytesExt};

        let version = cursor.read_u8()?;
        if version != 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Unsupported segment version",
            ));
        }

        let index_offset = cursor.read_u64::<LittleEndian>()?;
        let index_len = cursor.read_u64::<LittleEndian>()?;
        let bloom_offset = cursor.read_u64::<LittleEndian>()?;
        let bloom_len = cursor.read_u64::<LittleEndian>()?;
        let quantizer_offset = cursor.read_u64::<LittleEndian>()?;
        let quantizer_len = cursor.read_u64::<LittleEndian>()?;

        // Skip padding (7 bytes) to reach Magic
        cursor.set_position(cursor.position() + 7);

        let mut magic = [0u8; 8];
        cursor.read_exact(&mut magic)?;

        if magic != *MAGIC_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid magic bytes",
            ));
        }

        // 2. Read Index
        manager.seek(index_offset).await?;
        let index_bytes = manager.read_exact(index_len as usize).await?;
        let config = bincode::config::standard();
        let index: SegmentIndex = bincode::serde::decode_from_slice(&index_bytes, config)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?
            .0;

        // 3. Read Bloom Filter
        manager.seek(bloom_offset).await?;
        let bloom_bytes = manager.read_exact(bloom_len as usize).await?;
        let bloom: BloomFilter = bincode::serde::decode_from_slice(&bloom_bytes, config)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?
            .0;

        // 4. Read Quantizer
        manager.seek(quantizer_offset).await?;
        let quantizer = manager.read_exact(quantizer_len as usize).await?;

        Ok(Self {
            manager,
            index,
            bloom,
            quantizer,
        })
    }

    pub fn might_contain(&self, id: u64) -> bool {
        self.bloom.contains(&id.to_le_bytes())
    }

    pub fn read_metadata(&self) -> &[u8] {
        &self.quantizer
    }

    /// Reads both IDs and Vectors from a specific bucket.
    pub async fn read_bucket(&mut self, bucket_id: u32) -> io::Result<(Vec<u64>, Vec<Vec<f32>>)> {
        let loc = self
            .index
            .buckets
            .get(&bucket_id)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "Bucket not found"))?;

        // 1. Read IDs Column (NEW)
        self.manager.seek(loc.ids_offset).await?;
        let ids_bytes = self.manager.read_exact(loc.ids_length as usize).await?;
        let (ids, _): (Vec<u64>, usize) =
            bincode::decode_from_slice(&ids_bytes, bincode::config::standard())
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        // 2. Read Vector Data
        self.manager.seek(loc.offset).await?;
        let blob = self.manager.read_exact(loc.length as usize).await?;

        // 3. Verify Vector Integrity (CRC32)
        let mut hasher = Hasher::new();
        hasher.update(&blob);
        if hasher.finalize() != loc.checksum {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Data corruption detected (Checksum Mismatch)",
            ));
        }

        // 4. Parse Columnar Stream
        let mut cursor = Cursor::new(&blob);
        let mut columns = Vec::new();
        let mut bytes_read = 0;

        while bytes_read < loc.length {
            let mut len_buf = [0u8; 4];
            cursor.read_exact(&mut len_buf)?;
            let col_len = u32::from_le_bytes(len_buf) as usize;
            bytes_read += 4;

            let mut col_data = vec![0u8; col_len];
            cursor.read_exact(&mut col_data)?;
            bytes_read += col_len as u64;

            let floats =
                decompress_vectors(&col_data, loc.vector_count, CompressionStrategy::AlpRd);
            columns.push(floats);
        }

        // 5. Inverse Transpose
        let vectors = transpose_from_columns(&columns, loc.vector_count);

        Ok((ids, vectors))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::disk_manager::DiskManager;
    use crate::segment_writer::SegmentWriter;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_random_access_correctness() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("multi_bucket.drift");

        let vec_a = vec![vec![1.0], vec![1.1]];
        let ids_a = vec![10, 11];

        let vec_b = vec![vec![2.0], vec![2.2]];
        let ids_b = vec![20, 21];

        let meta = vec![0xAA, 0xBB];

        {
            let manager = DiskManager::open(&path).await.unwrap();
            let mut writer = SegmentWriter::new(manager, meta.clone());
            writer.write_bucket(10, &ids_a, &vec_a).await.unwrap();
            writer.write_bucket(20, &ids_b, &vec_b).await.unwrap();
            writer.finalize().await.unwrap();
        }

        let mut reader = SegmentReader::open(&path).await.unwrap();

        assert_eq!(reader.read_metadata(), &meta);
        assert!(reader.might_contain(10));
        assert!(reader.might_contain(21));

        // Verify ID and Vector pairs match
        let (read_ids_b, read_vec_b) = reader.read_bucket(20).await.unwrap();
        assert_eq!(read_ids_b, ids_b);
        assert_eq!(read_vec_b, vec_b);

        let (read_ids_a, read_vec_a) = reader.read_bucket(10).await.unwrap();
        assert_eq!(read_ids_a, ids_a);
        assert_eq!(read_vec_a, vec_a);
    }
}
