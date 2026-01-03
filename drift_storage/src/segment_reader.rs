use crate::compression::wrapper::{
    CompressionStrategy, decompress_vectors, transpose_from_columns,
};
use crate::disk_manager::DiskManager;
use crate::segment_writer::SegmentIndex;
use crate::{DriftFooter, FOOTER_SIZE, MAGIC_BYTES};
use drift_core::bucket::BucketData;
use drift_traits::Cacheable;
use fastbloom::BloomFilter;
use opendal::Operator;
use std::io::{self, Cursor, Read};

pub struct SegmentReader {
    manager: DiskManager,
    pub index: SegmentIndex,
    pub bloom: BloomFilter,
    pub quantizer: Vec<u8>,
}

impl SegmentReader {
    pub async fn open_with_op(op: Operator, path: &str) -> io::Result<Self> {
        let manager = DiskManager::new(op, path.to_string());

        // 1. Check Size
        let file_len = manager.len().await?;

        // ⚡ FIX: Explicit check preventing underflow in subtraction
        if file_len < FOOTER_SIZE as u64 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "File too small"));
        }

        // 2. Read Footer
        let footer_pos = file_len - FOOTER_SIZE as u64;
        let footer_bytes = manager.read_at(footer_pos, FOOTER_SIZE).await?;

        // Use shared struct to parse
        let footer =
            DriftFooter::from_bytes(&footer_bytes.try_into().map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidData, "Footer size mismatch")
            })?)?;

        // Validation
        if footer.magic != *MAGIC_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid Magic Bytes",
            ));
        }
        if footer.version != 1 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Bad Version"));
        }

        // 3. Read Index
        let index_bytes = manager
            .read_at(footer.index_offset, footer.index_length as usize)
            .await?;
        let index: SegmentIndex =
            bincode::serde::decode_from_slice(&index_bytes, bincode::config::standard())
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?
                .0;

        // 4. Read Bloom
        let bloom_bytes = manager
            .read_at(footer.bloom_offset, footer.bloom_length as usize)
            .await?;
        let bloom: BloomFilter =
            bincode::serde::decode_from_slice(&bloom_bytes, bincode::config::standard())
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?
                .0;

        // 5. Read Quantizer
        let quantizer = manager
            .read_at(footer.quantizer_offset, footer.quantizer_length as usize)
            .await?;

        Ok(Self {
            manager,
            index,
            bloom,
            quantizer,
        })
    }

    /// ⚡ FAST PATH: Reads Hot Index Blob.
    pub async fn read_bucket(&self, bucket_id: u32) -> io::Result<(Vec<u64>, Vec<u8>)> {
        let loc = self
            .index
            .buckets
            .get(&bucket_id)
            .ok_or(io::Error::new(io::ErrorKind::NotFound, "Bucket not found"))?;

        // 1. Read the FULL BucketData blob (Magic + Metadata + Codes + IDs)
        let blob = self
            .manager
            .read_at(loc.index_offset, loc.index_length as usize)
            .await?;

        // 2. Parse (Checks 0xBD47001 Magic)
        let bucket = BucketData::from_bytes(&blob)?;

        // 3. Return IDs and Codes
        let codes = bucket.codes.as_slice().to_vec();
        Ok((bucket.vids, codes))
    }

    /// 💿 COLD PATH: Reads High-Fidelity Float Vectors.
    pub async fn read_bucket_high_fidelity(&self, bucket_id: u32) -> io::Result<Vec<Vec<f32>>> {
        let loc = self
            .index
            .buckets
            .get(&bucket_id)
            .ok_or(io::Error::new(io::ErrorKind::NotFound, "Bucket not found"))?;

        let blob = self
            .manager
            .read_at(loc.data_offset, loc.data_length as usize)
            .await?;
        let mut cursor = Cursor::new(&blob);
        let mut columns = Vec::new();
        let mut bytes_read = 0;

        while bytes_read < loc.data_length {
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

        Ok(transpose_from_columns(&columns, loc.vector_count))
    }

    pub fn read_metadata(&self) -> &[u8] {
        &self.quantizer
    }

    pub fn might_contain(&self, id: u64) -> bool {
        self.bloom.contains(&id.to_le_bytes())
    }
}
