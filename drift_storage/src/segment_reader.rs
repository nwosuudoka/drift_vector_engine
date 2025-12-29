use crate::disk_manager::DiskManager;
use crate::segment_writer::SegmentIndex;
use crate::{FOOTER_SIZE, MAGIC_BYTES};
use byteorder::{LittleEndian, ReadBytesExt};
use fastbloom::BloomFilter;
use std::io::{self, Cursor, Read};

// For the Cold Path (ALP Decompression)
use crate::compression::wrapper::{
    CompressionStrategy, decompress_vectors, transpose_from_columns,
};

pub struct SegmentReader {
    manager: DiskManager,
    pub index: SegmentIndex,
    pub bloom: BloomFilter,
    pub quantizer: Vec<u8>,
}

impl SegmentReader {
    /// Opens a segment via URI (s3:// or file://)
    pub async fn open(uri: &str) -> io::Result<Self> {
        let manager = DiskManager::open(uri).await?;

        // 1. Get Length (Network Call 1)
        let file_len = manager.len().await?;
        if file_len < FOOTER_SIZE as u64 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "File too small"));
        }

        // 2. Read Footer (Network Call 2 - Range Read)
        let footer_pos = file_len - FOOTER_SIZE as u64;
        let footer_bytes = manager.read_at(footer_pos, FOOTER_SIZE).await?;

        let mut cursor = Cursor::new(&footer_bytes);
        let version = cursor.read_u8()?;
        if version != 1 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Bad Version"));
        }

        let index_offset = cursor.read_u64::<LittleEndian>()?;
        let index_len = cursor.read_u64::<LittleEndian>()?;
        let bloom_offset = cursor.read_u64::<LittleEndian>()?;
        let bloom_len = cursor.read_u64::<LittleEndian>()?;
        let quantizer_offset = cursor.read_u64::<LittleEndian>()?;
        let quantizer_len = cursor.read_u64::<LittleEndian>()?;

        // Magic Check
        // Skip padding (7 bytes)
        cursor.set_position(cursor.position() + 7);
        let mut magic = [0u8; 8];
        cursor.read_exact(&mut magic)?;
        if &magic != MAGIC_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid Magic Bytes",
            ));
        }

        // 3. Read Index (Network Call 3)
        let index_bytes = manager.read_at(index_offset, index_len as usize).await?;
        let index: SegmentIndex =
            bincode::serde::decode_from_slice(&index_bytes, bincode::config::standard())
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?
                .0;

        // 4. Read Bloom (Network Call 4)
        let bloom_bytes = manager.read_at(bloom_offset, bloom_len as usize).await?;
        let bloom: BloomFilter =
            bincode::serde::decode_from_slice(&bloom_bytes, bincode::config::standard())
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?
                .0;

        // 5. Read Quantizer (Network Call 5)
        let quantizer = manager
            .read_at(quantizer_offset, quantizer_len as usize)
            .await?;

        Ok(Self {
            manager,
            index,
            bloom,
            quantizer,
        })
    }

    /// ⚡ FAST PATH: Reads IDs and Raw SQ8 Codes.
    /// Used by the Cache to populate `BucketData` efficiently.
    /// Returns: `(IDs, Raw SQ8 Bytes)`
    pub async fn read_bucket(&self, bucket_id: u32) -> io::Result<(Vec<u64>, Vec<u8>)> {
        let loc = self
            .index
            .buckets
            .get(&bucket_id)
            .ok_or(io::Error::new(io::ErrorKind::NotFound, "Bucket not found"))?;

        // 1. Read IDs
        let ids_bytes = self
            .manager
            .read_at(loc.ids_offset, loc.ids_length as usize)
            .await?;
        let (ids, _): (Vec<u64>, usize) =
            bincode::decode_from_slice(&ids_bytes, bincode::config::standard())
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        // 2. Read SQ8 Index Blob
        // This fetches the `index_offset` (Fast Tier)
        let sq8_blob = self
            .manager
            .read_at(loc.index_offset, loc.index_length as usize)
            .await?;

        Ok((ids, sq8_blob))
    }

    /// 💿 COLD PATH: Reads High-Fidelity Float Vectors.
    /// Decompresses ALP data. Used for Re-ranking or Crash Recovery if needed.
    pub async fn read_bucket_high_fidelity(&self, bucket_id: u32) -> io::Result<Vec<Vec<f32>>> {
        let loc = self
            .index
            .buckets
            .get(&bucket_id)
            .ok_or(io::Error::new(io::ErrorKind::NotFound, "Bucket not found"))?;

        if loc.data_length == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "No high-fidelity data available",
            ));
        }

        // 1. Read the ALP Blob
        let blob = self
            .manager
            .read_at(loc.data_offset, loc.data_length as usize)
            .await?;

        // 2. Decompress ALP Columns
        let mut cursor = Cursor::new(&blob);
        let mut columns = Vec::new();
        let mut bytes_read = 0;

        while bytes_read < loc.data_length {
            // Read Length Prefix (u32)
            let mut len_buf = [0u8; 4];
            cursor.read_exact(&mut len_buf)?;
            let col_len = u32::from_le_bytes(len_buf) as usize;
            bytes_read += 4;

            // Read Column Data
            let mut col_data = vec![0u8; col_len];
            cursor.read_exact(&mut col_data)?;
            bytes_read += col_len as u64;

            // Decompress
            let floats =
                decompress_vectors(&col_data, loc.vector_count, CompressionStrategy::AlpRd);
            columns.push(floats);
        }

        // 3. Transpose back to Row-Major (Vec<Vec<f32>>)
        let vectors = transpose_from_columns(&columns, loc.vector_count);
        Ok(vectors)
    }

    pub fn might_contain(&self, id: u64) -> bool {
        self.bloom.contains(&id.to_le_bytes())
    }

    pub fn read_metadata(&self) -> &[u8] {
        &self.quantizer
    }
}
