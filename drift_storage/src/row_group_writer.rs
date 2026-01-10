use crate::compression::wrapper::{CompressedColumn, CompressionStrategy, transpose_flat};
use crate::format::{ROW_GROUP_HEADER_SIZE, RowGroupHeader};
use byteorder::{LittleEndian, WriteBytesExt};
use crc32fast::Hasher;
use std::io::{self, Write};
use zerocopy::IntoBytes;

/// Constants for Alignment
const ALIGNMENT: usize = 64; // Cache-line alignment

/// A buffered writer that accumulates vectors and flushes them as a persistent RowGroup.
///
/// Lifecycle:
/// 1. `new()`
/// 2. `write_group()` -> Flushes a batch of vectors to the underlying writer.
pub struct RowGroupWriter<W: Write> {
    writer: W,
    current_offset: u64, // Tracks absolute file position for offsets
}

impl<W: Write> RowGroupWriter<W> {
    pub fn new(writer: W, start_offset: u64) -> Self {
        Self {
            writer,
            current_offset: start_offset,
        }
    }

    /// Flushes a batch of vectors as a single Row Group.
    ///
    /// # Flow
    /// 1. Transpose vectors (Row -> Col)
    /// 2. Compress Cold Data (ALP) -> Buffer
    /// 3. Prepare Hot Index (SQ8 + IDs) -> Buffer
    /// 4. Calculate Checksum
    /// 5. Calculate Alignment Padding
    /// 6. Write [Header] -> [Hot] -> [Padding] -> [Cold]
    pub fn write_group(
        &mut self,
        ids: &[u64],
        flat_vectors: &[f32],
        _tombstones: Option<&bit_set::BitSet>, // Future: Serialize BitSet
        quantizer: &drift_core::quantizer::Quantizer,
        dim: usize,
    ) -> io::Result<RowGroupHeader> {
        let count = ids.len();

        // Validation check
        if flat_vectors.len() != count * dim {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Flat vector buffer length mismatch",
            ));
        }

        let count = ids.len();
        if count == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Empty RowGroup",
            ));
        }

        // --- STEP 1: PREPARE IN MEMORY (CPU INTENSIVE) ---

        // A. Transpose (Row -> Col) for Columnar Compression
        let columns = transpose_flat(flat_vectors, count, dim);

        // B. Compress Cold Data (ALP) -> Buffer
        // Format: [Len: u32][Bytes]...
        let mut cold_buffer = Vec::with_capacity(count * dim * 4);
        for col_floats in &columns {
            let compressed = CompressedColumn::compress(col_floats, CompressionStrategy::AlpRd);
            cold_buffer.write_u32::<LittleEndian>(compressed.data.len() as u32)?;
            cold_buffer.write_all(&compressed.data)?;
        }

        // C. Prepare Hot Index (SQ8 + IDs) -> Buffer
        // Hot Buffer Layout: [IDs (u64 array)] [SQ8 Codes (u8 array)]
        // We write IDs first to ensure 8-byte alignment for the u64s.
        let mut hot_buffer = Vec::with_capacity(count * 8 + count * dim);

        // 1. IDs
        for &id in ids {
            hot_buffer.write_u64::<LittleEndian>(id)?;
        }

        // 2. SQ8 Codes (Row-Major for streaming scan)
        // for vec in vectors {
        //     let codes = quantizer.encode(vec);
        //     hot_buffer.write_all(&codes)?;
        // }
        // C. Hot Index
        // 2. SQ8 Codes
        // Quantizer also needs to support flat slice or we iterate chunks
        for i in 0..count {
            let start = i * dim;
            let vec_slice = &flat_vectors[start..start + dim];
            let codes = quantizer.encode(vec_slice);
            hot_buffer.write_all(&codes)?;
        }

        // 3. Tombstones (Placeholder for now)
        // In the future, we write the BitSet bytes here.
        hot_buffer.write_u32::<LittleEndian>(0)?; // Length 0

        // --- STEP 2: CALCULATE CHECKSUM ---
        let mut hasher = Hasher::new();
        hasher.update(&hot_buffer);
        hasher.update(&cold_buffer);
        let checksum = hasher.finalize();

        // --- STEP 3: ALIGNMENT CALCULATION ---

        // Header size is fixed (64 bytes)
        let header_len = ROW_GROUP_HEADER_SIZE as u64;

        // Hot Offset starts immediately after Header
        let hot_start_rel = header_len;
        let hot_len = hot_buffer.len() as u32;

        // Calculate where Hot Data ends
        let current_pos = hot_start_rel + hot_len as u64;

        // We want Cold Data to start on a 64-byte boundary (ALIGNMENT)
        let padding_needed =
            (ALIGNMENT as u64 - (current_pos % ALIGNMENT as u64)) % ALIGNMENT as u64;

        let cold_start_rel = current_pos + padding_needed;
        let cold_len = cold_buffer.len() as u32;

        // --- STEP 4: WRITE TO DISK ---

        // Offsets are absolute (file-relative)
        let base_offset = self.current_offset;

        let header = RowGroupHeader::new(
            count as u32,
            checksum,
            base_offset + hot_start_rel,
            hot_len,
            base_offset + cold_start_rel,
            cold_len,
        );

        // 1. Write Header (Zero-Copy)
        self.writer.write_all(header.as_bytes())?;
        self.current_offset += header_len;

        // 2. Write Hot Data
        self.writer.write_all(&hot_buffer)?;
        self.current_offset += hot_len as u64;

        // 3. Write Padding (Zeros)
        if padding_needed > 0 {
            let pads = vec![0u8; padding_needed as usize];
            self.writer.write_all(&pads)?;
            self.current_offset += padding_needed;
        }

        // 4. Write Cold Data
        self.writer.write_all(&cold_buffer)?;
        self.current_offset += cold_len as u64;

        Ok(header)
    }

    /// Accessor to flush underlying writer
    pub fn inner_mut(&mut self) -> &mut W {
        &mut self.writer
    }
}
