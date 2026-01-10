use crate::compression::wrapper::{
    CompressionStrategy, decompress_vectors, transpose_from_columns,
};
use crate::format::{
    DriftFooter, FOOTER_SIZE, HEADER_SIZE, MAGIC_V2, ROW_GROUP_HEADER_SIZE, RowGroupHeader,
};
use byteorder::{LittleEndian, ReadBytesExt};
use drift_core::bucket::compute_distance_lut; // SIMD Kernel
use drift_core::quantizer::Quantizer;
use drift_traits::{PageId, PageManager, TombstoneView};
use std::collections::BinaryHeap;
use std::io::{self, Cursor, Read};
use std::sync::Arc;
use zerocopy::FromBytes;

#[derive(PartialEq)]
struct SearchCandidate(f32, u64);

impl Eq for SearchCandidate {}

impl Ord for SearchCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Rust's Heap is a MaxHeap. We want to keep the K smallest distances.
        // So we want the MaxHeap to pop the LARGEST distance.
        // Default f32 comparison.
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl PartialOrd for SearchCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

pub struct BucketFileReader {
    storage: Arc<dyn PageManager>,
    file_id: u32,
    quantizer: Option<Quantizer>,
}

impl BucketFileReader {
    pub fn new(storage: Arc<dyn PageManager>, file_id: u32) -> Self {
        Self {
            storage,
            file_id,
            quantizer: None,
        }
    }

    /// Loads the Quantizer from the file footer.
    pub async fn load_quantizer(&mut self) -> io::Result<()> {
        if self.quantizer.is_some() {
            return Ok(());
        }

        let file_len = self.storage.len(self.file_id).await?;
        if file_len < FOOTER_SIZE as u64 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "File too small"));
        }

        // 1. Read Footer
        let footer_page = PageId {
            file_id: self.file_id,
            offset: file_len - FOOTER_SIZE as u64,
            length: FOOTER_SIZE as u32,
        };
        let footer_bytes = self.storage.read_page(footer_page).await?;

        let footer = DriftFooter::read_from_bytes(&footer_bytes).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("Invalid Footer {e}"))
        })?;

        if footer.magic != MAGIC_V2 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid Magic"));
        }

        // 2. Read Quantizer Blob
        let q_page = PageId {
            file_id: self.file_id,
            offset: footer.quantizer_offset,
            length: footer.quantizer_length,
        };
        let q_bytes = self.storage.read_page(q_page).await?;

        let (q, _): (Quantizer, usize) =
            bincode::decode_from_slice(&q_bytes, bincode::config::standard())
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        self.quantizer = Some(q);
        Ok(())
    }

    pub async fn scan(
        &mut self,
        query: &[f32],
        k: usize,
        tombstones: &dyn TombstoneView, // ⚡ Filter Source
    ) -> io::Result<Vec<(u64, f32)>> {
        if self.quantizer.is_none() {
            self.load_quantizer().await?;
        }
        let q = self.quantizer.as_ref().unwrap();
        let lut = q.precompute_lut(query);
        let dim = query.len();

        // MaxHeap to keep the "Worst" of the "Best K".
        // If we find a candidate better than heap.peek(), we swap.
        let mut heap: BinaryHeap<SearchCandidate> = BinaryHeap::with_capacity(k + 1);

        let mut current_offset = HEADER_SIZE as u64;

        loop {
            // A. Read RG Header
            let head_page = PageId {
                file_id: self.file_id,
                offset: current_offset,
                length: ROW_GROUP_HEADER_SIZE as u32,
            };

            let rg_bytes = match self.storage.read_page(head_page).await {
                Ok(b) if b.len() == ROW_GROUP_HEADER_SIZE => b,
                _ => break, // EOF
            };

            let rg = RowGroupHeader::read_from_bytes(&rg_bytes).map_err(|e| {
                io::Error::new(io::ErrorKind::InvalidData, format!("Bad RG Header {e}"))
            })?;

            // B. Read Hot Index (IDs + SQ8)
            let hot_len = rg.hot_length;
            let hot_page = PageId {
                file_id: self.file_id,
                offset: rg.hot_offset,
                length: hot_len,
            };
            let hot_blob = self.storage.read_page(hot_page).await?;

            // C. Scan Row Group (Pushed down logic)
            self.scan_row_group_filtered(&rg, &hot_blob, &lut, dim, k, tombstones, &mut heap);

            // D. Advance
            current_offset = rg.cold_offset + rg.cold_length as u64;
        }

        // Convert Heap -> Sorted Vec
        let mut results = heap.into_vec();
        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        Ok(results.into_iter().map(|c| (c.1, c.0)).collect())
    }

    fn scan_row_group_filtered(
        &self,
        header: &RowGroupHeader,
        hot_blob: &[u8],
        lut: &[f32],
        dim: usize,
        k: usize,
        tombstones: &dyn TombstoneView,         // ⚡ Filter
        heap: &mut BinaryHeap<SearchCandidate>, // ⚡ Heap updated in-place
    ) {
        let count = header.vector_count as usize;
        let mut cursor = Cursor::new(hot_blob);

        // 1. Decode IDs first (Cheap)
        let mut ids = Vec::with_capacity(count);
        for _ in 0..count {
            if let Ok(id) = cursor.read_u64::<LittleEndian>() {
                ids.push(id);
            }
        }

        // 2. Locate Codes
        let ids_size = cursor.position() as usize;
        if ids_size >= hot_blob.len() {
            return;
        }
        let codes = &hot_blob[ids_size..];

        // 3. Iterate
        let lut_ptr = lut.as_ptr();
        let codes_ptr = codes.as_ptr();

        for (i, &id) in ids.iter().enumerate() {
            // ⚡ CHECK TOMBSTONE BEFORE MATH ⚡
            if tombstones.contains(id) {
                continue;
            }

            // Unsafe SIMD calc
            let offset = i * dim;
            // Bound check safety
            if offset + dim > codes.len() {
                break;
            }

            let dist = unsafe { compute_distance_lut(codes_ptr.add(offset), lut_ptr, dim) };

            // ⚡ MAINTAIN TOP-K HEAP ⚡
            if heap.len() < k {
                heap.push(SearchCandidate(dist, id));
            } else if let Some(worst) = heap.peek() {
                if dist < worst.0 {
                    heap.pop();
                    heap.push(SearchCandidate(dist, id));
                }
            }
        }
    }

    fn scan_row_group(
        &self,
        header: &RowGroupHeader,
        hot_blob: &[u8],
        lut: &[f32],
        dim: usize,
    ) -> Vec<(u64, f32)> {
        let count = header.vector_count as usize;
        let mut results = Vec::with_capacity(count);

        // 1. Decode IDs
        let mut cursor = Cursor::new(hot_blob);
        let mut ids = Vec::with_capacity(count);
        for _ in 0..count {
            if let Ok(id) = cursor.read_u64::<LittleEndian>() {
                ids.push(id);
            }
        }

        // 2. Decode SQ8 Codes
        let ids_size = cursor.position() as usize;
        if ids_size >= hot_blob.len() {
            return results;
        }

        let codes = &hot_blob[ids_size..];
        if codes.len() < count * dim {
            return results;
        }

        // 3. Run ADC Kernel (Unsafe SIMD)
        let lut_ptr = lut.as_ptr();
        let codes_ptr = codes.as_ptr();

        for (i, &id) in ids.iter().enumerate() {
            let offset = i * dim;
            let dist = unsafe { compute_distance_lut(codes_ptr.add(offset), lut_ptr, dim) };
            results.push((id, dist));
        }

        results
    }

    /// Reads ALL high-fidelity vectors from the file into a single flat buffer.
    /// Used for Maintenance (Split/Merge).
    pub async fn read_all_vectors(&mut self, dim: usize) -> io::Result<(Vec<u64>, Vec<f32>)> {
        let mut all_ids = Vec::new();
        let mut all_vecs = Vec::new();

        // Start after File Header
        let mut current_offset = HEADER_SIZE as u64;

        loop {
            // 1. Read RowGroup Header
            let head_page = PageId {
                file_id: self.file_id,
                offset: current_offset,
                length: ROW_GROUP_HEADER_SIZE as u32,
            };

            // Graceful EOF check
            let head_bytes_res = self.storage.read_page(head_page).await;
            let head_bytes = match head_bytes_res {
                Ok(b) if b.len() == ROW_GROUP_HEADER_SIZE => b,
                _ => break, // EOF
            };

            let rg_header = RowGroupHeader::read_from_bytes(&head_bytes)
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Bad RG Header"))?;

            // 2. Read Hot Index (to get IDs)
            // Hot Layout: [IDs] [SQ8] ...
            // We only need the IDs part.
            let count = rg_header.vector_count as usize;
            let id_bytes_len = count * 8;

            let id_page = PageId {
                file_id: self.file_id,
                offset: rg_header.hot_offset,
                length: id_bytes_len as u32,
            };

            let id_blob = self.storage.read_page(id_page).await?;
            let mut cursor = Cursor::new(id_blob);
            for _ in 0..count {
                all_ids.push(cursor.read_u64::<LittleEndian>()?);
            }

            // 3. Read Cold Data (High Fidelity Floats)
            let cold_page = PageId {
                file_id: self.file_id,
                offset: rg_header.cold_offset,
                length: rg_header.cold_length,
            };
            let cold_blob = self.storage.read_page(cold_page).await?;

            // 4. Decompress
            // Format: [Len][Bytes] repeated for each column (dim times)
            let mut cursor = Cursor::new(&cold_blob);
            let mut columns = Vec::with_capacity(dim);

            for _ in 0..dim {
                let chunk_len = cursor.read_u32::<LittleEndian>()? as usize;
                let mut chunk = vec![0u8; chunk_len];
                cursor.read_exact(&mut chunk)?;

                let col_floats = decompress_vectors(&chunk, count, CompressionStrategy::AlpRd);
                columns.push(col_floats);
            }

            // Transpose back to Row-Major [N][D]
            let rows = transpose_from_columns(&columns, count);

            // Flatten into main buffer
            for row in rows {
                all_vecs.extend_from_slice(&row);
            }

            // 5. Advance
            current_offset = rg_header.cold_offset + rg_header.cold_length as u64;
        }

        Ok((all_ids, all_vecs))
    }
}
