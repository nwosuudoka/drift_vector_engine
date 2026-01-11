use crate::compression::wrapper::{
    CompressionStrategy, decompress_vectors, transpose_from_columns,
};
use crate::format::{DriftHeader, HEADER_SIZE, MAGIC_V2, ROW_GROUP_HEADER_SIZE, RowGroupHeader};
use byteorder::{LittleEndian, ReadBytesExt};
use drift_core::bucket::compute_distance_lut;
use drift_core::quantizer::Quantizer;
use drift_traits::{IoContext, PageId, PageManager, SearchCandidate, TombstoneView};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};
use std::io::{self, Cursor, Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;
use zerocopy::{FromBytes, FromZeros};

#[derive(PartialEq)]
struct CandidateWrapper(SearchCandidate);

impl Eq for CandidateWrapper {}

impl Ord for CandidateWrapper {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .approx_dist
            .partial_cmp(&other.0.approx_dist)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for CandidateWrapper {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct BucketFileReader {
    storage: Arc<dyn PageManager>,
    file_id: u32,
    quantizer: Option<Quantizer>,
    data_start_offset: u64,
}

impl BucketFileReader {
    pub fn new(storage: Arc<dyn PageManager>, file_id: u32) -> Self {
        Self {
            storage,
            file_id,
            quantizer: None,
            data_start_offset: HEADER_SIZE as u64,
        }
    }

    pub async fn load_quantizer(&mut self) -> io::Result<()> {
        if self.quantizer.is_some() {
            return Ok(());
        }

        // 1. Read Header
        let head_page = PageId {
            file_id: self.file_id,
            offset: 0,
            length: HEADER_SIZE as u32,
        };
        let head_bytes = self.storage.read_page(head_page).await?;
        let header = DriftHeader::read_from_bytes(&head_bytes)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid Header"))?;

        if header.magic != MAGIC_V2 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid Magic"));
        }

        // 2. Read Quantizer
        let q_page = PageId {
            file_id: self.file_id,
            offset: header.quantizer_offset,
            length: header.quantizer_length,
        };
        let q_bytes = self.storage.read_page(q_page).await?;

        let (q, _): (Quantizer, usize) =
            bincode::decode_from_slice(&q_bytes, bincode::config::standard())
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        self.quantizer = Some(q);

        // ⚡ Update Start Offset so Scans Skip the Quantizer
        self.data_start_offset = header.quantizer_offset + header.quantizer_length as u64;

        Ok(())
    }

    fn get_row_header(buf: Vec<u8>) -> RowGroupHeader {
        let mut header = RowGroupHeader::new_zeroed();
        // This is safe because both are raw bytes and size is checked above.
        unsafe {
            std::ptr::copy_nonoverlapping(
                buf.as_ptr(),
                &mut header as *mut _ as *mut u8,
                ROW_GROUP_HEADER_SIZE,
            );
        }
        header
    }

    /// Step 1: Scan Hot Index
    pub async fn scan(
        &mut self,
        query: &[f32],
        k: usize,
        tombstones: &dyn TombstoneView,
    ) -> io::Result<Vec<SearchCandidate>> {
        if self.quantizer.is_none() {
            self.load_quantizer().await?;
        }
        let q = self.quantizer.as_ref().unwrap();
        let lut = q.precompute_lut(query);
        let dim = query.len();

        let mut heap: BinaryHeap<CandidateWrapper> = BinaryHeap::with_capacity(k + 1);
        let mut current_offset = self.data_start_offset;

        loop {
            // A. Read RG Header
            let head_page = PageId {
                file_id: self.file_id,
                offset: current_offset,
                length: ROW_GROUP_HEADER_SIZE as u32,
            };

            let head_bytes = match self.storage.read_page(head_page).await {
                Ok(b) if b.len() == ROW_GROUP_HEADER_SIZE => b,
                _ => break, // EOF
            };

            // let rg = *RowGroupHeader::ref_from_bytes(&head_bytes)
            // .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            let rg = Self::get_row_header(head_bytes);

            // B. Read Hot Index
            let hot_page = PageId {
                file_id: self.file_id,
                offset: rg.hot_offset,
                length: rg.hot_length,
            };
            let hot_blob = self.storage.read_page(hot_page).await?;

            // C. Scan
            self.scan_row_group_filtered(&rg, &hot_blob, &lut, dim, k, tombstones, &mut heap);

            // D. Advance
            current_offset = rg.cold_offset.clone() + rg.cold_length as u64;
        }

        // Return sorted candidates
        let mut results: Vec<SearchCandidate> = heap.into_vec().into_iter().map(|w| w.0).collect();
        results.sort_by(|a, b| {
            a.approx_dist
                .partial_cmp(&b.approx_dist)
                .unwrap_or(Ordering::Equal)
        });
        Ok(results)
    }

    fn scan_row_group_filtered(
        &self,
        header: &RowGroupHeader,
        hot_blob: &[u8],
        lut: &[f32],
        dim: usize,
        k: usize,
        tombstones: &dyn TombstoneView,
        heap: &mut BinaryHeap<CandidateWrapper>,
    ) {
        let count = header.vector_count as usize;
        let mut cursor = Cursor::new(hot_blob);

        // 1. Decode IDs
        let mut ids = Vec::with_capacity(count);
        for _ in 0..count {
            if let Ok(id) = cursor.read_u64::<LittleEndian>() {
                ids.push(id);
            }
        }

        // 2. Decode Codes
        let ids_size = cursor.position() as usize;
        if ids_size >= hot_blob.len() {
            return;
        }
        let codes = &hot_blob[ids_size..];

        // 3. Scan
        let lut_ptr = lut.as_ptr();
        let codes_ptr = codes.as_ptr();

        for (i, &id) in ids.iter().enumerate() {
            if tombstones.contains(id) {
                continue;
            }

            let offset = i * dim;
            if offset + dim > codes.len() {
                break;
            }

            let dist = unsafe { compute_distance_lut(codes_ptr.add(offset), lut_ptr, dim) };

            let candidate = SearchCandidate {
                id,
                approx_dist: dist,
                file_id: self.file_id,
                cold_offset: header.cold_offset,
                cold_length: header.cold_length,
                index_in_rg: i as u16,
                vector_count: count as u16, // ⚡ The size of THIS block
            };

            if heap.len() < k {
                heap.push(CandidateWrapper(candidate));
            } else if let Some(worst) = heap.peek() {
                if dist < worst.0.approx_dist {
                    heap.pop();
                    heap.push(CandidateWrapper(candidate));
                }
            }
        }
    }

    /// Step 2: Refine (Batch Fetch Cold Data)
    pub async fn refine(
        &self,
        candidates: Vec<SearchCandidate>,
        query: &[f32],
        dim: usize,
    ) -> io::Result<Vec<(u64, f32)>> {
        let mut results = Vec::with_capacity(candidates.len());
        let mut groups: BTreeMap<u64, (u32, u16, Vec<SearchCandidate>)> = BTreeMap::new();

        // Group by RowGroup
        for c in candidates {
            if c.file_id != self.file_id {
                continue;
            }
            groups
                .entry(c.cold_offset)
                .or_insert_with(|| (c.cold_length, c.vector_count, Vec::new()))
                .2
                .push(c);
        }

        // Fetch & Compute
        for (offset, (length, count, group)) in groups {
            let page = PageId {
                file_id: self.file_id,
                offset,
                length,
            };
            let cold_blob = self.storage.read_page(page).await?;

            let vectors = self.decompress_row_group(&cold_blob, dim, count as usize)?;

            for c in group {
                let idx = c.index_in_rg as usize;
                let start = idx * dim;
                if start + dim <= vectors.len() {
                    let vec = &vectors[start..start + dim];
                    let exact_dist = drift_core::math::l2_sq(query, vec);
                    results.push((c.id, exact_dist));
                }
            }
        }
        Ok(results)
    }

    /// Helper: Decompresses a raw cold blob (ALP/LZ4) into flat f32s
    fn decompress_row_group(&self, blob: &[u8], dim: usize, count: usize) -> io::Result<Vec<f32>> {
        let mut cursor = Cursor::new(blob);
        let mut columns = Vec::with_capacity(dim);

        for _ in 0..dim {
            let chunk_len = cursor.read_u32::<LittleEndian>()? as usize;
            let mut chunk = vec![0u8; chunk_len];
            cursor.read_exact(&mut chunk)?;

            let col_floats = decompress_vectors(&chunk, count, CompressionStrategy::AlpRd);
            columns.push(col_floats);
        }

        let rows = transpose_from_columns(&columns, count);
        let mut flat = Vec::with_capacity(count * dim);
        for row in rows {
            flat.extend_from_slice(&row);
        }
        Ok(flat)
    }

    /// Reads ALL high-fidelity vectors. Used for Maintenance.
    pub async fn read_all_vectors(&mut self, dim: usize) -> io::Result<(Vec<u64>, Vec<f32>)> {
        if self.quantizer.is_none() {
            // In read_all, we might need quantizer info to verify file structure,
            // but technically we can read header without it.
            // However, to know where data starts, we MUST read the header.
            self.load_quantizer().await?;
        }

        let mut all_ids = Vec::new();
        let mut all_vecs = Vec::new();
        let mut current_offset = self.data_start_offset;

        loop {
            // 1. RG Header
            let head_page = PageId {
                file_id: self.file_id,
                offset: current_offset,
                length: ROW_GROUP_HEADER_SIZE as u32,
            };
            let head_bytes = match self.storage.read_page(head_page).await {
                Ok(b) if b.len() == ROW_GROUP_HEADER_SIZE => b,
                _ => break,
            };
            let rg = RowGroupHeader::read_from_bytes(&head_bytes)
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Bad RG Header"))?;

            // 2. Hot Index (Get IDs)
            let count = rg.vector_count as usize;
            let id_bytes_len = count * 8;
            let id_page = PageId {
                file_id: self.file_id,
                offset: rg.hot_offset,
                length: id_bytes_len as u32,
            };
            let id_blob = self.storage.read_page(id_page).await?;
            let mut cursor = Cursor::new(id_blob);
            for _ in 0..count {
                all_ids.push(cursor.read_u64::<LittleEndian>()?);
            }

            // 3. Cold Data (High Fidelity)
            let cold_page = PageId {
                file_id: self.file_id,
                offset: rg.cold_offset,
                length: rg.cold_length,
            };
            let cold_blob = self.storage.read_page(cold_page).await?;

            // 4. Decompress using Helper (⚡ DRY Fix)
            let vectors = self.decompress_row_group(&cold_blob, dim, count)?;
            all_vecs.extend(vectors);

            // 5. Advance
            current_offset = rg.cold_offset + rg.cold_length as u64;
        }

        Ok((all_ids, all_vecs))
    }

    pub fn get_quantizer(path: &Path) -> io::Result<Quantizer> {
        let mut file = std::fs::File::open(path)?;
        let mut head_buf = [0u8; HEADER_SIZE];
        file.read_exact(&mut head_buf)?;

        // Offset 48 is quantizer_offset (u64), 56 is length (u32) based on our layout.
        // Let's rely on `DriftHeader` being Pod/ZeroCopy.
        // let header = DriftHeader::ref_from_bytes(&head_buf).map_err(io::Error::other)?;
        let header = DriftHeader::force_copy(&head_buf);

        // Validate
        if header.magic != MAGIC_V2 {
            tracing::error!("Invalid magic bytes {} != {}", header.magic, MAGIC_V2);
            panic!("Invalid magic bytes")
        }

        // Read Quantizer Blob
        file.seek(SeekFrom::Start(header.quantizer_offset))?;
        let mut q_buf = vec![0u8; header.quantizer_length as usize];
        file.read_exact(&mut q_buf)?;

        let (q, _): (Quantizer, usize) =
            bincode::decode_from_slice(&q_buf, bincode::config::standard())
                .map_err(io::Error::other)
                .context("Failed to decode quantizer")?;

        Ok(q)
    }
}
