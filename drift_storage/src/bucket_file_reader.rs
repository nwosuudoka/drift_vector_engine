use crate::compression::wrapper::{
    CompressionStrategy, decompress_vectors, transpose_from_columns,
};
use crate::disk_manager::DiskManager;
use crate::format::{
    DriftFooter, DriftHeader, HEADER_SIZE, MAGIC_V2, ROW_GROUP_HEADER_SIZE, RowGroupHeader,
};
use byteorder::{LittleEndian, ReadBytesExt};
use drift_core::bucket::compute_distance_lut;
use drift_core::quantizer::Quantizer;
use drift_traits::{IoContext, SearchCandidate, TombstoneView};
use fastbloom::BloomFilter;
use opendal::Operator;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};
use std::io::{self, Cursor, Read, Seek, SeekFrom};
use std::path::Path;
use zerocopy::FromBytes;

/// Wrapper to make SearchCandidate sortable in a BinaryHeap (MaxHeap).
#[derive(PartialEq)]
struct CandidateWrapper(SearchCandidate);

struct ScanParams<'a> {
    lut: &'a [f32],
    dim: usize,
    k: usize,
    tombstones: &'a dyn TombstoneView,
}

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
    manager: DiskManager,
    pub footer: DriftFooter,
    pub row_groups: Vec<RowGroupHeader>, // Cached headers
    pub quantizer: Option<Quantizer>,
    pub bloom: Option<BloomFilter>,
}

impl BucketFileReader {
    /// Opens a bucket file from an Operator.
    /// Reads Footer, Index, and Header (Quantizer) immediately.
    pub async fn open(op: Operator, path: &str) -> io::Result<Self> {
        let manager = DiskManager::new(op, path.to_string());
        let file_len = manager.len().await?;

        if file_len < crate::format::FOOTER_SIZE as u64 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "File too small"));
        }

        // 1. Read Footer
        let footer_pos = file_len - crate::format::FOOTER_SIZE as u64;
        let footer_bytes = manager
            .read_at(footer_pos, crate::format::FOOTER_SIZE)
            .await?;
        let footer = DriftFooter::read_from_bytes(&footer_bytes)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid Footer"))?;

        if footer.magic != MAGIC_V2 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Bad Magic"));
        }

        // 2. Read Index (RowGroup Headers)
        // Format: [Count: u32] [Header 1] [Header 2] ...
        // We trust the footer's row_group_count and index_start_offset.
        // Size = 4 bytes (count) + (N * 64 bytes)
        let index_size = 4 + (footer.row_group_count as usize * ROW_GROUP_HEADER_SIZE);
        let index_bytes = manager
            .read_at(footer.index_start_offset, index_size)
            .await?;

        let mut cursor = Cursor::new(index_bytes);
        let _stored_count = cursor.read_u32::<LittleEndian>()?;

        let mut row_groups = Vec::with_capacity(footer.row_group_count as usize);
        let mut rg_buf = [0u8; ROW_GROUP_HEADER_SIZE];

        for _ in 0..footer.row_group_count {
            cursor.read_exact(&mut rg_buf)?;
            let rg = RowGroupHeader::read_from_bytes(&rg_buf)
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Bad RG Header"))?;
            row_groups.push(rg);
        }

        // 3. Read Header & Quantizer
        // We need the header to find the Quantizer offset.
        let head_bytes = manager.read_at(0, HEADER_SIZE).await?;
        let header = DriftHeader::read_from_bytes(&head_bytes)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid Header"))?;

        // We load the quantizer eagerly or lazy?
        // Let's load Lazy to keep open() fast, but we need data_start_offset.
        let _data_start_offset = header.quantizer_offset + header.quantizer_length as u64;

        // 4. Read Bloom (Optional)
        let bloom = if footer.bloom_filter_length > 0 {
            let b_bytes = manager
                .read_at(
                    footer.bloom_filter_offset,
                    footer.bloom_filter_length as usize,
                )
                .await?;
            let (b, _): (BloomFilter, usize) =
                bincode::serde::decode_from_slice(&b_bytes, bincode::config::standard())
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            Some(b)
        } else {
            None
        };

        Ok(Self {
            manager,
            footer,
            row_groups,
            quantizer: None,
            bloom,
        })
    }

    pub async fn load_quantizer(&mut self) -> io::Result<()> {
        if self.quantizer.is_some() {
            return Ok(());
        }

        // Re-read header to get offset (or we could have stored it).
        // Since HEADER_SIZE is small, re-reading is negligible.
        let head_bytes = self.manager.read_at(0, HEADER_SIZE).await?;
        let header = DriftHeader::read_from_bytes(&head_bytes)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid Header"))?;

        let q_bytes = self
            .manager
            .read_at(header.quantizer_offset, header.quantizer_length as usize)
            .await?;
        let (q, _): (Quantizer, usize) =
            bincode::decode_from_slice(&q_bytes, bincode::config::standard())
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        self.quantizer = Some(q);
        Ok(())
    }

    /// ⚡ SCAN PATH: Approx Search using SQ8
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
        let q_dim = q.min.len();

        if q_dim != dim {
            tracing::warn!(
                "⚠️ Scan dim mismatch: path={} query_dim={} quant_dim={}",
                self.manager.path,
                dim,
                q_dim
            );
        }
        tracing::info!(
            "🔎 Scan start: path={} k={} query_dim={} quant_dim={} row_groups={}",
            self.manager.path,
            k,
            dim,
            q_dim,
            self.row_groups.len()
        );

        let mut heap: BinaryHeap<CandidateWrapper> = BinaryHeap::with_capacity(k + 1);

        let params = ScanParams {
            lut: &lut,
            dim,
            k,
            tombstones,
        };

        // Iterate Cached RowGroups
        let mut total_vectors = 0usize;
        let mut total_hot_bytes = 0usize;
        for rg in &self.row_groups {
            // Read Hot Index Blob
            // Hot Layout: [IDs: u64 * N] [Codes: u8 * N * D] [Tombstones...]
            // We read just enough for IDs + Codes.
            // ID Size: N * 8
            // Code Size: N * D
            let count = rg.vector_count as usize;
            let _hot_size = (count * 8) + (count * dim); // Roughly. 
            // Better: Read the whole hot_length as defined in header.

            let hot_blob = self
                .manager
                .read_at(rg.hot_offset, rg.hot_length as usize)
                .await?;

            total_vectors += count;
            total_hot_bytes += hot_blob.len();
            self.scan_row_group_filtered(rg, &hot_blob, &params, &mut heap);
        }

        let mut results: Vec<SearchCandidate> = heap.into_vec().into_iter().map(|w| w.0).collect();
        results.sort_by(|a, b| {
            a.approx_dist
                .partial_cmp(&b.approx_dist)
                .unwrap_or(Ordering::Equal)
        });
        tracing::info!(
            "🔎 Scan done: path={} total_vectors={} total_hot_bytes={} candidates={}",
            self.manager.path,
            total_vectors,
            total_hot_bytes,
            results.len()
        );
        Ok(results)
    }

    fn scan_row_group_filtered(
        &self,
        header: &RowGroupHeader,
        hot_blob: &[u8],
        params: &ScanParams,
        heap: &mut BinaryHeap<CandidateWrapper>,
    ) {
        let count = header.vector_count as usize;
        let mut cursor = Cursor::new(hot_blob);

        let mut ids = Vec::with_capacity(count);
        for _ in 0..count {
            if let Ok(id) = cursor.read_u64::<LittleEndian>() {
                ids.push(id);
            }
        }

        let ids_size = cursor.position() as usize;
        if ids_size >= hot_blob.len() {
            tracing::warn!(
                "⚠️ Scan RG: ids_size {} >= hot_blob_len {} path={} cold_offset={} cold_len={}",
                ids_size,
                hot_blob.len(),
                self.manager.path,
                header.cold_offset,
                header.cold_length
            );
            return;
        }
        let codes = &hot_blob[ids_size..];

        let lut_ptr = params.lut.as_ptr();
        let codes_ptr = codes.as_ptr();

        let mut considered = 0usize;
        let mut tombstoned = 0usize;
        let mut min_dist = f32::MAX;
        let mut max_dist = 0.0f32;

        for (i, &id) in ids.iter().enumerate() {
            if params.tombstones.contains(id) {
                tombstoned += 1;
                continue;
            }

            let offset = i * params.dim;
            if offset + params.dim > codes.len() {
                break;
            }

            let dist = unsafe { compute_distance_lut(codes_ptr.add(offset), lut_ptr, params.dim) };
            considered += 1;
            if dist < min_dist {
                min_dist = dist;
            }
            if dist > max_dist {
                max_dist = dist;
            }

            let candidate = SearchCandidate {
                id,
                approx_dist: dist,
                file_id: 0, // Managed by BucketManager
                cold_offset: header.cold_offset,
                cold_length: header.cold_length,
                index_in_rg: i as u32,
                vector_count: count as u32,
            };

            if heap.len() < params.k {
                heap.push(CandidateWrapper(candidate));
            } else if let Some(worst) = heap.peek()
                && dist < worst.0.approx_dist
            {
                heap.pop();
                heap.push(CandidateWrapper(candidate));
            }
        }

        tracing::info!(
            "🔎 Scan RG: path={} vectors={} hot_len={} ids_bytes={} codes_len={} dim={} considered={} tombstoned={} min_dist={:.4} max_dist={:.4}",
            self.manager.path,
            count,
            hot_blob.len(),
            ids_size,
            codes.len(),
            params.dim,
            considered,
            tombstoned,
            min_dist,
            max_dist
        );
    }

    /// REFINE PATH: Fetch & Decompress Cold Data
    pub async fn refine(
        &self,
        candidates: Vec<SearchCandidate>,
        query: &[f32],
        dim: usize,
    ) -> io::Result<Vec<(u64, f32)>> {
        let candidate_count = candidates.len();
        let mut results = Vec::with_capacity(candidate_count);
        let mut groups: BTreeMap<u64, (u32, u32, Vec<SearchCandidate>)> = BTreeMap::new();

        for c in candidates {
            groups
                .entry(c.cold_offset)
                .or_insert_with(|| (c.cold_length, c.vector_count, Vec::new()))
                .2
                .push(c);
        }

        tracing::info!(
            "🔍 Refine start: path={} groups={} candidates={} dim={}",
            self.manager.path,
            groups.len(),
            candidate_count,
            dim
        );

        for (offset, (length, count, group)) in groups {
            tracing::info!(
                "🔍 Refine: Reading Cold Blob at Offset {} (Len {}). Expecting {} vectors. Candidates={}",
                offset,
                length,
                count,
                group.len()
            );
            let cold_blob = self.manager.read_at(offset, length as usize).await?;

            if !cold_blob.is_empty() && cold_blob.iter().all(|&b| b == 0) {
                tracing::error!(
                    "🚨 DATA LOSS: Read {} bytes at offset {}, but ALL were zeros! path={}",
                    length,
                    offset,
                    self.manager.path
                );
            }

            let vectors = self.decompress_row_group(&cold_blob, dim, count as usize)?;
            tracing::info!(
                "🔍 Refine: Decompressed vectors len={} (expected={}) path={}",
                vectors.len(),
                count as usize * dim,
                self.manager.path
            );

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

    /// Returns (IDs, Vectors)
    pub async fn read_all_vectors(&mut self) -> io::Result<(Vec<u64>, Vec<Vec<f32>>)> {
        if self.quantizer.is_none() {
            self.load_quantizer().await?;
        }
        let dim = self.quantizer.as_ref().unwrap().min.len();

        let mut all_ids = Vec::new();
        let mut all_vecs = Vec::new();

        for rg in &self.row_groups {
            let count = rg.vector_count as usize;

            // 1. IDs (Hot)
            let id_block_size = count * 8;
            let id_bytes = self.manager.read_at(rg.hot_offset, id_block_size).await?;
            let mut id_cursor = Cursor::new(id_bytes);
            for _ in 0..count {
                all_ids.push(id_cursor.read_u64::<LittleEndian>()?);
            }

            // 2. Vectors (Cold)
            let cold_bytes = self
                .manager
                .read_at(rg.cold_offset, rg.cold_length as usize)
                .await?;
            let mut col_cursor = Cursor::new(&cold_bytes);
            let mut columns = Vec::with_capacity(dim);

            for _ in 0..dim {
                let len = col_cursor.read_u32::<LittleEndian>()? as usize;
                let mut col_data = vec![0u8; len];
                col_cursor.read_exact(&mut col_data)?;

                let floats = decompress_vectors(&col_data, count, CompressionStrategy::AlpRd);
                columns.push(floats);
            }

            let rows = transpose_from_columns(&columns, count);
            all_vecs.extend(rows);
        }

        Ok((all_ids, all_vecs))
    }

    fn decompress_row_group(&self, blob: &[u8], dim: usize, count: usize) -> io::Result<Vec<f32>> {
        let mut cursor = Cursor::new(blob);
        let mut columns = Vec::with_capacity(dim);

        for _ in 0..dim {
            let chunk_len = cursor.read_u32::<LittleEndian>()? as usize;
            let mut chunk = vec![0u8; chunk_len];
            cursor.read_exact(&mut chunk)?;

            let col_floats = decompress_vectors(&chunk, count, CompressionStrategy::AlpRd);
            if col_floats.len() != count {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "ALP Decode Mismatch: Expected {} items, got {}",
                        count,
                        col_floats.len()
                    ),
                ));
            }
            columns.push(col_floats);
        }

        let rows = transpose_from_columns(&columns, count);
        let mut flat = Vec::with_capacity(count * dim);
        for row in rows {
            flat.extend_from_slice(&row);
        }
        Ok(flat)
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
