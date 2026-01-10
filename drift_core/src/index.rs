use crate::aligned::AlignedBytes;
use crate::bucket::{Bucket, BucketData, BucketHeader};
use crate::kmeans::KMeansTrainer;
use crate::memtable::{MemTable, MemTableOptions};
use crate::quantizer::Quantizer;
use crate::wal::{WalEntry, WalReader, WalWriter};
use drift_cache::block_cache::BlockCache;
use drift_kv::bitstore::BitStore;
use drift_traits::{PageId, PageManager};
use parking_lot::{Mutex, RwLock};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::cmp::Ordering as CmpOrdering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use tracing::{error, info};

#[derive(Debug, PartialEq, Eq)]
pub enum MaintenanceStatus {
    Completed,
    SkippedSingularity,
    SkippedTooSmall,
    SkippedLocked,
}

impl MaintenanceStatus {
    pub fn to_str(&self) -> &'static str {
        match self {
            MaintenanceStatus::Completed => "Completed",
            MaintenanceStatus::SkippedSingularity => "Skipped Singularity",
            MaintenanceStatus::SkippedTooSmall => "Skipped Too Small",
            MaintenanceStatus::SkippedLocked => "SkippedLocked",
        }
    }
}

#[derive(Clone)]
pub struct IndexOptions {
    pub dim: usize,
    pub num_centroids: usize,
    pub training_sample_size: usize,
    pub max_bucket_capacity: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
}

impl Default for IndexOptions {
    fn default() -> Self {
        Self {
            dim: 0,
            num_centroids: 0,
            training_sample_size: 0,
            max_bucket_capacity: 0,
            ef_construction: 40,
            ef_search: 15,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: u64,
    pub distance: f32,
}

impl PartialEq for SearchResult {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}
impl Eq for SearchResult {}
impl PartialOrd for SearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        self.distance.partial_cmp(&other.distance)
    }
}
impl Ord for SearchResult {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        self.partial_cmp(other).unwrap_or(CmpOrdering::Equal)
    }
}

#[derive(Clone)]
pub(crate) struct CentroidEntry {
    pub(crate) id: u32,
    pub(crate) vector: Vec<f32>,
    pub(crate) active: bool,
}

pub struct PartitionResult {
    pub bucket_id: u32,
    pub ids: Vec<u64>,
    pub indices: Vec<usize>,
    pub codes: Vec<u8>,
    pub centroid: Vec<f32>,
}

pub struct VectorIndex {
    pub config: IndexOptions,
    pub(crate) centroids: RwLock<Arc<Vec<CentroidEntry>>>,

    pub memtable: RwLock<Arc<MemTable>>,
    pub frozen_memtable: RwLock<Option<Arc<MemTable>>>,

    pub(crate) buckets: RwLock<Arc<HashMap<u32, BucketHeader>>>,
    pub cache: Arc<BlockCache<BucketData>>,
    pub(crate) quantizer: RwLock<Option<Arc<Quantizer>>>,
    pub(crate) next_bucket_id: AtomicU32,
    pub(crate) wal: Mutex<WalWriter>,
    pub kv: Arc<BitStore>,
    pub deleted_ids: RwLock<HashSet<u64>>,
}

impl VectorIndex {
    pub fn new(
        config: IndexOptions,
        wal_path: &Path,
        storage: Arc<dyn PageManager>,
    ) -> io::Result<Self> {
        let memtable = Arc::new(MemTable::new(MemTableOptions {
            capacity: config.max_bucket_capacity * 10,
            dim: config.dim,
        }));

        let mut recovered_deletes = HashSet::new();

        if wal_path.exists() {
            let reader = WalReader::open(wal_path)?;
            for entry in reader.read_all() {
                match entry {
                    WalEntry::Insert { id, vector } => {
                        memtable.insert(id, &vector);
                        recovered_deletes.remove(&id);
                    }
                    WalEntry::Delete { id } => {
                        memtable.delete(id);
                        recovered_deletes.insert(id);
                    }
                }
            }
        }

        let writer = WalWriter::new(wal_path)?;
        let base_dir = wal_path.parent().unwrap_or(Path::new("."));
        let kv_path = base_dir.join("id_map");
        let kv = Arc::new(BitStore::new(&kv_path).map_err(|e| io::Error::other(e.to_string()))?);

        let cache = Arc::new(BlockCache::new(storage, 1000, 16));

        Ok(Self {
            config,
            wal: Mutex::new(writer),

            memtable: RwLock::new(memtable),
            frozen_memtable: RwLock::new(None),

            quantizer: RwLock::new(None),
            centroids: RwLock::new(Arc::new(Vec::new())),
            next_bucket_id: AtomicU32::new(0),
            buckets: RwLock::new(Arc::new(HashMap::new())),
            kv,
            cache,
            deleted_ids: RwLock::new(recovered_deletes),
        })
    }

    // --- 1. ATOMIC ROTATION ---
    pub fn rotate_and_freeze(&self) -> io::Result<Option<Arc<MemTable>>> {
        // 1. Backpressure
        if self.frozen_memtable.read().is_some() {
            return Ok(None);
        }

        // 2. Acquire WAL Lock (Pauses Inserts)
        let _wal_lock = self.wal.lock();

        // 3. Create Fresh MemTable
        let new_active = Arc::new(MemTable::new(MemTableOptions {
            capacity: self.config.max_bucket_capacity * 10,
            dim: self.config.dim,
        }));

        // 4. Atomic Swap (using RwLock Write Guard)
        let old_arc = {
            let mut guard = self.memtable.write();
            let old = guard.clone();
            *guard = new_active;
            old
        };

        // 5. Install Frozen State
        {
            let mut frozen_guard = self.frozen_memtable.write();
            *frozen_guard = Some(old_arc.clone());
        }

        info!(
            "MemTable Rotated. {} vectors moved to Frozen state.",
            old_arc.len()
        );
        Ok(Some(old_arc))
    }

    // --- 2. FLUSH CONFIRMATION ---
    /// Called by the Janitor after the segment is safely on disk.
    pub fn confirm_flush(&self) -> io::Result<()> {
        // 1. Truncate WAL (Data is safe on disk)
        {
            let mut wal = self.wal.lock();
            wal.truncate()?;
        }

        // 2. Clear Frozen Slot (Releasing RAM)
        {
            let mut frozen_guard = self.frozen_memtable.write();
            *frozen_guard = None;
        }

        info!("Flush Confirmed. WAL truncated and Frozen slot cleared.");
        Ok(())
    }

    // --- Helpers using RwLock ---
    pub(crate) fn update_centroids<F>(&self, mut f: F)
    where
        F: FnMut(&Vec<CentroidEntry>) -> Vec<CentroidEntry>,
    {
        let mut guard = self.centroids.write();
        let new_vec = f(guard.as_ref());
        *guard = Arc::new(new_vec);
    }

    pub(crate) fn update_buckets<F>(&self, mut f: F)
    where
        F: FnMut(&HashMap<u32, BucketHeader>) -> HashMap<u32, BucketHeader>,
    {
        let mut guard = self.buckets.write();
        let new_map = f(guard.as_ref());
        *guard = Arc::new(new_map);
    }

    // --- 3. ZERO-COPY TRAINING ---
    pub async fn train_from_memtable(&self, memtable: &MemTable) -> io::Result<()> {
        info!("Janitor: Training Quantizer from MemTable (Zero-Copy)...");

        if memtable.len() == 0 {
            return Ok(());
        }

        let dim = self.config.dim;

        // ⚡ FIX: Scope the locks!
        // We acquire locks, copy the sample, and release them immediately inside this block.
        let sample_data = {
            let (_ids_guard, data_guard, _tomb_guard) = memtable.get_data_guards();

            assert_eq!(
                data_guard.len() % dim,
                0,
                "MemTable corruption: data length alignment"
            );

            let sample_limit = 10_000 * dim;
            if data_guard.len() > sample_limit {
                data_guard[..sample_limit].to_vec()
            } else {
                data_guard.to_vec()
            }
        }; // <--- 🔓 Locks are dropped here.

        // 4. Compute (Async)
        // Now safe to await because we hold no !Send types.
        let q = tokio::task::spawn_blocking(move || Quantizer::train(&sample_data, dim))
            .await
            .map_err(|e| io::Error::other(format!("Quantizer Train failed: {}", e)))?;

        // 5. Update State
        let q_arc = Arc::new(q);
        *self.quantizer.write() = Some(q_arc);

        info!("Janitor: Quantizer Trained.");
        Ok(())
    }

    pub fn insert(&self, id: u64, vector: &[f32]) -> io::Result<()> {
        let mut wal = self.wal.lock();
        wal.write_insert(id, vector)?;
        let memtable = self.memtable.read();
        memtable.insert(id, vector);
        self.deleted_ids.write().remove(&id);
        Ok(())
    }

    pub fn insert_batch(&self, vectors: &[(u64, Vec<f32>)]) -> std::io::Result<()> {
        {
            let mut wal = self.wal.lock();
            for (id, vec) in vectors {
                wal.write_insert(*id, vec)?;
            }
        }
        let memtable = self.memtable.read();
        memtable.insert_batch(vectors);
        {
            let mut deleted = self.deleted_ids.write();
            for (id, _) in vectors {
                deleted.remove(id);
            }
        }
        Ok(())
    }

    pub fn delete(&self, id: u64) -> io::Result<()> {
        let mut wal = self.wal.lock();
        wal.write_delete(id)?;

        let memtable = self.memtable.read();
        memtable.delete(id);

        self.deleted_ids.write().insert(id);

        let id_bytes = id.to_le_bytes();
        if let Ok(Some(bucket_id_bytes)) = self.kv.get(&id_bytes)
            && let Ok(bucket_id) = bucket_id_bytes.try_into().map(u32::from_le_bytes)
        {
            let buckets = self.buckets.read();
            if let Some(header) = buckets.get(&bucket_id) {
                header.mark_tombstone();
            }
        }
        let _ = self.kv.remove(&id_bytes);
        Ok(())
    }

    // --- 5. UNIFIED SEARCH ---
    pub async fn search_async(
        &self,
        query: &[f32],
        k: usize,
        target_confidence: f32,
        lambda: f32,
        tau: f32,
    ) -> io::Result<Vec<SearchResult>> {
        let internal_k = k * 2;

        let lut: Option<Vec<f32>> = {
            let guard = self.quantizer.read();
            guard.as_ref().map(|q| q.precompute_lut(query))
        };

        let deleted_snapshot = {
            let guard = self.deleted_ids.read();
            guard.clone()
        };

        // ⚡ UNIFIED RAM SEARCH (Active + Frozen)
        let mem_results = {
            // A. Search Active Table
            let active = self.memtable.read();
            let mut results = active.search(query, internal_k);

            // B. Search Frozen Table (if exists)
            {
                let frozen_guard = self.frozen_memtable.read();
                if let Some(frozen) = frozen_guard.as_ref() {
                    results.extend(frozen.search(query, internal_k));
                }
            }
            results
        };

        let selected_headers = self.get_selected_headers(query, target_confidence, lambda, tau);

        let mut disk_candidates = Vec::new();
        let mut l0_found = HashSet::new();
        for (id, _) in &mem_results {
            l0_found.insert(*id);
        }

        if let Some(lut_vec) = &lut {
            let dim = self.config.dim;
            for header in &selected_headers {
                if let Ok(data) = self.cache.get(&header.page_id).await {
                    let hits = Bucket::scan_with_lut(&data, lut_vec, dim);
                    for res in hits {
                        if !deleted_snapshot.contains(&res.id) && !l0_found.contains(&res.id) {
                            disk_candidates.push((header.id, res));
                        }
                    }
                }
            }
        }

        disk_candidates.sort_by(|a, b| {
            a.1.distance
                .partial_cmp(&b.1.distance)
                .unwrap_or(CmpOrdering::Equal)
        });
        if disk_candidates.len() > internal_k {
            disk_candidates.truncate(internal_k);
        }

        let mut final_heap = BinaryHeap::with_capacity(k);
        for (id, dist) in mem_results {
            if !deleted_snapshot.contains(&id) {
                Self::push_to_heap(&mut final_heap, SearchResult { id, distance: dist }, k);
            }
        }

        // Disk Refinement logic (simplified copy from previous)
        let mut buckets_to_refine: HashMap<u32, Vec<u64>> = HashMap::new();
        for (bid, res) in &disk_candidates {
            buckets_to_refine.entry(*bid).or_default().push(res.id);
        }

        for (bid, target_ids) in buckets_to_refine {
            let page_id_opt = self.get_bucket_page_id(bid);
            if page_id_opt.is_none() {
                continue;
            }
            let page_id = page_id_opt.unwrap();

            let bucket_vecs_res = self.cache.storage().read_high_fidelity(bid).await;
            let hot_data_res = self.cache.get(&page_id).await;

            if let (Ok(bucket_vecs), Ok(hot_data)) = (bucket_vecs_res, hot_data_res) {
                let safe_limit = bucket_vecs.len().min(hot_data.vids.len());
                for i in 0..safe_limit {
                    let id = hot_data.vids[i];
                    if target_ids.contains(&id) {
                        let vec = &bucket_vecs[i];
                        let true_dist = crate::math::l2_sq(query, vec);
                        Self::push_to_heap(
                            &mut final_heap,
                            SearchResult {
                                id,
                                distance: true_dist,
                            },
                            k,
                        );
                    }
                }
            } else {
                for (_, res) in disk_candidates.iter().filter(|(b, _)| *b == bid) {
                    Self::push_to_heap(&mut final_heap, res.clone(), k);
                }
            }
        }

        let mut sorted = final_heap.into_vec();
        sorted.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(CmpOrdering::Equal)
        });
        Ok(sorted)
    }

    /// Internal helper for bucket selection (Saturating Density Model). [cite: 70, 120]
    fn get_selected_headers(
        &self,
        query: &[f32],
        target_confidence: f32,
        lambda: f32,
        tau: f32,
    ) -> Vec<BucketHeader> {
        // let guard = epoch::pin();
        // let buckets_map = unsafe { self.buckets.load(Ordering::Acquire, &guard).as_ref() }.unwrap();
        let buckets_map = self.buckets.read();
        if buckets_map.is_empty() {
            return Vec::new();
        }

        if buckets_map.is_empty() {
            return Vec::new();
        }

        let mut clusters: HashMap<Vec<u32>, Vec<&BucketHeader>> = HashMap::new();
        for header in buckets_map.values() {
            let key: Vec<u32> = header.centroid.iter().map(|f| f.to_bits()).collect();
            clusters.entry(key).or_default().push(header);
        }

        let mut candidates: Vec<(Vec<&BucketHeader>, f32)> = clusters
            .into_values()
            .map(|headers| {
                let centroid = &headers[0].centroid;
                let total_count: u32 = headers.iter().map(|h| h.count).sum();
                let dist = crate::math::l2_sq(query, centroid).sqrt();
                let p_geom = (-lambda * dist).exp();
                let reliability = 1.0 - (-(total_count as f32) / tau).exp();
                (headers, p_geom * reliability)
            })
            .collect();

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(CmpOrdering::Equal));
        let total_score: f32 = candidates.iter().map(|(_, p)| p).sum();
        let mut headers = Vec::new();
        let mut acc_conf = 0.0;
        let mut visited_ids = HashSet::new();

        let min_scan = (buckets_map.len() / 10).max(10);

        for (group, score) in candidates {
            let norm_score = if total_score > 1e-6 {
                score / total_score
            } else {
                0.0
            };
            acc_conf += norm_score;
            for h in group {
                if visited_ids.insert(h.id) {
                    headers.push((*h).clone());
                }
            }
            if acc_conf >= target_confidence && headers.len() >= min_scan {
                break;
            }
        }

        // Geometric Guardrail: Force 5 closest buckets to catch drift tails. [cite: 3, 124]
        let mut all_headers: Vec<&BucketHeader> = buckets_map.values().collect();
        all_headers.sort_by(|a, b| {
            crate::math::l2_sq(query, &a.centroid)
                .partial_cmp(&crate::math::l2_sq(query, &b.centroid))
                .unwrap_or(CmpOrdering::Equal)
        });
        for h in all_headers.iter().take(5) {
            if visited_ids.insert(h.id) {
                headers.push((*h).clone());
            }
        }

        for h in &headers {
            h.touch();
        } // EWMA Heat tracking [cite: 80, 521]
        headers
    }

    fn get_bucket_page_id(&self, bucket_id: u32) -> Option<PageId> {
        self.buckets
            .read()
            .get(&bucket_id)
            .map(|h| h.page_id.clone())
    }
    // Helper for Heap
    fn push_to_heap(heap: &mut BinaryHeap<SearchResult>, res: SearchResult, k: usize) {
        if heap.len() < k {
            heap.push(res);
        } else if res.distance <= heap.peek().unwrap().distance {
            heap.pop();
            heap.push(res);
        }
    }

    pub async fn force_register_bucket_with_ids(
        &self,
        id: u32,
        ids: &[u64],
        vectors: &[Vec<f32>],
    ) -> io::Result<()> {
        if vectors.is_empty() {
            return Ok(());
        }

        let deleted_snapshot: HashSet<u64> = self.deleted_ids.read().clone();

        let (valid_ids, valid_vecs): (Vec<u64>, Vec<Vec<f32>>) = ids
            .iter()
            .zip(vectors.iter())
            .filter(|(id, _)| !deleted_snapshot.contains(id))
            .map(|(id, vec)| (*id, vec.clone()))
            .unzip();

        if valid_ids.is_empty() {
            return Ok(());
        }

        let dim = self.config.dim;
        let mut vector_sum = vec![0.0; dim];
        for v in &valid_vecs {
            for i in 0..dim {
                vector_sum[i] += v[i];
            }
        }

        let q_arc = self
            .quantizer
            .read()
            .as_ref()
            .expect("Quantizer missing")
            .clone();

        let mut centroid = vec![0.0; dim];
        for v in &valid_vecs {
            for i in 0..dim {
                centroid[i] += v[i];
            }
        }

        for val in centroid.iter_mut() {
            *val /= valid_vecs.len() as f32;
        }

        let mut data = BucketData {
            codes: AlignedBytes::new(valid_vecs.len() * dim),
            vids: Vec::with_capacity(valid_vecs.len()),
            tombstones: bit_set::BitSet::with_capacity(valid_vecs.len()),
        };
        for (i, v) in valid_vecs.iter().enumerate() {
            let code = q_arc.encode(v);
            data.vids.push(valid_ids[i]);
            for b in code {
                data.codes.push(b);
            }
        }

        let bytes = data.to_bytes(dim)?;
        self.cache.storage().write_page(id, 0, &bytes).await?;
        let page_id = PageId {
            file_id: id,
            offset: 0,
            length: bytes.len() as u32,
        };

        let mut current = self.next_bucket_id.load(Ordering::Relaxed);
        while current <= id {
            match self.next_bucket_id.compare_exchange(
                current,
                id + 1,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current = actual,
            }
        }

        self.update_centroids(|c| {
            let mut new = c.clone();
            if let Some(pos) = new.iter().position(|x| x.id == id) {
                new.remove(pos);
            }
            new.push(CentroidEntry {
                id,
                vector: centroid.clone(),
                active: true,
            });
            new
        });

        self.update_buckets(|b| {
            let mut new = b.clone();
            let header = BucketHeader::new(
                id,
                centroid.clone(),
                valid_ids.len() as u32,
                page_id.clone(),
            );

            *header.stats.vector_sum.write() = vector_sum.clone();
            new.insert(id, header);
            new
        });

        for &vid in &valid_ids {
            let _ = self.kv.put(&vid.to_le_bytes(), &id.to_le_bytes());
        }

        Ok(())
    }

    // --- 4. ZERO-COPY PARTITIONING ---
    /// Partitions the MemTable into buckets using K-Means.
    pub fn partition_memtable(&self, memtable: &MemTable) -> io::Result<Vec<PartitionResult>> {
        info!("Janitor: Partitioning MemTable (Zero-Copy K-Means)...");

        let (ids_guard, data_guard, tomb_guard) = memtable.get_data_guards();
        let num_vectors = ids_guard.len();
        let dim = self.config.dim;

        if num_vectors == 0 {
            return Ok(Vec::new());
        }

        // 1. Filter Deleted Items
        let deleted_snapshot = self.deleted_ids.read();
        let valid_indices: Vec<usize> = (0..num_vectors)
            .filter(|&i| {
                let id = ids_guard[i];
                !tomb_guard.contains(&id) && !deleted_snapshot.contains(&id)
            })
            .collect();

        if valid_indices.is_empty() {
            return Ok(Vec::new());
        }

        // 2. Determine K
        let target_cap = self.config.max_bucket_capacity;
        let count_based_k =
            (valid_indices.len() as f32 / (target_cap as f32 * 0.6)).ceil() as usize;
        let k = count_based_k.max(2);

        // 3. K-Means Training (Zero-Copy Sample)
        let sample_limit = 50_000.min(valid_indices.len());
        // We create a flat buffer sample for training
        let training_samples: Vec<f32> = valid_indices
            .iter()
            .take(sample_limit)
            .flat_map(|&idx| {
                let start = idx * dim;
                data_guard[start..start + dim].iter().copied()
            })
            .collect();

        let trainer = KMeansTrainer::new(k, dim, 10).with_mini_batch(1024);
        let result = trainer.train(&training_samples); // Uses new flat-buffer train()

        // 4. Assignment
        let centroids = result.centroids;
        let assignments: Vec<usize> = valid_indices
            .par_iter()
            .map(|&idx| {
                let start = idx * dim;
                let vec = &data_guard[start..start + dim];
                Self::nearest_centroid_index(vec, &centroids)
            })
            .collect();

        // 5. Group Indices (Metadata Only - No Vector Copying)
        let q_arc = self.get_quantizer().expect("Quantizer missing");

        let mut cluster_indices = vec![Vec::new(); k];
        for (i, &assignment) in assignments.iter().enumerate() {
            if assignment < k {
                cluster_indices[assignment].push(valid_indices[i]);
            }
        }

        let mut partitions = Vec::new();

        for (cluster_idx, indices) in cluster_indices.into_iter().enumerate() {
            if indices.is_empty() {
                continue;
            }

            // Batch Encode SQ8 (Fast)
            let flat_codes: Vec<u8> = indices
                .par_iter()
                .flat_map(|&idx| {
                    let start = idx * dim;
                    let vec = &data_guard[start..start + dim];
                    q_arc.encode(vec)
                })
                .collect();

            // Collect IDs
            let cluster_vids: Vec<u64> = indices.iter().map(|&idx| ids_guard[idx]).collect();

            // ⚡ ZERO-COPY: We do NOT clone the float vectors here.
            // We just pass the 'indices' (pointers) to the persistence layer.

            let bucket_id = self.next_bucket_id.fetch_add(1, Ordering::Relaxed);
            partitions.push(PartitionResult {
                bucket_id,
                ids: cluster_vids,
                indices: indices.clone(), // Just indices!
                codes: flat_codes,
                centroid: centroids[cluster_idx].clone(),
            });
        }

        Ok(partitions)
    }

    /// Internal helper utilizing the SIMD-friendly L2 distance kernel.
    /// This prevents the math itself from being a bottleneck during assignment.
    fn nearest_centroid_index(vec: &[f32], centroids: &[Vec<f32>]) -> usize {
        let mut min_dist = f32::MAX;
        let mut best_idx = 0;
        for (i, c) in centroids.iter().enumerate() {
            // Uses the optimized math from memtable.rs
            let d_sq = crate::math::l2_sq(vec, c);
            if d_sq < min_dist {
                min_dist = d_sq;
                best_idx = i;
            }
        }
        best_idx
    }

    // --- 6. MAINTENANCE OPERATIONS ---

    pub async fn split_and_steal(&self, bucket_id: u32) -> io::Result<MaintenanceStatus> {
        let q_arc = self.quantizer.read().as_ref().unwrap().clone();
        let deleted_snapshot: HashSet<u64> = self.deleted_ids.read().clone();

        // 1. Load Target Bucket Header & Global Centroids
        // ⚡ CHANGE: Use RwLock read() instead of epoch::pin()
        let (header, all_centroids) = {
            let buckets = self.buckets.read();
            let centroids = self.centroids.read();
            let h = match buckets.get(&bucket_id) {
                Some(h) => h.clone(),
                None => return Ok(MaintenanceStatus::SkippedTooSmall),
            };
            (h, centroids.clone())
        }; // Drop locks here

        // 2. Load Data
        let data_arc = match self.cache.get(&header.page_id).await {
            Ok(d) => d,
            Err(_) => return Ok(MaintenanceStatus::SkippedLocked),
        };
        let (raw_vecs, raw_ids) = data_arc.reconstruct(&q_arc);

        let mut vecs = Vec::new();
        let mut ids = Vec::new();
        for (v, id) in raw_vecs.into_iter().zip(raw_ids.into_iter()) {
            if !deleted_snapshot.contains(&id) {
                vecs.push(v);
                ids.push(id);
            }
        }

        if vecs.len() < 10 {
            return Ok(MaintenanceStatus::SkippedTooSmall);
        }

        // 3. Drift Check
        let dim = self.config.dim;
        let mut mean = vec![0.0; dim];
        for v in &vecs {
            for i in 0..dim {
                mean[i] += v[i];
            }
        }
        for mean_val in &mut mean {
            *mean_val /= vecs.len() as f32;
        }

        let drift = crate::math::l2_sq(&mean, &header.centroid).sqrt();
        let capacity_ratio = vecs.len() as f32 / self.config.max_bucket_capacity as f32;

        let capacity_breach = capacity_ratio > 0.8;
        let drift_breach = drift > 0.15;

        if !capacity_breach && !drift_breach {
            return Ok(MaintenanceStatus::SkippedTooSmall);
        }

        // 4. K-Means Split (2-Means)
        // Flatten for training
        let flat_training_data: Vec<f32> = vecs.iter().flatten().copied().collect();
        let trainer = KMeansTrainer::new(2, self.config.dim, 10);
        let result = trainer.train(&flat_training_data); // Uses new signature

        let mut vecs_a = Vec::new();
        let mut ids_a = Vec::new();
        let mut vecs_b = Vec::new();
        let mut ids_b = Vec::new();

        for (i, &c_idx) in result.assignments.iter().enumerate() {
            if c_idx == 0 {
                vecs_a.push(vecs[i].clone());
                ids_a.push(ids[i]);
            } else {
                vecs_b.push(vecs[i].clone());
                ids_b.push(ids[i]);
            }
        }

        if vecs_a.len() < 5 || vecs_b.len() < 5 {
            return Ok(MaintenanceStatus::SkippedSingularity);
        }

        // 5. Neighbor Stealing (Budgeted)
        let neighbors = {
            let mut candidates: Vec<(u32, f32)> = all_centroids
                .iter()
                .filter(|c| c.active && c.id != bucket_id)
                .map(|c| (c.id, crate::math::l2_sq(&header.centroid, &c.vector)))
                .collect();
            candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            candidates
                .into_iter()
                .take(3)
                .map(|(id, _)| id)
                .collect::<Vec<u32>>()
        };

        let mut modified_neighbors = HashMap::new();
        let mut budget = 0;
        const MAX_STEAL_BUDGET: usize = 200;

        for nid in neighbors {
            if budget >= MAX_STEAL_BUDGET {
                break;
            }

            // Load Neighbor Header
            // ⚡ CHANGE: Use RwLock read()
            let n_header = {
                let buckets = self.buckets.read();
                match buckets.get(&nid) {
                    Some(h) => h.clone(),
                    None => continue,
                }
            };

            // Load Neighbor Data
            let n_data = match self.cache.get(&n_header.page_id).await {
                Ok(d) => d,
                Err(_) => continue,
            };

            let (raw_n_vecs, raw_n_ids) = n_data.reconstruct(&q_arc);
            let mut n_vecs = Vec::new();
            let mut n_ids = Vec::new();

            for (v, id) in raw_n_vecs.into_iter().zip(raw_n_ids.into_iter()) {
                if !deleted_snapshot.contains(&id) {
                    n_vecs.push(v);
                    n_ids.push(id);
                }
            }

            // Check for defectors
            let mut steal_indices = Vec::new();
            let mut stolen_items = Vec::new();

            for (i, vec) in n_vecs.iter().enumerate() {
                if budget >= MAX_STEAL_BUDGET {
                    break;
                }

                let d_curr = crate::math::l2_sq(vec, &n_header.centroid);
                let d_a = crate::math::l2_sq(vec, &result.centroids[0]);
                let d_b = crate::math::l2_sq(vec, &result.centroids[1]);

                // If point is significantly closer to A or B than current parent
                if d_a.min(d_b) < (d_curr - 0.025) {
                    budget += 1;
                    steal_indices.push(i);
                    stolen_items.push((vec.clone(), n_ids[i], d_a < d_b));
                }
            }

            if !steal_indices.is_empty() {
                // Remove stolen items from neighbor vectors
                steal_indices.sort_unstable_by(|a, b| b.cmp(a)); // Descending order
                for idx in steal_indices {
                    n_vecs.swap_remove(idx);
                    n_ids.swap_remove(idx);
                }

                // Add stolen items to A or B
                for (vec, id, is_a) in stolen_items {
                    if is_a {
                        vecs_a.push(vec);
                        ids_a.push(id);
                    } else {
                        vecs_b.push(vec);
                        ids_b.push(id);
                    }
                }

                // Queue neighbor for rewrite
                modified_neighbors.insert(nid, (n_vecs, n_ids));
            }
        }

        // 6. Write New Buckets & Updates

        let calc_centroid = |vecs: &[Vec<f32>], dim: usize| -> Vec<f32> {
            let mut c = vec![0.0; dim];
            if vecs.is_empty() {
                return c;
            }
            for v in vecs {
                for i in 0..dim {
                    c[i] += v[i];
                }
            }
            for x in c.iter_mut() {
                *x /= vecs.len() as f32;
            }
            c
        };

        let calc_sum = |vecs: &[Vec<f32>], dim: usize| -> Vec<f32> {
            let mut s = vec![0.0; dim];
            for v in vecs {
                for i in 0..dim {
                    s[i] += v[i];
                }
            }
            s
        };

        // Recalculate Centroids for A & B (Post-Steal)
        let centroid_a = calc_centroid(&vecs_a, dim);
        let centroid_b = calc_centroid(&vecs_b, dim);
        let sum_a = calc_sum(&vecs_a, dim);
        let sum_b = calc_sum(&vecs_b, dim);

        // Helper to write a page
        let write_page =
            async |bid: u32, vs: &[Vec<f32>], is: &[u64]| -> io::Result<(PageId, u32)> {
                let mut data = BucketData {
                    codes: crate::aligned::AlignedBytes::new(vs.len() * dim),
                    vids: Vec::with_capacity(vs.len()),
                    tombstones: bit_set::BitSet::with_capacity(vs.len()),
                };
                for (i, v) in vs.iter().enumerate() {
                    let code = q_arc.encode(v);
                    data.vids.push(is[i]);
                    for b in code {
                        data.codes.push(b);
                    }
                }
                let bytes = data.to_bytes(dim)?;
                self.cache.storage().write_page(bid, 0, &bytes).await?;
                Ok((
                    PageId {
                        file_id: bid,
                        offset: 0,
                        length: bytes.len() as u32,
                    },
                    vs.len() as u32,
                ))
            };

        let id_a = self.next_bucket_id.fetch_add(1, Ordering::Relaxed);
        let id_b = self.next_bucket_id.fetch_add(1, Ordering::Relaxed);

        let (page_a, count_a) = write_page(id_a, &vecs_a, &ids_a).await?;
        let (page_b, count_b) = write_page(id_b, &vecs_b, &ids_b).await?;

        // Process Modified Neighbors
        let mut neighbor_updates = Vec::new();
        for (nid, (vs, is)) in modified_neighbors {
            let (p, c) = write_page(nid, &vs, &is).await?;
            let new_c = calc_centroid(&vs, dim);
            let new_s = calc_sum(&vs, dim);
            neighbor_updates.push((nid, p, c, new_c, new_s));
        }

        // 7. Atomic Metadata Update

        // Update Centroids List
        self.update_centroids(|c| {
            let mut new = c.clone();
            if let Some(pos) = new.iter().position(|x| x.id == bucket_id) {
                new.remove(pos);
            }
            // Add new split buckets
            new.push(CentroidEntry {
                id: id_a,
                vector: centroid_a.clone(),
                active: true,
            });
            new.push(CentroidEntry {
                id: id_b,
                vector: centroid_b.clone(),
                active: true,
            });

            // Update modified neighbors
            for (nid, _, _, new_c, _) in &neighbor_updates {
                if let Some(entry) = new.iter_mut().find(|x| x.id == *nid) {
                    entry.vector = new_c.clone();
                }
            }
            new
        });

        // Update Buckets Map
        self.update_buckets(|b| {
            let mut new = b.clone();
            new.remove(&bucket_id);

            let header_a = BucketHeader::new(id_a, centroid_a.clone(), count_a, page_a.clone());
            *header_a.stats.vector_sum.write() = sum_a.clone();
            new.insert(id_a, header_a);

            let header_b = BucketHeader::new(id_b, centroid_b.clone(), count_b, page_b.clone());
            *header_b.stats.vector_sum.write() = sum_b.clone();
            new.insert(id_b, header_b);

            for (nid, p, c, new_c, new_s) in &neighbor_updates {
                if let Some(h) = new.get_mut(nid) {
                    h.count = *c;
                    h.page_id = p.clone();
                    h.centroid = new_c.clone(); // ⚡ Update Centroid
                    *h.stats.vector_sum.write() = new_s.clone(); // ⚡ Update Sum
                }
            }
            new
        });

        // Update KV Pointers
        let update_kv = |ids: &[u64], bid: u32| {
            for id in ids {
                let _ = self.kv.put(&id.to_le_bytes(), &bid.to_le_bytes());
            }
        };
        update_kv(&ids_a, id_a);
        update_kv(&ids_b, id_b);

        Ok(MaintenanceStatus::Completed)
    }

    pub async fn scatter_merge(&self, zombie_id: u32) -> io::Result<MaintenanceStatus> {
        let q_arc = self.quantizer.read().as_ref().unwrap().clone();
        let deleted_snapshot: HashSet<u64> = self.deleted_ids.read().clone();

        // 1. Load Zombie Header
        // ⚡ CHANGE: Use RwLock read()
        let (z_header, all_centroids) = {
            let buckets = self.buckets.read();
            let centroids = self.centroids.read();
            let h = match buckets.get(&zombie_id) {
                Some(h) => h.clone(),
                None => return Ok(MaintenanceStatus::SkippedTooSmall),
            };
            (h, centroids.clone())
        };

        // 2. Load Zombie Data
        let data_arc = match self.cache.get(&z_header.page_id).await {
            Ok(d) => d,
            Err(_) => return Ok(MaintenanceStatus::SkippedLocked),
        };

        // 3. Reconstruct & Filter Survivors
        let (raw_vecs, raw_ids) = data_arc.reconstruct(&q_arc);
        let mut orphans_vec = Vec::new();
        let mut orphans_id = Vec::new();

        for (v, id) in raw_vecs.into_iter().zip(raw_ids.into_iter()) {
            if !deleted_snapshot.contains(&id) {
                orphans_vec.push(v);
                orphans_id.push(id);
            }
        }

        // 4. Budget Enforcement
        if orphans_vec.len() > 50 {
            return Ok(MaintenanceStatus::SkippedTooSmall);
        }

        if orphans_vec.is_empty() {
            self.atomic_remove_bucket(zombie_id);
            return Ok(MaintenanceStatus::Completed);
        }

        // 5. Find Neighbors (Top-3)
        let mut candidates: Vec<(u32, f32)> = all_centroids
            .iter()
            .filter(|c| c.active && c.id != zombie_id)
            .map(|c| (c.id, crate::math::l2_sq(&z_header.centroid, &c.vector)))
            .collect();

        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let neighbor_ids: Vec<u32> = candidates.into_iter().take(3).map(|(id, _)| id).collect();

        if neighbor_ids.is_empty() {
            // Fallback to L0 if no neighbors exist
            for (i, vec) in orphans_vec.iter().enumerate() {
                self.insert(orphans_id[i], vec)?;
            }
            self.atomic_remove_bucket(zombie_id);
            return Ok(MaintenanceStatus::Completed);
        }

        // 6. Assign Orphans
        let mut adoption_map: HashMap<u32, Vec<(u64, Vec<f32>)>> = HashMap::new();
        // ⚡ SAFETY THRESHOLD
        const MAX_MERGE_DIST_SQ: f32 = 5000.0;

        for (i, vec) in orphans_vec.iter().enumerate() {
            let mut best_neighbor = None;
            let mut min_dist = f32::MAX;

            for &nid in &neighbor_ids {
                if let Some(c) = all_centroids.iter().find(|c| c.id == nid) {
                    let d = crate::math::l2_sq(vec, &c.vector);
                    if d < min_dist {
                        min_dist = d;
                        best_neighbor = Some(nid);
                    }
                }
            }

            if let Some(bid) = best_neighbor
                && min_dist < MAX_MERGE_DIST_SQ
            {
                adoption_map
                    .entry(bid)
                    .or_default()
                    .push((orphans_id[i], vec.clone()));
            } else {
                // Too far -> Send back to L0
                self.insert(orphans_id[i], vec)?;
            }
        }

        // 7. Rewrite Neighbors
        for (target_id, orphans) in adoption_map {
            // A. Load Neighbor
            // ⚡ CHANGE: Use RwLock read()
            let t_header = {
                let buckets = self.buckets.read();
                match buckets.get(&target_id) {
                    Some(h) => h.clone(),
                    None => {
                        for (id, v) in orphans {
                            self.insert(id, &v)?;
                        }
                        continue;
                    }
                }
            };

            // B. Load Data
            let t_data = match self.cache.get(&t_header.page_id).await {
                Ok(d) => d,
                Err(_) => {
                    for (id, v) in orphans {
                        self.insert(id, &v)?;
                    }
                    continue;
                }
            };

            // C. Merge
            let (raw_t_vecs, raw_t_ids) = t_data.reconstruct(&q_arc);
            let mut vs = Vec::new();
            let mut is = Vec::new();

            for (v, id) in raw_t_vecs.into_iter().zip(raw_t_ids.into_iter()) {
                if !deleted_snapshot.contains(&id) {
                    vs.push(v);
                    is.push(id);
                }
            }
            for (id, v) in &orphans {
                vs.push(v.clone());
                is.push(*id);
            }

            // D. Encode
            let dim = self.config.dim;
            let mut new_data = BucketData {
                codes: crate::aligned::AlignedBytes::new(vs.len() * dim),
                vids: Vec::with_capacity(vs.len()),
                tombstones: bit_set::BitSet::with_capacity(vs.len()),
            };
            for (i, v) in vs.iter().enumerate() {
                let code = q_arc.encode(v);
                new_data.vids.push(is[i]);
                for b in code {
                    new_data.codes.push(b);
                }
            }

            // Recalculate Sum & Centroid
            let mut new_sum = vec![0.0; dim];
            for v in &vs {
                for i in 0..dim {
                    new_sum[i] += v[i];
                }
            }
            let new_count = vs.len() as f32;
            let mut new_centroid = vec![0.0; dim];
            if new_count > 0.0 {
                for i in 0..dim {
                    new_centroid[i] = new_sum[i] / new_count;
                }
            }

            let bytes = new_data.to_bytes(dim)?;
            self.cache
                .storage()
                .write_page(target_id, 0, &bytes)
                .await?;

            // E. Update Metadata
            self.update_buckets(|current| {
                let mut new = current.clone();
                if let Some(h) = new.get_mut(&target_id) {
                    h.count = vs.len() as u32;
                    h.page_id.length = bytes.len() as u32;
                    // ⚡ UPDATE
                    h.centroid = new_centroid.clone();
                    *h.stats.vector_sum.write() = new_sum.clone();
                }
                new
            });

            self.update_centroids(|c| {
                let mut new = c.clone();
                if let Some(entry) = new.iter_mut().find(|x| x.id == target_id) {
                    entry.vector = new_centroid.clone();
                }
                new
            });

            // F. Update KV
            for (id, _) in orphans {
                let _ = self.kv.put(&id.to_le_bytes(), &target_id.to_le_bytes());
            }
        }

        self.atomic_remove_bucket(zombie_id);
        Ok(MaintenanceStatus::Completed)
    }

    fn atomic_remove_bucket(&self, id: u32) {
        self.update_centroids(|c| {
            let mut new = c.clone();
            if let Some(pos) = new.iter().position(|x| x.id == id) {
                new.remove(pos);
            }
            new
        });
        self.update_buckets(|b| {
            let mut new = b.clone();
            new.remove(&id);
            new
        });
    }

    pub fn set_quantizer(&self, q: Quantizer) {
        *self.quantizer.write() = Some(Arc::new(q));
    }

    pub fn get_all_bucket_headers(&self) -> Vec<BucketHeader> {
        self.buckets.read().values().cloned().collect()
    }

    pub fn allocate_next_bucket_id(&self) -> u32 {
        self.next_bucket_id.fetch_add(1, Ordering::Relaxed)
    }

    pub fn memtable_len(&self) -> usize {
        self.memtable.read().len()
    }

    pub fn get_quantizer(&self) -> Option<Arc<Quantizer>> {
        self.quantizer.read().clone()
    }

    // ⚡ NEW: The Scavenger Logic (Copy-On-Write Compaction)
    pub async fn compact_bucket(&self, bucket_id: u32) -> io::Result<Option<Vec<u64>>> {
        // 1. Snapshot Global Deletes
        let global_tombstones = self.deleted_ids.read().clone();

        // 2. Load Header
        // ⚡ CHANGE: Use RwLock read() instead of epoch::pin()
        let (header, centroid) = {
            let buckets = self.buckets.read();
            match buckets.get(&bucket_id) {
                Some(h) => (h.clone(), h.centroid.clone()),
                None => return Ok(None), // Bucket already gone/merged
            }
        };

        // 3. Load Data (Async I/O)
        let data = match self.cache.get(&header.page_id).await {
            Ok(d) => d,
            Err(e) => {
                error!("Compaction failed to load bucket {}: {}", bucket_id, e);
                return Ok(None);
            }
        };

        // 4. Reconstruct & Filter
        let q_arc = self.quantizer.read().as_ref().unwrap().clone();
        let (raw_vecs, raw_ids) = data.reconstruct(&q_arc);

        let mut live_vecs = Vec::new();
        let mut live_ids = Vec::new();
        let mut compacted_ids = Vec::new();

        for (vec, id) in raw_vecs.into_iter().zip(raw_ids.into_iter()) {
            if global_tombstones.contains(&id) {
                compacted_ids.push(id);
            } else {
                live_vecs.push(vec);
                live_ids.push(id);
            }
        }

        if compacted_ids.is_empty() {
            return Ok(None);
        }

        // 5. Handle "Empty Bucket" Case
        if live_vecs.is_empty() {
            self.atomic_remove_bucket(bucket_id);
            for id in compacted_ids.iter() {
                let _ = self.kv.remove(&id.to_le_bytes());
            }
            return Ok(Some(compacted_ids));
        }

        // 6. Write New Bucket (Copy-on-Write)
        let new_id = self.next_bucket_id.fetch_add(1, Ordering::Relaxed);
        let dim = self.config.dim;

        let mut new_data = BucketData {
            codes: crate::aligned::AlignedBytes::new(live_vecs.len() * dim),
            vids: Vec::with_capacity(live_vecs.len()),
            tombstones: bit_set::BitSet::with_capacity(live_vecs.len()),
        };

        for (i, v) in live_vecs.iter().enumerate() {
            let code = q_arc.encode(v);
            new_data.vids.push(live_ids[i]);
            for b in code {
                new_data.codes.push(b);
            }
        }

        let bytes = new_data.to_bytes(dim)?;
        self.cache.storage().write_page(new_id, 0, &bytes).await?;

        let page_id = PageId {
            file_id: new_id,
            offset: 0,
            length: bytes.len() as u32,
        };

        // 7. Atomic Swap
        self.update_buckets(|current| {
            let mut new_map = current.clone();
            new_map.remove(&bucket_id);
            new_map.insert(
                new_id,
                BucketHeader::new(
                    new_id,
                    centroid.clone(),
                    live_vecs.len() as u32,
                    page_id.clone(),
                ),
            );
            new_map
        });

        self.update_centroids(|current| {
            let mut new_list = current.clone();
            if let Some(pos) = new_list.iter().position(|c| c.id == bucket_id) {
                new_list.remove(pos);
            }
            new_list.push(crate::index::CentroidEntry {
                id: new_id,
                vector: centroid.clone(),
                active: true,
            });
            new_list
        });

        // 8. Update KV
        for &vid in &live_ids {
            let _ = self.kv.put(&vid.to_le_bytes(), &new_id.to_le_bytes());
        }

        Ok(Some(compacted_ids))
    }

    /// ⚡ STEP 3: REGISTER PARTITIONS (State Update)
    /// Called AFTER the segment is safely on S3.
    /// Updates in-memory maps to point to the new data.
    pub async fn register_partitions(
        &self,
        partitions: &[PartitionResult],
        segment_id: &str,
        offsets: &HashMap<u32, (u64, u32)>,
    ) -> io::Result<()> {
        let dim = self.config.dim; // Need dim for reconstruction
        let segment_filename = format!("segment_{}.drift", segment_id);

        for p in partitions {
            let (off, len) = offsets
                .get(&p.bucket_id)
                .ok_or(io::Error::other("Missing offset"))?;

            // 1. Register the File Mapping
            // This informs TieredPageManager -> Remote that Bucket X is in Segment Y.
            self.cache
                .storage()
                .register_file(p.bucket_id, PathBuf::from(&segment_filename));

            // 2. Cache Warming (Optional but recommended)
            // Since we have the data in memory right now, we can write it to the Local Cache immediately.
            // This avoids a read-back from S3 later.
            // Because we decoupled Local/Remote, this writes to `cache/ID.bin` safely.
            let bucket_data = BucketData {
                codes: AlignedBytes::from_slice(&p.codes),
                vids: p.ids.clone(),
                tombstones: bit_set::BitSet::with_capacity(p.ids.len()),
            };
            let bytes = bucket_data.to_bytes(dim)?;

            // Write to Local Cache (offset 0)
            self.cache
                .storage()
                .write_page(p.bucket_id, 0, &bytes)
                .await?;

            // 3. Update Maps
            // PageId points to the REMOTE location (Source of Truth)
            let page_id = PageId {
                file_id: p.bucket_id,
                offset: *off,
                length: *len,
            };

            self.update_buckets(|b| {
                let mut new = b.clone();
                new.insert(
                    p.bucket_id,
                    BucketHeader::new(
                        p.bucket_id,
                        p.centroid.clone(),
                        p.ids.len() as u32,
                        page_id.clone(),
                    ),
                );
                new
            });

            self.update_centroids(|c| {
                let mut new = c.clone();
                new.push(CentroidEntry {
                    id: p.bucket_id,
                    vector: p.centroid.clone(),
                    active: true,
                });
                new
            });

            for &vid in &p.ids {
                let _ = self.kv.put(&vid.to_le_bytes(), &p.bucket_id.to_le_bytes());
            }
        }
        Ok(())
    }

    // Helper to find WHERE a specific vector ID lives
    pub fn locate_vector(&self, id: u64) -> Option<u32> {
        let id_bytes = id.to_le_bytes();
        if let Ok(Some(val)) = self.kv.get(&id_bytes) {
            return Some(u32::from_le_bytes(val.try_into().unwrap()));
        }
        // Check MemTable
        // ...
        None
    }

    /// Also persists the training data as the initial L1 segments.
    pub async fn train(&self, samples: &[Vec<f32>]) -> std::io::Result<()> {
        info!("Index: Starting Public Train API");
        assert!(!samples.is_empty(), "Empty training set");

        let dim = self.config.dim;
        let n_centroids = self.config.num_centroids;

        // 1. FLATTEN DATA (Adapt for new Flat-Buffer Math)
        // Since input is &[Vec<f32>], we must flatten it once to use our optimized math kernels.
        // This allocation is acceptable for the "Training" phase which is rare/one-time.
        let flat_samples: Vec<f32> = samples.iter().flatten().copied().collect();

        // 2. OFFLOAD MATH (Quantizer + K-Means)
        let (q, result) = tokio::task::spawn_blocking(move || {
            // A. Train Quantizer
            // We use the new signature: train(flat_data, dim)
            let q = Quantizer::train(&flat_samples, dim);

            // B. Train K-Means
            // We use the new signature: train(flat_data)
            let batch_size = 1024.min(flat_samples.len() / dim);
            let trainer = KMeansTrainer::new(n_centroids, dim, 20).with_mini_batch(batch_size);

            info!("Index: Executing K-Means...");
            let res = trainer.train(&flat_samples);

            (q, res)
        })
        .await
        .map_err(|e| std::io::Error::other(format!("Training Task Failed: {}", e)))?;

        // 3. UPDATE QUANTIZER (RwLock)
        let q_arc = Arc::new(q);
        {
            let mut q_guard = self.quantizer.write();
            *q_guard = Some(q_arc.clone());
        }

        info!("Index: Math Complete. Writing Initial Segments...");

        // 4. PARTITION & PERSIST
        // We group the original 'samples' based on the K-Means result.
        let mut cluster_indices = vec![Vec::new(); n_centroids];
        for (i, &assignment) in result.assignments.iter().enumerate() {
            if assignment < n_centroids {
                cluster_indices[assignment].push(i);
            }
        }

        let mut new_centroids = Vec::with_capacity(n_centroids);
        let mut new_buckets = HashMap::with_capacity(n_centroids);

        // Reset IDs
        self.next_bucket_id.store(0, Ordering::Relaxed);

        for (cluster_idx, sample_indices) in cluster_indices.into_iter().enumerate() {
            if sample_indices.is_empty() {
                continue;
            }

            let id = self.next_bucket_id.fetch_add(1, Ordering::Relaxed);
            let center = result.centroids[cluster_idx].clone();

            // A. Batch Encode (SQ8) using Rayon
            // We read from the original 'samples' ragged array here since we have indices into it
            let flat_codes: Vec<u8> = sample_indices
                .par_iter()
                .flat_map(|&idx| q_arc.encode(&samples[idx]))
                .collect();

            // B. Collect IDs (Training data usually implies ID = Index, or we generate them)
            // For simplicity in train(), we assume the index in 'samples' is the ID.
            let vids: Vec<u64> = sample_indices.iter().map(|&idx| idx as u64).collect();

            // C. Create Bucket Data
            let bucket_data = BucketData {
                codes: AlignedBytes::from_slice(&flat_codes),
                vids: vids.clone(),
                tombstones: bit_set::BitSet::with_capacity(sample_indices.len()),
            };

            // D. Write Page to Disk
            let bytes = bucket_data.to_bytes(dim)?;
            self.cache.storage().write_page(id, 0, &bytes).await?;

            let page_id = PageId {
                file_id: id,
                offset: 0,
                length: bytes.len() as u32,
            };

            // E. Calculate Vector Sum (For Drift Tracking)
            let mut vector_sum = vec![0.0; dim];
            for &idx in &sample_indices {
                let v = &samples[idx];
                for d in 0..dim {
                    vector_sum[d] += v[d];
                }
            }

            // F. Create Header
            let header = BucketHeader::new(id, center.clone(), vids.len() as u32, page_id);
            *header.stats.vector_sum.write() = vector_sum;

            new_buckets.insert(id, header);
            new_centroids.push(CentroidEntry {
                id,
                vector: center,
                active: true,
            });

            // G. Update KV Store
            for &vid in &vids {
                let _ = self.kv.put(&vid.to_le_bytes(), &id.to_le_bytes());
            }
        }

        // 5. ATOMIC STATE SWAP (RwLock)
        {
            let mut c_guard = self.centroids.write();
            *c_guard = Arc::new(new_centroids);
        }
        {
            let mut b_guard = self.buckets.write();
            *b_guard = Arc::new(new_buckets);
        }

        info!("Index: Training and Bootstrapping Complete.");
        Ok(())
    }
}
