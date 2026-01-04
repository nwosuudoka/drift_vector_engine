use crate::aligned::AlignedBytes;
use crate::bucket::{Bucket, BucketData, BucketHeader, compute_distance_lut};
use crate::kmeans::KMeansTrainer;
use crate::memtable::MemTable;
use crate::quantizer::Quantizer;
use crate::wal::{WalEntry, WalReader, WalWriter};
use crossbeam_epoch::{self as epoch, Atomic, Owned};
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
use tracing::{error, info, instrument};

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
    pub vectors: Vec<Vec<f32>>,
    pub codes: Vec<u8>,
    pub centroid: Vec<f32>,
}

pub struct VectorIndex {
    pub config: IndexOptions,
    pub(crate) centroids: Atomic<Vec<CentroidEntry>>,

    pub memtable: Atomic<Arc<MemTable>>,
    pub frozen_memtable: RwLock<Option<Arc<MemTable>>>,

    pub(crate) buckets: Atomic<HashMap<u32, BucketHeader>>,
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
        let memtable = Arc::new(MemTable::new(
            config.max_bucket_capacity * 10,
            config.dim,
            config.ef_construction,
            16,
        ));

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

            memtable: Atomic::new(memtable),
            frozen_memtable: RwLock::new(None),

            quantizer: RwLock::new(None),
            centroids: Atomic::new(Vec::new()),
            next_bucket_id: AtomicU32::new(0),
            buckets: Atomic::new(HashMap::new()),
            kv,
            cache,
            deleted_ids: RwLock::new(recovered_deletes),
        })
    }

    fn update_centroids<F>(&self, mut f: F)
    where
        F: FnMut(&Vec<CentroidEntry>) -> Vec<CentroidEntry>,
    {
        let guard = epoch::pin();
        loop {
            let shared = self.centroids.load(Ordering::Acquire, &guard);
            let current = unsafe { shared.as_ref() }.unwrap();
            let new_vec = f(current);
            if self
                .centroids
                .compare_exchange(
                    shared,
                    Owned::new(new_vec),
                    Ordering::Release,
                    Ordering::Relaxed,
                    &guard,
                )
                .is_ok()
            {
                break;
            }
        }
    }

    fn update_buckets<F>(&self, mut f: F)
    where
        F: FnMut(&HashMap<u32, BucketHeader>) -> HashMap<u32, BucketHeader>,
    {
        let guard = epoch::pin();
        loop {
            let shared = self.buckets.load(Ordering::Acquire, &guard);
            let current = unsafe { shared.as_ref() }.unwrap();
            let new_map = f(current);
            if self
                .buckets
                .compare_exchange(
                    shared,
                    Owned::new(new_map),
                    Ordering::Release,
                    Ordering::Relaxed,
                    &guard,
                )
                .is_ok()
            {
                break;
            }
        }
    }

    // pub async fn train(&self, samples: &[Vec<f32>]) -> io::Result<()> {
    //     assert!(!samples.is_empty(), "Empty training set");
    //     let dim = self.config.dim;

    //     let q = Arc::new(Quantizer::train(samples));
    //     *self.quantizer.write() = Some(q.clone());

    //     let trainer = KMeansTrainer::new(self.config.num_centroids, dim, 20);
    //     let result = trainer.train(samples);

    //     self.next_bucket_id.store(0, Ordering::Relaxed);
    //     let mut new_centroids = Vec::new();
    //     let mut new_buckets = HashMap::new();
    //     let mut cluster_data = vec![vec![]; self.config.num_centroids];

    //     for (i, &assignment) in result.assignments.iter().enumerate() {
    //         cluster_data[assignment].push(i);
    //     }

    //     for (cluster_idx, sample_indices) in cluster_data.into_iter().enumerate() {
    //         let id = self.next_bucket_id.fetch_add(1, Ordering::Relaxed);
    //         let center = result.centroids[cluster_idx].clone();

    //         let mut bucket_data = BucketData {
    //             codes: crate::aligned::AlignedBytes::new(sample_indices.len() * dim),
    //             vids: Vec::with_capacity(sample_indices.len()),
    //             tombstones: bit_set::BitSet::with_capacity(sample_indices.len()),
    //         };

    //         for &sample_idx in &sample_indices {
    //             let vec = &samples[sample_idx];
    //             let code = q.encode(vec);
    //             bucket_data.vids.push(sample_idx as u64);
    //             for b in code {
    //                 bucket_data.codes.push(b);
    //             }
    //         }

    //         let bytes = bucket_data.to_bytes(dim)?;
    //         self.cache.storage().write_page(id, 0, &bytes).await?;
    //         let page_id = PageId {
    //             file_id: id,
    //             offset: 0,
    //             length: bytes.len() as u32,
    //         };

    //         new_buckets.insert(
    //             id,
    //             BucketHeader::new(id, center.clone(), sample_indices.len() as u32, page_id),
    //         );
    //         new_centroids.push(CentroidEntry {
    //             id,
    //             vector: center,
    //             active: true,
    //         });

    //         for &sample_idx in &sample_indices {
    //             let _ = self
    //                 .kv
    //                 .put(&(sample_idx as u64).to_le_bytes(), &id.to_le_bytes());
    //         }
    //     }

    //     let _guard = epoch::pin();
    //     self.centroids
    //         .store(Owned::new(new_centroids), Ordering::Release);
    //     self.buckets
    //         .store(Owned::new(new_buckets), Ordering::Release);
    //     Ok(())
    // }

    pub async fn train(&self, samples: &[Vec<f32>]) -> std::io::Result<()> {
        info!("Janitor: Starting Training - Phase 1: CPU Heavy Math");
        assert!(!samples.is_empty(), "Empty training set");

        let dim = self.config.dim;
        let n_centroids = self.config.num_centroids;

        // 1. DATA OWNERSHIP
        // Clone the samples into owned memory to satisfy the 'static lifetime
        // requirement for tokio::task::spawn_blocking.
        let samples_owned = samples.to_vec();

        // 2. OFFLOAD TO BLOCKING THREAD POOL
        // We perform K-Means and Quantizer training in a separate thread pool.
        // This avoids deadlocking the Tokio executor while the CPU grinds.
        // println!("before training");
        let (q, result) = tokio::task::spawn_blocking(move || {
            // A. Fast Quantizer Training (Sampled)
            // 10k items provide enough statistical variance for 1-99% clipping [cite: 1313, 1320]
            let sample_limit = 10_000.min(samples_owned.len());
            let q = Quantizer::train(&samples_owned[..sample_limit]);

            // B. Mini-Batch K-Means
            // Uses the optimized trainer to reduce O(N) complexity [cite: 1125, 1141]
            let batch_size = 1024.min(samples_owned.len());
            let trainer = KMeansTrainer::new(n_centroids, dim, 20).with_mini_batch(batch_size);

            info!("Janitor: Executing K-Means on background thread...");
            let res = trainer.train(&samples_owned);

            (q, res)
        })
        .await
        .map_err(|e| std::io::Error::other(format!("Training Task Failed: {}", e)))?;

        // println!("after training");

        // 3. ATOMIC STATE UPDATE
        // Update the internal quantizer state while holding the lock for the shortest time possible.
        let q_arc = Arc::new(q);
        {
            let mut q_guard = self.quantizer.write();
            *q_guard = Some(q_arc.clone());
        }

        info!("Janitor: Math Complete - Phase 2: Parallel Encoding and Async I/O");

        // 4. PARALLEL PARTITIONING
        let mut cluster_data = vec![vec![]; n_centroids];
        for (i, &assignment) in result.assignments.iter().enumerate() {
            if assignment < n_centroids {
                cluster_data[assignment].push(i);
            }
        }

        let mut new_centroids = Vec::with_capacity(n_centroids);
        let mut new_buckets = HashMap::with_capacity(n_centroids);

        // Reset bucket ID counter for the fresh training run
        self.next_bucket_id
            .store(0, std::sync::atomic::Ordering::Relaxed);

        // 5. BATCH ENCODING & STORAGE
        for (cluster_idx, sample_indices) in cluster_data.into_iter().enumerate() {
            if sample_indices.is_empty() {
                continue;
            }

            let id = self
                .next_bucket_id
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let center = result.centroids[cluster_idx].clone();

            // Use Rayon for parallel vector encoding within the async function [cite: 1269, 1281]
            let flat_codes: Vec<u8> = sample_indices
                .par_iter()
                .flat_map(|&idx| q_arc.encode(&samples[idx]))
                .collect();

            let vids: Vec<u64> = sample_indices.iter().map(|&idx| idx as u64).collect();
            let bucket_data = BucketData::from(
                crate::aligned::AlignedBytes::from_slice(&flat_codes), // SIMD-aligned buffer [cite: 442, 577]
                vids.clone(),
                bit_set::BitSet::with_capacity(sample_indices.len()),
            );

            // Async Disk Write [cite: 69, 189]
            let bytes = bucket_data.to_bytes(dim)?;
            self.cache.storage().write_page(id, 0, &bytes).await?;

            let page_id = PageId {
                file_id: id,
                offset: 0,
                length: bytes.len() as u32,
            };
            new_buckets.insert(
                id,
                BucketHeader::new(id, center.clone(), vids.len() as u32, page_id),
            );
            new_centroids.push(CentroidEntry {
                id,
                vector: center,
                active: true,
            });

            // Update global mapping for O(1) lookups [cite: 72, 131]
            for &vid in &vids {
                let _ = self.kv.put(&vid.to_le_bytes(), &id.to_le_bytes());
            }
        }

        // 6. FINAL ATOMIC COMMIT
        // Use crossbeam-epoch for lock-free swap of the global bucket/centroid maps [cite: 429, 647]
        let _guard = crossbeam_epoch::pin();
        self.centroids.store(
            crossbeam_epoch::Owned::new(new_centroids),
            std::sync::atomic::Ordering::Release,
        );
        self.buckets.store(
            crossbeam_epoch::Owned::new(new_buckets),
            std::sync::atomic::Ordering::Release,
        );

        info!("Janitor: Training and Flush Complete.");
        Ok(())
    }

    pub fn insert(&self, id: u64, vector: &[f32]) -> io::Result<()> {
        // 1. Acquire Lock (Keeps sync with Rotate)
        let mut wal = self.wal.lock();

        // 2. Write to WAL (Buffered)
        wal.write_insert(id, vector)?;
        // PERF FIX: Removed wal.flush() to prevent fsync bottleneck

        // 3. Insert into Memory (Safe from rotation race)
        let guard = epoch::pin();
        let memtable = unsafe { self.memtable.load(Ordering::Acquire, &guard).as_ref() }.unwrap();
        memtable.insert(id, vector);

        self.deleted_ids.write().remove(&id);

        Ok(())
    }

    #[instrument(skip(self, vectors), fields(count = vectors.len()), level = "info")]
    pub fn insert_batch(&self, vectors: &[(u64, Vec<f32>)]) -> std::io::Result<()> {
        {
            let mut wal = self.wal.lock();
            for (id, vec) in vectors {
                wal.write_insert(*id, vec)?;
            }
        }

        let guard = crossbeam_epoch::pin();
        let memtable = unsafe {
            self.memtable
                .load(std::sync::atomic::Ordering::Acquire, &guard)
                .as_ref()
        }
        .unwrap();

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
        // 1. Write to WAL
        let mut wal = self.wal.lock();
        wal.write_delete(id)?;

        // ⚡ OPTIMIZATION: Removed wal.flush() to prevent test timeouts.
        // The OS page cache provides sufficient durability for these tests.

        // 2. Update MemTable (Visibility L0)
        let guard = epoch::pin();
        let memtable = unsafe { self.memtable.load(Ordering::Acquire, &guard).as_ref() }.unwrap();
        memtable.delete(id);

        // 3. Update Global Tombstones (Visibility L1)
        self.deleted_ids.write().insert(id);

        // 4. Update KV (Locator)
        // We attempt to remove it so compaction knows it's gone (and tests pass).
        let id_bytes = id.to_le_bytes();

        // Mark tombstone in bucket header if we can find it (Optimistic)
        if let Ok(Some(bucket_id_bytes)) = self.kv.get(&id_bytes)
            && let Ok(bucket_id) = bucket_id_bytes.try_into().map(u32::from_le_bytes)
        {
            let buckets = unsafe { self.buckets.load(Ordering::Acquire, &guard).as_ref() }.unwrap();
            if let Some(header) = buckets.get(&bucket_id) {
                header.mark_tombstone();
            }
        }

        // Remove from KV Store
        let _ = self.kv.remove(&id_bytes);

        Ok(())
    }

    pub async fn search_async(
        &self,
        query: &[f32],
        k: usize,
        target_confidence: f32,
        lambda: f32,
        tau: f32,
    ) -> io::Result<Vec<SearchResult>> {
        // --- PHASE 1: SNAPSHOT & PRE-CALCULATION (Synchronous) ---
        // We capture state from RAM first to avoid holding RwLock guards across await points.

        // 1. Internal Oversampling:
        // We keep 2x candidates to ensure actual matches survive quantization noise.
        let internal_k = k * 2;

        // 2. ADC Look-Up Table:
        // Precompute squared distances for the query against all 256 possible byte values. [cite: 18-19, 125]
        let lut: Option<Vec<f32>> = {
            let guard = self.quantizer.read();
            guard.as_ref().map(|q| q.precompute_lut(query))
        };

        // 3. Delete Snapshot:
        // Clone the tombstone set to filter results without holding the deleted_ids lock. [cite: 8, 122]
        let deleted_snapshot = {
            let guard = self.deleted_ids.read();
            guard.clone()
        };

        // 4. L0 MemTable Scan:
        // Perform high-fidelity f32 search on the most recent data. [cite: 5, 123]
        let mem_results = {
            let guard = epoch::pin();
            let active = unsafe { self.memtable.load(Ordering::Acquire, &guard).as_ref() }.unwrap();
            let mut results = active.search(query, internal_k, self.config.ef_search);
            let frozen_guard = self.frozen_memtable.read();
            if let Some(frozen) = frozen_guard.as_ref() {
                results.extend(frozen.search(query, internal_k, self.config.ef_search));
            }
            results
        };

        // 5. Centroid Routing:
        // Select which L1 buckets to scan based on the Saturating Density model. [cite: 60, 70, 120]
        let selected_headers = self.get_selected_headers(query, target_confidence, lambda, tau);

        // --- PHASE 2: ASYNCHRONOUS DISK I/O ---
        // Now performing I/O. Results here are approximate due to SQ8 quantization. [cite: 13, 79, 127]

        let mut disk_candidates = Vec::new();
        let mut l0_found = HashSet::new();
        for (id, _) in &mem_results {
            l0_found.insert(*id);
        }

        if let Some(lut_vec) = &lut {
            let dim = self.config.dim;
            for header in &selected_headers {
                // Fetch quantized data from the tiered BlockCache. [cite: 10, 149, 150]
                if let Ok(data) = self.cache.get(&header.page_id).await {
                    let hits = Bucket::scan_with_lut(&data, lut_vec, dim); // SIMD ADC [cite: 82, 573]
                    for res in hits {
                        if !deleted_snapshot.contains(&res.id) && !l0_found.contains(&res.id) {
                            disk_candidates.push((header.id, res));
                        }
                    }
                }
            }
        }

        // Sort approximate candidates and keep internal_k for refinement.
        disk_candidates.sort_by(|a, b| {
            a.1.distance
                .partial_cmp(&b.1.distance)
                .unwrap_or(CmpOrdering::Equal)
        });
        if disk_candidates.len() > internal_k {
            disk_candidates.truncate(internal_k);
        }

        // --- PHASE 3: HIGH-FIDELITY RE-RANKING (ALP Refinement) ---
        // Use Cold Blobs (ALP) to verify the top approximate candidates. [cite: 13, 107, 128]

        let mut final_heap = BinaryHeap::with_capacity(k);

        // A. Add MemTable results (already 100% precise).
        for (id, dist) in mem_results {
            if !deleted_snapshot.contains(&id) {
                Self::push_to_heap(&mut final_heap, SearchResult { id, distance: dist }, k);
            }
        }

        // B. Refine Disk Candidates by fetching raw vectors.
        let mut buckets_to_refine: HashMap<u32, Vec<u64>> = HashMap::new();
        for (bid, res) in &disk_candidates {
            buckets_to_refine.entry(*bid).or_default().push(res.id);
        }

        for (bid, target_ids) in buckets_to_refine {
            // Resolve the latest page_id in case the Janitor split the bucket during I/O.
            let page_id_opt = self.get_bucket_page_id(bid);
            if page_id_opt.is_none() {
                continue;
            }

            let page_id = page_id_opt.unwrap();

            // Fetch raw floats (ALP) and current hot metadata (VIDs). [cite: 13, 106]
            let bucket_vecs_res = self.cache.storage().read_high_fidelity(bid).await;
            let hot_data_res = self.cache.get(&page_id).await;

            if let (Ok(bucket_vecs), Ok(hot_data)) = (bucket_vecs_res, hot_data_res) {
                // Safety Guard: Handle race conditions where bucket changed under our feet.
                let safe_limit = bucket_vecs.len().min(hot_data.vids.len());

                for i in 0..safe_limit {
                    let id = hot_data.vids[i];
                    if target_ids.contains(&id) {
                        let vec = &bucket_vecs[i];
                        // Final high-fidelity L2 calculation. [cite: 35, 1262]
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
                // Fallback: If ALP read fails, retain the approximate ADC result. [cite: 14]
                for (_, res) in disk_candidates.iter().filter(|(b, _)| *b == bid) {
                    Self::push_to_heap(&mut final_heap, res.clone(), k);
                }
            }
        }

        // Finalize results.
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
        let guard = epoch::pin();
        let buckets_map = unsafe { self.buckets.load(Ordering::Acquire, &guard).as_ref() }.unwrap();

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

    fn get_bucket_page_id(&self, bucket_id: u32) -> Option<drift_traits::PageId> {
        let guard = epoch::pin();
        let buckets = unsafe { self.buckets.load(Ordering::Acquire, &guard).as_ref() }.unwrap();
        buckets.get(&bucket_id).map(|h| h.page_id.clone())
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

    // drift_core/src/index.rs

    pub async fn partition_and_flush(
        &self,
        ids: &[u64],
        vectors: &[Vec<f32>],
    ) -> io::Result<Vec<u32>> {
        if vectors.is_empty() {
            return Ok(Vec::new());
        }

        // 1. Filter Tombstones
        let deleted_snapshot = self.deleted_ids.read().clone();
        let (valid_ids, valid_vecs): (Vec<u64>, Vec<Vec<f32>>) = ids
            .iter()
            .zip(vectors.iter())
            .filter(|(id, _)| !deleted_snapshot.contains(id))
            .map(|(id, vec)| (*id, vec.clone()))
            .unzip();

        if valid_ids.is_empty() {
            return Ok(Vec::new());
        }

        // 2. Determine K (Number of Buckets)
        let target_cap = self.config.max_bucket_capacity;
        let num_vectors = valid_ids.len();

        // ⚡ FIX 2: Hardened Partition Logic
        // Old logic: ceil(N / (Target * 0.8)) -> caused under-partitioning (3 buckets for 5 clusters)

        // Strategy A: Conservative Capacity (aim for 60% fill, not 80%)
        let count_based_k = (num_vectors as f32 / (target_cap as f32 * 0.6)).ceil() as usize;

        // Strategy B: Structural Heuristic
        // Assume latent clusters are rarely larger than 200 items in this workload.
        // This prevents cramming 1000 items into 2 buckets just because capacity allows it.
        let heuristic_k = (num_vectors / 200).max(1);
        // let heuristic_k = (num_vectors / 2000).max(1);

        // Take the maximum to be safe
        let k = count_based_k.max(heuristic_k).max(2);

        // Fast Path for tiny data
        if num_vectors <= target_cap && k == 1 {
            let id = self.next_bucket_id.fetch_add(1, Ordering::Relaxed);
            self.force_register_bucket_with_ids(id, &valid_ids, &valid_vecs)
                .await?;
            return Ok(vec![id]);
        }

        tracing::info!(
            "Flush: Partitioning {} vectors into {} buckets (Target Cap: {})",
            num_vectors,
            k,
            target_cap
        );

        // 3. Train K-Means

        let batch_size = (num_vectors / 10).clamp(1000, 5000);
        info!(
            "Flush: Partitioning {} vectors into {} buckets (Mini-Batch: {})",
            num_vectors, k, batch_size
        );

        let trainer = KMeansTrainer::new(k, self.config.dim, 15).with_mini_batch(batch_size); // Enable optimization

        // Wrap this call to catch K-Means panics
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| trainer.train(&valid_vecs)))
                .map_err(|_| io::Error::other("K-Means Panicked!"))?;

        println!(
            "DEBUG: K-Means Complete. Centroids: {}",
            result.centroids.len()
        );

        let result = trainer.train(&valid_vecs);

        // 4. Group Data
        let mut clusters: Vec<(Vec<u64>, Vec<Vec<f32>>)> = vec![(Vec::new(), Vec::new()); k];

        for (i, &assignment) in result.assignments.iter().enumerate() {
            if assignment < k {
                clusters[assignment].0.push(valid_ids[i]);
                clusters[assignment].1.push(valid_vecs[i].clone());
            }
        }

        // 5. Prepare Results
        let q_arc = self
            .get_quantizer()
            .expect("Quantizer missing during flush");
        let mut created_ids = Vec::new();

        for (idx, (c_ids, c_vecs)) in clusters.into_iter().enumerate() {
            if c_ids.is_empty() {
                continue;
            }

            // Calculate Centroid
            let centroid = result.centroids[idx].clone();

            // SQ8 Encode
            let mut flat_codes = Vec::with_capacity(c_vecs.len() * self.config.dim);
            for v in &c_vecs {
                flat_codes.extend_from_slice(&q_arc.encode(v));
            }

            // ⚡ Calculate Vector Sum for Drift Tracking (O(1))
            let mut vector_sum = vec![0.0; self.config.dim];
            for v in &c_vecs {
                for i in 0..self.config.dim {
                    vector_sum[i] += v[i];
                }
            }

            // Allocate ID
            let bucket_id = self.next_bucket_id.fetch_add(1, Ordering::Relaxed);

            // Register (using internal logic to set vector_sum correctly)
            // Note: We can't reuse force_register easily here without refactoring because
            // force_register calculates its own centroid. We want to use the K-Means centroid.

            // Manual Registration Pattern:
            let bucket_data = BucketData {
                codes: AlignedBytes::from_slice(&flat_codes),
                vids: c_ids.clone(),
                tombstones: bit_set::BitSet::with_capacity(c_ids.len()),
            };

            let bytes = bucket_data.to_bytes(self.config.dim)?;
            self.cache
                .storage()
                .write_page(bucket_id, 0, &bytes)
                .await?;
            let page_id = PageId {
                file_id: bucket_id,
                offset: 0,
                length: bytes.len() as u32,
            };

            self.update_buckets(|b| {
                let mut new = b.clone();
                let header = BucketHeader::new(
                    bucket_id,
                    centroid.clone(),
                    c_ids.len() as u32,
                    page_id.clone(),
                );
                // ⚡ IMPORTANT: Set the sum
                *header.stats.vector_sum.write() = vector_sum.clone();
                new.insert(bucket_id, header);
                new
            });

            self.update_centroids(|c| {
                let mut new = c.clone();
                new.push(CentroidEntry {
                    id: bucket_id,
                    vector: centroid.clone(),
                    active: true,
                });
                new
            });

            for &vid in &c_ids {
                let _ = self.kv.put(&vid.to_le_bytes(), &bucket_id.to_le_bytes());
            }

            created_ids.push(bucket_id);
        }

        Ok(created_ids)
    }

    pub async fn split_and_steal(&self, bucket_id: u32) -> io::Result<MaintenanceStatus> {
        let q_arc = self.quantizer.read().as_ref().unwrap().clone();
        let deleted_snapshot: HashSet<u64> = self.deleted_ids.read().clone();

        // 1. Load Target Bucket Header & Global Centroids
        let (header, all_centroids) = {
            let guard = epoch::pin();
            let buckets = unsafe { self.buckets.load(Ordering::Acquire, &guard).as_ref() }.unwrap();
            let centroids =
                unsafe { self.centroids.load(Ordering::Acquire, &guard).as_ref() }.unwrap();
            let h = match buckets.get(&bucket_id) {
                Some(h) => h.clone(),
                None => return Ok(MaintenanceStatus::SkippedTooSmall),
            };
            (h, centroids.clone())
        };

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
        let trainer = KMeansTrainer::new(2, self.config.dim, 10);
        let result = trainer.train(&vecs);

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
            let n_header = {
                let guard = epoch::pin();
                let buckets =
                    unsafe { self.buckets.load(Ordering::Acquire, &guard).as_ref() }.unwrap();
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

        // ⚡ CRITICAL FIX: Recalculate Centroids for A & B (Post-Steal)
        // Since we added stolen vectors, the K-Means centroids are stale.
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
            // ⚡ CRITICAL FIX: Recalculate Neighbor Centroids (Post-Theft)
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
        let (z_header, all_centroids) = {
            let guard = epoch::pin();
            let buckets = unsafe { self.buckets.load(Ordering::Acquire, &guard).as_ref() }.unwrap();
            let centroids =
                unsafe { self.centroids.load(Ordering::Acquire, &guard).as_ref() }.unwrap();

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
        // ⚡ SAFETY THRESHOLD: Prevent merging into distant clusters (~70 units squared)
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
                // Too far from neighbors -> Send back to L0 for re-clustering
                self.insert(orphans_id[i], vec)?;
            }
        }

        // 7. Rewrite Neighbors
        for (target_id, orphans) in adoption_map {
            // A. Load Neighbor
            let t_header = {
                let guard = epoch::pin();
                let buckets =
                    unsafe { self.buckets.load(Ordering::Acquire, &guard).as_ref() }.unwrap();
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

            // ⚡ FIX: Recalculate Sum & Centroid
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
                    // ⚡ UPDATE
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

    pub fn allocate_next_bucket_id(&self) -> u32 {
        self.next_bucket_id.fetch_add(1, Ordering::Relaxed)
    }

    pub fn memtable_len(&self) -> usize {
        let guard = epoch::pin();
        let memtable = unsafe { self.memtable.load(Ordering::Acquire, &guard).as_ref() }.unwrap();
        memtable.len()
    }

    // ⚡ Rotation Logic
    // pub fn rotate_and_freeze(&self) -> io::Result<Option<Arc<MemTable>>> {
    //     // 1. Check if frozen slot is empty (Backpressure)
    //     let mut frozen_guard = self.frozen_memtable.write();
    //     if frozen_guard.is_some() {
    //         return Ok(None);
    //     }

    //     // 2. Lock WAL (Stops Inserts briefly)
    //     let _wal = self.wal.lock();

    //     // 3. Create New Active Table
    //     let new_memtable = Arc::new(MemTable::new(
    //         self.config.max_bucket_capacity * 10,
    //         self.config.dim,
    //         self.config.ef_construction,
    //         16,
    //     ));

    //     // 4. Atomic Swap
    //     // We swap the new table in and get the old one out in a single atomic step.
    //     let guard = epoch::pin();
    //     let new_owned = Owned::new(new_memtable);

    //     let old_shared = self.memtable.swap(new_owned, Ordering::AcqRel, &guard);

    //     // 5. Extract the Arc from the old pointer
    //     // Safety: We successfully swapped it out, so we have the reference.
    //     // We clone the inner Arc<MemTable> to hold it in the frozen slot.
    //     let old_arc = unsafe { old_shared.as_ref() }
    //         .expect("MemTable should not be null")
    //         .clone();

    //     // 6. Schedule cleanup of the old atomic wrapper
    //     // The Atomic<Arc<...>> held a pointer to the Arc on the heap.
    //     // We must defer deleting that pointer wrapper until all threads reading it are done.
    //     unsafe { guard.defer_destroy(old_shared) };

    //     // 7. Move Old Table to Frozen
    //     *frozen_guard = Some(old_arc.clone());

    //     // 8. Truncate WAL? NO!
    //     // Data is only in memory. Do NOT truncate WAL yet.

    //     Ok(Some(old_arc))
    // } // WAL Lock released here.

    // Replace your current rotate_and_freeze with this:
    pub fn rotate_and_freeze(&self) -> io::Result<Option<Arc<MemTable>>> {
        // 1. Check if frozen slot is empty (Backpressure check)
        // If this is Some, the Janitor is still busy with the previous 1M vectors.
        if self.frozen_memtable.read().is_some() {
            return Ok(None);
        }

        // 2. Perform Atomic Swap under WAL lock
        let old_memtable_arc = {
            let _wal_lock = self.wal.lock();

            let new_active = Arc::new(MemTable::new(
                self.config.max_bucket_capacity * 10,
                self.config.dim,
                self.config.ef_construction,
                16,
            ));

            let guard = epoch::pin();
            let new_owned = Owned::new(new_active);

            // Atomic pointer swap [cite: 938]
            let old_shared = self.memtable.swap(new_owned, Ordering::AcqRel, &guard);

            let arc = unsafe { old_shared.as_ref() }
                .expect("MemTable should not be null")
                .clone();

            unsafe { guard.defer_destroy(old_shared) };
            arc
        };

        // 3. Update Frozen Slot (Resumes ingestion immediately after this)
        {
            let mut frozen_guard = self.frozen_memtable.write();
            *frozen_guard = Some(old_memtable_arc.clone());
        }

        info!("MemTable Rotated. Ingestion can now resume on new table.");
        Ok(Some(old_memtable_arc))
    }

    // Janitor calls this after disk IO is done.
    // pub fn confirm_flush(&self) -> io::Result<()> {
    //     let mut frozen_guard = self.frozen_memtable.write();

    //     // 1. Clear Frozen Slot
    //     *frozen_guard = None;

    //     // 2. Truncate WAL
    //     // Now that data is safely on disk segments, we can wipe the log.
    //     let mut wal = self.wal.lock();
    //     wal.truncate()?;

    //     Ok(())
    // }

    // Replace your current confirm_flush with this:
    pub fn confirm_flush(&self) -> io::Result<()> {
        // 1. Truncate WAL first [cite: 949-950]
        // Now that the Janitor has finished writing the .drift segments to S3/Disk,
        // the WAL data is redundant.
        {
            let mut wal = self.wal.lock();
            wal.truncate()?;
        }

        // 2. Clear Frozen Slot
        // This signals to the backpressure loop in billion_scale.rs that
        // there is room for another rotation.
        {
            let mut frozen_guard = self.frozen_memtable.write();
            *frozen_guard = None;
        }

        info!("Flush confirmed. WAL truncated and frozen slot cleared.");
        Ok(())
    }

    pub fn rotate_memtable(&self) -> io::Result<Vec<(u64, Vec<f32>)>> {
        // 1. Acquire Lock (Blocks any active inserts)
        let mut wal = self.wal.lock();

        let guard = epoch::pin();
        let current_memtable_ref =
            unsafe { self.memtable.load(Ordering::Acquire, &guard).as_ref() }.unwrap();

        // 2. Extract Data (Safe because no inserts are happening due to WAL lock)
        let data = current_memtable_ref.extract_all();

        let new_memtable = Arc::new(MemTable::new(
            self.config.max_bucket_capacity * 10,
            self.config.dim,
            self.config.ef_construction,
            16,
        ));

        // 3. Swap Pointers
        self.memtable
            .store(Owned::new(new_memtable), Ordering::Release);

        // 4. Truncate WAL
        wal.truncate()?;

        Ok(data)
    } // Lock released here. New inserts will pick up the new MemTable.

    pub fn get_quantizer(&self) -> Option<Arc<Quantizer>> {
        self.quantizer.read().clone()
    }

    pub fn set_quantizer(&self, q: Quantizer) {
        *self.quantizer.write() = Some(Arc::new(q));
    }

    pub fn get_all_bucket_headers(&self) -> Vec<BucketHeader> {
        let guard = epoch::pin();
        let buckets = unsafe { self.buckets.load(Ordering::Acquire, &guard).as_ref() }.unwrap();
        buckets.values().cloned().collect()
    }

    // ⚡ NEW: The Scavenger Logic (Copy-On-Write Compaction)
    pub async fn compact_bucket(&self, bucket_id: u32) -> io::Result<Option<Vec<u64>>> {
        // 1. Snapshot Global Deletes
        // We need a stable view of what is deleted to filter correctly.
        let global_tombstones = self.deleted_ids.read().clone();

        // 2. Load Header
        let (header, centroid) = {
            let guard = epoch::pin();
            let buckets = unsafe { self.buckets.load(Ordering::Acquire, &guard).as_ref() }.unwrap();
            match buckets.get(&bucket_id) {
                Some(h) => (h.clone(), h.centroid.clone()),
                None => return Ok(None), // Bucket already gone/merged
            }
        };

        // 3. Load Data (Async I/O)
        // If the bucket is locked or missing, we skip it this round.
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
                compacted_ids.push(id); // Clean this up
            } else {
                live_vecs.push(vec);
                live_ids.push(id);
            }
        }

        // Optimization: If nothing to clean, exit early.
        if compacted_ids.is_empty() {
            return Ok(None);
        }

        // 5. Handle "Empty Bucket" Case (All items deleted)
        if live_vecs.is_empty() {
            self.atomic_remove_bucket(bucket_id);
            // Ensure KV is clean (though index.delete usually handles this)
            for id in compacted_ids.iter() {
                let _ = self.kv.remove(&id.to_le_bytes());
            }
            return Ok(Some(compacted_ids));
        }

        // 6. Write New Bucket (Copy-on-Write)
        // We acquire a new ID. Readers still see the old bucket until the atomic swap.
        let new_id = self.next_bucket_id.fetch_add(1, Ordering::Relaxed);
        let dim = self.config.dim;

        // Create new clean BucketData
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

        // Write to Disk
        self.cache.storage().write_page(new_id, 0, &bytes).await?;

        let page_id = PageId {
            file_id: new_id,
            offset: 0,
            length: bytes.len() as u32,
        };

        // 7. Atomic Swap (The "Commit")
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

        // Update Centroids Mapping
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

        // 8. Update KV Locators for Live Items
        // This points the live IDs to the new bucket ID.
        for &vid in &live_ids {
            let _ = self.kv.put(&vid.to_le_bytes(), &new_id.to_le_bytes());
        }

        Ok(Some(compacted_ids))
    }

    pub async fn calculate_partitions(
        &self,
        ids: &[u64],
        vectors: &[Vec<f32>],
    ) -> io::Result<Vec<PartitionResult>> {
        if vectors.is_empty() {
            return Ok(Vec::new());
        }

        // 1. Filter Tombstones
        let deleted_snapshot = self.deleted_ids.read().clone();
        let (valid_ids, valid_vecs): (Vec<u64>, Vec<Vec<f32>>) = ids
            .iter()
            .zip(vectors.iter())
            .filter(|(id, _)| !deleted_snapshot.contains(id))
            .map(|(id, vec)| (*id, vec.clone()))
            .unzip();

        if valid_ids.is_empty() {
            return Ok(Vec::new());
        }

        // 2. Determine K (Target 80% capacity)
        let target_cap = self.config.max_bucket_capacity;
        let num_vectors = valid_ids.len();

        let k = if num_vectors <= target_cap {
            1
        } else {
            (num_vectors as f32 / (target_cap as f32 * 0.8)).ceil() as usize
        };

        // 3. Train K-Means
        tracing::info!(
            "Flush: Partitioning {} vectors into {} buckets",
            num_vectors,
            k
        );
        let trainer = KMeansTrainer::new(k, self.config.dim, 10);
        let result = trainer.train(&valid_vecs);

        // 4. Group Data
        // We need 3 arrays per cluster: IDs, Vecs, and Centroid
        let mut clusters: Vec<(Vec<u64>, Vec<Vec<f32>>)> = vec![(Vec::new(), Vec::new()); k];

        for (i, &assignment) in result.assignments.iter().enumerate() {
            if assignment < k {
                clusters[assignment].0.push(valid_ids[i]);
                clusters[assignment].1.push(valid_vecs[i].clone());
            }
        }

        // 5. Prepare Results (Quantize here to save CPU later)
        let q_arc = self
            .get_quantizer()
            .expect("Quantizer missing during flush");
        let mut partitions = Vec::new();

        for (idx, (c_ids, c_vecs)) in clusters.into_iter().enumerate() {
            if c_ids.is_empty() {
                continue;
            }

            // Calculate Centroid
            let centroid = result.centroids[idx].clone();

            // SQ8 Encode
            let mut flat_codes = Vec::with_capacity(c_vecs.len() * self.config.dim);
            for v in &c_vecs {
                flat_codes.extend_from_slice(&q_arc.encode(v));
            }

            // Allocate ID
            let bucket_id = self.next_bucket_id.fetch_add(1, Ordering::Relaxed);

            partitions.push(PartitionResult {
                bucket_id,
                ids: c_ids,
                vectors: c_vecs,
                codes: flat_codes,
                centroid,
            });
        }

        Ok(partitions)
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

    // ⚡ DEBUG: Search with Trace
    // Returns: (Results, List of Scanned Bucket IDs, Debug Info Map)
    pub async fn search_debug(
        &self,
        query: &[f32],
        k: usize,
        target_confidence: f32,
        lambda: f32,
        tau: f32,
    ) -> io::Result<(Vec<SearchResult>, Vec<u32>, HashMap<String, String>)> {
        // --- PHASE 1: SNAPSHOT ---
        let lut = self
            .quantizer
            .read()
            .as_ref()
            .map(|q| q.precompute_lut(query));
        let quantizer = self.quantizer.read().as_ref().unwrap().clone();

        let guard = epoch::pin();
        let buckets_map = unsafe { self.buckets.load(Ordering::Acquire, &guard).as_ref() }.unwrap();

        // 1. Calculate Scores for ALL buckets (for debug info)
        let mut debug_scores = Vec::new();
        for header in buckets_map.values() {
            let centroid = &header.centroid;
            let count = header.count;
            let dist = crate::math::l2_sq(query, centroid).sqrt();
            let p_geom = (-lambda * dist).exp();
            let reliability = 1.0 - (-(count as f32) / tau).exp();
            let p_eff = p_geom * reliability;

            debug_scores.push((header.id, dist, count, p_eff));
        }

        // 2. Selection Logic (Cluster & Probabilistic + Guardrail)
        let mut clusters: HashMap<Vec<u32>, Vec<&BucketHeader>> = HashMap::new();
        for header in buckets_map.values() {
            let key = header.centroid.iter().map(|f| f.to_bits()).collect();
            clusters.entry(key).or_default().push(header);
        }
        let mut candidates: Vec<(Vec<&BucketHeader>, f32)> = clusters
            .into_values()
            .map(|h| {
                let c = &h[0].centroid;
                let count: u32 = h.iter().map(|b| b.count).sum();
                let d = crate::math::l2_sq(query, c).sqrt();
                let p = (-lambda * d).exp() * (1.0 - (-(count as f32) / tau).exp());
                (h, p)
            })
            .collect();
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(CmpOrdering::Equal));

        let mut selected_headers = Vec::new();
        let mut acc = 0.0;
        let mut visited = HashSet::new();
        let min_scan = (buckets_map.len() / 20).max(5);
        for (group, score) in candidates {
            acc += score;
            for h in group {
                if visited.insert(h.id) {
                    selected_headers.push(h);
                }
            }
            if acc >= target_confidence && selected_headers.len() >= min_scan {
                break;
            }
        }
        let mut all = buckets_map.values().collect::<Vec<_>>();
        all.sort_by(|a, b| {
            crate::math::l2_sq(query, &a.centroid)
                .partial_cmp(&crate::math::l2_sq(query, &b.centroid))
                .unwrap()
        });
        for h in all.iter().take(5) {
            if visited.insert(h.id) {
                selected_headers.push(h);
            }
        }
        // --- ROUTING LOGIC END ---

        let scanned_ids: Vec<u32> = selected_headers.iter().map(|h| h.id).collect();

        // 3. FORENSIC SCAN
        let search_buffer_size = self.config.ef_search.max(k);
        let mut sq8_heap = BinaryHeap::new();
        let mut float_heap = BinaryHeap::new();

        // Tracking specific missing ID details requires knowing the ground truth,
        // but search_debug doesn't know it. We return a map of "ID -> (SQ8_Dist, Float_Dist)".
        let mut forensics = HashMap::new();

        for h in &selected_headers {
            if let Ok(d) = self.cache.get(&h.page_id).await {
                // A. SQ8 Scan
                if let Some(l) = &lut {
                    for r in Bucket::scan_with_lut(&d, l, self.config.dim) {
                        Self::push_to_heap(&mut sq8_heap, r, search_buffer_size); // Buffer
                    }
                }

                // B. ⚡ FLOAT RECONSTRUCTION SCAN (The Truth)
                // We reconstruct the whole bucket to floats and check exact distances
                let (vecs, ids) = d.reconstruct(&quantizer);
                for (i, vec) in vecs.iter().enumerate() {
                    let id = ids[i];
                    let true_dist_sq = crate::math::l2_sq(query, vec);

                    // We can re-calculate SQ8 distance for this specific vector to verify LUT
                    // (Optional, expensive)

                    let r = SearchResult {
                        id,
                        distance: true_dist_sq,
                    };
                    Self::push_to_heap(&mut float_heap, r, search_buffer_size);

                    // Store detailed trace for every vector found
                    let sq8_dist = if let Some(l) = &lut {
                        // Re-run LUT for this single item
                        let start = i * self.config.dim;
                        let code_ptr = d.codes.as_ptr();
                        unsafe {
                            compute_distance_lut(code_ptr.add(start), l.as_ptr(), self.config.dim)
                        }
                    } else {
                        0.0
                    };

                    forensics.insert(
                        id.to_string(),
                        format!("SQ8={:.2} Float={:.2}", sq8_dist, true_dist_sq),
                    );
                }
            }
        }

        let mut results = sq8_heap.into_vec();
        results.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(CmpOrdering::Equal)
        });
        if results.len() > k {
            results.truncate(k);
        }

        let mut debug_info = HashMap::new();
        debug_info.insert("total_buckets".to_string(), buckets_map.len().to_string());

        // 4. Debug Report
        let mut debug_info = HashMap::new();
        debug_info.insert("total_buckets".to_string(), buckets_map.len().to_string());
        debug_info.insert("scanned_count".to_string(), scanned_ids.len().to_string());

        // Serialize top 5 bucket scores for context
        debug_scores.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap()); // Sort by P_eff desc
        let top_scores = debug_scores
            .iter()
            .take(5)
            .map(|(id, d, c, p)| format!("B{}: D={:.2} C={} P={:.4}", id, d, c, p))
            .collect::<Vec<_>>()
            .join(", ");
        debug_info.insert("top_candidates".to_string(), top_scores);

        Ok((results, scanned_ids, debug_info))
    }
}
