use crate::bucket::{Bucket, BucketData, BucketHeader};
use crate::kmeans::KMeansTrainer;
use crate::memtable::MemTable;
use crate::quantizer::Quantizer;
use crate::wal::{WalEntry, WalReader, WalWriter};
use crossbeam_epoch::{self as epoch, Atomic, Owned};
use drift_cache::block_cache::BlockCache;
use drift_kv::bitstore::BitStore;
use drift_traits::{PageId, PageManager};
use parking_lot::{Mutex, RwLock};
use std::cmp::Ordering as CmpOrdering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::io;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use tracing::{error, instrument};

#[derive(Debug, PartialEq, Eq)]
pub enum MaintenanceStatus {
    Completed,
    SkippedSingularity,
    SkippedTooSmall,
    SkippedLocked,
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

pub struct VectorIndex {
    pub config: IndexOptions,
    pub(crate) centroids: Atomic<Vec<CentroidEntry>>,

    pub(crate) memtable: Atomic<Arc<MemTable>>,
    pub(crate) frozen_memtable: RwLock<Option<Arc<MemTable>>>,

    pub(crate) buckets: Atomic<HashMap<u32, BucketHeader>>,
    pub cache: Arc<BlockCache<BucketData>>,
    pub(crate) quantizer: RwLock<Option<Arc<Quantizer>>>,
    pub(crate) next_bucket_id: AtomicU32,
    pub(crate) wal: Mutex<WalWriter>,
    pub kv: Arc<BitStore>,
    pub(crate) deleted_ids: RwLock<HashSet<u64>>,
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

    pub async fn train(&self, samples: &[Vec<f32>]) -> io::Result<()> {
        assert!(!samples.is_empty(), "Empty training set");
        let dim = self.config.dim;

        let q = Arc::new(Quantizer::train(samples));
        *self.quantizer.write() = Some(q.clone());

        let trainer = KMeansTrainer::new(self.config.num_centroids, dim, 20);
        let result = trainer.train(samples);

        self.next_bucket_id.store(0, Ordering::Relaxed);
        let mut new_centroids = Vec::new();
        let mut new_buckets = HashMap::new();
        let mut cluster_data = vec![vec![]; self.config.num_centroids];

        for (i, &assignment) in result.assignments.iter().enumerate() {
            cluster_data[assignment].push(i);
        }

        for (cluster_idx, sample_indices) in cluster_data.into_iter().enumerate() {
            let id = self.next_bucket_id.fetch_add(1, Ordering::Relaxed);
            let center = result.centroids[cluster_idx].clone();

            let mut bucket_data = BucketData {
                codes: crate::aligned::AlignedBytes::new(sample_indices.len() * dim),
                vids: Vec::with_capacity(sample_indices.len()),
                tombstones: bit_set::BitSet::with_capacity(sample_indices.len()),
            };

            for &sample_idx in &sample_indices {
                let vec = &samples[sample_idx];
                let code = q.encode(vec);
                bucket_data.vids.push(sample_idx as u64);
                for b in code {
                    bucket_data.codes.push(b);
                }
            }

            let bytes = bucket_data.to_bytes(dim)?;
            self.cache.storage().write_page(id, 0, &bytes).await?;
            let page_id = PageId {
                file_id: id,
                offset: 0,
                length: bytes.len() as u32,
            };

            new_buckets.insert(
                id,
                BucketHeader::new(id, center.clone(), sample_indices.len() as u32, page_id),
            );
            new_centroids.push(CentroidEntry {
                id,
                vector: center,
                active: true,
            });

            for &sample_idx in &sample_indices {
                let _ = self
                    .kv
                    .put(&(sample_idx as u64).to_le_bytes(), &id.to_le_bytes());
            }
        }

        let _guard = epoch::pin();
        self.centroids
            .store(Owned::new(new_centroids), Ordering::Release);
        self.buckets
            .store(Owned::new(new_buckets), Ordering::Release);
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
        let mut wal = self.wal.lock();
        wal.write_delete(id)?;
        wal.flush()?;

        let guard = epoch::pin();
        let memtable = unsafe { self.memtable.load(Ordering::Acquire, &guard).as_ref() }.unwrap();
        memtable.delete(id);

        self.deleted_ids.write().insert(id);
        let _ = self.kv.remove(&id.to_le_bytes());
        Ok(())
    }

    // ⚡ NEW: The Multi-Stage Search
    pub async fn search_async(
        &self,
        query: &[f32],
        k: usize,
        target_confidence: f32,
        lambda: f32,
        tau: f32,
    ) -> io::Result<Vec<SearchResult>> {
        // 1. Search Active + Frozen MemTables
        let mut mem_results = {
            let guard = epoch::pin();
            let active = unsafe { self.memtable.load(Ordering::Acquire, &guard).as_ref() }.unwrap();

            // Search Active
            let mut results = active.search(query, k, self.config.ef_search);

            // Search Frozen (if exists)
            let frozen_guard = self.frozen_memtable.read();
            if let Some(frozen) = frozen_guard.as_ref() {
                let frozen_res = frozen.search(query, k, self.config.ef_search);
                results.extend(frozen_res);
            }
            results
        };

        // Note: mem_results now contains results from Active and Frozen.
        // Duplicates might exist if an ID was updated, but downstream logic handles it via `deleted_ids` checks
        // and usually `Active` is fresher.

        let quantizer_arc = {
            let guard = self.quantizer.read();
            match guard.as_ref() {
                Some(q) => q.clone(),
                None => {
                    // If no quantizer, return only memory results
                    return Ok(mem_results
                        .into_iter()
                        .map(|(id, d)| SearchResult { id, distance: d })
                        .collect());
                }
            }
        };

        // ... [Bucket Selection Logic - KEEP AS IS] ...
        // (This part remains exactly the same as your old code)
        let selected_headers = {
            let guard = epoch::pin();
            let buckets_map =
                unsafe { self.buckets.load(Ordering::Acquire, &guard).as_ref() }.unwrap();

            if buckets_map.is_empty() {
                Vec::new()
            } else {
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

                        let dist_sq = crate::math::l2_sq(query, centroid);
                        let dist = dist_sq.sqrt();
                        let p_geom = (-lambda * dist).exp();
                        let p_density = 1.0 - (-(total_count as f32) / tau).exp();

                        (headers, p_geom * p_density)
                    })
                    .collect();

                candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(CmpOrdering::Equal));

                let mut headers = Vec::new();
                let mut acc_conf = 0.0;
                for (group, score) in candidates {
                    acc_conf += score;
                    for h in group {
                        headers.push((*h).clone());
                    }
                    if acc_conf >= target_confidence {
                        break;
                    }
                }

                if headers.is_empty() {
                    if let Some(best) = buckets_map.values().min_by(|a, b| {
                        crate::math::l2_sq(query, &a.centroid)
                            .partial_cmp(&crate::math::l2_sq(query, &b.centroid))
                            .unwrap()
                    }) {
                        headers.push((*best).clone());
                    }
                }
                headers
            }
        };

        let mut loaded_data = Vec::with_capacity(selected_headers.len());
        for header in selected_headers {
            match self.cache.get(&header.page_id).await {
                Ok(data) => loaded_data.push(data),
                Err(e) => error!("Failed to load bucket {}: {}", header.id, e),
            }
        }

        // ... [Merging Logic - MODIFIED] ...
        let mut heap = BinaryHeap::with_capacity(k);
        let deleted_guard = self.deleted_ids.read();
        let mut l0_found = HashSet::new();

        // Process Memory Results (Active + Frozen)
        for (id, dist) in mem_results {
            if deleted_guard.contains(&id) {
                continue;
            }
            // Use l0_found to prevent adding older versions from Disk
            l0_found.insert(id);

            let res = SearchResult { id, distance: dist };
            if heap.len() < k {
                heap.push(res);
            } else if dist <= heap.peek().unwrap().distance {
                heap.pop();
                heap.push(res);
            }
        }

        // Process Disk Results
        for data in loaded_data {
            let hits = Bucket::scan_static(&data, &quantizer_arc, query);
            for res in hits {
                if deleted_guard.contains(&res.id) {
                    continue;
                }
                // If ID was found in Memory (Active or Frozen), skip Disk version (it's stale)
                if l0_found.contains(&res.id) {
                    continue;
                }

                if heap.len() < k {
                    heap.push(res);
                } else if res.distance <= heap.peek().unwrap().distance {
                    heap.pop();
                    heap.push(res);
                }
            }
        }

        let mut sorted = heap.into_vec();
        sorted.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(CmpOrdering::Equal)
        });
        Ok(sorted)
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

        let q_arc = self
            .quantizer
            .read()
            .as_ref()
            .expect("Quantizer missing")
            .clone();
        let dim = self.config.dim;

        let mut centroid = vec![0.0; dim];
        for v in &valid_vecs {
            for i in 0..dim {
                centroid[i] += v[i];
            }
        }
        for i in 0..dim {
            centroid[i] /= valid_vecs.len() as f32;
        }

        let mut data = BucketData {
            codes: crate::aligned::AlignedBytes::new(valid_vecs.len() * dim),
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
            new.insert(
                id,
                BucketHeader::new(
                    id,
                    centroid.clone(),
                    valid_ids.len() as u32,
                    page_id.clone(),
                ),
            );
            new
        });

        for &vid in &valid_ids {
            let _ = self.kv.put(&vid.to_le_bytes(), &id.to_le_bytes());
        }

        Ok(())
    }

    pub async fn split_and_steal(&self, bucket_id: u32) -> io::Result<MaintenanceStatus> {
        let q_arc = self.quantizer.read().as_ref().unwrap().clone();
        let deleted_snapshot: HashSet<u64> = self.deleted_ids.read().clone();

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

        // --- NEW: DRIFT/VARIANCE CHECK ---
        // Implement your suggestion: Check if the bucket has enough internal variance to split.
        // If variance is near zero, it's a Singularity.
        let dim = self.config.dim;
        let mut mean = vec![0.0; dim];
        for v in &vecs {
            for i in 0..dim {
                mean[i] += v[i];
            }
        }
        for i in 0..dim {
            mean[i] /= vecs.len() as f32;
        }

        let mut total_variance = 0.0;
        for v in &vecs {
            total_variance += crate::math::l2_sq(v, &mean);
        }
        let avg_variance = total_variance / vecs.len() as f32;

        // Threshold 0.01 is conservative for SQ8 (which has ~0.5 error).
        // If real variance is < 0.01, points are identical.
        if avg_variance < 0.01 {
            return Ok(MaintenanceStatus::SkippedSingularity);
        }

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

        for nid in neighbors {
            if budget >= 200 {
                break;
            }
            let n_header = {
                let guard = epoch::pin();
                let buckets =
                    unsafe { self.buckets.load(Ordering::Acquire, &guard).as_ref() }.unwrap();
                match buckets.get(&nid) {
                    Some(h) => h.clone(),
                    None => continue,
                }
            };
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
            let mut steal_indices = Vec::new();
            let mut stolen_items = Vec::new();
            for (i, vec) in n_vecs.iter().enumerate() {
                if budget >= 200 {
                    break;
                }
                budget += 1;
                let d_curr = crate::math::l2_sq(vec, &n_header.centroid);
                let d_a = crate::math::l2_sq(vec, &result.centroids[0]);
                let d_b = crate::math::l2_sq(vec, &result.centroids[1]);
                if d_a.min(d_b) < (d_curr - 0.025) {
                    steal_indices.push(i);
                    stolen_items.push((vec.clone(), n_ids[i], d_a < d_b));
                }
            }
            if !steal_indices.is_empty() {
                steal_indices.sort_unstable_by(|a, b| b.cmp(a));
                for idx in steal_indices {
                    n_vecs.swap_remove(idx);
                    n_ids.swap_remove(idx);
                }
                modified_neighbors.insert(nid, (n_vecs, n_ids));
                for (vec, id, is_a) in stolen_items {
                    if is_a {
                        vecs_a.push(vec);
                        ids_a.push(id);
                    } else {
                        vecs_b.push(vec);
                        ids_b.push(id);
                    }
                }
            }
        }

        let write_page =
            async |bid: u32, vs: &[Vec<f32>], is: &[u64]| -> io::Result<(PageId, u32)> {
                let dim = self.config.dim;
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
        let mut neighbor_updates = Vec::new();
        for (nid, (vs, is)) in modified_neighbors {
            let (p, c) = write_page(nid, &vs, &is).await?;
            neighbor_updates.push((nid, p, c));
        }

        self.update_centroids(|c| {
            let mut new = c.clone();
            if let Some(x) = new.iter_mut().find(|x| x.id == bucket_id) {
                x.active = false;
            }
            new.push(CentroidEntry {
                id: id_a,
                vector: result.centroids[0].clone(),
                active: true,
            });
            new.push(CentroidEntry {
                id: id_b,
                vector: result.centroids[1].clone(),
                active: true,
            });
            new
        });
        self.update_buckets(|b| {
            let mut new = b.clone();
            new.remove(&bucket_id);
            new.insert(
                id_a,
                BucketHeader::new(id_a, result.centroids[0].clone(), count_a, page_a.clone()),
            );
            new.insert(
                id_b,
                BucketHeader::new(id_b, result.centroids[1].clone(), count_b, page_b.clone()),
            );
            for (nid, p, c) in &neighbor_updates {
                if let Some(h) = new.get_mut(nid) {
                    h.count = *c;
                    h.page_id = p.clone();
                }
            }
            new
        });
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
        let data_arc = match self.cache.get(&z_header.page_id).await {
            Ok(d) => d,
            Err(_) => return Ok(MaintenanceStatus::SkippedLocked),
        };
        let (raw_vecs, raw_ids) = data_arc.reconstruct(&q_arc);

        let mut orphans_vec = Vec::new();
        let mut orphans_id = Vec::new();
        for (v, id) in raw_vecs.into_iter().zip(raw_ids.into_iter()) {
            if !deleted_snapshot.contains(&id) {
                orphans_vec.push(v);
                orphans_id.push(id);
            }
        }

        if orphans_vec.is_empty() {
            self.atomic_remove_bucket(zombie_id);
            return Ok(MaintenanceStatus::Completed);
        }

        let mut candidates: Vec<(u32, f32)> = all_centroids
            .iter()
            .filter(|c| c.active && c.id != zombie_id)
            .map(|c| (c.id, crate::math::l2_sq(&z_header.centroid, &c.vector)))
            .collect();
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let neighbor_ids: Vec<u32> = candidates.into_iter().take(3).map(|(id, _)| id).collect();

        if neighbor_ids.is_empty() {
            for (i, vec) in orphans_vec.iter().enumerate() {
                self.insert(orphans_id[i], vec)?;
            }
            self.atomic_remove_bucket(zombie_id);
            return Ok(MaintenanceStatus::Completed);
        }

        let mut adoption_map: HashMap<u32, Vec<(u64, Vec<f32>)>> = HashMap::new();
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
            if let Some(bid) = best_neighbor {
                adoption_map
                    .entry(bid)
                    .or_default()
                    .push((orphans_id[i], vec.clone()));
            } else {
                self.insert(orphans_id[i], vec)?;
            }
        }
        for (target_id, orphans) in adoption_map {
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
            let t_data = match self.cache.get(&t_header.page_id).await {
                Ok(d) => d,
                Err(_) => {
                    for (id, v) in orphans {
                        self.insert(id, &v)?;
                    }
                    continue;
                }
            };
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
            let bytes = new_data.to_bytes(dim)?;
            self.cache
                .storage()
                .write_page(target_id, 0, &bytes)
                .await?;
            self.update_buckets(|current| {
                let mut new = current.clone();
                if let Some(h) = new.get_mut(&target_id) {
                    h.count = vs.len() as u32;
                    h.page_id.length = bytes.len() as u32;
                }
                new
            });
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
    pub fn rotate_and_freeze(&self) -> io::Result<Option<Arc<MemTable>>> {
        // 1. Check if frozen slot is empty (Backpressure)
        let mut frozen_guard = self.frozen_memtable.write();
        if frozen_guard.is_some() {
            return Ok(None);
        }

        // 2. Lock WAL (Stops Inserts briefly)
        let _wal = self.wal.lock();

        // 3. Create New Active Table
        let new_memtable = Arc::new(MemTable::new(
            self.config.max_bucket_capacity * 10,
            self.config.dim,
            self.config.ef_construction,
            16,
        ));

        // 4. Atomic Swap
        // We swap the new table in and get the old one out in a single atomic step.
        let guard = epoch::pin();
        let new_owned = Owned::new(new_memtable);

        let old_shared = self.memtable.swap(new_owned, Ordering::AcqRel, &guard);

        // 5. Extract the Arc from the old pointer
        // Safety: We successfully swapped it out, so we have the reference.
        // We clone the inner Arc<MemTable> to hold it in the frozen slot.
        let old_arc = unsafe { old_shared.as_ref() }
            .expect("MemTable should not be null")
            .clone();

        // 6. Schedule cleanup of the old atomic wrapper
        // The Atomic<Arc<...>> held a pointer to the Arc on the heap.
        // We must defer deleting that pointer wrapper until all threads reading it are done.
        unsafe { guard.defer_destroy(old_shared) };

        // 7. Move Old Table to Frozen
        *frozen_guard = Some(old_arc.clone());

        // 8. Truncate WAL? NO!
        // Data is only in memory. Do NOT truncate WAL yet.

        Ok(Some(old_arc))
    } // WAL Lock released here.

    // Janitor calls this after disk IO is done.
    pub fn confirm_flush(&self) -> io::Result<()> {
        let mut frozen_guard = self.frozen_memtable.write();

        // 1. Clear Frozen Slot
        *frozen_guard = None;

        // 2. Truncate WAL
        // Now that data is safely on disk segments, we can wipe the log.
        let mut wal = self.wal.lock();
        wal.truncate()?;

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
}
