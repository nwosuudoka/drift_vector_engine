use crate::bucket::Bucket;
use crate::kmeans::KMeansTrainer;
use crate::memtable::MemTable;
use crate::quantizer::Quantizer;
use crate::wal::{WalEntry, WalReader, WalWriter};
use crossbeam_epoch::{self as epoch, Atomic, Owned};
use drift_kv::bitstore::BitStore;
use rayon::prelude::*;
use std::cmp::Ordering as CmpOrdering;
use std::collections::{BinaryHeap, HashMap};
use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, RwLock};

#[derive(Clone)]
pub struct IndexOptions {
    pub dim: usize,
    pub num_centroids: usize,
    pub training_sample_size: usize,
    /// Threshold to trigger a split (e.g., 2000 vectors)
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
            ef_construction: 40, // Default for production
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

/// Dynamic Centroid Table Entry
#[derive(Clone)]
pub(crate) struct CentroidEntry {
    pub(crate) id: u32,
    pub(crate) vector: Vec<f32>,
    pub(crate) active: bool, // Soft Delete support
}

pub struct VectorIndex {
    pub config: IndexOptions,
    // Quantizer is now Arc so buckets can share it cheaply
    pub(crate) quantizer: RwLock<Option<Arc<Quantizer>>>,

    // Routing Table: Supports Append + Soft Delete
    pub(crate) centroids: Atomic<Vec<CentroidEntry>>,

    // Data Table: Maps ID -> Bucket
    pub(crate) buckets: Atomic<HashMap<u32, Arc<Bucket>>>,

    // ID Generator for new buckets (Splits)
    pub(crate) next_bucket_id: AtomicU32,

    pub(crate) wal: Mutex<WalWriter>,
    // L0: MemTable
    pub(crate) memtable: Atomic<Arc<MemTable>>,

    pub(crate) kv: Arc<BitStore>,
}

impl VectorIndex {
    pub fn new(config: IndexOptions, wal_path: &Path) -> io::Result<Self> {
        // 1. Init L0
        let memtable = Arc::new(MemTable::new(
            config.max_bucket_capacity * 10,
            config.dim,
            // 40,
            config.ef_construction,
            16,
        ));

        // 2. Replay WAL (Crash Recovery)
        if wal_path.exists() {
            let reader = WalReader::open(wal_path)?;
            let entries = reader.read_all();
            if !entries.is_empty() {
                // Bulk load could be optimized, but simple insert is safe
                for entry in entries {
                    match entry {
                        WalEntry::Insert { id, vector } => {
                            memtable.insert(id, &vector);
                        } // TODO: implement delete
                        WalEntry::Delete { id } => {
                            memtable.delete(id);
                        }
                    }
                }
            }
        }

        // 3. Open WAL for writing (Append mode)
        let writer = WalWriter::new(wal_path)?;

        let base_dir = wal_path.parent().unwrap_or(Path::new("."));
        let kv_path = base_dir.join("id_map");
        let kv = Arc::new(BitStore::new(&kv_path).map_err(|e| io::Error::other(e.to_string()))?);

        Ok(Self {
            config,
            wal: Mutex::new(writer),
            memtable: Atomic::new(memtable),
            quantizer: RwLock::new(None),
            centroids: Atomic::new(Vec::new()),
            buckets: Atomic::new(HashMap::new()),
            next_bucket_id: AtomicU32::new(0),
            kv,
        })
    }

    /// Helper: Safely swap the Centroids map using Compare-and-Swap loop
    fn update_centroids<F>(&self, mut f: F)
    where
        F: FnMut(&Vec<CentroidEntry>) -> Vec<CentroidEntry>,
    {
        let guard = epoch::pin();
        loop {
            // 1. Load shared pointer
            let shared = self.centroids.load(Ordering::Acquire, &guard);

            // 2. Dereference safely (returns &Vec<...>)
            let current = unsafe { shared.as_ref() }.unwrap();

            // 3. Create modified copy
            let new_vec = f(current);

            // 4. Atomic CAS
            // If `shared` hasn't changed, replace with `new_vec`.
            // Owned::new allocates the new data on the heap managed by epoch.
            match self.centroids.compare_exchange(
                shared,
                Owned::new(new_vec),
                Ordering::Release,
                Ordering::Relaxed,
                &guard,
            ) {
                Ok(_) => break,     // Success
                Err(_) => continue, // Retry: someone else updated it
            }
        }
    }

    /// Helper: Safely swap the Buckets map using Compare-and-Swap loop
    fn update_buckets<F>(&self, mut f: F)
    where
        F: FnMut(&HashMap<u32, Arc<Bucket>>) -> HashMap<u32, Arc<Bucket>>,
    {
        let guard = epoch::pin();
        loop {
            let shared = self.buckets.load(Ordering::Acquire, &guard);
            let current = unsafe { shared.as_ref() }.unwrap();
            let new_map = f(current);

            match self.buckets.compare_exchange(
                shared,
                Owned::new(new_map),
                Ordering::Release,
                Ordering::Relaxed,
                &guard,
            ) {
                Ok(_) => break,
                Err(_) => continue,
            }
        }
    }

    pub fn train(&self, samples: &[Vec<f32>]) {
        assert!(!samples.is_empty(), "Empty training set");
        let dim = self.config.dim;

        // 1. Train Quantizer
        let q = Arc::new(Quantizer::train(samples));
        *self.quantizer.write().unwrap() = Some(q.clone());

        // 2. Train Initial K-Means
        let trainer = KMeansTrainer::new(self.config.num_centroids, dim, 20);
        let result = trainer.train(samples);

        // 3. Initialize Data (Atomic Swap)
        self.next_bucket_id.store(0, Ordering::Relaxed);

        let mut new_centroids = Vec::new();
        let mut new_buckets = HashMap::new();

        for center in result.centroids.into_iter() {
            let id = self.next_bucket_id.fetch_add(1, Ordering::Relaxed);
            new_centroids.push(CentroidEntry {
                id,
                vector: center,
                active: true,
            });
            let cap = (samples.len() / self.config.num_centroids).max(100);
            let bucket = Arc::new(Bucket::new(id, cap, dim, q.clone()));
            new_buckets.insert(id, bucket);
        }

        // Unconditional store since we are initializing/resetting
        let _guard = epoch::pin();
        self.centroids
            .store(Owned::new(new_centroids), Ordering::Release);
        self.buckets
            .store(Owned::new(new_buckets), Ordering::Release);
    }

    pub fn insert(&self, id: u64, vector: &[f32]) -> io::Result<()> {
        // 1. DURABILITY: Append to WAL
        {
            let mut wal = self.wal.lock().unwrap();
            wal.write_insert(id, vector)?;
            wal.flush()?; // Ensure it hits OS buffers at minimum
        }

        // 2. VISIBILITY: Insert into HNSW (L0)
        let guard = epoch::pin();
        let memtable = unsafe { self.memtable.load(Ordering::Acquire, &guard).as_ref() }.unwrap();
        memtable.insert(id, vector);

        Ok(())
    }

    pub fn split_bucket(&self, bucket_id: u32) {
        let q_arc = {
            let g = self.quantizer.read().unwrap();
            g.as_ref().unwrap().clone()
        };

        // 1. Extract Data (Lock-Free Map lookup, then Read Lock on Bucket)
        let guard = epoch::pin();
        let buckets_ref = unsafe { self.buckets.load(Ordering::Acquire, &guard).as_ref() }.unwrap();

        let (vecs, ids) = match buckets_ref.get(&bucket_id) {
            Some(b) => b.extract_reconstructed(),
            None => return, // Bucket already gone
        };
        // Drop guard early - we have the data
        drop(guard);

        if vecs.len() < 20 {
            return;
        }

        // 2. Run Local K-Means
        let trainer: KMeansTrainer = KMeansTrainer::new(2, self.config.dim, 10);
        let result = trainer.train(&vecs);

        // 3. Create New Buckets
        let id_1 = self.next_bucket_id.fetch_add(1, Ordering::Relaxed);
        let id_2 = self.next_bucket_id.fetch_add(1, Ordering::Relaxed);

        let cap = vecs.len() / 2 + 100;
        let b1 = Arc::new(Bucket::new(id_1, cap, self.config.dim, q_arc.clone()));
        let b2 = Arc::new(Bucket::new(id_2, cap, self.config.dim, q_arc.clone()));

        *b1.centroid.write() = result.centroids[0].clone();
        *b2.centroid.write() = result.centroids[1].clone();

        for (i, &cluster_idx) in result.assignments.iter().enumerate() {
            let target = if cluster_idx == 0 { &b1 } else { &b2 };
            let code = q_arc.encode(&vecs[i]);
            target.insert(ids[i], &code);
        }

        // 4. ATOMIC UPDATE (COW)
        // We update centroids first, then buckets.
        // There is a tiny window where centroids exist but buckets don't,
        // causing searches to miss. This is acceptable in eventual consistency.

        self.update_centroids(|current| {
            let mut new = current.clone();
            if let Some(c) = new.iter_mut().find(|c| c.id == bucket_id) {
                c.active = false;
            }
            new.push(CentroidEntry {
                id: id_1,
                vector: result.centroids[0].clone(),
                active: true,
            });
            new.push(CentroidEntry {
                id: id_2,
                vector: result.centroids[1].clone(),
                active: true,
            });
            new
        });

        self.update_buckets(|current| {
            let mut new = current.clone();
            new.remove(&bucket_id);
            new.insert(id_1, b1.clone());
            new.insert(id_2, b2.clone());
            new
        });
    }

    // New Helper: Force insert to L1 (Bypassing L0) - useful for hydration/training
    pub fn force_insert_l1(&self, id: u64, vector: &[f32]) {
        let q_guard = self.quantizer.read().unwrap();
        let q = q_guard.as_ref().expect("Not trained");
        let guard = epoch::pin();

        let centroids = unsafe { self.centroids.load(Ordering::Acquire, &guard).as_ref() }.unwrap();
        let best_id = self.find_nearest_bucket(vector, centroids);
        let code = q.encode(vector);

        let buckets = unsafe { self.buckets.load(Ordering::Acquire, &guard).as_ref() }.unwrap();
        if let Some(bucket) = buckets.get(&best_id) {
            bucket.insert(id, &code);

            // Log error but don't crash (Eventual consistency?)
            // Ideally, we want this to succeed.
            if let Err(e) = self.kv.put(
                id.to_le_bytes().as_slice(),
                best_id.to_le_bytes().as_slice(),
            ) {
                eprintln!("KV Put Error for ID {}: {}", id, e);
            }
        }
    }

    pub fn search_drift_aware(
        &self,
        query: &[f32],
        k: usize,
        target_confidence: f32,
        lambda: f32,
        tau: f32,
    ) -> Vec<SearchResult> {
        let q_guard = self.quantizer.read().unwrap();
        let has_l1 = q_guard.is_some();

        // --- 1. SEARCH LEVEL 0 (MemTable) ---
        let guard = epoch::pin();
        let memtable = unsafe { self.memtable.load(Ordering::Acquire, &guard).as_ref() }.unwrap();

        // Use config.ef_search
        let l0_results = memtable.search(query, k, self.config.ef_search);

        // Optimization: If no L1 exists (not trained), return L0 immediately.
        if !has_l1 {
            return l0_results
                .into_iter()
                .map(|(id, dist)| SearchResult { id, distance: dist })
                .collect();
        }

        // --- 2. SEARCH LEVEL 1 (Buckets) ---
        let q = q_guard.as_ref().unwrap();
        let buckets = unsafe { self.buckets.load(Ordering::Acquire, &guard).as_ref() }.unwrap();
        let centroids = unsafe { self.centroids.load(Ordering::Acquire, &guard).as_ref() }.unwrap();

        // Calculate Scores (Saturating Density)
        let mut candidates: Vec<(u32, f32, f32)> = centroids
            .iter()
            .filter(|c| c.active)
            .map(|c| {
                let dist_sq = distance_sq(query, &c.vector);
                let count = if let Some(b) = buckets.get(&c.id) {
                    b.count.load(Ordering::Relaxed) as f32
                } else {
                    0.0
                };
                (c.id, dist_sq, count)
            })
            .collect();

        // Sort by Probability
        candidates.sort_by(|a, b| {
            let dist_a = a.1.sqrt();
            let p_geom_a = (-lambda * dist_a).exp();
            let r_a = 1.0 - (-a.2 / tau).exp();
            let score_a = p_geom_a * r_a;

            let dist_b = b.1.sqrt();
            let p_geom_b = (-lambda * dist_b).exp();
            let r_b = 1.0 - (-b.2 / tau).exp();
            let score_b = p_geom_b * r_b;

            score_b.partial_cmp(&score_a).unwrap_or(CmpOrdering::Equal)
        });

        // Select Buckets
        let mut target_ids = Vec::new();
        let mut accumulated_confidence = 0.0;

        for (id, dist_sq, count) in candidates {
            if accumulated_confidence >= target_confidence {
                break;
            }
            let p_eff = (-lambda * dist_sq.sqrt()).exp() * (1.0 - (-count / tau).exp());
            accumulated_confidence += p_eff;
            target_ids.push(id);
        }

        // Scan L1 Buckets
        let lut = q.precompute_lut(query);
        let l1_results: Vec<SearchResult> = target_ids
            .par_iter()
            .flat_map(|id| {
                if let Some(b) = buckets.get(id) {
                    b.scan_adc(&lut, k)
                } else {
                    Vec::new()
                }
            })
            .collect();

        // --- 3. MERGE L0 AND L1 ---
        // Use a binary heap to keep the top K results globally
        let mut heap = BinaryHeap::with_capacity(k);

        // Push L0 Results
        for (id, dist) in l0_results {
            let res = SearchResult { id, distance: dist };
            if heap.len() < k {
                heap.push(res);
            } else if dist < heap.peek().unwrap().distance {
                heap.pop();
                heap.push(res);
            }
        }

        // Push L1 Results
        for res in l1_results {
            if heap.len() < k {
                heap.push(res);
            } else if res.distance < heap.peek().unwrap().distance {
                heap.pop();
                heap.push(res);
            }
        }

        let mut sorted = heap.into_vec();
        sorted.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(CmpOrdering::Equal)
        });
        sorted
    }

    fn find_nearest_bucket(&self, vec: &[f32], centroids: &[CentroidEntry]) -> u32 {
        let mut best_id = 0;
        let mut min_dist = f32::MAX;
        for c in centroids {
            if !c.active {
                continue;
            }
            let d = distance_sq(vec, &c.vector);
            if d < min_dist {
                min_dist = d;
                best_id = c.id;
            }
        }
        best_id
    }

    // =========================================================================
    //  PRIMITIVE 1: SCATTER MERGE (The "Heal" Operation)
    // =========================================================================

    pub fn scatter_merge(&self, zombie_id: u32) -> io::Result<()> {
        // 1. Atomic Dissolve: Remove bucket from Index
        // We use the CAS loops to remove it from Maps

        let mut removed_bucket: Option<Arc<Bucket>> = None;

        self.update_centroids(|current| {
            let mut new = current.clone();
            if let Some(pos) = new.iter().position(|c| c.id == zombie_id) {
                new.remove(pos);
            }
            new
        });

        self.update_buckets(|current| {
            let mut new = current.clone();
            removed_bucket = new.remove(&zombie_id);
            new
        });

        // 2. Extract & Scatter
        if let Some(bucket) = removed_bucket {
            let (vecs, ids) = bucket.extract_reconstructed();
            for (i, vec) in vecs.iter().enumerate() {
                let vid = ids[i];
                self.insert(vid, vec)?;

                let _ = self.kv.remove(vid.to_le_bytes().as_slice());
            }
        }

        Ok(())
    }

    pub fn get_quantizer(&self) -> Option<Arc<Quantizer>> {
        self.quantizer.read().unwrap().clone()
    }

    pub fn get_all_buckets(&self) -> Vec<Arc<Bucket>> {
        let guard = epoch::pin();
        let buckets = unsafe { self.buckets.load(Ordering::Acquire, &guard).as_ref() }.unwrap();
        buckets.values().cloned().collect()
    }

    pub fn set_quantizer(&self, q: Quantizer) {
        *self.quantizer.write().unwrap() = Some(Arc::new(q));
    }

    // =========================================================================
    //  PRIMITIVE 2: SPLIT WITH NEIGHBOR STEALING (The "Growth" Operation)
    // =========================================================================

    pub fn split_and_steal(&self, bucket_id: u32) {
        let q_arc = {
            let g = self.quantizer.read().unwrap();
            g.as_ref().unwrap().clone()
        };

        // --- PHASE 1: STANDARD SPLIT ---

        // 1. Extract Data (Read-only access to Index map)
        // We scope the guard to drop it as soon as we have the data vectors.
        let (vecs, ids, old_centroid) = {
            let guard = epoch::pin();

            // Access Maps
            let buckets = unsafe { self.buckets.load(Ordering::Acquire, &guard).as_ref() }.unwrap();
            let centroids =
                unsafe { self.centroids.load(Ordering::Acquire, &guard).as_ref() }.unwrap();

            let bucket = match buckets.get(&bucket_id) {
                Some(b) => b,
                None => return, // Bucket might have been merged/deleted concurrently
            };

            let vecs_data = bucket.extract_reconstructed();

            let centroid = centroids
                .iter()
                .find(|c| c.id == bucket_id)
                .map(|c| c.vector.clone())
                // If centroid is missing, index state is inconsistent, abort safely
                .unwrap_or_default();

            (vecs_data.0, vecs_data.1, centroid)
        };

        if vecs.is_empty() || vecs.len() < 20 {
            return;
        }

        // 2. Run Local 2-Means (Heavy Compute - No Index Locks held)
        let trainer = KMeansTrainer::new(2, self.config.dim, 10);
        let result = trainer.train(&vecs);

        // 3. Create New Buckets
        let id_a = self.next_bucket_id.fetch_add(1, Ordering::Relaxed);
        let id_b = self.next_bucket_id.fetch_add(1, Ordering::Relaxed);

        let capacity = vecs.len() / 2 + 100;
        let ba = Arc::new(Bucket::new(id_a, capacity, self.config.dim, q_arc.clone()));
        let bb = Arc::new(Bucket::new(id_b, capacity, self.config.dim, q_arc.clone()));

        *ba.centroid.write() = result.centroids[0].clone();
        *bb.centroid.write() = result.centroids[1].clone();

        // 4. Populate New Buckets
        for (i, &cluster_idx) in result.assignments.iter().enumerate() {
            let target = if cluster_idx == 0 { &ba } else { &bb };
            let code = q_arc.encode(&vecs[i]);
            let vid = ids[i]; // The vector ID

            target.insert(ids[i], &code);

            let _ = self.kv.put(
                vid.to_le_bytes().as_slice(),
                target.id.to_le_bytes().as_slice(),
            );
        }

        // --- PHASE 2: NEIGHBOR STEALING (Budgeted Maintenance) ---

        // 1. Find Neighbors
        // We need a fresh guard to read the *current* state of the index
        // because it might have changed while we were running K-Means.
        let neighbors = {
            let guard = epoch::pin();
            let centroids =
                unsafe { self.centroids.load(Ordering::Acquire, &guard).as_ref() }.unwrap();

            // Re-implement find_top_k inline or via helper that accepts slice
            let mut candidates: Vec<(u32, f32)> = centroids
                .iter()
                .filter(|c| c.active && c.id != bucket_id)
                .map(|c| (c.id, distance_sq(&old_centroid, &c.vector)))
                .collect();

            candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(CmpOrdering::Equal));
            candidates
                .into_iter()
                .take(3)
                .map(|(id, _)| id)
                .collect::<Vec<u32>>()
        };

        // 2. Audit Neighbors & Steal
        let delta = 0.025;
        let mut stolen_data = Vec::new();
        let max_check = 200;
        let mut budget_checked = 0;

        // We pin again to access neighbor buckets
        let guard = epoch::pin();
        let buckets = unsafe { self.buckets.load(Ordering::Acquire, &guard).as_ref() }.unwrap();

        for neighbor_id in neighbors {
            if budget_checked >= max_check {
                break;
            }

            if let Some(n_bucket) = buckets.get(&neighbor_id) {
                // Extract copies for checking (expensive, but safe)
                let (n_vecs, n_ids) = n_bucket.extract_reconstructed();
                let n_centroid = n_bucket.centroid.read();

                let mut ids_to_steal = Vec::new();

                for (i, vec) in n_vecs.iter().enumerate() {
                    if budget_checked >= max_check {
                        break;
                    }
                    budget_checked += 1;

                    let dist_current = distance_sq(vec, &n_centroid);
                    let dist_a = distance_sq(vec, &result.centroids[0]);
                    let dist_b = distance_sq(vec, &result.centroids[1]);

                    let dist_new = dist_a.min(dist_b);

                    if dist_new < (dist_current - delta) {
                        ids_to_steal.push(n_ids[i]);
                    }
                }

                // Execute Steal (Write to Neighbor Bucket)
                // We can do this while holding the epoch guard because Bucket internal locks are fine.
                if !ids_to_steal.is_empty() {
                    let stolen = n_bucket.steal_vectors(&ids_to_steal);
                    stolen_data.extend(stolen.into_iter().zip(ids_to_steal.into_iter()));
                }
            }
        }
        drop(guard); // Release map access

        // 3. Insert Stolen Items into NEW buckets (ba/bb)
        for (vec, id) in stolen_data {
            let da = distance_sq(&vec, &result.centroids[0]);
            let db = distance_sq(&vec, &result.centroids[1]);
            let target = if da < db { &ba } else { &bb };
            let code = q_arc.encode(&vec);
            target.insert(id, &code);
        }

        // --- PHASE 3: ATOMIC COMMIT ---

        self.update_centroids(|current| {
            let mut new = current.clone();
            if let Some(c) = new.iter_mut().find(|c| c.id == bucket_id) {
                c.active = false;
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

        self.update_buckets(|current| {
            let mut new = current.clone();
            new.remove(&bucket_id);
            new.insert(id_a, ba.clone());
            new.insert(id_b, bb.clone());
            new
        });
    }

    pub fn rebalance_buckets(&self, id_a: u32, id_b: u32) {
        let q_arc = self.quantizer.read().unwrap().as_ref().unwrap().clone();

        // 1. Extract Data
        let guard = epoch::pin();
        let buckets = unsafe { self.buckets.load(Ordering::Acquire, &guard).as_ref() }.unwrap();

        let b_a = buckets.get(&id_a);
        let b_b = buckets.get(&id_b);

        if b_a.is_none() || b_b.is_none() {
            return;
        }

        let (vecs_a, ids_a) = b_a.unwrap().extract_reconstructed();
        let (vecs_b, ids_b) = b_b.unwrap().extract_reconstructed();
        drop(guard);

        // 2. Pool
        let mut all_vecs = vecs_a;
        all_vecs.extend(vecs_b);
        let mut all_ids = ids_a;
        all_ids.extend(ids_b);

        if all_vecs.is_empty() {
            return;
        }

        // 3. K-Means
        let trainer = KMeansTrainer::new(2, self.config.dim, 10);
        let result = trainer.train(&all_vecs);

        // 4. New Buckets
        // We create NEW buckets with the OLD IDs to "refresh" them
        let b1 = Arc::new(Bucket::new(
            id_a,
            all_vecs.len(),
            self.config.dim,
            q_arc.clone(),
        ));
        let b2 = Arc::new(Bucket::new(
            id_b,
            all_vecs.len(),
            self.config.dim,
            q_arc.clone(),
        ));

        *b1.centroid.write() = result.centroids[0].clone();
        *b2.centroid.write() = result.centroids[1].clone();

        for (i, &cluster_idx) in result.assignments.iter().enumerate() {
            let target = if cluster_idx == 0 { &b1 } else { &b2 };
            let code = q_arc.encode(&all_vecs[i]);
            target.insert(all_ids[i], &code);
        }

        // 5. Atomic Update
        // We only update if the centroids still exist (didn't get deleted by someone else)
        self.update_centroids(|current| {
            let mut new = current.clone();
            if let Some(c) = new.iter_mut().find(|c| c.id == id_a) {
                c.vector = result.centroids[0].clone();
                c.active = true;
            }
            if let Some(c) = new.iter_mut().find(|c| c.id == id_b) {
                c.vector = result.centroids[1].clone();
                c.active = true;
            }
            new
        });

        self.update_buckets(|current| {
            let mut new = current.clone();
            new.insert(id_a, b1.clone());
            new.insert(id_b, b2.clone());
            new
        });
    }

    /// Returns the number of items in the active MemTable.
    /// Used by the Janitor to trigger flushes.
    pub fn memtable_len(&self) -> usize {
        let guard = epoch::pin();
        let memtable = unsafe { self.memtable.load(Ordering::Acquire, &guard).as_ref() }.unwrap();
        memtable.len()
    }

    /// ROTATE: Swaps the current MemTable with a fresh one, clears the WAL,
    /// and returns the old data for flushing.
    ///
    /// This is a critical section that briefly blocks inserts to ensure consistency.
    pub fn rotate_memtable(&self) -> io::Result<Vec<(u64, Vec<f32>)>> {
        // 1. Acquire WAL Lock (Stop Writes)
        // We hold this lock for the duration of the swap to ensure no one writes
        // to the old WAL/MemTable while we are resetting them.
        let mut wal = self.wal.lock().unwrap();

        // 2. Extract Data from Current MemTable
        // We pin the epoch to access the shared atomic pointer
        let guard = epoch::pin();
        let current_memtable_ref =
            unsafe { self.memtable.load(Ordering::Acquire, &guard).as_ref() }.unwrap();

        // Copy data out. Since we hold the WAL lock, no *new* inserts can happen,
        // but readers might still be accessing the HNSW graph (which is fine).
        let data = current_memtable_ref.extract_all();

        // 3. Create New MemTable
        // We use the same capacity/config as the original
        // Note: In a real system, you might dynamically adjust capacity based on load
        let new_memtable = Arc::new(MemTable::new(
            self.config.max_bucket_capacity * 10,
            self.config.dim,
            self.config.ef_construction,
            16,
        ));

        // 4. Atomic Swap
        // This makes the new, empty MemTable visible to readers immediately.
        // The old MemTable will be deallocated by crossbeam once all threads drop their guards.
        self.memtable
            .store(Owned::new(new_memtable), Ordering::Release);

        // 5. Truncate WAL
        // Now that the active MemTable is empty, the WAL (which backs it) should be empty too.
        // The old data is safely captured in the `data` Vec and will be flushed to L1 by the caller.
        wal.truncate()?;

        // WAL lock drops here automatically, allowing writes to resume to the NEW MemTable/WAL.
        Ok(data)
    }

    pub fn force_register_bucket_with_ids(&self, id: u32, ids: &[u64], vectors: &[Vec<f32>]) {
        if vectors.is_empty() {
            return;
        }

        let q_arc = self
            .quantizer
            .read()
            .unwrap()
            .as_ref()
            .expect("Quantizer missing during hydration")
            .clone();

        // 1. Calculate Centroid for the routing table
        let dim = self.config.dim;
        let mut centroid = vec![0.0; dim];
        for v in vectors {
            for i in 0..dim {
                centroid[i] += v[i];
            }
        }
        for i in 0..dim {
            centroid[i] /= vectors.len() as f32;
        }

        // 2. Create the L1 Bucket
        let bucket = Arc::new(Bucket::new(id, vectors.len() + 100, dim, q_arc.clone()));
        *bucket.centroid.write() = centroid.clone();

        // 3. Hydrate with REAL IDs (The Fix)
        for (i, v) in vectors.iter().enumerate() {
            let code = q_arc.encode(v);
            // We use ids[i] instead of the loop index i
            bucket.insert(ids[i], &code);
        }

        // 4. Atomic Registration into Routing Table
        self.update_centroids(|c| {
            let mut new = c.clone();
            new.push(CentroidEntry {
                id,
                vector: centroid.clone(),
                active: true,
            });
            new
        });

        self.update_buckets(|b| {
            let mut new = b.clone();
            new.insert(id, bucket.clone());
            new
        });
    }

    /// Reserves a new unique Bucket ID.
    /// Essential for hydration to prevent collisions when loading multiple L0 segments.
    pub fn allocate_next_bucket_id(&self) -> u32 {
        self.next_bucket_id.fetch_add(1, Ordering::Relaxed)
    }

    pub fn delete(&self, id: u64) -> io::Result<()> {
        // 1. Durable: Write to WAL
        {
            let mut wal = self.wal.lock().unwrap();
            wal.write_delete(id)?;
            // We don't necessarily flush to disk for every delete (perf trade-off)
        }

        // 2. L0: MemTable
        let guard = epoch::pin();
        let memtable = unsafe { self.memtable.load(Ordering::Acquire, &guard).as_ref() }.unwrap();
        memtable.delete(id);

        // 3: L1: Fast Lookup via KV
        let k_bytes = id.to_le_bytes();
        match self.kv.get(&k_bytes) {
            Ok(Some(v_buf)) => {
                if v_buf.len() == 4 {
                    let bucket_id = u32::from_le_bytes(v_buf.try_into().unwrap());
                    let buckets =
                        unsafe { self.buckets.load(Ordering::Acquire, &guard).as_ref() }.unwrap();
                    if let Some(bucket) = buckets.get(&bucket_id) {
                        bucket.delete(id);
                        // Optional: We keep the KV entry pointing to the bucket so subsequent deletes are fast,
                        // OR we remove it.
                        // If we remove it from KV, we lose track of the tombstone location.
                        // Better to KEEP it in KV so we know where the tombstone lives (to prevent resurrection).
                        // Actually, if we delete from KV, `get` returns None, and we assume it's gone.
                        // But physically, the vector remains in the bucket (marked as tombstone).
                        // Let's REMOVE from KV to signify "Not Searchable".
                        let _ = self.kv.remove(&k_bytes);
                    }
                }
            }
            Ok(None) => {} // Not in L1
            Err(e) => {
                eprintln!("KV Error: {}", e);
            }
        }

        // Might want to keep this as a backup??
        // 3.1. L1: Scan Buckets (Parallel)
        // Since we don't know which bucket has the ID, we check them all.
        // This is fast in RAM.
        // let buckets_map = unsafe { self.buckets.load(Ordering::Acquire, &guard).as_ref() }.unwrap();

        // buckets_map.values().par_bridge().for_each(|bucket| {
        //     // Bucket::delete checks if ID exists and sets the bitset if so.
        //     // We need to implement bucket.delete(id)
        //     bucket.delete(id);
        // });

        Ok(())
    }
}

fn distance_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}
