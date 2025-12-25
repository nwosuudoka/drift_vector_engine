use crate::bucket::{Bucket, BucketData, BucketHeader};
use crate::kmeans::KMeansTrainer;
use crate::memtable::MemTable;
use crate::quantizer::Quantizer;
use crate::wal::{WalEntry, WalReader, WalWriter};
use crossbeam_epoch::{self as epoch, Atomic, Owned};
use drift_cache::block_cache::BlockCache;
use drift_cache::{PageId, PageManager};
use drift_kv::bitstore::BitStore;
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
    pub(crate) buckets: Atomic<HashMap<u32, BucketHeader>>,
    pub cache: Arc<BlockCache<BucketData>>,
    pub(crate) quantizer: RwLock<Option<Arc<Quantizer>>>,
    pub(crate) next_bucket_id: AtomicU32,
    pub(crate) wal: Mutex<WalWriter>,
    pub(crate) kv: Arc<BitStore>,
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

        if wal_path.exists() {
            let reader = WalReader::open(wal_path)?;
            for entry in reader.read_all() {
                match entry {
                    WalEntry::Insert { id, vector } => memtable.insert(id, &vector),
                    WalEntry::Delete { id } => memtable.delete(id),
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
            quantizer: RwLock::new(None),
            centroids: Atomic::new(Vec::new()),
            next_bucket_id: AtomicU32::new(0),
            buckets: Atomic::new(HashMap::new()),
            kv,
            cache,
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

    // =========================================================================
    //  CORE OPERATIONS
    // =========================================================================

    pub async fn train(&self, samples: &[Vec<f32>]) -> io::Result<()> {
        assert!(!samples.is_empty(), "Empty training set");
        let dim = self.config.dim;

        let q = Arc::new(Quantizer::train(samples));
        *self.quantizer.write().unwrap() = Some(q.clone());

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

            // Populate KV for initial data
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
        {
            let mut wal = self.wal.lock().unwrap();
            wal.write_insert(id, vector)?;
            wal.flush()?;
        }
        let guard = epoch::pin();
        let memtable = unsafe { self.memtable.load(Ordering::Acquire, &guard).as_ref() }.unwrap();
        memtable.insert(id, vector);
        Ok(())
    }

    pub fn delete(&self, id: u64) -> io::Result<()> {
        {
            let mut wal = self.wal.lock().unwrap();
            wal.write_delete(id)?;
            // wal.flush()?;
        }
        let guard = epoch::pin();
        let memtable = unsafe { self.memtable.load(Ordering::Acquire, &guard).as_ref() }.unwrap();
        memtable.delete(id);
        let _ = self.kv.remove(&id.to_le_bytes());
        Ok(())
    }

    // =========================================================================
    //  SEARCH (Async, Scoped Guards)
    // =========================================================================

    pub async fn search_async(
        &self,
        query: &[f32],
        k: usize,
        target_confidence: f32,
        lambda: f32,
        tau: f32,
    ) -> io::Result<Vec<SearchResult>> {
        let l0_results = {
            let guard = epoch::pin();
            let memtable =
                unsafe { self.memtable.load(Ordering::Acquire, &guard).as_ref() }.unwrap();
            memtable.search(query, k, self.config.ef_search)
        };

        let quantizer_arc = {
            let guard = self.quantizer.read().unwrap();
            match guard.as_ref() {
                Some(q) => q.clone(),
                None => {
                    return Ok(l0_results
                        .into_iter()
                        .map(|(id, d)| SearchResult { id, distance: d })
                        .collect());
                }
            }
        };

        let selected_headers = {
            let guard = epoch::pin();
            let buckets_map =
                unsafe { self.buckets.load(Ordering::Acquire, &guard).as_ref() }.unwrap();

            if buckets_map.is_empty() {
                Vec::new()
            } else {
                let mut candidates: Vec<(&BucketHeader, f32)> = buckets_map
                    .values()
                    .map(|header| {
                        let dist = crate::math::l2_sq(query, &header.centroid).sqrt();
                        let p_geom = (-lambda * dist).exp();
                        let p_density = 1.0 - (-(header.count as f32) / tau).exp();
                        (header, p_geom * p_density)
                    })
                    .collect();

                candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(CmpOrdering::Equal));

                let mut headers = Vec::new();
                let mut acc_conf = 0.0;
                for (header, score) in &candidates {
                    acc_conf += score;
                    headers.push((*header).clone());
                    if acc_conf >= target_confidence {
                        break;
                    }
                }

                // FALLBACK: Ensure at least top 1 bucket is probed
                if headers.is_empty() && !candidates.is_empty() {
                    headers.push(candidates[0].0.clone());
                }
                headers
            }
        };

        let mut loaded_data = Vec::with_capacity(selected_headers.len());
        for header in selected_headers {
            match self.cache.get(&header.page_id).await {
                Ok(data) => loaded_data.push(data),
                Err(e) => eprintln!("Failed to load bucket {}: {}", header.id, e),
            }
        }

        let mut heap = BinaryHeap::with_capacity(k);
        for (id, dist) in l0_results {
            let res = SearchResult { id, distance: dist };
            if heap.len() < k {
                heap.push(res);
            } else if dist < heap.peek().unwrap().distance {
                heap.pop();
                heap.push(res);
            }
        }

        for data in loaded_data {
            let hits = Bucket::scan_static(&data, &quantizer_arc, query);
            for res in hits {
                if heap.len() < k {
                    heap.push(res);
                } else if res.distance < heap.peek().unwrap().distance {
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

    // =========================================================================
    //  MAINTENANCE (Hydration & Split)
    // =========================================================================

    pub async fn force_register_bucket_with_ids(
        &self,
        id: u32,
        ids: &[u64],
        vectors: &[Vec<f32>],
    ) -> io::Result<()> {
        if vectors.is_empty() {
            return Ok(());
        }
        let q_arc = self
            .quantizer
            .read()
            .unwrap()
            .as_ref()
            .expect("Quantizer missing")
            .clone();
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

        let mut data = BucketData {
            codes: crate::aligned::AlignedBytes::new(vectors.len() * dim),
            vids: Vec::with_capacity(vectors.len()),
            tombstones: bit_set::BitSet::with_capacity(vectors.len()),
        };
        for (i, v) in vectors.iter().enumerate() {
            let code = q_arc.encode(v);
            data.vids.push(ids[i]);
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

        // FIX: Remove existing centroid before adding new one to avoid duplicates
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
                BucketHeader::new(id, centroid.clone(), ids.len() as u32, page_id.clone()),
            );
            new
        });

        // FIX: Update KV Store
        for &vid in ids {
            let _ = self.kv.put(&vid.to_le_bytes(), &id.to_le_bytes());
        }

        Ok(())
    }

    pub async fn split_and_steal(&self, bucket_id: u32) -> io::Result<()> {
        let q_arc = self.quantizer.read().unwrap().as_ref().unwrap().clone();

        let (header, all_centroids) = {
            let guard = epoch::pin();
            let buckets = unsafe { self.buckets.load(Ordering::Acquire, &guard).as_ref() }.unwrap();
            let centroids =
                unsafe { self.centroids.load(Ordering::Acquire, &guard).as_ref() }.unwrap();
            let h = match buckets.get(&bucket_id) {
                Some(h) => h.clone(),
                None => return Ok(()),
            };
            (h, centroids.clone())
        };

        let data_arc = match self.cache.get(&header.page_id).await {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };
        let (vecs, ids) = data_arc.reconstruct(&q_arc);

        // FIX: Lower threshold for testing
        if vecs.len() < 10 {
            return Ok(());
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
            let (mut n_vecs, mut n_ids) = n_data.reconstruct(&q_arc);

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

        Ok(())
    }

    // =========================================================================
    //  MAINTENANCE: SCATTER MERGE
    // =========================================================================

    fn find_nearest_bucket_exclude(
        &self,
        vec: &[f32],
        centroids: &[CentroidEntry],
        exclude: u32,
    ) -> Option<u32> {
        let mut best_id = None;
        let mut min_dist = f32::MAX;
        for c in centroids {
            if !c.active || c.id == exclude {
                continue;
            }
            let d = crate::math::l2_sq(vec, &c.vector);
            if d < min_dist {
                min_dist = d;
                best_id = Some(c.id);
            }
        }
        best_id
    }

    fn find_top_k_neighbors(
        &self,
        query_centroid: &[f32],
        k: usize,
        exclude_id: u32,
        all_centroids: &[CentroidEntry],
    ) -> Vec<u32> {
        let mut candidates: Vec<(u32, f32)> = all_centroids
            .iter()
            .filter(|c| c.active && c.id != exclude_id)
            .map(|c| (c.id, crate::math::l2_sq(query_centroid, &c.vector)))
            .collect();
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        candidates.into_iter().take(k).map(|(id, _)| id).collect()
    }

    pub async fn scatter_merge(&self, zombie_id: u32) -> io::Result<()> {
        let q_arc = self.quantizer.read().unwrap().as_ref().unwrap().clone();

        let (z_header, all_centroids) = {
            let guard = epoch::pin();
            let buckets = unsafe { self.buckets.load(Ordering::Acquire, &guard).as_ref() }.unwrap();
            let centroids =
                unsafe { self.centroids.load(Ordering::Acquire, &guard).as_ref() }.unwrap();
            let h = match buckets.get(&zombie_id) {
                Some(h) => h.clone(),
                None => return Ok(()),
            };
            (h, centroids.clone())
        };

        let data_arc = match self.cache.get(&z_header.page_id).await {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };
        let (orphans_vec, orphans_id) = data_arc.reconstruct(&q_arc);

        if orphans_vec.is_empty() {
            self.atomic_remove_bucket(zombie_id);
            return Ok(());
        }

        // FIX: Top-3 Constraint
        let neighbor_ids =
            self.find_top_k_neighbors(&z_header.centroid, 3, zombie_id, &all_centroids);
        if neighbor_ids.is_empty() {
            for (i, vec) in orphans_vec.iter().enumerate() {
                self.insert(orphans_id[i], vec)?;
            }
            self.atomic_remove_bucket(zombie_id);
            return Ok(());
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

            let (mut vs, mut is) = t_data.reconstruct(&q_arc);
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
        Ok(())
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

    pub fn rotate_memtable(&self) -> io::Result<Vec<(u64, Vec<f32>)>> {
        let mut wal = self.wal.lock().unwrap();
        let guard = epoch::pin();
        let current_memtable_ref =
            unsafe { self.memtable.load(Ordering::Acquire, &guard).as_ref() }.unwrap();
        let data = current_memtable_ref.extract_all();

        let new_memtable = Arc::new(MemTable::new(
            self.config.max_bucket_capacity * 10,
            self.config.dim,
            self.config.ef_construction,
            16,
        ));
        self.memtable
            .store(Owned::new(new_memtable), Ordering::Release);
        wal.truncate()?;
        Ok(data)
    }

    pub fn get_quantizer(&self) -> Option<Arc<Quantizer>> {
        self.quantizer.read().unwrap().clone()
    }

    pub fn set_quantizer(&self, q: Quantizer) {
        *self.quantizer.write().unwrap() = Some(Arc::new(q));
    }

    pub async fn rebalance_buckets(&self, id_a: u32, id_b: u32) -> io::Result<()> {
        let q_arc = self.quantizer.read().unwrap().as_ref().unwrap().clone();
        let (h_a, h_b) = {
            let guard = epoch::pin();
            let buckets = unsafe { self.buckets.load(Ordering::Acquire, &guard).as_ref() }.unwrap();
            match (buckets.get(&id_a), buckets.get(&id_b)) {
                (Some(a), Some(b)) => (a.clone(), b.clone()),
                _ => return Ok(()),
            }
        };

        let d_a = match self.cache.get(&h_a.page_id).await {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };
        let d_b = match self.cache.get(&h_b.page_id).await {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        let (mut vs, mut is) = d_a.reconstruct(&q_arc);
        let (v_b, i_b) = d_b.reconstruct(&q_arc);
        vs.extend(v_b);
        is.extend(i_b);
        if vs.is_empty() {
            return Ok(());
        }

        let trainer = KMeansTrainer::new(2, self.config.dim, 10);
        let result = trainer.train(&vs);

        let mut v1 = Vec::new();
        let mut i1 = Vec::new();
        let mut v2 = Vec::new();
        let mut i2 = Vec::new();

        for (i, &c) in result.assignments.iter().enumerate() {
            if c == 0 {
                v1.push(vs[i].clone());
                i1.push(is[i]);
            } else {
                v2.push(vs[i].clone());
                i2.push(is[i]);
            }
        }

        let write_page =
            async |bid: u32, vecs: &[Vec<f32>], vids: &[u64]| -> io::Result<(PageId, u32)> {
                let dim = self.config.dim;
                let mut data = BucketData {
                    codes: crate::aligned::AlignedBytes::new(vecs.len() * dim),
                    vids: Vec::with_capacity(vecs.len()),
                    tombstones: bit_set::BitSet::with_capacity(vecs.len()),
                };
                for (i, v) in vecs.iter().enumerate() {
                    let code = q_arc.encode(v);
                    data.vids.push(vids[i]);
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
                    vecs.len() as u32,
                ))
            };

        let (p1, c1) = write_page(id_a, &v1, &i1).await?;
        let (p2, c2) = write_page(id_b, &v2, &i2).await?;

        self.update_centroids(|c| {
            let mut new = c.clone();
            if let Some(x) = new.iter_mut().find(|x| x.id == id_a) {
                x.vector = result.centroids[0].clone();
                x.active = true;
            }
            if let Some(x) = new.iter_mut().find(|x| x.id == id_b) {
                x.vector = result.centroids[1].clone();
                x.active = true;
            }
            new
        });
        self.update_buckets(|b| {
            let mut new = b.clone();
            new.insert(
                id_a,
                BucketHeader::new(id_a, result.centroids[0].clone(), c1, p1.clone()),
            );
            new.insert(
                id_b,
                BucketHeader::new(id_b, result.centroids[1].clone(), c2, p2.clone()),
            );
            new
        });

        for id in i1 {
            let _ = self.kv.put(&id.to_le_bytes(), &id_a.to_le_bytes());
        }
        for id in i2 {
            let _ = self.kv.put(&id.to_le_bytes(), &id_b.to_le_bytes());
        }

        Ok(())
    }
}
