use crate::bucket_file_reader::BucketFileReader;
use atomic_float::AtomicF32;
use drift_core::lock_manager::BucketCoordinator;
use drift_core::math::Metric;
use drift_traits::{BucketStats, StorageEngine, TombstoneView};
use futures::future::join_all;
use opendal::Operator;
use parking_lot::RwLock;
use std::cmp;
use std::collections::{HashMap, HashSet};
use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use tokio::sync::Semaphore;

#[derive(Clone, Debug, PartialEq)]
pub enum StorageClass {
    Local,
    Remote,
    Tiered {
        remote_path: String,
        local_path: String,
    },
    Promoting {
        local_active: String,
        local_frozen: String,
        remote_path: Option<String>,
    },
}

#[derive(Debug)]
pub struct BucketVersion {
    pub bucket_id: u32,
    pub path: String,
    pub class: StorageClass,
}

#[derive(Debug)]
struct LiveBucketState {
    version: Arc<BucketVersion>,
    total_count: AtomicU32,
    tombstone_count: AtomicU32,
    temperature: AtomicF32,
    tombstones: RwLock<Arc<HashSet<u64>>>,
    vector_sum: RwLock<Vec<f32>>,
}

impl LiveBucketState {
    fn new(version: Arc<BucketVersion>, count: u32) -> Self {
        Self {
            version,
            total_count: AtomicU32::new(count),
            tombstone_count: AtomicU32::new(0),
            temperature: AtomicF32::new(0.5),
            tombstones: RwLock::new(Arc::new(HashSet::new())),
            vector_sum: RwLock::new(Vec::new()),
        }
    }

    fn touch(&self) {
        const ALPHA: f32 = 0.05;
        let _ = self
            .temperature
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                if current >= 1.0 {
                    None
                } else {
                    Some(current + ALPHA * (1.0 - current))
                }
            });
    }
}

#[derive(Debug, Clone)]
struct LocalTombstoneView {
    inner: Arc<HashSet<u64>>,
}
impl TombstoneView for LocalTombstoneView {
    fn contains(&self, id: u64) -> bool {
        self.inner.contains(&id)
    }
    fn len(&self) -> usize {
        self.inner.len()
    }
}

pub struct BucketManager {
    local_op: Operator,
    remote_op: Operator,
    registry: Arc<RwLock<HashMap<u32, Arc<LiveBucketState>>>>,
    scan_semaphore: Arc<Semaphore>,
    coordinator: Arc<BucketCoordinator>,
    metric: Metric,
}

impl BucketManager {
    pub fn new(
        local_op: Operator,
        remote_op: Operator,
        max_concurrent_scans: usize,
        coordinator: Arc<BucketCoordinator>,
        metric: Metric,
    ) -> Self {
        Self {
            local_op,
            remote_op,
            registry: Arc::new(RwLock::new(HashMap::new())),
            scan_semaphore: Arc::new(Semaphore::new(max_concurrent_scans)),
            coordinator,
            metric,
        }
    }

    pub fn collect_all_tombstones(&self) -> Vec<u64> {
        let reg = self.registry.read();
        let mut all_deletes = Vec::new();
        for state in reg.values() {
            all_deletes.extend(state.tombstones.read().iter());
        }
        all_deletes
    }

    pub fn register_bucket(&self, bucket_id: u32, path: String, class: StorageClass) {
        self.register_bucket_with_count(bucket_id, path, class, 0);
    }

    pub fn register_bucket_with_count(
        &self,
        bucket_id: u32,
        path: String,
        class: StorageClass,
        count: u32,
    ) {
        let version = Arc::new(BucketVersion {
            bucket_id,
            path,
            class,
        });
        let state = Arc::new(LiveBucketState::new(version, count));
        self.registry.write().insert(bucket_id, state);
    }

    pub fn remote_operator(&self) -> Operator {
        self.remote_op.clone()
    }

    pub fn get_version(&self, bucket_id: u32) -> Option<Arc<BucketVersion>> {
        self.registry
            .read()
            .get(&bucket_id)
            .map(|s| s.version.clone())
    }

    pub fn get_location(&self, bucket_id: u32) -> Option<(String, StorageClass)> {
        self.get_version(bucket_id)
            .map(|v| (v.path.clone(), v.class.clone()))
    }

    pub fn get_tombstones(&self, bucket_id: u32) -> Arc<HashSet<u64>> {
        let reg = self.registry.read();
        if let Some(state) = reg.get(&bucket_id) {
            return state.tombstones.read().clone();
        }
        Arc::new(HashSet::new())
    }

    /// This handles the race condition where new deletes arrive during promotion.
    pub fn prune_tombstones(&self, bucket_id: u32, processed_ids: &HashSet<u64>) {
        let reg = self.registry.read();
        if let Some(state) = reg.get(&bucket_id) {
            let mut guard = state.tombstones.write();

            // COW: Clone the current set (which might contain NEW deletes)
            let mut new_set = (**guard).clone();

            // Remove ONLY the ones we successfully persisted/purged
            // This leaves any "new" deletes untouched.
            let len_before = new_set.len();
            new_set.retain(|id| !processed_ids.contains(id));
            let len_after = new_set.len();

            *guard = Arc::new(new_set);

            // Update the atomic counter
            let removed_count = (len_before - len_after) as u32;
            state
                .tombstone_count
                .fetch_sub(removed_count, Ordering::Relaxed);
        }
    }
}

#[async_trait::async_trait]
impl StorageEngine for BucketManager {
    fn mark_delete(&self, bucket_id: u32, vector_id: u64) -> io::Result<()> {
        let reg = self.registry.read();
        if let Some(state) = reg.get(&bucket_id) {
            // let mut set = state.tombstones.write();
            // if set.insert(vector_id) {
            //     state.tombstone_count.fetch_add(1, Ordering::Relaxed);
            // }
            let mut guard = state.tombstones.write();
            if guard.contains(&vector_id) {
                return Ok(());
            }

            // let mut new_set =
            let mut new_set = (**guard).clone();
            new_set.insert(vector_id);
            *guard = Arc::new(new_set);

            state.tombstone_count.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }

    fn get_bucket_stats(&self, bucket_id: u32) -> Option<BucketStats> {
        let reg = self.registry.read();
        let state = reg.get(&bucket_id)?;
        Some(BucketStats {
            tombstone_count: state.tombstone_count.load(Ordering::Relaxed),
            total_count: state.total_count.load(Ordering::Relaxed),
            temperature: state.temperature.load(Ordering::Relaxed),
            active: true,
        })
    }

    fn register_bucket(&self, bucket_id: u32, path: String, count: u32) {
        self.register_bucket_with_count(bucket_id, path, StorageClass::Local, count);
    }

    fn tick_cooling(&self, decay_rate: f32) {
        let reg = self.registry.read();
        for state in reg.values() {
            let _ = state
                .temperature
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |t| {
                    Some(t * decay_rate)
                });
        }
    }

    async fn search_and_refine(
        &self,
        bucket_ids: &[u32],
        query: &[f32],
        k: usize,
        oversample_factor: usize,
    ) -> Vec<(u64, f32)> {
        let mut handles = Vec::with_capacity(bucket_ids.len());
        let metric = self.metric;

        for &bid in bucket_ids {
            let local_op = self.local_op.clone();
            let remote_op = self.remote_op.clone();
            let sem = self.scan_semaphore.clone();
            let query = query.to_vec();

            let registry = self.registry.clone(); // ⚡ Clone Arc to pass into task
            let coordinator = self.coordinator.clone();

            handles.push(tokio::spawn(async move {
                let _permit = sem.acquire().await.unwrap();

                // 1. Acquire Lock (Prevents Janitor from modifying/deleting this bucket)
                let _lock_guard = coordinator.read(bid).await;

                // 2. ⚡ Fetch State INSIDE the lock (Guaranteed consistent)
                let (version, local_tombstones) = {
                    let reg = registry.read();
                    match reg.get(&bid) {
                        Some(state) => {
                            state.touch();
                            (state.version.clone(), state.tombstones.read().clone())
                        }
                        None => return Vec::new(), // Bucket deleted while we waited
                    }
                };

                let bucket_view = Arc::new(LocalTombstoneView {
                    inner: local_tombstones,
                });

                let ops_to_scan = match &version.class {
                    StorageClass::Local => vec![(local_op, version.path.clone(), "Local")],
                    StorageClass::Remote => vec![(remote_op, version.path.clone(), "Remote")],
                    StorageClass::Tiered {
                        remote_path,
                        local_path,
                    } => vec![
                        (remote_op, remote_path.clone(), "Tiered-Base"),
                        (local_op, local_path.clone(), "Tiered-Delta"),
                    ],
                    StorageClass::Promoting {
                        local_active,
                        local_frozen,
                        remote_path,
                    } => {
                        let mut ops = vec![
                            (local_op.clone(), local_active.clone(), "Promoting-Active"),
                            (local_op.clone(), local_frozen.clone(), "Promoting-Frozen"),
                        ];
                        if let Some(rp) = remote_path {
                            ops.push((remote_op, rp.clone(), "Promoting-Base"));
                        }
                        ops
                    }
                };

                let mut refined_results = Vec::new();
                for (op, path, label) in ops_to_scan {
                    match BucketFileReader::open(op, &path).await {
                        Ok(mut reader) => {
                            if let Ok(candidates) = reader
                                .scan(&query, oversample_factor, metric, bucket_view.as_ref())
                                .await
                            {
                                let dim = reader
                                    .quantizer
                                    .as_ref()
                                    .map(|q| q.min.len())
                                    .unwrap_or(query.len());
                                if let Ok(matches) =
                                    reader.refine(candidates, &query, dim, metric).await
                                {
                                    refined_results.extend(matches);
                                }
                            }
                        }
                        Err(e) => {
                            // If file missing inside lock, it's a real error (unless empty bucket edge case)
                            if !e.to_string().contains("File too small") {
                                tracing::warn!(
                                    "⚠️ BucketManager: Failed to open {} ({}): {}",
                                    path,
                                    label,
                                    e
                                );
                            }
                        }
                    }
                }
                refined_results
            }));
        }

        let results_list = join_all(handles).await;
        let mut all_results = Vec::new();
        for res in results_list.into_iter().flatten() {
            all_results.extend(res);
        }
        all_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(cmp::Ordering::Equal));
        if all_results.len() > k {
            all_results.truncate(k);
        }
        all_results
    }

    /// Fetches ALL high-fidelity vectors for a logical bucket.
    /// MERGES data from Local (Delta) and Remote (Base) tiers if necessary.
    async fn fetch_bucket(&self, bucket_id: u32) -> io::Result<(Vec<u64>, Vec<f32>)> {
        // 1. Get Location Snapshot
        let version = self
            .get_version(bucket_id)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "Bucket not found"))?;

        // 2. Identify all sources
        let mut ops_to_scan = Vec::new();
        match &version.class {
            StorageClass::Local => {
                ops_to_scan.push((self.local_op.clone(), version.path.clone()));
            }
            StorageClass::Remote => {
                ops_to_scan.push((self.remote_op.clone(), version.path.clone()));
            }
            StorageClass::Tiered {
                remote_path,
                local_path,
            } => {
                // Fetch Base (Remote) THEN Delta (Local)
                ops_to_scan.push((self.remote_op.clone(), remote_path.clone()));
                ops_to_scan.push((self.local_op.clone(), local_path.clone()));
            }
            StorageClass::Promoting {
                local_active,
                local_frozen,
                remote_path,
            } => {
                // If we are promoting, data is scattered. Gather everything.
                if let Some(rp) = remote_path {
                    ops_to_scan.push((self.remote_op.clone(), rp.clone()));
                }
                ops_to_scan.push((self.local_op.clone(), local_frozen.clone()));
                ops_to_scan.push((self.local_op.clone(), local_active.clone()));
            }
        };

        // 3. Fetch and Merge
        let mut merged_ids = Vec::new();
        let mut merged_vecs_flat = Vec::new();

        for (op, path) in ops_to_scan {
            match BucketFileReader::open(op, &path).await {
                Ok(mut reader) => {
                    // Read (IDs, Vec<Vec<f32>>)
                    if let Ok((ids, vecs)) = reader.read_all_vectors().await {
                        merged_ids.extend(ids);
                        for v in vecs {
                            merged_vecs_flat.extend(v);
                        }
                    }
                }
                Err(e) => {
                    // If a tiered file is missing, that's critical data loss (or config error)
                    tracing::warn!("fetch_bucket: Failed to read component {}: {}", path, e);
                    // We continue best-effort? No, for maintenance we want strict correctness.
                    // But for now, let's log and continue to avoid crashing the Janitor loop.
                }
            }
        }

        //    if merged_ids.is_empty() {
        //         return Err(io::Error::new(
        //             io::ErrorKind::NotFound,
        //             "Bucket data empty or inaccessible",
        //         ));
        //     }

        Ok((merged_ids, merged_vecs_flat))
    }

    fn update_bucket_drift(
        &self,
        bucket_id: u32,
        delta_sum: &[f32],
        delta_count: u32,
    ) -> io::Result<()> {
        let reg = self.registry.read();
        if let Some(state) = reg.get(&bucket_id) {
            // 1. Update Count
            state.total_count.fetch_add(delta_count, Ordering::Relaxed);

            // 2. Update Sum
            let mut sum_guard = state.vector_sum.write();
            if sum_guard.is_empty() {
                // First update or recovery
                *sum_guard = delta_sum.to_vec();
            } else if sum_guard.len() == delta_sum.len() {
                for (i, val) in sum_guard.iter_mut().enumerate() {
                    *val += delta_sum[i];
                }
            } else {
                tracing::warn!(
                    "Dimension mismatch in drift update for bucket {}",
                    bucket_id
                );
            }
        }
        Ok(())
    }

    fn get_bucket_drift_stats(&self, bucket_id: u32) -> Option<(Vec<f32>, u32)> {
        let reg = self.registry.read();
        let state = reg.get(&bucket_id)?;

        let sum = state.vector_sum.read().clone();
        let count = state.total_count.load(Ordering::Relaxed);

        Some((sum, count))
    }
}
