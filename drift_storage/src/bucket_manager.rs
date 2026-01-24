use crate::bucket_file_reader::BucketFileReader;
use async_trait::async_trait;
use atomic_float::AtomicF32;
use drift_core::lock_manager::BucketCoordinator;
use drift_traits::{BucketStats, DataProvider, StorageEngine, TombstoneView};
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
}

impl LiveBucketState {
    fn new(version: Arc<BucketVersion>, count: u32) -> Self {
        Self {
            version,
            total_count: AtomicU32::new(count),
            tombstone_count: AtomicU32::new(0),
            temperature: AtomicF32::new(0.5),
            tombstones: RwLock::new(Arc::new(HashSet::new())),
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
}

impl BucketManager {
    pub fn new(
        local_op: Operator,
        remote_op: Operator,
        max_concurrent_scans: usize,
        coordinator: Arc<BucketCoordinator>,
    ) -> Self {
        Self {
            local_op,
            remote_op,
            registry: Arc::new(RwLock::new(HashMap::new())),
            scan_semaphore: Arc::new(Semaphore::new(max_concurrent_scans)),
            coordinator,
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
                                .scan(&query, oversample_factor, bucket_view.as_ref())
                                .await
                            {
                                let dim = reader
                                    .quantizer
                                    .as_ref()
                                    .map(|q| q.min.len())
                                    .unwrap_or(query.len());
                                if let Ok(matches) = reader.refine(candidates, &query, dim).await {
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
}

#[async_trait]
impl DataProvider for BucketManager {
    async fn fetch_bucket(&self, bucket_id: u32) -> io::Result<(Vec<u64>, Vec<f32>)> {
        // 1. Get Location
        let version = self
            .get_version(bucket_id)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "Bucket not found"))?;

        let ops_to_try = match &version.class {
            StorageClass::Local => vec![(self.local_op.clone(), version.path.clone())],
            StorageClass::Remote => vec![(self.remote_op.clone(), version.path.clone())],
            // For Tiered, we ideally need the merged view.
            // MVP: Just fetch Remote (Base) + Local (Delta) and merge in memory.
            // For now, let's implement the simple case (Remote or Local).
            StorageClass::Tiered { remote_path, .. } => {
                vec![(self.remote_op.clone(), remote_path.clone())]
            }
            StorageClass::Promoting { local_active, .. } => {
                vec![(self.local_op.clone(), local_active.clone())]
            }
        };

        for (op, path) in ops_to_try {
            if let Ok(mut reader) = BucketFileReader::open(op, &path).await {
                // We need a method on Reader to get EVERYTHING (IDs + Floats)
                // BucketFileReader already has read_all_vectors() -> (Vec<u64>, Vec<Vec<f32>>)
                // We need flattened floats.
                if let Ok((ids, vecs)) = reader.read_all_vectors().await {
                    let flat: Vec<f32> = vecs.into_iter().flatten().collect();
                    return Ok((ids, flat));
                }
            }
        }

        Err(io::Error::new(
            io::ErrorKind::NotFound,
            "Failed to read bucket data",
        ))
    }
}
