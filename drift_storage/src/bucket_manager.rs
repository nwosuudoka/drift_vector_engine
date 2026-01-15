use crate::bucket_file_reader::BucketFileReader;
use drift_core::lock_manager::BucketCoordinator;
use drift_traits::{DiskSearcher, TombstoneView};
use futures::future::join_all;
use opendal::Operator;
use parking_lot::RwLock;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::Arc;
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

pub struct BucketManager {
    local_op: Operator,
    remote_op: Operator,
    registry: Arc<RwLock<HashMap<u32, Arc<BucketVersion>>>>,
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

    pub fn register_bucket(
        &self,
        bucket_id: u32,
        path: String,
        class: StorageClass,
    ) -> Option<Arc<BucketVersion>> {
        let version = Arc::new(BucketVersion {
            bucket_id,
            path,
            class,
        });
        self.registry.write().insert(bucket_id, version)
    }

    pub fn deregister_bucket(&self, bucket_id: u32) -> Option<Arc<BucketVersion>> {
        self.registry.write().remove(&bucket_id)
    }

    pub fn get_version(&self, bucket_id: u32) -> Option<Arc<BucketVersion>> {
        self.registry.read().get(&bucket_id).cloned()
    }

    pub fn get_location(&self, bucket_id: u32) -> Option<(String, StorageClass)> {
        self.get_version(bucket_id)
            .map(|v| (v.path.clone(), v.class.clone()))
    }
}

// #[derive(PartialEq)]
// struct CandidateWrapper(SearchCandidate);
// impl Eq for CandidateWrapper {}
// impl Ord for CandidateWrapper {
//     fn cmp(&self, other: &Self) -> Ordering {
//         self.0
//             .approx_dist
//             .partial_cmp(&other.0.approx_dist)
//             .unwrap_or(Ordering::Equal)
//     }
// }
// impl PartialOrd for CandidateWrapper {
//     fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
//         Some(self.cmp(other))
//     }
// }

#[async_trait::async_trait]
impl DiskSearcher for BucketManager {
    async fn search_and_refine(
        &self,
        bucket_ids: &[u32],
        query: &[f32],
        k: usize,
        oversample_factor: usize,
        tombstones: Arc<dyn TombstoneView>,
    ) -> Vec<(u64, f32)> {
        let mut handles = Vec::with_capacity(bucket_ids.len());

        for &bid in bucket_ids {
            let local_op = self.local_op.clone();
            let remote_op = self.remote_op.clone();
            let sem = self.scan_semaphore.clone();
            let query = query.to_vec();
            let tombstones = tombstones.clone();
            let coordinator = self.coordinator.clone();
            let registry_ref = self.registry.clone();

            handles.push(tokio::spawn(async move {
                let _permit = sem.acquire().await.unwrap();

                //  CRITICAL: Acquire Lock ONCE
                // This lock is held for both the Scan AND Refine phases.
                // The Janitor cannot swap the file underneath us.
                let _lock_guard = coordinator.read(bid).await;

                // 2. Get Version
                let version = {
                    let reg = registry_ref.read();
                    reg.get(&bid).cloned()
                };

                let version = match version {
                    Some(v) => v,
                    None => return Vec::new(),
                };

                // 3. Resolve Paths
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
                let scan_k = oversample_factor; // Scan more than we need

                for (op, path, label) in ops_to_scan {
                    match BucketFileReader::open(op, &path).await {
                        Ok(mut reader) => {
                            // A. SCAN (Approximate)
                            if let Ok(candidates) =
                                reader.scan(&query, scan_k, tombstones.as_ref()).await
                            {
                                // B. REFINE (Exact) - IMMEDIATELY
                                // Since we still hold _lock_guard and the reader is open on the correct file,
                                // the offsets in 'candidates' are guaranteed to be valid for this 'reader'.
                                let dim = reader
                                    .quantizer
                                    .as_ref()
                                    .map(|q| q.min.len())
                                    .unwrap_or(query.len());

                                match reader.refine(candidates, &query, dim).await {
                                    Ok(exact_matches) => refined_results.extend(exact_matches),
                                    Err(e) => {
                                        tracing::error!(
                                            "Refine failed for {} ({}): {}",
                                            path,
                                            label,
                                            e
                                        );
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            if !e.to_string().contains("File too small") {
                                tracing::warn!("⚠️ Open Failed for {} ({}): {}", path, label, e);
                            }
                        }
                    }
                }

                // _lock_guard is dropped HERE.
                // It is safe for the Janitor to delete the file now, because we have extracted the data.
                refined_results
            }));
        }

        // 4. Merge Results
        let results_list = join_all(handles).await;

        let mut all_results = Vec::new();
        for res in results_list.into_iter().flatten() {
            all_results.extend(res);
        }

        // 5. Global Sort & Top-K
        // Sort by distance ascending
        all_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        if all_results.len() > k {
            all_results.truncate(k);
        }

        all_results
    }
}
