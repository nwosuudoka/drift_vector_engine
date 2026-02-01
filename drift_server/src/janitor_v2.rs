use crate::local_staging::LocalStagingManager;
use crate::manifest::ServerManifestManager;
use crate::persistence_v2::PersistenceManager;
use crate::reaper::Reaper;
use drift_core::{index_v2::VectorIndex, lock_manager::BucketCoordinator};
use drift_storage::bucket_manager::{BucketManager, StorageClass};
use drift_traits::StorageEngine;
use std::sync::Arc;
use std::time::Duration;
use std::{collections::HashMap, io};
use tokio::sync::Mutex;
use tokio::time;
use tracing::{error, info};

pub struct JanitorConfig {
    pub index: Arc<VectorIndex>,
    pub manifest: Arc<ServerManifestManager>,
    pub staging: Arc<LocalStagingManager>,
    pub persistence: Arc<PersistenceManager>,
    pub bucket_manager: Arc<BucketManager>,
    pub check_interval: Duration,
    pub promotion_threshold_bytes: u64,
    pub coordinator: Arc<BucketCoordinator>,
    pub max_bucket_capacity: usize,
    pub drift_threshold: f32, // Default 0.15
    pub split_threshold: f32, // Default 0.8
}

pub struct Janitor {
    index: Arc<VectorIndex>,
    manifest: Arc<ServerManifestManager>,
    staging: Arc<LocalStagingManager>,
    persistence: Arc<PersistenceManager>,
    bucket_manager: Arc<BucketManager>,
    check_interval: Duration,
    promotion_threshold_bytes: u64,
    reaper: Mutex<Reaper>,
    coordinator: Arc<BucketCoordinator>,
    max_bucket_capacity: usize,
    drift_threshold: f32,
    split_threshold: f32,
}

impl Janitor {
    pub fn new(config: JanitorConfig) -> Self {
        let reaper = Mutex::new(Reaper::new(
            config.staging.clone(),
            config.persistence.clone(),
        ));
        Self {
            index: config.index,
            manifest: config.manifest,
            staging: config.staging,
            persistence: config.persistence,
            bucket_manager: config.bucket_manager,
            check_interval: config.check_interval,
            promotion_threshold_bytes: config.promotion_threshold_bytes,
            reaper,
            coordinator: config.coordinator,
            max_bucket_capacity: config.max_bucket_capacity,
            drift_threshold: config.drift_threshold,
            split_threshold: config.split_threshold,
        }
    }

    pub async fn run(&self) {
        let mut interval = time::interval(self.check_interval);
        info!("Janitor: Started.");

        loop {
            interval.tick().await;

            // 1. Flush Frozen MemTables
            if let Err(e) = self.perform_flush().await {
                error!("Janitor: Flush failed: {}", e);
            }

            // 2. Maintenance (Splits)
            self.check_maintainance().await;

            // 3. Run Reaper (Garbage Collection)
            let mut reaper = self.reaper.lock().await;
            reaper.run_cycle().await;
            drop(reaper); // Release lock before long operations

            // 4. Check for Promotions (Local -> S3)
            if let Err(e) = self.promote_segments().await {
                error!("Janitor: Promotion failed: {}", e);
            }
        }
    }

    pub(crate) async fn check_maintainance(&self) {
        // Simple heuristic: Scan one bucket per tick to find candidates
        // In prod, use a PriorityQueue or "Dirty Set"

        // This relies on BucketManager implementing get_all_bucket_ids or similar
        // Or we iterate Manifest state.
        let buckets = self.manifest.get_state().get_buckets().clone();
        let max_cap = self.max_bucket_capacity as f32;

        let (router_centroids, router_ids) = self.index.get_router().read().get_snapshot();

        // Map BucketID -> Centroid
        let centroid_map: HashMap<u32, Vec<f32>> = router_ids
            .iter()
            .zip(router_centroids.chunks(self.index.get_dim()))
            .map(|(id, vec)| (*id, vec.to_vec()))
            .collect();

        for b in buckets {
            // 1. Get Live Stats from BucketManager
            let (current_sum, current_count) =
                match self.bucket_manager.get_bucket_drift_stats(b.id) {
                    Some(s) => s,
                    None => continue, // Bucket might be deleted
                };

            // 2. Get Target Centroid
            let target_centroid = match centroid_map.get(&b.id) {
                Some(c) => c,
                None => continue,
            };

            // 3. Calculate Drift
            // Drift = Dist(Mean, Centroid)
            let drift_score = if current_count > 0 && !current_sum.is_empty() {
                let dim = current_sum.len();
                let n = current_count as f32;
                let mut dist_sq = 0.0;

                for i in 0..dim {
                    let mean = current_sum[i] / n;
                    let diff = mean - target_centroid[i];
                    dist_sq += diff * diff;
                }
                dist_sq.sqrt()
            } else {
                0.0
            };

            let cap_ratio = current_count as f32 / max_cap;

            // 4. Decision: Split if Full OR (Mostly Full AND Drifted)
            // 0.15 is the Drift Threshold from the paper.
            if cap_ratio > 1.0
                || (cap_ratio > self.split_threshold && drift_score > self.drift_threshold)
            {
                info!(
                    "Janitor: Triggering Split for Bucket {} (Cap: {:.2}, Drift: {:.4})",
                    b.id, cap_ratio, drift_score
                );

                if let Err(e) = self.perform_split(b.id).await {
                    error!("Janitor: Split failed for {}: {}", b.id, e);
                }
                break; // Throttle
            }
        }
    }

    async fn perform_flush(&self) -> io::Result<()> {
        // 1. Data Flush Logic (Unchanged from previous plan)
        let (partitions, wal_ids) = match self.index.flush_frozen() {
            Some((p, w)) => (p, w),
            // Important: Even if no data to flush, we should occasionally flush tombstones?
            // For simplicity, we couple them. If no data flush, no tombstone flush yet.
            None => return Ok(()),
        };

        let mut updates = Vec::new();
        for (bucket_id, group) in &partitions {
            let new_count = self.staging.append_batch(*bucket_id, group).await?;
            updates.push((*bucket_id, new_count, group.centroid.clone()));

            // If we update stats for a non-existent bucket, BucketManager ignores it.
            if self.bucket_manager.get_version(*bucket_id).is_none() {
                let filename = self.staging.get_active_filename(*bucket_id);
                self.bucket_manager.register_bucket(
                    *bucket_id,
                    filename,
                    drift_storage::bucket_manager::StorageClass::Local,
                );
            }

            // Calculate Delta Sum for Drift Tracking
            // We iterate the flat vector buffer
            if group.count > 0 {
                let dim = group.flat_vectors.len() / group.count;
                let mut delta_sum = vec![0.0; dim];

                for chunk in group.flat_vectors.chunks_exact(dim) {
                    for (i, val) in chunk.iter().enumerate() {
                        delta_sum[i] += val;
                    }
                }

                // Push update to BucketManager
                self.bucket_manager.update_bucket_drift(
                    *bucket_id,
                    &delta_sum,
                    group.count as u32,
                )?;
            }

            if self.bucket_manager.get_version(*bucket_id).is_none() {
                let filename = self.staging.get_active_filename(*bucket_id);
                self.bucket_manager.register_bucket(
                    *bucket_id,
                    filename,
                    drift_storage::bucket_manager::StorageClass::Local,
                );
            }
        }

        // 2. TOMBSTONE PERSISTENCE (Merged L0 + L1)
        let mut all_tombstones = Vec::new();

        // A. Collect L0 (MemTable Deletes)
        {
            // Lock, Clone Arc (cheap), iterate
            let l0_arc = self.index.get_tombstones();
            all_tombstones.extend(l0_arc.iter());
        }

        // B. Collect L1 (Bucket Deletes)
        let l1_deletes = self.bucket_manager.collect_all_tombstones();
        all_tombstones.extend(l1_deletes);

        let mut tombstone_file_opt = None;

        if !all_tombstones.is_empty() {
            // Deduplicate
            all_tombstones.sort_unstable();
            all_tombstones.dedup();

            let run_id = uuid::Uuid::new_v4().to_string();
            let ts_file = self
                .persistence
                .flush_tombstones(&all_tombstones, &run_id)
                .await?;

            info!(
                "Janitor: Persisted {} cumulative tombstones to {}",
                all_tombstones.len(),
                &ts_file
            );
            tombstone_file_opt = Some(ts_file);
        }

        // 3. Update Manifest (Atomic)
        self.manifest.apply_atomic(|m| {
            for (id, count, centroid_opt) in updates {
                let exists = m.get_buckets().iter().any(|b| b.id == id);
                if !exists {
                    m.add_bucket(id, String::new(), centroid_opt);
                }
                m.update_bucket_stats(id, count, 0);
            }

            // ⚡ Update Pointer to NEW cumulative file
            if let Some(tf) = tombstone_file_opt {
                m.inner.tombstone_files = vec![tf];
            }
        })?;

        // 4. Acknowledge
        self.index.acknowledge_flush(&wal_ids)?;

        Ok(())
    }

    async fn promote_segments(&self) -> io::Result<()> {
        let threshold = self.promotion_threshold_bytes;
        let candidates = self.staging.list_large_buckets(threshold)?;

        if candidates.is_empty() {
            return Ok(());
        }

        info!("JanitorV2: Promoting {} buckets to S3...", candidates.len());

        for bucket_id in candidates {
            // 🔒 ACQUIRE HEAVY WRITE LOCK
            // No searchers can read this bucket while we hold this.
            // This guarantees safety without complex Reaper logic.
            let _lock_guard = self.coordinator.write(bucket_id).await;

            // 1. GENERATE UNIQUE FILENAMES
            let staging_filename = format!(
                "bucket_{}_staging_{}.drift",
                bucket_id,
                uuid::Uuid::new_v4()
            );
            let new_filename = format!("bucket_{}_{}.drift", bucket_id, uuid::Uuid::new_v4());

            // 2. ROTATE (Active -> Staging)
            let rotated = self
                .staging
                .rotate_bucket_for_promotion(bucket_id, &staging_filename, &new_filename)
                .await?;

            if !rotated {
                continue;
            }

            // 3. CHECK STATE & UPDATE REGISTRY (Atomic Swap to Promoting)
            // Even though we have a lock, we update registry so that IF we crash or unlock, state is valid.
            let remote_path_opt = if let Some(ver) = self.bucket_manager.get_version(bucket_id) {
                match &ver.class {
                    StorageClass::Tiered { remote_path, .. } => Some(remote_path.clone()),
                    StorageClass::Promoting { remote_path, .. } => remote_path.clone(),
                    _ => None,
                }
            } else {
                None
            };

            let promoting_class = StorageClass::Promoting {
                local_active: new_filename.clone(),
                local_frozen: staging_filename.clone(),
                remote_path: remote_path_opt.clone(),
            };

            // We do NOT need to send the old version to Reaper here because we hold the lock!
            // No one else can be reading the old version.
            self.bucket_manager
                .register_bucket(bucket_id, new_filename.clone(), promoting_class);

            // 4. UPLOAD (Slow IO - blocking search on this bucket)
            let (ids, vecs) = self.staging.read_file_content(&staging_filename).await?;
            if ids.is_empty() {
                let _ = self.staging.delete_file(&staging_filename).await;
                continue;
            }

            let old_run_id = {
                let m = self.manifest.get_state();
                m.get_buckets()
                    .iter()
                    .find(|b| b.id == bucket_id)
                    .map(|b| b.run_id.clone())
                    .filter(|s| !s.is_empty())
            };

            let dim = self.index.get_dim();
            let (new_run_id, final_count) = self
                .persistence
                .promote_to_s3(bucket_id, &ids, &vecs, old_run_id, dim)
                .await?;

            // 5. FINALIZE (Promoting -> Tiered)
            let new_remote_path = format!("bucket_{}_{}.drift", bucket_id, new_run_id);
            let tiered_class = StorageClass::Tiered {
                remote_path: new_remote_path.clone(),
                local_path: new_filename.clone(),
            };

            self.bucket_manager
                .register_bucket(bucket_id, new_remote_path, tiered_class);

            // 6. UPDATE MANIFEST
            self.manifest.apply_atomic(|m| {
                m.update_bucket_run_id(bucket_id, new_run_id.clone());
                m.update_bucket_stats(bucket_id, final_count, 0);
            })?;

            // 7. CLEANUP (Immediate & Safe)
            // Because we hold the Write Lock, no Searcher is accessing `staging_filename` or `old_remote_path`.
            // We can delete them immediately.

            // Delete Local Staging
            info!("JanitorV2: Cleaning up staging {}", staging_filename);
            let _ = self.staging.delete_file(&staging_filename).await;

            // Delete Old S3 File
            if let Some(old_path) = remote_path_opt {
                info!("JanitorV2: Cleaning up old S3 segment {}", old_path);
                let _ = self.persistence.delete_file(&old_path).await;
            }

            info!("JanitorV2: Promoted Bucket {} -> {}", bucket_id, new_run_id);
        } // 🔓 LOCK RELEASED HERE

        Ok(())
    }

    /// Executes the physical split operation.
    pub(crate) async fn perform_split(&self, bucket_id: u32) -> io::Result<()> {
        info!("Janitor: ✂️ Calculating split for Bucket {}", bucket_id);

        // 1. Calculate (The Brain)
        // This is read-only and safe.
        let proposal = match self.index.calculate_split(bucket_id).await? {
            Ok(p) => p,
            Err(status) => {
                info!("Janitor: Split aborted: {}", status.to_str());
                // TODO: Update Ignore Map here to prevent retry loops
                return Ok(());
            }
        };

        // 2. Write New Buckets (Staging)
        // We allocate new IDs for the children
        // Note: allocate_next_bucket_id is atomic on Index
        let id_left = self.index.allocate_next_bucket_id();
        let id_right = self.index.allocate_next_bucket_id();

        // Write Left
        let count_l = self.staging.append_batch(id_left, &proposal.left).await?;
        let file_l = self.staging.get_active_filename(id_left);

        // Write Right
        let count_r = self.staging.append_batch(id_right, &proposal.right).await?;
        let file_r = self.staging.get_active_filename(id_right);

        // 3. Register New Files (Local)
        self.bucket_manager
            .register_bucket(id_left, file_l, StorageClass::Local);
        self.bucket_manager
            .register_bucket(id_right, file_r, StorageClass::Local);

        // 4. Atomic Commit (Manifest + Router)
        // We perform all metadata updates in one go.
        self.manifest.apply_atomic(|m| {
            // Remove Old
            m.remove_bucket(bucket_id);

            // Add New
            m.add_bucket(id_left, String::new(), proposal.left.centroid.clone());
            m.update_bucket_stats(id_left, count_l, 0);

            m.add_bucket(id_right, String::new(), proposal.right.centroid.clone());
            m.update_bucket_stats(id_right, count_r, 0);
        })?;

        // 5. Update In-Memory Router (Critical for Search)
        // We need to expose a method on Index to update router, or do it here if we have access.
        // Ideally, Index listens to Manifest updates or we call a method.
        // For V2 MVP, we can call a helper on Index.
        self.index
            .apply_split_update(
                bucket_id,
                (id_left, proposal.left.centroid.unwrap()),
                (id_right, proposal.right.centroid.unwrap()),
            )
            .await;

        // 6. Handle Defectors (Loopback)
        if !proposal.loopback.is_empty() {
            info!(
                "Janitor: ↩️ Looping back {} defectors",
                proposal.loopback.len()
            );
            self.index.insert_batch(&proposal.loopback)?;
        }

        // 7. Cleanup Old
        // We delete the STAGING file for the old bucket immediately if it exists.
        // Remote files are handled by Reaper later.
        let old_staging = self.staging.get_active_filename(bucket_id);
        let _ = self.staging.delete_file(&old_staging).await;

        info!(
            "Janitor: Split Complete. {} -> {}, {}",
            bucket_id, id_left, id_right
        );
        Ok(())
    }
}
