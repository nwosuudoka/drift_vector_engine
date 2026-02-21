use crate::cleanup::CleanupApi;
use crate::local_staging::LocalStagingManager;
use crate::manifest::ServerManifestManager;
use crate::persistence::PersistenceManager;
use crate::reaper::Reaper;
use drift_core::partitioner::PartitionGroup;
use drift_core::{index::VectorIndex, lock_manager::BucketCoordinator};
use drift_storage::bucket_manager::{BucketManager, StorageClass};
use drift_traits::StorageEngine;
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::{collections::HashMap, io};
use tokio::sync::Mutex;
use tokio::time;
use tracing::{error, info, warn};

const KV_SYNC_INTERVAL_MS_ENV: &str = "DRIFT_KV_SYNC_INTERVAL_MS";

fn kv_sync_interval_from_env() -> Duration {
    let default = Duration::from_millis(5_000);
    match std::env::var(KV_SYNC_INTERVAL_MS_ENV) {
        Ok(raw) => match raw.trim().parse::<u64>() {
            Ok(ms) if ms > 0 => Duration::from_millis(ms),
            _ => {
                warn!(
                    "Janitor: invalid {}='{}'; using default {}ms",
                    KV_SYNC_INTERVAL_MS_ENV,
                    raw,
                    default.as_millis()
                );
                default
            }
        },
        Err(_) => default,
    }
}

pub struct JanitorVars {
    pub promotion_threshold_bytes: u64,
    pub max_bucket_capacity: usize,
    pub drift_threshold: f32,         // Default 0.15
    pub split_threshold: f32,         // Default 0.8
    pub temperature_cool_factor: f32, // default to 0.98
    pub check_interval: Duration,
    pub urgency_threshold: f32, // default to 1.5
}

impl Default for JanitorVars {
    fn default() -> Self {
        Self {
            promotion_threshold_bytes: 1024,
            max_bucket_capacity: 2000,
            drift_threshold: 0.15,
            split_threshold: 0.8,
            temperature_cool_factor: 0.98,
            check_interval: Duration::from_millis(100),
            urgency_threshold: 1.5,
        }
    }
}

pub struct JanitorConfig {
    pub index: Arc<VectorIndex>,
    pub manifest: Arc<ServerManifestManager>,
    pub staging: Arc<LocalStagingManager>,
    pub persistence: Arc<PersistenceManager>,
    pub bucket_manager: Arc<BucketManager>,
    pub coordinator: Arc<BucketCoordinator>,
    pub vars: JanitorVars,
}

pub struct Janitor {
    index: Arc<VectorIndex>,
    manifest: Arc<ServerManifestManager>,
    staging: Arc<LocalStagingManager>,
    persistence: Arc<PersistenceManager>,
    cleanup: CleanupApi,
    bucket_manager: Arc<BucketManager>,
    reaper: Mutex<Reaper>,
    coordinator: Arc<BucketCoordinator>,
    vars: JanitorVars,
}

impl Janitor {
    pub fn new(config: JanitorConfig) -> Self {
        let cleanup = CleanupApi::new(config.staging.clone(), config.persistence.clone());
        let reaper = Mutex::new(Reaper::new(
            config.staging.clone(),
            config.persistence.clone(),
        ));
        Self {
            index: config.index,
            manifest: config.manifest,
            staging: config.staging,
            persistence: config.persistence,
            cleanup,
            bucket_manager: config.bucket_manager,
            reaper,
            coordinator: config.coordinator,
            vars: config.vars,
        }
    }

    fn update_kv_mapping(&self, bucket_id: u32, ids: &[u64]) {
        let kv = self.index.get_kv();
        let bucket_bytes = bucket_id.to_le_bytes();
        for id in ids {
            if let Err(e) = kv.put(&id.to_le_bytes(), &bucket_bytes) {
                warn!(
                    "Janitor: failed kv.put for id={} bucket={}: {}",
                    id, bucket_id, e
                );
            }
        }
    }

    fn remove_kv_mapping(&self, ids: &[u64]) {
        let kv = self.index.get_kv();
        for id in ids {
            if let Err(e) = kv.remove(&id.to_le_bytes()) {
                warn!("Janitor: failed kv.remove for id={}: {}", id, e);
            }
        }
    }

    fn sync_kv_best_effort(&self, reason: &str) {
        if let Err(e) = self.index.get_kv().sync() {
            warn!("Janitor: kv.sync failed ({reason}): {e}");
        }
    }

    pub async fn run(&self) {
        let mut interval = time::interval(self.vars.check_interval);
        let kv_sync_interval = kv_sync_interval_from_env();
        let mut last_kv_sync = Instant::now();
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

            if last_kv_sync.elapsed() >= kv_sync_interval {
                self.sync_kv_best_effort("periodic");
                last_kv_sync = Instant::now();
            }
        }
    }

    async fn perform_flush(&self) -> io::Result<()> {
        // 1. Data Flush Logic
        let (partitions, wal_ids) = match self.index.flush_frozen() {
            Some((p, w)) => (p, w),
            None => return Ok(()),
        };

        let mut manifest_updates = Vec::new();
        // ⚡ CHANGE: Store updates for the Router (ID, Count, Centroid)
        let mut router_updates = Vec::new();

        for (bucket_id, group) in &partitions {
            // A. Append to Local Staging
            let new_count = self.staging.append_batch(*bucket_id, group).await?;

            // A1. Update KV (VectorID -> BucketID)
            self.update_kv_mapping(*bucket_id, &group.ids);

            // B. Ensure Registered
            if self.bucket_manager.get_version(*bucket_id).is_none() {
                let filename = self.staging.get_active_filename(*bucket_id);
                self.bucket_manager.register_bucket(
                    *bucket_id,
                    filename,
                    drift_storage::bucket_manager::StorageClass::Local,
                );
            }

            // C. Calculate Delta Sum (for Drift Tracking)
            let dim = self.index.get_dim();
            let mut delta_sum = vec![0.0; dim];
            if group.count > 0 {
                // Sum the new vectors
                for chunk in group.flat_vectors.chunks_exact(dim) {
                    for (i, val) in chunk.iter().enumerate() {
                        delta_sum[i] += val;
                    }
                }
            }

            // D. Update Persistent Stats
            self.bucket_manager
                .update_bucket_drift(*bucket_id, &delta_sum, group.count as u32)?;

            // E. ⚡ FETCH GLOBAL TRUTH
            // We get the TOTAL sum and TOTAL count from the manager
            let (total_sum, total_count) =
                self.bucket_manager
                    .get_bucket_drift_stats(*bucket_id)
                    .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "Failed to read stats"))?;

            // F. ⚡ RECALCULATE CENTROID
            let global_centroid: Vec<f32> = if total_count > 0 {
                total_sum.iter().map(|s| s / total_count as f32).collect()
            } else {
                vec![0.0; dim]
            };

            manifest_updates.push((*bucket_id, new_count, Some(global_centroid.clone())));
            router_updates.push((*bucket_id, total_count, global_centroid));
        }

        // 2. Tombstone Persistence
        let mut all_tombstones = Vec::new();
        {
            let l0_arc = self.index.get_tombstones();
            all_tombstones.extend(l0_arc.iter());
        }
        let l1_deletes = self.bucket_manager.collect_all_tombstones();
        all_tombstones.extend(l1_deletes);

        let mut tombstone_file_opt = None;
        if !all_tombstones.is_empty() {
            all_tombstones.sort_unstable();
            all_tombstones.dedup();
            let run_id = uuid::Uuid::new_v4().to_string();
            let ts_file = self
                .persistence
                .flush_tombstones(&all_tombstones, &run_id)
                .await?;
            tombstone_file_opt = Some(ts_file);
        }

        // 3. Update Manifest
        self.manifest.apply_atomic(|m| {
            for (id, count, centroid_opt) in manifest_updates {
                let exists = m.get_buckets().iter().any(|b| b.id == id);
                if !exists {
                    m.add_bucket(id, String::new(), centroid_opt);
                }
                m.update_bucket_stats(id, count, 0);
            }
            if let Some(tf) = tombstone_file_opt {
                m.inner.tombstone_files = vec![tf];
            }
        })?;

        // 4. ⚡ UPDATE ROUTER
        // We need a helper on Index to access the Router's `update_bucket` (not update_bucket_count)
        // If it doesn't exist, we'll need to add `index.update_router_bucket(...)`
        // Assuming we can access the router lock via the index:
        {
            let mut r = self.index.get_router().write();
            for (id, count, vec) in router_updates {
                // If bucket exists, update count AND centroid.
                // If it doesn't exist, add it.
                if r.get_centroid(id).is_some() {
                    r.update_bucket(id, count, vec);
                } else {
                    r.add_bucket(id, vec);
                    // Ensure count is set correctly after add (add_bucket sets count to 0)
                    r.update_bucket_count(id, count);
                }
            }
        }

        // 5. Acknowledge
        self.index.acknowledge_flush(&wal_ids)?;
        self.sync_kv_best_effort("flush");

        Ok(())
    }

    async fn _perform_flush(&self) -> io::Result<()> {
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

            // Keep router counts in sync with actual bucket size.
            self.index
                .update_router_count(*bucket_id, new_count as u32, group.centroid.clone());

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

    pub(crate) async fn promote_segments(&self) -> io::Result<()> {
        let threshold = self.vars.promotion_threshold_bytes;
        let candidates = self.staging.list_large_buckets(threshold)?;

        if candidates.is_empty() {
            return Ok(());
        }

        info!("Janitor: Promoting {} buckets to S3...", candidates.len());

        for bucket_id in candidates {
            // 🔒 Lock Bucket (Prevents split/merge/compact during promotion)
            let _lock_guard = self.coordinator.write(bucket_id).await;

            let tombstone_snapshot = self.bucket_manager.get_tombstones(bucket_id);

            // 1. Rotate Local File
            let staging_filename = format!(
                "bucket_{}_staging_{}.drift",
                bucket_id,
                uuid::Uuid::new_v4()
            );
            let new_filename = format!("bucket_{}_{}.drift", bucket_id, uuid::Uuid::new_v4());

            let rotated = self
                .staging
                .rotate_bucket_for_promotion(bucket_id, &staging_filename, &new_filename)
                .await?;
            if !rotated {
                continue;
            }

            // 2. Update Registry to "Promoting" state
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

            let current_count = self
                .bucket_manager
                .get_bucket_stats(bucket_id)
                .map(|s| s.total_count)
                .unwrap_or(0);

            self.bucket_manager.register_bucket_with_count(
                bucket_id,
                new_filename.clone(),
                promoting_class,
                current_count,
            );

            // --- ⚡ EXPLICIT MERGE & FILTER LOGIC ---

            // A. Read Local Staging
            let (mut merged_ids, mut merged_vecs) =
                self.staging.read_file_content(&staging_filename).await?;
            if merged_ids.is_empty() {
                self.cleanup
                    .delete_local_best_effort(&staging_filename, "promotion-empty-staging")
                    .await;
                continue;
            }

            // B. Read Remote (if exists)
            if let Some(path) = &remote_path_opt {
                // Extract RunID from path "bucket_{id}_{uuid}.drift"
                if let Some(run_id) = path
                    .strip_prefix(&format!("bucket_{}_", bucket_id))
                    .and_then(|s| s.strip_suffix(".drift"))
                {
                    let (r_ids, r_vecs) = self
                        .persistence
                        .read_remote_bucket(bucket_id, run_id)
                        .await?;
                    merged_ids.extend(r_ids);
                    merged_vecs.extend(r_vecs);
                }
            }

            // C. Snapshot Tombstones (Atomic Arc clone)

            // D. Filter (Purge)
            let mut final_ids = Vec::with_capacity(merged_ids.len());
            let mut final_vecs = Vec::with_capacity(merged_vecs.len());

            for (id, vec) in merged_ids.into_iter().zip(merged_vecs.into_iter()) {
                if !tombstone_snapshot.contains(&id) {
                    final_ids.push(id);
                    final_vecs.push(vec);
                }
            }

            let final_count = final_ids.len() as u32;

            // E. Write to S3
            let dim = self.index.get_dim();
            let (new_run_id, _) = self
                .persistence
                .write_remote_bucket(bucket_id, &final_ids, &final_vecs, dim)
                .await?;

            // --- END EXPLICIT LOGIC ---

            // 3. Finalize Registry (Tiered)
            let new_remote_path = format!("bucket_{}_{}.drift", bucket_id, new_run_id);
            let new_remote_fingerprint = match self
                .persistence
                .object_fingerprint_for_path(&new_remote_path)
                .await
            {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!(
                        "Janitor: failed to read remote fingerprint for {}: {}",
                        new_remote_path,
                        e
                    );
                    String::new()
                }
            };
            let tiered_class = StorageClass::Tiered {
                remote_path: new_remote_path.clone(),
                local_path: new_filename.clone(),
            };
            self.bucket_manager.register_bucket_with_count(
                bucket_id,
                new_remote_path.clone(),
                tiered_class,
                final_count,
            );

            // Keep router counts in sync after promotion.
            self.index.update_router_count(bucket_id, final_count, None);

            // 4. ⚡ RECONCILE: Prune the specific deletions we just handled
            self.bucket_manager
                .prune_tombstones(bucket_id, &tombstone_snapshot);

            // 5. Update Manifest
            // Get fresh stats for atomic update (retains any NEW tombstones that arrived during upload)
            if let Some(stats) = self.bucket_manager.get_bucket_stats(bucket_id) {
                self.manifest.apply_atomic(|m| {
                    m.update_bucket_remote_meta(
                        bucket_id,
                        new_run_id.clone(),
                        new_remote_path.clone(),
                        new_remote_fingerprint.clone(),
                    );
                    m.update_bucket_stats(bucket_id, final_count as u64, stats.tombstone_count);
                })?;
            }

            // 6. Cleanup
            self.cleanup
                .delete_local_best_effort(&staging_filename, "promotion-rotated-staging")
                .await;
            if let Some(old_path) = remote_path_opt {
                self.cleanup
                    .delete_remote_best_effort(&old_path, "promotion-old-remote")
                    .await;
            }
        }
        Ok(())
    }

    /// Executes the physical split operation.
    pub(crate) async fn perform_split(&self, bucket_id: u32) -> io::Result<()> {
        info!("Janitor: ✂️ Calculating split for Bucket {}", bucket_id);

        // 1. Snapshot Parent State
        let parent_stats = self
            .bucket_manager
            .get_bucket_stats(bucket_id)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "Parent bucket missing"))?;

        // a. Calculate
        // b. Calculate (The Brain)
        // This is read-only and safe.
        let proposal = match self.index.calculate_split(bucket_id).await? {
            Ok(p) => p,
            Err(status) => {
                info!("Janitor: Split aborted: {}", status.to_str());
                // TODO: Update Ignore Map here to prevent retry loops
                return Ok(());
            }
        };

        // c. ️ SAFETY CHECK ️
        let child_sum = proposal.left.count + proposal.right.count + proposal.loopback.len();
        if (child_sum as u32) < parent_stats.total_count {
            tracing::error!(
                "Janitor: 🚨 CRITICAL SPLIT FAILURE! Data loss detected. Parent: {}, Children: {}. Aborting.",
                parent_stats.total_count,
                child_sum
            );
            return Ok(());
        }

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

        // Update KV mapping for new buckets
        self.update_kv_mapping(id_left, &proposal.left.ids);
        self.update_kv_mapping(id_right, &proposal.right.ids);

        // Remove KV mapping for loopback (now L0-only)
        if !proposal.loopback.is_empty() {
            let loopback_ids: Vec<u64> = proposal.loopback.iter().map(|(id, _)| *id).collect();
            self.remove_kv_mapping(&loopback_ids);
        }

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
        // Call a helper on Index for the split update.
        self.index
            .apply_split_update(
                bucket_id,
                (id_left, proposal.left.centroid.unwrap(), count_l as u32),
                (id_right, proposal.right.centroid.unwrap(), count_r as u32),
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
        self.cleanup
            .delete_local_best_effort(&old_staging, "split-old-staging")
            .await;
        self.sync_kv_best_effort("split");

        info!(
            "Janitor: Split Complete. {} -> {}, {}",
            bucket_id, id_left, id_right
        );
        Ok(())
    }

    async fn perform_merge(&self, zombie_id: u32) -> io::Result<()> {
        info!("Janitor: 🚑 Merging Zombie Bucket {}", zombie_id);

        let proposal = match self.index.calculate_merge(zombie_id).await? {
            Ok(p) => p,
            Err(status) => {
                info!("Janitor: Merge aborted: {}", status.to_str());
                return Ok(());
            }
        };

        // Handle Empty/No-Neighbor case (Delete Logic)
        if proposal.moves.is_empty() {
            info!(
                "Janitor: Zombie Bucket {} is empty or isolated. Deleting.",
                zombie_id
            );

            // 1. Remove from Metadata
            self.manifest.apply_atomic(|m| m.remove_bucket(zombie_id))?;
            self.index.apply_merge_update(zombie_id, &[]).await;

            // 2. ⚡ NEW: Delete Physical File
            let zombie_file = self.staging.get_active_filename(zombie_id);
            self.cleanup
                .delete_local_best_effort(&zombie_file, "merge-empty-zombie")
                .await;

            return Ok(());
        }

        if proposal.moves.is_empty() {
            self.manifest.apply_atomic(|m| m.remove_bucket(zombie_id))?;
            self.index.apply_merge_update(zombie_id, &[]).await;
            return Ok(());
        }

        let mut manifest_updates = Vec::new();
        let mut files_to_delete = Vec::new();

        for (target_id, group) in &proposal.moves {
            // A. New File Name
            let new_filename = format!("bucket_{}_{}.drift", target_id, uuid::Uuid::new_v4());

            // B. Read Old Data (to merge)
            let (mut ids, mut vecs) = self.staging.read_full_bucket(*target_id).await?;
            let old_filename = self.staging.get_active_filename(*target_id);

            // C. Merge Vectors
            ids.extend(&group.ids);
            let dim = self.index.get_dim();
            for chunk in group.flat_vectors.chunks_exact(dim) {
                vecs.push(chunk.to_vec());
            }

            // D. Recalculate Stats (Sum & Centroid)
            let count = ids.len();
            let mut new_sum = vec![0.0; dim];

            // Flatten for writing & summing
            let mut flat_vecs = Vec::with_capacity(count * dim);
            for v in &vecs {
                flat_vecs.extend_from_slice(v);
                for i in 0..dim {
                    new_sum[i] += v[i];
                }
            }

            // ⚡ NEW: Calculate Centroid
            let mut new_centroid = vec![0.0; dim];
            if count > 0 {
                for i in 0..dim {
                    new_centroid[i] = new_sum[i] / count as f32;
                }
            }

            // E. Write File
            let mut merged_group = PartitionGroup::new(dim, None);
            merged_group.ids = ids;
            merged_group.flat_vectors = flat_vecs;
            merged_group.count = count;

            self.staging
                .write_new_file(&new_filename, &merged_group)
                .await?;

            // Update KV mapping for all IDs now owned by target_id
            self.update_kv_mapping(*target_id, &merged_group.ids);

            // Track update: (ID, Count, Sum, Centroid, Filename)
            manifest_updates.push((
                *target_id,
                count as u64,
                new_sum,
                new_centroid,
                new_filename,
            ));
            files_to_delete.push(old_filename);
        }

        // 3. Atomic Commit (Manifest)
        self.manifest.apply_atomic(|m| {
            m.remove_bucket(zombie_id);
            for (id, count, _, centroid, _) in &manifest_updates {
                // Use add_bucket to update Centroid + Count in Manifest
                // add_bucket handles upsert correctly
                m.add_bucket(*id, String::new(), Some(centroid.clone()));
                m.update_bucket_stats(*id, *count, 0);
            }
        })?;

        // 4. Update Runtime (Router)
        // Convert to format required by apply_merge_update: (id, count, sum, centroid)
        let router_updates: Vec<_> = manifest_updates
            .iter()
            .map(|(id, c, s, cent, _)| (*id, *c, s.clone(), cent.clone()))
            .collect();

        self.index
            .apply_merge_update(zombie_id, &router_updates)
            .await;

        // 5. Update Storage (BucketManager)
        for (id, count, sum, _, filename) in &manifest_updates {
            // Register overwrites the entry in BucketManager with a fresh state (empty sum)
            self.bucket_manager
                .register_bucket(*id, filename.clone(), StorageClass::Local);
            self.staging.set_active_filename(*id, filename.clone());

            // Re-inject the correct sum so Drift Calculation works immediately
            self.bucket_manager
                .update_bucket_drift(*id, sum, *count as u32)?;
        }

        // 6. Cleanup
        let zombie_file = self.staging.get_active_filename(zombie_id);
        self.cleanup
            .delete_local_best_effort(&zombie_file, "merge-zombie-old")
            .await;

        for f in files_to_delete {
            if !manifest_updates.iter().any(|(_, _, _, _, new)| *new == f) {
                self.cleanup
                    .delete_local_best_effort(&f, "merge-neighbor-old")
                    .await;
            }
        }
        self.sync_kv_best_effort("merge");

        info!("Janitor: Merge Complete. {} scattered.", zombie_id);
        Ok(())
    }

    pub(crate) async fn check_maintainance(&self) {
        // 1. Global Cooling (Decay temperature for all buckets)
        // Ask the storage layer to decay active temperatures.
        // const TEMPERATURE_COOL_FACTOR: f32 = 0.98;
        self.bucket_manager
            .tick_cooling(self.vars.temperature_cool_factor);

        // 2. Snapshot State
        let buckets = self.manifest.get_state().get_buckets().clone();
        let max_cap = self.vars.max_bucket_capacity as f32;

        // Snapshot Router for Centroids (needed for Drift calc)
        let (router_centroids, router_ids) = self.index.get_router().read().get_snapshot();
        let dim = self.index.get_dim();

        // Helper to find centroid for a bucket ID
        let centroid_map: HashMap<u32, Vec<f32>> = router_ids
            .iter()
            .zip(router_centroids.chunks(dim))
            .map(|(id, vec)| (*id, vec.to_vec()))
            .collect();

        for b in buckets {
            // 3. Fetch live stats from BucketManager
            // returns BucketStats { tombstone_count, total_count, temperature, ... }
            let stats = match self.bucket_manager.get_bucket_stats(b.id) {
                Some(s) => s,
                None => continue, // Bucket might be deleted or not yet registered
            };

            // Fetch vector sum for drift calculation
            let (current_sum, _drift_count) = self
                .bucket_manager
                .get_bucket_drift_stats(b.id)
                .unwrap_or((vec![], 0));

            let current_count = stats.total_count;

            // 4. Calculate Metrics (Manually, since V1 BucketHeader logic is gone)

            // --- A. Calculate Urgency ---
            // Formula: (Emptiness / (Temp + epsilon)) + (Beta * ZombieRatio)
            let total = current_count as f32;
            let dead = stats.tombstone_count as f32;
            let temp = stats.temperature; // Already decayed by tick_cooling above

            let live = (total - dead).max(0.0);

            let emptiness = if live < max_cap {
                (max_cap - live) / max_cap
            } else {
                0.0
            };

            let zombie_ratio = if total > 0.0 { dead / total } else { 0.0 };

            const EPSILON: f32 = 0.001;
            const BETA: f32 = 3.0;

            let urgency = (emptiness / (temp + EPSILON)) + (BETA * zombie_ratio);

            // --- B. Calculate Drift ---
            // Formula: Distance(Mean, Centroid)
            let drift_score = if current_count > 0 && !current_sum.is_empty() {
                if let Some(target_centroid) = centroid_map.get(&b.id) {
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
                }
            } else {
                0.0
            };

            let cap_ratio = current_count as f32 / max_cap;

            // 5. Decision Matrix

            // A. SPLIT: Too Full OR High Drift
            if cap_ratio > 1.0
                || (cap_ratio > self.vars.split_threshold
                    && drift_score > self.vars.drift_threshold)
            {
                info!(
                    "Janitor: ✂️ Triggering Split for Bucket {} (Cap: {:.2}, Drift: {:.4})",
                    b.id, cap_ratio, drift_score
                );

                if let Err(e) = self.perform_split(b.id).await {
                    error!("Janitor: Split failed for {}: {}", b.id, e);
                }
                break; // One op per tick
            }
            // B. MERGE: High Urgency
            // Using the urgency score we calculated manually above
            else if urgency > self.vars.urgency_threshold {
                info!(
                    "Janitor: 🚑 Triggering Merge for Zombie Bucket {} (Urgency: {:.2}, Count: {})",
                    b.id, urgency, current_count
                );

                if let Err(e) = self.perform_merge(b.id).await {
                    error!("Janitor: Merge failed for {}: {}", b.id, e);
                }
                break; // One op per tick
            }
        }
    }
}
