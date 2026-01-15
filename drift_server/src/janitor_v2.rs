use crate::local_staging::LocalStagingManager;
use crate::manifest::ServerManifestManager;
use crate::persistence_v2::PersistenceManager;
use crate::reaper::Reaper;
use drift_core::{index_v2::VectorIndex, lock_manager::BucketCoordinator};
use drift_storage::bucket_manager::{BucketManager, StorageClass};
use std::io;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tokio::time;
use tracing::{error, info};

pub struct JanitorConfig {
    pub index: Arc<VectorIndex>,
    pub manifest: Arc<ServerManifestManager>,
    pub staging: Arc<LocalStagingManager>,
    pub persistence: PersistenceManager,
    pub bucket_manager: Arc<BucketManager>,
    pub check_interval: Duration,
    pub promotion_threshold_bytes: u64,
    pub coordinator: Arc<BucketCoordinator>,
}

pub struct Janitor {
    index: Arc<VectorIndex>,
    manifest: Arc<ServerManifestManager>,
    staging: Arc<LocalStagingManager>,
    persistence: PersistenceManager,
    bucket_manager: Arc<BucketManager>,
    check_interval: Duration,
    promotion_threshold_bytes: u64,
    reaper: Mutex<Reaper>,
    coordinator: Arc<BucketCoordinator>,
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

            // 2. Run Reaper (Garbage Collection)
            let mut reaper = self.reaper.lock().await;
            reaper.run_cycle().await;
            drop(reaper); // Release lock before long operations

            // 3. Check for Promotions (Local -> S3)
            if let Err(e) = self.promote_segments().await {
                error!("Janitor: Promotion failed: {}", e);
            }
        }
    }

    async fn perform_flush(&self) -> io::Result<()> {
        // 1. Flush Frozen Logic
        // Returns (partitions, wal_ids)
        let (partitions, wal_ids) = match self.index.flush_frozen() {
            Some((p, w)) if !p.is_empty() => (p, w),
            _ => return Ok(()),
        };

        info!(
            "Janitor: Flushing {} buckets (WALs: {:?})",
            partitions.len(),
            wal_ids
        );

        let mut updates = Vec::new();

        // 2. Write to Local Staging
        for (bucket_id, group) in &partitions {
            let new_count = self.staging.append_batch(*bucket_id, group).await?;
            updates.push((*bucket_id, new_count, group.centroid.clone()));

            // ⚡ CRITICAL FIX: SMART REGISTRY UPDATE ⚡
            // Do NOT blindly overwrite with StorageClass::Local.
            // Only register if it's a NEW bucket.
            // Existing buckets (Local or Tiered) keep their state because the filename implies the active local file.

            let current_version = self.bucket_manager.get_version(*bucket_id);
            if current_version.is_none() {
                // New Bucket: Register as Local
                let filename = self.staging.get_active_filename(*bucket_id);
                self.bucket_manager
                    .register_bucket(*bucket_id, filename, StorageClass::Local);
            }
            // Else: It exists. We just appended to the file it already points to.
            // If it was Tiered, it stays Tiered. If Local, stays Local.
        }

        // 3. Update Manifest
        self.manifest.apply_atomic(|m| {
            for (id, count, centroid_opt) in updates {
                let exists = m.get_buckets().iter().any(|b| b.id == id);
                if !exists {
                    m.add_bucket(id, String::new(), centroid_opt);
                }
                m.update_bucket_stats(id, count, 0);
            }
        })?;

        // 4. Acknowledge Flush
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
            let _old_version = self.bucket_manager.register_bucket(
                bucket_id,
                new_filename.clone(),
                promoting_class,
            );

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
}
