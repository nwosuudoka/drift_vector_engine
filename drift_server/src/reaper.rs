use crate::local_staging::LocalStagingManager;
use crate::persistence_v2::PersistenceManager; // ⚡ Import this
use drift_storage::bucket_manager::{BucketVersion, StorageClass};
use std::collections::VecDeque;
use std::sync::Arc;
use tracing::{info, warn};

pub struct Reaper {
    queue: VecDeque<Arc<BucketVersion>>,
    staging: Arc<LocalStagingManager>,
    persistence: Arc<PersistenceManager>, // ⚡ Add field
}

impl Reaper {
    pub fn new(staging: Arc<LocalStagingManager>, persistence: Arc<PersistenceManager>) -> Self {
        Self {
            queue: VecDeque::new(),
            staging,
            persistence,
        }
    }

    pub fn schedule_deletion(&mut self, version: Arc<BucketVersion>) {
        self.queue.push_back(version);
    }

    pub async fn run_cycle(&mut self) {
        let count = self.queue.len();
        if count == 0 {
            return;
        }

        for _ in 0..count {
            if let Some(version) = self.queue.pop_front() {
                // ⚡ SAFETY CHECK: Only delete if NO active searchers hold this version
                if Arc::strong_count(&version) == 1 {
                    match &version.class {
                        StorageClass::Local => {
                            // Delete local active file
                            info!("Reaper: 💀 Reaping Local {}", version.path);
                            let _ = self.staging.delete_file(&version.path).await;
                        }
                        StorageClass::Promoting {
                            local_frozen,
                            remote_path,
                            ..
                        } => {
                            // 1. Delete the unique Staging File (Always safe now due to UUID)
                            info!("Reaper: 💀 Reaping Frozen {}", local_frozen);
                            let _ = self.staging.delete_file(local_frozen).await;

                            // 2. ⚡ Delete the OLD S3 File
                            if let Some(path) = remote_path {
                                info!("Reaper: 💀 Reaping Obsolete S3 Segment {}", path);
                                if let Err(e) = self.persistence.delete_file(path).await {
                                    warn!("Reaper: Failed to delete S3 file {}: {}", path, e);
                                }
                            }
                        }
                        _ => {}
                    }
                } else {
                    // Still in use, retry later
                    self.queue.push_back(version);
                }
            }
        }
    }
}
