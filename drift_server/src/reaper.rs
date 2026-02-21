use crate::cleanup::CleanupApi;
use crate::local_staging::LocalStagingManager;
use crate::persistence::PersistenceManager;
use drift_storage::bucket_manager::{BucketVersion, StorageClass};
use std::collections::VecDeque;
use std::sync::Arc;
use tracing::info;

pub struct Reaper {
    queue: VecDeque<Arc<BucketVersion>>,
    cleanup: CleanupApi,
}

impl Reaper {
    pub fn new(staging: Arc<LocalStagingManager>, persistence: Arc<PersistenceManager>) -> Self {
        Self {
            queue: VecDeque::new(),
            cleanup: CleanupApi::new(staging, persistence),
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
                            self.cleanup
                                .delete_local_best_effort(&version.path, "reaper-local")
                                .await;
                        }
                        StorageClass::Promoting {
                            local_frozen,
                            remote_path,
                            ..
                        } => {
                            // 1. Delete the unique Staging File (Always safe now due to UUID)
                            info!("Reaper: 💀 Reaping Frozen {}", local_frozen);
                            self.cleanup
                                .delete_local_best_effort(local_frozen, "reaper-promoting-local")
                                .await;

                            // 2. ⚡ Delete the OLD S3 File
                            if let Some(path) = remote_path {
                                info!("Reaper: 💀 Reaping Obsolete S3 Segment {}", path);
                                self.cleanup
                                    .delete_remote_best_effort(path, "reaper-promoting-remote")
                                    .await;
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
