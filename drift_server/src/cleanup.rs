use crate::local_staging::LocalStagingManager;
use crate::persistence::PersistenceManager;
use std::io;
use std::sync::Arc;
use tracing::warn;

#[derive(Clone)]
pub struct CleanupApi {
    staging: Arc<LocalStagingManager>,
    persistence: Arc<PersistenceManager>,
}

impl CleanupApi {
    pub fn new(staging: Arc<LocalStagingManager>, persistence: Arc<PersistenceManager>) -> Self {
        Self {
            staging,
            persistence,
        }
    }

    pub async fn delete_local_best_effort(&self, filename: &str, context: &str) {
        if filename.is_empty() {
            return;
        }
        if let Err(e) = self.staging.delete_file(filename).await {
            warn!(
                "Cleanup: failed to delete local file {} ({}): {}",
                filename, context, e
            );
        }
    }

    pub async fn delete_remote(&self, path: &str) -> io::Result<()> {
        self.persistence.delete_file(path).await
    }

    pub async fn delete_remote_best_effort(&self, path: &str, context: &str) {
        if path.is_empty() {
            return;
        }
        if let Err(e) = self.delete_remote(path).await {
            warn!(
                "Cleanup: failed to delete remote file {} ({}): {}",
                path, context, e
            );
        }
    }
}
