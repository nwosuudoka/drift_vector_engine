use parking_lot::RwLock as SyncRwLock;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock};

/// Central authority for Bucket locking.
/// Ensures Janitor (Write) and Searchers (Read) never collide on the same bucket.
#[derive(Clone, Default, Debug)]
pub struct BucketCoordinator {
    // Map BucketID -> Async RwLock
    locks: Arc<SyncRwLock<HashMap<u32, Arc<RwLock<()>>>>>,
}

impl BucketCoordinator {
    pub fn new() -> Self {
        Self::default()
    }

    fn get_lock(&self, bucket_id: u32) -> Arc<RwLock<()>> {
        // Fast path: Check if lock exists (Read lock on map)
        {
            let map = self.locks.read();
            if let Some(lock) = map.get(&bucket_id) {
                return lock.clone();
            }
        }
        // Slow path: Create lock (Write lock on map)
        let mut map = self.locks.write();
        map.entry(bucket_id)
            .or_insert_with(|| Arc::new(RwLock::new(())))
            .clone()
    }

    /// Acquisitions for Searchers (Shared access).
    /// Will wait if a Write Lock (Janitor) is active.
    pub async fn read(&self, bucket_id: u32) -> OwnedRwLockReadGuard<()> {
        let lock = self.get_lock(bucket_id);
        lock.read_owned().await
    }

    /// Acquisitions for Janitor/Reaper (Exclusive access).
    /// Will wait for all Searchers to finish.
    pub async fn write(&self, bucket_id: u32) -> OwnedRwLockWriteGuard<()> {
        let lock = self.get_lock(bucket_id);
        lock.write_owned().await
    }
}
