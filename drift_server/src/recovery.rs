use crate::manifest::ServerManifestManager;
use drift_core::router::Router;
use drift_core::wal_v2::{WalEntry, WalReader};
use drift_storage::bucket_manager::{BucketManager, StorageClass};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::io;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{info, warn};

pub struct RecoveryManager {
    base_path: PathBuf,
    manifest: Arc<ServerManifestManager>,
    wal_path: PathBuf,
}

impl RecoveryManager {
    pub fn new(base_path: impl Into<PathBuf>, manifest: Arc<ServerManifestManager>) -> Self {
        let p = base_path.into();
        Self {
            wal_path: p.join("write_ahead_log.wal"),
            base_path: p,
            manifest,
        }
    }

    /// Full System Recovery
    /// Always returns a Router (empty if Day 0) and pending WAL entries.
    pub async fn recover(
        &self,
        bucket_manager: &BucketManager,
        dim: usize,
    ) -> io::Result<(Arc<RwLock<Router>>, Vec<(u64, Vec<f32>)>)> {
        info!("Recovery: Starting...");

        // 1. Get Snapshot of Manifest
        let wrapper = self.manifest.get_state();

        // --- STEP A: REBUILD ROUTER ---
        let bucket_stats: HashMap<u32, u64> = wrapper
            .get_buckets()
            .iter()
            .map(|b| (b.id, b.vector_count))
            .collect();

        let mut pb_centroids = Vec::new();
        let mut counts = Vec::new();

        for c in wrapper.get_centroids() {
            pb_centroids.push(drift_core::manifest::pb::Centroid {
                id: c.id,
                vector: c.vector.clone(),
            });
            let count = bucket_stats.get(&c.id).copied().unwrap_or(0);
            counts.push(count as u32);
        }

        //  Use Router::empty() for Day 0 logic
        let router = if pb_centroids.is_empty() {
            info!("Recovery: No existing state found (Day 0). Bootstrapping empty router.");
            Arc::new(RwLock::new(Router::empty(
                dim,
                drift_core::router::Metric::L2,
            )))
        } else {
            let r = Router::new(&pb_centroids, &counts, dim, drift_core::router::Metric::L2)
                .ok_or_else(|| io::Error::other("Failed to rebuild router from non-empty state"))?;
            info!(
                "Recovery: Router rebuilt with {} buckets.",
                pb_centroids.len()
            );
            Arc::new(RwLock::new(r))
        };

        // --- STEP B: REBUILD STORAGE ---
        for b in wrapper.get_buckets() {
            let local_filename = format!("bucket_{}.drift", b.id);
            let local_full_path = self.base_path.join("data").join(&local_filename);

            if local_full_path.exists() {
                // ⚡ Register as LOCAL
                bucket_manager.register_bucket(b.id, local_filename, StorageClass::Local);
            } else if !b.run_id.is_empty() {
                // ⚡ Register as REMOTE
                let remote_filename = format!("bucket_{}_{}.drift", b.id, b.run_id);
                bucket_manager.register_bucket(b.id, remote_filename, StorageClass::Remote);
            } else {
                warn!("Recovery: Bucket {} is registered but has no file!", b.id);
            }
        }

        // --- STEP C: REPLAY WAL ---
        let wal_vectors = if self.wal_path.exists() {
            let reader = WalReader::open(&self.wal_path)?;
            let entries = reader.read_committed();
            entries
                .into_iter()
                .filter_map(|e| match e {
                    WalEntry::Insert { id, vector } => Some((id, vector)),
                    _ => None,
                })
                .collect()
        } else {
            Vec::new()
        };

        Ok((router, wal_vectors))
    }
}
