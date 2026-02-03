use crate::manifest::ServerManifestManager;
use drift_core::math::Metric;
use drift_core::router::Router;
use drift_core::wal_v2::{WalEntry, WalReader};
use drift_storage::bucket_manager::{BucketManager, StorageClass};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{info, warn};

pub struct ReplayData {
    pub inserts: Vec<(u64, Vec<f32>)>,
    pub deletes: Vec<u64>,
}

pub struct RecoveryManager {
    root: PathBuf,
    manifest: Arc<ServerManifestManager>,
}

impl RecoveryManager {
    pub fn new(root: &Path, manifest: Arc<ServerManifestManager>) -> Self {
        Self {
            root: root.to_path_buf(),
            manifest,
        }
    }

    /// Full System Recovery
    /// Rebuilds Router, Registers Buckets, and Scans WAL for replay.
    pub async fn recover(
        &self,
        bucket_manager: &BucketManager,
        dim: usize,
        wal_dir: &Path, // ⚡ Explicit WAL path required
    ) -> io::Result<(Arc<RwLock<Router>>, ReplayData)> {
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

        let router = if pb_centroids.is_empty() {
            info!("Recovery: No existing state found (Day 0). Bootstrapping empty router.");
            Arc::new(RwLock::new(Router::empty(dim, Metric::L2)))
        } else {
            let r = Router::new(&pb_centroids, &counts, dim, Metric::L2)
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
            // Check staging dir specifically
            let local_full_path = self.root.join("staging").join(&local_filename);

            if local_full_path.exists() {
                bucket_manager.register_bucket(b.id, local_filename, StorageClass::Local);
            } else if !b.run_id.is_empty() {
                let remote_filename = format!("bucket_{}_{}.drift", b.id, b.run_id);
                bucket_manager.register_bucket(b.id, remote_filename, StorageClass::Remote);
            } else {
                warn!("Recovery: Bucket {} is registered but has no file!", b.id);
            }
        }

        // --- STEP C: SCAN WAL ---
        let mut inserts = Vec::new();
        let mut deletes = Vec::new();

        if wal_dir.exists() {
            let mut entries = std::fs::read_dir(wal_dir)?
                .map(|res| res.map(|e| e.path()))
                .collect::<Result<Vec<_>, std::io::Error>>()?;

            entries.sort(); // Replay order is critical

            for path in entries {
                if path.extension().is_some_and(|e| e == "log")
                    && let Ok(reader) = WalReader::open(&path)
                {
                    for entry in reader.read_committed() {
                        match entry {
                            WalEntry::Insert { id, vector } => inserts.push((id, vector)),
                            WalEntry::Delete { id } => deletes.push(id),
                            _ => {}
                        }
                    }
                }
            }
        }

        info!(
            "Recovery: Scanned WAL at {:?}: {} inserts, {} deletes",
            wal_dir,
            inserts.len(),
            deletes.len()
        );

        Ok((router, ReplayData { inserts, deletes }))
    }
}
