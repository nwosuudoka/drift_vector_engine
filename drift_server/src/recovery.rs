use crate::manifest::ServerManifestManager;
use drift_core::router::Router;
use drift_core::wal::{WalEntry, WalReader};
use drift_storage::bucket_manager::{BucketManager, StorageClass};
use drift_storage::disk_manager::DiskManager;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::{info, warn};

pub struct ReplayData {
    pub inserts: Vec<(u64, Vec<f32>)>,
    pub deletes: Vec<u64>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RecoveryGuardMetricsSnapshot {
    pub mismatches_detected: u64,
    pub invalidations_performed: u64,
    pub fail_fast_aborts: u64,
}

#[derive(Default)]
struct RecoveryGuardMetrics {
    mismatches_detected: AtomicU64,
    invalidations_performed: AtomicU64,
    fail_fast_aborts: AtomicU64,
}

impl RecoveryGuardMetrics {
    fn snapshot(&self) -> RecoveryGuardMetricsSnapshot {
        RecoveryGuardMetricsSnapshot {
            mismatches_detected: self.mismatches_detected.load(Ordering::Relaxed),
            invalidations_performed: self.invalidations_performed.load(Ordering::Relaxed),
            fail_fast_aborts: self.fail_fast_aborts.load(Ordering::Relaxed),
        }
    }

    #[cfg(test)]
    fn reset_for_tests(&self) {
        self.mismatches_detected.store(0, Ordering::Relaxed);
        self.invalidations_performed.store(0, Ordering::Relaxed);
        self.fail_fast_aborts.store(0, Ordering::Relaxed);
    }
}

static RECOVERY_GUARD_METRICS: OnceLock<RecoveryGuardMetrics> = OnceLock::new();

fn recovery_guard_metrics() -> &'static RecoveryGuardMetrics {
    RECOVERY_GUARD_METRICS.get_or_init(RecoveryGuardMetrics::default)
}

pub struct RecoveryManager {
    root: PathBuf,
    manifest: Arc<ServerManifestManager>,
    fingerprint_guard: RecoveryFingerprintGuardConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FingerprintMismatchPolicy {
    InvalidateAndContinue,
    FailFast,
}

#[derive(Debug, Clone)]
pub struct RecoveryFingerprintGuardConfig {
    pub policy: FingerprintMismatchPolicy,
    pub max_mismatches: Option<usize>,
}

impl Default for RecoveryFingerprintGuardConfig {
    fn default() -> Self {
        Self {
            policy: FingerprintMismatchPolicy::InvalidateAndContinue,
            max_mismatches: None,
        }
    }
}

impl RecoveryFingerprintGuardConfig {
    fn from_env() -> Self {
        let policy = std::env::var("DRIFT_RECOVERY_FINGERPRINT_POLICY")
            .ok()
            .map(|raw| raw.trim().to_ascii_lowercase())
            .and_then(|raw| match raw.as_str() {
                "invalidate_and_continue" => Some(FingerprintMismatchPolicy::InvalidateAndContinue),
                "fail_fast" => Some(FingerprintMismatchPolicy::FailFast),
                other => {
                    warn!(
                        "Recovery: Unknown DRIFT_RECOVERY_FINGERPRINT_POLICY='{}'. \
                         Falling back to invalidate_and_continue.",
                        other
                    );
                    None
                }
            })
            .unwrap_or(FingerprintMismatchPolicy::InvalidateAndContinue);

        let max_mismatches = std::env::var("DRIFT_RECOVERY_FINGERPRINT_MAX_MISMATCHES")
            .ok()
            .and_then(|raw| match raw.trim().parse::<usize>() {
                Ok(v) => Some(v),
                Err(_) => {
                    warn!(
                        "Recovery: Invalid DRIFT_RECOVERY_FINGERPRINT_MAX_MISMATCHES='{}'. \
                         Ignoring this setting.",
                        raw
                    );
                    None
                }
            });

        Self {
            policy,
            max_mismatches,
        }
    }
}

impl RecoveryManager {
    pub fn new(root: &Path, manifest: Arc<ServerManifestManager>) -> Self {
        Self::new_with_fingerprint_guard(root, manifest, RecoveryFingerprintGuardConfig::from_env())
    }

    pub fn new_with_fingerprint_guard(
        root: &Path,
        manifest: Arc<ServerManifestManager>,
        fingerprint_guard: RecoveryFingerprintGuardConfig,
    ) -> Self {
        Self {
            root: root.to_path_buf(),
            manifest,
            fingerprint_guard,
        }
    }

    pub fn global_fingerprint_guard_metrics() -> RecoveryGuardMetricsSnapshot {
        recovery_guard_metrics().snapshot()
    }

    #[cfg(test)]
    pub fn reset_fingerprint_guard_metrics_for_tests() {
        recovery_guard_metrics().reset_for_tests();
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
        let metric = wrapper
            .metric()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

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
            Arc::new(RwLock::new(Router::empty(dim, metric.clone())))
        } else {
            let r = Router::new(&pb_centroids, &counts, dim, metric)
                .ok_or_else(|| io::Error::other("Failed to rebuild router from non-empty state"))?;
            info!(
                "Recovery: Router rebuilt with {} buckets.",
                pb_centroids.len()
            );
            Arc::new(RwLock::new(r))
        };

        // --- STEP B: REBUILD STORAGE ---
        let remote_op = bucket_manager.remote_operator();
        let mut mismatch_count: usize = 0;
        for b in wrapper.get_buckets() {
            let local_filename = format!("bucket_{}.driftu", b.id);
            // Check staging dir specifically
            let local_full_path = self.root.join("staging").join(&local_filename);

            // Local staging is authoritative during recovery; skip remote fingerprint guard.
            if local_full_path.exists() {
                bucket_manager.register_bucket(b.id, local_filename, StorageClass::Local);
                continue;
            }

            let remote_filename = if !b.object_path.is_empty() {
                Some(b.object_path.clone())
            } else if !b.run_id.is_empty() {
                Some(format!("bucket_{}_{}.driftu", b.id, b.run_id))
            } else {
                None
            };

            if let Some(remote_path) = &remote_filename
                && !b.object_fingerprint.is_empty()
                && let Some(cached_fingerprint) =
                    DiskManager::nvme_cached_fingerprint_for_object(&remote_op, remote_path)
                && cached_fingerprint != b.object_fingerprint
            {
                recovery_guard_metrics()
                    .mismatches_detected
                    .fetch_add(1, Ordering::Relaxed);
                mismatch_count = mismatch_count.saturating_add(1);
                match self.fingerprint_guard.policy {
                    FingerprintMismatchPolicy::InvalidateAndContinue => {
                        warn!(
                            "Recovery: NVMe cache fingerprint mismatch for bucket {} (path: {}). \
                             invalidating stale cache entry.",
                            b.id, remote_path
                        );
                        DiskManager::invalidate_nvme_cache_for_object(&remote_op, remote_path)
                            .await?;
                        recovery_guard_metrics()
                            .invalidations_performed
                            .fetch_add(1, Ordering::Relaxed);

                        if let Some(max) = self.fingerprint_guard.max_mismatches
                            && mismatch_count > max
                        {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                format!(
                                    "Recovery fingerprint mismatches exceeded configured limit: \
                                     observed={} max={}",
                                    mismatch_count, max
                                ),
                            ));
                        }
                    }
                    FingerprintMismatchPolicy::FailFast => {
                        recovery_guard_metrics()
                            .fail_fast_aborts
                            .fetch_add(1, Ordering::Relaxed);
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "Recovery fingerprint mismatch detected for bucket {} at {} \
                                 with fail_fast policy enabled",
                                b.id, remote_path
                            ),
                        ));
                    }
                }
            }

            if let Some(remote_path) = remote_filename {
                bucket_manager.register_bucket(b.id, remote_path, StorageClass::Remote);
            } else {
                warn!("Recovery: Bucket {} is registered but has no file!", b.id);
            }
        }

        if mismatch_count > 0 {
            info!(
                "Recovery: processed {} manifest/cache fingerprint mismatch(es) with policy {:?}.",
                mismatch_count, self.fingerprint_guard.policy
            );
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
