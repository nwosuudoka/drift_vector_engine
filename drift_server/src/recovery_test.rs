#[cfg(test)]
mod tests {
    use crate::manifest::ServerManifestManager;
    use crate::recovery::{
        FingerprintMismatchPolicy, RecoveryFingerprintGuardConfig, RecoveryManager,
    };
    use drift_core::lock_manager::BucketCoordinator;
    use drift_core::math::Metric;
    use drift_core::wal::WalWriter;
    use drift_storage::bucket_manager::{BucketManager, StorageClass};
    use drift_storage::disk_manager::DiskManager;
    use opendal::Operator;
    use opendal::services::{Fs, Memory};
    use std::sync::{Arc, OnceLock};
    use tempfile::tempdir;

    static ENV_LOCK: OnceLock<std::sync::Mutex<()>> = OnceLock::new();

    struct EnvVarGuard {
        key: &'static str,
        prev: Option<String>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let prev = std::env::var(key).ok();
            // SAFETY: Guarded by ENV_LOCK in tests mutating process env.
            unsafe { std::env::set_var(key, value) };
            Self { key, prev }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            if let Some(prev) = &self.prev {
                // SAFETY: Guarded by ENV_LOCK in tests mutating process env.
                unsafe { std::env::set_var(self.key, prev) };
            } else {
                // SAFETY: Guarded by ENV_LOCK in tests mutating process env.
                unsafe { std::env::remove_var(self.key) };
            }
        }
    }

    fn create_noop_manager() -> BucketManager {
        let builder = Fs::default().root("/tmp");
        let op = Operator::new(builder).unwrap().finish();
        BucketManager::new(
            op.clone(),
            op,
            1,
            Arc::new(BucketCoordinator::new()),
            Metric::L2,
        )
    }

    fn create_memory_operator() -> Operator {
        Operator::new(Memory::default()).unwrap().finish()
    }

    fn create_memory_manager(op: Operator) -> BucketManager {
        BucketManager::new(
            op.clone(),
            op,
            1,
            Arc::new(BucketCoordinator::new()),
            Metric::L2,
        )
    }

    fn create_broken_cache_entry(
        root: &std::path::Path,
        dir_name: &str,
        object_key: &str,
        with_object_file: bool,
        with_meta_file: bool,
    ) -> std::path::PathBuf {
        let dir = root.join(dir_name);
        std::fs::create_dir_all(&dir).unwrap();
        if with_object_file {
            std::fs::write(dir.join("object.cache"), vec![1u8; 16]).unwrap();
        }
        if with_meta_file {
            let body = format!("{object_key}\nmock_fingerprint\n");
            std::fs::write(dir.join("cache.meta"), body.as_bytes()).unwrap();
        }
        dir
    }

    #[tokio::test]
    async fn test_recover_empty_state() {
        let dir = tempdir().unwrap();
        let manifest = Arc::new(ServerManifestManager::new(dir.path(), 128).unwrap());
        let mgr = RecoveryManager::new(dir.path(), manifest);
        let bucket_mgr = create_noop_manager();

        // No WAL dir exists yet
        let wal_dir = dir.path().join("wal");

        let (router, replay) = mgr.recover(&bucket_mgr, 128, &wal_dir).await.unwrap();

        assert!(router.read().get_centroid(0).is_none());
        assert!(replay.inserts.is_empty());
    }

    #[tokio::test]
    async fn test_recover_registers_local_files() {
        let dir = tempdir().unwrap();
        let staging = dir.path().join("staging");
        std::fs::create_dir_all(&staging).unwrap();

        // Mock a local bucket file
        std::fs::File::create(staging.join("bucket_1.drift")).unwrap();

        let manifest = Arc::new(ServerManifestManager::new(dir.path(), 128).unwrap());

        // Add bucket 1 to manifest
        manifest
            .apply_atomic(|m| {
                m.add_bucket(1, "run_remote".into(), Some(vec![0.0; 128]));
            })
            .unwrap();

        let mgr = RecoveryManager::new(dir.path(), manifest);
        let bucket_mgr = create_noop_manager();
        let wal_dir = dir.path().join("wal");

        mgr.recover(&bucket_mgr, 128, &wal_dir).await.unwrap();

        // Should find local file and prefer it over remote run_id
        let (path, class) = bucket_mgr.get_location(1).unwrap();
        assert!(path.contains("bucket_1.drift"));
        assert_eq!(class, StorageClass::Local);
    }

    #[tokio::test]
    async fn test_recover_prefers_manifest_object_path_for_remote() {
        let dir = tempdir().unwrap();
        let manifest = Arc::new(ServerManifestManager::new(dir.path(), 8).unwrap());
        manifest
            .apply_atomic(|m| {
                m.add_bucket(7, "run_remote".into(), Some(vec![0.0; 8]));
                m.update_bucket_remote_meta(
                    7,
                    "run_remote".into(),
                    "custom/provider/object-7.drift".into(),
                    "len=64|etag=test".into(),
                );
            })
            .unwrap();

        let mgr = RecoveryManager::new(dir.path(), manifest);
        let bucket_mgr = create_noop_manager();
        let wal_dir = dir.path().join("wal");

        mgr.recover(&bucket_mgr, 8, &wal_dir).await.unwrap();

        let (path, class) = bucket_mgr.get_location(7).unwrap();
        assert_eq!(path, "custom/provider/object-7.drift");
        assert_eq!(class, StorageClass::Remote);
    }

    #[tokio::test]
    async fn test_recover_invalidates_stale_nvme_cache_by_manifest_fingerprint() {
        let _env_guard = ENV_LOCK
            .get_or_init(|| std::sync::Mutex::new(()))
            .lock()
            .unwrap();
        RecoveryManager::reset_fingerprint_guard_metrics_for_tests();
        let before = RecoveryManager::global_fingerprint_guard_metrics();

        let dir = tempdir().unwrap();
        let cache_root = tempdir().unwrap();
        let _cache_dir =
            EnvVarGuard::set("DRIFT_NVME_CACHE_DIR", cache_root.path().to_str().unwrap());

        let path = "custom/provider/object-9.drift";
        let op = create_memory_operator();
        op.write(path, vec![42u8; 64]).await.unwrap();

        let disk = DiskManager::new(op.clone(), path.to_string());
        let _ = disk.read_at(0, 8).await.unwrap();

        let cached_fp = DiskManager::nvme_cached_fingerprint_for_object(&op, path)
            .expect("cache fingerprint should exist after initial read");
        assert!(!cached_fp.is_empty());

        let manifest = Arc::new(ServerManifestManager::new(dir.path(), 8).unwrap());
        manifest
            .apply_atomic(|m| {
                m.add_bucket(9, "run_remote".into(), Some(vec![0.0; 8]));
                m.update_bucket_remote_meta(
                    9,
                    "run_remote".into(),
                    path.into(),
                    format!("{cached_fp}-stale"),
                );
            })
            .unwrap();

        let mgr = RecoveryManager::new_with_fingerprint_guard(
            dir.path(),
            manifest,
            RecoveryFingerprintGuardConfig {
                policy: FingerprintMismatchPolicy::InvalidateAndContinue,
                max_mismatches: None,
            },
        );
        let bucket_mgr = create_memory_manager(op.clone());
        let wal_dir = dir.path().join("wal");
        mgr.recover(&bucket_mgr, 8, &wal_dir).await.unwrap();

        let cache_after = DiskManager::nvme_cached_fingerprint_for_object(&op, path);
        assert!(
            cache_after.is_none(),
            "stale cache entry should be invalidated on recovery"
        );
        let after = RecoveryManager::global_fingerprint_guard_metrics();
        assert!(
            after
                .mismatches_detected
                .saturating_sub(before.mismatches_detected)
                >= 1
        );
        assert!(
            after
                .invalidations_performed
                .saturating_sub(before.invalidations_performed)
                >= 1
        );
    }

    #[tokio::test]
    async fn test_recover_keeps_nvme_cache_when_manifest_fingerprint_matches() {
        let _env_guard = ENV_LOCK
            .get_or_init(|| std::sync::Mutex::new(()))
            .lock()
            .unwrap();

        let dir = tempdir().unwrap();
        let cache_root = tempdir().unwrap();
        let _cache_dir =
            EnvVarGuard::set("DRIFT_NVME_CACHE_DIR", cache_root.path().to_str().unwrap());

        let path = "custom/provider/object-10.drift";
        let op = create_memory_operator();
        op.write(path, vec![7u8; 64]).await.unwrap();

        let disk = DiskManager::new(op.clone(), path.to_string());
        let _ = disk.read_at(0, 8).await.unwrap();

        let cached_fp = DiskManager::nvme_cached_fingerprint_for_object(&op, path)
            .expect("cache fingerprint should exist after initial read");
        assert!(!cached_fp.is_empty());

        let manifest = Arc::new(ServerManifestManager::new(dir.path(), 8).unwrap());
        manifest
            .apply_atomic(|m| {
                m.add_bucket(10, "run_remote".into(), Some(vec![0.0; 8]));
                m.update_bucket_remote_meta(10, "run_remote".into(), path.into(), cached_fp);
            })
            .unwrap();

        let mgr = RecoveryManager::new_with_fingerprint_guard(
            dir.path(),
            manifest,
            RecoveryFingerprintGuardConfig {
                policy: FingerprintMismatchPolicy::InvalidateAndContinue,
                max_mismatches: None,
            },
        );
        let bucket_mgr = create_memory_manager(op.clone());
        let wal_dir = dir.path().join("wal");
        mgr.recover(&bucket_mgr, 8, &wal_dir).await.unwrap();

        let cache_after = DiskManager::nvme_cached_fingerprint_for_object(&op, path);
        assert!(
            cache_after.is_some(),
            "matching fingerprint should preserve cache entry"
        );
    }

    #[tokio::test]
    async fn test_recover_handles_missing_cache_meta_file_gracefully() {
        let _env_guard = ENV_LOCK
            .get_or_init(|| std::sync::Mutex::new(()))
            .lock()
            .unwrap();
        RecoveryManager::reset_fingerprint_guard_metrics_for_tests();
        let before = RecoveryManager::global_fingerprint_guard_metrics();

        let dir = tempdir().unwrap();
        let cache_root = tempdir().unwrap();
        let _cache_dir =
            EnvVarGuard::set("DRIFT_NVME_CACHE_DIR", cache_root.path().to_str().unwrap());

        let path = "custom/provider/object-20.drift";
        let op = create_memory_operator();
        op.write(path, vec![1u8; 32]).await.unwrap();

        let broken_dir = create_broken_cache_entry(
            cache_root.path(),
            "broken_missing_meta",
            "memory://::unused",
            true,
            false,
        );

        let manifest = Arc::new(ServerManifestManager::new(dir.path(), 8).unwrap());
        manifest
            .apply_atomic(|m| {
                m.add_bucket(20, "run_remote".into(), Some(vec![0.0; 8]));
                m.update_bucket_remote_meta(
                    20,
                    "run_remote".into(),
                    path.into(),
                    "len=32|etag=expected".into(),
                );
            })
            .unwrap();

        let mgr = RecoveryManager::new_with_fingerprint_guard(
            dir.path(),
            manifest,
            RecoveryFingerprintGuardConfig {
                policy: FingerprintMismatchPolicy::InvalidateAndContinue,
                max_mismatches: Some(0),
            },
        );
        let bucket_mgr = create_memory_manager(op.clone());
        let wal_dir = dir.path().join("wal");
        mgr.recover(&bucket_mgr, 8, &wal_dir).await.unwrap();

        assert!(
            !broken_dir.exists(),
            "recovery-time cache load should purge entries missing cache.meta"
        );

        let after = RecoveryManager::global_fingerprint_guard_metrics();
        assert_eq!(
            after
                .mismatches_detected
                .saturating_sub(before.mismatches_detected),
            0
        );
        assert_eq!(
            after
                .invalidations_performed
                .saturating_sub(before.invalidations_performed),
            0
        );
    }

    #[tokio::test]
    async fn test_recover_handles_missing_cached_object_file_gracefully() {
        let _env_guard = ENV_LOCK
            .get_or_init(|| std::sync::Mutex::new(()))
            .lock()
            .unwrap();
        RecoveryManager::reset_fingerprint_guard_metrics_for_tests();
        let before = RecoveryManager::global_fingerprint_guard_metrics();

        let dir = tempdir().unwrap();
        let cache_root = tempdir().unwrap();
        let _cache_dir =
            EnvVarGuard::set("DRIFT_NVME_CACHE_DIR", cache_root.path().to_str().unwrap());

        let path = "custom/provider/object-21.drift";
        let op = create_memory_operator();
        op.write(path, vec![1u8; 32]).await.unwrap();

        let broken_dir = create_broken_cache_entry(
            cache_root.path(),
            "broken_missing_object",
            "memory://::unused",
            false,
            true,
        );

        let manifest = Arc::new(ServerManifestManager::new(dir.path(), 8).unwrap());
        manifest
            .apply_atomic(|m| {
                m.add_bucket(21, "run_remote".into(), Some(vec![0.0; 8]));
                m.update_bucket_remote_meta(
                    21,
                    "run_remote".into(),
                    path.into(),
                    "len=32|etag=expected".into(),
                );
            })
            .unwrap();

        let mgr = RecoveryManager::new_with_fingerprint_guard(
            dir.path(),
            manifest,
            RecoveryFingerprintGuardConfig {
                policy: FingerprintMismatchPolicy::InvalidateAndContinue,
                max_mismatches: Some(0),
            },
        );
        let bucket_mgr = create_memory_manager(op.clone());
        let wal_dir = dir.path().join("wal");
        mgr.recover(&bucket_mgr, 8, &wal_dir).await.unwrap();

        assert!(
            !broken_dir.exists(),
            "recovery-time cache load should purge entries missing object.cache"
        );

        let after = RecoveryManager::global_fingerprint_guard_metrics();
        assert_eq!(
            after
                .mismatches_detected
                .saturating_sub(before.mismatches_detected),
            0
        );
        assert_eq!(
            after
                .invalidations_performed
                .saturating_sub(before.invalidations_performed),
            0
        );
    }

    #[tokio::test]
    async fn test_recover_skips_guard_when_manifest_fingerprint_empty() {
        let _env_guard = ENV_LOCK
            .get_or_init(|| std::sync::Mutex::new(()))
            .lock()
            .unwrap();
        RecoveryManager::reset_fingerprint_guard_metrics_for_tests();
        let before = RecoveryManager::global_fingerprint_guard_metrics();

        let dir = tempdir().unwrap();
        let cache_root = tempdir().unwrap();
        let _cache_dir =
            EnvVarGuard::set("DRIFT_NVME_CACHE_DIR", cache_root.path().to_str().unwrap());

        let path = "custom/provider/object-22.drift";
        let op = create_memory_operator();
        op.write(path, vec![9u8; 64]).await.unwrap();

        let disk = DiskManager::new(op.clone(), path.to_string());
        let _ = disk.read_at(0, 8).await.unwrap();
        assert!(
            DiskManager::nvme_cached_fingerprint_for_object(&op, path).is_some(),
            "cache should be populated before recovery"
        );

        let manifest = Arc::new(ServerManifestManager::new(dir.path(), 8).unwrap());
        manifest
            .apply_atomic(|m| {
                m.add_bucket(22, "run_remote".into(), Some(vec![0.0; 8]));
                m.update_bucket_remote_meta(22, "run_remote".into(), path.into(), String::new());
            })
            .unwrap();

        let mgr = RecoveryManager::new_with_fingerprint_guard(
            dir.path(),
            manifest,
            RecoveryFingerprintGuardConfig {
                policy: FingerprintMismatchPolicy::InvalidateAndContinue,
                max_mismatches: Some(0),
            },
        );
        let bucket_mgr = create_memory_manager(op.clone());
        let wal_dir = dir.path().join("wal");
        mgr.recover(&bucket_mgr, 8, &wal_dir).await.unwrap();

        let cache_after = DiskManager::nvme_cached_fingerprint_for_object(&op, path);
        assert!(
            cache_after.is_some(),
            "empty manifest fingerprint should skip guard and preserve cache"
        );

        let after = RecoveryManager::global_fingerprint_guard_metrics();
        assert_eq!(
            after
                .mismatches_detected
                .saturating_sub(before.mismatches_detected),
            0
        );
    }

    #[tokio::test]
    async fn test_recover_local_bucket_does_not_touch_remote_guard() {
        let _env_guard = ENV_LOCK
            .get_or_init(|| std::sync::Mutex::new(()))
            .lock()
            .unwrap();
        RecoveryManager::reset_fingerprint_guard_metrics_for_tests();
        let before = RecoveryManager::global_fingerprint_guard_metrics();

        let dir = tempdir().unwrap();
        let staging_dir = dir.path().join("staging");
        std::fs::create_dir_all(&staging_dir).unwrap();
        std::fs::write(staging_dir.join("bucket_23.drift"), vec![0u8; 4]).unwrap();

        let cache_root = tempdir().unwrap();
        let _cache_dir =
            EnvVarGuard::set("DRIFT_NVME_CACHE_DIR", cache_root.path().to_str().unwrap());

        let path = "custom/provider/object-23.drift";
        let op = create_memory_operator();
        op.write(path, vec![4u8; 64]).await.unwrap();
        let disk = DiskManager::new(op.clone(), path.to_string());
        let _ = disk.read_at(0, 8).await.unwrap();
        let cached_fp = DiskManager::nvme_cached_fingerprint_for_object(&op, path).unwrap();

        let manifest = Arc::new(ServerManifestManager::new(dir.path(), 8).unwrap());
        manifest
            .apply_atomic(|m| {
                m.add_bucket(23, "run_remote".into(), Some(vec![0.0; 8]));
                m.update_bucket_remote_meta(
                    23,
                    "run_remote".into(),
                    path.into(),
                    format!("{cached_fp}-stale"),
                );
            })
            .unwrap();

        let mgr = RecoveryManager::new_with_fingerprint_guard(
            dir.path(),
            manifest,
            RecoveryFingerprintGuardConfig {
                policy: FingerprintMismatchPolicy::InvalidateAndContinue,
                max_mismatches: Some(0),
            },
        );
        let bucket_mgr = create_memory_manager(op.clone());
        let wal_dir = dir.path().join("wal");
        mgr.recover(&bucket_mgr, 8, &wal_dir).await.unwrap();

        let (local_path, class) = bucket_mgr.get_location(23).unwrap();
        assert_eq!(class, StorageClass::Local);
        assert!(local_path.contains("bucket_23.drift"));
        assert!(
            DiskManager::nvme_cached_fingerprint_for_object(&op, path).is_some(),
            "local bucket recovery should not invalidate remote cache"
        );

        let after = RecoveryManager::global_fingerprint_guard_metrics();
        assert_eq!(
            after
                .mismatches_detected
                .saturating_sub(before.mismatches_detected),
            0
        );
    }

    #[tokio::test]
    async fn test_recover_invalid_env_policy_falls_back_to_invalidate_and_continue() {
        let _env_guard = ENV_LOCK
            .get_or_init(|| std::sync::Mutex::new(()))
            .lock()
            .unwrap();
        RecoveryManager::reset_fingerprint_guard_metrics_for_tests();
        let before = RecoveryManager::global_fingerprint_guard_metrics();

        let dir = tempdir().unwrap();
        let cache_root = tempdir().unwrap();
        let _cache_dir =
            EnvVarGuard::set("DRIFT_NVME_CACHE_DIR", cache_root.path().to_str().unwrap());
        let _policy = EnvVarGuard::set("DRIFT_RECOVERY_FINGERPRINT_POLICY", "not_a_real_policy");
        let _max = EnvVarGuard::set("DRIFT_RECOVERY_FINGERPRINT_MAX_MISMATCHES", "not_a_number");

        let path = "custom/provider/object-24.drift";
        let op = create_memory_operator();
        op.write(path, vec![2u8; 64]).await.unwrap();
        let disk = DiskManager::new(op.clone(), path.to_string());
        let _ = disk.read_at(0, 8).await.unwrap();
        let cached_fp = DiskManager::nvme_cached_fingerprint_for_object(&op, path).unwrap();

        let manifest = Arc::new(ServerManifestManager::new(dir.path(), 8).unwrap());
        manifest
            .apply_atomic(|m| {
                m.add_bucket(24, "run_remote".into(), Some(vec![0.0; 8]));
                m.update_bucket_remote_meta(
                    24,
                    "run_remote".into(),
                    path.into(),
                    format!("{cached_fp}-stale"),
                );
            })
            .unwrap();

        let mgr = RecoveryManager::new(dir.path(), manifest);
        let bucket_mgr = create_memory_manager(op.clone());
        let wal_dir = dir.path().join("wal");
        mgr.recover(&bucket_mgr, 8, &wal_dir).await.unwrap();

        assert!(
            DiskManager::nvme_cached_fingerprint_for_object(&op, path).is_none(),
            "invalid env policy should fallback to invalidate_and_continue"
        );

        let after = RecoveryManager::global_fingerprint_guard_metrics();
        assert!(
            after
                .mismatches_detected
                .saturating_sub(before.mismatches_detected)
                >= 1
        );
        assert!(
            after
                .invalidations_performed
                .saturating_sub(before.invalidations_performed)
                >= 1
        );
        assert_eq!(
            after
                .fail_fast_aborts
                .saturating_sub(before.fail_fast_aborts),
            0
        );
    }

    #[tokio::test]
    async fn test_recover_fingerprint_policy_fail_fast_aborts_startup() {
        let _env_guard = ENV_LOCK
            .get_or_init(|| std::sync::Mutex::new(()))
            .lock()
            .unwrap();
        RecoveryManager::reset_fingerprint_guard_metrics_for_tests();
        let before = RecoveryManager::global_fingerprint_guard_metrics();

        let dir = tempdir().unwrap();
        let cache_root = tempdir().unwrap();
        let _cache_dir =
            EnvVarGuard::set("DRIFT_NVME_CACHE_DIR", cache_root.path().to_str().unwrap());

        let path = "custom/provider/object-11.drift";
        let op = create_memory_operator();
        op.write(path, vec![5u8; 64]).await.unwrap();

        let disk = DiskManager::new(op.clone(), path.to_string());
        let _ = disk.read_at(0, 8).await.unwrap();

        let cached_fp = DiskManager::nvme_cached_fingerprint_for_object(&op, path).unwrap();
        let manifest = Arc::new(ServerManifestManager::new(dir.path(), 8).unwrap());
        manifest
            .apply_atomic(|m| {
                m.add_bucket(11, "run_remote".into(), Some(vec![0.0; 8]));
                m.update_bucket_remote_meta(
                    11,
                    "run_remote".into(),
                    path.into(),
                    format!("{cached_fp}-stale"),
                );
            })
            .unwrap();

        let mgr = RecoveryManager::new_with_fingerprint_guard(
            dir.path(),
            manifest,
            RecoveryFingerprintGuardConfig {
                policy: FingerprintMismatchPolicy::FailFast,
                max_mismatches: None,
            },
        );
        let bucket_mgr = create_memory_manager(op.clone());
        let wal_dir = dir.path().join("wal");
        let err = match mgr.recover(&bucket_mgr, 8, &wal_dir).await {
            Ok(_) => panic!("fail_fast policy should abort on mismatch"),
            Err(e) => e,
        };
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);

        let cache_after = DiskManager::nvme_cached_fingerprint_for_object(&op, path);
        assert!(
            cache_after.is_some(),
            "fail_fast should not invalidate cache as startup aborts immediately"
        );
        let after = RecoveryManager::global_fingerprint_guard_metrics();
        assert!(
            after
                .mismatches_detected
                .saturating_sub(before.mismatches_detected)
                >= 1
        );
        assert!(
            after
                .fail_fast_aborts
                .saturating_sub(before.fail_fast_aborts)
                >= 1
        );
    }

    #[tokio::test]
    async fn test_recover_fingerprint_policy_max_mismatches_enforced() {
        let _env_guard = ENV_LOCK
            .get_or_init(|| std::sync::Mutex::new(()))
            .lock()
            .unwrap();

        let dir = tempdir().unwrap();
        let cache_root = tempdir().unwrap();
        let _cache_dir =
            EnvVarGuard::set("DRIFT_NVME_CACHE_DIR", cache_root.path().to_str().unwrap());

        let path = "custom/provider/object-12.drift";
        let op = create_memory_operator();
        op.write(path, vec![6u8; 64]).await.unwrap();

        let disk = DiskManager::new(op.clone(), path.to_string());
        let _ = disk.read_at(0, 8).await.unwrap();

        let cached_fp = DiskManager::nvme_cached_fingerprint_for_object(&op, path).unwrap();
        let manifest = Arc::new(ServerManifestManager::new(dir.path(), 8).unwrap());
        manifest
            .apply_atomic(|m| {
                m.add_bucket(12, "run_remote".into(), Some(vec![0.0; 8]));
                m.update_bucket_remote_meta(
                    12,
                    "run_remote".into(),
                    path.into(),
                    format!("{cached_fp}-stale"),
                );
            })
            .unwrap();

        let mgr = RecoveryManager::new_with_fingerprint_guard(
            dir.path(),
            manifest,
            RecoveryFingerprintGuardConfig {
                policy: FingerprintMismatchPolicy::InvalidateAndContinue,
                max_mismatches: Some(0),
            },
        );
        let bucket_mgr = create_memory_manager(op.clone());
        let wal_dir = dir.path().join("wal");
        let err = match mgr.recover(&bucket_mgr, 8, &wal_dir).await {
            Ok(_) => panic!("max_mismatches=0 should fail on first mismatch"),
            Err(e) => e,
        };
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    #[tokio::test]
    async fn test_recover_uses_manifest_metric_for_router() {
        let dir = tempdir().unwrap();
        let manifest = Arc::new(
            ServerManifestManager::new_with_metric(dir.path(), 2, Metric::COSINE).unwrap(),
        );
        manifest
            .apply_atomic(|m| {
                // This setup routes differently under L2 vs COSINE for query [1, 0]:
                // L2 => bucket 2; COSINE => bucket 1.
                m.add_bucket(1, "run_1".into(), Some(vec![2.0, 0.0]));
                m.add_bucket(2, "run_2".into(), Some(vec![0.9, 0.9]));
            })
            .unwrap();

        let mgr = RecoveryManager::new(dir.path(), manifest);
        let bucket_mgr = create_noop_manager();
        let wal_dir = dir.path().join("wal");

        let (router, _) = mgr.recover(&bucket_mgr, 2, &wal_dir).await.unwrap();
        let selected = router.read().route(&[1.0, 0.0]);
        assert_eq!(selected, 1, "Recovery should honor COSINE metric");
    }

    #[tokio::test]
    async fn test_recover_scans_wal() {
        let dir = tempdir().unwrap();
        let wal_dir = dir.path().join("wal");
        std::fs::create_dir_all(&wal_dir).unwrap();

        // Create WAL file
        let mut writer = WalWriter::new(wal_dir.join("log.log")).unwrap();
        writer.write_insert(10, &vec![1.0]).unwrap();
        writer.write_delete(20).unwrap();
        writer.sync().unwrap();

        let manifest = Arc::new(ServerManifestManager::new(dir.path(), 1).unwrap());
        let mgr = RecoveryManager::new(dir.path(), manifest);
        let bucket_mgr = create_noop_manager();

        let (_, replay) = mgr.recover(&bucket_mgr, 1, &wal_dir).await.unwrap();

        assert_eq!(replay.inserts.len(), 1);
        assert_eq!(replay.inserts[0].0, 10);
        assert_eq!(replay.deletes.len(), 1);
        assert_eq!(replay.deletes[0], 20);
    }

    #[tokio::test]
    async fn test_recover_empty_state_defaults() {
        let dir = tempdir().unwrap();
        // Manifest creates and saves default state
        let manifest = Arc::new(ServerManifestManager::new(dir.path(), 128).unwrap());
        let mgr = RecoveryManager::new(dir.path(), manifest);

        let bucket_mgr = create_noop_manager();
        let wal_dir = dir.path().join("wal"); // Empty dir

        let (router, replay) = mgr.recover(&bucket_mgr, 128, &wal_dir).await.unwrap();

        // Should return empty router (Day 0)
        assert!(router.read().get_centroid(0).is_none());
        assert!(replay.inserts.is_empty());
    }

    #[tokio::test]
    async fn test_recover_scans_wal_for_inserts_and_deletes() {
        let dir = tempdir().unwrap();
        let wal_dir = dir.path().join("wal");
        std::fs::create_dir_all(&wal_dir).unwrap();

        // Create a WAL file manually
        let mut writer = WalWriter::new(wal_dir.join("current.log")).unwrap();
        writer.write_insert(10, &vec![1.0]).unwrap();
        writer.write_delete(20).unwrap();
        writer.sync().unwrap();

        let manifest = Arc::new(ServerManifestManager::new(dir.path(), 1).unwrap());
        let mgr = RecoveryManager::new(dir.path(), manifest);
        let bucket_mgr = create_noop_manager();

        let (_, replay) = mgr.recover(&bucket_mgr, 1, &wal_dir).await.unwrap();

        assert_eq!(replay.inserts.len(), 1);
        assert_eq!(replay.inserts[0].0, 10);

        assert_eq!(replay.deletes.len(), 1);
        assert_eq!(replay.deletes[0], 20);
    }
}
