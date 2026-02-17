#[cfg(test)]
mod tests {
    use crate::manifest::ServerManifestManager;
    use crate::recovery::RecoveryManager;
    use drift_core::lock_manager::BucketCoordinator;
    use drift_core::wal::WalWriter;
    use drift_storage::bucket_manager::{BucketManager, StorageClass};
    use opendal::Operator;
    use opendal::services::Fs;
    use std::sync::Arc;
    use tempfile::tempdir;

    fn create_noop_manager() -> BucketManager {
        let builder = Fs::default().root("/tmp");
        let op = Operator::new(builder).unwrap().finish();
        BucketManager::new(op.clone(), op, 1, Arc::new(BucketCoordinator::new()))
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
