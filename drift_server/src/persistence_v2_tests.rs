#[cfg(test)]
mod tests {
    use crate::janitor_v2::{Janitor, JanitorConfig, JanitorVars};
    use crate::local_staging::LocalStagingManager;
    use crate::manifest::ServerManifestManager;
    use crate::persistence_v2::PersistenceManager;
    use crate::recovery::RecoveryManager;
    use drift_core::index_v2::VectorIndex;
    use drift_core::lock_manager::BucketCoordinator;
    use drift_core::manifest::pb::Centroid;
    use drift_core::math::Metric;
    use drift_core::router::Router;
    use drift_core::wal_v2::WalManager;
    use drift_kv::bitstore::BitStore;
    use drift_storage::bucket_manager::BucketManager;
    use opendal::Operator;
    use opendal::services::Fs;
    use parking_lot::{Mutex, RwLock};
    use std::path::Path;
    use std::sync::Arc;
    use std::time::Duration;
    use tempfile::tempdir;

    fn create_local_operator(path: &Path) -> std::io::Result<Operator> {
        let builder = Fs::default().root(path.to_str().unwrap());
        Ok(Operator::new(builder)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?
            .finish())
    }

    // --- SETUP HELPER ---
    async fn setup_env(
        path: &std::path::Path,
        capacity: usize,
        dim: usize,
    ) -> (
        Arc<VectorIndex>,
        Arc<BucketManager>,
        Janitor,
        Arc<ServerManifestManager>,
        Arc<PersistenceManager>,
    ) {
        let data_dir = path.join("data");
        let wal_dir = path.join("wal");
        let kv_dir = data_dir.join("kv");
        let staging_dir = data_dir.join("staging");

        std::fs::create_dir_all(&data_dir).unwrap();
        std::fs::create_dir_all(&wal_dir).unwrap();
        std::fs::create_dir_all(&kv_dir).unwrap();
        std::fs::create_dir_all(&staging_dir).unwrap();

        let op = create_local_operator(&staging_dir).unwrap();
        let persistence = Arc::new(PersistenceManager::new(op.clone()));
        let staging = Arc::new(LocalStagingManager::new(&staging_dir).unwrap());
        let coordinator = Arc::new(BucketCoordinator::new());
        let bucket_manager = Arc::new(BucketManager::new(
            op.clone(),
            op.clone(),
            4,
            coordinator.clone(),
        ));

        let manifest = Arc::new(ServerManifestManager::new(&data_dir, dim as u32).unwrap());

        let centroids = vec![Centroid {
            id: 1,
            vector: vec![0.0; dim],
        }];
        let router = Arc::new(RwLock::new(
            Router::new(&centroids, &[0], dim, Metric::L2).unwrap(),
        ));

        let wal = Arc::new(Mutex::new(WalManager::new(&wal_dir).unwrap()));
        let kv = Arc::new(BitStore::new(&kv_dir).unwrap());

        let index = Arc::new(VectorIndex::new(
            dim,
            capacity,
            router,
            wal,
            bucket_manager.clone(),
            kv,
        ));

        let janitor = Janitor::new(JanitorConfig {
            index: index.clone(),
            manifest: manifest.clone(),
            staging,
            persistence: persistence.clone(),
            bucket_manager: bucket_manager.clone(),
            coordinator,
            vars: JanitorVars {
                check_interval: Duration::from_millis(100),
                promotion_threshold_bytes: 1024,
                max_bucket_capacity: 2000,
                split_threshold: 0.8,
                drift_threshold: 0.15,
                ..Default::default()
            },
        });

        (index, bucket_manager, janitor, manifest, persistence)
    }

    // --- TEST 1: WAL Recovery (Crash before Flush) ---
    #[tokio::test]
    async fn test_wal_delete_recovery_logic() {
        let dir = tempdir().unwrap();
        let root = dir.path();

        // 1. Setup Components
        let (index, bucket_manager, _, manifest, _) = setup_env(root, 100, 2).await;

        // 2. Insert & Delete
        index.insert(100, &[1.0, 1.0]).unwrap();
        index.delete(100).unwrap(); // Writes Delete to WAL

        // Force WAL flush by writing another entry
        index.delete(101).unwrap();

        // 3. Drop to flush/close
        drop(index);

        // 4. Recover
        let wal_path = root.join("wal");

        let recovery_mgr = RecoveryManager::new(&root.join("data"), manifest.clone());

        // ⚡ NEW API: Pass WAL Path directly
        let (_, replay_data) = recovery_mgr
            .recover(&bucket_manager, 2, &wal_path)
            .await
            .unwrap();

        assert!(
            replay_data.deletes.contains(&100),
            "WAL Scan missed the delete of ID 100"
        );
        assert!(replay_data.deletes.contains(&101));
    }

    // --- TEST 2: S3 Tombstone Hydration (Crash after Flush) ---
    #[tokio::test]
    async fn test_s3_tombstone_hydration_lifecycle() {
        let dir = tempdir().unwrap();
        let (index, _, janitor, _, persistence) = setup_env(dir.path(), 100, 2).await;

        // 1. Insert & Delete
        index.insert(200, &[0.0, 0.0]).unwrap();
        index.delete(200).unwrap();

        // 2. FORCE FLUSH
        persistence
            .flush_tombstones(&[200], "run_simulation")
            .await
            .unwrap();

        // 3. Restart
        drop(index);
        drop(janitor);

        // 4. Hydrate
        let recovered_deletes = persistence.load_all_tombstones().await.unwrap();

        assert!(
            recovered_deletes.contains(&200),
            "Failed to hydrate deletions from disk (S3)"
        );
    }
}
