#[cfg(test)]
mod tests {
    use crate::janitor_v2::{Janitor, JanitorConfig};
    use crate::local_staging::LocalStagingManager;
    use crate::manifest::ServerManifestManager;
    use crate::persistence_v2::PersistenceManager;
    use drift_core::index_v2::VectorIndex;
    use drift_core::lock_manager::BucketCoordinator;
    use drift_core::router::{Metric, Router};
    use drift_core::wal_v2::WalManager;
    use drift_storage::bucket_manager::BucketManager;
    use opendal::{Operator, services};
    use parking_lot::{Mutex, RwLock};
    use std::sync::Arc;
    use std::time::Duration;
    use tempfile::tempdir;

    fn create_fs_operator(path: &std::path::Path) -> Operator {
        let builder = services::Fs::default().root(path.to_str().unwrap());
        Operator::new(builder).unwrap().finish()
    }

    #[tokio::test]
    async fn test_janitor_v2_flush_lifecycle() {
        let dir = tempdir().unwrap();
        let data_dir = dir.path().join("data");
        std::fs::create_dir(&data_dir).unwrap();
        let dim = 2;

        let manifest = Arc::new(ServerManifestManager::new(dir.path(), dim as u32).unwrap());
        let staging = Arc::new(LocalStagingManager::new(&data_dir).unwrap());
        let op = create_fs_operator(&data_dir);
        let persistence = PersistenceManager::new(op.clone());
        let coordinator = Arc::new(BucketCoordinator::new());

        let bucket_manager = Arc::new(BucketManager::new(
            op.clone(),
            op.clone(),
            8,
            coordinator.clone(),
        ));

        manifest
            .apply_atomic(|m| {
                m.add_bucket(0, "run_init".to_string(), Some(vec![0.0; dim]));
            })
            .unwrap();

        let wal_dir = dir.path().join("wal");
        let wal_mgr = Arc::new(Mutex::new(WalManager::new(&wal_dir).unwrap()));

        let centroids = vec![drift_core::manifest::pb::Centroid {
            id: 0,
            vector: vec![0.0; dim],
        }];
        let router = Arc::new(RwLock::new(
            Router::new(&centroids, &[0], dim, Metric::L2).unwrap(),
        ));

        let index = Arc::new(VectorIndex::new(
            dim,
            10,
            router,
            wal_mgr,
            bucket_manager.clone(),
        ));

        let janitor = Janitor::new(JanitorConfig {
            index: index.clone(),
            manifest: manifest.clone(),
            staging: staging.clone(),
            persistence,
            bucket_manager: bucket_manager.clone(),
            check_interval: Duration::from_millis(10),
            promotion_threshold_bytes: 100,
            coordinator: coordinator.clone(),
        });

        // 4. Trigger Flush Condition (Split Batches)
        // Batch A: Fill Capacity (10 items) -> Triggers Rotation
        let batch_a: Vec<(u64, Vec<f32>)> = (0..10).map(|i| (i, vec![0.0; dim])).collect();
        let rotated = index.insert_batch(&batch_a).unwrap();
        assert!(rotated, "First batch should fill capacity and rotate");

        // Batch B: New Active Data (5 items) -> No Rotation
        let batch_b: Vec<(u64, Vec<f32>)> = (10..15).map(|i| (i, vec![0.0; dim])).collect();
        let rotated_b = index.insert_batch(&batch_b).unwrap();
        assert!(!rotated_b, "Second batch should fit in new active table");

        // 5. Run Janitor (Flushes Frozen)
        let handle = tokio::spawn(async move { janitor.run().await });
        tokio::time::sleep(Duration::from_millis(200)).await;
        handle.abort();

        // 6. Verify Persistence
        let state = manifest.get_state();
        let b0 = state.get_buckets().iter().find(|b| b.id == 0).unwrap();

        // Assert: Only the 10 frozen items were flushed.
        assert_eq!(
            b0.vector_count, 10,
            "Manifest should reflect ONLY flushed frozen items"
        );

        //  Expect None, because Janitor should have cleared it!
        assert!(
            index.flush_frozen().is_none(),
            "Frozen slot should be cleared after flush"
        );

        // Verify WAL Rotation:
        // 1 Frozen WAL was deleted. 1 Active WAL should remain.
        let files = std::fs::read_dir(&wal_dir).unwrap().count();
        assert_eq!(files, 1, "Should have exactly 1 active WAL file remaining");
    }
}

#[cfg(test)]
mod stress_tests {
    use crate::janitor_v2::{Janitor, JanitorConfig};
    use crate::local_staging::LocalStagingManager;
    use crate::manifest::ServerManifestManager;
    use crate::persistence_v2::PersistenceManager;
    use drift_core::index_v2::VectorIndex;
    use drift_core::lock_manager::BucketCoordinator;
    use drift_core::router::{Metric, Router};
    use drift_core::wal_v2::WalManager;
    use drift_storage::bucket_manager::BucketManager;
    use opendal::{Operator, services};
    use parking_lot::{Mutex, RwLock};
    use std::sync::Arc;
    use std::time::Duration;
    use tempfile::tempdir;

    fn create_fs_operator(path: &std::path::Path) -> Operator {
        let mut builder = services::Fs::default();
        builder = builder.root(path.to_str().unwrap());
        Operator::new(builder).unwrap().finish()
    }

    async fn setup_env(
        dir: &std::path::Path,
        capacity: usize,
        dim: usize,
    ) -> (
        Arc<VectorIndex>,
        Arc<BucketManager>,
        Janitor,
        Arc<ServerManifestManager>,
    ) {
        let _ = tracing_subscriber::fmt()
            .with_env_filter("drift_storage=info,drift_server=info") // Filter for relevant crates
            .with_test_writer() // Important: Prints to test output
            .try_init();

        let data_dir = dir.join("data");
        std::fs::create_dir(&data_dir).unwrap();

        let manifest = Arc::new(ServerManifestManager::new(dir, dim as u32).unwrap());
        let staging = Arc::new(LocalStagingManager::new(&data_dir).unwrap());
        let op = create_fs_operator(&data_dir);
        let persistence = PersistenceManager::new(op.clone());

        let coordinator = Arc::new(BucketCoordinator::new());

        let bucket_manager = Arc::new(BucketManager::new(
            op.clone(),
            op.clone(),
            8,
            coordinator.clone(),
        ));

        manifest
            .apply_atomic(|m| {
                m.add_bucket(0, "".to_string(), Some(vec![0.0; dim]));
            })
            .unwrap();

        let wal_dir = dir.join("wal");
        let wal_mgr = Arc::new(Mutex::new(WalManager::new(&wal_dir).unwrap()));

        let centroids = vec![drift_core::manifest::pb::Centroid {
            id: 0,
            vector: vec![0.0; dim],
        }];
        let router = Arc::new(RwLock::new(
            Router::new(&centroids, &[0], dim, Metric::L2).unwrap(),
        ));

        let index = Arc::new(VectorIndex::new(
            dim,
            capacity,
            router,
            wal_mgr,
            bucket_manager.clone(),
        ));

        // Low threshold to force S3 promotion
        let promotion_threshold = 1024;

        let janitor = Janitor::new(JanitorConfig {
            index: index.clone(),
            manifest: manifest.clone(),
            staging: staging,
            persistence,
            bucket_manager: bucket_manager.clone(),
            check_interval: Duration::from_millis(5),
            promotion_threshold_bytes: promotion_threshold,
            coordinator: coordinator.clone(),
        });

        (index, bucket_manager, janitor, manifest)
    }

    #[tokio::test]
    async fn test_consistent_reads_during_tiering() {
        let dir = tempdir().unwrap();
        let dim = 8;
        let capacity = 10;
        let (index, bucket_manager, janitor, _) = setup_env(dir.path(), capacity, dim).await;

        let janitor_task = tokio::spawn(async move { janitor.run().await });

        // 1. Insert Target Data (ID 999)
        let target_vec = vec![100.0; dim];
        index.insert(999, target_vec.clone()).unwrap();

        // 2. Spawn Reader Thread
        let index_ref = index.clone();
        let bucket_mgr_ref = bucket_manager.clone();

        let reader_handle = tokio::spawn(async move {
            for i in 0..50 {
                let results = index_ref
                    .search(&target_vec, 5, 0.9, 1.0, 10.0)
                    .await
                    .unwrap();

                let found = results.iter().any(|(id, _)| *id == 999);

                if !found {
                    // Debugging
                    println!("❌ Read #{}: MISSING ID 999.", i);
                    if let Some(v) = bucket_mgr_ref.get_version(0) {
                        println!("   -> Bucket 0 is at: {:?} (Path: {})", v.class, v.path);
                    } else {
                        println!("   -> Bucket 0 NOT FOUND in Registry!");
                    }
                    return false;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            true
        });

        // 3. Trigger Tiering Moves (Flush -> Promote)
        for i in 0..100 {
            let vec = vec![i as f32; dim];
            index.insert(i, vec).unwrap();
            tokio::time::sleep(Duration::from_millis(5)).await;
        }

        // Wait for reader
        let success = reader_handle.await.unwrap();
        janitor_task.abort();

        assert!(
            success,
            "Read Consistency Violation! Lost data during tiering transition."
        );
    }
}
