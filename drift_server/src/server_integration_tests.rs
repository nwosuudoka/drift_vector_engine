#[cfg(test)]
mod tests {
    use crate::local_staging::LocalStagingManager;
    use crate::manifest::ServerManifestManager;
    use crate::persistence_v2::PersistenceManager;
    use crate::recovery::RecoveryManager;
    use drift_core::lock_manager::BucketCoordinator;
    use drift_core::partitioner::PartitionGroup;
    use drift_core::quantizer::Quantizer;
    use drift_core::wal_v2::WalWriter;
    use drift_storage::bucket_file_writer::BucketFileWriter;
    use drift_storage::bucket_manager::{BucketManager, StorageClass};
    use drift_traits::DiskSearcher;
    use opendal::{Operator, services};
    use std::sync::Arc;
    use tempfile::tempdir;

    fn create_fs_operator(path: &std::path::Path) -> Operator {
        let mut builder = services::Fs::default();
        builder = builder.root(path.to_str().unwrap());
        Operator::new(builder).unwrap().finish()
    }

    fn mock_batch(start_id: u64, count: usize, dim: usize, val_offset: f32) -> PartitionGroup {
        let mut ids = Vec::new();
        let mut vecs = Vec::new();
        for i in 0..count {
            let id = start_id + i as u64;
            ids.push(id);
            vecs.extend(vec![id as f32 + val_offset; dim]);
        }

        PartitionGroup {
            ids,
            flat_vectors: vecs,
            count,
            centroid: Some(vec![0.0; dim]),
        }
    }

    #[tokio::test]
    async fn test_full_lifecycle_flush_promote_recover() {
        let dir = tempdir().unwrap();
        let data_dir = dir.path().join("data");
        std::fs::create_dir(&data_dir).unwrap();

        let dim = 8;
        let bucket_id = 1;

        let manifest = Arc::new(ServerManifestManager::new(dir.path(), dim as u32).unwrap());
        let staging = Arc::new(LocalStagingManager::new(&data_dir).unwrap());
        let op = create_fs_operator(&data_dir);
        let persistence = PersistenceManager::new(op.clone());

        // A. Flush to Local Staging
        let group = mock_batch(0, 10, dim, 0.0);
        staging.append_batch(bucket_id, &group).await.unwrap();

        manifest
            .apply_atomic(|m| {
                m.add_bucket(bucket_id, String::new(), group.centroid.clone());
                m.update_bucket_stats(bucket_id, 10, 0);
            })
            .unwrap();

        // B. Promote to S3
        let (local_ids, local_vecs) = staging.read_full_bucket(bucket_id).await.unwrap();
        let (new_run_id, _) = persistence
            .promote_to_s3(bucket_id, &local_ids, &local_vecs, None, dim)
            .await
            .unwrap();

        manifest
            .apply_atomic(|m| {
                m.update_bucket_run_id(bucket_id, new_run_id.clone());
            })
            .unwrap();
        staging.delete_bucket(bucket_id).await.unwrap();

        let coordinator = Arc::new(BucketCoordinator::new());

        // C. Recover
        //  Pass both operators (Local/Remote) to BucketManager
        let bucket_manager = BucketManager::new(op.clone(), op.clone(), 4, coordinator.clone());
        let recovery = RecoveryManager::new(dir.path(), manifest.clone());

        let (router_opt, _) = recovery.recover(&bucket_manager, dim).await.unwrap();

        // D. Verify
        let router = router_opt.read();
        assert!(router.get_centroid(bucket_id).is_some());

        let (reg_path, class) = bucket_manager
            .get_location(bucket_id)
            .expect("Bucket registered");
        assert!(reg_path.contains(&new_run_id));
        assert_eq!(class, StorageClass::Remote);

        // Search
        let query = vec![5.0; dim];

        // We drop the router lock before await to be safe
        drop(router);

        //  Use search_and_refine (Atomic API)
        let results = bucket_manager
            .search_and_refine(
                &[bucket_id],
                &query,
                5,  // k
                15, // oversample
                Arc::new(drift_traits::mock::NoTombstones),
            )
            .await;

        assert!(!results.is_empty(), "Should find results");
        // Check ID 5 (which should have value 5.0, distance 0.0)
        assert_eq!(results[0].0, 5);
        assert!(results[0].1 < 0.001);
    }

    #[tokio::test]
    async fn test_recovery_local_priority() {
        let dir = tempdir().unwrap();
        let data_dir = dir.path().join("data");
        std::fs::create_dir(&data_dir).unwrap();

        let dim = 8;
        let bucket_id = 1;

        let manifest = Arc::new(ServerManifestManager::new(dir.path(), dim as u32).unwrap());
        let op = create_fs_operator(&data_dir);

        // 1. Create "Old" S3 File
        let run_id = "run_OLD";
        let s3_path = data_dir.join(format!("bucket_{}_{}.drift", bucket_id, run_id));
        {
            let file = std::fs::File::create(s3_path).unwrap();
            let q = Quantizer::train(&vec![0.0; dim], dim);
            let mut w = BucketFileWriter::new_streaming(file, [0u8; 16], q, dim).unwrap();
            w.write_batch(&[100], &vec![100.0; dim]).unwrap();
            w.finalize().unwrap();
        }

        // 2. Create "New" Local File
        let local_path = data_dir.join(format!("bucket_{}.drift", bucket_id));
        {
            let file = std::fs::File::create(local_path).unwrap();
            let q = Quantizer::train(&vec![0.0; dim], dim);
            let mut w = BucketFileWriter::new_streaming(file, [0u8; 16], q, dim).unwrap();
            w.write_batch(&[200], &vec![200.0; dim]).unwrap();
            w.finalize().unwrap();
        }

        manifest
            .apply_atomic(|m| {
                m.add_bucket(bucket_id, run_id.to_string(), Some(vec![0.0; dim]));
            })
            .unwrap();

        // 4. Recover
        //  Pass dual operators
        let coordinator = Arc::new(BucketCoordinator::new());
        let bucket_manager = BucketManager::new(op.clone(), op.clone(), 1, coordinator.clone());
        let recovery = RecoveryManager::new(dir.path(), manifest.clone());
        let (_router_opt, _) = recovery.recover(&bucket_manager, dim).await.unwrap();

        // 5. Verify Priority
        let (path, class) = bucket_manager.get_location(bucket_id).unwrap();
        assert!(path.contains("bucket_1.drift"));
        assert_eq!(class, StorageClass::Local); // Local Priority

        let query = vec![200.0; dim];

        //  Use search_and_refine
        let results = bucket_manager
            .search_and_refine(
                &[bucket_id],
                &query,
                1,
                3,
                Arc::new(drift_traits::mock::NoTombstones),
            )
            .await;

        assert!(!results.is_empty());
        assert_eq!(results[0].0, 200, "Recovery must prefer Local Staging");
    }

    // --- TEST 3: Corner Case - WAL Replay (Day 0) ---
    #[tokio::test]
    async fn test_wal_replay() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("write_ahead_log.wal");

        let dim = 8;
        // Day 0: Empty Manifest (No buckets yet)
        let manifest = Arc::new(ServerManifestManager::new(dir.path(), dim as u32).unwrap());
        let op = create_fs_operator(dir.path());

        //  Use Dual Operator constructor (Mocking same op for both)
        let coordinator = Arc::new(BucketCoordinator::new());
        let bucket_manager = BucketManager::new(op.clone(), op.clone(), 1, coordinator.clone());

        // 1. Write to WAL
        {
            let mut wal = WalWriter::new(&wal_path).unwrap();
            wal.write_insert(999, &vec![1.0; dim]).unwrap();
            wal.write_insert(888, &vec![2.0; dim]).unwrap();
            wal.sync().unwrap();
        }

        // 2. Recover
        let recovery = RecoveryManager::new(dir.path(), manifest.clone());
        let (router, wal_vectors) = recovery.recover(&bucket_manager, dim).await.unwrap();

        // 3. Verify
        // We verify it has 0 centroids.
        assert!(
            router.read().get_centroid(0).is_none(),
            "Router should be empty"
        );

        // WAL items must be returned
        assert_eq!(wal_vectors.len(), 2);
        assert_eq!(wal_vectors[0].0, 999);
        assert_eq!(wal_vectors[1].0, 888);
    }
}

#[cfg(test)]
mod wal_tests_tests {
    use drift_core::index_v2::VectorIndex;
    use drift_core::lock_manager::BucketCoordinator;
    use drift_core::router::{Metric, Router};
    use drift_core::wal_v2::WalManager; // Use WalManager
    use drift_storage::bucket_manager::BucketManager;
    use opendal::{Operator, services};
    use parking_lot::{Mutex, RwLock};
    use std::sync::Arc;
    use tempfile::tempdir;

    fn create_fs_operator(path: &std::path::Path) -> Operator {
        let builder = services::Fs::default().root(path.to_str().unwrap());
        Operator::new(builder).unwrap().finish()
    }

    #[tokio::test]
    async fn test_wal_rotation_persistence() {
        let dir = tempdir().unwrap();
        let wal_dir = dir.path().join("wal");
        std::fs::create_dir_all(&wal_dir).unwrap();
        let data_dir = dir.path().join("data");
        std::fs::create_dir_all(&data_dir).unwrap();

        let dim = 2;
        let cap = 5; // Capacity 5

        let op = create_fs_operator(&data_dir);
        let coordinator = Arc::new(BucketCoordinator::new());
        let storage = Arc::new(BucketManager::new(
            op.clone(),
            op.clone(),
            1,
            coordinator.clone(),
        ));

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
            cap,
            router,
            wal_mgr.clone(),
            storage.clone(),
        ));

        // 2. Insert Batch 1 (5 Items)
        // ⚡ insert() triggers rotate_active() immediately when len >= cap.
        // So after the 5th insert, rotation happens.
        for i in 0..5 {
            index.insert(i, vec![0.0; dim]).unwrap();
        }

        // *State check:*
        // 1 WAL file (Closed/Frozen) containing the 5 items.
        // 1 WAL file (Active) that is empty.
        let files_1 = std::fs::read_dir(&wal_dir).unwrap().count();
        assert_eq!(
            files_1, 2,
            "Should have rotated: 1 frozen WAL + 1 active WAL"
        );

        // 3. Force Rotation (Insert 6th item)
        // This goes into the NEW active table/WAL.
        // It does not trigger another rotation until that table hits 5 items.
        index.insert(5, vec![0.0; dim]).unwrap();

        let files_2 = std::fs::read_dir(&wal_dir).unwrap().count();
        assert_eq!(files_2, 2, "Still 2 files (Frozen + Active)");

        // 4. Flush Frozen
        // This takes the Old WAL (ID 1) and flushes it.
        let (_parts, wal_ids) = index.flush_frozen().expect("Should have frozen data");
        assert_eq!(wal_ids.len(), 1);

        // 5. Confirm Flush (Deletes WAL 1)
        index.acknowledge_flush(&wal_ids).unwrap();

        // 6. Verify Files on Disk
        let files_after = std::fs::read_dir(&wal_dir).unwrap();
        let filenames: Vec<String> = files_after
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();

        // We expect only the ACTIVE WAL (ID 2 or greater) to remain.
        assert_eq!(filenames.len(), 1, "Old WAL should be deleted");
        println!("Remaining WAL: {}", filenames[0]);
    }
}
