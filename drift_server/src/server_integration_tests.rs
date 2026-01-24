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
    use drift_traits::StorageEngine;
    use opendal::{Operator, services};
    use std::sync::Arc;
    use tempfile::tempdir;

    fn create_fs_operator(path: &std::path::Path) -> Operator {
        let builder = services::Fs::default().root(path.to_str().unwrap());
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
        let wal_dir = dir.path().join("wal");
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
        let bucket_manager = BucketManager::new(op.clone(), op.clone(), 4, coordinator.clone());
        // Fix: Pass data_dir so it looks in data_dir/staging
        let recovery = RecoveryManager::new(&data_dir, manifest.clone());

        let (router_lock, _replay) = recovery
            .recover(&bucket_manager, dim, &wal_dir)
            .await
            .unwrap();

        // D. Verify
        let router = router_lock.read();
        assert!(router.get_centroid(bucket_id).is_some());

        let (reg_path, class) = bucket_manager
            .get_location(bucket_id)
            .expect("Bucket registered");
        assert!(reg_path.contains(&new_run_id));
        assert_eq!(class, StorageClass::Remote);

        let query = vec![5.0; dim];
        drop(router);

        let results = bucket_manager
            .search_and_refine(&[bucket_id], &query, 5, 15)
            .await;

        assert!(!results.is_empty(), "Should find results");
        assert_eq!(results[0].0, 5);
    }

    #[tokio::test]
    async fn test_recovery_local_priority() {
        let dir = tempdir().unwrap();
        let data_dir = dir.path().join("data");
        let staging_dir = data_dir.join("staging");
        std::fs::create_dir_all(&staging_dir).unwrap();

        let dim = 8;
        let bucket_id = 1;

        let manifest = Arc::new(ServerManifestManager::new(dir.path(), dim as u32).unwrap());

        // ⚡ Operators
        let remote_op = create_fs_operator(&data_dir);
        let local_op = create_fs_operator(&staging_dir); // Rooted at staging

        // 1. Create "Old" S3 File in data_dir (Remote)
        let run_id = "run_OLD";
        let s3_path = data_dir.join(format!("bucket_{}_{}.drift", bucket_id, run_id));
        {
            let file = std::fs::File::create(s3_path).unwrap();
            let q = Quantizer::train(&vec![0.0; dim], dim);
            let mut w = BucketFileWriter::new_streaming(file, [0u8; 16], q, dim).unwrap();
            w.write_batch(&[100], &vec![100.0; dim]).unwrap();
            w.finalize().unwrap();
        }

        // 2. Create "New" Local File in staging_dir
        let local_path = staging_dir.join(format!("bucket_{}.drift", bucket_id));
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
        let coordinator = Arc::new(BucketCoordinator::new());
        // ⚡ Pass local_op (staging rooted)
        let bucket_manager = BucketManager::new(local_op, remote_op, 1, coordinator.clone());

        let recovery = RecoveryManager::new(&data_dir, manifest.clone());
        let wal_dir = data_dir.join("wal");
        let _ = recovery
            .recover(&bucket_manager, dim, &wal_dir)
            .await
            .unwrap();

        // 5. Verify Priority
        let (path, class) = bucket_manager.get_location(bucket_id).unwrap();

        // RecoveryManager registers "bucket_1.drift".
        // Since bucket_manager.local_op is rooted at staging/, this resolves correctly.
        assert!(
            path.contains("bucket_1.drift"),
            "Recovery failed to prefer Local Staging. Got: {}",
            path
        );
        assert_eq!(class, StorageClass::Local);

        let query = vec![200.0; dim];
        let results = bucket_manager
            .search_and_refine(&[bucket_id], &query, 1, 3)
            .await;

        assert!(!results.is_empty(), "Search should find data in local file");
        assert_eq!(results[0].0, 200, "Should find ID 200 from Local Staging");
    }

    #[tokio::test]
    async fn test_wal_replay() {
        let dir = tempdir().unwrap();
        let wal_dir = dir.path().join("wal").join("test_col");
        std::fs::create_dir_all(&wal_dir).unwrap();

        // Recovery expects .log extensions for WAL segments
        let wal_path = wal_dir.join("wal_1.log");

        let dim = 8;
        let manifest = Arc::new(ServerManifestManager::new(dir.path(), dim as u32).unwrap());
        let op = create_fs_operator(dir.path());
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
        // We use dir.path() here because we aren't testing staging resolution
        let recovery = RecoveryManager::new(dir.path(), manifest.clone());

        // Pass the directory containing the WAL file
        let (router, replay_data) = recovery
            .recover(&bucket_manager, dim, &wal_dir)
            .await
            .unwrap();

        assert!(
            router.read().get_centroid(0).is_none(),
            "Router should be empty"
        );

        assert_eq!(replay_data.inserts.len(), 2);
        assert_eq!(replay_data.inserts[0].0, 999);
        assert_eq!(replay_data.inserts[1].0, 888);
    }
}
