#[cfg(test)]
mod tests {
    use crate::config::{Config, FileConfig, StorageCommand};
    use crate::manager::CollectionManager;
    use std::sync::Arc;
    use tempfile::tempdir;

    const DIM: usize = 128;

    fn test_config(root: &std::path::Path) -> Config {
        Config {
            port: 50051,
            wal_dir: root.join("wal"),
            data_dir: root.join("data"),
            storage: StorageCommand::File(FileConfig {
                path: root.join("storage"),
            }),
            default_dim: DIM,
            max_bucket_capacity: 1000,
            ef_construction: 50,
            ef_search: 50,
        }
    }

    #[tokio::test]
    async fn test_manager_initializes_v2_architecture() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path());
        let data_dir = config.data_dir.clone();
        let manager = Arc::new(CollectionManager::new(config));

        let collection_name = "v2_init_test";

        // 1. Initialize Collection
        let collection = manager
            .get_or_create(collection_name, None, None)
            .await
            .expect("Failed to create collection");

        // 2. Verify Directory Structure (The V2 Signature)
        // Root: data/v2_init_test/
        let coll_root = data_dir.join(collection_name);

        // Assert these directories were created by the Manager
        assert!(coll_root.join("kv").exists(), "KV directory missing");
        assert!(
            coll_root.join("staging").exists(),
            "Staging directory missing"
        );

        // The WAL dir is separate in config
        let wal_dir = dir.path().join("wal").join(collection_name);
        assert!(wal_dir.exists(), "WAL directory missing");

        // 3. Verify Components are Live
        let index = &collection.index;

        // Check V2 specific components
        // KV should be accessible
        let test_key = 100u64.to_le_bytes();
        assert!(index.get_kv().get(&test_key).is_ok(), "KV store not active");

        // Check Dimensions
        assert_eq!(index.get_dim(), DIM);
    }

    #[tokio::test]
    async fn test_manager_recovers_state() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path());
        let data_dir = config.data_dir.clone();
        let manager = Arc::new(CollectionManager::new(config.clone()));
        let name = "recovery_test";

        // 1. Create and Modify
        {
            let coll = manager.get_or_create(name, None, None).await.unwrap();
            // Insert something to trigger WAL/KV creation
            coll.index.insert(1, &[0.0; DIM]).unwrap();
        } // Drop collection ref

        // 2. Re-Initialize (Simulate Restart)
        let manager2 = Arc::new(CollectionManager::new(config));
        let _coll2 = manager2.get_or_create(name, None, None).await.unwrap();

        // 3. Verify Persistence
        // The KV/WAL should still exist on disk.
        // Note: Actual data recovery is tested in core/server integration tests.
        // Here we just ensure the Manager doesn't crash or blow up the folders.

        let kv_path = data_dir.join(name).join("kv");
        assert!(kv_path.exists());
    }

    #[tokio::test]
    async fn test_dimension_mismatch_prevention() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path());
        let manager = Arc::new(CollectionManager::new(config));

        // 1. Create with Dim 128
        let _ = manager
            .get_or_create("dim_test", Some(128), Some(2000))
            .await
            .unwrap();

        // 2. Try to get with Dim 64 -> Should Fail
        let result = manager
            .get_or_create("dim_test", Some(64), Some(2000))
            .await;

        assert!(result.is_err());

        match result {
            Ok(_) => assert!(false, "should not get here"),
            Err(e) => assert!(e.to_string().contains("Dimension mismatch")),
        }
    }
}

#[cfg(test)]
mod load_and_unload_collection_test {
    use crate::config::{Config, FileConfig, StorageCommand};
    use crate::manager::CollectionManager;
    use std::sync::Arc;
    use tempfile::tempdir;

    fn test_config(root: &std::path::Path) -> Config {
        Config {
            port: 50051,
            wal_dir: root.join("wal"),
            data_dir: root.join("data"), // Explicit data root
            storage: StorageCommand::File(FileConfig {
                path: root.join("storage_unused"),
            }),
            default_dim: 128,
            max_bucket_capacity: 1000,
            ef_construction: 50,
            ef_search: 50,
        }
    }

    #[tokio::test]
    async fn test_manager_creates_directory_structure() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path());
        let manager = Arc::new(CollectionManager::new(config));

        let name = "test_struct";
        manager.get_or_create(name, None, None).await.unwrap();

        // Verify Structure:
        // root/data/test_struct/
        //   |-- staging/
        //   |-- kv/
        //   |-- manifest.pb
        // root/wal/test_struct/

        let coll_root = dir.path().join("data").join(name);
        assert!(coll_root.exists());
        assert!(coll_root.join("staging").exists());
        assert!(coll_root.join("kv").exists());

        // let err_log = format!("{:?}", coll_root.join("manifest.pb"));
        assert!(coll_root.join("manifest.pb").exists());

        let wal_root = dir.path().join("wal").join(name);
        assert!(wal_root.exists());
    }

    #[tokio::test]
    async fn test_manager_reloads_collection() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path());
        let name = "persist";

        // 1. Initial Run
        {
            let manager = Arc::new(CollectionManager::new(config.clone()));
            let coll = manager.get_or_create(name, None, None).await.unwrap();

            // Insert Data (Writes to WAL)
            coll.index.insert(1, &[0.0; 128]).unwrap();

            // Force WAL sync to ensure data is on disk before we drop
            // We lock the WAL and call sync manually
            // (Note: In production, OS page cache usually handles this on process exit,
            // but for unit tests with immediate reopen, explicit sync is safer).
            coll.index.get_wal().lock().current().sync().unwrap();
        }
        // Drop manager -> Closes files

        // 2. Re-open (Simulate Restart)
        let manager2 = Arc::new(CollectionManager::new(config));
        let coll2 = manager2.get_or_create(name, None, None).await.unwrap();

        // 3. Verify data via search (WAL replay should have restored ID 1)
        let res = coll2
            .index
            .search(&[0.0; 128], 1, 0.9, 1.0, 100.0)
            .await
            .unwrap();

        assert_eq!(res.len(), 1, "WAL Replay failed: Index is empty");
        assert_eq!(res[0].0, 1, "WAL Replay failed: ID mismatch");
    }
}

#[cfg(test)]
mod bucket_fetch_test {
    // use crate::bucket_file_writer::BucketFileWriter;
    // use crate::bucket_manager::{BucketManager, StorageClass};
    use drift_core::lock_manager::BucketCoordinator;
    use drift_core::quantizer::Quantizer;
    use drift_storage::{
        bucket_file_writer::BucketFileWriter,
        bucket_manager::{BucketManager, StorageClass},
    };
    use drift_traits::StorageEngine;
    use opendal::{Operator, services};
    use std::sync::Arc;
    use tempfile::tempdir;

    // --- Helpers ---

    fn create_local_operator(path: &std::path::Path) -> Operator {
        let builder = services::Fs::default().root(path.to_str().unwrap());
        Operator::new(builder).unwrap().finish()
    }

    /// Creates a valid .drift file with specific data
    async fn create_bucket_file(
        dir: &std::path::Path,
        filename: &str,
        ids: &[u64],
        vecs: &[Vec<f32>],
        dim: usize,
    ) {
        let path = dir.join(filename);
        let file = std::fs::File::create(&path).unwrap();

        let flat: Vec<f32> = vecs.iter().flatten().copied().collect();
        let q = Quantizer::train(&flat, dim);
        let mut writer = BucketFileWriter::new_streaming(file, [0u8; 16], q, dim).unwrap();
        writer.write_batch(ids, &flat).unwrap();
        writer.finalize().unwrap();
    }

    // --- TEST 1: Tiered Fetch (Base + Delta) ---
    #[tokio::test]
    async fn test_fetch_bucket_merges_tiered_storage() {
        let dir = tempdir().unwrap();
        let op = create_local_operator(dir.path());
        let coordinator = Arc::new(BucketCoordinator::new());
        let manager = BucketManager::new(op.clone(), op.clone(), 4, coordinator);

        let dim = 2;
        let bucket_id = 100;

        // 1. Create Remote Base (IDs 0-9)
        let base_ids: Vec<u64> = (0..10).collect();
        let base_vecs = vec![vec![1.0, 1.0]; 10];
        create_bucket_file(dir.path(), "base.drift", &base_ids, &base_vecs, dim).await;

        // 2. Create Local Delta (IDs 10-14)
        let delta_ids: Vec<u64> = (10..15).collect();
        let delta_vecs = vec![vec![2.0, 2.0]; 5];
        create_bucket_file(dir.path(), "delta.drift", &delta_ids, &delta_vecs, dim).await;

        // 3. Register Tiered
        manager.register_bucket_with_count(
            bucket_id,
            "ignored.drift".to_string(),
            StorageClass::Tiered {
                remote_path: "base.drift".to_string(),
                local_path: "delta.drift".to_string(),
            },
            15,
        );

        // 4. Fetch
        let (ids, flat_vecs) = manager.fetch_bucket(bucket_id).await.expect("Fetch failed");

        // 5. Verify Merge
        assert_eq!(ids.len(), 15, "Should have 15 total items");
        assert_eq!(flat_vecs.len(), 30, "Should have 30 floats (15 * 2)");

        // Check content
        // Base items
        assert!(ids.contains(&0));
        assert!(ids.contains(&9));
        // Delta items
        assert!(ids.contains(&10));
        assert!(ids.contains(&14));

        // Verify values (approx check due to quantization)
        // Find index of ID 0
        let idx_0 = ids.iter().position(|&x| x == 0).unwrap();
        assert!(
            (flat_vecs[idx_0 * 2] - 1.0).abs() < 0.05,
            "Base value mismatch"
        );

        // Find index of ID 10
        let idx_10 = ids.iter().position(|&x| x == 10).unwrap();
        assert!(
            (flat_vecs[idx_10 * 2] - 2.0).abs() < 0.05,
            "Delta value mismatch"
        );
    }

    // --- TEST 2: Promoting Fetch (Active + Frozen + Remote) ---
    #[tokio::test]
    async fn test_fetch_bucket_merges_promoting_storage() {
        let dir = tempdir().unwrap();
        let op = create_local_operator(dir.path());
        let coordinator = Arc::new(BucketCoordinator::new());
        let manager = BucketManager::new(op.clone(), op.clone(), 4, coordinator);
        let dim = 2;
        let bucket_id = 200;

        // 1. Remote Base (ID 1)
        create_bucket_file(dir.path(), "remote.drift", &[1], &[vec![1.0; dim]], dim).await;

        // 2. Local Frozen (ID 2)
        create_bucket_file(dir.path(), "frozen.drift", &[2], &[vec![2.0; dim]], dim).await;

        // 3. Local Active (ID 3)
        create_bucket_file(dir.path(), "active.drift", &[3], &[vec![3.0; dim]], dim).await;

        // 4. Register Promoting
        manager.register_bucket(
            bucket_id,
            "active.drift".to_string(),
            StorageClass::Promoting {
                local_active: "active.drift".to_string(),
                local_frozen: "frozen.drift".to_string(),
                remote_path: Some("remote.drift".to_string()),
            },
        );

        // 5. Fetch
        let (ids, _) = manager.fetch_bucket(bucket_id).await.expect("Fetch failed");

        // 6. Verify Full Merge
        assert_eq!(ids.len(), 3);
        assert!(ids.contains(&1), "Missing Remote");
        assert!(ids.contains(&2), "Missing Frozen");
        assert!(ids.contains(&3), "Missing Active");
    }

    // --- TEST 3: Missing Component Failure ---
    // --- TEST 3: Missing Component Failure ---
    #[tokio::test]
    async fn test_fetch_bucket_returns_empty_on_missing_files() {
        let dir = tempdir().unwrap();
        let op = create_local_operator(dir.path());
        let coordinator = Arc::new(BucketCoordinator::new());
        let manager = BucketManager::new(op.clone(), op.clone(), 4, coordinator);

        // Register bucket pointing to non-existent file
        manager.register_bucket(999, "ghost.drift".to_string(), StorageClass::Local);

        // Fetch should NOT fail. It should return empty data.
        let res = manager.fetch_bucket(999).await;

        assert!(
            res.is_ok(),
            "Should not error on missing file (treated as empty)"
        );

        let (ids, vecs) = res.unwrap();
        assert!(ids.is_empty(), "IDs should be empty");
        assert!(vecs.is_empty(), "Vectors should be empty");
    }
}
