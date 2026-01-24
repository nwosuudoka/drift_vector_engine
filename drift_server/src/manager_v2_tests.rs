#[cfg(test)]
mod tests {
    use crate::config::{Config, FileConfig, StorageCommand};
    use crate::manager_v2::CollectionManager;
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
            .get_or_create(collection_name, None)
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
            let coll = manager.get_or_create(name, None).await.unwrap();
            // Insert something to trigger WAL/KV creation
            coll.index.insert(1, &[0.0; DIM]).unwrap();
        } // Drop collection ref

        // 2. Re-Initialize (Simulate Restart)
        let manager2 = Arc::new(CollectionManager::new(config));
        let _coll2 = manager2.get_or_create(name, None).await.unwrap();

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
        let _ = manager.get_or_create("dim_test", Some(128)).await.unwrap();

        // 2. Try to get with Dim 64 -> Should Fail
        let result = manager.get_or_create("dim_test", Some(64)).await;

        assert!(result.is_err());

        match result {
            Ok(_) => assert!(false, "should not get here"),
            Err(e) => assert!(e.to_string().contains("Dimension mismatch")),
        }
    }
}

#[cfg(test)]
mod more_tests {
    use crate::config::{Config, FileConfig, StorageCommand};
    use crate::manager_v2::CollectionManager;
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
        manager.get_or_create(name, None).await.unwrap();

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
            let coll = manager.get_or_create(name, None).await.unwrap();

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
        let coll2 = manager2.get_or_create(name, None).await.unwrap();

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
