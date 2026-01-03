#[cfg(test)]
mod tests {
    use crate::persistence::PersistenceManager;
    use drift_cache::local_store::LocalDiskManager;
    use drift_core::index::{IndexOptions, VectorIndex};
    use std::fs::OpenOptions;
    use std::io::Write;
    use std::sync::Arc;
    use tempfile::tempdir;

    fn make_options() -> IndexOptions {
        IndexOptions {
            dim: 2,
            num_centroids: 1,
            training_sample_size: 50,
            max_bucket_capacity: 100,
            ef_construction: 100,
            ef_search: 100,
        }
    }

    fn create_index(dir: &std::path::Path, wal_name: &str) -> Arc<VectorIndex> {
        let wal_path = dir.join(wal_name);
        let storage_path = dir.join("storage");
        std::fs::create_dir_all(&storage_path).unwrap();
        let storage = Arc::new(LocalDiskManager::new(storage_path));
        Arc::new(VectorIndex::new(make_options(), &wal_path, storage).unwrap())
    }

    #[tokio::test]
    async fn test_recovery_with_corrupt_wal_tail() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("current.wal");

        {
            let index = create_index(dir.path(), "current.wal");
            index
                .train(&vec![vec![0.0, 0.0], vec![100.0, 100.0]])
                .await
                .unwrap();
            index.insert(1, &vec![1.0, 1.0]).unwrap();
            index.insert(2, &vec![2.0, 2.0]).unwrap();
        }

        // Corrupt the WAL
        {
            let mut file = OpenOptions::new().append(true).open(&wal_path).unwrap();
            file.write_all(&[0xDE, 0xAD, 0xBE, 0xEF]).unwrap();
        }

        // Recover
        let index_recovered = create_index(dir.path(), "current.wal");

        // Should still work
        let res = index_recovered.insert(3, &vec![3.0, 3.0]);
        assert!(res.is_ok());

        let search = index_recovered
            .search_async(&vec![1.0, 1.0], 1, 0.9, 1.0, 100.0)
            .await
            .unwrap();
        assert!(!search.is_empty());
    }

    #[tokio::test]
    async fn test_persistence_manager_loads_l1_and_wal() {
        let dir = tempdir().unwrap();
        let persistence = PersistenceManager::new(dir.path());

        // 1. Setup L1 Data
        let index_l1 = create_index(dir.path(), "current.wal");
        index_l1
            .train(&vec![vec![0.0, 0.0], vec![100.0, 100.0]])
            .await
            .unwrap();

        // Insert L1 Data (using force_register with explicit ID 0)
        index_l1
            .force_register_bucket_with_ids(0, &[100], &[vec![10.0, 10.0]])
            .await
            .unwrap();

        let seg_path = persistence
            .flush_memtable_to_segment(&index_l1, "test")
            .await
            .unwrap();
        drop(index_l1);

        // 2. Setup L0 Data (Separate Index run on same WAL)
        let index_l0 = create_index(dir.path(), "current.wal");
        index_l0.insert(200, &vec![20.0, 20.0]).unwrap();
        drop(index_l0);

        // 3. Load Combined (Simulate startup via persistence)
        // Note: load_from_segment creates its own index.
        // Real startup would use CollectionManager to hydrate.
        // For this test, we load the segment into a new index.
        let loaded = persistence.load_from_segment(&seg_path).await.unwrap();

        // Since load_from_segment initializes a new index pointing to "current.wal",
        // it should replay L0 (ID 200).
        // And it loads L1 from the segment (ID 100).

        let res = loaded
            .search_async(&vec![0.0, 0.0], 10, 0.9, 1.0, 100.0)
            .await
            .unwrap();
        println!("Search Results: {:?}", res);

        let found_l1 = res.iter().any(|r| r.id == 100);
        let found_l0 = res.iter().any(|r| r.id == 200);

        assert!(found_l1, "Missed L1 data (ID 100)");
        assert!(found_l0, "Missed L0 data from WAL (ID 200)");
    }
}
