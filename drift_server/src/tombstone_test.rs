#[cfg(test)]
mod tests {
    use crate::persistence::PersistenceManager;
    use drift_cache::local_store::LocalDiskManager;
    use drift_core::index::{IndexOptions, VectorIndex};
    use opendal::{Operator, services};
    use std::sync::Arc;
    use tempfile::tempdir;

    fn create_index_with_storage(dir: &std::path::Path, wal_name: &str) -> Arc<VectorIndex> {
        let wal_path = dir.join(wal_name);
        let storage_path = dir.join("storage");
        std::fs::create_dir_all(&storage_path).unwrap();
        let storage = Arc::new(LocalDiskManager::new(storage_path));
        let options = IndexOptions {
            dim: 2,
            num_centroids: 2,
            training_sample_size: 50,
            max_bucket_capacity: 100,
            ef_construction: 100,
            ef_search: 100,
        };
        Arc::new(VectorIndex::new(options, &wal_path, storage).unwrap())
    }

    // Helper to create a local FS operator
    fn create_local_operator(path: &std::path::Path) -> Operator {
        let mut builder = services::Fs::default();
        builder = builder.root(path.to_str().unwrap());
        Operator::new(builder).unwrap().finish()
    }

    #[tokio::test]
    async fn test_tombstone_persistence_prevents_resurrection() {
        let dir = tempdir().unwrap();

        // ⚡ CHANGE: Create Operator and inject
        let op = create_local_operator(dir.path());
        let persistence = PersistenceManager::new(op, dir.path());

        // 1. Create Index
        let index = create_index_with_storage(dir.path(), "test.wal");

        // Train implicitly inserts ID 0 (Training Data)
        index.train(&vec![vec![0.0, 0.0]]).await.unwrap();

        // Insert ID 1 (User Data)
        index.insert(1, &vec![0.0, 0.0]).unwrap();

        // 2. Delete ID 1
        index.delete(1).unwrap();

        // Verify ID 1 is gone in RAM
        let res = index
            .search_async(&vec![0.0, 0.0], 10, 0.9, 1.0, 100.0)
            .await
            .unwrap();

        // ⚡ FIX: We assert that ID 1 is missing. We do NOT check is_empty() because ID 0 exists.
        assert!(
            !res.iter().any(|r| r.id == 1),
            "ID 1 should be deleted in RAM check"
        );

        // 3. Flush (Writes Segment + Tombstones)
        persistence
            .flush_memtable_to_segment(&[], &index, "run_1")
            .await
            .unwrap();

        // Manually flush tombstones (Simulating Janitor)
        let deleted: Vec<u64> = index.deleted_ids.read().iter().cloned().collect();
        persistence
            .flush_tombstones(&deleted, "run_1")
            .await
            .unwrap();

        // 4. Restart (Simulate Crash/Restart)
        drop(index);

        // Load Segment
        // ⚡ CHANGE: Use relative object key string, not PathBuf
        let index_2 = persistence
            .load_from_segment("segment_l0_run_1.drift")
            .await
            .unwrap();

        // Load Tombstones
        let loaded_deletes = persistence.load_all_tombstones().await.unwrap();
        index_2.deleted_ids.write().extend(loaded_deletes);

        // 5. Verify ID 1 is STILL gone
        let res_2 = index_2
            .search_async(&vec![0.0, 0.0], 10, 0.9, 1.0, 100.0)
            .await
            .unwrap();

        assert!(
            !res_2.iter().any(|r| r.id == 1),
            "Zombie resurrected! ID 1 was found after restart."
        );
    }
}
