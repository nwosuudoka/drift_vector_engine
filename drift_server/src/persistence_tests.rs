#[cfg(test)]
mod tests {
    use crate::persistence::PersistenceManager;
    use drift_cache::local_store::LocalDiskManager;
    use drift_core::index::{IndexOptions, VectorIndex};
    use std::sync::Arc;
    use tempfile::tempdir;

    // Helper to create an index with storage
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

    #[tokio::test]
    async fn test_end_to_end_persistence_lifecycle() {
        let dir = tempdir().unwrap();
        let persistence = PersistenceManager::new(dir.path());

        let index_original = create_index_with_storage(dir.path(), "current.wal");

        // Train
        let train_data = vec![vec![10.0, 10.0], vec![-10.0, -10.0]];
        index_original.train(&train_data).await.unwrap();

        // Insert Data to L0 (MemTable) -> Persisted via WAL
        for i in 0..50 {
            index_original.insert(i, &vec![10.0, 10.0]).unwrap();
        }
        for i in 50..100 {
            index_original.insert(i, &vec![-10.0, -10.0]).unwrap();
        }

        // Verify state before "Crash"
        let pre_crash = index_original
            .search_async(&vec![10.0, 10.0], 1, 0.9, 1.0, 10.0)
            .await
            .unwrap();
        assert!(!pre_crash.is_empty());

        // Snapshot L1 (buckets) to disk
        // Note: This takes existing L1 buckets (from train) and writes them.
        // L0 remains in WAL.
        let segment_path = persistence
            .flush_to_segment(&index_original, "run_1")
            .await
            .expect("Flush failed");

        // "Crash"
        drop(index_original);

        // Recover
        let index_recovered = persistence
            .load_from_segment(&segment_path)
            .await
            .expect("Load failed");

        // Verify L0 Recovery (from WAL)
        let results_a = index_recovered
            .search_async(&vec![10.0, 10.0], 5, 0.9, 1.0, 10.0)
            .await
            .unwrap();

        assert_eq!(
            results_a.len(),
            5,
            "Failed to recover L0 data for Cluster A"
        );

        let results_b = index_recovered
            .search_async(&vec![-10.0, -10.0], 5, 0.9, 1.0, 10.0)
            .await
            .unwrap();

        assert_eq!(
            results_b.len(),
            5,
            "Failed to recover L0 data for Cluster B"
        );

        // Verify Data Fidelity
        let top_dist = results_a[0].distance;
        assert!(
            top_dist < 0.02,
            "Recovered distance too high: {}. Expected near 0.0",
            top_dist
        );

        println!("Persistence Lifecycle Passed!");
    }
}
