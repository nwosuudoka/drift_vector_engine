#[cfg(test)]
mod tests {
    use crate::persistence::PersistenceManager;
    use drift_cache::local_store::LocalDiskManager;
    use drift_core::index::{IndexOptions, VectorIndex};
    use opendal::{Operator, services};
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
        // ⚡ CHANGE: Build the Operator manually for the test
        let mut builder = services::Fs::default();
        builder = builder.root(dir.path().to_str().unwrap());
        let op = Operator::new(builder).unwrap().finish();

        // ⚡ CHANGE: Inject Operator into Manager
        let persistence = PersistenceManager::new(op, dir.path());
        let index_original = create_index_with_storage(dir.path(), "current.wal");

        // Train
        let train_data = vec![vec![10.0, 10.0], vec![-10.0, -10.0]];
        index_original.train(&train_data).await.unwrap();

        // 1. Snapshot L1 (Partitioned Flush)
        // We manually trigger the calculation to get partitions
        let ids: Vec<u64> = (0..50).collect();
        let vectors: Vec<Vec<f32>> = (0..50).map(|_| vec![10.0, 10.0]).collect();

        let partitions = index_original
            .calculate_partitions(&ids, &vectors)
            .await
            .unwrap();

        // Write Segment
        let (run_id, _locations) = persistence
            .write_partitioned_segment(&partitions, &index_original)
            .await
            .expect("Flush failed");

        let segment_key = format!("segment_{}.drift", run_id);

        // "Crash"
        drop(index_original);

        // 2. Recover
        let index_recovered = persistence
            .load_from_segment(&segment_key)
            .await
            .expect("Load failed");

        // 3. Verify L1 Recovery
        // We wrote 50 items at [10.0, 10.0]. They should be searchable.
        let results = index_recovered
            .search_async(&vec![10.0, 10.0], 5, 0.9, 1.0, 10.0)
            .await
            .unwrap();

        assert_eq!(results.len(), 5, "Failed to recover L1 data from segment");

        // Verify Data Fidelity
        let top_dist = results[0].distance;
        assert!(
            top_dist < 0.02,
            "Recovered distance too high: {}. Expected near 0.0",
            top_dist
        );

        println!("Persistence Lifecycle Passed!");
    }

    #[tokio::test]
    async fn test_flush_memtable_l0_persistence() {
        let dir = tempdir().unwrap();
        let mut builder = services::Fs::default();
        builder = builder.root(dir.path().to_str().unwrap());
        let op = Operator::new(builder).unwrap().finish();

        let persistence = PersistenceManager::new(op, dir.path());
        let index = create_index_with_storage(dir.path(), "l0.wal");

        index.train(&vec![vec![1.0, 1.0]]).await.unwrap();

        let data = vec![(100, vec![1.0, 1.0]), (200, vec![2.0, 2.0])];

        // Write L0 Segment
        let run_id = "test_l0";
        let file_name = persistence
            .flush_memtable_to_segment(&data, &index, run_id)
            .await
            .unwrap();

        // Load Back
        let loaded_index = persistence.load_from_segment(&file_name).await.unwrap();

        // Verify
        let res = loaded_index
            .search_async(&vec![1.0, 1.0], 1, 0.9, 1.0, 10.0)
            .await
            .unwrap();
        assert_eq!(res[0].id, 100);
    }
}
