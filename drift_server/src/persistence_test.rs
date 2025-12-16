#[cfg(test)]
mod tests {
    use crate::persistence::PersistenceManager;
    use drift_core::index::{IndexOptions, VectorIndex};
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_end_to_end_persistence_lifecycle() {
        let dir = tempdir().unwrap();
        let persistence = PersistenceManager::new(dir.path());
        let wal_path = dir.path().join("current.wal");

        let options = IndexOptions {
            dim: 2,
            num_centroids: 2,
            training_sample_size: 50,
            max_bucket_capacity: 100,
            ef_construction: 100,
            ef_search: 100,
        };

        let index_original = VectorIndex::new(options, &wal_path).unwrap();

        // Train
        let train_data = vec![vec![10.0, 10.0], vec![-10.0, -10.0]];
        index_original.train(&train_data);

        // Insert Data to L0 (MemTable) -> Persisted via WAL
        for i in 0..50 {
            index_original.insert(i, &vec![10.0, 10.0]).unwrap();
        }
        for i in 50..100 {
            index_original.insert(i, &vec![-10.0, -10.0]).unwrap();
        }

        // Verify state before "Crash"
        let pre_crash = index_original.search_drift_aware(&vec![10.0, 10.0], 1, 0.9, 1.0, 10.0);
        assert!(!pre_crash.is_empty());

        // Snapshot L1 (buckets) to disk
        // Note: This does NOT flush L0 to L1. L0 remains in WAL.
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
        let results_a = index_recovered.search_drift_aware(&vec![10.0, 10.0], 5, 0.9, 1.0, 10.0);
        assert_eq!(
            results_a.len(),
            5,
            "Failed to recover L0 data for Cluster A"
        );

        let results_b = index_recovered.search_drift_aware(&vec![-10.0, -10.0], 5, 0.9, 1.0, 10.0);
        assert_eq!(
            results_b.len(),
            5,
            "Failed to recover L0 data for Cluster B"
        );

        // Verify Data Fidelity
        // We relax tolerance slightly for ANN behavior after rebuild
        let top_dist = results_a[0].distance;
        assert!(
            top_dist < 0.02,
            "Recovered distance too high: {}. Expected near 0.0",
            top_dist
        );

        println!("Persistence Lifecycle Passed!");
    }
}
