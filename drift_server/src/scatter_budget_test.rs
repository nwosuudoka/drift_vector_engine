#[cfg(test)]
mod tests {
    use drift_cache::local_store::LocalDiskManager;
    use drift_core::index::{IndexOptions, MaintenanceStatus, VectorIndex};
    use std::sync::Arc;
    use tempfile::tempdir;

    // --- Setup ---
    async fn setup_index(dir: &std::path::Path) -> Arc<VectorIndex> {
        let wal_path = dir.join("test.wal");
        let storage_path = dir.join("storage");
        std::fs::create_dir_all(&storage_path).unwrap();

        let storage = Arc::new(LocalDiskManager::new(storage_path));
        let options = IndexOptions {
            dim: 2,
            num_centroids: 2,
            training_sample_size: 10,
            max_bucket_capacity: 100,
            ef_construction: 10,
            ef_search: 10,
        };

        let index = Arc::new(VectorIndex::new(options, &wal_path, storage).unwrap());

        // Train initializes the Quantizer
        let train_data = vec![vec![0.0, 0.0], vec![100.0, 100.0]];
        index.train(&train_data).await.unwrap();

        index
    }

    async fn flush_to_bucket(index: &Arc<VectorIndex>, bucket_id: u32, vectors: &[Vec<f32>]) {
        let ids: Vec<u64> = (0..vectors.len()).map(|i| i as u64).collect();
        index
            .force_register_bucket_with_ids(bucket_id, &ids, vectors)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn test_scatter_merge_respects_budget() {
        let dir = tempdir().unwrap();
        let index = setup_index(dir.path()).await;

        // 1. Create a "Small Zombie" Bucket (ID 0)
        // 10 items. This is under the budget of 50.
        let small_zombie = vec![vec![0.0, 0.0]; 10];
        flush_to_bucket(&index, 0, &small_zombie).await;

        // 2. Create a Neighbor Bucket (ID 1)
        let neighbor = vec![vec![0.0, 0.0]; 10];
        flush_to_bucket(&index, 1, &neighbor).await;

        // 3. Attempt Merge on Small Zombie
        // Should succeed because 10 < 50
        let status_small = index.scatter_merge(0).await.unwrap();
        assert_eq!(
            status_small,
            MaintenanceStatus::Completed,
            "Small zombie should merge"
        );

        // 4. Create a "Large Zombie" Bucket (ID 2)
        // 60 items. This is OVER the budget of 50.
        let large_zombie = vec![vec![0.0, 0.0]; 60];
        flush_to_bucket(&index, 2, &large_zombie).await;

        // 5. Attempt Merge on Large Zombie
        // Should fail/skip because 60 > 50
        let status_large = index.scatter_merge(2).await.unwrap();
        assert_eq!(
            status_large,
            MaintenanceStatus::SkippedTooSmall,
            "Large zombie should be skipped due to budget"
        );

        // Verify Bucket 2 still exists
        let headers = index.get_all_bucket_headers();
        assert!(
            headers.iter().any(|h| h.id == 2),
            "Large zombie should not be deleted"
        );
    }
}
