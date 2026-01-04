#[cfg(test)]
mod tests {
    use crate::index::{IndexOptions, VectorIndex};
    use crate::memtable::MemTable;
    use crate::memtable::MemTableSnapshot;
    use drift_cache::local_store::LocalDiskManager;
    use std::sync::Arc;
    use std::time::Duration;
    use tempfile::TempDir;
    use tokio::time::sleep;

    const DIM: usize = 2;

    // Helper to create an Async Index
    async fn create_test_index() -> (Arc<VectorIndex>, TempDir) {
        let dir = TempDir::new().unwrap();
        let wal_path = dir.path().join("test.wal");
        let storage_path = dir.path().join("storage");
        std::fs::create_dir(&storage_path).unwrap();

        let options = IndexOptions {
            dim: DIM,
            num_centroids: 2,
            training_sample_size: 50,
            max_bucket_capacity: 100,
            ef_construction: 100,
            ef_search: 100,
        };

        let storage = Arc::new(LocalDiskManager::new(storage_path));
        let index = Arc::new(VectorIndex::new(options, &wal_path, storage).unwrap());
        (index, dir)
    }

    #[test]
    fn test_memtable_basic_ops() {
        // Unit test for MemTable logic (Synchronous part is fine here)
        let memtable = MemTable::new(100, DIM, 100, 16);

        memtable.insert(1, &vec![0.0, 0.0]);
        memtable.insert(2, &vec![10.0, 10.0]);
        memtable.insert(3, &vec![1.0, 1.0]);

        // Search near [0,0]
        let results = memtable.search(&vec![0.1, 0.1], 2);
        assert!(results.len() >= 2);

        // Closest should be ID 1 (dist ~0.02) or ID 3 (dist ~1.62)
        // ID 2 is at dist ~196.
        assert_eq!(results[0].0, 1);

        // Test Delete
        memtable.delete(1);
        let results_after = memtable.search(&vec![0.1, 0.1], 2);
        assert!(
            !results_after.iter().any(|(id, _)| *id == 1),
            "Deleted item 1 found"
        );
    }

    #[tokio::test]
    async fn test_hybrid_search_correctness() {
        let (index, _guard) = create_test_index().await;

        // 1. Train L1 (Disk) with buckets at [100,100] and [-100,-100]
        let train_data = vec![vec![100.0, 100.0], vec![-100.0, -100.0]];
        index.train(&train_data).await.unwrap();

        // 2. Insert L1 Item (Async) at [100, 100]
        index
            .force_register_bucket_with_ids(0, &[10], &[vec![100.0, 100.0]])
            .await
            .unwrap();

        // 3. Insert L0 Item (Sync/MemTable) at [0.5, 0.5]
        // This goes into the MemTable via the WAL.
        index.insert(20, &vec![0.5, 0.5]).unwrap();

        // 4. Search at [0,0]
        // L0 item (20) is dist ~0.5. L1 item (10) is dist ~141.
        // We use a very low lambda to ensure the far L1 item is even considered/scored.
        let results = index
            .search_async(
                &vec![0.0, 0.0],
                5,
                0.5,   // Target Confidence
                0.01,  // Low Lambda -> decay is slow -> far items have score > 0
                100.0, // Tau
            )
            .await
            .unwrap();

        assert!(!results.is_empty());

        // Check ranking
        assert_eq!(results[0].id, 20, "L0 item (20) should be closest");

        // Check if L1 was found (depends on lambda/k)
        if results.len() > 1 {
            assert!(
                results.iter().any(|r| r.id == 10),
                "L1 item (10) missing from hybrid results"
            );
        }
    }

    #[tokio::test]
    async fn test_concurrent_hybrid_traffic() {
        let (index, _guard) = create_test_index().await;

        // Setup base index
        index.train(&vec![vec![0.0, 0.0]]).await.unwrap();

        let index_write = index.clone();
        let index_read = index.clone();

        // Spawn Writer Task
        let writer = tokio::spawn(async move {
            for i in 0..100 {
                // Insert distinct items to avoid HNSW duplicate issues
                let vec = vec![i as f32, i as f32];
                // Blocking insert is fast, but we wrap it block_in_place if needed,
                // or just call it since it takes a RwLock (not async mutex).
                // Ideally, in an async context, you'd use `spawn_blocking`.
                // For this test, calling it directly is acceptable as it's purely memory ops.
                index_write.insert(i as u64, &vec).unwrap();

                if i % 10 == 0 {
                    sleep(Duration::from_millis(1)).await;
                }
            }
        });

        // Spawn Reader Task
        let reader = tokio::spawn(async move {
            for _ in 0..10 {
                let results = index_read
                    .search_async(&vec![0.0, 0.0], 5, 0.9, 1.0, 10.0)
                    .await
                    .unwrap();

                // Assert we found something or nothing, but didn't crash
                assert!(results.len() > 0);
                sleep(Duration::from_millis(2)).await;
            }
        });

        let _ = tokio::join!(writer, reader);
    }

    #[test]
    fn test_memtable_flat_buffer_alignment() {
        const DIM: usize = 4;
        let memtable = MemTable::new(10, DIM, 0, 0);

        // Insert recognizable patterns
        memtable.insert(1, &[1.0, 1.1, 1.2, 1.3]);
        memtable.insert(2, &[2.0, 2.1, 2.2, 2.3]);

        assert_eq!(memtable.len(), 2);

        // Verify internal search finds exact matches
        let results = memtable.search(&[1.0, 1.1, 1.2, 1.3], 1);
        assert_eq!(results[0].0, 1);
        assert!(results[0].1 < 1e-6);
    }

    #[test]
    fn test_snapshot_zero_copy_integrity() {
        const DIM: usize = 4;
        let memtable = MemTable::new(100, DIM, 0, 0);
        for i in 0..50 {
            memtable.insert(i as u64, &[i as f32; DIM]);
        }

        // ⚡ Perform the hardware-native freeze
        let snapshot = memtable.freeze_snapshot();

        // 1. Verify IDs are preserved
        assert_eq!(snapshot.ids.len(), 50);
        assert_eq!(snapshot.ids[10], 10);

        // 2. Verify Vectors are contiguous (Offset = Index * DIM)
        assert_eq!(snapshot.vectors[10 * DIM], 10.0);
        assert_eq!(snapshot.vectors[10 * DIM + (DIM - 1)], 10.0);

        // 3. Verify search on immutable snapshot
        let query = vec![25.0; DIM];
        let hits = snapshot.search(&query, 1);
        assert_eq!(hits[0].0, 25);
    }
}
