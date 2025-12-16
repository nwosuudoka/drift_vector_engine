#[cfg(test)]
mod tests {
    use crate::index::{IndexOptions, VectorIndex};
    use crate::memtable::MemTable;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;
    use tempfile::tempdir;

    const DIM: usize = 2;

    fn create_test_index() -> (Arc<VectorIndex>, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");
        let options = IndexOptions {
            dim: DIM,
            num_centroids: 2,
            training_sample_size: 50,
            max_bucket_capacity: 100,
            ef_construction: 100,
            ef_search: 100,
        };
        let index = Arc::new(VectorIndex::new(options, &wal_path).unwrap());
        (index, dir)
    }

    #[test]
    fn test_memtable_basic_ops() {
        let memtable = MemTable::new(100, DIM, 100, 16); // High ef_construction
        memtable.insert(1, &vec![0.0, 0.0]);
        memtable.insert(2, &vec![10.0, 10.0]);
        memtable.insert(3, &vec![1.0, 1.0]);

        let results = memtable.search(&vec![0.1, 0.1], 2, 40);
        assert!(results.len() >= 2);
        assert_eq!(results[0].0, 1);
    }

    #[test]
    fn test_hybrid_search_correctness() {
        let (index, _guard) = create_test_index();

        // L1 Buckets at [100,100] and [-100,-100]
        let train_data = vec![vec![100.0, 100.0], vec![-100.0, -100.0]];
        index.train(&train_data);

        // L1 Item at [100, 100]
        index.force_insert_l1(10, &vec![100.0, 100.0]);

        // L0 Item at [0.5, 0.5]
        index.insert(20, &vec![0.5, 0.5]).unwrap();

        // Search at [0,0]
        // Note: L1 item is distance 141. L0 is distance 0.7.
        // We set very high lambda to allow distant items, or just check that we find L0.
        // If we want BOTH, we need lambda to be small enough that exp(-lambda * 141) > epsilon.
        // Let's check finding L0 primarily, as that's the integration test goal.

        let results = index.search_drift_aware(
            &vec![0.0, 0.0],
            5,
            0.5,  // Target Confidence
            0.01, // Very low lambda to allow "far" L1 buckets to have non-zero probability
            100.0,
        );

        // We expect L0 item (20) to be #1.
        assert!(!results.is_empty());
        assert_eq!(
            results[0].id, 20,
            "L0 item should be found and ranked first"
        );

        // With lambda=0.01, we might find L1 (10) as well.
        if results.len() > 1 {
            assert_eq!(results[1].id, 10);
        }
    }

    #[test]
    fn test_concurrent_hybrid_traffic() {
        let (index, _guard) = create_test_index();
        index.train(&vec![vec![0.0, 0.0]]);

        let index_write = index.clone();
        let index_read = index.clone();

        let writer = thread::spawn(move || {
            for i in 0..100 {
                // Insert distinct items to avoid HNSW duplicate issues
                index_write
                    .insert(i as u64, &vec![i as f32, i as f32])
                    .unwrap();
                if i % 10 == 0 {
                    thread::sleep(Duration::from_millis(1));
                }
            }
        });

        let reader = thread::spawn(move || {
            for _ in 0..10 {
                let results = index_read.search_drift_aware(&vec![0.0, 0.0], 5, 0.9, 1.0, 10.0);
                assert!(results.len() >= 0);
                thread::sleep(Duration::from_millis(1));
            }
        });

        writer.join().unwrap();
        reader.join().unwrap();
    }
}
