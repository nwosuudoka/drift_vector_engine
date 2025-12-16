#[cfg(test)]
mod tests {
    use crate::index::{IndexOptions, VectorIndex};
    use crossbeam_epoch::{self as epoch};
    use std::sync::atomic::Ordering;
    use tempfile::{TempDir, tempdir};

    fn create_test_index(dim: usize, num_centroids: usize) -> (VectorIndex, TempDir) {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        let options = IndexOptions {
            dim,
            num_centroids,
            training_sample_size: 50,
            max_bucket_capacity: 100,
            // High parameters for test stability on small N
            ef_construction: 100,
            ef_search: 100,
        };

        let index = VectorIndex::new(options, &wal_path).unwrap();
        (index, dir)
    }

    #[test]
    fn test_scatter_merge_logic() {
        let (index, _guard) = create_test_index(2, 2);

        // Train 2 clusters
        let train_data = vec![vec![0.0, 0.0], vec![100.0, 100.0]];
        index.train(&train_data);

        // Identify Bucket 0 (Near [0,0])
        let guard = epoch::pin();
        let buckets = unsafe { index.buckets.load(Ordering::Acquire, &guard).as_ref() }.unwrap();
        let (id_0, _) = buckets
            .iter()
            .find(|(_, b)| b.centroid.read()[0] < 50.0)
            .map(|(id, b)| (*id, b.clone()))
            .unwrap();
        drop(guard);

        // 1. Insert Resident (Goes to L0)
        index.insert(999, &vec![1.0, 1.0]).unwrap();

        // 2. Insert Bridge Node (Crucial for HNSW connectivity)
        index.insert(555, &vec![50.0, 50.0]).unwrap();

        // 3. Manual Injection of "Drifter" into L1 Bucket 0
        {
            let guard = epoch::pin();
            let buckets =
                unsafe { index.buckets.load(Ordering::Acquire, &guard).as_ref() }.unwrap();
            let b0 = buckets.get(&id_0).unwrap();
            let q = b0.quantizer.clone();
            b0.insert(888, &q.encode(&vec![99.0, 99.0]));
        }

        // 4. Trigger Scatter Merge (Moves 888 from L1 to L0)
        index.scatter_merge(id_0);

        // 5. Verify Search
        let results = index.search_drift_aware(&vec![100.0, 100.0], 5, 0.5, 1.0, 100.0);

        assert!(
            results.iter().any(|r| r.id == 888),
            "Drifting vector 888 lost (Should be in L0)"
        );
    }

    #[test]
    fn test_drift_aware_routing_logic() {
        let (index, _guard) = create_test_index(2, 2);

        // Use distinct data to ensure valid Quantizer
        let training: Vec<Vec<f32>> = (0..50).map(|i| vec![i as f32, i as f32]).collect();
        index.train(&training);

        // Insert DISTINCT vectors
        index.insert(1, &vec![0.0, 0.0]).unwrap();
        index.insert(2, &vec![0.1, 0.1]).unwrap();

        let res = index.search_drift_aware(&vec![0.0, 0.0], 5, 0.99, 1.0, 100.0);

        assert!(res.len() >= 2, "Should find both L0 items");
    }

    #[test]
    fn test_saturating_density_stopping() {
        let (index, _guard) = create_test_index(2, 2);
        let train_data = vec![vec![0.0, 0.0], vec![10.0, 10.0]];
        index.train(&train_data);
        // L1 injection
        index.force_insert_l1(1, &vec![10.0, 10.0]);
        // Search
        let results = index.search_drift_aware(&vec![0.0, 0.0], 5, 0.5, 1.0, 10.0);
        assert!(!results.is_empty());
        assert_eq!(results[0].id, 1);
    }

    #[test]
    fn test_index_split_lifecycle() {
        let (index, _guard) = create_test_index(2, 1);
        let training: Vec<Vec<f32>> = (0..50).map(|_| vec![0.0; 2]).collect();
        index.train(&training);
        for i in 0..50 {
            index.force_insert_l1(i, &vec![10.0, 10.0]);
        }
        for i in 50..100 {
            index.force_insert_l1(i, &vec![-10.0, -10.0]);
        }

        index.split_bucket(0);

        let res_a = index.search_drift_aware(&vec![10.0, 10.0], 5, 0.95, 25.0, 100.0);
        assert!(!res_a.is_empty());
    }

    #[test]
    fn test_split_and_steal_atomic() {
        let (index, _guard) = create_test_index(2, 2);
        let train = vec![vec![-10.0, 0.0], vec![10.0, 0.0]];
        index.train(&train);

        let (left_id, right_id) = {
            let guard = epoch::pin();
            let centroids =
                unsafe { index.centroids.load(Ordering::Acquire, &guard).as_ref() }.unwrap();
            (
                centroids.iter().find(|c| c.vector[0] < 0.0).unwrap().id,
                centroids.iter().find(|c| c.vector[0] > 0.0).unwrap().id,
            )
        };

        let q = index.get_quantizer().unwrap();
        let defector_code = q.encode(&vec![-9.0, 0.0]);
        {
            let guard = epoch::pin();
            let buckets =
                unsafe { index.buckets.load(Ordering::Acquire, &guard).as_ref() }.unwrap();
            let b_right = buckets.get(&right_id).unwrap();
            b_right.insert(777, &defector_code);
        }

        for i in 0..30 {
            index.force_insert_l1(i, &vec![-10.0, 0.0]);
        }

        index.split_and_steal(left_id);

        {
            let guard = epoch::pin();
            let buckets =
                unsafe { index.buckets.load(Ordering::Acquire, &guard).as_ref() }.unwrap();
            let b_right = buckets.get(&right_id).unwrap();
            let data = b_right.data.read();
            let idx = data.vids.iter().position(|&x| x == 777).unwrap();
            assert!(data.tombstones.contains(idx));
        }
    }
}
