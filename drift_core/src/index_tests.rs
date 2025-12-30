#[cfg(test)]
mod tests {
    use crate::index::{IndexOptions, MaintenanceStatus, VectorIndex};
    use crossbeam_epoch::{self as epoch};
    use drift_cache::local_store::LocalDiskManager;
    use std::sync::Arc;
    use std::sync::atomic::Ordering;
    use tempfile::TempDir;

    // Helper to setup Async Index
    async fn create_test_index(dim: usize, num_centroids: usize) -> (VectorIndex, TempDir) {
        let dir = TempDir::new().unwrap();
        let wal_path = dir.path().join("test.wal");
        let storage_path = dir.path().join("storage");
        std::fs::create_dir(&storage_path).unwrap();

        let options = IndexOptions {
            dim,
            num_centroids,
            training_sample_size: 50,
            max_bucket_capacity: 100,
            ef_construction: 100,
            ef_search: 100,
        };

        let storage = Arc::new(LocalDiskManager::new(storage_path));
        let index = VectorIndex::new(options, &wal_path, storage).unwrap();
        (index, dir)
    }

    #[tokio::test]
    async fn test_split_and_steal_atomic() {
        let (index, _guard) = create_test_index(2, 2).await;

        // 1. Train 2 clusters with variance
        // Left Range: [-10.0 ... -7.5]
        // Right Range: [10.0 ... 12.5]
        let mut train = Vec::new();
        for i in 0..25 {
            train.push(vec![-10.0 + (i as f32 * 0.1), 0.0]);
        }
        for i in 0..25 {
            train.push(vec![10.0 + (i as f32 * 0.1), 0.0]);
        }

        index.train(&train).await.unwrap();

        // 2. Identify the "Left" Bucket ID
        let left_id = {
            let guard = epoch::pin();
            let centroids =
                unsafe { index.centroids.load(Ordering::Acquire, &guard).as_ref() }.unwrap();
            centroids.iter().find(|c| c.vector[0] < 0.0).unwrap().id
        };

        // 3. Identify the "Right" Bucket ID
        let right_id = {
            let guard = epoch::pin();
            let centroids =
                unsafe { index.centroids.load(Ordering::Acquire, &guard).as_ref() }.unwrap();
            centroids.iter().find(|c| c.vector[0] > 0.0).unwrap().id
        };

        // 4. Inject a "Defector" into the "Right" Bucket (using in-bounds value)
        index
            .force_register_bucket_with_ids(right_id, &[777], &[vec![-9.0, 0.0]])
            .await
            .unwrap();

        // HACK: Manually reset centroid to force drift
        {
            let guard = epoch::pin();
            loop {
                let shared = index.centroids.load(Ordering::Acquire, &guard);
                let current = unsafe { shared.as_ref() }.unwrap();
                let mut new_vec = current.clone();
                if let Some(c) = new_vec.iter_mut().find(|c| c.id == right_id) {
                    c.vector = vec![10.0, 0.0];
                }
                if index
                    .centroids
                    .compare_exchange(
                        shared,
                        epoch::Owned::new(new_vec),
                        Ordering::Release,
                        Ordering::Relaxed,
                        &guard,
                    )
                    .is_ok()
                {
                    break;
                }
            }
            loop {
                let shared = index.buckets.load(Ordering::Acquire, &guard);
                let current = unsafe { shared.as_ref() }.unwrap();
                let mut new_map = current.clone();
                if let Some(h) = new_map.get_mut(&right_id) {
                    h.centroid = vec![10.0, 0.0];
                }
                if index
                    .buckets
                    .compare_exchange(
                        shared,
                        epoch::Owned::new(new_map),
                        Ordering::Release,
                        Ordering::Relaxed,
                        &guard,
                    )
                    .is_ok()
                {
                    break;
                }
            }
        }

        // Verify precondition
        let k_bytes = 777u64.to_le_bytes();
        let v_buf = index
            .kv
            .get(&k_bytes)
            .unwrap()
            .expect("ID 777 should be in KV");
        let bucket_id_before = u32::from_le_bytes(v_buf.try_into().unwrap());
        assert_eq!(bucket_id_before, right_id, "Precondition failed");

        // Inject High Variance Data into Left Bucket
        // Use values INSIDE the training range (-10.0 to -7.5) to avoid clipping.
        // -10.0 vs -8.0 is distinct.
        // (-14.0 would be clipped to -10.0, creating a singularity).
        let mut left_ids = Vec::new();
        let mut left_vecs = Vec::new();
        for i in 0..30 {
            left_ids.push(1000 + i as u64);
            let val = if i % 2 == 0 { -10.0 } else { -8.0 };
            left_vecs.push(vec![val, 0.0]);
        }
        index
            .force_register_bucket_with_ids(left_id, &left_ids, &left_vecs)
            .await
            .unwrap();

        // 5. Trigger Split
        let status = index.split_and_steal(left_id).await.unwrap();
        assert_eq!(status, MaintenanceStatus::Completed, "Split was skipped!");

        // 6. Verify Steal
        let v_buf = index.kv.get(&k_bytes).unwrap().expect("ID 777 missing");
        let new_bucket_id = u32::from_le_bytes(v_buf.try_into().unwrap());

        assert_ne!(
            new_bucket_id, right_id,
            "Vector 777 should have been stolen"
        );
        assert_ne!(
            new_bucket_id, left_id,
            "Vector 777 should not be in old Left ID"
        );

        // 7. Search
        let res = index
            .search_async(&vec![-9.0, 0.0], 5, 0.9, 100.0, 100.0)
            .await
            .unwrap();
        assert!(res.iter().any(|r| r.id == 777), "777 not found via search");
    }

    #[tokio::test]
    async fn test_split_and_steal_requires_populated_split_bucket_and_steals() {
        let (index, _guard) = create_test_index(2, 2).await;

        // Train
        let mut train = Vec::new();
        for i in 0..25 {
            train.push(vec![-10.0 + (i as f32 * 0.1), 0.0]);
        }
        for i in 0..25 {
            train.push(vec![10.0 + (i as f32 * 0.1), 0.0]);
        }
        index.train(&train).await.unwrap();

        let (left_id, right_id) = {
            let guard = epoch::pin();
            let centroids =
                unsafe { index.centroids.load(Ordering::Acquire, &guard).as_ref() }.unwrap();
            let left = centroids.iter().find(|c| c.vector[0] < 0.0).unwrap().id;
            let right = centroids.iter().find(|c| c.vector[0] > 0.0).unwrap().id;
            (left, right)
        };

        // Inject High Variance Data (In-Bounds)
        let mut left_ids = Vec::new();
        let mut left_vecs = Vec::new();
        for i in 0..30u64 {
            left_ids.push(10_000 + i as u64);
            // Use -8.0 instead of -12.0 to stay in range
            if i % 2 == 0 {
                left_vecs.push(vec![-10.0, 0.0]);
            } else {
                left_vecs.push(vec![-8.0, 0.0]);
            }
        }
        index
            .force_register_bucket_with_ids(left_id, &left_ids, &left_vecs)
            .await
            .unwrap();

        // Seed RIGHT bucket
        let mut right_ids = vec![777u64];
        let mut right_vecs = vec![vec![-9.0, 0.0]];

        for i in 0..25u64 {
            right_ids.push(20_000 + i);
            right_vecs.push(vec![10.0, 0.0]);
        }

        index
            .force_register_bucket_with_ids(right_id, &right_ids, &right_vecs)
            .await
            .unwrap();

        // Split
        let status = index.split_and_steal(left_id).await.unwrap();
        assert_eq!(
            status,
            MaintenanceStatus::Completed,
            "Split aborted due to low variance"
        );

        // Verify Steal
        let k_bytes = 777u64.to_le_bytes();
        let after = index
            .kv
            .get(&k_bytes)
            .unwrap()
            .expect("777 missing from KV");
        let mapped_after = u32::from_le_bytes(after.try_into().unwrap());

        assert_ne!(mapped_after, right_id, "Expected 777 to be stolen");
        assert_ne!(mapped_after, left_id, "Old left bucket should be gone");

        let res = index
            .search_async(&vec![-9.0, 0.0], 20, 0.9, 100.0, 100.0)
            .await
            .unwrap();
        assert!(res.iter().any(|r| r.id == 777), "777 not found via search");
    }

    #[tokio::test]
    async fn test_saturating_density_stopping_nonflaky() {
        let (index, _guard) = create_test_index(2, 2).await;
        let mut train_data = Vec::new();
        for i in 0..25 {
            train_data.push(vec![0.0 + i as f32 * 0.01, 0.0]);
        }
        for i in 0..25 {
            train_data.push(vec![10.0 + i as f32 * 0.01, 10.0]);
        }
        index.train(&train_data).await.unwrap();

        let near_zero_bucket = {
            let guard = epoch::pin();
            let buckets =
                unsafe { index.buckets.load(Ordering::Acquire, &guard).as_ref() }.unwrap();
            buckets
                .iter()
                .min_by(|(_, a), (_, b)| {
                    let da = crate::math::l2_sq(&a.centroid, &[0.0, 0.0]);
                    let db = crate::math::l2_sq(&b.centroid, &[0.0, 0.0]);
                    da.partial_cmp(&db).unwrap()
                })
                .map(|(id, _)| *id)
                .unwrap()
        };

        index
            .force_register_bucket_with_ids(near_zero_bucket, &[999], &[vec![0.0, 0.0]])
            .await
            .unwrap();
        let results = index
            .search_async(&vec![0.0, 0.0], 50, 0.5, 1.0, 10.0)
            .await
            .unwrap();
        assert!(results.iter().any(|r| r.id == 999));
    }

    #[tokio::test]
    async fn test_scatter_merge_logic() {
        let (index, _guard) = create_test_index(2, 2).await;
        let mut train_data = Vec::new();
        for _ in 0..25 {
            train_data.push(vec![0.0, 0.0]);
        }
        for _ in 0..25 {
            train_data.push(vec![100.0, 100.0]);
        }
        index.train(&train_data).await.unwrap();

        let (id_0, _) = {
            let guard = epoch::pin();
            let buckets =
                unsafe { index.buckets.load(Ordering::Acquire, &guard).as_ref() }.unwrap();
            buckets
                .iter()
                .find(|(_, b)| b.centroid[0] < 50.0)
                .map(|(id, b)| (*id, b.clone()))
                .unwrap()
        };

        index
            .force_register_bucket_with_ids(id_0, &[888], &[vec![99.0, 99.0]])
            .await
            .unwrap();
        index.scatter_merge(id_0).await.unwrap();

        let results = index
            .search_async(&vec![100.0, 100.0], 30, 0.9, 1.0, 100.0)
            .await
            .unwrap();
        assert!(results.iter().any(|r| r.id == 888));
    }
}
