#[cfg(test)]
mod tests {
    use crate::bucket::BucketHeader; // Needed for internal inspection if necessary
    use crate::index::{IndexOptions, VectorIndex};
    use crossbeam_epoch::{self as epoch};
    use drift_cache::local_store::LocalDiskManager;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::sync::atomic::Ordering;
    use tempfile::TempDir;

    // Helper to setup Async Index
    async fn create_test_index(dim: usize, num_centroids: usize) -> (VectorIndex, TempDir) {
        let dir = TempDir::new().unwrap();
        let wal_path = dir.path().join("test.wal");
        // Use a subdir for cache storage to keep it clean
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
    async fn test_saturating_density_stopping_nonflaky() {
        let (index, _guard) = create_test_index(2, 2).await;

        // Train 2 clusters: Near [0,0] and Near [10,10]
        let mut train_data = Vec::new();
        for _ in 0..25 {
            train_data.push(vec![0.0, 0.0]);
        }
        for _ in 0..25 {
            train_data.push(vec![10.0, 10.0]);
        }
        index.train(&train_data).await.unwrap();

        // Find the bucket whose centroid is closest to [0,0] (don't assume id=0)
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

        // Inject high-density L1 item into the correct bucket
        index
            .force_register_bucket_with_ids(near_zero_bucket, &[999], &[vec![0.0, 0.0]])
            .await
            .unwrap();

        // Use a larger k so we don't lose 999 due to top-k truncation / ties
        let results = index
            .search_async(&vec![0.0, 0.0], 50, 0.5, 1.0, 10.0)
            .await
            .unwrap();

        assert!(results.iter().any(|r| r.id == 999), "L1 item 999 not found");

        // Also verify KV mapping (stronger signal than top-k search)
        let k_bytes = 999u64.to_le_bytes();
        let v_buf = index
            .kv
            .get(&k_bytes)
            .unwrap()
            .expect("999 missing from KV");
        let mapped_bucket = u32::from_le_bytes(v_buf.try_into().unwrap());
        assert_eq!(
            mapped_bucket, near_zero_bucket,
            "KV should map 999 to the injected L1 bucket"
        );
    }

    #[tokio::test]
    async fn test_drift_aware_routing_logic_nonflaky() {
        let (index, _guard) = create_test_index(2, 2).await;

        // Use distinct data to ensure valid Quantizer training
        let training: Vec<Vec<f32>> = (0..50).map(|i| vec![i as f32, i as f32]).collect();
        index.train(&training).await.unwrap();

        // Insert DISTINCT vectors into L0 (MemTable)
        index.insert(100, &vec![0.0, 0.0]).unwrap();
        index.insert(101, &vec![0.1, 0.1]).unwrap();

        // Use larger k so L1 points can't crowd out our L0 inserts.
        // We only want to test that L0 is searched/merged at all.
        let res = index
            .search_async(&vec![0.0, 0.0], 100, 0.99, 1.0, 100.0)
            .await
            .unwrap();

        assert!(res.iter().any(|r| r.id == 100), "L0 item 100 not found");
        assert!(res.iter().any(|r| r.id == 101), "L0 item 101 not found");
    }

    #[tokio::test]
    async fn test_scatter_merge_logic() {
        let (index, _guard) = create_test_index(2, 2).await;

        // 1. Train 2 clusters
        let mut train_data = Vec::new();
        for _ in 0..25 {
            train_data.push(vec![0.0, 0.0]);
        }
        for _ in 0..25 {
            train_data.push(vec![100.0, 100.0]);
        }
        index.train(&train_data).await.unwrap();

        // 2. Identify Bucket 0 (Near [0,0])
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

        // 3. Inject "Drifter" into Bucket 0
        // It is physically in Bucket 0 ([0,0]), but its value is [99,99] (Close to Bucket 1)
        index
            .force_register_bucket_with_ids(id_0, &[888], &[vec![99.0, 99.0]])
            .await
            .unwrap();

        // 4. Trigger Scatter Merge on Bucket 0
        // This destroys Bucket 0.
        // It should adopt "orphans" into their nearest neighbors.
        // ID 888 [99,99] is clearly closer to Bucket 1 [100,100].
        index.scatter_merge(id_0).await.unwrap();

        // 5. Verify: Bucket 0 should be gone from the map
        {
            let guard = epoch::pin();
            let buckets =
                unsafe { index.buckets.load(Ordering::Acquire, &guard).as_ref() }.unwrap();
            assert!(buckets.get(&id_0).is_none(), "Bucket 0 should be deleted");
        }

        // 6. Verify: 888 should be searchable (Adopted by Bucket 1)
        // We use k=5. Since Bucket 1 has 25 items at dist=0, and 888 is at dist=1.41,
        // typically 888 would be the 26th result.
        // HOWEVER, scatter_merge rewrites the target bucket.
        // If it appends 888 to Bucket 1, it exists there physically.
        // We search with k=30 to ensure we find it even if it's ranked lower than the exact matches.
        let results = index
            .search_async(&vec![100.0, 100.0], 30, 0.9, 1.0, 100.0)
            .await
            .unwrap();

        assert!(results.iter().any(|r| r.id == 888), "Drifter 888 was lost!");

        // 7. Verify KV Consistency
        let k_bytes = 888u64.to_le_bytes();
        let v_buf = index
            .kv
            .get(&k_bytes)
            .unwrap()
            .expect("888 missing from KV");
        let new_id = u32::from_le_bytes(v_buf.try_into().unwrap());

        assert_ne!(new_id, id_0, "KV still points to dead Bucket 0");
    }

    #[tokio::test]
    async fn test_split_and_steal_atomic() {
        let (index, _guard) = create_test_index(2, 2).await;

        // 1. Setup: Two clear clusters
        // Left: [-10, 0], Right: [10, 0]
        let mut train = Vec::new();
        for _ in 0..25 {
            train.push(vec![-10.0, 0.0]);
        }
        for _ in 0..25 {
            train.push(vec![10.0, 0.0]);
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

        // 4. Inject a "Defector" into the "Right" Bucket
        // force_register will set the centroid to [-9.0] (matching the data).
        // We must RESET it to [10.0] to simulate that this point is an outlier/drifter.
        index
            .force_register_bucket_with_ids(right_id, &[777], &[vec![-9.0, 0.0]])
            .await
            .unwrap();

        // HACK: Manually reset the centroid to [10.0, 0.0] to create "Drift" condition
        {
            // Update Centroids Map
            let guard = epoch::pin();
            loop {
                let shared = index.centroids.load(Ordering::Acquire, &guard);
                let current = unsafe { shared.as_ref() }.unwrap();
                let mut new_vec = current.clone();
                if let Some(c) = new_vec.iter_mut().find(|c| c.id == right_id) {
                    c.vector = vec![10.0, 0.0]; // Reset to original center
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

            // Update Buckets Map (Header)
            loop {
                let shared = index.buckets.load(Ordering::Acquire, &guard);
                let current = unsafe { shared.as_ref() }.unwrap();
                let mut new_map = current.clone();
                if let Some(h) = new_map.get_mut(&right_id) {
                    h.centroid = vec![10.0, 0.0]; // Reset to original center
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

        // Verify precondition: 777 is in Right ID
        let k_bytes = 777u64.to_le_bytes();
        let v_buf = index
            .kv
            .get(&k_bytes)
            .unwrap()
            .expect("ID 777 should be in KV");
        let bucket_id_before = u32::from_le_bytes(v_buf.try_into().unwrap());
        assert_eq!(
            bucket_id_before, right_id,
            "Precondition failed: 777 not in Right Bucket"
        );

        // 5. Trigger Split on LEFT Bucket
        // It checks neighbors (Right ID).
        // It sees 777 at [-9.0].
        // Right Centroid is [10.0] (Dist = 19^2 = 361).
        // Left Centroid is [-10.0] (Dist = 1^2 = 1).
        // 1 < 361 -> STEAL!
        index.split_and_steal(left_id).await.unwrap();

        // 6. Verify Steal via KV Store
        let v_buf = index
            .kv
            .get(&k_bytes)
            .unwrap()
            .expect("ID 777 missing from KV after split");
        let new_bucket_id = u32::from_le_bytes(v_buf.try_into().unwrap());

        assert_ne!(
            new_bucket_id, right_id,
            "Vector 777 should have been stolen from Right ID"
        );
        assert_ne!(
            new_bucket_id, left_id,
            "Vector 777 should not be in old Left ID (it was deleted)"
        );

        // 7. Verify Searchability
        let res = index
            .search_async(&vec![-9.0, 0.0], 5, 0.9, 100.0, 100.0)
            .await
            .unwrap();
        assert!(res.iter().any(|r| r.id == 777), "777 not found via search");
    }

    // ============== MORE TESTS ==============

    // A richer helper for tests that need to reopen the index / inspect WAL path.
    async fn create_test_index_with_paths(
        dim: usize,
        num_centroids: usize,
    ) -> (VectorIndex, TempDir, IndexOptions, PathBuf, PathBuf) {
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

        let storage = Arc::new(LocalDiskManager::new(storage_path.clone()));
        let index = VectorIndex::new(options.clone(), &wal_path, storage).unwrap();
        (index, dir, options, wal_path, storage_path)
    }

    #[tokio::test]
    async fn test_rotate_memtable_drains_and_truncates_wal() {
        let (index, _guard, _opts, wal_path, _storage_path) =
            create_test_index_with_paths(2, 2).await;

        // No training required for rotate_memtable coverage.
        index.insert(1, &vec![1.0, 1.0]).unwrap();
        index.insert(2, &vec![2.0, 2.0]).unwrap();
        index.insert(3, &vec![3.0, 3.0]).unwrap();

        // WAL should have content.
        let before = std::fs::metadata(&wal_path).unwrap().len();
        assert!(before > 0, "WAL should be non-empty after inserts");

        let drained = index.rotate_memtable().unwrap();
        assert_eq!(
            drained.len(),
            3,
            "rotate_memtable should return all memtable items"
        );
        assert_eq!(
            index.memtable_len(),
            0,
            "memtable should be empty after rotate_memtable"
        );

        // WAL should be truncated.
        let after = std::fs::metadata(&wal_path).unwrap().len();
        assert_eq!(after, 0, "WAL should be truncated after rotate_memtable");
    }

    #[tokio::test]
    async fn test_wal_replay_persists_inserts_and_applies_deletes() {
        let (index, _guard, opts, wal_path, storage_path) =
            create_test_index_with_paths(2, 2).await;

        // Train so search_async routing is available (even though we only use L0 here).
        let training: Vec<Vec<f32>> = (0..50).map(|i| vec![i as f32, i as f32]).collect();
        index.train(&training).await.unwrap();

        index.insert(10, &vec![0.0, 0.0]).unwrap();
        index.insert(11, &vec![0.1, 0.1]).unwrap();
        index.delete(11).unwrap();

        // Drop the index (simulate restart).
        drop(index);

        // Reopen index pointing at same WAL + storage dir
        let storage = Arc::new(LocalDiskManager::new(storage_path));
        let reopened = VectorIndex::new(opts, &wal_path, storage).unwrap();

        // Re-train so search_async has centroids (WAL replay restores memtable+KV, not the quantizer).
        let training2: Vec<Vec<f32>> = (0..50).map(|i| vec![i as f32, i as f32]).collect();
        reopened.train(&training2).await.unwrap();

        let res = reopened
            .search_async(&vec![0.0, 0.0], 10, 0.99, 10.0, 100.0)
            .await
            .unwrap();

        assert!(
            res.iter().any(|r| r.id == 10),
            "Inserted id=10 should survive restart"
        );
        assert!(
            !res.iter().any(|r| r.id == 11),
            "Deleted id=11 should NOT survive restart"
        );
    }

    #[tokio::test]
    async fn test_force_register_bucket_updates_kv_mapping() {
        let (index, _guard) = create_test_index(2, 2).await;

        // Train 2 clusters: Near [0,0] and Near [10,10]
        let mut train_data = Vec::new();
        for _ in 0..25 {
            train_data.push(vec![0.0, 0.0]);
        }
        for _ in 0..25 {
            train_data.push(vec![10.0, 10.0]);
        }
        index.train(&train_data).await.unwrap();

        // Choose a bucket near [0,0]
        let bucket_id = {
            let guard = epoch::pin();
            let buckets =
                unsafe { index.buckets.load(Ordering::Acquire, &guard).as_ref() }.unwrap();
            buckets
                .iter()
                .find(|(_, b)| b.centroid[0] < 5.0)
                .map(|(id, _)| *id)
                .unwrap()
        };

        index
            .force_register_bucket_with_ids(bucket_id, &[4242], &[vec![0.0, 0.0]])
            .await
            .unwrap();

        // Verify KV points to that bucket (same pattern as existing tests).
        let k_bytes = 4242u64.to_le_bytes();
        let v_buf = index
            .kv
            .get(&k_bytes)
            .unwrap()
            .expect("4242 missing from KV");
        let mapped = u32::from_le_bytes(v_buf.try_into().unwrap());

        assert_eq!(mapped, bucket_id, "KV should map inserted id to its bucket");
    }

    #[tokio::test]
    async fn test_search_async_merges_l0_and_l1_hits() {
        let (index, _guard) = create_test_index(2, 2).await;

        // Train
        let training: Vec<Vec<f32>> = (0..50).map(|i| vec![i as f32, i as f32]).collect();
        index.train(&training).await.unwrap();

        // L0 insert
        index.insert(9001, &vec![1.0, 1.0]).unwrap();

        // Pick some L1 bucket and inject 9002 into it
        let bucket_id = {
            let guard = epoch::pin();
            let buckets =
                unsafe { index.buckets.load(Ordering::Acquire, &guard).as_ref() }.unwrap();
            *buckets.keys().next().unwrap()
        };

        index
            .force_register_bucket_with_ids(bucket_id, &[9002], &[vec![1.1, 1.1]])
            .await
            .unwrap();

        let res = index
            .search_async(&vec![1.0, 1.0], 10, 0.9, 10.0, 100.0)
            .await
            .unwrap();

        assert!(
            res.iter().any(|r| r.id == 9001),
            "Expected L0 id=9001 in results"
        );
        assert!(
            res.iter().any(|r| r.id == 9002),
            "Expected L1 id=9002 in results"
        );
    }

    #[tokio::test]
    async fn test_delete_removes_from_l0_and_search() {
        let (index, _guard) = create_test_index(2, 2).await;

        let training: Vec<Vec<f32>> = (0..50).map(|i| vec![i as f32, i as f32]).collect();
        index.train(&training).await.unwrap();

        index.insert(7007, &vec![0.0, 0.0]).unwrap();

        // Search should find it before delete (L0)
        let res_before = index
            .search_async(&vec![0.0, 0.0], 10, 0.99, 10.0, 100.0)
            .await
            .unwrap();
        assert!(
            res_before.iter().any(|r| r.id == 7007),
            "Insert not searchable"
        );

        index.delete(7007).unwrap();

        // KV may or may not contain L0 ids by design; but delete removes it if it exists.
        let k_bytes = 7007u64.to_le_bytes();
        assert!(
            index.kv.get(&k_bytes).unwrap().is_none(),
            "Expected KV entry removed after delete (if it existed)"
        );

        // Search should not find it after delete
        let res_after = index
            .search_async(&vec![0.0, 0.0], 10, 0.99, 10.0, 100.0)
            .await
            .unwrap();
        assert!(
            !res_after.iter().any(|r| r.id == 7007),
            "Deleted id should not be searchable"
        );
    }

    #[tokio::test]
    async fn test_split_and_steal_requires_populated_split_bucket_and_steals() {
        let (index, _guard) = create_test_index(2, 2).await;

        // Train: Left [-10,0], Right [10,0]
        let mut train = Vec::new();
        for _ in 0..25 {
            train.push(vec![-10.0, 0.0]);
        }
        for _ in 0..25 {
            train.push(vec![10.0, 0.0]);
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

        // IMPORTANT: seed LEFT bucket with enough points so split actually executes
        // (make it robust whether the threshold is 10 or 20 by inserting 30 points).
        let mut left_ids = Vec::new();
        let mut left_vecs = Vec::new();
        for i in 0..30u64 {
            left_ids.push(10_000 + i as u64);
            // two tight subclusters so kmeans split is well-defined
            if i % 2 == 0 {
                left_vecs.push(vec![-10.0, 0.0]);
            } else {
                left_vecs.push(vec![-11.0, 0.0]);
            }
        }
        index
            .force_register_bucket_with_ids(left_id, &left_ids, &left_vecs)
            .await
            .unwrap();

        // Seed RIGHT bucket (neighbors exist) + add the "defector"
        let mut right_ids = vec![777u64];
        let mut right_vecs = vec![vec![-9.0, 0.0]]; // closer to left than to right centroid

        for i in 0..25u64 {
            right_ids.push(20_000 + i);
            right_vecs.push(vec![10.0, 0.0]);
        }

        index
            .force_register_bucket_with_ids(right_id, &right_ids, &right_vecs)
            .await
            .unwrap();

        // Precondition: 777 maps to right
        let k_bytes = 777u64.to_le_bytes();
        let before = index
            .kv
            .get(&k_bytes)
            .unwrap()
            .expect("777 missing from KV");
        let mapped_before = u32::from_le_bytes(before.try_into().unwrap());
        assert_eq!(
            mapped_before, right_id,
            "Precondition failed: 777 not in right bucket"
        );

        // Split left and steal from neighbors
        index.split_and_steal(left_id).await.unwrap();

        // After: 777 should no longer map to the old right bucket
        let after = index
            .kv
            .get(&k_bytes)
            .unwrap()
            .expect("777 missing from KV after split");
        let mapped_after = u32::from_le_bytes(after.try_into().unwrap());
        assert_ne!(
            mapped_after, right_id,
            "Expected 777 to be stolen from right"
        );
        assert_ne!(
            mapped_after, left_id,
            "Old left bucket should be gone after split"
        );

        // Searchability: should still be findable near [-9,0]
        let res = index
            .search_async(&vec![-9.0, 0.0], 20, 0.9, 100.0, 100.0)
            .await
            .unwrap();
        assert!(
            res.iter().any(|r| r.id == 777),
            "777 not found via search after steal"
        );
    }

    #[tokio::test]
    async fn test_drift_aware_routing_logic() {
        let (index, _guard) = create_test_index(2, 2).await;

        let training: Vec<Vec<f32>> = (0..50).map(|i| vec![i as f32, i as f32]).collect();
        index.train(&training).await.unwrap();

        // Insert DISTINCT vectors into L0
        // Use values that are clearly distinct from training data (integers)
        // L0: [0.5, 0.5] (Dist to 0,0 is 0.5)
        // L1: [0,0] (Dist 0), [1,1] (Dist 1.41)
        index.insert(100, &vec![0.5, 0.5]).unwrap();

        // Search
        let res = index
            .search_async(&vec![0.0, 0.0], 10, 0.99, 1.0, 100.0)
            .await
            .unwrap();

        assert!(
            res.iter().any(|r| r.id == 100),
            "L0 item 100 not found in results: {:?}",
            res
        );
    }

    #[tokio::test]
    async fn test_saturating_density_stopping_2() {
        let (index, _guard) = create_test_index(2, 2).await;

        // 1. Train with TWO Clusters
        // Dense Cluster A: [100,100] (25 items)
        // Dense Cluster B: [200,200] (25 items)
        // Query will be at [0,0].
        let mut train_data = Vec::new();
        for _ in 0..25 {
            train_data.push(vec![100.0, 100.0]);
        }
        for _ in 0..25 {
            train_data.push(vec![200.0, 200.0]);
        }

        index.train(&train_data).await.unwrap();

        // 2. Identify a bucket ID to overwrite (Any existing ID works)
        let target_id = 0;

        // 3. Inject SPARSE L1 Data at [0,0]
        // This overwrites Bucket 0 with a single item at [0,0].
        // Now Bucket 0 is "Sparse" (Count 1) but "Close" (Dist 0).
        // The other bucket is "Dense" (Count 25) but "Far" (Dist 141).
        index
            .force_register_bucket_with_ids(target_id, &[999], &[vec![0.0, 0.0]])
            .await
            .unwrap();

        // 4. Search [0,0]
        // Density Score:
        // Bucket 0 (Sparse/Close): Geom=1.0 * Dens=0.09 = 0.09
        // Bucket 1 (Dense/Far):    Geom=e^-141 * Dens=1.0 ~ 0.0
        //
        // Correct logic must pick Bucket 0 despite low density because it's the ONLY geometric match.
        let results = index
            .search_async(&vec![0.0, 0.0], 5, 0.05, 1.0, 10.0) // Low confidence target
            .await
            .unwrap();

        assert!(!results.is_empty());
        assert!(
            results.iter().any(|r| r.id == 999),
            "L1 item 999 not found. Results: {:?}",
            results
        );
    }
}
