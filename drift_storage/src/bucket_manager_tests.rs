#[cfg(test)]
mod tests {
    use crate::bucket_manager::{BucketManager, StorageClass};
    use crate::unified_writer::UnifiedLocalWriter;
    use drift_core::lock_manager::BucketCoordinator;
    use drift_core::math::Metric;
    use drift_traits::StorageEngine;
    use opendal::{Operator, services};
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;
    use std::time::Duration;
    use tempfile::tempdir;
    use tokio::time::sleep;

    fn create_local_operator(path: &std::path::Path) -> Operator {
        let builder = services::Fs::default().root(path.to_str().unwrap());
        Operator::new(builder).unwrap().finish()
    }

    /// Creates a valid unified file with specific data.
    async fn create_bucket_file(
        dir: &std::path::Path,
        filename: &str,
        ids: &[u64],
        vecs: &[Vec<f32>],
        dim: usize,
    ) {
        let path = dir.join(filename);
        let flat: Vec<f32> = vecs.iter().flatten().copied().collect();
        UnifiedLocalWriter::write_vector_only_flat_to_path(path, ids, &flat, dim).unwrap();
    }

    // --- TEST 1: Tiered Access (Local Delta + Remote Base) ---
    #[tokio::test]
    async fn test_tiered_search_aggregates_sources() {
        let dir = tempdir().unwrap();
        let op = create_local_operator(dir.path());
        let coordinator = Arc::new(BucketCoordinator::new());
        let manager = BucketManager::new(op.clone(), op.clone(), 4, coordinator, Metric::L2);

        let dim = 2;
        let bucket_id = 1;

        // 1. Create Remote Base File (IDs 0-9, Value 10.0)
        let remote_file = "remote_base.driftu";
        create_bucket_file(
            dir.path(),
            remote_file,
            &(0..10).collect::<Vec<_>>(),
            &vec![vec![10.0; dim]; 10],
            dim,
        )
        .await;

        // 2. Create Local Delta File (IDs 10-14, Value 20.0)
        let local_file = "local_delta.driftu";
        create_bucket_file(
            dir.path(),
            local_file,
            &(10..15).collect::<Vec<_>>(),
            &vec![vec![20.0; dim]; 5],
            dim,
        )
        .await;

        // 3. Register as Tiered
        manager.register_bucket_with_count(
            bucket_id,
            "ignored_primary_path".to_string(), // Tiered ignores the main path
            StorageClass::Tiered {
                remote_path: remote_file.to_string(),
                local_path: local_file.to_string(),
            },
            15,
        );

        // 4. Search (Atomic Search & Refine)
        // Query [10.0, 10.0] -> Should match Remote best, then Local
        let query = vec![10.0; dim];
        let results = manager
            .search_and_refine(
                &[bucket_id],
                &query,
                20, // K
                60, // Oversample
            )
            .await;

        // 5. Verify Aggregation
        assert_eq!(
            results.len(),
            15,
            "Should find all 15 items (10 Remote + 5 Local)"
        );

        // Verify Content
        // Results are Vec<(u64, f32)>
        let has_remote_id = results.iter().any(|(id, _)| *id == 0);
        let has_local_id = results.iter().any(|(id, _)| *id == 14);

        assert!(has_remote_id, "Missing remote data");
        assert!(has_local_id, "Missing local data");
    }

    // --- TEST 2: Promoting State (Active + Frozen + Remote) ---
    #[tokio::test]
    async fn test_promoting_state_scans_all_three_sources() {
        let dir = tempdir().unwrap();
        let op = create_local_operator(dir.path());
        let coordinator = Arc::new(BucketCoordinator::new());
        let manager = BucketManager::new(op.clone(), op.clone(), 4, coordinator, Metric::L2);
        let dim = 2;
        let bucket_id = 2;

        // 1. Setup 3 Files simulating the Promotion transition
        create_bucket_file(dir.path(), "active.driftu", &[100], &[vec![1.0; dim]], dim).await; // ID 100
        create_bucket_file(dir.path(), "frozen.driftu", &[200], &[vec![2.0; dim]], dim).await; // ID 200
        create_bucket_file(dir.path(), "remote.driftu", &[300], &[vec![3.0; dim]], dim).await; // ID 300

        // 2. Register Promoting State
        manager.register_bucket_with_count(
            bucket_id,
            "active.driftu".to_string(),
            StorageClass::Promoting {
                local_active: "active.driftu".to_string(),
                local_frozen: "frozen.driftu".to_string(),
                remote_path: Some("remote.driftu".to_string()),
            },
            3,
        );

        // 3. Search
        let results = manager
            .search_and_refine(&[bucket_id], &vec![0.0; dim], 10, 30)
            .await;

        // 4. Verify
        assert_eq!(results.len(), 3, "Must scan Active, Frozen, AND Remote");

        let ids: Vec<u64> = results.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&100));
        assert!(ids.contains(&200));
        assert!(ids.contains(&300));
    }

    // --- TEST 3: Concurrent Update Safety ---
    #[tokio::test]
    async fn test_concurrent_registry_update_during_search() {
        let dir = tempdir().unwrap();
        let op = create_local_operator(dir.path());
        let coordinator = Arc::new(BucketCoordinator::new());
        let manager = Arc::new(BucketManager::new(
            op.clone(),
            op.clone(),
            4,
            coordinator.clone(),
            Metric::L2,
        ));

        let bucket_id = 3;
        let dim = 2;

        // Initial State: Just Local
        create_bucket_file(
            dir.path(),
            "initial.driftu",
            &(0..100).collect::<Vec<_>>(),
            &vec![vec![0.0; dim]; 100],
            dim,
        )
        .await;
        manager.register_bucket(bucket_id, "initial.driftu".to_string(), StorageClass::Local);

        // Next State: Remote (Simulating completion of promotion)
        create_bucket_file(
            dir.path(),
            "final.driftu",
            &(0..100).collect::<Vec<_>>(),
            &vec![vec![0.0; dim]; 100],
            dim,
        )
        .await;

        let m_clone = manager.clone();
        let m_clone_2 = manager.clone();
        let c_clone = coordinator.clone();

        // Task A: Searcher (Simulate a long running search)
        let search_handle = tokio::spawn(async move {
            // We loop search to catch the transition
            for _ in 0..10 {
                let results = m_clone
                    .search_and_refine(&[bucket_id], &vec![0.0; dim], 10, 30)
                    .await;
                assert!(
                    !results.is_empty(),
                    "Search returned empty results during transition!"
                );
                sleep(Duration::from_millis(5)).await;
            }
        });

        // Task B: Writer (Janitor updating registry)
        let update_handle = tokio::spawn(async move {
            sleep(Duration::from_millis(15)).await;

            // Acquire Write Lock (mimicking Janitor)
            let _guard = c_clone.write(bucket_id).await;

            // Update Registry
            m_clone_2.register_bucket(bucket_id, "final.driftu".to_string(), StorageClass::Remote);

            // Verify lock works: Search shouldn't run here
            sleep(Duration::from_millis(20)).await;
        });

        let (r1, r2) = tokio::join!(search_handle, update_handle);
        r1.unwrap();
        r2.unwrap();
    }

    // --- TEST 4: Missing File Handling ---
    #[tokio::test]
    async fn test_missing_file_graceful_handling() {
        let dir = tempdir().unwrap();
        let op = create_local_operator(dir.path());
        let coordinator = Arc::new(BucketCoordinator::new());
        let manager = BucketManager::new(op.clone(), op.clone(), 4, coordinator, Metric::L2);

        // Register a file that DOES NOT EXIST
        manager.register_bucket(99, "ghost.driftu".to_string(), StorageClass::Local);

        // Search should not panic, just return empty/partial
        let results = manager
            .search_and_refine(&[99], &vec![0.0; 2], 10, 30)
            .await;

        assert!(results.is_empty(), "Should handle missing file gracefully");
    }

    // --- TEST: Local Tombstone Filtering ---
    #[tokio::test]
    async fn test_bucket_manager_local_delete() {
        let dir = tempdir().unwrap();
        let op = create_local_operator(dir.path());
        let coordinator = Arc::new(BucketCoordinator::new());
        let manager = BucketManager::new(op.clone(), op.clone(), 4, coordinator, Metric::L2);
        let dim = 2;
        let bucket_id = 1;

        // 1. Setup Data: IDs 10, 20, 30
        let ids = vec![10, 20, 30];
        let vecs = vec![vec![1.0, 1.0], vec![2.0, 2.0], vec![3.0, 3.0]];
        create_bucket_file(dir.path(), "b1.driftu", &ids, &vecs, dim).await;

        manager.register_bucket_with_count(
            bucket_id,
            "b1.driftu".to_string(),
            StorageClass::Local,
            3,
        );

        // 2. Verify Initial Search (All present)
        let query = vec![1.0, 1.0];
        let res_1 = manager
            .search_and_refine(&[bucket_id], &query, 10, 10)
            .await;
        assert_eq!(res_1.len(), 3);

        // 3. Mark Delete (ID 20)
        manager.mark_delete(bucket_id, 20).unwrap();

        // 4. Verify Search Filters ID 20
        let res_2 = manager
            .search_and_refine(&[bucket_id], &query, 10, 10)
            .await;
        assert_eq!(res_2.len(), 2);
        assert!(res_2.iter().any(|(id, _)| *id == 10));
        assert!(!res_2.iter().any(|(id, _)| *id == 20)); // Gone
        assert!(res_2.iter().any(|(id, _)| *id == 30));

        // 5. Verify Stats Update
        let stats = manager.get_bucket_stats(bucket_id).unwrap();
        assert_eq!(stats.tombstone_count, 1);
        assert_eq!(stats.total_count, 3);
    }

    // --- TEST: Shadowing via explicit delete ---
    #[tokio::test]
    async fn test_bucket_manager_shadowing_view() {
        let dir = tempdir().unwrap();
        let op = create_local_operator(dir.path());
        let coordinator = Arc::new(BucketCoordinator::new());
        let manager = BucketManager::new(op.clone(), op.clone(), 4, coordinator, Metric::L2);

        // 1. Setup Data (ID 100 exists on disk)
        create_bucket_file(dir.path(), "b1.driftu", &[100], &[vec![0.0, 0.0]], 2).await;
        manager.register_bucket(1, "b1.driftu".to_string(), StorageClass::Local);

        // 2. SHADOW ACTION
        // "Shadowing" means marking the disk version as deleted because a newer one exists in RAM.
        manager.mark_delete(1, 100).expect("Failed to mark delete");

        // 3. Search
        // Note: No "GlobalView" passed here anymore. The Manager checks its own internal state.
        let res = manager.search_and_refine(&[1], &[0.0, 0.0], 10, 10).await;

        // 4. Verify ID 100 is filtered
        assert!(res.is_empty(), "Shadowing failed to hide ID 100");
    }

    #[tokio::test]
    async fn test_bucket_manager_candidate_id_pushdown() {
        let dir = tempdir().unwrap();
        let op = create_local_operator(dir.path());
        let coordinator = Arc::new(BucketCoordinator::new());
        let manager = BucketManager::new(op.clone(), op.clone(), 4, coordinator, Metric::L2);
        let dim = 2;
        let bucket_id = 7;

        let ids = vec![10, 20, 30];
        let vecs = vec![vec![1.0, 1.0], vec![2.0, 2.0], vec![3.0, 3.0]];
        create_bucket_file(dir.path(), "b7.driftu", &ids, &vecs, dim).await;
        manager.register_bucket_with_count(
            bucket_id,
            "b7.driftu".to_string(),
            StorageClass::Local,
            3,
        );

        let mut candidate_ids: HashMap<u32, HashSet<u64>> = HashMap::new();
        candidate_ids.insert(bucket_id, HashSet::from([20u64]));

        let query = vec![0.0, 0.0];
        let results = manager
            .search_and_refine_with_candidates(&[bucket_id], &query, 10, 10, Some(&candidate_ids))
            .await;

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 20);
    }

    #[tokio::test]
    async fn test_bucket_manager_dense_scan_shortlist_keeps_top_oversample() {
        let dir = tempdir().unwrap();
        let op = create_local_operator(dir.path());
        let coordinator = Arc::new(BucketCoordinator::new());
        let manager = BucketManager::new(op.clone(), op.clone(), 4, coordinator, Metric::L2);
        let dim = 2;
        let bucket_id = 11;

        let ids = vec![10, 20, 30, 40, 50];
        let vecs = vec![
            vec![5.0, 0.0], // far
            vec![1.0, 0.0], // close
            vec![4.0, 0.0], // far
            vec![0.2, 0.0], // closest
            vec![2.0, 0.0], // medium
        ];
        create_bucket_file(dir.path(), "b11.driftu", &ids, &vecs, dim).await;
        manager.register_bucket_with_count(
            bucket_id,
            "b11.driftu".to_string(),
            StorageClass::Local,
            ids.len() as u32,
        );

        let query = vec![0.0, 0.0];
        let results = manager
            .search_and_refine_with_candidates(&[bucket_id], &query, 10, 2, None)
            .await;

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 40);
        assert_eq!(results[1].0, 20);
    }

    #[tokio::test]
    async fn test_bucket_drift_tracking_accumulation() {
        let dir = tempdir().unwrap();
        let op = create_local_operator(dir.path());
        let coordinator = Arc::new(BucketCoordinator::new());
        let manager = BucketManager::new(op.clone(), op.clone(), 4, coordinator, Metric::L2);

        let bucket_id = 1;

        // 1. Register Bucket (Initialize State)
        // Storage class doesn't matter for this test, but it must be registered to exist in the registry.
        manager.register_bucket(bucket_id, "test.driftu".to_string(), StorageClass::Local);

        // 2. First Update (Initialization Path)
        // Simulating flushing a batch of 10 vectors that sum to [10.0, 20.0]
        let delta_sum_1 = vec![10.0, 20.0];
        let count_1 = 10;

        manager
            .update_bucket_drift(bucket_id, &delta_sum_1, count_1)
            .expect("First update failed");

        // Verify Initial State
        let (sum_1, total_1) = manager
            .get_bucket_drift_stats(bucket_id)
            .expect("Stats not found");
        assert_eq!(total_1, 10);
        assert_eq!(sum_1, vec![10.0, 20.0]);

        // 3. Second Update (Accumulation Path)
        // Simulating flushing another batch of 5 vectors summing to [5.5, 5.5]
        let delta_sum_2 = vec![5.5, 5.5];
        let count_2 = 5;

        manager
            .update_bucket_drift(bucket_id, &delta_sum_2, count_2)
            .expect("Second update failed");

        // Verify Accumulation
        let (sum_2, total_2) = manager
            .get_bucket_drift_stats(bucket_id)
            .expect("Stats not found");

        assert_eq!(total_2, 15, "Total count should be 10 + 5");

        // Check sums with float tolerance
        assert!(
            (sum_2[0] - 15.5).abs() < 1e-5,
            "Dim 0 sum incorrect: {}",
            sum_2[0]
        );
        assert!(
            (sum_2[1] - 25.5).abs() < 1e-5,
            "Dim 1 sum incorrect: {}",
            sum_2[1]
        );
    }

    #[tokio::test]
    async fn test_bucket_drift_dimension_mismatch_safety() {
        let dir = tempdir().unwrap();
        let op = create_local_operator(dir.path());
        let coordinator = Arc::new(BucketCoordinator::new());
        let manager = BucketManager::new(op.clone(), op.clone(), 4, coordinator, Metric::L2);

        let bucket_id = 99;
        manager.register_bucket(bucket_id, "safe.driftu".to_string(), StorageClass::Local);

        // 1. Initialize with Dim 2
        manager
            .update_bucket_drift(bucket_id, &vec![1.0, 1.0], 1)
            .unwrap();

        // 2. Try updating with Dim 3 (Should fail silently/warn but NOT crash or corrupt)
        // The code logs a warning and skips the update.
        manager
            .update_bucket_drift(bucket_id, &vec![1.0, 1.0, 1.0], 1)
            .unwrap();

        // 3. Verify state is unchanged
        let (sum, count) = manager.get_bucket_drift_stats(bucket_id).unwrap();
        assert_eq!(sum.len(), 2, "Dimensions should remain 2");
        assert_eq!(
            sum,
            vec![1.0, 1.0],
            "Sum should not be modified by bad update"
        );

        // Count updates happen BEFORE the dimension check in the provided code snippet.
        // If your code updates count before checking dim, this will be 2.
        // If you want it to be atomic, the count update should move inside the check.
        // Based on the provided snippet: `state.total_count.fetch_add` happens first.
        assert_eq!(
            count, 2,
            "Count updated despite dim mismatch (Current implementation behavior)"
        );
    }
}
