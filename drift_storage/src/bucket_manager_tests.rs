#[cfg(test)]
mod tests {
    use crate::bucket_file_writer::BucketFileWriter;
    use crate::bucket_manager::{BucketManager, StorageClass};
    use drift_core::lock_manager::BucketCoordinator;
    use drift_core::quantizer::Quantizer;
    use drift_traits::{DiskSearcher, TombstoneView};
    use opendal::{Operator, services};
    use std::sync::Arc;
    use tempfile::tempdir;
    use tokio::time::{Duration, sleep};

    // --- Mocks & Helpers ---

    #[derive(Debug)]
    struct NoTombstones;
    impl TombstoneView for NoTombstones {
        fn contains(&self, _id: u64) -> bool {
            false
        }
        fn len(&self) -> usize {
            0
        }
    }

    fn create_local_operator(path: &std::path::Path) -> Operator {
        let builder = services::Fs::default().root(path.to_str().unwrap());
        Operator::new(builder).unwrap().finish()
    }

    /// Creates a valid .drift file with specific data
    async fn create_bucket_file(
        dir: &std::path::Path,
        filename: &str,
        start_id: u64,
        count: usize,
        dim: usize,
        val: f32,
    ) {
        let path = dir.join(filename);
        let file = std::fs::File::create(&path).unwrap();

        let ids: Vec<u64> = (0..count as u64).map(|i| start_id + i).collect();
        let vecs: Vec<Vec<f32>> = (0..count).map(|_| vec![val; dim]).collect();
        let flat_vecs: Vec<f32> = vecs.into_iter().flatten().collect();

        let q = Quantizer::train(&flat_vecs, dim);
        let mut writer = BucketFileWriter::new_streaming(file, [0u8; 16], q, dim).unwrap();
        writer.write_batch(&ids, &flat_vecs).unwrap();
        writer.finalize().unwrap();
    }

    // --- TEST 1: Tiered Access (Local Delta + Remote Base) ---
    #[tokio::test]
    async fn test_tiered_search_aggregates_sources() {
        let dir = tempdir().unwrap();
        let op = create_local_operator(dir.path());
        let coordinator = Arc::new(BucketCoordinator::new());
        let manager = BucketManager::new(op.clone(), op.clone(), 4, coordinator);

        let dim = 2;
        let bucket_id = 1;

        // 1. Create Remote Base File (IDs 0-9, Value 10.0)
        let remote_file = "remote_base.drift";
        create_bucket_file(dir.path(), remote_file, 0, 10, dim, 10.0).await;

        // 2. Create Local Delta File (IDs 10-14, Value 20.0)
        let local_file = "local_delta.drift";
        create_bucket_file(dir.path(), local_file, 10, 5, dim, 20.0).await;

        // 3. Register as Tiered
        manager.register_bucket(
            bucket_id,
            "ignored_primary_path".to_string(), // Tiered ignores the main path
            StorageClass::Tiered {
                remote_path: remote_file.to_string(),
                local_path: local_file.to_string(),
            },
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
                Arc::new(NoTombstones),
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
        let manager = BucketManager::new(op.clone(), op.clone(), 4, coordinator);
        let dim = 2;
        let bucket_id = 2;

        // 1. Setup 3 Files simulating the Promotion transition
        create_bucket_file(dir.path(), "active.drift", 100, 1, dim, 1.0).await; // ID 100
        create_bucket_file(dir.path(), "frozen.drift", 200, 1, dim, 2.0).await; // ID 200
        create_bucket_file(dir.path(), "remote.drift", 300, 1, dim, 3.0).await; // ID 300

        // 2. Register Promoting State
        manager.register_bucket(
            bucket_id,
            "active.drift".to_string(),
            StorageClass::Promoting {
                local_active: "active.drift".to_string(),
                local_frozen: "frozen.drift".to_string(),
                remote_path: Some("remote.drift".to_string()),
            },
        );

        // 3. Search
        let results = manager
            .search_and_refine(
                &[bucket_id],
                &vec![0.0; dim],
                10,
                30,
                Arc::new(NoTombstones),
            )
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
        ));

        let bucket_id = 3;
        let dim = 2;

        // Initial State: Just Local
        create_bucket_file(dir.path(), "initial.drift", 0, 100, dim, 0.0).await;
        manager.register_bucket(bucket_id, "initial.drift".to_string(), StorageClass::Local);

        // Next State: Remote (Simulating completion of promotion)
        create_bucket_file(dir.path(), "final.drift", 0, 100, dim, 0.0).await;

        let m_clone = manager.clone();
        let c_clone = coordinator.clone();

        // Task A: Searcher (Simulate a long running search)
        let search_handle = tokio::spawn(async move {
            // We loop search to catch the transition
            for _ in 0..10 {
                let results = m_clone
                    .search_and_refine(
                        &[bucket_id],
                        &vec![0.0; dim],
                        10,
                        30,
                        Arc::new(NoTombstones),
                    )
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
            manager.register_bucket(bucket_id, "final.drift".to_string(), StorageClass::Remote);

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
        let manager = BucketManager::new(op.clone(), op.clone(), 4, coordinator);

        // Register a file that DOES NOT EXIST
        manager.register_bucket(99, "ghost.drift".to_string(), StorageClass::Local);

        // Search should not panic, just return empty/partial
        let results = manager
            .search_and_refine(&[99], &vec![0.0; 2], 10, 30, Arc::new(NoTombstones))
            .await;

        assert!(results.is_empty(), "Should handle missing file gracefully");
    }
}
