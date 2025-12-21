#[cfg(test)]
mod tests {
    use crate::manager::CollectionManager;
    use crate::persistence::PersistenceManager;
    use drift_core::index::{IndexOptions, VectorIndex};
    use std::sync::Arc;
    use std::time::Duration;
    use tempfile::tempdir;
    use tokio::time::sleep;

    /// Helper to generate a vector of specific dimension
    fn gen_vec(val: f32, dim: usize) -> Vec<f32> {
        vec![val; dim]
    }

    /// TEST 2: Collection Manager Lifecycle
    /// Verifies isolation and persistence.
    #[tokio::test]
    async fn test_collection_manager_isolation_and_recovery() {
        let dir = tempdir().unwrap();
        // Server defaults to 128 dims in manager.rs
        let dim = 128;

        // Phase 1: Create Data
        {
            let manager = CollectionManager::new(dir.path());

            // Collection A: "users"
            let users = manager.get_or_create("users").await.unwrap();
            users
                .index
                .train(&vec![gen_vec(0.0, dim), gen_vec(1.0, dim)]);

            for i in 0..10 {
                users.index.insert(i, &gen_vec(0.1, dim)).unwrap();
            }

            // Force flush
            let data = users.index.rotate_memtable().unwrap();
            let p_mgr = PersistenceManager::new(dir.path().join("users"));
            p_mgr
                .flush_memtable_to_segment(&data, &users.index, "init")
                .await
                .unwrap();

            // Collection B: "products"
            let products = manager.get_or_create("products").await.unwrap();
            products.index.train(&vec![gen_vec(10.0, dim)]);
            products.index.insert(0, &gen_vec(10.0, dim)).unwrap();
        }

        println!("--- RESTARTING MANAGER ---");

        // Phase 2: Reload
        let manager_2 = CollectionManager::new(dir.path());

        // Load "users"
        let users_2 = manager_2.get_or_create("users").await.unwrap();
        let res_u = users_2
            .index
            .search_drift_aware(&gen_vec(0.1, dim), 1, 0.1, 1.0, 100.0);

        assert!(!res_u.is_empty());
        // Relaxed assertion: We found *a* user with correct data
        assert!(
            res_u[0].distance < 0.05,
            "Failed to recover valid user data. Distance: {}",
            res_u[0].distance
        );

        // Load "products"
        let products_2 = manager_2.get_or_create("products").await.unwrap();
        let res_p = products_2
            .index
            .search_drift_aware(&gen_vec(10.0, dim), 1, 0.1, 1.0, 100.0);
        assert!(!res_p.is_empty());
        assert_eq!(
            res_p[0].id, 0,
            "Failed to recover 'products' collection data"
        );

        // Phase 3: Verify Isolation
        let res_cross = products_2
            .index
            .search_drift_aware(&gen_vec(0.1, dim), 1, 0.0, 1.0, 100.0);
        if !res_cross.is_empty() {
            // If found, score should be very bad (high distance)
            assert!(
                res_cross[0].distance > 10.0,
                "Cross-contamination: Found user data in products!"
            );
        }

        println!("✅ test_collection_manager_isolation_and_recovery Passed!");
    }

    /// TEST 1: Hydration Logic
    /// Verifies that PersistenceManager can reconstruct the index state.
    #[tokio::test]
    async fn test_persistence_hydration() {
        let dir = tempdir().unwrap();
        let persistence = PersistenceManager::new(dir.path());
        let wal_path = dir.path().join("current.wal");

        let dim = 128;

        // A. Setup: Create an index and generate segments on disk
        {
            let options = IndexOptions {
                dim,
                num_centroids: 2,
                training_sample_size: 20,
                max_bucket_capacity: 50,
                ef_construction: 50,
                ef_search: 50,
            };
            let index = VectorIndex::new(options, &wal_path).unwrap();

            index.train(&vec![gen_vec(1.0, dim), gen_vec(-1.0, dim)]);

            // Create Segment 1 (IDs 0-9)
            let batch1: Vec<(u64, Vec<f32>)> = (0..10).map(|i| (i, gen_vec(1.0, dim))).collect();
            persistence
                .flush_memtable_to_segment(&batch1, &index, "batch_1")
                .await
                .unwrap();

            // Create Segment 2 (IDs 10-19)
            let batch2: Vec<(u64, Vec<f32>)> = (10..20).map(|i| (i, gen_vec(-1.0, dim))).collect();
            persistence
                .flush_memtable_to_segment(&batch2, &index, "batch_2")
                .await
                .unwrap();

            // Insert into WAL (IDs 20-24)
            for i in 20..25 {
                index.insert(i, &gen_vec(0.5, dim)).unwrap();
            }
        }

        println!("--- SIMULATING RESTART ---");

        // B. Action: Create FRESH index and Hydrate
        let options = IndexOptions {
            dim,
            num_centroids: 2,
            training_sample_size: 20,
            max_bucket_capacity: 50,
            ef_construction: 50,
            ef_search: 50,
        };
        let new_index = VectorIndex::new(options, &wal_path).unwrap();

        // Verify WAL replay (L0)
        assert_eq!(new_index.memtable_len(), 5, "WAL replay failed");

        // Run Hydration (L1)
        persistence.hydrate_index(&new_index).await.unwrap();

        // C. Assertion: Search

        // 1. Search for Segment 1 data (IDs 0-9)
        let res_1 = new_index.search_drift_aware(&gen_vec(1.0, dim), 1, 0.1, 1.0, 100.0);
        assert!(!res_1.is_empty());
        assert!(res_1[0].distance < 0.05);
        assert!(
            res_1[0].id < 10,
            "Result ID {} should be from Segment 1 (0-9)",
            res_1[0].id
        );

        // 2. Search for Segment 2 data (IDs 10-19)
        let res_2 = new_index.search_drift_aware(&gen_vec(-1.0, dim), 1, 0.1, 1.0, 100.0);
        assert!(!res_2.is_empty());
        assert!(res_2[0].distance < 0.05);
        assert!(
            res_2[0].id >= 10 && res_2[0].id < 20,
            "Result ID {} should be from Segment 2 (10-19)",
            res_2[0].id
        );

        // 3. Search for WAL data (IDs 20-24)
        let res_wal = new_index.search_drift_aware(&gen_vec(0.5, dim), 1, 0.1, 1.0, 100.0);
        assert!(!res_wal.is_empty(), "Result empty for WAL search");

        // Key Fix: Check that the returned ID is *any* of the WAL IDs.
        // Since all 5 vectors are identical, returning any of them is correct.
        let wal_id = res_wal[0].id;
        assert!(
            (20..25).contains(&wal_id),
            "Failed to find WAL data. Got ID {}, expected 20..24",
            wal_id
        );

        println!("✅ test_persistence_hydration Passed!");
    }
}
