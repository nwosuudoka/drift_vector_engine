#[cfg(test)]
mod tests {
    use crate::manager::CollectionManager;
    use crate::persistence::PersistenceManager;
    use tempfile::tempdir;

    fn gen_vec(val: f32, dim: usize) -> Vec<f32> {
        vec![val; dim]
    }

    #[tokio::test]
    async fn test_collection_manager_isolation_and_recovery() {
        let dir = tempdir().unwrap();
        let dim = 128;

        // Phase 1: Create Data
        {
            let manager = CollectionManager::new(dir.path());

            // Collection A: "users"
            let users = manager.get_or_create("users").await.unwrap();
            users
                .index
                .train(&vec![gen_vec(0.0, dim), gen_vec(1.0, dim)])
                .await
                .unwrap();

            for i in 0..10 {
                users.index.insert(i, &gen_vec(0.1, dim)).unwrap();
            }

            // Force flush manually for test speed
            let data = users.index.rotate_memtable().unwrap();
            let p_mgr = PersistenceManager::new(dir.path().join("users"));
            p_mgr
                .flush_memtable_to_segment(&data, &users.index, "init")
                .await
                .unwrap();

            // Collection B: "products"
            let products = manager.get_or_create("products").await.unwrap();
            products
                .index
                .train(&vec![gen_vec(10.0, dim)])
                .await
                .unwrap();
            products.index.insert(0, &gen_vec(10.0, dim)).unwrap();
        }

        println!("--- RESTARTING MANAGER ---");

        // Phase 2: Reload
        let manager_2 = CollectionManager::new(dir.path());

        // Load "users"
        let users_2 = manager_2.get_or_create("users").await.unwrap();
        let res_u = users_2
            .index
            .search_async(&gen_vec(0.1, dim), 1, 0.1, 1.0, 100.0)
            .await
            .unwrap();

        assert!(!res_u.is_empty());
        assert!(res_u[0].distance < 0.05, "Failed to recover user data");

        // Load "products"
        let products_2 = manager_2.get_or_create("products").await.unwrap();
        let res_p = products_2
            .index
            .search_async(&gen_vec(10.0, dim), 1, 0.1, 1.0, 100.0)
            .await
            .unwrap();

        assert!(!res_p.is_empty());
        assert_eq!(res_p[0].id, 0, "Failed to recover products data");

        // Phase 3: Verify Isolation
        // Searching "products" for user data should yield nothing (or very far distance)
        let res_cross = products_2
            .index
            .search_async(&gen_vec(0.1, dim), 1, 0.0, 1.0, 100.0)
            .await
            .unwrap();

        if !res_cross.is_empty() {
            assert!(
                res_cross[0].distance > 10.0,
                "Cross-contamination: Found user data in products!"
            );
        }

        println!("✅ test_collection_manager_isolation_and_recovery Passed!");
    }
}
