#[cfg(test)]
mod tests {
    use crate::config::Config;
    use crate::manager::CollectionManager;
    use crate::persistence::PersistenceManager;
    use tempfile::tempdir;

    fn gen_vec(val: f32, dim: usize) -> Vec<f32> {
        vec![val; dim]
    }

    fn default_test_config(path: &std::path::Path) -> Config {
        Config {
            port: 50051,
            storage_uri: format!("file://{}", path.join("storage").to_string_lossy()),
            wal_dir: path.join("wal"),
            default_dim: 128,
            max_bucket_capacity: 1000,
            ef_construction: 50,
            ef_search: 50,
        }
    }

    #[tokio::test]
    async fn test_collection_manager_isolation_and_recovery() {
        let dir = tempdir().unwrap();
        let dim = 128;

        let config = default_test_config(dir.path());

        // Phase 1: Create Data
        {
            let manager = CollectionManager::new(config.clone());

            // --- Collection A: "users" ---
            let users = manager.get_or_create("users", Some(dim)).await.unwrap();
            users
                .index
                .train(&vec![gen_vec(0.0, dim), gen_vec(1.0, dim)])
                .await
                .unwrap();

            for i in 0..10 {
                users.index.insert(i, &gen_vec(0.1, dim)).unwrap();
            }

            // Force flush "users"
            let data_u = users.index.rotate_memtable().unwrap();

            // ⚡ FIX: Point to WAL directory, just like CollectionManager does
            let users_wal_path = config.wal_dir.join("users");
            std::fs::create_dir_all(&users_wal_path).unwrap(); // Ensure dir exists

            let p_mgr_u = PersistenceManager::new(users_wal_path);
            p_mgr_u
                .flush_memtable_to_segment(&data_u, &users.index, "init_users")
                .await
                .unwrap();

            // --- Collection B: "products" ---
            let products = manager.get_or_create("products", Some(dim)).await.unwrap();
            products
                .index
                .train(&vec![gen_vec(10.0, dim)])
                .await
                .unwrap();
            products.index.insert(0, &gen_vec(10.0, dim)).unwrap();

            // Force flush "products"
            let data_p = products.index.rotate_memtable().unwrap();

            // ⚡ FIX: Point to WAL directory here too
            let products_wal_path = config.wal_dir.join("products");
            std::fs::create_dir_all(&products_wal_path).unwrap();

            let p_mgr_p = PersistenceManager::new(products_wal_path);
            p_mgr_p
                .flush_memtable_to_segment(&data_p, &products.index, "init_products")
                .await
                .unwrap();
        }

        println!("--- RESTARTING MANAGER ---");

        // Phase 2: Reload
        let manager_2 = CollectionManager::new(config.clone());

        // Load "users"
        let users_2 = manager_2.get_or_create("users", Some(dim)).await.unwrap();
        let res_u = users_2
            .index
            .search_async(&gen_vec(0.1, dim), 1, 0.1, 1.0, 100.0)
            .await
            .unwrap();

        assert!(!res_u.is_empty());
        assert!(res_u[0].distance < 0.05, "Failed to recover user data");

        // Load "products"
        let products_2 = manager_2
            .get_or_create("products", Some(dim))
            .await
            .unwrap();
        let res_p = products_2
            .index
            .search_async(&gen_vec(10.0, dim), 1, 0.1, 1.0, 100.0)
            .await
            .unwrap();

        assert!(!res_p.is_empty());
        assert_eq!(res_p[0].id, 0, "Failed to recover products data");

        // Phase 3: Verify Isolation
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
