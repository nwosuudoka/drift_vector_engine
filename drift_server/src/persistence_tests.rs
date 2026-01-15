#[cfg(test)]
mod tests {
    use crate::persistence::PersistenceManager;
    use drift_cache::local_store::LocalDiskManager;
    use drift_core::index::{IndexOptions, VectorIndex};
    use opendal::{Operator, services};
    use std::sync::Arc;
    use tempfile::tempdir;

    // --- Helpers ---

    fn create_local_operator(path: &std::path::Path) -> Operator {
        let mut builder = services::Fs::default();
        builder = builder.root(path.to_str().unwrap());
        Operator::new(builder).unwrap().finish()
    }

    fn create_index_with_storage(dir: &std::path::Path, wal_name: &str) -> Arc<VectorIndex> {
        let wal_path = dir.join(wal_name);
        let storage_path = dir.join("storage");
        std::fs::create_dir_all(&storage_path).unwrap();
        let storage = Arc::new(LocalDiskManager::new(storage_path));

        let options = IndexOptions {
            dim: 2,
            num_centroids: 2,
            training_sample_size: 50,
            max_bucket_capacity: 100,
            ef_construction: 100,
            ef_search: 100,
        };
        Arc::new(VectorIndex::new(options, &wal_path, storage).unwrap())
    }

    // --- Tests ---

    #[tokio::test]
    async fn test_end_to_end_persistence_lifecycle() {
        let dir = tempdir().unwrap();
        let op = create_local_operator(dir.path());
        let persistence = PersistenceManager::new(op, dir.path());
        let index_original = create_index_with_storage(dir.path(), "current.wal");

        // 1. Train & Fill MemTable
        let train_vecs: Vec<Vec<f32>> = (0..50).map(|_| vec![10.0, 10.0]).collect();
        for (i, vec) in train_vecs.iter().enumerate() {
            index_original.insert(i as u64, vec).unwrap();
        }

        // 2. Snapshot (Zero-Copy)
        let memtable = index_original.memtable.read().clone();

        index_original.train_from_memtable(&memtable).await.unwrap();

        let partitions = index_original.partition_memtable(&memtable).unwrap();

        // 3. Write Segment
        // We mock the freeze so persistence can access data via the "Frozen" slot
        {
            let mut guard = index_original.frozen_memtable.write();
            *guard = Some(memtable.clone());
        }

        let (run_id, _locations) = persistence
            .write_partitioned_segment(&partitions, &index_original)
            .await
            .expect("Flush failed");

        let segment_key = format!("segment_{}.drift", run_id);

        // 4. Recover Specific Segment
        let index_recovered = persistence
            .load_from_segment(&segment_key)
            .await
            .expect("Load failed");

        // 5. Verify L1 Recovery
        let results = index_recovered
            .search_async(&vec![10.0, 10.0], 5, 0.9, 1.0, 10.0)
            .await
            .unwrap();

        assert_eq!(results.len(), 5, "Failed to recover L1 data from segment");
        assert!(results[0].distance < 0.02);
    }

    #[tokio::test]
    async fn test_full_index_hydration_recovery() {
        let dir = tempdir().unwrap();
        let op = create_local_operator(dir.path());
        let persistence = PersistenceManager::new(op, dir.path());

        // --- PHASE 1: Create and Persist Data ---
        {
            let index = create_index_with_storage(dir.path(), "run1.wal");

            //  Use distinct data so ranking is deterministic.
            for i in 0..50 {
                // ID 0 = [10.0, 10.0]
                // ID 1 = [10.1, 10.1]
                let val = 10.0 + (i as f32 * 0.1);
                index.insert(i, &vec![val, val]).unwrap();
            }

            // Prepare Flush
            let memtable = index.memtable.read().clone();
            index.train_from_memtable(&memtable).await.unwrap();

            // ⚡ SYNC Partition (Updated API)
            let partitions = index.partition_memtable(&memtable).unwrap();

            // Freeze
            {
                *index.frozen_memtable.write() = Some(memtable);
            }

            // Write Segment 1
            persistence
                .write_partitioned_segment(&partitions, &index)
                .await
                .unwrap();
        }
        // Index 1 is dropped here. RAM is clear.

        // --- PHASE 2: Hydration (Server Restart) ---

        // 1. Create a FRESH, empty index
        let new_index = create_index_with_storage(dir.path(), "run2.wal");

        // 2. Hydrate
        persistence
            .hydrate_index(&new_index)
            .await
            .expect("Hydration failed");

        // 3. Verify
        let headers_after = new_index.get_all_bucket_headers();
        assert!(!headers_after.is_empty(), "Buckets were not restored");

        // 4. Search for the exact vector of ID 0
        let results = new_index
            .search_async(&vec![10.0, 10.0], 5, 0.9, 1.0, 10.0)
            .await
            .unwrap();

        assert_eq!(results.len(), 5, "Hydrated index returned no results");

        // ⚡ NOW THIS IS SAFE: ID 0 is strictly the closest vector.
        assert_eq!(results[0].id, 0, "Top result should be ID 0");
        assert!(results[0].distance < 0.001, "Distance should be near zero");
    }

    #[tokio::test]
    async fn test_flush_memtable_l0_persistence() {
        let dir = tempdir().unwrap();
        let op = create_local_operator(dir.path());
        let persistence = PersistenceManager::new(op, dir.path());
        let index = create_index_with_storage(dir.path(), "l0.wal");

        index.insert(0, &vec![1.0, 1.0]).unwrap();

        let memtable = index.memtable.read().clone();
        index.train_from_memtable(&memtable).await.unwrap();

        let data = vec![(100, vec![1.0, 1.0]), (200, vec![2.0, 2.0])];

        // Write L0 Segment
        let run_id = "test_l0";
        let file_name = persistence
            .flush_memtable_to_segment(&data, &index, run_id)
            .await
            .unwrap();

        // Load Back
        let loaded_index = persistence.load_from_segment(&file_name).await.unwrap();

        // Verify
        let res = loaded_index
            .search_async(&vec![1.0, 1.0], 1, 0.9, 1.0, 10.0)
            .await
            .unwrap();
        assert_eq!(res[0].id, 100);
    }
}

#[cfg(test)]
mod persistence_integration_tests {
    use crate::persistence_v2::PersistenceManager;
    use drift_storage::bucket_file_reader::BucketFileReader;
    use opendal::{Operator, services};
    use tempfile::tempdir;

    // --- Helpers ---

    fn create_local_operator(path: &std::path::Path) -> Operator {
        let builder = services::Fs::default().root(path.to_str().unwrap());
        Operator::new(builder).unwrap().finish()
    }

    fn mock_data(start_id: u64, count: usize, dim: usize, val: f32) -> (Vec<u64>, Vec<Vec<f32>>) {
        let ids: Vec<u64> = (0..count as u64).map(|i| start_id + i).collect();
        let vecs: Vec<Vec<f32>> = (0..count).map(|_| vec![val; dim]).collect();
        (ids, vecs)
    }

    #[tokio::test]
    async fn test_persistence_promotion_and_merge_flow() {
        // Setup
        let dir = tempdir().unwrap();
        let op = create_local_operator(dir.path());
        let persistence = PersistenceManager::new(op.clone());
        let bucket_id = 1;
        let dim = 4;

        // ==========================================================
        // PHASE 1: Initial Promotion (Local Staging -> S3)
        // ==========================================================
        println!("Phase 1: Initial Promotion...");
        let (ids_1, vecs_1) = mock_data(0, 10, dim, 1.0); // IDs 0-9, Value 1.0

        let (run_id_1, count_1) = persistence
            .promote_to_s3(bucket_id, &ids_1, &vecs_1, None, dim)
            .await
            .expect("Initial promotion failed");

        assert_eq!(count_1, 10);
        let key_1 = format!("bucket_{}_{}.drift", bucket_id, run_id_1);

        // Verify file exists
        assert!(op.exists(&key_1).await.unwrap());

        // ==========================================================
        // PHASE 2: Merge Promotion (New Local + Old S3 -> New S3)
        // ==========================================================
        println!("Phase 2: Merge Promotion...");
        let (ids_2, vecs_2) = mock_data(10, 10, dim, 2.0); // IDs 10-19, Value 2.0

        let (run_id_2, count_2) = persistence
            .promote_to_s3(bucket_id, &ids_2, &vecs_2, Some(run_id_1.clone()), dim)
            .await
            .expect("Merge promotion failed");

        assert_eq!(count_2, 20, "Should contain 10 old + 10 new items");
        let key_2 = format!("bucket_{}_{}.drift", bucket_id, run_id_2);

        // ==========================================================
        // PHASE 3: Verification (Read Back)
        // ==========================================================
        println!("Phase 3: Verify Merged Data...");

        let mut reader = BucketFileReader::open(op.clone(), &key_2)
            .await
            .expect("Failed to open merged file");

        let (read_ids, read_vecs) = reader
            .read_all_vectors()
            .await
            .expect("Failed to read vectors");

        // Check Counts
        assert_eq!(read_ids.len(), 20);
        assert_eq!(read_vecs.len(), 20);

        // Check Content (Order is preserved: Local appended to Remote)
        // Note: The implementation in persistence_v2 appends remote to local?
        // Let's check the code:
        // let mut merged_ids = local_ids.to_vec(); ... merged_ids.extend(remote_ids);
        // So Local (10..20) comes FIRST, then Remote (0..9).

        // Verify ID 10 (Local) is present and has value 2.0
        let idx_10 = read_ids.iter().position(|&x| x == 10).unwrap();
        assert_eq!(read_vecs[idx_10][0], 2.0);

        // Verify ID 0 (Remote) is present and has value 1.0
        let idx_0 = read_ids.iter().position(|&x| x == 0).unwrap();
        assert_eq!(read_vecs[idx_0][0], 1.0);

        println!("✅ Persistence Integration Test Passed!");
    }
}
