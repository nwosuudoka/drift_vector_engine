#[cfg(test)]
mod tests {
    use crate::persistence::PersistenceManager;
    use drift_core::index::{IndexOptions, VectorIndex};
    use drift_storage::disk_manager::DriftPageManager;
    use opendal::{Operator, services};
    use std::collections::HashMap;
    use std::sync::Arc;
    use tempfile::tempdir;

    const DIM: usize = 128;
    const NUM_VECTORS: usize = 1000;

    fn create_local_operator(path: &std::path::Path) -> Operator {
        let mut builder = services::Fs::default();
        builder = builder.root(path.to_str().unwrap());
        Operator::new(builder).unwrap().finish()
    }

    #[tokio::test]
    async fn test_write_scale_pipeline_end_to_end() {
        let _ = tracing_subscriber::fmt()
            .with_env_filter("drift_server=info")
            .with_test_writer()
            .try_init();

        let dir = tempdir().unwrap();
        let op = create_local_operator(dir.path());
        let persistence = PersistenceManager::new(op.clone(), dir.path());

        // 1. Setup Index with DriftPageManager (Required for Segment mapping)
        let wal_path = dir.path().join("test.wal");
        let storage = Arc::new(DriftPageManager::new(op.clone()));

        let options = IndexOptions {
            dim: DIM,
            num_centroids: 4, // Enough to force partitioning
            training_sample_size: 100,
            max_bucket_capacity: 100, // Small capacity to force logic
            ef_construction: 64,
            ef_search: 64,
        };
        let index = Arc::new(VectorIndex::new(options.clone(), &wal_path, storage).unwrap());

        // -------------------------------------------------------------
        // PHASE 1: INGESTION (MemTable)
        // -------------------------------------------------------------
        println!("🚀 Phase 1: Ingestion");

        // Train first
        let train_data = vec![vec![0.0; DIM]; 50];
        index.train(&train_data).await.unwrap();

        // Insert Data
        let mut vectors = Vec::with_capacity(NUM_VECTORS);
        for i in 0..NUM_VECTORS {
            let val = (i % 10) as f32; // Predictable pattern
            vectors.push(vec![val; DIM]);
            index.insert(i as u64, &vectors[i]).unwrap();
        }

        // Verify Visibility (MemTable)
        let results_l0 = index
            .search_async(&vec![0.0; DIM], 5, 0.9, 1.0, 100.0)
            .await
            .unwrap();
        assert!(!results_l0.is_empty(), "Search failed on MemTable");
        assert_eq!(index.memtable_len(), NUM_VECTORS);

        // -------------------------------------------------------------
        // PHASE 2: ROTATION (Zero-Copy Snapshot)
        // -------------------------------------------------------------
        println!("❄️ Phase 2: Rotation to Snapshot");

        let snapshot = index.rotate_and_freeze().unwrap().expect("Rotation failed");

        assert_eq!(snapshot.ids.len(), NUM_VECTORS, "Snapshot lost data");
        assert_eq!(index.memtable_len(), 0, "Active MemTable not cleared");

        // Verify Search Visibility (Snapshot)
        // search_async internally checks the frozen_memtable slot
        let results_snap = index
            .search_async(&vec![0.0; DIM], 5, 0.9, 1.0, 100.0)
            .await
            .unwrap();
        assert!(!results_snap.is_empty(), "Search failed on Snapshot");

        // -------------------------------------------------------------
        // PHASE 3: PARTITIONING (Flat Buffer)
        // -------------------------------------------------------------
        println!("🧮 Phase 3: Partitioning (Flat)");

        let (partitions, cluster_assignments) = index
            .partition_and_flush_flat(&snapshot.ids, &snapshot.vectors, DIM)
            .await
            .unwrap();

        assert!(!partitions.is_empty(), "No partitions generated");
        assert_eq!(partitions.len(), cluster_assignments.len());

        let total_assigned: usize = cluster_assignments.iter().map(|c| c.len()).sum();
        assert_eq!(
            total_assigned, NUM_VECTORS,
            "Lost vectors during partitioning"
        );

        // -------------------------------------------------------------
        // PHASE 4: PERSISTENCE (Parallel ALP)
        // -------------------------------------------------------------
        println!("💾 Phase 4: Persistence (Parallel ALP)");

        let (run_id, locations) = persistence
            .write_partitioned_segment(
                &partitions,
                &cluster_assignments,
                &snapshot.vectors,
                DIM,
                &index,
            )
            .await
            .expect("Persistence failed");

        // Verify Map Generation
        let offsets_map: HashMap<u32, (u64, u32)> = locations
            .into_iter()
            .map(|(id, loc)| (id, (loc.index_offset, loc.index_length as u32)))
            .collect();

        assert_eq!(offsets_map.len(), partitions.len());

        // -------------------------------------------------------------
        // PHASE 5: REGISTRATION & FINALIZATION
        // -------------------------------------------------------------
        println!("📝 Phase 5: Registration");

        index
            .register_partitions(&partitions, &run_id, &offsets_map)
            .await
            .unwrap();

        // Clear Snapshot
        index.confirm_flush().unwrap();

        // Verify Search Visibility (Disk)
        let results_disk = index
            .search_async(&vec![0.0; DIM], 5, 0.9, 1.0, 100.0)
            .await
            .unwrap();
        assert!(!results_disk.is_empty(), "Search failed on Disk Segments");

        // -------------------------------------------------------------
        // PHASE 6: CRASH RECOVERY (Rehydration)
        // -------------------------------------------------------------
        println!("♻️ Phase 6: Crash Recovery");

        // Drop original index to simulate shutdown
        drop(index);

        // Re-open
        let storage_recovered = Arc::new(DriftPageManager::new(op.clone()));
        let index_recovered =
            Arc::new(VectorIndex::new(options.clone(), &wal_path, storage_recovered).unwrap());

        // Hydrate from Disk
        persistence
            .hydrate_index(&index_recovered)
            .await
            .expect("Hydration failed");

        // Verify Data is back
        let results_recovered = index_recovered
            .search_async(&vec![0.0; DIM], 5, 0.9, 1.0, 100.0)
            .await
            .unwrap();
        assert!(
            !results_recovered.is_empty(),
            "Search failed after recovery"
        );

        // Verify Quantizer was loaded
        assert!(
            index_recovered.get_quantizer().is_some(),
            "Quantizer not restored"
        );

        println!("✅ Billion-Scale Pipeline Verified Successfully");
    }
}

#[cfg(test)]
mod tests_2 {
    use crate::persistence::PersistenceManager;
    use drift_core::index::{IndexOptions, VectorIndex};
    use drift_storage::disk_manager::DriftPageManager;
    use opendal::{Operator, services};
    use std::collections::HashMap;
    use std::sync::Arc;
    use tempfile::tempdir;

    const DIM: usize = 128;
    const NUM_VECTORS: usize = 1000;

    fn create_local_operator(path: &std::path::Path) -> Operator {
        let mut builder = services::Fs::default();
        builder = builder.root(path.to_str().unwrap());
        Operator::new(builder).unwrap().finish()
    }

    #[tokio::test]
    async fn test_billion_scale_pipeline_end_to_end() {
        // Enable detailed logging
        let _ = tracing_subscriber::fmt()
            .with_env_filter("drift_server=debug,drift_core=debug,drift_storage=debug")
            .with_test_writer()
            .try_init();

        let dir = tempdir().unwrap();
        let op = create_local_operator(dir.path());
        let persistence = PersistenceManager::new(op.clone(), dir.path());

        let wal_path = dir.path().join("test.wal");
        let storage = Arc::new(DriftPageManager::new(op.clone()));

        // Options designed to force multiple buckets
        let options = IndexOptions {
            dim: DIM,
            num_centroids: 10,
            training_sample_size: 100,
            max_bucket_capacity: 100,
            ef_construction: 64,
            ef_search: 64,
        };
        let index = Arc::new(VectorIndex::new(options, &wal_path, storage).unwrap());

        // -------------------------------------------------------------
        // PHASE 1: INGESTION
        // -------------------------------------------------------------
        println!("🚀 Phase 1: Ingestion");

        // Train
        let train_data = vec![vec![0.0; DIM]; 50];
        index.train(&train_data).await.unwrap();

        // Insert
        for i in 0..NUM_VECTORS {
            let val = (i % 10) as f32;
            let vec = vec![val; DIM];
            index.insert(i as u64, &vec).unwrap();
        }

        // -------------------------------------------------------------
        // PHASE 2: SNAPSHOT
        // -------------------------------------------------------------
        println!("❄️ Phase 2: Rotation");
        let snapshot = index.rotate_and_freeze().unwrap().expect("Rotation failed");
        assert_eq!(snapshot.ids.len(), NUM_VECTORS);

        // -------------------------------------------------------------
        // PHASE 3: PARTITIONING (Instrumented)
        // -------------------------------------------------------------
        println!("🧮 Phase 3: Partitioning");

        let (partitions, assignments) = index
            .partition_and_flush_flat(&snapshot.ids, &snapshot.vectors, DIM)
            .await
            .unwrap();

        // 🔍 DEBUG: Validate Partition Integrity
        println!("   -> Generated {} partitions", partitions.len());
        for (i, p) in partitions.iter().enumerate() {
            let ids_len = p.ids.len();
            let codes_len = p.codes.len();
            let expected_codes = ids_len * DIM;

            println!(
                "   [Part #{}] ID: {}, Count: {}, Codes: {} bytes",
                i, p.bucket_id, ids_len, codes_len
            );

            if codes_len != expected_codes {
                panic!(
                    "💥 DATA CORRUPTION DETECTED IN INDEX!\nBucket {}: Has {} IDs but {} code bytes (Expected {}). Alignment broken?",
                    p.bucket_id, ids_len, codes_len, expected_codes
                );
            }

            // Validate Assignment Alignment
            if assignments[i].len() != ids_len {
                panic!(
                    "💥 ASSIGNMENT MISMATCH!\nBucket {}: Has {} IDs but {} assignments.",
                    p.bucket_id,
                    ids_len,
                    assignments[i].len()
                );
            }
        }

        // -------------------------------------------------------------
        // PHASE 4: PERSISTENCE
        // -------------------------------------------------------------
        println!("💾 Phase 4: Persistence");

        let (run_id, locations) = persistence
            .write_partitioned_segment(&partitions, &assignments, &snapshot.vectors, DIM, &index)
            .await
            .expect("Persistence failed");

        // 🔍 DEBUG: Check File Size
        let segment_file = format!("segment_{}.drift", run_id);
        let meta = op.stat(&segment_file).await.unwrap();
        println!(
            "   -> Segment Written: {} (Size: {} bytes)",
            segment_file,
            meta.content_length()
        );

        if meta.content_length() < 64 {
            panic!(
                "💥 Segment file is too small ({} bytes)! Header/Footer missing?",
                meta.content_length()
            );
        }

        let offsets_map: HashMap<u32, (u64, u32)> = locations
            .into_iter()
            .map(|(id, loc)| (id, (loc.index_offset, loc.index_length as u32)))
            .collect();

        // -------------------------------------------------------------
        // PHASE 5: REGISTRATION
        // -------------------------------------------------------------
        index
            .register_partitions(&partitions, &run_id, &offsets_map)
            .await
            .unwrap();
        index.confirm_flush().unwrap();

        // -------------------------------------------------------------
        // PHASE 6: CRASH RECOVERY (The Failure Point)
        // -------------------------------------------------------------
        println!("♻️ Phase 6: Crash Recovery");

        drop(index); // Close original

        // Re-open
        let storage_recovered = Arc::new(DriftPageManager::new(op.clone()));
        let index_recovered = Arc::new(
            VectorIndex::new(
                IndexOptions {
                    dim: DIM,
                    ..Default::default()
                },
                &wal_path,
                storage_recovered,
            )
            .unwrap(),
        );

        println!("   -> Hydrating Index...");

        // Use match to catch error details
        match persistence.hydrate_index(&index_recovered).await {
            Ok(_) => println!("   ✅ Hydration Complete"),
            Err(e) => {
                println!("   ❌ HYDRATION FAILED: {:?}", e);
                // List file contents for debug
                let entries = op.list("").await.unwrap();
                println!("   📂 Directory Contents:");
                for entry in entries {
                    let m = op.stat(entry.path()).await.unwrap();
                    println!("      - {} ({} bytes)", entry.path(), m.content_length());
                }
                panic!("Test failed during hydration");
            }
        }

        // Final verification
        let results = index_recovered
            .search_async(&vec![0.0; DIM], 5, 0.9, 1.0, 100.0)
            .await
            .unwrap();
        assert!(!results.is_empty(), "Search failed after recovery");

        println!("✅ Test Passed");
    }
}
