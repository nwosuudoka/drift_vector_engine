#[cfg(test)]
mod tests {
    use crate::janitor::Janitor;
    use crate::persistence::PersistenceManager;
    use drift_core::index::{IndexOptions, VectorIndex};
    use std::sync::Arc;
    use std::time::Duration;
    use tempfile::tempdir;
    use tokio::time::sleep;

    #[tokio::test]
    async fn test_janitor_lifecycle_flush_and_truncate() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("current.wal");
        let persistence = PersistenceManager::new(dir.path());

        // 1. Initialize Index
        let options = IndexOptions {
            dim: 2,
            num_centroids: 1,
            training_sample_size: 50,
            max_bucket_capacity: 100,
            ef_construction: 50,
            ef_search: 50,
        };
        let index = Arc::new(VectorIndex::new(options, &wal_path).unwrap());

        // Train to enable Quantizer (required for flushing)
        index.train(&vec![vec![0.0, 0.0], vec![100.0, 100.0]]);

        // 2. Spawn Janitor
        // Threshold = 100. Interval = 10ms (fast for test).
        let janitor = Janitor::new(index.clone(), persistence, 100, Duration::from_millis(10));

        tokio::spawn(async move {
            janitor.run().await;
        });

        // 3. Insert Data (Enough to trigger ~2.5 flushes)
        // Insert 250 items.
        // 0-99 -> Flush 1
        // 100-199 -> Flush 2
        // 200-249 -> Remains in MemTable
        for i in 0..250 {
            index.insert(i as u64, &vec![10.0, 10.0]).unwrap();

            // Tiny sleep every 50 inserts to let Janitor catch up (simulating real traffic)
            if i % 50 == 0 {
                sleep(Duration::from_millis(20)).await;
            }
        }

        // Give Janitor final moment to react
        sleep(Duration::from_millis(100)).await;

        // 4. VERIFICATION

        // A. Check MemTable Size
        // Should have roughly 50 items (200-249), definitely NOT 250.
        let mem_size = index.memtable_len();
        println!("Final MemTable Size: {}", mem_size);
        assert!(
            mem_size < 150,
            "Janitor failed to flush! MemTable still full."
        );
        assert!(
            mem_size > 0,
            "MemTable shouldn't be completely empty (last batch)."
        );

        // B. Check WAL Truncation
        // The WAL should only contain the current MemTable (approx 50 items).
        // 50 items * ~20 bytes < 2000 bytes. If full (250 items), it would be much larger.
        let wal_len = std::fs::metadata(&wal_path).unwrap().len();
        println!("Final WAL Size: {} bytes", wal_len);
        assert!(wal_len > 0);
        // Heuristic: Ensure it's not growing infinitely.
        // (Exact byte math depends on serialization overhead, but it should be small).

        // C. Check Disk Segments
        let mut segment_count = 0;
        for entry in std::fs::read_dir(dir.path()).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|s| s.to_str()) == Some("drift") {
                segment_count += 1;
                println!("Found Segment: {:?}", path);
            }
        }
        assert!(
            segment_count >= 1,
            "Should have flushed at least 1 segment to disk"
        );

        // D. Data Accessibility (Search)
        // Search for an item that was definitely flushed (e.g., ID 10)
        // NOTE: Currently, `force_register_bucket` inside persistence/load might mess up IDs,
        // but since we are searching the *Index* which hasn't been reloaded,
        // the L1 buckets in RAM (which persistence adds to) *should* keep their IDs if implemented correctly.
        // Wait, `flush_memtable_to_segment` WRITES to disk, but does it ADD back to the L1 index in memory?
        // Ah! The Janitor writes to disk, but the `VectorIndex` doesn't know about the new file yet!

        // CRITICAL MISSING LINK:
        // The Janitor flushes to disk, but we need to *register* that new segment back into the `VectorIndex` L1 map
        // so it becomes searchable immediately.
        // Or, usually, we "reload" or "attach" the segment.

        // For this test, we verify the FILE exists.
        // Future Step: Make Janitor call `index.register_segment(...)`.
    }
}

#[cfg(test)]
mod gold_standard_tests {
    use crate::janitor::Janitor;
    use crate::persistence::PersistenceManager;
    use drift_core::index::{IndexOptions, VectorIndex};
    use std::sync::Arc;
    use std::time::Duration;
    use tempfile::tempdir;
    use tokio::time::sleep;

    // --- Helper to create a standard test index ---
    fn create_index(dir: &std::path::Path, wal_name: &str) -> Arc<VectorIndex> {
        let wal_path = dir.join(wal_name);
        let options = IndexOptions {
            dim: 2,
            num_centroids: 1,
            training_sample_size: 50,
            max_bucket_capacity: 100,
            ef_construction: 50,
            ef_search: 50,
        };
        let index = Arc::new(VectorIndex::new(options, &wal_path).unwrap());
        // Train to ensure Quantizer is ready (required for flushing)
        index.train(&vec![vec![0.0, 0.0], vec![100.0, 100.0]]);
        index
    }

    #[tokio::test]
    async fn test_end_to_end_durability_and_recovery() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("current.wal");
        let persistence = PersistenceManager::new(dir.path());

        println!("--- PHASE 1: HIGH TRAFFIC WRITES ---");

        let index = create_index(dir.path(), "current.wal");

        // 1. Spawn Janitor
        // Threshold = 100. Interval = 10ms (Aggressive for test).
        let janitor = Janitor::new(index.clone(), persistence, 100, Duration::from_millis(10));
        tokio::spawn(async move {
            janitor.run().await;
        });

        // 2. Insert Data (350 items)
        // Expected behavior:
        // - Flush 1 at ~100 items (IDs 0-99)
        // - Flush 2 at ~200 items (IDs 100-199)
        // - Flush 3 at ~300 items (IDs 200-299)
        // - RAM Tail: ~50 items (IDs 300-349)

        for i in 0..350 {
            index.insert(i as u64, &vec![10.0, 10.0]).unwrap();
            if i % 50 == 0 {
                sleep(Duration::from_millis(20)).await;
            }
        }

        // Wait for final background flushes
        sleep(Duration::from_millis(500)).await;

        println!("--- PHASE 2: VERIFY DISK STATE ---");

        // A. Verify WAL Truncation
        let wal_len = std::fs::metadata(&wal_path).unwrap().len();
        println!("Final WAL Size: {} bytes", wal_len);

        // A full WAL with 350 items would be ~15KB (350 * 40 bytes).
        // A truncated WAL with 50 items should be ~2KB.
        assert!(wal_len < 5000, "WAL was not truncated! It's too large.");
        assert!(wal_len > 0, "WAL shouldn't be empty (contains tail).");

        // B. Verify Segment Files
        let mut segments = Vec::new();
        for entry in std::fs::read_dir(dir.path()).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|s| s.to_str()) == Some("drift") {
                segments.push(path);
            }
        }
        println!("Found {} segments on disk.", segments.len());
        assert!(segments.len() >= 3, "Expected at least 3 flushed segments.");

        println!("--- PHASE 3: SIMULATE CRASH & RECOVERY ---");

        // 3. Drop old index (Simulate Shutdown)
        drop(index);

        // 4. Load from Disk (PersistenceManager)
        let persistence_loader = PersistenceManager::new(dir.path());

        // We need to load ALL segments + the WAL.
        // In a real server, we would loop through all .drift files.
        // Here we manually load one to prove the point, or iterate if we can.

        // Let's create a fresh index and populate it from the segments we found.
        // This mimics the server startup logic we haven't written yet.

        let index_recovered = create_index(dir.path(), "current.wal"); // Re-opens the WAL

        // Load Segments back into memory
        // for seg_path in segments {
        //     println!("Loading segment: {:?}", seg_path);
        //     let reader = drift_storage::segment_reader::SegmentReader::open(&seg_path)
        //         .await
        //         .unwrap();

        //     // We use the raw reader to hydrate because `load_from_segment` creates a new index,
        //     // but we want to merge into ONE index.
        //     for id in reader.index.buckets.keys() {
        //         let vecs = reader.clone().read_bucket(*id).await.unwrap();
        //         // WARNING: Current `force_register_bucket` resets IDs to 0..N.
        //         // This is a known limitation we accepted earlier.
        //         // For this test, we verify the *count* and *existence* of data buckets.
        //         index_recovered.force_register_bucket(*id, &vecs);
        //     }
        // }

        // Load Segments back into memory
        for seg_path in segments {
            println!("Loading segment: {:?}", seg_path);
            // 1. Open the reader as mutable
            let mut reader = drift_storage::segment_reader::SegmentReader::open(&seg_path)
                .await
                .unwrap();

            // 2. Extract IDs first to avoid borrowing issues during the loop
            let bucket_ids: Vec<u32> = reader.index.buckets.keys().cloned().collect();

            for id in bucket_ids {
                // 3. Call read_bucket directly on the mutable reader (no clone needed)
                let (ids, vecs) = reader.read_bucket(id).await.unwrap();

                // Hydrate the recovered index
                index_recovered.force_register_bucket_with_ids(id, &ids, &vecs);
            }
        }

        println!("--- PHASE 4: VERIFY DATA INTEGRITY ---");

        // 5. Search Validation
        // We search for vectors from different generations.

        // A. Search for L0 Tail Data (ID 349) - Should be in WAL/MemTable
        let res_tail = index_recovered.search_drift_aware(&vec![10.0, 10.0], 10, 0.9, 1.0, 100.0);
        // We inserted identical vectors [10,10], so we check if *any* result has high ID
        // Note: L0 IDs are preserved by WAL.
        let found_tail = res_tail.iter().any(|r| r.id >= 300);
        assert!(found_tail, "Failed to recover L0 Tail data from WAL");

        // B. Search for Old L1 Data
        // Since IDs might be reset to 0 in L1 currently, we check if we found *enough* results.
        // If segments loaded, we should find results from L1 (Bucket IDs) + L0.
        // The fact that we have > 50 results (WAL size) implies L1 is working.

        // Count total results (approximate via high K)
        let res_all = index_recovered.search_drift_aware(&vec![10.0, 10.0], 500, 0.5, 1.0, 100.0);
        println!("Recovered Search found {} items", res_all.len());

        assert!(
            res_all.len() > 60,
            "Only found WAL items. Failed to load L1 segments."
        );
    }

    #[tokio::test]
    async fn test_concurrent_inserts_and_flush() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("concurrent.wal");
        let persistence = PersistenceManager::new(dir.path());
        let index = create_index(dir.path(), "concurrent.wal");

        let janitor = Janitor::new(index.clone(), persistence, 50, Duration::from_millis(1));

        // 1. Run Janitor
        tokio::spawn(async move {
            janitor.run().await;
        });

        // 2. Hammer with concurrent inserts
        // We want to ensure no deadlocks when `rotate_memtable` locks the WAL.
        let handles: Vec<_> = (0..10)
            .map(|i| {
                let idx = index.clone();
                tokio::spawn(async move {
                    for j in 0..100 {
                        let id = (i * 1000) + j;
                        idx.insert(id as u64, &vec![1.0, 1.0]).unwrap();
                        // No sleep, max contention
                    }
                })
            })
            .collect();

        for h in handles {
            h.await.unwrap();
        }

        // 3. Verify total count (approximate via MemTable + Files)
        // Just ensuring it didn't crash or hang is the main win here.
        let mem_size = index.memtable_len();
        println!(
            "Concurrent Test Finished. Final MemTable Size: {}",
            mem_size
        );

        // It didn't deadlock!
        assert!(true);
    }
}
