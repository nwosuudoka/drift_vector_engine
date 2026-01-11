#[cfg(test)]
mod tests {
    use crate::index_v2::VectorIndexV2;
    use crate::manifest::pb::Centroid; // Assuming this moved or is available
    use crate::router::Router;
    use crate::wal_v2::WalWriter;
    use async_trait::async_trait;
    use drift_traits::{DiskSearcher, SearchCandidate, TombstoneView};
    use parking_lot::{Mutex, RwLock};
    use std::collections::HashMap;
    use std::sync::Arc;
    use tempfile::tempdir;

    #[test]
    fn test_saturating_density_selection() {
        // Setup: 4 Centroids to satisfy potential Top-3 Guardrail
        // C1: [0, 0]   (Dist 0)    Count 10   (Tiny but closest)
        // C3: [2, 2]   (Dist ~2.8) Count 100  (Medium)
        // C4: [3, 3]   (Dist ~4.2) Count 100  (Medium)
        // C2: [10, 10] (Dist ~14)  Count 1000 (Huge but far)

        let centroids = vec![
            Centroid {
                id: 1,
                vector: vec![0.0, 0.0],
            },
            Centroid {
                id: 2,
                vector: vec![10.0, 10.0],
            }, // The one we want to exclude
            Centroid {
                id: 3,
                vector: vec![2.0, 2.0],
            }, // Distractor 1
            Centroid {
                id: 4,
                vector: vec![3.0, 3.0],
            }, // Distractor 2
        ];

        // Counts: C1 is risky, C2 is safe, C3/C4 average
        let counts = vec![10, 1000, 100, 100];

        let router = Router::new(&centroids, &counts, 2, "L2").unwrap();
        let query = vec![0.0, 0.0];

        // CASE A: Strict Lambda, Low Tau.
        // Target Confidence 0.8.
        // Logic Breakdown:
        // 1. C1 (Dist 0): High prob, Low reliability. Score ~0.81. -> Selected.
        //    Cumulative confidence > 0.8. STOP.
        //
        // 2. Guardrail Check (Top-K):
        //    If K=3, it adds Top 3 Closest: C1, C3, C4.
        //    C2 is 4th closest. It should be EXCLUDED.

        let res_a = router.select_buckets(&query, 0.8, 1.0, 5.0);

        assert!(res_a.contains(&1), "Should contain closest");
        assert!(
            !res_a.contains(&2),
            "Should exclude far bucket (C2) even if huge count"
        );

        // Optional: Verify Guardrail behavior
        // If your router defaults to min_k=3, these might be present:
        // assert!(res_a.contains(&3));
        // assert!(res_a.contains(&4));
    }

    // --- MOCK DISK SEARCHER ---
    // Simulates the Storage Layer returning results from S3/Disk
    #[derive(Debug)]
    struct MockDisk {
        // Map BucketID -> List of (VectorID, Distance)
        data: RwLock<HashMap<u32, Vec<(u64, f32)>>>,
    }

    impl MockDisk {
        fn new() -> Self {
            Self {
                data: RwLock::new(HashMap::new()),
            }
        }

        fn insert(&self, bucket_id: u32, id: u64, distance: f32) {
            self.data
                .write()
                .entry(bucket_id)
                .or_default()
                .push((id, distance));
        }
    }

    #[async_trait]
    impl DiskSearcher for MockDisk {
        async fn search(
            &self,
            bucket_ids: &[u32],
            _query: &[f32],
            _k: usize,
            tv: Arc<dyn TombstoneView>, // ⚡ Update Signature
        ) -> Vec<SearchCandidate> {
            // ⚡ Return SearchCandidate
            let data = self.data.read();
            let mut results = Vec::new();
            for bid in bucket_ids {
                if let Some(items) = data.get(bid) {
                    for (id, dist) in items {
                        if !tv.contains(*id) {
                            // Mock Candidate creation
                            results.push(SearchCandidate {
                                id: *id,
                                approx_dist: *dist,
                                file_id: 0,
                                cold_offset: 0,
                                cold_length: 0,
                                index_in_rg: 0,
                                vector_count: 0,
                            });
                        }
                    }
                }
            }
            results
        }

        // Mock Refine (Identity function for mock)
        async fn refine(
            &self,
            candidates: Vec<SearchCandidate>,
            _query: &[f32],
        ) -> Vec<(u64, f32)> {
            candidates
                .into_iter()
                .map(|c| (c.id, c.approx_dist))
                .collect()
        }
    }

    // --- TEST 1: WAL INTEGRATION ---
    #[test]
    fn test_wal_durability() {
        let dir = tempdir().unwrap();
        let (index, _) = create_index(&dir, 100);

        // Insert Item
        index.insert(1, vec![1.0, 1.0]).unwrap();

        // Verify WAL file grew
        let wal_path = dir.path().join("test.wal");
        let metadata = std::fs::metadata(wal_path).unwrap();
        assert!(metadata.len() > 0, "WAL should contain data after insert");
    }

    // --- TEST 3: UNIFIED SEARCH (RAM + DISK) ---
    #[tokio::test]
    async fn test_unified_search_scatter_gather() {
        let dir = tempdir().unwrap();
        let (index, disk) = create_index(&dir, 2);

        // SCENARIO: Query near Bucket 0 ([-10, -10])
        // We want results from:
        // 1. Active MemTable (RAM)
        // 2. Frozen MemTable (RAM)
        // 3. Disk Segment (Mock)

        // A. Setup Disk Data (ID 100) -> Bucket 0
        disk.insert(0, 100, 0.1); // ID 100, Distance 0.1

        // B. Setup Frozen Data (ID 200) -> Bucket 0
        // Insert enough to rotate
        index.insert(200, vec![-10.0, -10.0]).unwrap(); // Exact match centroid
        index.insert(999, vec![10.0, 10.0]).unwrap(); // Noise (Bucket 1)
        index.insert(300, vec![-10.1, -10.1]).unwrap(); // Triggers Rotation. 300 is Active.

        // Now:
        // Disk:   [100]
        // Frozen: [200, 999]
        // Active: [300]

        // C. Execute Search
        // Query [-10, -10]. Should route to Bucket 0.
        // Should find: 100 (Disk), 200 (Frozen), 300 (Active).
        // Should ignore 999 (It's in Bucket 1, or far away in distance sorting).

        let query = vec![-10.0, -10.0];
        let results = index.search(&query, 10, 0.9, 1.0, 100.0).await.unwrap();

        // D. Verify
        let ids: Vec<u64> = results.iter().map(|(id, _)| *id).collect();

        assert!(ids.contains(&100), "Missed Disk result");
        assert!(ids.contains(&200), "Missed Frozen result");
        assert!(ids.contains(&300), "Missed Active result");

        // Check Ranking (200 is exact match dist=0, 300 is dist=0.02, 100 is dist=0.1)
        assert_eq!(results[0].0, 200, "Exact match should be first");

        // Ensure we didn't search Bucket 1 (ID 999) on disk
        // (MockDisk only returns what we asked for. If we asked for Bucket 1, we'd get nothing as we didn't seed it)
        // But 999 IS in Frozen RAM, so it *might* appear if we scan all RAM.
        // Current impl scans ALL RAM (Active+Frozen) regardless of routing.
        // So 999 will be in the list but distance will be huge.

        // Let's verify distance sorting pushed 999 to the end
        if ids.contains(&999) {
            let last = results.last().unwrap();
            assert_eq!(last.0, 999);
            assert!(last.1 > 100.0);
        }
    }

    // --- TEST 3: END-TO-END FLOW ---
    #[tokio::test]
    async fn test_end_to_end_v2_flow() {
        let dir = tempdir().unwrap();
        let (index, disk) = create_index(&dir, 2);

        // 1. Insert & Rotate
        index.insert(1, vec![1.0, 1.0]).unwrap();
        // Rotation happens here (Capacity hit)
        let rotate = index.insert(2, vec![2.0, 2.0]).unwrap();
        assert!(rotate, "Should have triggered rotation");

        // Active table
        index.insert(3, vec![3.0, 3.0]).unwrap();

        // 2. Setup Mock Disk Data (ID 999)
        disk.insert(0, 999, 0.1);

        // 3. Search (RAM + Disk)
        let results = index
            .search(&[0.0, 0.0], 10, 0.9, 1.0, 100.0)
            .await
            .unwrap();

        let ids: Vec<u64> = results.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&1)); // Frozen
        assert!(ids.contains(&2)); // Frozen
        assert!(ids.contains(&3)); // Active
        assert!(ids.contains(&999)); // Disk

        // 4. Flush
        let part = index.flush_frozen();
        assert!(part.is_some());
    }

    // // --- MOCK DISK SEARCHER ---
    // #[derive(Debug)]
    // struct MockDisk {
    //     data: RwLock<HashMap<u32, Vec<(u64, f32)>>>,
    // }

    // impl MockDisk {
    //     fn new() -> Self {
    //         Self {
    //             data: RwLock::new(HashMap::new()),
    //         }
    //     }

    //     fn insert(&self, bucket_id: u32, id: u64, distance: f32) {
    //         self.data
    //             .write()
    //             .entry(bucket_id)
    //             .or_default()
    //             .push((id, distance));
    //     }
    // }

    // --- SETUP HELPER ---
    fn create_index(dir: &tempfile::TempDir, cap: usize) -> (Arc<VectorIndexV2>, Arc<MockDisk>) {
        let dim = 2;
        let wal_path = dir.path().join("test.wal");

        let centroids = vec![
            Centroid {
                id: 0,
                vector: vec![-10.0, -10.0],
            },
            Centroid {
                id: 1,
                vector: vec![10.0, 10.0],
            },
        ];
        let counts = vec![0, 0];

        let router = Arc::new(RwLock::new(
            Router::new(&centroids, &counts, dim, "L2").unwrap(),
        ));
        let wal = Arc::new(Mutex::new(WalWriter::new(&wal_path).unwrap()));
        let disk = Arc::new(MockDisk::new());

        let index = Arc::new(VectorIndexV2::new(dim, cap, router, wal, disk.clone()));
        (index, disk)
    }

    // --- TEST: UNIFIED SEARCH WITH DELETES ---
    #[tokio::test]
    async fn test_search_respects_deletes() {
        let dir = tempdir().unwrap();
        let (index, disk) = create_index(&dir, 10);

        // 1. Setup Data
        // Disk has ID 100
        disk.insert(0, 100, 0.1);

        // MemTable has ID 200
        index.insert(200, vec![-10.0, -10.0]).unwrap();

        // 2. Verify both exist
        let results = index
            .search(&[-10.0, -10.0], 10, 0.9, 1.0, 100.0)
            .await
            .unwrap();
        assert_eq!(results.len(), 2);

        // 3. Delete ID 100 (Disk item)
        index.delete(100).unwrap();

        // 4. Verify Disk Searcher filters it out via TombstoneView
        let results_after = index
            .search(&[-10.0, -10.0], 10, 0.9, 1.0, 100.0)
            .await
            .unwrap();
        assert_eq!(results_after.len(), 1);
        assert_eq!(results_after[0].0, 200);

        // 5. Delete ID 200 (MemTable item)
        index.delete(200).unwrap();

        let results_empty = index
            .search(&[-10.0, -10.0], 10, 0.9, 1.0, 100.0)
            .await
            .unwrap();
        assert!(results_empty.is_empty());
    }

    #[tokio::test]
    async fn test_resurrection() {
        // Scenario: User deletes ID 100, realizes mistake, re-inserts ID 100.
        // The system must NOT filter out the new ID 100.
        let dir = tempdir().unwrap();
        let (index, _disk) = create_index(&dir, 10);

        // 1. Insert & Delete
        index.insert(100, vec![1.0, 1.0]).unwrap();
        index.delete(100).unwrap();

        // Verify it's gone
        let res = index.search(&[1.0, 1.0], 5, 0.0, 1.0, 100.0).await.unwrap();
        assert!(res.is_empty(), "Should be deleted");

        // 2. Resurrection (Re-insert)
        index.insert(100, vec![1.0, 1.0]).unwrap();

        // 3. Verify it's back
        let res_back = index.search(&[1.0, 1.0], 5, 0.0, 1.0, 100.0).await.unwrap();
        assert_eq!(res_back.len(), 1, "Should be resurrected");
        assert_eq!(res_back[0].0, 100);
    }

    #[tokio::test]
    async fn test_idempotent_deletes() {
        // Scenario: Sending delete command multiple times for the same ID.
        // Should not crash or cause logical errors.
        let dir = tempdir().unwrap();
        let (index, disk) = create_index(&dir, 10);

        disk.insert(0, 500, 0.1);

        // Double Delete
        index.delete(500).unwrap();
        index.delete(500).unwrap(); // Should be fine

        let res = index.search(&[0.0, 0.0], 5, 0.0, 1.0, 100.0).await.unwrap();
        assert!(res.is_empty());
    }

    #[tokio::test]
    async fn test_tombstone_view_snapshot_isolation() {
        // Scenario: We get a view for a long-running search.
        // A delete happens during the search.
        // The search should use the OLD view (Snapshot Isolation).

        let dir = tempdir().unwrap();
        let (index, _disk) = create_index(&dir, 10);

        // Data Setup
        index.insert(1, vec![1.0, 1.0]).unwrap();
        index.insert(2, vec![1.0, 1.0]).unwrap();

        // 1. Manually grab a view (Simulate start of search query)
        // We need to access the internal tracker to do this test properly,
        // or just rely on the public API behavior.
        // Let's use the public API but simulate the timing.

        // Since we can't pause the 'search' function in a unit test easily without mocks,
        // we test the Tracker behavior directly here as a proxy.
        let tracker = &index.tombstones;

        let view_before_delete = tracker.get_view();

        // 2. Perform Delete
        index.delete(1).unwrap();

        // 3. Check View (Should still see 1 as ALIVE if it was empty before)
        // Wait, 'view_before_delete' was created when 1 was NOT deleted.
        // So view.contains(1) should be FALSE.
        assert!(
            !view_before_delete.contains(1),
            "Old snapshot should not know about new delete"
        );

        // 4. New View
        let view_after = tracker.get_view();
        assert!(view_after.contains(1), "New snapshot MUST see the delete");
    }

    #[tokio::test]
    async fn test_insert_overwrites_tombstone_on_disk() {
        // Edge Case:
        // 1. ID 10 is on DISK.
        // 2. We DELETE ID 10.
        // 3. We INSERT ID 10 (moves to MemTable).
        // Result: We should find the MemTable version.
        // Trap: If we didn't unmark_delete, the DiskSearcher would ignore the old one (good),
        // but the MemTable searcher might ALSO ignore the new one (bad).

        let dir = tempdir().unwrap();
        let (index, disk) = create_index(&dir, 10);

        // 1. Disk has ID 10
        disk.insert(0, 10, 0.5); // Dist 0.5

        // 2. Delete ID 10
        index.delete(10).unwrap();

        // 3. Re-insert ID 10 (Better distance 0.1)
        index.insert(10, vec![-10.0, -10.0]).unwrap(); // Matches query perfectly

        let res = index
            .search(&[-10.0, -10.0], 5, 0.0, 1.0, 100.0)
            .await
            .unwrap();

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].0, 10);
        // We can't easily check distance here without inspecting internals,
        // but existence proves we didn't filter it out.
    }

    #[tokio::test]
    async fn test_two_stage_search_l0_l1_integration() {
        let dir = tempdir().unwrap();
        let (index, disk) = create_index(&dir, 10);

        // 1. SETUP L0 (RAM)
        // Insert ID 100: Exactly at [-10, -10] (Dist 0.0)
        index.insert(100, vec![-10.0, -10.0]).unwrap();

        // 2. SETUP L1 (MOCK DISK)
        // Insert ID 200: Slightly offset [-10.1, -10.0] (Dist ~0.01)
        // We simulate this directly in the MockDisk.
        // NOTE: Our MockDisk uses the "distance" arg as both approx and exact for simplicity.
        // In a real scenario, approx (SQ8) would differ from exact (f32).
        disk.insert(0, 200, 0.01);

        // 3. SETUP DISTRACTOR (L1)
        // Insert ID 300: Far away (Dist 100.0)
        disk.insert(0, 300, 100.0);

        // 4. EXECUTE UNIFIED SEARCH
        // Query: [-10, -10]
        // Router should select Bucket 0.
        // Expectation:
        // - RAM Search finds ID 100.
        // - Disk Search (Stage 1) finds ID 200, 300.
        // - Disk Refine (Stage 2) confirms distances for 200, 300.
        // - Merge step combines 100, 200, 300.
        // - Top-K (k=2) keeps 100 and 200.

        let results = index
            .search(
                &[-10.0, -10.0], // Query
                2,               // K = 2
                0.9,             // Target Confidence (Router)
                1.0,             // Lambda
                10.0,            // Tau
            )
            .await
            .unwrap();

        // 5. ASSERTIONS
        assert_eq!(results.len(), 2, "Should return Top-2 results");

        // First result should be ID 100 (RAM, Dist 0.0)
        assert_eq!(results[0].0, 100);
        assert!(results[0].1.abs() < 0.001);

        // Second result should be ID 200 (Disk, Dist 0.01)
        assert_eq!(results[1].0, 200);
        assert!(results[1].1 > 0.0); // Distance should be ~0.01

        // ID 300 should be truncated
    }

    // --- TEST 2: FLUSH FROZEN LOGIC ---
    #[test]
    fn test_flush_frozen_partitioning() {
        let dir = tempdir().unwrap();
        // Capacity 2: Forces rotation ON the 2nd insert.
        let (index, _) = create_index(&dir, 2);

        // 1. Fill Active Table
        index.insert(1, vec![-9.0, -9.0]).unwrap(); // Bucket 0

        // 2. Trigger Rotation
        let rotated = index.insert(2, vec![9.0, 9.0]).unwrap(); // Bucket 1
        assert!(rotated, "Should have triggered rotation at capacity limit");

        // 3. New Active Table
        let rotated_again = index.insert(3, vec![-8.0, -8.0]).unwrap(); // Bucket 0
        assert!(!rotated_again, "New table should not rotate yet");

        // 4. Flush Frozen (Peek)
        let partitions = index.flush_frozen().expect("Should return partition data");

        // 5. Verify Partitioning
        assert_eq!(partitions.len(), 2, "Should partition into 2 buckets");

        let b0 = partitions.get(&0).unwrap();
        assert!(b0.ids.contains(&1));
        assert!(!b0.ids.contains(&3)); // 3 is active, not frozen!
        assert_eq!(b0.count, 1);

        let b1 = partitions.get(&1).unwrap();
        assert!(b1.ids.contains(&2));
        assert_eq!(b1.count, 1);

        // ⚡ NEW STEP: Acknowledge Flush (Clear Memory & WAL)
        index
            .acknowledge_flush()
            .expect("Failed to acknowledge flush");

        // 6. Verify Frozen Cleared
        assert!(index.flush_frozen().is_none());
    }
}
