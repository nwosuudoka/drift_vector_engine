#[cfg(test)]
mod tests {
    use crate::index_new::VectorIndexV2;
    use crate::manifest::pb::Centroid; // Assuming this moved or is available
    use crate::router::Router;
    use crate::wal::WalWriter;
    use async_trait::async_trait;
    use drift_traits::DiskSearcher;
    use parking_lot::{Mutex, RwLock};
    use std::collections::HashMap;
    use std::sync::Arc;
    use tempfile::tempdir;

    #[test]
    fn test_saturating_density_selection() {
        // Setup: 2 Centroids
        // C1: [0, 0] (Distance 0 to query), Count 10 (Tiny)
        // C2: [10, 10] (Distance ~14 to query), Count 1000 (Huge)

        let centroids = vec![
            Centroid {
                id: 1,
                vector: vec![0.0, 0.0],
            },
            Centroid {
                id: 2,
                vector: vec![10.0, 10.0],
            },
        ];
        let counts = vec![10, 1000]; // C1 is tiny, C2 is huge

        let router = Router::new(&centroids, &counts, 2, "L2").unwrap();
        let query = vec![0.0, 0.0];

        // CASE A: Strict Lambda, Low Tau.
        // C1 is perfect match distance-wise. C2 is far.
        // Lambda=1.0, Tau=5.
        // C1: Dist=0 -> P_geom=1.0. Reliability=1-exp(-10/5)=0.86. Score=0.86.
        // C2: Dist=14 -> P_geom~0. Reliability=1. Score~0.
        // Should pick C1.
        let res_a = router.select_buckets(&query, 0.8, 1.0, 5.0);
        assert!(res_a.contains(&1));
        assert!(!res_a.contains(&2));

        // CASE B: "Hot Zombie" / Noise suppression (The Logic Check)
        // Query is exactly at C1. But C1 has only 1 item. C2 has 1000 items.
        // Tau = 100.
        // C1: Reliability = 1 - exp(-1/100) ≈ 0.01. Score ≈ 0.01.
        // C2: If query was halfway...

        // Let's test Guardrail.
        // Even if C1 is tiny/unreliable, it is the CLOSEST.
        // The Guardrail (Step 4) forces top-3 closest.
        // So C1 MUST be returned even if density score is trash.
        let centroids_2 = vec![
            Centroid {
                id: 1,
                vector: vec![0.0, 0.0],
            }, // Closest
            Centroid {
                id: 2,
                vector: vec![100.0, 100.0],
            }, // Far
        ];
        let counts_2 = vec![1, 1000]; // 1 item vs 1000
        let router_2 = Router::new(&centroids_2, &counts_2, 2, "L2").unwrap();

        let res_guard = router_2.select_buckets(&query, 0.9, 1.0, 100.0);
        assert!(
            res_guard.contains(&1),
            "Guardrail failed to include closest bucket"
        );
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
        async fn search(&self, bucket_ids: &[u32], _query: &[f32], _k: usize) -> Vec<(u64, f32)> {
            let data = self.data.read();
            let mut results = Vec::new();
            for bid in bucket_ids {
                if let Some(items) = data.get(bid) {
                    results.extend_from_slice(items);
                }
            }
            results
        }
    }

    // --- SETUP HELPER ---
    fn create_index(dir: &tempfile::TempDir, cap: usize) -> (Arc<VectorIndexV2>, Arc<MockDisk>) {
        let dim = 2;
        let wal_path = dir.path().join("test.wal");

        // 1. Setup Router (2 Buckets: Left and Right)
        // Bucket 0: [-10, -10]
        // Bucket 1: [10, 10]
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
        let counts = vec![0, 0]; // Counts ignored for simple L2 routing
        let router = Arc::new(RwLock::new(
            Router::new(&centroids, &counts, dim, "L2").unwrap(),
        ));

        // 2. Setup Dependencies
        let wal = Arc::new(Mutex::new(WalWriter::new(&wal_path).unwrap()));
        let disk = Arc::new(MockDisk::new());

        let index = Arc::new(VectorIndexV2::new(dim, cap, router, wal, disk.clone()));
        (index, disk)
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

    // --- TEST 2: FLUSH FROZEN LOGIC ---
    #[test]
    fn test_flush_frozen_partitioning() {
        let dir = tempdir().unwrap();
        // Capacity 2: Forces rotation ON the 2nd insert.
        let (index, _) = create_index(&dir, 2);

        // 1. Fill Active Table
        index.insert(1, vec![-9.0, -9.0]).unwrap(); // Bucket 0

        // 2. Trigger Rotation
        // This insert fills capacity (len becomes 2).
        // The implementation eagerly rotates immediately.
        let rotated = index.insert(2, vec![9.0, 9.0]).unwrap(); // Bucket 1
        assert!(rotated, "Should have triggered rotation at capacity limit");

        // 3. New Active Table
        // This goes into the NEW active table (len 1). Should NOT rotate.
        let rotated_again = index.insert(3, vec![-8.0, -8.0]).unwrap(); // Bucket 0
        assert!(!rotated_again, "New table should not rotate yet");

        // 4. Flush Frozen
        // We expect [1, 2] to be flushed. [3] stays in RAM.
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

        // 6. Verify Frozen Cleared
        assert!(index.flush_frozen().is_none());
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
}
