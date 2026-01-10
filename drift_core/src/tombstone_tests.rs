#[cfg(test)]
mod tests {
    use crate::tombstone_v2::InMemoryTombstoneTracker;
    use drift_traits::TombstoneTracker;

    #[test]
    fn test_mark_and_check_delete() {
        let tracker = InMemoryTombstoneTracker::new();

        // 1. Initially empty
        assert!(!tracker.is_deleted(1));

        // 2. Mark delete (goes to Delta)
        tracker.mark_delete(1);

        // 3. Verify visibility
        assert!(tracker.is_deleted(1));
        assert!(!tracker.is_deleted(2));
    }

    #[test]
    fn test_snapshot_isolation() {
        let tracker = InMemoryTombstoneTracker::new();

        // 1. Mark ID 100
        tracker.mark_delete(100);

        // 2. Take Snapshot (View A)
        // This triggers internal flush_delta() -> base
        let view_a = tracker.get_view();

        assert!(view_a.contains(100));
        assert!(!view_a.contains(200));

        // 3. Mark ID 200 (goes to Delta)
        tracker.mark_delete(200);

        // 4. Verify View A is UNCHANGED (Snapshot Isolation)
        assert!(view_a.contains(100));
        assert!(!view_a.contains(200), "Snapshot should not see new writes");

        // 5. Verify Tracker sees new write
        assert!(tracker.is_deleted(200));

        // 6. Take New Snapshot (View B)
        let view_b = tracker.get_view();
        assert!(view_b.contains(100));
        assert!(view_b.contains(200));
    }

    #[test]
    fn test_dissolve_batch_lifecycle() {
        let tracker = InMemoryTombstoneTracker::new();

        // Scenario:
        // We have deleted vectors 1, 2, 3.
        // We compact bucket A (containing 1 and 2).
        // We want to dissolve 1 and 2 from RAM, but keep 3.

        tracker.mark_delete(1);
        tracker.mark_delete(2);
        tracker.mark_delete(3);

        // Force flush to base (simulate get_view usage)
        let _ = tracker.get_view();

        // Dissolve 1 and 2
        tracker.dissolve_batch(&[1, 2]);

        // Verify 1 and 2 are gone from tracker (RAM freed)
        assert!(!tracker.is_deleted(1));
        assert!(!tracker.is_deleted(2));

        // Verify 3 is still there
        assert!(tracker.is_deleted(3));
    }

    #[test]
    fn test_dissolve_race_condition_safety() {
        let tracker = InMemoryTombstoneTracker::new();

        // Scenario:
        // ID 1 is in Base.
        // ID 1 is ALSO marked again in Delta (race condition or redundant delete).
        // Dissolve should clear BOTH.

        tracker.mark_delete(1);
        let _ = tracker.get_view(); // 1 moves to Base

        tracker.mark_delete(1); // 1 added to Delta again

        // Verify state
        assert!(tracker.is_deleted(1));

        // Dissolve
        tracker.dissolve_batch(&[1]);

        // Verify completely gone
        assert!(!tracker.is_deleted(1));
    }
}

#[cfg(test)]
mod edge_cases {
    use crate::index_new::VectorIndexV2;
    use crate::manifest::pb::Centroid;
    use crate::router::Router;
    use crate::wal::WalWriter;
    use async_trait::async_trait;
    use drift_traits::{DiskSearcher, TombstoneView}; // Import Trait
    use parking_lot::{Mutex, RwLock};
    use std::collections::HashMap;
    use std::sync::Arc;
    use tempfile::tempdir;

    // --- MOCK DISK SEARCHER ---
    #[derive(Debug)]
    struct MockDisk {
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
            tv: &dyn TombstoneView, // ⚡ Filter Applied Here
        ) -> Vec<(u64, f32)> {
            let data = self.data.read();
            let mut results = Vec::new();
            for bid in bucket_ids {
                if let Some(items) = data.get(bid) {
                    for (id, dist) in items {
                        // Check if deleted before returning
                        if !tv.contains(*id) {
                            results.push((*id, *dist));
                        }
                    }
                }
            }
            results
        }
    }

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
}
