#[cfg(test)]
mod tests {
    use crate::memtable::MemTable;
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::Duration;

    const DIM: usize = 2;

    // --- 1. Basic Correctness Tests ---

    #[test]
    fn test_memtable_basic_crud() {
        let memtable = MemTable::new(100, DIM, 16, 16);

        // A. Insert
        memtable.insert(1, &[10.0, 10.0]); // Far
        memtable.insert(2, &[0.0, 0.0]); // Target
        memtable.insert(3, &[0.1, 0.1]); // Close to Target

        // B. Search (Expect ID 2 and 3)
        let results = memtable.search(&[0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 2, "ID 2 should be exact match");
        assert_eq!(results[1].0, 3, "ID 3 should be second closest");

        // C. Delete
        memtable.delete(2);

        // D. Search again (ID 2 should be gone)
        let results_after = memtable.search(&[0.0, 0.0], 2);
        assert_eq!(results_after.len(), 2);
        assert_eq!(results_after[0].0, 3, "ID 3 promoted to first");
        assert_eq!(results_after[1].0, 1, "ID 1 included to fill K");
    }

    #[test]
    fn test_flat_buffer_alignment() {
        // Verify that the contiguous buffer logic actually packs floats correctly
        const TEST_DIM: usize = 4;
        let memtable = MemTable::new(10, TEST_DIM, 16, 16);

        memtable.insert(100, &[1.0, 2.0, 3.0, 4.0]);
        memtable.insert(101, &[5.0, 6.0, 7.0, 8.0]);

        // Manually inspect the lock-protected data
        let (_, data, _) = memtable.get_data_guards();

        // Vector 1
        assert_eq!(data[0], 1.0);
        assert_eq!(data[3], 4.0);

        // Vector 2 starts immediately after
        assert_eq!(data[4], 5.0);
        assert_eq!(data[7], 8.0);
    }

    // --- 2. Zero-Copy Architecture Tests ---

    #[test]
    fn test_freeze_is_cheap_pointer_copy() {
        let memtable = Arc::new(MemTable::new(1000, DIM, 16, 16));

        // Insert some data
        for i in 0..100 {
            memtable.insert(i, &[0.0, 0.0]);
        }

        // "Freeze" the memtable (Simulating Index::rotate_and_freeze)
        // This is just an Arc clone.
        let frozen_ref = memtable.clone();

        // 1. Verify Data Visibility
        assert_eq!(frozen_ref.len(), 100);

        // 2. Verify Shared Underlying Memory
        // Modifying the "active" handle (if we allowed it) would be visible
        // to the "frozen" handle because they point to the same RwLocks.
        // In the real architecture, we drop the 'active' pointer from the Index
        // after cloning, so writes stop going to this instance.

        // Let's verify they are indeed the same object in memory
        assert!(Arc::ptr_eq(&memtable, &frozen_ref));
    }

    #[test]
    fn test_concurrent_janitor_and_search_access() {
        // This is the CRITICAL test for the "Shared Frozen State" architecture.
        // It proves that the Janitor reading data (to flush) does NOT block Search.

        let memtable = Arc::new(MemTable::new(100, DIM, 16, 16));
        for i in 0..50 {
            memtable.insert(i, &[i as f32, i as f32]);
        }

        let janitor_ref = memtable.clone();
        let search_ref = memtable.clone();

        let barrier = Arc::new(Barrier::new(2));
        let barrier_c = barrier.clone();

        // --- Thread A: Janitor (Flush) ---
        let janitor_handle = thread::spawn(move || {
            // 1. Acquire Read Lock (Simulating extraction for flush)
            let (_ids, _vecs, _) = janitor_ref.get_data_guards();

            // 2. Signal we have the lock
            barrier_c.wait();

            // 3. Sleep to simulate heavy K-Means/Compression
            thread::sleep(Duration::from_millis(500));

            // Lock is held this entire time!
        });

        // --- Thread B: Searcher ---
        // Wait for Janitor to acquire lock
        barrier.wait();

        // Try to Search IMMEDIATELY.
        // If RwLock implementation is correct, this should NOT block.
        let start = std::time::Instant::now();

        let results = search_ref.search(&[0.0, 0.0], 5);

        let duration = start.elapsed();

        assert_eq!(results.len(), 5);

        // Assert it was fast (non-blocking).
        // If it blocked, duration would be > 500ms.
        assert!(
            duration < Duration::from_millis(50),
            "Search was blocked by Janitor!"
        );

        janitor_handle.join().unwrap();
    }

    #[test]
    fn test_tombstones_propagate_to_shared_view() {
        let memtable = Arc::new(MemTable::new(100, DIM, 16, 16));
        memtable.insert(1, &[10.0, 10.0]);

        let frozen_view = memtable.clone();

        // User deletes ID 1
        memtable.delete(1);

        // The "Frozen" view (used by search) should immediately see the tombstone
        // because it shares the same RwLock<HashSet>.
        let results = frozen_view.search(&[10.0, 10.0], 1);

        assert!(results.is_empty(), "Frozen view failed to see deletion");
    }
}
