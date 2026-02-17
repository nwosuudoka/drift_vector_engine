#[cfg(test)]
mod tests {
    use crate::memtable::{MemTable, MemTableOptions};
    use drift_traits::TombstoneView;
    use std::collections::HashSet;
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::{Duration, Instant};

    #[derive(Debug, Default)]
    struct TestView {
        ids: HashSet<u64>,
    }

    impl TestView {
        fn with_ids(ids: &[u64]) -> Self {
            let mut view = Self::default();
            for id in ids {
                view.ids.insert(*id);
            }
            view
        }
    }

    impl TombstoneView for TestView {
        fn contains(&self, id: u64) -> bool {
            self.ids.contains(&id)
        }

        fn len(&self) -> usize {
            self.ids.len()
        }
    }

    fn make_table(capacity: usize, dim: usize) -> MemTable {
        MemTable::new(MemTableOptions { capacity, dim })
    }

    #[test]
    fn test_empty_and_len() {
        let table = make_table(4, 2);
        assert!(table.is_empty());
        assert_eq!(table.len(), 0);
    }

    #[test]
    fn test_insert_capacity_threshold() {
        let table = make_table(3, 2);

        assert!(!table.insert(1, &[0.0, 0.0]));
        assert!(!table.insert(2, &[1.0, 1.0]));
        assert!(table.insert(3, &[2.0, 2.0]));

        assert_eq!(table.len(), 3);
        assert!(!table.is_empty());
    }

    #[test]
    #[should_panic]
    fn test_insert_dim_mismatch_panics() {
        let table = make_table(2, 2);
        table.insert(1, &[0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_insert_batch_empty_noop() {
        let table = make_table(4, 2);
        let batch: Vec<(u64, Vec<f32>)> = Vec::new();
        assert!(!table.insert_batch(&batch));
        assert_eq!(table.len(), 0);
    }

    #[test]
    fn test_insert_batch_capacity_threshold() {
        let table = make_table(5, 3);

        let batch_a = vec![
            (1, vec![0.0, 0.0, 0.0]),
            (2, vec![1.0, 1.0, 1.0]),
            (3, vec![2.0, 2.0, 2.0]),
            (4, vec![3.0, 3.0, 3.0]),
        ];
        assert!(!table.insert_batch(&batch_a));
        assert_eq!(table.len(), 4);

        let batch_b = vec![(5, vec![4.0, 4.0, 4.0]), (6, vec![5.0, 5.0, 5.0])];
        assert!(table.insert_batch(&batch_b));
        assert_eq!(table.len(), 6);
    }

    #[test]
    fn test_data_layout_ids_and_vectors() {
        let table = make_table(4, 2);
        table.insert(10, &[1.0, 2.0]);
        table.insert(20, &[3.0, 4.0]);

        let (ids, data) = table.get_data_guards();
        assert_eq!(ids.as_slice(), &[10, 20]);
        assert_eq!(data.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(data.len(), ids.len() * 2);
    }

    #[test]
    fn test_snapshot_isolated_copy() {
        let table = make_table(10, 2);
        table.insert(1, &[0.0, 0.0]);
        table.insert(2, &[1.0, 1.0]);

        let (ids_snapshot, data_snapshot) = table.snapshot();
        assert_eq!(ids_snapshot, vec![1, 2]);
        assert_eq!(data_snapshot, vec![0.0, 0.0, 1.0, 1.0]);

        table.insert(3, &[2.0, 2.0]);
        let (ids_after, data_after) = table.snapshot();

        assert_eq!(ids_snapshot, vec![1, 2]);
        assert_eq!(data_snapshot, vec![0.0, 0.0, 1.0, 1.0]);
        assert_eq!(ids_after, vec![1, 2, 3]);
        assert_eq!(data_after, vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);
    }

    #[test]
    fn test_search_orders_by_distance() {
        let table = make_table(10, 2);
        table.insert(1, &[0.0, 0.0]);
        table.insert(2, &[1.0, 1.0]);
        table.insert(3, &[2.0, 2.0]);

        let view = TestView::default();
        let results = table.search(&[0.0, 0.0], 3, &view);
        let ids: Vec<u64> = results.iter().map(|(id, _)| *id).collect();

        assert_eq!(ids, vec![1, 2, 3]);
        assert!(results[0].1 <= results[1].1);
        assert!(results[1].1 <= results[2].1);
    }

    #[test]
    fn test_search_k_greater_than_len() {
        let table = make_table(10, 2);
        table.insert(1, &[0.0, 0.0]);
        table.insert(2, &[1.0, 1.0]);

        let view = TestView::default();
        let results = table.search(&[0.5, 0.5], 10, &view);

        assert_eq!(results.len(), 2);
    }

    #[test]
    #[should_panic]
    fn test_search_k_zero_panics() {
        let table = make_table(10, 2);
        table.insert(1, &[0.0, 0.0]);
        let view = TestView::default();
        let _ = table.search(&[0.0, 0.0], 0, &view);
    }

    #[test]
    fn test_search_respects_tombstones() {
        let table = make_table(10, 2);
        table.insert(10, &[0.0, 0.0]);
        table.insert(20, &[1.0, 1.0]);
        table.insert(30, &[2.0, 2.0]);

        let view = TestView::with_ids(&[20]);
        let results = table.search(&[0.0, 0.0], 3, &view);
        let ids: Vec<u64> = results.iter().map(|(id, _)| *id).collect();

        assert_eq!(ids, vec![10, 30]);
    }

    #[test]
    fn test_get_data_guards_allows_concurrent_reads() {
        let table = Arc::new(make_table(100, 2));
        for i in 0..50 {
            table.insert(i, &[i as f32, i as f32]);
        }

        let reader = table.clone();
        let searcher = table.clone();
        let barrier = Arc::new(Barrier::new(2));
        let barrier_a = barrier.clone();

        let handle = thread::spawn(move || {
            let (_ids, _data) = reader.get_data_guards();
            barrier_a.wait();
            thread::sleep(Duration::from_millis(200));
        });

        barrier.wait();
        let view = TestView::default();
        let start = Instant::now();
        let results = searcher.search(&[0.0, 0.0], 5, &view);
        let elapsed = start.elapsed();

        assert_eq!(results.len(), 5);
        assert!(elapsed < Duration::from_millis(50));

        handle.join().unwrap();
    }

    #[test]
    fn test_insert_batch_then_single_insert() {
        let table = make_table(10, 2);
        let batch = vec![(1, vec![0.0, 0.0]), (2, vec![1.0, 1.0])];
        assert!(!table.insert_batch(&batch));

        assert!(!table.insert(3, &[2.0, 2.0]));

        let (ids, data) = table.get_data_guards();
        assert_eq!(ids.as_slice(), &[1, 2, 3]);
        assert_eq!(data.as_slice(), &[0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);
    }
}
