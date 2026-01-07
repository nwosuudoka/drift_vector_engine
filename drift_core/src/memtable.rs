use parking_lot::{RwLock, RwLockReadGuard};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

/// Level 0: High-Density In-Memory Buffer.
/// Used for both Active Ingestion and Frozen Flushing.
pub struct MemTable {
    ids: RwLock<Vec<u64>>,
    data: RwLock<Vec<f32>>,
    tombstones: RwLock<HashSet<u64>>,
    dim: usize,
}

impl MemTable {
    pub fn new(capacity: usize, dim: usize, _ef: usize, _layers: usize) -> Self {
        Self {
            ids: RwLock::new(Vec::with_capacity(capacity)),
            data: RwLock::new(Vec::with_capacity(capacity * dim)),
            tombstones: RwLock::new(HashSet::new()),
            dim,
        }
    }

    pub fn insert(&self, id: u64, vector: &[f32]) {
        assert!(
            vector.len() == self.dim,
            "mismatch dims {} != {}",
            self.dim,
            vector.len()
        );
        let mut ids = self.ids.write();
        let mut data = self.data.write();
        ids.push(id);
        data.extend_from_slice(vector);
        self.tombstones.write().remove(&id);
    }

    pub fn insert_batch(&self, batch: &[(u64, Vec<f32>)]) {
        let mut ids = self.ids.write();
        let mut data = self.data.write();
        let mut tombstones = self.tombstones.write();
        for (id, vector) in batch {
            ids.push(*id);
            data.extend_from_slice(vector);
            tombstones.remove(id);
        }
    }

    pub fn delete(&self, id: u64) {
        self.tombstones.write().insert(id);
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.ids.read().len()
    }

    #[allow(clippy::type_complexity)]
    /// Accessors for the Janitor (Zero-Copy Read)
    /// Returns ReadGuards. As long as these are held, no WRITES can happen.
    /// But since this table is Frozen, writes shouldn't happen anyway.
    /// Reads (Search) can still happen concurrently.
    pub fn get_data_guards(
        &self,
    ) -> (
        RwLockReadGuard<'_, Vec<u64>>,
        RwLockReadGuard<'_, Vec<f32>>,
        RwLockReadGuard<'_, HashSet<u64>>,
    ) {
        (self.ids.read(), self.data.read(), self.tombstones.read())
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Parallel Scan Search
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        let ids = self.ids.read();
        let data = self.data.read();
        let tombstones = self.tombstones.read();
        let dim = self.dim;

        let n = ids.len();
        let chunk_size = dim;

        let final_heap = (0..n)
            .into_par_iter()
            .fold(
                || BinaryHeap::with_capacity(k + 1),
                |mut heap, i| {
                    let id = ids[i];
                    if tombstones.contains(&id) {
                        return heap;
                    }

                    let start = i * chunk_size;
                    let vector = &data[start..start + chunk_size];
                    let dist_sq = l2_sq_simd_friendly(query, vector);

                    let item = HeapItem {
                        distance: OrderedFloat(dist_sq),
                        id,
                    };

                    if heap.len() < k {
                        heap.push(item);
                    } else if item.distance < heap.peek().unwrap().distance {
                        heap.pop();
                        heap.push(item);
                    }
                    heap
                },
            )
            .reduce(
                || BinaryHeap::with_capacity(k),
                |mut a, b| {
                    for item in b {
                        if a.len() < k {
                            a.push(item);
                        } else if item.distance < a.peek().unwrap().distance {
                            a.pop();
                            a.push(item);
                        }
                    }
                    a
                },
            );

        let mut result: Vec<(u64, f32)> = final_heap
            .into_iter()
            .map(|item| (item.id, item.distance.0))
            .collect();
        result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        result
    }
}

// Helpers
#[inline(always)]
pub fn l2_sq_simd_friendly(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum
}

#[derive(Debug, PartialEq)]
struct OrderedFloat(f32);
impl Eq for OrderedFloat {}
impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}
impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Less)
    }
}
struct HeapItem {
    distance: OrderedFloat,
    id: u64,
}
impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}
impl Eq for HeapItem {}
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}
impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.cmp(&other.distance)
    }
}
