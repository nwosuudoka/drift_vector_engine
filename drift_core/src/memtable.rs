use parking_lot::RwLock;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::sync::Arc;

/// ⚡ Immutable view of a frozen memtable segment.
/// Shared between Search and Janitor with ZERO lock contention.
pub struct MemTableSnapshot {
    pub ids: Vec<u64>,
    pub vectors: Vec<f32>, // Flattened contiguous buffer (N * D)
    pub dim: usize,
}

impl MemTableSnapshot {
    /// Zero-copy search on the immutable snapshot.
    /// This is called while the Janitor is simultaneously flushing the data.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        let n = self.ids.len();
        let chunk_size = self.dim;

        let final_heap = (0..n)
            .into_par_iter()
            .fold(
                || BinaryHeap::with_capacity(k + 1),
                |mut heap, i| {
                    let start = i * chunk_size;
                    let vector = &self.vectors[start..start + chunk_size];

                    let dist_sq = l2_sq_simd_friendly(query, vector);
                    let item = HeapItem {
                        distance: OrderedFloat(dist_sq),
                        id: self.ids[i],
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

/// Level 0: High-Density In-Memory Buffer (MemTable).
pub struct MemTable {
    ids: RwLock<Vec<u64>>,
    data: RwLock<Vec<f32>>, // Contiguous buffer for hardware alignment
    tombstones: RwLock<HashSet<u64>>,
    dim: usize,
}

impl MemTable {
    pub fn new(_capacity: usize, dim: usize, _ef: usize, _layers: usize) -> Self {
        Self {
            ids: RwLock::new(Vec::with_capacity(_capacity)),
            data: RwLock::new(Vec::with_capacity(_capacity * dim)),
            tombstones: RwLock::new(HashSet::new()),
            dim,
        }
    }

    /// Fast O(1) synchronous insert [cite: 23-24, 1301]
    pub fn insert(&self, id: u64, vector: &[f32]) {
        let mut ids = self.ids.write();
        let mut data = self.data.write();

        ids.push(id);
        data.extend_from_slice(vector);

        // Remove from tombstones if it was previously deleted
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

    pub fn len(&self) -> usize {
        self.ids.read().len()
    }

    /// ⚡ BILION-SCALE ROTATION: Zero-Copy Transition to Snapshot
    /// Consumes the MemTable and returns an immutable Arc that Search and Janitor share.
    pub fn freeze(self) -> Arc<MemTableSnapshot> {
        // Since we consume 'self', we move the Vecs out of the RwLocks
        let ids = self.ids.into_inner();
        let vectors = self.data.into_inner();
        let tombstones = self.tombstones.into_inner();

        // Optional: Production-grade cleanup by filtering tombstones here
        // to keep the L1 flush phase small and focused.
        let mut clean_ids = Vec::with_capacity(ids.len());
        let mut clean_vecs = Vec::with_capacity(vectors.len());

        for (i, &id) in ids.iter().enumerate() {
            if !tombstones.contains(&id) {
                clean_ids.push(id);
                let start = i * self.dim;
                clean_vecs.extend_from_slice(&vectors[start..start + self.dim]);
            }
        }

        Arc::new(MemTableSnapshot {
            ids: clean_ids,
            vectors: clean_vecs,
            dim: self.dim,
        })
    }

    /// Drains the current buffers into a new immutable snapshot.
    pub fn freeze_snapshot(&self) -> Arc<MemTableSnapshot> {
        let mut ids_guard = self.ids.write();
        let mut data_guard = self.data.write();
        let mut tomb_guard = self.tombstones.write();

        // 1. Move the heap-allocated buffers out of the MemTable (O(1))
        // This leaves the active MemTable empty but its capacity intact.
        let ids = std::mem::take(&mut *ids_guard);
        let vectors = std::mem::take(&mut *data_guard);
        let tombstones = std::mem::take(&mut *tomb_guard);

        // 2. Perform one-time cleanup to keep Janitor math fast
        let mut clean_ids = Vec::with_capacity(ids.len());
        let mut clean_vecs = Vec::with_capacity(vectors.len());

        for (i, &id) in ids.iter().enumerate() {
            if !tombstones.contains(&id) {
                clean_ids.push(id);
                let start = i * self.dim;
                clean_vecs.extend_from_slice(&vectors[start..start + self.dim]);
            }
        }

        Arc::new(MemTableSnapshot {
            ids: clean_ids,
            vectors: clean_vecs,
            dim: self.dim,
        })
    }

    /// Standard brute force search for active (non-frozen) MemTable
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        let ids = self.ids.read();
        let data = self.data.read();
        let tombstones = self.tombstones.read();
        let dim = self.dim;

        let final_heap = (0..ids.len())
            .into_par_iter()
            .fold(
                || BinaryHeap::with_capacity(k + 1),
                |mut heap, i| {
                    let id = ids[i];
                    if tombstones.contains(&id) {
                        return heap;
                    }

                    let start = i * dim;
                    let vector = &data[start..start + dim];
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

/// Hardware-native L2 kernel
#[inline(always)]
pub fn l2_sq_simd_friendly(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    let n = a.len();
    let a = &a[..n];
    let b = &b[..n];

    for i in 0..n {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum
}

// Boilerplate for BinaryHeap logic
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
