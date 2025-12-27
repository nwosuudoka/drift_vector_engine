// drift_core/src/memtable.rs

use hnsw_rs::prelude::*;
use parking_lot::RwLock;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use tracing::{Level, instrument, span};

/// Wrapper to allow f32 in BinaryHeap (Max-Heap)
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
        // Handle NaNs by pushing them to the end
        self.partial_cmp(other).unwrap_or(Ordering::Less)
    }
}

/// Helper struct to keep track of top-k candidates in the heap
#[derive(Debug, PartialEq, Eq)]
struct HeapItem {
    distance: OrderedFloat,
    id: u64,
}

// Order by distance so BinaryHeap acts as a Max-Heap on distance.
// We keep the K smallest items. If new_item < max_in_heap, replace max.
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

/// Level 0: The In-Memory Buffer (MemTable).
///
/// Stores raw f32 vectors.
/// - Writes are O(1) (HashMap insert).
/// - Reads are O(N) (Linear Scan).
///
/// The HNSW field is kept for compatibility but is NOT updated synchronously.
pub struct MemTable {
    // Hnsw is thread-safe for searching, but we wrap our secondary storage in locks
    pub hnsw: RwLock<Hnsw<'static, f32, DistL2>>,
    #[allow(dead_code)]
    capacity: usize,
    #[allow(dead_code)]
    dim: usize,

    // Stores raw data for flushing. Wrapped in RwLock for concurrent access.
    data: RwLock<HashMap<u64, Vec<f32>>>,
    tombstones: RwLock<HashSet<u64>>,
}

impl MemTable {
    pub fn new(capacity: usize, dim: usize, ef_construction: usize, max_layers: usize) -> Self {
        let hnsw = Hnsw::new(
            16, // max_nb_connection (M)
            capacity,
            max_layers,
            ef_construction,
            DistL2,
        );

        Self {
            hnsw: RwLock::new(hnsw),
            capacity,
            dim,
            data: RwLock::new(HashMap::new()),
            tombstones: RwLock::new(HashSet::new()),
        }
    }

    // Inside impl MemTable { ... }
    pub fn insert(&self, id: u64, vector: &[f32]) {
        // 1. Tombstone Logic
        {
            // "entered()" starts the timer immediately
            let _span = span!(Level::TRACE, "lock_tombstones").entered();
            let mut tombstones = self.tombstones.write();
            if tombstones.contains(&id) {
                tombstones.remove(&id);
            }
        }

        // 2. Data Map Insert
        {
            let _span = span!(Level::TRACE, "lock_data_map").entered();
            // This measures waiting for the lock + writing the data
            self.data.write().insert(id, vector.to_vec());
        }

        // 3. HNSW Insert (The likely bottleneck)
        {
            let _span = span!(Level::INFO, "lock_hnsw_insert").entered();
            // This is the heavy operation.
            // If this span takes 1.5ms, it's CPU bound.
            // If it takes 500ms, it means multiple threads are fighting for this lock.
            self.hnsw.write().insert((vector, id as usize));
        }
    }

    // 👇 ADD THIS NEW METHOD
    pub fn insert_batch(&self, batch: &[(u64, Vec<f32>)]) {
        // 1. Lock Tombstones ONCE
        {
            let _span = span!(Level::TRACE, "lock_tombstones_batch").entered();
            let mut tombstones = self.tombstones.write();
            for (id, _) in batch {
                if tombstones.contains(id) {
                    tombstones.remove(id);
                }
            }
        }

        // 2. Lock Data Map ONCE
        {
            let _span = span!(Level::TRACE, "lock_data_map_batch").entered();
            let mut data = self.data.write();
            for (id, vector) in batch {
                data.insert(*id, vector.clone());
            }
        }

        // 3. Lock HNSW ONCE (Critical Optimization)

        {
            let _span = span!(Level::INFO, "lock_hnsw_batch").entered();
            let hnsw = self.hnsw.write();

            let capacity = self.capacity; // Get the configured capacity
            let current_count = hnsw.get_nb_point();

            // ⚡ SAFETY CHECK:
            // If we are about to exceed capacity, log a warning or panic.
            if current_count + batch.len() > capacity {
                tracing::error!(
                    "CRITICAL: MemTable HNSW capacity exceeded! Cap: {}, Current: {}, Batch: {}",
                    capacity,
                    current_count,
                    batch.len()
                );
                // In a real DB, you might trigger an emergency flush here.
                // For now, let's panic so tests fail loudly instead of silently.
                if cfg!(test) {
                    panic!("MemTable Capacity Exceeded");
                }
            }

            for (id, vector) in batch {
                hnsw.insert((vector, *id as usize));
            }
        }
    }

    pub fn delete(&self, id: u64) {
        let mut set = self.tombstones.write();
        set.insert(id);

        // Also remove from data map so it doesn't get flushed to disk
        let mut data = self.data.write();
        data.remove(&id);
    }

    // Update search to filter results
    pub fn search(&self, query: &[f32], k: usize, ef_search: usize) -> Vec<(u64, f32)> {
        let tombstones = self.tombstones.read();
        self.hnsw
            .read()
            .search(query, k + tombstones.len(), ef_search) // Ask for more to account for filtering
            .into_iter()
            .map(|n| (n.d_id as u64, n.distance * n.distance))
            .filter(|(id, _)| !tombstones.contains(id)) // Filter
            .take(k)
            .collect()
    }

    pub fn len(&self) -> usize {
        self.hnsw.read().get_nb_point()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn extract_all(&self) -> Vec<(u64, Vec<f32>)> {
        let guard = self.tombstones.read();
        self.data
            .read()
            .iter()
            .filter(|(id, _)| !guard.contains(id))
            .map(|(k, v)| (*k, v.clone()))
            .collect()
    }
}
