// drift_core/src/memtable.rs

use hnsw_rs::prelude::*;
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};

/// Level 0: The In-Memory Graph (MemTable).
/// Stores uncompressed f32 vectors for high recall on recent data.
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

    pub fn insert(&self, id: u64, vector: &[f32]) {
        {
            let mut tombstones = self.tombstones.write();
            if tombstones.contains(&id) {
                tombstones.remove(&id);
            }
        }

        self.tombstones.write().remove(&id);

        self.data.write().insert(id, vector.to_vec());
        self.hnsw.write().insert((vector, id as usize));
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
            .map(|n| (n.d_id as u64, n.distance))
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
