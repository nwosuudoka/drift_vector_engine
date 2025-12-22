// drift_core/src/memtable.rs

use hnsw_rs::prelude::*;
use std::{
    collections::{HashMap, HashSet},
    sync::RwLock,
};

/// Level 0: The In-Memory Graph (MemTable).
/// Stores uncompressed f32 vectors for high recall on recent data.
pub struct MemTable {
    // Hnsw is thread-safe for searching, but we wrap our secondary storage in locks
    pub hnsw: Hnsw<'static, f32, DistL2>,
    capacity: usize,
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
            hnsw,
            capacity,
            dim,
            data: RwLock::new(HashMap::new()),
            tombstones: RwLock::new(HashSet::new()),
        }
    }

    pub fn insert(&self, id: u64, vector: &[f32]) {
        // 1. Insert into HNSW (Search Graph)
        // hnsw_rs insert signature: (&[T], usize)
        self.hnsw.insert((vector, id as usize));

        // 2. Insert into Data Map (Storage for Flush)
        let mut map = self.data.write().unwrap();
        map.insert(id, vector.to_vec());
    }

    pub fn delete(&self, id: u64) {
        let mut set = self.tombstones.write().unwrap();
        set.insert(id);

        // Also remove from data map so it doesn't get flushed to disk
        let mut data = self.data.write().unwrap();
        data.remove(&id);
    }

    // Update search to filter results
    pub fn search(&self, query: &[f32], k: usize, ef_search: usize) -> Vec<(u64, f32)> {
        let tombstones = self.tombstones.read().unwrap();
        self.hnsw
            .search(query, k + tombstones.len(), ef_search) // Ask for more to account for filtering
            .into_iter()
            .map(|n| (n.d_id as u64, n.distance))
            .filter(|(id, _)| !tombstones.contains(id)) // Filter
            .take(k)
            .collect()
    }

    // pub fn search(&self, query: &[f32], k: usize, ef_search: usize) -> Vec<(u64, f32)> {
    //     self.hnsw
    //         .search(query, k, ef_search)
    //         .into_iter()
    //         .map(|n| (n.d_id as u64, n.distance)) // hnsw_rs returns distance, not squared
    //         .collect()
    // }

    pub fn len(&self) -> usize {
        self.hnsw.get_nb_point()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Extract all data for flushing.
    /// Returns a copy of the data so the MemTable remains valid for
    /// any straggling readers until it is fully dropped.
    pub fn extract_all(&self) -> Vec<(u64, Vec<f32>)> {
        let map = self.data.read().unwrap();
        map.iter().map(|(k, v)| (*k, v.clone())).collect()
    }
}
