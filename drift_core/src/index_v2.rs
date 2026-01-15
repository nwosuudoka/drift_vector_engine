use crate::memtable_v2::{MemTableOptions, MemTableV2};
use crate::partitioner::{IncrementalPartitioner, PartitionGroup, PartitionResult};
use crate::router::Router;
use crate::tombstone_v2::InMemoryTombstoneTracker;
use crate::wal_v2::WalManager;
use drift_traits::{DiskSearcher, TombstoneTracker};
use parking_lot::{Mutex, RwLock};
use std::cmp::Ordering;
use std::collections::HashSet;
use std::io;
use std::sync::Arc;
use tokio::task;

#[derive(Clone)]
pub struct FrozenTable {
    pub table: Arc<MemTableV2>,
    pub wal_id: u64,
}

pub struct VectorIndex {
    active: RwLock<Arc<MemTableV2>>,
    frozen: RwLock<Vec<FrozenTable>>,
    wal_manager: Arc<Mutex<WalManager>>,
    disk: Arc<dyn DiskSearcher>,
    pub(crate) tombstones: Arc<dyn TombstoneTracker>,
    router: Arc<RwLock<Router>>,
    dim: usize,
    capacity: usize,
}

impl VectorIndex {
    pub fn new(
        dim: usize,
        capacity: usize,
        router: Arc<RwLock<Router>>,
        wal_manager: Arc<Mutex<WalManager>>,
        disk: Arc<dyn DiskSearcher>,
    ) -> Self {
        Self {
            active: RwLock::new(Arc::new(MemTableV2::new(MemTableOptions { capacity, dim }))),
            frozen: RwLock::new(Vec::new()),
            wal_manager,
            disk,
            router,
            dim,
            capacity,
            tombstones: Arc::new(InMemoryTombstoneTracker::new()),
        }
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) -> io::Result<bool> {
        {
            let mut mgr = self.wal_manager.lock();
            mgr.current().write_insert(id, &vector)?;
        }

        let active_ptr = { self.active.read().clone() };
        let needs_rotate = active_ptr.insert(id, &vector);
        self.tombstones.unmark_delete(id);

        if needs_rotate {
            return self.rotate_active();
        }
        Ok(false)
    }

    pub fn insert_batch(&self, batch: &[(u64, Vec<f32>)]) -> io::Result<bool> {
        if batch.is_empty() {
            return Ok(false);
        }

        {
            let mut mgr = self.wal_manager.lock();
            let current = mgr.current();
            let tx_id = current.begin_transaction()?;
            for (id, vector) in batch {
                current.write_insert(*id, vector)?;
            }
            current.commit_transaction(tx_id)?;
        }

        let ids: Vec<u64> = batch.iter().map(|(id, _)| *id).collect();
        self.tombstones.unmark_delete_batch(&ids);

        let active_ptr = { self.active.read().clone() };
        let needs_rotate = active_ptr.insert_batch(batch);

        if needs_rotate {
            return self.rotate_active();
        }
        Ok(false)
    }

    fn rotate_active(&self) -> io::Result<bool> {
        let mut active_guard = self.active.write();
        if active_guard.len() < self.capacity {
            return Ok(false);
        }

        let mut frozen_guard = self.frozen.write();
        let mut wal_guard = self.wal_manager.lock();

        let old_wal_id = wal_guard.rotate()?; // ⚡ Rotate WAL

        let old_table = active_guard.clone();
        frozen_guard.push(FrozenTable {
            table: old_table,
            wal_id: old_wal_id,
        });

        *active_guard = Arc::new(MemTableV2::new(MemTableOptions {
            capacity: self.capacity,
            dim: self.dim,
        }));

        Ok(true)
    }

    pub fn delete(&self, id: u64) -> io::Result<()> {
        self.wal_manager.lock().current().write_delete(id)?;
        self.tombstones.mark_delete(id);
        Ok(())
    }

    pub async fn search(
        &self,
        query: &[f32],
        k: usize,
        target: f32,
        lambda: f32,
        tau: f32,
    ) -> io::Result<Vec<(u64, f32)>> {
        let (ram_tables, view) = {
            let active = self.active.read();
            let frozen = self.frozen.read();
            let view = self.tombstones.get_view();
            let mut tables = vec![active.clone()];
            tables.extend(frozen.iter().map(|f| f.table.clone()));
            (tables, view)
        };

        let ram_results_raw = task::spawn_blocking({
            let query = query.to_vec();
            move || {
                let mut results = Vec::new();
                for table in ram_tables {
                    results.extend(table.search(&query, k));
                }
                results
            }
        })
        .await
        .map_err(io::Error::other)?;

        let bucket_ids = self
            .router
            .read()
            .select_buckets(query, target, lambda, tau);
        let oversample_factor = k * 3;
        // ⚡ ATOMIC DISK SEARCH
        // We pass 'k' as the final target, but 'oversample_factor' for the internal scan
        let disk_results = self
            .disk
            .search_and_refine(&bucket_ids, query, k, oversample_factor, view.clone())
            .await;

        let mut final_results = Vec::new();
        let mut seen = HashSet::new();

        // Merge RAM (already refined/exact) + Disk (now refined/exact)
        for (id, dist) in ram_results_raw {
            if !view.contains(id) && seen.insert(id) {
                final_results.push((id, dist));
            }
        }
        for (id, dist) in disk_results {
            if seen.insert(id) && !view.contains(id) {
                final_results.push((id, dist));
            }
        }

        final_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        if final_results.len() > k {
            final_results.truncate(k);
        }
        Ok(final_results)
    }

    pub fn flush_frozen(&self) -> Option<(PartitionResult, Vec<u64>)> {
        let tables_to_flush = {
            let f = self.frozen.read();
            if f.is_empty() {
                return None;
            }
            f.clone()
        };

        let router_guard = self.router.read();
        let mut global_partition = std::collections::HashMap::new();
        let mut wal_ids = Vec::new();

        for ft in tables_to_flush {
            wal_ids.push(ft.wal_id);
            let (ids, flat_vecs) = ft.table.snapshot();
            let part = IncrementalPartitioner::partition(&ids, &flat_vecs, self.dim, &router_guard);

            for (bucket_id, group) in part {
                let entry = global_partition
                    .entry(bucket_id)
                    .or_insert_with(|| PartitionGroup::new(self.dim, group.centroid.clone()));
                if entry.centroid.is_none() {
                    entry.centroid = group.centroid;
                }
                entry.ids.extend(group.ids);
                entry.flat_vectors.extend(group.flat_vectors);
                entry.count += group.count;
            }
        }
        Some((global_partition, wal_ids))
    }

    pub fn acknowledge_flush(&self, flushed_wal_ids: &[u64]) -> io::Result<()> {
        {
            let mut f = self.frozen.write();
            f.retain(|ft| !flushed_wal_ids.contains(&ft.wal_id));
        }
        let mgr = self.wal_manager.lock();
        for &id in flushed_wal_ids {
            mgr.delete_wal(id)?;
        }
        Ok(())
    }

    pub fn get_dim(&self) -> usize {
        self.dim
    }
}
