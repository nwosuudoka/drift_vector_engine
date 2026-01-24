use crate::memtable_v2::{MemTableOptions, MemTableV2};
use crate::partitioner::{IncrementalPartitioner, PartitionGroup, PartitionResult};
use crate::router::Router;
use crate::wal_v2::WalManager;
use drift_kv::bitstore::BitStore;
use drift_traits::{StorageEngine, TombstoneView};
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

#[derive(Debug, PartialEq)]
pub enum MaintenanceStatus {
    Completed,
    SkippedSingularity { variance: f32 }, // ⚡ NEW: Too dense to split
    SkippedTooSmall,                      // Below split threshold
    SkippedLocked,                        // Currently being read/written
}

impl MaintenanceStatus {
    pub fn to_str(&self) -> String {
        match self {
            MaintenanceStatus::Completed => "Completed".to_string(),
            MaintenanceStatus::SkippedSingularity { variance } => {
                format!("Skipped Singularity (Var: {:.4})", variance)
            }
            MaintenanceStatus::SkippedTooSmall => "Skipped Too Small".to_string(),
            MaintenanceStatus::SkippedLocked => "Skipped Locked".to_string(),
        }
    }
}

pub struct SplitProposal {
    pub target_bucket: u32,
    pub left: PartitionGroup,
    pub right: PartitionGroup,
    pub loopback: Vec<(u64, Vec<f32>)>,
}

// Helper View for L0
#[derive(Debug, Clone)]
pub struct L0TombstoneView {
    inner: Arc<HashSet<u64>>,
}
impl TombstoneView for L0TombstoneView {
    fn contains(&self, id: u64) -> bool {
        self.inner.contains(&id)
    }
    fn len(&self) -> usize {
        self.inner.len()
    }
}

pub struct VectorIndex {
    active: RwLock<Arc<MemTableV2>>,
    frozen: RwLock<Vec<FrozenTable>>,
    wal_manager: Arc<Mutex<WalManager>>,
    storage: Arc<dyn StorageEngine>,
    kv: Arc<BitStore>,
    router: Arc<RwLock<Router>>,
    deleted_ids: RwLock<Arc<HashSet<u64>>>,
    dim: usize,
    capacity: usize,
}

impl VectorIndex {
    pub fn new(
        dim: usize,
        capacity: usize,
        router: Arc<RwLock<Router>>,
        wal_manager: Arc<Mutex<WalManager>>,
        storage: Arc<dyn StorageEngine>,
        kv: Arc<BitStore>,
    ) -> Self {
        Self {
            active: RwLock::new(Arc::new(MemTableV2::new(MemTableOptions { capacity, dim }))),
            frozen: RwLock::new(Vec::new()),
            deleted_ids: RwLock::new(Arc::new(HashSet::new())),
            wal_manager,
            storage,
            kv,
            router,
            dim,
            capacity,
        }
    }

    pub fn insert(&self, id: u64, vector: &[f32]) -> io::Result<bool> {
        // 1. WAL (Durability First)
        // If this fails, we return Error and state is unchanged.
        {
            let mut mgr = self.wal_manager.lock();
            mgr.current().write_insert(id, vector)?;
        }

        // 2. Shadowing (Side Effect)
        // We do this BEFORE L0 insert so that search queries don't see duplicates.
        // If this fails (e.g. KV error), we log it but don't abort, because WAL is already written.
        // Replay will fix consistency eventually.
        let id_bytes = id.to_le_bytes();
        if let Ok(Some(val)) = self.kv.get(&id_bytes)
            && let Ok(bucket_id) = val.try_into().map(u32::from_le_bytes)
        {
            // Tell Manager to update its L1 state (Thread-safe)
            if let Err(e) = self.storage.mark_delete(bucket_id, id) {
                tracing::warn!(
                    "Failed to shadow vector {} in bucket {}: {}",
                    id,
                    bucket_id,
                    e
                );
            }
        }

        // 3. MemTable (Visibility)
        let active_ptr = { self.active.read().clone() };
        let needs_rotate = active_ptr.insert(id, vector);

        // Resurrection: Unmark L0 tombstone if present
        self.unmark_l0_delete(id);

        if needs_rotate {
            return self.rotate_active();
        }
        Ok(false)
    }

    /// Internal helper for COW deletes (L0)
    fn mark_l0_delete(&self, id: u64) {
        let mut guard = self.deleted_ids.write();
        if guard.contains(&id) {
            return;
        }

        let mut new_set = (**guard).clone();
        new_set.insert(id);
        *guard = Arc::new(new_set);
    }

    /// Internal helper for COW un-delete (Resurrection)
    fn unmark_l0_delete(&self, id: u64) {
        let mut guard = self.deleted_ids.write();
        if !guard.contains(&id) {
            return;
        }

        let mut new_set = (**guard).clone();
        new_set.remove(&id);
        *guard = Arc::new(new_set);
    }

    fn unmark_l0_delete_batch(&self, ids: &[u64]) {
        let mut guard = self.deleted_ids.write();
        // Optimization: Check if any need removal before cloning
        if !ids.iter().any(|id| guard.contains(id)) {
            return;
        }

        let mut new_set = (**guard).clone();
        for id in ids {
            new_set.remove(id);
        }
        *guard = Arc::new(new_set);
    }

    pub fn insert_batch(&self, batch: &[(u64, Vec<f32>)]) -> io::Result<bool> {
        if batch.is_empty() {
            return Ok(false);
        }

        // 1. WAL Transaction (Durability First)
        {
            let mut mgr = self.wal_manager.lock();
            let current = mgr.current();
            let tx_id = current.begin_transaction()?;
            for (id, vector) in batch {
                current.write_insert(*id, vector)?;
            }
            current.commit_transaction(tx_id)?; // FSync happens here
        }

        // 2. Batch Shadowing
        for (id, _) in batch {
            let id_bytes = id.to_le_bytes();
            if let Ok(Some(val)) = self.kv.get(&id_bytes)
                && let Ok(bucket_id) = val.try_into().map(u32::from_le_bytes)
            {
                let _ = self.storage.mark_delete(bucket_id, *id);
            }
        }

        // 3. MemTable
        let ids: Vec<u64> = batch.iter().map(|(id, _)| *id).collect();
        self.unmark_l0_delete_batch(&ids);

        let active_ptr = { self.active.read().clone() };
        let needs_rotate = active_ptr.insert_batch(batch);

        if needs_rotate {
            return self.rotate_active();
        }
        Ok(false)
    }

    pub fn delete(&self, id: u64) -> io::Result<()> {
        // 1. WAL (Durability)
        self.wal_manager.lock().current().write_delete(id)?;

        // 2. L0 Tombstone (Fast path)
        self.mark_l0_delete(id);

        // 3. L1 Tombstone (Persistent disk search)
        let id_bytes = id.to_le_bytes();
        if let Ok(Some(val)) = self.kv.get(&id_bytes)
            && let Ok(bucket_id) = val.try_into().map(u32::from_le_bytes)
        {
            let _ = self.storage.mark_delete(bucket_id, id);
        }
        Ok(())
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

    pub async fn search(
        &self,
        query: &[f32],
        k: usize,
        target: f32,
        lambda: f32,
        tau: f32,
    ) -> io::Result<Vec<(u64, f32)>> {
        let (ram_tables, l0_view) = {
            let active = self.active.read();
            let frozen = self.frozen.read();
            // ⚡ FAST SNAPSHOT
            let view = self.deleted_ids.read().clone();

            let mut tables = vec![active.clone()];
            tables.extend(frozen.iter().map(|f| f.table.clone()));
            (tables, view)
        };

        // Wrap Arc in View trait
        let view_struct = L0TombstoneView {
            inner: l0_view.clone(),
        };

        // A. RAM Search
        let ram_results_raw = task::spawn_blocking({
            let query = query.to_vec();
            move || {
                let mut results = Vec::new();
                for table in ram_tables {
                    // Pass view down
                    results.extend(table.search(&query, k, &view_struct));
                }
                results
            }
        })
        .await
        .map_err(io::Error::other)?;

        // B. Disk Search
        let bucket_ids = self
            .router
            .read()
            .select_buckets(query, target, lambda, tau);
        let oversample_factor = k * 3;

        let disk_results = self
            .storage
            .search_and_refine(&bucket_ids, query, k, oversample_factor)
            .await;

        // C. Merge
        let mut final_results = Vec::new();
        let mut seen = HashSet::new();

        // ⚡ Filter L0 results using the snapshot we took at start
        for (id, dist) in ram_results_raw {
            if !l0_view.contains(&id) && seen.insert(id) {
                final_results.push((id, dist));
            }
        }
        // Disk results are already filtered by BucketManager's local tombstones.
        // We also check L0 tombstones here in case an L0 delete targets an L1 item
        // (though L1 should have been marked too, safety check).
        for (id, dist) in disk_results {
            if !l0_view.contains(&id) && seen.insert(id) {
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

    pub fn get_tombstones(&self) -> Arc<HashSet<u64>> {
        self.deleted_ids.read().clone()
    }

    pub fn get_deleted_ids_inner(&self) -> &RwLock<Arc<HashSet<u64>>> {
        &self.deleted_ids
    }

    pub fn get_kv(&self) -> Arc<BitStore> {
        self.kv.clone()
    }

    pub fn get_wal(&self) -> Arc<Mutex<WalManager>> {
        self.wal_manager.clone()
    }
}
