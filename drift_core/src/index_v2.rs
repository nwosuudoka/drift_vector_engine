use crate::memtable_v2::{MemTableOptions, MemTableV2};
use crate::partitioner::{IncrementalPartitioner, PartitionResult};
use crate::router::Router;
use crate::tombstone_v2::InMemoryTombstoneTracker;
use crate::wal_v2::WalWriter;
use drift_traits::{DiskSearcher, TombstoneTracker};
use parking_lot::{Mutex, RwLock};
use std::cmp::Ordering;
use std::collections::HashSet;
// Using parking_lot for perf
use std::io;
use std::sync::Arc;
use tokio::task;

/// The Coordinator for the V2 Architecture.
/// Manages the "Active" vs "Frozen" lifecycle and orchestrates Unified Search.
pub struct VectorIndexV2 {
    // 1. The Active Table (Receives Writes)
    active: RwLock<Arc<MemTableV2>>,

    // 2. The Frozen Tables (Read-Only, waiting for flush)
    frozen: RwLock<Vec<Arc<MemTableV2>>>,

    // 3. Durability (Write-Ahead Log)
    // Protected by Mutex to ensure serial writes (append-only safety)
    wal: Arc<Mutex<WalWriter>>,

    // 4. Disk Tier (Injected Dependency)
    disk: Arc<dyn DiskSearcher>,

    // tombstones
    pub(crate) tombstones: Arc<dyn TombstoneTracker>,

    // 5. Logic
    router: Arc<RwLock<Router>>,
    dim: usize,
    capacity: usize,
}

impl VectorIndexV2 {
    pub fn new(
        dim: usize,
        capacity: usize,
        router: Arc<RwLock<Router>>,
        wal: Arc<Mutex<WalWriter>>,
        disk: Arc<dyn DiskSearcher>,
    ) -> Self {
        Self {
            active: RwLock::new(Arc::new(MemTableV2::new(MemTableOptions { capacity, dim }))),
            frozen: RwLock::new(Vec::new()),
            wal,
            disk,
            router,
            dim,
            capacity,
            tombstones: Arc::new(InMemoryTombstoneTracker::new()),
        }
    }

    /// Fast Insert with WAL Durability.
    /// Returns `true` if rotation is needed.
    pub fn insert(&self, id: u64, vector: Vec<f32>) -> io::Result<bool> {
        // 1. Write to WAL (Durability First)
        {
            let mut wal_guard = self.wal.lock();
            wal_guard.write_insert(id, &vector)?;
            // We rely on background fsync or O_DIRECT for pure speed,
            // explicit sync() can be called by batch logic if needed.
            wal_guard.sync()?;
        }

        // 2. Insert to MemTable
        // Get Read Lock on Ptr (Fast)
        let active_ptr = { self.active.read().clone() };

        // Insert (Internal locks handle concurrency)
        let needs_rotate = active_ptr.insert(id, &vector);

        // If this ID was previously deleted, we MUST unmark it now.
        // Otherwise, the Searcher will filter out this new insert.
        self.tombstones.unmark_delete(id);

        if needs_rotate {
            // Upgrade to write lock to rotate
            let mut active_guard = self.active.write();

            // Double-check optimization
            if active_guard.len() >= self.capacity {
                // Move Active -> Frozen
                let old_table = active_guard.clone();
                self.frozen.write().push(old_table);

                let m_opts = MemTableOptions {
                    capacity: self.capacity,
                    dim: self.dim,
                };
                // Create New
                *active_guard = Arc::new(MemTableV2::new(m_opts));
                return Ok(true);
            }
        }

        Ok(false)
    }

    // /// Efficiently writes a batch to WAL and MemTable with minimal lock contention.
    // pub fn insert_batch(&self, batch: &[(u64, Vec<f32>)]) -> io::Result<bool> {
    //     if batch.is_empty() {
    //         return Ok(false);
    //     }

    //     // 1. Batch Write to WAL (Durability)
    //     // We lock once for the whole batch.
    //     {
    //         let mut wal_guard = self.wal.lock();
    //         for (id, vector) in batch {
    //             wal_guard.write_insert(*id, vector)?;
    //         }
    //         // Sync once at the end of the batch
    //         wal_guard.sync()?;
    //     }

    //     // 2. Batch Unmark Tombstones
    //     // Collect IDs to avoid cloning vectors if possible, or just iterate.
    //     // Since unmark_delete_batch takes &[u64], we map.
    //     let ids: Vec<u64> = batch.iter().map(|(id, _)| *id).collect();
    //     self.tombstones.unmark_delete_batch(&ids);

    //     // 3. Batch Insert to MemTable
    //     // Get Read Lock on Ptr (Fast)
    //     let active_ptr = { self.active.read().clone() };

    //     // Use optimized MemTable batch insert
    //     let needs_rotate = active_ptr.insert_batch(batch);

    //     // 4. Check Rotation
    //     if needs_rotate {
    //         return self.try_rotate();
    //     }

    //     Ok(false)
    // }

    pub fn insert_batch(&self, batch: &[(u64, Vec<f32>)]) -> io::Result<bool> {
        if batch.is_empty() {
            return Ok(false);
        }

        // 1. Write to WAL (Transactional)
        {
            let mut wal_guard = self.wal.lock();
            let tx_id = wal_guard.begin_transaction()?; // BEGIN

            for (id, vector) in batch {
                wal_guard.write_insert(*id, vector)?;
            }

            wal_guard.commit_transaction(tx_id)?; // COMMIT (Syncs here)
        }

        // 2. MemTable & Tombstones (In-Memory)
        let ids: Vec<u64> = batch.iter().map(|(id, _)| *id).collect();
        self.tombstones.unmark_delete_batch(&ids);

        let active_ptr = { self.active.read().clone() };
        let needs_rotate = active_ptr.insert_batch(batch);

        if needs_rotate {
            return self.try_rotate();
        }

        Ok(false)
    }

    // Helper to rotate active table
    fn try_rotate(&self) -> io::Result<bool> {
        let mut active_guard = self.active.write();

        // Check condition again under write lock
        if active_guard.len() >= self.capacity {
            let old_table = active_guard.clone();
            self.frozen.write().push(old_table);

            let m_opts = MemTableOptions {
                capacity: self.capacity,
                dim: self.dim,
            };
            *active_guard = Arc::new(MemTableV2::new(m_opts));
            return Ok(true);
        }
        Ok(false)
    }

    pub fn delete(&self, id: u64) -> io::Result<()> {
        // 1. WAL
        self.wal.lock().write_delete(id)?;

        // 2. MemTable
        // We only need to mark it in the Active table.
        // The Search logic merges tombstones from all RAM tables.
        self.tombstones.mark_delete(id);

        Ok(())
    }

    /// Unified Search: RAM (Exact) + Disk (Approx -> Exact Refine)
    pub async fn search(
        &self,
        query: &[f32],
        k: usize,
        target_confidence: f32,
        lambda: f32,
        tau: f32,
    ) -> io::Result<Vec<(u64, f32)>> {
        // 1. Snapshot Everything
        let (ram_tables, view) = {
            let active = self.active.read();
            let frozen = self.frozen.read();
            // Get consistent view of deletes
            let view = self.tombstones.get_view();

            let mut tables = vec![active.clone()];
            tables.extend(frozen.iter().cloned());
            (tables, view)
        };

        // 2. RAM Search (CPU) - Exact L2
        // We pass 'view' to RAM search or filter results?
        // Let's filter post-search for RAM to keep MemTable simple,
        // or update MemTable signature later. For now, filter post.
        let ram_results_raw = task::spawn_blocking({
            let query = query.to_vec();
            move || {
                let mut results = Vec::new();
                for table in ram_tables {
                    // MemTable search returns (id, dist)
                    results.extend(table.search(&query, k));
                }
                results
            }
        })
        .await
        .map_err(io::Error::other)?;

        // 3. Disk Search (IO) - Two Stage
        let bucket_ids = {
            let router = self.router.read();
            router.select_buckets(query, target_confidence, lambda, tau)
        };

        // OVERSAMPLE: We need more candidates because SQ8 is approximate.
        let oversample_k = k * 3;

        // Stage A: Scatter (SQ8 Scan)
        // Returns candidates with location metadata
        let candidates = self
            .disk
            .search(&bucket_ids, query, oversample_k, view.clone())
            .await;

        // Stage B: Gather (ALP Refine)
        // Returns (id, exact_dist)
        let disk_results = self.disk.refine(candidates, query).await;

        // 4. Merge & Deduplicate & Filter
        let mut final_results = Vec::new();
        let mut seen = HashSet::new();

        // RAM results need filtering against the view (Snapshot Isolation)
        for (id, dist) in ram_results_raw {
            if !view.contains(id) && seen.insert(id) {
                final_results.push((id, dist));
            }
        }

        // Disk results were already filtered during Scan (Stage A)
        // but we check 'seen' to avoid duplicates if RAM has the same ID (fresh update)
        for (id, dist) in disk_results {
            if seen.insert(id) {
                // Double check tombstone?
                // Technically unnecessary if DiskSearcher respected view, but safe.
                if !view.contains(id) {
                    final_results.push((id, dist));
                }
            }
        }

        // 5. Final Sort
        final_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        if final_results.len() > k {
            final_results.truncate(k);
        }

        Ok(final_results)
    }

    /// Used by Janitor to prepare writes.
    pub fn flush_frozen(&self) -> Option<PartitionResult> {
        let tables_to_flush = {
            let f = self.frozen.read(); // Read lock only!
            if f.is_empty() {
                return None;
            }
            f.clone()
        };

        let router_guard = self.router.read();
        let mut global_partition = std::collections::HashMap::new();

        for table in tables_to_flush {
            let (ids, flat_vecs) = table.snapshot();
            let part = IncrementalPartitioner::partition(&ids, &flat_vecs, self.dim, &router_guard);

            // Merge
            for (bucket_id, group) in part {
                let entry = global_partition.entry(bucket_id).or_insert_with(|| {
                    crate::partitioner::PartitionGroup {
                        ids: vec![],
                        flat_vectors: vec![],
                        count: 0,
                    }
                });
                entry.ids.extend(group.ids);
                entry.flat_vectors.extend(group.flat_vectors);
                entry.count += group.count;
            }
        }

        Some(global_partition)
    }

    /// Call this ONLY after data is safely persisted to disk/staging.
    pub fn acknowledge_flush(&self) -> io::Result<()> {
        // 1. Clear Frozen (Write Lock)
        {
            let mut f = self.frozen.write();
            f.clear();
        }

        // 2. Truncate WAL (Mutex Lock)
        {
            let mut wal = self.wal.lock();
            wal.truncate()?;
        }

        Ok(())
    }
}
