use crate::memtable::{MemTable, MemTableOptions};
use crate::partitioner::{IncrementalPartitioner, PartitionResult};
use crate::router::Router;
use crate::tombstone_v2::InMemoryTombstoneTracker;
use crate::wal::WalWriter;
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
    active: RwLock<Arc<MemTable>>,

    // 2. The Frozen Tables (Read-Only, waiting for flush)
    frozen: RwLock<Vec<Arc<MemTable>>>,

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
            active: RwLock::new(Arc::new(MemTable::new(MemTableOptions { capacity, dim }))),
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
                *active_guard = Arc::new(MemTable::new(m_opts));
                return Ok(true);
            }
        }

        Ok(false)
    }

    pub fn delete(&self, id: u64) -> io::Result<()> {
        // 1. WAL
        self.wal.lock().write_delete(id)?;

        // 2. MemTable
        // We only need to mark it in the Active table.
        // The Search logic merges tombstones from all RAM tables.
        self.active.read().delete(id);
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

    /// Flushes ALL frozen tables.
    /// Returns data to be written by the caller (Server).
    pub fn flush_frozen(&self) -> Option<PartitionResult> {
        let tables_to_flush = {
            let mut f = self.frozen.write();
            if f.is_empty() {
                return None;
            }
            let pending = f.clone();
            f.clear();
            pending
        };

        let router_guard = self.router.read();
        let mut global_partition = std::collections::HashMap::new();

        for table in tables_to_flush {
            let (ids, flat_vecs, _tomb) = table.snapshot();

            // Adapt MemTable data for Partitioner
            // TODO: Update Partitioner to accept flat slices directly to avoid this allocation
            let mut batch = Vec::with_capacity(ids.len());
            for (i, id) in ids.iter().enumerate() {
                let s = i * self.dim;
                let e = s + self.dim;
                batch.push((*id, flat_vecs[s..e].to_vec()));
            }

            let part = IncrementalPartitioner::partition(&batch, &router_guard);

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

        // ⚠️ CRITICAL: After successful return, the caller MUST truncate the WAL.
        Some(global_partition)
    }
}
