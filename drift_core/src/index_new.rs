use crate::memtable::{MemTable, MemTableOptions};
use crate::partitioner::{IncrementalPartitioner, PartitionResult};
use crate::router::Router;
use crate::wal::WalWriter;
use drift_traits::DiskSearcher;
use parking_lot::{Mutex, RwLock}; // Using parking_lot for perf
use std::cmp::Ordering;
use std::io;
use std::sync::Arc;

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

    /// Unified Scatter-Gather Search (RAM + Disk).
    /// Implements the "Saturating Density" search policy.
    pub async fn search(
        &self,
        query: &[f32],
        k: usize,
        // Search Parameters
        target_confidence: f32, // e.g. 0.95
        lambda: f32,            // e.g. 1.0
        tau: f32,               // e.g. 100.0
    ) -> io::Result<Vec<(u64, f32)>> {
        // 1. Snapshot RAM Tables (Active + Frozen)
        let ram_tables: Vec<Arc<MemTable>> = {
            let mut tables = Vec::new();
            tables.push(self.active.read().clone());
            tables.extend(self.frozen.read().iter().cloned());
            tables
        };

        // 2. RAM Search (CPU Bound - Exact Scan)
        // We scan everything in RAM because it's cheap and high-priority.
        let mut candidates = tokio::task::spawn_blocking({
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

        // 3. Route to Disk Buckets (Probabilistic Selection)
        // This is where we apply the Paper's logic.
        let bucket_ids = {
            let router = self.router.read();
            router.select_buckets(query, target_confidence, lambda, tau)
        };

        // 4. Disk Search (Async I/O)
        // We only scan the buckets selected by the math above.
        let disk_results = self.disk.search(&bucket_ids, query, k).await;
        candidates.extend(disk_results);

        // 5. Merge & Sort (Gather)
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        // Deduplicate
        let mut unique_candidates = Vec::with_capacity(k);
        let mut seen = std::collections::HashSet::new();
        for c in candidates {
            if seen.insert(c.0) {
                unique_candidates.push(c);
                if unique_candidates.len() >= k {
                    break;
                }
            }
        }

        Ok(unique_candidates)
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
