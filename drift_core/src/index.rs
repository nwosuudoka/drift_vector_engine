use crate::kmeans::KMeansAlgorithm;
use crate::math::Metric;
use crate::memtable::{MemTable, MemTableOptions};
use crate::metric_strategy::strategy_for;
use crate::partitioner::{IncrementalPartitioner, PartitionGroup, PartitionResult};
use crate::payload::{PayloadRow, PayloadSchema};
use crate::router::Router;
use crate::wal::WalManager;
use drift_kv::bitstore::BitStore;
use drift_traits::{StorageEngine, TombstoneView};
use parking_lot::{Mutex, RwLock};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering as AtomicOrdering};
use tokio::task;

const SPLIT_THRESHOLD: usize = 50;
const SINGULARITY_THRESHOLD: f32 = 0.01;

#[derive(Clone)]
pub struct FrozenTable {
    pub table: Arc<MemTable>,
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

pub struct MergeProposal {
    pub zombie_id: u32,
    pub moves: HashMap<u32, PartitionGroup>,
}

#[derive(Debug, Clone, Default)]
pub struct SearchHintStats {
    pub selected_bucket_count: usize,
    pub candidate_bucket_count: usize,
    pub candidate_id_count: usize,
    pub estimated_total_bucket_ids: usize,
    pub estimated_scanned_ids: usize,
}

pub struct VectorIndex {
    active: RwLock<Arc<MemTable>>,
    frozen: RwLock<Vec<FrozenTable>>,
    wal_manager: Arc<Mutex<WalManager>>,
    storage: Arc<dyn StorageEngine>,
    kv: Arc<BitStore>,
    router: Arc<RwLock<Router>>,
    deleted_ids: RwLock<Arc<HashSet<u64>>>,

    next_bucket_id: AtomicU32,
    last_search_hint_stats: RwLock<SearchHintStats>,

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
        let max_id = router.read().max_bucket_id();

        Self {
            active: RwLock::new(Arc::new(MemTable::new(MemTableOptions { capacity, dim }))),
            frozen: RwLock::new(Vec::new()),
            deleted_ids: RwLock::new(Arc::new(HashSet::new())),
            wal_manager,
            storage,
            kv,
            router,
            dim,
            capacity,
            next_bucket_id: AtomicU32::new(max_id + 1),
            last_search_hint_stats: RwLock::new(SearchHintStats::default()),
        }
    }

    pub fn allocate_next_bucket_id(&self) -> u32 {
        self.next_bucket_id.fetch_add(1, AtomicOrdering::Relaxed)
    }

    pub fn insert(&self, id: u64, vector: &[f32]) -> io::Result<bool> {
        self.insert_with_payload(id, vector, None, None)
    }

    pub fn insert_with_payload(
        &self,
        id: u64,
        vector: &[f32],
        payload_schema: Option<&PayloadSchema>,
        payload_row: Option<&PayloadRow>,
    ) -> io::Result<bool> {
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
        let needs_rotate =
            active_ptr.insert_with_payload(id, vector, payload_schema, payload_row)?;

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
        self.insert_batch_with_payload(batch, None, None)
    }

    pub fn insert_batch_with_payload(
        &self,
        batch: &[(u64, Vec<f32>)],
        payload_schema: Option<&PayloadSchema>,
        payload_rows: Option<&[PayloadRow]>,
    ) -> io::Result<bool> {
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
        let needs_rotate =
            active_ptr.insert_batch_with_payload(batch, payload_schema, payload_rows)?;

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

        *active_guard = Arc::new(MemTable::new(MemTableOptions {
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
        self.search_with_hints(query, k, target, lambda, tau, None, None)
            .await
    }

    pub fn select_buckets(&self, query: &[f32], target: f32, lambda: f32, tau: f32) -> Vec<u32> {
        let router = self.router.read();
        router.select_buckets(query, target, lambda, tau)
    }

    pub fn all_routable_bucket_ids(&self) -> Vec<u32> {
        let router = self.router.read();
        let (_, ids) = router.get_snapshot();
        ids
    }

    pub fn rank_bucket_ids_by_query_distance(&self, query: &[f32], bucket_ids: &[u32]) -> Vec<u32> {
        if bucket_ids.len() <= 1 {
            return bucket_ids.to_vec();
        }

        let (flat_centroids, ids, dim, metric) = {
            let router = self.router.read();
            let (flat, ids) = router.get_snapshot();
            (flat, ids, router.dim(), router.metric())
        };

        if query.len() != dim {
            return bucket_ids.to_vec();
        }

        let scorer = strategy_for(metric);
        let mut centroid_scores: HashMap<u32, f32> = HashMap::with_capacity(ids.len());
        for (idx, bucket_id) in ids.iter().enumerate() {
            let start = idx * dim;
            let end = start + dim;
            if end > flat_centroids.len() {
                continue;
            }
            let centroid = &flat_centroids[start..end];
            centroid_scores.insert(*bucket_id, scorer.score(query, centroid));
        }

        let mut ranked: Vec<(u32, f32, usize)> = bucket_ids
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, bucket_id)| {
                (
                    bucket_id,
                    *centroid_scores.get(&bucket_id).unwrap_or(&f32::INFINITY),
                    idx,
                )
            })
            .collect();

        ranked.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.2.cmp(&b.2))
        });
        ranked
            .into_iter()
            .map(|(bucket_id, _, _)| bucket_id)
            .collect()
    }

    pub async fn search_with_bucket_hint(
        &self,
        query: &[f32],
        k: usize,
        target: f32,
        lambda: f32,
        tau: f32,
        bucket_hint: Option<&[u32]>,
    ) -> io::Result<Vec<(u64, f32)>> {
        self.search_with_hints(query, k, target, lambda, tau, bucket_hint, None)
            .await
    }

    pub async fn search_with_hints(
        &self,
        query: &[f32],
        k: usize,
        target: f32,
        lambda: f32,
        tau: f32,
        bucket_hint: Option<&[u32]>,
        candidate_ids: Option<&HashMap<u32, HashSet<u64>>>,
    ) -> io::Result<Vec<(u64, f32)>> {
        let (bucket_ids, metric) = {
            let router = self.router.read();
            let selected = if let Some(hint) = bucket_hint {
                hint.to_vec()
            } else {
                router.select_buckets(query, target, lambda, tau)
            };
            (selected, router.metric())
        };
        let mut hint_stats = SearchHintStats {
            selected_bucket_count: bucket_ids.len(),
            ..SearchHintStats::default()
        };
        if let Some(candidate_map) = candidate_ids {
            let mut candidate_bucket_count = 0usize;
            let mut candidate_id_count = 0usize;
            for bucket_id in &bucket_ids {
                if let Some(ids) = candidate_map.get(bucket_id)
                    && !ids.is_empty()
                {
                    candidate_bucket_count += 1;
                    candidate_id_count += ids.len();
                }
            }
            hint_stats.candidate_bucket_count = candidate_bucket_count;
            hint_stats.candidate_id_count = candidate_id_count;
        }
        for bucket_id in &bucket_ids {
            let live_count = self
                .storage
                .get_bucket_stats(*bucket_id)
                .map(|stats| stats.total_count.saturating_sub(stats.tombstone_count) as usize)
                .unwrap_or(0);
            hint_stats.estimated_total_bucket_ids += live_count;
            let scanned_for_bucket = candidate_ids
                .and_then(|candidate_map| candidate_map.get(bucket_id))
                .map(|ids| ids.len())
                .unwrap_or(live_count);
            hint_stats.estimated_scanned_ids += scanned_for_bucket;
        }
        if hint_stats.estimated_total_bucket_ids == 0 && hint_stats.candidate_id_count > 0 {
            hint_stats.estimated_total_bucket_ids = hint_stats.candidate_id_count;
        }
        if hint_stats.estimated_scanned_ids == 0 && hint_stats.candidate_id_count > 0 {
            hint_stats.estimated_scanned_ids = hint_stats.candidate_id_count;
        }
        *self.last_search_hint_stats.write() = hint_stats;

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
                    results.extend(table.search(&query, k, metric, &view_struct));
                }
                results
            }
        })
        .await
        .map_err(io::Error::other)?;

        // B. Disk Search
        let oversample_factor = k * 3;

        let disk_results = self
            .storage
            .search_and_refine_with_candidates(
                &bucket_ids,
                query,
                k,
                oversample_factor,
                candidate_ids,
            )
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

    pub fn metric(&self) -> Metric {
        self.router.read().metric()
    }

    pub fn last_search_hint_stats(&self) -> SearchHintStats {
        self.last_search_hint_stats.read().clone()
    }

    pub fn lookup_l0_payload_rows(&self, ids: &[u64]) -> HashMap<u64, PayloadRow> {
        if ids.is_empty() {
            return HashMap::new();
        }

        let wanted: HashSet<u64> = ids.iter().copied().collect();
        let mut rows_by_id: HashMap<u64, PayloadRow> = HashMap::new();

        // Apply older frozen rows first, then let active rows override.
        let frozen_tables = self.frozen.read().clone();
        for frozen in frozen_tables {
            let (table_ids, payload_rows) = frozen.table.snapshot_payload_rows();
            if let Some(rows) = payload_rows {
                for (id, row) in table_ids.into_iter().zip(rows.into_iter()) {
                    if wanted.contains(&id) {
                        rows_by_id.entry(id).or_insert(row);
                    }
                }
            }
        }

        let active_table = self.active.read().clone();
        let (active_ids, active_rows) = active_table.snapshot_payload_rows();
        if let Some(rows) = active_rows {
            for (id, row) in active_ids.into_iter().zip(rows.into_iter()) {
                if wanted.contains(&id) {
                    rows_by_id.insert(id, row);
                }
            }
        }

        rows_by_id
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
            let (ids, flat_vecs, payload_schema, payload_rows) = ft.table.snapshot_with_payload();
            let part = IncrementalPartitioner::partition_with_payload(
                &ids,
                &flat_vecs,
                self.dim,
                &router_guard,
                payload_schema.as_ref(),
                payload_rows.as_deref(),
            )
            .expect("frozen payload partitioning invariants should hold");

            for (bucket_id, group) in part {
                let PartitionGroup {
                    ids,
                    flat_vectors,
                    count,
                    centroid,
                    payload_schema,
                    payload_rows,
                } = group;
                let entry = global_partition.entry(bucket_id).or_insert_with(|| {
                    PartitionGroup::new_with_payload(
                        self.dim,
                        centroid.clone(),
                        payload_schema.clone(),
                    )
                });
                if entry.centroid.is_none() {
                    entry.centroid = centroid;
                }
                if entry.payload_schema.is_none() {
                    entry.payload_schema = payload_schema.clone();
                } else if payload_schema.is_some() && entry.payload_schema != payload_schema {
                    panic!("payload schema mismatch while merging frozen partition groups");
                }
                entry.ids.extend(ids);
                entry.flat_vectors.extend(flat_vectors);
                entry.count += count;
                match (&mut entry.payload_rows, payload_rows) {
                    (Some(existing), Some(mut incoming)) => existing.append(&mut incoming),
                    (None, Some(incoming)) => entry.payload_rows = Some(incoming),
                    (Some(_), None) => {
                        panic!(
                            "payload row presence mismatch while merging frozen partition groups"
                        )
                    }
                    (None, None) => {}
                }
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

    pub async fn calculate_split(
        &self,
        bucket_id: u32,
    ) -> io::Result<Result<SplitProposal, MaintenanceStatus>> {
        // 1. Fetch High-Fidelity Data (Async I/O)
        let fetch_res = self.storage.fetch_bucket(bucket_id).await;

        let (ids, flat_vecs) = match fetch_res {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!(
                    "calculate_split: Failed to fetch bucket {}: {}",
                    bucket_id,
                    e
                );
                return Ok(Err(MaintenanceStatus::SkippedLocked));
            }
        };

        if ids.len() < SPLIT_THRESHOLD {
            return Ok(Err(MaintenanceStatus::SkippedTooSmall));
        }

        // 2. Snapshot Global Context (for Defector Check)
        // We capture the state of the world NOW.
        let (global_centroids, global_ids, metric) = {
            let router = self.router.read();
            let (centroids, ids) = router.get_snapshot();
            (centroids, ids, router.metric())
        };
        let dim = self.dim;

        // 3. Offload Compute to Blocking Thread
        let proposal_result = task::spawn_blocking(move || {
            let strategy = crate::metric_strategy::strategy_for(metric);
            // A. Train K-Means (K=2)
            let trainer = KMeansAlgorithm::for_metric(metric, 2, dim, 15);
            let result = trainer.train(&flat_vecs);

            // B. Singularity Check (Variance)
            let dist_sq = strategy.score(&result.centroids[0], &result.centroids[1]);
            if dist_sq < SINGULARITY_THRESHOLD {
                return Err(MaintenanceStatus::SkippedSingularity { variance: dist_sq });
            }

            // C. Partition & Defector Logic
            let mut left = PartitionGroup::new(dim, Some(result.centroids[0].clone()));
            let mut right = PartitionGroup::new(dim, Some(result.centroids[1].clone()));
            let mut loopback = Vec::new();

            let c0 = &result.centroids[0];
            let c1 = &result.centroids[1];

            // Pre-calculate defector threshold (Hysteresis)
            // A neighbor must be at least 10% closer to be worth moving.
            const HYSTERESIS: f32 = 0.90;

            for (i, &assignment) in result.assignments.iter().enumerate() {
                let start = i * dim;
                let end = start + dim;
                let vec = &flat_vecs[start..end];
                let id = ids[i];

                // 1. Distance to assigned Local Child
                let dist_local = if assignment == 0 {
                    strategy.score(vec, c0)
                } else {
                    strategy.score(vec, c1)
                };

                // 2. Global Defector Check
                // Find distance to the NEAREST global neighbor (excluding the current bucket).
                // Optimization: We scan the flat buffer directly.
                let mut best_global_dist = f32::MAX;

                // We iterate chunks of 'dim' in global_centroids
                for (g_idx, g_id) in global_ids.iter().enumerate() {
                    if *g_id == bucket_id {
                        continue;
                    } // Skip self

                    let g_start = g_idx * dim;
                    let g_vec = &global_centroids[g_start..g_start + dim];

                    let d = strategy.score(vec, g_vec);
                    if d < best_global_dist {
                        best_global_dist = d;
                    }
                }

                // 3. Decision
                if best_global_dist < (dist_local * HYSTERESIS) {
                    // DEFECTOR FOUND: Sending back to MemTable
                    loopback.push((id, vec.to_vec()));
                } else {
                    // Stay Local
                    if assignment == 0 {
                        left.ids.push(id);
                        left.flat_vectors.extend_from_slice(vec);
                        left.count += 1;
                    } else {
                        right.ids.push(id);
                        right.flat_vectors.extend_from_slice(vec);
                        right.count += 1;
                    }
                }
            }

            Ok(SplitProposal {
                target_bucket: bucket_id,
                left,
                right,
                loopback,
            })
        })
        .await
        .map_err(io::Error::other)?;

        Ok(proposal_result)
    }

    pub async fn apply_split_update(
        &self,
        old_id: u32,
        left: (u32, Vec<f32>, u32),
        right: (u32, Vec<f32>, u32),
    ) {
        let mut guard = self.router.write();
        // Remove old
        guard.remove_bucket(old_id);
        // Add new
        guard.add_bucket(left.0, left.1.clone());
        guard.update_bucket(left.0, left.2, left.1);
        guard.add_bucket(right.0, right.1.clone());
        guard.update_bucket(right.0, right.2, right.1);
    }

    // For testing purposes
    pub fn get_router(&self) -> &Arc<RwLock<Router>> {
        &self.router
    }

    pub fn update_router_count(&self, bucket_id: u32, count: u32, centroid: Option<Vec<f32>>) {
        let mut r = self.router.write();
        if !r.update_bucket_count(bucket_id, count)
            && let Some(c) = centroid
        {
            r.add_bucket(bucket_id, c);
            let _ = r.update_bucket_count(bucket_id, count);
        }
    }

    /// This is Read-Only. It does not modify disk or memory.
    pub async fn calculate_merge(
        &self,
        zombie_id: u32,
    ) -> io::Result<Result<MergeProposal, MaintenanceStatus>> {
        // 1. Fetch Data
        // This pulls High-Fidelity data (merging L1 tiers if needed)
        let fetch_res = self.storage.fetch_bucket(zombie_id).await;

        let (ids, flat_vecs) = match fetch_res {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!(
                    "calculate_merge: Failed to fetch bucket {}: {}",
                    zombie_id,
                    e
                );
                println!("skip locked here {}", e);
                // If the file is gone, it's already locked or deleted. Skip.
                return Ok(Err(MaintenanceStatus::SkippedLocked));
            }
        };

        // 2. Budget / Sanity Checks
        // If it's empty, we just return an empty proposal (signal to delete).
        if ids.is_empty() {
            return Ok(Ok(MergeProposal {
                zombie_id,
                moves: HashMap::new(),
            }));
        }

        // If it grew significantly since the check (e.g. unexpected ingest), abort.
        // We don't want to scatter a huge healthy bucket.
        if ids.len() > 100 {
            return Ok(Err(MaintenanceStatus::SkippedTooSmall)); // Actually too big
        }

        // 3. Snapshot Router (Global Context)
        // We need this to find neighbors.
        let (centroids, bucket_ids, metric) = {
            let router = self.router.read();
            let (centroids, ids) = router.get_snapshot();
            (centroids, ids, router.metric())
        };
        let dim = self.dim;

        // 4. Offload Calculation (CPU Heavy)
        let proposal_result = task::spawn_blocking(move || {
            let strategy = crate::metric_strategy::strategy_for(metric);
            let mut moves: HashMap<u32, PartitionGroup> = HashMap::new();

            for (i, &id) in ids.iter().enumerate() {
                let start = i * dim;
                let end = start + dim;
                let vec = &flat_vecs[start..end];

                // Find Nearest Neighbor that is NOT the zombie
                let mut best_dist = f32::MAX;
                let mut best_id = None;
                let mut best_centroid = None;

                // Iterate all global centroids
                for (c_idx, &c_id) in bucket_ids.iter().enumerate() {
                    if c_id == zombie_id {
                        continue;
                    }

                    let c_start = c_idx * dim;
                    let centroid = &centroids[c_start..c_start + dim];

                    let dist = strategy.score(vec, centroid);

                    if dist < best_dist {
                        best_dist = dist;
                        best_id = Some(c_id);
                        best_centroid = Some(centroid.to_vec());
                    }
                }

                if let Some(target) = best_id {
                    let group = moves.entry(target).or_insert_with(|| {
                        // We pass the Target's centroid so PartitionGroup is valid
                        PartitionGroup::new(dim, best_centroid)
                    });

                    group.ids.push(id);
                    group.flat_vectors.extend_from_slice(vec);
                    group.count += 1;
                } else {
                    // No neighbors found?
                    // This happens if:
                    // 1. There is only 1 bucket in the system.
                    // 2. All other buckets are empty/inactive (unlikely).
                    // We abort the merge.
                    return Err(MaintenanceStatus::SkippedTooSmall);
                }
            }

            Ok(MergeProposal { zombie_id, moves })
        })
        .await
        .map_err(io::Error::other)?;

        Ok(proposal_result)
    }

    /// Now accepts `new_centroid` to keep drift calculations valid.
    pub async fn apply_merge_update(
        &self,
        zombie_id: u32,
        updates: &[(u32, u64, Vec<f32>, Vec<f32>)], // (TargetID, Count, Sum, Centroid)
    ) {
        let mut r = self.router.write();

        // 1. Kill Zombie
        r.remove_bucket(zombie_id);

        // 2. Update Neighbors (Count + Centroid)
        for (target_id, new_count, _, new_centroid) in updates {
            r.update_bucket(*target_id, *new_count as u32, new_centroid.clone());
        }
    }

    pub fn debug_find_needles_in_l0(&self, needle_ids: &HashSet<u64>) -> HashSet<u64> {
        let mut found = HashSet::new();

        let active = self.active.read();
        let (ids, _) = active.snapshot();
        for id in ids {
            if needle_ids.contains(&id) {
                found.insert(id);
            }
        }

        let frozen = self.frozen.read();
        for ft in frozen.iter() {
            let (ids, _) = ft.table.snapshot();
            for id in ids {
                if needle_ids.contains(&id) {
                    found.insert(id);
                }
            }
        }

        found
    }

    pub async fn debug_fetch_bucket_ids(&self, bucket_id: u32) -> io::Result<Vec<u64>> {
        let (ids, _vecs) = self.storage.fetch_bucket(bucket_id).await?;
        Ok(ids)
    }

    pub async fn debug_fetch_bucket(&self, bucket_id: u32) -> io::Result<(Vec<u64>, Vec<f32>)> {
        self.storage.fetch_bucket(bucket_id).await
    }

    /// Returns the number of vectors in the active MemTable.
    pub fn memtable_len(&self) -> usize {
        self.active.read().len()
    }

    /// Returns the number of frozen MemTables waiting to be flushed.
    pub fn get_frozen_count(&self) -> usize {
        self.frozen.read().len()
    }
}
