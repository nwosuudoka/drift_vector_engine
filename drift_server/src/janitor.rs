use crate::persistence::PersistenceManager;
use drift_core::index::{MaintenanceStatus, VectorIndex};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};
use tokio::time;
use tracing::{error, info, instrument};

const BUCKET_SPLIT_THRESHOLD: f32 = 0.8;
const TEMPERATURE_COOL_FACTOR: f32 = 0.98;
const DRIFT_THRESHOLD: f32 = 0.15;

#[derive(Clone, Copy, Debug)]
struct IgnoreState {
    retries: u32,
    next_retry_at: Instant,
}

impl IgnoreState {
    /// Create a fresh ignore state (first time we see a singularity).
    fn first(now: Instant) -> Self {
        let retries = 1;
        Self {
            retries,
            next_retry_at: now + Self::backoff_for(retries),
        }
    }

    /// Whether we should skip maintenance for this bucket right now.
    fn should_skip(&self, now: Instant) -> bool {
        now < self.next_retry_at
    }

    /// Record another singularity and push out the next retry time.
    fn on_singularity(mut self, now: Instant) -> Self {
        self.retries = self.retries.saturating_add(1);
        self.next_retry_at = now + Self::backoff_for(self.retries);
        self
    }

    /// Exponential backoff with a cap (tune to taste).
    fn backoff_for(retries: u32) -> Duration {
        // 50ms, 100ms, 200ms, 400ms, ... capped
        let exp = retries.min(6); // caps growth
        let ms = 50u64.saturating_mul(1u64 << exp);
        Duration::from_millis(ms).min(Duration::from_secs(2))
    }
}

pub struct Janitor {
    index: Arc<VectorIndex>,
    persistence: PersistenceManager,
    flush_threshold: usize,
    check_interval: Duration,
    ignore_map: std::sync::Mutex<HashMap<u32, IgnoreState>>,
}

impl Janitor {
    pub fn new(
        index: Arc<VectorIndex>,
        persistence: PersistenceManager,
        flush_threshold: usize,
        check_interval: Duration,
    ) -> Self {
        Self {
            index,
            persistence,
            flush_threshold,
            check_interval,
            ignore_map: std::sync::Mutex::new(HashMap::new()),
        }
    }

    pub async fn run(&self) {
        let mut interval = time::interval(self.check_interval);
        loop {
            interval.tick().await;

            // 1. Flush Logic
            // Check if we need to flush (Threshold met OR pending frozen work from a failed retry)
            let mem_size = self.index.memtable_len();
            let has_frozen = self.index.frozen_memtable.read().is_some();

            if mem_size >= self.flush_threshold || has_frozen {
                match self.perform_flush().await {
                    Ok(Some(p)) => info!("Janitor: Flushed to {:?}", p),
                    Ok(None) => {} // No work needed
                    Err(e) => error!("Janitor Error: Failed to flush: {}", e),
                }
            }

            // 2. Scavenge (Disk -> Disk Compaction)
            if let Err(e) = self.scavenge().await {
                error!("Janitor: Scavenge failed: {}", e);
            }

            // 3. Maintenance
            self.perform_maintenance().await;
        }
    }

    async fn perform_maintenance(&self) {
        let buckets = self.index.get_all_bucket_headers();
        let target_cap = self.index.config.max_bucket_capacity as u32;
        let mut ops_budget = 1;
        let now = Instant::now();

        for header in buckets {
            if ops_budget == 0 {
                break;
            }
            {
                let mut guard = self.ignore_map.lock().unwrap();
                if let Some(state) = guard.get(&header.id)
                    && state.should_skip(now)
                {
                    if state.should_skip(now) {
                        continue;
                    } else {
                        guard.remove(&header.id);
                    }
                }
            }

            header.cool(TEMPERATURE_COOL_FACTOR);

            let _ =
                header
                    .stats
                    .temperature
                    .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |t| Some(t * 0.98));

            let urgency = header.calculate_urgency(target_cap as usize);
            let capacity_ratio = header.count as f32 / target_cap as f32;
            let drift = header.calculate_drift();

            // Make that value configurable
            if (capacity_ratio > BUCKET_SPLIT_THRESHOLD) || (drift > DRIFT_THRESHOLD) {
                info!(
                    "Janitor: ✂️ Splitting Bucket {} (Count {})",
                    header.id, header.count
                );

                match self.index.split_and_steal(header.id).await {
                    Ok(MaintenanceStatus::Completed) => {
                        info!(
                            "Janitor: ✂️ Split Bucket {} (Ratio {:.2})",
                            header.id, capacity_ratio
                        );
                        ops_budget -= 1;
                        continue; // Done with this bucket
                    }
                    Ok(MaintenanceStatus::SkippedSingularity) => {
                        info!("Janitor: Bucket {} is a Singularity. Ignoring.", header.id);

                        let now = Instant::now();
                        let mut guard = self.ignore_map.lock().unwrap();

                        guard
                            .entry(header.id)
                            .and_modify(|s| *s = s.on_singularity(now))
                            .or_insert_with(|| IgnoreState::first(now));
                    }
                    Ok(_) => {}
                    Err(e) => error!("Split failed: {}", e),
                }
            } else if urgency > 1.5 {
                info!(
                    "Janitor: 🚑 Scatter Merging Bucket {} (Count {})",
                    header.id, header.count
                );
                self.ignore_map.lock().unwrap().remove(&header.id);
                match self.index.scatter_merge(header.id).await {
                    Ok(MaintenanceStatus::Completed) => ops_budget -= 1,
                    Ok(_) => {}
                    Err(e) => error!("Merge failed: {}", e),
                }
            }
            // header.temperature()
        }
    }

    #[instrument(skip(self), level = "info")]
    async fn perform_flush(&self) -> std::io::Result<Option<std::path::PathBuf>> {
        let start = std::time::Instant::now();

        // 1. Extract Data
        let memtable_arc = {
            let frozen = self.index.frozen_memtable.read();
            if let Some(m) = frozen.as_ref() {
                m.clone()
            } else {
                drop(frozen);
                match self.index.rotate_and_freeze()? {
                    Some(m) => m,
                    None => return Ok(None),
                }
            }
        };

        let data = memtable_arc.extract_all();
        if data.is_empty() {
            self.index.confirm_flush()?;
            return Ok(None);
        }

        let vectors: Vec<Vec<f32>> = data.iter().map(|(_, v)| v.clone()).collect();
        let ids: Vec<u64> = data.iter().map(|(id, _)| *id).collect();

        // 2. Train (if needed)
        if self.index.get_quantizer().is_none() {
            self.index.train(&vectors).await?;
        }

        // 3. Partition
        let partitions = self.index.calculate_partitions(&ids, &vectors).await?;
        if partitions.is_empty() {
            self.index.confirm_flush()?;
            return Ok(None);
        }

        info!("calculated {} partitions", partitions.len());

        // 4. ⚡ PERSIST (Get Offsets) ⚡
        let (run_id, locations) = self
            .persistence
            .write_partitioned_segment(&partitions, &self.index)
            .await?;

        // Extract just the (offset, len) for the Index Blob (SQ8) needed by BlockCache
        // BucketLocation has many fields, we need 'index_offset' and 'index_length'
        let offsets_map: HashMap<u32, (u64, u32)> = locations
            .into_iter()
            .map(|(id, loc)| (id, (loc.index_offset, loc.index_length as u32)))
            .collect();

        // 5. ⚡ REGISTER (Pass Offsets) ⚡
        self.index
            .register_partitions(&partitions, &run_id, &offsets_map)
            .await?;

        // 6. Cleanup
        let deleted_snapshot: Vec<u64> = self.index.deleted_ids.read().iter().cloned().collect();
        if !deleted_snapshot.is_empty() {
            self.persistence
                .flush_tombstones(&deleted_snapshot, &run_id)
                .await?;
        }
        self.index.confirm_flush()?;

        info!(
            vectors = data.len(),
            buckets = partitions.len(),
            duration_ms = start.elapsed().as_millis(),
            "Flush Complete"
        );
        Ok(Some(PathBuf::from(format!("segment_{}.drift", run_id))))
    }

    pub async fn scavenge(&self) -> std::io::Result<()> {
        // Optimization: If global deleted set is small, don't bother scanning.
        if self.index.deleted_ids.read().len() < 100 {
            return Ok(());
        }

        // 1. Identify Dirty Buckets
        // We scan headers and check the AtomicU32 tombstone_count.
        let headers = self.index.get_all_bucket_headers();
        let mut dirty_candidates = Vec::new();

        for header in headers {
            let total = header.count as f32;
            if total == 0.0 {
                continue;
            }

            // This load is atomic and safe.
            // It reflects the current count updated by delete() calls.
            let dead = header
                .stats
                .tombstone_count
                .load(std::sync::atomic::Ordering::Relaxed) as f32;

            // Threshold: > 20% dead items
            if (dead / total) > 0.20 {
                dirty_candidates.push(header.id);
            }
        }

        // 2. Compact One Candidate
        // We process one per cycle to be gentle on system resources.
        if let Some(bucket_id) = dirty_candidates.first() {
            info!("Janitor: Scavenging bucket {} (Dirty)", bucket_id);

            if let Ok(Some(purged_ids)) = self.index.compact_bucket(*bucket_id).await
                && !purged_ids.is_empty()
            {
                info!("Janitor: Purged {} IDs from memory.", purged_ids.len());

                // 3. FREE MEMORY
                // Now we can safely remove these IDs from the global filter
                let mut write_guard = self.index.deleted_ids.write();
                for id in purged_ids {
                    write_guard.remove(&id);
                }
            }
        }

        Ok(())
    }
}
