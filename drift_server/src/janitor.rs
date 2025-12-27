use crate::persistence::PersistenceManager;
use drift_core::index::{MaintenanceStatus, VectorIndex};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;
use tokio::time;
use tracing::{error, info, instrument};

pub struct Janitor {
    index: Arc<VectorIndex>,
    persistence: PersistenceManager,
    flush_threshold: usize,
    check_interval: Duration,
    // Track buckets that are "Unsplittable" to prevent infinite loops
    ignore_set: std::sync::Mutex<HashSet<u32>>,
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
            ignore_set: std::sync::Mutex::new(HashSet::new()),
        }
    }

    pub async fn run(&self) {
        let mut interval = time::interval(self.check_interval);

        loop {
            interval.tick().await;

            // 1. Flush (Priority High)
            let size = self.index.memtable_len();
            if size >= self.flush_threshold {
                match self.perform_flush().await {
                    Ok(Some(p)) => info!("Janitor: Flushed to {:?}", p),
                    Ok(None) => {}
                    Err(e) => error!("Janitor Error: Failed to flush: {}", e),
                }
            }

            // 2. Maintenance (Priority Low)
            self.perform_maintenance().await;
        }
    }

    async fn perform_maintenance(&self) {
        let buckets = self.index.get_all_bucket_headers();
        let target_cap = self.index.config.max_bucket_capacity as u32;

        // BUDGET: Only do 1 heavy op per tick to prevent starvation
        let mut ops_budget = 1;

        for header in buckets {
            if ops_budget == 0 {
                break;
            }

            // Check Ignore List (Scoped)
            {
                let guard = self.ignore_set.lock().unwrap();
                if guard.contains(&header.id) {
                    continue;
                }
            }

            // 1. SPLIT
            if header.count > (target_cap as f32 * 1.5) as u32 {
                info!(
                    "Janitor: ✂️ Splitting Bucket {} (Count {})",
                    header.id, header.count
                );

                match self.index.split_and_steal(header.id).await {
                    Ok(MaintenanceStatus::Completed) => {
                        ops_budget -= 1;
                    }
                    Ok(MaintenanceStatus::SkippedSingularity) => {
                        info!("Janitor: Bucket {} is a Singularity. Ignoring.", header.id);
                        self.ignore_set.lock().unwrap().insert(header.id);
                    }
                    Ok(_) => {}
                    Err(e) => error!("Split failed: {}", e),
                }
            }
            // 2. MERGE
            else if header.count == 0 {
                info!(
                    "Janitor: 🚑 Scatter Merging Bucket {} (Count {})",
                    header.id, header.count
                );
                self.ignore_set.lock().unwrap().remove(&header.id); // Cleanup

                match self.index.scatter_merge(header.id).await {
                    Ok(MaintenanceStatus::Completed) => {
                        ops_budget -= 1;
                    }
                    Ok(_) => {}
                    Err(e) => error!("Merge failed: {}", e),
                }
            }
        }
    }

    // In drift_server/src/janitor.rs
    #[instrument(skip(self), level = "info")]
    async fn perform_flush(&self) -> std::io::Result<Option<std::path::PathBuf>> {
        let start = std::time::Instant::now();

        // 1. Rotate & Freeze (Non-Blocking)
        // This is fast. It just swaps pointers.
        let memtable_arc = match self.index.rotate_and_freeze()? {
            Some(m) => m,
            None => {
                // If None, it means the previous flush is still frozen (backpressure).
                // We simply return and try again next tick.
                return Ok(None);
            }
        };

        let data = memtable_arc.extract_all();
        if data.is_empty() {
            // Even if empty, we must confirm flush to unfreeze the slot!
            self.index.confirm_flush()?;
            return Ok(None);
        }

        let vectors: Vec<Vec<f32>> = data.iter().map(|(_, v)| v.clone()).collect();
        let ids: Vec<u64> = data.iter().map(|(id, _)| *id).collect();

        // 2. Train (Heavy CPU)
        {
            let q_guard = self.index.get_quantizer();
            if q_guard.is_none() {
                info!(count = data.len(), "Starting Index Training");
                self.index.train(&vectors).await?;
                info!(
                    duration_ms = start.elapsed().as_millis(),
                    "Training Complete"
                );
            }
        }

        // 3. Persist (Heavy IO)
        let new_id = self.index.allocate_next_bucket_id();
        self.index
            .force_register_bucket_with_ids(new_id, &ids, &vectors)
            .await?;

        let run_id = uuid::Uuid::new_v4().to_string();
        let path = self
            .persistence
            .flush_memtable_to_segment(&data, &self.index, &run_id)
            .await?;

        // 4. Confirm Flush (Cleanup)
        // This clears the frozen slot and truncates the WAL.
        self.index.confirm_flush()?;

        info!(
            vectors = data.len(),
            duration_ms = start.elapsed().as_millis(),
            "Flush Complete"
        );
        Ok(Some(path))
    }
}
