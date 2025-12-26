use crate::persistence::PersistenceManager;
use drift_core::index::{MaintenanceStatus, VectorIndex};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;
use tokio::time;

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
                    Ok(Some(p)) => println!("Janitor: Flushed to {:?}", p),
                    Ok(None) => {}
                    Err(e) => eprintln!("Janitor Error: Failed to flush: {}", e),
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
                println!(
                    "Janitor: ✂️ Splitting Bucket {} (Count {})",
                    header.id, header.count
                );

                match self.index.split_and_steal(header.id).await {
                    Ok(MaintenanceStatus::Completed) => {
                        ops_budget -= 1;
                    }
                    Ok(MaintenanceStatus::SkippedSingularity) => {
                        println!("Janitor: Bucket {} is a Singularity. Ignoring.", header.id);
                        self.ignore_set.lock().unwrap().insert(header.id);
                    }
                    Ok(_) => {}
                    Err(e) => eprintln!("Split failed: {}", e),
                }
            }
            // 2. MERGE
            else if header.count == 0 {
                println!(
                    "Janitor: 🚑 Scatter Merging Bucket {} (Count {})",
                    header.id, header.count
                );
                self.ignore_set.lock().unwrap().remove(&header.id); // Cleanup

                match self.index.scatter_merge(header.id).await {
                    Ok(MaintenanceStatus::Completed) => {
                        ops_budget -= 1;
                    }
                    Ok(_) => {}
                    Err(e) => eprintln!("Merge failed: {}", e),
                }
            }
        }
    }

    async fn perform_flush(&self) -> std::io::Result<Option<std::path::PathBuf>> {
        let data = self.index.rotate_memtable()?;
        if data.is_empty() {
            return Ok(None);
        }

        let vectors: Vec<Vec<f32>> = data.iter().map(|(_, v)| v.clone()).collect();
        let ids: Vec<u64> = data.iter().map(|(id, _)| *id).collect();

        {
            let q_guard = self.index.get_quantizer();
            if q_guard.is_none() {
                println!(
                    "Janitor: First flush. Training index on {} samples...",
                    vectors.len()
                );
                self.index.train(&vectors).await?;
                // return Ok(None);
            }
        }

        let new_id = self.index.allocate_next_bucket_id();
        self.index
            .force_register_bucket_with_ids(new_id, &ids, &vectors)
            .await?;

        let run_id = uuid::Uuid::new_v4().to_string();
        let path = self
            .persistence
            .flush_memtable_to_segment(&data, &self.index, &run_id)
            .await?;
        Ok(Some(path))
    }
}
