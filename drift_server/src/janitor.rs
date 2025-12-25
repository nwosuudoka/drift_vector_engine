use crate::persistence::PersistenceManager;
use drift_core::index::VectorIndex;
use std::sync::Arc;
use std::time::Duration;
use tokio::time;

pub struct Janitor {
    index: Arc<VectorIndex>,
    persistence: PersistenceManager,
    flush_threshold: usize,
    check_interval: Duration,
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
        }
    }

    /// Runs the background loop. Call this via `tokio::spawn`.
    pub async fn run(&self) {
        let mut interval = time::interval(self.check_interval);

        // Run maintenance less frequently than flush checks (e.g. every 10 ticks)
        let mut ticks = 0;

        loop {
            interval.tick().await;
            ticks += 1;

            // 1. Flush Logic (L0 -> L1)
            let size = self.index.memtable_len();
            if size >= self.flush_threshold {
                match self.perform_flush().await {
                    Ok(Some(p)) => println!("Janitor: Flushed to {:?}", p),
                    Ok(None) => {}
                    Err(e) => eprintln!("Janitor Error: Failed to flush: {}", e),
                }
            }

            // 2. Maintenance Logic (L1 -> L1) - Run every 10th tick
            // This is the "Self-Healing" heartbeat.
            if ticks % 10 == 0 {
                self.perform_maintenance();
            }
        }
    }

    fn perform_maintenance(&self) {
        let buckets = self.index.get_all_buckets();
        let target_cap = self.index.config.max_bucket_capacity;

        let mut best_merge_candidate = None;
        let mut max_urgency = 0.0;
        let mut best_split_candidate = None;

        for bucket in &buckets {
            bucket.decay_temperature();

            // 1. Check Merge Urgency (Healing)
            let urgency = bucket.calculate_urgency(target_cap);
            if urgency > max_urgency {
                max_urgency = urgency;
                best_merge_candidate = Some(bucket.id);
            }

            // 2. Check Split Criteria (Growth) - SDD Section 3.C
            // We prioritize the *first* valid split candidate we find to avoid scanning everything strictly
            if best_split_candidate.is_none() && bucket.should_split(target_cap) {
                best_split_candidate = Some(bucket.id);
            }
        }

        // DECISION LOGIC: Heal first, then Grow.
        if max_urgency > 1.0
            && let Some(id) = best_merge_candidate
        {
            println!(
                "Janitor: 🚑 Scatter Merging Bucket {} (Urgency {:.2})",
                id, max_urgency
            );
            let _ = self.index.scatter_merge(id);
            return; // One op per tick
        }

        if let Some(id) = best_split_candidate {
            println!(
                "Janitor: ✂️ Splitting Bucket {} (Drift/Capacity detected)",
                id
            );
            self.index.split_and_steal(id);
        }
    }

    /// Performs the atomic swap, training (if needed), promotion, and persistence.
    async fn perform_flush(&self) -> std::io::Result<Option<std::path::PathBuf>> {
        // 1. Rotate & Get Data (Blocking operation, holds WAL lock briefly)
        let data = self.index.rotate_memtable()?;

        if data.is_empty() {
            return Ok(None);
        }

        // Prepare data vectors for processing
        let vectors: Vec<Vec<f32>> = data.iter().map(|(_, v)| v.clone()).collect();
        let ids: Vec<u64> = data.iter().map(|(id, _)| *id).collect();

        // 2. AUTO-TRAINING (Cold Start)
        let is_first_run = self.index.get_quantizer().is_none();

        if is_first_run {
            println!(
                "Janitor: First flush detected. Training index on {} samples...",
                vectors.len()
            );
            self.index.train(&vectors);
        }

        // 3. PROMOTE TO L1 (Memory)
        for (i, vec) in vectors.iter().enumerate() {
            self.index.force_insert_l1(ids[i], vec);
        }

        // 4. PERSIST TO DISK (Durability)
        let run_id = uuid::Uuid::new_v4().to_string();

        let path = self
            .persistence
            .flush_memtable_to_segment(&data, &self.index, &run_id)
            .await?;

        Ok(Some(path))
    }
}
