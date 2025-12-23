use crate::persistence::PersistenceManager;
use drift_core::index::VectorIndex;
use std::sync::Arc;
use std::sync::atomic::Ordering;
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

    /// Scans L1 buckets, decays temperature, and triggers Split or Merge if urgency is high.
    fn perform_maintenance(&self) {
        // We only want ONE heavy operation per cycle to avoid thrashing.
        // We scan all buckets to find the "best" candidate for maintenance.

        let buckets = self.index.get_all_buckets();
        if buckets.is_empty() {
            return;
        }

        let max_capacity = self.index.config.max_bucket_capacity;
        let mut best_bucket_id = None;
        let mut max_urgency = 0.0;
        let mut is_zombie_candidate = false;

        for bucket in &buckets {
            // 1. Thermodynamics: Cool down the bucket
            // If it hasn't been searched recently, its temperature drops.
            bucket.decay_temperature();

            // 2. Calculate Urgency (The "Hot Zombie" Formula)
            let urgency = bucket.calculate_urgency(max_capacity);

            // 3. Track the worst offender
            if urgency > max_urgency {
                max_urgency = urgency;
                best_bucket_id = Some(bucket.id);

                // Classify the problem: Is it dead (Zombie) or just full?
                let count = bucket.count.load(Ordering::Relaxed) as f32;
                let dead = bucket.tombstone_count.load(Ordering::Relaxed) as f32;

                // If > 30% of data is dead, it's a Zombie.
                // Even if it's hot, we want to merge it to reclaim space/efficiency.
                is_zombie_candidate = count > 0.0 && (dead / count) > 0.3;
            }
        }

        // 4. Execute Maintenance (If Threshold Met)
        // Threshold 1.0 is the baseline for "Needs Attention"
        if max_urgency > 1.0
            && let Some(id) = best_bucket_id
        {
            println!(
                "Janitor: Maintenance triggered for Bucket {} (Urgency: {:.2})",
                id, max_urgency
            );

            if is_zombie_candidate {
                println!(
                    "Janitor: 🚑 Scatter Merging Bucket {} (Zombie Detected)",
                    id
                );
                if let Err(e) = self.index.scatter_merge(id) {
                    eprintln!("Janitor Error during Merge: {}", e);
                }
            } else {
                println!("Janitor: ✂️ Splitting Bucket {} (Capacity Pressure)", id);
                // Split is CPU bound. In a real system, we might spawn_blocking.
                self.index.split_and_steal(id);
            }
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
