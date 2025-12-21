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

        loop {
            interval.tick().await;

            let size = self.index.memtable_len();
            if size >= self.flush_threshold {
                // println!("Janitor: Threshold reached ({} >= {}). Flushing...", size, self.flush_threshold);

                match self.perform_flush().await {
                    Ok(path) => {
                        if let Some(p) = path {
                            println!("Janitor: Flushed to {:?}", p);
                        }
                    }
                    Err(e) => eprintln!("Janitor Error: Failed to flush: {}", e),
                }
            }
        }
    }

    /// Performs the atomic swap and persistence.
    // async fn perform_flush(&self) -> std::io::Result<Option<std::path::PathBuf>> {
    //     // 1. Rotate & Get Data (Blocking operation, holds WAL lock briefly)
    //     let data = self.index.rotate_memtable()?;

    //     if data.is_empty() {
    //         return Ok(None);
    //     }

    //     // 2. Flush to Segment (Async I/O)
    //     let run_id = uuid::Uuid::new_v4().to_string();

    //     let path = self
    //         .persistence
    //         .flush_memtable_to_segment(&data, &self.index, &run_id)
    //         .await?;

    //     Ok(Some(path))
    // }

    /// Performs the atomic swap, training (if needed), promotion, and persistence.
    async fn perform_flush(&self) -> std::io::Result<Option<std::path::PathBuf>> {
        // 1. Rotate & Get Data (Blocking operation, holds WAL lock briefly)
        // The old MemTable is now disconnected. We own this data.
        let data = self.index.rotate_memtable()?;

        if data.is_empty() {
            return Ok(None);
        }

        // Prepare data vectors for processing
        let vectors: Vec<Vec<f32>> = data.iter().map(|(_, v)| v.clone()).collect();
        let ids: Vec<u64> = data.iter().map(|(id, _)| *id).collect();

        // 2. AUTO-TRAINING (Cold Start)
        // If the Quantizer is not set, we MUST train it now using this batch of data.
        // Otherwise, we can't compress (quantize) the vectors for L1 storage or Disk.
        let is_first_run = self.index.get_quantizer().is_none();

        if is_first_run {
            println!(
                "Janitor: First flush detected. Training index on {} samples...",
                vectors.len()
            );
            // This initializes Quantizer, Centroids, and Empty L1 Buckets
            self.index.train(&vectors);
        }

        // 3. PROMOTE TO L1 (Memory)
        // Now that we have a trained index (either existing or just trained),
        // we compress these raw vectors and insert them into the L1 buckets in RAM.
        // This keeps them searchable (compressed) while freeing the massive MemTable RAM usage.
        // Note: force_insert_l1 is thread-safe (uses internal RwLocks).
        for (i, vec) in vectors.iter().enumerate() {
            self.index.force_insert_l1(ids[i], vec);
        }

        // 4. PERSIST TO DISK (Durability)
        // We write the raw data to a "L0 Segment" on disk.
        // This ensures if we crash, we can recover this batch.
        let run_id = uuid::Uuid::new_v4().to_string();

        let path = self
            .persistence
            .flush_memtable_to_segment(
                &data,
                &self.index, // Index now has the quantizer set, so this will succeed.
                &run_id,
            )
            .await?;

        Ok(Some(path))
    }
}
