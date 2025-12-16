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
    async fn perform_flush(&self) -> std::io::Result<Option<std::path::PathBuf>> {
        // 1. Rotate & Get Data (Blocking operation, holds WAL lock briefly)
        let data = self.index.rotate_memtable()?;

        if data.is_empty() {
            return Ok(None);
        }

        // 2. Flush to Segment (Async I/O)
        let run_id = uuid::Uuid::new_v4().to_string();

        let path = self
            .persistence
            .flush_memtable_to_segment(&data, &self.index, &run_id)
            .await?;

        Ok(Some(path))
    }
}
