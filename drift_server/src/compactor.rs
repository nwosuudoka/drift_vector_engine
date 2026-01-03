use drift_core::index::VectorIndex;
use opendal::Operator;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{info, instrument};

pub struct SegmentCompactor {
    index: Arc<VectorIndex>,
    op: Operator,
}

impl SegmentCompactor {
    pub fn new(index: Arc<VectorIndex>, op: Operator) -> Self {
        Self { index, op }
    }

    /// The main entry point. Runs both cleaning cycles.
    pub async fn run_cycle(&self) -> std::io::Result<()> {
        self.compact_tombstones().await?;
        self.vacuum_segments().await?;
        Ok(())
    }

    /// 🧹 PHASE 1: VACUUM SEGMENTS
    /// Deletes "segment_*.drift" files that are no longer referenced by any live bucket.
    #[instrument(skip(self), level = "info")]
    pub async fn vacuum_segments(&self) -> std::io::Result<()> {
        // 1. Identify Live Files (Mark)
        let mut live_files = HashSet::new();

        // Get all headers from the index
        let headers = self.index.get_all_bucket_headers();
        let storage = self.index.cache.storage();

        for header in headers {
            let file_id = header.page_id.file_id;
            // Ask storage layer which file this ID maps to
            if let Some(path) = storage.get_physical_path(file_id) {
                live_files.insert(path);
            }
        }

        // 2. List All Files (Sweep)
        let entries = self.op.list("").await.map_err(std::io::Error::other)?;
        let mut deleted_count = 0;
        let mut reclaimed_bytes = 0;

        for entry in entries {
            let path = entry.path();

            // Only touch segment files
            if !path.starts_with("segment_") || !path.ends_with(".drift") {
                continue;
            }

            // If it's NOT in our live set, kill it.
            if !live_files.contains(path) {
                // Safety check: Don't delete brand new files created during a race?
                // The Janitor registers files BEFORE they are visible, so live_files
                // should capture them. But we can check mod time if we want to be paranoid.

                let meta = self.op.stat(path).await.map_err(std::io::Error::other)?;
                reclaimed_bytes += meta.content_length();

                info!("Compactor: 🗑️ Deleting orphan segment: {}", path);
                self.op.delete(path).await.map_err(std::io::Error::other)?;
                deleted_count += 1;
            }
        }

        if deleted_count > 0 {
            info!(
                "Compactor: Vacuum complete. Deleted {} segments, freed {} bytes.",
                deleted_count, reclaimed_bytes
            );
        }

        Ok(())
    }

    /// 🪦 PHASE 2: COMPACT TOMBSTONES
    /// Merges all active deletes into a single file and deletes old logs.
    #[instrument(skip(self), level = "info")]
    pub async fn compact_tombstones(&self) -> std::io::Result<()> {
        // 1. Snapshot Memory
        let deleted_snapshot: Vec<u64> = self.index.deleted_ids.read().iter().cloned().collect();

        if deleted_snapshot.is_empty() {
            return Ok(());
        }

        // 2. Write Unified Tombstone File
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let new_filename = format!("tombstones_compacted_{}.drift", timestamp);

        // Use PersistenceManager logic (re-implemented here briefly or helper)
        let file = drift_core::tombstone::TombstoneFile::new(deleted_snapshot);
        let bytes = file.to_bytes()?;

        self.op
            .write(&new_filename, bytes)
            .await
            .map_err(std::io::Error::other)?;
        info!(
            "Compactor: Wrote consolidated tombstones to {}",
            new_filename
        );

        // 3. Delete OLD Tombstone Files
        let entries = self.op.list("").await.map_err(std::io::Error::other)?;
        for entry in entries {
            let path = entry.path();

            // Delete old tombstone files, but NOT the one we just wrote
            if path.starts_with("tombstones_") && path.ends_with(".drift") && path != new_filename {
                info!("Compactor: 🗑️ Deleting obsolete tombstone log: {}", path);
                self.op.delete(path).await.map_err(std::io::Error::other)?;
            }
        }

        Ok(())
    }
}
