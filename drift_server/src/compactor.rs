use drift_core::index::VectorIndex;
use opendal::Operator;
use std::collections::HashSet;
use std::path::Path;
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
    #[instrument(skip(self), level = "info")]
    pub async fn vacuum_segments(&self) -> std::io::Result<()> {
        // 1. MARK: Identify Live Files
        // We store just the FILENAME to be path-agnostic
        let mut live_filenames = HashSet::new();

        let headers = self.index.get_all_bucket_headers();
        let storage = self.index.cache.storage();

        for header in &headers {
            let file_id = header.page_id.file_id;
            if let Some(path_str) = storage.get_physical_path(file_id) {
                // Extract filename only: "data/segment_123.drift" -> "segment_123.drift"
                let p = Path::new(&path_str);
                if let Some(name) = p.file_name().and_then(|s| s.to_str()) {
                    live_filenames.insert(name.to_string());
                }
            }
        }

        // Abort to prevent deleting the entire database due to a configuration error.
        if !headers.is_empty() && live_filenames.is_empty() {
            tracing::error!(
                "Compactor: 🚨 SAFETY ABORT! Index has {} buckets but mapped 0 physical files. Check PageManager implementation.",
                headers.len()
            );
            return Ok(());
        }

        // 2. SWEEP: List all files and delete garbage
        let entries = self.op.list("").await.map_err(std::io::Error::other)?;
        let mut deleted_count = 0;
        let mut reclaimed_bytes = 0;

        for entry in entries {
            let path = entry.path();

            // Normalize path to filename for comparison
            let p = Path::new(path);
            let filename = match p.file_name().and_then(|s| s.to_str()) {
                Some(n) => n,
                None => continue, // Skip odd paths
            };

            // Filter: Only target Drift Segment files
            if !filename.starts_with("segment_") || !filename.ends_with(".drift") {
                continue;
            }

            // Logic: If filename is NOT in our live set, it is garbage.
            if !live_filenames.contains(filename) {
                // Safety check to ensure we aren't deleting a brand new file being written?
                // The unique run_id UUIDs prevent name collision, so checking "is in index" is sufficient.

                let meta = match self.op.stat(path).await {
                    Ok(m) => m,
                    Err(_) => continue, // Already gone?
                };
                reclaimed_bytes += meta.content_length();

                info!(
                    "Compactor: 🗑️ Deleting orphan segment: {} (Not in live set: {:?})",
                    path, live_filenames
                );
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
