use drift_core::{quantizer::Quantizer, tombstone::TombstoneFile};
use drift_storage::bucket_file_reader::BucketFileReader;
use drift_storage::bucket_file_writer::BucketFileWriter;
use opendal::Operator;
use std::io::{self, Cursor};
use tracing::info;

#[derive(Clone)]
pub struct PersistenceManager {
    op: Operator,
}

impl PersistenceManager {
    pub fn new(op: Operator) -> Self {
        Self { op }
    }

    /// Merges local staging data with an existing S3 segment to create a new, optimized S3 segment.
    pub async fn promote_to_s3(
        &self,
        bucket_id: u32,
        local_ids: &[u64],
        local_vecs: &[Vec<f32>], // Row-major vectors
        old_run_id: Option<String>,
        dim: usize,
    ) -> io::Result<(String, u64)> {
        let mut merged_ids = local_ids.to_vec();
        let mut merged_vecs = local_vecs.to_vec();

        // 1. Download & Merge Remote Data (if exists)
        if let Some(rid) = &old_run_id {
            let key = format!("bucket_{}_{}.drift", bucket_id, rid);
            if self.op.exists(&key).await.map_err(io::Error::other)? {
                info!("Persistence: Pulling remote segment {} for merge", key);

                // Open Reader
                let mut reader = BucketFileReader::open(self.op.clone(), &key).await?;

                // Read ALL data (Decompresses ALP -> f32)
                let (remote_ids, remote_vecs) = reader.read_all_vectors().await?;

                merged_ids.extend(remote_ids);
                merged_vecs.extend(remote_vecs);
            }
        }

        if merged_ids.is_empty() {
            return Ok((String::new(), 0));
        }

        // 2. Retrain Quantizer (Adapt to data drift)
        // Flatten for training
        let flat_samples: Vec<f32> = merged_vecs.iter().flatten().copied().collect();
        let q = Quantizer::train(&flat_samples, dim);

        // 3. Stream Write to Memory Buffer -> S3
        // We use Cursor<Vec<u8>> because BucketFileWriter requires the Truncatable trait.
        let mut buffer = Cursor::new(Vec::new());
        let new_run_id = uuid::Uuid::new_v4().to_string();
        let new_key = format!("bucket_{}_{}.drift", bucket_id, new_run_id);

        // Start writer with NEW quantizer
        let mut writer = BucketFileWriter::new_streaming(&mut buffer, [0u8; 16], q, dim)?;

        // Write the merged data as ONE large batch
        let flat_merged: Vec<f32> = merged_vecs.into_iter().flatten().collect();
        writer.write_batch(&merged_ids, &flat_merged)?;

        // Finalize (Writes Footer & Syncs)
        let (_, total_count) = writer.finalize()?;

        // Upload the bytes
        let final_bytes = buffer.into_inner();
        self.op
            .write(&new_key, final_bytes)
            .await
            .map_err(io::Error::other)?;

        info!(
            "Persistence: Promoted bucket {} to {} ({} items)",
            bucket_id, new_key, total_count
        );

        // Note: Old file deletion is handled by Reaper, not here.

        Ok((new_run_id, total_count))
    }

    pub async fn delete_file(&self, path: &str) -> std::io::Result<()> {
        self.op.delete(path).await.map_err(std::io::Error::other)
    }

    pub async fn flush_tombstones(&self, ids: &[u64], run_id: &str) -> std::io::Result<String> {
        if ids.is_empty() {
            return Ok("".to_string());
        }

        let file_name = format!("tombstones_{}.drift", run_id);
        let file = TombstoneFile::new(ids.to_vec());
        let bytes = file.to_bytes()?;

        self.op
            .write(&file_name, bytes)
            .await
            .map_err(std::io::Error::other)?;

        info!(
            "Persistence: Flushed {} tombstones to {}",
            ids.len(),
            file_name
        );
        Ok(file_name)
    }

    // Used by CollectionManager on startup to hydrate the deletion set.
    pub async fn load_all_tombstones(&self) -> std::io::Result<Vec<u64>> {
        let mut all_deleted = Vec::new();

        // List all files in the root
        let entries = self.op.list("").await.map_err(std::io::Error::other)?;

        for entry in entries {
            let path = entry.path();
            if path.starts_with("tombstones_") && path.ends_with(".drift") {
                let bytes = self
                    .op
                    .read(path)
                    .await
                    .map_err(std::io::Error::other)?
                    .to_vec();

                match TombstoneFile::from_bytes(&bytes) {
                    Ok(file) => all_deleted.extend(file.deleted_ids),
                    Err(e) => tracing::warn!("Failed to load tombstone file {}: {}", path, e),
                }
            }
        }
        Ok(all_deleted)
    }
}
