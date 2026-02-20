use drift_core::{quantizer::Quantizer, tombstone::TombstoneFile};
use drift_storage::bucket_file_reader::BucketFileReader;
use drift_storage::bucket_file_writer::BucketFileWriter;
use drift_storage::disk_manager::DiskManager;
use opendal::Operator;
use std::io::{self, Cursor};
use tracing::{info, warn};

#[derive(Clone)]
pub struct PersistenceManager {
    op: Operator,
}

impl PersistenceManager {
    pub fn new(op: Operator) -> Self {
        Self { op }
    }

    /// READ: Fetches raw vectors from an existing S3 segment.
    pub async fn read_remote_bucket(
        &self,
        bucket_id: u32,
        run_id: &str,
    ) -> io::Result<(Vec<u64>, Vec<Vec<f32>>)> {
        let key = format!("bucket_{}_{}.drift", bucket_id, run_id);

        if !self.op.exists(&key).await.map_err(io::Error::other)? {
            return Ok((vec![], vec![]));
        }

        info!("Persistence: Reading remote segment {}", key);
        let mut reader = BucketFileReader::open(self.op.clone(), &key).await?;
        reader.read_all_vectors().await
    }

    /// ⚡ WRITE: Persists clean vectors to a new S3 segment.
    /// Handles Quantization and Serialization.
    pub async fn write_remote_bucket(
        &self,
        bucket_id: u32,
        ids: &[u64],
        vectors: &[Vec<f32>],
        dim: usize,
    ) -> io::Result<(String, u64)> {
        if ids.is_empty() {
            return Ok((String::new(), 0));
        }

        // 1. Train Quantizer
        let flat_samples: Vec<f32> = vectors.iter().flatten().copied().collect();
        let q = Quantizer::train(&flat_samples, dim);

        // 2. Stream Write
        let mut buffer = Cursor::new(Vec::new());
        let new_run_id = uuid::Uuid::new_v4().to_string();
        let new_key = format!("bucket_{}_{}.drift", bucket_id, new_run_id);

        let mut writer = BucketFileWriter::new_streaming(&mut buffer, [0u8; 16], q, dim)?;
        writer.write_batch(ids, &flat_samples)?;

        let (_, total_count) = writer.finalize()?;

        // 3. Upload
        let final_bytes = buffer.into_inner();
        self.op
            .write(&new_key, final_bytes)
            .await
            .map_err(io::Error::other)?;

        info!(
            "Persistence: Wrote bucket {} to {} ({} items)",
            bucket_id, new_key, total_count
        );

        Ok((new_run_id, total_count))
    }

    pub async fn delete_file(&self, path: &str) -> std::io::Result<()> {
        self.op.delete(path).await.map_err(std::io::Error::other)?;
        if let Err(e) = DiskManager::invalidate_nvme_cache_for_object(&self.op, path).await {
            warn!(
                "Persistence: failed to invalidate NVMe cache for {}: {}",
                path, e
            );
        }
        Ok(())
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
