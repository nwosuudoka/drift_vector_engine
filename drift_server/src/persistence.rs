use crate::global_metadata_snapshot::GlobalMetadataSnapshot;
use drift_core::manifest::BucketPayloadIndexMeta;
use drift_core::tombstone::TombstoneFile;
use drift_storage::disk_manager::DiskManager;
use drift_storage::unified_format::{
    UNIFIED_FLAG_HAS_EXACT_INDEX, UNIFIED_FLAG_HAS_PAYLOAD_COLUMNS, UNIFIED_FLAG_HAS_PAYLOAD_STATS,
    UNIFIED_HEADER_SIZE, UnifiedBlockType, UnifiedHeader, UnifiedPayloadRow, UnifiedPayloadSchema,
};
use drift_storage::unified_reader::UnifiedReader;
use drift_storage::unified_writer::UnifiedRemoteWriter;
use opendal::{Metadata, Operator};
use std::io;
use tracing::{info, warn};

#[derive(Clone)]
pub struct PersistenceManager {
    op: Operator,
}

#[derive(Debug, Clone)]
pub struct RemoteUnifiedWriteResult {
    pub run_id: String,
    pub object_path: String,
    pub row_count: u64,
    pub payload_index_meta: BucketPayloadIndexMeta,
}

#[derive(Debug, Clone)]
pub struct RemoteGlobalMetadataWriteResult {
    pub run_id: String,
    pub object_path: String,
    pub object_fingerprint: String,
    pub format_version: u32,
    pub encoded_size_bytes: usize,
}

impl PersistenceManager {
    fn object_fingerprint(meta: &Metadata) -> String {
        let mut fields = Vec::with_capacity(5);
        fields.push(format!("len={}", meta.content_length()));
        if let Some(v) = meta.version() {
            fields.push(format!("version={}", v));
        }
        if let Some(etag) = meta.etag() {
            fields.push(format!("etag={}", etag));
        }
        if let Some(md5) = meta.content_md5() {
            fields.push(format!("md5={}", md5));
        }
        if let Some(ts) = meta.last_modified() {
            fields.push(format!("last_modified={:?}", ts));
        }
        fields.join("|")
    }

    pub fn new(op: Operator) -> Self {
        Self { op }
    }

    pub fn operator(&self) -> Operator {
        self.op.clone()
    }

    pub fn remote_bucket_path(&self, bucket_id: u32, run_id: &str) -> String {
        format!("bucket_{}_{}.driftu", bucket_id, run_id)
    }

    pub fn remote_global_metadata_path(&self, run_id: &str) -> String {
        format!("global_metadata_{}.driftmeta", run_id)
    }

    pub async fn read_remote_bucket_path(
        &self,
        path: &str,
    ) -> io::Result<(Vec<u64>, Vec<Vec<f32>>)> {
        if !self.op.exists(path).await.map_err(io::Error::other)? {
            return Ok((vec![], vec![]));
        }

        info!("Persistence: Reading remote segment {}", path);
        let mut reader = UnifiedReader::open(self.op.clone(), path).await?;
        reader.read_all_vectors().await
    }

    pub async fn read_remote_bucket_path_flat(
        &self,
        path: &str,
    ) -> io::Result<(Vec<u64>, Vec<f32>)> {
        if !self.op.exists(path).await.map_err(io::Error::other)? {
            return Ok((vec![], vec![]));
        }

        info!("Persistence: Reading remote segment {}", path);
        let mut reader = UnifiedReader::open(self.op.clone(), path).await?;
        reader.read_all_vectors_flat().await
    }

    pub async fn read_remote_bucket_payload_schema_path(
        &self,
        path: &str,
    ) -> io::Result<Option<UnifiedPayloadSchema>> {
        if !self.op.exists(path).await.map_err(io::Error::other)? {
            return Ok(None);
        }

        info!("Persistence: Reading payload schema from {}", path);
        let reader = UnifiedReader::open(self.op.clone(), path).await?;
        reader.read_payload_schema().await
    }

    pub async fn read_remote_bucket_payload_rows_path(
        &self,
        path: &str,
    ) -> io::Result<Vec<UnifiedPayloadRow>> {
        if !self.op.exists(path).await.map_err(io::Error::other)? {
            return Ok(Vec::new());
        }

        info!("Persistence: Reading payload rows from {}", path);
        let reader = UnifiedReader::open(self.op.clone(), path).await?;
        reader.read_payload_rows().await
    }

    pub async fn read_remote_bucket_payload_rows_path_optional(
        &self,
        path: &str,
    ) -> io::Result<Option<Vec<UnifiedPayloadRow>>> {
        if !self.op.exists(path).await.map_err(io::Error::other)? {
            return Ok(None);
        }

        info!("Persistence: Reading payload rows (optional) from {}", path);
        let reader = UnifiedReader::open(self.op.clone(), path).await?;
        let has_payload_columns = reader
            .blocks
            .iter()
            .any(|b| b.block_type == UnifiedBlockType::PayloadColumn);
        if !has_payload_columns {
            return Ok(None);
        }
        Ok(Some(reader.read_payload_rows().await?))
    }

    /// READ: Fetches raw vectors from an existing S3 segment.
    pub async fn read_remote_bucket(
        &self,
        bucket_id: u32,
        run_id: &str,
    ) -> io::Result<(Vec<u64>, Vec<Vec<f32>>)> {
        let key = self.remote_bucket_path(bucket_id, run_id);
        self.read_remote_bucket_path(&key).await
    }

    pub async fn read_remote_bucket_flat(
        &self,
        bucket_id: u32,
        run_id: &str,
    ) -> io::Result<(Vec<u64>, Vec<f32>)> {
        let key = self.remote_bucket_path(bucket_id, run_id);
        self.read_remote_bucket_path_flat(&key).await
    }

    pub async fn read_remote_bucket_payload_schema(
        &self,
        bucket_id: u32,
        run_id: &str,
    ) -> io::Result<Option<UnifiedPayloadSchema>> {
        let key = self.remote_bucket_path(bucket_id, run_id);
        self.read_remote_bucket_payload_schema_path(&key).await
    }

    pub async fn read_remote_bucket_payload_rows(
        &self,
        bucket_id: u32,
        run_id: &str,
    ) -> io::Result<Vec<UnifiedPayloadRow>> {
        let key = self.remote_bucket_path(bucket_id, run_id);
        self.read_remote_bucket_payload_rows_path(&key).await
    }

    pub async fn read_remote_bucket_payload_rows_optional(
        &self,
        bucket_id: u32,
        run_id: &str,
    ) -> io::Result<Option<Vec<UnifiedPayloadRow>>> {
        let key = self.remote_bucket_path(bucket_id, run_id);
        self.read_remote_bucket_payload_rows_path_optional(&key)
            .await
    }

    /// WRITE: Persists vectors to a new unified immutable segment (`.driftu`).
    pub async fn write_remote_bucket(
        &self,
        bucket_id: u32,
        ids: &[u64],
        vectors: &[Vec<f32>],
        dim: usize,
    ) -> io::Result<(String, u64)> {
        let flat: Vec<f32> = vectors.iter().flatten().copied().collect();
        self.write_remote_bucket_unified_flat(bucket_id, ids, &flat, dim)
            .await
    }

    /// WRITE: Persists vectors to a new unified immutable segment (`.driftu`).
    pub async fn write_remote_bucket_unified(
        &self,
        bucket_id: u32,
        ids: &[u64],
        vectors: &[Vec<f32>],
        dim: usize,
    ) -> io::Result<(String, u64)> {
        self.write_remote_bucket_unified_with_schema(bucket_id, ids, vectors, dim, None)
            .await
    }

    pub async fn write_remote_bucket_unified_with_schema(
        &self,
        bucket_id: u32,
        ids: &[u64],
        vectors: &[Vec<f32>],
        dim: usize,
        payload_schema: Option<&UnifiedPayloadSchema>,
    ) -> io::Result<(String, u64)> {
        let flat: Vec<f32> = vectors.iter().flatten().copied().collect();
        self.write_remote_bucket_unified_flat_with_schema(
            bucket_id,
            ids,
            &flat,
            dim,
            payload_schema,
        )
        .await
    }

    pub async fn write_remote_bucket_unified_flat(
        &self,
        bucket_id: u32,
        ids: &[u64],
        flat_vectors: &[f32],
        dim: usize,
    ) -> io::Result<(String, u64)> {
        self.write_remote_bucket_unified_flat_with_schema(bucket_id, ids, flat_vectors, dim, None)
            .await
    }

    pub async fn write_remote_bucket_unified_flat_with_schema(
        &self,
        bucket_id: u32,
        ids: &[u64],
        flat_vectors: &[f32],
        dim: usize,
        payload_schema: Option<&UnifiedPayloadSchema>,
    ) -> io::Result<(String, u64)> {
        self.write_remote_bucket_unified_flat_with_payload(
            bucket_id,
            ids,
            flat_vectors,
            dim,
            payload_schema,
            None,
        )
        .await
    }

    pub async fn write_remote_bucket_unified_flat_with_payload(
        &self,
        bucket_id: u32,
        ids: &[u64],
        flat_vectors: &[f32],
        dim: usize,
        payload_schema: Option<&UnifiedPayloadSchema>,
        payload_rows: Option<&[UnifiedPayloadRow]>,
    ) -> io::Result<(String, u64)> {
        let result = self
            .write_remote_bucket_unified_flat_with_payload_result(
                bucket_id,
                ids,
                flat_vectors,
                dim,
                payload_schema,
                payload_rows,
            )
            .await?;
        Ok((result.run_id, result.row_count))
    }

    pub async fn write_remote_bucket_unified_flat_with_payload_result(
        &self,
        bucket_id: u32,
        ids: &[u64],
        flat_vectors: &[f32],
        dim: usize,
        payload_schema: Option<&UnifiedPayloadSchema>,
        payload_rows: Option<&[UnifiedPayloadRow]>,
    ) -> io::Result<RemoteUnifiedWriteResult> {
        if ids.is_empty() {
            return Ok(RemoteUnifiedWriteResult {
                run_id: String::new(),
                object_path: String::new(),
                row_count: 0,
                payload_index_meta: BucketPayloadIndexMeta::default(),
            });
        }

        let new_run_id = uuid::Uuid::new_v4().to_string();
        let new_key = format!("bucket_{}_{}.driftu", bucket_id, new_run_id);
        let bytes = UnifiedRemoteWriter::write_vector_with_payload_flat_to_bytes(
            ids,
            flat_vectors,
            dim,
            payload_schema,
            payload_rows,
        )?;
        let payload_index_meta = Self::payload_index_meta_from_bytes(&bytes)?;

        self.op
            .write(&new_key, bytes)
            .await
            .map_err(io::Error::other)?;

        info!(
            "Persistence: Wrote unified bucket {} to {} ({} items, schema={}, payload_rows={})",
            bucket_id,
            new_key,
            ids.len(),
            payload_index_meta.has_payload_columns,
            payload_rows.is_some()
        );

        Ok(RemoteUnifiedWriteResult {
            run_id: new_run_id,
            object_path: new_key,
            row_count: ids.len() as u64,
            payload_index_meta,
        })
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

    pub async fn object_fingerprint_for_path(&self, path: &str) -> io::Result<String> {
        let meta = self.op.stat(path).await.map_err(io::Error::other)?;
        Ok(Self::object_fingerprint(&meta))
    }

    pub async fn write_global_metadata_snapshot(
        &self,
        snapshot: &GlobalMetadataSnapshot,
    ) -> io::Result<RemoteGlobalMetadataWriteResult> {
        let run_id = uuid::Uuid::new_v4().to_string();
        let object_path = self.remote_global_metadata_path(&run_id);
        let bytes = snapshot.encode_to_bytes()?;

        self.op
            .write(&object_path, bytes.clone())
            .await
            .map_err(io::Error::other)?;
        let object_fingerprint = self.object_fingerprint_for_path(&object_path).await?;

        info!(
            "Persistence: wrote global metadata snapshot {} ({} bytes, format_version={})",
            object_path,
            bytes.len(),
            snapshot.format_version
        );

        Ok(RemoteGlobalMetadataWriteResult {
            run_id,
            object_path,
            object_fingerprint,
            format_version: snapshot.format_version,
            encoded_size_bytes: bytes.len(),
        })
    }

    pub async fn read_global_metadata_snapshot_path(
        &self,
        path: &str,
    ) -> io::Result<Option<GlobalMetadataSnapshot>> {
        if !self.op.exists(path).await.map_err(io::Error::other)? {
            return Ok(None);
        }

        let bytes = self.op.read(path).await.map_err(io::Error::other)?.to_vec();
        let snapshot = GlobalMetadataSnapshot::decode_from_bytes(&bytes)?;
        Ok(Some(snapshot))
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

    fn payload_index_meta_from_bytes(bytes: &[u8]) -> io::Result<BucketPayloadIndexMeta> {
        if bytes.len() < UNIFIED_HEADER_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unified object too small to decode header",
            ));
        }

        let header = UnifiedHeader::decode(&bytes[..UNIFIED_HEADER_SIZE])?;
        Ok(BucketPayloadIndexMeta {
            has_payload_columns: (header.flags & UNIFIED_FLAG_HAS_PAYLOAD_COLUMNS) != 0,
            has_exact_index: (header.flags & UNIFIED_FLAG_HAS_EXACT_INDEX) != 0,
            has_payload_stats: (header.flags & UNIFIED_FLAG_HAS_PAYLOAD_STATS) != 0,
            payload_schema_hash: header.payload_schema_hash,
        })
    }
}
