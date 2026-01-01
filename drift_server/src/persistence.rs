use drift_core::index::{IndexOptions, VectorIndex};
use drift_core::quantizer::Quantizer;
use drift_core::tombstone::TombstoneFile;
use drift_storage::disk_manager::{DiskManager, DriftPageManager};
use drift_storage::segment_reader::SegmentReader;
use drift_storage::segment_writer::SegmentWriter;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{info, warn};

#[derive(Clone)]
pub struct PersistenceManager {
    base_path: PathBuf,
}

impl PersistenceManager {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            base_path: path.into(),
        }
    }

    fn to_uri(path: &Path) -> String {
        let abs = std::fs::canonicalize(path).unwrap_or(path.to_path_buf());
        format!("file://{}", abs.to_string_lossy())
    }

    pub async fn flush_to_segment(
        &self,
        index: &VectorIndex,
        run_id: &str,
    ) -> std::io::Result<PathBuf> {
        let file_name = format!("segment_{}.drift", run_id);
        let file_path = self.base_path.join(&file_name);
        let uri = Self::to_uri(&file_path);

        let quantizer_arc = index.get_quantizer().ok_or(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Untrained",
        ))?;

        let q_bytes = bincode::encode_to_vec(&*quantizer_arc, bincode::config::standard())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let manager = DiskManager::open(&uri).await?;
        let mut writer = SegmentWriter::new(manager, q_bytes).await?;

        let headers = index.get_all_bucket_headers();
        for header in headers {
            if let Ok(data) = index.cache.get(&header.page_id).await
                && !data.vids.is_empty()
            {
                writer
                    .write_bucket_sq8(
                        header.id,
                        &data.vids,
                        data.codes.as_slice(),
                        index.config.dim,
                    )
                    .await?;
            }
        }
        writer.finalize().await?;
        Ok(file_path)
    }

    pub async fn flush_memtable_to_segment(
        &self,
        data: &[(u64, Vec<f32>)],
        index: &VectorIndex,
        run_id: &str,
    ) -> std::io::Result<PathBuf> {
        let file_name = format!("segment_l0_{}.drift", run_id);
        let file_path = self.base_path.join(&file_name);
        let uri = Self::to_uri(&file_path);

        let quantizer_arc = index.get_quantizer().ok_or(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Untrained",
        ))?;

        let q_bytes = bincode::encode_to_vec(&*quantizer_arc, bincode::config::standard())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let manager = DiskManager::open(&uri).await?;
        let mut writer = SegmentWriter::new(manager, q_bytes).await?;

        let ids: Vec<u64> = data.iter().map(|(id, _)| *id).collect();
        let vecs: Vec<Vec<f32>> = data.iter().map(|(_, v)| v.clone()).collect();

        if !vecs.is_empty() {
            let dim = index.config.dim;
            let mut flat_codes = Vec::with_capacity(vecs.len() * dim);
            for v in &vecs {
                flat_codes.extend_from_slice(&quantizer_arc.encode(v));
            }

            writer
                .write_bucket_dual(0, &ids, &vecs, &flat_codes, dim)
                .await?;
        }

        writer.finalize().await?;
        Ok(file_path)
    }

    pub async fn load_from_segment(&self, path: &Path) -> std::io::Result<VectorIndex> {
        let uri = Self::to_uri(path);
        let reader = SegmentReader::open(&uri).await?;

        let q_bytes = reader.read_metadata();
        let (quantizer, _): (Quantizer, usize) =
            bincode::decode_from_slice(q_bytes, bincode::config::standard()).map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "Corrupt Quantizer")
            })?;

        let dim = quantizer.min.len();
        let options = IndexOptions {
            dim,
            num_centroids: 0,
            training_sample_size: 0,
            max_bucket_capacity: 1000,
            ..Default::default()
        };

        let storage_path = self.base_path.join("storage");
        std::fs::create_dir_all(&storage_path)?;
        let storage_uri = Self::to_uri(&storage_path);
        let storage = Arc::new(DriftPageManager::new(&storage_uri).await?);

        let wal_path = self.base_path.join("current.wal");
        let index = VectorIndex::new(options, &wal_path, storage)?;
        index.set_quantizer(quantizer.clone()); // Clone needed for fallback logic below

        let bucket_ids: Vec<u32> = reader.index.buckets.keys().cloned().collect();
        for id in bucket_ids {
            // ⚡ RECOVERY FALLBACK
            // Try to read High Fidelity data first.
            // If missing (e.g. flushed from RAM-only state), recover from SQ8 codes.
            let vectors = match reader.read_bucket_high_fidelity(id).await {
                Ok(v) => v,
                Err(_) => {
                    info!("Recovering bucket {} from SQ8 (Low Fidelity)", id);
                    // Fast path read
                    let (_, codes) = reader.read_bucket(id).await?;

                    // Manual reconstruction
                    let dim = index.config.dim;
                    let count = codes.len() / dim;
                    let mut rec_vecs = Vec::with_capacity(count);

                    for i in 0..count {
                        let start = i * dim;
                        let code_slice = &codes[start..start + dim];
                        rec_vecs.push(quantizer.reconstruct(code_slice));
                    }
                    rec_vecs
                }
            };

            // Fetch IDs (always available in fast path)
            let (ids, _) = reader.read_bucket(id).await?;

            index
                .force_register_bucket_with_ids(id, &ids, &vectors)
                .await?;
        }
        Ok(index)
    }

    pub async fn hydrate_index(&self, index: &VectorIndex) -> std::io::Result<()> {
        let mut read_dir = tokio::fs::read_dir(&self.base_path).await?;
        let mut paths = Vec::new();
        while let Some(entry) = read_dir.next_entry().await? {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("drift") {
                paths.push(path);
            }
        }
        paths.sort();

        for path in paths {
            let uri = Self::to_uri(&path);
            let reader = SegmentReader::open(&uri).await?;

            if index.get_quantizer().is_none() {
                let q_bytes = reader.read_metadata();
                let (quantizer, _): (Quantizer, usize) = bincode::decode_from_slice(
                    q_bytes,
                    bincode::config::standard(),
                )
                .map_err(|_| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, "Corrupt Quantizer")
                })?;
                index.set_quantizer(quantizer);
            }

            // We need a handle to the quantizer for fallback reconstruction
            // (Only if hydration fails high-fidelity read)
            let q_ref = index.get_quantizer();

            let bucket_ids: Vec<u32> = reader.index.buckets.keys().cloned().collect();
            for _old_id in bucket_ids {
                // ⚡ HYDRATION FALLBACK
                let vectors = match reader.read_bucket_high_fidelity(_old_id).await {
                    Ok(v) => v,
                    Err(_) => {
                        info!("Hydrating bucket {} from SQ8 (Low Fidelity)", _old_id);
                        let (_, codes) = reader.read_bucket(_old_id).await?;

                        if let Some(q) = &q_ref {
                            let dim = index.config.dim;
                            let count = codes.len() / dim;
                            let mut rec_vecs = Vec::with_capacity(count);
                            for i in 0..count {
                                let start = i * dim;
                                let code_slice = &codes[start..start + dim];
                                rec_vecs.push(q.reconstruct(code_slice));
                            }
                            rec_vecs
                        } else {
                            // Should be impossible if we just set it above
                            return Err(std::io::Error::other(
                                "Missing Quantizer during hydration",
                            ));
                        }
                    }
                };

                let (ids, _) = reader.read_bucket(_old_id).await?;
                let new_id = index.allocate_next_bucket_id();
                index
                    .force_register_bucket_with_ids(new_id, &ids, &vectors)
                    .await?;
            }
        }
        Ok(())
    }

    /// Flushes a snapshot of deleted IDs to a sidecar file.
    pub async fn flush_tombstones(&self, ids: &[u64], run_id: &str) -> std::io::Result<PathBuf> {
        if ids.is_empty() {
            return Ok(self.base_path.clone());
        }

        let file_name = format!("tombstones_{}.drift", run_id);
        let file_path = self.base_path.join(&file_name);
        let uri = Self::to_uri(&file_path);

        let file = TombstoneFile::new(ids.to_vec());
        let bytes = file.to_bytes()?;

        let manager = DiskManager::open(&uri).await?;
        manager.upload(bytes).await?;

        info!("Flushed {} tombstones to {}", ids.len(), file_name);
        Ok(file_path)
    }

    /// Loads ALL tombstone files and returns the aggregated set.
    pub async fn load_all_tombstones(&self) -> std::io::Result<Vec<u64>> {
        let mut all_deleted = Vec::new();

        if !self.base_path.exists() {
            return Ok(vec![]);
        }

        let mut read_dir = tokio::fs::read_dir(&self.base_path).await?;

        while let Some(entry) = read_dir.next_entry().await? {
            let path = entry.path();
            let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");

            if name.starts_with("tombstones_") && name.ends_with(".drift") {
                let uri = Self::to_uri(&path);
                let manager = DiskManager::open(&uri).await?;

                let len = manager.len().await?;
                if len > 0 {
                    let bytes = manager.read_at(0, len as usize).await?;
                    match TombstoneFile::from_bytes(&bytes) {
                        Ok(file) => all_deleted.extend(file.deleted_ids),
                        Err(e) => warn!("Failed to load tombstone file {:?}: {}", path, e),
                    }
                }
            }
        }
        Ok(all_deleted)
    }
}
