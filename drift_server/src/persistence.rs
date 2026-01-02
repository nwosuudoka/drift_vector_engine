use drift_core::index::{IndexOptions, VectorIndex};
use drift_core::quantizer::Quantizer;
use drift_core::tombstone::TombstoneFile;
use drift_storage::disk_manager::{DiskManager, DriftPageManager};
use drift_storage::segment_reader::SegmentReader;
use drift_storage::segment_writer::SegmentWriter;
use opendal::Operator;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{info, warn};

#[derive(Clone)]
pub struct PersistenceManager {
    op: Operator,
    local_base_path: PathBuf,
}

impl PersistenceManager {
    pub fn new(op: Operator, local_path: impl Into<PathBuf>) -> Self {
        Self {
            op,
            local_base_path: local_path.into(),
        }
    }

    pub async fn flush_to_segment(
        &self,
        index: &VectorIndex,
        run_id: &str,
    ) -> std::io::Result<String> {
        // Returns object key (filename)
        let file_name = format!("segment_{}.drift", run_id);

        let quantizer_arc = index.get_quantizer().ok_or(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Untrained",
        ))?;
        let q_bytes = bincode::encode_to_vec(&*quantizer_arc, bincode::config::standard())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        // ⚡ Use DI: DiskManager::new(op, path)
        let manager = DiskManager::new(self.op.clone(), file_name.clone());
        let mut writer = SegmentWriter::new(manager, q_bytes).await?;

        let headers = index.get_all_bucket_headers();
        for header in headers {
            if let Ok(data) = index.cache.get(&header.page_id).await {
                if !data.vids.is_empty() {
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
        }
        writer.finalize().await?;
        Ok(file_name)
    }

    pub async fn flush_memtable_to_segment(
        &self,
        data: &[(u64, Vec<f32>)],
        index: &VectorIndex,
        run_id: &str,
    ) -> std::io::Result<String> {
        let file_name = format!("segment_l0_{}.drift", run_id);

        let quantizer_arc = index.get_quantizer().ok_or(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Untrained",
        ))?;
        let q_bytes = bincode::encode_to_vec(&*quantizer_arc, bincode::config::standard())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        // ⚡ Use DI
        let manager = DiskManager::new(self.op.clone(), file_name.clone());
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
        Ok(file_name)
    }

    // Note: 'path' here is a relative object key string, not a PathBuf
    pub async fn load_from_segment(&self, object_key: &str) -> std::io::Result<VectorIndex> {
        // ⚡ Use DI: SegmentReader must act on the injected operator
        // We will add a helper to SegmentReader to accept (Operator, path)
        let reader = SegmentReader::open_with_op(self.op.clone(), object_key).await?;

        let q_bytes = reader.read_metadata();
        let (quantizer, _): (Quantizer, usize) =
            bincode::decode_from_slice(q_bytes, bincode::config::standard()).map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "Corrupt Quantizer")
            })?;
        let dim = quantizer.min.len();

        // Use default options for recovery
        let options = IndexOptions {
            dim,
            num_centroids: 0,
            training_sample_size: 0,
            max_bucket_capacity: 1000,
            ..Default::default()
        };

        // Rebuild storage layout
        let storage_path = self.local_base_path.join("storage");
        std::fs::create_dir_all(&storage_path)?;

        // Use DriftPageManager with the same Operator!
        let storage = Arc::new(DriftPageManager::new(self.op.clone()));
        let wal_path = self.local_base_path.join("current.wal");

        let index = VectorIndex::new(options, &wal_path, storage)?;
        index.set_quantizer(quantizer.clone());

        let bucket_ids: Vec<u32> = reader.index.buckets.keys().cloned().collect();
        for id in bucket_ids {
            let vectors = match reader.read_bucket_high_fidelity(id).await {
                Ok(v) => v,
                Err(_) => {
                    info!("Recovering bucket {} from SQ8", id);
                    let (_, codes) = reader.read_bucket(id).await?;
                    let count = codes.len() / dim;
                    let mut rec_vecs = Vec::with_capacity(count);
                    for i in 0..count {
                        let start = i * dim;
                        rec_vecs.push(quantizer.reconstruct(&codes[start..start + dim]));
                    }
                    rec_vecs
                }
            };
            let (ids, _) = reader.read_bucket(id).await?;
            index
                .force_register_bucket_with_ids(id, &ids, &vectors)
                .await?;
        }
        Ok(index)
    }

    pub async fn hydrate_index(&self, index: &VectorIndex) -> std::io::Result<()> {
        // List files from the Operator
        let lister = self.op.lister("").await.map_err(std::io::Error::other)?;
        // We need to collect and filter
        // OpenDAL Lister is async stream
        use opendal::Entry;
        // Basic list loop (simplified for brevity, assume < 1000 files for now)
        // In production use streaming
        let entries: Vec<Entry> = self.op.list("").await.map_err(std::io::Error::other)?;

        let mut paths = Vec::new();
        for entry in entries {
            let path = entry.path();
            if path.ends_with(".drift") {
                paths.push(path.to_string());
            }
        }
        paths.sort();

        for path in paths {
            // ⚡ Use DI
            let reader = SegmentReader::open_with_op(self.op.clone(), &path).await?;

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

            let q_ref = index.get_quantizer();
            let bucket_ids: Vec<u32> = reader.index.buckets.keys().cloned().collect();
            for _old_id in bucket_ids {
                let vectors = match reader.read_bucket_high_fidelity(_old_id).await {
                    Ok(v) => v,
                    Err(_) => {
                        // Fallback logic (same as before)
                        let (_, codes) = reader.read_bucket(_old_id).await?;
                        if let Some(q) = &q_ref {
                            let dim = index.config.dim;
                            let count = codes.len() / dim;
                            let mut rec_vecs = Vec::with_capacity(count);
                            for i in 0..count {
                                rec_vecs.push(q.reconstruct(&codes[i * dim..(i + 1) * dim]));
                            }
                            rec_vecs
                        } else {
                            return Err(std::io::Error::new(
                                std::io::ErrorKind::Other,
                                "Missing Quantizer",
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

    pub async fn flush_tombstones(&self, ids: &[u64], run_id: &str) -> std::io::Result<String> {
        if ids.is_empty() {
            return Ok("".to_string());
        }

        let file_name = format!("tombstones_{}.drift", run_id);
        let file = TombstoneFile::new(ids.to_vec());
        let bytes = file.to_bytes()?;

        let manager = DiskManager::new(self.op.clone(), file_name.clone());
        manager.upload(bytes).await?;

        info!("Flushed {} tombstones to {}", ids.len(), file_name);
        Ok(file_name)
    }

    pub async fn load_all_tombstones(&self) -> std::io::Result<Vec<u64>> {
        let mut all_deleted = Vec::new();
        // Listing
        let entries: Vec<opendal::Entry> = self.op.list("").await.map_err(std::io::Error::other)?;

        for entry in entries {
            let path = entry.path();
            if path.starts_with("tombstones_") && path.ends_with(".drift") {
                let manager = DiskManager::new(self.op.clone(), path.to_string());
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
