use drift_cache::LocalDiskManager;
use drift_core::index::{IndexOptions, VectorIndex};
use drift_core::quantizer::Quantizer;
use drift_storage::disk_manager::DiskManager;
use drift_storage::segment_reader::SegmentReader;
use drift_storage::segment_writer::SegmentWriter;
use std::path::{Path, PathBuf};

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

    /// FLUSH L1: Persist existing L1 buckets to a new Segment.
    pub async fn flush_to_segment(
        &self,
        index: &VectorIndex,
        run_id: &str,
    ) -> std::io::Result<PathBuf> {
        let file_name = format!("segment_{}.drift", run_id);
        let file_path = self.base_path.join(&file_name);

        let quantizer_arc = index.get_quantizer().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "Untrained index")
        })?;

        let q_bytes = bincode::encode_to_vec(&*quantizer_arc, bincode::config::standard())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let manager = DiskManager::open(&file_path).await?;
        let mut writer = SegmentWriter::new(manager, q_bytes);

        // Iterate Headers
        let headers = index.get_all_bucket_headers();

        for header in headers {
            // LOAD DATA from Cache (Async)
            // If it's not in cache, this fetches from disk.
            match index.cache.get(&header.page_id).await {
                Ok(data_arc) => {
                    let (vecs, ids) = data_arc.reconstruct(&quantizer_arc);
                    if !vecs.is_empty() {
                        writer.write_bucket(header.id, &ids, &vecs).await?;
                    }
                }
                Err(e) => eprintln!(
                    "Persistence Warning: Failed to load bucket {}: {}",
                    header.id, e
                ),
            }
        }

        writer.finalize().await?;
        println!("Flushed L1 index to {:?}", file_path);
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

        let quantizer_arc = index.get_quantizer().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "Untrained index")
        })?;

        let q_bytes = bincode::encode_to_vec(&*quantizer_arc, bincode::config::standard())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let manager = DiskManager::open(&file_path).await?;
        let mut writer = SegmentWriter::new(manager, q_bytes);

        // Write as single bucket ID 0 (will be remapped on load)
        let ids: Vec<u64> = data.iter().map(|(id, _)| *id).collect();
        let vecs: Vec<Vec<f32>> = data.iter().map(|(_, v)| v.clone()).collect();

        if !vecs.is_empty() {
            writer.write_bucket(0, &ids, &vecs).await?;
        }

        writer.finalize().await?;
        Ok(file_path)
    }

    pub async fn load_from_segment(&self, path: &Path) -> std::io::Result<VectorIndex> {
        let mut reader = SegmentReader::open(path).await?;

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

        // We need a storage backend for the new index
        // Assume sibling directory 'storage' relative to segment or CWD
        let storage_path = self.base_path.join("storage");
        let storage = std::sync::Arc::new(LocalDiskManager::new(storage_path));

        let wal_path = self.base_path.join("current.wal");
        let index = VectorIndex::new(options, &wal_path, storage)?;
        index.set_quantizer(quantizer);

        let bucket_ids: Vec<u32> = reader.index.buckets.keys().cloned().collect();
        for id in bucket_ids {
            let (ids, vectors) = reader.read_bucket(id).await?;
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
            println!(
                "Persistence: Hydrating segment {:?}",
                path.file_name().unwrap()
            );
            let mut reader = SegmentReader::open(&path).await?;

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

            let bucket_ids: Vec<u32> = reader.index.buckets.keys().cloned().collect();
            for _old_id in bucket_ids {
                let (ids, vectors) = reader.read_bucket(_old_id).await?;
                // Always allocate new ID to prevent collision
                let new_id = index.allocate_next_bucket_id();
                index
                    .force_register_bucket_with_ids(new_id, &ids, &vectors)
                    .await?;
            }
        }
        Ok(())
    }
}
