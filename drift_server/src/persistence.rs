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

        // 1. Get Quantizer Snapshot
        let quantizer_arc = index.get_quantizer().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Cannot flush untrained index",
            )
        })?;

        // Serialize Quantizer for Metadata
        let q_bytes = bincode::encode_to_vec(&*quantizer_arc, bincode::config::standard())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        // 2. Init Writer
        let manager = DiskManager::open(&file_path).await?;
        let mut writer = SegmentWriter::new(manager, q_bytes);

        // 3. Iterate Buckets (L1 Data)
        let buckets = index.get_all_buckets();

        for bucket in buckets {
            let (vecs, ids) = bucket.extract_reconstructed();

            if !vecs.is_empty() {
                writer.write_bucket(bucket.id, &ids, &vecs).await?;
            }
        }

        writer.finalize().await?;
        println!("Flushed L1 index to {:?}", file_path);
        Ok(file_path)
    }

    /// NEW: FLUSH L0 (MemTable) -> Disk (Segment)
    /// This takes raw (ID, Vector) pairs extracted from MemTable and writes them to a new Segment.
    pub async fn flush_memtable_to_segment(
        &self,
        data: &[(u64, Vec<f32>)],
        index: &VectorIndex,
        run_id: &str,
    ) -> std::io::Result<PathBuf> {
        let file_name = format!("segment_l0_{}.drift", run_id);
        let file_path = self.base_path.join(&file_name);

        // 1. Get Quantizer (Must be trained to write L1 format)
        let quantizer_arc = index.get_quantizer().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Cannot flush L0 without trained Quantizer",
            )
        })?;

        let q_bytes = bincode::encode_to_vec(&*quantizer_arc, bincode::config::standard())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        // 2. Init Writer
        let manager = DiskManager::open(&file_path).await?;
        let mut writer = SegmentWriter::new(manager, q_bytes);

        // 3. Prepare Data
        // We write the entire MemTable snapshot as a single bucket (ID 0 for now).
        // The VectorIndex re-assigns/splits this upon loading/compaction later.
        let bucket_id = 0;

        let ids: Vec<u64> = data.iter().map(|(id, _)| *id).collect();
        let vecs: Vec<Vec<f32>> = data.iter().map(|(_, v)| v.clone()).collect();

        if !vecs.is_empty() {
            writer.write_bucket(bucket_id, &ids, &vecs).await?;
        }

        writer.finalize().await?;
        println!("Flushed L0 MemTable to {:?}", file_path);
        Ok(file_path)
    }

    /// LOAD: Disk (Segment + WAL) -> Memory (Hybrid L0+L1)
    pub async fn load_from_segment(&self, path: &Path) -> std::io::Result<VectorIndex> {
        // 1. Open Segment Reader
        let mut reader = SegmentReader::open(path).await?;

        // 2. Restore Quantizer
        let q_bytes = reader.read_metadata();
        let (quantizer, _): (Quantizer, usize) =
            bincode::decode_from_slice(q_bytes, bincode::config::standard()).map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Corrupt Quantizer Metadata",
                )
            })?;

        // 3. Initialize Index (Restores L0 from WAL)
        let dim = quantizer.min.len();
        let options = IndexOptions {
            dim,
            num_centroids: 0,
            training_sample_size: 0,
            max_bucket_capacity: 1000,
            ..Default::default()
        };

        // Standard WAL location
        let wal_path = self.base_path.join("current.wal");

        let index = VectorIndex::new(options, &wal_path)?;

        // Set Quantizer
        index.set_quantizer(quantizer);

        // 4. Hydrate Buckets (Restores L1 from Segment)
        let bucket_ids: Vec<u32> = reader.index.buckets.keys().cloned().collect();

        for id in bucket_ids {
            let (ids, vectors) = reader.read_bucket(id).await?;
            index.force_register_bucket_with_ids(id, &ids, &vectors);
        }

        Ok(index)
    }

    /// HYDRATION: Scans the directory for .drift files and restores them into the Index.
    pub async fn hydrate_index(&self, index: &VectorIndex) -> std::io::Result<()> {
        let mut read_dir = tokio::fs::read_dir(&self.base_path).await?;
        let mut segments_found = 0;

        // Collect paths to sort them for deterministic loading order
        let mut paths = Vec::new();
        while let Some(entry) = read_dir.next_entry().await? {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("drift") {
                paths.push(path);
            }
        }
        paths.sort(); // Sort by filename (e.g. timestamp or UUID)

        for path in paths {
            println!(
                "Persistence: Hydrating segment {:?}",
                path.file_name().unwrap()
            );

            let mut reader = SegmentReader::open(&path).await?;

            // 1. Restore Quantizer (Cold Start)
            let needs_quantizer = index.get_quantizer().is_none();
            if needs_quantizer {
                let q_bytes = reader.read_metadata();
                // ... (keep existing deserialization logic) ...
                let (quantizer, _): (Quantizer, usize) = bincode::decode_from_slice(
                    q_bytes,
                    bincode::config::standard(),
                )
                .map_err(|_| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, "Corrupt Quantizer")
                })?;

                println!("Persistence: Restored Quantizer from segment.");
                index.set_quantizer(quantizer);
            }

            // 2. Load Buckets with NEW IDs
            // L0 segments (from MemTable flushes) often have ID=0.
            // We MUST allocate a new ID for them in the current index to avoid overwrites.
            let bucket_ids_in_file: Vec<u32> = reader.index.buckets.keys().cloned().collect();

            for _old_id in bucket_ids_in_file {
                // Read the data associated with the old ID
                let (ids, vectors) = reader.read_bucket(_old_id).await?;

                // ALLOCATE FRESH ID
                let new_id = index.allocate_next_bucket_id();

                // Register with the new ID
                index.force_register_bucket_with_ids(new_id, &ids, &vectors);
            }

            segments_found += 1;
        }

        if segments_found > 0 {
            println!(
                "Persistence: Hydration complete. Loaded {} segments.",
                segments_found
            );
        }

        Ok(())
    }
}
