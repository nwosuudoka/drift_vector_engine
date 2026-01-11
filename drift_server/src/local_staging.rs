use drift_core::partitioner::PartitionGroup;
use drift_core::quantizer::Quantizer;
use drift_storage::bucket_file_reader::BucketFileReader;
use drift_storage::bucket_file_writer::BucketFileWriter;
use drift_traits::IoContext;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::Mutex;

/// Manages local .drift files on NVMe/Disk.
/// Handles locking and append operations.
pub struct LocalStagingManager {
    base_path: PathBuf,
    // Locks per BucketID to ensure serial appends
    locks: RwLock<HashMap<u32, Arc<Mutex<()>>>>,
}

impl LocalStagingManager {
    pub fn new(base_path: impl AsRef<Path>) -> io::Result<Self> {
        let p = base_path.as_ref().to_path_buf();
        fs::create_dir_all(&p).context("Failed to create staging dir")?;
        Ok(Self {
            base_path: p,
            locks: RwLock::new(HashMap::new()),
        })
    }

    /// Gets or creates a lock for a specific bucket.
    fn get_lock(&self, bucket_id: u32) -> Arc<Mutex<()>> {
        // Optimistic read
        if let Some(lock) = self.locks.read().get(&bucket_id) {
            return lock.clone();
        }

        // Write path
        let mut map = self.locks.write();
        map.entry(bucket_id)
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    }

    /// Appends a batch of vectors to the local bucket file.
    /// If the file doesn't exist, it creates it and trains a new Quantizer.
    pub async fn append_batch(&self, bucket_id: u32, batch: &PartitionGroup) -> io::Result<()> {
        let dim = if !batch.flat_vectors.is_empty() {
            batch.flat_vectors.len() / batch.ids.len()
        } else {
            return Ok(()); // Nothing to write
        };

        // 1. Acquire Bucket Lock
        let lock = self.get_lock(bucket_id);
        let _guard = lock.lock().await;

        let file_path = self.base_path.join(format!("bucket_{}.drift", bucket_id));
        let file_exists = file_path.exists();

        // 2. Open File
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&file_path)
            .context("Failed to open bucket file")?;

        let file_len = file.metadata()?.len();

        // 3. Initialize Writer
        let mut writer = if file_exists && file_len > 0 {
            let q = BucketFileReader::get_quantizer(&file_path)?;
            BucketFileWriter::new_append(
                file, [0u8; 16], // RunID (TODO: Manage RunIDs properly)
                q, dim, file_len,
            )?
        } else {
            // --- NEW FILE MODE ---
            // Train Quantizer on this batch
            // Note: If batch is tiny, this is suboptimal, but acceptable for MVP.
            let q = Quantizer::train(&batch.flat_vectors, dim);
            BucketFileWriter::new_streaming(file, [0u8; 16], q, dim)?
        };

        // 4. Write Data
        writer.write_batch(&batch.ids, &batch.flat_vectors)?;

        // 5. Finalize (Flush + Truncate)
        // If we are appending, this safely updates the footer.
        if file_exists {
            writer.finalize_and_truncate()?;
        } else {
            writer.finalize()?;
        }

        Ok(())
    }

    // /// Helper to extract the immutable Quantizer from a file's header.
    // fn recover_quantizer(&self, path: &Path) -> io::Result<Quantizer> {
    //     let mut file = std::fs::File::open(path)?;
    //     let mut head_buf = [0u8; HEADER_SIZE];
    //     file.read_exact(&mut head_buf)?;

    //     // Offset 48 is quantizer_offset (u64), 56 is length (u32) based on our layout.
    //     // Let's rely on `DriftHeader` being Pod/ZeroCopy.
    //     // let header = DriftHeader::ref_from_bytes(&head_buf).map_err(io::Error::other)?;
    //     let header = DriftHeader::force_copy(&head_buf);

    //     // Validate
    //     if header.magic != MAGIC_V2 {
    //         tracing::error!("Invalid magic bytes {} != {}", header.magic, MAGIC_V2);
    //         panic!("Invalid magic bytes")
    //     }

    //     // Read Quantizer Blob
    //     file.seek(SeekFrom::Start(header.quantizer_offset))?;
    //     let mut q_buf = vec![0u8; header.quantizer_length as usize];
    //     file.read_exact(&mut q_buf)?;

    //     let (q, _): (Quantizer, usize) =
    //         bincode::decode_from_slice(&q_buf, bincode::config::standard())
    //             .map_err(io::Error::other)
    //             .context("Failed to decode quantizer")?;

    //     Ok(q)
    // }
}
