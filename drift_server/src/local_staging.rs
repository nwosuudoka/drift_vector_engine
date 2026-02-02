use drift_core::partitioner::PartitionGroup;
use drift_core::quantizer::Quantizer;
use drift_storage::bucket_file_reader::BucketFileReader;
use drift_storage::bucket_file_writer::BucketFileWriter;
use drift_traits::IoContext;
use opendal::{Operator, services};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};

pub struct LocalStagingManager {
    base_path: PathBuf,
    // ⚡ Active Generation Map: BucketID -> Filename
    active_files: RwLock<HashMap<u32, String>>,
}

impl LocalStagingManager {
    pub fn new(base_path: impl AsRef<Path>) -> io::Result<Self> {
        let p = base_path.as_ref().to_path_buf();
        fs::create_dir_all(&p).context("Failed to create staging dir")?;
        Ok(Self {
            base_path: p,
            active_files: RwLock::new(HashMap::new()),
        })
    }

    pub fn get_base_path(&self) -> &Path {
        &self.base_path
    }

    /// Returns the currently active filename for this bucket.
    pub fn get_active_filename(&self, bucket_id: u32) -> String {
        self.active_files
            .read()
            .get(&bucket_id)
            .cloned()
            .unwrap_or_else(|| format!("bucket_{}.drift", bucket_id))
    }

    /// ⚡ ATOMIC ROTATION (Caller holds Write Lock via Coordinator)
    pub async fn rotate_bucket_for_promotion(
        &self,
        bucket_id: u32,
        staging_name: &str,
        new_active_name: &str,
    ) -> io::Result<bool> {
        // No internal lock needed - Janitor holds the BucketCoordinator lock.

        let current_filename = self.get_active_filename(bucket_id);
        let src = self.base_path.join(&current_filename);
        let dst = self.base_path.join(staging_name);

        if !src.exists() {
            return Ok(false);
        }

        // 1. Rotate
        fs::rename(&src, &dst).context("Failed to rotate bucket file")?;

        // 2. Create New Active File (Empty)
        let new_path = self.base_path.join(new_active_name);
        OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&new_path)
            .context("Failed to create new active file")?;

        // 3. Update Pointer
        self.active_files
            .write()
            .insert(bucket_id, new_active_name.to_string());

        Ok(true)
    }

    /// Sets the active filename manually (used in tests or recovery).
    pub fn set_active_filename(&self, bucket_id: u32, filename: String) {
        self.active_files.write().insert(bucket_id, filename);
    }

    /// Creates an empty file to initialize a new generation.
    pub async fn create_empty_file(&self, filename: &str) -> io::Result<()> {
        let path = self.base_path.join(filename);
        OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;
        Ok(())
    }

    pub async fn append_batch(&self, bucket_id: u32, batch: &PartitionGroup) -> io::Result<u64> {
        if batch.count == 0 {
            return Ok(0);
        }
        let dim = batch.flat_vectors.len() / batch.ids.len();

        // No internal lock needed.

        // Resolve Dynamic Filename
        let filename = self.get_active_filename(bucket_id);
        let path = self.base_path.join(&filename);
        let exists = path.exists();

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .context("Failed to open bucket file")?;

        let initial_len = file.metadata()?.len();

        let mut writer = if exists && initial_len > 0 {
            let op = self.create_local_op()?;
            let mut reader = BucketFileReader::open(op, &filename).await?;
            reader.load_quantizer().await?;
            let q = reader
                .quantizer
                .ok_or_else(|| io::Error::other("Quantizer missing"))?;
            BucketFileWriter::new_append(file, [0u8; 16], q, dim, initial_len)?
        } else {
            let q = Quantizer::train(&batch.flat_vectors, dim);
            BucketFileWriter::new_streaming(file, [0u8; 16], q, dim)?
        };

        writer.write_batch(&batch.ids, &batch.flat_vectors)?;
        let (_, total_count) = writer.finalize_and_truncate()?;

        Ok(total_count)
    }

    pub async fn read_file_content(&self, filename: &str) -> io::Result<(Vec<u64>, Vec<Vec<f32>>)> {
        let path = self.base_path.join(filename);
        if !path.exists() {
            return Ok((vec![], vec![]));
        }
        let op = self.create_local_op()?;
        let mut reader = BucketFileReader::open(op, filename).await?;
        reader.load_quantizer().await?;
        reader.read_all_vectors().await
    }

    /// Reads the current active file for the bucket.
    /// ⚡ This method is lock-free (unsafe if concurrent writes happen, but fine for tests).
    pub async fn read_full_bucket(&self, bucket_id: u32) -> io::Result<(Vec<u64>, Vec<Vec<f32>>)> {
        let filename = self.get_active_filename(bucket_id);
        self.read_file_content(&filename).await
    }

    pub async fn delete_file(&self, filename: &str) -> io::Result<()> {
        let path = self.base_path.join(filename);
        if path.exists() {
            fs::remove_file(path)?;
        }
        Ok(())
    }

    // Deletes the active file (used during total bucket removal)
    pub async fn delete_bucket(&self, bucket_id: u32) -> io::Result<()> {
        // No lock.
        let filename = self.get_active_filename(bucket_id);
        self.delete_file(&filename).await
    }

    pub fn list_large_buckets(&self, threshold: u64) -> io::Result<Vec<u32>> {
        let mut res = Vec::new();
        let map = self.active_files.read();

        if map.is_empty() {
            for entry in fs::read_dir(&self.base_path)? {
                let entry = entry?;
                let path = entry.path();
                if let Some(name) = path.file_name().and_then(|s| s.to_str())
                    && name.starts_with("bucket_")
                    && name.ends_with(".drift")
                {
                    let id_part = name
                        .trim_start_matches("bucket_")
                        .trim_end_matches(".drift");
                    if let Ok(id) = id_part.parse::<u32>()
                        && entry.metadata()?.len() >= threshold
                    {
                        res.push(id);
                    }
                }
            }
        } else {
            for (&id, filename) in map.iter() {
                let path = self.base_path.join(filename);
                if let Ok(meta) = fs::metadata(path)
                    && meta.len() >= threshold
                {
                    res.push(id);
                }
            }
        }
        Ok(res)
    }

    fn create_local_op(&self) -> io::Result<Operator> {
        let root = self.base_path.to_str().unwrap();
        let builder = services::Fs::default().root(root);
        Ok(Operator::new(builder)?.finish())
    }

    /// NEW: Writes a fresh file for a bucket (CoW support).
    /// Used during Scatter-Merge to rewrite a neighbor bucket with new data.
    pub async fn write_new_file(&self, filename: &str, group: &PartitionGroup) -> io::Result<u64> {
        let path = self.base_path.join(filename);
        let dim = group.flat_vectors.len() / group.ids.len();

        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .context("Failed to create new staging file")?;

        let q = Quantizer::train(&group.flat_vectors, dim);
        let mut writer = BucketFileWriter::new_streaming(file, [0u8; 16], q, dim)?;

        writer.write_batch(&group.ids, &group.flat_vectors)?;
        let (_, total_count) = writer.finalize()?;

        Ok(total_count)
    }
}
