use async_trait::async_trait;
use drift_traits::{PageId, PageManager};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::Result;
use std::os::unix::fs::FileExt; // Linux/Mac specific optimization
use std::path::PathBuf;
use std::sync::Arc;
use tokio::task;

/// A Disk Manager optimized for NVMe Random Reads.
/// Uses `pread` to allow true parallel reads on shared file handles.
pub struct LocalDiskManager {
    base_path: PathBuf,
    // Map ID -> Path
    paths: RwLock<HashMap<u32, PathBuf>>,
    // Map ID -> Open File Handle
    // We use std::fs::File because it is Sync (on Unix) and allows parallel read_at.
    files: RwLock<HashMap<u32, Arc<File>>>,
}

impl LocalDiskManager {
    pub fn new(base_path: impl Into<PathBuf>) -> Self {
        let path = base_path.into();
        std::fs::create_dir_all(&path).ok(); // Ensure dir exists
        Self {
            base_path: path,
            paths: RwLock::new(HashMap::new()),
            files: RwLock::new(HashMap::new()),
        }
    }

    /// Helper to get or open a file handle. Auto-registers if missing.
    fn get_file(&self, file_id: u32) -> Result<Arc<File>> {
        // 1. Check if already open (Fast Read Lock)
        {
            let handles = self.files.read();
            if let Some(f) = handles.get(&file_id) {
                return Ok(f.clone());
            }
        }

        // 2. Check if path registered, if not, AUTO-REGISTER (Write Lock)
        let mut paths_guard = self.paths.write();
        let mut files_guard = self.files.write();

        // Double check after lock
        if let Some(f) = files_guard.get(&file_id) {
            return Ok(f.clone());
        }

        // Resolve Path
        let path = paths_guard
            .entry(file_id)
            .or_insert_with(|| self.base_path.join(format!("{}.drift", file_id)))
            .clone();

        // Open File
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&path)?;

        let arc_file = Arc::new(file);
        files_guard.insert(file_id, arc_file.clone());

        Ok(arc_file)
    }
}

#[async_trait]
impl PageManager for LocalDiskManager {
    fn register_file(&self, file_id: u32, path: PathBuf) {
        let mut map = self.paths.write();
        map.insert(file_id, path);
    }

    async fn read_page(&self, page_id: PageId) -> Result<Vec<u8>> {
        let file = self.get_file(page_id.file_id)?;
        let offset = page_id.offset;
        let len = page_id.length as usize;

        // Offload blocking I/O to a specialized thread
        task::spawn_blocking(move || {
            let mut buf = vec![0u8; len];
            // pread: Thread-safe, offset-based read
            file.read_exact_at(&mut buf, offset)?;
            Ok(buf)
        })
        .await? // Handle JoinError
    }

    async fn write_page(&self, file_id: u32, offset: u64, data: &[u8]) -> Result<()> {
        let file = self.get_file(file_id)?;
        // Clone data to move into thread (overhead is minimal for typical page sizes)
        // For massive writes, we might want to pass ownership of a Vec.
        let data = data.to_vec();

        task::spawn_blocking(move || {
            file.write_all_at(&data, offset)?;
            // Optional: file.sync_data()?; // For strict durability
            Ok(())
        })
        .await?
    }
}
