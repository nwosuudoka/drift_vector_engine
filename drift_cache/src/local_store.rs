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
    // Map ID -> Open File Handle
    // files: RwLock<HashMap<u32, Arc<File>>>,
}

impl LocalDiskManager {
    pub fn new(base_path: impl Into<PathBuf>) -> Self {
        let path = base_path.into();
        std::fs::create_dir_all(&path).ok(); // Ensure dir exists
        Self { base_path: path }
    }

    /// Helper to open a file.
    /// Does NOT cache the handle.
    fn open_file(&self, file_id: u32, for_write: bool) -> Result<File> {
        // 1. Resolve Path
        let path = self.base_path.join(format!("{}.bin", file_id));

        // 2. Configure Options
        let mut options = OpenOptions::new();
        options.read(true);

        if for_write {
            options.write(true).create(true).truncate(false);
        } else {
            // Read-only: Fail if not found
            options.write(false).create(false);
        }

        // 3. Open
        options.open(&path)
    }
}

#[async_trait]
impl PageManager for LocalDiskManager {
    fn register_file(&self, _file_id: u32, _path: PathBuf) {
        // No-op for LocalDiskManager in Decoupled Mode.
        // It always infers path from ID.
    }

    async fn read_page(&self, page_id: PageId) -> Result<Vec<u8>> {
        let file = self.open_file(page_id.file_id, false)?;

        let offset = page_id.offset;
        let len = page_id.length as usize;

        // Offload blocking IO
        task::spawn_blocking(move || {
            let mut buf = vec![0u8; len];
            // Read exact bytes
            file.read_exact_at(&mut buf, offset)?;
            Ok(buf)
        })
        .await?
    }

    async fn write_page(&self, file_id: u32, offset: u64, data: &[u8]) -> Result<()> {
        let file = self.open_file(file_id, true)?;
        let data = data.to_vec();

        task::spawn_blocking(move || {
            file.write_all_at(&data, offset)?;
            Ok(())
        })
        .await?
    }
}
