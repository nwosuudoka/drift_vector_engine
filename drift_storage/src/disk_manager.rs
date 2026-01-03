use async_trait::async_trait;
use drift_traits::{PageId, PageManager};
use opendal::Operator;
use std::collections::HashMap;
use std::io;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

// --- DiskManager (Flat File Access) ---
#[derive(Clone)]
pub struct DiskManager {
    op: Operator,
    pub path: String, // Relative path inside the bucket/root
}

impl DiskManager {
    /// Create a manager using an existing Operator.
    /// The 'path' is relative to the Operator's root.
    pub fn new(op: Operator, path: String) -> Self {
        Self { op, path }
    }

    // Removed: pub async fn open(uri: &str) -> ...

    pub async fn read_at(&self, offset: u64, length: usize) -> io::Result<Vec<u8>> {
        let range = offset..offset + length as u64;
        let data = self
            .op
            .read_with(&self.path)
            .range(range)
            .await
            .map_err(io::Error::other)?;

        if data.len() != length {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "Short read"));
        }
        Ok(data.to_vec())
    }

    pub async fn len(&self) -> io::Result<u64> {
        let meta = self.op.stat(&self.path).await.map_err(io::Error::other)?;
        Ok(meta.content_length())
    }

    pub async fn upload(&self, data: Vec<u8>) -> io::Result<()> {
        self.op
            .write(&self.path, data)
            .await
            .map_err(io::Error::other)?;
        Ok(())
    }
}

// --- DriftPageManager (Page-Based Access) ---
#[derive(Clone)]
pub struct DriftPageManager {
    op: Operator,
    // Maps FileID -> Relative Path (e.g., 1 -> "segment_1.drift")
    files: Arc<RwLock<HashMap<u32, String>>>,
}

impl DriftPageManager {
    /// Creates a new PageManager wrapping an OpenDAL Operator.
    /// The Operator should already be configured with the correct Root/Bucket.
    pub fn new(op: Operator) -> Self {
        Self {
            op,
            files: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl PageManager for DriftPageManager {
    fn register_file(&self, file_id: u32, path: PathBuf) {
        // We only care about the filename (relative path) since Operator is rooted.
        let filename = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown.drift")
            .to_string();
        let mut map = self.files.write().unwrap();
        map.insert(file_id, filename);
    }

    async fn read_page(&self, page_id: PageId) -> io::Result<Vec<u8>> {
        let path = {
            let map = self.files.read().unwrap();
            map.get(&page_id.file_id).cloned()
        };

        if let Some(p) = path {
            let range = page_id.offset..page_id.offset + page_id.length as u64;
            let data = self.op.read_with(&p).range(range).await?;
            if data.len() != page_id.length as usize {
                return Err(io::Error::other("Short read"));
            }
            Ok(data.to_vec())
        } else {
            Err(io::Error::other(format!(
                "File ID {} not registered",
                page_id.file_id,
            )))
        }
    }

    async fn write_page(&self, file_id: u32, offset: u64, data: &[u8]) -> io::Result<()> {
        let filename = {
            let mut map = self.files.write().unwrap();
            map.entry(file_id)
                .or_insert_with(|| format!("page_{}", file_id))
                .clone()
        };

        // RMW Strategy (Simulated for Object Storage)
        let exists = self.op.exists(&filename).await?;
        let mut full_data = if exists {
            self.op.read(&filename).await?.to_vec()
        } else {
            Vec::new()
        };

        let end = offset as usize + data.len();
        if full_data.len() < end {
            full_data.resize(end, 0);
        }
        full_data[offset as usize..end].copy_from_slice(data);
        self.op.write(&filename, full_data).await?;
        Ok(())
    }

    /// Used by Compactor to determine liveness.
    /// Returns None if the ID is not registered or purely in-memory.
    fn get_physical_path(&self, file_id: u32) -> Option<String> {
        self.files.read().unwrap().get(&file_id).cloned()
    }
}
