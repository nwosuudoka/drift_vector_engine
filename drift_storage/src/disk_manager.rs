use async_trait::async_trait;
use drift_traits::{PageId, PageManager};
use opendal::Operator;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::io;
use std::io::SeekFrom;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use tokio::io::{AsyncReadExt, AsyncSeekExt};
use tracing::debug;

#[derive(Clone)]
struct ObjectReadCache {
    root: PathBuf,
    namespace: String,
    max_cached_file_bytes: Option<u64>,
}

// --- DiskManager (Flat File Access) ---
#[derive(Clone)]
pub struct DiskManager {
    op: Operator,
    pub path: String, // Relative path inside the bucket/root
    cache: Option<ObjectReadCache>,
}

impl DiskManager {
    /// Create a manager using an existing Operator.
    /// The 'path' is relative to the Operator's root.
    pub fn new(op: Operator, path: String) -> Self {
        tracing::debug!("DiskManager: Attached to {}", path);
        Self {
            cache: Self::build_object_cache(&op),
            op,
            path,
        }
    }

    fn object_namespace(op: &Operator) -> String {
        let info = op.info();
        format!("{}://{}{}", info.scheme(), info.name(), info.root())
    }

    fn nvme_cache_root() -> Option<PathBuf> {
        let root = std::env::var("DRIFT_NVME_CACHE_DIR").ok()?;
        let root = PathBuf::from(root);
        std::fs::create_dir_all(&root).ok()?;
        Some(root)
    }

    fn is_remote_scheme(scheme: &str) -> bool {
        !matches!(scheme, "fs" | "memory")
    }

    fn build_object_cache(op: &Operator) -> Option<ObjectReadCache> {
        let scheme = op.info().scheme();
        if !Self::is_remote_scheme(scheme) {
            return None;
        }

        let root = Self::nvme_cache_root()?;
        let namespace = Self::object_namespace(op);
        let max_cached_file_bytes = std::env::var("DRIFT_NVME_CACHE_MAX_FILE_BYTES")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|v| *v > 0);

        Some(ObjectReadCache {
            root,
            namespace,
            max_cached_file_bytes,
        })
    }

    fn object_cache_dir(cache_root: &Path, namespace: &str, path: &str) -> PathBuf {
        let mut hasher = DefaultHasher::new();
        namespace.hash(&mut hasher);
        path.hash(&mut hasher);
        let hash = hasher.finish();
        cache_root.join(format!("{hash:016x}"))
    }

    fn object_cache_path(cache_root: &Path, namespace: &str, path: &str) -> PathBuf {
        let dir = Self::object_cache_dir(cache_root, namespace, path);
        dir.join("object.cache")
    }

    async fn read_range_from_disk_cache(
        path: &Path,
        offset: u64,
        length: usize,
    ) -> io::Result<Option<Vec<u8>>> {
        let mut file = match tokio::fs::File::open(path).await {
            Ok(f) => f,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e),
        };

        let end = offset
            .checked_add(length as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Read overflow"))?;
        let file_len = file.metadata().await?.len();
        if end > file_len {
            return Ok(None);
        }

        file.seek(SeekFrom::Start(offset)).await?;
        let mut out = vec![0u8; length];
        match file.read_exact(&mut out).await {
            Ok(_) => Ok(Some(out)),
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => Ok(None),
            Err(e) => Err(e),
        }
    }

    async fn write_full_to_disk_cache(path: PathBuf, payload: Vec<u8>) -> io::Result<()> {
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let tmp_path = path.with_extension(format!("tmp-{nonce}"));
        tokio::fs::write(&tmp_path, payload).await?;
        tokio::fs::rename(tmp_path, path).await
    }

    pub async fn invalidate_nvme_cache_for_object(op: &Operator, path: &str) -> io::Result<()> {
        if !Self::is_remote_scheme(op.info().scheme()) {
            return Ok(());
        }
        let Some(root) = Self::nvme_cache_root() else {
            return Ok(());
        };
        let namespace = Self::object_namespace(op);
        let dir = Self::object_cache_dir(&root, &namespace, path);
        match tokio::fs::remove_dir_all(dir).await {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e),
        }
    }

    async fn read_remote_range(&self, offset: u64, length: usize) -> io::Result<Vec<u8>> {
        let end = offset
            .checked_add(length as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Read overflow"))?;
        let range = offset..end;
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

    // Removed: pub async fn open(uri: &str) -> ...

    pub async fn read_at(&self, offset: u64, length: usize) -> io::Result<Vec<u8>> {
        if length == 0 {
            return Ok(Vec::new());
        }

        if let Some(cache) = &self.cache {
            let cache_path = Self::object_cache_path(&cache.root, &cache.namespace, &self.path);

            // 1) Local disk hit (full object cached on NVMe, read only requested range)
            match Self::read_range_from_disk_cache(&cache_path, offset, length).await {
                Ok(Some(bytes)) => return Ok(bytes),
                Ok(None) => {}
                Err(e) => {
                    debug!(
                        "DiskManager: local cache read failed for {}: {}",
                        self.path, e
                    );
                }
            }

            if let Some(max_bytes) = cache.max_cached_file_bytes {
                match self.op.stat(&self.path).await {
                    Ok(meta) if meta.content_length() > max_bytes => {
                        return self.read_remote_range(offset, length).await;
                    }
                    Ok(_) => {}
                    Err(e) => {
                        debug!(
                            "DiskManager: stat failed for {} before full-cache read: {}",
                            self.path, e
                        );
                    }
                }
            }

            // 2) Remote fetch full object, then slice requested range.
            let full = self.op.read(&self.path).await.map_err(io::Error::other)?;
            let payload = full.to_vec();
            let end = offset
                .checked_add(length as u64)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Read overflow"))?;
            if end > payload.len() as u64 {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "Short read"));
            }

            let start_idx = offset as usize;
            let end_idx = end as usize;
            let out = payload[start_idx..end_idx].to_vec();

            // Best effort async populate of local cache file with full object.
            let cache_path_for_disk = cache_path;
            let payload_for_disk = payload;
            tokio::spawn(async move {
                let _ = Self::write_full_to_disk_cache(cache_path_for_disk, payload_for_disk).await;
            });

            return Ok(out);
        }

        self.read_remote_range(offset, length).await
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

    async fn len(&self, file_id: u32) -> std::io::Result<u64> {
        let path = {
            let map = self.files.read().unwrap();
            map.get(&file_id).cloned()
        };

        if let Some(p) = path {
            let meta = self.op.stat(&p).await?;
            Ok(meta.content_length())
        } else {
            Err(io::Error::other(format!(
                "File ID {} not registered",
                file_id
            )))
        }
    }
}
