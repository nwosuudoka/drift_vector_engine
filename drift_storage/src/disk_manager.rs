use async_trait::async_trait;
use drift_traits::{PageId, PageManager};
use opendal::{Operator, services};
use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use url::Url;

// --- Existing DiskManager (Single File Handle) ---
#[derive(Clone)]
pub struct DiskManager {
    op: Operator,
    pub path: String,
}

impl DiskManager {
    pub async fn open(uri: &str) -> io::Result<Self> {
        let (op, relative_path) = Self::parse_uri(uri)?;
        Ok(Self {
            op,
            path: relative_path,
        })
    }

    // Helper to parse URI into Operator + Path
    fn parse_uri(uri: &str) -> io::Result<(Operator, String)> {
        let url = Url::parse(uri).map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
        match url.scheme() {
            "file" => {
                let builder = services::Fs::default();
                let path = Path::new(url.path());
                let parent = path.parent().unwrap_or(Path::new("/")).to_str().unwrap();
                let filename = path.file_name().unwrap_or_default().to_str().unwrap();
                let op = Operator::new(builder.root(parent))
                    .map_err(io::Error::other)?
                    .finish();
                Ok((op, filename.to_string()))
            }
            "s3" => {
                let builder = services::S3::default();
                let bucket = url.host_str().ok_or(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "Missing bucket",
                ))?;
                let op = Operator::new(builder.bucket(bucket).region("us-east-1"))
                    .map_err(io::Error::other)?
                    .finish();
                let p = url.path().trim_start_matches('/');
                Ok((op, p.to_string()))
            }
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Unsupported scheme",
            )),
        }
    }

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

#[derive(Clone)]
pub struct DriftPageManager {
    op: Operator,
    // Maps FileID -> Relative Path (e.g., 1 -> "segment_1.drift")
    files: Arc<RwLock<HashMap<u32, String>>>,
}

impl DriftPageManager {
    /// Opens a manager rooted at a specific location (directory/bucket prefix).
    /// URI Example: file:///data/collection_name/storage/ or s3://bucket/collection/storage/
    pub async fn new(uri: &str) -> io::Result<Self> {
        // We use the same parsing logic, but we treat the 'path' as the root.
        // For 'file://.../storage', we want the Operator root to be '.../storage'.

        let url = Url::parse(uri).map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
        let op = match url.scheme() {
            "file" => {
                let builder = services::Fs::default();
                // Root is the directory itself
                Operator::new(builder.root(url.path()))
                    .map_err(io::Error::other)?
                    .finish()
            }
            "s3" => {
                let bucket = url.host_str().ok_or(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "Missing bucket",
                ))?;

                let builder = {
                    let builder = services::S3::default().bucket(bucket).region("us-east-1");
                    let root = url.path().trim_start_matches('/');
                    match root.is_empty() {
                        true => builder,
                        false => builder.root(root),
                    }
                };

                // For S3, we might need a root prefix if the URI has a path
                Operator::new(builder).map_err(io::Error::other)?.finish()
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "Unsupported scheme",
                ));
            }
        };

        Ok(Self {
            op,
            files: Arc::new(RwLock::new(HashMap::new())),
        })
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
                .or_insert_with(|| format!("page_{}", file_id)) // e.g. "page_0"
                .clone()
        };

        // RMW Strategy
        // 1. Check/Read existing
        let exists = self.op.exists(&filename).await?;
        let mut full_data = if exists {
            self.op.read(&filename).await?.to_vec()
        } else {
            Vec::new()
        };

        // 2. Patch
        let end = offset as usize + data.len();
        if full_data.len() < end {
            full_data.resize(end, 0);
        }
        full_data[offset as usize..end].copy_from_slice(data);

        // 3. Write Full
        self.op.write(&filename, full_data).await?;
        Ok(())
    }
}
