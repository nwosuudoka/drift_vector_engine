use async_trait::async_trait;
use drift_traits::{PageId, PageManager};
use std::io::{Error, ErrorKind, Result};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::task;
use tracing::{debug, warn};

// 4MB Cache Line for S3 (Standard Part Size)
pub(crate) const REMOTE_CHUNK_SIZE: u64 = 4 * 1024 * 1024;

pub struct TieredPageManager {
    local: Arc<dyn PageManager>,
    remote: Arc<dyn PageManager>,
}

impl TieredPageManager {
    pub fn new(local: Arc<dyn PageManager>, remote: Arc<dyn PageManager>) -> Self {
        Self { local, remote }
    }

    fn get_chunk_start(offset: u64) -> u64 {
        (offset / REMOTE_CHUNK_SIZE) * REMOTE_CHUNK_SIZE
    }
}

#[async_trait]
impl PageManager for TieredPageManager {
    fn register_file(&self, file_id: u32, path: PathBuf) {
        self.local.register_file(file_id, path.clone());
        self.remote.register_file(file_id, path);
    }

    async fn read_page(&self, page_id: PageId) -> Result<Vec<u8>> {
        // 1. FAST PATH: Check Local NVMe
        if let Ok(data) = self.local.read_page(page_id.clone()).await {
            return Ok(data);
        }

        // 2. CACHE MISS: Calculate 4MB Chunk
        let chunk_start = Self::get_chunk_start(page_id.offset);

        // Safety: If a request straddles a chunk boundary, fallback to direct read.
        if page_id.offset + page_id.length as u64 > chunk_start + REMOTE_CHUNK_SIZE {
            warn!("Request spans chunk boundary. Bypassing cache population.");
            return self.remote.read_page(page_id).await;
        }

        let chunk_id = PageId {
            file_id: page_id.file_id,
            offset: chunk_start,
            length: REMOTE_CHUNK_SIZE as u32,
        };

        // 3. Fetch from Remote
        let fetch_result = self.remote.read_page(chunk_id.clone()).await;

        let (data_to_cache, offset_to_cache, is_full_chunk) = match fetch_result {
            Ok(data) => (data, chunk_start, true),
            Err(_) => {
                // If 4MB read fails (e.g. strict reader on small file), fallback to exact read.
                let exact_data = self.remote.read_page(page_id.clone()).await?;

                // ⚡ OPPORTUNISTIC CACHE: Cache exactly what we found.
                // This ensures small files are still cached, even if we couldn't prefetch neighbors.
                (exact_data, page_id.offset, false)
            }
        };

        // 4. Populate Local Cache (Async Write-Back)
        let local = self.local.clone();
        let fid = page_id.file_id;
        let cache_payload = data_to_cache.clone();

        task::spawn(async move {
            if let Err(e) = local.write_page(fid, offset_to_cache, &cache_payload).await {
                debug!("Failed to populate tiered cache: {}", e);
            }
        });

        // 5. Return Data
        if is_full_chunk {
            // Slice from the big chunk
            let relative_offset = (page_id.offset - chunk_start) as usize;
            let req_len = page_id.length as usize;
            if relative_offset + req_len <= data_to_cache.len() {
                Ok(data_to_cache[relative_offset..relative_offset + req_len].to_vec())
            } else {
                Err(Error::new(ErrorKind::UnexpectedEof, "Chunk truncated"))
            }
        } else {
            // We performed an exact read in the fallback, so return as-is
            Ok(data_to_cache)
        }
    }

    async fn write_page(&self, file_id: u32, offset: u64, data: &[u8]) -> Result<()> {
        self.local.write_page(file_id, offset, data).await
    }
}
