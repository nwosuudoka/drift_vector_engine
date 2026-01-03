use async_trait::async_trait;
use drift_traits::{PageId, PageManager};
use std::io::Result;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::debug;

pub struct TieredPageManager {
    local: Arc<dyn PageManager>,
    remote: Arc<dyn PageManager>,
}

impl TieredPageManager {
    pub fn new(local: Arc<dyn PageManager>, remote: Arc<dyn PageManager>) -> Self {
        Self { local, remote }
    }
}

#[async_trait]
impl PageManager for TieredPageManager {
    fn register_file(&self, file_id: u32, path: PathBuf) {
        // Only Remote needs to know the Segment mapping.
        // Local assumes ID -> Filename mapping.
        self.remote.register_file(file_id, path);
    }

    async fn read_page(&self, page_id: PageId) -> Result<Vec<u8>> {
        // 1. FAST PATH: Check Local Cache (Bucket File)
        // We assume the local file contains JUST this bucket, starting at 0.
        // This decouples the local cache structure from the remote segment structure.
        let local_read = self
            .local
            .read_page(PageId {
                file_id: page_id.file_id,
                offset: 0, // ⚡ Always 0 for local cache files
                length: page_id.length,
            })
            .await;

        if let Ok(data) = local_read {
            return Ok(data);
        }

        // 2. CACHE MISS: Read from Remote Segment
        // This uses the REAL offset inside the S3 segment file.
        let data = self.remote.read_page(page_id.clone()).await?;

        // 3. POPULATE: Write to Local Cache
        // We write the data to a new local file (ID.bin) at offset 0.
        let local = self.local.clone();
        let fid = page_id.file_id;
        let cache_data = data.clone();

        tokio::spawn(async move {
            if let Err(e) = local.write_page(fid, 0, &cache_data).await {
                debug!("Failed to populate tiered cache for {}: {}", fid, e);
            }
        });

        Ok(data)
    }

    async fn write_page(&self, file_id: u32, offset: u64, data: &[u8]) -> Result<()> {
        // Writes go to Local (Cache Warming / MemTable Flush simulation)
        self.local.write_page(file_id, offset, data).await
    }

    // Forward physical path lookup to remote storage (S3/Disk)
    // This allows the Compactor to see what files are actually live.
    fn get_physical_path(&self, file_id: u32) -> Option<String> {
        self.remote.get_physical_path(file_id)
    }

    async fn read_high_fidelity(&self, file_id: u32) -> Result<Vec<Vec<f32>>> {
        self.remote.read_high_fidelity(file_id).await
    }
}
