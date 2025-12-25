use crate::sharded_fifo::ShardedFastS3Fifo;
use crate::store::{Cacheable, PageId, PageManager};
use std::io::Result;
use std::sync::Arc;

pub struct BlockCache<V: Cacheable> {
    storage: Arc<dyn PageManager>,
    ram: Arc<ShardedFastS3Fifo<PageId, V>>,
}

impl<V: Cacheable> BlockCache<V> {
    pub fn new(storage: Arc<dyn PageManager>, capacity_items: usize, concurrency: usize) -> Self {
        Self {
            storage,
            ram: Arc::new(ShardedFastS3Fifo::new(capacity_items, concurrency)),
        }
    }

    /// The Magic Method: Transparently loads data from disk if missing from RAM.
    pub async fn get(&self, page_id: &PageId) -> Result<Arc<V>> {
        // 1. Fast Path: Check RAM (Zero-Copy)
        if let Some(val) = self.ram.get(page_id) {
            return Ok(val);
        }

        // 2. Slow Path: Disk I/O
        // Note: In a high-concurrency scenario, multiple threads might race here
        // and fetch the same page twice. This is acceptable for V1 (OS page cache handles it).
        let raw_bytes = self.storage.read_page(page_id.clone()).await?;

        // 3. Deserialize (CPU bound)
        let obj = V::from_bytes(&raw_bytes)?;
        // let arc_obj = Arc::new(obj);

        // 4. Cache It (S3FIFO decides admission/eviction)
        self.ram.put(page_id.clone(), obj);

        // 5: return the value.
        // Ok(self.ram.get(page_id).expect("Just inserted"))
        let cached_arc = self
            .ram
            .get(page_id)
            .ok_or_else(|| std::io::Error::other("Cache rejection or race condition"))?;

        Ok(cached_arc)
    }

    // Optimized implementation to avoid the double lookup above:
    pub async fn get_optimized(&self, page_id: &PageId) -> Result<Arc<V>> {
        if let Some(val) = self.ram.get(page_id) {
            return Ok(val);
        }

        let raw_bytes = self.storage.read_page(page_id.clone()).await?;
        let obj = V::from_bytes(&raw_bytes)?;

        // We put it in. S3FIFO wraps it in Arc internally.
        self.ram.put(page_id.clone(), obj);

        // We retrieve it immediately.
        // (Optimally S3FIFO put would return the Arc it created, but get() is fast enough)
        Ok(self.ram.get(page_id).expect("Just inserted"))
    }
}
