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

    pub fn storage(&self) -> &Arc<dyn PageManager> {
        &self.storage
    }

    /// Transparently loads data from disk if missing from RAM.
    pub async fn get(&self, page_id: &PageId) -> Result<Arc<V>> {
        // 1. Fast Path: Check RAM
        if let Some(val) = self.ram.get(page_id) {
            return Ok(val);
        }

        // 2. Slow Path: Disk I/O
        let raw_bytes = self.storage.read_page(page_id.clone()).await?;
        let obj = V::from_bytes(&raw_bytes)?;

        // FIX: Wrap in Arc immediately to allow sharing regardless of caching status
        let arc_obj = Arc::new(obj);

        // 3. Try to Cache (S3FIFO may reject it)
        // We pass a clone of the Arc, so if rejected, we still hold the data.
        self.ram.put(page_id.clone(), arc_obj.clone());

        // 4. Return the data (Guaranteed success)
        Ok(arc_obj)
    }

    // Optimized implementation (No change needed since get() logic is now fixed)
    pub async fn get_optimized(&self, page_id: &PageId) -> Result<Arc<V>> {
        self.get(page_id).await
    }
}
