use async_trait::async_trait;
use std::io::Result;
use std::path::PathBuf;

/// A lightweight handle to a chunk of data.
/// This struct is the "Key" in our S3FIFO BlockCache.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct PageId {
    pub file_id: u32, // Mapped internally to a file path
    pub offset: u64,  // Byte offset in the file
    pub length: u32,  // Length of the blob
}

#[async_trait]
pub trait PageManager: Send + Sync {
    /// Maps a logical ID (e.g., 1) to a physical path (e.g., "/data/segment_123.drift")
    fn register_file(&self, file_id: u32, path: PathBuf);

    /// Fetches a raw blob from storage.
    /// This is the "Cache Miss" path.
    async fn read_page(&self, page_id: PageId) -> Result<Vec<u8>>;

    /// Writes a blob to storage.
    /// Used during Flush/Compaction.
    async fn write_page(&self, file_id: u32, offset: u64, data: &[u8]) -> Result<()>;

    /// Used by Compactor to determine liveness.
    /// Returns None if the ID is not registered or purely in-memory.
    fn get_physical_path(&self, file_id: u32) -> Option<String> {
        None
    }
}

// Trait for object that can be cached (e.g Buckets).
pub trait Cacheable: Send + Sync + 'static {
    /// Decode raw bytes from disk in to the object.
    fn from_bytes(data: &[u8]) -> Result<Self>
    where
        Self: Sized;
    // We strictly use Arc for cache entries to allow cheap cloning
}
