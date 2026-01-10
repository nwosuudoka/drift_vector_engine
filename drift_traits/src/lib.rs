use async_trait::async_trait;
use std::fmt::Debug;
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
    /// Maps a logical ID (BucketID) to a physical path (e.g., "segment_uuid.drift")
    fn register_file(&self, file_id: u32, path: PathBuf);

    /// Fetches a raw blob from storage.
    /// This is the "Cache Miss" path.
    async fn read_page(&self, page_id: PageId) -> Result<Vec<u8>>;

    /// Writes a blob to storage.
    /// Used during Flush/Compaction.
    async fn write_page(&self, file_id: u32, offset: u64, data: &[u8]) -> Result<()>;

    /// Used by Compactor to determine liveness.
    /// Returns None if the ID is not registered or purely in-memory.
    fn get_physical_path(&self, _file_id: u32) -> Option<String> {
        None
    }

    /// 💿 NEW: Fetches high-fidelity (ALP) vectors for a logical bucket ID.
    /// This bypasses the block cache to provide raw floats for re-ranking.
    async fn read_high_fidelity(&self, _file_id: u32) -> Result<Vec<Vec<f32>>> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "High-fidelity reads not supported by this manager",
        ))
    }

    async fn len(&self, file_id: u32) -> Result<u64>;
}

// Trait for object that can be cached (e.g Buckets).
pub trait Cacheable: Send + Sync + 'static {
    /// Decode raw bytes from disk in to the object.
    fn from_bytes(data: &[u8]) -> Result<Self>
    where
        Self: Sized;
    // We strictly use Arc for cache entries to allow cheap cloning
}

/// Represents a component capable of searching stored vectors.
/// Implemented by `BucketManager` in the storage layer.
#[async_trait]
pub trait DiskSearcher: Send + Sync {
    /// Searches specific buckets on disk.
    ///
    /// # Arguments
    /// * `bucket_ids`: The list of buckets to query (determined by Router).
    /// * `query`: The vector to search for.
    /// * `k`: Number of results.
    ///
    /// Returns a flat list of (id, distance).
    async fn search(&self, bucket_ids: &[u32], query: &[f32], k: usize) -> Vec<(u64, f32)>;
}
