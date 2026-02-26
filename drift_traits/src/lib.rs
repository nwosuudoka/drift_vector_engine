pub mod mock;

use async_trait::async_trait;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::io::Result;
use std::path::PathBuf;
use std::sync::Arc;

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

#[derive(Debug, Clone, PartialEq)]
pub struct SearchCandidate {
    pub id: u64,
    pub approx_dist: f32,
    // The "Cookie" for retrieving cold data
    pub file_id: u32,
    pub cold_offset: u64,  // Offset of the Cold RowGroup blob
    pub cold_length: u32,  // Length of the Cold RowGroup blob
    pub index_in_rg: u32,  // Index of this vector inside the RG
    pub vector_count: u32, // Vector count for the RG
}

// ⚡ NEW: Shared Stats Struct
#[derive(Debug, Clone, Copy)]
pub struct BucketStats {
    pub tombstone_count: u32,
    pub total_count: u32,
    pub temperature: f32, // 0.0 (Cold) to 1.0 (hot)
    pub active: bool,
}

impl Default for BucketStats {
    fn default() -> Self {
        Self {
            tombstone_count: 0,
            total_count: 0,
            temperature: 0.5, // Start warm
            active: true,
        }
    }
}

impl BucketStats {
    pub fn cool(&mut self, decay_rate: f32) {
        self.temperature *= decay_rate;
    }
}

#[async_trait]
pub trait DiskSearcher: Send + Sync {
    /// Performs a complete search: Scan Index (Hot) -> Fetch Data (Cold) -> Refine.
    /// Guarantees consistency by holding locks across both phases.
    async fn search_and_refine(
        &self,
        bucket_ids: &[u32],
        query: &[f32],
        k: usize,                 // Top-K to return
        oversample_factor: usize, // How many candidates to scan (e.g. k * 3)
        tombstones: Arc<dyn TombstoneView>,
    ) -> Vec<(u64, f32)>; // Returns (ID, Exact Distance)
}

/// Represents a component capable of searching stored vectors.
/// Implemented by `BucketManager` in the storage layer.
#[async_trait]
pub trait StorageEngine: Send + Sync {
    /// Performs a complete search: Scan Index (Hot) -> Fetch Data (Cold) -> Refine.
    /// Guarantees consistency by holding locks across both phases.
    async fn search_and_refine(
        &self,
        bucket_ids: &[u32],
        query: &[f32],
        k: usize,                 // Top-K to return
        oversample_factor: usize, // How many candidates to scan (e.g. k * 3)
    ) -> Vec<(u64, f32)>; // Returns (ID, Exact Distance)

    /// Optional ID-level pushdown hint for disk search.
    /// Keys are bucket_ids; values are allowlisted vector IDs within those buckets.
    async fn search_and_refine_with_candidates(
        &self,
        bucket_ids: &[u32],
        query: &[f32],
        k: usize,
        oversample_factor: usize,
        candidate_ids: Option<&HashMap<u32, HashSet<u64>>>,
    ) -> Vec<(u64, f32)> {
        let _ = candidate_ids;
        self.search_and_refine(bucket_ids, query, k, oversample_factor)
            .await
    }

    fn mark_delete(&self, bucket_id: u32, vector_id: u64) -> Result<()>;
    fn get_bucket_stats(&self, bucket_id: u32) -> Option<BucketStats>;

    /// Called by Janitor after flushing new vectors to disk.
    fn update_bucket_drift(
        &self,
        bucket_id: u32,
        delta_sum: &[f32],
        delta_count: u32,
    ) -> Result<()>;
    /// Returns None if bucket not found.
    fn get_bucket_drift_stats(&self, bucket_id: u32) -> Option<(Vec<f32>, u32)>;

    // Called by Janitor to simulate cooling over time
    fn tick_cooling(&self, decay_rate: f32);

    // Registers a new bucket (e.g. from Split/Flush)
    fn register_bucket(&self, bucket_id: u32, path: String, count: u32);

    async fn fetch_bucket(&self, bucket_id: u32) -> Result<(Vec<u64>, Vec<f32>)>;
}

/// Abstraction for managing deleted vectors (Write Path).
#[async_trait]
pub trait TombstoneTracker: Send + Sync + Debug {
    /// Marks a vector ID as deleted.
    fn mark_delete(&self, id: u64);
    fn mark_delete_batch(&self, id: &[u64]);

    /// Allow resurrection
    fn unmark_delete(&self, id: u64);
    fn unmark_delete_batch(&self, id: &[u64]);

    /// Checks if a single ID is deleted (Fast path for MemTable/Ingest).
    fn is_deleted(&self, id: u64) -> bool;

    /// Returns a thread-safe snapshot for a query execution.
    fn get_view(&self) -> Arc<dyn TombstoneView>;

    /// Frees up storage after compaction.
    fn dissolve_batch(&self, compacted_ids: &[u64]);
}

/// This completely hides whether we use HashSets, Bitmaps, or Bloom Filters.
#[allow(clippy::len_without_is_empty)]
pub trait TombstoneView: Send + Sync + Debug {
    fn contains(&self, id: u64) -> bool;
    /// Optional: Approximate count for statistics
    fn len(&self) -> usize;
}

pub trait IoContext<T> {
    fn context(self, msg: &str) -> std::io::Result<T>;
}

impl<T> IoContext<T> for std::io::Result<T> {
    fn context(self, msg: &str) -> std::io::Result<T> {
        self.map_err(|e| std::io::Error::new(e.kind(), format!("{msg}: {e}")))
    }
}
