# drift_core

Core engine for the drift-aware ANN described in `PAPER.pdf`: HNSW-backed memtable, PQ-compressed L1 buckets, WAL durability, and maintenance primitives that keep the index healthy as data drifts.

## Implemented features
- **L0 MemTable**: thread-safe HNSW over raw f32 vectors for high-recall recent data; exposes `insert`, `search`, and snapshot extraction for flushing.
- **Write-Ahead Log**: framed WAL (`WalWriter`/`WalReader`) with CRC32, length prefixing, append-only inserts, and replay on startup for crash recovery.
- **Quantization**: percentile-clipped SQ8 quantizer (1–99% clipping) with encode/reconstruct and LUT-based ADC for PQ buckets.
- **Bucketed L1**: centroid routing table, SIMD ADC scans, temperature/urgency tracking, tombstones, and 64-byte aligned storage for PQ codes.
- **Async search + train**: `search_async` and async K-Means training for disk-native buckets without blocking the runtime.
- **Disk-native plumbing**: `BlockCache` for bucket pages plus a persistent `BitStore` for O(1) `VectorID -> BucketID` lookups.
- **Maintenance**: bucket splitting, neighbor stealing, rebalance, and scatter-merge; atomic COW updates of centroid/bucket maps using epoch GC.
- **Training**: k-means++ trainer with empty-cluster rescue, used for initial centroid placement and splitting; helper to force-register hydrated buckets.

## Key modules
- `index.rs`: `VectorIndex`, async search/train, maintenance ops, memtable rotation, WAL integration, cache + KV wiring.
- `memtable.rs`: HNSW-backed mutable tier with snapshot extraction.
- `wal.rs`: WAL writer/reader for inserts.
- `bucket.rs`: PQ bucket layout, SIMD ADC kernels (AVX2 + scalar fallback), tombstone handling, and urgency/temperature bookkeeping.
- `quantizer.rs`: SQ8 encode/reconstruct + LUT generation.
- `kmeans.rs`: k-means++ trainer with robustness tests.
- `aligned.rs`: 64-byte aligned byte buffer used by buckets.

## Usage sketch
```rust
use drift_cache::local_store::LocalDiskManager;
use drift_core::index::{IndexOptions, VectorIndex};
use std::{path::Path, sync::Arc};

# #[tokio::main]
# async fn main() -> Result<(), Box<dyn std::error::Error>> {
let opts = IndexOptions { dim: 128, num_centroids: 64, training_sample_size: 10_000, max_bucket_capacity: 2000, ..Default::default() };
let wal_path = Path::new("data/current.wal");
let storage = Arc::new(LocalDiskManager::new("data/storage"));
let index = VectorIndex::new(opts, wal_path, storage)?;

// Train quantizer/centroids before L1 inserts:
// index.train(&training_samples).await?;
index.insert(42, &[0.1_f32; 128])?;
let hits = index.search_async(&[0.1_f32; 128], 10, 0.95, 25.0, 100.0).await?;
# Ok(())
# }
```

See `drift_server` for persistence flows and `PAPER.pdf` for the target design this crate implements.
