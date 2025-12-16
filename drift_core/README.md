# drift_core

Core engine for the drift-aware ANN described in `PAPER.pdf`: HNSW-backed memtable, PQ-compressed L1 buckets, WAL durability, and maintenance primitives that keep the index healthy as data drifts.

## Implemented features
- **L0 MemTable**: thread-safe HNSW over raw f32 vectors for high-recall recent data; exposes `insert`, `search`, and snapshot extraction for flushing.
- **Write-Ahead Log**: framed WAL (`WalWriter`/`WalReader`) with CRC32, length prefixing, append-only inserts, and replay on startup for crash recovery.
- **Quantization**: percentile-clipped SQ8 quantizer (1–99% clipping) with encode/reconstruct and LUT-based ADC for PQ buckets.
- **Bucketed L1**: centroid routing table, SIMD ADC scans, temperature/urgency tracking, tombstones, and 64-byte aligned storage for PQ codes.
- **Search**: `search_drift_aware` merges L0 HNSW hits with L1 ADC results, prioritizing buckets via geometric/density scoring (`lambda`/`tau`) to follow data drift.
- **Maintenance**: bucket splitting, neighbor stealing, rebalance, and scatter-merge; atomic COW updates of centroid/bucket maps using epoch GC.
- **Training**: k-means++ trainer with empty-cluster rescue, used for initial centroid placement and splitting; helper to force-register hydrated buckets.

## Key modules
- `index.rs`: `VectorIndex`, search path, maintenance ops, memtable rotation, WAL integration.
- `memtable.rs`: HNSW-backed mutable tier with snapshot extraction.
- `wal.rs`: WAL writer/reader for inserts.
- `bucket.rs`: PQ bucket layout, SIMD ADC kernels (AVX2 + scalar fallback), tombstone handling, and urgency/temperature bookkeeping.
- `quantizer.rs`: SQ8 encode/reconstruct + LUT generation.
- `kmeans.rs`: k-means++ trainer with robustness tests.
- `aligned.rs`: 64-byte aligned byte buffer used by buckets.

## Usage sketch
```rust
use drift_core::index::{IndexOptions, VectorIndex};
use std::path::Path;

let opts = IndexOptions { dim: 128, num_centroids: 64, training_sample_size: 10_000, max_bucket_capacity: 2000, ..Default::default() };
let wal_path = Path::new("data/current.wal");
let index = VectorIndex::new(opts, wal_path)?;
// Train quantizer/centroids before L1 inserts:
// index.train(&training_samples);
index.insert(42, &[0.1_f32; 128])?;
let hits = index.search_drift_aware(&[0.1_f32; 128], 10, 0.95, 0.7, 100.0);
```

See `drift_server` for persistence flows and `PAPER.pdf` for the target design this crate implements.
