# Changelog

## [0.2.0] - Async Disk-Native Architecture

### Added

- **Async Core:** `VectorIndex` operations (`train`, `search`, `split`) are now async to support non-blocking disk I/O.
- **Drift-Aware Routing:** `search_async` implements the "Saturating Density" model (Lambda/Tau parameters) to filter low-confidence buckets.
- **Disk-Native L1:** Level 1 buckets are now backed by `BlockCache` and `PageManager`, loading data on-demand instead of holding everything in RAM.
- **Auto-Registration:** `LocalDiskManager` now automatically manages file creation for new buckets.
- **`drift_cache` Crate:** Created library to handle memory management and disk I/O with S3-FIFO eviction.
- **`drift_kv` Crate:** Integrated persistent `BitStore` for O(1) `VectorID -> BucketID` lookups.

### Changed

- **Strong Consistency:** `split_and_steal` and `scatter_merge` now guarantee atomic KV Store updates, preventing vector data loss during maintenance.
- **Storage Format:** Buckets are serialized as `BucketData` pages (Header + Codes + VIDs + Tombstones) with Magic Bytes validation (`0xBD47001`).
- **Quantization:** Improved SQ8 quantization with rounding for higher precision.
- **Maintenance Logic:** `Janitor` now strictly separates "Healing" and "Growth" logic.
- **Deletion Logic:** Upgraded `delete(id)` to O(1) using the KV store.

### Removed

- Synchronous `Bucket` access in the public API.
- In-memory `Vec<Bucket>` storage (replaced by `Atomic<HashMap<u32, BucketHeader>>`).

## [0.1.0] - Initial Prototype

### Added

- **HNSW Graph:** Thread-safe MemTable for hot data (Level 0).
- **Hybrid Search:** Merges L0 (Graph) and L1 (Disk) results.
- **Write-Ahead Log (WAL):** Durability for L0 inserts.
- **gRPC Interface:** Basic `DriftService` implementation.
- **Bitpacking:** Scalar bit-packing for Tombstone blocks.
