# Changelog

## [0.3.0] - Server Async Migration

### Added

- **gRPC Search:** The `Search` RPC endpoint now utilizes the async core engine (`search_async`), enabling non-blocking concurrent searches over disk-resident data.
- **Tunable Parameters:** The `SearchRequest` protobuf message now supports optional `target_confidence`, `lambda`, and `tau` parameters for runtime tuning of the Saturating Density model.

## [0.2.1] - Stability & Correctness Hardening

### Added

- **Singularity Guard:** `split_and_steal` now calculates vector variance. If data is too clustered (variance < 0.01) or geometrically inseparable, the split is aborted to prevent CPU busy-loops.
- **MaintenanceStatus Enum:** Core operations now return `Completed`, `SkippedSingularity`, `SkippedTooSmall`, or `SkippedLocked` instead of opaque booleans.
- **Janitor Blacklist:** The Janitor maintains an ephemeral `ignore_set` for buckets identified as unsplittable singularities.
- **Operation Budgeting:** The Janitor now limits maintenance to 1 major operation per tick to ensure `perform_flush` is never starved during high-churn events ("Split Storms").

### Changed

- **Split Logic:** Switched from blind splitting to Drift-Aware splitting using the variance check.
- **Merge Logic:** Enforced "Strict Hysteresis"—only completely empty buckets (`count == 0`) are candidates for scatter-merge.
- **Stress Tests:** Updated `split_storm` and `scatter_split_race` to use high-variance random data, ensuring valid geometric separability during tests.

## [0.2.0] - Async Disk-Native Architecture

### Added

- **Async Core:** `VectorIndex` operations (`train`, `search`, `split`) are now async to support non-blocking disk I/O.
- **Drift-Aware Routing:** `search_async` implements the "Saturating Density" model (Lambda/Tau parameters).
- **Disk-Native L1:** Level 1 buckets are now backed by `BlockCache` and `PageManager`.
- **Auto-Registration:** `LocalDiskManager` manages file creation.
- **`drift_cache` Crate:** S3-FIFO eviction policy.
- **`drift_kv` Crate:** Persistent `BitStore` for O(1) `VectorID -> BucketID` lookups.

### Changed

- **Strong Consistency:** Atomic KV Store updates prevents data loss during maintenance.
- **Storage Format:** Serialized as `BucketData` pages with Magic Bytes (`0xBD47001`).
- **Deletion Logic:** Upgraded `delete(id)` to O(1) using the KV store.

### Removed

- Synchronous `Bucket` access.
- In-memory `Vec<Bucket>` storage.
