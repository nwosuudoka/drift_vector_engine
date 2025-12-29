# Changelog

## [0.6.2] - Persistence Hardening & Dual-Tier Storage

**Tag:** `v0.6.2-storage-stable`

### Added

- **Dual-Tier Segments:** Segments now contain both a `Hot Blob` (SQ8 codes for O(1) search) and a `Cold Blob` (ALP compressed floats for re-ranking).
- **Recovery Fallback:** `PersistenceManager::load_from_segment` and `hydrate_index` now attempt to read High-Fidelity data first, but gracefully degrade to reconstructing from SQ8 codes if the cold blob is missing or corrupt. This ensures the system can always recover.

### Changed

- **Storage Protocol:** `BucketData` is no longer reconstructed during flush; raw SQ8 bytes are written directly to disk to minimize CPU overhead.
- **Test Suite:** Added `test_end_to_end_persistence_lifecycle` to verify crash recovery using the new Dual-Tier format.

## [0.6.0] - The Performance Fix (ADC LUT)

**Work in Progress**

### Changed

- **Hot Search Loop:** Replaced per-vector floating point calculations with a precomputed Look-Up Table (LUT).
  - _Old:_ `dist += (q[i] - reconstruct(code[i]))^2` (CPU bound)
  - _New:_ `dist += lut[i * 256 + code[i]]` (Memory bound, 10x faster).
- **Bucket API:** `scan_static` now accepts `&[f32]` (LUT) instead of `&[f32]` (Query).

### Added

- **Unrolled Kernel:** Manually unrolled 4x loop in `compute_distance_lut` to saturate instruction pipelines.

## [0.5.4] - Lazy Indexing & High-Throughput Ingestion

**Tag:** `v0.5.4-lazy-indexing-stable`

### Added

- **Lazy Indexing Architecture:** Decoupled ingestion from indexing. Writes now go to an $O(1)$ `MemTable` buffer, while indexing happens asynchronously during flush.
- **Parallel Brute-Force Search:** Implemented a Rayon-based parallel scanner for the MemTable, maintaining low query latency (~8ms) even with large unindexed buffers.
- **Robust Janitor Logic:** Updated `Janitor` to correctly detect MemTable size via `HashMap` length and retry failed flushes gracefully without deadlocking.
- **Stress Tests:** Added `server_heavy_load_test` to validate throughput at ~670k vec/sec and ensure eventual consistency under load.

### Changed

- **Write Throughput:** Massive performance boost (~500x). Batch inserts no longer block on HNSW graph updates.
- **Search Logic:** `search_async` now performs a hybrid search: Parallel Scan (MemTable) + HNSW Search (Disk Segments).
- **Test Suite:** Updated `janitor_tests` and `stress_tests` to use robust polling (eventual consistency) instead of brittle sleeps.

### Fixed

- **Silent Disk Errors:** `DiskManager::upload` now correctly propagates IO errors instead of swallowing them.
- **MemTable Length Bug:** `memtable.len()` now reports the correct count from the backing `HashMap`, fixing Janitor flush triggers.
- **Test Race Conditions:** Fixed flaky tests in `janitor_tests.rs` by ensuring disk operations complete before aborting background tasks.

## [0.5.2] - Metric Unification & Config

### Added

- **Global Config:** Server now configurable via CLI arguments (`--port`, `--storage-uri`) and Environment Variables (`DRIFT_PORT`, etc.).

### Changed

- **Metric Standardization:** The Search API now consistently returns Euclidean Distance. Internally, the engine operates on Squared Euclidean Distance for performance and correctness during ranking.

## [0.5.0] - Cloud-Native Storage & Stability

**Tag:** `v0.5.0-cloud-native-beta`

### Added

- **Cloud Storage Support:** Integrated `apache-opendal` to support S3, GCS, Azure, and Local FS uniformly.
- **URI Configuration:** Server now accepts storage paths as URIs (e.g., `file:///data`, `s3://my-bucket`).
- **Scratch File Writing:** Segments are buffered to a local temporary file and uploaded atomically on completion, ensuring compatibility with immutable object stores.
- **Auto-Registration:** `DriftPageManager` automatically registers new file IDs during write operations, preventing "File ID not found" errors during training.

### Changed

- **Dynamic Dimensions:** Server no longer panics on dimension mismatch; it adapts to the input data dimension or validates against the existing schema.
- **Refactored Persistence:** `PersistenceManager` now uses `DriftPageManager` instead of `LocalDiskManager`.

### Fixed

- **Write Page Panic:** Implemented Read-Modify-Write logic for `write_page` to support random-access writes required by `drift_core` on top of object storage.

## [0.4.0] - Single-Node Stable Release

**Tag:** `v0.4.0-single-node-stable`

### Added

- **`drift-cli`:** A command-line tool to Train, Insert, and Search without writing code.
- **Dynamic Dimension Sizing:** Server automatically configures index dimensions based on input data.
- **End-to-End Verification:** Validated the full stack from CLI -> gRPC -> Async Core -> Disk -> WAL.

### Fixed

- **Bloom Filter Integration:** `drift_storage` now correctly uses `fastbloom` for O(1) negative lookups.
- **Panic on Dimension Mismatch:** Server now validates or adapts to vector dimensions instead of crashing.

---

## [0.3.3] - Server API Completion

### Added

- **Full gRPC Suite:** Implemented `Train`, `Insert`, and `Search` endpoints in `DriftService`.
- **Async Training:** Exposed the heavy K-Means index construction as a non-blocking async RPC.
- **Server Verification:** Validated the full lifecycle (Train -> Insert -> Search) via `server_tests.rs`.

## [0.3.3] - Train API Implementation

### Added

- **gRPC Train Handler:** Implemented the `train` RPC endpoint. It accepts a batch of vectors, converts them from Protobuf format to internal `Vec<f32>`, and awaits the async `index.train()` method.
- **Server Tests:** Added `server_tests.rs` to verify the full API lifecycle (Train -> Insert -> Search) via the `DriftService` abstraction.

## [0.3.2] - Protobuf Schema Definition

### Added

- **`drift.proto`:** Defined the gRPC service contract, including `Vector`, `InsertRequest`, `SearchRequest` (with tunable parameters), and `TrainRequest`.
- **`build.rs`:** Added Tonic build script to compile the Protobuf schema during the build process.

## [0.3.1] - Async Train Endpoint

### Added

- **gRPC Train:** Implemented `DriftService::train` to accept a batch of vectors, convert them to the internal format, and invoke the async `index.train()` method. This allows non-blocking index construction.

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
