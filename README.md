# Drift Vector Engine

Rust workspace implementing the ideas sketched in `PAPER.pdf`: a drift-aware vector index with fast in-memory writes, SIMD-accelerated approximate search, and columnar on-disk persistence.

## Implemented features

- Hybrid two-level index: L0 `MemTable` (HNSW over raw f32 vectors) plus L1 product-quantized buckets with centroid routing and soft deletes.
- Lazy ingestion + durability: O(1) memtable writes, async indexing on flush, WAL replay, and atomic rotation that truncates the WAL after flush.
- Async search & training: `search_async` and async K-Means training with RAM-first search, phase separation (async IO vs sync CPU), and a recall guardrail.
- Drift-aware retrieval: merges L0 HNSW results with L1 ADC scans using bucket temperature/density scoring (lambda/tau) and configurable `ef_search`.
- Adaptive maintenance: bucket splitting with local k-means, scatter-merge healing, neighbor stealing, rebalancing, and Scavenger compaction to purge tombstones.
- Quantization & training: percentile-clipped SQ8 quantizer, LUT-based ADC, and k-means++ training helpers with empty-cluster rescue.
- SIMD hot paths: 64-byte aligned buffers, AVX2 ADC kernels (with scalar/NEON fallback), and cache-aware heap selection for top-k.
- Storage format: columnar `.drift` segments with hot SQ8 blobs + cold ALP/ALP_RD blobs, checksummed blobs, Bloom filters, embedded quantizer metadata, and a 64-byte footer (`MAGIC_BYTES`).
- Cloud-native IO + cache: opendal-backed storage URIs (local/S3) with a tiered page cache (local + remote, read-ahead) and a persistent `BitStore` for O(1) `VectorID -> BucketID` lookups.
- Server API + tools: gRPC `Train`, `Insert`, and `Search`, multi-tenant collections, a `drift` CLI client, and `drift_sim` drift harness.
- Background persistence: Janitor monitors L0 size, flushes to new segments via `PersistenceManager`, and persists tombstones for crash-safe deletes.
- Tests: unit and integration coverage for k-means, quantizer, WAL, bucket logic, compression, segment IO, persistence, and janitor/WAL end-to-end flow.

## Workspace layout

- `drift_core`: core vector index (L0+L1), WAL, quantizer, bucket maintenance, k-means, SIMD scanning, and async search/training.
- `drift_storage`: on-disk segment format, compression pipelines, Bloom filter/index footer, and async disk manager.
- `drift_cache`: block cache (S3-FIFO) and page manager implementations for disk-native buckets.
- `drift_kv`: persistent `BitStore` used for `VectorID -> BucketID` mappings.
- `drift_server`: gRPC server, multi-tenant collection manager, janitor, persistence wiring, and CLI/simulation binaries.
- Supporting docs: `PAPER.pdf` (design target), `TODO.md`.

## How to use

- Build/tests: `cargo test` (or `cargo test -p drift_core`, `-p drift_storage`, `-p drift_server`).
- Run the gRPC server: `cargo run -p drift_server` (listens on `127.0.0.1:50051`).
- Sample client: `cargo run -p drift_server --bin client` (multi-collection insert/search demo).
- CLI client: `cargo run -p drift_server --bin drift -- --help`.
- Drift harness: `cargo run -p drift_server --bin drift_sim`.
- API definitions: `drift_server/proto/drift.proto` defines `Train`, `Insert`, and `Search` requests.
- Persistence: see `drift_server/src/persistence.rs` for flushing/loading patterns and `drift_server/src/janitor.rs` for the background loop.
- Paper alignment: Each crate README notes which portions of `PAPER.pdf` are implemented today.

## Status

This codebase implements the core mechanics from `PAPER.pdf`—a drift-aware ANN with WAL-backed memtables, PQ buckets, compressed disk segments, and a gRPC service layer for training and search. Distributed consensus for stateless workers is the next major milestone (see `TODO.md`).
