# Drift Vector Engine

Rust workspace implementing the ideas sketched in `PAPER.pdf`: a drift-aware vector index with fast in-memory writes, SIMD-accelerated approximate search, and columnar on-disk persistence.

## Implemented features

- Hybrid two-level index: L0 `MemTable` (HNSW over raw f32 vectors) plus L1 product-quantized buckets with centroid routing and soft deletes.
- Durability: write-ahead log (`WalWriter`/`WalReader`) with checksum framing, crash replay, and atomic memtable rotation that truncates the WAL after flush.
- Async search & training: `search_async` and async K-Means training to keep disk-backed operations non-blocking.
- Drift-aware retrieval: merges L0 HNSW results with L1 ADC scans, using bucket temperature/density scoring (lambda/tau) and configurable `ef_search`.
- Adaptive maintenance: bucket splitting with local k-means, scatter-merge healing, neighbor stealing, and rebalancing to keep centroids healthy as data drifts.
- Quantization & training: percentile-clipped SQ8 quantizer, LUT-based ADC, and k-means++ training helpers with empty-cluster rescue.
- SIMD hot paths: 64-byte aligned buffers, AVX2 ADC kernels (with scalar/NEON fallback), and cache-aware heap selection for top-k.
- Storage format: columnar segment writer/reader with ALP/ALP_RD compression, checksumed blobs, Bloom filters for ID existence, embedded quantizer metadata, and a 64-byte footer (`MAGIC_BYTES`).
- Block cache + KV: S3-FIFO block cache for bucket pages and a persistent `BitStore` for O(1) `VectorID -> BucketID` lookups.
- Server API: gRPC `Train`, `Insert`, and `Search` endpoints with multi-tenant collections and tunable search parameters.
- Background persistence: Janitor task monitors L0 size and flushes to new segments via `PersistenceManager`, keeping WAL small while retaining crash recovery.
- Tests: unit and integration coverage for k-means, quantizer, WAL, bucket logic, compression, segment IO, persistence, and janitor/WAL end-to-end flow.

## Workspace layout

- `drift_core`: core vector index (L0+L1), WAL, quantizer, bucket maintenance, k-means, SIMD scanning, and async search/training.
- `drift_storage`: on-disk segment format, compression pipelines, Bloom filter/index footer, and async disk manager.
- `drift_cache`: block cache (S3-FIFO) and page manager implementations for disk-native buckets.
- `drift_kv`: persistent `BitStore` used for `VectorID -> BucketID` mappings.
- `drift_server`: gRPC server, multi-tenant collection manager, janitor, and persistence wiring.
- Supporting docs: `PAPER.pdf` (design target), `TODO.md`.

## How to use

- Build/tests: `cargo test` (or `cargo test -p drift_core`, `-p drift_storage`, `-p drift_server`).
- Run the gRPC server: `cargo run -p drift_server` (listens on `127.0.0.1:50051`).
- Sample client: `cargo run -p drift_server --bin client` (multi-collection insert/search demo).
- API definitions: `drift_server/proto/drift.proto` defines `Train`, `Insert`, and `Search` requests.
- Persistence: see `drift_server/src/persistence.rs` for flushing/loading patterns and `drift_server/src/janitor.rs` for the background loop.
- Paper alignment: Each crate README notes which portions of `PAPER.pdf` are implemented today.

## Status

This codebase implements the core mechanics from `PAPER.pdf`—a drift-aware ANN with WAL-backed memtables, PQ buckets, compressed disk segments, and a gRPC service layer for training and search.
