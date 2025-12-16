# Drift Vector Engine

Rust workspace implementing the ideas sketched in `PAPER.pdf`: a drift-aware vector index with fast in-memory writes, SIMD-accelerated approximate search, and columnar on-disk persistence.

## Implemented features

- Hybrid two-level index: L0 `MemTable` (HNSW over raw f32 vectors) plus L1 product-quantized buckets with centroid routing and soft deletes.
- Durability: write-ahead log (`WalWriter`/`WalReader`) with checksum framing, crash replay, and atomic memtable rotation that truncates the WAL after flush.
- Search: drift-aware retrieval that merges L0 HNSW results with L1 ADC scans, using bucket temperature/density scoring (lambda/tau) and configurable `ef_search`.
- Adaptive maintenance: bucket splitting with local k-means, scatter-merge healing, neighbor stealing, and rebalancing to keep centroids healthy as data drifts.
- Quantization & training: percentile-clipped SQ8 quantizer, LUT-based ADC, and k-means++ training helpers with empty-cluster rescue.
- SIMD hot paths: 64-byte aligned buffers, AVX2 ADC kernels (with scalar/NEON fallback), and cache-aware heap selection for top-k.
- Storage format: columnar segment writer/reader with ALP/ALP_RD compression, checksumed blobs, Bloom filters for ID existence, embedded quantizer metadata, and a 64-byte footer (`MAGIC_BYTES`).
- Background persistence: Janitor task monitors L0 size and flushes to new segments via `PersistenceManager`, keeping WAL small while retaining crash recovery.
- Tests: unit and integration coverage for k-means, quantizer, WAL, bucket logic, compression, segment IO, persistence, and janitor/WAL end-to-end flow.

## Workspace layout

- `drift_core`: core vector index (L0+L1), WAL, quantizer, bucket maintenance, k-means, and SIMD scanning primitives.
- `drift_storage`: on-disk segment format, compression pipelines, Bloom filter/index footer, and async disk manager.
- `drift_server`: orchestration layer (janitor + persistence manager) that flushes/loads indices between memory and disk.
- Supporting docs: `PAPER.pdf` (design target), `TODO.md`.

## How to use

- Build/tests: `cargo test` (or `cargo test -p drift_core`, `-p drift_storage`, `-p drift_server`).
- Persistence: see `drift_server/src/persistence.rs` for flushing/loading patterns and `drift_server/src/janitor.rs` for the background loop.
- Paper alignment: Each crate README notes which portions of `PAPER.pdf` are implemented today.

## Status

This codebase implements the core mechanics from `PAPER.pdf`—a drift-aware ANN with WAL-backed memtables, PQ buckets, and compressed disk segments. Extend modules or add APIs/CLIs on top of the crates to integrate with your application.
