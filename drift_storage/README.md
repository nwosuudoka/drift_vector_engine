# drift_storage

On-disk storage layer for the drift vector engine described in `PAPER.pdf`. Provides a compressed, columnar segment format with integrity checks and fast membership hints.

## Implemented features
- **Segment writer/reader**: persists buckets into `.drift` files with columnar vectors, separate ID columns, CRC32 checksums, and embedded quantizer bytes.
- **Footer + magic**: fixed 64-byte footer containing offsets/lengths for index, bloom filter, and quantizer blobs; validated on read via `MAGIC_BYTES`.
- **Bloom filter**: per-segment Bloom filter for quick “might contain ID” checks before disk scans.
- **Compression**: ALP/ALP_RD codecs for float columns, with transpose to column-major layout for higher compression ratios; lossless round-trips covered by tests.
- **Async disk manager**: `DiskManager` wraps Tokio file IO for aligned reads/writes, cursor management, and syncing.

## Key modules
- `segment_writer.rs`: writes IDs, compressed vector columns, index map, bloom filter, quantizer metadata, and footer.
- `segment_reader.rs`: validates footer, restores index/bloom/quantizer, and reads buckets back into (ids, vectors) pairs.
- `compression/`: ALP/ALP_RD codecs (`alp.rs`, `alp_rd.rs`), transposition helpers, and fuzz-style tests for edge cases.
- `block.rs`: 4KB-aligned page blocks for O_DIRECT-friendly buffers.
- `disk_manager.rs`: async helpers for offset-based IO and metadata.

## Usage sketch
```rust
use drift_storage::{segment_reader::SegmentReader, segment_writer::SegmentWriter};
// See drift_server/persistence.rs for full end-to-end usage with the core index.
```

This crate is the persistence substrate the server layer uses to flush L0/L1 data; it follows the format outlined in `PAPER.pdf`.
