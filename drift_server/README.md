# drift_server

Orchestration layer that wires the core index and storage format from `PAPER.pdf` together: background flushes, WAL rotation, and segment hydration.

## Implemented features
- **gRPC service**: `Train`, `Insert`, and `Search` endpoints generated from `drift.proto`, wired via Tonic.
- **Multi-tenant collections**: per-collection storage directories under `./data/<collection>/` with isolated WAL + segments.
- **Janitor loop**: configurable background task that watches the L0 memtable size, triggers `rotate_memtable`, and flushes snapshots to disk segments.
- **Persistence manager**: writes both L1 buckets and L0 snapshots to new `.drift` segments, embedding quantizer bytes and checksums; can rebuild a `VectorIndex` from a segment plus WAL.
- **WAL integration**: flush paths preserve durability guarantees by coordinating WAL truncation with memtable rotation.
- **Integration tests**: cover janitor flush paths and persistence round-trips (`wal_integration_test.rs`, `janitor_tests.rs`, `persistence_test.rs`, `server_tests.rs`).

## Usage sketch
```rust
use drift_core::index::{IndexOptions, VectorIndex};
use drift_server::{janitor::Janitor, persistence::PersistenceManager};
use std::{path::PathBuf, sync::Arc, time::Duration};

let opts = IndexOptions { dim: 128, num_centroids: 64, training_sample_size: 10_000, max_bucket_capacity: 2000, ..Default::default() };
let wal_path = PathBuf::from("data/current.wal");
let index = Arc::new(VectorIndex::new(opts, &wal_path)?);

let persistence = PersistenceManager::new("data");
let janitor = Janitor::new(index.clone(), persistence, 10_000, Duration::from_secs(30));
tokio::spawn(async move { janitor.run().await; });
```

## gRPC quickstart
- Run the server: `cargo run -p drift_server` (listens on `127.0.0.1:50051`).
- Use the bundled client demo: `cargo run -p drift_server --bin client`.
- Protobuf definitions live in `drift_server/proto/drift.proto` (fields include `collection_name`, `vector`, `k`, `target_confidence`, `lambda`, `tau`).

This crate binds together `drift_core` and `drift_storage` and exposes the gRPC API for training, inserting, and searching the drift-aware index.
