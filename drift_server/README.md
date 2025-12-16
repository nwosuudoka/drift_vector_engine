# drift_server

Orchestration layer that wires the core index and storage format from `PAPER.pdf` together: background flushes, WAL rotation, and segment hydration.

## Implemented features
- **Janitor loop**: configurable background task that watches the L0 memtable size, triggers `rotate_memtable`, and flushes snapshots to disk segments.
- **Persistence manager**: writes both L1 buckets and L0 snapshots to new `.drift` segments, embedding quantizer bytes and checksums; can rebuild a `VectorIndex` from a segment plus WAL.
- **WAL integration**: flush paths preserve durability guarantees by coordinating WAL truncation with memtable rotation.
- **Integration tests**: cover janitor flush paths and persistence round-trips (`wal_integration_test.rs`, `janitor_tests.rs`, `persistence_test.rs`).

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

This crate binds together `drift_core` and `drift_storage` to deliver the durability and background maintenance described in `PAPER.pdf`.
