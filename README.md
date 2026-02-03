# Drift Vector Engine (v2)

Rust workspace implementing a drift-aware vector index with WAL-backed ingestion, unified
on-disk row-group storage, and background maintenance for evolving data distributions.
This README describes the **current v2 system** as implemented in this repository.

## Overview

v2 is organized around **Logical Buckets** that can live in multiple physical locations:

- **Local staging**: mutable `.drift` files under `data/<collection>/staging/`.
- **Remote base**: immutable `.drift` files in the configured storage root (local or S3).
- **Tiered**: a remote base plus a local delta.

Search and maintenance operate on this logical view, while the Janitor keeps metadata
and files consistent.

## Architecture (v2)

### Core components

- **L0 MemTable (v2)**: append-only in-memory buffer with parallel scan search.
- **WAL**: per-collection WAL for durable inserts/deletes, replayed on recovery.
- **L1 Bucket Files**: columnar `.drift` files with hot SQ8 + cold ALP data, multiple
  row groups for local staging, and a single run id for remote base files.
- **Router**: centroid routing table that selects buckets using target_confidence, lambda,
  tau, plus a distance guardrail.
- **BucketManager**: tracks bucket state (Local/Remote/Tiered/Promoting), tombstones,
  and provides unified search across local/remote tiers.
- **BitStore (KV)**: persistent `VectorID -> BucketID` mapping used for L1 tombstones.
- **Janitor v2**: background loop for flush, promotion, split, scatter merge, tombstone
  persistence, and reaper cleanup.

### Data layout (on disk)

For a collection named `my_collection`:

- WAL: `WAL_DIR/my_collection/`
- Local staging: `DATA_DIR/my_collection/staging/`
- KV store: `DATA_DIR/my_collection/kv/`
- Remote storage root (local or S3): `STORAGE_ROOT/my_collection/`

## Write path (v2)

1. **Insert/Train** writes to WAL first.
2. **MemTable** receives the vector (L0 visibility).
3. When L0 reaches capacity it is rotated to a frozen table.
4. **Janitor flush** partitions frozen vectors using the Router and appends row groups
   to local staging files.
5. **Router and KV** are updated: centroids/counts are recalculated and KV mappings
   are written for each flushed ID.

## Read path (v2)

1. **Snapshot** L0 tombstones and active/frozen tables.
2. **L0 scan**: parallel scan of memtables, filtered by L0 tombstones.
3. **Bucket selection**: Router chooses buckets using target_confidence/lambda/tau plus
   a distance guardrail.
4. **L1 scan**: BucketManager scans local/remote/tiered files, applying per-bucket
   tombstones and refining candidates.
5. **Merge** L0 and L1 results, re-check L0 tombstones, return top-K.

## Promotion (Local -> Remote)

When a local staging file exceeds a threshold, the Janitor promotes it:

1. **Lock** the bucket with the BucketCoordinator.
2. **Rotate** the local file (active -> frozen) and create a new empty local file.
3. **Merge** frozen local data with the remote base (if any), apply tombstones, and
   write a new remote file.
4. **Finalize**: bucket becomes Tiered, manifest run id is updated, tombstones are
   pruned, and old files are scheduled for cleanup.

Result: remote is compacted, local continues as a delta, and search reads both.

## Split and Defector Loopback (Split/Steal update)

When a bucket is too large or drifted:

1. **Fetch** the bucket (merging tiers if needed).
2. **K-Means (K=2)** produces two child centroids.
3. **Partition** vectors into left/right children.
4. **Defector check**: vectors that are much closer to a different global centroid are
   looped back to L0 instead of being written to children.
5. **Write** new local files for children, update manifest/router, delete old staging file.

This replaces the older explicit neighbor stealing with a safer defector loopback guard.

## Scatter Merge (Zombie healing)

When a bucket becomes a zombie (low count / high urgency):

1. **Fetch** the bucket data.
2. **Scatter** each vector to the nearest neighbor centroid.
3. **Rewrite neighbors** by writing new local files (copy-on-write).
4. **Update** manifest/router/bucket manager and delete the zombie file.

## Tombstones and deletes

- **L0 tombstones** are maintained in-memory for fast filtering.
- **L1 tombstones** are tracked per bucket; KV mapping provides `VectorID -> BucketID`.
- **Persistence**: Janitor periodically persists a cumulative tombstone file and updates
  the manifest pointer.

## Recovery

On startup, RecoveryManager:

1. Loads the manifest and rebuilds the Router.
2. Registers buckets in BucketManager (local or remote).
3. Replays WAL inserts/deletes.
4. Hydrates persisted tombstones.

## Configuration

CLI flags (also available via environment variables):

- `DRIFT_PORT` (default 50051)
- `DRIFT_WAL_DIR` (default ./data/wal)
- `DRIFT_DATA` (default ./data/drift)
- `DRIFT_DEFAULT_DIM` (default 128)
- `DRIFT_MAX_BUCKET_CAPACITY` (default 1000)
- `DRIFT_EF_CONSTRUCTION` (default 128)
- `DRIFT_EF_SEARCH` (default 50)

Storage:

- File backend: `DRIFT_DATA_DIR` (root directory for remote storage)
- S3 backend: `DRIFT_S3_BUCKET`, `DRIFT_S3_REGION`, `DRIFT_S3_ENDPOINT`,
  `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`

## How to use (v2)

### Run the server

```
cargo run -p drift_server
```

### Client / CLI

```
cargo run -p drift_server --bin client
cargo run -p drift_server --bin drift -- --help
```

### Simulations

```
# v2 drift simulation
cargo run -p drift_server --bin drift_sim --release

# v2 churn/tombstone simulation
cargo run -p drift_server --bin churn_sim --release
```

### Benchmarking (v2)

```
# Read/write benchmark (in-process)
scripts/bench_rw.sh

# Custom parameters
scripts/bench_rw.sh -- --total-vectors 50000 --query-count 500
```

## Workspace layout

- `drift_core`: v2 index, router, memtable, WAL, maintenance algorithms.
- `drift_storage`: `.drift` file format, row groups, quantization, compression.
- `drift_kv`: BitStore mapping for `VectorID -> BucketID`.
- `drift_server`: gRPC server, v2 manager, janitor, persistence, and sims.

## Current status (v2)

- v2 server/manager/janitor are active and used by default (`drift_server` binary).
- L0 uses a parallel scan MemTable; HNSW-based L0 is legacy v1.
- v2 maintenance includes split with defector loopback and scatter merge.
- Promotion supports Local/Remote/Tiered/Promoting states.

If you want to map features to v1/v2 explicitly, see `SYSTEM_VIEW.md` and `TODO.v2.2.md`.
