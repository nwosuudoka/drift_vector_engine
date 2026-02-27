# Drift Vector Engine

Rust workspace implementing drift-aware vector search with WAL-backed ingestion, unified `.driftu`
storage, typed payload columns, and background maintenance.

This document reflects the current v3 implementation in this repository.

## Overview

System state is modeled as logical buckets that can span multiple physical files:

- Local staging: mutable `.driftu` files under `data/<collection>/staging/`.
- Remote base: immutable `.driftu` files in the configured storage backend root.
- Tiered: remote base plus local delta.
- Promoting: temporary merge state during local -> remote promotion.

Search and maintenance operate on this logical view through `BucketManager`.

## V3 status

- Storage format is strict v3-only (`.driftu`) on read and write paths.
- Collection metric is explicit at creation time and validated on reopen/reuse.
- Typed payload rows are stored in unified files alongside vectors.
- Payload schema management RPCs are available for create/update/get/validate field definitions.
- Search API supports field filters (`exact`, `any_of`, `range`) and payload projection.
- Recovery validates payload/index metadata against manifest state and reports diagnostics.
- Remote reads can use local NVMe full-object cache with fingerprint-aware invalidation.
- Optional Prometheus metrics exporter exposes cache and recovery counters.

## Architecture

### Core components

- L0 MemTable: append-only in-memory buffer with parallel search scan.
- WAL: per-collection durability for inserts/deletes before L0 mutation.
- L1 bucket files: unified `.driftu` files with vector data, payload schema/rows, and payload indexes.
- Router: centroid routing for bucket selection (`target_confidence`, `lambda`, `tau`).
- BucketManager: tracks bucket location/state (Local/Remote/Tiered/Promoting).
- KV store (`drift_kv`): persistent `VectorID -> BucketID` mapping for tombstone propagation.
- Janitor: flush, promote, split, scatter-merge, tombstone persistence, and cleanup.
- DiskManager: optional local cache for remote object reads.

## Data layout (on disk)

For collection `my_collection`:

- WAL: `WAL_DIR/my_collection/`
- Local collection root: `DRIFT_DATA/my_collection/`
- Local staging: `DRIFT_DATA/my_collection/staging/`
- KV store: `DRIFT_DATA/my_collection/kv/`
- Remote storage root: `<storage-root>/my_collection/` (`file` or `s3` backend)

## Write path

1. Insert/InsertBatch writes to WAL first.
2. Vector (and optional payload row) is appended to L0 MemTable.
3. MemTable rotation creates frozen tables when capacity is reached.
4. Janitor flush partitions rows by router centroid and appends unified row groups to local staging.
5. KV and router are updated to keep `VectorID -> BucketID` and centroid stats consistent.

## Read path

1. Snapshot L0 state (active/frozen tables + tombstones).
2. Router selects candidate buckets using drift parameters.
3. Vector search returns top-K candidate IDs from L0 and L1.
4. If request has payload filters/projection:
   - candidate K is expanded (`k * 8`, capped at 8192),
   - payload rows are loaded from L1 (plus L0 fallback),
   - filters are applied in server layer,
   - optional payload projection is attached to results.
5. L0/L1 results merge and final top-K is returned.

## Promotion (Local -> Remote)

When local staging exceeds threshold:

1. Bucket is locked.
2. Active local file is rotated into frozen staging.
3. Frozen local rows merge with existing remote base (if present), applying tombstones.
4. New remote `.driftu` file is written.
5. Manifest metadata is updated atomically (run/path/fingerprint + payload/index flags/hash).
6. Bucket transitions to Tiered and old files are scheduled for reaper cleanup.

## Recovery

On startup:

1. Manifest is loaded and router is rebuilt.
2. Bucket locations/classes are re-registered.
3. WAL inserts/deletes are replayed.
4. Tombstones are hydrated.
5. Recovery guard validates remote-object fingerprints and payload/index metadata alignment.

## Configuration

Main server flags (also available as env vars):

- `DRIFT_PORT` (default `50051`)
- `DRIFT_WAL_DIR` (default `./data/wal`)
- `DRIFT_DATA` (default `./data/drift/`)
- `DRIFT_DEFAULT_DIM` (default `128`)
- `DRIFT_MAX_BUCKET_CAPACITY` (default `1000`)
- `DRIFT_EF_CONSTRUCTION` (default `128`)
- `DRIFT_EF_SEARCH` (default `50`)

Storage backend:

- File backend: `DRIFT_DATA_DIR` (subcommand `file`)
- S3 backend: `DRIFT_S3_BUCKET`, `DRIFT_S3_REGION`, `DRIFT_S3_ENDPOINT`,
  `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` (subcommand `s3`)

NVMe object cache:

- `DRIFT_NVME_CACHE_DIR`
- `DRIFT_NVME_CACHE_MAX_BYTES`
- `DRIFT_NVME_CACHE_MAX_FILES`
- `DRIFT_NVME_CACHE_MAX_FILE_BYTES`
- `DRIFT_NVME_FINGERPRINT_VERIFY_INTERVAL_MS`
- `DRIFT_NVME_CACHE_FORCE_ALL_SCHEMES=1` (test-only override)

Metrics exporter:

- `DRIFT_METRICS_ADDR` (preferred, example `0.0.0.0:9091`)
- `DRIFT_METRICS_PORT` (fallback)

KV durability startup controls:

- `DRIFT_KV_SYNC_INTERVAL_MS`
- `DRIFT_KV_FORCE_REBUILD_ON_STARTUP`
- `DRIFT_KV_VALIDATE_MAX_BUCKETS`
- `DRIFT_KV_VALIDATE_IDS_PER_BUCKET`

## Run and use

### Start server (local filesystem backend)

```bash
cargo run -p drift_server --bin drift_server -- file --path ./data
```

### Start server (S3 backend)

```bash
cargo run -p drift_server --bin drift_server -- s3 --bucket <bucket> --region us-east-1
```

### CLI client (vector-focused convenience wrapper)

```bash
cargo run -p drift_server --bin drift -- --help
```

`drift` CLI currently covers create/train/insert/search for vectors.
For payload inserts, filters, and projection, use the gRPC API directly.

### API reference

- gRPC proto: `drift_server/proto/drift.proto`
- Developer API spec: `docs/API_SPEC.md`
- Architecture flow: `SYSTEM_VIEW.md`

### Simulations

```bash
cargo run -p drift_server --bin drift_sim --release
cargo run -p drift_server --bin churn_sim --release
```

### Benchmarking

```bash
cargo run -p drift_server --bin bench_rw --release -- --total-vectors 20000 --query-count 200
```

Calibration profile commands (emit JSON artifacts for threshold tuning):

```bash
cargo run -p drift_server --bin bench_rw --release -- \
  --dim 32 --total-vectors 4000 --batch-size 500 \
  --query-count 80 --warmup-queries 15 \
  --filtered-query-count 80 --filtered-warmup-queries 15 \
  --filter-cardinality 32 --k 10 \
  --summary-json-path /tmp/bench_rw_calib_small.json

cargo run -p drift_server --bin bench_rw --release -- \
  --dim 64 --total-vectors 20000 --batch-size 1000 \
  --query-count 120 --warmup-queries 20 \
  --filtered-query-count 120 --filtered-warmup-queries 20 \
  --filter-cardinality 64 --k 10 \
  --summary-json-path /tmp/bench_rw_calib_medium.json

cargo run -p drift_server --bin bench_rw --release -- \
  --dim 64 --total-vectors 60000 --batch-size 1500 \
  --query-count 150 --warmup-queries 25 \
  --filtered-query-count 150 --filtered-warmup-queries 25 \
  --filter-cardinality 128 --k 10 \
  --summary-json-path /tmp/bench_rw_calib_large.json
```

CI default filtered guardrails (when `CI=1` and explicit filtered limits are not passed):
- small tier (`total_vectors <= 10_000`): `max_filtered_p95_ms=12`, `max_filtered_overhead_ratio=7.0`
- medium tier (`10_001..=50_000`): `max_filtered_p95_ms=35`, `max_filtered_overhead_ratio=7.0`
- large tier (`> 50_000`): `max_filtered_p95_ms=90`, `max_filtered_overhead_ratio=10.0`

Optional planner diagnostics for filtered benchmarks:
- Enable with `DRIFT_FILTER_PLANNER_DIAGNOSTICS=1`.
- `bench_rw` will print planner decision ratios per probed bucket and include these summary JSON fields:
  - `filtered_candidate_fanout` (candidate / pre-filter routable live IDs)
  - `filtered_post_prune_candidate_fanout` (candidate / post-prune live IDs)
  - `filtered_estimated_scan_ratio` (scanned / pre-filter routable live IDs)
  - `filtered_post_prune_estimated_scan_ratio` (scanned / post-prune live IDs)
  - `filtered_scan_accounting_fallback_query_count` (queries where scanned/live accounting used fallback estimation)
  - `filtered_scan_accounting_fallback_query_ratio` (fallback query count / filtered query count)
  - `filtered_prefilter_routable_live_ids_avg` (full routable universe snapshot per query)
  - `filtered_estimated_global_scan_ratio` (same denominator as `filtered_estimated_scan_ratio`; retained for compatibility)
  - `filtered_planner_produced_bucket_ratio`
  - `filtered_planner_applied_bucket_ratio`
  - `filtered_planner_gated_bucket_ratio`
  - `filtered_planner_probe_error_bucket_ratio`
  - `filtered_planner_empty_exact_bucket_ratio`
  - `filtered_planner_no_index_bucket_ratio`
  - `filtered_planner_range_stats_only_bucket_ratio`
  - `filtered_planner_other_absence_bucket_ratio`
  - `filtered_planner_catalog_eligible_query_ratio`
  - `filtered_planner_catalog_pruned_bucket_ratio`
  - `filtered_planner_catalog_complete_may_match_bucket_ratio`
  - `filtered_planner_catalog_incomplete_bucket_ratio`
  - `filtered_planner_catalog_stale_bucket_ratio`
  - `filtered_planner_catalog_missing_bucket_ratio`
  - `filtered_planner_diagnostics_enabled`

Locality experiment controls:
- `--tenant-assignment-mode`:
  - `round-robin` (default): `tenant_i = id % filter_cardinality`
  - `vector-bin`: tenant from quantized first vector dimension (weak geometry correlation probe)
  - `tenant-clustered`: vectors generated around tenant centroids (strong locality probe)
- `--tenant-cluster-noise`: jitter range for `tenant-clustered` generation.
- Filter predicate controls:
  - `--filtered-predicate-mode {tenant-exact|price-range}` (default `tenant-exact`)
  - `--filtered-range-window`: contiguous ID window size for `price-range` queries.
- `bench_rw` summary JSON also emits locality diagnostics:
  - `filtered_predicate_mode`
  - `filtered_range_window`
  - `tenant_assignment_mode`
  - `configured_filter_cardinality`
  - `effective_filter_cardinality`
  - `tenant_locality_bucket_count`
  - `tenant_locality_kv_entries_scanned`
  - `tenant_locality_kv_entries_skipped`
  - `tenant_locality_avg_distinct_tenants_per_bucket`
  - `tenant_locality_avg_dominant_tenant_share`
  - `tenant_locality_avg_tenant_bucket_coverage_ratio`

## Workspace layout

- `drift_core`: index, router, memtable, WAL, payload model, maintenance algorithms.
- `drift_storage`: unified `.driftu` format, payload encoding/indexing, disk cache.
- `drift_kv`: persistent `VectorID -> BucketID` mapping.
- `drift_server`: gRPC service, collection manager, janitor, recovery, benchmarks/sims.
- `drift_traits`: storage abstractions.

## Current status

- Server/janitor/recovery path is active by default in `drift_server`.
- Payload-preserving flush/promotion/split/scatter paths are wired.
- API includes payload schema management, typed payload insert, and field-filtered search with projection.
- Filter execution is currently post-vector candidate stage (planner/index pushdown is next phase).
