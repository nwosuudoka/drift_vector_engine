# Drift V3 Strategy

This document captures the V3 direction aligned with current implementation and team decisions.

## Decisions (aligned with pushback)

1. Row-group tombstones are not serialized in V3 hot blobs.
- Rationale: deletes are already applied before persistence (promotion/merge path) and query-time tombstones are tracked separately in memory + persisted tombstone files.
- Result: V3 hot layout is now `[ids][sq8 codes]`.
- Reader behavior: strict V3 layout is enforced (no legacy trailer bytes accepted).

2. L0 remains MemTable + parallel scans (no HNSW in hot ingest path).
- Rationale: high-throughput inserts are prioritized over graph maintenance cost.
- Next improvement path: SIMD/LUT acceleration in L0 scan, adaptive chunking, and optional lightweight coarse routing for very large L0.

3. Validation is now first-class in file open path.
- Footer magic/version support: V3 only.
- Structural checks: index bounds, row-group count consistency, quantizer bounds, bloom bounds, row-group range layout validation.
- Header validation: strict magic/version pair validation.

4. NVMe cache layer is introduced for remote reads.
- Read path now uses disk-only cache for `DiskManager::read_at`:
  - Local NVMe full-object files: hashed files under the configured cache directory.
  - Reads are served by local range slicing from the cached object file.
  - Remote/object storage is used as fallback on miss.
- Runtime has singleflight protection so concurrent misses for the same object wait on one downloader.
- Runtime tracks in-memory cache metadata (`size`, `last-access`, `access-count`, fingerprint) and rebuilds it by scanning cache files on restart.
- Eviction is S3FIFO-inspired for disk objects: one-hit entries are evicted before recurring entries when budgets are exceeded.
- No in-memory byte payload cache is used in the read path.
- Intended usage: remote/object-store backed reads (S3-like operators).

## Format Versioning

- V3 magic/version are now the write default.
- Reader accepts V3 only.
- Non-V3 files are rejected by strict header/footer validation.

## Current V3 Foundation Delivered

- V3 default write path in format header/footer.
- V3-only read compatibility (no legacy V2 fallback).
- Stronger reader-side structural validation.
- V3 hot layout cleanup (no tombstone placeholder bytes).
- Remote read cache with local NVMe full-object files (no RAM payload cache).
- Delete/cleanup calls routed through `CleanupApi` and `PersistenceManager::delete_file` for consistent remote cache invalidation.
- Cache metrics are exposed via `DiskManager::global_nvme_cache_metrics()`.
- Provider-agnostic object fingerprints are stored in manifest bucket metadata.
- Recovery fingerprint guard metrics are exposed in health responses and optional Prometheus `/metrics`.

## Cache Configuration

Set the following env vars to enable local NVMe object-file caching for remote operators:

- `DRIFT_NVME_CACHE_DIR` (required to enable)
- `DRIFT_NVME_CACHE_MAX_BYTES` (optional global disk budget by bytes)
- `DRIFT_NVME_CACHE_MAX_FILES` (optional global disk budget by object-file count)
- `DRIFT_NVME_CACHE_MAX_FILE_BYTES` (optional; if set, larger objects bypass full-file cache and use direct range reads)
- `DRIFT_NVME_FINGERPRINT_VERIFY_INTERVAL_MS` (optional; how often cached objects are re-verified against remote metadata)

If `DRIFT_NVME_CACHE_DIR` is not set, behavior stays unchanged (no local object-file cache).

## Cache Invalidation (Current)

- On remote object delete via `PersistenceManager::delete_file`, the matching NVMe cache directory for that object is invalidated.
- Janitor/Reaper cleanup paths route delete calls through `CleanupApi`, which delegates remote deletion to `PersistenceManager`.
- In normal segment lifecycle, file names are immutable (`bucket_{id}_{run_id}.drift`) so stale-read risk is low because replaced state uses a new object key.
- If any workflow reuses the same object key for overwrite-in-place, fingerprint mismatch detection invalidates stale local cache before refresh.
- Disk cache enforces optional byte/count budgets when configured.

## Migration Plan (incremental)

1. V3 rollout (now)
- Write/read V3 only.
- No legacy V2 compatibility path is maintained.

2. Hardening
- Add corruption tests for each validation check (bad offsets, bad counts, overlapping ranges).
- Add strict V3 lifecycle integration tests in promotion/search/recovery loops.

3. Query/runtime upgrades
- Use bloom as actual scan pre-filter in query path.
- Add optional per-row-group checksum verification mode during refine/debug.
- Expand metrics export beyond health payload:
  - Implemented: optional Prometheus `/metrics` endpoint for NVMe cache + recovery guard counters.
  - Next: OpenTelemetry integration and richer labels/histograms.
- Tune eviction policy for workload-specific hot/cold behavior.
- Add manifest/runtime cross-check path for fingerprint mismatch observability.

4. Payload/index expansion (next major step)
- Add sidecar payload store and secondary indexes (metadata/text/range).
- Introduce query planner for hybrid search (vector + filters + text).
