# Drift V3 Strategy

This document captures the V3 direction aligned with current implementation and team decisions.

## Decisions (aligned with pushback)

1. Row-group tombstones are not serialized in V3 hot blobs.
- Rationale: deletes are already applied before persistence (promotion/merge path) and query-time tombstones are tracked separately in memory + persisted tombstone files.
- Result: V3 hot layout is now `[ids][sq8 codes]`.
- Backward compatibility: reader still accepts legacy V2 trailing bytes in hot blobs.

2. L0 remains MemTable + parallel scans (no HNSW in hot ingest path).
- Rationale: high-throughput inserts are prioritized over graph maintenance cost.
- Next improvement path: SIMD/LUT acceleration in L0 scan, adaptive chunking, and optional lightweight coarse routing for very large L0.

3. Validation is now first-class in file open path.
- Footer magic/version support: V2 and V3.
- Structural checks: index bounds, row-group count consistency, quantizer bounds, bloom bounds, row-group range layout validation.
- Header validation: strict magic/version pair validation.

4. NVMe cache layer is introduced for remote reads.
- Read path now uses disk-only cache for `DiskManager::read_at`:
  - Local NVMe full-object files: hashed files under the configured cache directory.
  - Reads are served by local range slicing from the cached object file.
  - Remote/object storage is used as fallback on miss.
- No in-memory byte payload cache is used in the read path.
- Intended usage: remote/object-store backed reads (S3-like operators).

## Format Versioning

- V3 magic/version are now the write default.
- Reader remains compatible with both V2 and V3 files.
- Existing V2 files are readable without migration.

## Current V3 Foundation Delivered

- V3 default write path in format header/footer.
- V2+V3 read compatibility.
- Stronger reader-side structural validation.
- V3 hot layout cleanup (no tombstone placeholder bytes).
- Remote read cache with local NVMe full-object files (no RAM payload cache).
- Delete/cleanup calls routed through `CleanupApi` and `PersistenceManager::delete_file` for consistent remote cache invalidation.

## Cache Configuration

Set the following env vars to enable local NVMe object-file caching for remote operators:

- `DRIFT_NVME_CACHE_DIR` (required to enable)
- `DRIFT_NVME_CACHE_MAX_FILE_BYTES` (optional; if set, larger objects bypass full-file cache and use direct range reads)

If `DRIFT_NVME_CACHE_DIR` is not set, behavior stays unchanged (no local object-file cache).

## Cache Invalidation (Current)

- On remote object delete via `PersistenceManager::delete_file`, the matching NVMe cache directory for that object is invalidated.
- Janitor/Reaper cleanup paths route delete calls through `CleanupApi`, which delegates remote deletion to `PersistenceManager`.
- In normal segment lifecycle, file names are immutable (`bucket_{id}_{run_id}.drift`) so stale-read risk is low because replaced state uses a new object key.
- If any workflow reuses the same object key for overwrite-in-place, explicit invalidation must happen on write/replace as well.
- Disk cache is currently best-effort and not yet budgeted by size.

## Migration Plan (incremental)

1. V3 rollout (now)
- Write V3, read V2/V3.
- Keep all existing segment files valid.

2. Hardening
- Add corruption tests for each validation check (bad offsets, bad counts, overlapping ranges).
- Add mixed V2/V3 integration tests in promotion/search/recovery loops.

3. Query/runtime upgrades
- Use bloom as actual scan pre-filter in query path.
- Add optional per-row-group checksum verification mode during refine/debug.
- Add cache metrics emission (disk cache hit/miss, remote fetch ratio).
- Add disk cache budget + sweeper (size-based eviction of stale object cache files).
- Add object-version guard (etag/version marker) for overwrite-in-place safety.

4. Payload/index expansion (next major step)
- Add sidecar payload store and secondary indexes (metadata/text/range).
- Introduce query planner for hybrid search (vector + filters + text).
