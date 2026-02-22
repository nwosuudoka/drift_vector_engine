# Payload and Secondary Index V1 Spec

Status: Draft  
Owner: Drift V3 Track  
Scope: Metadata/text payload storage and secondary indexes aligned with current bucket/run architecture.

## 1. Goals

1. Add non-vector fields (metadata and text) without slowing current insert/query hot paths.
2. Keep object storage (S3-compatible) as source of truth.
3. Preserve immutable run semantics (`bucket_{id}_{run_id}.*`).
4. Support first filter types:
- exact match (`=`, `IN`) for keyword/string/bool/enum
- numeric/date range (`<`, `<=`, `>`, `>=`, `BETWEEN`)

## 2. Non-Goals (V1)

1. Full-text ranking (BM25/ANN fusion) is out of scope.
2. Distributed/sharded query planner is out of scope.
3. Replacing vector routing/scoring is out of scope.

## 3. Design Principles

1. S3/object storage is authoritative; local NVMe cache is a derived acceleration layer.
2. Payload/index artifacts are immutable per `run_id`.
3. Query path remains efficient by pruning early (stats/indexes) and late materialization.
4. Rebuildability: all payload/index artifacts can be reconstructed from source objects.

## 4. Logical Data Model

Per collection:
1. Vector record:
- `id: u64`
- `vector: [f32; dim]`
2. Payload record:
- `id: u64` (same identity as vector record)
- `fields: map<field_name, typed_value>`

Supported V1 field types:
1. `BOOL`
2. `INT64`
3. `FLOAT64`
4. `TIMESTAMP_MICROS` (stored as `INT64`)
5. `KEYWORD` (UTF-8 exact-match string)
6. `TEXT` (stored raw; no ranking in V1)

## 5. Storage Artifacts

For each bucket run:
1. Vector object (existing):
- `bucket_{bucket_id}_{run_id}.drift`
2. Payload sidecar (new):
- `bucket_{bucket_id}_{run_id}.payload`
3. Secondary index sidecars (new, optional per field):
- exact: `bucket_{bucket_id}_{run_id}.idx_exact_{field_id}`
- range: `bucket_{bucket_id}_{run_id}.idx_range_{field_id}`

All sidecars are written/uploaded atomically with run promotion/split/merge.

## 6. Payload File Format (V1)

## 6.1 Header (fixed, 128 bytes)

Fields:
1. `magic` (`u64`) = `DRIFTPL1`
2. `version` (`u16`) = `1`
3. `flags` (`u16`)
4. `schema_hash` (`u64`) (collection payload schema fingerprint)
5. `run_id` (`[u8; 16]`) (same logical run identity as vector object)
6. offsets/lengths for:
- schema block
- row-group directory
- footer

## 6.2 Schema Block

Repeated field descriptors:
1. `field_id: u16`
2. `name: string`
3. `type: enum`
4. `nullable: bool`
5. `index_mode: NONE | EXACT | RANGE | EXACT_RANGE`
6. `codec: enum`

## 6.3 Row-Group Blocks

Payload row groups are aligned 1:1 with vector row groups by `row_group_id` and `vector_count`.

Each row group stores:
1. `id_block`:
- ordered `u64` IDs for deterministic join/validation
2. column chunks:
- one chunk per field
- chunk format depends on field type/codec
3. per-column stats:
- null count
- min/max (for range-capable types)
- optional dictionary cardinality

## 6.4 Footer (fixed, 128 bytes)

Contains:
1. total row-group count
2. directory offsets/lengths
3. file checksum/magic/version guards

## 7. Index Sidecar Formats (V1)

## 7.1 Exact Index (`idx_exact`)

Logical structure:
1. dictionary of normalized field values
2. postings list per value:
- sorted `u64` vector IDs (or delta-encoded IDs)
3. optional row-group postings to accelerate pruning

Use cases:
1. `field = value`
2. `field IN (...)`

## 7.2 Range Index (`idx_range`)

Logical structure:
1. sorted `(value, id)` entries in blocks
2. sparse block index (min/max value -> block offset)
3. optional row-group min/max mirrors

Use cases:
1. `field BETWEEN a AND b`
2. `field >= a`, `field < b`

## 8. Manifest Extensions

Extend `manifest.Bucket` with sidecar metadata:
1. `payload_path: string`
2. `payload_fingerprint: string`
3. repeated `secondary_indexes`:
- `field_id`
- `index_kind` (`EXACT`/`RANGE`)
- `object_path`
- `object_fingerprint`
- `row_count`

Requirement:
Manifest updates for vector object + payload + indexes are atomic in one version bump.

## 9. Write/Maintenance Path

## 9.1 Flush

1. MemTable flush partitions vectors (existing).
2. Payload buffers for same IDs are flushed into payload row groups aligned to vector row groups.
3. KV mapping remains `id -> bucket` accelerator.

## 9.2 Promotion

1. Merge local + remote vectors (existing behavior).
2. Merge corresponding payload sidecars by ID.
3. Rebuild exact/range indexes for resulting run.
4. Upload vector/payload/index artifacts.
5. Commit manifest atomically.

## 9.3 Split/Merge (Janitor)

1. Split/merge operations must rewrite both vector and payload artifacts.
2. Secondary indexes are rebuilt for new child/target runs.
3. Old run artifacts are reaped after manifest commit and grace period.

## 10. Query Planner (V1)

Planner modes:
1. vector-only:
- unchanged current path
2. filter-first:
- use exact/range indexes to produce candidate IDs
- route candidates by KV (`id -> bucket`)
- run vector refine on reduced candidate set
3. vector-first + filter-late:
- current vector path first
- apply payload predicate on top candidates

Heuristic inputs:
1. estimated selectivity from index stats
2. candidate count budget
3. k / oversample factor

## 11. Recovery and Consistency

1. On startup, validate that manifest-listed payload/index objects exist.
2. If missing/corrupt:
- mark sidecar state degraded
- fallback behavior:
  - vector-only queries still available
  - filtered queries return clear error or fallback to scan (configurable)
3. Rebuild tool can regenerate sidecars from source run objects.

## 12. Efficiency Notes

1. No payload decode on vector-only queries.
2. Late materialization: decode payload only for rows that survive filter/vector pruning.
3. Row-group stats (min/max/null) provide cheap pre-pruning before index or data reads.
4. Immutable sidecars maximize cacheability and simplify invalidation.

## 13. API Evolution (Post-Spec)

gRPC additions (future):
1. collection payload schema definition/update APIs
2. insert/update APIs carrying payload fields
3. search request filter clause DSL

Backward compatibility:
1. existing APIs remain valid for vector-only collections.

## 14. Rollout Plan

Phase 1:
1. Manifest schema extensions
2. Payload sidecar writer/reader (no planner usage yet)

Phase 2:
1. Exact index implementation
2. Query planner exact-filter path

Phase 3:
1. Range index implementation
2. Planner selectivity heuristics and tuning

Phase 4:
1. Operational tooling (rebuild, validation, metrics)
2. Production hardening and load tests

## 15. Acceptance Criteria

1. Vector-only latency regression <= 5% p95.
2. Exact/range filter queries show measurable scan reduction for selective predicates.
3. Manifest commit stays atomic across vector + payload + index artifacts.
4. Recovery from missing/stale sidecars is deterministic and observable.
