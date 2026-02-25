# System View (v3)

This document describes the current end-to-end runtime flow for vectors and payloads in Drift v3.
The core design unit is a logical bucket that may span local and remote unified files.

## 0) API contract (current)

- Collections are created explicitly with `CreateCollection` (dim + metric).
- Inserts can carry typed payload rows (`PayloadRow`).
- Search supports:
  - vector nearest-neighbor retrieval,
  - payload field filters (`exact`, `any_of`, `range`),
  - payload field projection in results.

All payload data is stored in the same `.driftu` unified files as vectors.

## 1) Logical bucket model

`BucketManager` exposes one logical bucket state per bucket ID:

- `Local`: active mutable staging file.
- `Remote`: immutable remote base file.
- `Tiered`: remote base + local delta.
- `Promoting`: transient state while frozen local data is being merged into remote.

Search and maintenance read through this abstraction, not directly from one file path.

## 2) Ingest path (WAL -> L0)

For each insert or insert-batch row:

1. Entry is written to WAL.
2. Vector is inserted into L0 MemTable.
3. Optional payload row is inserted alongside the vector.
4. L0 tombstones track deletes with copy-on-write snapshots for lock-light reads.

Payload schema is inferred per insert-batch request:

- field logical type must be consistent for each `field_id`,
- null-only fields are rejected,
- nullable is inferred from null/missing observations.

## 3) Flush path (L0 -> local unified staging)

When MemTable rotates/freeze threshold is hit:

1. Janitor partitions rows by router centroid.
2. Each partition appends a row group into bucket-local `.driftu` staging file.
3. Payload schema/rows are written with vectors into the same unified row groups.
4. KV mapping (`VectorID -> BucketID`) is updated.
5. Router centroid/count stats are updated.
6. Manifest metadata is atomically updated.

Result: vector and payload data move together from L0 to L1.

## 4) Promotion path (local -> remote)

When a local staging file crosses promotion threshold:

1. Bucket lock is acquired.
2. Active local file is rotated to frozen staging.
3. Frozen local + remote base (if any) are merged.
4. Tombstones are applied during merge.
5. New remote `.driftu` object is written.
6. Manifest remote metadata is updated atomically, including payload/index flags and schema hash.
7. Bucket transitions to `Tiered`; old files are queued for reaper cleanup.

Result: remote holds compacted base, local continues as mutable delta.

## 5) Search path (vector + payload)

Search execution is two-stage today:

1. Vector stage:
   - Router selects buckets from drift params (`target_confidence`, `lambda`, `tau`).
   - Index returns vector candidates by score.
2. Payload stage (only when filters/projection requested):
   - candidate `k` is expanded (`k * 8`, capped at 8192),
   - payload rows are loaded for candidate IDs from L1 buckets, with L0 fallback,
   - filters are evaluated in server layer,
   - projection fields are attached to output rows.

If no filters and no projection are requested, search returns vector results without payload lookup.

## 6) Filter semantics (server implementation)

Filters are combined with logical AND:

- `exact`: field value must equal target value.
- `any_of`: field value must match one of provided values.
- `range`: lower/upper bounds; inclusivity defaults to `true` if omitted.

Comparison behavior:

- Numeric comparison is supported across `int64`, `float32`, `float64`, and `timestamp_micros`.
- `keyword` and `text` compare lexicographically.
- Missing field behaves as `null` for `exact`/`any_of`.
- `range` fails when field is missing, null, or non-comparable type.

Projection behavior:

- Empty `payload_projection_fields` => `SearchResult.payload` is omitted.
- Non-empty projection => payload object includes only requested fields that exist.

## 7) Recovery and startup guards

On startup:

1. Manifest is loaded.
2. Bucket classes/paths are re-registered.
3. WAL replay rehydrates recent writes/deletes.
4. Persisted tombstones are hydrated.
5. Recovery guard validates remote fingerprint and payload/index metadata consistency.
6. Health/metrics expose recovery guard diagnostics.

## 8) Current trade-offs

- Vector retrieval happens before payload filtering, so selective filters can still over-fetch.
- Candidate expansion (`k * 8`) improves filtered recall but adds read/decode work.
- Dedicated schema-management RPCs are not yet shipped; schema is currently inferred from writes.

This is the current production mental model and should be treated as the source of truth
for v3 runtime behavior.
