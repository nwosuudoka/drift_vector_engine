# Payload and Secondary Index V1 Spec (Unified `.driftu`)

Status: Active draft  
Owner: Drift V3 Track  
Scope: typed payload storage and secondary index behavior inside unified bucket files.

## 1. Canonical Scope

This spec assumes payload and index data live inside the same `.driftu` object as vectors.

1. No standalone `.payload` artifacts
2. No standalone `.idx_exact_*` or `.idx_range_*` artifacts
3. One immutable `.driftu` object per published run

`FORMAT.md` is the canonical wire-layout reference. This file defines payload/index semantics and planner expectations.

## 2. Goals

1. Add metadata/text payload fields without regressing vector-only hot paths.
2. Preserve immutable run semantics and manifest CAS publication.
3. Support exact-match filtering first, then range filtering.
4. Keep rebuildability: index blocks can be reconstructed from payload column blocks.

## 3. Non-Goals (V1)

1. Full-text ranking (BM25/hybrid lexical scoring)
2. Distributed query planner changes
3. Replacing existing vector routing/scoring behavior

## 4. Logical Data Model

Per record:

1. `id: u64`
2. `vector: [f32; dim]`
3. `payload: map<field_id, typed_value>`

Supported payload types in unified V1:

1. `Bool`
2. `Int64`
3. `Float32`
4. `Float64`
5. `TimestampMicros`
6. `Keyword`
7. `Text`
8. `Bytes`
9. `LobRef`

## 5. Unified Block Types Used for Payload/Indexes

1. `PayloadSchema`
2. `PayloadColumn`
3. `PayloadExactIndex`
4. Range index block type: planned, not active yet

All blocks are row-range scoped via `(row_start, row_count)` and validated against header/footer directory metadata.

## 6. Payload Schema Contract

Schema block stores a versioned list of field descriptors:

1. `field_id`
2. `name`
3. `logical_type`
4. `nullable`
5. `indexed`

Rules:

1. Multiple schema blocks in one file must decode identically.
2. Payload column chunks must reference known schema fields.
3. Non-nullable fields cannot materialize null rows.

## 7. Payload Column Contract

Each payload column chunk contains:

1. field identity/type
2. row range
3. codec
4. optional validity bitmap
5. encoded non-null data

Decode rules:

1. Descriptor codec and in-chunk codec must match.
2. Decoded value count must equal non-null row count.
3. Validity + data must reconstruct exactly `row_count` output values.
4. Unknown codec for a logical type is a hard error.

## 8. Encoding Policy by Type

Current policy in writer/reader implementation:

1. `Bool` -> `Bitset`
2. `Int64` -> `ForBitpack`
3. `TimestampMicros` -> `ForBitpack`
4. `Float32` -> `AlpRd`
5. `Float64` -> `AlpRd`
6. `Keyword` -> `DictBitpack`
7. `Text` -> `VarLen` or `DictBitpack` (smaller payload wins)
8. `Bytes` -> `VarLen` or `DictBitpack` (smaller payload wins)
9. `LobRef` -> `VarLen`

Profile split target:

1. Local append profile should remain low-latency and append-safe.
2. Remote rewrite profile can apply heavier compression policies.

## 9. Exact Index Contract (V1)

Exact index blocks provide value-to-ID postings for indexed fields.

Current wire payload:

1. dictionary values (`Vec<Vec<u8>>`)
2. postings lists encoded as delta + bitpack payloads
3. block codec marker: `DictPostingsBitpack`
4. legacy `DictPostings` runs are not supported and must be rewritten

Reader behavior:

1. Validate dictionary/postings length parity.
2. Merge same-field index blocks across row ranges.
3. Enforce logical-type consistency across merged blocks.
4. Sort and deduplicate merged postings.

Current implementation:

1. Dictionary values are stored as exact-key bytes.
2. Posting lists are reconstructed from bitpacked deltas at read time.

## 10. Range Index Contract (Planned)

Target for numeric/time fields:

1. sorted `(value, id)` entries
2. sparse min/max directory for block pruning
3. optional row-group min/max mirrors

APIs and planner paths for range filtering are deferred until this block format is implemented.

## 11. Query Planner Expectations

### 11.1 Current

1. vector-only path remains unchanged
2. exact filter API can use unified exact index blocks when present
3. payload columns are materialized on demand

### 11.2 Planned

1. filter-first mode for selective exact/range predicates
2. vector-first + filter-late mode for weakly selective predicates
3. selectivity heuristics from row-group stats and index metadata

## 12. Consistency and Recovery

1. `.driftu` object is uploaded before manifest CAS swap.
2. Manifest CAS is the publish boundary for vectors + payload + indexes together.
3. Startup/recovery validation checks header/footer/directory/block checksums.
4. Missing or corrupt payload/index blocks must fail deterministically with clear errors.

## 13. Rollout Phases

1. Phase A: payload schema + payload column blocks (implemented)
2. Phase B: exact index blocks + merge semantics (implemented baseline)
3. Phase C: range index block format + query path integration
4. Phase D: stats-driven planner and operational hardening

## 14. Acceptance Criteria

1. Vector-only p95 regression remains within budget.
2. Exact-filter queries show measurable candidate pruning.
3. Unified read/write paths remain append-safe and corruption-detecting.
4. Planner-facing payload/index metadata is deterministic across rewrite cycles.
