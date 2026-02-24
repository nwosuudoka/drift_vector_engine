# Drift Unified Bucket Format (`.driftu`) - V1

Status: Active draft (canonical for implementation)  
Owner: V3/VNext storage track  
Goal: one immutable object format for vectors, payload, and secondary indexes.

## 1. Canonical Direction

This repository standardizes on a single unified file format:

1. One run object per bucket: `bucket_<id>_<run_id>.driftu`
2. No payload/index sidecar files in the unified path
3. Payload schema, payload columns, and secondary indexes are stored as typed blocks inside `.driftu`

If any older docs mention separate `.payload` or `.idx_*` sidecars, treat those as historical design notes.

## 2. Design Objectives

1. Keep vector-first scan path hot (`SQ8` block contiguous and cheap to read).
2. Support typed payload fields with strict validation and deterministic decode.
3. Keep local flush append-safe and crash-safe.
4. Keep remote promotion immutable and rewrite-based (no in-place mutation).
5. Allow per-type and per-block codec choices.
6. Preserve future extension for planner-driven filter pushdown and range indexes.

## 3. File Lifecycle

1. Local active file: `bucket_<id>_active.driftu`
2. Local rotated staging snapshot: `bucket_<id>_staging_<uuid>.driftu`
3. Remote immutable run file: `bucket_<id>_<run_id>.driftu`

Lifecycle rules:

1. Flush appends chunk-aligned blocks to local active files.
2. Promotion/split/merge rewrites new immutable `.driftu` object(s).
3. Manifest CAS swap publishes the new run object path/fingerprint.
4. Old objects are reaped asynchronously.

## 4. Physical Layout

```text
+-----------------------------------+
| Header (fixed 128 bytes)          |
+-----------------------------------+
| Payload Region (framed blocks)    |
|   - Quantizer                     |
|   - IDs                           |
|   - Vector Codes                  |
|   - Optional Payload Schema       |
|   - Optional Payload Columns      |
|   - Optional Secondary Indexes    |
+-----------------------------------+
| Block Directory                   |
+-----------------------------------+
| Footer (fixed 64 bytes)           |
+-----------------------------------+
```

Current fixed constants are in `drift_storage/src/unified_format.rs`.

## 5. Header, Footer, and Block Framing

### 5.1 Header (fixed)

Header includes:

1. magic/version/header_len
2. flags
3. dimension (`dim`)
4. total `row_count`
5. quantizer location metadata
6. block directory location/count
7. footer location/length
8. creation timestamp

### 5.2 Footer (fixed)

Footer includes:

1. magic/version/footer_len
2. flags (must match header)
3. total `row_count` (must match header)
4. block directory offset/count (must match header)
5. block directory CRC32

### 5.3 Block Descriptor

Each block descriptor contains:

1. `block_type`
2. `codec`
3. `row_start`
4. `row_count`
5. `offset`
6. `compressed_len`
7. `raw_len`
8. `crc32`

This descriptor applies uniformly to vectors, payload columns, and index blocks.

## 6. Logical Row-Group Model

1. A chunk is defined by `(row_start, row_count)` and includes quantizer + ids + vector-codes blocks.
2. Payload and index blocks reference the same row ranges.
3. Chunks must form contiguous row coverage from `0..row_count`.
4. File must remain readable after each successful append.

## 7. Payload Type System

Supported logical payload types:

1. `Bool`
2. `Int64`
3. `Float32`
4. `Float64`
5. `TimestampMicros`
6. `Keyword`
7. `Text`
8. `Bytes`
9. `LobRef`

Nullability is represented by per-column validity bitmaps.

## 8. Encoding Policy (Per Type)

Codec selection is per column-chunk and may differ between chunks.

### 8.1 Current Default Matrix

| Logical Type | Local Append Profile | Remote Rewrite Profile | Current Implementation |
|---|---|---|---|
| `Bool` | `Bitset` | `Bitset` (RLE candidate later) | Implemented |
| `Int64` | `ForBitpack` | `ForBitpack` (lane variants later) | Implemented |
| `TimestampMicros` | `ForBitpack` | `ForBitpack` | Implemented |
| `Float32` | `AlpRd` | `AlpRd` (plain fallback policy pending) | Implemented |
| `Float64` | `AlpRd` | `AlpRd` (plain fallback policy pending) | Implemented |
| `Keyword` | `DictBitpack` | `DictBitpack` | Implemented |
| `Text` | `VarLen` or `DictBitpack` by size | same policy for now | Implemented |
| `Bytes` | `VarLen` or `DictBitpack` by size | same policy for now | Implemented |
| `LobRef` | `VarLen` | `VarLen` | Implemented |

### 8.2 Rules

1. Codec is always stored in both block descriptor and column chunk.
2. Reader must reject codec/type mismatch.
3. Dictionary encodings must fail on invalid dictionary IDs.
4. Bitpacked encodings must fail on width/length mismatch.

## 9. Payload and Index Blocks

### 9.1 Payload Schema Block

Stores schema (`field_id`, name, logical type, nullability, indexed flag).

### 9.2 Payload Column Block

Stores one field chunk for one row range:

1. `field_id`
2. `logical_type`
3. `codec`
4. `row_start`
5. `row_count`
6. optional `validity`
7. encoded `data`

### 9.3 Exact Index Block

Current V1 exact index stores:

1. normalized value dictionary (`Vec<Vec<u8>>`)
2. postings encoded as delta + bitpack payloads
3. one block per indexed field per chunk as needed

Block codec:

1. `DictPostingsBitpack` only
2. older exact-index postings must be rewritten by current writer

### 9.4 Payload Stats Block

Stores one stats payload per chunk row range:

1. `row_start`
2. `row_count`
3. per-field stats entries:
   - `field_id`
   - `logical_type`
   - `null_count`
   - `min` (optional, absent for all-null chunk field)
   - `max` (optional, absent for all-null chunk field)
   - `cardinality_hint`

Reader validates:

1. one stats block per vector chunk when stats flag is set
2. field coverage matches schema
3. logical types and min/max ordering are consistent

### 9.5 Range Index

Range index is not yet implemented in `.driftu` V1 runtime. It remains planned next for numeric/time predicates.

## 10. Append and Rewrite Semantics

### 10.1 Local Append

1. Validate incoming shape (`ids`, vectors, `dim`, payload rows).
2. Validate schema compatibility with existing file schema.
3. Append chunk blocks.
4. Rebuild block directory and footer.
5. Rewrite header/footer metadata atomically.

### 10.2 Remote Rewrite (Promotion/Split/Merge)

1. Read source runs and apply tombstones/dedupe.
2. Re-emit unified blocks in a new immutable object.
3. Publish object by manifest CAS.

## 11. Read-Path Guarantees

1. Header/footer and block directory are validated before scan.
2. Block CRC32 is verified on read.
3. Payload schema flag/block consistency is enforced.
4. Payload columns decode only when requested.
5. Exact index blocks for the same field are mergeable across chunks.

## 12. Open Work

1. Header extensions for metric and schema hash.
2. Exact postings compressed wire format (delta + bitpack).
3. Range index block type and codec contract.
4. Stats-aware planner heuristics and pruning strategy wiring.
5. Explicit local-vs-remote codec policy object in writer path.

## 13. Implementation Milestones

1. Keep `drift_storage::unified_format` as single source of wire contracts.
2. Keep append writer and reader validation strict.
3. Add corruption matrix tests for each new codec/index addition.
4. Wire unified format into persistence/manifest flows end-to-end.
