# Drift Unified Bucket Format (`.driftu`) - Draft

Status: Draft for discussion  
Owner: V3/VNext storage track  
Goal: One unified immutable object format for vectors + payload + secondary indexes.

## Execution Plan (Phase-by-Phase)

1. Phase 1 (now): Add payload schema metadata block to `.driftu` with read/write APIs.
2. Phase 2: Add typed payload column blocks (BOOL/INT64/FLOAT/TEXT/BYTES/LOB_REF) without indexes.
3. Phase 3: Add predicate planner and late materialization for payload filters.
4. Phase 4: Add `EXACT` side index blocks (dictionary + postings) and planner integration.
5. Phase 5: Add remote promotion compaction policies for payload/index rewrite and validation.

Current status:
1. Phase 1 is in progress in `drift_storage` (schema block type + reader/writer support).

## 1. What We Optimize For

1. Fast vector-first scans (SQ8 path stays hot).
2. High ingest throughput (append-only local flush path).
3. Safe copy-on-write split/merge/promotion (no in-place remote mutation).
4. Object-store source of truth with manifest CAS pointer swaps.
5. Typed payload support without sidecar coordination complexity.
6. Efficient handling of arbitrary large payloads (images/audio/video/docs/blobs).

## 2. Architecture Decision

Keep current architecture and evolve file format:

1. `WAL + MemTable + router + janitor` remains.
2. Local staging remains append-oriented.
3. Remote objects remain immutable per `run_id`.
4. Split/merge/compaction remain rewrite jobs (create new object(s), then manifest swap).

Do not mutate remote objects in place.

## 3. Core Strategy for Fast Encoding/Scanning

Two-level design:

1. Physical layout is vector-first contiguous superblocks.
2. Logical row groups are metadata ranges in footer (for pruning and parallel scheduling).
3. Keep large payload bytes off the hot vector scan path.

Implication:

1. Vector scan reads contiguous bytes (low IO overhead, SIMD-friendly).
2. Payload/index reads are selective and columnar.
3. Row-group-level stats and postings reduce unnecessary payload decoding.
4. Large object fetch is explicit and lazy (only when requested).

## 4. File Naming and Lifecycle

1. Local active: `bucket_<id>_active.driftu`
2. Local rotated snapshot: `bucket_<id>_staging_<uuid>.driftu`
3. Remote immutable: `bucket_<id>_<run_id>.driftu`

Lifecycle:

1. Flush appends to local active file.
2. Promotion/split/merge writes brand new object(s).
3. Manifest CAS publishes new object path/fingerprint.
4. Old objects are reaped asynchronously.

## 5. Physical File Layout (V1 draft)

```text
+-------------------------------+
| Header (fixed)                |
+-------------------------------+
| Block: Quantizer              |
+-------------------------------+
| Block: IDs                    |
+-------------------------------+
| Block: Vector Codes (SQ8)     |
+-------------------------------+
| Block: Optional Norms         |
+-------------------------------+
| Block: Payload Columns...     |
+-------------------------------+
| Block: LOB Pointer Columns... |
+-------------------------------+
| Block: Secondary Indexes...   |
+-------------------------------+
| Footer (directory + stats)    |
+-------------------------------+
```

### 5.1 Header (fixed)

Fields (conceptual):

1. magic/version
2. flags
3. dim
4. metric (`L2`, `COSINE`)
5. row_count
6. payload_schema_hash
7. created_at
8. footer_offset/footer_length
9. checksum mode

### 5.2 Block Framing

Every block is framed:

1. `block_type`
2. `codec`
3. `row_start`
4. `row_count`
5. `offset`
6. `compressed_len`
7. `raw_len`
8. `crc32`

This gives uniform validation and easier future extension.

### 5.3 Footer

Footer contains:

1. row-group directory (`row_start`, `row_count`, block refs)
2. payload field schema (typed, nullable, codec hints)
3. column stats per row group (min/max/null_count/cardinality_hint)
4. index directory (EXACT now, others later)
5. manifest consistency fields (schema hash, optional object fingerprint hint)
6. footer checksum + magic

## 6. Logical Row Group Model

Row groups are logical row ranges, not necessarily independent vector blobs.

Each row group tracks:

1. row range (`start`, `count`)
2. payload block references for that range
3. stats for predicate pruning
4. optional local postings pointers

Vector blocks may remain global contiguous blocks while row groups map ranges into them.

## 7. Type System and Codec Policy (V1)

### 7.1 Supported Logical Types

V1 payload types:

1. `BOOL`
2. `INT64` (canonical integer type; narrower ints normalize to this)
3. `FLOAT32`
4. `FLOAT64` (optional for precision-sensitive collections)
5. `TIMESTAMP_MICROS` (physical `INT64`)
6. `KEYWORD` (exact string)
7. `TEXT` (stored text, no ranking in V1)
8. `BYTES` (opaque binary payload)
9. `LOB_REF` (logical external large object reference)

Nullability for all nullable fields is represented via a dedicated validity bitmap block.

### 7.2 Vortex/FastLanes-Inspired Principles

1. Block-local encoding decisions (Vortex style): choose codecs per block/column chunk, not globally for the file.
2. Lane-friendly integer blocks (FastLanes style): fixed-size integer blocks for SIMD-friendly decode and predictable scans.
3. Keep encoded blocks self-describing: each block carries codec id + lengths + checksum.
4. Separate write profiles:
   - local flush profile favors low CPU overhead.
   - remote promotion profile favors compactness and scan efficiency.

### 7.3 Default Codec Matrix

| Logical Type | Local Flush (append path) | Remote Promotion (rewrite path) | Notes |
|---|---|---|---|
| `BOOL` | bitset | bitset or RLE-bitset | choose RLE when runs are long |
| `INT64` | FOR + bitpack | FOR + bitpack (or FastLanes block for low-card runs) | block size fixed for SIMD decode |
| `TIMESTAMP_MICROS` | FOR + bitpack | FOR + bitpack | same as `INT64` |
| `FLOAT32` | plain/fixed | ALP_RD (fallback plain/fixed) | remote rewrite pays CPU once |
| `FLOAT64` | plain/fixed | ALP_RD (fallback plain/fixed) | optional type in V1 |
| `KEYWORD` | dictionary + bitpacked ids | dictionary + bitpacked ids (optional zstd on dict payload) | supports `EXACT` index well |
| `TEXT` | plain + lz4 | plain + zstd | V1 stores/retrieves; no BM25 |
| `BYTES` | length-delimited + lz4 | length-delimited + zstd | no semantic indexing in V1 |
| `LOB_REF` | plain/fixed ref tuple | plain/fixed ref tuple | points to external blob/chunk object(s) |

### 7.4 Secondary Index Encoding (V1)

1. `EXACT` first:
   - value dictionary
   - postings lists of row ids (delta + bitpack)
2. `RANGE` later:
   - sorted `(value, row_id)` blocks
   - sparse min/max directory for pruning

### 7.5 Adaptive Fallback Rules

1. If dictionary cardinality is too high for a block, fallback from dictionary to plain compressed block.
2. If ALP_RD does not beat plain by a minimum ratio threshold, store plain/fixed block.
3. If integer blocks are highly irregular, stay on FOR + bitpack rather than forcing advanced lane encodings.

### 7.6 Arbitrary Large Input Strategy (images/audio/video/docs)

For big payloads, the unified bucket file stores references, not full raw bytes:

1. Inline only lightweight metadata in payload columns:
   - mime/content type
   - byte size
   - width/height/duration (if available)
   - hash/fingerprint
2. Store heavy bytes in external immutable blob objects:
   - content-addressed key (hash-based), or
   - run-scoped chunk object key
3. `LOB_REF` physical tuple:
   - `blob_key`
   - `offset`
   - `length`
   - `fingerprint` (optional)
4. Query/search path never reads blob bytes unless explicitly requested.

This preserves vector/filter latency while still supporting arbitrary binary inputs.

### 7.7 Why This Is Better for Big Data Inputs

1. Prevents read amplification in vector scans.
2. Avoids repeatedly rewriting huge media bytes during split/merge/compaction.
3. Keeps copy-on-write object churn bounded to vector/index data.
4. Enables deduplication via content-addressed blob keys.

## 8. Write Path Stages

### 8.1 Flush (MemTable -> local active file)

1. Partition by bucket.
2. Append new logical row group data.
3. Update local footer/checkpoint atomically.
4. Keep file readable after each append.

Local goal: write throughput and crash safety.

### 8.2 Promotion (local + remote -> new remote object)

1. Read remote run + rotated local snapshot.
2. Apply tombstones and dedupe.
3. Rewrite into one new immutable remote `.driftu`.
4. Build payload/index blocks during rewrite.
5. Upload object.
6. Commit manifest CAS to point to new run.

Remote goal: read efficiency and compactness.

### 8.3 Split/Merge

1. Compute output partitions.
2. Write child/merged objects as fresh `.driftu` files.
3. CAS update manifest topology and pointers.
4. Reap old objects asynchronously.

## 9. Read Path

1. Load header + footer first.
2. Plan scan using metric + row-group stats + available indexes.
3. For vector-first query:
   - scan SQ8 contiguous vector block
   - apply tombstones
   - refine candidates
4. For filtered query:
   - apply EXACT/RANGE pruning first
   - evaluate vector only on survivors
   - late materialize payload values

## 10. Consistency and Failure Semantics

1. Data object upload happens before manifest swap.
2. Manifest CAS is the publication boundary.
3. Crash before CAS: new object is orphan, safe to reap.
4. Crash after CAS: new object is authoritative.
5. Recovery validates:
   - manifest path/fingerprint
   - header/footer checksums
   - schema hash consistency

## 11. What We Are Explicitly Not Doing Now

1. No sidecar payload/index artifacts.
2. No in-place remote patching.
3. No query-path row-group checkpointing overhead.
4. No broad codec auto-tuning in first cut (start with stable defaults).

## 12. Implementation Milestones

1. Add `drift_storage::unified_format` structs + encoder/decoder + validators.
2. Add local append writer (`AppendLocal`) with checkpoint footer updates.
3. Add remote rewrite writer (`RewriteRemote`) for janitor promotion/split/merge.
4. Wire persistence manager to read/write `.driftu`.
5. Wire manifest bucket metadata to track format version/object fingerprint.
6. Add corruption/recovery/integration tests for flush/promotion/split/merge.

## 13. Open Decisions (for next discussion)

1. Target row-group row count / byte target defaults.
2. Norm block required vs optional for cosine.
3. Exact posting format (`u64 delta+bitpack` vs hybrid).
4. Whether to keep bloom in unified footer for cheap negative checks.
5. Final magic/version naming (`V3 unified` vs `V4` bump).
