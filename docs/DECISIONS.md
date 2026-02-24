# Technical Decisions

Last updated: 2026-02-23

## 2026-02-23: Row-group payload stats stored as first-class unified blocks
- Decision:
  - Add `PayloadStats` block type with one block per vector chunk row range.
  - Track per-field `null_count`, `min`, `max`, and `cardinality_hint`.
  - Gate presence with a dedicated header/footer flag bit.
- Rationale:
  - Make pruning metadata deterministic and colocated with payload columns/index blocks.
  - Keep reader startup checks strict on metadata flags and force explicit rewrites when formats change.
- Impact:
  - New files with payload rows emit stats blocks by default.
  - Reader exposes `read_payload_stats()` and validates field coverage/type/range consistency.
  - Append only emits stats on files that already advertise stats support.

## 2026-02-23: Exact postings switched to delta+bitpack wire codec
- Decision:
  - Write exact index postings with `DictPostingsBitpack` codec and delta+bitpack payload.
  - Reject legacy `DictPostings` exact-index blocks on read.
- Rationale:
  - Reduce exact-index block size while preserving exact filter semantics.
  - Keep mixed-version safety while the fleet transitions.
- Impact:
  - New files use compact postings encoding by default.
  - Older runs must be rewritten with current writer before exact filtering.

## 2026-02-23: Header/Footer now carry metric and payload schema hash
- Decision:
  - Add `metric` and `payload_schema_hash` fields to both unified header and footer.
  - Validate header/footer parity on read.
- Rationale:
  - Detect metadata drift/corruption early at open time.
  - Prepare for metric-aware and schema-consistency-aware query planning/recovery.
- Impact:
  - Reader rejects metric/hash mismatches between header and footer.
  - Append path preserves metric and schema hash metadata.

## 2026-02-23: Unified `.driftu` is canonical (no payload/index sidecars)
- Decision:
  - Standardize on one run object (`.driftu`) containing vectors, payload, and indexes.
  - Treat sidecar-based payload/index design as historical and non-canonical.
- Rationale:
  - Keep metadata publication atomic via single manifest object pointer swap.
  - Avoid sidecar consistency coordination and simplify recovery validation.
- Impact:
  - `FORMAT.md` and `PAYLOAD_INDEX_V1_SPEC.md` must remain aligned to unified block model.
  - Next encoding work is in unified header/footer metadata and index block evolution.

## 2026-02-23: Payload codec expansion in unified storage
- Decision:
  - Introduce `ForBitpack`, `AlpRd`, and `DictBitpack` codecs for payload columns.
- Rationale:
  - Improve storage efficiency and align decoding with type-specific encoding.
- Impact:
  - Reader and writer must stay codec-synchronized.
  - Tests must validate codec matrix roundtrip and malformed inputs.

## 2026-02-23: Append semantics for payload-bearing files
- Decision:
  - Permit append with payload rows when schema is consistent.
  - Reject append attempts that violate payload/schema invariants.
- Rationale:
  - Preserve data integrity and exact-index correctness across chunk appends.
- Impact:
  - Append API callers must pass payload rows for payload-enabled files.

## 2026-02-23: Exact index behavior across multiple blocks
- Decision:
  - Merge exact index entries for a field across blocks, sort/dedup postings.
- Rationale:
  - Appended chunks naturally produce multiple blocks; query path needs unified view.
- Impact:
  - Reader does extra merge work during exact-index reads.
  - Integrity checks enforce logical type consistency between blocks.
