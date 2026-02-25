# Next Steps

Last updated: 2026-02-25

## Current Execution Phase
- Phase D payload/filter API is shipped; remaining Phase D gap is payload schema management API.
- Documentation surface is now synced for current API/runtime (`README.md`, `SYSTEM_VIEW.md`, `docs/API_SPEC.md`).

## Active Queue (Post-Item 15)
- [ ] 16. Finish remaining Phase D gap: payload schema management API.
  - Scope: create/update/validate payload field definitions.
  - File focus: `drift_server/proto/drift.proto` + server handlers.
- [ ] 17. Start Phase E performance work for filter-aware planning.
  - Goal: move from vector-first filtering toward selectivity-aware candidate planning and index pushdown.
- [ ] 18. Add targeted filtered-search benchmark + p95 regression guardrails.
  - Goal: track filtered query latency impact over time and catch regressions early.
- [ ] 19. Add remaining Phase B E2E durability test.
  - Goal: verify payload survives `flush -> promote -> recover` lifecycle.
- [ ] 20. Prepare follow-up commit after items 16-19 are green.

## Priority Queue
- [x] 1. Run full workspace regression once before commit.
  - Command: `cargo test --workspace`
- [x] 2. Extend unified header/footer with metric + schema hash fields.
  - Goal: align runtime wire contract with unified format spec.
  - Suggested command after changes: `cargo test -p drift_storage unified_header`
- [x] 3. Add row-group stats metadata for payload pruning.
  - Fields: `null_count`, `min`, `max`, `cardinality_hint`.
  - Suggested command: `cargo test -p drift_storage payload_stats`
- [x] 4. Encode exact postings as delta + bitpack wire format.
  - Goal: keep API semantics while improving index compactness.
  - Suggested command: `cargo test -p drift_storage exact_index`
- [x] 5. Drop backward-compat read path for legacy exact-index codec.
  - Goal: fail fast on old exact-index codecs and require rewrite with current writer.
  - Suggested command: `cargo test -p drift_storage exact_index`
- [x] 6. Add malformed input tests for new codec decoders.
  - Cases: invalid bit width, truncated payloads, out-of-range dictionary IDs, trailing bytes.
  - Suggested command: `cargo test -p drift_storage malformed_payload`
- [x] 7. Add append edge-case tests.
  - Cases: null-heavy payload rows, mixed dictionary cardinality growth, schema optionality checks.
  - Suggested command: `cargo test -p drift_storage unified_local_append`
- [x] 8. Update storage docs/spec to include unified contract and codec encodings.
  - Files: `FORMAT.md`, `PAYLOAD_INDEX_V1_SPEC.md`.
- [x] 9. Capture compression and read-latency impact for payload-heavy datasets.
  - Suggested command pattern: add and run targeted bench/test harness in `drift_storage`.
- [x] 10. Wire payload rows through staging append and promotion write path.
  - Completed:
    - payload-aware local staging append/read APIs
    - promotion merge + tombstone purge preserving payload rows
    - promotion remote write switched to `write_remote_bucket_unified_flat_with_payload(...)`
- [x] 11. Carry payload rows from ingest/partition into staging batches.
  - Completed:
    - Added core payload model module in `drift_core` (`payload.rs`).
    - Extended `MemTable`, `VectorIndex`, and partitioner with payload-aware ingest/flush APIs.
    - Janitor flush now converts core payload rows/schema to unified payload rows/schema before staging append.
    - Server insert/train surfaces now call payload-aware index methods (currently passing `None` until protobuf payload fields land).
- [x] 12. Extend manifest schema for payload/index metadata and recovery diagnostics.
  - Completed:
    - `manifest.proto` bucket fields (`has_payload_columns`, `has_exact_index`, `has_payload_stats`, `payload_schema_hash`)
    - persistence write result exports unified header-derived payload/index metadata
    - promotion manifest update writes payload/index metadata atomically with run metadata
    - recovery compares manifest metadata vs remote unified header and reports diagnostics
    - health/metrics surface new recovery diagnostics counters
- [x] 13. Preserve payload rows during split/merge/scatter rewrite flows.
  - File focus: `drift_server/src/janitor.rs` split/merge/scatter paths.
  - Completed:
    - Added janitor payload-aware bucket read helper that merges Local/Remote/Tiered/Promoting components in `BucketManager::fetch_bucket` order.
    - Split path now preserves payload schema/rows when rewriting child files and re-inserting loopback defectors.
    - Scatter-merge path now preserves payload schema/rows while rewriting target buckets and validates schema/row consistency.
    - Added payload preservation tests:
      - `janitor_split_test::test_janitor_split_preserves_payload_rows`
      - `janitor_scatter_merge_test::test_scatter_merge_preserves_payload_rows`
- [x] 14. Draft API changes for payload insert + filter search in protobuf.
  - File: `drift_server/proto/drift.proto`.
  - Goal: add payload-bearing insert fields and filter clauses.
  - Completed:
    - Added protobuf payload value/row model (`PayloadValue`, `PayloadRow`, `PayloadLobRef`).
    - Added payload-bearing insert contracts:
      - `InsertRequest.payload`
      - `InsertBatchRequest.payload_rows`
    - Added filter clauses to `SearchRequest`:
      - `FieldFilter` with `exact`, `any_of`, and `range`
      - optional `payload_projection_fields`
    - Added `SearchResult.payload` projection field.
    - Wired server-side filter execution (exact/any_of/range) with payload row lookup from L1 and L0 fallback.
    - Added integration coverage:
      - `server_integration_tests::test_search_field_filters_exact_anyof_range_and_projection`
- [x] 15. Prepare commit with focused message once above items are green.
  - Suggested command sequence:
    - `git status --short`
    - `git add drift_core/src/{payload.rs,memtable.rs,index.rs,partitioner.rs,partitioner_tests.rs,index_tests.rs} drift_server/src/{janitor.rs,server.rs,janitor_tests.rs,local_staging_test.rs,server_integration_tests.rs,manager.rs} drift_server/proto/drift.proto docs/{NEXT.md,SESSION_LOG.md} TODO.md`
    - `git commit -m "feat(api): add payload inserts and field-filtered search contract"`

## If Starting a New Session
- Open `docs/CONTEXT.md`, `docs/NEXT.md`, and the active implementation files.
- Prompt pattern: "Read `@docs/CONTEXT.md` and `@docs/NEXT.md`. Execute top unchecked step."
