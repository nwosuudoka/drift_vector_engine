# Global Master Plan: Drift Vector Engine (V3)

## Recently Completed

- [x] V3-only storage format enforcement in `drift_storage` (legacy V2 read path removed).
- [x] Hot row-group layout simplified to `[ids][sq8_codes]` with no serialized tombstone trailer bytes.
- [x] Reader validation hardening:
  - [x] Header/footer strict magic+version checks.
  - [x] Row-group count/offset/range consistency checks.
  - [x] Quantizer and bloom range bounds checks.
  - [x] Corruption matrix tests for malformed files.
- [x] Metric strategy and collection contract:
  - [x] Metric chosen at collection creation.
  - [x] Metric mismatch rejected on reopen/use.
  - [x] Spherical K-Means path added for cosine workflows.
- [x] NVMe full-object cache for remote reads:
  - [x] Cache full object file on local disk.
  - [x] Serve request ranges by local slicing.
  - [x] Singleflight download guard for concurrent misses.
  - [x] S3FIFO-inspired eviction with count and byte budgets.
  - [x] Fingerprint verification/invalidation path.
  - [x] Runtime metadata recovery after restart.
- [x] Unified delete/cleanup path through `CleanupApi` and `PersistenceManager::delete_file`.
- [x] Remote delete invalidates matching NVMe cache entries.
- [x] Health endpoint now exports NVMe cache metrics payload.
- [x] Chaos durability test hardening:
  - [x] Health-gated startup/restart checks.
  - [x] Epoch-specific recovery assertions.
  - [x] Bounded retry verification loop.
  - [x] Stale binary prevention in test spawn flow.

## In Progress

- [x] Startup guard: manifest fingerprint vs local cache metadata cross-check.
- [x] Configurable startup policy on fingerprint mismatches (invalidate-and-continue vs fail-fast).
- [x] Metrics export for runtime observability:
  - [x] Health RPC includes recovery-guard counters.
  - [x] Optional Prometheus `/metrics` exporter for NVMe cache + recovery guard counters.
- [x] Remove stale V2 naming from remaining logs/tests/docs.
- [x] KV durability hardening:
  - [x] Janitor periodic + lifecycle `kv.sync()` policy.
  - [x] Startup KV validation with rebuild on missing/stale mappings.

## Immediate Next Execution Steps (Post-Item 15)

- [ ] 1. Finish remaining Phase D gap: payload schema management API (create/update/validate field definitions).
- [ ] 2. Start Phase E performance work: move from vector-first filtering toward filter-aware planning and index pushdown.
- [ ] 3. Add targeted filtered-search benchmarks and p95 guardrails to track regressions.
- [ ] 4. Add remaining Phase B E2E durability test: payload survives flush -> promote -> recover.

## Next Major Step: Unified Payload and Secondary Index Expansion

### Phase A: Storage Wire Contract Hardening (`drift_storage`)

- [x] Extend unified header/footer with metric + payload schema hash.
- [x] Add row-group stats metadata (`null_count`, `min`, `max`, `cardinality_hint`) for payload pruning.
- [x] Switch exact-index postings from raw vectors to delta + bitpacked wire encoding.
- [ ] Define and implement range-index block format for numeric/time payload fields.
- [x] Add malformed-input test matrix for all new codec/index paths.

### Phase B: Data Path Wiring for Payload Preservation (`drift_core` + `drift_server`)

- [x] Extend ingest/partition structures to carry payload rows along with vectors.
- [x] Wire payload rows through staging append path (not schema-only).
- [x] Preserve payload rows during promotion merge/rewrite (local + remote + tombstone purge).
- [x] Preserve payload rows during split/merge/scatter rewrite flows.
- [ ] Add end-to-end tests that verify payload survives flush -> promote -> recover.

### Phase C: Manifest and Recovery Metadata (`manifest.proto` + server recovery)

- [x] Extend manifest bucket metadata for payload/index capabilities and schema hash consistency checks.
- [x] Record payload/index state atomically with run metadata updates.
- [x] Validate payload/index metadata during startup recovery and expose degraded-state diagnostics.

### Phase D: API and Query Surface (`drift.proto` + server handlers)

- [ ] Add payload schema management API (create/update/validate field definitions).
- [x] Add insert/insert-batch payload fields in protobuf API.
- [x] Add filter DSL to search request (`exact`, `in`, `range`, boolean composition baseline).
- [x] Add optional payload projection in search response (late materialization path).
- [x] Keep vector-only API behavior fully backward compatible.

### Phase E: Planner and Execution

- [ ] Implement filter-first vs vector-first planning heuristics using index/stats selectivity.
- [ ] Integrate exact-index candidate pruning into query execution.
- [ ] Integrate range-index candidate pruning and fallback scan behavior.
- [ ] Add instrumentation for candidate reduction, filter latency, and decode costs.
- [ ] Validate p95 latency/regression and selective-filter win targets.

## Deferred (Post-Expansion Hardening)

- [ ] Cache integrity verification off query path:
  - [ ] Keep query read path free of per-request checksum/fingerprint verification overhead.
  - [ ] Add background cache scrubber to sample cached objects and invalidate mismatches asynchronously.
  - [ ] Keep startup/recovery fingerprint guard as the primary online integrity gate.

## Future: Distributed Drift Cluster

- [ ] Consensus layer for shard ownership (OpenRaft or etcd-backed).
- [ ] Remote WAL abstraction (Kafka/Redpanda/S3-append compatible).
- [ ] Stateless workers loading collections from object storage.
- [ ] Gateway/router node for request routing by collection and vector id.
