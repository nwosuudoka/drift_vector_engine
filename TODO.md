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
- [ ] Remove stale V2 naming from remaining logs/tests/docs.

## Next Major Step: Payload and Secondary Index Expansion

- [ ] Sidecar payload format for metadata and full text values.
- [ ] Query planner for hybrid vector + filter execution.
- [ ] Pluggable index trait for metadata/text/range fields.
- [ ] First production secondary index implementation (exact match + range baseline).

## Future: Distributed Drift Cluster

- [ ] Consensus layer for shard ownership (OpenRaft or etcd-backed).
- [ ] Remote WAL abstraction (Kafka/Redpanda/S3-append compatible).
- [ ] Stateless workers loading collections from object storage.
- [ ] Gateway/router node for request routing by collection and vector id.
