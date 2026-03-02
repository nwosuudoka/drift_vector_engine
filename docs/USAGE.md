# Drift Production API Usage Spec (v3)

Last updated: 2026-03-01

This guide is the release-facing usage spec for developers integrating with the current Drift API.
It focuses on what is production-effective now.

Source of truth for wire contract: `drift_server/proto/drift.proto`.

## 1) Release scope

This release is ready for:
- vector search with payload-bearing inserts
- payload schema management APIs
- filtered search (`exact`, `any_of`, `range`)
- payload projection in search responses

This release is not yet optimized for:
- highly selective range-heavy workloads at large tier

## 2) Stable RPC surface

Service: `drift.Drift`

RPCs:
- `Health`
- `CreateCollection`
- `CreatePayloadSchema`
- `UpdatePayloadSchema`
- `GetPayloadSchema`
- `ValidatePayload`
- `Insert`
- `InsertBatch`
- `Search`
- `Train`

## 3) Production bootstrap contract

Use this startup sequence per collection:

1. `CreateCollection`
2. `GetPayloadSchema`
3. If schema not found: `CreatePayloadSchema`
4. Optional validation gate: `ValidatePayload` on sampled or staged rows
5. Start `InsertBatch` traffic

Important:
- `UpdatePayloadSchema` requires an empty collection (no buffered or persisted vectors).
- Schema definitions are runtime state and should be treated as part of your deployment config.

## 4) Ingestion patterns for best results

Preferred write API:
- use `InsertBatch` over repeated `Insert`

Rules:
- keep `payload_rows` length exactly aligned with `vectors` length
- keep payload types consistent for each `field_id`
- use explicit schema registration instead of relying on request-time schema inference

Recommended schema design:
- mark frequently filtered fields as `indexed: true`
- use `PAYLOAD_LOGICAL_TYPE_KEYWORD` for tenant/category exact matches
- keep range fields typed as numeric/timestamp logical types

## 5) Query patterns for maximum benefit now

### Best current performance profile

Use:
- selective `exact` / non-null `any_of` filters on indexed fields
- small `payload_projection_fields` sets
- practical `k` values (avoid over-large `k` unless required)

Why:
- planner + candidate pushdown are strongest on indexed exact-style predicates.
- when filters or projection are present, search expands internal candidate count (`k * 8`, cap `8192`).

### Range filters (current state)

Range filters are fully supported and correct, but on large-tier workloads they can still be scan-heavy.

For now:
- combine range with selective exact tenant/category clauses when possible
- keep range windows narrow and bounded
- monitor filtered p95 and scan-ratio metrics in CI

## 6) Operational checklist

Before sending traffic:
- ensure `Health.ready == true`
- confirm collection/schema bootstrap sequence completed

During operation:
- track `Health` response diagnostics:
  - `nvme_cache.*`
  - `recovery_guard.*`
- run filtered benchmark guardrails in CI using `bench_rw`

Recommended benchmark mode for filter planning visibility:
- set `DRIFT_FILTER_PLANNER_DIAGNOSTICS=1` during benchmark runs
- capture `--summary-json-path` artifacts for release comparisons

## 7) Error handling contract

Map gRPC status classes as:
- `INVALID_ARGUMENT`: request shape/type/validation errors
- `NOT_FOUND`: missing resources
- `FAILED_PRECONDITION`: state preconditions not met
- `INTERNAL`: storage/engine/internal failures

Client policy:
- do not retry `INVALID_ARGUMENT` without request correction
- gate retries for `INTERNAL` with backoff and request idempotency controls

## 8) Known release limitations

- Payload schema registry is runtime collection state and should be explicitly managed by deploy/bootstrap automation.
- CLI convenience binary is vector-focused; payload/filter/projection production flows should use gRPC clients.
- Range-heavy filter optimization is the next patch-track item.

## 9) Minimal grpcurl golden path

Create collection:

```bash
grpcurl -plaintext \
  -import-path drift_server/proto \
  -proto drift.proto \
  -d '{
    "collectionName": "products",
    "dim": 3,
    "metric": "METRIC_TYPE_COSINE"
  }' \
  127.0.0.1:50051 \
  drift.Drift/CreateCollection
```

Create payload schema:

```bash
grpcurl -plaintext \
  -import-path drift_server/proto \
  -proto drift.proto \
  -d '{
    "collectionName": "products",
    "schema": {
      "fields": [
        {
          "fieldId": 1,
          "name": "tenant",
          "logicalType": "PAYLOAD_LOGICAL_TYPE_KEYWORD",
          "nullable": false,
          "indexed": true
        },
        {
          "fieldId": 2,
          "name": "price",
          "logicalType": "PAYLOAD_LOGICAL_TYPE_FLOAT64",
          "nullable": true,
          "indexed": true
        }
      ]
    }
  }' \
  127.0.0.1:50051 \
  drift.Drift/CreatePayloadSchema
```

Batch insert:

```bash
grpcurl -plaintext \
  -import-path drift_server/proto \
  -proto drift.proto \
  -d '{
    "collectionName": "products",
    "vectors": [
      { "id": "1001", "values": [0.11, 0.72, 0.33] },
      { "id": "1002", "values": [0.10, 0.70, 0.30] }
    ],
    "payloadRows": [
      { "fields": { "1": { "keywordValue": "tenant_a" }, "2": { "float64Value": 19.99 } } },
      { "fields": { "1": { "keywordValue": "tenant_b" }, "2": { "float64Value": 29.99 } } }
    ]
  }' \
  127.0.0.1:50051 \
  drift.Drift/InsertBatch
```

Search with exact filter + projection:

```bash
grpcurl -plaintext \
  -import-path drift_server/proto \
  -proto drift.proto \
  -d '{
    "collectionName": "products",
    "vector": [0.1, 0.7, 0.3],
    "k": 10,
    "filters": [
      { "fieldId": 1, "exact": { "keywordValue": "tenant_a" } }
    ],
    "payloadProjectionFields": [1, 2]
  }' \
  127.0.0.1:50051 \
  drift.Drift/Search
```
