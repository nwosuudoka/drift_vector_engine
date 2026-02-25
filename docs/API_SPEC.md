# Drift API Spec (Current v3 Surface)

This document describes the current developer-facing API behavior in this repository.
Source of truth for message definitions is `drift_server/proto/drift.proto`.

## 1) Service endpoint and startup

Service: `drift.Drift` (gRPC, protobuf)

Default endpoint: `127.0.0.1:50051`

Start server (filesystem backend):

```bash
cargo run -p drift_server --bin drift_server -- file --path ./data
```

Start server (S3 backend):

```bash
cargo run -p drift_server --bin drift_server -- s3 --bucket <bucket> --region us-east-1
```

Important:

- You must provide a storage backend subcommand (`file` or `s3`).
- `CreateCollection` should be called before insert/train/search.

## 2) RPC summary

- `Health(HealthRequest) -> HealthResponse`
- `CreateCollection(CreateCollectionRequest) -> CreateCollectionResponse`
- `CreatePayloadSchema(CreatePayloadSchemaRequest) -> PayloadSchemaResponse`
- `UpdatePayloadSchema(UpdatePayloadSchemaRequest) -> PayloadSchemaResponse`
- `GetPayloadSchema(GetPayloadSchemaRequest) -> GetPayloadSchemaResponse`
- `ValidatePayload(ValidatePayloadRequest) -> ValidatePayloadResponse`
- `Insert(InsertRequest) -> InsertResponse`
- `InsertBatch(InsertBatchRequest) -> InsertResponse`
- `Search(SearchRequest) -> SearchResponse`
- `Train(TrainRequest) -> TrainResponse`

## 3) Collection contract

`CreateCollectionRequest`:

- `collection_name` (required, non-empty)
- `dim` (required, `> 0`)
- `metric` (required): `METRIC_TYPE_L2` or `METRIC_TYPE_COSINE`
- `max_bucket_capacity` (optional, `0` means server default)

Behavior:

- First call creates collection metadata.
- Subsequent calls validate existing dim/metric compatibility.

## 4) Payload data model

Payload row is `map<uint32, PayloadValue>` where key = `field_id`.

`PayloadValue` oneof currently supports:

- `bool_value`
- `int64_value`
- `float32_value`
- `float64_value`
- `keyword_value`
- `text_value`
- `bytes_value`
- `timestamp_micros_value`
- `lob_ref_value` (`blob_key`, `offset`, `length`, optional `fingerprint`)
- `null_value` (must be `true` when present)

## 5) Payload schema management

Schema model:

- `PayloadSchemaDefinition` contains `PayloadFieldDefinition[]`.
- Each field has `field_id`, `name`, `logical_type`, `nullable`, `indexed`.

RPCs:

- `CreatePayloadSchema`:
  - requires schema to be present and non-empty.
  - fails with `ALREADY_EXISTS` if schema is already registered.
- `UpdatePayloadSchema`:
  - replaces the currently registered schema.
  - currently requires an empty collection (no buffered/persisted vectors), otherwise `FAILED_PRECONDITION`.
- `GetPayloadSchema`:
  - returns `found=false` when no schema is registered.
- `ValidatePayload`:
  - validates rows against current schema and returns per-row error strings.
  - fails with `FAILED_PRECONDITION` if schema is not configured.

Current persistence note:

- payload schema definitions are runtime collection state in current implementation.
- schema definitions are not yet persisted across server restart.

## 6) Insert and InsertBatch

### Insert

`InsertRequest` fields:

- `collection_name`
- `vector` (`id`, `values`)
- `payload` (optional `PayloadRow`)

Validation highlights:

- payload without vector is rejected (`INVALID_ARGUMENT`).
- payload value must set a valid oneof kind.
- `null_value` must be `true` if present.
- when a payload schema is registered, payload rows are validated against field definitions.

### InsertBatch

`InsertBatchRequest` fields:

- `collection_name`
- `vectors[]`
- `payload_rows[]` (optional, positional with `vectors`)

Validation highlights:

- if `payload_rows` is provided, length must equal `vectors` length.
- field logical type must be consistent per `field_id` within the request.
- null-only fields are rejected (at least one typed value required).
- when a payload schema is registered, each row is validated against that schema.

Schema note:

- If schema is registered: inserts must satisfy registered definitions.
- If schema is not registered: schema is inferred from payload rows in each request.

## 7) Search with filters and payload projection

`SearchRequest`:

- `collection_name`
- `vector` (query)
- `k`
- drift params: `target_confidence`, `lambda`, `tau`
- `filters[]` (optional)
- `payload_projection_fields[]` (optional)

`SearchResult`:

- `id`
- `score`
- `payload` (optional `PayloadRow`)

### Filter operators

`FieldFilter` supports one condition:

- `exact: PayloadValue`
- `any_of: PayloadValueList`
- `range: RangeFilter`

`RangeFilter`:

- `lower` optional
- `upper` optional
- `lower_inclusive` optional (defaults `true`)
- `upper_inclusive` optional (defaults `true`)
- at least one bound required

### Current evaluation semantics

- All filters are ANDed.
- `exact` / `any_of`: missing field is treated as `null`.
- `range`: missing/null/non-comparable field fails the filter.
- Numeric cross-type comparisons are supported for:
  - `int64`, `float32`, `float64`, `timestamp_micros`
- `keyword` and `text` compare lexicographically.

### Projection semantics

- Empty `payload_projection_fields`:
  - response omits payload (`SearchResult.payload = null`).
- Non-empty projection:
  - payload object is returned with requested fields that exist.

### Candidate expansion behavior

When filters or projection are requested, server expands candidate count before payload eval:

- `candidate_k = clamp(k * 8, min = k, max = 8192)`

Filtering/projection currently happens after vector candidate retrieval.

## 8) Error mapping

- Validation issues -> `INVALID_ARGUMENT`
- Missing collection or missing resources -> `NOT_FOUND`
- Precondition failures (for example missing schema or non-empty collection during schema update) -> `FAILED_PRECONDITION`
- Internal IO/engine failures -> `INTERNAL`

## 9) grpcurl quickstart examples

All examples assume server at `127.0.0.1:50051`.
Use `-plaintext` for local non-TLS testing.

Health:

```bash
grpcurl -plaintext \
  -import-path drift_server/proto \
  -proto drift.proto \
  -d '{}' \
  127.0.0.1:50051 \
  drift.Drift/Health
```

Create collection:

```bash
grpcurl -plaintext \
  -import-path drift_server/proto \
  -proto drift.proto \
  -d '{
    "collectionName": "products",
    "dim": 3,
    "metric": "METRIC_TYPE_COSINE",
    "maxBucketCapacity": 0
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

Validate payload rows:

```bash
grpcurl -plaintext \
  -import-path drift_server/proto \
  -proto drift.proto \
  -d '{
    "collectionName": "products",
    "rows": [
      { "fields": { "1": { "keywordValue": "tenant_a" }, "2": { "float64Value": 19.99 } } },
      { "fields": { "1": { "int64Value": "7" } } }
    ]
  }' \
  127.0.0.1:50051 \
  drift.Drift/ValidatePayload
```

Insert one vector with payload:

```bash
grpcurl -plaintext \
  -import-path drift_server/proto \
  -proto drift.proto \
  -d '{
    "collectionName": "products",
    "vector": { "id": "1001", "values": [0.11, 0.72, 0.33] },
    "payload": {
      "fields": {
        "1": { "keywordValue": "tenant_a" },
        "2": { "int64Value": "42" },
        "3": { "float64Value": 19.99 }
      }
    }
  }' \
  127.0.0.1:50051 \
  drift.Drift/Insert
```

Batch insert with aligned payload rows:

```bash
grpcurl -plaintext \
  -import-path drift_server/proto \
  -proto drift.proto \
  -d '{
    "collectionName": "products",
    "vectors": [
      { "id": "1002", "values": [0.12, 0.70, 0.31] },
      { "id": "1003", "values": [0.15, 0.69, 0.29] }
    ],
    "payloadRows": [
      { "fields": { "1": { "keywordValue": "tenant_a" }, "2": { "int64Value": "10" } } },
      { "fields": { "1": { "keywordValue": "tenant_b" }, "2": { "int64Value": "20" } } }
    ]
  }' \
  127.0.0.1:50051 \
  drift.Drift/InsertBatch
```

Search with exact filter + payload projection:

```bash
grpcurl -plaintext \
  -import-path drift_server/proto \
  -proto drift.proto \
  -d '{
    "collectionName": "products",
    "vector": [0.1, 0.7, 0.3],
    "k": 5,
    "targetConfidence": 0.95,
    "lambda": 25.0,
    "tau": 100.0,
    "filters": [
      {
        "fieldId": 1,
        "exact": { "keywordValue": "tenant_a" }
      }
    ],
    "payloadProjectionFields": [1, 2]
  }' \
  127.0.0.1:50051 \
  drift.Drift/Search
```

Search with range filter:

```bash
grpcurl -plaintext \
  -import-path drift_server/proto \
  -proto drift.proto \
  -d '{
    "collectionName": "products",
    "vector": [0.1, 0.7, 0.3],
    "k": 5,
    "filters": [
      {
        "fieldId": 2,
        "range": {
          "lower": { "int64Value": "10" },
          "upper": { "int64Value": "30" },
          "lowerInclusive": true,
          "upperInclusive": false
        }
      }
    ]
  }' \
  127.0.0.1:50051 \
  drift.Drift/Search
```

## 10) CLI note

`cargo run -p drift_server --bin drift -- --help` is useful for basic vector flows, but it does
not yet expose payload/filter/projection arguments. For those features, use gRPC clients directly.
