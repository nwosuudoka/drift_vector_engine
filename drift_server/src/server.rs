use crate::drift_proto::{
    CreateCollectionRequest, CreateCollectionResponse, CreatePayloadSchemaRequest, FieldFilter,
    GetPayloadSchemaRequest, GetPayloadSchemaResponse, InsertBatchRequest, InsertRequest,
    MetricType, PayloadFieldDefinition, PayloadLogicalType, PayloadRow, PayloadSchemaDefinition,
    PayloadSchemaResponse, PayloadValue, SearchRequest, SearchResponse, SearchResult,
    UpdatePayloadSchemaRequest, ValidatePayloadRequest, ValidatePayloadResponse,
    drift_server::Drift,
};
use crate::drift_proto::{
    HealthRequest, HealthResponse, InsertResponse, NvmeCacheMetrics, RecoveryGuardMetrics,
    TrainRequest, TrainResponse,
};
use crate::filter_metadata_catalog::{
    BucketExactClauseCoverage, BucketProbeObservation, BucketRangeClauseCoverage,
    ExactFieldQueryClause, ExactValueMembershipKey, ExactValuePresence, RangeFieldQueryClause,
    RangeFieldZoneMap, logical_type_tag,
};
use crate::filter_planner_diagnostics::{
    FilterPlannerDiagnosticsSnapshot, diagnostics_enabled_from_env,
};
use crate::global_filter_routing_index::GlobalFilterRoutingIndex;
use crate::manager::{Collection, CollectionManager};
use crate::recovery::RecoveryManager;
use drift_core::math::Metric;
use drift_core::payload::{
    PayloadFieldSchema as CorePayloadFieldSchema, PayloadLogicalType as CorePayloadLogicalType,
    PayloadRow as CorePayloadRow, PayloadSchema as CorePayloadSchema,
    PayloadValue as CorePayloadValue,
};
use drift_storage::bucket_manager::StorageClass;
use drift_storage::disk_manager::DiskManager;
use drift_storage::unified_format::{
    UnifiedLobRef, UnifiedLogicalType, UnifiedPayloadFieldStats, UnifiedPayloadRow,
    UnifiedPayloadStatsChunk, UnifiedPayloadValue, encode_exact_key,
};
use drift_storage::unified_reader::UnifiedReader;
use drift_traits::StorageEngine;
use opendal::{Operator, services};
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::io;
use std::sync::Arc;
use tonic::{Request, Response, Status};

pub struct DriftService {
    pub manager: Arc<CollectionManager>,
}

fn map_collection_error(err: io::Error) -> Status {
    match err.kind() {
        io::ErrorKind::NotFound => Status::not_found(err.to_string()),
        io::ErrorKind::InvalidInput => Status::invalid_argument(err.to_string()),
        _ => Status::internal(err.to_string()),
    }
}

fn metric_from_proto(metric: i32) -> Result<Metric, Status> {
    let parsed = MetricType::try_from(metric)
        .map_err(|_| Status::invalid_argument(format!("Unknown metric enum value: {metric}")))?;

    match parsed {
        MetricType::L2 => Ok(Metric::L2),
        MetricType::Cosine => Ok(Metric::COSINE),
        MetricType::Unspecified => Err(Status::invalid_argument("Metric must be specified")),
    }
}

#[derive(Debug, Clone)]
enum ParsedFilter {
    Exact {
        field_id: u32,
        value: UnifiedPayloadValue,
    },
    AnyOf {
        field_id: u32,
        values: Vec<UnifiedPayloadValue>,
    },
    Range {
        field_id: u32,
        lower: Option<UnifiedPayloadValue>,
        lower_inclusive: bool,
        upper: Option<UnifiedPayloadValue>,
        upper_inclusive: bool,
    },
}

enum PayloadSource {
    Local(String),
    Remote(String),
}

fn parse_proto_payload_value_to_core(value: &PayloadValue) -> Result<CorePayloadValue, Status> {
    let Some(kind) = value.kind.as_ref() else {
        return Err(Status::invalid_argument("payload value missing kind"));
    };
    match kind {
        crate::drift_proto::payload_value::Kind::BoolValue(v) => Ok(CorePayloadValue::Bool(*v)),
        crate::drift_proto::payload_value::Kind::Int64Value(v) => Ok(CorePayloadValue::Int64(*v)),
        crate::drift_proto::payload_value::Kind::Float32Value(v) => {
            Ok(CorePayloadValue::Float32(*v))
        }
        crate::drift_proto::payload_value::Kind::Float64Value(v) => {
            Ok(CorePayloadValue::Float64(*v))
        }
        crate::drift_proto::payload_value::Kind::KeywordValue(v) => {
            Ok(CorePayloadValue::Keyword(v.clone()))
        }
        crate::drift_proto::payload_value::Kind::TextValue(v) => {
            Ok(CorePayloadValue::Text(v.clone()))
        }
        crate::drift_proto::payload_value::Kind::BytesValue(v) => {
            Ok(CorePayloadValue::Bytes(v.clone()))
        }
        crate::drift_proto::payload_value::Kind::TimestampMicrosValue(v) => {
            Ok(CorePayloadValue::TimestampMicros(*v))
        }
        crate::drift_proto::payload_value::Kind::LobRefValue(v) => Ok(CorePayloadValue::LobRef(
            drift_core::payload::PayloadLobRef {
                blob_key: v.blob_key.clone(),
                offset: v.offset,
                length: v.length,
                fingerprint: if v.fingerprint.is_empty() {
                    None
                } else {
                    Some(v.fingerprint.clone())
                },
            },
        )),
        crate::drift_proto::payload_value::Kind::NullValue(v) => {
            if !*v {
                return Err(Status::invalid_argument(
                    "payload null_value must be true when provided",
                ));
            }
            Ok(CorePayloadValue::Null)
        }
    }
}

fn parse_proto_payload_value_to_unified(
    value: &PayloadValue,
) -> Result<UnifiedPayloadValue, Status> {
    let Some(kind) = value.kind.as_ref() else {
        return Err(Status::invalid_argument("payload value missing kind"));
    };
    match kind {
        crate::drift_proto::payload_value::Kind::BoolValue(v) => Ok(UnifiedPayloadValue::Bool(*v)),
        crate::drift_proto::payload_value::Kind::Int64Value(v) => {
            Ok(UnifiedPayloadValue::Int64(*v))
        }
        crate::drift_proto::payload_value::Kind::Float32Value(v) => {
            Ok(UnifiedPayloadValue::Float32(*v))
        }
        crate::drift_proto::payload_value::Kind::Float64Value(v) => {
            Ok(UnifiedPayloadValue::Float64(*v))
        }
        crate::drift_proto::payload_value::Kind::KeywordValue(v) => {
            Ok(UnifiedPayloadValue::Keyword(v.clone()))
        }
        crate::drift_proto::payload_value::Kind::TextValue(v) => {
            Ok(UnifiedPayloadValue::Text(v.clone()))
        }
        crate::drift_proto::payload_value::Kind::BytesValue(v) => {
            Ok(UnifiedPayloadValue::Bytes(v.clone()))
        }
        crate::drift_proto::payload_value::Kind::TimestampMicrosValue(v) => {
            Ok(UnifiedPayloadValue::TimestampMicros(*v))
        }
        crate::drift_proto::payload_value::Kind::LobRefValue(v) => {
            Ok(UnifiedPayloadValue::LobRef(UnifiedLobRef {
                blob_key: v.blob_key.clone(),
                offset: v.offset,
                length: v.length,
                fingerprint: if v.fingerprint.is_empty() {
                    None
                } else {
                    Some(v.fingerprint.clone())
                },
            }))
        }
        crate::drift_proto::payload_value::Kind::NullValue(v) => {
            if !*v {
                return Err(Status::invalid_argument(
                    "payload null_value must be true when provided",
                ));
            }
            Ok(UnifiedPayloadValue::Null)
        }
    }
}

fn core_payload_value_logical_type(value: &CorePayloadValue) -> Option<CorePayloadLogicalType> {
    match value {
        CorePayloadValue::Bool(_) => Some(CorePayloadLogicalType::Bool),
        CorePayloadValue::Int64(_) => Some(CorePayloadLogicalType::Int64),
        CorePayloadValue::Float32(_) => Some(CorePayloadLogicalType::Float32),
        CorePayloadValue::Float64(_) => Some(CorePayloadLogicalType::Float64),
        CorePayloadValue::Keyword(_) => Some(CorePayloadLogicalType::Keyword),
        CorePayloadValue::Text(_) => Some(CorePayloadLogicalType::Text),
        CorePayloadValue::Bytes(_) => Some(CorePayloadLogicalType::Bytes),
        CorePayloadValue::TimestampMicros(_) => Some(CorePayloadLogicalType::TimestampMicros),
        CorePayloadValue::LobRef(_) => Some(CorePayloadLogicalType::LobRef),
        CorePayloadValue::Null => None,
    }
}

fn infer_core_payload_schema(rows: &[CorePayloadRow]) -> Result<Option<CorePayloadSchema>, Status> {
    let mut field_ids: HashSet<u32> = HashSet::new();
    let mut field_types: HashMap<u32, CorePayloadLogicalType> = HashMap::new();

    for row in rows {
        for (field_id, value) in row {
            field_ids.insert(*field_id);
            if let Some(logical_type) = core_payload_value_logical_type(value) {
                if let Some(existing) = field_types.get(field_id) {
                    if existing != &logical_type {
                        return Err(Status::invalid_argument(format!(
                            "payload field {} has inconsistent logical types",
                            field_id
                        )));
                    }
                } else {
                    field_types.insert(*field_id, logical_type);
                }
            }
        }
    }

    if field_ids.is_empty() {
        return Ok(None);
    }

    let mut ids: Vec<u32> = field_ids.into_iter().collect();
    ids.sort_unstable();
    let mut fields = Vec::with_capacity(ids.len());

    for field_id in ids {
        let Some(logical_type) = field_types.get(&field_id).cloned() else {
            return Err(Status::invalid_argument(format!(
                "payload field {} is null-only; provide at least one typed value",
                field_id
            )));
        };
        let nullable = rows.iter().any(|row| match row.get(&field_id) {
            Some(CorePayloadValue::Null) => true,
            Some(_) => false,
            None => true,
        });

        fields.push(CorePayloadFieldSchema {
            field_id,
            name: format!("field_{field_id}"),
            logical_type,
            nullable,
            indexed: true,
        });
    }

    Ok(Some(CorePayloadSchema::new(fields)))
}

fn payload_logical_type_from_proto(logical_type: i32) -> Result<CorePayloadLogicalType, Status> {
    let parsed = PayloadLogicalType::try_from(logical_type).map_err(|_| {
        Status::invalid_argument(format!(
            "unknown payload logical type enum value: {}",
            logical_type
        ))
    })?;
    match parsed {
        PayloadLogicalType::Bool => Ok(CorePayloadLogicalType::Bool),
        PayloadLogicalType::Int64 => Ok(CorePayloadLogicalType::Int64),
        PayloadLogicalType::Float32 => Ok(CorePayloadLogicalType::Float32),
        PayloadLogicalType::Float64 => Ok(CorePayloadLogicalType::Float64),
        PayloadLogicalType::Keyword => Ok(CorePayloadLogicalType::Keyword),
        PayloadLogicalType::Text => Ok(CorePayloadLogicalType::Text),
        PayloadLogicalType::Bytes => Ok(CorePayloadLogicalType::Bytes),
        PayloadLogicalType::TimestampMicros => Ok(CorePayloadLogicalType::TimestampMicros),
        PayloadLogicalType::LobRef => Ok(CorePayloadLogicalType::LobRef),
        PayloadLogicalType::Unspecified => Err(Status::invalid_argument(
            "payload logical type must be specified",
        )),
    }
}

fn payload_logical_type_to_proto(logical_type: &CorePayloadLogicalType) -> i32 {
    match logical_type {
        CorePayloadLogicalType::Bool => PayloadLogicalType::Bool as i32,
        CorePayloadLogicalType::Int64 => PayloadLogicalType::Int64 as i32,
        CorePayloadLogicalType::Float32 => PayloadLogicalType::Float32 as i32,
        CorePayloadLogicalType::Float64 => PayloadLogicalType::Float64 as i32,
        CorePayloadLogicalType::Keyword => PayloadLogicalType::Keyword as i32,
        CorePayloadLogicalType::Text => PayloadLogicalType::Text as i32,
        CorePayloadLogicalType::Bytes => PayloadLogicalType::Bytes as i32,
        CorePayloadLogicalType::TimestampMicros => PayloadLogicalType::TimestampMicros as i32,
        CorePayloadLogicalType::LobRef => PayloadLogicalType::LobRef as i32,
    }
}

fn payload_logical_type_label(logical_type: &CorePayloadLogicalType) -> &'static str {
    match logical_type {
        CorePayloadLogicalType::Bool => "bool",
        CorePayloadLogicalType::Int64 => "int64",
        CorePayloadLogicalType::Float32 => "float32",
        CorePayloadLogicalType::Float64 => "float64",
        CorePayloadLogicalType::Keyword => "keyword",
        CorePayloadLogicalType::Text => "text",
        CorePayloadLogicalType::Bytes => "bytes",
        CorePayloadLogicalType::TimestampMicros => "timestamp_micros",
        CorePayloadLogicalType::LobRef => "lob_ref",
    }
}

fn core_payload_schema_to_proto(schema: &CorePayloadSchema) -> PayloadSchemaDefinition {
    let fields = schema
        .fields
        .iter()
        .map(|field| PayloadFieldDefinition {
            field_id: field.field_id,
            name: field.name.clone(),
            logical_type: payload_logical_type_to_proto(&field.logical_type),
            nullable: field.nullable,
            indexed: field.indexed,
        })
        .collect();
    PayloadSchemaDefinition { fields }
}

fn validate_core_payload_schema(schema: &CorePayloadSchema) -> Result<(), Status> {
    if schema.fields.is_empty() {
        return Err(Status::invalid_argument(
            "payload schema must include at least one field definition",
        ));
    }
    let mut seen = HashSet::with_capacity(schema.fields.len());
    for field in &schema.fields {
        if !seen.insert(field.field_id) {
            return Err(Status::invalid_argument(format!(
                "duplicate payload field_id {} in schema definition",
                field.field_id
            )));
        }
        if field.name.trim().is_empty() {
            return Err(Status::invalid_argument(format!(
                "payload field {} has empty name",
                field.field_id
            )));
        }
    }
    Ok(())
}

fn proto_payload_schema_to_core(
    schema: PayloadSchemaDefinition,
) -> Result<CorePayloadSchema, Status> {
    if schema.fields.is_empty() {
        return Err(Status::invalid_argument(
            "payload schema must include at least one field definition",
        ));
    }

    let mut fields = Vec::with_capacity(schema.fields.len());
    for field in schema.fields {
        let logical_type = payload_logical_type_from_proto(field.logical_type)?;
        let name = if field.name.trim().is_empty() {
            format!("field_{}", field.field_id)
        } else {
            field.name
        };
        fields.push(CorePayloadFieldSchema {
            field_id: field.field_id,
            name,
            logical_type,
            nullable: field.nullable,
            indexed: field.indexed,
        });
    }
    fields.sort_by_key(|field| field.field_id);
    let schema = CorePayloadSchema::new(fields);
    validate_core_payload_schema(&schema)?;
    Ok(schema)
}

fn validate_payload_rows_against_schema(
    rows: &[CorePayloadRow],
    schema: &CorePayloadSchema,
) -> Vec<String> {
    let field_by_id: HashMap<u32, &CorePayloadFieldSchema> = schema
        .fields
        .iter()
        .map(|field| (field.field_id, field))
        .collect();
    let mut errors = Vec::new();

    for (row_idx, row) in rows.iter().enumerate() {
        for field_id in row.keys() {
            if !field_by_id.contains_key(field_id) {
                errors.push(format!(
                    "row {} contains unknown field_id {}",
                    row_idx, field_id
                ));
            }
        }

        for field in &schema.fields {
            let Some(value) = row.get(&field.field_id) else {
                if !field.nullable {
                    errors.push(format!(
                        "row {} missing non-nullable field_id {} ({})",
                        row_idx, field.field_id, field.name
                    ));
                }
                continue;
            };

            if matches!(value, CorePayloadValue::Null) {
                if !field.nullable {
                    errors.push(format!(
                        "row {} has null for non-nullable field_id {} ({})",
                        row_idx, field.field_id, field.name
                    ));
                }
                continue;
            }

            let Some(actual_type) = core_payload_value_logical_type(value) else {
                continue;
            };
            if actual_type != field.logical_type {
                errors.push(format!(
                    "row {} field_id {} ({}) type mismatch: expected {}, got {}",
                    row_idx,
                    field.field_id,
                    field.name,
                    payload_logical_type_label(&field.logical_type),
                    payload_logical_type_label(&actual_type)
                ));
            }
        }
    }

    errors
}

fn proto_payload_row_to_core(row: PayloadRow) -> Result<CorePayloadRow, Status> {
    let mut out: BTreeMap<u32, CorePayloadValue> = BTreeMap::new();
    for (field_id, value) in row.fields {
        out.insert(field_id, parse_proto_payload_value_to_core(&value)?);
    }
    Ok(out)
}

fn core_payload_value_to_unified(value: &CorePayloadValue) -> UnifiedPayloadValue {
    match value {
        CorePayloadValue::Bool(v) => UnifiedPayloadValue::Bool(*v),
        CorePayloadValue::Int64(v) => UnifiedPayloadValue::Int64(*v),
        CorePayloadValue::Float32(v) => UnifiedPayloadValue::Float32(*v),
        CorePayloadValue::Float64(v) => UnifiedPayloadValue::Float64(*v),
        CorePayloadValue::Keyword(v) => UnifiedPayloadValue::Keyword(v.clone()),
        CorePayloadValue::Text(v) => UnifiedPayloadValue::Text(v.clone()),
        CorePayloadValue::Bytes(v) => UnifiedPayloadValue::Bytes(v.clone()),
        CorePayloadValue::TimestampMicros(v) => UnifiedPayloadValue::TimestampMicros(*v),
        CorePayloadValue::LobRef(v) => UnifiedPayloadValue::LobRef(UnifiedLobRef {
            blob_key: v.blob_key.clone(),
            offset: v.offset,
            length: v.length,
            fingerprint: v.fingerprint.clone(),
        }),
        CorePayloadValue::Null => UnifiedPayloadValue::Null,
    }
}

fn core_payload_row_to_unified(row: &CorePayloadRow) -> UnifiedPayloadRow {
    row.iter()
        .map(|(field_id, value)| (*field_id, core_payload_value_to_unified(value)))
        .collect()
}

fn unified_payload_value_to_proto(value: &UnifiedPayloadValue) -> PayloadValue {
    let kind = match value {
        UnifiedPayloadValue::Null => crate::drift_proto::payload_value::Kind::NullValue(true),
        UnifiedPayloadValue::Bool(v) => crate::drift_proto::payload_value::Kind::BoolValue(*v),
        UnifiedPayloadValue::Int64(v) => crate::drift_proto::payload_value::Kind::Int64Value(*v),
        UnifiedPayloadValue::Float32(v) => {
            crate::drift_proto::payload_value::Kind::Float32Value(*v)
        }
        UnifiedPayloadValue::Float64(v) => {
            crate::drift_proto::payload_value::Kind::Float64Value(*v)
        }
        UnifiedPayloadValue::TimestampMicros(v) => {
            crate::drift_proto::payload_value::Kind::TimestampMicrosValue(*v)
        }
        UnifiedPayloadValue::Keyword(v) => {
            crate::drift_proto::payload_value::Kind::KeywordValue(v.clone())
        }
        UnifiedPayloadValue::Text(v) => {
            crate::drift_proto::payload_value::Kind::TextValue(v.clone())
        }
        UnifiedPayloadValue::Bytes(v) => {
            crate::drift_proto::payload_value::Kind::BytesValue(v.clone())
        }
        UnifiedPayloadValue::LobRef(v) => crate::drift_proto::payload_value::Kind::LobRefValue(
            crate::drift_proto::PayloadLobRef {
                blob_key: v.blob_key.clone(),
                offset: v.offset,
                length: v.length,
                fingerprint: v.fingerprint.clone().unwrap_or_default(),
            },
        ),
    };
    PayloadValue { kind: Some(kind) }
}

fn parse_filters(filters: &[FieldFilter]) -> Result<Vec<ParsedFilter>, Status> {
    let mut parsed = Vec::with_capacity(filters.len());
    for filter in filters {
        let field_id = filter.field_id;
        let Some(condition) = filter.condition.as_ref() else {
            return Err(Status::invalid_argument(format!(
                "filter for field {} missing condition",
                field_id
            )));
        };
        match condition {
            crate::drift_proto::field_filter::Condition::Exact(value) => {
                parsed.push(ParsedFilter::Exact {
                    field_id,
                    value: parse_proto_payload_value_to_unified(value)?,
                });
            }
            crate::drift_proto::field_filter::Condition::AnyOf(values) => {
                if values.values.is_empty() {
                    return Err(Status::invalid_argument(format!(
                        "any_of filter for field {} requires at least one value",
                        field_id
                    )));
                }
                let mut parsed_values = Vec::with_capacity(values.values.len());
                for value in &values.values {
                    parsed_values.push(parse_proto_payload_value_to_unified(value)?);
                }
                parsed.push(ParsedFilter::AnyOf {
                    field_id,
                    values: parsed_values,
                });
            }
            crate::drift_proto::field_filter::Condition::Range(range) => {
                let lower = range
                    .lower
                    .as_ref()
                    .map(parse_proto_payload_value_to_unified)
                    .transpose()?;
                let upper = range
                    .upper
                    .as_ref()
                    .map(parse_proto_payload_value_to_unified)
                    .transpose()?;
                if lower.is_none() && upper.is_none() {
                    return Err(Status::invalid_argument(format!(
                        "range filter for field {} requires lower or upper bound",
                        field_id
                    )));
                }
                parsed.push(ParsedFilter::Range {
                    field_id,
                    lower,
                    lower_inclusive: range.lower_inclusive.unwrap_or(true),
                    upper,
                    upper_inclusive: range.upper_inclusive.unwrap_or(true),
                });
            }
        }
    }
    Ok(parsed)
}

fn numeric_payload_value(value: &UnifiedPayloadValue) -> Option<f64> {
    match value {
        UnifiedPayloadValue::Int64(v) => Some(*v as f64),
        UnifiedPayloadValue::Float32(v) => Some(*v as f64),
        UnifiedPayloadValue::Float64(v) => Some(*v),
        UnifiedPayloadValue::TimestampMicros(v) => Some(*v as f64),
        _ => None,
    }
}

fn compare_payload_values(
    lhs: &UnifiedPayloadValue,
    rhs: &UnifiedPayloadValue,
) -> Option<Ordering> {
    if let (Some(a), Some(b)) = (numeric_payload_value(lhs), numeric_payload_value(rhs)) {
        return a.partial_cmp(&b);
    }

    match (lhs, rhs) {
        (UnifiedPayloadValue::Bool(a), UnifiedPayloadValue::Bool(b)) => Some(a.cmp(b)),
        (UnifiedPayloadValue::Keyword(a), UnifiedPayloadValue::Keyword(b)) => Some(a.cmp(b)),
        (UnifiedPayloadValue::Text(a), UnifiedPayloadValue::Text(b)) => Some(a.cmp(b)),
        (UnifiedPayloadValue::Keyword(a), UnifiedPayloadValue::Text(b)) => Some(a.cmp(b)),
        (UnifiedPayloadValue::Text(a), UnifiedPayloadValue::Keyword(b)) => Some(a.cmp(b)),
        (UnifiedPayloadValue::Bytes(a), UnifiedPayloadValue::Bytes(b)) => Some(a.cmp(b)),
        (UnifiedPayloadValue::TimestampMicros(a), UnifiedPayloadValue::TimestampMicros(b)) => {
            Some(a.cmp(b))
        }
        _ => None,
    }
}

fn row_matches_filters(row: Option<&UnifiedPayloadRow>, filters: &[ParsedFilter]) -> bool {
    if filters.is_empty() {
        return true;
    }
    let Some(row) = row else {
        return false;
    };

    filters.iter().all(|filter| match filter {
        ParsedFilter::Exact { field_id, value } => {
            row.get(field_id).unwrap_or(&UnifiedPayloadValue::Null) == value
        }
        ParsedFilter::AnyOf { field_id, values } => {
            let current = row.get(field_id).unwrap_or(&UnifiedPayloadValue::Null);
            values.iter().any(|candidate| candidate == current)
        }
        ParsedFilter::Range {
            field_id,
            lower,
            lower_inclusive,
            upper,
            upper_inclusive,
        } => {
            let Some(current) = row.get(field_id) else {
                return false;
            };
            if matches!(current, UnifiedPayloadValue::Null) {
                return false;
            }

            if let Some(lower_bound) = lower {
                let Some(ordering) = compare_payload_values(current, lower_bound) else {
                    return false;
                };
                if *lower_inclusive {
                    if ordering == Ordering::Less {
                        return false;
                    }
                } else if ordering != Ordering::Greater {
                    return false;
                }
            }

            if let Some(upper_bound) = upper {
                let Some(ordering) = compare_payload_values(current, upper_bound) else {
                    return false;
                };
                if *upper_inclusive {
                    if ordering == Ordering::Greater {
                        return false;
                    }
                } else if ordering != Ordering::Less {
                    return false;
                }
            }

            true
        }
    })
}

fn project_payload_row(row: Option<&UnifiedPayloadRow>, projection: &HashSet<u32>) -> PayloadRow {
    if projection.is_empty() {
        return PayloadRow {
            fields: HashMap::new(),
        };
    }
    let mut fields = HashMap::new();
    if let Some(row) = row {
        for field_id in projection {
            if let Some(value) = row.get(field_id) {
                fields.insert(*field_id, unified_payload_value_to_proto(value));
            }
        }
    }
    PayloadRow { fields }
}

fn decode_bucket_id(raw: &[u8]) -> Option<u32> {
    let bytes: [u8; 4] = raw.try_into().ok()?;
    Some(u32::from_le_bytes(bytes))
}

fn payload_sources_for_class(class: &StorageClass, primary_path: &str) -> Vec<PayloadSource> {
    match class {
        StorageClass::Local => vec![PayloadSource::Local(primary_path.to_string())],
        StorageClass::Remote => vec![PayloadSource::Remote(primary_path.to_string())],
        StorageClass::Tiered {
            remote_path,
            local_path,
        } => vec![
            PayloadSource::Remote(remote_path.clone()),
            PayloadSource::Local(local_path.clone()),
        ],
        StorageClass::Promoting {
            local_active,
            local_frozen,
            remote_path,
        } => {
            let mut sources = Vec::new();
            if let Some(path) = remote_path {
                sources.push(PayloadSource::Remote(path.clone()));
            }
            sources.push(PayloadSource::Local(local_frozen.clone()));
            sources.push(PayloadSource::Local(local_active.clone()));
            sources
        }
    }
}

fn create_staging_operator(collection: &Collection) -> io::Result<Operator> {
    let root = collection
        .staging
        .get_base_path()
        .to_str()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "invalid staging path"))?;
    let builder = services::Fs::default().root(root);
    Ok(Operator::new(builder).map_err(io::Error::other)?.finish())
}

fn range_filter_overlaps_field_stats(
    stats: &UnifiedPayloadFieldStats,
    row_count: u32,
    lower: Option<&UnifiedPayloadValue>,
    lower_inclusive: bool,
    upper: Option<&UnifiedPayloadValue>,
    upper_inclusive: bool,
) -> bool {
    if stats.null_count >= row_count {
        return false;
    }

    let (Some(min), Some(max)) = (stats.min.as_ref(), stats.max.as_ref()) else {
        // Keep the bucket/source if stats are partial; final row-level evaluation remains authoritative.
        return true;
    };

    if let Some(lower_bound) = lower {
        let Some(ordering) = compare_payload_values(max, lower_bound) else {
            return false;
        };
        if lower_inclusive {
            if ordering == Ordering::Less {
                return false;
            }
        } else if ordering != Ordering::Greater {
            return false;
        }
    }

    if let Some(upper_bound) = upper {
        let Some(ordering) = compare_payload_values(min, upper_bound) else {
            return false;
        };
        if upper_inclusive {
            if ordering == Ordering::Greater {
                return false;
            }
        } else if ordering != Ordering::Less {
            return false;
        }
    }

    true
}

fn range_filter_might_match_chunk(
    chunk: &UnifiedPayloadStatsChunk,
    field_id: u32,
    lower: Option<&UnifiedPayloadValue>,
    lower_inclusive: bool,
    upper: Option<&UnifiedPayloadValue>,
    upper_inclusive: bool,
) -> bool {
    let Some(stats) = chunk.fields.iter().find(|f| f.field_id == field_id) else {
        return false;
    };
    range_filter_overlaps_field_stats(
        stats,
        chunk.row_count,
        lower,
        lower_inclusive,
        upper,
        upper_inclusive,
    )
}

fn aggregate_range_zone_map_for_field(
    chunks: &[UnifiedPayloadStatsChunk],
    field_id: u32,
) -> Option<RangeFieldZoneMap> {
    let mut saw_field_stats = false;
    let mut has_non_null_values = false;
    let mut min_value: Option<UnifiedPayloadValue> = None;
    let mut max_value: Option<UnifiedPayloadValue> = None;

    for chunk in chunks {
        let Some(stats) = chunk.fields.iter().find(|field| field.field_id == field_id) else {
            continue;
        };
        saw_field_stats = true;

        if stats.null_count >= chunk.row_count {
            continue;
        }
        has_non_null_values = true;

        let (Some(chunk_min), Some(chunk_max)) = (stats.min.clone(), stats.max.clone()) else {
            return Some(RangeFieldZoneMap {
                has_non_null_values: true,
                min: None,
                max: None,
            });
        };

        min_value = match min_value.take() {
            Some(current) => match compare_payload_values(&current, &chunk_min) {
                Some(Ordering::Greater) => Some(chunk_min),
                Some(_) => Some(current),
                None => None,
            },
            None => Some(chunk_min),
        };
        max_value = match max_value.take() {
            Some(current) => match compare_payload_values(&current, &chunk_max) {
                Some(Ordering::Less) => Some(chunk_max),
                Some(_) => Some(current),
                None => None,
            },
            None => Some(chunk_max),
        };
    }

    if !saw_field_stats {
        return None;
    }
    Some(RangeFieldZoneMap {
        has_non_null_values,
        min: min_value,
        max: max_value,
    })
}

fn intersect_candidate_ids(existing: &mut HashSet<u64>, next: &HashSet<u64>) {
    existing.retain(|id| next.contains(id));
}

#[derive(Default)]
struct FilterSourceProbe {
    might_match: bool,
    exact_candidate_ids: Option<HashSet<u64>>,
    saw_exact_filter: bool,
    saw_indexed_exact_filter: bool,
    saw_empty_exact_match: bool,
    saw_range_filter: bool,
    indexed_exact_field_ids: HashSet<u32>,
    range_filter_field_ids: HashSet<u32>,
    exact_value_presence: HashMap<ExactValueMembershipKey, ExactValuePresence>,
    range_field_zone_maps: HashMap<u32, RangeFieldZoneMap>,
}

#[derive(Default)]
struct FilterAwareExecutionPlan {
    bucket_ids: Vec<u32>,
    candidate_ids: HashMap<u32, HashSet<u64>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CandidateAbsenceReason {
    EmptyExactMatch,
    NoIndexedExact,
    RangeStatsOnly,
    Other,
}

fn classify_candidate_absence_reason(
    saw_exact_filter: bool,
    saw_indexed_exact_filter: bool,
    saw_empty_exact_match: bool,
    saw_range_filter: bool,
) -> CandidateAbsenceReason {
    if saw_empty_exact_match {
        return CandidateAbsenceReason::EmptyExactMatch;
    }
    if saw_exact_filter && !saw_indexed_exact_filter {
        return CandidateAbsenceReason::NoIndexedExact;
    }
    if saw_range_filter && !saw_exact_filter {
        return CandidateAbsenceReason::RangeStatsOnly;
    }
    CandidateAbsenceReason::Other
}

fn record_candidate_decision(
    snapshot: &mut FilterPlannerDiagnosticsSnapshot,
    has_bucket_candidates: bool,
    candidate_applied: bool,
    candidate_gated: bool,
    disable_candidates: bool,
    bucket_candidate_count: usize,
    saw_exact_filter: bool,
    saw_indexed_exact_filter: bool,
    saw_empty_exact_match: bool,
    saw_range_filter: bool,
) {
    if has_bucket_candidates {
        snapshot.candidate_produced_bucket_count += 1;
        snapshot.candidate_id_count += bucket_candidate_count;
        if candidate_applied {
            snapshot.candidate_applied_bucket_count += 1;
        } else if candidate_gated {
            snapshot.candidate_gated_broad_selectivity_bucket_count += 1;
        } else {
            snapshot.candidate_other_absence_bucket_count += 1;
        }
        return;
    }

    if disable_candidates {
        snapshot.candidate_disabled_probe_error_bucket_count += 1;
        return;
    }

    match classify_candidate_absence_reason(
        saw_exact_filter,
        saw_indexed_exact_filter,
        saw_empty_exact_match,
        saw_range_filter,
    ) {
        CandidateAbsenceReason::EmptyExactMatch => {
            snapshot.candidate_empty_exact_match_bucket_count += 1;
        }
        CandidateAbsenceReason::NoIndexedExact => {
            snapshot.candidate_no_indexed_exact_bucket_count += 1;
        }
        CandidateAbsenceReason::RangeStatsOnly => {
            snapshot.candidate_range_stats_only_bucket_count += 1;
        }
        CandidateAbsenceReason::Other => {
            snapshot.candidate_other_absence_bucket_count += 1;
        }
    }
}

fn write_last_filter_planner_diagnostics(
    collection: &Collection,
    snapshot: FilterPlannerDiagnosticsSnapshot,
) {
    *collection.last_filter_planner_diagnostics.write() = snapshot;
}

const CANDIDATE_PUSHDOWN_MAX_SELECTIVITY: f64 = 0.85;

fn should_apply_candidate_pushdown(bucket_live_ids: usize, candidate_count: usize) -> bool {
    if candidate_count == 0 {
        return false;
    }
    if bucket_live_ids == 0 {
        // Keep pushdown enabled when stats are unavailable; broad-case guardrail depends on stats.
        return true;
    }
    let effective_live = bucket_live_ids.max(candidate_count);
    let selectivity = candidate_count as f64 / effective_live as f64;
    selectivity <= CANDIDATE_PUSHDOWN_MAX_SELECTIVITY
}

fn logical_type_for_exact_filter_value(value: &UnifiedPayloadValue) -> Option<UnifiedLogicalType> {
    match value {
        UnifiedPayloadValue::Null => None,
        UnifiedPayloadValue::Bool(_) => Some(UnifiedLogicalType::Bool),
        UnifiedPayloadValue::Int64(_) => Some(UnifiedLogicalType::Int64),
        UnifiedPayloadValue::Float32(_) => Some(UnifiedLogicalType::Float32),
        UnifiedPayloadValue::Float64(_) => Some(UnifiedLogicalType::Float64),
        UnifiedPayloadValue::TimestampMicros(_) => Some(UnifiedLogicalType::TimestampMicros),
        UnifiedPayloadValue::Keyword(_) => Some(UnifiedLogicalType::Keyword),
        UnifiedPayloadValue::Text(_) => Some(UnifiedLogicalType::Text),
        UnifiedPayloadValue::Bytes(_) => Some(UnifiedLogicalType::Bytes),
        UnifiedPayloadValue::LobRef(_) => Some(UnifiedLogicalType::LobRef),
    }
}

fn exact_value_membership_key(
    field_id: u32,
    logical_type: &UnifiedLogicalType,
    value: &UnifiedPayloadValue,
) -> io::Result<Option<ExactValueMembershipKey>> {
    Ok(
        encode_exact_key(logical_type, value)?.map(|encoded_value| ExactValueMembershipKey {
            field_id,
            logical_type_tag: logical_type_tag(logical_type),
            encoded_value,
        }),
    )
}

fn exact_value_membership_key_from_value(
    field_id: u32,
    value: &UnifiedPayloadValue,
) -> io::Result<Option<ExactValueMembershipKey>> {
    let Some(logical_type) = logical_type_for_exact_filter_value(value) else {
        return Ok(None);
    };
    exact_value_membership_key(field_id, &logical_type, value)
}

struct CatalogFilterClauses {
    exact_clauses: Vec<ExactFieldQueryClause>,
    range_clauses: Vec<RangeFieldQueryClause>,
}

fn extract_catalog_filter_clauses(
    filters: &[ParsedFilter],
) -> io::Result<Option<CatalogFilterClauses>> {
    if filters.is_empty() {
        return Ok(None);
    }

    let mut exact_clauses = Vec::new();
    let mut range_clauses = Vec::new();
    for filter in filters {
        match filter {
            ParsedFilter::Exact { field_id, value } => {
                let Some(key) = exact_value_membership_key_from_value(*field_id, value)? else {
                    // Null exact filters are not covered by exact index postings.
                    continue;
                };
                exact_clauses.push(ExactFieldQueryClause {
                    field_id: *field_id,
                    value_keys: vec![key],
                });
            }
            ParsedFilter::AnyOf { field_id, values } => {
                if values
                    .iter()
                    .any(|value| matches!(value, UnifiedPayloadValue::Null))
                {
                    continue;
                }
                let mut seen = HashSet::new();
                let mut keys = Vec::new();
                for value in values {
                    let Some(key) = exact_value_membership_key_from_value(*field_id, value)? else {
                        continue;
                    };
                    if seen.insert(key.clone()) {
                        keys.push(key);
                    }
                }
                if keys.is_empty() {
                    continue;
                }
                exact_clauses.push(ExactFieldQueryClause {
                    field_id: *field_id,
                    value_keys: keys,
                });
            }
            ParsedFilter::Range {
                field_id,
                lower,
                lower_inclusive,
                upper,
                upper_inclusive,
            } => range_clauses.push(RangeFieldQueryClause {
                field_id: *field_id,
                lower: lower.clone(),
                lower_inclusive: *lower_inclusive,
                upper: upper.clone(),
                upper_inclusive: *upper_inclusive,
            }),
        }
    }
    if exact_clauses.is_empty() && range_clauses.is_empty() {
        return Ok(None);
    }
    Ok(Some(CatalogFilterClauses {
        exact_clauses,
        range_clauses,
    }))
}

async fn collect_exact_routing_values_for_fields(
    reader: &UnifiedReader,
    field_ids: &HashSet<u32>,
) -> io::Result<(
    HashMap<u32, HashMap<u64, HashSet<ExactValueMembershipKey>>>,
    HashSet<u32>,
)> {
    if field_ids.is_empty() {
        return Ok((HashMap::new(), HashSet::new()));
    }

    let schema = reader.read_payload_schema().await?;
    let Some(schema) = schema else {
        return Ok((HashMap::new(), HashSet::new()));
    };

    let mut field_values: HashMap<u32, HashMap<u64, HashSet<ExactValueMembershipKey>>> =
        HashMap::new();
    let mut indexed_fields = HashSet::new();
    for field in &schema.fields {
        if !field.indexed || !field_ids.contains(&field.field_id) {
            continue;
        }
        indexed_fields.insert(field.field_id);

        let Some(index) = reader.read_exact_index(field.field_id).await? else {
            continue;
        };
        let logical_type_tag = logical_type_tag(&index.logical_type);
        let per_id = field_values.entry(field.field_id).or_default();
        for (encoded_value, postings) in
            index.dictionary.into_iter().zip(index.postings.into_iter())
        {
            let key = ExactValueMembershipKey {
                field_id: field.field_id,
                logical_type_tag,
                encoded_value,
            };
            for id in postings {
                per_id.entry(id).or_default().insert(key.clone());
            }
        }
    }
    Ok((field_values, indexed_fields))
}

async fn probe_source_with_exact_routing_hydration(
    reader: &UnifiedReader,
    filters: &[ParsedFilter],
    missing_exact_fields: &HashSet<u32>,
    routing_exact_field_values: &mut HashMap<u32, HashMap<u64, HashSet<ExactValueMembershipKey>>>,
    routing_complete_exact_fields: &mut HashSet<u32>,
) -> io::Result<FilterSourceProbe> {
    let probe = source_filter_probe(reader, filters).await?;
    if missing_exact_fields.is_empty() {
        return Ok(probe);
    }

    let (source_field_values, indexed_fields_in_source) =
        collect_exact_routing_values_for_fields(reader, missing_exact_fields).await?;
    for (field_id, source_values_by_id) in source_field_values {
        let merged_values_by_id = routing_exact_field_values.entry(field_id).or_default();
        for (id, keys) in source_values_by_id {
            merged_values_by_id.entry(id).or_default().extend(keys);
        }
    }
    routing_complete_exact_fields.retain(|field_id| indexed_fields_in_source.contains(field_id));
    Ok(probe)
}

fn preselect_bucket_ids_with_global_routing_index(
    routing: &GlobalFilterRoutingIndex,
    probe_bucket_ids: &[u32],
    exact_clauses: &[ExactFieldQueryClause],
) -> Vec<u32> {
    if exact_clauses.is_empty() {
        return probe_bucket_ids.to_vec();
    }

    let clause_buckets: Vec<HashSet<u32>> = exact_clauses
        .iter()
        .map(|clause| routing.buckets_for_exact_clause(clause))
        .collect();
    let mut selected = Vec::with_capacity(probe_bucket_ids.len());

    'bucket: for &bucket_id in probe_bucket_ids {
        for (clause, clause_bucket_ids) in exact_clauses.iter().zip(clause_buckets.iter()) {
            if clause_bucket_ids.contains(&bucket_id) {
                continue;
            }
            if routing.bucket_exact_field_complete(bucket_id, clause.field_id) {
                continue 'bucket;
            }
        }
        selected.push(bucket_id);
    }

    selected
}

fn preselect_probe_buckets_with_global_exact_routing(
    collection: &Collection,
    probe_bucket_ids: &[u32],
    exact_clauses: &[ExactFieldQueryClause],
) -> Vec<u32> {
    let routing = collection.global_filter_routing_index.read();
    preselect_bucket_ids_with_global_routing_index(&routing, probe_bucket_ids, exact_clauses)
}

#[derive(Default, Clone, Copy)]
struct CatalogPreselectionStats {
    input_bucket_count: usize,
    pruned_bucket_count: usize,
    complete_may_match_bucket_count: usize,
    incomplete_bucket_count: usize,
    stale_bucket_count: usize,
    missing_bucket_count: usize,
}

struct CatalogPreselectionResult {
    selected_bucket_ids: Vec<u32>,
    stats: CatalogPreselectionStats,
}

fn preselect_probe_buckets_with_catalog(
    collection: &Collection,
    probe_bucket_ids: &[u32],
    clauses: &CatalogFilterClauses,
) -> CatalogPreselectionResult {
    let mut catalog = collection.filter_metadata_catalog.write();
    let mut selected_bucket_ids = Vec::with_capacity(probe_bucket_ids.len());
    let mut stats = CatalogPreselectionStats {
        input_bucket_count: probe_bucket_ids.len(),
        ..CatalogPreselectionStats::default()
    };

    for &bucket_id in probe_bucket_ids {
        let Some(version) = collection.bucket_manager.get_version(bucket_id) else {
            stats.missing_bucket_count += 1;
            selected_bucket_ids.push(bucket_id);
            continue;
        };
        let live_count = collection
            .bucket_manager
            .get_bucket_stats(bucket_id)
            .map(|stats| stats.total_count.saturating_sub(stats.tombstone_count));
        let Some(live_count) = live_count else {
            stats.missing_bucket_count += 1;
            selected_bucket_ids.push(bucket_id);
            continue;
        };
        let exact_coverage = if clauses.exact_clauses.is_empty() {
            BucketExactClauseCoverage::CompleteMayMatch
        } else {
            catalog.classify_bucket_exact_clauses(
                bucket_id,
                &version.path,
                Some(live_count),
                &clauses.exact_clauses,
            )
        };
        let range_coverage = if clauses.range_clauses.is_empty() {
            BucketRangeClauseCoverage::CompleteMayMatch
        } else {
            catalog.classify_bucket_range_clauses(
                bucket_id,
                &version.path,
                Some(live_count),
                &clauses.range_clauses,
            )
        };

        if matches!(
            exact_coverage,
            BucketExactClauseCoverage::StaleBucketPath
                | BucketExactClauseCoverage::StaleBucketStats
        ) || matches!(
            range_coverage,
            BucketRangeClauseCoverage::StaleBucketPath
                | BucketRangeClauseCoverage::StaleBucketStats
        ) {
            stats.stale_bucket_count += 1;
            catalog.invalidate_bucket(bucket_id);
            selected_bucket_ids.push(bucket_id);
            continue;
        }

        if matches!(exact_coverage, BucketExactClauseCoverage::MissingBucket)
            || matches!(range_coverage, BucketRangeClauseCoverage::MissingBucket)
        {
            stats.missing_bucket_count += 1;
            selected_bucket_ids.push(bucket_id);
            continue;
        }

        if matches!(exact_coverage, BucketExactClauseCoverage::CompleteNoMatch)
            || matches!(range_coverage, BucketRangeClauseCoverage::CompleteNoMatch)
        {
            stats.pruned_bucket_count += 1;
            continue;
        }

        if matches!(exact_coverage, BucketExactClauseCoverage::Incomplete)
            || matches!(range_coverage, BucketRangeClauseCoverage::Incomplete)
        {
            stats.incomplete_bucket_count += 1;
            selected_bucket_ids.push(bucket_id);
            continue;
        }

        stats.complete_may_match_bucket_count += 1;
        selected_bucket_ids.push(bucket_id);
    }

    CatalogPreselectionResult {
        selected_bucket_ids,
        stats,
    }
}

async fn source_filter_probe(
    reader: &UnifiedReader,
    filters: &[ParsedFilter],
) -> io::Result<FilterSourceProbe> {
    if filters.is_empty() {
        return Ok(FilterSourceProbe {
            might_match: true,
            ..FilterSourceProbe::default()
        });
    }

    let schema = reader.read_payload_schema().await?;
    let needs_range_stats = filters
        .iter()
        .any(|filter| matches!(filter, ParsedFilter::Range { .. }));
    let payload_stats = if needs_range_stats {
        let stats = reader.read_payload_stats().await?;
        if stats.is_empty() { None } else { Some(stats) }
    } else {
        None
    };

    let mut probe = FilterSourceProbe {
        might_match: true,
        ..FilterSourceProbe::default()
    };
    let mut used_exact_pushdown = false;

    for filter in filters {
        match filter {
            ParsedFilter::Exact { field_id, value } => {
                probe.saw_exact_filter = true;
                if matches!(value, UnifiedPayloadValue::Null) {
                    continue;
                }

                let Some(schema) = schema.as_ref() else {
                    probe.might_match = false;
                    return Ok(probe);
                };
                let Some(field) = schema.fields.iter().find(|f| f.field_id == *field_id) else {
                    probe.might_match = false;
                    return Ok(probe);
                };

                if !field.indexed {
                    continue;
                }
                probe.saw_indexed_exact_filter = true;
                probe.indexed_exact_field_ids.insert(*field_id);

                let matches: HashSet<u64> = reader
                    .filter_ids_exact(*field_id, value)
                    .await?
                    .into_iter()
                    .collect();
                let membership_key =
                    exact_value_membership_key(*field_id, &field.logical_type, value)?;
                if matches.is_empty() {
                    if let Some(key) = membership_key {
                        probe
                            .exact_value_presence
                            .insert(key, ExactValuePresence::Absent);
                    }
                    probe.saw_empty_exact_match = true;
                    probe.might_match = false;
                    probe.exact_candidate_ids = None;
                    return Ok(probe);
                }

                if let Some(key) = membership_key {
                    probe
                        .exact_value_presence
                        .insert(key, ExactValuePresence::Present);
                }

                used_exact_pushdown = true;
                if let Some(existing) = probe.exact_candidate_ids.as_mut() {
                    intersect_candidate_ids(existing, &matches);
                    if existing.is_empty() {
                        probe.saw_empty_exact_match = true;
                        probe.might_match = false;
                        probe.exact_candidate_ids = None;
                        return Ok(probe);
                    }
                } else {
                    probe.exact_candidate_ids = Some(matches);
                }
            }
            ParsedFilter::AnyOf { field_id, values } => {
                probe.saw_exact_filter = true;
                if values
                    .iter()
                    .any(|value| matches!(value, UnifiedPayloadValue::Null))
                {
                    continue;
                }

                let Some(schema) = schema.as_ref() else {
                    probe.might_match = false;
                    return Ok(probe);
                };
                let Some(field) = schema.fields.iter().find(|f| f.field_id == *field_id) else {
                    probe.might_match = false;
                    return Ok(probe);
                };

                if !field.indexed {
                    continue;
                }
                probe.saw_indexed_exact_filter = true;
                probe.indexed_exact_field_ids.insert(*field_id);

                let mut matches = HashSet::new();
                for value in values {
                    let exact_ids = reader.filter_ids_exact(*field_id, value).await?;
                    let membership_key =
                        exact_value_membership_key(*field_id, &field.logical_type, value)?;
                    if exact_ids.is_empty() {
                        if let Some(key) = membership_key {
                            probe
                                .exact_value_presence
                                .insert(key, ExactValuePresence::Absent);
                        }
                        continue;
                    }
                    matches.extend(exact_ids);
                    if let Some(key) = membership_key {
                        probe
                            .exact_value_presence
                            .insert(key, ExactValuePresence::Present);
                    }
                }
                if matches.is_empty() {
                    probe.saw_empty_exact_match = true;
                    probe.might_match = false;
                    probe.exact_candidate_ids = None;
                    return Ok(probe);
                }

                used_exact_pushdown = true;
                if let Some(existing) = probe.exact_candidate_ids.as_mut() {
                    intersect_candidate_ids(existing, &matches);
                    if existing.is_empty() {
                        probe.saw_empty_exact_match = true;
                        probe.might_match = false;
                        probe.exact_candidate_ids = None;
                        return Ok(probe);
                    }
                } else {
                    probe.exact_candidate_ids = Some(matches);
                }
            }
            ParsedFilter::Range {
                field_id,
                lower,
                lower_inclusive,
                upper,
                upper_inclusive,
            } => {
                probe.saw_range_filter = true;
                probe.range_filter_field_ids.insert(*field_id);
                let Some(schema) = schema.as_ref() else {
                    probe.might_match = false;
                    return Ok(probe);
                };
                if schema
                    .fields
                    .iter()
                    .all(|field| field.field_id != *field_id)
                {
                    probe.might_match = false;
                    return Ok(probe);
                }

                let Some(stats) = payload_stats.as_ref() else {
                    continue;
                };
                if let Some(zone_map) = aggregate_range_zone_map_for_field(stats, *field_id) {
                    probe.range_field_zone_maps.insert(*field_id, zone_map);
                }
                let any_chunk_matches = stats.iter().any(|chunk| {
                    range_filter_might_match_chunk(
                        chunk,
                        *field_id,
                        lower.as_ref(),
                        *lower_inclusive,
                        upper.as_ref(),
                        *upper_inclusive,
                    )
                });
                if !any_chunk_matches {
                    probe.might_match = false;
                    return Ok(probe);
                }
            }
        }
    }

    if !used_exact_pushdown {
        probe.exact_candidate_ids = None;
    }
    Ok(probe)
}

async fn plan_filter_aware_execution(
    collection: &Collection,
    planning_bucket_ids: &[u32],
    routed_bucket_ids: &[u32],
    query: &[f32],
    filters: &[ParsedFilter],
) -> FilterAwareExecutionPlan {
    let diagnostics_enabled = diagnostics_enabled_from_env();
    let mut diagnostics = FilterPlannerDiagnosticsSnapshot {
        enabled: diagnostics_enabled,
        query_has_filters: !filters.is_empty(),
        ..FilterPlannerDiagnosticsSnapshot::default()
    };

    if filters.is_empty() {
        let plan = FilterAwareExecutionPlan {
            bucket_ids: routed_bucket_ids.to_vec(),
            candidate_ids: HashMap::new(),
        };
        write_last_filter_planner_diagnostics(collection, diagnostics);
        return plan;
    }

    let mut probe_bucket_ids = if planning_bucket_ids.is_empty() {
        routed_bucket_ids.to_vec()
    } else {
        planning_bucket_ids.to_vec()
    };
    let mut queried_exact_field_ids = HashSet::new();
    match extract_catalog_filter_clauses(filters) {
        Ok(Some(clauses)) => {
            queried_exact_field_ids
                .extend(clauses.exact_clauses.iter().map(|clause| clause.field_id));
            if !clauses.exact_clauses.is_empty() {
                let global_exact_input_bucket_count = probe_bucket_ids.len();
                let routed = preselect_probe_buckets_with_global_exact_routing(
                    collection,
                    &probe_bucket_ids,
                    &clauses.exact_clauses,
                );
                let global_exact_pruned_bucket_count =
                    global_exact_input_bucket_count.saturating_sub(routed.len());
                if routed.len() < probe_bucket_ids.len() {
                    tracing::debug!(
                        "Filter planner: global exact routing pruned {} bucket(s) from probe set of {}",
                        probe_bucket_ids.len().saturating_sub(routed.len()),
                        probe_bucket_ids.len()
                    );
                }
                if diagnostics_enabled {
                    diagnostics.global_exact_preselect_eligible_query = true;
                    diagnostics.global_exact_preselect_input_bucket_count =
                        global_exact_input_bucket_count;
                    diagnostics.global_exact_preselect_pruned_bucket_count =
                        global_exact_pruned_bucket_count;
                }
                probe_bucket_ids = routed;
            }
            let result =
                preselect_probe_buckets_with_catalog(collection, &probe_bucket_ids, &clauses);
            if diagnostics_enabled {
                diagnostics.catalog_exact_clause_eligible_query = true;
                diagnostics.catalog_preselect_input_bucket_count = result.stats.input_bucket_count;
                diagnostics.catalog_preselect_pruned_bucket_count =
                    result.stats.pruned_bucket_count;
                diagnostics.catalog_preselect_complete_may_match_bucket_count =
                    result.stats.complete_may_match_bucket_count;
                diagnostics.catalog_preselect_incomplete_bucket_count =
                    result.stats.incomplete_bucket_count;
                diagnostics.catalog_preselect_stale_bucket_count = result.stats.stale_bucket_count;
                diagnostics.catalog_preselect_missing_bucket_count =
                    result.stats.missing_bucket_count;
            }
            if result.stats.pruned_bucket_count > 0 {
                tracing::debug!(
                    "Filter planner: catalog preselection pruned {} bucket(s) from probe set of {}",
                    result.stats.pruned_bucket_count,
                    probe_bucket_ids.len()
                );
            }
            probe_bucket_ids = result.selected_bucket_ids;
        }
        Ok(None) => {}
        Err(err) => {
            tracing::debug!(
                "Filter planner: failed to derive catalog clauses: {}; falling back to full probe set",
                err
            );
        }
    }
    if probe_bucket_ids.is_empty() {
        let plan = FilterAwareExecutionPlan {
            bucket_ids: routed_bucket_ids.to_vec(),
            candidate_ids: HashMap::new(),
        };
        write_last_filter_planner_diagnostics(collection, diagnostics);
        return plan;
    }

    let local_op = match create_staging_operator(collection) {
        Ok(op) => op,
        Err(err) => {
            tracing::debug!(
                "Filter planner: failed to create staging operator: {}; disabling pushdown",
                err
            );
            if diagnostics_enabled {
                diagnostics.candidate_disabled_probe_error_bucket_count = probe_bucket_ids.len();
            }
            let plan = FilterAwareExecutionPlan {
                bucket_ids: routed_bucket_ids.to_vec(),
                candidate_ids: HashMap::new(),
            };
            write_last_filter_planner_diagnostics(collection, diagnostics);
            return plan;
        }
    };
    let remote_op = collection.persistence.operator();

    let mut plan = FilterAwareExecutionPlan {
        bucket_ids: Vec::with_capacity(probe_bucket_ids.len()),
        candidate_ids: HashMap::new(),
    };
    for &bucket_id in &probe_bucket_ids {
        if diagnostics_enabled {
            diagnostics.probed_bucket_count += 1;
        }

        let Some(version) = collection.bucket_manager.get_version(bucket_id) else {
            plan.bucket_ids.push(bucket_id);
            if diagnostics_enabled {
                diagnostics.kept_bucket_count += 1;
                record_candidate_decision(
                    &mut diagnostics,
                    false,
                    false,
                    false,
                    false,
                    0,
                    false,
                    false,
                    false,
                    false,
                );
            }
            continue;
        };
        let bucket_path = version.path.clone();
        let sources = payload_sources_for_class(&version.class, &version.path);
        let missing_routing_exact_fields: HashSet<u32> = if queried_exact_field_ids.is_empty() {
            HashSet::new()
        } else {
            let routing = collection.global_filter_routing_index.read();
            queried_exact_field_ids
                .iter()
                .copied()
                .filter(|field_id| !routing.bucket_exact_field_complete(bucket_id, *field_id))
                .collect()
        };
        let mut routing_exact_field_values: HashMap<
            u32,
            HashMap<u64, HashSet<ExactValueMembershipKey>>,
        > = HashMap::new();
        let mut routing_complete_exact_fields = missing_routing_exact_fields.clone();

        let mut keep_bucket = false;
        let mut disable_candidates = false;
        let mut bucket_candidates = HashSet::new();
        let mut has_bucket_candidates = false;
        let mut saw_exact_filter = false;
        let mut saw_indexed_exact_filter = false;
        let mut saw_empty_exact_match = false;
        let mut saw_range_filter = false;
        let mut candidate_applied = false;
        let mut candidate_gated = false;
        let mut observed_indexed_exact_fields = HashSet::new();
        let mut observed_range_stats_fields = HashSet::new();
        let mut observed_exact_value_presence: HashMap<
            ExactValueMembershipKey,
            ExactValuePresence,
        > = HashMap::new();
        let mut observed_range_field_zone_maps: HashMap<u32, RangeFieldZoneMap> = HashMap::new();

        for source in sources {
            let probe = match source {
                PayloadSource::Local(path) => {
                    match UnifiedReader::open(local_op.clone(), &path).await {
                        Ok(reader) => {
                            probe_source_with_exact_routing_hydration(
                                &reader,
                                filters,
                                &missing_routing_exact_fields,
                                &mut routing_exact_field_values,
                                &mut routing_complete_exact_fields,
                            )
                            .await
                        }
                        Err(err) => Err(err),
                    }
                }
                PayloadSource::Remote(path) => {
                    match UnifiedReader::open(remote_op.clone(), &path).await {
                        Ok(reader) => {
                            probe_source_with_exact_routing_hydration(
                                &reader,
                                filters,
                                &missing_routing_exact_fields,
                                &mut routing_exact_field_values,
                                &mut routing_complete_exact_fields,
                            )
                            .await
                        }
                        Err(err) => Err(err),
                    }
                }
            };

            match probe {
                Ok(probe) => {
                    let FilterSourceProbe {
                        might_match,
                        exact_candidate_ids,
                        saw_exact_filter: source_saw_exact_filter,
                        saw_indexed_exact_filter: source_saw_indexed_exact_filter,
                        saw_empty_exact_match: source_saw_empty_exact_match,
                        saw_range_filter: source_saw_range_filter,
                        indexed_exact_field_ids,
                        range_filter_field_ids,
                        exact_value_presence,
                        range_field_zone_maps,
                    } = probe;
                    saw_exact_filter |= source_saw_exact_filter;
                    saw_indexed_exact_filter |= source_saw_indexed_exact_filter;
                    saw_empty_exact_match |= source_saw_empty_exact_match;
                    saw_range_filter |= source_saw_range_filter;
                    observed_indexed_exact_fields.extend(indexed_exact_field_ids);
                    observed_range_stats_fields.extend(range_filter_field_ids);
                    for (key, source_presence) in exact_value_presence {
                        observed_exact_value_presence
                            .entry(key)
                            .and_modify(|existing| {
                                if matches!(source_presence, ExactValuePresence::Present) {
                                    *existing = ExactValuePresence::Present;
                                }
                            })
                            .or_insert(source_presence);
                    }
                    observed_range_field_zone_maps.extend(range_field_zone_maps);
                    if !might_match {
                        continue;
                    }
                    keep_bucket = true;
                    if disable_candidates {
                        continue;
                    }
                    if let Some(source_candidates) = exact_candidate_ids {
                        has_bucket_candidates = true;
                        bucket_candidates.extend(source_candidates);
                    }
                }
                Err(err) => {
                    tracing::debug!(
                        "Filter planner: metadata probe failed for bucket {}: {}; keeping bucket",
                        bucket_id,
                        err
                    );
                    keep_bucket = true;
                    disable_candidates = true;
                }
            }
        }

        if !disable_candidates && !missing_routing_exact_fields.is_empty() {
            let mut routing = collection.global_filter_routing_index.write();
            for field_id in &missing_routing_exact_fields {
                let field_values = routing_exact_field_values
                    .remove(field_id)
                    .unwrap_or_default();
                routing.replace_bucket_exact_field_values(bucket_id, *field_id, field_values);
                if routing_complete_exact_fields.contains(field_id) {
                    routing.mark_bucket_exact_field_complete(bucket_id, *field_id);
                }
            }
        }

        if !disable_candidates {
            collection
                .filter_metadata_catalog
                .write()
                .observe_bucket_probe(
                    bucket_id,
                    BucketProbeObservation {
                        bucket_path,
                        bucket_live_count: collection
                            .bucket_manager
                            .get_bucket_stats(bucket_id)
                            .map(|stats| stats.total_count.saturating_sub(stats.tombstone_count)),
                        indexed_exact_fields: observed_indexed_exact_fields,
                        range_stats_fields: observed_range_stats_fields,
                        exact_value_presence: observed_exact_value_presence,
                        range_field_zone_maps: observed_range_field_zone_maps,
                    },
                );
        }

        if diagnostics_enabled {
            if keep_bucket {
                diagnostics.kept_bucket_count += 1;
            } else {
                diagnostics.pruned_bucket_count += 1;
            }
        }

        let candidate_count = bucket_candidates.len();
        if keep_bucket {
            plan.bucket_ids.push(bucket_id);
            if !disable_candidates && has_bucket_candidates {
                let bucket_live_ids = collection
                    .bucket_manager
                    .get_bucket_stats(bucket_id)
                    .map(|stats| stats.total_count.saturating_sub(stats.tombstone_count) as usize)
                    .unwrap_or(0);
                if should_apply_candidate_pushdown(bucket_live_ids, bucket_candidates.len()) {
                    plan.candidate_ids.insert(bucket_id, bucket_candidates);
                    candidate_applied = true;
                } else {
                    tracing::debug!(
                        "Filter planner: skipping candidate pushdown for bucket {} due to broad selectivity (candidates={}, live_ids={})",
                        bucket_id,
                        bucket_candidates.len(),
                        bucket_live_ids
                    );
                    candidate_gated = true;
                }
            }
        }

        if diagnostics_enabled {
            record_candidate_decision(
                &mut diagnostics,
                has_bucket_candidates,
                candidate_applied,
                candidate_gated,
                disable_candidates,
                candidate_count,
                saw_exact_filter,
                saw_indexed_exact_filter,
                saw_empty_exact_match,
                saw_range_filter,
            );
        }
    }
    if plan.bucket_ids.is_empty() {
        let plan = FilterAwareExecutionPlan {
            bucket_ids: routed_bucket_ids.to_vec(),
            candidate_ids: HashMap::new(),
        };
        write_last_filter_planner_diagnostics(collection, diagnostics);
        return plan;
    }
    plan.bucket_ids = collection
        .index
        .rank_bucket_ids_by_query_distance(query, &plan.bucket_ids);
    write_last_filter_planner_diagnostics(collection, diagnostics);
    plan
}

#[cfg(test)]
mod planner_heuristic_tests {
    use super::{
        CandidateAbsenceReason, FilterPlannerDiagnosticsSnapshot,
        classify_candidate_absence_reason, preselect_bucket_ids_with_global_routing_index,
        record_candidate_decision, should_apply_candidate_pushdown,
    };
    use crate::filter_metadata_catalog::{ExactFieldQueryClause, ExactValueMembershipKey};
    use crate::global_filter_routing_index::GlobalFilterRoutingIndex;
    use std::collections::HashSet;

    fn exact_key(field_id: u32, value: &str) -> ExactValueMembershipKey {
        ExactValueMembershipKey {
            field_id,
            logical_type_tag: 1,
            encoded_value: value.as_bytes().to_vec(),
        }
    }

    #[test]
    fn global_routing_preselect_prunes_only_complete_absent_buckets() {
        let mut routing = GlobalFilterRoutingIndex::default();
        let tenant_a = exact_key(1, "tenant_a");

        routing.upsert_id_values(10, 1, vec![tenant_a.clone()]);
        routing.set_bucket_complete_exact_fields(1, HashSet::from([1]));
        routing.set_bucket_complete_exact_fields(2, HashSet::from([1]));

        let selected = preselect_bucket_ids_with_global_routing_index(
            &routing,
            &[1, 2],
            &[ExactFieldQueryClause {
                field_id: 1,
                value_keys: vec![tenant_a.clone()],
            }],
        );

        assert_eq!(selected, vec![1]);
    }

    #[test]
    fn global_routing_preselect_keeps_incomplete_absent_buckets() {
        let mut routing = GlobalFilterRoutingIndex::default();
        let tenant_a = exact_key(1, "tenant_a");

        routing.upsert_id_values(10, 1, vec![tenant_a.clone()]);
        routing.set_bucket_complete_exact_fields(1, HashSet::from([1]));

        let selected = preselect_bucket_ids_with_global_routing_index(
            &routing,
            &[1, 2],
            &[ExactFieldQueryClause {
                field_id: 1,
                value_keys: vec![tenant_a],
            }],
        );

        assert_eq!(selected, vec![1, 2]);
    }

    #[test]
    fn candidate_pushdown_disables_broad_selectivity() {
        assert!(!should_apply_candidate_pushdown(100, 95));
        assert!(!should_apply_candidate_pushdown(1_000, 900));
    }

    #[test]
    fn candidate_pushdown_keeps_selective_filters() {
        assert!(should_apply_candidate_pushdown(100, 10));
        assert!(should_apply_candidate_pushdown(1_000, 200));
    }

    #[test]
    fn candidate_pushdown_keeps_candidates_when_stats_missing() {
        assert!(should_apply_candidate_pushdown(0, 16));
    }

    #[test]
    fn candidate_absence_classification_precedence() {
        assert_eq!(
            classify_candidate_absence_reason(true, true, true, false),
            CandidateAbsenceReason::EmptyExactMatch
        );
        assert_eq!(
            classify_candidate_absence_reason(true, false, false, true),
            CandidateAbsenceReason::NoIndexedExact
        );
        assert_eq!(
            classify_candidate_absence_reason(false, false, false, true),
            CandidateAbsenceReason::RangeStatsOnly
        );
        assert_eq!(
            classify_candidate_absence_reason(false, false, false, false),
            CandidateAbsenceReason::Other
        );
    }

    #[test]
    fn candidate_decision_records_produced_and_applied() {
        let mut snapshot = FilterPlannerDiagnosticsSnapshot::default();
        record_candidate_decision(
            &mut snapshot,
            true,
            true,
            false,
            false,
            9,
            true,
            true,
            false,
            false,
        );
        assert_eq!(snapshot.candidate_produced_bucket_count, 1);
        assert_eq!(snapshot.candidate_applied_bucket_count, 1);
        assert_eq!(snapshot.candidate_gated_broad_selectivity_bucket_count, 0);
        assert_eq!(snapshot.candidate_id_count, 9);
    }

    #[test]
    fn candidate_decision_records_produced_and_gated() {
        let mut snapshot = FilterPlannerDiagnosticsSnapshot::default();
        record_candidate_decision(
            &mut snapshot,
            true,
            false,
            true,
            false,
            12,
            true,
            true,
            false,
            false,
        );
        assert_eq!(snapshot.candidate_produced_bucket_count, 1);
        assert_eq!(snapshot.candidate_applied_bucket_count, 0);
        assert_eq!(snapshot.candidate_gated_broad_selectivity_bucket_count, 1);
        assert_eq!(snapshot.candidate_id_count, 12);
    }

    #[test]
    fn candidate_decision_records_probe_error_disable() {
        let mut snapshot = FilterPlannerDiagnosticsSnapshot::default();
        record_candidate_decision(
            &mut snapshot,
            false,
            false,
            false,
            true,
            0,
            true,
            true,
            false,
            false,
        );
        assert_eq!(snapshot.candidate_disabled_probe_error_bucket_count, 1);
    }

    #[test]
    fn candidate_decision_records_empty_exact_match_reason() {
        let mut snapshot = FilterPlannerDiagnosticsSnapshot::default();
        record_candidate_decision(
            &mut snapshot,
            false,
            false,
            false,
            false,
            0,
            true,
            true,
            true,
            false,
        );
        assert_eq!(snapshot.candidate_empty_exact_match_bucket_count, 1);
    }

    #[test]
    fn candidate_decision_records_no_indexed_exact_reason() {
        let mut snapshot = FilterPlannerDiagnosticsSnapshot::default();
        record_candidate_decision(
            &mut snapshot,
            false,
            false,
            false,
            false,
            0,
            true,
            false,
            false,
            false,
        );
        assert_eq!(snapshot.candidate_no_indexed_exact_bucket_count, 1);
    }

    #[test]
    fn candidate_decision_records_range_stats_only_reason() {
        let mut snapshot = FilterPlannerDiagnosticsSnapshot::default();
        record_candidate_decision(
            &mut snapshot,
            false,
            false,
            false,
            false,
            0,
            false,
            false,
            false,
            true,
        );
        assert_eq!(snapshot.candidate_range_stats_only_bucket_count, 1);
    }
}

async fn load_bucket_payload_rows(
    collection: &Collection,
    bucket_id: u32,
) -> io::Result<HashMap<u64, UnifiedPayloadRow>> {
    let version = collection
        .bucket_manager
        .get_version(bucket_id)
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "bucket not found"))?;
    let sources = payload_sources_for_class(&version.class, &version.path);

    let mut rows_by_id = HashMap::new();
    for source in sources {
        let (ids, rows_opt) = match source {
            PayloadSource::Local(path) => {
                let (ids, _) = collection.staging.read_file_content_flat(&path).await?;
                let rows = collection.staging.read_file_payload_rows(&path).await?;
                (ids, rows)
            }
            PayloadSource::Remote(path) => {
                let (ids, _) = collection
                    .persistence
                    .read_remote_bucket_path_flat(&path)
                    .await?;
                let rows = collection
                    .persistence
                    .read_remote_bucket_payload_rows_path_optional(&path)
                    .await?;
                (ids, rows)
            }
        };

        let Some(rows) = rows_opt else {
            continue;
        };
        if rows.len() != ids.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "bucket {} payload row mismatch: ids={}, payload_rows={}",
                    bucket_id,
                    ids.len(),
                    rows.len()
                ),
            ));
        }

        for (id, row) in ids.into_iter().zip(rows.into_iter()) {
            rows_by_id.insert(id, row);
        }
    }

    Ok(rows_by_id)
}

#[tonic::async_trait]
impl Drift for DriftService {
    async fn health(
        &self,
        _request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        let (cache_enabled, cache_snapshot) = match DiskManager::global_nvme_cache_metrics() {
            Some(snapshot) => (true, snapshot),
            None => (false, Default::default()),
        };
        let recovery_guard = RecoveryManager::global_fingerprint_guard_metrics();

        Ok(Response::new(HealthResponse {
            ready: true,
            version: env!("CARGO_PKG_VERSION").to_string(),
            nvme_cache: Some(NvmeCacheMetrics {
                enabled: cache_enabled,
                hits: cache_snapshot.hits,
                misses: cache_snapshot.misses,
                remote_fetches: cache_snapshot.remote_fetches,
                singleflight_waits: cache_snapshot.singleflight_waits,
                evictions: cache_snapshot.evictions,
                bytes_cached: cache_snapshot.bytes_cached,
                bytes_evicted: cache_snapshot.bytes_evicted,
                invalidations: cache_snapshot.invalidations,
                fingerprint_mismatches: cache_snapshot.fingerprint_mismatches,
                recovered_entries: cache_snapshot.recovered_entries,
            }),
            recovery_guard: Some(RecoveryGuardMetrics {
                mismatches_detected: recovery_guard.mismatches_detected,
                invalidations_performed: recovery_guard.invalidations_performed,
                fail_fast_aborts: recovery_guard.fail_fast_aborts,
                payload_index_mismatches_detected: recovery_guard.payload_index_mismatches_detected,
                payload_index_validation_errors: recovery_guard.payload_index_validation_errors,
            }),
        }))
    }

    async fn create_collection(
        &self,
        request: Request<CreateCollectionRequest>,
    ) -> Result<Response<CreateCollectionResponse>, Status> {
        let req = request.into_inner();
        if req.collection_name.trim().is_empty() {
            return Err(Status::invalid_argument("collection_name cannot be empty"));
        }
        if req.dim == 0 {
            return Err(Status::invalid_argument("dim must be > 0"));
        }

        let metric = metric_from_proto(req.metric)?;
        let max_bucket_capacity = if req.max_bucket_capacity == 0 {
            None
        } else {
            Some(req.max_bucket_capacity as usize)
        };

        self.manager
            .get_or_create(
                &req.collection_name,
                Some(req.dim as usize),
                max_bucket_capacity,
                Some(metric),
            )
            .await
            .map_err(map_collection_error)?;

        Ok(Response::new(CreateCollectionResponse { success: true }))
    }

    async fn create_payload_schema(
        &self,
        request: Request<CreatePayloadSchemaRequest>,
    ) -> Result<Response<PayloadSchemaResponse>, Status> {
        let req = request.into_inner();
        if req.collection_name.trim().is_empty() {
            return Err(Status::invalid_argument("collection_name cannot be empty"));
        }
        let schema_req = req
            .schema
            .ok_or_else(|| Status::invalid_argument("schema is required"))?;
        let schema = proto_payload_schema_to_core(schema_req)?;

        let collection = self
            .manager
            .get_or_create(&req.collection_name, None, None, None)
            .await
            .map_err(map_collection_error)?;

        {
            let mut guard = collection.payload_schema.write();
            if guard.is_some() {
                return Err(Status::already_exists(
                    "payload schema already exists; use UpdatePayloadSchema",
                ));
            }
            *guard = Some(schema.clone());
        }

        Ok(Response::new(PayloadSchemaResponse {
            success: true,
            schema: Some(core_payload_schema_to_proto(&schema)),
        }))
    }

    async fn update_payload_schema(
        &self,
        request: Request<UpdatePayloadSchemaRequest>,
    ) -> Result<Response<PayloadSchemaResponse>, Status> {
        let req = request.into_inner();
        if req.collection_name.trim().is_empty() {
            return Err(Status::invalid_argument("collection_name cannot be empty"));
        }
        let schema_req = req
            .schema
            .ok_or_else(|| Status::invalid_argument("schema is required"))?;
        let schema = proto_payload_schema_to_core(schema_req)?;

        let collection = self
            .manager
            .get_or_create(&req.collection_name, None, None, None)
            .await
            .map_err(map_collection_error)?;

        if collection.index.memtable_len() > 0
            || collection.index.get_frozen_count() > 0
            || collection.bucket_manager.bucket_count() > 0
        {
            return Err(Status::failed_precondition(
                "updating payload schema requires an empty collection (no buffered or persisted vectors)",
            ));
        }

        {
            let mut guard = collection.payload_schema.write();
            if guard.is_none() {
                return Err(Status::not_found(
                    "payload schema does not exist; use CreatePayloadSchema first",
                ));
            }
            *guard = Some(schema.clone());
        }

        Ok(Response::new(PayloadSchemaResponse {
            success: true,
            schema: Some(core_payload_schema_to_proto(&schema)),
        }))
    }

    async fn get_payload_schema(
        &self,
        request: Request<GetPayloadSchemaRequest>,
    ) -> Result<Response<GetPayloadSchemaResponse>, Status> {
        let req = request.into_inner();
        if req.collection_name.trim().is_empty() {
            return Err(Status::invalid_argument("collection_name cannot be empty"));
        }

        let collection = self
            .manager
            .get_or_create(&req.collection_name, None, None, None)
            .await
            .map_err(map_collection_error)?;
        let schema = collection.payload_schema.read().clone();

        Ok(Response::new(GetPayloadSchemaResponse {
            found: schema.is_some(),
            schema: schema.as_ref().map(core_payload_schema_to_proto),
        }))
    }

    async fn validate_payload(
        &self,
        request: Request<ValidatePayloadRequest>,
    ) -> Result<Response<ValidatePayloadResponse>, Status> {
        let req = request.into_inner();
        if req.collection_name.trim().is_empty() {
            return Err(Status::invalid_argument("collection_name cannot be empty"));
        }
        let collection = self
            .manager
            .get_or_create(&req.collection_name, None, None, None)
            .await
            .map_err(map_collection_error)?;
        let Some(schema) = collection.payload_schema.read().clone() else {
            return Err(Status::failed_precondition(
                "payload schema is not configured for collection",
            ));
        };

        let mut rows = Vec::with_capacity(req.rows.len());
        for row in req.rows {
            rows.push(proto_payload_row_to_core(row)?);
        }
        let errors = validate_payload_rows_against_schema(rows.as_slice(), &schema);

        Ok(Response::new(ValidatePayloadResponse {
            valid: errors.is_empty(),
            errors,
        }))
    }

    async fn train(
        &self,
        request: Request<TrainRequest>,
    ) -> Result<Response<TrainResponse>, Status> {
        let req = request.into_inner();

        // Extract dimension from training data
        let dim_hint = req.vectors.first().map(|v| v.values.len());

        let collection = self
            .manager
            .get_or_create(&req.collection_name, dim_hint, None, None)
            .await
            .map_err(map_collection_error)?;

        let batch: Vec<(u64, Vec<f32>)> =
            req.vectors.into_iter().map(|v| (v.id, v.values)).collect();

        if let Some(schema) = collection.payload_schema.read().clone() {
            let payload_rows = vec![BTreeMap::new(); batch.len()];
            let errors = validate_payload_rows_against_schema(payload_rows.as_slice(), &schema);
            if !errors.is_empty() {
                return Err(Status::invalid_argument(format!(
                    "payload validation failed: {}",
                    errors.join("; ")
                )));
            }
            collection
                .index
                .insert_batch_with_payload(&batch, Some(&schema), Some(payload_rows.as_slice()))
                .map_err(|e| Status::internal(e.to_string()))?;
        } else {
            // Treat training data as just another batch of inserts.
            // The Janitor will see the volume and trigger training automatically.
            collection
                .index
                .insert_batch_with_payload(&batch, None, None)
                .map_err(|e| Status::internal(e.to_string()))?;
        }

        Ok(Response::new(TrainResponse { success: true }))
    }

    async fn insert(
        &self,
        request: Request<InsertRequest>,
    ) -> Result<Response<InsertResponse>, Status> {
        let req = request.into_inner();
        let collection_name = req.collection_name;
        if req.payload.is_some() && req.vector.is_none() {
            return Err(Status::invalid_argument(
                "payload provided without vector in InsertRequest",
            ));
        }

        let dim_hint = req.vector.as_ref().map(|v| v.values.len());

        let collection = self
            .manager
            .get_or_create(&collection_name, dim_hint, None, None) // Add dim hint support in proto if needed
            .await
            .map_err(map_collection_error)?;

        if let Some(vec) = req.vector {
            let configured_schema = collection.payload_schema.read().clone();
            let payload_row = req.payload.map(proto_payload_row_to_core).transpose()?;
            let payload_row = payload_row.unwrap_or_default();

            let (payload_schema, payload_row_for_insert) = if let Some(schema) = configured_schema {
                let errors = validate_payload_rows_against_schema(
                    std::slice::from_ref(&payload_row),
                    &schema,
                );
                if !errors.is_empty() {
                    return Err(Status::invalid_argument(format!(
                        "payload validation failed: {}",
                        errors.join("; ")
                    )));
                }
                (Some(schema), Some(payload_row))
            } else {
                let payload_row = if payload_row.is_empty() {
                    None
                } else {
                    Some(payload_row)
                };
                let payload_schema = if let Some(row) = payload_row.as_ref() {
                    infer_core_payload_schema(std::slice::from_ref(row))?
                } else {
                    None
                };
                (payload_schema, payload_row)
            };

            collection
                .index
                .insert_with_payload(
                    vec.id,
                    &vec.values,
                    payload_schema.as_ref(),
                    payload_row_for_insert.as_ref(),
                )
                .map_err(|e| Status::internal(e.to_string()))?;
        }

        Ok(Response::new(InsertResponse { success: true }))
    }

    async fn insert_batch(
        &self,
        request: Request<InsertBatchRequest>,
    ) -> Result<Response<InsertResponse>, Status> {
        let req = request.into_inner();
        if !req.payload_rows.is_empty() && req.payload_rows.len() != req.vectors.len() {
            return Err(Status::invalid_argument(format!(
                "payload_rows length mismatch: vectors={}, payload_rows={}",
                req.vectors.len(),
                req.payload_rows.len()
            )));
        }

        let dim_hint = req.vectors.first().map(|v| v.values.len());

        let collection = self
            .manager
            .get_or_create(&req.collection_name, dim_hint, None, None)
            .await
            .map_err(map_collection_error)?;

        let batch: Vec<(u64, Vec<f32>)> =
            req.vectors.into_iter().map(|v| (v.id, v.values)).collect();
        if batch.is_empty() {
            return Ok(Response::new(InsertResponse { success: true }));
        }

        let configured_schema = collection.payload_schema.read().clone();
        if let Some(schema) = configured_schema {
            let payload_rows = if req.payload_rows.is_empty() {
                vec![BTreeMap::new(); batch.len()]
            } else {
                let mut rows: Vec<CorePayloadRow> = Vec::with_capacity(req.payload_rows.len());
                for row in req.payload_rows {
                    rows.push(proto_payload_row_to_core(row)?);
                }
                rows
            };

            let errors = validate_payload_rows_against_schema(payload_rows.as_slice(), &schema);
            if !errors.is_empty() {
                return Err(Status::invalid_argument(format!(
                    "payload validation failed: {}",
                    errors.join("; ")
                )));
            }

            collection
                .index
                .insert_batch_with_payload(&batch, Some(&schema), Some(payload_rows.as_slice()))
                .map_err(|e| Status::internal(e.to_string()))?;
        } else {
            if req.payload_rows.is_empty() {
                collection
                    .index
                    .insert_batch_with_payload(&batch, None, None)
                    .map_err(|e| Status::internal(e.to_string()))?;
            } else {
                let mut payload_rows: Vec<CorePayloadRow> =
                    Vec::with_capacity(req.payload_rows.len());
                for row in req.payload_rows {
                    payload_rows.push(proto_payload_row_to_core(row)?);
                }
                let payload_schema = infer_core_payload_schema(&payload_rows)?;
                if payload_schema.is_none() && payload_rows.iter().any(|row| !row.is_empty()) {
                    return Err(Status::invalid_argument(
                        "unable to infer payload schema for non-empty payload rows",
                    ));
                }
                if payload_schema.is_none() {
                    collection
                        .index
                        .insert_batch_with_payload(&batch, None, None)
                        .map_err(|e| Status::internal(e.to_string()))?;
                } else {
                    collection
                        .index
                        .insert_batch_with_payload(
                            &batch,
                            payload_schema.as_ref(),
                            Some(payload_rows.as_slice()),
                        )
                        .map_err(|e| Status::internal(e.to_string()))?;
                }
            }
        }

        Ok(Response::new(InsertResponse { success: true }))
    }

    async fn search(
        &self,
        request: Request<SearchRequest>,
    ) -> Result<Response<SearchResponse>, Status> {
        let req = request.into_inner();
        let dim_hint = Some(req.vector.len());
        let collection = self
            .manager
            .get_or_create(&req.collection_name, dim_hint, None, None)
            .await
            .map_err(map_collection_error)?;
        let requested_k = req.k.max(1) as usize;
        let parsed_filters = parse_filters(&req.filters)?;
        let projection: HashSet<u32> = req.payload_projection_fields.iter().copied().collect();
        let needs_payload_eval = !parsed_filters.is_empty() || !projection.is_empty();
        let filter_plan = if parsed_filters.is_empty() {
            write_last_filter_planner_diagnostics(
                &collection,
                FilterPlannerDiagnosticsSnapshot {
                    enabled: diagnostics_enabled_from_env(),
                    query_has_filters: false,
                    ..FilterPlannerDiagnosticsSnapshot::default()
                },
            );
            None
        } else {
            let planning_bucket_ids = collection.index.all_routable_bucket_ids();
            let routed_bucket_ids = collection.index.select_buckets(
                &req.vector,
                req.target_confidence,
                req.lambda,
                req.tau,
            );
            Some(
                plan_filter_aware_execution(
                    &collection,
                    &planning_bucket_ids,
                    &routed_bucket_ids,
                    &req.vector,
                    &parsed_filters,
                )
                .await,
            )
        };
        let bucket_hint = filter_plan.as_ref().map(|p| p.bucket_ids.as_slice());
        let candidate_ids_hint = filter_plan
            .as_ref()
            .and_then(|p| (!p.candidate_ids.is_empty()).then_some(&p.candidate_ids));
        let candidate_k = if needs_payload_eval {
            requested_k.saturating_mul(8).clamp(requested_k, 8192)
        } else {
            requested_k
        };

        let results = collection
            .index
            .search_with_hints(
                &req.vector,
                candidate_k,
                req.target_confidence,
                req.lambda,
                req.tau,
                bucket_hint,
                candidate_ids_hint,
            )
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        if !needs_payload_eval {
            let search_results = results
                .into_iter()
                .map(|(id, score)| SearchResult {
                    id,
                    score,
                    payload: None,
                })
                .collect();
            return Ok(Response::new(SearchResponse {
                results: search_results,
            }));
        }

        let mut bucket_candidates: HashMap<u32, Vec<u64>> = HashMap::new();
        let mut unresolved_ids = Vec::new();
        for (id, _) in &results {
            match collection.index.get_kv().get(&id.to_le_bytes()) {
                Ok(Some(raw)) => {
                    if let Some(bucket_id) = decode_bucket_id(&raw) {
                        bucket_candidates.entry(bucket_id).or_default().push(*id);
                    } else {
                        unresolved_ids.push(*id);
                    }
                }
                Ok(None) | Err(_) => unresolved_ids.push(*id),
            }
        }

        let mut payload_by_id: HashMap<u64, UnifiedPayloadRow> = HashMap::new();
        for (bucket_id, ids) in bucket_candidates {
            match load_bucket_payload_rows(&collection, bucket_id).await {
                Ok(rows) => {
                    for id in ids {
                        if let Some(row) = rows.get(&id) {
                            payload_by_id.insert(id, row.clone());
                        } else {
                            unresolved_ids.push(id);
                        }
                    }
                }
                Err(err) if err.kind() == io::ErrorKind::NotFound => {
                    tracing::debug!(
                        "Search payload lookup: bucket {} missing (likely concurrent maintenance); retrying unresolved IDs via L0 fallback",
                        bucket_id
                    );
                    unresolved_ids.extend(ids);
                }
                Err(err) => return Err(Status::internal(err.to_string())),
            }
        }

        if !unresolved_ids.is_empty() {
            let l0_rows = collection.index.lookup_l0_payload_rows(&unresolved_ids);
            for (id, row) in l0_rows {
                payload_by_id.insert(id, core_payload_row_to_unified(&row));
            }
        }

        let mut search_results = Vec::with_capacity(requested_k);
        for (id, score) in results {
            let payload_row = payload_by_id.get(&id);
            if !row_matches_filters(payload_row, &parsed_filters) {
                continue;
            }
            let payload = if projection.is_empty() {
                None
            } else {
                Some(project_payload_row(payload_row, &projection))
            };
            search_results.push(SearchResult { id, score, payload });
            if search_results.len() >= requested_k {
                break;
            }
        }

        Ok(Response::new(SearchResponse {
            results: search_results,
        }))
    }
}
