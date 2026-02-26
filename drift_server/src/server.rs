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
    UnifiedLobRef, UnifiedPayloadFieldStats, UnifiedPayloadRow, UnifiedPayloadStatsChunk,
    UnifiedPayloadValue,
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

fn intersect_candidate_ids(existing: &mut HashSet<u64>, next: &HashSet<u64>) {
    existing.retain(|id| next.contains(id));
}

#[derive(Default)]
struct FilterSourceProbe {
    might_match: bool,
    exact_candidate_ids: Option<HashSet<u64>>,
}

#[derive(Default)]
struct FilterAwareExecutionPlan {
    bucket_ids: Vec<u32>,
    candidate_ids: HashMap<u32, HashSet<u64>>,
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

async fn source_filter_probe(
    reader: &UnifiedReader,
    filters: &[ParsedFilter],
) -> io::Result<FilterSourceProbe> {
    if filters.is_empty() {
        return Ok(FilterSourceProbe {
            might_match: true,
            exact_candidate_ids: None,
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

    let mut exact_candidate_ids: Option<HashSet<u64>> = None;
    let mut used_exact_pushdown = false;

    for filter in filters {
        match filter {
            ParsedFilter::Exact { field_id, value } => {
                if matches!(value, UnifiedPayloadValue::Null) {
                    continue;
                }

                let Some(schema) = schema.as_ref() else {
                    return Ok(FilterSourceProbe::default());
                };
                let Some(field) = schema.fields.iter().find(|f| f.field_id == *field_id) else {
                    return Ok(FilterSourceProbe::default());
                };

                if !field.indexed {
                    continue;
                }

                let matches: HashSet<u64> = reader
                    .filter_ids_exact(*field_id, value)
                    .await?
                    .into_iter()
                    .collect();
                if matches.is_empty() {
                    return Ok(FilterSourceProbe::default());
                }

                used_exact_pushdown = true;
                if let Some(existing) = exact_candidate_ids.as_mut() {
                    intersect_candidate_ids(existing, &matches);
                    if existing.is_empty() {
                        return Ok(FilterSourceProbe::default());
                    }
                } else {
                    exact_candidate_ids = Some(matches);
                }
            }
            ParsedFilter::AnyOf { field_id, values } => {
                if values
                    .iter()
                    .any(|value| matches!(value, UnifiedPayloadValue::Null))
                {
                    continue;
                }

                let Some(schema) = schema.as_ref() else {
                    return Ok(FilterSourceProbe::default());
                };
                let Some(field) = schema.fields.iter().find(|f| f.field_id == *field_id) else {
                    return Ok(FilterSourceProbe::default());
                };

                if !field.indexed {
                    continue;
                }

                let mut matches = HashSet::new();
                for value in values {
                    matches.extend(reader.filter_ids_exact(*field_id, value).await?);
                }
                if matches.is_empty() {
                    return Ok(FilterSourceProbe::default());
                }

                used_exact_pushdown = true;
                if let Some(existing) = exact_candidate_ids.as_mut() {
                    intersect_candidate_ids(existing, &matches);
                    if existing.is_empty() {
                        return Ok(FilterSourceProbe::default());
                    }
                } else {
                    exact_candidate_ids = Some(matches);
                }
            }
            ParsedFilter::Range {
                field_id,
                lower,
                lower_inclusive,
                upper,
                upper_inclusive,
            } => {
                let Some(schema) = schema.as_ref() else {
                    return Ok(FilterSourceProbe::default());
                };
                if schema
                    .fields
                    .iter()
                    .all(|field| field.field_id != *field_id)
                {
                    return Ok(FilterSourceProbe::default());
                }

                let Some(stats) = payload_stats.as_ref() else {
                    continue;
                };
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
                    return Ok(FilterSourceProbe::default());
                }
            }
        }
    }

    Ok(FilterSourceProbe {
        might_match: true,
        exact_candidate_ids: if used_exact_pushdown {
            exact_candidate_ids
        } else {
            None
        },
    })
}

async fn plan_filter_aware_execution(
    collection: &Collection,
    routed_bucket_ids: &[u32],
    filters: &[ParsedFilter],
) -> FilterAwareExecutionPlan {
    if routed_bucket_ids.is_empty() || filters.is_empty() {
        return FilterAwareExecutionPlan {
            bucket_ids: routed_bucket_ids.to_vec(),
            candidate_ids: HashMap::new(),
        };
    }

    let local_op = match create_staging_operator(collection) {
        Ok(op) => op,
        Err(err) => {
            tracing::debug!(
                "Filter planner: failed to create staging operator: {}; disabling pushdown",
                err
            );
            return FilterAwareExecutionPlan {
                bucket_ids: routed_bucket_ids.to_vec(),
                candidate_ids: HashMap::new(),
            };
        }
    };
    let remote_op = collection.persistence.operator();

    let mut plan = FilterAwareExecutionPlan {
        bucket_ids: Vec::with_capacity(routed_bucket_ids.len()),
        candidate_ids: HashMap::new(),
    };
    for &bucket_id in routed_bucket_ids {
        let Some(version) = collection.bucket_manager.get_version(bucket_id) else {
            plan.bucket_ids.push(bucket_id);
            continue;
        };
        let sources = payload_sources_for_class(&version.class, &version.path);

        let mut keep_bucket = false;
        let mut disable_candidates = false;
        let mut bucket_candidates = HashSet::new();
        let mut has_bucket_candidates = false;

        for source in sources {
            let probe = match source {
                PayloadSource::Local(path) => {
                    match UnifiedReader::open(local_op.clone(), &path).await {
                        Ok(reader) => source_filter_probe(&reader, filters).await,
                        Err(err) => Err(err),
                    }
                }
                PayloadSource::Remote(path) => {
                    match UnifiedReader::open(remote_op.clone(), &path).await {
                        Ok(reader) => source_filter_probe(&reader, filters).await,
                        Err(err) => Err(err),
                    }
                }
            };

            match probe {
                Ok(probe) => {
                    if !probe.might_match {
                        continue;
                    }
                    keep_bucket = true;
                    if disable_candidates {
                        continue;
                    }
                    if let Some(source_candidates) = probe.exact_candidate_ids {
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
                } else {
                    tracing::debug!(
                        "Filter planner: skipping candidate pushdown for bucket {} due to broad selectivity (candidates={}, live_ids={})",
                        bucket_id,
                        bucket_candidates.len(),
                        bucket_live_ids
                    );
                }
            }
        }
    }
    plan
}

#[cfg(test)]
mod planner_heuristic_tests {
    use super::should_apply_candidate_pushdown;

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
            None
        } else {
            let routed_bucket_ids = collection.index.select_buckets(
                &req.vector,
                req.target_confidence,
                req.lambda,
                req.tau,
            );
            Some(
                plan_filter_aware_execution(&collection, &routed_bucket_ids, &parsed_filters).await,
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
