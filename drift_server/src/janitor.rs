use crate::cleanup::CleanupApi;
use crate::filter_metadata_catalog::FilterMetadataCatalog;
use crate::global_filter_routing_index::{
    GlobalFilterRoutingIndex, extract_indexed_exact_value_keys, indexed_exact_field_ids,
};
use crate::global_metadata_snapshot::{
    GLOBAL_METADATA_SNAPSHOT_FORMAT_VERSION, GlobalMetadataSnapshot, RoutingBucketTokenSnapshot,
};
use crate::local_staging::LocalStagingManager;
use crate::manifest::ServerManifestManager;
use crate::persistence::PersistenceManager;
use crate::reaper::Reaper;
use drift_core::partitioner::PartitionGroup;
use drift_core::payload::{
    PayloadLogicalType as CorePayloadLogicalType, PayloadRow as CorePayloadRow,
    PayloadSchema as CorePayloadSchema, PayloadValue as CorePayloadValue,
};
use drift_core::{index::VectorIndex, lock_manager::BucketCoordinator};
use drift_storage::bucket_manager::{BucketManager, StorageClass};
use drift_storage::unified_format::{
    UnifiedFieldSchema, UnifiedLobRef, UnifiedLogicalType, UnifiedPayloadRow, UnifiedPayloadSchema,
    UnifiedPayloadValue,
};
use drift_traits::StorageEngine;
use parking_lot::RwLock as ParkingRwLock;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use std::{
    collections::{HashMap, VecDeque},
    io,
};
use tokio::sync::Mutex;
use tokio::time;
use tracing::{error, info, warn};

const KV_SYNC_INTERVAL_MS_ENV: &str = "DRIFT_KV_SYNC_INTERVAL_MS";
const GLOBAL_METADATA_PERSIST_ENV: &str = "DRIFT_GLOBAL_METADATA_PERSIST";
const GLOBAL_METADATA_CHECKPOINT_INTERVAL_MS_ENV: &str =
    "DRIFT_GLOBAL_METADATA_CHECKPOINT_INTERVAL_MS";

fn kv_sync_interval_from_env() -> Duration {
    let default = Duration::from_millis(5_000);
    match std::env::var(KV_SYNC_INTERVAL_MS_ENV) {
        Ok(raw) => match raw.trim().parse::<u64>() {
            Ok(ms) if ms > 0 => Duration::from_millis(ms),
            _ => {
                warn!(
                    "Janitor: invalid {}='{}'; using default {}ms",
                    KV_SYNC_INTERVAL_MS_ENV,
                    raw,
                    default.as_millis()
                );
                default
            }
        },
        Err(_) => default,
    }
}

fn global_metadata_persist_enabled_from_env() -> bool {
    match std::env::var(GLOBAL_METADATA_PERSIST_ENV) {
        Ok(raw) => match raw.trim().to_ascii_lowercase().as_str() {
            "0" | "false" | "no" | "off" => false,
            "1" | "true" | "yes" | "on" => true,
            other => {
                warn!(
                    "Janitor: invalid {}='{}'; defaulting to enabled",
                    GLOBAL_METADATA_PERSIST_ENV, other
                );
                true
            }
        },
        Err(_) => true,
    }
}

fn global_metadata_checkpoint_interval_from_env() -> Duration {
    let default = Duration::from_millis(15_000);
    match std::env::var(GLOBAL_METADATA_CHECKPOINT_INTERVAL_MS_ENV) {
        Ok(raw) => match raw.trim().parse::<u64>() {
            Ok(ms) if ms > 0 => Duration::from_millis(ms),
            _ => {
                warn!(
                    "Janitor: invalid {}='{}'; using default {}ms",
                    GLOBAL_METADATA_CHECKPOINT_INTERVAL_MS_ENV,
                    raw,
                    default.as_millis()
                );
                default
            }
        },
        Err(_) => default,
    }
}

fn core_payload_schema_to_unified(schema: &CorePayloadSchema) -> UnifiedPayloadSchema {
    UnifiedPayloadSchema::new(
        schema
            .fields
            .iter()
            .map(|field| UnifiedFieldSchema {
                field_id: field.field_id,
                name: field.name.clone(),
                logical_type: core_payload_logical_type_to_unified(&field.logical_type),
                nullable: field.nullable,
                indexed: field.indexed,
            })
            .collect(),
    )
}

fn core_payload_rows_to_unified(rows: &[CorePayloadRow]) -> Vec<UnifiedPayloadRow> {
    rows.iter()
        .map(|row| {
            row.iter()
                .map(|(field_id, value)| (*field_id, core_payload_value_to_unified(value)))
                .collect()
        })
        .collect()
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

fn core_payload_logical_type_to_unified(
    logical_type: &CorePayloadLogicalType,
) -> UnifiedLogicalType {
    match logical_type {
        CorePayloadLogicalType::Bool => UnifiedLogicalType::Bool,
        CorePayloadLogicalType::Int64 => UnifiedLogicalType::Int64,
        CorePayloadLogicalType::Float32 => UnifiedLogicalType::Float32,
        CorePayloadLogicalType::Float64 => UnifiedLogicalType::Float64,
        CorePayloadLogicalType::Keyword => UnifiedLogicalType::Keyword,
        CorePayloadLogicalType::Text => UnifiedLogicalType::Text,
        CorePayloadLogicalType::Bytes => UnifiedLogicalType::Bytes,
        CorePayloadLogicalType::TimestampMicros => UnifiedLogicalType::TimestampMicros,
        CorePayloadLogicalType::LobRef => UnifiedLogicalType::LobRef,
    }
}

fn unified_payload_schema_to_core(schema: &UnifiedPayloadSchema) -> CorePayloadSchema {
    CorePayloadSchema::new(
        schema
            .fields
            .iter()
            .map(|field| drift_core::payload::PayloadFieldSchema {
                field_id: field.field_id,
                name: field.name.clone(),
                logical_type: unified_payload_logical_type_to_core(&field.logical_type),
                nullable: field.nullable,
                indexed: field.indexed,
            })
            .collect(),
    )
}

fn unified_payload_rows_to_core(rows: &[UnifiedPayloadRow]) -> Vec<CorePayloadRow> {
    rows.iter()
        .map(|row| {
            row.iter()
                .map(|(field_id, value)| (*field_id, unified_payload_value_to_core(value)))
                .collect()
        })
        .collect()
}

fn unified_payload_value_to_core(value: &UnifiedPayloadValue) -> CorePayloadValue {
    match value {
        UnifiedPayloadValue::Bool(v) => CorePayloadValue::Bool(*v),
        UnifiedPayloadValue::Int64(v) => CorePayloadValue::Int64(*v),
        UnifiedPayloadValue::Float32(v) => CorePayloadValue::Float32(*v),
        UnifiedPayloadValue::Float64(v) => CorePayloadValue::Float64(*v),
        UnifiedPayloadValue::Keyword(v) => CorePayloadValue::Keyword(v.clone()),
        UnifiedPayloadValue::Text(v) => CorePayloadValue::Text(v.clone()),
        UnifiedPayloadValue::Bytes(v) => CorePayloadValue::Bytes(v.clone()),
        UnifiedPayloadValue::TimestampMicros(v) => CorePayloadValue::TimestampMicros(*v),
        UnifiedPayloadValue::LobRef(v) => {
            CorePayloadValue::LobRef(drift_core::payload::PayloadLobRef {
                blob_key: v.blob_key.clone(),
                offset: v.offset,
                length: v.length,
                fingerprint: v.fingerprint.clone(),
            })
        }
        UnifiedPayloadValue::Null => CorePayloadValue::Null,
    }
}

fn unified_payload_logical_type_to_core(
    logical_type: &UnifiedLogicalType,
) -> CorePayloadLogicalType {
    match logical_type {
        UnifiedLogicalType::Bool => CorePayloadLogicalType::Bool,
        UnifiedLogicalType::Int64 => CorePayloadLogicalType::Int64,
        UnifiedLogicalType::Float32 => CorePayloadLogicalType::Float32,
        UnifiedLogicalType::Float64 => CorePayloadLogicalType::Float64,
        UnifiedLogicalType::Keyword => CorePayloadLogicalType::Keyword,
        UnifiedLogicalType::Text => CorePayloadLogicalType::Text,
        UnifiedLogicalType::Bytes => CorePayloadLogicalType::Bytes,
        UnifiedLogicalType::TimestampMicros => CorePayloadLogicalType::TimestampMicros,
        UnifiedLogicalType::LobRef => CorePayloadLogicalType::LobRef,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PayloadRowLookupKey {
    id: u64,
    vector_bits: Vec<u32>,
}

type PayloadRowLookup = HashMap<PayloadRowLookupKey, VecDeque<UnifiedPayloadRow>>;

fn payload_lookup_key(id: u64, vector: &[f32]) -> PayloadRowLookupKey {
    PayloadRowLookupKey {
        id,
        vector_bits: vector.iter().map(|v| v.to_bits()).collect(),
    }
}

fn build_payload_row_lookup(
    bucket_id: u32,
    context: &str,
    ids: &[u64],
    flat_vectors: &[f32],
    dim: usize,
    rows: &[UnifiedPayloadRow],
) -> io::Result<PayloadRowLookup> {
    if rows.len() != ids.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{context} payload row mismatch for bucket {bucket_id}: ids={}, payload_rows={}",
                ids.len(),
                rows.len()
            ),
        ));
    }
    if flat_vectors.len() != ids.len() * dim {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{context} vector length mismatch for bucket {bucket_id}: ids={}, flat={}",
                ids.len(),
                flat_vectors.len()
            ),
        ));
    }

    let mut lookup: PayloadRowLookup = HashMap::new();
    for (row_idx, id) in ids.iter().enumerate() {
        let start = row_idx * dim;
        let end = start + dim;
        let key = payload_lookup_key(*id, &flat_vectors[start..end]);
        lookup
            .entry(key)
            .or_default()
            .push_back(rows[row_idx].clone());
    }
    Ok(lookup)
}

fn take_payload_rows_for_group(
    bucket_id: u32,
    context: &str,
    group: &PartitionGroup,
    dim: usize,
    lookup: &mut PayloadRowLookup,
) -> io::Result<Vec<UnifiedPayloadRow>> {
    if group.flat_vectors.len() != group.ids.len() * dim {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{context} vector length mismatch for bucket {bucket_id}: ids={}, flat={}",
                group.ids.len(),
                group.flat_vectors.len()
            ),
        ));
    }

    let mut rows = Vec::with_capacity(group.ids.len());
    for (row_idx, id) in group.ids.iter().enumerate() {
        let start = row_idx * dim;
        let end = start + dim;
        let key = payload_lookup_key(*id, &group.flat_vectors[start..end]);
        let (row, should_remove) = match lookup.get_mut(&key) {
            Some(queue) => {
                let value = queue.pop_front().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "{context} payload queue underflow for bucket {bucket_id} (id={id})"
                        ),
                    )
                })?;
                (value, queue.is_empty())
            }
            None => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("{context} payload lookup miss for bucket {bucket_id} (id={id})"),
                ));
            }
        };
        if should_remove {
            lookup.remove(&key);
        }
        rows.push(row);
    }
    Ok(rows)
}

fn take_payload_rows_for_loopback(
    bucket_id: u32,
    context: &str,
    loopback: &[(u64, Vec<f32>)],
    dim: usize,
    lookup: &mut PayloadRowLookup,
) -> io::Result<Vec<UnifiedPayloadRow>> {
    let mut rows = Vec::with_capacity(loopback.len());
    for (id, vector) in loopback {
        if vector.len() != dim {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "{context} loopback vector length mismatch for bucket {bucket_id} (id={id})"
                ),
            ));
        }

        let key = payload_lookup_key(*id, vector);
        let (row, should_remove) = match lookup.get_mut(&key) {
            Some(queue) => {
                let value = queue.pop_front().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "{context} payload queue underflow for bucket {bucket_id} (loopback id={id})"
                        ),
                    )
                })?;
                (value, queue.is_empty())
            }
            None => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "{context} payload lookup miss for bucket {bucket_id} (loopback id={id})"
                    ),
                ));
            }
        };
        if should_remove {
            lookup.remove(&key);
        }
        rows.push(row);
    }
    Ok(rows)
}

fn ensure_payload_lookup_drained(
    bucket_id: u32,
    context: &str,
    lookup: &PayloadRowLookup,
) -> io::Result<()> {
    let remaining: usize = lookup.values().map(VecDeque::len).sum();
    if remaining != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{context} payload lookup not fully consumed for bucket {bucket_id}: remaining_rows={remaining}"
            ),
        ));
    }
    Ok(())
}

pub struct JanitorVars {
    pub promotion_threshold_bytes: u64,
    pub max_bucket_capacity: usize,
    pub drift_threshold: f32,         // Default 0.15
    pub split_threshold: f32,         // Default 0.8
    pub temperature_cool_factor: f32, // default to 0.98
    pub check_interval: Duration,
    pub urgency_threshold: f32, // default to 1.5
}

impl Default for JanitorVars {
    fn default() -> Self {
        Self {
            promotion_threshold_bytes: 1024,
            max_bucket_capacity: 2000,
            drift_threshold: 0.15,
            split_threshold: 0.8,
            temperature_cool_factor: 0.98,
            check_interval: Duration::from_millis(100),
            urgency_threshold: 1.5,
        }
    }
}

pub struct JanitorConfig {
    pub index: Arc<VectorIndex>,
    pub manifest: Arc<ServerManifestManager>,
    pub staging: Arc<LocalStagingManager>,
    pub persistence: Arc<PersistenceManager>,
    pub bucket_manager: Arc<BucketManager>,
    pub filter_metadata_catalog: Arc<ParkingRwLock<FilterMetadataCatalog>>,
    pub global_filter_routing_index: Arc<ParkingRwLock<GlobalFilterRoutingIndex>>,
    pub coordinator: Arc<BucketCoordinator>,
    pub vars: JanitorVars,
}

pub struct Janitor {
    index: Arc<VectorIndex>,
    manifest: Arc<ServerManifestManager>,
    staging: Arc<LocalStagingManager>,
    persistence: Arc<PersistenceManager>,
    cleanup: CleanupApi,
    bucket_manager: Arc<BucketManager>,
    filter_metadata_catalog: Arc<ParkingRwLock<FilterMetadataCatalog>>,
    global_filter_routing_index: Arc<ParkingRwLock<GlobalFilterRoutingIndex>>,
    reaper: Mutex<Reaper>,
    coordinator: Arc<BucketCoordinator>,
    vars: JanitorVars,
    global_metadata_persist_enabled: bool,
    global_metadata_checkpoint_interval: Duration,
    global_metadata_dirty: AtomicBool,
}

impl Janitor {
    pub fn new(config: JanitorConfig) -> Self {
        let cleanup = CleanupApi::new(config.staging.clone(), config.persistence.clone());
        let reaper = Mutex::new(Reaper::new(
            config.staging.clone(),
            config.persistence.clone(),
        ));
        let global_metadata_persist_enabled = global_metadata_persist_enabled_from_env();
        let global_metadata_checkpoint_interval = global_metadata_checkpoint_interval_from_env();
        Self {
            index: config.index,
            manifest: config.manifest,
            staging: config.staging,
            persistence: config.persistence,
            cleanup,
            bucket_manager: config.bucket_manager,
            filter_metadata_catalog: config.filter_metadata_catalog,
            global_filter_routing_index: config.global_filter_routing_index,
            reaper,
            coordinator: config.coordinator,
            vars: config.vars,
            global_metadata_persist_enabled,
            global_metadata_checkpoint_interval,
            global_metadata_dirty: AtomicBool::new(false),
        }
    }

    fn mark_global_metadata_dirty(&self) {
        if self.global_metadata_persist_enabled {
            self.global_metadata_dirty.store(true, Ordering::Relaxed);
        }
    }

    async fn checkpoint_global_metadata_if_due(
        &self,
        last_checkpoint: &mut Instant,
        reason: &str,
        force: bool,
    ) {
        if !self.global_metadata_persist_enabled {
            return;
        }
        if !force && last_checkpoint.elapsed() < self.global_metadata_checkpoint_interval {
            return;
        }
        if !force && !self.global_metadata_dirty.load(Ordering::Relaxed) {
            return;
        }

        let mut snapshot = GlobalMetadataSnapshot::default();
        snapshot.format_version = GLOBAL_METADATA_SNAPSHOT_FORMAT_VERSION;
        snapshot.routing = self.global_filter_routing_index.read().export_snapshot();
        snapshot.catalog = self.filter_metadata_catalog.read().export_snapshot();
        snapshot.routing.bucket_tokens = self.capture_routing_bucket_tokens();

        let previous_pointer = self.manifest.get_state().global_metadata_pointer();
        let write_result = match self
            .persistence
            .write_global_metadata_snapshot(&snapshot)
            .await
        {
            Ok(result) => result,
            Err(err) => {
                warn!(
                    "Janitor: failed to persist global metadata snapshot (reason={}): {}",
                    reason, err
                );
                *last_checkpoint = Instant::now();
                return;
            }
        };

        let update_result = self.manifest.apply_atomic(|manifest| {
            manifest.update_global_metadata_pointer(
                write_result.object_path.clone(),
                write_result.object_fingerprint.clone(),
                write_result.format_version,
            );
        });
        if let Err(err) = update_result {
            warn!(
                "Janitor: failed to publish global metadata pointer to manifest (reason={}): {}",
                reason, err
            );
            *last_checkpoint = Instant::now();
            return;
        }

        if let Some(previous) = previous_pointer
            && !previous.path.is_empty()
            && previous.path != write_result.object_path
        {
            self.cleanup
                .delete_remote_best_effort(&previous.path, "global-metadata-rotate")
                .await;
        }

        self.global_metadata_dirty.store(false, Ordering::Relaxed);
        *last_checkpoint = Instant::now();
    }

    fn capture_routing_bucket_tokens(&self) -> Vec<RoutingBucketTokenSnapshot> {
        let manifest_state = self.manifest.get_state();
        let mut tokens = Vec::new();
        for bucket in manifest_state.get_buckets() {
            let Some(version) = self.bucket_manager.get_version(bucket.id) else {
                continue;
            };
            let Some(stats) = self.bucket_manager.get_bucket_stats(bucket.id) else {
                continue;
            };
            tokens.push(RoutingBucketTokenSnapshot {
                bucket_id: bucket.id,
                bucket_path: version.path.clone(),
                bucket_live_count: stats.total_count.saturating_sub(stats.tombstone_count),
            });
        }
        tokens.sort_by(|lhs, rhs| lhs.bucket_id.cmp(&rhs.bucket_id));
        tokens
    }

    fn invalidate_catalog_buckets(&self, bucket_ids: &[u32]) {
        if bucket_ids.is_empty() {
            return;
        }
        let mut catalog = self.filter_metadata_catalog.write();
        for bucket_id in bucket_ids {
            catalog.invalidate_bucket(*bucket_id);
        }
        self.mark_global_metadata_dirty();
    }

    fn invalidate_routing_buckets(&self, bucket_ids: &[u32]) {
        if bucket_ids.is_empty() {
            return;
        }
        let mut routing = self.global_filter_routing_index.write();
        for bucket_id in bucket_ids {
            routing.invalidate_bucket(*bucket_id);
        }
        self.mark_global_metadata_dirty();
    }

    fn remove_routing_ids(&self, ids: &[u64]) {
        if ids.is_empty() {
            return;
        }
        let mut routing = self.global_filter_routing_index.write();
        for id in ids {
            routing.remove_id(*id);
        }
        self.mark_global_metadata_dirty();
    }

    fn upsert_routing_entries_incremental(
        &self,
        bucket_id: u32,
        payload_schema: Option<&UnifiedPayloadSchema>,
        payload_rows: Option<&[UnifiedPayloadRow]>,
        ids: &[u64],
    ) -> io::Result<()> {
        if ids.is_empty() {
            return Ok(());
        }

        let mut routing = self.global_filter_routing_index.write();
        match (payload_schema, payload_rows) {
            (Some(schema), Some(rows)) => {
                if rows.len() != ids.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "routing incremental payload row mismatch for bucket {}: ids={}, payload_rows={}",
                            bucket_id,
                            ids.len(),
                            rows.len()
                        ),
                    ));
                }
                for (id, row) in ids.iter().zip(rows.iter()) {
                    let keys = extract_indexed_exact_value_keys(schema, row)?;
                    routing.upsert_id_values(*id, bucket_id, keys);
                }
            }
            (Some(_), None) if !ids.is_empty() => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "routing incremental payload rows missing for bucket {} with {} ids",
                        bucket_id,
                        ids.len()
                    ),
                ));
            }
            (None, Some(rows)) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "routing incremental payload rows present without schema for bucket {} (payload_rows={})",
                        bucket_id,
                        rows.len()
                    ),
                ));
            }
            (None, None) | (Some(_), None) => {
                for id in ids {
                    routing.remove_id(*id);
                }
            }
        }
        self.mark_global_metadata_dirty();
        Ok(())
    }

    fn rebuild_routing_bucket_from_snapshot(
        &self,
        bucket_id: u32,
        payload_schema: Option<&UnifiedPayloadSchema>,
        payload_rows: Option<&[UnifiedPayloadRow]>,
        ids: &[u64],
    ) -> io::Result<()> {
        let mut routing = self.global_filter_routing_index.write();
        routing.invalidate_bucket(bucket_id);

        match (payload_schema, payload_rows) {
            (Some(schema), Some(rows)) => {
                if rows.len() != ids.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "routing rebuild payload row mismatch for bucket {}: ids={}, payload_rows={}",
                            bucket_id,
                            ids.len(),
                            rows.len()
                        ),
                    ));
                }
                for (id, row) in ids.iter().zip(rows.iter()) {
                    let keys = extract_indexed_exact_value_keys(schema, row)?;
                    routing.upsert_id_values(*id, bucket_id, keys);
                }
                routing
                    .set_bucket_complete_exact_fields(bucket_id, indexed_exact_field_ids(schema));
            }
            (Some(_schema), None) if !ids.is_empty() => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "routing rebuild payload rows missing for bucket {} with {} ids",
                        bucket_id,
                        ids.len()
                    ),
                ));
            }
            (Some(schema), None) => {
                routing
                    .set_bucket_complete_exact_fields(bucket_id, indexed_exact_field_ids(schema));
            }
            (None, Some(rows)) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "routing rebuild payload rows present without schema for bucket {} (payload_rows={})",
                        bucket_id,
                        rows.len()
                    ),
                ));
            }
            (None, None) => {
                for id in ids {
                    routing.remove_id(*id);
                }
            }
        }
        self.mark_global_metadata_dirty();
        Ok(())
    }

    fn update_kv_mapping(&self, bucket_id: u32, ids: &[u64]) {
        let kv = self.index.get_kv();
        let bucket_bytes = bucket_id.to_le_bytes();
        for id in ids {
            if let Err(e) = kv.put(&id.to_le_bytes(), &bucket_bytes) {
                warn!(
                    "Janitor: failed kv.put for id={} bucket={}: {}",
                    id, bucket_id, e
                );
            }
        }
    }

    fn remove_kv_mapping(&self, ids: &[u64]) {
        let kv = self.index.get_kv();
        for id in ids {
            if let Err(e) = kv.remove(&id.to_le_bytes()) {
                warn!("Janitor: failed kv.remove for id={}: {}", id, e);
            }
        }
    }

    fn sync_kv_best_effort(&self, reason: &str) {
        if let Err(e) = self.index.get_kv().sync() {
            warn!("Janitor: kv.sync failed ({reason}): {e}");
        }
    }

    #[allow(clippy::type_complexity)]
    async fn read_bucket_flat_with_payload(
        &self,
        bucket_id: u32,
        context: &str,
    ) -> io::Result<(
        Vec<u64>,
        Vec<f32>,
        Option<UnifiedPayloadSchema>,
        Option<Vec<UnifiedPayloadRow>>,
    )> {
        #[derive(Clone)]
        enum Source {
            Local(String),
            Remote(String),
        }

        let version = self.bucket_manager.get_version(bucket_id).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("bucket {bucket_id} not found"),
            )
        })?;

        let mut sources = Vec::new();
        match &version.class {
            StorageClass::Local => {
                sources.push(Source::Local(version.path.clone()));
            }
            StorageClass::Remote => {
                sources.push(Source::Remote(version.path.clone()));
            }
            StorageClass::Tiered {
                remote_path,
                local_path,
            } => {
                sources.push(Source::Remote(remote_path.clone()));
                sources.push(Source::Local(local_path.clone()));
            }
            StorageClass::Promoting {
                local_active,
                local_frozen,
                remote_path,
            } => {
                if let Some(path) = remote_path {
                    sources.push(Source::Remote(path.clone()));
                }
                sources.push(Source::Local(local_frozen.clone()));
                sources.push(Source::Local(local_active.clone()));
            }
        }

        let dim = self.index.get_dim();
        let mut merged_ids = Vec::new();
        let mut merged_flat = Vec::new();
        let mut merged_schema: Option<UnifiedPayloadSchema> = None;
        let mut merged_rows: Option<Vec<UnifiedPayloadRow>> = None;

        for source in sources {
            let (source_label, ids, flat, schema, rows) = match &source {
                Source::Local(path) => {
                    let ids_and_flat = self.staging.read_file_content_flat(path).await?;
                    let schema = self.staging.read_file_payload_schema(path).await?;
                    let rows = self.staging.read_file_payload_rows(path).await?;
                    (
                        format!("local:{path}"),
                        ids_and_flat.0,
                        ids_and_flat.1,
                        schema,
                        rows,
                    )
                }
                Source::Remote(path) => {
                    let ids_and_flat = self.persistence.read_remote_bucket_path_flat(path).await?;
                    let schema = self
                        .persistence
                        .read_remote_bucket_payload_schema_path(path)
                        .await?;
                    let rows = self
                        .persistence
                        .read_remote_bucket_payload_rows_path_optional(path)
                        .await?;
                    (
                        format!("remote:{path}"),
                        ids_and_flat.0,
                        ids_and_flat.1,
                        schema,
                        rows,
                    )
                }
            };

            if flat.len() != ids.len() * dim {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "{context} vector length mismatch for bucket {bucket_id} source {source_label}: ids={}, flat={}",
                        ids.len(),
                        flat.len()
                    ),
                ));
            }
            if rows.is_some() && schema.is_none() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "{context} payload rows present without schema for bucket {bucket_id} source {source_label}"
                    ),
                ));
            }
            if schema.is_some() && rows.is_none() && !ids.is_empty() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "{context} payload schema present without rows for bucket {bucket_id} source {source_label}"
                    ),
                ));
            }
            if let Some(source_rows) = rows.as_ref()
                && source_rows.len() != ids.len()
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "{context} payload row mismatch for bucket {bucket_id} source {source_label}: ids={}, payload_rows={}",
                        ids.len(),
                        source_rows.len()
                    ),
                ));
            }

            if let Some(source_schema) = schema.as_ref() {
                if let Some(existing) = merged_schema.as_ref() {
                    if existing != source_schema {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "{context} payload schema mismatch while reading bucket {bucket_id} source {source_label}"
                            ),
                        ));
                    }
                } else {
                    merged_schema = Some(source_schema.clone());
                }
            } else if merged_schema.is_some() && !ids.is_empty() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "{context} payload schema missing for bucket {bucket_id} source {source_label}"
                    ),
                ));
            }

            match (rows, merged_rows.as_mut()) {
                (Some(mut source_rows), Some(target_rows)) => target_rows.append(&mut source_rows),
                (Some(source_rows), None) => merged_rows = Some(source_rows),
                (None, Some(_)) if !ids.is_empty() => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "{context} payload rows missing for bucket {bucket_id} source {source_label}"
                        ),
                    ));
                }
                (None, Some(_)) => {}
                (None, None) => {}
            }

            merged_ids.extend(ids);
            merged_flat.extend(flat);
        }

        if merged_flat.len() != merged_ids.len() * dim {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "{context} merged vector length mismatch for bucket {bucket_id}: ids={}, flat={}",
                    merged_ids.len(),
                    merged_flat.len()
                ),
            ));
        }
        if let Some(rows) = merged_rows.as_ref() {
            if rows.len() != merged_ids.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "{context} merged payload row mismatch for bucket {bucket_id}: ids={}, payload_rows={}",
                        merged_ids.len(),
                        rows.len()
                    ),
                ));
            }
        }
        if merged_rows.is_some() && merged_schema.is_none() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{context} payload rows present without schema for bucket {bucket_id}"),
            ));
        }
        if merged_schema.is_some() && merged_rows.is_none() && !merged_ids.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{context} payload schema present without rows for bucket {bucket_id}"),
            ));
        }

        Ok((merged_ids, merged_flat, merged_schema, merged_rows))
    }

    pub async fn run(&self) {
        let mut interval = time::interval(self.vars.check_interval);
        let kv_sync_interval = kv_sync_interval_from_env();
        let mut last_kv_sync = Instant::now();
        let mut last_global_metadata_checkpoint = Instant::now();
        info!("Janitor: Started.");

        loop {
            interval.tick().await;

            // 1. Flush Frozen MemTables
            if let Err(e) = self.perform_flush().await {
                error!("Janitor: Flush failed: {}", e);
            }

            // 2. Maintenance (Splits)
            self.check_maintainance().await;

            // 3. Run Reaper (Garbage Collection)
            let mut reaper = self.reaper.lock().await;
            reaper.run_cycle().await;
            drop(reaper); // Release lock before long operations

            // 4. Check for Promotions (Local -> S3)
            if let Err(e) = self.promote_segments().await {
                error!("Janitor: Promotion failed: {}", e);
            }

            if last_kv_sync.elapsed() >= kv_sync_interval {
                self.sync_kv_best_effort("periodic");
                last_kv_sync = Instant::now();
            }

            self.checkpoint_global_metadata_if_due(
                &mut last_global_metadata_checkpoint,
                "periodic",
                false,
            )
            .await;
        }
    }

    async fn perform_flush(&self) -> io::Result<()> {
        // 1. Data Flush Logic
        let (partitions, wal_ids) = match self.index.flush_frozen() {
            Some((p, w)) => (p, w),
            None => return Ok(()),
        };

        let mut manifest_updates = Vec::new();
        let mut flushed_bucket_ids = Vec::new();
        // ⚡ CHANGE: Store updates for the Router (ID, Count, Centroid)
        let mut router_updates = Vec::new();

        for (bucket_id, group) in &partitions {
            flushed_bucket_ids.push(*bucket_id);
            let payload_schema = group
                .payload_schema
                .as_ref()
                .map(core_payload_schema_to_unified);
            let payload_rows = group
                .payload_rows
                .as_ref()
                .map(|rows| core_payload_rows_to_unified(rows));

            // A. Append to Local Staging
            let new_count = self
                .staging
                .append_batch_with_payload(
                    *bucket_id,
                    group,
                    payload_schema.as_ref(),
                    payload_rows.as_deref(),
                )
                .await?;

            // A1. Update KV (VectorID -> BucketID)
            self.update_kv_mapping(*bucket_id, &group.ids);
            self.upsert_routing_entries_incremental(
                *bucket_id,
                payload_schema.as_ref(),
                payload_rows.as_deref(),
                &group.ids,
            )?;

            // B. Ensure Registered
            if self.bucket_manager.get_version(*bucket_id).is_none() {
                let filename = self.staging.get_active_filename(*bucket_id);
                self.bucket_manager.register_bucket(
                    *bucket_id,
                    filename,
                    drift_storage::bucket_manager::StorageClass::Local,
                );
            }

            // C. Calculate Delta Sum (for Drift Tracking)
            let dim = self.index.get_dim();
            let mut delta_sum = vec![0.0; dim];
            if group.count > 0 {
                // Sum the new vectors
                for chunk in group.flat_vectors.chunks_exact(dim) {
                    for (i, val) in chunk.iter().enumerate() {
                        delta_sum[i] += val;
                    }
                }
            }

            // D. Update Persistent Stats
            self.bucket_manager
                .update_bucket_drift(*bucket_id, &delta_sum, group.count as u32)?;

            // E. ⚡ FETCH GLOBAL TRUTH
            // We get the TOTAL sum and TOTAL count from the manager
            let (total_sum, total_count) =
                self.bucket_manager
                    .get_bucket_drift_stats(*bucket_id)
                    .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "Failed to read stats"))?;

            // F. ⚡ RECALCULATE CENTROID
            let global_centroid: Vec<f32> = if total_count > 0 {
                total_sum.iter().map(|s| s / total_count as f32).collect()
            } else {
                vec![0.0; dim]
            };

            manifest_updates.push((*bucket_id, new_count, Some(global_centroid.clone())));
            router_updates.push((*bucket_id, total_count, global_centroid));
        }

        // 2. Tombstone Persistence
        let mut all_tombstones = Vec::new();
        {
            let l0_arc = self.index.get_tombstones();
            all_tombstones.extend(l0_arc.iter());
        }
        let l1_deletes = self.bucket_manager.collect_all_tombstones();
        all_tombstones.extend(l1_deletes);

        let mut tombstone_file_opt = None;
        if !all_tombstones.is_empty() {
            all_tombstones.sort_unstable();
            all_tombstones.dedup();
            self.remove_routing_ids(&all_tombstones);
            let run_id = uuid::Uuid::new_v4().to_string();
            let ts_file = self
                .persistence
                .flush_tombstones(&all_tombstones, &run_id)
                .await?;
            tombstone_file_opt = Some(ts_file);
        }

        // 3. Update Manifest
        self.manifest.apply_atomic(|m| {
            for (id, count, centroid_opt) in manifest_updates {
                let exists = m.get_buckets().iter().any(|b| b.id == id);
                if !exists {
                    m.add_bucket(id, String::new(), centroid_opt);
                }
                m.update_bucket_stats(id, count, 0);
            }
            if let Some(tf) = tombstone_file_opt {
                m.inner.tombstone_files = vec![tf];
            }
        })?;

        // 4. ⚡ UPDATE ROUTER
        // We need a helper on Index to access the Router's `update_bucket` (not update_bucket_count)
        // If it doesn't exist, we'll need to add `index.update_router_bucket(...)`
        // Assuming we can access the router lock via the index:
        {
            let mut r = self.index.get_router().write();
            for (id, count, vec) in router_updates {
                // If bucket exists, update count AND centroid.
                // If it doesn't exist, add it.
                if r.get_centroid(id).is_some() {
                    r.update_bucket(id, count, vec);
                } else {
                    r.add_bucket(id, vec);
                    // Ensure count is set correctly after add (add_bucket sets count to 0)
                    r.update_bucket_count(id, count);
                }
            }
        }

        // 5. Acknowledge
        self.index.acknowledge_flush(&wal_ids)?;
        self.sync_kv_best_effort("flush");
        self.invalidate_catalog_buckets(flushed_bucket_ids.as_slice());

        Ok(())
    }

    async fn _perform_flush(&self) -> io::Result<()> {
        // 1. Data Flush Logic (Unchanged from previous plan)
        let (partitions, wal_ids) = match self.index.flush_frozen() {
            Some((p, w)) => (p, w),
            // Important: Even if no data to flush, we should occasionally flush tombstones?
            // For simplicity, we couple them. If no data flush, no tombstone flush yet.
            None => return Ok(()),
        };

        let mut updates = Vec::new();
        for (bucket_id, group) in &partitions {
            let payload_schema = group
                .payload_schema
                .as_ref()
                .map(core_payload_schema_to_unified);
            let payload_rows = group
                .payload_rows
                .as_ref()
                .map(|rows| core_payload_rows_to_unified(rows));
            let new_count = self
                .staging
                .append_batch_with_payload(
                    *bucket_id,
                    group,
                    payload_schema.as_ref(),
                    payload_rows.as_deref(),
                )
                .await?;
            updates.push((*bucket_id, new_count, group.centroid.clone()));

            // Keep router counts in sync with actual bucket size.
            self.index
                .update_router_count(*bucket_id, new_count as u32, group.centroid.clone());

            // If we update stats for a non-existent bucket, BucketManager ignores it.
            if self.bucket_manager.get_version(*bucket_id).is_none() {
                let filename = self.staging.get_active_filename(*bucket_id);
                self.bucket_manager.register_bucket(
                    *bucket_id,
                    filename,
                    drift_storage::bucket_manager::StorageClass::Local,
                );
            }

            // Calculate Delta Sum for Drift Tracking
            // We iterate the flat vector buffer
            if group.count > 0 {
                let dim = group.flat_vectors.len() / group.count;
                let mut delta_sum = vec![0.0; dim];

                for chunk in group.flat_vectors.chunks_exact(dim) {
                    for (i, val) in chunk.iter().enumerate() {
                        delta_sum[i] += val;
                    }
                }

                // Push update to BucketManager
                self.bucket_manager.update_bucket_drift(
                    *bucket_id,
                    &delta_sum,
                    group.count as u32,
                )?;
            }

            if self.bucket_manager.get_version(*bucket_id).is_none() {
                let filename = self.staging.get_active_filename(*bucket_id);
                self.bucket_manager.register_bucket(
                    *bucket_id,
                    filename,
                    drift_storage::bucket_manager::StorageClass::Local,
                );
            }
        }

        // 2. TOMBSTONE PERSISTENCE (Merged L0 + L1)
        let mut all_tombstones = Vec::new();

        // A. Collect L0 (MemTable Deletes)
        {
            // Lock, Clone Arc (cheap), iterate
            let l0_arc = self.index.get_tombstones();
            all_tombstones.extend(l0_arc.iter());
        }

        // B. Collect L1 (Bucket Deletes)
        let l1_deletes = self.bucket_manager.collect_all_tombstones();
        all_tombstones.extend(l1_deletes);

        let mut tombstone_file_opt = None;

        if !all_tombstones.is_empty() {
            // Deduplicate
            all_tombstones.sort_unstable();
            all_tombstones.dedup();

            let run_id = uuid::Uuid::new_v4().to_string();
            let ts_file = self
                .persistence
                .flush_tombstones(&all_tombstones, &run_id)
                .await?;

            info!(
                "Janitor: Persisted {} cumulative tombstones to {}",
                all_tombstones.len(),
                &ts_file
            );
            tombstone_file_opt = Some(ts_file);
        }

        // 3. Update Manifest (Atomic)
        self.manifest.apply_atomic(|m| {
            for (id, count, centroid_opt) in updates {
                let exists = m.get_buckets().iter().any(|b| b.id == id);
                if !exists {
                    m.add_bucket(id, String::new(), centroid_opt);
                }
                m.update_bucket_stats(id, count, 0);
            }

            // ⚡ Update Pointer to NEW cumulative file
            if let Some(tf) = tombstone_file_opt {
                m.inner.tombstone_files = vec![tf];
            }
        })?;

        // 4. Acknowledge
        self.index.acknowledge_flush(&wal_ids)?;

        Ok(())
    }

    pub(crate) async fn promote_segments(&self) -> io::Result<()> {
        let threshold = self.vars.promotion_threshold_bytes;
        let candidates = self.staging.list_large_buckets(threshold)?;

        if candidates.is_empty() {
            return Ok(());
        }

        info!("Janitor: Promoting {} buckets to S3...", candidates.len());

        for bucket_id in candidates {
            // 🔒 Lock Bucket (Prevents split/merge/compact during promotion)
            let _lock_guard = self.coordinator.write(bucket_id).await;

            let tombstone_snapshot = self.bucket_manager.get_tombstones(bucket_id);

            // 1. Rotate Local File
            let staging_filename = format!(
                "bucket_{}_staging_{}.driftu",
                bucket_id,
                uuid::Uuid::new_v4()
            );
            let new_filename = format!("bucket_{}_{}.driftu", bucket_id, uuid::Uuid::new_v4());

            let rotated = self
                .staging
                .rotate_bucket_for_promotion(bucket_id, &staging_filename, &new_filename)
                .await?;
            if !rotated {
                continue;
            }

            // 2. Update Registry to "Promoting" state
            let remote_path_opt = if let Some(ver) = self.bucket_manager.get_version(bucket_id) {
                match &ver.class {
                    StorageClass::Tiered { remote_path, .. } => Some(remote_path.clone()),
                    StorageClass::Promoting { remote_path, .. } => remote_path.clone(),
                    _ => None,
                }
            } else {
                None
            };

            let promoting_class = StorageClass::Promoting {
                local_active: new_filename.clone(),
                local_frozen: staging_filename.clone(),
                remote_path: remote_path_opt.clone(),
            };

            let current_count = self
                .bucket_manager
                .get_bucket_stats(bucket_id)
                .map(|s| s.total_count)
                .unwrap_or(0);

            self.bucket_manager.register_bucket_with_count(
                bucket_id,
                new_filename.clone(),
                promoting_class,
                current_count,
            );
            self.invalidate_catalog_buckets(&[bucket_id]);

            // --- ⚡ EXPLICIT MERGE & FILTER LOGIC ---
            let dim = self.index.get_dim();

            // A. Read Local Staging
            let (mut merged_ids, mut merged_flat) = self
                .staging
                .read_file_content_flat(&staging_filename)
                .await?;
            let mut merged_schema = self
                .staging
                .read_file_payload_schema(&staging_filename)
                .await?;
            let mut merged_payload_rows = self
                .staging
                .read_file_payload_rows(&staging_filename)
                .await?;
            if merged_ids.is_empty() {
                self.cleanup
                    .delete_local_best_effort(&staging_filename, "promotion-empty-staging")
                    .await;
                continue;
            }
            if let Some(rows) = merged_payload_rows.as_ref() {
                if rows.len() != merged_ids.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "promotion staging payload row mismatch for bucket {}: ids={}, payload_rows={}",
                            bucket_id,
                            merged_ids.len(),
                            rows.len()
                        ),
                    ));
                }
            }

            // B. Read Remote (if exists)
            if let Some(path) = &remote_path_opt {
                let (r_ids, r_flat) = self.persistence.read_remote_bucket_path_flat(path).await?;
                let r_schema = self
                    .persistence
                    .read_remote_bucket_payload_schema_path(path)
                    .await?;
                let r_payload_rows = self
                    .persistence
                    .read_remote_bucket_payload_rows_path_optional(path)
                    .await?;
                if let Some(remote_schema) = r_schema {
                    if let Some(local_schema) = &merged_schema {
                        if local_schema != &remote_schema {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                format!(
                                    "promotion payload schema mismatch for bucket {}",
                                    bucket_id
                                ),
                            ));
                        }
                    } else {
                        merged_schema = Some(remote_schema);
                    }
                }

                if let Some(rows) = r_payload_rows.as_ref()
                    && rows.len() != r_ids.len()
                {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "promotion remote payload row mismatch for bucket {}: ids={}, payload_rows={}",
                            bucket_id,
                            r_ids.len(),
                            rows.len()
                        ),
                    ));
                }

                match (&mut merged_payload_rows, r_payload_rows) {
                    (Some(local_rows), Some(mut remote_rows)) => {
                        local_rows.append(&mut remote_rows);
                    }
                    (None, None) => {}
                    _ => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "promotion payload row presence mismatch for bucket {}",
                                bucket_id
                            ),
                        ));
                    }
                }
                merged_ids.extend(r_ids);
                merged_flat.extend(r_flat);
            }

            // C. Snapshot Tombstones (Atomic Arc clone)
            let rows_from_vectors = merged_flat.len() / dim;
            if rows_from_vectors < merged_ids.len() {
                tracing::warn!(
                    "Janitor: merged row mismatch for bucket {} (ids={}, vectors={})",
                    bucket_id,
                    merged_ids.len(),
                    rows_from_vectors
                );
                merged_ids.truncate(rows_from_vectors);
                merged_flat.truncate(rows_from_vectors * dim);
                if let Some(rows) = merged_payload_rows.as_mut() {
                    rows.truncate(rows_from_vectors);
                }
            }

            // D. Filter (Purge)
            let mut final_ids = Vec::with_capacity(merged_ids.len());
            let mut final_flat = Vec::with_capacity(merged_flat.len());
            let mut final_payload_rows: Option<Vec<UnifiedPayloadRow>> = merged_payload_rows
                .as_ref()
                .map(|rows| Vec::with_capacity(rows.len()));

            for (row_idx, id) in merged_ids.into_iter().enumerate() {
                if !tombstone_snapshot.contains(&id) {
                    let start = row_idx * dim;
                    let end = start + dim;
                    if end > merged_flat.len() {
                        break;
                    }
                    final_ids.push(id);
                    final_flat.extend_from_slice(&merged_flat[start..end]);
                    if let Some(rows) = merged_payload_rows.as_ref() {
                        if row_idx >= rows.len() {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                format!(
                                    "promotion payload row index out of bounds for bucket {}",
                                    bucket_id
                                ),
                            ));
                        }
                        if let Some(target_rows) = final_payload_rows.as_mut() {
                            target_rows.push(rows[row_idx].clone());
                        }
                    }
                }
            }
            if final_payload_rows.is_some() && merged_schema.is_none() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "promotion payload rows present without schema for bucket {}",
                        bucket_id
                    ),
                ));
            }

            let final_count = final_ids.len() as u32;

            // E. Write to S3
            let write_result = self
                .persistence
                .write_remote_bucket_unified_flat_with_payload_result(
                    bucket_id,
                    &final_ids,
                    &final_flat,
                    dim,
                    merged_schema.as_ref(),
                    final_payload_rows.as_deref(),
                )
                .await?;
            self.rebuild_routing_bucket_from_snapshot(
                bucket_id,
                merged_schema.as_ref(),
                final_payload_rows.as_deref(),
                &final_ids,
            )?;

            // --- END EXPLICIT LOGIC ---

            // 3. Finalize Registry (Tiered)
            let crate::persistence::RemoteUnifiedWriteResult {
                run_id: new_run_id,
                object_path: new_remote_path,
                payload_index_meta,
                ..
            } = write_result;
            let new_remote_fingerprint = match self
                .persistence
                .object_fingerprint_for_path(&new_remote_path)
                .await
            {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!(
                        "Janitor: failed to read remote fingerprint for {}: {}",
                        new_remote_path,
                        e
                    );
                    String::new()
                }
            };
            let tiered_class = StorageClass::Tiered {
                remote_path: new_remote_path.clone(),
                local_path: new_filename.clone(),
            };
            self.bucket_manager.register_bucket_with_count(
                bucket_id,
                new_remote_path.clone(),
                tiered_class,
                final_count,
            );
            self.invalidate_catalog_buckets(&[bucket_id]);

            // Keep router counts in sync after promotion.
            self.index.update_router_count(bucket_id, final_count, None);

            // 4. ⚡ RECONCILE: Prune the specific deletions we just handled
            self.bucket_manager
                .prune_tombstones(bucket_id, &tombstone_snapshot);

            // 5. Update Manifest
            // Get fresh stats for atomic update (retains any NEW tombstones that arrived during upload)
            if let Some(stats) = self.bucket_manager.get_bucket_stats(bucket_id) {
                self.manifest.apply_atomic(|m| {
                    m.update_bucket_remote_meta_with_payload_index(
                        bucket_id,
                        new_run_id.clone(),
                        new_remote_path.clone(),
                        new_remote_fingerprint.clone(),
                        payload_index_meta,
                    );
                    m.update_bucket_stats(bucket_id, final_count as u64, stats.tombstone_count);
                })?;
            }

            // 6. Cleanup
            self.cleanup
                .delete_local_best_effort(&staging_filename, "promotion-rotated-staging")
                .await;
            if let Some(old_path) = remote_path_opt {
                self.cleanup
                    .delete_remote_best_effort(&old_path, "promotion-old-remote")
                    .await;
            }
        }
        Ok(())
    }

    /// Executes the physical split operation.
    pub(crate) async fn perform_split(&self, bucket_id: u32) -> io::Result<()> {
        info!("Janitor: ✂️ Calculating split for Bucket {}", bucket_id);

        // 1. Snapshot Parent State
        let parent_stats = self
            .bucket_manager
            .get_bucket_stats(bucket_id)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "Parent bucket missing"))?;

        // a. Calculate
        // b. Calculate (The Brain)
        // This is read-only and safe.
        let proposal = match self.index.calculate_split(bucket_id).await? {
            Ok(p) => p,
            Err(status) => {
                info!("Janitor: Split aborted: {}", status.to_str());
                // TODO: Update Ignore Map here to prevent retry loops
                return Ok(());
            }
        };

        // c. ️ SAFETY CHECK ️
        let child_sum = proposal.left.count + proposal.right.count + proposal.loopback.len();
        if (child_sum as u32) < parent_stats.total_count {
            tracing::error!(
                "Janitor: 🚨 CRITICAL SPLIT FAILURE! Data loss detected. Parent: {}, Children: {}. Aborting.",
                parent_stats.total_count,
                child_sum
            );
            return Ok(());
        }

        let dim = self.index.get_dim();
        let (source_ids, source_flat, source_schema, source_rows) = self
            .read_bucket_flat_with_payload(bucket_id, "split-source")
            .await?;
        if source_ids.len() != child_sum {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "split source row mismatch for bucket {}: source_rows={}, split_rows={}",
                    bucket_id,
                    source_ids.len(),
                    child_sum
                ),
            ));
        }

        let mut payload_lookup = match source_rows.as_ref() {
            Some(rows) => Some(build_payload_row_lookup(
                bucket_id,
                "split-source",
                &source_ids,
                &source_flat,
                dim,
                rows,
            )?),
            None => None,
        };
        let left_payload_rows = if let Some(lookup) = payload_lookup.as_mut() {
            Some(take_payload_rows_for_group(
                bucket_id,
                "split-left",
                &proposal.left,
                dim,
                lookup,
            )?)
        } else {
            None
        };
        let right_payload_rows = if let Some(lookup) = payload_lookup.as_mut() {
            Some(take_payload_rows_for_group(
                bucket_id,
                "split-right",
                &proposal.right,
                dim,
                lookup,
            )?)
        } else {
            None
        };
        let loopback_payload_rows = if let Some(lookup) = payload_lookup.as_mut() {
            Some(take_payload_rows_for_loopback(
                bucket_id,
                "split-loopback",
                &proposal.loopback,
                dim,
                lookup,
            )?)
        } else {
            None
        };
        if let Some(lookup) = payload_lookup.as_ref() {
            ensure_payload_lookup_drained(bucket_id, "split-source", lookup)?;
        }

        // 2. Write New Buckets (Staging)
        // We allocate new IDs for the children
        // Note: allocate_next_bucket_id is atomic on Index
        let id_left = self.index.allocate_next_bucket_id();
        let id_right = self.index.allocate_next_bucket_id();

        // Write Left
        let count_l = self
            .staging
            .append_batch_with_payload(
                id_left,
                &proposal.left,
                source_schema.as_ref(),
                left_payload_rows.as_deref(),
            )
            .await?;
        let file_l = self.staging.get_active_filename(id_left);

        // Write Right
        let count_r = self
            .staging
            .append_batch_with_payload(
                id_right,
                &proposal.right,
                source_schema.as_ref(),
                right_payload_rows.as_deref(),
            )
            .await?;
        let file_r = self.staging.get_active_filename(id_right);

        // Update KV mapping for new buckets
        self.update_kv_mapping(id_left, &proposal.left.ids);
        self.update_kv_mapping(id_right, &proposal.right.ids);
        let loopback_ids: Vec<u64> = proposal.loopback.iter().map(|(id, _)| *id).collect();

        // Remove KV mapping for loopback (now L0-only)
        if !loopback_ids.is_empty() {
            self.remove_kv_mapping(&loopback_ids);
        }

        // 3. Register New Files (Local)
        self.bucket_manager
            .register_bucket(id_left, file_l, StorageClass::Local);
        self.bucket_manager
            .register_bucket(id_right, file_r, StorageClass::Local);
        self.rebuild_routing_bucket_from_snapshot(
            id_left,
            source_schema.as_ref(),
            left_payload_rows.as_deref(),
            &proposal.left.ids,
        )?;
        self.rebuild_routing_bucket_from_snapshot(
            id_right,
            source_schema.as_ref(),
            right_payload_rows.as_deref(),
            &proposal.right.ids,
        )?;

        // 4. Atomic Commit (Manifest + Router)
        // We perform all metadata updates in one go.
        self.manifest.apply_atomic(|m| {
            // Remove Old
            m.remove_bucket(bucket_id);

            // Add New
            m.add_bucket(id_left, String::new(), proposal.left.centroid.clone());
            m.update_bucket_stats(id_left, count_l, 0);

            m.add_bucket(id_right, String::new(), proposal.right.centroid.clone());
            m.update_bucket_stats(id_right, count_r, 0);
        })?;

        // 5. Update In-Memory Router (Critical for Search)
        // We need to expose a method on Index to update router, or do it here if we have access.
        // Ideally, Index listens to Manifest updates or we call a method.
        // Call a helper on Index for the split update.
        self.index
            .apply_split_update(
                bucket_id,
                (id_left, proposal.left.centroid.unwrap(), count_l as u32),
                (id_right, proposal.right.centroid.unwrap(), count_r as u32),
            )
            .await;

        // 6. Handle Defectors (Loopback)
        if !proposal.loopback.is_empty() {
            info!(
                "Janitor: ↩️ Looping back {} defectors",
                proposal.loopback.len()
            );
            if let Some(rows) = loopback_payload_rows.as_ref() {
                let core_schema = source_schema.as_ref().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "split loopback payload rows present without schema for bucket {}",
                            bucket_id
                        ),
                    )
                })?;
                let core_schema = unified_payload_schema_to_core(core_schema);
                let core_rows = unified_payload_rows_to_core(rows);
                self.index.insert_batch_with_payload(
                    &proposal.loopback,
                    Some(&core_schema),
                    Some(core_rows.as_slice()),
                )?;
            } else {
                self.index.insert_batch(&proposal.loopback)?;
            }
        }

        // 7. Cleanup Old
        // We delete the STAGING file for the old bucket immediately if it exists.
        // Remote files are handled by Reaper later.
        let old_staging = self.staging.get_active_filename(bucket_id);
        self.cleanup
            .delete_local_best_effort(&old_staging, "split-old-staging")
            .await;
        self.sync_kv_best_effort("split");
        self.invalidate_routing_buckets(&[bucket_id]);
        if !loopback_ids.is_empty() {
            self.remove_routing_ids(&loopback_ids);
        }
        self.invalidate_catalog_buckets(&[bucket_id, id_left, id_right]);

        info!(
            "Janitor: Split Complete. {} -> {}, {}",
            bucket_id, id_left, id_right
        );
        Ok(())
    }

    async fn perform_merge(&self, zombie_id: u32) -> io::Result<()> {
        info!("Janitor: 🚑 Merging Zombie Bucket {}", zombie_id);

        let proposal = match self.index.calculate_merge(zombie_id).await? {
            Ok(p) => p,
            Err(status) => {
                info!("Janitor: Merge aborted: {}", status.to_str());
                return Ok(());
            }
        };

        // Handle Empty/No-Neighbor case (Delete Logic)
        if proposal.moves.is_empty() {
            info!(
                "Janitor: Zombie Bucket {} is empty or isolated. Deleting.",
                zombie_id
            );

            // 1. Remove from Metadata
            self.manifest.apply_atomic(|m| m.remove_bucket(zombie_id))?;
            self.index.apply_merge_update(zombie_id, &[]).await;
            self.invalidate_routing_buckets(&[zombie_id]);
            self.invalidate_catalog_buckets(&[zombie_id]);

            // 2. ⚡ NEW: Delete Physical File
            let zombie_file = self.staging.get_active_filename(zombie_id);
            self.cleanup
                .delete_local_best_effort(&zombie_file, "merge-empty-zombie")
                .await;

            return Ok(());
        }

        if proposal.moves.is_empty() {
            self.manifest.apply_atomic(|m| m.remove_bucket(zombie_id))?;
            self.index.apply_merge_update(zombie_id, &[]).await;
            self.invalidate_routing_buckets(&[zombie_id]);
            self.invalidate_catalog_buckets(&[zombie_id]);
            return Ok(());
        }

        let dim = self.index.get_dim();
        let (zombie_ids, zombie_flat, zombie_schema, zombie_rows) = self
            .read_bucket_flat_with_payload(zombie_id, "merge-zombie")
            .await?;
        let mut zombie_payload_lookup = match zombie_rows.as_ref() {
            Some(rows) => Some(build_payload_row_lookup(
                zombie_id,
                "merge-zombie",
                &zombie_ids,
                &zombie_flat,
                dim,
                rows,
            )?),
            None => None,
        };

        let mut manifest_updates = Vec::new();
        let mut files_to_delete = Vec::new();

        for (target_id, group) in &proposal.moves {
            // A. New File Name
            let new_filename = format!("bucket_{}_{}.driftu", target_id, uuid::Uuid::new_v4());

            // B. Read Old Data (to merge)
            let (mut ids, mut flat_vecs, target_schema, target_rows) = self
                .read_bucket_flat_with_payload(*target_id, "merge-target")
                .await?;
            let old_filename = self.staging.get_active_filename(*target_id);
            let target_count = ids.len();
            if group.flat_vectors.len() != group.ids.len() * dim {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "merge proposal vector mismatch for target {}: ids={}, flat={}",
                        target_id,
                        group.ids.len(),
                        group.flat_vectors.len()
                    ),
                ));
            }
            let incoming_rows = if let Some(lookup) = zombie_payload_lookup.as_mut() {
                Some(take_payload_rows_for_group(
                    zombie_id,
                    "merge-zombie-moves",
                    group,
                    dim,
                    lookup,
                )?)
            } else {
                None
            };

            // C. Merge Vectors
            ids.extend(&group.ids);
            flat_vecs.extend_from_slice(&group.flat_vectors);

            let merged_schema = match (target_schema.as_ref(), zombie_schema.as_ref()) {
                (Some(lhs), Some(rhs)) => {
                    if lhs != rhs {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "merge payload schema mismatch between target {} and zombie {}",
                                target_id, zombie_id
                            ),
                        ));
                    }
                    Some(lhs.clone())
                }
                (Some(schema), None) => Some(schema.clone()),
                (None, Some(schema)) => Some(schema.clone()),
                (None, None) => None,
            };
            let mut merged_rows: Option<Vec<UnifiedPayloadRow>> = None;
            if merged_schema.is_some() {
                let mut rows = Vec::with_capacity(ids.len());
                if target_count > 0 {
                    let existing_rows = target_rows.ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "merge target payload rows missing for bucket {} (target={})",
                                zombie_id, target_id
                            ),
                        )
                    })?;
                    rows.extend(existing_rows);
                } else if let Some(existing_rows) = target_rows {
                    rows.extend(existing_rows);
                }

                if !group.ids.is_empty() {
                    let mut moved_rows = incoming_rows.ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "merge incoming payload rows missing for zombie {} -> target {}",
                                zombie_id, target_id
                            ),
                        )
                    })?;
                    rows.append(&mut moved_rows);
                } else if let Some(mut moved_rows) = incoming_rows {
                    rows.append(&mut moved_rows);
                }

                if rows.len() != ids.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "merge payload row mismatch for target {}: ids={}, payload_rows={}",
                            target_id,
                            ids.len(),
                            rows.len()
                        ),
                    ));
                }
                merged_rows = Some(rows);
            } else if target_rows.is_some()
                || incoming_rows.is_some()
                || target_schema.is_some()
                || zombie_schema.is_some()
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "merge payload presence mismatch for target {} and zombie {}",
                        target_id, zombie_id
                    ),
                ));
            }

            // D. Recalculate Stats (Sum & Centroid)
            let count = ids.len();
            let mut new_sum = vec![0.0; dim];
            for chunk in flat_vecs.chunks_exact(dim) {
                for i in 0..dim {
                    new_sum[i] += chunk[i];
                }
            }

            // ⚡ NEW: Calculate Centroid
            let mut new_centroid = vec![0.0; dim];
            if count > 0 {
                for i in 0..dim {
                    new_centroid[i] = new_sum[i] / count as f32;
                }
            }

            // E. Write File
            let mut merged_group = PartitionGroup::new(dim, None);
            merged_group.ids = ids;
            merged_group.flat_vectors = flat_vecs;
            merged_group.count = count;

            self.staging
                .write_new_file_with_payload(
                    &new_filename,
                    &merged_group,
                    merged_schema.as_ref(),
                    merged_rows.as_deref(),
                )
                .await?;

            // Update KV mapping for all IDs now owned by target_id
            self.update_kv_mapping(*target_id, &merged_group.ids);
            self.rebuild_routing_bucket_from_snapshot(
                *target_id,
                merged_schema.as_ref(),
                merged_rows.as_deref(),
                &merged_group.ids,
            )?;

            // Track update: (ID, Count, Sum, Centroid, Filename)
            manifest_updates.push((
                *target_id,
                count as u64,
                new_sum,
                new_centroid,
                new_filename,
            ));
            files_to_delete.push(old_filename);
        }
        if let Some(lookup) = zombie_payload_lookup.as_ref() {
            ensure_payload_lookup_drained(zombie_id, "merge-zombie", lookup)?;
        }

        // 3. Atomic Commit (Manifest)
        self.manifest.apply_atomic(|m| {
            m.remove_bucket(zombie_id);
            for (id, count, _, centroid, _) in &manifest_updates {
                // Use add_bucket to update Centroid + Count in Manifest
                // add_bucket handles upsert correctly
                m.add_bucket(*id, String::new(), Some(centroid.clone()));
                m.update_bucket_stats(*id, *count, 0);
            }
        })?;

        // 4. Update Runtime (Router)
        // Convert to format required by apply_merge_update: (id, count, sum, centroid)
        let router_updates: Vec<_> = manifest_updates
            .iter()
            .map(|(id, c, s, cent, _)| (*id, *c, s.clone(), cent.clone()))
            .collect();

        self.index
            .apply_merge_update(zombie_id, &router_updates)
            .await;

        // 5. Update Storage (BucketManager)
        for (id, count, sum, _, filename) in &manifest_updates {
            // Register overwrites the entry in BucketManager with a fresh state (empty sum)
            self.bucket_manager
                .register_bucket(*id, filename.clone(), StorageClass::Local);
            self.staging.set_active_filename(*id, filename.clone());

            // Re-inject the correct sum so Drift Calculation works immediately
            self.bucket_manager
                .update_bucket_drift(*id, sum, *count as u32)?;
        }
        let mut invalidated = Vec::with_capacity(manifest_updates.len() + 1);
        invalidated.push(zombie_id);
        invalidated.extend(manifest_updates.iter().map(|(id, _, _, _, _)| *id));
        self.invalidate_routing_buckets(&[zombie_id]);
        self.invalidate_catalog_buckets(invalidated.as_slice());

        // 6. Cleanup
        let zombie_file = self.staging.get_active_filename(zombie_id);
        self.cleanup
            .delete_local_best_effort(&zombie_file, "merge-zombie-old")
            .await;

        for f in files_to_delete {
            if !manifest_updates.iter().any(|(_, _, _, _, new)| *new == f) {
                self.cleanup
                    .delete_local_best_effort(&f, "merge-neighbor-old")
                    .await;
            }
        }
        self.sync_kv_best_effort("merge");

        info!("Janitor: Merge Complete. {} scattered.", zombie_id);
        Ok(())
    }

    pub(crate) async fn check_maintainance(&self) {
        // 1. Global Cooling (Decay temperature for all buckets)
        // Ask the storage layer to decay active temperatures.
        // const TEMPERATURE_COOL_FACTOR: f32 = 0.98;
        self.bucket_manager
            .tick_cooling(self.vars.temperature_cool_factor);

        // 2. Snapshot State
        let buckets = self.manifest.get_state().get_buckets().clone();
        let max_cap = self.vars.max_bucket_capacity as f32;

        // Snapshot Router for Centroids (needed for Drift calc)
        let (router_centroids, router_ids) = self.index.get_router().read().get_snapshot();
        let dim = self.index.get_dim();

        // Helper to find centroid for a bucket ID
        let centroid_map: HashMap<u32, Vec<f32>> = router_ids
            .iter()
            .zip(router_centroids.chunks(dim))
            .map(|(id, vec)| (*id, vec.to_vec()))
            .collect();

        for b in buckets {
            // 3. Fetch live stats from BucketManager
            // returns BucketStats { tombstone_count, total_count, temperature, ... }
            let stats = match self.bucket_manager.get_bucket_stats(b.id) {
                Some(s) => s,
                None => continue, // Bucket might be deleted or not yet registered
            };

            // Fetch vector sum for drift calculation
            let (current_sum, _drift_count) = self
                .bucket_manager
                .get_bucket_drift_stats(b.id)
                .unwrap_or((vec![], 0));

            let current_count = stats.total_count;

            // 4. Calculate Metrics (Manually, since V1 BucketHeader logic is gone)

            // --- A. Calculate Urgency ---
            // Formula: (Emptiness / (Temp + epsilon)) + (Beta * ZombieRatio)
            let total = current_count as f32;
            let dead = stats.tombstone_count as f32;
            let temp = stats.temperature; // Already decayed by tick_cooling above

            let live = (total - dead).max(0.0);

            let emptiness = if live < max_cap {
                (max_cap - live) / max_cap
            } else {
                0.0
            };

            let zombie_ratio = if total > 0.0 { dead / total } else { 0.0 };

            const EPSILON: f32 = 0.001;
            const BETA: f32 = 3.0;

            let urgency = (emptiness / (temp + EPSILON)) + (BETA * zombie_ratio);

            // --- B. Calculate Drift ---
            // Formula: Distance(Mean, Centroid)
            let drift_score = if current_count > 0 && !current_sum.is_empty() {
                if let Some(target_centroid) = centroid_map.get(&b.id) {
                    let n = current_count as f32;
                    let mut dist_sq = 0.0;
                    for i in 0..dim {
                        let mean = current_sum[i] / n;
                        let diff = mean - target_centroid[i];
                        dist_sq += diff * diff;
                    }
                    dist_sq.sqrt()
                } else {
                    0.0
                }
            } else {
                0.0
            };

            let cap_ratio = current_count as f32 / max_cap;

            // 5. Decision Matrix

            // A. SPLIT: Too Full OR High Drift
            if cap_ratio > 1.0
                || (cap_ratio > self.vars.split_threshold
                    && drift_score > self.vars.drift_threshold)
            {
                info!(
                    "Janitor: ✂️ Triggering Split for Bucket {} (Cap: {:.2}, Drift: {:.4})",
                    b.id, cap_ratio, drift_score
                );

                if let Err(e) = self.perform_split(b.id).await {
                    error!("Janitor: Split failed for {}: {}", b.id, e);
                }
                break; // One op per tick
            }
            // B. MERGE: High Urgency
            // Using the urgency score we calculated manually above
            else if urgency > self.vars.urgency_threshold {
                info!(
                    "Janitor: 🚑 Triggering Merge for Zombie Bucket {} (Urgency: {:.2}, Count: {})",
                    b.id, urgency, current_count
                );

                if let Err(e) = self.perform_merge(b.id).await {
                    error!("Janitor: Merge failed for {}: {}", b.id, e);
                }
                break; // One op per tick
            }
        }
    }
}
