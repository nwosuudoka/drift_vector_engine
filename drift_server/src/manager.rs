use crate::config::Config;
use crate::filter_metadata_catalog::FilterMetadataCatalog;
use crate::filter_planner_diagnostics::FilterPlannerDiagnosticsSnapshot;
use crate::global_filter_routing_index::GlobalFilterRoutingIndex;
use crate::global_metadata_snapshot::{FilterCatalogSnapshot, GlobalRoutingSnapshot};
use crate::janitor::{Janitor, JanitorConfig, JanitorVars};
use crate::local_staging::LocalStagingManager;
use crate::manifest::ServerManifestManager;
use crate::persistence::PersistenceManager;
use crate::recovery::RecoveryManager;
use crate::storage_factory::StorageFactory;
use drift_core::index::VectorIndex;
use drift_core::lock_manager::BucketCoordinator;
use drift_core::manifest::ManifestWrapper;
use drift_core::math::Metric;
use drift_core::payload::PayloadSchema as CorePayloadSchema;
use drift_core::wal::WalManager;
use drift_kv::bitstore::BitStore;
use drift_storage::bucket_manager::BucketManager;
use drift_traits::StorageEngine;
use opendal::Operator;
use opendal::services::Fs;
use parking_lot::{Mutex, RwLock as ParkingRwLock};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{error, info, warn};

const KV_FORCE_REBUILD_ON_STARTUP_ENV: &str = "DRIFT_KV_FORCE_REBUILD_ON_STARTUP";
const KV_VALIDATE_MAX_BUCKETS_ENV: &str = "DRIFT_KV_VALIDATE_MAX_BUCKETS";
const KV_VALIDATE_IDS_PER_BUCKET_ENV: &str = "DRIFT_KV_VALIDATE_IDS_PER_BUCKET";
const KV_VALIDATE_MAX_BUCKETS_DEFAULT: usize = 8;
const KV_VALIDATE_IDS_PER_BUCKET_DEFAULT: usize = 4;
const GLOBAL_METADATA_PERSIST_ENV: &str = "DRIFT_GLOBAL_METADATA_PERSIST";

pub struct Collection {
    pub index: Arc<VectorIndex>,
    pub name: String,
    pub staging: Arc<LocalStagingManager>,
    pub persistence: Arc<PersistenceManager>,
    pub bucket_manager: Arc<BucketManager>,
    pub payload_schema: Arc<ParkingRwLock<Option<CorePayloadSchema>>>,
    pub last_filter_planner_diagnostics: Arc<ParkingRwLock<FilterPlannerDiagnosticsSnapshot>>,
    pub filter_metadata_catalog: Arc<ParkingRwLock<FilterMetadataCatalog>>,
    pub global_filter_routing_index: Arc<ParkingRwLock<GlobalFilterRoutingIndex>>,
    // We hold the handle so it runs in the background. Dropping this struct (e.g. shutdown) will abort it.
    pub janitor_task: tokio::task::JoinHandle<()>,
}

pub struct CollectionManager {
    config: Config,
    collections: RwLock<HashMap<String, Arc<Collection>>>,
}

impl CollectionManager {
    pub fn new(config: Config) -> Self {
        std::fs::create_dir_all(&config.wal_dir).expect("Failed to create WAL root");
        Self {
            config,
            collections: RwLock::new(HashMap::new()),
        }
    }

    fn load_manifest_meta(coll_root: &Path) -> std::io::Result<Option<(usize, Metric)>> {
        let path = coll_root.join("manifest.pb");
        if !path.exists() {
            return Ok(None);
        }

        let bytes = std::fs::read(&path)?;
        let wrapper = ManifestWrapper::from_bytes(&bytes)
            .map_err(|e| std::io::Error::other(format!("Protobuf decode error: {}", e)))?;
        let metric = wrapper
            .metric()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(Some((wrapper.get_dim() as usize, metric)))
    }

    fn create_local_operator(path: &Path) -> std::io::Result<Operator> {
        let builder = Fs::default().root(path.to_str().unwrap());
        Ok(Operator::new(builder)
            .map_err(std::io::Error::other)?
            .finish())
    }

    fn env_truthy(name: &str) -> bool {
        std::env::var(name)
            .ok()
            .map(|v| {
                matches!(
                    v.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false)
    }

    fn env_usize(name: &str, default: usize) -> usize {
        match std::env::var(name) {
            Ok(raw) => match raw.trim().parse::<usize>() {
                Ok(v) if v > 0 => v,
                _ => {
                    warn!(
                        "Manager: invalid {}='{}'; using default {}",
                        name, raw, default
                    );
                    default
                }
            },
            Err(_) => default,
        }
    }

    fn clear_filter_metadata_catalog_for_recovery(
        collection_name: &str,
        catalog: &Arc<ParkingRwLock<FilterMetadataCatalog>>,
    ) {
        let mut guard = catalog.write();
        let stats = guard.stats();
        if stats.bucket_count > 0 {
            info!(
                "Manager: clearing filter metadata catalog for '{}' before recovery (buckets={}, exact_values={})",
                collection_name, stats.bucket_count, stats.exact_value_memberships
            );
        }
        guard.clear();
    }

    fn clear_global_filter_routing_index_for_recovery(
        collection_name: &str,
        index: &Arc<ParkingRwLock<GlobalFilterRoutingIndex>>,
    ) {
        let mut guard = index.write();
        let stats = guard.stats();
        if stats.id_entry_count > 0 || stats.value_entry_count > 0 {
            info!(
                "Manager: clearing global filter routing index for '{}' before recovery (ids={}, values={}, value_bucket_pairs={})",
                collection_name,
                stats.id_entry_count,
                stats.value_entry_count,
                stats.value_bucket_pair_count
            );
        }
        guard.clear();
    }

    fn global_metadata_persist_enabled() -> bool {
        match std::env::var(GLOBAL_METADATA_PERSIST_ENV) {
            Ok(raw) => match raw.trim().to_ascii_lowercase().as_str() {
                "0" | "false" | "no" | "off" => false,
                "1" | "true" | "yes" | "on" => true,
                other => {
                    warn!(
                        "Manager: invalid {}='{}'; defaulting to enabled",
                        GLOBAL_METADATA_PERSIST_ENV, other
                    );
                    true
                }
            },
            Err(_) => true,
        }
    }

    fn bucket_live_count(bucket_manager: &Arc<BucketManager>, bucket_id: u32) -> Option<u32> {
        bucket_manager
            .get_bucket_stats(bucket_id)
            .map(|stats| stats.total_count.saturating_sub(stats.tombstone_count))
    }

    async fn hydrate_global_metadata_from_manifest(
        collection_name: &str,
        manifest: &Arc<ServerManifestManager>,
        persistence: &Arc<PersistenceManager>,
        bucket_manager: &Arc<BucketManager>,
        filter_metadata_catalog: &Arc<ParkingRwLock<FilterMetadataCatalog>>,
        global_filter_routing_index: &Arc<ParkingRwLock<GlobalFilterRoutingIndex>>,
    ) {
        let pointer = manifest.get_state().global_metadata_pointer();
        let Some(pointer) = pointer else {
            return;
        };

        let snapshot = match persistence
            .read_global_metadata_snapshot_path(&pointer.path)
            .await
        {
            Ok(Some(snapshot)) => snapshot,
            Ok(None) => {
                warn!(
                    "Manager: global metadata pointer for '{}' references missing object '{}'",
                    collection_name, pointer.path
                );
                return;
            }
            Err(err) => {
                warn!(
                    "Manager: failed to read global metadata snapshot for '{}' from '{}': {}",
                    collection_name, pointer.path, err
                );
                return;
            }
        };

        if pointer.format_version != 0 && pointer.format_version != snapshot.format_version {
            warn!(
                "Manager: global metadata format mismatch for '{}': manifest={} snapshot={}; skipping hydration",
                collection_name, pointer.format_version, snapshot.format_version
            );
            return;
        }

        let valid_routing_buckets: HashSet<u32> = snapshot
            .routing
            .bucket_tokens
            .iter()
            .filter_map(|token| {
                let version = bucket_manager.get_version(token.bucket_id)?;
                let live_count = Self::bucket_live_count(bucket_manager, token.bucket_id)?;
                (version.path == token.bucket_path && live_count == token.bucket_live_count)
                    .then_some(token.bucket_id)
            })
            .collect();

        let mut routing_snapshot = GlobalRoutingSnapshot::default();
        routing_snapshot.bucket_tokens = snapshot
            .routing
            .bucket_tokens
            .iter()
            .filter(|token| valid_routing_buckets.contains(&token.bucket_id))
            .cloned()
            .collect();
        routing_snapshot.id_entries = snapshot
            .routing
            .id_entries
            .iter()
            .filter(|entry| valid_routing_buckets.contains(&entry.bucket_id))
            .cloned()
            .collect();
        routing_snapshot.complete_exact_fields_by_bucket = snapshot
            .routing
            .complete_exact_fields_by_bucket
            .iter()
            .filter(|entry| valid_routing_buckets.contains(&entry.bucket_id))
            .cloned()
            .collect();

        let mut catalog_snapshot = FilterCatalogSnapshot::default();
        catalog_snapshot.buckets = snapshot
            .catalog
            .buckets
            .iter()
            .filter(|bucket| {
                let Some(version) = bucket_manager.get_version(bucket.bucket_id) else {
                    return false;
                };
                if version.path != bucket.bucket_path {
                    return false;
                }
                match bucket.bucket_live_count {
                    Some(expected) => {
                        Self::bucket_live_count(bucket_manager, bucket.bucket_id) == Some(expected)
                    }
                    None => true,
                }
            })
            .cloned()
            .collect();

        {
            let mut routing = global_filter_routing_index.write();
            routing.import_snapshot(&routing_snapshot);
        }
        {
            let mut catalog = filter_metadata_catalog.write();
            catalog.import_snapshot(&catalog_snapshot);
        }

        let routing_stats = global_filter_routing_index.read().stats();
        let catalog_stats = filter_metadata_catalog.read().stats();
        info!(
            "Manager: hydrated global metadata for '{}' (routing_ids={}, routing_values={}, catalog_buckets={})",
            collection_name,
            routing_stats.id_entry_count,
            routing_stats.value_entry_count,
            catalog_stats.bucket_count
        );
    }

    fn recreate_kv_store(kv_base_path: &Path) -> std::io::Result<Arc<BitStore>> {
        let idx = kv_base_path.with_extension("idx");
        let dat = kv_base_path.with_extension("dat");

        let _ = std::fs::remove_file(&idx);
        let _ = std::fs::remove_file(&dat);

        Ok(Arc::new(
            BitStore::new(kv_base_path).map_err(std::io::Error::other)?,
        ))
    }

    async fn should_rebuild_kv_mapping(
        collection_name: &str,
        manifest: &Arc<ServerManifestManager>,
        bucket_manager: &Arc<BucketManager>,
        kv: &Arc<BitStore>,
    ) -> bool {
        if Self::env_truthy(KV_FORCE_REBUILD_ON_STARTUP_ENV) {
            info!(
                "Manager: KV startup rebuild forced for '{}' via {}",
                collection_name, KV_FORCE_REBUILD_ON_STARTUP_ENV
            );
            return true;
        }

        let buckets_with_data: Vec<u32> = manifest
            .get_state()
            .get_buckets()
            .iter()
            .filter(|b| b.vector_count > 0)
            .map(|b| b.id)
            .collect();

        if buckets_with_data.is_empty() {
            return false;
        }

        let kv_has_entries = match kv.iter().next() {
            Some(Ok(_)) => true,
            Some(Err(e)) => {
                warn!(
                    "Manager: KV iterator error during startup validation for '{}': {}",
                    collection_name, e
                );
                return true;
            }
            None => false,
        };

        if !kv_has_entries {
            info!(
                "Manager: KV appears empty for '{}' with {} non-empty buckets; scheduling rebuild.",
                collection_name,
                buckets_with_data.len()
            );
            return true;
        }

        let max_buckets =
            Self::env_usize(KV_VALIDATE_MAX_BUCKETS_ENV, KV_VALIDATE_MAX_BUCKETS_DEFAULT);
        let ids_per_bucket = Self::env_usize(
            KV_VALIDATE_IDS_PER_BUCKET_ENV,
            KV_VALIDATE_IDS_PER_BUCKET_DEFAULT,
        );

        let mut checked_ids = 0usize;
        for bucket_id in buckets_with_data.into_iter().take(max_buckets) {
            let ids = match bucket_manager.fetch_bucket(bucket_id).await {
                Ok((ids, _)) => ids,
                Err(e) => {
                    warn!(
                        "Manager: KV startup validation could not fetch bucket {} for '{}': {}",
                        bucket_id, collection_name, e
                    );
                    continue;
                }
            };

            let expected_bucket = bucket_id;
            for id in ids.into_iter().take(ids_per_bucket) {
                let id_bytes = id.to_le_bytes();
                let mapped_bucket = match kv.get(&id_bytes) {
                    Ok(Some(val)) => match <[u8; 4]>::try_from(val.as_slice()) {
                        Ok(raw) => u32::from_le_bytes(raw),
                        Err(_) => {
                            warn!(
                                "Manager: KV startup validation found malformed value for id={} in '{}'",
                                id, collection_name
                            );
                            return true;
                        }
                    },
                    Ok(None) => {
                        warn!(
                            "Manager: KV startup validation missing mapping for id={} in '{}'",
                            id, collection_name
                        );
                        return true;
                    }
                    Err(e) => {
                        warn!(
                            "Manager: KV startup validation read error for id={} in '{}': {}",
                            id, collection_name, e
                        );
                        return true;
                    }
                };

                if mapped_bucket != expected_bucket {
                    warn!(
                        "Manager: KV startup validation mismatch for id={} in '{}': expected bucket {}, got {}",
                        id, collection_name, expected_bucket, mapped_bucket
                    );
                    return true;
                }
                checked_ids = checked_ids.saturating_add(1);
            }
        }

        info!(
            "Manager: KV startup validation passed for '{}' (checked {} id mapping(s)).",
            collection_name, checked_ids
        );
        false
    }

    async fn rebuild_kv_mapping(
        collection_name: &str,
        manifest: &Arc<ServerManifestManager>,
        bucket_manager: &Arc<BucketManager>,
        kv: &Arc<BitStore>,
    ) -> std::io::Result<()> {
        let buckets = manifest.get_state().get_buckets().clone();
        let mut rebuilt_ids = 0usize;
        let mut scanned_buckets = 0usize;

        for b in buckets.into_iter().filter(|b| b.vector_count > 0) {
            match bucket_manager.fetch_bucket(b.id).await {
                Ok((ids, _)) => {
                    scanned_buckets = scanned_buckets.saturating_add(1);
                    let bucket_bytes = b.id.to_le_bytes();
                    for id in ids {
                        kv.put(&id.to_le_bytes(), &bucket_bytes)
                            .map_err(std::io::Error::other)?;
                        rebuilt_ids = rebuilt_ids.saturating_add(1);
                    }
                }
                Err(e) => {
                    warn!(
                        "Manager: KV rebuild skipped bucket {} for '{}': {}",
                        b.id, collection_name, e
                    );
                }
            }
        }

        kv.sync().map_err(std::io::Error::other)?;
        info!(
            "Manager: KV rebuild complete for '{}' (buckets_scanned={}, ids_mapped={})",
            collection_name, scanned_buckets, rebuilt_ids
        );
        Ok(())
    }

    async fn init_kv_store(
        collection_name: &str,
        kv_base_path: &Path,
        manifest: &Arc<ServerManifestManager>,
        bucket_manager: &Arc<BucketManager>,
    ) -> std::io::Result<Arc<BitStore>> {
        let mut kv = Arc::new(BitStore::new(kv_base_path).map_err(std::io::Error::other)?);
        if Self::should_rebuild_kv_mapping(collection_name, manifest, bucket_manager, &kv).await {
            info!("Manager: rebuilding KV mapping for '{}'", collection_name);
            kv = Self::recreate_kv_store(kv_base_path)?;
            Self::rebuild_kv_mapping(collection_name, manifest, bucket_manager, &kv).await?;
        }
        Ok(kv)
    }

    pub async fn get_or_create(
        &self,
        name: &str,
        dim_hint: Option<usize>,
        max_bucket_capacity: Option<usize>,
        create_metric: Option<Metric>,
    ) -> std::io::Result<Arc<Collection>> {
        // 1. Fast Path
        {
            // await the lock
            let map = self.collections.read().await;
            if let Some(coll) = map.get(name) {
                if let Some(d) = dim_hint
                    && coll.index.get_dim() != d
                {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        format!(
                            "Dimension mismatch: Existing={}, Requested={}",
                            coll.index.get_dim(),
                            d
                        ),
                    ));
                }
                if let Some(requested_metric) = create_metric
                    && coll.index.metric() != requested_metric
                {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        format!(
                            "Metric mismatch: Existing={}, Requested={}",
                            coll.index.metric(),
                            requested_metric
                        ),
                    ));
                }
                return Ok(coll.clone());
            }
        }

        // 2. Slow Path
        let mut map = self.collections.write().await;
        if let Some(coll) = map.get(name) {
            if let Some(d) = dim_hint
                && coll.index.get_dim() != d
            {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!(
                        "Dimension mismatch: Existing={}, Requested={}",
                        coll.index.get_dim(),
                        d
                    ),
                ));
            }
            if let Some(requested_metric) = create_metric
                && coll.index.metric() != requested_metric
            {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!(
                        "Metric mismatch: Existing={}, Requested={}",
                        coll.index.metric(),
                        requested_metric
                    ),
                ));
            }
            return Ok(coll.clone());
        }

        info!("Manager: Initializing collection '{}' (v3)", name);

        // --- PATH RESOLUTION ---
        let wal_dir = self.config.wal_dir.join(name);
        let coll_root = self.config.data_dir.join(name);
        let staging_dir = coll_root.join("staging");
        let kv_dir = coll_root.join("kv");

        let existing_meta = Self::load_manifest_meta(&coll_root)?;
        let metric = match (create_metric, existing_meta.map(|(_, m)| m)) {
            (Some(requested), Some(existing)) if requested != existing => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!(
                        "Metric mismatch: Existing={}, Requested={}",
                        existing, requested
                    ),
                ));
            }
            (Some(requested), _) => requested,
            (None, Some(existing)) => existing,
            (None, None) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!(
                        "Collection '{}' does not exist. Create it first with CreateCollection",
                        name
                    ),
                ));
            }
        };

        let dim = if let Some((existing_dim, _)) = existing_meta {
            if let Some(requested_dim) = dim_hint
                && requested_dim != existing_dim
            {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!(
                        "Dimension mismatch: Existing={}, Requested={}",
                        existing_dim, requested_dim
                    ),
                ));
            }
            existing_dim
        } else {
            dim_hint.ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "Dimension is required when creating a new collection",
                )
            })?
        };

        // Note: fs::create_dir_all is blocking. In a high-throughput scenario,
        // we might wrap this in spawn_blocking, but for initialization it's usually acceptable.
        std::fs::create_dir_all(&wal_dir)?;
        std::fs::create_dir_all(&coll_root)?;
        std::fs::create_dir_all(&staging_dir)?;
        std::fs::create_dir_all(&kv_dir)?;

        // --- COMPONENTS ---
        let manifest = Arc::new(ServerManifestManager::new_with_metric(
            &coll_root, dim as u32, metric,
        )?);
        let staging = Arc::new(LocalStagingManager::new(&staging_dir)?);

        let remote_op = StorageFactory::build(&self.config.storage, name)?;
        let persistence = Arc::new(PersistenceManager::new(remote_op.clone()));
        let local_op = Self::create_local_operator(&staging_dir)?;

        let coordinator = Arc::new(BucketCoordinator::new());
        let bucket_manager = Arc::new(BucketManager::new(
            local_op,
            remote_op,
            16,
            coordinator.clone(),
            metric,
        ));
        let filter_metadata_catalog =
            Arc::new(ParkingRwLock::new(FilterMetadataCatalog::default()));
        Self::clear_filter_metadata_catalog_for_recovery(name, &filter_metadata_catalog);
        let global_filter_routing_index =
            Arc::new(ParkingRwLock::new(GlobalFilterRoutingIndex::default()));
        Self::clear_global_filter_routing_index_for_recovery(name, &global_filter_routing_index);

        // --- RECOVERY (Step 1: Router) ---
        // ⚡ This .await is why we needed tokio::sync::RwLock!
        let recovery = RecoveryManager::new(&coll_root, manifest.clone());
        let (router, replay_data) = recovery.recover(&bucket_manager, dim, &wal_dir).await?;

        // --- INDEX ---
        let wal = Arc::new(Mutex::new(WalManager::new(&wal_dir)?));
        let kv = Self::init_kv_store(name, &kv_dir, &manifest, &bucket_manager).await?;

        let index = Arc::new(VectorIndex::new(
            dim,
            self.config.max_bucket_capacity,
            router,
            wal,
            bucket_manager.clone(),
            kv.clone(),
        ));

        // --- RECOVERY (Step 2: Hydrate S3 Tombstones) ---
        let persisted_deletes = persistence.load_all_tombstones().await?;
        if !persisted_deletes.is_empty() {
            info!(
                "Manager: Hydrating {} deletions from persistence...",
                persisted_deletes.len()
            );
            // L0
            {
                let mut guard = index.get_deleted_ids_inner().write();
                let mut set = (**guard).clone();
                set.extend(persisted_deletes.iter());
                *guard = Arc::new(set);
            }
            // L1
            for &id in &persisted_deletes {
                let id_bytes = id.to_le_bytes();
                if let Ok(Some(val)) = kv.get(&id_bytes)
                    && let Ok(bucket_id) = val.try_into().map(u32::from_le_bytes)
                {
                    let _ = bucket_manager.mark_delete(bucket_id, id);
                }
            }
        }

        if Self::global_metadata_persist_enabled() {
            Self::hydrate_global_metadata_from_manifest(
                name,
                &manifest,
                &persistence,
                &bucket_manager,
                &filter_metadata_catalog,
                &global_filter_routing_index,
            )
            .await;
            if !persisted_deletes.is_empty() {
                let mut routing = global_filter_routing_index.write();
                for id in &persisted_deletes {
                    routing.remove_id(*id);
                }
            }
        }

        // --- RECOVERY (Step 3: Replay WAL) ---
        if !replay_data.inserts.is_empty() {
            info!(
                "Manager: Replaying {} inserts from WAL...",
                replay_data.inserts.len()
            );
            index.insert_batch(&replay_data.inserts)?;
        }
        if !replay_data.deletes.is_empty() {
            info!(
                "Manager: Replaying {} deletes from WAL...",
                replay_data.deletes.len()
            );
            for id in replay_data.deletes {
                index.delete(id)?;
                if Self::global_metadata_persist_enabled() {
                    global_filter_routing_index.write().remove_id(id);
                }
            }
        }

        let max_bucket_capacity = max_bucket_capacity.unwrap_or(self.config.max_bucket_capacity);

        // --- JANITOR ---
        let janitor_config = JanitorConfig {
            index: index.clone(),
            manifest: manifest.clone(),
            staging: staging.clone(),
            persistence: persistence.clone(),
            bucket_manager: bucket_manager.clone(),
            filter_metadata_catalog: filter_metadata_catalog.clone(),
            global_filter_routing_index: global_filter_routing_index.clone(),
            coordinator,
            vars: JanitorVars {
                promotion_threshold_bytes: 16 * 1024 * 1024,
                check_interval: Duration::from_millis(100),
                max_bucket_capacity,
                split_threshold: 0.8,
                drift_threshold: 0.15,
                temperature_cool_factor: 0.98,
                urgency_threshold: 1.5,
            },
        };

        let janitor = Janitor::new(janitor_config);
        let task_name = name.to_string();
        let janitor_task = tokio::spawn(async move {
            info!("Janitor started for '{}'", task_name);
            janitor.run().await;
            error!("Janitor stopped for '{}'", task_name);
        });

        let collection = Arc::new(Collection {
            index,
            name: name.to_string(),
            staging,
            persistence,
            bucket_manager,
            payload_schema: Arc::new(ParkingRwLock::new(None)),
            last_filter_planner_diagnostics: Arc::new(ParkingRwLock::new(
                FilterPlannerDiagnosticsSnapshot::default(),
            )),
            filter_metadata_catalog,
            global_filter_routing_index,
            janitor_task,
        });

        map.insert(name.to_string(), collection.clone());
        info!("Manager: Collection '{}' ready.", name);
        Ok(collection)
    }

    pub async fn list_collections(&self) -> Vec<String> {
        self.collections.read().await.keys().cloned().collect()
    }
}
