use crate::config::Config;
use crate::janitor_v2::{Janitor, JanitorConfig};
use crate::local_staging::LocalStagingManager;
use crate::manifest::ServerManifestManager;
use crate::persistence_v2::PersistenceManager;
use crate::recovery::RecoveryManager;
use crate::storage_factory::StorageFactory;
use drift_core::index_v2::VectorIndex;
use drift_core::lock_manager::BucketCoordinator;
use drift_core::wal_v2::WalManager;
use drift_kv::bitstore::BitStore;
use drift_storage::bucket_manager::BucketManager;
use drift_traits::StorageEngine;
use opendal::Operator;
use opendal::services::Fs;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{error, info};

pub struct Collection {
    pub index: Arc<VectorIndex>,
    pub name: String,
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

    fn create_local_operator(path: &Path) -> std::io::Result<Operator> {
        let builder = Fs::default().root(path.to_str().unwrap());
        Ok(Operator::new(builder)
            .map_err(std::io::Error::other)?
            .finish())
    }

    pub async fn get_or_create(
        &self,
        name: &str,
        dim_hint: Option<usize>,
        max_bucket_capacity: Option<usize>,
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
                return Ok(coll.clone());
            }
        }

        // 2. Slow Path
        let mut map = self.collections.write().await;
        if let Some(coll) = map.get(name) {
            return Ok(coll.clone());
        }

        info!("Manager: Initializing collection '{}' (v2)", name);

        // --- PATH RESOLUTION ---
        let wal_dir = self.config.wal_dir.join(name);
        let coll_root = self.config.data_dir.join(name);
        let staging_dir = coll_root.join("staging");
        let kv_dir = coll_root.join("kv");

        // Note: fs::create_dir_all is blocking. In a high-throughput scenario,
        // we might wrap this in spawn_blocking, but for initialization it's usually acceptable.
        std::fs::create_dir_all(&wal_dir)?;
        std::fs::create_dir_all(&coll_root)?;
        std::fs::create_dir_all(&staging_dir)?;
        std::fs::create_dir_all(&kv_dir)?;

        let dim = dim_hint.unwrap_or(self.config.default_dim);

        // --- COMPONENTS ---
        let manifest = Arc::new(ServerManifestManager::new(&coll_root, dim as u32)?);
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
        ));

        // --- RECOVERY (Step 1: Router) ---
        // ⚡ This .await is why we needed tokio::sync::RwLock!
        let recovery = RecoveryManager::new(&coll_root, manifest.clone());
        let (router, replay_data) = recovery.recover(&bucket_manager, dim, &wal_dir).await?;

        // --- INDEX ---
        let wal = Arc::new(Mutex::new(WalManager::new(&wal_dir)?));
        let kv = Arc::new(BitStore::new(&kv_dir).map_err(std::io::Error::other)?);

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
            }
        }

        let max_bucket_capacity = max_bucket_capacity.unwrap_or(self.config.max_bucket_capacity);

        // --- JANITOR ---
        let janitor_config = JanitorConfig {
            index: index.clone(),
            manifest: manifest.clone(),
            staging,
            persistence,
            bucket_manager: bucket_manager.clone(),
            check_interval: Duration::from_millis(100),
            promotion_threshold_bytes: 16 * 1024 * 1024,
            coordinator,
            max_bucket_capacity,
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
