use crate::config::Config;
use crate::janitor::Janitor;
use crate::persistence::PersistenceManager;
use drift_cache::LocalDiskManager;
use drift_cache::tiered_store::TieredPageManager;
use drift_core::index::{IndexOptions, VectorIndex};
use drift_storage::disk_manager::DriftPageManager;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::info; // Updated import

pub struct Collection {
    pub index: Arc<VectorIndex>,
    pub name: String,
}

pub struct CollectionManager {
    // base_path: PathBuf,
    config: Config,
    collections: RwLock<HashMap<String, Arc<Collection>>>,
}

impl CollectionManager {
    pub fn new(config: Config) -> Self {
        // let path = base_path.into();
        // std::fs::create_dir_all(&path).expect("Failed to create data root");
        std::fs::create_dir_all(&config.wal_dir).expect("Failed to create WAL root");

        Self {
            config,
            collections: RwLock::new(HashMap::new()),
        }
    }

    pub async fn get_or_create(
        &self,
        name: &str,
        dim_hint: Option<usize>,
    ) -> std::io::Result<Arc<Collection>> {
        // 1. Fast Path
        {
            let map = self.collections.read().await;
            if let Some(coll) = map.get(name) {
                if let Some(d) = dim_hint
                    && coll.index.config.dim != d
                {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        format!(
                            "Dimension mismatch: Existing={}, Requested={}",
                            coll.index.config.dim, d
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

        info!("Manager: Initializing collection '{}'", name);

        // A. Resolve Paths
        let coll_wal_dir = self.config.wal_dir.join(name);
        std::fs::create_dir_all(&coll_wal_dir)?;
        let wal_path = coll_wal_dir.join("current.wal");

        // B. Storage Setup (Tiered)
        let storage_uri = if self.config.storage_uri.ends_with('/') {
            format!("{}{}", self.config.storage_uri, name)
        } else {
            format!("{}/{}", self.config.storage_uri, name)
        };

        if storage_uri.starts_with("file://") {
            let path_str = storage_uri.strip_prefix("file://").unwrap();
            std::fs::create_dir_all(path_str)?;
        }

        // 1. Remote (Source of Truth)
        let remote_storage = Arc::new(DriftPageManager::new(&storage_uri).await?);

        // 2. Local (NVMe Cache)
        // Located at: ../cache/{collection_name} relative to WAL dir
        let cache_dir = self
            .config
            .wal_dir
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .join("cache")
            .join(name);
        std::fs::create_dir_all(&cache_dir)?;
        let local_storage = Arc::new(LocalDiskManager::new(cache_dir));

        // 3. Tiered (The Bridge)
        let storage = Arc::new(TieredPageManager::new(local_storage, remote_storage));

        let persistence = PersistenceManager::new(&coll_wal_dir);
        let dim = dim_hint.unwrap_or(self.config.default_dim);

        let options = IndexOptions {
            dim,
            num_centroids: 16,
            training_sample_size: 1000,
            max_bucket_capacity: self.config.max_bucket_capacity,
            ef_construction: self.config.ef_construction,
            ef_search: self.config.ef_search,
        };

        let index = Arc::new(VectorIndex::new(options, &wal_path, storage)?);

        // Hydrate
        persistence.hydrate_index(&index).await?;

        // Start Janitor
        let j_idx = index.clone();
        let j_persist = persistence.clone();
        let flush_threshold = self.config.max_bucket_capacity;
        let refresh_rate = 100;

        tokio::spawn(async move {
            let janitor = Janitor::new(
                j_idx,
                j_persist,
                flush_threshold,
                Duration::from_millis(refresh_rate),
            );
            janitor.run().await;
        });

        let collection = Arc::new(Collection {
            index,
            name: name.to_string(),
        });

        map.insert(name.to_string(), collection.clone());
        info!(
            "Manager: Collection '{}' ready (dim: {}, uri: {})",
            name, dim, storage_uri
        );
        Ok(collection)
    }

    pub async fn list_collections(&self) -> Vec<String> {
        let map = self.collections.read().await;
        map.keys().cloned().collect()
    }
}
