use crate::config::Config;
use crate::janitor::Janitor;
use crate::persistence::PersistenceManager;
use drift_core::index::{IndexOptions, VectorIndex};
use drift_storage::disk_manager::DriftPageManager; // Updated import
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

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

        println!("Manager: Initializing collection '{}'", name);

        // A. Resolve WAL Path (Local)
        // WALs are always local for speed/durability guarantees
        let coll_wal_dir = self.config.wal_dir.join(name);
        std::fs::create_dir_all(&coll_wal_dir)?;
        let wal_path = coll_wal_dir.join("current.wal");

        // B. Resolve Storage URI (Cloud/Local)
        // If config is "s3://bucket/data", collection is "s3://bucket/data/{name}"
        // If config is "file:///data", collection is "file:///data/{name}"
        let storage_uri = if self.config.storage_uri.ends_with('/') {
            format!("{}{}", self.config.storage_uri, name)
        } else {
            format!("{}/{}", self.config.storage_uri, name)
        };

        // For file://, ensure the directory exists physically
        if storage_uri.starts_with("file://") {
            let path_str = storage_uri.strip_prefix("file://").unwrap();
            std::fs::create_dir_all(path_str)?;
        }

        let storage = Arc::new(DriftPageManager::new(&storage_uri).await?);

        // Persistence needs the base WAL dir for temp files/recovery logic
        let persistence = PersistenceManager::new(&coll_wal_dir);

        let dim = dim_hint.unwrap_or(self.config.default_dim);

        let options = IndexOptions {
            dim,
            num_centroids: 16,
            training_sample_size: 1000,
            max_bucket_capacity: self.config.max_bucket_capacity, // Use Config
            ef_construction: self.config.ef_construction,
            ef_search: self.config.ef_search,
        };

        let index = Arc::new(VectorIndex::new(options, &wal_path, storage)?);

        // Hydrate
        persistence.hydrate_index(&index).await?;

        // Start Janitor
        let j_idx = index.clone();
        let j_persist = persistence.clone();
        tokio::spawn(async move {
            let janitor = Janitor::new(j_idx, j_persist, 2000, Duration::from_secs(2));
            janitor.run().await;
        });

        let collection = Arc::new(Collection {
            index,
            name: name.to_string(),
        });

        map.insert(name.to_string(), collection.clone());
        println!(
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
