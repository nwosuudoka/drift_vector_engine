use crate::janitor::Janitor;
use crate::persistence::PersistenceManager;
use drift_core::index::{IndexOptions, VectorIndex};
use drift_storage::disk_manager::DriftPageManager; // Updated import
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

pub struct Collection {
    pub index: Arc<VectorIndex>,
    pub name: String,
}

pub struct CollectionManager {
    base_path: PathBuf,
    collections: RwLock<HashMap<String, Arc<Collection>>>,
}

impl CollectionManager {
    pub fn new(base_path: impl Into<PathBuf>) -> Self {
        let path = base_path.into();
        std::fs::create_dir_all(&path).expect("Failed to create data root");

        Self {
            base_path: path,
            collections: RwLock::new(HashMap::new()),
        }
    }

    /// Retrieves or creates a collection.
    /// `dim_hint` is required if creating a NEW collection.
    pub async fn get_or_create(
        &self,
        name: &str,
        dim_hint: Option<usize>,
    ) -> std::io::Result<Arc<Collection>> {
        // 1. Fast Path (Read Lock)
        {
            let map = self.collections.read().await;
            if let Some(coll) = map.get(name) {
                // Safety Check: If hint provided, ensure it matches
                if let Some(d) = dim_hint {
                    if coll.index.config.dim != d {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            format!(
                                "Dimension Mismatch: Collection has {}, requested {}",
                                coll.index.config.dim, d
                            ),
                        ));
                    }
                }
                return Ok(coll.clone());
            }
        }

        // 2. Slow Path (Write Lock)
        let mut map = self.collections.write().await;
        if let Some(coll) = map.get(name) {
            return Ok(coll.clone());
        }

        println!("Manager: Loading/Creating collection '{}'...", name);

        // Setup Paths
        let coll_dir = self.base_path.join(name);
        std::fs::create_dir_all(&coll_dir)?;

        let wal_path = coll_dir.join("current.wal");

        let storage_dir = coll_dir.join("storage");
        std::fs::create_dir_all(&storage_dir)?;

        let abs_storage_path = std::fs::canonicalize(&storage_dir).unwrap_or(storage_dir);
        let storage_uri = format!("file://{}", abs_storage_path.to_string_lossy());

        // Open DiskManager via OpenDAL
        let storage = Arc::new(DriftPageManager::new(&storage_uri).await?);
        let persistence = PersistenceManager::new(&coll_dir);

        // Dynamic Dimension or Default
        let dim = dim_hint.unwrap_or(128);

        let options = IndexOptions {
            dim,
            num_centroids: 16, // In prod, this might scale with data size
            training_sample_size: 1000,
            max_bucket_capacity: 1000,
            ef_construction: 50,
            ef_search: 20,
        };

        // Initialize Index
        let index = Arc::new(VectorIndex::new(options, &wal_path, storage)?);

        // Hydrate from Disk (Loads existing segments)
        // Note: If we are recovering, we might want to check the loaded segments to confirm 'dim'
        // matches, but VectorIndex handles safety internally usually.
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
        println!("Manager: Collection '{}' (dim: {}) ready.", name, dim);

        Ok(collection)
    }

    pub async fn list_collections(&self) -> Vec<String> {
        let map = self.collections.read().await;
        map.keys().cloned().collect()
    }
}
