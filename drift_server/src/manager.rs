use crate::janitor::Janitor;
use crate::persistence::PersistenceManager;
use drift_core::index::{IndexOptions, VectorIndex};
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

    /// Retrieves a collection. If it doesn't exist, creates it using the provided `dim`.
    /// If `dim` is None for a new collection, defaults to 128.
    pub async fn get_or_create(
        &self,
        name: &str,
        dim_hint: Option<usize>,
    ) -> std::io::Result<Arc<Collection>> {
        // 1. Fast Path
        {
            let map = self.collections.read().await;
            if let Some(coll) = map.get(name) {
                // Optional: Verify dimension matches hint if provided
                if let Some(d) = dim_hint {
                    if coll.index.config.dim != d {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            format!(
                                "Collection '{}' exists with dim {}, but requested {}",
                                name, coll.index.config.dim, d
                            ),
                        ));
                    }
                }
                return Ok(coll.clone());
            }
        }

        // 2. Slow Path
        let mut map = self.collections.write().await;
        if let Some(coll) = map.get(name) {
            return Ok(coll.clone());
        }

        println!("Manager: Creating/Loading collection '{}'...", name);

        let coll_dir = self.base_path.join(name);
        std::fs::create_dir_all(&coll_dir)?;

        let wal_path = coll_dir.join("current.wal");
        let storage_path = coll_dir.join("storage");
        let storage = Arc::new(drift_cache::local_store::LocalDiskManager::new(
            storage_path,
        ));
        let persistence = PersistenceManager::new(&coll_dir);

        // Determine Dimension: Hint -> 128 (Default)
        let dim = dim_hint.unwrap_or(128);

        let options = IndexOptions {
            dim, // Use dynamic dimension
            num_centroids: 16,
            training_sample_size: 1000,
            max_bucket_capacity: 1000,
            ef_construction: 50,
            ef_search: 20,
        };

        let index = Arc::new(VectorIndex::new(options, &wal_path, storage)?);

        // Hydrate from disk
        persistence.hydrate_index(&index).await?;

        // Start Janitor
        let janitor_idx = index.clone();
        let janitor_persist = persistence.clone();
        tokio::spawn(async move {
            let janitor = Janitor::new(janitor_idx, janitor_persist, 2000, Duration::from_secs(2));
            janitor.run().await;
        });

        let collection = Arc::new(Collection {
            index,
            name: name.to_string(),
        });

        map.insert(name.to_string(), collection.clone());
        println!("Manager: Collection '{}' (dim: {}) is ready.", name, dim);

        Ok(collection)
    }

    pub async fn list_collections(&self) -> Vec<String> {
        let map = self.collections.read().await;
        map.keys().cloned().collect()
    }
}
