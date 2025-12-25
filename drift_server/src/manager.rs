use crate::janitor::Janitor;
use crate::persistence::PersistenceManager;
use drift_core::index::{IndexOptions, VectorIndex};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

/// A wrapper around a running Index + its background tasks
pub struct Collection {
    pub index: Arc<VectorIndex>,
    pub name: String,
}

pub struct CollectionManager {
    base_path: PathBuf,
    // Thread-safe map of loaded collections
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

    /// Retrieves a collection by name, creating/loading it if it doesn't exist.
    pub async fn get_or_create(&self, name: &str) -> std::io::Result<Arc<Collection>> {
        // 1. Fast Path: Read Lock
        {
            let map = self.collections.read().await;
            if let Some(coll) = map.get(name) {
                return Ok(coll.clone());
            }
        }

        // 2. Slow Path: Write Lock (Double-checked locking pattern)
        let mut map = self.collections.write().await;

        // Re-check in case another thread created it while we waited for the write lock
        if let Some(coll) = map.get(name) {
            return Ok(coll.clone());
        }

        println!("Manager: Loading collection '{}'...", name);

        // 3. Setup Paths
        // Layout: ./data/<name>/
        let coll_dir = self.base_path.join(name);
        std::fs::create_dir_all(&coll_dir)?;

        let wal_path = coll_dir.join("current.wal");

        let storage_path = coll_dir.join("storage");
        let storage = Arc::new(drift_cache::local_store::LocalDiskManager::new(
            storage_path,
        ));

        // 4. Initialize Components
        let persistence = PersistenceManager::new(&coll_dir);

        let options = IndexOptions {
            dim: 128,
            num_centroids: 16,
            training_sample_size: 1000,
            max_bucket_capacity: 1000,
            ef_construction: 50,
            ef_search: 20,
        };

        // 5. Init Index (Replays WAL automatically)
        let index = Arc::new(VectorIndex::new(options, &wal_path, storage)?);

        // 6. HYDRATION (The Fix)
        // We use the persistence instance to load historical segments from disk.
        // This makes the variable 'persistence' useful and necessary.
        persistence.hydrate_index(&index).await?;

        // 7. Spawn Dedicated Janitor
        // We clone the persistence manager (cheap PathBuf copy) for the background thread.
        let janitor_idx = index.clone();
        let janitor_persist = persistence.clone();

        tokio::spawn(async move {
            let janitor = Janitor::new(janitor_idx, janitor_persist, 2000, Duration::from_secs(2));
            janitor.run().await;
        });

        // 8. Store and Return
        let collection = Arc::new(Collection {
            index,
            name: name.to_string(),
        });

        map.insert(name.to_string(), collection.clone());
        println!("Manager: Collection '{}' is ready.", name);

        Ok(collection)
    }

    pub async fn list_collections(&self) -> Vec<String> {
        let map = self.collections.read().await;
        map.keys().cloned().collect()
    }
}
