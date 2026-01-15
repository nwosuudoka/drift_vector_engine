use crate::janitor_v2::{Janitor, JanitorConfig};
use crate::local_staging::LocalStagingManager;
use crate::manifest::ServerManifestManager;
use crate::persistence_v2::PersistenceManager;
use crate::recovery::RecoveryManager;
use drift_core::index_v2::VectorIndex;
use drift_core::lock_manager::BucketCoordinator;
use drift_core::wal_v2::WalManager;
use drift_storage::bucket_manager::BucketManager;
use opendal::{Operator, services};
use parking_lot::Mutex;
use std::io;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::broadcast;
use tracing::info;

pub struct DriftServer {
    pub index: Arc<VectorIndex>,
    pub manager: Arc<BucketManager>, // Exposed for metrics/tests
    janitor_task: tokio::task::JoinHandle<()>,
    shutdown_tx: broadcast::Sender<()>,
}

pub struct DriftServerOpts {
    pub dim: usize,
    pub capacity: usize,
    pub promotion_threshold: u64,
}

impl DriftServer {
    pub async fn new(
        data_dir: impl AsRef<Path>,
        remote_op: Operator,
        opts: DriftServerOpts,
    ) -> io::Result<Self> {
        let root = data_dir.as_ref().to_path_buf();
        let data_subdir = root.join("data");
        std::fs::create_dir_all(&data_subdir)?;

        info!("Server: Booting up in {:?}", root);

        // 1. Storage Components
        let manifest = Arc::new(ServerManifestManager::new(&root, opts.dim as u32)?);
        let staging = Arc::new(LocalStagingManager::new(&data_subdir)?);
        let persistence = PersistenceManager::new(remote_op.clone());

        // 2. Bucket Manager (Dual Operator)
        // Local OP points to ./data/
        let local_builder = services::Fs::default().root(data_subdir.to_str().unwrap());
        let local_op = Operator::new(local_builder)?.finish();

        let coordinator = Arc::new(BucketCoordinator::new());

        let bucket_manager = Arc::new(BucketManager::new(
            local_op,
            remote_op,
            16,
            coordinator.clone(),
        ));

        // 3. Recovery
        let recovery = RecoveryManager::new(&root, manifest.clone());
        let (router, wal_replay) = recovery.recover(&bucket_manager, opts.dim).await?;

        // 4. Index Initialization
        let wal_dir = root.join("wal");
        let wal = Arc::new(Mutex::new(WalManager::new(&wal_dir)?));

        // let capacity = 2000; // Tuning param

        let index = Arc::new(VectorIndex::new(
            opts.dim,
            opts.capacity,
            router,
            wal,
            bucket_manager.clone(), // Injected as DiskSearcher
        ));

        // 5. Replay WAL
        if !wal_replay.is_empty() {
            info!("Server: Replaying {} vectors...", wal_replay.len());
            index.insert_batch(&wal_replay)?;
        }

        // 6. Start Janitor V2
        let (shutdown_tx, _) = broadcast::channel(1);
        let janitor = Janitor::new(JanitorConfig {
            index: index.clone(),
            manifest,
            staging,
            persistence,
            bucket_manager: bucket_manager.clone(), // Needs this to update registry after promotion!
            check_interval: std::time::Duration::from_secs(5),
            promotion_threshold_bytes: opts.promotion_threshold,
            coordinator: coordinator.clone(),
        });

        let mut rx = shutdown_tx.subscribe();
        let janitor_task = tokio::spawn(async move {
            info!("Janitor: Started.");
            tokio::select! {
                _ = janitor.run() => { tracing::error!("Janitor died"); }
                _ = rx.recv() => { info!("Janitor: Shutdown"); }
            }
        });

        Ok(Self {
            index,
            manager: bucket_manager,
            janitor_task,
            shutdown_tx,
        })
    }

    pub async fn shutdown(self) -> io::Result<()> {
        self.shutdown_tx.send(()).map_err(io::Error::other)?;
        self.janitor_task.await?;
        Ok(())
    }
}
