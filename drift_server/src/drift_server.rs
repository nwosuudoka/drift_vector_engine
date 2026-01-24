use crate::manager::CollectionManager;
use std::io;
use std::sync::Arc;
use tokio::sync::broadcast;
use tracing::info;

pub struct DriftServer {
    pub manager: Arc<CollectionManager>,
    shutdown_tx: broadcast::Sender<()>,
}

impl DriftServer {
    pub fn new(manager: Arc<CollectionManager>) -> Self {
        let (shutdown_tx, _) = broadcast::channel(1);
        Self {
            manager,
            shutdown_tx,
        }
    }

    pub async fn shutdown(self) -> io::Result<()> {
        info!("Server: Shutting down...");
        self.shutdown_tx.send(()).map_err(io::Error::other)?;
        // Manager holds the collections, dropping them drops the Janitor handles.
        // If we need explicit graceful shutdown for janitors, we can iterate collections here.
        Ok(())
    }
}
