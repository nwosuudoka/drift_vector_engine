use drift_server::manager::CollectionManager;
use std::sync::Arc;
use tonic::transport::Server;

use drift_server::server::DriftService;
use drift_server::drift_proto::drift_server::DriftServer;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "127.0.0.1:50051".parse()?;

    println!("Initializing Drift Manager...");
    let data_dir = std::path::Path::new("./data");

    // Initialize Manager (Handles directories and lazy loading)
    let manager = Arc::new(CollectionManager::new(data_dir));

    let service = DriftService { manager };

    println!("Drift Multi-Tenant Server listening on {}", addr);
    Server::builder()
        .add_service(DriftServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}
