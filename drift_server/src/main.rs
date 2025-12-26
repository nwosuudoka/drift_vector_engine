use clap::Parser;
use drift_server::config::Config;
use drift_server::drift_proto::drift_server::DriftServer;
use drift_server::manager::CollectionManager;
use drift_server::server::DriftService;
use std::sync::Arc;
use tonic::transport::Server;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::parse();

    println!("🚀 Starting Drift Server with Config: {:?}", config);
    let addr = format!("0.0.0.0:{}", config.port).parse()?;

    println!("Initializing Drift Manager...");
    // let data_dir = std::path::Path::new("./data");

    let path_str = if config.storage_uri.starts_with("file://") {
        config.storage_uri.strip_prefix("file://").unwrap()
    } else {
        "./data" // Fallback for now if S3 is passed but manager expects Path
    };

    let manager = Arc::new(CollectionManager::new(path_str));

    let service = DriftService { manager };

    println!("Drift Multi-Tenant Server listening on {}", addr);
    Server::builder()
        .add_service(DriftServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}
