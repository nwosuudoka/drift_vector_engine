use clap::Parser;
use drift_server::config::Config;
use drift_server::drift_proto::drift_server::DriftServer;
use drift_server::manager::CollectionManager;
use drift_server::server::DriftService;
use std::sync::Arc;
use tonic::transport::Server;
use tracing_subscriber::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (chrome_layer, _guard) = tracing_chrome::ChromeLayerBuilder::new().build();
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(chrome_layer)
        .init();

    let config = Config::parse();

    // Banner
    println!("========================================");
    println!("   🚀 DRIFT VECTOR ENGINE v0.5.4");
    println!("========================================");
    println!("Config Loaded:");
    println!("  • Port:            {}", config.port);
    // println!("  • Storage URI:     {}", config.storage_uri);
    println!("  • WAL Path:        {:?}", config.wal_dir);
    println!("  • Default Dim:     {}", config.default_dim);
    println!("  • Max Bucket Cap:  {}", config.max_bucket_capacity);
    println!("  • HNSW Ef_Const:   {}", config.ef_construction);
    println!("  • HNSW Ef_Search:  {}", config.ef_search);
    println!("----------------------------------------");

    let addr = format!("0.0.0.0:{}", config.port).parse()?;

    // 2. Initialize Manager (Inject Config)
    let manager = Arc::new(CollectionManager::new(config));

    let service = DriftService { manager };

    println!("✅ Server ready and listening on {}", addr);

    Server::builder()
        .add_service(DriftServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}
