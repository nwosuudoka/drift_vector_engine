use clap::Parser;
use drift_server::config::Config;
use drift_server::drift_proto::drift_server::DriftServer as GrpcServer;
use drift_server::drift_server::DriftServer; // The App Struct
use drift_server::manager::CollectionManager;
use drift_server::server::DriftService; // The gRPC impl
use std::sync::Arc;
use tonic::transport::Server;
use tracing::info;
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
    println!("   🚀 DRIFT VECTOR ENGINE v2.0.0");
    println!("========================================");
    println!("Config Loaded:");
    println!("  • Port:            {}", config.port);
    println!("  • WAL Path:        {:?}", config.wal_dir);
    println!("  • Data Path:        {:?}", config.data_dir);
    println!("  • Default Dim:     {}", config.default_dim);
    println!("  • Max Bucket Cap:  {}", config.max_bucket_capacity); // TODO: change this
    println!("----------------------------------------");

    // 2. Initialize Manager
    let manager = Arc::new(CollectionManager::new(config.clone()));

    // 3. Initialize App
    let _app = DriftServer::new(manager.clone());

    // 4. Start gRPC Service
    let addr = format!("0.0.0.0:{}", config.port).parse()?;
    let drift_service = DriftService { manager };

    info!("gRPC Listening on {}", addr);

    Server::builder()
        .add_service(GrpcServer::new(drift_service))
        .serve(addr)
        .await?;

    Ok(())
}
