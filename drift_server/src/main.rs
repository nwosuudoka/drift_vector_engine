use clap::Parser;
use drift_server::config::Config;
use drift_server::drift_proto::drift_server::DriftServer as GrpcServer;
use drift_server::manager::CollectionManager;
use drift_server::server::DriftService;
use std::sync::Arc;
use tonic::transport::Server;
use tracing::info;
use tracing_subscriber::{EnvFilter, prelude::*};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| "drift_server=info,warn".into());

    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(filter)
        .init();

    let config = Config::parse();

    // Banner
    println!("========================================");
    println!("   🚀 DRIFT VECTOR ENGINE v2.0.0 (LBR)");
    println!("========================================");
    println!("Config Loaded:");
    println!("  • Port:            {}", config.port);
    println!("  • WAL Path:        {:?}", config.wal_dir);
    println!("  • Data Path:       {:?}", config.data_dir);
    println!("  • Default Dim:     {}", config.default_dim);
    println!("  • Max Bucket Cap:  {}", config.max_bucket_capacity);
    println!("----------------------------------------");

    // 2. Initialize Manager (V2)
    // This spins up the RecoveryManager, JanitorV2, and Reaper automatically.
    let manager = Arc::new(CollectionManager::new(config.clone()));

    // 3. Start gRPC Service
    let addr = format!("0.0.0.0:{}", config.port).parse()?;
    let drift_service = DriftService { manager };

    info!("gRPC Listening on {}", addr);

    Server::builder()
        .add_service(GrpcServer::new(drift_service))
        .serve(addr)
        .await?;

    Ok(())
}
