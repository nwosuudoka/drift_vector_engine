use clap::{Parser, Subcommand};
use drift_server::drift_proto::{
    InsertRequest, SearchRequest, TrainRequest, Vector, drift_client::DriftClient,
};
use serde_json::from_str;

#[derive(Parser)]
#[command(name = "drift")]
#[command(about = "CLI for Drift Vector Database", long_about = None)]
struct Cli {
    /// Server URL
    #[arg(short, long, default_value = "http://127.0.0.1:50051")]
    url: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a new collection with a batch of vectors
    Train {
        /// Collection name
        #[arg(short, long)]
        collection: String,

        /// Path to a JSON file containing [[f32; dim], ...] or list of vectors
        /// For simplicity in this v1, we accept a JSON string of a list of vectors: "[[0.1, 0.2], [0.3, 0.4]]"
        #[arg(long)]
        data: String,
    },
    /// Insert a single vector
    Insert {
        /// Collection name
        #[arg(short, long)]
        collection: String,

        /// Vector ID
        #[arg(long)]
        id: u64,

        /// Vector values as JSON string, e.g. "[0.1, 0.2, 0.3]"
        #[arg(short, long)]
        vector: String,
    },
    /// Search for nearest neighbors
    Search {
        /// Collection name
        #[arg(short, long)]
        collection: String,

        /// Query vector as JSON string, e.g. "[0.1, 0.2, 0.3]"
        #[arg(short, long)]
        vector: String,

        /// Number of results to return
        #[arg(short, long, default_value_t = 10)]
        k: u32,

        /// Drift: Target Confidence (0.0 - 1.0)
        #[arg(long, default_value_t = 0.9)]
        confidence: f32,

        /// Drift: Lambda (Decay rate)
        #[arg(long, default_value_t = 25.0)]
        lambda: f32,

        /// Drift: Tau (Critical Mass)
        #[arg(long, default_value_t = 100.0)]
        tau: f32,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // Connect to gRPC Server
    let mut client = DriftClient::connect(cli.url.clone())
        .await
        .map_err(|e| format!("Failed to connect to {}: {}", cli.url, e))?;

    match &cli.command {
        Commands::Train { collection, data } => {
            let raw_data: Vec<Vec<f32>> = from_str(data)
                .map_err(|e| format!("Invalid JSON data. Expected [[f32; N], ...]: {}", e))?;

            let vectors = raw_data
                .into_iter()
                .enumerate()
                .map(|(i, val)| Vector {
                    id: i as u64, // Auto-assign IDs for bulk train for now
                    values: val,
                })
                .collect();

            let req = tonic::Request::new(TrainRequest {
                collection_name: collection.clone(),
                vectors,
            });

            let response = client.train(req).await?;
            println!("Train Response: {:?}", response.into_inner());
        }
        Commands::Insert {
            collection,
            id,
            vector,
        } => {
            let values: Vec<f32> = from_str(vector)
                .map_err(|e| format!("Invalid vector JSON. Expected [f32; N]: {}", e))?;

            let req = tonic::Request::new(InsertRequest {
                collection_name: collection.clone(),
                vector: Some(Vector { id: *id, values }),
            });

            let response = client.insert(req).await?;
            println!("Insert Response: {:?}", response.into_inner());
        }
        Commands::Search {
            collection,
            vector,
            k,
            confidence,
            lambda,
            tau,
        } => {
            let values: Vec<f32> = from_str(vector)
                .map_err(|e| format!("Invalid vector JSON. Expected [f32; N]: {}", e))?;

            let req = tonic::Request::new(SearchRequest {
                collection_name: collection.clone(),
                vector: values,
                k: *k,
                target_confidence: *confidence,
                lambda: *lambda,
                tau: *tau,
            });

            let response = client.search(req).await?;
            let results = response.into_inner().results;

            println!("--- Search Results ({}) ---", results.len());
            for (i, res) in results.iter().enumerate() {
                println!("#{}: ID={}, Score={:.4}", i + 1, res.id, res.score);
            }
        }
    }

    Ok(())
}
