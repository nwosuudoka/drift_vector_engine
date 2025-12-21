pub mod janitor;
pub mod persistence;

pub(crate) mod janitor_tests;
pub(crate) mod persistence_test;
pub(crate) mod wal_integration_test;

use std::sync::Arc;
use std::time::Duration;

use drift_core::index::{IndexOptions, VectorIndex};
use tonic::{Request, Response, Status, transport::Server};

pub mod drift_proto {
    tonic::include_proto!("drift");
}

use drift_proto::drift_server::{Drift, DriftServer};
use drift_proto::{InsertRequest, InsertResponse, SearchRequest, SearchResponse};

use crate::janitor::Janitor;
use crate::persistence::PersistenceManager;

pub struct DriftService {
    index: Arc<VectorIndex>,
}

#[tonic::async_trait]
impl Drift for DriftService {
    async fn insert(
        &self,
        request: Request<InsertRequest>,
    ) -> Result<Response<InsertResponse>, Status> {
        let req = request.into_inner();
        let vec_data = req
            .vector
            .ok_or_else(|| Status::invalid_argument("vector is missing"))?;

        // Perform some validation here
        if vec_data.values.len() != self.index.config.dim {
            return Err(Status::invalid_argument(format!(
                "vector dimension mismatch. expected {} got {}",
                self.index.config.dim,
                vec_data.values.len(),
            )));
        }

        // Execute insert
        // This writes to the (WAL) durable and MemTable (Searchable)
        match self.index.insert(vec_data.id, &vec_data.values) {
            Ok(_) => Ok(Response::new(InsertResponse { success: true })),
            Err(e) => {
                eprintln!("Error inserting vector: {:?}", e);
                Err(Status::internal("failed to insert vector"))
            }
        }
    }

    async fn search(
        &self,
        request: Request<SearchRequest>,
    ) -> Result<Response<SearchResponse>, Status> {
        println!("Got a request: {:?}", request);

        let req = request.into_inner();
        let k = if req.k == 0 { 10 } else { req.k as usize };

        const TARGET_CONFIDENCE: f32 = 0.9;
        const LAMBDA: f32 = 1.0;
        const TAU: f32 = 100.0;

        let result = self
            .index
            .search_drift_aware(&req.vector, k, TARGET_CONFIDENCE, LAMBDA, TAU);

        let proto_results = result
            .into_iter()
            .map(|r| drift_proto::SearchResult {
                id: r.id,
                score: r.distance,
            })
            .collect();

        Ok(Response::new(SearchResponse {
            results: proto_results,
        }))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // let addr = "[::1]:50051".parse()?;
    let addr = "127.0.0.1:50051".parse()?;

    println!("init drift engine");

    let data_dir = std::path::Path::new("./data");
    std::fs::create_dir_all(data_dir)?;

    let wal_path = data_dir.join("current.wal");
    let persistence = PersistenceManager::new(data_dir);

    // Standard Config
    let options = IndexOptions {
        dim: 128, // Example: OpenAI embedding size (often 1536, using 128 for testing)
        num_centroids: 16,
        training_sample_size: 1000,
        max_bucket_capacity: 1000,
        ef_construction: 50,
        ef_search: 20,
    };

    // Initialize Index (This will replay WAL if it exists)
    // TODO: Need robust loading logic (Load Segments + Replay WAL).
    // For now, we create fresh/recover-WAL style.
    let index = Arc::new(VectorIndex::new(options, &wal_path)?);

    // --- B. Start Background Janitor ---
    let janitor = Janitor::new(
        index.clone(),
        persistence,
        2000, // Flush when MemTable hits 2000 items
        Duration::from_secs(2),
    );
    tokio::spawn(async move {
        janitor.run().await;
    });

    println!("Drift Vector DB listening on {}", addr);

    let service = DriftService { index };

    Server::builder()
        .add_service(DriftServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}
