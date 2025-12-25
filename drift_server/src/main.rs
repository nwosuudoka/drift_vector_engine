pub mod janitor;
pub mod manager;
pub mod persistence;

mod janitor_tests;
mod manager_tests;
mod persistence_tests;
mod wal_integration_test;

use std::sync::Arc;

use drift_server::manager::CollectionManager;
use tonic::{Request, Response, Status, transport::Server};

pub mod drift_proto {
    tonic::include_proto!("drift");
}

use drift_proto::drift_server::{Drift, DriftServer};
use drift_proto::{InsertRequest, InsertResponse, SearchRequest, SearchResponse, SearchResult};

pub struct DriftService {
    manager: Arc<CollectionManager>,
}

const TARGET_CONFIDENCE: f32 = 0.9;
const LAMBDA: f32 = 1.0;
const TAU: f32 = 100.0;

#[tonic::async_trait]
impl Drift for DriftService {
    async fn insert(
        &self,
        request: Request<InsertRequest>,
    ) -> Result<Response<InsertResponse>, Status> {
        let req = request.into_inner();
        let vec_data = req
            .vector
            .ok_or_else(|| Status::invalid_argument("Vector missing"))?;

        // 2. Resolve Collection
        let collection_name = if req.collection_name.is_empty() {
            "default".to_string()
        } else {
            req.collection_name
        };

        let collection = self
            .manager
            .get_or_create(&collection_name)
            .await
            .map_err(|e| Status::internal(format!("Failed to load collection: {}", e)))?;

        // 3. Insert into specific index
        match collection.index.insert(vec_data.id, &vec_data.values) {
            Ok(_) => Ok(Response::new(InsertResponse { success: true })),
            Err(e) => {
                eprintln!("Insert error: {}", e);
                Err(Status::internal("Failed to insert vector"))
            }
        }
    }

    // async fn search(
    //     &self,
    //     request: Request<SearchRequest>,
    // ) -> Result<Response<SearchResponse>, Status> {
    //     println!("Got a request: {:?}", request);

    //     let req = request.into_inner();

    //     let collection_name = if req.collection_name.is_empty() {
    //         "default".to_string()
    //     } else {
    //         req.collection_name
    //     };

    //     // Resolve Collection
    //     let collection = self
    //         .manager
    //         .get_or_create(&collection_name)
    //         .await
    //         .map_err(|e| Status::internal(format!("Failed to load collection: {}", e)))?;

    //     let k = if req.k == 0 { 10 } else { req.k as usize };
    //     let results = collection
    //         .index
    //         .search_async(&req.vector, k, TARGET_CONFIDENCE, LAMBDA, TAU)
    //         .await
    //         .map_err(|e| Status::internal(format!("Search failed: {}", e)))?;

    //     let proto_results = results
    //         .into_iter()
    //         .map(|r| SearchResult {
    //             id: r.id,
    //             score: r.distance,
    //         })
    //         .collect();

    //     Ok(Response::new(SearchResponse {
    //         results: proto_results,
    //     }))
    // }

    async fn search(
        &self,
        request: Request<SearchRequest>,
    ) -> Result<Response<SearchResponse>, Status> {
        let req = request.into_inner();

        let collection_name = if req.collection_name.is_empty() {
            "default".to_string()
        } else {
            req.collection_name
        };

        let collection = self
            .manager
            .get_or_create(&collection_name)
            .await
            .map_err(|e| Status::internal(format!("Failed to load collection: {}", e)))?;

        let k = if req.k == 0 { 10 } else { req.k as usize };

        // Parse Parameters with Defaults
        let target_confidence = if req.target_confidence == 0.0 {
            0.90
        } else {
            req.target_confidence
        };
        let lambda = if req.lambda == 0.0 { 25.0 } else { req.lambda };
        let tau = if req.tau == 0.0 { 100.0 } else { req.tau };

        // Call the new Async Search
        let results = collection
            .index
            .search_async(&req.vector, k, target_confidence, lambda, tau)
            .await
            .map_err(|e| Status::internal(format!("Search failed: {}", e)))?;

        let proto_results = results
            .into_iter()
            .map(|r| SearchResult {
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
