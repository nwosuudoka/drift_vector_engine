use std::sync::Arc;
use tonic::{Request, Response, Status};

use crate::{
    drift_proto::{
        InsertRequest, InsertResponse, SearchRequest, SearchResponse, SearchResult, TrainRequest,
        TrainResponse, drift_server::Drift,
    },
    manager::CollectionManager,
};

#[derive(Clone)]
pub struct DriftService {
    pub manager: Arc<CollectionManager>,
}

// const TARGET_CONFIDENCE: f32 = 0.9;
// const LAMBDA: f32 = 1.0;
// const TAU: f32 = 100.0;

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
        let collection_name = if req.collection_name.is_empty() {
            "default".to_string()
        } else {
            req.collection_name
        };

        // Pass dimension hint from the vector itself
        let dim = vec_data.values.len();

        let collection = self
            .manager
            .get_or_create(&collection_name, Some(dim))
            .await
            .map_err(|e| Status::internal(format!("Failed to load collection: {}", e)))?;

        match collection.index.insert(vec_data.id, &vec_data.values) {
            Ok(_) => Ok(Response::new(InsertResponse { success: true })),
            Err(e) => {
                eprintln!("Insert error: {}", e);
                Err(Status::internal(format!("Failed to insert: {}", e)))
            }
        }
    }

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

        // Pass dimension hint
        let dim = req.vector.len();

        let collection = self
            .manager
            .get_or_create(&collection_name, Some(dim))
            .await
            .map_err(|e| Status::internal(format!("Failed to load collection: {}", e)))?;

        // ... rest of search logic ...
        let k = if req.k == 0 { 10 } else { req.k as usize };
        let target_confidence = if req.target_confidence == 0.0 {
            0.90
        } else {
            req.target_confidence
        };
        let lambda = if req.lambda == 0.0 { 25.0 } else { req.lambda };
        let tau = if req.tau == 0.0 { 100.0 } else { req.tau };

        let results = collection
            .index
            .search_async(&req.vector, k, target_confidence, lambda, tau)
            .await
            .map_err(|e| Status::internal(format!("Search failed: {}", e)))?;

        let proto_results = results
            .into_iter()
            .map(|r| SearchResult {
                id: r.id,
                score: r.distance.sqrt(),
            })
            .collect();

        Ok(Response::new(SearchResponse {
            results: proto_results,
        }))
    }

    async fn train(
        &self,
        request: Request<TrainRequest>,
    ) -> Result<Response<TrainResponse>, Status> {
        let req = request.into_inner();
        let collection_name = if req.collection_name.is_empty() {
            "default".to_string()
        } else {
            req.collection_name
        };

        if req.vectors.is_empty() {
            return Err(Status::invalid_argument("Training dataset cannot be empty"));
        }

        // Get dimension from first vector
        let dim = req.vectors[0].values.len();

        // Pass hint
        let collection = self
            .manager
            .get_or_create(&collection_name, Some(dim))
            .await
            .map_err(|e| Status::internal(format!("Failed to load collection: {}", e)))?;

        let training_data: Vec<Vec<f32>> = req.vectors.into_iter().map(|v| v.values).collect();

        match collection.index.train(&training_data).await {
            Ok(_) => Ok(Response::new(TrainResponse { success: true })),
            Err(e) => Err(Status::internal(format!("Training failed: {}", e))),
        }
    }
}
