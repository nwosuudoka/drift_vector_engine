use crate::drift_proto::drift_server::Drift;
use crate::{
    drift_proto::{
        InsertBatchRequest, InsertRequest, InsertResponse, SearchRequest, SearchResponse,
        SearchResult, TrainRequest, TrainResponse,
    },
    manager::CollectionManager,
};
use std::sync::Arc;
use std::time::Instant; // NEW: For metrics
use tonic::{Request, Response, Status};
use tracing::{error, info, instrument};

#[derive(Clone)]
pub struct DriftService {
    pub manager: Arc<CollectionManager>,
}

#[tonic::async_trait]
impl Drift for DriftService {
    async fn insert(
        &self,
        request: Request<InsertRequest>,
    ) -> Result<Response<InsertResponse>, Status> {
        let start = Instant::now(); // Metric Start
        let req = request.into_inner();
        let count = 1; // Single vector insert

        let vec_data = req
            .vector
            .ok_or_else(|| Status::invalid_argument("Vector missing"))?;
        let collection_name = if req.collection_name.is_empty() {
            "default".to_string()
        } else {
            req.collection_name
        };
        let dim = vec_data.values.len();

        let collection = self
            .manager
            .get_or_create(&collection_name, Some(dim))
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        match collection.index.insert(vec_data.id, &vec_data.values) {
            Ok(_) => {
                // Metric Log
                info!(
                    "[METRIC] Insert | Coll: {} | Dim: {} | Latency: {:.2?} | Count: {}",
                    collection_name,
                    dim,
                    start.elapsed(),
                    count
                );
                Ok(Response::new(InsertResponse { success: true }))
            }
            Err(e) => {
                error!("[ERROR] Insert failed: {}", e);
                Err(Status::internal(format!("Failed to insert: {}", e)))
            }
        }
    }

    #[instrument(skip(self, request), fields(batch_size = request.get_ref().vectors.len()))]
    async fn insert_batch(
        &self,
        request: Request<InsertBatchRequest>,
    ) -> Result<Response<InsertResponse>, Status> {
        let start = Instant::now();
        let req = request.into_inner();
        let count = req.vectors.len();

        let collection_name = if req.collection_name.is_empty() {
            "default".to_string()
        } else {
            req.collection_name
        };

        if count == 0 {
            return Ok(Response::new(InsertResponse { success: true }));
        }

        let dim = req.vectors[0].values.len();

        let collection = self
            .manager
            .get_or_create(&collection_name, Some(dim))
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        // Map Protobuf Vector -> Tuple
        let batch: Vec<(u64, Vec<f32>)> =
            req.vectors.into_iter().map(|v| (v.id, v.values)).collect();

        match collection.index.insert_batch(&batch) {
            Ok(_) => {
                info!(
                    "[METRIC] InsertBatch | Coll: {} | Count: {} | Latency: {:.2?}",
                    collection_name,
                    count,
                    start.elapsed()
                );
                Ok(Response::new(InsertResponse { success: true }))
            }
            Err(e) => Err(Status::internal(format!("Batch insert failed: {}", e))),
        }
    }

    async fn search(
        &self,
        request: Request<SearchRequest>,
    ) -> Result<Response<SearchResponse>, Status> {
        let start = Instant::now(); // Metric Start
        let req = request.into_inner();
        let collection_name = if req.collection_name.is_empty() {
            "default".to_string()
        } else {
            req.collection_name
        };
        let dim = req.vector.len();

        let collection = self
            .manager
            .get_or_create(&collection_name, Some(dim))
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        let k = if req.k == 0 { 10 } else { req.k as usize };
        // Default params if 0.0 provided
        let target_confidence = if req.target_confidence == 0.0 {
            0.90
        } else {
            req.target_confidence
        };
        let lambda = if req.lambda == 0.0 { 1.0 } else { req.lambda }; // Default updated to 1.0
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
            .collect::<Vec<_>>();

        info!(
            "[METRIC] Search | Coll: {} | K: {} | Hits: {} | Latency: {:.2?}",
            collection_name,
            k,
            proto_results.len(),
            start.elapsed()
        );

        Ok(Response::new(SearchResponse {
            results: proto_results,
        }))
    }

    async fn train(
        &self,
        request: Request<TrainRequest>,
    ) -> Result<Response<TrainResponse>, Status> {
        let start = Instant::now();
        let req = request.into_inner();
        let collection_name = if req.collection_name.is_empty() {
            "default".to_string()
        } else {
            req.collection_name
        };

        if req.vectors.is_empty() {
            return Err(Status::invalid_argument("Empty training set"));
        }
        let dim = req.vectors[0].values.len();
        let count = req.vectors.len();

        let collection = self
            .manager
            .get_or_create(&collection_name, Some(dim))
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        let training_data: Vec<Vec<f32>> = req.vectors.into_iter().map(|v| v.values).collect();

        match collection.index.train(&training_data).await {
            Ok(_) => {
                info!(
                    "[METRIC] Train | Coll: {} | Samples: {} | Latency: {:.2?}",
                    collection_name,
                    count,
                    start.elapsed()
                );
                Ok(Response::new(TrainResponse { success: true }))
            }
            Err(e) => Err(Status::internal(format!("Training failed: {}", e))),
        }
    }
}
