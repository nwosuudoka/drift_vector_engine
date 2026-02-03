use crate::drift_proto::{
    InsertBatchRequest, InsertRequest, SearchRequest, SearchResponse, SearchResult,
    drift_server::Drift,
};
use crate::drift_proto::{InsertResponse, TrainRequest, TrainResponse};
use crate::manager::CollectionManager;
use std::sync::Arc;
use tonic::{Request, Response, Status};

pub struct DriftService {
    pub manager: Arc<CollectionManager>,
}

#[tonic::async_trait]
impl Drift for DriftService {
    async fn train(
        &self,
        request: Request<TrainRequest>,
    ) -> Result<Response<TrainResponse>, Status> {
        let req = request.into_inner();

        // Extract dimension from training data
        let dim_hint = req.vectors.first().map(|v| v.values.len());

        let collection = self
            .manager
            .get_or_create(&req.collection_name, dim_hint, None)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        let batch: Vec<(u64, Vec<f32>)> =
            req.vectors.into_iter().map(|v| (v.id, v.values)).collect();

        // Treat training data as just another batch of inserts.
        // The Janitor will see the volume and trigger training automatically.
        collection
            .index
            .insert_batch(&batch)
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(TrainResponse { success: true }))
    }

    async fn insert(
        &self,
        request: Request<InsertRequest>,
    ) -> Result<Response<InsertResponse>, Status> {
        let req = request.into_inner();
        let collection_name = req.collection_name;

        let dim_hint = req.vector.as_ref().map(|v| v.values.len());

        let collection = self
            .manager
            .get_or_create(&collection_name, dim_hint, None) // Add dim hint support in proto if needed
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        if let Some(vec) = req.vector {
            collection
                .index
                .insert(vec.id, &vec.values)
                .map_err(|e| Status::internal(e.to_string()))?;
        }

        Ok(Response::new(InsertResponse { success: true }))
    }

    async fn insert_batch(
        &self,
        request: Request<InsertBatchRequest>,
    ) -> Result<Response<InsertResponse>, Status> {
        let req = request.into_inner();

        let dim_hint = req.vectors.first().map(|v| v.values.len());

        let collection = self
            .manager
            .get_or_create(&req.collection_name, dim_hint, None)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        let batch: Vec<(u64, Vec<f32>)> =
            req.vectors.into_iter().map(|v| (v.id, v.values)).collect();

        collection
            .index
            .insert_batch(&batch)
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(InsertResponse { success: true }))
    }

    async fn search(
        &self,
        request: Request<SearchRequest>,
    ) -> Result<Response<SearchResponse>, Status> {
        let req = request.into_inner();
        let dim_hint = Some(req.vector.len());
        let collection = self
            .manager
            .get_or_create(&req.collection_name, dim_hint, None)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        let results = collection
            .index
            .search(
                &req.vector,
                req.k as usize,
                req.target_confidence,
                req.lambda,
                req.tau,
            )
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        let search_results = results
            .into_iter()
            .map(|(id, score)| SearchResult {
                id,
                score,
                // metadata: String::new(), // Metadata support pending
            })
            .collect();

        Ok(Response::new(SearchResponse {
            results: search_results,
        }))
    }
}
