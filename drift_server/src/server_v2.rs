use crate::drift_proto::{
    InsertBatchRequest, InsertRequest, SearchRequest, SearchResponse, SearchResult,
    drift_server::Drift,
};
use crate::drift_proto::{InsertResponse, TrainRequest, TrainResponse};
use crate::manager_v2::CollectionManager;
use std::sync::Arc;
use tonic::{Request, Response, Status};

pub struct DriftService {
    pub manager: Arc<CollectionManager>,
}

#[tonic::async_trait]
impl Drift for DriftService {
    async fn train(
        &self,
        _request: Request<TrainRequest>,
    ) -> Result<Response<TrainResponse>, Status> {
        todo!()
    }

    async fn insert(
        &self,
        request: Request<InsertRequest>,
    ) -> Result<Response<InsertResponse>, Status> {
        let req = request.into_inner();
        let collection_name = req.collection_name;

        let collection = self
            .manager
            .get_or_create(&collection_name, None, None) // Add dim hint support in proto if needed
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
        let collection = self
            .manager
            .get_or_create(&req.collection_name, None, None)
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
        let collection = self
            .manager
            .get_or_create(&req.collection_name, None, None)
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
