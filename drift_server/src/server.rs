use crate::drift_proto::{
    CreateCollectionRequest, CreateCollectionResponse, InsertBatchRequest, InsertRequest,
    MetricType, SearchRequest, SearchResponse, SearchResult, drift_server::Drift,
};
use crate::drift_proto::{
    HealthRequest, HealthResponse, InsertResponse, TrainRequest, TrainResponse,
};
use crate::manager::CollectionManager;
use drift_core::math::Metric;
use std::io;
use std::sync::Arc;
use tonic::{Request, Response, Status};

pub struct DriftService {
    pub manager: Arc<CollectionManager>,
}

fn map_collection_error(err: io::Error) -> Status {
    match err.kind() {
        io::ErrorKind::NotFound => Status::not_found(err.to_string()),
        io::ErrorKind::InvalidInput => Status::invalid_argument(err.to_string()),
        _ => Status::internal(err.to_string()),
    }
}

fn metric_from_proto(metric: i32) -> Result<Metric, Status> {
    let parsed = MetricType::try_from(metric)
        .map_err(|_| Status::invalid_argument(format!("Unknown metric enum value: {metric}")))?;

    match parsed {
        MetricType::L2 => Ok(Metric::L2),
        MetricType::Cosine => Ok(Metric::COSINE),
        MetricType::Unspecified => Err(Status::invalid_argument("Metric must be specified")),
    }
}

#[tonic::async_trait]
impl Drift for DriftService {
    async fn health(
        &self,
        _request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        Ok(Response::new(HealthResponse {
            ready: true,
            version: env!("CARGO_PKG_VERSION").to_string(),
        }))
    }

    async fn create_collection(
        &self,
        request: Request<CreateCollectionRequest>,
    ) -> Result<Response<CreateCollectionResponse>, Status> {
        let req = request.into_inner();
        if req.collection_name.trim().is_empty() {
            return Err(Status::invalid_argument("collection_name cannot be empty"));
        }
        if req.dim == 0 {
            return Err(Status::invalid_argument("dim must be > 0"));
        }

        let metric = metric_from_proto(req.metric)?;
        let max_bucket_capacity = if req.max_bucket_capacity == 0 {
            None
        } else {
            Some(req.max_bucket_capacity as usize)
        };

        self.manager
            .get_or_create(
                &req.collection_name,
                Some(req.dim as usize),
                max_bucket_capacity,
                Some(metric),
            )
            .await
            .map_err(map_collection_error)?;

        Ok(Response::new(CreateCollectionResponse { success: true }))
    }

    async fn train(
        &self,
        request: Request<TrainRequest>,
    ) -> Result<Response<TrainResponse>, Status> {
        let req = request.into_inner();

        // Extract dimension from training data
        let dim_hint = req.vectors.first().map(|v| v.values.len());

        let collection = self
            .manager
            .get_or_create(&req.collection_name, dim_hint, None, None)
            .await
            .map_err(map_collection_error)?;

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
            .get_or_create(&collection_name, dim_hint, None, None) // Add dim hint support in proto if needed
            .await
            .map_err(map_collection_error)?;

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
            .get_or_create(&req.collection_name, dim_hint, None, None)
            .await
            .map_err(map_collection_error)?;

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
            .get_or_create(&req.collection_name, dim_hint, None, None)
            .await
            .map_err(map_collection_error)?;

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
