#[cfg(test)]
mod tests {
    use crate::drift_proto::{
        InsertRequest, SearchRequest, TrainRequest, Vector, drift_server::Drift,
    };
    use crate::manager::CollectionManager;
    use crate::server::DriftService;
    use std::sync::Arc;
    use tempfile::tempdir;
    use tonic::Request;

    #[tokio::test]
    async fn test_server_full_lifecycle() {
        let dir = tempdir().unwrap();
        let manager = Arc::new(CollectionManager::new(dir.path()));
        let service = DriftService { manager };
        let collection = "lifecycle_test";
        let dim = 128; // Must match Manager default

        // 1. TRAIN
        let train_vecs = (0..50)
            .map(|i| Vector {
                id: i as u64,
                values: vec![i as f32; dim], // Distinct vectors
            })
            .collect();

        let train_req = Request::new(TrainRequest {
            collection_name: collection.to_string(),
            vectors: train_vecs,
        });

        let train_resp = service.train(train_req).await;
        assert!(train_resp.is_ok(), "Train failed");
        assert!(train_resp.unwrap().into_inner().success);

        // 2. INSERT (L0)
        let insert_req = Request::new(InsertRequest {
            collection_name: collection.to_string(),
            vector: Some(Vector {
                id: 999,
                values: vec![0.0; dim], // Matches ID 0 from training
            }),
        });

        let insert_resp = service.insert(insert_req).await;
        assert!(insert_resp.is_ok());

        // 3. SEARCH
        // Should find the L0 item (999) and the L1 item (0) which are identical
        let search_req = Request::new(SearchRequest {
            collection_name: collection.to_string(),
            vector: vec![0.0; dim],
            k: 5,
            target_confidence: 0.9,
            lambda: 0.0,
            tau: 0.0,
        });

        let search_resp = service.search(search_req).await;
        assert!(search_resp.is_ok());

        let results = search_resp.unwrap().into_inner().results;
        assert!(!results.is_empty());

        let found_999 = results.iter().any(|r| r.id == 999);
        let found_0 = results.iter().any(|r| r.id == 0);

        assert!(found_999, "Failed to find L0 item");
        assert!(found_0, "Failed to find L1 item");
    }
}
