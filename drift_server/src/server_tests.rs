#[cfg(test)]
mod tests {
    use crate::drift_proto::{TrainRequest, Vector, drift_server::Drift};
    use crate::manager::CollectionManager;
    use crate::server::DriftService;
    use std::sync::Arc;
    use tempfile::tempdir;
    use tonic::Request;

    #[tokio::test]
    async fn test_server_train_flow() {
        // 1. Setup Service
        let dir = tempdir().unwrap();
        let manager = Arc::new(CollectionManager::new(dir.path()));
        let service = DriftService { manager };

        // 2. Prepare Data
        // The CollectionManager defaults to dim=128. We must match this.
        let dim = 128;
        let vectors = (0..50)
            .map(|i| Vector {
                id: i as u64,
                // Create a vector where all elements are 'i' (e.g., [0.0, 0.0, ...], [1.0, 1.0, ...])
                // These are distinct from each other, so they won't trigger the Singularity Guard.
                values: vec![i as f32; dim],
            })
            .collect();

        let req = Request::new(TrainRequest {
            collection_name: "test_collection".to_string(),
            vectors,
        });

        // 3. Call Train
        let resp = service.train(req).await;

        // 4. Verify
        if let Err(e) = &resp {
            println!("Train failed with status: {:?}", e);
        }
        assert!(resp.is_ok(), "Train request failed");
        assert!(
            resp.unwrap().into_inner().success,
            "Train response indicated failure"
        );

        // 5. Verify Side Effects
        // The collection directory should now exist
        assert!(dir.path().join("test_collection").exists());
        // The WAL should exist
        assert!(
            dir.path()
                .join("test_collection")
                .join("current.wal")
                .exists()
        );
    }
}
