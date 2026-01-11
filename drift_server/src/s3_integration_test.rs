#[cfg(test)]
#[cfg(feature = "integration-test")]
mod s3_tests {
    use crate::config::{Config, S3Config, StorageCommand};
    use crate::drift_proto::{
        InsertRequest, SearchRequest, TrainRequest, Vector, drift_server::Drift,
    };
    use crate::manager::CollectionManager;
    use crate::server::DriftService;

    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use aws_config::BehaviorVersion;
    use tempfile::tempdir;
    use tonic::Request;

    // ✅ testcontainers 0.26.3 imports
    use testcontainers::core::IntoContainerPort;
    use testcontainers::core::wait::WaitFor;
    use testcontainers::runners::AsyncRunner;
    use testcontainers::{GenericImage, ImageExt};

    // Helper to generate a random vector
    fn random_vector(dim: usize) -> Vector {
        let values: Vec<f32> = (0..dim).map(|_| rand::random::<f32>()).collect();
        Vector {
            id: rand::random(),
            values,
        }
    }

    #[tokio::test]
    async fn test_s3_minio_end_to_end() {
        // 1) Setup MinIO Container using testcontainers 0.26.3 API
        let minio_image = GenericImage::new("minio/minio", "latest")
            .with_exposed_port(9000.tcp())
            // Log-based wait can be flaky across MinIO versions; time-based is often safer.
            // If you *know* this log line exists for your MinIO tag, keep message_on_stdout instead.
            .with_wait_for(WaitFor::seconds(2))
            .with_env_var("MINIO_ROOT_USER", "minioadmin")
            .with_env_var("MINIO_ROOT_PASSWORD", "minioadmin")
            .with_cmd(["server", "/data"]);

        let container = minio_image
            .start()
            .await
            .expect("Failed to start MinIO container");

        // ✅ 0.26.x: pass the internal port number here (not 9000.tcp())
        let port = container
            .get_host_port_ipv4(9000)
            .await
            .expect("Failed to get mapped MinIO port");
        let endpoint = format!("http://127.0.0.1:{port}");

        println!("🚀 MinIO started at {endpoint}");

        // 2) Setup AWS SDK to create the bucket
        let aws_config = aws_config::defaults(BehaviorVersion::latest())
            .endpoint_url(&endpoint)
            .region(aws_config::Region::new("us-east-1"))
            .credentials_provider(aws_credential_types::Credentials::new(
                "minioadmin",
                "minioadmin",
                None,
                None,
                "test",
            ))
            .load()
            .await;

        let s3_client = aws_sdk_s3::Client::new(&aws_config);

        // MinIO can still be warming up; retry create_bucket briefly.
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            match s3_client.create_bucket().bucket("drift-test").send().await {
                Ok(_) => break,
                Err(e) if Instant::now() < deadline => {
                    eprintln!("create_bucket not ready yet ({e}); retrying...");
                    tokio::time::sleep(Duration::from_millis(500)).await;
                }
                Err(e) => panic!("Failed to create MinIO bucket: {e}"),
            }
        }

        // 3) Configure Drift with S3 Strategy
        let dir = tempdir().unwrap();

        let drift_config = Config {
            port: 50099,
            wal_dir: dir.path().join("wal"),
            storage: StorageCommand::S3(S3Config {
                bucket: "drift-test".to_string(),
                region: "us-east-1".to_string(),
                endpoint: Some(endpoint.clone()),
                access_key: Some("minioadmin".to_string()),
                secret_key: Some("minioadmin".to_string()),
            }),
            default_dim: 128,
            max_bucket_capacity: 100, // Small capacity to force flush
            ef_construction: 50,
            ef_search: 50,
        };

        let manager = Arc::new(CollectionManager::new(drift_config));
        let service = DriftService { manager };
        let collection = "s3_test_collection";

        // 4) Run Lifecycle

        // A) TRAIN
        let train_vecs: Vec<Vector> = (0..50).map(|_| random_vector(128)).collect();
        service
            .train(Request::new(TrainRequest {
                collection_name: collection.to_string(),
                vectors: train_vecs.clone(),
            }))
            .await
            .expect("Train failed");

        // B) INSERT (Exceed capacity to force flush)
        println!("📥 Inserting data to force flush...");
        for _ in 0..150 {
            service
                .insert(Request::new(InsertRequest {
                    collection_name: collection.to_string(),
                    vector: Some(random_vector(128)),
                }))
                .await
                .expect("Insert failed");
        }

        // Wait for Janitor to flush to S3
        println!("⏳ Waiting for S3 flush...");
        tokio::time::sleep(Duration::from_secs(5)).await;

        // 5) Verify Data in S3
        let objects = s3_client
            .list_objects_v2()
            .bucket("drift-test")
            .send()
            .await
            .unwrap();
        let contents = objects.contents();

        println!("📦 S3 Contents:");
        let mut found_segment = false;
        for obj in contents {
            let key = obj.key().unwrap_or("<none>");
            println!(" - {key}");
            if key.contains(collection) && key.ends_with(".drift") {
                found_segment = true;
            }
        }

        assert!(
            found_segment,
            "Failed to find flushed .drift segment in MinIO bucket"
        );

        // 6) Verify Read (Search)
        println!("🔍 Verifying Read...");
        let search_res = service
            .search(Request::new(SearchRequest {
                collection_name: collection.to_string(),
                vector: train_vecs[0].values.clone(),
                k: 5,
                target_confidence: 0.8,
                lambda: 1.0,
                tau: 100.0,
            }))
            .await;

        assert!(search_res.is_ok(), "Search failed");
        let results = search_res.unwrap().into_inner().results;
        assert!(
            !results.is_empty(),
            "Search returned no results from S3 data"
        );

        println!("✅ S3 Integration Test Passed!");
    }
}
