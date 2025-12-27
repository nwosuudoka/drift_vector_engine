#[cfg(test)]
#[cfg(feature = "stress-test")]
mod tests {
    use crate::config::Config;
    use crate::drift_proto::{
        InsertBatchRequest, InsertRequest, SearchRequest, Vector, drift_server::Drift,
    };
    use crate::manager::CollectionManager;
    use crate::server::DriftService;
    use rand::Rng;
    use std::sync::Arc;
    use std::time::{Duration, Instant};
    use tempfile::tempdir;
    use tonic::Request;

    const DIM: usize = 128;
    const TOTAL_VECTORS: usize = 100_000;
    const BATCH_SIZE: usize = 1_000;
    const QUERY_COUNT: usize = 100;

    fn generate_random_vector(rng: &mut impl Rng) -> Vec<f32> {
        (0..DIM).map(|_| rng.random::<f32>()).collect()
    }

    #[tokio::test]
    // #[ignore]
    async fn test_server_heavy_load_performance() {
        let dir = tempdir().unwrap();

        let config = Config {
            port: 50051,
            storage_uri: format!("file://{}", dir.path().join("storage").to_string_lossy()),
            wal_dir: dir.path().join("wal"),
            default_dim: DIM,
            max_bucket_capacity: 2000,
            // Tuning for 100k Vectors:
            ef_construction: 200,
            ef_search: 200,
        };

        let manager = Arc::new(CollectionManager::new(config));
        let service = DriftService {
            manager: manager.clone(),
        };
        let collection = "stress_test";

        println!("🚀 Starting Load Test: {} Vectors", TOTAL_VECTORS);

        // 2. Ingestion Phase
        let start_ingest = Instant::now();
        let mut rng = rand::rng();

        for batch_idx in 0..(TOTAL_VECTORS / BATCH_SIZE) {
            let start_id = (batch_idx * BATCH_SIZE) as u64;

            let mut batch_vecs = Vec::with_capacity(BATCH_SIZE);
            for i in 0..BATCH_SIZE {
                let id = start_id + i as u64;
                let vector = generate_random_vector(&mut rng);
                batch_vecs.push(Vector { id, values: vector });
            }

            let req = Request::new(InsertBatchRequest {
                collection_name: collection.to_string(),
                vectors: batch_vecs,
            });

            service
                .insert_batch(req)
                .await
                .expect("Batch Insert failed");

            if batch_idx % 10 == 0 {
                println!("   Inserted {} vectors...", start_id + BATCH_SIZE as u64);
            }
        }

        let ingest_duration = start_ingest.elapsed();
        println!("✅ Ingestion Complete in {:.2?}", ingest_duration);
        println!(
            "   Throughput: {:.0} vec/sec",
            TOTAL_VECTORS as f64 / ingest_duration.as_secs_f64()
        );

        // 3. Inject "Needle" for verification
        let needle_id = 999_999;
        let needle_vec = vec![0.5; DIM];
        println!("💉 Injecting Needle (ID {})...", needle_id);
        service
            .insert(Request::new(InsertRequest {
                collection_name: collection.to_string(),
                vector: Some(Vector {
                    id: needle_id,
                    values: needle_vec.clone(),
                }),
            }))
            .await
            .unwrap();
        println!("✅ Needle Injected.");

        // 4. Wait for Janitor / Indexing
        println!("⏳ Waiting for background flush/indexing...");
        tokio::time::sleep(Duration::from_secs(4)).await; // Increased wait

        // 5. Query Phase
        println!("🔍 Starting Query Benchmark...");
        let mut latencies = Vec::with_capacity(QUERY_COUNT);

        for _ in 0..QUERY_COUNT {
            let query_vec = generate_random_vector(&mut rng);
            let start_q = Instant::now();
            let req = Request::new(SearchRequest {
                collection_name: collection.to_string(),
                vector: query_vec,
                k: 10,
                target_confidence: 0.95,
                lambda: 1.0,
                tau: 100.0,
            });
            let _ = service.search(req).await.expect("Search failed");
            latencies.push(start_q.elapsed());
        }

        // 6. Verification Query
        println!("🔎 Verifying Needle...");
        let verify_req = Request::new(SearchRequest {
            collection_name: collection.to_string(),
            vector: needle_vec,
            k: 5,
            target_confidence: 0.99,
            lambda: 1.0,
            tau: 100.0,
        });

        let verify_res = service.search(verify_req).await.unwrap().into_inner();
        let found = verify_res.results.iter().any(|r| r.id == needle_id);

        // 7. Report
        latencies.sort();
        let p50 = latencies[latencies.len() / 2];
        let p99 = latencies[(latencies.len() as f64 * 0.99) as usize];

        println!("========================================");
        println!("📊 LOAD TEST RESULTS");
        println!("Total Vectors: {}", TOTAL_VECTORS);
        println!("P50 Latency:   {:.2?}", p50);
        println!("P99 Latency:   {:.2?}", p99);
        println!(
            "Correctness:   {}",
            if found { "PASS ✅" } else { "FAIL ❌" }
        );
        println!("========================================");

        assert!(found, "Failed to find the needle vector!");
    }
}

#[cfg(test)]
#[cfg(feature = "stress-test")]
mod pretrained_tests {
    use crate::config::Config;
    use crate::drift_proto::{
        InsertBatchRequest, InsertRequest, SearchRequest, Vector, drift_server::Drift,
    };
    use crate::manager::CollectionManager;
    use crate::server::DriftService;
    use rand::Rng;
    use std::sync::Arc;
    use std::time::{Duration, Instant};
    use tempfile::tempdir;
    use tonic::Request;

    const DIM: usize = 128;
    const TOTAL_VECTORS: usize = 100_000;
    const BATCH_SIZE: usize = 1_000;
    const QUERY_COUNT: usize = 100;

    fn generate_random_vector(rng: &mut impl Rng) -> Vec<f32> {
        (0..DIM).map(|_| rng.random::<f32>()).collect()
    }

    #[tokio::test]
    // #[ignore]
    async fn test_server_heavy_load_performance() {
        // =================================================================
        // 1. SETUP
        // =================================================================

        let _ = tracing_subscriber::fmt()
            .with_env_filter("drift_server=trace,info") // Show traces for your app, info for others
            .with_test_writer() // Print to test output
            .try_init(); // Safe to call multiple times

        let dir = tempdir().unwrap();

        let config = Config {
            port: 50051,
            storage_uri: format!("file://{}", dir.path().join("storage").to_string_lossy()),
            wal_dir: dir.path().join("wal"),
            default_dim: DIM,
            max_bucket_capacity: 110_000, // Trigger flushes often
            ef_construction: 80,
            ef_search: 200,
        };

        let manager = Arc::new(CollectionManager::new(config));
        let service = DriftService {
            manager: manager.clone(),
        };
        let collection = "stress_test";
        let mut rng = rand::rng();

        // =================================================================
        // 🚦 PHASE 1: WARM-UP & TRAINING
        // =================================================================
        // We insert enough vectors to exceed max_bucket_capacity (2000)
        // to force the FIRST flush and TRAIN the index now.
        println!("🔥 WARM-UP: Inserting 3,000 vectors to force initial training...");

        let warmup_count = 3000;
        let mut warmup_vecs = Vec::with_capacity(warmup_count);

        for i in 0..warmup_count {
            // Use high IDs for warmup to avoid collision with main test (optional but cleaner)
            let id = 10_000_000 + i as u64;
            warmup_vecs.push(Vector {
                id,
                values: generate_random_vector(&mut rng),
            });
        }

        service
            .insert_batch(Request::new(InsertBatchRequest {
                collection_name: collection.to_string(),
                vectors: warmup_vecs,
            }))
            .await
            .expect("Warmup insert failed");

        println!("⏳ WARM-UP: Sleeping 5s to allow Janitor to train and persist...");
        tokio::time::sleep(Duration::from_secs(5)).await;
        println!("✅ WARM-UP: Complete. Index is trained. Starting Main Load Test.");

        // =================================================================
        // 🚀 PHASE 2: MAIN INGESTION
        // =================================================================
        println!("🚀 Starting Load Test: {} Vectors", TOTAL_VECTORS);
        let start_ingest = Instant::now();

        for batch_idx in 0..(TOTAL_VECTORS / BATCH_SIZE) {
            let start_id = (batch_idx * BATCH_SIZE) as u64;

            let mut batch_vecs = Vec::with_capacity(BATCH_SIZE);
            for i in 0..BATCH_SIZE {
                let id = start_id + i as u64;
                let vector = generate_random_vector(&mut rng);
                batch_vecs.push(Vector { id, values: vector });
            }

            let req = Request::new(InsertBatchRequest {
                collection_name: collection.to_string(),
                vectors: batch_vecs,
            });

            service
                .insert_batch(req)
                .await
                .expect("Batch Insert failed");

            // Optional: Print rate every 10 batches
            if batch_idx > 0 && batch_idx % 10 == 0 {
                let elapsed = start_ingest.elapsed();
                let count = (batch_idx + 1) * BATCH_SIZE;
                let rate = count as f64 / elapsed.as_secs_f64();
                println!(
                    "    Inserted {} vectors... (Avg Rate: {:.0} vec/s)",
                    count, rate
                );
            }
        }

        let ingest_duration = start_ingest.elapsed();
        println!("✅ Ingestion Complete in {:.2?}", ingest_duration);
        println!(
            "    Throughput: {:.0} vec/sec",
            TOTAL_VECTORS as f64 / ingest_duration.as_secs_f64()
        );

        // =================================================================
        // 3. INJECT NEEDLE
        // =================================================================
        let needle_id = 999_999;
        let needle_vec = vec![0.5; DIM];
        println!("💉 Injecting Needle (ID {})...", needle_id);
        service
            .insert(Request::new(InsertRequest {
                collection_name: collection.to_string(),
                vector: Some(Vector {
                    id: needle_id,
                    values: needle_vec.clone(),
                }),
            }))
            .await
            .unwrap();
        println!("✅ Needle Injected.");

        // =================================================================
        // 4. WAIT FOR JANITOR
        // =================================================================
        println!("⏳ Waiting for background flush/indexing...");
        tokio::time::sleep(Duration::from_secs(4)).await;

        // =================================================================
        // 5. QUERY PHASE
        // =================================================================
        println!("🔍 Starting Query Benchmark...");
        let mut latencies = Vec::with_capacity(QUERY_COUNT);

        for _ in 0..QUERY_COUNT {
            let query_vec = generate_random_vector(&mut rng);
            let start_q = Instant::now();
            let req = Request::new(SearchRequest {
                collection_name: collection.to_string(),
                vector: query_vec,
                k: 10,
                target_confidence: 0.95,
                lambda: 1.0,
                tau: 100.0,
            });
            let _ = service.search(req).await.expect("Search failed");
            latencies.push(start_q.elapsed());
        }

        // =================================================================
        // 6. VERIFICATION
        // =================================================================
        println!("🔎 Verifying Needle...");
        let verify_req = Request::new(SearchRequest {
            collection_name: collection.to_string(),
            vector: needle_vec,
            k: 5,
            target_confidence: 0.99,
            lambda: 1.0,
            tau: 100.0,
        });

        let verify_res = service.search(verify_req).await.unwrap().into_inner();
        let found = verify_res.results.iter().any(|r| r.id == needle_id);

        // =================================================================
        // 7. REPORT
        // =================================================================
        latencies.sort();
        let p50 = latencies[latencies.len() / 2];
        let p99 = latencies[(latencies.len() as f64 * 0.99) as usize];

        println!("========================================");
        println!("📊 LOAD TEST RESULTS (Pre-Trained)");
        println!("Total Vectors: {}", TOTAL_VECTORS);
        println!("P50 Latency:   {:.2?}", p50);
        println!("P99 Latency:   {:.2?}", p99);
        println!(
            "Correctness:   {}",
            if found { "PASS ✅" } else { "FAIL ❌" }
        );
        println!("========================================");

        assert!(found, "Failed to find the needle vector!");
    }
}
