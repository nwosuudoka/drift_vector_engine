use drift_server::drift_proto::{InsertRequest, SearchRequest, Vector, drift_client::DriftClient};
use rand::Rng;
use std::time::{Duration, Instant};
use tokio::time::sleep;

// We reuse the proto module from the library crate to avoid recompiling protos twice
// Ensure drift_server/src/lib.rs exports the proto module (pub mod drift_proto ...)

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "http://127.0.0.1:50051";
    println!("Connecting to Drift Server at {}...", addr);

    // 1. Establish Connection
    // In production, we would configure TLS, timeouts, and keep-alive here.
    let mut client = match DriftClient::connect(addr).await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to connect: {}. Is the server running?", e);
            return Ok(());
        }
    };

    println!("--- TEST 1: HIGH THROUGHPUT INSERT ---");
    let dim = 128; // Must match server config
    let num_vectors = 2500; // Enough to trigger Janitor (limit 2000)
    let batch_size = 100;

    let mut rng = rand::rng();
    let start = Instant::now();

    for i in 0..num_vectors {
        let id = i as u64;
        // Generate random vector
        let values: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();

        let request = tonic::Request::new(InsertRequest {
            collection_name: "default".to_string(),
            vector: Some(Vector { id, values }),
        });

        if let Err(e) = client.insert(request).await {
            eprintln!("Insert failed at ID {}: {}", id, e);
        }

        if i % batch_size == 0 && i > 0 {
            println!("Inserted {} vectors...", i);
        }
    }

    let duration = start.elapsed();
    println!(
        "Inserted {} vectors in {:?}. Throughput: {:.2} ops/sec",
        num_vectors,
        duration,
        num_vectors as f64 / duration.as_secs_f64()
    );

    // Allow Janitor time to flush in the background
    println!("Waiting for background flush...");
    sleep(Duration::from_secs(3)).await;

    println!("--- TEST 2: SEARCH ACCURACY ---");
    // We search for a specific ID we inserted (e.g., ID 1000).
    // To verify accuracy, we need to know the vector.
    // For this generic test, we'll insert a KNOWN "Query Vector" at ID 99999.

    let query_id = 99999;
    let query_vec: Vec<f32> = vec![0.5; dim]; // Distinctive vector

    client
        .insert(tonic::Request::new(InsertRequest {
            collection_name: "default".to_string(),
            vector: Some(Vector {
                id: query_id,
                values: query_vec.clone(),
            }),
        }))
        .await?;

    println!("Inserted Query Target (ID: {})", query_id);

    // Search for it
    let search_req = tonic::Request::new(SearchRequest {
        collection_name: "default".to_string(),
        vector: query_vec, // Search with the exact same vector
        k: 5,
    });

    let response = client.search(search_req).await?.into_inner();

    println!("Search Results:");
    let mut found = false;
    for res in response.results {
        println!(" - ID: {}, Score: {:.4}", res.id, res.score);
        if res.id == query_id {
            found = true;
        }
    }

    if found {
        println!("✅ SUCCESS: Found inserted vector!");
    } else {
        println!("❌ FAILURE: Did not find vector ID {}", query_id);
    }

    Ok(())
}
