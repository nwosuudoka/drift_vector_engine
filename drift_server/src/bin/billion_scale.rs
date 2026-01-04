use clap::Parser;
use drift_server::config::{Config, FileConfig, StorageCommand};
use drift_server::drift_proto::drift_server::Drift;
use drift_server::drift_proto::{InsertBatchRequest, SearchRequest, Vector};
use drift_server::manager::CollectionManager;
use drift_server::server::DriftService;
use rand::prelude::*;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tempfile::tempdir;
use tokio::time::sleep;
use tonic::Request;

const DIM: usize = 128;
const BATCH_SIZE: usize = 5_000;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 100_000)]
    count: usize,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Safety: Catch background panics
    let orig_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        orig_hook(panic_info);
        std::process::exit(1);
    }));

    tracing_subscriber::fmt()
        .with_env_filter("drift_server=info")
        .init();

    let args = Args::parse();
    let dir = tempdir()?;
    let storage_path = dir.path().join("storage");

    // 2. Anti-Fragmentation Strategy
    // We want 1-5 large segments, not 100 small ones.
    // let optimal_capacity = args.count.max(2000);
    let optimal_capacity = 2000;

    let config = Config {
        port: 50099,
        wal_dir: dir.path().join("wal"),
        storage: StorageCommand::File(FileConfig { path: storage_path }),
        default_dim: DIM,
        max_bucket_capacity: optimal_capacity,
        ef_construction: 128,
        ef_search: 256,
    };

    let manager = Arc::new(CollectionManager::new(config));
    let service = DriftService {
        manager: manager.clone(),
    };
    let collection = "scale_test";

    println!("🚀 Starting Test: {} Vectors (Dim: {})", args.count, DIM);

    // ---------------------------------------------------------
    // 3. PLANT NEEDLES
    // ---------------------------------------------------------
    println!("\n🧵 Planting Needles...");
    let mut rng = StdRng::seed_from_u64(42);
    let mut queries = Vec::new();
    let mut needles = Vec::new();

    for q_idx in 0..10 {
        // Query in range [0, 100]
        let query_vec: Vec<f32> = (0..DIM).map(|_| rng.random::<f32>() * 100.0).collect();
        queries.push(query_vec.clone());
        for k in 0..10 {
            let mut needle = query_vec.clone();
            // Tiny noise (Dist ~0.1)
            for val in needle.iter_mut() {
                *val += (rng.random::<f32>() - 0.5) * 0.01;
            }

            needles.push(Vector {
                id: (q_idx * 10 + k) as u64,
                values: needle,
            });
        }
    }
    service
        .insert_batch(Request::new(InsertBatchRequest {
            collection_name: collection.to_string(),
            vectors: needles.clone(),
        }))
        .await?;

    // ---------------------------------------------------------
    // 4. FILL HAYSTACK
    // ---------------------------------------------------------
    println!("\n🚜 Filling Haystack...");
    let start_fill = Instant::now();
    let mut inserted = 0;
    let mut next_id = 1000;

    let coll = manager.get_or_create(collection, None).await.unwrap();
    let mut batch_vecs = Vec::with_capacity(BATCH_SIZE);
    for _ in 0..BATCH_SIZE {
        batch_vecs.push(vec![0.0; DIM]);
    }

    while inserted < args.count {
        while coll.index.memtable_len() > 500_000 {
            let n = coll.index.memtable_len();

            println!("   ✋ Backpressure: MemTable full. Waiting... {}", n);
            sleep(Duration::from_millis(500)).await;
        }

        let mut batch_proto = Vec::with_capacity(BATCH_SIZE);
        for i in 0..BATCH_SIZE {
            for d in 0..DIM {
                // Same geometric range [0, 100]
                batch_vecs[i][d] = rng.random::<f32>() * 100.0;
            }
            batch_proto.push(Vector {
                id: next_id,
                values: batch_vecs[i].clone(),
            });
            next_id += 1;
        }

        service
            .insert_batch(Request::new(InsertBatchRequest {
                collection_name: collection.to_string(),
                vectors: batch_proto,
            }))
            .await?;

        inserted += BATCH_SIZE;
        if inserted % 100_000 == 0 || inserted == args.count {
            let elapsed = start_fill.elapsed();
            println!(
                "   Progress: {} vectors | {:.0} vec/sec",
                inserted,
                inserted as f64 / elapsed.as_secs_f64()
            );
        }
    }

    // ---------------------------------------------------------
    // 5. FLUSH & DRAIN
    // ---------------------------------------------------------
    println!("\n⏳ Final Drain...");
    while coll.index.memtable_len() > 0 {
        sleep(Duration::from_secs(2)).await;
        println!("   Draining... ({} remaining)", coll.index.memtable_len());
    }
    loop {
        let has_frozen = coll.index.frozen_memtable.read().is_some();
        if !has_frozen {
            break;
        }
        println!("   Waiting for active flush...");
        sleep(Duration::from_secs(2)).await;
    }
    sleep(Duration::from_secs(2)).await;

    let headers = coll.index.get_all_bucket_headers();
    println!("✅ Index Ready. Total Buckets: {}", headers.len());

    // ---------------------------------------------------------
    // 6. RECALL TEST
    // ---------------------------------------------------------
    println!("\n🔍 Measuring Recall...");
    let mut total_hits = 0;

    for (i, query) in queries.iter().enumerate() {
        let start_q = Instant::now();
        let response = service
            .search(Request::new(SearchRequest {
                collection_name: collection.to_string(),
                vector: query.clone(),
                k: 10,
                target_confidence: 0.99,
                lambda: 0.05,
                tau: 20.0,
            }))
            .await?
            .into_inner();

        let duration = start_q.elapsed();
        let expected_start = (i * 10) as u64;
        let hits = response
            .results
            .iter()
            .filter(|r| r.id >= expected_start && r.id < expected_start + 10)
            .count();

        println!("   Query #{}: Hits {}/10 | {:.2?}ms", i, hits, duration);
        total_hits += hits;
    }

    let recall = (total_hits as f32 / 100.0) * 100.0;
    println!("\n📊 FINAL RECALL: {:.2}%", recall);

    if recall < 95.0 {
        panic!("❌ Failed Recall Target!");
    }

    Ok(())
}
