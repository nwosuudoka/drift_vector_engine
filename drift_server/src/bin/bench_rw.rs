use clap::Parser;
use drift_core::math::Metric;
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

fn gen_random_vector(rng: &mut impl Rng, dim: usize) -> Vec<f32> {
    (0..dim).map(|_| rng.random::<f32>()).collect()
}

fn percentile(sorted: &[Duration], pct: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::from_millis(0);
    }
    let idx = ((sorted.len() as f64) * pct).floor() as usize;
    let idx = idx.min(sorted.len() - 1);
    sorted[idx]
}

async fn wait_for_flush(
    coll: &Arc<drift_server::manager::Collection>,
    wait_active_empty: bool,
    timeout: Duration,
) -> (usize, usize) {
    let start = Instant::now();
    loop {
        let mem_len = coll.index.memtable_len();
        let frozen = coll.index.get_frozen_count();
        let active_ok = !wait_active_empty || mem_len == 0;
        if frozen == 0 && active_ok {
            return (mem_len, frozen);
        }
        if start.elapsed() >= timeout {
            return (mem_len, frozen);
        }
        sleep(Duration::from_millis(100)).await;
    }
}

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 128)]
    dim: usize,
    #[arg(long, default_value_t = 20_000)]
    total_vectors: usize,
    #[arg(long, default_value_t = 1_000)]
    batch_size: usize,
    #[arg(long, default_value_t = 200)]
    query_count: usize,
    #[arg(long, default_value_t = 10)]
    k: usize,
    #[arg(long, default_value_t = 0.99)]
    target_confidence: f32,
    #[arg(long, default_value_t = 0.05)]
    lambda: f32,
    #[arg(long, default_value_t = 20.0)]
    tau: f32,
    #[arg(long, default_value_t = 1_000)]
    max_bucket_capacity: usize,
    #[arg(long, default_value_t = 64)]
    ef_construction: usize,
    #[arg(long, default_value_t = 128)]
    ef_search: usize,
    #[arg(long, default_value_t = 999)]
    seed: u64,
    #[arg(long, default_value_t = 20)]
    warmup_queries: usize,
    #[arg(long, default_value_t = 3_000)]
    flush_timeout_ms: u64,
    #[arg(long, default_value_t = false)]
    wait_active_empty: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter("drift_server=info")
        .init();

    let args = Args::parse();

    let dir = tempdir()?;
    let config = Config {
        port: 50056,
        wal_dir: dir.path().join("wal"),
        data_dir: dir.path().join("data"),
        storage: StorageCommand::File(FileConfig {
            path: dir.path().join("storage"),
        }),
        default_dim: args.dim,
        max_bucket_capacity: args.max_bucket_capacity,
        ef_construction: args.ef_construction,
        ef_search: args.ef_search,
    };

    let manager = Arc::new(CollectionManager::new(config));
    let service = DriftService {
        manager: manager.clone(),
    };
    let collection = "bench_rw";
    manager
        .get_or_create(
            collection,
            Some(args.dim),
            Some(args.max_bucket_capacity),
            Some(Metric::L2),
        )
        .await?;

    println!("🏁 Bench RW (v2)");
    println!(
        "   • Vectors: {} (batch {})",
        args.total_vectors, args.batch_size
    );
    println!("   • Queries: {} (k={})", args.query_count, args.k);
    println!(
        "   • Routing: target_confidence={:.2}, lambda={:.3}, tau={:.1}",
        args.target_confidence, args.lambda, args.tau
    );
    println!(
        "   • Index: max_bucket_capacity={}, ef_construction={}, ef_search={}",
        args.max_bucket_capacity, args.ef_construction, args.ef_search
    );

    let mut rng = StdRng::seed_from_u64(args.seed);

    // -------------------------------
    // WRITE BENCH
    // -------------------------------
    println!("\n✍️  Write Benchmark...");
    let start_ingest = Instant::now();
    let mut batch_latencies = Vec::new();
    let mut next_id: u64 = 0;

    while (next_id as usize) < args.total_vectors {
        let remaining = args.total_vectors - next_id as usize;
        let batch_count = remaining.min(args.batch_size);
        let mut batch_vecs = Vec::with_capacity(batch_count);

        for _ in 0..batch_count {
            let vector = gen_random_vector(&mut rng, args.dim);
            batch_vecs.push(Vector {
                id: next_id,
                values: vector,
            });
            next_id += 1;
        }

        let batch_start = Instant::now();
        service
            .insert_batch(Request::new(InsertBatchRequest {
                collection_name: collection.to_string(),
                vectors: batch_vecs,
            }))
            .await?;
        batch_latencies.push(batch_start.elapsed());
    }

    let ingest_duration = start_ingest.elapsed();
    let write_throughput = args.total_vectors as f64 / ingest_duration.as_secs_f64();
    batch_latencies.sort();
    let w_p50 = percentile(&batch_latencies, 0.50);
    let w_p95 = percentile(&batch_latencies, 0.95);
    let w_p99 = percentile(&batch_latencies, 0.99);

    println!("✅ Write Complete in {:.2?}", ingest_duration);
    println!("   • Throughput: {:.0} vec/sec", write_throughput);
    println!(
        "   • Batch Latency p50/p95/p99: {:.2?} / {:.2?} / {:.2?}",
        w_p50, w_p95, w_p99
    );

    // Flush wait (best-effort)
    let coll = manager
        .get_or_create(collection, Some(args.dim), None, Some(Metric::L2))
        .await
        .unwrap();
    let (mem_len, frozen) = wait_for_flush(
        &coll,
        args.wait_active_empty,
        Duration::from_millis(args.flush_timeout_ms),
    )
    .await;
    if frozen > 0 || (args.wait_active_empty && mem_len > 0) {
        println!(
            "⚠️  Flush wait timed out (active={}, frozen={})",
            mem_len, frozen
        );
    }

    // -------------------------------
    // READ BENCH
    // -------------------------------
    println!("\n🔍 Read Benchmark...");

    for _ in 0..args.warmup_queries {
        let q = gen_random_vector(&mut rng, args.dim);
        let _ = service
            .search(Request::new(SearchRequest {
                collection_name: collection.to_string(),
                vector: q,
                k: args.k as u32,
                target_confidence: args.target_confidence,
                lambda: args.lambda,
                tau: args.tau,
            }))
            .await?;
    }

    let mut read_latencies = Vec::with_capacity(args.query_count);
    let start_reads = Instant::now();
    for _ in 0..args.query_count {
        let q = gen_random_vector(&mut rng, args.dim);
        let q_start = Instant::now();
        let _ = service
            .search(Request::new(SearchRequest {
                collection_name: collection.to_string(),
                vector: q,
                k: args.k as u32,
                target_confidence: args.target_confidence,
                lambda: args.lambda,
                tau: args.tau,
            }))
            .await?;
        read_latencies.push(q_start.elapsed());
    }
    let read_duration = start_reads.elapsed();
    let read_qps = args.query_count as f64 / read_duration.as_secs_f64();

    read_latencies.sort();
    let r_p50 = percentile(&read_latencies, 0.50);
    let r_p95 = percentile(&read_latencies, 0.95);
    let r_p99 = percentile(&read_latencies, 0.99);

    println!("✅ Read Complete in {:.2?}", read_duration);
    println!("   • QPS: {:.0} q/s", read_qps);
    println!(
        "   • Query Latency p50/p95/p99: {:.2?} / {:.2?} / {:.2?}",
        r_p50, r_p95, r_p99
    );

    println!("\n✅ Bench Complete.");
    Ok(())
}
