use clap::Parser;
use drift_core::index::VectorIndex;
use drift_core::math::Metric;
use drift_server::config::{Config, FileConfig, StorageCommand};
use drift_server::drift_proto::drift_server::Drift;
use drift_server::drift_proto::{SearchRequest, Vector};
use drift_server::manager::CollectionManager; // ⚡ UPDATED IMPORT
use drift_server::server::DriftService; // ⚡ UPDATED IMPORT
use rand::prelude::*;
use rand_distr::{Distribution, Normal};
use std::collections::{HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tempfile::tempdir;
use tokio::time::sleep;
use tonic::Request;
use walkdir::WalkDir;

// =================================================================
// HELPERS
// =================================================================

struct StaticCluster {
    center: Vec<f32>,
    std_dev: f32,
}

impl StaticCluster {
    fn new(rng: &mut impl Rng, dim: usize) -> Self {
        let center: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() * 100.0).collect();
        Self {
            center,
            std_dev: 15.0,
        }
    }

    fn generate_point(&self, rng: &mut impl Rng) -> Vec<f32> {
        let normal = Normal::new(0.0, self.std_dev).unwrap();
        self.center
            .iter()
            .map(|&c| c + normal.sample(rng))
            .collect()
    }
}

fn calculate_ground_truth(all_vectors: &[(u64, Vec<f32>)], query: &[f32], k: usize) -> Vec<u64> {
    let mut scored: Vec<(u64, f32)> = all_vectors
        .iter()
        .map(|(id, vec)| {
            let dist_sq: f32 = vec.iter().zip(query).map(|(a, b)| (a - b).powi(2)).sum();
            (*id, dist_sq)
        })
        .collect();

    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    scored.into_iter().take(k).map(|(id, _)| id).collect()
}

fn calculate_recall_at_k(results: &[u64], ground_truth: &[u64], eval_k: usize) -> f32 {
    let denom = ground_truth.len().min(eval_k);
    if denom == 0 {
        return 0.0;
    }
    let hits = results
        .iter()
        .take(eval_k)
        .filter(|id| ground_truth.contains(id))
        .count();
    hits as f32 / denom as f32
}

fn count_deleted_hits(results: &[u64], deleted_ids: &HashSet<u64>) -> usize {
    results.iter().filter(|id| deleted_ids.contains(id)).count()
}

fn summarize_counts(counts: &[u32]) -> Option<(u32, u32, u32)> {
    if counts.is_empty() {
        return None;
    }
    let mut sorted = counts.to_vec();
    sorted.sort_unstable();
    let min = sorted[0];
    let max = sorted[sorted.len() - 1];
    let p50 = sorted[sorted.len() / 2];
    Some((min, p50, max))
}

fn estimate_kv_coverage<R: Rng>(
    index: &VectorIndex,
    shadow_db: &[(u64, Vec<f32>)],
    sample: usize,
    rng: &mut R,
) -> f32 {
    if shadow_db.is_empty() || sample == 0 {
        return 0.0;
    }
    let total = sample.min(shadow_db.len());
    let kv = index.get_kv();
    let mut hits = 0usize;
    for _ in 0..total {
        let idx = rng.random_range(0..shadow_db.len());
        let id = shadow_db[idx].0;
        if kv.get(&id.to_le_bytes()).ok().flatten().is_some() {
            hits += 1;
        }
    }
    hits as f32 / total as f32
}

fn get_dir_size(path: &std::path::Path) -> u64 {
    if !path.exists() {
        return 0;
    }
    WalkDir::new(path)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter_map(|e| e.metadata().ok())
        .filter(|m| m.is_file())
        .map(|m| m.len())
        .sum()
}

async fn force_flush_and_wait(coll: &Arc<drift_server::manager::Collection>) {
    let mut retries = 0;
    // Wait for MemTable to empty
    while coll.index.memtable_len() > 0 {
        sleep(Duration::from_millis(100)).await;
        retries += 1;
        if retries > 50 {
            break;
        }
    }

    // Wait for Frozen tables to flush
    loop {
        if coll.index.get_frozen_count() == 0 {
            break;
        }
        sleep(Duration::from_millis(100)).await;
    }

    sleep(Duration::from_millis(500)).await; // Wait for FS sync
}

async fn wait_for_router_ready(
    coll: &Arc<drift_server::manager::Collection>,
    min_buckets: usize,
    timeout: Duration,
) -> usize {
    let start = Instant::now();
    loop {
        let bucket_count = coll.index.get_router().read().get_snapshot().1.len();
        if bucket_count >= min_buckets {
            return bucket_count;
        }
        if start.elapsed() >= timeout {
            return bucket_count;
        }
        sleep(Duration::from_millis(100)).await;
    }
}

struct QueryMetrics {
    avg_recall: f32,
    min_recall: f32,
    deleted_hit_rate: f32,
    avg_latency_ms: f32,
}

async fn measure_recall<R: Rng>(
    service: &DriftService,
    collection: &str,
    cluster: &StaticCluster,
    rng: &mut R,
    shadow_db: &[(u64, Vec<f32>)],
    deleted_ids: &HashSet<u64>,
    args: &Args,
) -> Result<QueryMetrics, Box<dyn std::error::Error>> {
    let mut total_recall = 0.0f32;
    let mut min_recall = 1.0f32;
    let mut deleted_hits = 0usize;
    let mut results_seen = 0usize;
    let mut total_latency = Duration::from_millis(0);

    for _ in 0..args.queries_per_cycle {
        let query_vec = cluster.generate_point(rng);
        let ground_truth = calculate_ground_truth(shadow_db, &query_vec, args.eval_k);

        let start = Instant::now();
        let res = service
            .search(Request::new(SearchRequest {
                collection_name: collection.to_string(),
                vector: query_vec,
                k: args.search_k as u32,
                target_confidence: args.target_confidence,
                lambda: args.lambda,
                tau: args.tau,
            }))
            .await?
            .into_inner();
        total_latency += start.elapsed();

        let result_ids: Vec<u64> = res.results.iter().map(|r| r.id).collect();
        let recall = calculate_recall_at_k(&result_ids, &ground_truth, args.eval_k);
        total_recall += recall;
        if recall < min_recall {
            min_recall = recall;
        }
        deleted_hits += count_deleted_hits(&result_ids, deleted_ids);
        results_seen += result_ids.len();
    }

    let queries = args.queries_per_cycle.max(1) as f32;
    let avg_recall = total_recall / queries;
    let avg_latency_ms = (total_latency.as_millis() as f32) / queries;
    let deleted_hit_rate = if results_seen > 0 {
        deleted_hits as f32 / results_seen as f32
    } else {
        0.0
    };

    Ok(QueryMetrics {
        avg_recall,
        min_recall,
        deleted_hit_rate,
        avg_latency_ms,
    })
}

// =================================================================
// SIMULATION
// =================================================================

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 128)]
    dim: usize,
    #[arg(long, default_value_t = 20_000)]
    base_size: usize,
    #[arg(long, default_value_t = 2_000)]
    churn_batch: usize,
    #[arg(long, default_value_t = 5)]
    cycles: usize,
    #[arg(long, default_value_t = 999)]
    seed: u64,
    #[arg(long, default_value_t = 500)]
    max_bucket_capacity: usize,
    #[arg(long, default_value_t = 0.99)]
    target_confidence: f32,
    #[arg(long, default_value_t = 0.05)]
    lambda: f32,
    #[arg(long, default_value_t = 20.0)]
    tau: f32,
    #[arg(long, default_value_t = 50)]
    search_k: usize,
    #[arg(long, default_value_t = 10)]
    eval_k: usize,
    #[arg(long, default_value_t = 20)]
    queries_per_cycle: usize,
    #[arg(long, default_value_t = 0.90)]
    min_recall: f32,
    #[arg(long, default_value_t = 200)]
    kv_sample: usize,
    #[arg(long, default_value_t = 0.50)]
    min_kv_coverage: f32,
    #[arg(long, default_value_t = 3_000)]
    router_wait_ms: u64,
    #[arg(long, default_value_t = false)]
    strict: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Safety Hook
    let orig_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        orig_hook(panic_info);
        std::process::exit(1);
    }));

    // Force info logging
    tracing_subscriber::fmt()
        .with_env_filter("drift_server=info")
        .init();

    let mut args = Args::parse();
    if args.queries_per_cycle == 0 {
        return Err("queries_per_cycle must be > 0".into());
    }
    if args.eval_k == 0 {
        return Err("eval_k must be > 0".into());
    }
    if args.search_k < args.eval_k {
        println!(
            "⚠️  search_k ({}) < eval_k ({}). Bumping search_k to {}.",
            args.search_k, args.eval_k, args.eval_k
        );
        args.search_k = args.eval_k;
    }

    let dir = tempdir()?;
    let root = dir.path();
    let storage_root = root.join("storage");
    std::fs::create_dir_all(&storage_root)?;

    //  Use StorageCommand strategy
    let config = Config {
        port: 50053,
        wal_dir: root.join("wal"),
        data_dir: root.join("data"),

        // Use File Strategy
        storage: StorageCommand::File(FileConfig {
            path: storage_root.clone(),
        }),

        default_dim: args.dim,
        max_bucket_capacity: args.max_bucket_capacity, // Small buckets to force splits
        ef_construction: 64,
        ef_search: 64,
    };

    // Initialize manager
    let manager = Arc::new(CollectionManager::new(config));
    let service = DriftService {
        manager: manager.clone(),
    };
    let collection = "churn_world";

    let mut rng = StdRng::seed_from_u64(args.seed);
    let cluster = StaticCluster::new(&mut rng, args.dim);

    // Shadow DB for Ground Truth
    let mut shadow_db: Vec<(u64, Vec<f32>)> = Vec::new();
    let mut active_ids: VecDeque<u64> = VecDeque::new();
    let mut deleted_all: HashSet<u64> = HashSet::new();
    let mut next_id = 0;

    println!("🌪️ Starting Churn Simulation (v3)");
    println!("   • Base Size: {}", args.base_size);
    println!("   • Churn Rate: {}/cycle", args.churn_batch);
    println!(
        "   • Search K (eval/search): {}/{}",
        args.eval_k, args.search_k
    );
    println!(
        "   • Routing: target_confidence={:.2}, lambda={:.3}, tau={:.1}",
        args.target_confidence, args.lambda, args.tau
    );
    println!("   • Queries/Cycle: {}", args.queries_per_cycle);
    println!("   • Min Recall Target: {:.1}%", args.min_recall * 100.0);
    if args.strict {
        println!("   • Strict Mode: ON");
    }

    // ---------------------------------------------------------
    // PHASE 1: FILL
    // ---------------------------------------------------------
    println!("\n📦 Phase 1: Filling Database...");

    // Get collection reference early for monitoring
    // Pass explicit dim hint to avoid panics
    let coll_ref = manager
        .get_or_create(collection, Some(args.dim), None, Some(Metric::L2))
        .await
        .unwrap();

    // Train first (Initial batch)
    let mut initial_batch = Vec::new();
    for _ in 0..2000 {
        let vec = cluster.generate_point(&mut rng);
        initial_batch.push(Vector {
            id: next_id,
            values: vec.clone(),
        });
        shadow_db.push((next_id, vec));
        active_ids.push_back(next_id);
        next_id += 1;
    }

    // Train is just an insert batch that triggers auto-training
    service
        .train(Request::new(drift_server::drift_proto::TrainRequest {
            collection_name: collection.to_string(),
            vectors: initial_batch.clone(),
        }))
        .await?;

    // Fill rest
    while shadow_db.len() < args.base_size {
        let mut batch = Vec::new();
        for _ in 0..1000 {
            let vec = cluster.generate_point(&mut rng);
            batch.push(Vector {
                id: next_id,
                values: vec.clone(),
            });
            shadow_db.push((next_id, vec));
            active_ids.push_back(next_id);
            next_id += 1;
        }
        service
            .insert_batch(Request::new(
                drift_server::drift_proto::InsertBatchRequest {
                    collection_name: collection.to_string(),
                    vectors: batch,
                },
            ))
            .await?;
    }

    force_flush_and_wait(&coll_ref).await;

    // Data is in root/data/<collection>/staging and root/storage/<collection>
    // We measure the whole root/data directory for simplicity
    let initial_size = get_dir_size(&root.join("data").join(collection));

    // Use get_router().read().get_snapshot().1.len()?
    // Or rely on the test logic. Index doesn't expose bucket headers publicly without
    // digging into index internals which are crate-private.
    // We'll skip printing bucket counts here unless `get_bucket_count()` is exposed.
    // `get_all_bucket_headers` is not available (it was part of older architecture).
    // Let's just print size.
    println!(
        "   💾 Initial Size: {:.2} MB",
        initial_size as f64 / 1024.0 / 1024.0
    );

    let router_bucket_count =
        wait_for_router_ready(&coll_ref, 1, Duration::from_millis(args.router_wait_ms)).await;
    if router_bucket_count == 0 {
        println!("   ⚠️  Router still empty after warmup. Disk search will be skipped.");
        if args.strict {
            return Err("router not ready".into());
        }
    }

    let (_centroids, router_ids, router_counts) = coll_ref
        .index
        .get_router()
        .read()
        .get_snapshot_with_counts();
    let suppressed = router_counts
        .iter()
        .filter(|&&c| (c as f32) < args.tau)
        .count();
    match summarize_counts(&router_counts) {
        Some((min, p50, max)) => {
            println!(
                "   🧭 Router: buckets={} | Size (Min/P50/Max): {}/{}/{} | <Tau: {}",
                router_ids.len(),
                min,
                p50,
                max,
                suppressed
            );
        }
        None => {
            println!("   🧭 Router: buckets=0");
        }
    }

    let kv_coverage = estimate_kv_coverage(
        coll_ref.index.as_ref(),
        &shadow_db,
        args.kv_sample,
        &mut rng,
    );
    println!(
        "   🔑 KV Coverage (sample {}): {:.1}%",
        args.kv_sample,
        kv_coverage * 100.0
    );
    if kv_coverage < args.min_kv_coverage {
        println!(
            "   ⚠️  Low KV coverage (min {:.0}%). L1 tombstone filtering may be ineffective.",
            args.min_kv_coverage * 100.0
        );
        if args.strict {
            return Err("kv coverage below threshold".into());
        }
    }

    // ---------------------------------------------------------
    // PHASE 2: CHURN (Random Replace) - Testing Recall
    // ---------------------------------------------------------
    for c in 1..=args.cycles {
        print!("\n🔄 Cycle {}: ", c);

        // A. DELETE Oldest (Simulate Churn)
        let mut deleted_ids = HashSet::new();
        for _ in 0..args.churn_batch {
            if let Some(id) = active_ids.pop_front() {
                coll_ref.index.delete(id).unwrap();
                deleted_ids.insert(id);
            }
        }
        deleted_all.extend(deleted_ids.iter());
        shadow_db.retain(|(id, _)| !deleted_ids.contains(id));

        // B. INSERT New
        let mut batch = Vec::new();
        for _ in 0..args.churn_batch {
            let vec = cluster.generate_point(&mut rng);
            batch.push(Vector {
                id: next_id,
                values: vec.clone(),
            });
            shadow_db.push((next_id, vec));
            active_ids.push_back(next_id);
            next_id += 1;
        }
        service
            .insert_batch(Request::new(
                drift_server::drift_proto::InsertBatchRequest {
                    collection_name: collection.to_string(),
                    vectors: batch,
                },
            ))
            .await?;

        force_flush_and_wait(&coll_ref).await;

        // C. MEASURE RECALL
        let metrics = measure_recall(
            &service,
            collection,
            &cluster,
            &mut rng,
            &shadow_db,
            &deleted_all,
            &args,
        )
        .await?;

        let current_size = get_dir_size(&root.join("data").join(collection));
        let growth = if initial_size > 0 {
            (current_size as f64 / initial_size as f64) * 100.0
        } else {
            0.0
        };

        let (_centroids, router_ids, router_counts) = coll_ref
            .index
            .get_router()
            .read()
            .get_snapshot_with_counts();
        let suppressed = router_counts
            .iter()
            .filter(|&&c| (c as f32) < args.tau)
            .count();

        match summarize_counts(&router_counts) {
            Some((min, p50, max)) => {
                println!(
                    "Recall avg/min: {:>6.2}%/{:>6.2}% | Deleted Hit Rate: {:>5.2}% | Q Avg: {:>4.1}ms | Buckets: {} | Size (Min/P50/Max): {}/{}/{} | <Tau: {} | Disk Growth: {:.1}%",
                    metrics.avg_recall * 100.0,
                    metrics.min_recall * 100.0,
                    metrics.deleted_hit_rate * 100.0,
                    metrics.avg_latency_ms,
                    router_ids.len(),
                    min,
                    p50,
                    max,
                    suppressed,
                    growth
                );
            }
            None => {
                println!(
                    "Recall avg/min: {:>6.2}%/{:>6.2}% | Deleted Hit Rate: {:>5.2}% | Q Avg: {:>4.1}ms | Buckets: 0 | Disk Growth: {:.1}%",
                    metrics.avg_recall * 100.0,
                    metrics.min_recall * 100.0,
                    metrics.deleted_hit_rate * 100.0,
                    metrics.avg_latency_ms,
                    growth
                );
            }
        }

        if metrics.avg_recall < args.min_recall {
            println!("   ⚠️  Recall Low! Tombstones or routing might be interfering.");
            if args.strict {
                return Err("recall below threshold".into());
            }
        }
        if metrics.deleted_hit_rate > 0.0 {
            println!("   ⚠️  Deleted IDs returned in results (tombstone path degraded).");
            if args.strict {
                return Err("deleted ids returned".into());
            }
        }
    }

    // ---------------------------------------------------------
    // PHASE 3: DECAY (Scatter Merge Test)
    // ---------------------------------------------------------
    println!("\n💀 Phase 3: Mass Deletion (Testing Scatter Merge)...");

    let kill_count = 6000;
    println!("   Deleting {} contiguous vectors...", kill_count);

    let mut deleted_ids = HashSet::new();
    for _ in 0..kill_count {
        if let Some(id) = active_ids.pop_front() {
            coll_ref.index.delete(id).unwrap();
            deleted_ids.insert(id);
        }
    }
    deleted_all.extend(deleted_ids.iter());
    shadow_db.retain(|(id, _)| !deleted_ids.contains(id));

    println!("   Waiting for Janitor to compact...");
    for _ in 0..10 {
        sleep(Duration::from_millis(500)).await;
        print!(".");
    }
    println!();

    force_flush_and_wait(&coll_ref).await;

    let final_size = get_dir_size(&root.join("data").join(collection));

    println!("   🏁 Final Report:");
    println!(
        "      Disk Size: {:.2} MB",
        final_size as f64 / 1024.0 / 1024.0
    );

    // Verify Recall after mass delete
    let final_metrics = measure_recall(
        &service,
        collection,
        &cluster,
        &mut rng,
        &shadow_db,
        &deleted_all,
        &args,
    )
    .await?;
    println!(
        "      Final Recall (avg/min): {:.1}% / {:.1}%",
        final_metrics.avg_recall * 100.0,
        final_metrics.min_recall * 100.0
    );
    if final_metrics.deleted_hit_rate > 0.0 {
        println!(
            "      ⚠️  Final Deleted Hit Rate: {:.2}%",
            final_metrics.deleted_hit_rate * 100.0
        );
    }

    // ⚡ CLEANUP: Gracefully stop the Janitor
    coll_ref.janitor_task.abort();
    println!("🧹 Janitor stopped.");

    Ok(())
}
