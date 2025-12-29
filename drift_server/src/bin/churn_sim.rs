use clap::Parser;
use drift_server::drift_proto::drift_server::Drift;
use drift_server::drift_proto::{SearchRequest, Vector};
use drift_server::manager::CollectionManager;
use drift_server::server::DriftService;
use rand::prelude::*;
use rand_distr::{Distribution, Normal};
use std::collections::{HashSet, VecDeque};
use std::sync::Arc;
use std::time::Duration;
use tempfile::tempdir;
use tokio::time::sleep;
use tonic::Request;
use walkdir::WalkDir; // Standard way to get dir size recursively

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

fn calculate_recall(results: &[u64], ground_truth: &[u64]) -> f32 {
    if ground_truth.is_empty() {
        return 0.0;
    }
    let hits = results
        .iter()
        .filter(|id| ground_truth.contains(id))
        .count();
    hits as f32 / ground_truth.len() as f32
}

fn get_dir_size(path: &std::path::Path) -> u64 {
    WalkDir::new(path)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter_map(|e| e.metadata().ok())
        .filter(|m| m.is_file())
        .map(|m| m.len())
        .sum()
}

async fn force_flush_and_wait(service: &DriftService, collection: &str) {
    let coll = service
        .manager
        .get_or_create(collection, None)
        .await
        .unwrap();
    let mut retries = 0;
    while coll.index.memtable_len() > 0 {
        sleep(Duration::from_millis(100)).await;
        retries += 1;
        if retries > 50 {
            break;
        }
    }
    sleep(Duration::from_millis(1000)).await; // Wait for FS sync
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
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let dir = tempdir()?;
    // Important: We look inside the collection folder for size
    let storage_root = dir.path().join("storage");
    let wal_path = dir.path().join("wal");

    // Config: Low capacity to ensure we have many buckets (easier to empty one)
    let config = drift_server::config::Config {
        port: 50053,
        storage_uri: format!("file://{}", storage_root.to_string_lossy()),
        wal_dir: wal_path.clone(),
        default_dim: args.dim,
        max_bucket_capacity: 500, // Small buckets
        ef_construction: 64,
        ef_search: 64,
    };

    let manager = Arc::new(CollectionManager::new(config));
    let service = DriftService {
        manager: manager.clone(),
    };
    let collection = "churn_world";

    // We need the full path to the collection data to measure size correctly
    let collection_data_path = storage_root.join(collection);

    let mut rng = StdRng::seed_from_u64(999);
    let cluster = StaticCluster::new(&mut rng, args.dim);

    // Shadow DB for Ground Truth
    // Using a HashMap for fast deletes by ID
    let mut shadow_db: Vec<(u64, Vec<f32>)> = Vec::new();
    let mut active_ids: VecDeque<u64> = VecDeque::new();
    let mut next_id = 0;

    println!("🌪️ Starting Churn Simulation");
    println!("   • Base Size: {}", args.base_size);
    println!("   • Churn Rate: {}/cycle", args.churn_batch);

    // ---------------------------------------------------------
    // PHASE 1: FILL
    // ---------------------------------------------------------
    println!("\n📦 Phase 1: Filling Database...");

    // Train first
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
    service
        .train(Request::new(drift_server::drift_proto::TrainRequest {
            collection_name: collection.to_string(),
            vectors: initial_batch.clone(),
        }))
        .await?;
    service
        .insert_batch(Request::new(
            drift_server::drift_proto::InsertBatchRequest {
                collection_name: collection.to_string(),
                vectors: initial_batch,
            },
        ))
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

    force_flush_and_wait(&service, collection).await;
    let initial_size = get_dir_size(&collection_data_path);
    let coll_ref = manager.get_or_create(collection, None).await.unwrap();
    let initial_buckets = coll_ref.index.get_all_bucket_headers().len();

    println!(
        "   💾 Initial Size: {:.2} MB | Buckets: {}",
        initial_size as f64 / 1024.0 / 1024.0,
        initial_buckets
    );

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
        // Update Shadow DB (Expensive but necessary for Ground Truth)
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

        force_flush_and_wait(&service, collection).await;

        // C. MEASURE RECALL
        let query_vec = cluster.generate_point(&mut rng);
        let k = 10;
        let ground_truth = calculate_ground_truth(&shadow_db, &query_vec, k);

        let res = service
            .search(Request::new(SearchRequest {
                collection_name: collection.to_string(),
                vector: query_vec,
                k: k as u32,
                target_confidence: 0.95,
                lambda: 1.0,
                tau: 100.0,
            }))
            .await?
            .into_inner();

        let result_ids: Vec<u64> = res.results.iter().map(|r| r.id).collect();
        let recall = calculate_recall(&result_ids, &ground_truth);

        let current_size = get_dir_size(&collection_data_path);
        let growth = (current_size as f64 / initial_size as f64) * 100.0;
        let buckets = coll_ref.index.get_all_bucket_headers().len();

        println!(
            "Recall: {:.1}% | Disk: {:.1}% | Buckets: {}",
            recall * 100.0,
            growth,
            buckets
        );

        if recall < 0.9 {
            println!("   ⚠️  Recall Low! Tombstones might be interfering.");
        }
    }

    // ---------------------------------------------------------
    // PHASE 3: DECAY (Scatter Merge Test)
    // ---------------------------------------------------------
    println!("\n💀 Phase 3: Mass Deletion (Testing Scatter Merge)...");

    // We delete 30% of the active IDs contiguous range to force empty buckets
    let kill_count = 6000;
    println!("   Deleting {} contiguous vectors...", kill_count);

    let mut deleted_ids = HashSet::new();
    for _ in 0..kill_count {
        if let Some(id) = active_ids.pop_front() {
            coll_ref.index.delete(id).unwrap();
            deleted_ids.insert(id);
        }
    }
    shadow_db.retain(|(id, _)| !deleted_ids.contains(id));

    // Wait for Janitor to notice and Merge
    println!("   Waiting for Janitor...");
    for _ in 0..10 {
        sleep(Duration::from_millis(500)).await;
        // Check bucket count
        let buckets = coll_ref.index.get_all_bucket_headers().len();
        // Also check if any buckets have 0 count
        let empty_buckets = coll_ref
            .index
            .get_all_bucket_headers()
            .iter()
            .filter(|h| h.count == 0)
            .count();
        if empty_buckets > 0 {
            print!(".");
        }
    }
    println!("");

    force_flush_and_wait(&service, collection).await;

    let final_buckets = coll_ref.index.get_all_bucket_headers().len();
    let final_size = get_dir_size(&collection_data_path);

    println!("   🏁 Final Report:");
    println!(
        "      Start Buckets: {} -> End Buckets: {}",
        initial_buckets, final_buckets
    );
    println!(
        "      Disk Size: {:.2} MB",
        final_size as f64 / 1024.0 / 1024.0
    );

    // Verify Merge happened
    if final_buckets < initial_buckets {
        println!("   ✅ SUCCESS: Bucket count dropped! Scatter Merge is working.");
    } else {
        println!("   ⚠️  WARNING: Bucket count did not drop. Check Janitor logic.");
    }

    // Verify Recall after mass delete
    let query_vec = cluster.generate_point(&mut rng);
    let ground_truth = calculate_ground_truth(&shadow_db, &query_vec, 10);
    let res = service
        .search(Request::new(SearchRequest {
            collection_name: collection.to_string(),
            vector: query_vec,
            k: 10,
            target_confidence: 0.95,
            lambda: 1.0,
            tau: 100.0,
        }))
        .await?
        .into_inner();
    let recall = calculate_recall(
        &res.results.iter().map(|r| r.id).collect::<Vec<_>>(),
        &ground_truth,
    );
    println!("      Final Recall: {:.1}%", recall * 100.0);

    Ok(())
}
