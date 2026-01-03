use clap::Parser;
use drift_core::math;
use drift_server::config::{Config, FileConfig, StorageCommand};
use drift_server::drift_proto::drift_server::Drift;
use drift_server::drift_proto::{InsertRequest, TrainRequest, Vector};
use drift_server::manager::CollectionManager;
use drift_server::server::DriftService;
use rand::prelude::*;
use rand_distr::{Distribution, Normal};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;
use tempfile::tempdir;
use tokio::time::sleep;
use tonic::Request; // Ensure this is available or use crate::math

// ... [DriftingCluster, World, calculate_ground_truth, calculate_recall, force_flush_and_wait remain the same] ...
// (I will omit them to save space, paste them from previous version)

// Copy DriftingCluster, World, Helpers here...
// =================================================================
// 1. DATA GENERATOR (Harder Mode)
// =================================================================

struct DriftingCluster {
    center: Vec<f32>,
    velocity: Vec<f32>,
    std_dev: f32,
}

impl DriftingCluster {
    fn new(rng: &mut impl Rng, dim: usize, drift_speed: f32) -> Self {
        let center: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() * 100.0).collect();
        let velocity: Vec<f32> = (0..dim)
            .map(|_| (rng.random::<f32>() - 0.5) * drift_speed)
            .collect();

        Self {
            center,
            velocity,
            std_dev: 15.0,
        }
    }

    fn tick(&mut self) {
        for (c, v) in self.center.iter_mut().zip(&self.velocity) {
            *c += *v;
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

struct World {
    clusters: Vec<DriftingCluster>,
    rng: StdRng,
}

impl World {
    fn new(dim: usize, num_clusters: usize, drift_speed: f32, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let clusters = (0..num_clusters)
            .map(|_| DriftingCluster::new(&mut rng, dim, drift_speed))
            .collect();
        Self { clusters, rng }
    }

    fn drift(&mut self) {
        for cluster in &mut self.clusters {
            cluster.tick();
        }
    }

    fn generate_batch(&mut self, batch_size: usize, start_id: u64) -> Vec<Vector> {
        let mut batch = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let cluster_idx = self.rng.random_range(0..self.clusters.len());
            let vec = self.clusters[cluster_idx].generate_point(&mut self.rng);

            batch.push(Vector {
                id: start_id + i as u64,
                values: vec,
            });
        }
        batch
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

async fn force_flush_and_wait(service: &DriftService, collection: &str) {
    let coll = service
        .manager
        .get_or_create(collection, None)
        .await
        .unwrap();

    // 1. Wait for MemTable drain
    let mut retries = 0;
    while coll.index.memtable_len() > 0 {
        sleep(Duration::from_millis(50)).await;
        retries += 1;
        if retries > 200 {
            break;
        } // 10s
    }

    // 2. Wait for Frozen to clear (Persistence complete)
    let mut frozen_retries = 0;
    loop {
        let has_frozen = coll.index.frozen_memtable.read().is_some();
        if !has_frozen {
            break;
        }
        sleep(Duration::from_millis(50)).await;
        frozen_retries += 1;
        if frozen_retries > 200 {
            break;
        }
    }

    sleep(Duration::from_millis(200)).await;
}

// =================================================================
// 3. MAIN
// =================================================================

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 128)]
    dim: usize,
    #[arg(long, default_value_t = 5)]
    clusters: usize,
    #[arg(long, default_value_t = 20)]
    batches: usize,
    #[arg(long, default_value_t = 1000)]
    batch_size: usize,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    let dir = tempdir()?;

    // ⚡ TUNED CONFIGURATION
    let config = Config {
        port: 50055,
        wal_dir: dir.path().join("wal"),
        storage: StorageCommand::File(FileConfig {
            path: dir.path().join("storage"),
        }),
        default_dim: args.dim,
        max_bucket_capacity: 300,
        ef_construction: 64,
        ef_search: 256,
    };

    let manager = Arc::new(CollectionManager::new(config));
    let service = DriftService {
        manager: manager.clone(),
    };
    let collection_name = "drift_simulation";

    let mut world = World::new(args.dim, args.clusters, 5.0, 999);
    let mut shadow_db: Vec<(u64, Vec<f32>)> = Vec::new();
    let mut next_id = 0;

    println!("🌍 Starting Concept Drift Simulation (Instrumented & Tuned)");
    println!("   • Clusters: {}", args.clusters);
    println!("   • Velocity: 5.0 (High)");
    println!("   • ef_search: 256");

    // ... (Training) ...
    println!("\n📚 Phase 1: Training...");
    let mut train_batch = world.generate_batch(2000, next_id);
    next_id += 2000;
    if !train_batch.is_empty() {
        train_batch[0].values = vec![-500.0; args.dim];
        train_batch[1].values = vec![1500.0; args.dim];
    }
    for v in &train_batch {
        shadow_db.push((v.id, v.values.clone()));
    }
    let train_req = Request::new(TrainRequest {
        collection_name: collection_name.to_string(),
        vectors: train_batch,
    });
    service.train(train_req).await?;
    println!("✅ Index Trained.");

    // Tuning Parameters
    let lambda = 0.05;
    let tau = 20.0;

    for b in 1..=args.batches {
        world.drift();
        let batch = world.generate_batch(args.batch_size, next_id);
        next_id += args.batch_size as u64;

        for v in &batch {
            shadow_db.push((v.id, v.values.clone()));
            service
                .insert(Request::new(InsertRequest {
                    collection_name: collection_name.to_string(),
                    vector: Some(v.clone()),
                }))
                .await?;
        }

        force_flush_and_wait(&service, collection_name).await;

        let query_vec = world.clusters[0].generate_point(&mut world.rng);
        let k = 10;
        let ground_truth = calculate_ground_truth(&shadow_db, &query_vec, k);

        let coll = manager.get_or_create(collection_name, None).await.unwrap();

        // Use DEBUG Search to track progress
        let (results, scanned_buckets, forensics) = coll
            .index
            .search_debug(
                &query_vec, k, 0.99, // Target Confidence
                lambda, tau,
            )
            .await
            .unwrap();

        let result_ids: HashSet<u64> = results.iter().map(|r| r.id).collect();
        let hits = ground_truth
            .iter()
            .filter(|id| result_ids.contains(id))
            .count();
        let recall = hits as f32 / k as f32;

        let bucket_count = coll.index.get_all_bucket_headers().len();

        // Stats
        let _q_lat = 0; // Removed timing to simplify, debug search is slow anyway

        println!(
            "Tick {:02}: Recall: {:>6.2}% | Buckets: {}",
            b,
            recall * 100.0,
            bucket_count
        );

        if recall < 0.9 {
            println!("\n🚨 RECALL FAILURE DETECTED ({}%)", recall * 100.0);
            println!(
                "   Scanned {} buckets: {:?}",
                scanned_buckets.len(),
                scanned_buckets
            );

            for &gt_id in &ground_truth {
                if !result_ids.contains(&gt_id) {
                    match coll.index.locate_vector(gt_id) {
                        Some(bucket_id) => {
                            let was_scanned = scanned_buckets.contains(&bucket_id);

                            // Get Bucket Stats
                            let headers = coll.index.get_all_bucket_headers();
                            let header = headers.iter().find(|h| h.id == bucket_id).unwrap();
                            let dist = crate::math::l2_sq(&query_vec, &header.centroid).sqrt();

                            // Check Forensic Data
                            let trace = forensics
                                .get(&gt_id.to_string())
                                .map(|s| s.as_str())
                                .unwrap_or("NOT_SCANNED");

                            println!(
                                "   ❌ Missed ID {}: Bucket {}. Scanned? {} | Dist={:.2} | Trace: {}",
                                gt_id, bucket_id, was_scanned, dist, trace
                            );
                        }
                        None => println!("   ❌ Missed ID {}: LOST!", gt_id),
                    }
                }
            }
        }
    }

    println!("\n✅ Simulation Complete.");
    Ok(())
}
