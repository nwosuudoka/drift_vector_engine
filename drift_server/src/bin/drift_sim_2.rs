use clap::Parser;
use drift_server::drift_proto::drift_server::Drift;
use drift_server::drift_proto::{InsertRequest, SearchRequest, TrainRequest, Vector};
use drift_server::manager::CollectionManager;
use drift_server::server::DriftService;
use rand::prelude::*;
use rand_distr::{Distribution, Normal};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tempfile::tempdir;
use tokio::time::sleep;
use tonic::Request;

// =================================================================
// 1. THE DRIFT SIMULATION ENGINE
// =================================================================

struct DriftingCluster {
    center: Vec<f32>,
    velocity: Vec<f32>,
    std_dev: f32,
}

impl DriftingCluster {
    fn new(rng: &mut impl Rng, dim: usize, drift_speed: f32) -> Self {
        // Initialize random position in 100x100 hypercube
        let center: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() * 100.0).collect();

        // Initialize random velocity vector
        let velocity: Vec<f32> = (0..dim)
            .map(|_| (rng.random::<f32>() - 0.5) * drift_speed)
            .collect();

        Self {
            center,
            velocity,
            std_dev: 15.0, // Spread of the cluster
        }
    }

    fn tick(&mut self) {
        // Apply velocity to position
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
            // Pick a random cluster for this point
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

// =================================================================
// 2. GROUND TRUTH & METRICS
// =================================================================

fn calculate_ground_truth(all_vectors: &[(u64, Vec<f32>)], query: &[f32], k: usize) -> Vec<u64> {
    let mut scored: Vec<(u64, f32)> = all_vectors
        .iter()
        .map(|(id, vec)| {
            let dist_sq: f32 = vec.iter().zip(query).map(|(a, b)| (a - b).powi(2)).sum();
            (*id, dist_sq)
        })
        .collect();

    // Sort by Distance ASC
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

// Wait for MemTable to flush so we test the Disk Index
async fn force_flush_and_wait(service: &DriftService, collection: &str) {
    let coll = service
        .manager
        .get_or_create(collection, None)
        .await
        .unwrap();

    // Busy wait for Janitor
    let mut retries = 0;
    while coll.index.memtable_len() > 0 {
        sleep(Duration::from_millis(100)).await;
        retries += 1;
        if retries > 100 {
            println!("⚠️ Warning: MemTable flush timeout.");
            break;
        }
    }
    // Buffer for FS sync
    sleep(Duration::from_millis(500)).await;
}

// =================================================================
// 3. MAIN HARNESS
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

    // 1. Spin up Server
    let dir = tempdir()?;
    let config = drift_server::config::Config {
        port: 50055,
        storage_uri: format!("file://{}", dir.path().join("storage").to_string_lossy()),
        wal_dir: dir.path().join("wal"),
        default_dim: args.dim,
        max_bucket_capacity: 600, // Small capacity to force frequent Splitting
        ef_construction: 64,
        ef_search: 64,
    };

    let manager = Arc::new(CollectionManager::new(config));
    let service = DriftService {
        manager: manager.clone(),
    };
    let collection_name = "drift_simulation";

    // 2. Initialize World
    // Drift Speed 5.0 is fast. It means clusters move 5 units per tick.
    // Over 20 ticks, they move 100 units (entire initial width).
    let mut world = World::new(args.dim, args.clusters, 5.0, 999);
    let mut shadow_db: Vec<(u64, Vec<f32>)> = Vec::new();
    let mut next_id = 0;

    println!("🌍 Starting Concept Drift Simulation");
    println!("   • Clusters: {}", args.clusters);
    println!("   • Velocity: 5.0 (High)");
    println!("   • Buckets:  Splitting enabled at 900 vectors");

    // 3. Initial Training
    println!("\n📚 Phase 1: Training...");
    let train_batch = world.generate_batch(2000, next_id);
    next_id += 2000;

    for v in &train_batch {
        shadow_db.push((v.id, v.values.clone()));
    }

    let train_req = Request::new(TrainRequest {
        collection_name: collection_name.to_string(),
        vectors: train_batch,
    });
    service.train(train_req).await?;
    println!("✅ Index Trained.");

    // 4. The Loop
    for b in 1..=args.batches {
        // A. Move the World
        world.drift();

        // B. Generate New Data (at new positions)
        let batch = world.generate_batch(args.batch_size, next_id);
        next_id += args.batch_size as u64;

        // C. Ingest
        for v in &batch {
            shadow_db.push((v.id, v.values.clone()));
            service
                .insert(Request::new(InsertRequest {
                    collection_name: collection_name.to_string(),
                    vector: Some(v.clone()),
                }))
                .await?;
        }

        // D. Flush (Push to Disk)
        force_flush_and_wait(&service, collection_name).await;

        // E. Verify Recall
        // We query for a point generated from the *current* cluster positions.
        // If the index hasn't adapted (split/merged), it will look in old locations and miss.
        let query_vec = world.clusters[0].generate_point(&mut world.rng);
        let k = 10;
        let ground_truth = calculate_ground_truth(&shadow_db, &query_vec, k);

        let search_req = Request::new(SearchRequest {
            collection_name: collection_name.to_string(),
            vector: query_vec,
            k: k as u32,
            target_confidence: 0.95,
            lambda: 0.5, // Decaying score
            tau: 100.0,
        });

        let start_q = Instant::now();
        let response = service.search(search_req).await?.into_inner();
        let q_lat = start_q.elapsed();

        let result_ids: Vec<u64> = response.results.iter().map(|r| r.id).collect();
        let recall = calculate_recall(&result_ids, &ground_truth);

        // Check Bucket Count (Adaptability Metric)
        let coll = manager.get_or_create(collection_name, None).await.unwrap();
        let bucket_count = coll.index.get_all_bucket_headers().len();

        println!(
            "Tick {:02}: Recall: {:>6.2}% | Buckets: {:>2} | Latency: {:>2}ms | Total: {}",
            b,
            recall * 100.0,
            bucket_count,
            q_lat.as_millis(),
            shadow_db.len()
        );

        if recall < 0.8 {
            println!("   ⚠️  Recall Drop Detected! Index might be lagging behind drift.");
        }
    }

    println!("\n✅ Simulation Complete.");
    Ok(())
}
