use clap::Parser;
use drift_server::config::{Config, FileConfig, StorageCommand};
use drift_server::drift_proto::drift_server::Drift;
use drift_server::drift_proto::{InsertBatchRequest, SearchRequest, Vector};
use drift_server::manager::CollectionManager;
use drift_server::server::DriftService;
use rand::prelude::*;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
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
    #[arg(long, default_value_t = false)]
    freeze_janitor: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let orig_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        orig_hook(panic_info);
        std::process::exit(1);
    }));

    tracing_subscriber::fmt()
        .with_env_filter("drift_server=info,drift_storage=info,drift_core=info")
        .init();

    let args = Args::parse();
    let dir = tempdir()?;
    let root = dir.path();
    let storage_path = root.join("storage");
    std::fs::create_dir_all(&storage_path)?;

    let optimal_capacity = 2000;

    let config = Config {
        port: 50099,
        wal_dir: root.join("wal"),
        data_dir: root.join("data"),
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

    // --- PLANTING ---
    println!("\n🧵 Planting Needles...");
    let mut rng = StdRng::seed_from_u64(42);
    let mut queries = Vec::new();
    let mut needles = Vec::new();

    for q_idx in 0..10 {
        let query_vec: Vec<f32> = (0..DIM).map(|_| rng.random::<f32>() * 100.0).collect();
        queries.push(query_vec.clone());
        for k in 0..10 {
            let mut needle = query_vec.clone();
            for val in needle.iter_mut() {
                *val += (rng.random::<f32>() - 0.5) * 0.01;
            }
            needles.push(Vector {
                id: (q_idx * 10 + k) as u64,
                values: needle,
            });
        }
    }
    let needle_ids: Vec<u64> = needles.iter().map(|v| v.id).collect();
    let needle_set: HashSet<u64> = needle_ids.iter().copied().collect();
    service
        .insert_batch(Request::new(InsertBatchRequest {
            collection_name: collection.to_string(),
            vectors: needles.clone(),
        }))
        .await?;

    // --- FILLING ---
    println!("\n🚜 Filling Haystack...");
    let start_fill = Instant::now();
    let mut inserted = 0;
    let mut next_id = 1000;

    let coll = manager
        .get_or_create(collection, Some(DIM), None)
        .await
        .unwrap();
    let mut batch_vecs = vec![vec![0.0; DIM]; BATCH_SIZE];

    while inserted < args.count {
        while coll.index.memtable_len() > 500_000 {
            let n = coll.index.memtable_len();
            println!("   ✋ Backpressure: MemTable full. Waiting... {}", n);
            sleep(Duration::from_millis(500)).await;
        }

        let mut batch_proto = Vec::with_capacity(BATCH_SIZE);
        for i in 0..BATCH_SIZE {
            let vec = &mut batch_vecs[i];
            for val in vec.iter_mut() {
                *val = rng.random::<f32>() * 100.0;
            }
            batch_proto.push(Vector {
                id: next_id,
                values: vec.clone(),
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

    // --- DRAINING ---
    println!("\n⏳ Final Drain...");
    while coll.index.memtable_len() > 0 {
        sleep(Duration::from_secs(2)).await;
        println!("   Draining... ({} remaining)", coll.index.memtable_len());
    }
    loop {
        let frozen_count = coll.index.get_frozen_count();
        if frozen_count == 0 {
            break;
        }
        println!(
            "   Waiting for active flush ({} frozen tables)...",
            frozen_count
        );
        sleep(Duration::from_secs(2)).await;
    }
    sleep(Duration::from_secs(2)).await;
    println!("✅ Index Ready.");

    if args.freeze_janitor {
        coll.janitor_task.abort();
        println!("🧊 Janitor frozen before tracing/recall.");
        sleep(Duration::from_secs(1)).await;
    }

    // --- NEEDLE TRACE ---
    println!("\n🧭 Needle Trace...");
    println!("   Janitor running: {}", !coll.janitor_task.is_finished());

    let l0_found = coll.index.debug_find_needles_in_l0(&needle_set);
    println!("   L0 needles: {}/{}", l0_found.len(), needle_set.len());

    let (_router_centroids, router_ids, router_counts) =
        coll.index.get_router().read().get_snapshot_with_counts();
    println!(
        "   Router snapshot: buckets={}, dim={}",
        router_ids.len(),
        DIM
    );
    if !router_counts.is_empty() {
        let mut min_count = u32::MAX;
        let mut max_count = 0u32;
        for c in &router_counts {
            min_count = min_count.min(*c);
            max_count = max_count.max(*c);
        }
        println!("   Router counts: min={} max={}", min_count, max_count);
    }

    let mut id_to_router_idx: HashMap<u32, usize> = HashMap::new();
    for (idx, id) in router_ids.iter().enumerate() {
        id_to_router_idx.insert(*id, idx);
    }

    let mut needle_locations: HashMap<u64, u32> = HashMap::new();
    let mut buckets_with_needles: HashMap<u32, usize> = HashMap::new();

    for bucket_id in router_ids.iter().copied() {
        match coll.index.debug_fetch_bucket_ids(bucket_id).await {
            Ok(ids) => {
                let mut count = 0usize;
                for id in ids {
                    if needle_set.contains(&id) {
                        needle_locations.entry(id).or_insert(bucket_id);
                        count += 1;
                    }
                }
                if count > 0 {
                    buckets_with_needles.insert(bucket_id, count);
                }
                if needle_locations.len() == needle_set.len() {
                    break;
                }
            }
            Err(e) => {
                println!("   ⚠️ bucket {} fetch failed: {}", bucket_id, e);
            }
        }
    }

    let mut needle_buckets: Vec<u32> = buckets_with_needles.keys().copied().collect();
    needle_buckets.sort_unstable();
    println!(
        "   Buckets with needles: {}",
        short_list_u32(&needle_buckets, 40)
    );

    let mut missing_needles: Vec<u64> = needle_ids
        .iter()
        .copied()
        .filter(|id| !needle_locations.contains_key(id))
        .collect();
    if !missing_needles.is_empty() {
        missing_needles.sort_unstable();
        println!(
            "   Missing needles on disk: {}",
            short_list_u64(&missing_needles, 40)
        );
    }

    // --- MEASURING ---
    println!("\n🔍 Measuring Recall...");
    let mut total_hits = 0;
    let mut brute_cache: HashMap<u32, (Vec<u64>, Vec<f32>)> = HashMap::new();

    for (i, query) in queries.iter().enumerate() {
        let (selected, q_centroids, q_ids, q_counts) = {
            let router = coll.index.get_router().read();
            let selected = router.select_buckets(query, 0.99, 0.05, 20.0);
            let (centroids, ids, counts) = router.get_snapshot_with_counts();
            (selected, centroids, ids, counts)
        };
        let mut selected_sorted = selected.clone();
        selected_sorted.sort_unstable();
        let selected_set: HashSet<u32> = selected_sorted.iter().copied().collect();

        let expected_start = (i * 10) as u64;
        let expected_ids: Vec<u64> = (expected_start..expected_start + 10).collect();
        let mut expected_buckets_set: HashSet<u32> = HashSet::new();
        for id in &expected_ids {
            if let Some(b) = needle_locations.get(id) {
                expected_buckets_set.insert(*b);
            }
        }
        let mut expected_buckets: Vec<u32> = expected_buckets_set.into_iter().collect();
        expected_buckets.sort_unstable();
        let overlap: Vec<u32> = expected_buckets
            .iter()
            .copied()
            .filter(|b| selected_set.contains(b))
            .collect();

        let mut q_id_to_idx: HashMap<u32, usize> = HashMap::new();
        for (idx, id) in q_ids.iter().enumerate() {
            q_id_to_idx.insert(*id, idx);
        }

        if !expected_buckets.is_empty() {
            let mut bucket_details = Vec::new();
            for b in &expected_buckets {
                if let Some(idx) = q_id_to_idx.get(b) {
                    let start = idx * DIM;
                    let end = start + DIM;
                    let centroid = &q_centroids[start..end];
                    let dist = l2_sq(query, centroid).sqrt();
                    let count = q_counts[*idx];
                    let p_geom = (-0.05 * dist).exp();
                    let reliability = 1.0 - (-(count as f32) / 20.0).exp();
                    let score = p_geom * reliability;
                    bucket_details.push(format!(
                        "{}(cnt={},dist={:.3},p={:.4},r={:.4},score={:.4})",
                        b, count, dist, p_geom, reliability, score
                    ));
                } else {
                    bucket_details.push(format!("{}(missing)", b));
                }
            }
            println!(
                "   Query #{} expected bucket details: {}",
                i,
                short_list_str(&bucket_details, 8)
            );
        }

        if !selected_sorted.is_empty() {
            let mut selected_details = Vec::new();
            for b in selected_sorted.iter().take(8) {
                if let Some(idx) = q_id_to_idx.get(b) {
                    let start = idx * DIM;
                    let end = start + DIM;
                    let centroid = &q_centroids[start..end];
                    let dist = l2_sq(query, centroid).sqrt();
                    let count = q_counts[*idx];
                    let p_geom = (-0.05 * dist).exp();
                    let reliability = 1.0 - (-(count as f32) / 20.0).exp();
                    let score = p_geom * reliability;
                    selected_details.push(format!(
                        "{}(cnt={},dist={:.3},p={:.4},r={:.4},score={:.4})",
                        b, count, dist, p_geom, reliability, score
                    ));
                }
            }
            println!(
                "   Query #{} selected bucket details: {}",
                i,
                short_list_str(&selected_details, 8)
            );
        }
        println!(
            "   Query #{} selected buckets: {} | expected buckets: {} | overlap: {}",
            i,
            short_list_u32(&selected_sorted, 20),
            short_list_u32(&expected_buckets, 20),
            short_list_u32(&overlap, 20)
        );

        if !expected_buckets.is_empty() {
            for b in &expected_buckets {
                if !brute_cache.contains_key(b) {
                    match coll.index.debug_fetch_bucket(*b).await {
                        Ok(res) => {
                            brute_cache.insert(*b, res);
                        }
                        Err(e) => {
                            println!("   Query #{} brute fetch failed for bucket {}: {}", i, b, e);
                            continue;
                        }
                    }
                }

                let cached = brute_cache.get(b).unwrap();
                let topk = brute_force_topk(&cached.0, &cached.1, DIM, query, 10);
                let brute_hits = topk
                    .iter()
                    .filter(|(id, _)| *id >= expected_start && *id < expected_start + 10)
                    .count();
                println!(
                    "   Query #{} brute force bucket {} hits {}/10",
                    i, b, brute_hits
                );
            }
        }

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

    // ⚡ CLEANUP: Gracefully stop the Janitor before TempDir is dropped
    coll.janitor_task.abort();
    println!("🧹 Janitor stopped.");

    if recall < 95.0 {
        panic!("❌ Failed Recall Target!");
    }

    Ok(())
}

fn short_list_u32(list: &[u32], max: usize) -> String {
    if list.is_empty() {
        return "[]".to_string();
    }
    let mut out = Vec::new();
    for item in list.iter().take(max) {
        out.push(item.to_string());
    }
    if list.len() > max {
        out.push("...".to_string());
    }
    format!("[{}]", out.join(", "))
}

fn short_list_u64(list: &[u64], max: usize) -> String {
    if list.is_empty() {
        return "[]".to_string();
    }
    let mut out = Vec::new();
    for item in list.iter().take(max) {
        out.push(item.to_string());
    }
    if list.len() > max {
        out.push("...".to_string());
    }
    format!("[{}]", out.join(", "))
}

fn short_list_str(list: &[String], max: usize) -> String {
    if list.is_empty() {
        return "[]".to_string();
    }
    let mut out = Vec::new();
    for item in list.iter().take(max) {
        out.push(item.clone());
    }
    if list.len() > max {
        out.push("...".to_string());
    }
    format!("[{}]", out.join(", "))
}

fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum
}

fn brute_force_topk(
    ids: &[u64],
    flat: &[f32],
    dim: usize,
    query: &[f32],
    k: usize,
) -> Vec<(u64, f32)> {
    if ids.is_empty() || flat.is_empty() {
        return Vec::new();
    }
    let mut dists: Vec<(f32, u64)> = Vec::with_capacity(ids.len());
    for (i, id) in ids.iter().enumerate() {
        let start = i * dim;
        if start + dim > flat.len() {
            break;
        }
        let dist = l2_sq(query, &flat[start..start + dim]);
        dists.push((dist, *id));
    }

    if dists.len() > k {
        let nth = k.min(dists.len() - 1);
        dists.select_nth_unstable_by(nth, |a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        dists.truncate(k);
    }
    dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    dists.into_iter().map(|(d, id)| (id, d)).collect()
}
