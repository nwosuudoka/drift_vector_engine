use clap::Parser;
use drift_core::math::Metric;
use drift_server::config::{Config, FileConfig, StorageCommand};
use drift_server::drift_proto::drift_server::Drift;
use drift_server::drift_proto::{
    FieldFilter, InsertBatchRequest, PayloadRow, PayloadValue, SearchRequest, Vector,
};
use drift_server::filter_planner_diagnostics::{
    FILTER_PLANNER_DIAGNOSTICS_ENV, diagnostics_enabled_from_env,
};
use drift_server::manager::CollectionManager;
use drift_server::server::DriftService;
use rand::prelude::*;
use serde::Serialize;
use std::collections::HashMap;
use std::path::PathBuf;
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

fn duration_ms(value: Duration) -> f64 {
    value.as_secs_f64() * 1_000.0
}

fn payload_keyword(value: impl Into<String>) -> PayloadValue {
    PayloadValue {
        kind: Some(drift_server::drift_proto::payload_value::Kind::KeywordValue(value.into())),
    }
}

fn payload_int64(value: i64) -> PayloadValue {
    PayloadValue {
        kind: Some(drift_server::drift_proto::payload_value::Kind::Int64Value(
            value,
        )),
    }
}

fn payload_row(entries: Vec<(u32, PayloadValue)>) -> PayloadRow {
    PayloadRow {
        fields: entries.into_iter().collect::<HashMap<_, _>>(),
    }
}

fn tenant_filter(field_id: u32, tenant_idx: usize) -> FieldFilter {
    FieldFilter {
        field_id,
        condition: Some(drift_server::drift_proto::field_filter::Condition::Exact(
            payload_keyword(format!("tenant_{tenant_idx}")),
        )),
    }
}

const CI_SMALL_TIER_MAX_VECTORS: usize = 10_000;
const CI_MEDIUM_TIER_MAX_VECTORS: usize = 50_000;
const CI_SMALL_MAX_FILTERED_P95_MS: f64 = 12.0;
const CI_MEDIUM_MAX_FILTERED_P95_MS: f64 = 35.0;
const CI_LARGE_MAX_FILTERED_P95_MS: f64 = 90.0;
const CI_SMALL_MAX_FILTERED_OVERHEAD_RATIO: f64 = 7.0;
const CI_MEDIUM_MAX_FILTERED_OVERHEAD_RATIO: f64 = 7.0;
const CI_LARGE_MAX_FILTERED_OVERHEAD_RATIO: f64 = 10.0;

#[derive(Debug, Clone, Copy)]
struct CiFilteredGuardrailDefaults {
    tier_label: &'static str,
    max_filtered_p95_ms: f64,
    max_filtered_overhead_ratio: f64,
}

fn ci_filtered_guardrail_defaults(total_vectors: usize) -> CiFilteredGuardrailDefaults {
    if total_vectors <= CI_SMALL_TIER_MAX_VECTORS {
        return CiFilteredGuardrailDefaults {
            tier_label: "small",
            max_filtered_p95_ms: CI_SMALL_MAX_FILTERED_P95_MS,
            max_filtered_overhead_ratio: CI_SMALL_MAX_FILTERED_OVERHEAD_RATIO,
        };
    }
    if total_vectors <= CI_MEDIUM_TIER_MAX_VECTORS {
        return CiFilteredGuardrailDefaults {
            tier_label: "medium",
            max_filtered_p95_ms: CI_MEDIUM_MAX_FILTERED_P95_MS,
            max_filtered_overhead_ratio: CI_MEDIUM_MAX_FILTERED_OVERHEAD_RATIO,
        };
    }
    CiFilteredGuardrailDefaults {
        tier_label: "large",
        max_filtered_p95_ms: CI_LARGE_MAX_FILTERED_P95_MS,
        max_filtered_overhead_ratio: CI_LARGE_MAX_FILTERED_OVERHEAD_RATIO,
    }
}

#[derive(Debug, Serialize)]
struct BenchSummary {
    dim: usize,
    total_vectors: usize,
    batch_size: usize,
    query_count: usize,
    filtered_query_count: usize,
    write_throughput_vec_per_sec: f64,
    write_batch_p95_ms: f64,
    unfiltered_qps: f64,
    unfiltered_p95_ms: f64,
    filtered_qps: Option<f64>,
    filtered_p95_ms: Option<f64>,
    filtered_avg_hits: Option<f64>,
    filtered_overhead_ratio: Option<f64>,
    filtered_candidate_fanout: Option<f64>,
    filtered_estimated_scanned_ids_avg: Option<f64>,
    filtered_estimated_scan_ratio: Option<f64>,
    filtered_planner_produced_bucket_ratio: Option<f64>,
    filtered_planner_applied_bucket_ratio: Option<f64>,
    filtered_planner_gated_bucket_ratio: Option<f64>,
    filtered_planner_probe_error_bucket_ratio: Option<f64>,
    filtered_planner_empty_exact_bucket_ratio: Option<f64>,
    filtered_planner_no_index_bucket_ratio: Option<f64>,
    filtered_planner_range_stats_only_bucket_ratio: Option<f64>,
    filtered_planner_other_absence_bucket_ratio: Option<f64>,
    filtered_planner_diagnostics_enabled: Option<bool>,
    ci_guardrail_tier: Option<String>,
    effective_max_filtered_p95_ms: Option<f64>,
    effective_max_filtered_overhead_ratio: Option<f64>,
}

struct JanitorAbortGuard {
    collection: Arc<drift_server::manager::Collection>,
}

impl JanitorAbortGuard {
    fn new(collection: Arc<drift_server::manager::Collection>) -> Self {
        Self { collection }
    }
}

impl Drop for JanitorAbortGuard {
    fn drop(&mut self) {
        self.collection.janitor_task.abort();
    }
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
    #[arg(long, default_value_t = 20)]
    filtered_warmup_queries: usize,
    #[arg(long, default_value_t = 200)]
    filtered_query_count: usize,
    #[arg(long, default_value_t = 64)]
    filter_cardinality: usize,
    #[arg(long, default_value_t = false)]
    filtered_projection: bool,
    #[arg(long)]
    max_unfiltered_p95_ms: Option<f64>,
    #[arg(long)]
    max_filtered_p95_ms: Option<f64>,
    #[arg(long)]
    max_filtered_overhead_ratio: Option<f64>,
    #[arg(long)]
    summary_json_path: Option<PathBuf>,
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
    let filter_cardinality = args.filter_cardinality.max(1);
    let ci_mode = std::env::var_os("CI").is_some();
    let ci_defaults = (ci_mode && args.filtered_query_count > 0)
        .then_some(ci_filtered_guardrail_defaults(args.total_vectors));
    let effective_max_filtered_p95_ms = args
        .max_filtered_p95_ms
        .or_else(|| ci_defaults.map(|defaults| defaults.max_filtered_p95_ms));
    let effective_max_filtered_overhead_ratio = args
        .max_filtered_overhead_ratio
        .or_else(|| ci_defaults.map(|defaults| defaults.max_filtered_overhead_ratio));
    let planner_diagnostics_enabled = diagnostics_enabled_from_env();
    const TENANT_FIELD_ID: u32 = 1;
    const PRICE_FIELD_ID: u32 = 2;

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
    let coll = manager
        .get_or_create(
            collection,
            Some(args.dim),
            Some(args.max_bucket_capacity),
            Some(Metric::L2),
        )
        .await?;
    let _janitor_guard = JanitorAbortGuard::new(coll.clone());

    println!("🏁 Bench RW (v3)");
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
    println!(
        "   • Filter workload: queries={}, tenant_cardinality={}, projection={}",
        args.filtered_query_count, filter_cardinality, args.filtered_projection
    );
    if ci_mode {
        if args.max_filtered_p95_ms.is_none() && args.filtered_query_count > 0 {
            let defaults =
                ci_defaults.expect("ci defaults should be present when filtered workload runs");
            println!(
                "   • CI guardrail defaults (tier={}): max_filtered_p95_ms={:.1}, max_filtered_overhead_ratio={:.2}",
                defaults.tier_label,
                defaults.max_filtered_p95_ms,
                defaults.max_filtered_overhead_ratio
            );
        }
    }

    let mut rng = StdRng::seed_from_u64(args.seed);
    let mut tenant_query_vectors: Vec<Option<Vec<f32>>> = vec![None; filter_cardinality];

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
        let mut payload_rows = Vec::with_capacity(batch_count);

        for _ in 0..batch_count {
            let id = next_id;
            let vector = gen_random_vector(&mut rng, args.dim);
            let tenant_idx = (id as usize) % filter_cardinality;
            if tenant_query_vectors[tenant_idx].is_none() {
                tenant_query_vectors[tenant_idx] = Some(vector.clone());
            }

            batch_vecs.push(Vector { id, values: vector });
            payload_rows.push(payload_row(vec![
                (
                    TENANT_FIELD_ID,
                    payload_keyword(format!("tenant_{tenant_idx}")),
                ),
                (PRICE_FIELD_ID, payload_int64(id as i64)),
            ]));
            next_id += 1;
        }

        let batch_start = Instant::now();
        service
            .insert_batch(Request::new(InsertBatchRequest {
                collection_name: collection.to_string(),
                vectors: batch_vecs,
                payload_rows,
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
                filters: vec![],
                payload_projection_fields: vec![],
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
                filters: vec![],
                payload_projection_fields: vec![],
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

    // -------------------------------
    // FILTERED READ BENCH
    // -------------------------------
    let (
        filtered_qps,
        filtered_p95,
        filtered_avg_hits,
        filtered_overhead_ratio,
        filtered_candidate_fanout,
        filtered_estimated_scanned_ids_avg,
        filtered_estimated_scan_ratio,
        filtered_planner_produced_bucket_ratio,
        filtered_planner_applied_bucket_ratio,
        filtered_planner_gated_bucket_ratio,
        filtered_planner_probe_error_bucket_ratio,
        filtered_planner_empty_exact_bucket_ratio,
        filtered_planner_no_index_bucket_ratio,
        filtered_planner_range_stats_only_bucket_ratio,
        filtered_planner_other_absence_bucket_ratio,
        filtered_planner_diagnostics_enabled,
    ) = if args.filtered_query_count == 0 {
        (
            None, None, None, None, None, None, None, None, None, None, None, None, None, None,
            None, None,
        )
    } else {
        println!("\n🧪 Filtered Read Benchmark...");

        for i in 0..args.filtered_warmup_queries {
            let tenant_idx = i % filter_cardinality;
            let q = tenant_query_vectors[tenant_idx]
                .clone()
                .unwrap_or_else(|| gen_random_vector(&mut rng, args.dim));
            let _ = service
                .search(Request::new(SearchRequest {
                    collection_name: collection.to_string(),
                    vector: q,
                    k: args.k as u32,
                    target_confidence: args.target_confidence,
                    lambda: args.lambda,
                    tau: args.tau,
                    filters: vec![tenant_filter(TENANT_FIELD_ID, tenant_idx)],
                    payload_projection_fields: if args.filtered_projection {
                        vec![TENANT_FIELD_ID, PRICE_FIELD_ID]
                    } else {
                        vec![]
                    },
                }))
                .await?;
        }

        let mut filtered_latencies = Vec::with_capacity(args.filtered_query_count);
        let mut total_hits = 0usize;
        let mut total_candidate_ids = 0usize;
        let mut total_scanned_ids = 0usize;
        let mut total_live_bucket_ids = 0usize;
        let mut total_planner_probed_buckets = 0usize;
        let mut total_planner_produced_buckets = 0usize;
        let mut total_planner_applied_buckets = 0usize;
        let mut total_planner_gated_buckets = 0usize;
        let mut total_planner_probe_error_buckets = 0usize;
        let mut total_planner_empty_exact_buckets = 0usize;
        let mut total_planner_no_index_buckets = 0usize;
        let mut total_planner_range_stats_only_buckets = 0usize;
        let mut total_planner_other_absence_buckets = 0usize;
        let start_filtered = Instant::now();
        for i in 0..args.filtered_query_count {
            let tenant_idx = i % filter_cardinality;
            let q = tenant_query_vectors[tenant_idx]
                .clone()
                .unwrap_or_else(|| gen_random_vector(&mut rng, args.dim));
            let q_start = Instant::now();
            let resp = service
                .search(Request::new(SearchRequest {
                    collection_name: collection.to_string(),
                    vector: q,
                    k: args.k as u32,
                    target_confidence: args.target_confidence,
                    lambda: args.lambda,
                    tau: args.tau,
                    filters: vec![tenant_filter(TENANT_FIELD_ID, tenant_idx)],
                    payload_projection_fields: if args.filtered_projection {
                        vec![TENANT_FIELD_ID, PRICE_FIELD_ID]
                    } else {
                        vec![]
                    },
                }))
                .await?
                .into_inner();
            filtered_latencies.push(q_start.elapsed());
            total_hits += resp.results.len();
            let hint_stats = coll.index.last_search_hint_stats();
            total_candidate_ids += hint_stats.candidate_id_count;
            total_scanned_ids += hint_stats.estimated_scanned_ids;
            total_live_bucket_ids += hint_stats.estimated_total_bucket_ids;
            if planner_diagnostics_enabled {
                let planner_diag = *coll.last_filter_planner_diagnostics.read();
                total_planner_probed_buckets += planner_diag.probed_bucket_count;
                total_planner_produced_buckets += planner_diag.candidate_produced_bucket_count;
                total_planner_applied_buckets += planner_diag.candidate_applied_bucket_count;
                total_planner_gated_buckets +=
                    planner_diag.candidate_gated_broad_selectivity_bucket_count;
                total_planner_probe_error_buckets +=
                    planner_diag.candidate_disabled_probe_error_bucket_count;
                total_planner_empty_exact_buckets +=
                    planner_diag.candidate_empty_exact_match_bucket_count;
                total_planner_no_index_buckets +=
                    planner_diag.candidate_no_indexed_exact_bucket_count;
                total_planner_range_stats_only_buckets +=
                    planner_diag.candidate_range_stats_only_bucket_count;
                total_planner_other_absence_buckets +=
                    planner_diag.candidate_other_absence_bucket_count;
            }
        }
        let filtered_duration = start_filtered.elapsed();
        let qps = args.filtered_query_count as f64 / filtered_duration.as_secs_f64();
        filtered_latencies.sort();
        let f_p50 = percentile(&filtered_latencies, 0.50);
        let f_p95 = percentile(&filtered_latencies, 0.95);
        let f_p99 = percentile(&filtered_latencies, 0.99);
        let avg_hits = total_hits as f64 / args.filtered_query_count as f64;
        let overhead = duration_ms(f_p95) / duration_ms(r_p95).max(1e-9);
        let candidate_fanout = if total_live_bucket_ids > 0 {
            total_candidate_ids as f64 / total_live_bucket_ids as f64
        } else {
            0.0
        };
        let estimated_scan_ratio = if total_live_bucket_ids > 0 {
            total_scanned_ids as f64 / total_live_bucket_ids as f64
        } else {
            0.0
        };
        let avg_scanned_ids = total_scanned_ids as f64 / args.filtered_query_count as f64;

        println!("✅ Filtered Read Complete in {:.2?}", filtered_duration);
        println!("   • Filtered QPS: {:.0} q/s", qps);
        println!(
            "   • Filtered Latency p50/p95/p99: {:.2?} / {:.2?} / {:.2?}",
            f_p50, f_p95, f_p99
        );
        println!("   • Average result count: {:.2}", avg_hits);
        println!("   • Filtered p95 / unfiltered p95: {:.2}x", overhead);
        println!(
            "   • Candidate fanout (candidate/live): {:.3}",
            candidate_fanout
        );
        println!(
            "   • Estimated scanned IDs/query: {:.1} ({:.3} of live IDs)",
            avg_scanned_ids, estimated_scan_ratio
        );

        let (
            planner_produced_bucket_ratio,
            planner_applied_bucket_ratio,
            planner_gated_bucket_ratio,
            planner_probe_error_bucket_ratio,
            planner_empty_exact_bucket_ratio,
            planner_no_index_bucket_ratio,
            planner_range_stats_only_bucket_ratio,
            planner_other_absence_bucket_ratio,
        ) = if planner_diagnostics_enabled {
            let planner_ratio = |value: usize, total: usize| {
                if total > 0 {
                    value as f64 / total as f64
                } else {
                    0.0
                }
            };
            let produced_ratio =
                planner_ratio(total_planner_produced_buckets, total_planner_probed_buckets);
            let applied_ratio =
                planner_ratio(total_planner_applied_buckets, total_planner_probed_buckets);
            let gated_ratio =
                planner_ratio(total_planner_gated_buckets, total_planner_probed_buckets);
            let probe_error_ratio = planner_ratio(
                total_planner_probe_error_buckets,
                total_planner_probed_buckets,
            );
            let empty_exact_ratio = planner_ratio(
                total_planner_empty_exact_buckets,
                total_planner_probed_buckets,
            );
            let no_index_ratio =
                planner_ratio(total_planner_no_index_buckets, total_planner_probed_buckets);
            let range_stats_only_ratio = planner_ratio(
                total_planner_range_stats_only_buckets,
                total_planner_probed_buckets,
            );
            let other_absence_ratio = planner_ratio(
                total_planner_other_absence_buckets,
                total_planner_probed_buckets,
            );

            println!(
                "   • Planner diagnostics (per probed bucket): produced={:.3}, applied={:.3}, gated={:.3}, probe_error={:.3}",
                produced_ratio, applied_ratio, gated_ratio, probe_error_ratio
            );
            println!(
                "   • Planner absence reasons: empty_exact={:.3}, no_index={:.3}, range_stats_only={:.3}, other={:.3}",
                empty_exact_ratio, no_index_ratio, range_stats_only_ratio, other_absence_ratio
            );
            (
                Some(produced_ratio),
                Some(applied_ratio),
                Some(gated_ratio),
                Some(probe_error_ratio),
                Some(empty_exact_ratio),
                Some(no_index_ratio),
                Some(range_stats_only_ratio),
                Some(other_absence_ratio),
            )
        } else {
            println!(
                "   • Planner diagnostics disabled (set {}=1 to enable)",
                FILTER_PLANNER_DIAGNOSTICS_ENV
            );
            (None, None, None, None, None, None, None, None)
        };

        (
            Some(qps),
            Some(f_p95),
            Some(avg_hits),
            Some(overhead),
            Some(candidate_fanout),
            Some(avg_scanned_ids),
            Some(estimated_scan_ratio),
            planner_produced_bucket_ratio,
            planner_applied_bucket_ratio,
            planner_gated_bucket_ratio,
            planner_probe_error_bucket_ratio,
            planner_empty_exact_bucket_ratio,
            planner_no_index_bucket_ratio,
            planner_range_stats_only_bucket_ratio,
            planner_other_absence_bucket_ratio,
            Some(planner_diagnostics_enabled),
        )
    };

    // -------------------------------
    // GUARDRAILS
    // -------------------------------
    let mut guardrail_failures = Vec::new();
    let unfiltered_p95_ms = duration_ms(r_p95);
    if let Some(limit_ms) = args.max_unfiltered_p95_ms
        && unfiltered_p95_ms > limit_ms
    {
        guardrail_failures.push(format!(
            "unfiltered p95 {:.2}ms exceeds limit {:.2}ms",
            unfiltered_p95_ms, limit_ms
        ));
    }

    let filtered_p95_ms = filtered_p95.map(duration_ms);
    if let Some(limit_ms) = effective_max_filtered_p95_ms {
        match filtered_p95_ms {
            Some(value) if value > limit_ms => guardrail_failures.push(format!(
                "filtered p95 {:.2}ms exceeds limit {:.2}ms",
                value, limit_ms
            )),
            Some(_) => {}
            None => guardrail_failures.push(
                "max-filtered-p95-ms configured but filtered workload did not run".to_string(),
            ),
        }
    }

    if let Some(limit_ratio) = effective_max_filtered_overhead_ratio {
        match filtered_overhead_ratio {
            Some(value) if value > limit_ratio => guardrail_failures.push(format!(
                "filtered/unfiltered p95 ratio {:.2}x exceeds limit {:.2}x",
                value, limit_ratio
            )),
            Some(_) => {}
            None => guardrail_failures.push(
                "max-filtered-overhead-ratio configured but filtered workload did not run"
                    .to_string(),
            ),
        }
    }

    if !guardrail_failures.is_empty() {
        eprintln!("\n❌ Guardrail failure(s):");
        for failure in &guardrail_failures {
            eprintln!("   • {failure}");
        }
        return Err(format!("benchmark guardrails failed ({})", guardrail_failures.len()).into());
    }

    // Optional summary export for tracking/pipelines.
    if let Some(path) = args.summary_json_path.as_ref() {
        let summary = BenchSummary {
            dim: args.dim,
            total_vectors: args.total_vectors,
            batch_size: args.batch_size,
            query_count: args.query_count,
            filtered_query_count: args.filtered_query_count,
            write_throughput_vec_per_sec: write_throughput,
            write_batch_p95_ms: duration_ms(w_p95),
            unfiltered_qps: read_qps,
            unfiltered_p95_ms,
            filtered_qps,
            filtered_p95_ms,
            filtered_avg_hits,
            filtered_overhead_ratio,
            filtered_candidate_fanout,
            filtered_estimated_scanned_ids_avg,
            filtered_estimated_scan_ratio,
            filtered_planner_produced_bucket_ratio,
            filtered_planner_applied_bucket_ratio,
            filtered_planner_gated_bucket_ratio,
            filtered_planner_probe_error_bucket_ratio,
            filtered_planner_empty_exact_bucket_ratio,
            filtered_planner_no_index_bucket_ratio,
            filtered_planner_range_stats_only_bucket_ratio,
            filtered_planner_other_absence_bucket_ratio,
            filtered_planner_diagnostics_enabled,
            ci_guardrail_tier: ci_defaults.map(|defaults| defaults.tier_label.to_string()),
            effective_max_filtered_p95_ms,
            effective_max_filtered_overhead_ratio,
        };
        let json = serde_json::to_string_pretty(&summary)?;
        std::fs::write(path, json)?;
        println!("   • Summary JSON: {}", path.display());
    }

    println!("\n✅ Bench Complete.");
    Ok(())
}
