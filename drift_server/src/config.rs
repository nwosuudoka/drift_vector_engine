use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
pub struct Config {
    /// Port to listen on
    #[arg(long, env = "DRIFT_PORT", default_value_t = 50051)]
    pub port: u16,

    /// Storage Backend URI (e.g., "file:///data" or "s3://my-bucket/data")
    #[arg(long, env = "DRIFT_STORAGE_URI", default_value = "file://./data")]
    pub storage_uri: String,

    /// WAL Directory (for local durability)
    #[arg(long, env = "DRIFT_WAL_DIR", default_value = "./data/wal")]
    pub wal_dir: PathBuf,

    /// Default Vector Dimension (if not inferred)
    #[arg(long, env = "DRIFT_DEFAULT_DIM", default_value_t = 128)]
    pub default_dim: usize,

    /// Max vectors per memory bucket before flush
    #[arg(long, env = "DRIFT_MAX_BUCKET_CAPACITY", default_value_t = 1000)]
    pub max_bucket_capacity: usize,

    /// HNSW Construction Depth (Higher = Better Graph, Slower Insert)
    #[arg(long, env = "DRIFT_EF_CONSTRUCTION", default_value_t = 128)]
    pub ef_construction: usize,

    /// HNSW Search Depth (Higher = Better Recall, Slower Search)
#[arg(long, env = "DRIFT_EF_SEARCH", default_value_t = 50)]
    pub ef_search: usize,
}
