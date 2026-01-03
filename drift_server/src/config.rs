use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
pub struct Config {
    /// Port to listen on
    #[arg(long, env = "DRIFT_PORT", default_value_t = 50051)]
    pub port: u16,

    // /// Storage Backend URI (e.g., "file:///data" or "s3://my-bucket/data")
    // #[arg(long, env = "DRIFT_STORAGE_URI", default_value = "file://./data")]
    // pub storage_uri: String,
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

    #[command(subcommand)]
    pub storage: StorageCommand,
}

#[derive(Subcommand, Debug, Clone)]
pub enum StorageCommand {
    /// Use Local Filesystem
    File(FileConfig),

    /// Use AWS S3 (or compatible like MinIO)
    S3(S3Config),
}

#[derive(Args, Debug, Clone)]
pub struct FileConfig {
    /// Root directory for data storage
    #[arg(long, env = "DRIFT_DATA_DIR", default_value = "./data")]
    pub path: PathBuf,
}

#[derive(Args, Debug, Clone)]
pub struct S3Config {
    /// Bucket Name
    #[arg(long, env = "DRIFT_S3_BUCKET")]
    pub bucket: String,

    /// Region (e.g., us-east-1)
    #[arg(long, env = "DRIFT_S3_REGION", default_value = "us-east-1")]
    pub region: String,

    /// Custom Endpoint (for MinIO/LocalStack)
    #[arg(long, env = "DRIFT_S3_ENDPOINT")]
    pub endpoint: Option<String>,

    /// Access Key
    #[arg(long, env = "AWS_ACCESS_KEY_ID")]
    pub access_key: Option<String>,

    /// Secret Key
    #[arg(long, env = "AWS_SECRET_ACCESS_KEY")]
    pub secret_key: Option<String>,
}
