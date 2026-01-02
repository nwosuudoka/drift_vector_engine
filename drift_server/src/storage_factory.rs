use crate::config::{FileConfig, S3Config, StorageCommand};
use opendal::{Operator, services};
use std::io;

pub struct StorageFactory;

impl StorageFactory {
    /// Builds an authenticated Operator rooted at the specific scope (collection).
    pub fn build(storage_config: &StorageCommand, scope: &str) -> io::Result<Operator> {
        match storage_config {
            StorageCommand::File(config) => Self::build_fs(config, scope),
            StorageCommand::S3(config) => Self::build_s3(config, scope),
        }
    }

    fn build_fs(config: &FileConfig, scope: &str) -> io::Result<Operator> {
        let mut builder = services::Fs::default();

        // Isolation: Root = base_path / collection_name
        // e.g., ./data/my_collection
        let root = config.path.join(scope);
        // Ensure local directory exists immediately
        std::fs::create_dir_all(&root)?;

        let op = Operator::new(builder.root(&root.to_string_lossy()))
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?
            .finish();
        Ok(op)
    }

    fn build_s3(config: &S3Config, scope: &str) -> io::Result<Operator> {
        let mut builder = services::S3::default()
            .bucket(&config.bucket)
            .region(&config.region)
            .root(scope);

        // Isolation: Root = collection_name
        // S3 keys will look like: s3://my-bucket/my_collection/segment_123.drift
        // OpenDAL handles the prefixing automatically.

        if let Some(endpoint) = &config.endpoint {
            builder = builder.endpoint(endpoint);
        }
        if let Some(key) = &config.access_key {
            builder = builder.access_key_id(key);
        }
        if let Some(secret) = &config.secret_key {
            builder = builder.secret_access_key(secret);
        }

        let op = Operator::new(builder)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?
            .finish();
        Ok(op)
    }
}
