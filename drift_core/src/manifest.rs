use prost::Message;
// use std::collections::HashMap;
// use std::io::Cursor;

pub mod pb {
    include!(concat!(env!("OUT_DIR"), "/manifest.rs"));
}

use pb::{Bucket, Centroid, Manifest};

#[derive(Debug, Clone)]
pub struct ManifestWrapper {
    pub inner: Manifest,
}

impl ManifestWrapper {
    pub fn new(dim: u32, metric: &str) -> Self {
        Self {
            inner: Manifest {
                version: 1,
                created_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                dim,
                metric: metric.to_string(),
                centroids: vec![],
                buckets: vec![],
                tombstone_files: vec![],
            },
        }
    }

    /// Decodes from bytes (e.g. from S3 or Disk)
    pub fn from_bytes(data: &[u8]) -> Result<Self, prost::DecodeError> {
        let manifest = Manifest::decode(data)?;
        Ok(Self { inner: manifest })
    }

    /// Encodes ot bytes (for saving)
    pub fn to_bytes(&self) -> Vec<u8> {
        self.inner.encode_to_vec()
    }

    pub fn version(&self) -> u64 {
        self.inner.version
    }

    pub fn bump_version(&mut self) {
        self.inner.version += 1;
    }

    // --- State Mutation Methods (The "Brain" Logic) ---

    pub fn add_bucket(&mut self, id: u32, run_id: String, centroid: Vec<f32>) {
        // 1. Update Registry
        // Remove existing if replacing
        self.inner.buckets.retain(|b| b.id != id);
        self.inner.buckets.push(Bucket {
            id,
            run_id,
            vector_count: 0, // Initial count, update later
            tombstone_count: 0,
            radius: 0.0,
        });

        // 2. Update Router
        self.inner.centroids.retain(|c| c.id != id);
        self.inner.centroids.push(Centroid {
            id,
            vector: centroid,
        });
    }

    pub fn update_bucket_stats(&mut self, id: u32, count: u64, tombstones: u32) {
        if let Some(b) = self.inner.buckets.iter_mut().find(|b| b.id == id) {
            b.vector_count = count;
            b.tombstone_count = tombstones;
        }
    }

    pub fn remove_bucket(&mut self, id: u32) {
        self.inner.buckets.retain(|b| b.id != id);
        self.inner.centroids.retain(|c| c.id != id);
    }

    pub fn get_centroids(&self) -> &Vec<Centroid> {
        &self.inner.centroids
    }

    pub fn get_buckets(&self) -> &Vec<Bucket> {
        &self.inner.buckets
    }

    pub fn get_dim(&self) -> u32 {
        self.inner.dim
    }
}
