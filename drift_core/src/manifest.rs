use prost::Message;
// use std::collections::HashMap;
// use std::io::Cursor;

pub mod pb {
    include!(concat!(env!("OUT_DIR"), "/manifest.rs"));
}

use crate::math::Metric;
use pb::{Bucket, Centroid, Manifest};

#[derive(Debug, Clone)]
pub struct ManifestWrapper {
    pub inner: Manifest,
}

impl ManifestWrapper {
    fn default_object_path(id: u32, run_id: &str) -> String {
        if run_id.is_empty() {
            String::new()
        } else {
            format!("bucket_{}_{}.drift", id, run_id)
        }
    }

    pub fn new(dim: u32, metric: Metric) -> Self {
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

    /// Adds a new bucket to the registry.
    /// Updates BOTH the physical registry (Bucket list) and the routing table (Centroid list).
    pub fn add_bucket(&mut self, id: u32, run_id: String, centroid: Option<Vec<f32>>) {
        // 1. Update Physical Registry
        // Remove existing entry if present (Upsert)
        if let Some(pos) = self.inner.buckets.iter().position(|b| b.id == id) {
            self.inner.buckets.remove(pos);
        }

        let object_path = Self::default_object_path(id, &run_id);
        self.inner.buckets.push(pb::Bucket {
            id,
            run_id,
            vector_count: 0, // Reset count on new run? Or pass it in. Usually 0 start.
            tombstone_count: 0,
            radius: 0.0,
            object_path,
            object_fingerprint: String::new(),
        });

        // 2. Update Routing Table (if centroid provided)
        if let Some(vec) = centroid {
            // Remove existing centroid for this ID
            if let Some(pos) = self.inner.centroids.iter().position(|c| c.id == id) {
                self.inner.centroids.remove(pos);
            }

            self.inner.centroids.push(pb::Centroid { id, vector: vec });
        }
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

    pub fn metric(&self) -> Result<Metric, String> {
        Metric::from_manifest_str(&self.inner.metric)
    }

    pub fn update_bucket_run_id(&mut self, id: u32, new_run_id: String) {
        if let Some(b) = self.inner.buckets.iter_mut().find(|b| b.id == id) {
            b.run_id = new_run_id;
            b.object_path = Self::default_object_path(id, &b.run_id);
            b.object_fingerprint.clear();
        }
    }

    pub fn update_bucket_remote_meta(
        &mut self,
        id: u32,
        run_id: String,
        object_path: String,
        object_fingerprint: String,
    ) {
        if let Some(b) = self.inner.buckets.iter_mut().find(|b| b.id == id) {
            b.run_id = run_id;
            b.object_path = object_path;
            b.object_fingerprint = object_fingerprint;
        }
    }
}
