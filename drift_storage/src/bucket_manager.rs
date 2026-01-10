use crate::bucket_file_reader::BucketFileReader;
use async_trait::async_trait;
use drift_core::manifest::ManifestWrapper;
use drift_traits::{DiskSearcher, PageManager, TombstoneView};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

pub struct BucketManager {
    storage: Arc<dyn PageManager>,
    manifest: Arc<RwLock<ManifestWrapper>>,
}

impl BucketManager {
    pub fn new(storage: Arc<dyn PageManager>, manifest: Arc<RwLock<ManifestWrapper>>) -> Self {
        Self { storage, manifest }
    }
}

#[async_trait]
impl DiskSearcher for BucketManager {
    async fn search(
        &self,
        bucket_ids: &[u32],
        query: &[f32],
        k: usize,
        tv: &dyn TombstoneView,
    ) -> Vec<(u64, f32)> {
        let mut results = Vec::new();

        // 1. Resolve Targets (Snapshotting Manifest state)
        // Map BucketID -> RunID
        let targets: Vec<(u32, String)> = {
            let m = self.manifest.read().unwrap();
            bucket_ids
                .iter()
                .filter_map(|&bid| {
                    m.get_buckets()
                        .iter()
                        .find(|b| b.id == bid)
                        .map(|b| (bid, b.run_id.clone()))
                })
                .collect()
        };

        // 2. Parallel Scan (Scatter)
        // For V2 MVP we iterate sequentially async.
        // In V3 we use streams/futures::join_all.
        for (bucket_id, run_id) in targets {
            // A. Register File
            // Map the BucketID (file_id) to the physical path.
            // This tells PageManager: "When asked for File {bucket_id}, go to {path}"
            let path = PathBuf::from(format!("segment_{}.drift", run_id));
            self.storage.register_file(bucket_id, path);

            // B. Create Reader & Scan
            let mut reader = BucketFileReader::new(self.storage.clone(), bucket_id);

            // We pass k just for info, but the scanner returns all candidates for now
            match reader.scan(query, k, tv).await {
                Ok(candidates) => results.extend(candidates),
                Err(e) => {
                    tracing::error!(
                        "Failed to scan bucket {} (Run {}): {}",
                        bucket_id,
                        run_id,
                        e
                    );
                }
            }
        }

        // 3. Gather & Sort
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        if results.len() > k {
            results.truncate(k);
        }

        results
    }
}

use drift_traits::DataProvider;

#[async_trait]
impl DataProvider for BucketManager {
    async fn fetch_bucket(&self, bucket_id: u32) -> std::io::Result<(Vec<u64>, Vec<f32>)> {
        // 1. Resolve RunID from Manifest
        let run_id = {
            let m = self.manifest.read().unwrap();
            m.get_buckets()
                .iter()
                .find(|b| b.id == bucket_id)
                .map(|b| b.run_id.clone())
                .ok_or(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "Bucket not in manifest",
                ))?
        };

        // 2. Register Path
        let path = std::path::PathBuf::from(format!("segment_{}.drift", run_id));
        self.storage.register_file(bucket_id, path);

        // 3. Create Reader & Fetch
        let mut reader = BucketFileReader::new(self.storage.clone(), bucket_id);

        // Get Dim from Manifest
        let dim = { self.manifest.read().unwrap().get_dim() as usize };

        reader.read_all_vectors(dim).await
    }
}
