use crate::bucket_file_reader::BucketFileReader;
use drift_traits::{DiskSearcher, PageManager, SearchCandidate, TombstoneView};
use futures::future::join_all;
use parking_lot::RwLock;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;
use tokio::sync::Semaphore; // For concurrency control

/// Manages the mapping between Logical Buckets and Physical Files.
/// Orchestrates parallel scans and refinement.
pub struct BucketManager {
    storage: Arc<dyn PageManager>,

    // Mapping: BucketID -> FileID (In V2, this comes from the Manifest)
    // For now, we assume a direct mapping or simple lookup.
    bucket_map: RwLock<HashMap<u32, u32>>,

    // Concurrency Limiter: Don't open 1000 files at once.
    scan_semaphore: Arc<Semaphore>,
}

impl BucketManager {
    pub fn new(storage: Arc<dyn PageManager>, max_concurrent_scans: usize) -> Self {
        Self {
            storage,
            bucket_map: RwLock::new(HashMap::new()),
            scan_semaphore: Arc::new(Semaphore::new(max_concurrent_scans)),
        }
    }

    pub fn update_mapping(&self, bucket_id: u32, file_id: u32) {
        self.bucket_map.write().insert(bucket_id, file_id);
    }

    pub fn get_file_id(&self, bucket_id: u32) -> Option<u32> {
        self.bucket_map.read().get(&bucket_id).copied()
    }
}

// Wrapper for Heap Sorting (same as in Reader)
#[derive(PartialEq)]
struct CandidateWrapper(SearchCandidate);
impl Eq for CandidateWrapper {}
impl Ord for CandidateWrapper {
    fn cmp(&self, other: &Self) -> Ordering {
        // MaxHeap: Pop largest distance
        self.0
            .approx_dist
            .partial_cmp(&other.0.approx_dist)
            .unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for CandidateWrapper {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[async_trait::async_trait]
impl DiskSearcher for BucketManager {
    async fn search(
        &self,
        bucket_ids: &[u32],
        query: &[f32],
        k: usize,
        tombstones: Arc<dyn TombstoneView>, // ⚡ Now owned Arc
    ) -> Vec<SearchCandidate> {
        let mut handles = Vec::with_capacity(bucket_ids.len());

        // 1. Scatter: Spawn Scan Tasks
        for &bid in bucket_ids {
            let file_id = match self.get_file_id(bid) {
                Some(f) => f,
                None => continue,
            };

            // Clone dependencies for the 'static task
            let storage = self.storage.clone();
            let query = query.to_vec(); // Vector query must be owned by task
            let sem = self.scan_semaphore.clone();
            let tombstones = tombstones.clone(); // ⚡ Cheap Arc clone

            // Spawn concurrent task
            handles.push(tokio::spawn(async move {
                // Rate limit concurrency
                let _permit = sem.acquire().await.unwrap();

                let mut reader = BucketFileReader::new(storage, file_id);
                // Reader expects &dyn, so we deref the Arc
                match reader.scan(&query, k, tombstones.as_ref()).await {
                    Ok(c) => c,
                    Err(e) => {
                        eprintln!("Error scanning bucket {}: {}", bid, e);
                        Vec::new()
                    }
                }
            }));
        }

        // 2. Gather: Wait for all tasks
        let results_list = join_all(handles).await;

        // 3. Merge Results (Top-K Reduction)
        let mut heap: BinaryHeap<CandidateWrapper> = BinaryHeap::with_capacity(k + 1);

        for candidates in results_list.into_iter().flatten() {
            for c in candidates {
                if heap.len() < k {
                    heap.push(CandidateWrapper(c));
                } else if let Some(worst) = heap.peek() {
                    if c.approx_dist < worst.0.approx_dist {
                        heap.pop();
                        heap.push(CandidateWrapper(c));
                    }
                }
            }
        }

        // 4. Sort
        let mut results: Vec<SearchCandidate> = heap.into_vec().into_iter().map(|w| w.0).collect();
        results.sort_by(|a, b| {
            a.approx_dist
                .partial_cmp(&b.approx_dist)
                .unwrap_or(Ordering::Equal)
        });

        results
    }

    /// Step 2: Batched Refinement
    async fn refine(&self, candidates: Vec<SearchCandidate>, query: &[f32]) -> Vec<(u64, f32)> {
        // Group by FileID
        let mut file_groups: HashMap<u32, Vec<SearchCandidate>> = HashMap::new();
        for c in candidates {
            file_groups.entry(c.file_id).or_default().push(c);
        }

        let mut final_results = Vec::new();

        // Process each file
        for (file_id, group) in file_groups {
            let reader = BucketFileReader::new(self.storage.clone(), file_id);
            // We use the same 'refine' logic we put in the Reader
            match reader.refine(group, query, query.len()).await {
                Ok(refined) => final_results.extend(refined),
                Err(e) => eprintln!("Error refining file {}: {}", file_id, e),
            }
        }

        final_results
    }
}
