use crate::router::Router;
use std::collections::HashMap;

/// Result of a partition operation.
/// Maps BucketID -> List of Flattened Vectors belonging to that bucket.
pub type PartitionResult = HashMap<u32, PartitionGroup>;

pub struct PartitionGroup {
    pub ids: Vec<u64>,
    pub flat_vectors: Vec<f32>, // Flattened for efficient writing
    pub count: usize,
}

impl PartitionGroup {
    fn new(_dim: usize) -> Self {
        Self {
            ids: Vec::new(),
            flat_vectors: Vec::new(), // Pre-allocating is hard without guessing count
            count: 0,
        }
    }
}

/// The Incremental Partitioner takes a batch of raw data and assigns it
/// to existing buckets using the Router.
pub struct IncrementalPartitioner;

impl IncrementalPartitioner {
    /// Partitions a batch of vectors into their respective buckets.
    ///
    /// # Arguments
    /// * `vectors`: Slice of (ID, Vector).
    /// * `router`: The read-only router to decide destination.
    pub fn partition(vectors: &[(u64, Vec<f32>)], router: &Router) -> PartitionResult {
        let dim = router.dim();
        let mut groups: PartitionResult = HashMap::new();

        for (id, vec) in vectors {
            // 1. Route
            let bucket_id = router.route(vec);

            // 2. Get or Create Group Buffer
            let group = groups
                .entry(bucket_id)
                .or_insert_with(|| PartitionGroup::new(dim));

            // 3. Append (Flattening on the fly)
            group.ids.push(*id);
            group.flat_vectors.extend_from_slice(vec);
            group.count += 1;
        }

        groups
    }
}
