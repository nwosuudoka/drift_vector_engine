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
    /// * `ids`: Slice of Vector IDs.
    /// * `flat_vectors`: Flattened vector data (layout: [v0_d0, v0_d1, ... v1_d0...]).
    /// * `dim`: Dimensionality of vectors.
    /// * `router`: The read-only router to decide destination.
    pub fn partition(
        ids: &[u64],
        flat_vectors: &[f32],
        dim: usize,
        router: &Router,
    ) -> PartitionResult {
        // Safety / Sanity check
        assert_eq!(
            flat_vectors.len(),
            ids.len() * dim,
            "Mismatch between IDs count and flat vector data length"
        );

        let mut groups: PartitionResult = HashMap::new();

        for (i, &id) in ids.iter().enumerate() {
            let start = i * dim;
            let end = start + dim;

            // Zero-copy slice
            let vec = &flat_vectors[start..end];

            // 1. Route
            let bucket_id = router.route(vec);

            // 2. Get or Create Group Buffer
            let group = groups
                .entry(bucket_id)
                .or_insert_with(|| PartitionGroup::new(dim));

            // 3. Append
            group.ids.push(id);
            group.flat_vectors.extend_from_slice(vec);
            group.count += 1;
        }

        groups
    }
}
