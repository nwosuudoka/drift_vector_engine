use crate::payload::{PayloadRow, PayloadSchema};
use crate::router::Router;
use std::collections::HashMap;
use std::io;

/// Result of a partition operation.
/// Maps BucketID -> List of Flattened Vectors belonging to that bucket.
pub type PartitionResult = HashMap<u32, PartitionGroup>;

pub struct PartitionGroup {
    pub ids: Vec<u64>,
    pub flat_vectors: Vec<f32>, // Flattened for efficient writing
    pub count: usize,
    pub centroid: Option<Vec<f32>>,
    pub payload_schema: Option<PayloadSchema>,
    pub payload_rows: Option<Vec<PayloadRow>>,
}

impl PartitionGroup {
    pub fn new(_dim: usize, centroid: Option<Vec<f32>>) -> Self {
        Self {
            ids: Vec::new(),
            flat_vectors: Vec::new(), // Pre-allocating is hard without guessing count
            count: 0,
            centroid,
            payload_schema: None,
            payload_rows: None,
        }
    }

    pub fn new_with_payload(
        _dim: usize,
        centroid: Option<Vec<f32>>,
        payload_schema: Option<PayloadSchema>,
    ) -> Self {
        let has_payload = payload_schema.is_some();
        Self {
            ids: Vec::new(),
            flat_vectors: Vec::new(),
            count: 0,
            centroid,
            payload_schema,
            payload_rows: if has_payload { Some(Vec::new()) } else { None },
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
        Self::partition_with_payload(ids, flat_vectors, dim, router, None, None)
            .expect("vector-only partition should be infallible")
    }

    pub fn partition_with_payload(
        ids: &[u64],
        flat_vectors: &[f32],
        dim: usize,
        router: &Router,
        payload_schema: Option<&PayloadSchema>,
        payload_rows: Option<&[PayloadRow]>,
    ) -> io::Result<PartitionResult> {
        // Safety / Sanity check
        if flat_vectors.len() != ids.len() * dim {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Mismatch between IDs count and flat vector data length",
            ));
        }
        if payload_rows.is_some() && payload_schema.is_none() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "payload rows provided without payload schema",
            ));
        }
        if payload_schema.is_some() && payload_rows.is_none() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "payload schema provided without payload rows",
            ));
        }
        if let Some(rows) = payload_rows
            && rows.len() != ids.len()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "payload row count mismatch: ids={}, payload_rows={}",
                    ids.len(),
                    rows.len()
                ),
            ));
        }

        let mut groups: PartitionResult = HashMap::new();

        for (i, &id) in ids.iter().enumerate() {
            let start = i * dim;
            let end = start + dim;

            // Zero-copy slice
            let vec = &flat_vectors[start..end];

            // 1. Route
            let bucket_id = router.route(vec);

            // 2. Get or Create Group Buffer
            let group = groups.entry(bucket_id).or_insert_with(|| {
                let centroid = router.get_centroid(bucket_id);
                PartitionGroup::new_with_payload(dim, centroid, payload_schema.cloned())
            });

            // 3. Append
            group.ids.push(id);
            group.flat_vectors.extend_from_slice(vec);
            group.count += 1;
            if let Some(rows) = payload_rows
                && let Some(group_rows) = group.payload_rows.as_mut()
            {
                group_rows.push(rows[i].clone());
            }
        }

        Ok(groups)
    }
}
