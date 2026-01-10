use crate::manifest::pb::Centroid;

/// The Router determines which Bucket a vector belongs to.
///
/// It holds a read-optimized, flat view of the global centroids.
/// Designed to be wrapped in an Arc for shared access across threads.
#[derive(Debug, Clone)]
pub struct Router {
    dim: usize,
    metric: String,
    // Flat storage for cache locaclity
    // [centroid1_dim1, centroid1_dim2, ..., centroid2_dim1, ...]
    flat_centroids: Vec<f32>,
    bucket_ids: Vec<u32>,
}

impl Router {
    /// Creates a new Router from centroids
    pub fn new(centriods: &[Centroid], dim: usize, metric: &str) -> Option<Self> {
        if centriods.is_empty() {
            return None;
        }

        let mut flat_centroids = Vec::with_capacity(centriods.len() * dim);
        let mut bucket_ids = Vec::with_capacity(centriods.len());

        for c in centriods {
            if c.vector.len() != dim {
                // Skip invalid centroids or panic? For prod, we skip and log, or fail hard.
                // Failing hard is safer for data integrity.
                panic!(
                    "Router init: Centroid {} dim mismatch (expected {}, got {})",
                    c.id,
                    dim,
                    c.vector.len()
                );
            }
            flat_centroids.extend_from_slice(&c.vector);
            bucket_ids.push(c.id);
        }

        Some(Self {
            dim,
            metric: metric.to_string(),
            flat_centroids,
            bucket_ids,
        })
    }

    /// Finds the nearest Bucket ID for the given vector.
    /// Returns the ID of the closest centroid.
    pub fn route(&self, vector: &[f32]) -> u32 {
        if vector.len() != self.dim {
            // Fallback or panic. In the hot path, panic signals a code bug.
            panic!("Router route: Vector dim mismatch");
        }

        let mut best_id = 0;
        let mut min_dist = f32::MAX;
        let mut max_sim = f32::MIN;

        let is_l2 = self.metric == "L2";
        let count = self.bucket_ids.len();

        for i in 0..count {
            let start = i * self.dim;
            let end = start + self.dim;
            let centroid_vec = &self.flat_centroids[start..end];
            let id = self.bucket_ids[i];

            if is_l2 {
                let dist = l2_sq(vector, centroid_vec);
                if dist < min_dist {
                    min_dist = dist;
                    best_id = id;
                }
            } else {
                // Cosine Similarity
                let sim = cosine_sim(vector, centroid_vec);
                if sim > max_sim {
                    max_sim = sim;
                    best_id = id;
                }
            }
        }

        best_id
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
}

// --- Inline Math Helpers (Candidates for SIMD Optimization later) ---
#[inline]
fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum()
}

#[inline(always)]
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}
