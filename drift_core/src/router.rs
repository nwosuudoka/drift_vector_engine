use std::cmp::Ordering;

use crate::manifest::pb::Centroid;

pub enum Metric {
    L2,
}

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

    bucket_counts: Vec<u32>,
}

impl Router {
    /// Creates a new Router from centroids
    pub fn new(centroids: &[Centroid], counts: &[u32], dim: usize, metric: &str) -> Option<Self> {
        if centroids.is_empty() || centroids.len() != counts.len() {
            return None;
        }

        let mut flat_centroids = Vec::with_capacity(centroids.len() * dim);
        let mut bucket_ids = Vec::with_capacity(centroids.len());
        let mut bucket_counts = Vec::with_capacity(centroids.len());

        for (i, c) in centroids.iter().enumerate() {
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
            bucket_counts.push(counts[i])
        }

        Some(Self {
            dim,
            metric: metric.to_string(),
            flat_centroids,
            bucket_ids,
            bucket_counts,
        })
    }

    /// ⚡ THE ALGORITHM: Drift-Aware Bucket Selection
    /// Returns a list of BucketIDs to scan.
    pub fn select_buckets(
        &self,
        query: &[f32],
        target_confidence: f32,
        lambda: f32,
        tau: f32,
    ) -> Vec<u32> {
        let count = self.bucket_ids.len();
        let mut scores = Vec::with_capacity(count);
        let mut total_score = 0.0;

        // 1. Calculate Unnormalized Scores
        for i in 0..count {
            let start = i * self.dim;
            let centroid = &self.flat_centroids[start..start + self.dim];

            // Distance (Euclidean)
            // Note: Paper uses L2 (Euclidean), not L2 Squared, for the exponential decay.
            let dist_sq = l2_sq(query, centroid);
            let dist = dist_sq.sqrt();

            // Term A: Geometric Probability P(b|q) ~ exp(-lambda * dist)
            let p_geom = (-lambda * dist).exp();

            // Term B: Reliability R(b) = 1 - exp(-count / tau)
            let n_b = self.bucket_counts[i] as f32;
            let reliability = 1.0 - (-n_b / tau).exp();

            let score = p_geom * reliability;

            scores.push((i, score, dist_sq)); // Keep dist_sq for guardrail
            total_score += score;
        }

        // 2. Sort by Score Descending
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        // 3. Accumulate until Target Confidence
        let mut selected_indices = std::collections::HashSet::new();
        let mut accumulated_confidence = 0.0;

        // Safety: Avoid div by zero
        let norm_factor = if total_score > 1e-9 {
            1.0 / total_score
        } else {
            0.0
        };

        for (idx, score, _) in &scores {
            let normalized = score * norm_factor;
            accumulated_confidence += normalized;

            selected_indices.insert(*idx);

            if accumulated_confidence >= target_confidence {
                break;
            }
        }

        // 4. Drift Guardrail (Recall Safety) [Cite: 124]
        // Force the top 3 closest buckets by pure distance, regardless of density.
        // This catches "new, small clusters" that the probabilistic model might ignore.

        // Re-sort/Find top K by distance (simple scan of our scores list, which has dist_sq)
        // We can just iterate the scores list again, or sort a copy.
        // Optimization: Just scan the scores list we already have.
        let mut by_dist = scores.clone();
        by_dist.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal));

        for i in 0..3.min(by_dist.len()) {
            selected_indices.insert(by_dist[i].0);
        }

        // 5. Map indices to Bucket IDs
        selected_indices
            .into_iter()
            .map(|idx| self.bucket_ids[idx])
            .collect()
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
