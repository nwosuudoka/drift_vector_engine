use rand::Rng;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct KMeansResult {
    pub centroids: Vec<Vec<f32>>,
    pub assignments: Vec<usize>,
}

pub struct KMeansTrainer {
    k: usize,
    max_iter: usize,
    tolerance: f32,
    dim: usize,
}

impl KMeansTrainer {
    pub fn new(k: usize, dim: usize, max_iter: usize) -> Self {
        Self {
            k,
            dim,
            max_iter,
            tolerance: 1e-4,
        }
    }

    pub fn train(&self, data: &[Vec<f32>]) -> KMeansResult {
        assert!(!data.is_empty(), "Cannot train on empty data");
        assert_eq!(data[0].len(), self.dim, "Dimension mismatch");

        // 1. Robust Initialization (K-Means++)
        // TODO(production): random sample is usually sufficient if data is shuffled
        let mut centroids = self.init_kmeans_plus_plus(data);
        let mut assignments = vec![usize::MAX; data.len()];

        for _iter in 0..self.max_iter {
            // 2. Assignment Step (Parallel)
            let changes = AtomicUsize::new(0);

            // Note: We compute squared errors for rescue strategy later
            assignments
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, assign)| {
                    let vec = &data[i];
                    let (best_idx, _) = Self::nearest_centroid(vec, &centroids);

                    if *assign != best_idx {
                        *assign = best_idx;
                        changes.fetch_add(1, Ordering::Relaxed);
                    }
                });

            let changed_count = changes.load(Ordering::Relaxed);
            let change_pct = changed_count as f32 / data.len() as f32;

            if change_pct < self.tolerance {
                break;
            }

            // 3. Update Step with Empty Cluster Rescue
            centroids = self.compute_new_centroids_robust(data, &assignments, &centroids);
        }

        KMeansResult {
            centroids,
            assignments,
        }
    }

    /// K-Means++ Initialization
    /// Picks centroids that are far away from each other to prevent collapse.
    fn init_kmeans_plus_plus(&self, data: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mut rng = rand::rng();
        let mut centroids = Vec::with_capacity(self.k);
        let n = data.len();

        // 1. Choose first centroid uniformly at random
        centroids.push(data[rng.random_range(0..n)].clone());

        // 2. Choose remaining K-1 centroids
        for _ in 1..self.k {
            // Calculate distance squared to nearest existing centroid for all points
            let mut dists = vec![0.0; n];
            let mut total_dist_sq = 0.0;

            for (i, vec) in data.iter().enumerate() {
                let (_, d_sq) = Self::nearest_centroid(vec, &centroids);
                dists[i] = d_sq;
                total_dist_sq += d_sq;
            }

            // Roulette Wheel Selection (Weighted Probability)
            // If all points are identical (total_dist_sq == 0), pick random to avoid infinite loop
            if total_dist_sq <= 0.0 {
                centroids.push(data[rng.random_range(0..n)].clone());
                continue;
            }

            let target = rng.random_range(0.0..total_dist_sq);
            let mut cumulative = 0.0;
            let mut selected_idx = 0;

            for (i, &d) in dists.iter().enumerate() {
                cumulative += d;
                if cumulative >= target {
                    selected_idx = i;
                    break;
                }
            }
            centroids.push(data[selected_idx].clone());
        }

        centroids
    }

    fn nearest_centroid(vec: &[f32], centroids: &[Vec<f32>]) -> (usize, f32) {
        let mut min_dist = f32::MAX;
        let mut best_idx = 0;

        for (i, c) in centroids.iter().enumerate() {
            let dist = distance_sq(vec, c);
            if dist < min_dist {
                min_dist = dist;
                best_idx = i;
            }
        }
        (best_idx, min_dist)
    }

    /// Robust Update: Handles empty clusters by respawning them
    fn compute_new_centroids_robust(
        &self,
        data: &[Vec<f32>],
        assignments: &[usize],
        _old_centroids: &[Vec<f32>],
    ) -> Vec<Vec<f32>> {
        let mut sums = vec![vec![0.0; self.dim]; self.k];
        let mut counts = vec![0; self.k];

        // 1. Accumulate
        for (i, &cluster_idx) in assignments.iter().enumerate() {
            if cluster_idx >= self.k {
                continue;
            }
            let vec = &data[i];
            let sum_vec = &mut sums[cluster_idx];
            for d in 0..self.dim {
                sum_vec[d] += vec[d];
            }
            counts[cluster_idx] += 1;
        }

        // 2. Average & Rescue
        let mut new_centroids = Vec::with_capacity(self.k);

        // Find the point with the highest error (worst fit) to respawn empty clusters there
        // This is a simplified "Rescue" strategy: Just pick a random point for V1 robustness
        // or (better) keep the old centroid if it had history.
        let mut rng = rand::rng();

        for i in 0..self.k {
            if counts[i] == 0 {
                // RESCUE STRATEGY: Empty Cluster
                // If we zero it out, it dies.
                // Instead, we respawn it at a random data point to give it a new life.
                let rescue_idx = rng.random_range(0..data.len());
                new_centroids.push(data[rescue_idx].clone());
            } else {
                let count = counts[i] as f32;
                let mut c = Vec::with_capacity(self.dim);
                #[allow(clippy::needless_range_loop)]
                for d in 0..self.dim {
                    c.push(sums[i][d] / count);
                }
                new_centroids.push(c);
            }
        }
        new_centroids
    }
}

// TODO (production): use simd here
#[inline]
fn distance_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_dimensions() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let trainer = KMeansTrainer::new(1, 3, 10);
        let result = trainer.train(&data);

        // With K=1, centroid should be average of all points
        // Avg([1,4]) = 2.5, Avg([2,5]) = 3.5, Avg([3,6]) = 4.5
        let c = &result.centroids[0];
        assert!((c[0] - 2.5).abs() < 0.001);
        assert!((c[1] - 3.5).abs() < 0.001);
        assert!((c[2] - 4.5).abs() < 0.001);
    }

    // ============================================================
    // Test 1: The "Happy Path" (Separation of distinct blobs)
    // ============================================================
    #[test]
    fn test_kmeans_convergence_simple() {
        // Cluster A: Around (10.0, 10.0)
        // Cluster B: Around (-10.0, -10.0)
        let mut data = Vec::new();
        for _ in 0..100 {
            data.push(vec![10.0, 10.0]);
        }
        for _ in 0..100 {
            data.push(vec![-10.0, -10.0]);
        }

        let trainer = KMeansTrainer::new(2, 2, 20);
        let result = trainer.train(&data);

        // Verify Centroids
        let c1 = &result.centroids[0];
        let c2 = &result.centroids[1];

        println!("Centroids: {:?} and {:?}", c1, c2);

        // One should be positive, one negative
        let c1_sum: f32 = c1.iter().sum();
        let c2_sum: f32 = c2.iter().sum();

        assert!(c1_sum.abs() > 15.0); // Should be near 20.0 or -20.0
        assert!(c2_sum.abs() > 15.0);
        assert_ne!(
            c1_sum.signum(),
            c2_sum.signum(),
            "Centroids failed to separate!"
        );
    }

    // ============================================================
    // Test 2: The "Regression Test" (K=1)
    // This previously failed. It verifies the averaging logic works
    // even when no "re-assignment" happens between clusters.
    // ============================================================
    #[test]
    fn test_kmeans_k_equals_1_averaging() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        // With K=1, the centroid MUST be the exact average of all points.
        // Avg([1,4])=2.5, Avg([2,5])=3.5, Avg([3,6])=4.5

        let trainer = KMeansTrainer::new(1, 3, 10);
        let result = trainer.train(&data);

        let c = &result.centroids[0];
        assert!((c[0] - 2.5).abs() < 0.001, "Dim 0 avg failed: {}", c[0]);
        assert!((c[1] - 3.5).abs() < 0.001, "Dim 1 avg failed: {}", c[1]);
        assert!((c[2] - 4.5).abs() < 0.001, "Dim 2 avg failed: {}", c[2]);
    }

    // ============================================================
    // Test 3: The "Embedding Simulation" (High Dimensionality)
    // ============================================================
    #[test]
    fn test_high_dimensional_clustering() {
        let dim = 128;
        let mut data = Vec::new();

        // Create 50 vectors of "All Ones" and 50 vectors of "All Zeros"
        for _ in 0..50 {
            data.push(vec![1.0; dim]);
        }
        for _ in 0..50 {
            data.push(vec![0.0; dim]);
        }

        let trainer = KMeansTrainer::new(2, dim, 10);
        let result = trainer.train(&data);

        // Verify that we recovered the two distinct states
        let c1_mag: f32 = result.centroids[0].iter().sum();
        let c2_mag: f32 = result.centroids[1].iter().sum();

        // One centroid should sum to ~128.0, the other to ~0.0
        // We don't know which is which, so we check the spread.
        let diff = (c1_mag - c2_mag).abs();
        assert!(
            diff > 100.0,
            "High-dim centroids collapsed! Spread was only {}",
            diff
        );
    }

    // ============================================================
    // Test 4: The "Singularity" (All points identical)
    // This tests division-by-zero protection. If all points are the same,
    // K-Means can sometimes panic if a centroid ends up with 0 points assigned.
    // ============================================================
    #[test]
    fn test_degenerate_data_identical_points() {
        let dim = 4;
        let mut data = Vec::new();
        // 100 points, ALL exactly [1.0, 1.0, 1.0, 1.0]
        for _ in 0..100 {
            data.push(vec![1.0; dim]);
        }

        // We ask for K=5. This is impossible (only 1 distinct point exists).
        // The algorithm should not crash (NaN) or panic.
        let trainer = KMeansTrainer::new(5, dim, 10);
        let result = trainer.train(&data);

        for c in &result.centroids {
            for val in c {
                assert!(!val.is_nan(), "NaN detected in centroid!");
                // It's acceptable for empty centroids to be 0.0 or random,
                // but the occupied one must be 1.0.
            }
        }
    }

    // ============================================================
    // Test 5: Input Validation
    // ============================================================
    #[test]
    #[should_panic(expected = "Cannot train on empty data")]
    fn test_panic_on_empty_input() {
        let data: Vec<Vec<f32>> = vec![];
        let trainer = KMeansTrainer::new(2, 4, 10);
        trainer.train(&data);
    }

    #[test]
    #[should_panic(expected = "Dimension mismatch")]
    fn test_panic_on_dimension_mismatch() {
        let data = vec![vec![1.0, 2.0]]; // 2D data
        let trainer = KMeansTrainer::new(2, 128, 10); // Trainer expects 128D
        trainer.train(&data);
    }
}
