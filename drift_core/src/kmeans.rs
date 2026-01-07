use rand::Rng;
use rayon::prelude::*;

pub struct KMeansResult {
    pub centroids: Vec<Vec<f32>>,
    pub assignments: Vec<usize>,
}

pub struct KMeansTrainer {
    k: usize,
    max_iter: usize,
    tolerance: f32,
    dim: usize,
    batch_size: Option<usize>,
}

impl KMeansTrainer {
    pub fn new(k: usize, dim: usize, max_iter: usize) -> Self {
        Self {
            k,
            dim,
            max_iter,
            tolerance: 1e-4,
            batch_size: None,
        }
    }

    pub fn with_mini_batch(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    /// Train using a flat buffer slice.
    pub fn train(&self, data: &[f32]) -> KMeansResult {
        assert!(!data.is_empty(), "Cannot train on empty data");
        assert_eq!(data.len() % self.dim, 0, "Dimension mismatch");
        let num_samples = data.len() / self.dim;

        // 1. Init
        let mut centroids = self.init_kmeans_plus_plus(data, num_samples);
        let mut cluster_history_counts = vec![0usize; self.k];
        let mut rng = rand::rng();

        for _iter in 0..self.max_iter {
            // A. Batch Selection
            let batch_indices: Vec<usize> = if let Some(bs) = self.batch_size {
                if bs >= num_samples {
                    (0..num_samples).collect()
                } else {
                    (0..bs).map(|_| rng.random_range(0..num_samples)).collect()
                }
            } else {
                (0..num_samples).collect()
            };

            // B. Parallel Assignment & Summation
            let batch_results: Vec<(usize, Vec<f32>, usize)> = batch_indices
                .par_iter()
                .fold(
                    || vec![(0, vec![0.0; self.dim], 0); self.k],
                    |mut acc, &idx| {
                        // ZERO-COPY ACCESS: Slice directly from flat buffer
                        let start = idx * self.dim;
                        let vec = &data[start..start + self.dim];

                        let (best_idx, _) = Self::nearest_centroid(vec, &centroids);
                        let (_, sum_vec, count) = &mut acc[best_idx];

                        for i in 0..self.dim {
                            sum_vec[i] += vec[i];
                        }
                        *count += 1;
                        acc
                    },
                )
                .reduce(
                    || vec![(0, vec![0.0; self.dim], 0); self.k],
                    |mut a, b| {
                        for i in 0..self.k {
                            a[i].2 += b[i].2;
                            for d in 0..self.dim {
                                a[i].1[d] += b[i].1[d];
                            }
                        }
                        a
                    },
                );

            // C. Centroid Update (Streaming Mean)
            let mut max_shift = 0.0f32;
            for (cluster_idx, (_, batch_sum, batch_count)) in batch_results.into_iter().enumerate()
            {
                if batch_count == 0 {
                    continue;
                }

                let old_count = cluster_history_counts[cluster_idx];
                let new_count = old_count + batch_count;
                cluster_history_counts[cluster_idx] = new_count;

                let lr = batch_count as f32 / new_count as f32;
                let momentum = 1.0 - lr;

                let mut current_shift_sq = 0.0;
                for d in 0..self.dim {
                    let old_val = centroids[cluster_idx][d];
                    let batch_avg = batch_sum[d] / batch_count as f32;
                    let new_val = (old_val * momentum) + (batch_avg * lr);
                    centroids[cluster_idx][d] = new_val;
                    let diff = new_val - old_val;
                    current_shift_sq += diff * diff;
                }
                if current_shift_sq > max_shift {
                    max_shift = current_shift_sq;
                }
            }

            if max_shift < self.tolerance {
                break;
            }
        }

        // Final Assignment Pass
        // We map simply to the index, no need to clone floats here.
        let final_assignments = (0..num_samples)
            .into_par_iter()
            .map(|i| {
                let start = i * self.dim;
                let vec = &data[start..start + self.dim];
                Self::nearest_centroid(vec, &centroids).0
            })
            .collect();

        KMeansResult {
            centroids,
            assignments: final_assignments,
        }
    }

    fn init_kmeans_plus_plus(&self, data: &[f32], num_samples: usize) -> Vec<Vec<f32>> {
        let mut rng = rand::rng();
        let mut centroids = Vec::with_capacity(self.k);

        if num_samples == 0 {
            return centroids;
        }

        let pool_limit = if let Some(bs) = self.batch_size {
            (bs * 10).clamp(1000, 20_000).min(num_samples)
        } else {
            50_000.min(num_samples)
        };

        // Pick indices
        let indices: Vec<usize> = if pool_limit < num_samples {
            (0..pool_limit)
                .map(|_| rng.random_range(0..num_samples))
                .collect()
        } else {
            (0..num_samples).collect()
        };

        // 1. First Centroid
        let first_idx = indices[rng.random_range(0..pool_limit)];
        let start = first_idx * self.dim;
        centroids.push(data[start..start + self.dim].to_vec());

        // 2. Remaining Centroids
        for _ in 1..self.k {
            let mut dists = vec![0.0; pool_limit];
            let mut total_dist_sq = 0.0;

            for (i, &data_idx) in indices.iter().enumerate() {
                let start = data_idx * self.dim;
                let vec = &data[start..start + self.dim];
                let (_, d_sq) = Self::nearest_centroid(vec, &centroids);
                dists[i] = d_sq;
                total_dist_sq += d_sq;
            }

            if total_dist_sq <= 0.0 {
                let idx = indices[rng.random_range(0..pool_limit)];
                let start = idx * self.dim;
                centroids.push(data[start..start + self.dim].to_vec());
                continue;
            }

            let target = rng.random_range(0.0..total_dist_sq);
            let mut cumulative = 0.0;
            for (i, &d) in dists.iter().enumerate() {
                cumulative += d;
                if cumulative >= target {
                    let idx = indices[i];
                    let start = idx * self.dim;
                    centroids.push(data[start..start + self.dim].to_vec());
                    break;
                }
            }
        }
        centroids
    }

    #[inline]
    fn nearest_centroid(vec: &[f32], centroids: &[Vec<f32>]) -> (usize, f32) {
        let mut min_dist = f32::MAX;
        let mut best_idx = 0;
        for (i, c) in centroids.iter().enumerate() {
            // Using l2_sq from crate::math would be better, but implementing inline for self-containment in this refactor
            let dist_sq: f32 = vec.iter().zip(c.iter()).map(|(a, b)| (a - b).powi(2)).sum();
            if dist_sq < min_dist {
                min_dist = dist_sq;
                best_idx = i;
            }
        }
        (best_idx, min_dist)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_dimensions() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let data = data.into_iter().flatten().collect::<Vec<f32>>();

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

        let data = data.into_iter().flatten().collect::<Vec<f32>>();

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

        let data = data.into_iter().flatten().collect::<Vec<f32>>();

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
        let data = data.into_iter().flatten().collect::<Vec<f32>>();

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
        let data = data.into_iter().flatten().collect::<Vec<f32>>();

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
        let data: Vec<f32> = vec![];
        let trainer = KMeansTrainer::new(2, 4, 10);
        trainer.train(&data);
    }

    #[test]
    #[should_panic(expected = "Dimension mismatch")]
    fn test_panic_on_dimension_mismatch() {
        let data = vec![vec![1.0, 2.0]]; // 2D data
        let data = data.into_iter().flatten().collect::<Vec<f32>>();

        let trainer = KMeansTrainer::new(5, 128, 10); // Trainer expects 128D
        trainer.train(&data);
    }

    #[test]
    fn test_mini_batch_speed_vs_full_2() {
        let dim = 128;
        let n_samples = 50_000; // Increased to 50k to ensure computation dominates overhead
        let k = 10;

        let mut data = Vec::with_capacity(n_samples);
        let mut rng = rand::rng();

        // Generate Clusters
        let gt_centroids: Vec<Vec<f32>> = (0..k).map(|i| vec![(i * 50) as f32; dim]).collect();

        for _ in 0..n_samples {
            let cluster_idx = rng.random_range(0..k);
            let center = &gt_centroids[cluster_idx];
            let point: Vec<f32> = center.iter().map(|&c| c + rng.random::<f32>()).collect();
            data.push(point);
        }

        let data = data.into_iter().flatten().collect::<Vec<f32>>();

        // A. Standard (Full)
        let start_full = std::time::Instant::now();
        let trainer_full = KMeansTrainer::new(k, dim, 10);
        let result_full = trainer_full.train(&data);
        let duration_full = start_full.elapsed();

        // B. Mini-Batch (Batch=500)
        let batch_size = 500;
        let start_mini = std::time::Instant::now();
        let trainer_mini = KMeansTrainer::new(k, dim, 10).with_mini_batch(batch_size);
        let result_mini = trainer_mini.train(&data);
        let duration_mini = start_mini.elapsed();

        println!("--- Performance Report ---");
        println!("Dataset: {} vectors, {} dim", n_samples, dim);
        println!("Full K-Means Time:       {:.2?}", duration_full);
        println!("Mini-Batch K-Means Time: {:.2?}", duration_mini);

        let speedup = duration_full.as_secs_f64() / duration_mini.as_secs_f64();
        println!("Speedup Factor:          {:.2}x", speedup);

        assert!(
            speedup > 3.0,
            "Mini-Batch should be significantly faster (>3x)"
        );

        // Correctness Check
        let mut mini_sums: Vec<f32> = result_mini.centroids.iter().map(|c| c[0]).collect();
        mini_sums.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut full_sums: Vec<f32> = result_full.centroids.iter().map(|c| c[0]).collect();
        full_sums.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Centroids should be roughly at 0, 50, 100, 150...
        // We verify that Mini-Batch found similar centers to Full.
        for (m, f) in mini_sums.iter().zip(full_sums.iter()) {
            assert!(
                (m - f).abs() < 10.0,
                "Centroids diverged! Mini: {}, Full: {}",
                m,
                f
            );
        }
    }
}
