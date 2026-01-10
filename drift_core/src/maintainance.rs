use crate::kmeans::KMeansTrainer;
use crate::manifest::pb::Centroid;
use crate::math::{calculate_mean, l2_sq};

pub struct SplitResult {
    pub left: Partition,
    pub right: Partition,
    /// Vectors that should be sent back to MemTable for re-routing
    pub loopback: Vec<(u64, Vec<f32>)>,
}

pub struct Partition {
    pub centroid: Centroid,
    pub ids: Vec<u64>,
    pub flat_vectors: Vec<f32>,
}

impl Partition {
    fn new(centroid_vec: Vec<f32>) -> Self {
        Self {
            centroid: Centroid {
                id: 0,
                vector: centroid_vec,
            },
            ids: Vec::new(),
            flat_vectors: Vec::new(),
        }
    }

    fn push(&mut self, id: u64, vec: &[f32]) {
        self.ids.push(id);
        self.flat_vectors.extend_from_slice(vec);
    }
}

pub struct MaintenanceCalculator;

impl MaintenanceCalculator {
    /// Pure Logic: Splits a bucket, allowing vectors to "defect" to neighbors.
    pub fn split_with_loopback(
        ids: &[u64],
        flat_vectors: &[f32],
        dim: usize,
        neighbor_centroids: &[Vec<f32>], // Global Context
    ) -> Option<SplitResult> {
        let count = ids.len();
        if count < 10 {
            return None;
        }
        if flat_vectors.len() != count * dim {
            return None;
        }

        // 1. Train K-Means (K=2)
        let trainer = KMeansTrainer::new(2, dim, 10);
        let result = trainer.train(flat_vectors);

        if result.centroids.len() < 2 {
            return None;
        }

        let c1 = &result.centroids[0];
        let c2 = &result.centroids[1];

        let mut left = Partition::new(c1.clone());
        let mut right = Partition::new(c2.clone());
        let mut loopback = Vec::new();

        // Safety Cap: Don't loopback more than 20% of data
        let max_loopback = count / 5;

        for (i, &_assignment) in result.assignments.iter().enumerate() {
            let start = i * dim;
            let vec = &flat_vectors[start..start + dim];
            let id = ids[i];

            // 1. Calculate Local Preference
            // Which child is closer?
            let d1 = l2_sq(vec, c1);
            let d2 = l2_sq(vec, c2);
            let (d_local, is_left) = if d1 < d2 { (d1, true) } else { (d2, false) };

            // 2. Calculate Neighbor Preference (Global Scan)
            let mut d_neighbor_min = f32::MAX;
            for n in neighbor_centroids {
                let d = l2_sq(vec, n);
                if d < d_neighbor_min {
                    d_neighbor_min = d;
                }
            }

            // 3. Decision Logic
            // Hysteresis: Only switch if neighbor is notably better (e.g. 10% closer squared)
            // AND we haven't hit the cap.
            let margin = d_local * 0.90;

            if d_neighbor_min < margin && loopback.len() < max_loopback {
                // Defect!
                loopback.push((id, vec.to_vec()));
            } else {
                // Stay
                if is_left {
                    left.push(id, vec);
                } else {
                    right.push(id, vec);
                }
            }
        }

        // Recalculate centroids for children since we removed points
        // (Optional but recommended for quality)
        left.centroid.vector = calculate_mean(&left.flat_vectors, dim);
        right.centroid.vector = calculate_mean(&right.flat_vectors, dim);

        Some(SplitResult {
            left,
            right,
            loopback,
        })
    }
}
