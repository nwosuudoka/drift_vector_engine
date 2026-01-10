/// Calculates the Squared Euclidean Distance (L2^2) between two vectors.
/// Used for Centroid routing (high precision, low call count).
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

pub fn mean_columns(data: &[f32], dim: usize) -> Vec<f32> {
    debug_assert!(dim > 0);
    debug_assert!(data.len() % dim == 0);

    let rows = data.len() / dim;
    let mut mean = vec![0.0f32; dim];

    for row in data.chunks_exact(dim) {
        // zip keeps it tight and tends to vectorize well
        for (m, &x) in mean.iter_mut().zip(row.iter()) {
            *m += x;
        }
    }

    let inv = 1.0 / rows as f32;
    for m in &mut mean {
        *m *= inv;
    }
    mean
}

// drift_core/src/math.rs

/// Calculates the geometric mean (centroid) of a flat buffer of vectors.
/// Input: `data` [v1_d1, v1_d2, ... v2_d1 ...]
/// Returns a vector of length `dim`.
pub fn calculate_mean(data: &[f32], dim: usize) -> Vec<f32> {
    if data.is_empty() {
        return vec![0.0; dim];
    }
    let count = data.len() / dim;
    let mut mean = vec![0.0; dim];

    // Iterate vectors
    for i in 0..count {
        let start = i * dim;
        let vec = &data[start..start + dim];
        for (d, val) in vec.iter().enumerate() {
            mean[d] += val;
        }
    }

    // Normalize
    let inv = 1.0 / count as f32;
    for val in mean.iter_mut() {
        *val *= inv;
    }
    mean
}

/// Calculates "Drift": The L2 distance between the data's actual mean and the target centroid.
pub fn calculate_drift(data: &[f32], target_centroid: &[f32], dim: usize) -> f32 {
    let current_mean = calculate_mean(data, dim);
    l2_sq(&current_mean, target_centroid).sqrt()
}
