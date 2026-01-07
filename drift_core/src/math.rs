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
