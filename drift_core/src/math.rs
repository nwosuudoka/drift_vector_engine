/// Calculates the Squared Euclidean Distance (L2^2) between two vectors.
/// Used for Centroid routing (high precision, low call count).
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}
