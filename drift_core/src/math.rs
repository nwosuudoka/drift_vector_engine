/// Calculates the Squared Euclidean Distance (L2^2) between two vectors.
/// Used for Centroid routing (high precision, low call count).
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

/// SIMD-friendly (unrolled) squared L2 distance.
/// Uses plain arithmetic in a tight loop so LLVM can auto-vectorize.
#[inline(always)]
pub fn l2_sq_simd_friendly(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut sum = 0.0f32;
    let mut i = 0usize;

    while i + 4 <= len {
        let d0 = a[i] - b[i];
        let d1 = a[i + 1] - b[i + 1];
        let d2 = a[i + 2] - b[i + 2];
        let d3 = a[i + 3] - b[i + 3];
        sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
        i += 4;
    }

    while i < len {
        let d = a[i] - b[i];
        sum += d * d;
        i += 1;
    }

    sum
}

/// SIMD-friendly (unrolled) cosine similarity.
/// Returns 0.0 if either vector has zero norm.
#[inline(always)]
pub fn cosine_similarity_simd_friendly(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    let mut i = 0usize;

    while i + 4 <= len {
        let a0 = a[i];
        let a1 = a[i + 1];
        let a2 = a[i + 2];
        let a3 = a[i + 3];

        let b0 = b[i];
        let b1 = b[i + 1];
        let b2 = b[i + 2];
        let b3 = b[i + 3];

        dot += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
        norm_a += a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3;
        norm_b += b0 * b0 + b1 * b1 + b2 * b2 + b3 * b3;
        i += 4;
    }

    while i < len {
        let av = a[i];
        let bv = b[i];
        dot += av * bv;
        norm_a += av * av;
        norm_b += bv * bv;
        i += 1;
    }

    if norm_a <= f32::EPSILON || norm_b <= f32::EPSILON {
        0.0
    } else {
        dot / (norm_a.sqrt() * norm_b.sqrt())
    }
}

/// Cosine distance in [0, 2] for non-degenerate vectors, lower is better.
#[inline(always)]
pub fn cosine_distance_simd_friendly(a: &[f32], b: &[f32]) -> f32 {
    let sim = cosine_similarity_simd_friendly(a, b).clamp(-1.0, 1.0);
    1.0 - sim
}

pub fn mean_columns(data: &[f32], dim: usize) -> Vec<f32> {
    debug_assert!(dim > 0);
    debug_assert!(data.len().is_multiple_of(dim));

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

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub enum Metric {
    L2,
    COSINE,
}

impl Metric {
    pub fn from_manifest_str(raw: &str) -> Result<Self, String> {
        let normalized = raw.trim();
        if normalized.is_empty() {
            // Backward-compat fallback for older manifests without metric populated.
            return Ok(Self::L2);
        }

        normalized.parse()
    }
}

impl std::fmt::Display for Metric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Metric::COSINE => write!(f, "COSINE"),
            Metric::L2 => write!(f, "L2"),
        }
    }
}

impl std::str::FromStr for Metric {
    type Err = String;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        match raw.trim().to_ascii_uppercase().as_str() {
            "L2" => Ok(Self::L2),
            "COSINE" => Ok(Self::COSINE),
            _ => Err(format!("unsupported metric '{}'", raw)),
        }
    }
}
