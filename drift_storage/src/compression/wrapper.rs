use crate::compression::alp::{alp_decode, alp_encode};
use crate::compression::alp_rd::{alp_rd_decode, alp_rd_encode};

/// Compress a set of vectors into a PageBlock-aligned buffer.
/// Returns the raw bytes to be written to disk.
pub fn compress_vectors(data: &[f32], strategy: CompressionStrategy) -> Vec<u8> {
    // 1. Cast f32 to f64 (ALP works natively on f64)
    // In production, we might port ALP to f32, but for now we promote.
    let data_f64: Vec<f64> = data.iter().map(|&x| x as f64).collect();

    match strategy {
        CompressionStrategy::Alp => alp_encode(&data_f64),
        CompressionStrategy::AlpRd => alp_rd_encode(&data_f64),
    }
}

/// Decompress bytes back into a Float Vector.
pub fn decompress_vectors(bytes: &[u8], count: usize, strategy: CompressionStrategy) -> Vec<f32> {
    let decoded_f64 = match strategy {
        // We need to expose the decode functions publically in your mod.rs
        CompressionStrategy::Alp => crate::compression::alp::alp_decode(bytes, count),
        CompressionStrategy::AlpRd => crate::compression::alp_rd::alp_rd_decode(bytes),
    };

    // Cast back to f32 for the engine
    decoded_f64.iter().map(|&x| x as f32).collect()
}

#[derive(Clone, Copy, Debug)]
pub enum CompressionStrategy {
    Alp,
    AlpRd,
}

pub struct CompressedColumn {
    pub data: Vec<u8>,
    pub count: usize,
    pub strategy: CompressionStrategy,
}

impl CompressedColumn {
    pub fn compress(raw_floats: &[f32], strategy: CompressionStrategy) -> Self {
        // ALP works natively on f64. Promote f32 -> f64.
        // This is a zero-copy cast if we were using f64 natively,
        // but for f32 embeddings we pay the cast cost for 3x compression gains.
        let data_f64: Vec<f64> = raw_floats.iter().map(|&x| x as f64).collect();
        let compressed_bytes = match strategy {
            CompressionStrategy::Alp => alp_encode(&data_f64),
            CompressionStrategy::AlpRd => alp_rd_encode(&data_f64),
        };

        Self {
            data: compressed_bytes,
            count: raw_floats.len(),
            strategy,
        }
    }

    /// Decompress back to floats.
    pub fn decompress(&self) -> Vec<f32> {
        let decoded_f64 = match self.strategy {
            CompressionStrategy::Alp => alp_decode(&self.data, self.count),
            CompressionStrategy::AlpRd => alp_rd_decode(&self.data),
        };

        // Demote back to f32
        decoded_f64.iter().map(|&x| x as f32).collect()
    }
}

/// Transpose Row-Major Vectors [N][D] -> Column-Major [D][N].
///
/// # Logic
/// Converts a list of vectors (rows) into a list of dimension columns.
/// This maximizes compression because values in a specific dimension (e.g., "warmth")
/// are usually correlated, whereas values across a single vector are not.
///
/// # Panics
/// Strictly enforces that all input vectors must have length equal to `dim`.
/// This prevents "Ragged Arrays" which would corrupt the columnar structure.
pub fn transpose(vectors: &[Vec<f32>], dim: usize) -> Vec<Vec<f32>> {
    if vectors.is_empty() {
        return vec![Vec::new(); dim];
    }

    let n = vectors.len();
    let mut columns = vec![Vec::with_capacity(n); dim];

    for (i, vec) in vectors.iter().enumerate() {
        // We explicitly assert length equality.
        assert_eq!(
            vec.len(),
            dim,
            "Vector at index {} has length {}, expected {}",
            i,
            vec.len(),
            dim
        );

        for (d, &val) in vec.iter().enumerate() {
            columns[d].push(val);
        }
    }
    columns
}

/// Inverse Transpose: Column-Major [D][N] -> Row-Major [N][D].
/// Used when loading data from disk into memory.
pub fn transpose_from_columns(columns: &[Vec<f32>], count: usize) -> Vec<Vec<f32>> {
    let dim = columns.len();
    if dim == 0 {
        return Vec::new();
    }

    let mut vectors = Vec::with_capacity(count);

    // Pre-allocate inner vectors
    for _ in 0..count {
        vectors.push(Vec::with_capacity(dim));
    }

    // Scatter columns into rows
    for col in columns {
        assert_eq!(col.len(), count, "Column length mismatch during restore");
        for (row_idx, &val) in col.iter().enumerate() {
            vectors[row_idx].push(val);
        }
    }
    vectors
}

/// Transpose scattered rows from a flat buffer.
///
/// # Arguments
/// * `source` - The massive flat MemTable buffer.
/// * `indices` - List of row indices to extract and transpose.
/// * `dim` - Vector dimension.
pub fn transpose_subset(source: &[f32], indices: &[usize], dim: usize) -> Vec<Vec<f32>> {
    assert!(source.len() % dim == 0, "Invalid source length");

    let n = indices.len();
    if n == 0 {
        return vec![Vec::new(); dim];
    }

    // Allocate columns (we must create these to compress them independently)
    let mut columns = vec![Vec::with_capacity(n); dim];

    for &row_idx in indices {
        let start = row_idx * dim;
        // Safety check
        if start + dim > source.len() {
            panic!("transpose_subset: Index out of bounds");
        }

        let vec_slice = &source[start..start + dim];
        for (d, &val) in vec_slice.iter().enumerate() {
            columns[d].push(val);
        }
    }
    columns
}

#[cfg(test)]
mod tests {
    use crate::compression::wrapper::{
        CompressedColumn, CompressionStrategy, transpose, transpose_subset,
    };
    use std::f32;

    #[test]
    fn test_transpose_logic() {
        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![4.0, 5.0, 6.0];
        let input = vec![vec1, vec2]; // 2 vectors, 3 dims

        let columns = transpose(&input, 3);

        assert_eq!(columns.len(), 3);
        assert_eq!(columns[0], vec![1.0, 4.0]); // Dim 0
        assert_eq!(columns[1], vec![2.0, 5.0]); // Dim 1
        assert_eq!(columns[2], vec![3.0, 6.0]); // Dim 2
    }

    #[test]
    fn test_alp_roundtrip_f32() {
        // Create random "Embedding-like" data
        // Small range, high precision
        let original: Vec<f32> = (0..1000).map(|i| (i as f32).sin()).collect();

        // Compress using ALP_RD (Real Doubles mode)
        let col = CompressedColumn::compress(&original, CompressionStrategy::AlpRd);

        println!("Original Size: {} bytes", original.len() * 4);
        println!("Compressed Size: {} bytes", col.data.len());
        println!(
            "Ratio: {:.2}x",
            (original.len() * 4) as f32 / col.data.len() as f32
        );

        // Decompress
        let recovered = col.decompress();

        // Verify Lossless (Exact bit match)
        for (a, b) in original.iter().zip(recovered.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "Mismatch found!");
        }
    }

    // ============================================================
    // Test Category 1: The Transposition Engine
    // ============================================================

    #[test]
    fn test_transpose_basic() {
        let vec1 = vec![1.0, 2.0];
        let vec2 = vec![3.0, 4.0];
        let input = vec![vec1, vec2];

        let cols = transpose(&input, 2);
        assert_eq!(cols.len(), 2);
        assert_eq!(cols[0], vec![1.0, 3.0]); // Column 0
        assert_eq!(cols[1], vec![2.0, 4.0]); // Column 1
    }

    #[test]
    #[should_panic(expected = "expected 3")]
    fn test_transpose_ragged_input_panics() {
        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![1.0, 2.0]; // Short vector
        let input = vec![vec1, vec2];
        transpose(&input, 3);
    }

    #[test]
    fn test_transpose_empty_input() {
        let input: Vec<Vec<f32>> = vec![];
        let cols = transpose(&input, 128);
        assert_eq!(cols.len(), 128); // Should have 128 empty columns
        assert!(cols[0].is_empty());
    }

    // ============================================================
    // Test Category 2: Data Distribution Edge Cases
    // ============================================================

    #[test]
    fn test_compress_constant_values() {
        // [Cite 250] Constant values (trailing zero / low variance) logic
        let data = vec![42.0; 1000]; // Zero variance

        let col = CompressedColumn::compress(&data, CompressionStrategy::AlpRd);
        let recovered = col.decompress();

        assert_floats_approx(&data, &recovered);

        // Compression check: 1000 * 4 bytes = 4000 bytes.
        // ALP should crush this to negligible size (headers + run length).
        println!("Constant Data Size: {} -> {}", 4000, col.data.len());
        assert!(col.data.len() < 200, "Constant compression failed");
    }

    #[test]
    fn test_compress_toxic_floats() {
        // ALP must handle NaNs, Infinities, and Denormals gracefully
        let data = vec![
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.0,
            -0.0,
            f32::MIN_POSITIVE, // Denormal
        ];

        let col = CompressedColumn::compress(&data, CompressionStrategy::AlpRd);
        let recovered = col.decompress();

        assert_floats_approx(&data, &recovered);
    }

    #[test]
    fn test_compress_high_entropy_random() {
        use rand::Rng;
        let mut rng = rand::rng();
        let data: Vec<f32> = (0..1000).map(|_| rng.random()).collect();

        // High entropy data is hard to compress. ALP_RD should handle it via exceptions/dictionary
        let col = CompressedColumn::compress(&data, CompressionStrategy::AlpRd);
        let recovered = col.decompress();

        assert_floats_approx(&data, &recovered);
    }

    // ============================================================
    // Test Category 3: Scale Tests
    // ============================================================

    #[test]
    fn test_openai_scale_simulation() {
        // Simulate a Bucket of 100 OpenAI vectors (1536 dims)
        // Total Floats: 153,600
        let n_vecs = 100;
        let dim = 1536;

        // Create synthetic embedding-like data (normalized, small range)
        let mut vectors = Vec::with_capacity(n_vecs);
        for i in 0..n_vecs {
            let v: Vec<f32> = (0..dim).map(|j| ((i + j) as f32).sin()).collect();
            vectors.push(v);
        }

        // 1. Transpose
        let columns = transpose(&vectors, dim);
        assert_eq!(columns.len(), dim);
        assert_eq!(columns[0].len(), n_vecs);

        // 2. Compress a single column (Simulate Segment Write)
        let col_0 = &columns[0];
        let compressed = CompressedColumn::compress(col_0, CompressionStrategy::AlpRd);

        // 3. Verify Ratio
        let raw_size = n_vecs * 4;
        let comp_size = compressed.data.len();
        println!(
            "OpenAI Column Compression: {}b -> {}b (Ratio: {:.2}x)",
            raw_size,
            comp_size,
            raw_size as f32 / comp_size as f32
        );

        // 4. Decompress
        let recovered = compressed.decompress();
        assert_floats_approx(col_0, &recovered);
    }

    #[allow(dead_code)]
    fn assert_floats_approx(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len(), "Length mismatch");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            if x.is_nan() {
                assert!(y.is_nan(), "Expected NaN at {}, got {}", i, y);
            } else {
                assert_eq!(x, y, "Mismatch at index {}", i);
            }
        }
    }

    #[test]
    fn test_transpose_logic_2() {
        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![4.0, 5.0, 6.0];
        let input = vec![vec1, vec2];
        let columns = transpose(&input, 3);
        assert_eq!(columns.len(), 3);
        assert_eq!(columns[0], vec![1.0, 4.0]);
    }

    #[test]
    fn test_transpose_subset_scattering() {
        let dim = 2;
        // 4 vectors [v0, v1, v2, v3] flattened
        // v0: [0.0, 0.1]
        // v1: [1.0, 1.1]
        // v2: [2.0, 2.1]
        // v3: [3.0, 3.1]
        let flat_data = vec![
            0.0, 0.1, // Row 0
            1.0, 1.1, // Row 1
            2.0, 2.1, // Row 2
            3.0, 3.1, // Row 3
        ];

        // We want to extract only Row 0 and Row 3 (Skipping 1 and 2)
        let indices = vec![0, 3];

        let columns = transpose_subset(&flat_data, &indices, dim);

        assert_eq!(columns.len(), 2, "Should produce 2 columns (dim=2)");
        assert_eq!(
            columns[0].len(),
            2,
            "Each column should have 2 items (indices.len())"
        );

        // Column 0 should contain [v0[0], v3[0]] -> [0.0, 3.0]
        assert_eq!(columns[0], vec![0.0, 3.0]);

        // Column 1 should contain [v0[1], v3[1]] -> [0.1, 3.1]
        assert_eq!(columns[1], vec![0.1, 3.1]);
    }

    #[test]
    fn test_transpose_subset_empty() {
        let flat_data = vec![1.0, 2.0];
        let indices = vec![];
        let dim = 2;

        let columns = transpose_subset(&flat_data, &indices, dim);

        assert_eq!(columns.len(), 2);
        assert!(columns[0].is_empty());
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_transpose_subset_out_of_bounds() {
        let flat_data = vec![1.0, 2.0]; // 1 vector
        let indices = vec![5]; // Requesting index 5
        let dim = 2;

        transpose_subset(&flat_data, &indices, dim);
    }
}
