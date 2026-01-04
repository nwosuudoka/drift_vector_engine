use bincode::{Decode, Encode};

#[derive(Encode, Decode, Clone)]
pub struct Quantizer {
    pub min: Vec<f32>,
    pub max: Vec<f32>,
    pub scale: Vec<f32>,
}

impl Quantizer {
    /// Train the quantizer using a flat buffer of vectors.
    /// Uses Percentile Clipping (1% - 99%) to ignore outliers[cite: 81, 423].
    ///
    /// # Performance
    /// * **Input:** Flat slice `&[f32]` (Zero-Copy input).
    /// * **Memory:** O(N) allocation *per column* (reused), not O(N*D).
    ///   For 1M vectors, this uses ~4MB temporary RAM instead of ~512MB.
    pub fn train(data: &[f32], dim: usize) -> Self {
        if data.is_empty() {
            panic!("Cannot train quantizer on empty data");
        }
        assert_eq!(data.len() % dim, 0, "Data length must be a multiple of dim");

        let count = data.len() / dim;
        let mut min = vec![0.0; dim];
        let mut max = vec![0.0; dim];
        let mut scale = vec![0.0; dim];

        // Reusable buffer to avoid re-allocating for every dimension
        let mut column_buffer = Vec::with_capacity(count);

        for d in 0..dim {
            // 1. Extract Column (Strided Read)
            column_buffer.clear();
            for i in 0..count {
                // Accessing the d-th component of the i-th vector
                // Layout: [v0_d0, v0_d1, ... | v1_d0, v1_d1 ... ]
                column_buffer.push(data[i * dim + d]);
            }

            // 2. Sort for Percentiles
            // Handle NaNs by defaulting to Equal (treats NaN as equal to itself for sorting stability)
            column_buffer.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let len = column_buffer.len();

            // 3. Percentile Logic (Restored from your code)
            // If N < 100, we don't drop anything.
            let count_to_drop = if len >= 100 { len / 100 } else { 0 };

            let p01_idx = count_to_drop.min(len - 1);
            let p99_idx = len
                .saturating_sub(count_to_drop)
                .saturating_sub(1)
                .min(len - 1);

            let actual_min_idx = p01_idx.min(p99_idx);
            let actual_max_idx = p01_idx.max(p99_idx);

            let p_min = column_buffer[actual_min_idx];
            let p_max = column_buffer[actual_max_idx];

            min[d] = p_min;
            max[d] = p_max;

            // 4. Calculate Scale
            let range = p_max - p_min;
            scale[d] = if range.abs() < 1e-6 {
                1.0 // Constant dimension protection
            } else {
                range / 255.0
            };
        }

        Self { min, max, scale }
    }

    /// Convert a float vector to SQ8 bytes.
    /// Values outside the trained p1-p99 range are clamped.
    pub fn encode(&self, vec: &[f32]) -> Vec<u8> {
        let mut codes = Vec::with_capacity(vec.len());
        for (i, &val) in vec.iter().enumerate() {
            // Formula: code = (val - min) / scale
            let normalized = (val - self.min[i]) / self.scale[i];

            // Use round() to minimize quantization error
            // Clamp to u8 range [0, 255] handles outliers automatically
            let byte = normalized.round().clamp(0.0, 255.0) as u8;
            codes.push(byte);
        }
        codes
    }

    /// Asymmetric Distance Calculation (ADC)
    /// Pre-computes the Squared L2 Distance Table for a query.
    /// Returns a flattened LUT of size [Dim * 256].
    pub fn precompute_lut(&self, query: &[f32]) -> Vec<f32> {
        let dim = query.len();
        let mut lut = vec![0.0; dim * 256];

        for i in 0..dim {
            let q_val = query[i];
            let min_val = self.min[i];
            let s = self.scale[i];

            for b in 0..=255 {
                // Reconstruct the value represented by this byte
                let reconstructed = min_val + (b as f32 * s);
                let diff = q_val - reconstructed;
                // Store squared difference
                lut[i * 256 + b] = diff * diff;
            }
        }
        lut
    }

    /// NEW: Reconstruct float vector from SQ8 bytes.
    /// Inverse of encode: val = min + (byte * scale)
    pub fn reconstruct(&self, code: &[u8]) -> Vec<f32> {
        let mut vec = Vec::with_capacity(code.len());
        for (i, &b) in code.iter().enumerate() {
            let val = self.min[i] + (b as f32 * self.scale[i]);
            vec.push(val);
        }
        vec
    }

    /// Asymmetric Distance Calculation (ADC)
    /// Computes distance between a precise Query (f32) and a quantized Code (u8).
    ///
    /// Formula: Sum( (q[i] - reconstruct(code[i]))^2 )
    #[inline]
    pub fn distance_adc(&self, query: &[f32], code: &[u8]) -> f32 {
        let mut sum_sq = 0.0;

        for (i, &c) in code.iter().enumerate() {
            // Reconstruct the coordinate from the byte 'c'
            // val = min + (c / 255.0) * (max - min)
            // or using precomputed tables if available.
            let reconstructed = self.reconstruct_coord(i, c);
            let diff = query[i] - reconstructed;
            sum_sq += diff * diff;
        }

        sum_sq
    }

    #[inline]
    fn reconstruct_coord(&self, dim_idx: usize, code: u8) -> f32 {
        // Assuming we store min/step per dimension
        // val = min[d] + code * step[d]
        self.min[dim_idx] + (code as f32 * self.scale[dim_idx])
    }
}
