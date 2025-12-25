use bincode::{Decode, Encode};

#[derive(Encode, Decode, Clone)]
pub struct Quantizer {
    pub min: Vec<f32>,
    pub max: Vec<f32>,
    pub scale: Vec<f32>,
}

impl Quantizer {
    /// Train the quantizer using Percentile Clipping (1% - 99%).
    /// This handles "Soap Bubble" distributions where outliers skew the grid.
    pub fn train(data: &[Vec<f32>]) -> Self {
        if data.is_empty() {
            panic!("Cannot train quantizer on empty data");
        }

        let dim = data[0].len();
        let mut min = vec![0.0; dim];
        let mut max = vec![0.0; dim];
        let mut scale = vec![0.0; dim];

        // Transpose and sort to find percentiles per dimension
        for i in 0..dim {
            let mut column: Vec<f32> = data.iter().map(|v| v[i]).collect();

            // Sort to find percentiles
            // Handle NaNs by pushing them to the end (or treat as equal)
            column.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let len = column.len();

            // Calculate how many items constitute 1%
            // Ensure we drop at least 1 item if len is small but > 0
            let count_to_drop = (len / 100).max(1);

            // If len is small (e.g., 1), min(len-1) ensures we don't go out of bounds.
            let p01_idx = count_to_drop.min(len - 1);

            // Use saturating_sub to prevent overflow when len is small.
            // Logic: We want the index `count_to_drop` positions from the end.
            let p99_idx = len
                .saturating_sub(count_to_drop)
                .saturating_sub(1)
                .min(len - 1);

            // 3: Inversion Protection
            // On very small datasets (len=2), p01_idx might be > p99_idx.
            // We ensure max >= min to prevent negative scaling.
            let actual_p01_idx = p01_idx.min(p99_idx);
            let actual_p99_idx = p01_idx.max(p99_idx);

            let p01 = column[actual_p01_idx];
            let p99 = column[actual_p99_idx];

            min[i] = p01;
            max[i] = p99;

            // Calculate scale
            let range = max[i] - min[i];
            // Prevent division by zero for constant dimensions
            scale[i] = if range.abs() < 1e-6 {
                1.0
            } else {
                range / 255.0
            };
        }

        Self { min, max, scale }
    }

    /// Convert a float vector to SQ8 bytes.
    /// Values outside the p1-p99 range are clamped.
    pub fn encode(&self, vec: &[f32]) -> Vec<u8> {
        let mut codes = Vec::with_capacity(vec.len());
        for i in 0..vec.len() {
            let val = (vec[i] - self.min[i]) / self.scale[i];
            // Use round() to minimize quantization error (vs floor)
            let byte = val.round().clamp(0.0, 255.0) as u8;
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

    // in drift_core/src/quantizer.rs

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
