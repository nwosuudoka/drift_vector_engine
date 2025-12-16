use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct Quantizer {
    pub min: Vec<f32>,
    pub max: Vec<f32>,
    pub scale: Vec<f32>,
}

impl Quantizer {
    pub fn new(min: Vec<f32>, max: Vec<f32>) -> Self {
        let mut scale = Vec::new();
        for i in 0..min.len() {
            scale.push((max[i] - min[i]) / 255.0);
        }
        Self { min, max, scale }
    }

    /// Convert a float vector to SQ8 bytes
    pub fn encode(&self, vec: &[f32]) -> Vec<u8> {
        let mut codes = Vec::with_capacity(vec.len());
        #[allow(clippy::needless_range_loop)]
        for i in 0..vec.len() {
            let val = (vec[i] - self.min[i]) / self.scale[i];
            let byte = val.clamp(0.0, 255.0) as u8;
            codes.push(byte);
        }
        codes
    }

    /// Pre-compute the LUT for a specific query.
    /// Returns a flattened table of size [Dim * 256].
    pub fn precompute_lut(&self, query: &[f32]) -> Vec<f32> {
        let dim = query.len();
        let mut lut = vec![0.0; dim * 256];

        for i in 0..dim {
            let q_val = query[i];
            let min_val = self.min[i];
            let s = self.scale[i];

            for b in 0..=255 {
                let reconstructed = min_val + (b as f32 * s);
                let diff = q_val - reconstructed;
                // Store squared difference
                lut[i * 256 + b] = diff * diff;
            }
        }
        lut
    }
}
