#[cfg(test)]
mod tests {
    use crate::quantizer::Quantizer;

    #[test]
    fn test_quantizer_reconstruction_accuracy() {
        // 1. Train on a simple range [0.0, 100.0]
        let mut data = Vec::new();
        for i in 0..100 {
            data.push(i as f32);
        }

        let q = Quantizer::train(&data, 1);

        // 2. Encode & Reconstruct a known value
        let input = vec![50.0];
        let code = q.encode(&input);
        let rec = q.reconstruct(&code);

        // SQ8 introduces loss, but 50.0 in [0, 100] range should be very close
        let diff = (input[0] - rec[0]).abs();
        assert!(diff < 0.5, "Reconstruction error too high: {}", diff);
    }

    #[test]
    fn test_quantizer_outlier_clipping() {
        // Train on [0, 100]
        let mut data = Vec::new();
        for i in 0..101 {
            data.push(i as f32);
        }

        let q = Quantizer::train(&data, 1);

        // Encode a massive outlier (1000.0)
        // The Quantizer should clamp this to 255 (the max value representing ~100.0)
        let outlier = vec![1000.0];
        let code = q.encode(&outlier);

        assert_eq!(code[0], 255);

        // Reconstructing it should give us the max trained value (~100.0), not 1000.0
        let rec = q.reconstruct(&code);
        assert!(rec[0] < 105.0); // Should be near the max of the training set
    }

    #[test]
    fn test_constant_dimension_safety() {
        // Dim 0 varies, Dim 1 is always 5.0
        let data = vec![vec![1.0, 5.0], vec![2.0, 5.0], vec![3.0, 5.0]]
            .into_iter()
            .flatten()
            .collect::<Vec<f32>>();

        // Should not panic (Division by Zero check)
        let q = Quantizer::train(&data, 2);

        let input = vec![1.5, 5.0];
        let _code = q.encode(&input);

        // Constant dimension usually maps to 0 or 127 depending on logic,
        // but critical part is scale is not Infinity.
        assert!(!q.scale[1].is_infinite());
    }
}
