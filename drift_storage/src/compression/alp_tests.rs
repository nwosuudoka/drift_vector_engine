#[cfg(test)]
mod tests {
    use crate::compression::alp::{alp_decode, alp_encode};

    /// Helper to assert float equality
    fn assert_floats_eq(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len(), "Vector lengths mismatch");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            if x.is_nan() && y.is_nan() {
                continue;
            }
            // Exact bit match required for Lossless Compression
            assert_eq!(x.to_bits(), y.to_bits(), "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_simple_integers() {
        // [cite: 147] Data with low variance/simple integers
        let data = vec![1.0, 2.0, 3.0, 100.0, 500.0];
        let encoded = alp_encode(&data);
        let decoded = alp_decode(&encoded, data.len());
        assert_floats_eq(&data, &decoded);
    }

    #[test]
    fn test_decimals_needs_exponent() {
        // [cite: 151] Time series often have small exponent deviation
        let data = vec![1.11, 2.22, 3.33, 4.44, 5.55];
        let encoded = alp_encode(&data);
        let decoded = alp_decode(&encoded, data.len());
        assert_floats_eq(&data, &decoded);
    }

    #[test]
    fn test_trailing_zeros_needs_factor() {
        // [cite: 250] Cutting trailing 0s with extra multiplication (factor f)
        // 12.5 encoded with exp=1 is 125.
        // If encoded with exp=2 (1250), factor=1 reduces it back to 125.
        // This tests if the adaptive sampler finds a good f.
        let data = vec![12.5, 30.5, 100.5];
        let encoded = alp_encode(&data);
        let decoded = alp_decode(&encoded, data.len());
        assert_floats_eq(&data, &decoded);
    }

    #[test]
    fn test_paper_example_hard_decimal() {
        //  8.0605 is the example used to show why simple rounding fails
        // and why high exponents are needed (exp=14).
        let data = vec![8.0605, 8.0605, 8.0605];
        let encoded = alp_encode(&data);
        let decoded = alp_decode(&encoded, data.len());
        assert_floats_eq(&data, &decoded);
    }

    #[test]
    fn test_exceptions_only() {
        // [cite: 406] Values which fail to be encoded become exceptions
        // Use values that cannot be represented as (int * 10^e * 10^-f)
        // e.g., extremely small numbers or irrational-like representations
        let data = vec![std::f64::consts::PI, std::f64::consts::E, 1.234567890123456];
        let encoded = alp_encode(&data);
        let decoded = alp_decode(&encoded, data.len());
        assert_floats_eq(&data, &decoded);
    }

    #[test]
    fn test_mixed_regular_and_exceptions() {
        // Mixed workload: some compressible, some not
        let mut data = vec![1.0, 2.0, 3.0]; // Compressible
        data.push(0.12345678912345); // Exception
        data.push(4.0); // Compressible

        let encoded = alp_encode(&data);
        let decoded = alp_decode(&encoded, data.len());
        assert_floats_eq(&data, &decoded);
    }

    #[test]
    fn test_alignment_boundaries() {
        // Force bit-widths that cause misalignment.
        // 3 values * 3 bits = 9 bits.
        // Header is byte aligned.
        // Deltas (9 bits) -> Flush (encodes to 16 bits).
        // Reader must correctly skip the padding 7 bits.
        let data = vec![1.0, 2.0, 3.0, 4.0];
        // This usually results in very small integers (deltas 0, 1, 2, 3), small bit width.
        let encoded = alp_encode(&data);
        let decoded = alp_decode(&encoded, data.len());
        assert_floats_eq(&data, &decoded);
    }

    #[test]
    fn test_negative_numbers() {
        // [cite: 116] IEEE 754 sign bit handling
        let data = vec![-1.5, -2.5, 3.5, -100.5];
        let encoded = alp_encode(&data);
        let decoded = alp_decode(&encoded, data.len());
        assert_floats_eq(&data, &decoded);
    }

    #[test]
    fn test_zeros() {
        // Zero handling is critical for FOR (Frame of Reference)
        let data = vec![0.0, 0.0, 0.0];
        let encoded = alp_encode(&data);
        let decoded = alp_decode(&encoded, data.len());
        assert_floats_eq(&data, &decoded);
    }

    #[test]
    fn test_large_dataset() {
        // Simulate a vector of 1024 values [cite: 90]
        let mut data = Vec::with_capacity(1024);
        for i in 0..1024 {
            data.push((i as f64) * 0.1);
        }
        let encoded = alp_encode(&data);
        let decoded = alp_decode(&encoded, data.len());
        assert_floats_eq(&data, &decoded);
    }
}
