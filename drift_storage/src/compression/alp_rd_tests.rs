#[cfg(test)]
mod tests {
    use crate::compression::alp_rd::{alp_rd_decode, alp_rd_encode};

    #[allow(dead_code)]
    fn bitwise_eq(a: f64, b: f64) -> bool {
        a.to_bits() == b.to_bits()
    }

    #[allow(dead_code)]
    fn vec_approx_eq(a: &[f64], b: &[f64]) -> bool {
        if a.len() != b.len() {
            return false;
        }
        // Use bitwise comparison for exact lossless roundtrip verification
        // This handles NaNs, Infinities, and -0.0 vs 0.0 correctly.
        a.iter().zip(b).all(|(x, y)| bitwise_eq(*x, *y))
    }

    // ======================================================
    // 1. BASIC ROUNDTRIP TESTS
    // ======================================================

    #[test]
    fn simple_small_list() {
        let v = vec![1.0, 2.0, 3.0];
        let encoded = alp_rd_encode(&v);
        let decoded = alp_rd_decode(&encoded);
        assert!(vec_approx_eq(&v, &decoded));
    }

    #[test]
    fn repeated_values_compress_well() {
        let v = vec![42.0; 1000];
        let encoded = alp_rd_encode(&v);
        assert!(encoded.len() < 500);
        let decoded = alp_rd_decode(&encoded);
        assert!(vec_approx_eq(&v, &decoded));
    }

    #[test]
    fn mixed_small_values() {
        let v = vec![1.25, -3.75, 99.001, 0.0, 1024.0, -2048.5];
        let encoded = alp_rd_encode(&v);
        let decoded = alp_rd_decode(&encoded);
        assert!(vec_approx_eq(&v, &decoded));
    }

    // ======================================================
    // 2. EDGE CASE FLOATS
    // ======================================================

    #[test]
    fn floating_edge_cases() {
        let specials = vec![
            0.0,
            -0.0,
            f64::MIN_POSITIVE,
            f64::EPSILON,
            f64::MAX,
            f64::MIN,
            1.0,
            -1.0,
            std::f64::consts::PI,
            std::f64::consts::E,
        ];

        let encoded = alp_rd_encode(&specials);
        let decoded = alp_rd_decode(&encoded);
        assert!(vec_approx_eq(&specials, &decoded));
    }

    #[test]
    fn denormals_test() {
        let values: Vec<f64> = (0..1000)
            .map(|i| f64::from_bits(i as u64)) // includes denormals, zeros, tiny numbers
            .collect();

        let encoded = alp_rd_encode(&values);
        let decoded = alp_rd_decode(&encoded);
        assert!(vec_approx_eq(&values, &decoded));
    }

    #[test]
    fn encode_nan_preserves_bit_pattern() {
        // NOTE: IEEE NaNs compare unequal to themselves, so we compare bits directly.
        let values = vec![
            f64::from_bits(0x7FF8_0000_0000_0001),
            f64::from_bits(0x7FFF_FFFF_FFFF_FFFF),
            f64::from_bits(0xFFF8_0000_0000_0001),
        ];

        let encoded = alp_rd_encode::<f64>(&values);
        let decoded = alp_rd_decode::<f64>(&encoded);

        for (a, b) in values.iter().zip(decoded) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    // ======================================================
    // 3. DICTIONARY EDGE CASES
    // ======================================================

    #[test]
    fn dictionary_single_unique_value() {
        let v = vec![123.456];
        let encoded = alp_rd_encode(&v);
        let decoded = alp_rd_decode(&encoded);
        assert!(vec_approx_eq(&v, &decoded));
    }

    #[test]
    fn dictionary_all_unique_values() {
        let v: Vec<f64> = (0..200).map(|i| i as f64 * 1.00123).collect();
        let encoded = alp_rd_encode(&v);
        let decoded = alp_rd_decode(&encoded);
        assert!(vec_approx_eq(&v, &decoded));
    }

    #[test]
    fn many_values_falling_into_exceptions() {
        let mut v = Vec::new();
        for i in 0..500 {
            // Choose values that differ in MSB significantly, forcing exceptions
            v.push(f64::from_bits(
                0x3FF0_0000_0000_0000 + i as u64 * 0x1000_0000_0000,
            ));
        }

        let encoded = alp_rd_encode(&v);
        let decoded = alp_rd_decode(&encoded);
        assert!(vec_approx_eq(&v, &decoded));
    }

    // ======================================================
    // 4. LARGE AND RANDOMIZED TESTING (FUZZ-LIKE)
    // ======================================================

    #[test]
    fn random_values_roundtrip() {
        for seed in 0..50u64 {
            let mut rng = seed;

            let random = |x: &mut u64| {
                *x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
                *x
            };

            let values: Vec<f64> = (0..2000)
                .map(|_| f64::from_bits(random(&mut rng)))
                .collect();

            let encoded = alp_rd_encode(&values);
            let decoded = alp_rd_decode(&encoded);

            assert!(vec_approx_eq(&values, &decoded));
            // assert!(vec_bits_eq(&values, &decoded));
        }
    }

    #[test]
    fn random_small_batches() {
        for seed in 0..200u64 {
            let mut rng = seed;
            let random = |x: &mut u64| {
                *x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
                *x
            };

            let len = (random(&mut rng) % 50) as usize + 1;

            let values: Vec<f64> = (0..len).map(|_| f64::from_bits(random(&mut rng))).collect();

            let encoded = alp_rd_encode(&values);
            let decoded = alp_rd_decode(&encoded);

            assert!(vec_approx_eq(&values, &decoded));
        }
    }

    // ======================================================
    // 5. STRESS TESTS
    // ======================================================

    #[test]
    fn very_large_input_stress_test() {
        let n = 200_000;
        let values: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();

        let encoded = alp_rd_encode(&values);
        let decoded = alp_rd_decode(&encoded);

        assert!(vec_approx_eq(&values, &decoded));
    }

    #[test]
    fn compress_monotonic_sequence() {
        let values: Vec<f64> = (0..50000).map(|i| (i as f64) * 0.000123456).collect();

        let encoded = alp_rd_encode(&values);
        let decoded = alp_rd_decode(&encoded);

        assert!(vec_approx_eq(&values, &decoded));
    }

    // ======================================================
    // 6. SERIALIZATION FORMAT INTEGRITY
    // ======================================================

    #[test]
    fn encoded_data_starts_with_correct_header() {
        let v = vec![1.1_f64, 2.2, 3.3];
        let encoded = alp_rd_encode(&v);

        let type_marker = encoded[0];
        let left_bw = encoded[1];
        let right_bw = encoded[2];
        let dict_size = encoded[3];

        assert!(type_marker == 64); // since we encoded f64
        assert!(left_bw >= 1 && left_bw <= 16);
        assert!(right_bw <= 63);
        assert!(dict_size >= 1 && dict_size <= 8);
    }

    #[test]
    fn dictionary_serialization_is_correct_length() {
        let v = vec![10.0_f64, 20.0, 30.0, 40.0];
        let encoded = alp_rd_encode(&v);

        let _ = encoded[0]; // f64 → should be 64
        let dict_size = encoded[3] as usize; // correct index

        // Header is 4 bytes:
        // [0] type marker
        // [1] left_bw
        // [2] right_bw
        // [3] dict_size
        let header_size = 4;
        let expected_dict_bytes = dict_size * 2;
        assert!(encoded.len() >= header_size + expected_dict_bytes);
    }

    // ======================================================
    // 7. DETERMINISM TESTS
    // ======================================================

    #[test]
    fn encoding_is_deterministic() {
        let v: Vec<f64> = (0..500).map(|i| (i as f64).cos()).collect();

        let e1 = alp_rd_encode(&v);
        let e2 = alp_rd_encode(&v);

        assert_eq!(e1, e2);
    }

    // ======================================================
    // 8. FORMAT REGRESSION TESTS
    // ======================================================

    #[test]
    fn roundtrip_regression_known_values() {
        let v = vec![
            3.14159_f64,
            -2.718281828,
            0.0000001,
            -0.0000001,
            123456789.123,
            -987654321.456,
        ];

        let expected_bits: Vec<u64> = v.iter().map(|x| x.to_bits()).collect();
        let encoded = alp_rd_encode::<f64>(&v);
        let decoded = alp_rd_decode::<f64>(&encoded);
        let out_bits: Vec<u64> = decoded.iter().map(|x| x.to_bits()).collect::<Vec<u64>>();
        assert_eq!(expected_bits, out_bits);
    }

    #[test]
    fn test_right_bw_min_max() {
        // Make values with extremely similar mantissas
        let mut v = Vec::new();
        for i in 0..1000 {
            v.push(f64::from_bits(0x3FF0_0000_0000_0000 + i));
        }

        let encoded = alp_rd_encode(&v);
        let decoded = alp_rd_decode(&encoded);

        assert!(vec_approx_eq(&v, &decoded));
    }

    #[test]
    fn test_randoms() {
        // let values = (0..1000).map(|i| i as f64).collect::<Vec<f64>>();
        use std::time::{SystemTime, UNIX_EPOCH};

        // Seed the generator with the current time in nanoseconds
        let mut state = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        // LCG Constants (Values from Knuth's MMIX)
        const A: u64 = 6364136223846793005;
        const C: u64 = 1442695040888963407;

        // Generate 1000 floating point numbers
        // We modify 'state' inside the map closure to generate the next number
        let numbers: Vec<f64> = (0..1024)
            .map(|_| {
                // Calculate next state: state = (a * state + c) (wrapping overflow)
                state = state.wrapping_mul(A).wrapping_add(C);

                // Convert to float between 0.0 and 1.0
                (state as f64) / (u64::MAX as f64)
            })
            .collect();

        let out = alp_rd_encode(&numbers);
        let got = alp_rd_decode(&out);
        assert!(vec_approx_eq(&numbers, &got));
    }
}
