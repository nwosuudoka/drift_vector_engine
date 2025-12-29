#[cfg(test)]
mod tests {
    use crate::aligned::AlignedBytes;
    use crate::bucket::{Bucket, BucketData, BucketHeader};
    use crate::quantizer::Quantizer;
    use bit_set::BitSet;
    use drift_traits::PageId;
    use std::sync::Arc;
    use std::sync::atomic::Ordering;

    // --- Helpers ---

    /// Helper to create a fully populated BucketData struct from raw vectors.
    fn create_bucket_data(q: &Quantizer, vecs: &[Vec<f32>]) -> BucketData {
        let dim = q.min.len();
        let count = vecs.len();

        let mut codes = AlignedBytes::new(count * dim);
        let mut vids = Vec::with_capacity(count);
        let tombstones = BitSet::with_capacity(count);

        for (i, vec) in vecs.iter().enumerate() {
            let code = q.encode(vec);
            for b in code {
                codes.push(b);
            }
            vids.push(i as u64); // ID = Index
        }

        BucketData {
            codes,
            vids,
            tombstones,
        }
    }

    /// Helper to create a dummy header for urgency tests
    fn create_header(count: u32) -> BucketHeader {
        BucketHeader::new(
            1,
            vec![],
            count,
            PageId {
                file_id: 0,
                offset: 0,
                length: 0,
            },
        )
    }

    // --- Tests ---

    #[test]
    fn test_bucket_data_reconstruct() {
        let data_raw = vec![vec![0.0], vec![100.0]];
        let q = Quantizer::train(&data_raw);

        // Simulate creating a bucket with 2 vectors
        let vec_a = vec![10.0];
        let vec_b = vec![90.0];
        let bucket_data = create_bucket_data(&q, &[vec_a, vec_b]);

        // Extract Reconstructed
        let (vecs, ids) = bucket_data.reconstruct(&q);

        assert_eq!(vecs.len(), 2);
        assert_eq!(ids.len(), 2);

        // Verify content (approximate due to SQ8)
        assert!((vecs[0][0] - 10.0).abs() < 1.0);
        assert_eq!(ids[0], 0);
    }

    #[test]
    fn test_bucket_tombstones_ignored_in_extract() {
        let data_raw = vec![vec![0.0]];
        let q = Quantizer::train(&data_raw);

        let input = vec![vec![10.0], vec![20.0], vec![30.0]];
        let mut bucket_data = create_bucket_data(&q, &input);

        // Soft Delete ID 1 (Value 20.0)
        bucket_data.tombstones.insert(1);

        // Extract
        let (vecs, ids) = bucket_data.reconstruct(&q);

        // Should only get ID 0 and 2
        assert_eq!(vecs.len(), 2);
        assert_eq!(ids, vec![0, 2]);
        assert!((vecs[0][0] - 10.0).abs() < 1.0);
        assert!((vecs[1][0] - 30.0).abs() < 1.0);
    }

    // This is our "Ground Truth" for LUT scanning.
    // It manually calculates distance using the LUT to verify the SIMD/Unrolled kernel.
    fn reference_lut_scan(data: &BucketData, lut: &[f32], dim: usize) -> Vec<(u64, f32)> {
        let mut results = Vec::new();

        for (i, &vid) in data.vids.iter().enumerate() {
            if data.tombstones.contains(i) {
                continue;
            }

            let start = i * dim;
            let code = &data.codes.as_slice()[start..start + dim];
            let mut dist = 0.0;

            for d in 0..dim {
                let code_byte = code[d] as usize;
                // Reference Look-up Logic
                let lut_idx = (d * 256) + code_byte;
                dist += lut[lut_idx];
            }
            results.push((vid, dist));
        }

        // Sort by distance (ASC)
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results
    }

    // --- TEST: SIMD Correctness (Fixed Tolerance) ---
    #[test]
    fn test_simd_vs_scalar_accuracy() {
        let dim = 128;
        let n_vecs = 100;

        let training_data: Vec<Vec<f32>> = (0..50).map(|i| vec![i as f32; dim]).collect();
        let q = Quantizer::train(&training_data);

        // Generate encoded data
        let mut vectors = Vec::new();
        for i in 0..n_vecs {
            let vec: Vec<f32> = (0..dim).map(|d| (i + d) as f32 * 0.1).collect();
            vectors.push(vec);
        }
        let bucket_data = create_bucket_data(&q, &vectors);

        // Create LUT
        let query = vec![5.0; dim];
        let lut = q.precompute_lut(&query);

        // Run Optimized Kernel
        let simd_results = Bucket::scan_with_lut(&bucket_data, &lut, dim);

        // Run Reference Implementation
        let ref_results = reference_lut_scan(&bucket_data, &lut, dim);

        assert_eq!(simd_results.len(), ref_results.len());

        for (simd, refer) in simd_results.iter().zip(ref_results.iter()) {
            assert_eq!(simd.id, refer.0);

            let diff = (simd.distance - refer.1).abs();

            // Allow small variance due to floating point association order
            if refer.1 > 1000.0 {
                let relative_error = diff / refer.1;
                assert!(
                    relative_error < 1e-5,
                    "Relative Error too high! ID: {}, SIMD: {}, Ref: {}",
                    simd.id,
                    simd.distance,
                    refer.1
                );
            } else {
                assert!(
                    diff < 1e-3,
                    "Absolute Error too high! ID: {}, SIMD: {}, Ref: {}",
                    simd.id,
                    simd.distance,
                    refer.1
                );
            }
        }
    }

    // --- TEST: Alignment & Remainder Torture ---
    #[test]
    fn test_weird_dimensions() {
        // Test Prime Number dimensions to break 4-step unrolling
        let dims = vec![1, 3, 7, 9, 13, 31, 65];

        for dim in dims {
            let data_raw = vec![vec![0.0; dim]; 10];
            let q = Quantizer::train(&data_raw);

            let mut vectors = Vec::new();
            for i in 0..10 {
                vectors.push(vec![i as f32; dim]);
            }
            let bucket_data = create_bucket_data(&q, &vectors);

            let query = vec![1.0; dim];
            let lut = q.precompute_lut(&query);

            let results = Bucket::scan_with_lut(&bucket_data, &lut, dim);

            // Just verifying it doesn't Segfault/Panic and returns data
            assert_eq!(results.len(), 10, "Failed at dim {}", dim);

            // Check ranking: Vector 1 (1.0) should be closest to Query (1.0)
            let pos_1 = results.iter().position(|r| r.id == 1).unwrap();
            let pos_9 = results.iter().position(|r| r.id == 9).unwrap();
            assert!(pos_1 < pos_9, "Ranking logic broken at dim {}", dim);
        }
    }

    // --- TEST 3: Tombstone Logic ---
    #[test]
    fn test_search_skips_tombstones() {
        let dim = 4;
        let data_raw = vec![vec![0.0; dim]];
        let q = Quantizer::train(&data_raw);

        let input = vec![vec![0.0; dim], vec![0.0; dim], vec![0.0; dim]]; // IDs 0, 1, 2
        let mut bucket_data = create_bucket_data(&q, &input);

        // Delete ID 1
        bucket_data.tombstones.insert(1);

        let query = vec![0.0; dim];
        let lut = q.precompute_lut(&query);

        let results = Bucket::scan_with_lut(&bucket_data, &lut, dim);

        // Should retrieve 0 and 2. 1 should be missing.
        assert_eq!(results.len(), 2);
        let ids: Vec<u64> = results.iter().map(|r| r.id).collect();
        assert!(ids.contains(&0));
        assert!(ids.contains(&2));
        assert!(!ids.contains(&1), "Tombstoned ID 1 was leaked!");
    }

    #[test]
    fn test_heat_protection_mechanism() {
        let h = create_header(10); // Live=10, Target=100 (Implicitly)

        // Case 1: Cold Bucket
        h.stats.temperature.store(0.0, Ordering::Relaxed);
        let u_cold = h.calculate_urgency(100);

        // Case 2: Hot Bucket
        h.stats.temperature.store(1.0, Ordering::Relaxed);
        let u_hot = h.calculate_urgency(100);

        println!("Cold Urgency: {}, Hot Urgency: {}", u_cold, u_hot);

        // Assertion: Heat MUST reduce urgency
        // Cold (0.9 / 0.001) = 900.0
        // Hot (0.9 / 1.001) = 0.89
        assert!(u_hot < u_cold, "Heat failed to protect the bucket!");
        assert!(u_hot < 1.0);
    }

    #[test]
    fn test_hot_zombie_paradox() {
        let h = create_header(100); // Total 100

        // Setup: Bucket is full of dead data (Count=100, Dead=90)
        h.stats.tombstone_count.store(90, Ordering::Relaxed);

        // Even if Hot (Temp=1.0), the Zombie Ratio should force a kill.
        h.stats.temperature.store(1.0, Ordering::Relaxed);

        // Live = 10. Emptiness = (100 - 10)/100 = 0.9.
        // Term 1 (Heat Shield): 0.9 / (1.0 + epsilon) ~= 0.9
        // Term 2 (Death): 3.0 * (90/100) = 2.7
        // Total ~= 3.6

        let u = h.calculate_urgency(100);

        // Threshold is usually 1.5. 3.6 is way above.
        assert!(
            u > 1.5,
            "Hot Zombie should still be killed (Urgency: {})",
            u
        );
    }
}
