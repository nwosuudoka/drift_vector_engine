/*
#[cfg(test)]

mod tests {
    use crate::bucket::Bucket;
    use crate::quantizer::Quantizer;
    use std::sync::Arc;
    use std::sync::atomic::Ordering;

    fn get_test_bucket() -> Bucket {
        let data = vec![vec![0.0], vec![100.0]];
        let q = Arc::new(Quantizer::train(&data));
        Bucket::new(1, 10, 1, q)
    }

    #[test]
    fn test_bucket_insert_and_extract() {
        let bucket = get_test_bucket();
        let q = &bucket.quantizer;

        // Insert Vector A: 10.0
        let vec_a = vec![10.0];
        bucket.insert(100, &q.encode(&vec_a));

        // Insert Vector B: 90.0
        let vec_b = vec![90.0];
        bucket.insert(200, &q.encode(&vec_b));

        // Extract Reconstructed (Simulating a Split operation)
        let (vecs, ids) = bucket.extract_reconstructed();

        assert_eq!(vecs.len(), 2);
        assert_eq!(ids.len(), 2);

        // Verify content (approximate due to SQ8)
        assert!((vecs[0][0] - 10.0).abs() < 1.0);
        assert_eq!(ids[0], 100);
    }

    #[test]
    fn test_bucket_tombstones_ignored_in_extract() {
        let bucket = get_test_bucket();
        let q = &bucket.quantizer;

        // Insert 3 items
        bucket.insert(1, &q.encode(&vec![10.0]));
        bucket.insert(2, &q.encode(&vec![20.0]));
        bucket.insert(3, &q.encode(&vec![30.0]));

        // Soft Delete ID 2 (Index 1)
        {
            let mut data = bucket.data.write();
            data.tombstones.insert(1); // Mark 2nd item as dead
        }

        // Extract
        let (vecs, ids) = bucket.extract_reconstructed();

        // Should only get ID 1 and 3
        assert_eq!(vecs.len(), 2);
        assert_eq!(ids, vec![1, 3]);
    }

    // This is our "Ground Truth". It is slow, simple, and obviously correct.
    fn reference_adc_scan(bucket: &Bucket, lut: &[f32]) -> Vec<(u64, f32)> {
        let data = bucket.data.read();
        let dim = bucket.quantizer.min.len();
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
        let q = Arc::new(Quantizer::train(&training_data));
        let bucket = Bucket::new(1, 1000, dim, q.clone());

        for i in 0..n_vecs {
            let vec: Vec<f32> = (0..dim).map(|d| (i + d) as f32 * 0.1).collect();
            let code = q.encode(&vec);
            bucket.insert(i as u64, &code);
        }

        let query = vec![5.0; dim];
        let lut = q.precompute_lut(&query);

        let simd_results = bucket.scan_adc(&lut, n_vecs);
        let ref_results = reference_adc_scan(&bucket, &lut);

        assert_eq!(simd_results.len(), ref_results.len());

        for (simd, refer) in simd_results.iter().zip(ref_results.iter()) {
            assert_eq!(simd.id, refer.0);

            let diff = (simd.distance - refer.1).abs();

            // Allow small variance due to SIMD addition order
            // If values are large (>1000), use relative error check
            if refer.1 > 1000.0 {
                let relative_error = diff / refer.1;
                assert!(
                    relative_error < 1e-6,
                    "Relative Error too high! ID: {}, SIMD: {}, Ref: {}, RelErr: {}",
                    simd.id,
                    simd.distance,
                    refer.1,
                    relative_error
                );
            } else {
                assert!(
                    diff < 1e-3,
                    "Absolute Error too high! ID: {}, SIMD: {}, Ref: {}, Diff: {}",
                    simd.id,
                    simd.distance,
                    refer.1,
                    diff
                );
            }
        }
    }

    // --- TEST: Alignment & Remainder Torture ---
    #[test]
    fn test_weird_dimensions() {
        // Test Prime Number dimensions to break 8-step strides
        let dims = vec![1, 3, 7, 9, 13, 31, 65];

        for dim in dims {
            let data = vec![vec![0.0; dim]; 10];
            let q = Arc::new(Quantizer::train(&data));
            let bucket = Bucket::new(1, 100, dim, q.clone());

            // Insert 10 vectors
            for i in 0..10 {
                let vec = vec![i as f32; dim];
                bucket.insert(i as u64, &q.encode(&vec));
            }

            let query = vec![1.0; dim];
            let lut = q.precompute_lut(&query);

            let results = bucket.scan_adc(&lut, 10);

            // Just verifying it doesn't Segfault/Panic and returns data
            assert_eq!(results.len(), 10, "Failed at dim {}", dim);

            // Quick spot check on distance logic
            // Vector 1 should be closer to Query(1.0) than Vector 9
            let pos_1 = results.iter().position(|r| r.id == 1).unwrap();
            let pos_9 = results.iter().position(|r| r.id == 9).unwrap();
            assert!(pos_1 < pos_9, "Ranking logic broken at dim {}", dim);
        }
    }

    // --- TEST 3: Tombstone Logic ---
    #[test]
    fn test_search_skips_tombstones() {
        let dim = 4;
        let data = vec![vec![0.0; dim]];
        let q = Arc::new(Quantizer::train(&data));
        let bucket = Bucket::new(1, 100, dim, q.clone());

        // Insert ID 0, 1, 2
        bucket.insert(0, &q.encode(&vec![0.0; dim]));
        bucket.insert(1, &q.encode(&vec![0.0; dim]));
        bucket.insert(2, &q.encode(&vec![0.0; dim]));

        // Delete ID 1
        {
            let mut d = bucket.data.write();
            d.tombstones.insert(1); // Index 1 corresponds to ID 1 here
        }

        let query = vec![0.0; dim];
        let lut = q.precompute_lut(&query);
        let results = bucket.scan_adc(&lut, 10);

        // Should retrieve 0 and 2. 1 should be missing.
        assert_eq!(results.len(), 2);
        let ids: Vec<u64> = results.iter().map(|r| r.id).collect();
        assert!(ids.contains(&0));
        assert!(ids.contains(&2));
        assert!(!ids.contains(&1), "Tombstoned ID 1 was leaked!");
    }

    #[test]
    fn test_drift_calculation_trigger() {
        // Create bucket with Capacity 10
        let bucket = get_test_bucket(); // capacity=10, dim=1, centroid=[0.0]
        let q = &bucket.quantizer;

        // 1. Insert 5 items (50% capacity) at 0.0 (No Drift)
        // should_split = False (Capacity < 80%)
        for i in 0..5 {
            bucket.insert(i, &q.encode(&vec![0.0]));
        }
        assert!(!bucket.should_split(10));

        // 2. Fill to 9 items (90% capacity) at 0.0 (No Drift)
        // should_split = False (Capacity OK, but Drift = 0.0 < 0.15)
        for i in 5..9 {
            bucket.insert(i, &q.encode(&vec![0.0]));
        }
        assert!(!bucket.should_split(10));

        // 3. Induce Drift
        // Insert a vector far away (e.g., 100.0).
        // This pulls the mean away from 0.0.
        // Mean becomes ~10.0. Distance(0, 10) = 10.0 > 0.15.
        bucket.insert(9, &q.encode(&vec![100.0]));

        // Now: Count=10 (100% > 80%) AND Drift is high.
        assert!(bucket.should_split(10), "Bucket failed to detect drift!");
    }

    fn make_bucket(dim: usize) -> Bucket {
        let data = vec![vec![0.0; dim]];
        let q = Arc::new(Quantizer::train(&data));
        Bucket::new(1, 100, dim, q)
    }

    #[test]
    fn test_heat_protection_mechanism() {
        let b = make_bucket(2);

        // Setup: Bucket is mostly empty (Live=10, Target=100) -> Emptiness = 0.9
        b.count.store(10, Ordering::Relaxed);

        // Case 1: Cold Bucket
        // Explicitly set Temp to 0.0 (it defaults to 1.0 in new())
        b.temperature.store(0.0, Ordering::Relaxed);
        let u_cold = b.calculate_urgency(100);

        // Case 2: Hot Bucket (Temp = 1.0)
        b.temperature.store(1.0, Ordering::Relaxed);
        let u_hot = b.calculate_urgency(100);

        // Debug prints to confirm
        println!("Cold Urgency: {}, Hot Urgency: {}", u_cold, u_hot);

        // Assertion: Heat MUST reduce urgency
        // Cold (0.9 / 0.001) = 900.0
        // Hot (0.9 / 1.001) = 0.89
        assert!(u_hot < u_cold, "Heat failed to protect the bucket!");
        assert!(u_hot < 1.0);
    }

    #[test]
    fn test_hot_zombie_paradox() {
        let b = make_bucket(2);

        // Setup: Bucket is full of dead data (Count=100, Dead=90)
        // Live = 10. Emptiness = 0.9.
        // ZombieRatio = 0.9.
        b.count.store(100, Ordering::Relaxed);
        b.tombstone_count.store(90, Ordering::Relaxed);

        // Even if Hot (Temp=1.0), the Zombie Ratio should force a kill.
        b.temperature.store(1.0, Ordering::Relaxed);

        // Term 1 (Heat Shield): 0.9 / 1.0 = 0.9
        // Term 2 (Death): 3.0 * 0.9 = 2.7
        // Total = 3.6

        let u = b.calculate_urgency(100);

        // Threshold is usually 1.5. 3.6 is way above.
        assert!(u > 1.5, "Hot Zombie should still be killed");
    }
}
*/
