#[cfg(test)]
mod tests {

    use rand::Rng;
    use std::sync::Arc;
    use std::thread;

    use crate::aligned::AlignedBytes;
    use crate::bucket::Bucket;
    use crate::quantizer::Quantizer;

    // ================================================================
    // TEST 1: The "Metal" Test (Memory Layout)
    // ================================================================
    #[test]
    fn test_simd_memory_alignment() {
        let capacity = 1024;
        let aligned = AlignedBytes::new(capacity);

        // Unsafe pointer arithmetic to verify alignment
        let ptr_addr = aligned.as_ptr() as usize;

        println!("Memory Address: 0x{:x}", ptr_addr);

        // Check if address is divisible by 64
        assert_eq!(
            ptr_addr % 64,
            0,
            "CRITICAL FAIL: Memory is not 64-byte aligned! AVX-512 will crash."
        );

        // Verify writing doesn't break things
        let mut aligned_mut = aligned; // Move to mut
        for i in 0..100 {
            aligned_mut.push(i as u8);
        }
        assert_eq!(aligned_mut[99], 99);
    }

    // ================================================================
    // TEST 2: The "Math" Test (ADC Accuracy)
    // ================================================================
    #[test]
    fn test_adc_quantization_accuracy() {
        let mut rng = rand::thread_rng();
        let dim = 128;

        // 1. Setup Quantizer with known bounds (-3.0 to 3.0)
        let min = vec![-3.0; dim];
        let max = vec![3.0; dim];
        let quantizer = Quantizer::new(min, max);

        // 2. Generate a random Query and a random Database Vector
        let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-2.0..2.0)).collect();
        let vec_db: Vec<f32> = (0..dim).map(|_| rng.gen_range(-2.0..2.0)).collect();

        // 3. True L2 Distance (Float32)
        let mut true_l2 = 0.0;
        for i in 0..dim {
            let diff = query[i] - vec_db[i];
            true_l2 += diff * diff;
        }

        // 4. ADC Distance (Quantized)
        // Encode DB vector only
        let code_db = quantizer.encode(&vec_db);

        // Precompute LUT for Query
        let lut = quantizer.precompute_lut(&query);

        // Scan (Simulate the inner loop)
        let mut adc_dist = 0.0;
        for i in 0..dim {
            let code_val = code_db[i];
            // LUT Lookup: Offset + ByteValue
            adc_dist += lut[i * 256 + code_val as usize];
        }

        println!("True L2: {:.4} | ADC L2: {:.4}", true_l2, adc_dist);

        // 5. Verification
        // ADC is an approximation, but it should be close.
        // We allow 5% error margin for SQ8.
        let error = (true_l2 - adc_dist).abs();
        let error_pct = error / true_l2;

        assert!(
            error_pct < 0.05,
            "Quantization Error too high! ADC Logic is flawed. Error: {:.2}%",
            error_pct * 100.0
        );
    }

    // ================================================================
    // TEST 3: The "Heat" Test (Concurrency)
    // ================================================================
    #[test]
    fn test_bucket_concurrency() {
        let dim = 128;
        let bucket = Arc::new(Bucket::new(1, 1000, dim));

        // Thread 1: The Writer (Ingest)
        let b_write = bucket.clone();
        let writer = thread::spawn(move || {
            for i in 0..100 {
                let mock_code = vec![0u8; dim];
                b_write.insert(i, &mock_code);
                // Simulate work
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        });

        // Thread 2: The Reader (Searcher checking metadata)
        let b_read = bucket.clone();
        let reader = thread::spawn(move || {
            for _ in 0..10 {
                // Check Atomic Stats (Lock-Free)
                let count = b_read.count.load(std::sync::atomic::Ordering::Relaxed);
                let temp = b_read
                    .temperature
                    .load(std::sync::atomic::Ordering::Relaxed);

                println!("Reader saw: Count={}, Temp={}", count, temp);

                // Acquire Read Lock on Data (Should not block writer significantly)
                let data = b_read.data.read();
                if data.vids.len() > 0 {
                    assert!(data.vids.len() <= 100);
                }
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        });

        writer.join().unwrap();
        reader.join().unwrap();

        // Final Consistency Check
        assert_eq!(bucket.count.load(std::sync::atomic::Ordering::SeqCst), 100);
        assert_eq!(bucket.data.read().vids.len(), 100);
    }
}
