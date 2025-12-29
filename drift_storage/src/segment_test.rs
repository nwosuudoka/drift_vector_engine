#[cfg(test)]
mod tests {
    use crate::compression::wrapper::CompressionStrategy;
    use crate::disk_manager::DiskManager;
    use crate::segment_reader::SegmentReader;
    use crate::segment_writer::{CompressionType, SegmentWriter};
    use rand::Rng;
    use tempfile::tempdir;

    // --- Helpers ---

    fn generate_random_vectors(count: usize, dim: usize) -> Vec<Vec<f32>> {
        let mut rng = rand::rng();
        (0..count)
            .map(|_| (0..dim).map(|_| rng.random::<f32>()).collect())
            .collect()
    }

    /// Simulates a Quantizer producing SQ8 codes from floats.
    /// Just maps float -> u8 via simple scaling for testing storage.
    fn mock_quantize(vecs: &[Vec<f32>]) -> Vec<u8> {
        let mut codes = Vec::with_capacity(vecs.len() * vecs[0].len());
        for v in vecs {
            for &val in v {
                // Mock quantization: (val * 10.0) % 255
                codes.push(((val * 10.0) as u32 % 255) as u8);
            }
        }
        codes
    }

    // --- Test 1: The "Dual-Tier" Flush (L0 -> L1) ---
    // This is the most critical path: persisting fresh data.
    #[tokio::test]
    async fn test_dual_tier_flush_and_read() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("dual_tier.drift");
        let uri = format!("file://{}", file_path.to_str().unwrap());

        let dim = 128;
        let count = 100;
        let bucket_id = 1;

        // 1. Generate Data
        let raw_vectors = generate_random_vectors(count, dim);
        let ids: Vec<u64> = (0..count as u64).collect();
        let sq8_codes = mock_quantize(&raw_vectors); // Simulate Quantizer output

        // Mock Quantizer Config (Metadata)
        let q_config = vec![0xAA, 0xBB, 0xCC];

        // 2. Write Segment (Dual Tier)
        {
            let manager = DiskManager::open(&uri).await.unwrap();
            let mut writer = SegmentWriter::new(manager, q_config.clone()).await.unwrap();

            writer
                .write_bucket_dual(bucket_id, &ids, &raw_vectors, &sq8_codes, dim)
                .await
                .expect("Write failed");

            writer.finalize().await.expect("Finalize failed");
        }

        // 3. Read Back
        let reader = SegmentReader::open(&uri).await.expect("Open failed");

        // A. Verify Metadata
        assert_eq!(reader.read_metadata(), &q_config);

        // B. Verify Index Location Data
        let loc = reader.index.buckets.get(&bucket_id).unwrap();
        assert_eq!(loc.vector_count, count);
        assert_eq!(loc.compression_type, CompressionType::Compressed as u8);
        assert!(loc.index_length > 0); // SQ8 Blob exists
        assert!(loc.data_length > 0); // ALP Blob exists

        // C. FAST PATH: Read SQ8 (Hot Cache Load)
        let (read_ids, read_codes) = reader.read_bucket(bucket_id).await.unwrap();
        assert_eq!(read_ids, ids);
        assert_eq!(read_codes, sq8_codes, "SQ8 Index blob corrupted");

        // D. COLD PATH: Read Floats (Re-ranking / Recovery)
        let read_floats = reader.read_bucket_high_fidelity(bucket_id).await.unwrap();
        assert_eq!(read_floats.len(), count);

        // Check fuzzy equality (ALP is lossy-ish depending on float precision, but usually exact for simple randoms)
        let f1 = &raw_vectors[0];
        let f2 = &read_floats[0];
        for (a, b) in f1.iter().zip(f2.iter()) {
            assert!((a - b).abs() < 1e-6, "Float data mismatch: {} vs {}", a, b);
        }
    }

    // --- Test 2: The "Maintenance" Path (L1 -> L1) ---
    // This tests Split/Merge where we only move SQ8 codes.
    #[tokio::test]
    async fn test_sq8_only_maintenance_write() {
        let dir = tempdir().unwrap();
        let uri = format!(
            "file://{}",
            dir.path().join("maintenance.drift").to_str().unwrap()
        );

        let dim = 64;
        let count = 50;
        let bucket_id = 99;

        // Data
        let ids: Vec<u64> = (1000..1050).collect();
        let sq8_codes = vec![0x7F; count * dim]; // All grey values

        // 1. Write Segment (SQ8 Only)
        {
            let manager = DiskManager::open(&uri).await.unwrap();
            let mut writer = SegmentWriter::new(manager, vec![]).await.unwrap();

            writer
                .write_bucket_sq8(bucket_id, &ids, &sq8_codes, dim)
                .await
                .expect("Write SQ8 failed");

            writer.finalize().await.unwrap();
        }

        // 2. Read Back
        let reader = SegmentReader::open(&uri).await.unwrap();
        let loc = reader.index.buckets.get(&bucket_id).unwrap();

        // Assertions
        assert_eq!(loc.compression_type, CompressionType::RawSQ8 as u8);
        assert_eq!(loc.data_length, 0, "Should have 0 data length for RawSQ8");

        // Fast Path should work
        let (read_ids, read_codes) = reader.read_bucket(bucket_id).await.unwrap();
        assert_eq!(read_ids, ids);
        assert_eq!(read_codes, sq8_codes);

        // Cold Path should FAIL safely
        let err = reader.read_bucket_high_fidelity(bucket_id).await;
        assert!(err.is_err());
        assert_eq!(err.unwrap_err().kind(), std::io::ErrorKind::InvalidData);
    }

    // --- Test 3: Validation & Errors ---
    #[tokio::test]
    async fn test_validations() {
        let dir = tempdir().unwrap();
        let uri = format!("file://{}", dir.path().join("bad.drift").to_str().unwrap());
        let manager = DiskManager::open(&uri).await.unwrap();
        let mut writer = SegmentWriter::new(manager, vec![]).await.unwrap();

        // 1. Dimension Mismatch
        let ids = vec![1];
        let codes = vec![0u8; 10]; // 10 bytes
        let dim = 128; // Expect 128 bytes

        let res = writer.write_bucket_sq8(1, &ids, &codes, dim).await;
        assert!(res.is_err());
        assert_eq!(res.unwrap_err().to_string(), "Dimension mismatch");

        // 2. ID Collision
        let ids_ok = vec![1];
        let codes_ok = vec![0u8; 10];
        let dim_ok = 10;

        writer
            .write_bucket_sq8(1, &ids_ok, &codes_ok, dim_ok)
            .await
            .unwrap();

        // Try writing Bucket 1 again
        let res_col = writer.write_bucket_sq8(1, &ids_ok, &codes_ok, dim_ok).await;
        assert!(res_col.is_err());
        assert_eq!(
            res_col.unwrap_err().kind(),
            std::io::ErrorKind::AlreadyExists
        );
    }

    // --- Test 4: Bloom Filter Integration ---
    #[tokio::test]
    async fn test_bloom_filter_persistence() {
        let dir = tempdir().unwrap();
        let uri = format!(
            "file://{}",
            dir.path().join("bloom.drift").to_str().unwrap()
        );

        let ids = vec![42, 999, 10_000];
        let dim = 4;
        let codes = vec![0u8; ids.len() * dim];

        // Write
        {
            let manager = DiskManager::open(&uri).await.unwrap();
            let mut writer = SegmentWriter::new(manager, vec![]).await.unwrap();
            writer.write_bucket_sq8(1, &ids, &codes, dim).await.unwrap();
            writer.finalize().await.unwrap();
        }

        // Read
        let reader = SegmentReader::open(&uri).await.unwrap();

        // Check Positive
        assert!(reader.might_contain(42));
        assert!(reader.might_contain(999));

        // Check Negative
        assert!(!reader.might_contain(1));
        assert!(!reader.might_contain(500));
    }
}
