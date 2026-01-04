#[cfg(test)]
mod tests {
    use crate::disk_manager::DiskManager;
    use crate::segment_reader::SegmentReader;
    use crate::segment_writer::SegmentWriter;
    use bit_set::BitSet;
    use drift_core::aligned::AlignedBytes; // ⚡ Need these
    use drift_core::bucket::BucketData; // ⚡ Need these
    use opendal::{Operator, services};
    use rand::Rng;
    use tempfile::tempdir;

    /// Helper to generate random vectors
    fn generate_vectors(count: usize, dim: usize) -> Vec<Vec<f32>> {
        let mut rng = rand::rng();
        (0..count)
            .map(|_| (0..dim).map(|_| rng.random::<f32>()).collect())
            .collect()
    }

    /// Helper to generate dummy SQ8 codes
    fn generate_sq8_blob(count: usize, dim: usize, pattern: u8) -> Vec<u8> {
        vec![pattern; count * dim]
    }

    /// ⚡ HELPER: Wraps raw data into a valid BucketData struct
    /// This is required because SegmentWriter::write_partition expects this structure.
    fn wrap_in_bucket(ids: &[u64], codes: &[u8]) -> BucketData {
        BucketData {
            codes: AlignedBytes::from_slice(codes),
            vids: ids.to_vec(),
            tombstones: BitSet::new(),
        }
    }

    // Create local filesystem operator
    fn create_local_operator(path: &std::path::Path) -> Operator {
        let mut builder = services::Fs::default();
        builder = builder.root(path.to_str().unwrap());
        Operator::new(builder).unwrap().finish()
    }

    #[tokio::test]
    async fn test_opendal_local_fs_flow() {
        let dir = tempdir().unwrap();
        let op = create_local_operator(dir.path());
        let filename = "test_seg.drift";

        let dim = 2;
        // Data setup
        let ids = vec![100, 200];
        let vecs = vec![vec![1.0, 1.0], vec![2.0, 2.0]];
        let codes = vec![0x10, 0x10, 0x20, 0x20]; // Mock SQ8 codes

        // 1. Write
        let manager = DiskManager::new(op.clone(), filename.to_string());
        let mut writer = SegmentWriter::new(manager, vec![0x01, 0x02]).await.unwrap();

        // ⚡ FIX: Use write_partition with BucketData
        let bucket = wrap_in_bucket(&ids, &codes);
        writer
            .write_partition(1, &bucket, &vecs, dim)
            .await
            .unwrap();

        writer.finalize().await.unwrap();

        // 2. Read
        let reader = SegmentReader::open_with_op(op.clone(), filename)
            .await
            .unwrap();

        assert_eq!(reader.read_metadata(), &[0x01, 0x02]);
        assert!(reader.might_contain(100));

        // A. Verify Fast Path (SQ8 Blob)
        let (read_ids, read_codes) = reader.read_bucket(1).await.unwrap();
        assert_eq!(read_ids, ids);
        assert_eq!(read_codes, codes, "SQ8 Index Blob corrupted");

        // B. Verify Cold Path (High Fidelity Floats)
        let read_vecs = reader.read_bucket_high_fidelity(1).await.unwrap();
        assert_eq!(read_vecs.len(), 2);

        // Check lossless roundtrip (ALP should match exactly for simple integers)
        assert_eq!(read_vecs[0][0], 1.0);
        assert_eq!(read_vecs[1][0], 2.0);
    }

    #[tokio::test]
    async fn test_heavy_integration_scenario() {
        let dir = tempdir().unwrap();
        let op = create_local_operator(dir.path());
        let filename = "heavy_segment.drift";

        // Setup Data
        let dim = 128;
        let quantizer_config = vec![1, 2, 3, 4, 5];

        // Bucket 1: Small, sequential IDs
        let b1_id = 10;
        let b1_ids = vec![100, 101, 102];
        let b1_vecs = generate_vectors(3, dim);
        let b1_codes = generate_sq8_blob(3, dim, 0xAA);

        // Bucket 2: Large, random IDs
        let b2_id = 20;
        let b2_ids: Vec<u64> = (2000..3000).collect();
        let b2_vecs = generate_vectors(1000, dim);
        let b2_codes = generate_sq8_blob(1000, dim, 0xBB);

        // Bucket 3: Edge case (Single vector)
        let b3_id = 30;
        let b3_ids = vec![99999];
        let b3_vecs = generate_vectors(1, dim);
        let b3_codes = generate_sq8_blob(1, dim, 0xCC);

        // --- WRITE PHASE ---
        {
            let manager = DiskManager::new(op.clone(), filename.to_string());
            let mut writer = SegmentWriter::new(manager, quantizer_config.clone())
                .await
                .unwrap();

            // ⚡ FIX: Wrap all in BucketData and use write_partition
            let b1 = wrap_in_bucket(&b1_ids, &b1_codes);
            writer
                .write_partition(b1_id, &b1, &b1_vecs, dim)
                .await
                .unwrap();

            let b2 = wrap_in_bucket(&b2_ids, &b2_codes);
            writer
                .write_partition(b2_id, &b2, &b2_vecs, dim)
                .await
                .unwrap();

            let b3 = wrap_in_bucket(&b3_ids, &b3_codes);
            writer
                .write_partition(b3_id, &b3, &b3_vecs, dim)
                .await
                .unwrap();

            writer.finalize().await.unwrap();
        }

        // --- READ PHASE ---
        let reader = SegmentReader::open_with_op(op.clone(), filename)
            .await
            .unwrap();

        // 1. Verify Metadata
        assert_eq!(
            reader.read_metadata(),
            &quantizer_config,
            "Quantizer config corrupted"
        );

        // 2. Verify Bloom Filter
        assert!(reader.might_contain(100));
        assert!(reader.might_contain(2500));
        assert!(reader.might_contain(99999));
        assert!(!reader.might_contain(1));
        assert!(!reader.might_contain(500000));

        // 3. Verify Bucket 1 (Fast Path)
        let (ids, codes) = reader.read_bucket(b1_id).await.unwrap();
        assert_eq!(ids, b1_ids);
        assert_eq!(codes, b1_codes);

        // 4. Verify Bucket 2 (Cold Path - High Fidelity)
        let vecs = reader.read_bucket_high_fidelity(b2_id).await.unwrap();
        assert_eq!(vecs.len(), 1000);
        // Spot check
        for i in 0..dim {
            assert!(
                (vecs[0][i] - b2_vecs[0][i]).abs() < 1e-5,
                "Vector data mismatch in Bucket 2"
            );
        }

        // 5. Verify Bucket 3 (Fast Path)
        let (ids, codes) = reader.read_bucket(b3_id).await.unwrap();
        assert_eq!(ids, b3_ids);
        assert_eq!(codes, b3_codes);

        // 6. Verify Error Handling
        let err = reader.read_bucket(999).await;
        assert!(err.is_err());
        assert_eq!(err.unwrap_err().kind(), std::io::ErrorKind::NotFound);
    }

    #[tokio::test]
    async fn test_empty_write_handling() {
        let dir = tempdir().unwrap();
        let op = create_local_operator(dir.path());
        let filename = "empty.drift";

        // Write a segment with NO buckets
        {
            let manager = DiskManager::new(op.clone(), filename.to_string());
            let writer = SegmentWriter::new(manager, vec![]).await.unwrap();
            writer.finalize().await.unwrap();
        }

        // Read it back
        let reader = SegmentReader::open_with_op(op.clone(), filename)
            .await
            .unwrap();
        assert!(reader.read_metadata().is_empty());

        // Should handle looking up non-existent bucket gracefully
        assert!(reader.read_bucket(1).await.is_err());
    }
}
