#[cfg(test)]
mod tests {
    use crate::bucket_file_reader::BucketFileReader;
    use crate::bucket_file_writer::BucketFileWriter;
    use drift_core::quantizer::Quantizer;
    use drift_traits::TombstoneView;
    use opendal::{Operator, services};
    use std::fs::OpenOptions;
    use tempfile::NamedTempFile;

    // --- Mocks & Helpers ---

    #[derive(Debug)]
    struct NoTombstones;
    impl TombstoneView for NoTombstones {
        fn contains(&self, _id: u64) -> bool {
            false
        }
        fn len(&self) -> usize {
            0
        }
    }

    fn create_fs_operator(root: &str) -> Operator {
        let builder = services::Fs::default().root(root);
        Operator::new(builder).unwrap().finish()
    }

    fn mock_batch(start_id: u64, count: usize, dim: usize) -> (Vec<u64>, Vec<f32>) {
        let mut ids = Vec::with_capacity(count);
        let mut vecs = Vec::with_capacity(count * dim);
        for i in 0..count {
            ids.push(start_id + i as u64);
            // Distinct Pattern: ID 0 -> [0.0, ...], ID 1 -> [1.0, ...]
            // This ensures every vector is distinct for quantization.
            let val = (start_id + i as u64) as f32;
            vecs.extend(std::iter::repeat(val).take(dim));
        }
        (ids, vecs)
    }

    // --- TEST 1: Write -> Read Round Trip (Maintenance Path) ---
    #[tokio::test]
    async fn test_write_read_round_trip() {
        let tmp_file = NamedTempFile::new().unwrap();
        let path = tmp_file.path().to_owned();
        let dim = 8;
        let count = 100;

        // 1. Write Data
        let (ids_orig, vecs_orig) = mock_batch(0, count, dim);
        let q = Quantizer::train(&vecs_orig, dim);

        {
            let file = OpenOptions::new().write(true).open(&path).unwrap();
            let mut writer =
                BucketFileWriter::new_streaming(file, [1u8; 16], q.clone(), dim).unwrap();
            writer.write_batch(&ids_orig, &vecs_orig).unwrap();
            writer.finalize().unwrap();
        }

        // 2. Read Data
        let root = path.parent().unwrap().to_str().unwrap();
        let filename = path.file_name().unwrap().to_str().unwrap();
        let op = create_fs_operator(root);

        let mut reader = BucketFileReader::open(op, filename).await.unwrap();
        let (ids_read, vecs_read) = reader.read_all_vectors().await.unwrap();

        // 3. Verify
        assert_eq!(ids_read, ids_orig, "IDs must match exactly");

        // Flatten for float comparison
        let flat_read: Vec<f32> = vecs_read.into_iter().flatten().collect();
        assert_eq!(flat_read.len(), vecs_orig.len());

        for (i, (a, b)) in flat_read.iter().zip(vecs_orig.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-3,
                "Vector mismatch at index {}: {} vs {}",
                i,
                a,
                b
            );
        }
    }

    // --- TEST 2: Append -> Read All (Integrity Check) ---
    #[tokio::test]
    async fn test_append_integrity() {
        let tmp_file = NamedTempFile::new().unwrap();
        let path = tmp_file.path().to_owned();
        let dim = 4;

        // Batch 1: IDs 0..50
        let (ids_1, vecs_1) = mock_batch(0, 50, dim);
        let q = Quantizer::train(&vecs_1, dim);

        // A. Create & Write Batch 1
        {
            let file = OpenOptions::new().write(true).open(&path).unwrap();
            let mut writer =
                BucketFileWriter::new_streaming(file, [1u8; 16], q.clone(), dim).unwrap();
            writer.write_batch(&ids_1, &vecs_1).unwrap();
            writer.finalize().unwrap();
        }

        // Batch 2: IDs 50..100
        let (ids_2, vecs_2) = mock_batch(50, 50, dim);

        // B. Open Append & Write Batch 2
        {
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)
                .unwrap();
            let len = file.metadata().unwrap().len();

            let mut writer =
                BucketFileWriter::new_append(file, [1u8; 16], q.clone(), dim, len).unwrap();

            // Verify internal state recovery
            assert_eq!(writer.get_total_count(), 50);

            writer.write_batch(&ids_2, &vecs_2).unwrap();
            let (_, total) = writer.finalize_and_truncate().unwrap();
            assert_eq!(total, 100);
        }

        // C. Read Back Everything
        let root = path.parent().unwrap().to_str().unwrap();
        let filename = path.file_name().unwrap().to_str().unwrap();
        let op = create_fs_operator(root);

        let mut reader = BucketFileReader::open(op, filename).await.unwrap();

        // Verify Footer via Reader
        assert_eq!(reader.footer.total_vector_count, 100);
        assert_eq!(reader.footer.row_group_count, 2);

        // Verify Data
        let (all_ids, _) = reader.read_all_vectors().await.unwrap();
        assert_eq!(all_ids.len(), 100);
        assert_eq!(all_ids[0], 0);
        assert_eq!(all_ids[99], 99);
    }

    // --- TEST 3: Search Accuracy (Scan Path) ---
    #[tokio::test]
    async fn test_search_accuracy() {
        let tmp_file = NamedTempFile::new().unwrap();
        let path = tmp_file.path().to_owned();
        let dim = 16;
        let count = 1000;

        // 1. Generate Robust Data
        // ID 500 is the "Target": [100.0, 100.0...]
        // Everyone else: [0.0, 0.0...]
        // This large delta ensures quantization puts them in different buckets.
        let mut ids = Vec::new();
        let mut vecs = Vec::new();
        for i in 0..count {
            ids.push(i as u64);
            if i == 500 {
                vecs.extend(vec![100.0f32; dim]);
            } else {
                vecs.extend(vec![0.0f32; dim]);
            }
        }

        // Train quantizer on this distribution
        let q = Quantizer::train(&vecs, dim);

        // 2. Write File
        {
            let file = OpenOptions::new().write(true).open(&path).unwrap();
            let mut writer = BucketFileWriter::new_streaming(file, [1u8; 16], q, dim).unwrap();
            writer.write_batch(&ids, &vecs).unwrap();
            writer.finalize().unwrap();
        }

        // 3. Search
        let root = path.parent().unwrap().to_str().unwrap();
        let filename = path.file_name().unwrap().to_str().unwrap();
        let op = create_fs_operator(root);

        let mut reader = BucketFileReader::open(op, filename).await.unwrap();

        // Query: Exact match for ID 500
        let query = vec![100.0f32; dim];
        let tombstones = NoTombstones;

        let results = reader.scan(&query, 10, &tombstones).await.unwrap();

        // 4. Verify
        assert!(!results.is_empty(), "Results should not be empty");

        // Verify Top 1
        let top = &results[0];
        println!("Top Result ID: {}, Dist: {}", top.id, top.approx_dist);

        assert_eq!(top.id, 500, "ID 500 should be the closest match");

        // Because of the massive gap (0 vs 100), SQ8 should easily distinguish them.
        // Target distance should be ~0. Others should be huge.
        assert!(top.approx_dist < 10.0, "Expected low distance for target");

        // Ensure others are far away
        if results.len() > 1 {
            assert!(results[1].approx_dist > 100.0, "Distractors should be far");
        }
    }
}
