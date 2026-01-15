#[cfg(test)]
mod tests {
    use crate::bucket_file_reader::BucketFileReader;
    use crate::bucket_file_writer::BucketFileWriter;
    use drift_core::quantizer::Quantizer;
    use opendal::{Operator, services};
    use std::fs::File;
    use tempfile::tempdir;

    fn create_local_operator(path: &std::path::Path) -> Operator {
        let builder = services::Fs::default().root(path.to_str().unwrap());
        Operator::new(builder).unwrap().finish()
    }

    fn mock_vectors(count: usize, dim: usize, val: f32) -> (Vec<u64>, Vec<Vec<f32>>) {
        let ids: Vec<u64> = (0..count as u64).collect();
        let vecs: Vec<Vec<f32>> = (0..count).map(|_| vec![val; dim]).collect();
        (ids, vecs)
    }

    #[tokio::test]
    async fn test_reproduce_merge_durability() {
        let dir = tempdir().unwrap();
        let op = create_local_operator(dir.path());
        let dim = 64; // Match the panic type marker check (f64=64)

        // 1. Data Setup
        let (ids, vecs) = mock_vectors(100, dim, 1.23);
        let flat_vecs: Vec<f32> = vecs.clone().into_iter().flatten().collect();
        let q = Quantizer::train(&flat_vecs, dim);

        let filename = "durability_test.drift";
        let path = dir.path().join(filename);

        // 2. WRITE (Streaming Mode - simulating Promotion)
        {
            let file = File::create(&path).unwrap();
            let mut writer =
                BucketFileWriter::new_streaming(file, [1u8; 16], q.clone(), dim).unwrap();
            writer.write_batch(&ids, &flat_vecs).unwrap();

            // ⚡ This MUST call sync_all() internally, or the test might fail/panic
            writer.finalize().unwrap();
        }

        // 3. READ (Simulating Searcher)
        let mut reader = BucketFileReader::open(op, filename).await.unwrap();

        // Use the correct API for BucketFileReader
        let result = reader.read_all_vectors().await;

        match result {
            Ok((_, vectors)) => {
                assert_eq!(vectors.len(), 100);
                assert_eq!(vectors[0][0], 1.23);
            }
            Err(e) => {
                // If we panic inside ALP, this won't even be reached (test crashes).
                // If we fix the panic but don't fix the sync, we get an IO error or empty vec.
                panic!("Read failed gracefully but data was lost: {}", e);
            }
        }
    }
}

#[cfg(test)]
mod mix_node_test {
    use crate::bucket_file_reader::BucketFileReader;
    use crate::bucket_file_writer::BucketFileWriter;
    use drift_core::quantizer::Quantizer;
    use opendal::{Operator, services};
    use std::fs::{File, OpenOptions};
    use std::io::{Seek, SeekFrom};
    use tempfile::tempdir;

    // --- Helpers ---
    fn create_local_operator(path: &std::path::Path) -> Operator {
        let builder = services::Fs::default().root(path.to_str().unwrap());
        Operator::new(builder).unwrap().finish()
    }

    fn mock_vectors(
        start_id: u64,
        count: usize,
        dim: usize,
        val_offset: f32,
    ) -> (Vec<u64>, Vec<Vec<f32>>) {
        let ids: Vec<u64> = (0..count as u64).map(|i| start_id + i).collect();
        let vecs: Vec<Vec<f32>> = (0..count)
            .map(|i| vec![(i as f32) + val_offset; dim])
            .collect();
        (ids, vecs)
    }

    #[tokio::test]
    async fn test_streaming_then_append_durability() {
        let dir = tempdir().unwrap();
        let op = create_local_operator(dir.path());
        let filename = "mixed_mode.drift";
        let path = dir.path().join(filename);
        let dim = 8;

        // Shared Quantizer (Must be consistent across appends)
        let (train_ids, train_vecs) = mock_vectors(0, 100, dim, 0.0);
        let train_flat: Vec<f32> = train_vecs.into_iter().flatten().collect();
        let quantizer = Quantizer::train(&train_flat, dim);

        // =========================================================================
        // PHASE 1: STREAMING WRITE (Initial Creation)
        // =========================================================================
        println!("Phase 1: Streaming Create...");
        let (ids_1, vecs_1) = mock_vectors(0, 50, dim, 10.0);
        let flat_1: Vec<f32> = vecs_1.clone().into_iter().flatten().collect();

        {
            let file = File::create(&path).unwrap();
            let mut writer = BucketFileWriter::new_streaming(
                file,
                [1u8; 16], // Run ID
                quantizer.clone(),
                dim,
            )
            .unwrap();

            writer.write_batch(&ids_1, &flat_1).unwrap();

            // ⚡ Checks durability fix: sync_all() is called here
            let (offset, count) = writer.finalize().unwrap();
            assert_eq!(count, 50);
            println!("   -> Finalized at offset {}", offset);
        }

        // =========================================================================
        // PHASE 2: APPEND WRITE (Extending the File)
        // =========================================================================
        println!("Phase 2: Appending...");
        let (ids_2, vecs_2) = mock_vectors(50, 50, dim, 20.0);
        let flat_2: Vec<f32> = vecs_2.clone().into_iter().flatten().collect();

        {
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)
                .unwrap();

            let initial_len = file.metadata().unwrap().len();

            let mut writer = BucketFileWriter::new_append(
                file,
                [1u8; 16], // Must match
                quantizer.clone(),
                dim,
                initial_len,
            )
            .expect("Failed to open for append");

            // Verify we recovered previous state
            assert_eq!(writer.get_total_count(), 50);

            writer.write_batch(&ids_2, &flat_2).unwrap();

            // ⚡ Checks durability fix: truncates old footer + syncs new data
            let (_, total) = writer.finalize_and_truncate().unwrap();
            assert_eq!(total, 100);
        }

        // =========================================================================
        // PHASE 3: READ BACK (Verification)
        // =========================================================================
        println!("Phase 3: Reading...");

        // Use BucketFileReader to verify the on-disk format is valid
        let mut reader = BucketFileReader::open(op, filename).await.unwrap();

        // 1. Verify Metadata
        assert_eq!(reader.footer.total_vector_count, 100);
        assert_eq!(reader.footer.row_group_count, 2); // 1 from Stream + 1 from Append

        // 2. Read All Vectors (Hot + Cold)
        let (read_ids, read_vecs) = reader.read_all_vectors().await.unwrap();

        assert_eq!(read_ids.len(), 100);
        assert_eq!(read_vecs.len(), 100);

        // 3. Verify Data Integrity (Batch 1)
        assert_eq!(read_ids[0], 0);
        assert_eq!(read_vecs[0][0], 10.0); // Offset from Phase 1

        // 4. Verify Data Integrity (Batch 2)
        assert_eq!(read_ids[50], 50);
        assert_eq!(read_vecs[50][0], 20.0); // Offset from Phase 2

        println!("✅ Mixed Mode Test Passed!");
    }
}
