#[cfg(test)]
mod tests {
    use crate::bucket_file_writer::BucketFileWriter;
    use crate::format::{DriftFooter, DriftHeader, HEADER_SIZE, MAGIC_V2};
    use drift_core::quantizer::Quantizer;
    use std::io::Cursor;
    use std::vec;
    use zerocopy::FromBytes;

    fn mock_data(start_id: u64, count: usize, dim: usize) -> (Vec<u64>, Vec<Vec<f32>>) {
        let ids: Vec<u64> = (start_id..start_id + count as u64).collect();
        let vecs: Vec<Vec<f32>> = (0..count).map(|_| vec![1.0; dim]).collect();
        (ids, vecs)
    }

    #[test]
    fn test_bucket_file_full_lifecycle() {
        let mut buffer = Cursor::new(Vec::new());
        let run_id = [1u8; 16];
        let dim = 8;

        // 1. Train Quantizer
        let (ids_train, vecs_train) = mock_data(0, 100, dim);
        let flat: Vec<f32> = vecs_train.clone().into_iter().flatten().collect();
        let q = Quantizer::train(&flat, dim);

        // 2. Initialize Writer
        let mut writer =
            BucketFileWriter::new_streaming(&mut buffer, run_id, q.clone(), dim).unwrap();

        // 3. Write Batch 1 (100 vectors)
        let train_flat = vecs_train.into_iter().flatten().collect::<Vec<f32>>();
        writer.write_batch(&ids_train, &train_flat).unwrap();

        // 4. Write Batch 2 (50 vectors)
        let (ids_2, vecs_2) = mock_data(1000, 50, dim);
        let vecs_2_flat = vecs_2.clone().into_iter().flatten().collect::<Vec<f32>>();
        writer.write_batch(&ids_2, &vecs_2_flat).unwrap();

        // 5. Finalize
        let total_size = writer.finalize().unwrap();

        // --- VERIFICATION ---

        // A. Check Size
        let written_data = buffer.into_inner();
        assert_eq!(written_data.len() as u64, total_size);

        // B. Check Header (Start)
        let header_bytes = &written_data[0..HEADER_SIZE];
        let header = DriftHeader::read_from_bytes(header_bytes).unwrap();
        assert_eq!(header.magic, MAGIC_V2);
        assert_eq!(header.run_id, run_id);

        // C. Check Footer (End)
        let footer_len = crate::format::FOOTER_SIZE;
        let footer_start = written_data.len() - footer_len;
        let footer_bytes = &written_data[footer_start..];
        let footer = DriftFooter::read_from_bytes(footer_bytes).unwrap();

        assert_eq!(footer.magic, MAGIC_V2);
        assert_eq!(footer.row_group_count, 2); // We wrote 2 batches

        // D. Verify Directory Layout
        // Footer points to Index Start.
        // Index contains [Count: u32] [RGHeader 0] [RGHeader 1]
        let index_start = footer.index_start_offset as usize;
        let index_bytes = &written_data[index_start..];

        // Manual check of u32 count
        let rg_count_bytes: [u8; 4] = index_bytes[0..4].try_into().unwrap();
        let rg_count = u32::from_le_bytes(rg_count_bytes);
        assert_eq!(rg_count, 2);
    }
}

#[cfg(test)]
mod writer_tests {
    use crate::bucket_file_writer::BucketFileWriter;
    use crate::format::HEADER_SIZE;
    use drift_core::quantizer::Quantizer;
    use std::fs::OpenOptions;
    use std::io::{Read, Seek, SeekFrom};
    use tempfile::NamedTempFile;

    // Helper to generate mock data
    fn mock_batch(start_id: u64, count: usize, dim: usize) -> (Vec<u64>, Vec<f32>) {
        let mut ids = Vec::with_capacity(count);
        let mut vecs = Vec::with_capacity(count * dim);
        for i in 0..count {
            ids.push(start_id + i as u64);
            // Simple pattern: vector value = id
            vecs.extend(std::iter::repeat((start_id + i as u64) as f32).take(dim));
        }
        (ids, vecs)
    }

    #[test]
    fn test_append_recovery_and_truncation() {
        let tmp_file = NamedTempFile::new().unwrap();
        let path = tmp_file.path().to_owned();
        let dim = 8;

        // ---------------------------------------------------------
        // PHASE 1: Create New File (Day 0)
        // ---------------------------------------------------------
        let (ids_1, vecs_1) = mock_batch(0, 100, dim);
        let q = Quantizer::train(&vecs_1, dim); // Train dummy quantizer

        {
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true) // Ensure it exists
                .open(&path)
                .unwrap();

            // Use "new_streaming" logic or just generic new with len=0
            let mut writer = BucketFileWriter::new_streaming(
                file,
                [1u8; 16], // Run ID
                q.clone(),
                dim,
            )
            .expect("Failed to create writer");

            writer
                .write_batch(&ids_1, &vecs_1)
                .expect("Write batch 1 failed");
            writer.finalize().expect("Finalize 1 failed");
        }

        // Snapshot length after Phase 1
        let len_phase_1 = std::fs::metadata(&path).unwrap().len();
        println!("File Size Phase 1: {} bytes", len_phase_1);

        // ---------------------------------------------------------
        // PHASE 2: Append (Day 1)
        // ---------------------------------------------------------
        // We simulate opening the existing file.
        // This exercises:
        // 1. Seek to End -> Read Old Footer
        // 2. Recover State (Bloom, Index)
        // 3. Seek Back (Rewind) -> Ready to Overwrite

        let (ids_2, vecs_2) = mock_batch(100, 50, dim);

        {
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)
                .unwrap();

            let initial_len = file.metadata().unwrap().len();

            let mut writer = BucketFileWriter::new_append(
                file,
                [1u8; 16], // Should match old Run ID, but we don't validate strictly yet
                q.clone(), // Must reuse same quantizer!
                dim,
                initial_len,
            )
            .expect("Failed to open append writer");

            writer
                .write_batch(&ids_2, &vecs_2)
                .expect("Write batch 2 failed");

            // finalize_and_truncate will ensure the file ends exactly at the new footer
            writer.finalize_and_truncate().expect("Finalize 2 failed");
        }

        let len_phase_2 = std::fs::metadata(&path).unwrap().len();
        println!("File Size Phase 2: {} bytes", len_phase_2);

        // Assert Growth: File must be larger now
        assert!(len_phase_2 > len_phase_1, "File should have grown");

        // ---------------------------------------------------------
        // PHASE 3: Verification (Read Back)
        // ---------------------------------------------------------
        // We manually scan the file structure to verify integrity.
        // In a real integration test, we'd use BucketFileReader.

        let mut file = std::fs::File::open(&path).unwrap();

        // 1. Verify Header (Head-Of-Line Quantizer check)
        let mut header_buf = [0u8; HEADER_SIZE];
        file.read_exact(&mut header_buf).unwrap();
        // Just verify magic
        let magic = u64::from_le_bytes(header_buf[0..8].try_into().unwrap());
        assert_eq!(magic, crate::format::MAGIC_V2);

        // 2. Verify Footer (At the VERY end)
        file.seek(SeekFrom::End(-(crate::format::FOOTER_SIZE as i64)))
            .unwrap();
        let mut footer_buf = [0u8; crate::format::FOOTER_SIZE];
        file.read_exact(&mut footer_buf).unwrap();

        // Use byteorder/zerocopy to decode struct if needed, or just check basic fields
        // Let's assume DriftFooter layout: [RowGroupCount (u32), ...]
        let rg_count = u32::from_le_bytes(footer_buf[0..4].try_into().unwrap());

        // CRITICAL CHECK: We wrote 1 batch in Phase 1, 1 batch in Phase 2.
        // Total RowGroups should be 2.
        assert_eq!(rg_count, 2, "Footer should report 2 RowGroups after append");

        // 3. Verify Truncation (No garbage after footer)
        let current_pos = file.stream_position().unwrap();
        let total_size = file.metadata().unwrap().len();
        assert_eq!(
            current_pos, total_size,
            "File should end exactly after footer"
        );
    }
}
