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
        let mut writer = BucketFileWriter::new(&mut buffer, run_id, q.clone(), dim).unwrap();

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

    /*
    #[test]
    fn test_reader_stream_skip_logic() {
        // 1. Setup Writer
        let mut buffer = Cursor::new(Vec::new());
        let run_id = [2u8; 16];
        let dim = 4;

        // Train Quantizer
        let (ids, vecs) = mock_data(0, 10, dim); // Batch 1
        let (ids2, vecs2) = mock_data(100, 10, dim); // Batch 2
        let flat: Vec<f32> = vecs.clone().into_iter().flatten().collect();
        let flat2: Vec<f32> = vecs2.clone().into_iter().flatten().collect();
        let q = Quantizer::train(&flat, dim);

        let mut writer = BucketFileWriter::new(&mut buffer, run_id, q.clone(), dim).unwrap();

        writer.write_batch(&ids, &flat).unwrap();
        writer.write_batch(&ids2, &flat2).unwrap();

        writer.finalize().unwrap();

        // 2. READ BACK
        buffer.set_position(0);
        let mut reader = BucketFileReader::new(buffer);

        // --- Batch 1: Read Hot, SKIP Cold ---
        {
            let mut group1 = reader.read_next_group().unwrap().unwrap();
            assert_eq!(group1.header.vector_count, 10);

            let (read_ids, _codes) = group1.decode_hot_index(dim).unwrap();
            assert_eq!(read_ids[0], 0);
            // We DROP group1 here without calling fetch_cold_vectors
        }

        // --- Batch 2: Read Hot, FETCH Cold ---
        // The reader should auto-skip Batch 1's cold data
        {
            let mut group2 = reader.read_next_group().unwrap().unwrap();
            assert_eq!(group2.header.vector_count, 10);

            let (read_ids2, _codes2) = group2.decode_hot_index(dim).unwrap();
            assert_eq!(read_ids2[0], 100); // Correctly advanced to batch 2

            let floats = group2.fetch_cold_vectors(dim).unwrap();
            assert_eq!(floats.len(), 40); // 10 vectors * 4 dim
            assert_eq!(floats[0], 1.0); // Value from mock_data
        }
    }
    */
}
