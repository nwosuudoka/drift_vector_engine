#[cfg(test)]
mod tests {
    use crate::{format::RowGroupHeader, row_group_writer::RowGroupWriter};
    use drift_core::quantizer::Quantizer;
    use std::io::Cursor;
    use zerocopy::FromBytes;

    // Helper to generate N dummy vectors of DIM
    fn mock_data(n: usize, dim: usize) -> (Vec<u64>, Vec<Vec<f32>>, Quantizer) {
        let ids: Vec<u64> = (0..n as u64).collect();
        let vecs: Vec<Vec<f32>> = (0..n).map(|i| vec![i as f32; dim]).collect();
        let flat: Vec<f32> = vecs.clone().into_iter().flatten().collect();
        let q = Quantizer::train(&flat, dim);
        (ids, vecs, q)
    }

    #[test]
    fn test_row_group_alignment_and_layout() {
        let mut buffer = Cursor::new(Vec::new());
        let mut writer = RowGroupWriter::new(&mut buffer, 0);

        let dim = 4;
        let ids = vec![100, 101];
        let vectors = vec![vec![1.0, 1.0, 1.0, 1.0], vec![2.0, 2.0, 2.0, 2.0]];

        // Train dummy quantizer
        let flat: Vec<f32> = vectors.clone().into_iter().flatten().collect();
        let q = Quantizer::train(&flat, dim);

        // WRITE
        let vectors_flat: Vec<f32> = vectors.clone().into_iter().flatten().collect();
        let header = writer
            .write_group(&ids, &vectors_flat, None, &q, dim)
            .unwrap();

        // ASSERTIONS
        // 1. Check Counts
        // Use .get() because these are U32/U64 wrappers now!
        assert_eq!(header.vector_count, 2);
        assert_eq!(header.hot_offset, 64);

        // 2. Check Alignment
        assert_eq!(
            header.cold_offset % 64,
            0,
            "Cold section must be 64-byte aligned"
        );

        // 3. Check Padding Content
        let bytes = buffer.into_inner();
        let padding_start = (header.hot_offset + header.hot_length as u64) as usize;
        let padding_end = header.cold_offset as usize;

        if padding_end > padding_start {
            let padding = &bytes[padding_start..padding_end];
            assert!(padding.iter().all(|&b| b == 0), "Padding must be zeroed");
        }
    }

    #[test]
    fn test_edge_case_perfect_alignment() {
        // We want the Hot Section to end EXACTLY on a 64-byte boundary.
        // Header = 64 bytes (Aligned).
        // Hot Section = (IDs: N*8) + (SQ8: N*dim) + (Tomb: 4).
        // We need (N*8 + N*dim + 4) % 64 == 0.
        // Let dim = 56.
        // Then (N*8 + N*56 + 4) = N*64 + 4.
        // This will always be 4 bytes off. Impossible to align perfectly with this layout?
        // Wait, Tombstones is [Len: u32] + [Bytes]. If Len=0, it's 4 bytes.
        // So Hot Size = N*8 + N*dim + 4.
        // We need N*8 + N*dim + 4 = K * 64.

        // Let's try dim = 60.
        // Hot = N*8 + N*60 + 4 = N*68 + 4.
        // If N=15: 15*68 + 4 = 1020 + 4 = 1024.
        // 1024 % 64 == 0. PERFECT ALIGNMENT.

        let (ids, vecs, q) = mock_data(15, 60);

        let mut buffer = Cursor::new(Vec::new());
        let mut writer = RowGroupWriter::new(&mut buffer, 0);
        let vecs_flat: Vec<f32> = vecs.clone().into_iter().flatten().collect();
        let header = writer.write_group(&ids, &vecs_flat, None, &q, 60).unwrap();

        // Check alignment
        // Hot Offset = 64.
        // Hot Length = 1024.
        // Cold Offset should be 64 + 1024 = 1088.
        assert_eq!(header.hot_length, 1024);
        assert_eq!(header.cold_offset, 1088);

        // Padding should be 0
        let bytes = buffer.into_inner();
        let padding_area = &bytes[1088..1088]; // Empty slice
        assert_eq!(padding_area.len(), 0);
    }

    #[test]
    fn test_edge_case_worst_misalignment() {
        // Force 1 byte of misalignment.
        // Hot Size % 64 == 1.
        // Hot = N*8 + N*dim + 4.
        // Let dim = 1.
        // Hot = N*9 + 4.
        // If N=7: 63 + 4 = 67. 67 % 64 = 3.
        // If N=14: 126 + 4 = 130.
        // Let's brute force a tiny config.
        // dim=1. N=1 -> 8+1+4 = 13.
        // Padding needed = 64 - 13 = 51.

        let (ids, vecs, q) = mock_data(1, 1);

        let mut buffer = Cursor::new(Vec::new());
        let mut writer = RowGroupWriter::new(&mut buffer, 0);
        let vecs_flat: Vec<f32> = vecs.clone().into_iter().flatten().collect();
        let header = writer.write_group(&ids, &vecs_flat, None, &q, 1).unwrap();

        // Hot Length = 13
        assert_eq!(header.hot_length, 13);

        // Cold Offset must be aligned (Multiple of 64)
        assert_eq!(header.cold_offset % 64, 0);

        // Verify we added padding
        assert!(header.cold_offset > header.hot_offset + header.hot_length as u64);
        let padding_len = header.cold_offset - (header.hot_offset + header.hot_length as u64);
        assert_eq!(padding_len, 51);
    }

    #[test]
    fn test_empty_input_errors() {
        let mut buffer = Cursor::new(Vec::new());
        let mut writer = RowGroupWriter::new(&mut buffer, 0);
        let q = Quantizer::train(&[1.0], 1);

        let res = writer.write_group(&[], &[], None, &q, 1);
        assert!(res.is_err());
        assert_eq!(res.unwrap_err().kind(), std::io::ErrorKind::InvalidInput);
    }

    #[test]
    fn test_mismatched_input_errors() {
        let mut buffer = Cursor::new(Vec::new());
        let mut writer = RowGroupWriter::new(&mut buffer, 0);
        let q = Quantizer::train(&[1.0], 1);

        let ids = vec![1];
        let vecs = vec![]; // Mismatch

        let res = writer.write_group(&ids, &vecs, None, &q, 1);
        assert!(res.is_err());
    }

    #[test]
    fn test_large_offset_persistence() {
        // Simulate writing a Row Group deep into a file (e.g., 5GB offset)
        let start_offset = 5 * 1024 * 1024 * 1024; // 5GB
        let mut buffer = Cursor::new(Vec::new());
        let mut writer = RowGroupWriter::new(&mut buffer, start_offset);

        let (ids, vecs, q) = mock_data(10, 4);
        let vecs_flat: Vec<f32> = vecs.clone().into_iter().flatten().collect();
        let header = writer.write_group(&ids, &vecs_flat, None, &q, 4).unwrap();

        // The header written to "disk" (buffer) should contain the huge offset
        assert_eq!(header.hot_offset, start_offset + 64);

        // Verify by reading back from buffer
        let bytes = buffer.into_inner();
        let read_header = RowGroupHeader::read_from_bytes(&bytes[0..64]).unwrap();
        assert_eq!(read_header.hot_offset, start_offset + 64);
    }
}
