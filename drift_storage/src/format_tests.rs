#[cfg(test)]
mod tests {
    use super::super::format::*;
    use zerocopy::{FromBytes, IntoBytes};

    #[test]
    fn test_drift_header_roundtrip() {
        let run_id = [7u8; 16];
        let original = DriftHeader::new(1000, run_id);

        // Zero-Copy serialization: as_bytes() returns a slice &[u8]
        let bytes = original.as_bytes();
        assert_eq!(bytes.len(), HEADER_SIZE);

        // Verify Magic (Little Endian)
        let magic_bytes = MAGIC_V2.to_le_bytes();
        assert_eq!(&bytes[0..8], &magic_bytes);

        // Zero-Copy deserialization
        let decoded = DriftHeader::read_from_bytes(bytes).unwrap();
        assert_eq!(decoded, original);
        assert!(decoded.validate());
    }

    #[test]
    fn test_row_group_header_roundtrip() {
        let header = RowGroupHeader::new(
            500,        // count
            0xDEADBEEF, // checksum
            128,        // hot_offset
            1024,       // hot_len
            2048,       // cold_offset
            4096,       // cold_len
        );

        let bytes = header.as_bytes();
        assert_eq!(bytes.len(), 64);

        // Verify manual padding alignment check
        // _pad_1 is at offset 20. It's a u32 (4 bytes).
        // It should be 0.
        assert_eq!(&bytes[20..24], &[0, 0, 0, 0]);

        let decoded = RowGroupHeader::read_from_bytes(bytes).unwrap();
        assert_eq!(decoded, header);
    }

    #[test]
    fn test_footer_magic_placement() {
        let footer = DriftFooter::new(10, 1000, 2000, 500);
        let bytes = footer.as_bytes();

        // Magic must be at the very end (120..128)
        let magic_bytes = MAGIC_V2.to_le_bytes();
        assert_eq!(&bytes[120..128], &magic_bytes);

        let decoded = DriftFooter::read_from_bytes(bytes).unwrap();
        assert_eq!(decoded, footer);
    }
}
