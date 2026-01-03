#[cfg(test)]
mod tests {
    use crate::{DriftFooter, FOOTER_SIZE, MAGIC_BYTES};

    #[test]
    fn test_footer_layout_strictness() {
        // Create a footer with recognizable pattern values
        let footer = DriftFooter::new(
            0x1111_1111_1111_1111, // Index Off
            0x2222_2222_2222_2222, // Index Len
            0x3333_3333_3333_3333, // Bloom Off
            0x4444_4444_4444_4444, // Bloom Len
            0x5555_5555_5555_5555, // Quant Off
            0x6666_6666_6666_6666, // Quant Len
        );

        let bytes = footer.to_bytes();

        // 1. Verify Size
        assert_eq!(bytes.len(), FOOTER_SIZE);

        // 2. Verify Version (Byte 0)
        assert_eq!(bytes[0], 1);

        // 3. Verify Offsets (Little Endian)
        // Index Offset starts at byte 1
        assert_eq!(&bytes[1..9], &0x1111_1111_1111_1111u64.to_le_bytes());

        // Quantizer Length (Last u64) starts at 1 + (5 * 8) = 41
        assert_eq!(&bytes[41..49], &0x6666_6666_6666_6666u64.to_le_bytes());

        // 4. Verify Padding (Bytes 49..56)
        // 7 bytes of zeros
        assert_eq!(&bytes[49..56], &[0u8; 7]);

        // 5. Verify Magic (Bytes 56..64)
        assert_eq!(&bytes[56..64], MAGIC_BYTES);
    }

    #[test]
    fn test_footer_roundtrip() {
        let original = DriftFooter::new(100, 200, 300, 400, 500, 600);
        let bytes = original.to_bytes();
        let decoded = DriftFooter::from_bytes(&bytes).expect("Failed to decode footer");

        assert_eq!(original, decoded);
        assert_eq!(decoded.magic, *MAGIC_BYTES);
    }
}
