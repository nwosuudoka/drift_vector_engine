#[cfg(test)]
mod tests {
    use crate::aligned::AlignedBytes;
    use crate::bucket::BucketData;
    use bit_set::BitSet;
    use drift_cache::store::Cacheable;

    #[test]
    fn test_bucket_data_roundtrip_with_bitpacking() {
        let count = 1000;
        let dim = 8;
        let target_size = count * dim; // 8000 bytes

        // 1. Setup Data
        // FIX: Use target_size for the loop, not codes.len() (which starts at 0)
        let mut codes = AlignedBytes::new(target_size);
        for i in 0..target_size {
            codes.push((i % 255) as u8);
        }

        // Verify we actually filled it
        assert_eq!(codes.len(), target_size);

        let mut vids = Vec::with_capacity(count);
        for i in 0..count {
            vids.push(i as u64 * 10);
        }

        let mut tombstones = BitSet::with_capacity(count);
        // Mark some random bits to test sparse bitpacking
        tombstones.insert(0);
        tombstones.insert(10);
        tombstones.insert(999);
        tombstones.insert(500);

        let original = BucketData {
            codes: codes.clone(),
            vids: vids.clone(),
            tombstones: tombstones.clone(),
        };

        // 2. Serialize
        let bytes = original.to_bytes(dim).expect("Serialization failed");

        println!("Serialized Size: {} bytes", bytes.len());

        // Sanity Check: Size should be roughly (Header) + (1000*8 Codes) + (1000*8 IDs) + (Tombstones)
        // Header ~12
        // Codes ~8000
        // IDs ~8000
        // Tombstones: 1000 bits / 8 = 125 bytes (compressed should be small)
        // Total ~16000 + overhead.
        assert!(bytes.len() > 16000);

        // 3. Deserialize
        let decoded = BucketData::from_bytes(&bytes).expect("Deserialization failed");

        // 4. Verify Content
        assert_eq!(decoded.vids, original.vids, "VIDs mismatch");
        assert_eq!(
            decoded.codes.as_slice(),
            original.codes.as_slice(),
            "Codes mismatch"
        );

        // 5. Verify Bitpacking (Tombstones)
        assert_eq!(
            decoded.tombstones.len(),
            original.tombstones.len(),
            "Tombstone count mismatch"
        );
        assert!(decoded.tombstones.contains(0));
        assert!(decoded.tombstones.contains(10));
        assert!(decoded.tombstones.contains(999));
        assert!(decoded.tombstones.contains(500));
        assert!(!decoded.tombstones.contains(1)); // Should be false
    }

    #[test]
    fn test_empty_bucket_serialization() {
        let dim = 128;
        let original = BucketData {
            codes: AlignedBytes::new(0),
            vids: vec![],
            tombstones: BitSet::new(),
        };

        let bytes = original.to_bytes(dim).unwrap();
        let decoded = BucketData::from_bytes(&bytes).unwrap();

        assert!(decoded.vids.is_empty());
        assert!(decoded.tombstones.is_empty());
    }
}
