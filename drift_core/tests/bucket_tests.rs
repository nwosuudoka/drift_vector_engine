use bit_set::BitSet;
use drift_core::aligned::AlignedBytes;
use drift_core::bucket::{Bucket, BucketData, BucketHeader};
use drift_traits::{Cacheable, PageId};
use rand::Rng;

// Helper to generate random aligned bytes
fn random_aligned_bytes(len: usize) -> AlignedBytes {
    let mut rng = rand::thread_rng();
    let mut bytes = AlignedBytes::new(len);
    // Unsafe access to write random data into aligned buffer
    unsafe {
        bytes.set_len(len);
        rng.fill(bytes.as_mut_slice());
    }
    bytes
}

#[test]
fn test_bucket_serialization_round_trip() {
    let dim = 128;
    let count = 100;

    // 1. Create Random Bucket Data
    let codes = random_aligned_bytes(count * dim);
    let vids: Vec<u64> = (0..count as u64).collect();
    let mut tombstones = BitSet::with_capacity(count);

    // Mark even indices as deleted
    for i in (0..count).step_by(2) {
        tombstones.insert(i);
    }

    let original = BucketData {
        codes: codes.clone(),
        vids: vids.clone(),
        tombstones: tombstones.clone(),
    };

    // 2. Serialize
    let bytes = original.to_bytes(dim).expect("Serialization failed");

    // 3. Deserialize
    let recovered = BucketData::from_bytes(&bytes).expect("Deserialization failed");

    // 4. Assert Equality
    assert_eq!(original.vids, recovered.vids, "Vector IDs mismatch");
    assert_eq!(
        original.tombstones, recovered.tombstones,
        "Tombstones mismatch"
    );

    // Check codes match exactly
    for i in 0..bytes.len() {
        if i < original.codes.len() {
            assert_eq!(
                original.codes[i], recovered.codes[i],
                "Code byte {} mismatch",
                i
            );
        }
    }
}

#[test]
fn test_bucket_scan_with_lut_and_tombstones() {
    // 1. Setup a tiny bucket (2 vectors, dim 4 for simplicity)
    let dim = 4;
    let vids = vec![10, 20];

    // Vector A (ID 10): [0, 0, 0, 0] (Codes)
    // Vector B (ID 20): [1, 1, 1, 1] (Codes)
    let mut codes = AlignedBytes::new(8);
    unsafe {
        codes.set_len(8);
        codes[0..4].copy_from_slice(&[0, 0, 0, 0]);
        codes[4..8].copy_from_slice(&[1, 1, 1, 1]);
    }

    let mut tombstones = BitSet::new();
    // ⚡ Kill Vector A (ID 10)
    tombstones.insert(0);

    let data = BucketData {
        codes,
        vids,
        tombstones,
    };

    // 2. Create a Mock LUT (Look-Up Table)
    // Size = dim * 256.
    // We want the distance to be predictable.
    // Let's say query is "perfect match" for 0, and "far" from 1.
    // LUT[dim_idx][byte_value] = distance_component

    let mut lut = vec![0.0f32; dim * 256];

    // Fill LUT such that value '0' adds 0.0 distance, value '1' adds 10.0 distance
    for d in 0..dim {
        lut[d * 256 + 0] = 0.0; // Code 0 -> Dist 0
        lut[d * 256 + 1] = 10.0; // Code 1 -> Dist 10
    }

    // 3. Execute Scan
    let results = Bucket::scan_with_lut(&data, &lut, dim);

    // 4. Verification

    // Should only have 1 result (ID 20) because ID 10 is a tombstone
    assert_eq!(results.len(), 1, "Should filter out tombstone (ID 10)");

    let result = &results[0];
    assert_eq!(result.id, 20);

    // Distance check:
    // Vector B has codes [1, 1, 1, 1].
    // LUT adds 10.0 for each '1'.
    // Total Dist = 10 + 10 + 10 + 10 = 40.0
    assert_eq!(result.distance, 40.0);
}

#[test]
fn test_bucket_header_stats_persistence() {
    // This ensures that metadata logic (like page_id) remains consistent
    let pid = PageId {
        file_id: 99,
        offset: 123,
        length: 456,
    };
    let header = BucketHeader::new(1, vec![1.0, 2.0], 500, pid.clone());

    assert_eq!(header.id, 1);
    assert_eq!(header.count, 500);
    assert_eq!(header.page_id.file_id, 99);

    // Stats start at default
    assert_eq!(header.temperature(), 0.0);

    header.touch();
    assert!(
        header.temperature() > 0.0,
        "Touch should increase temperature"
    );
}
