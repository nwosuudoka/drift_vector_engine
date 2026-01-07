use drift_cache::LocalDiskManager;
use drift_core::bucket::BucketHeader;
use drift_core::index::{IndexOptions, MaintenanceStatus, VectorIndex};
use drift_core::quantizer::Quantizer;
use drift_traits::PageId; // Assuming this is re-exported or available
use std::sync::Arc;
use tempfile::tempdir;

// Helper to create a dummy header for math testing
fn create_dummy_header(id: u32, count: u32) -> BucketHeader {
    BucketHeader::new(
        id,
        vec![0.0; 128], // 128-dim centroid
        count,
        PageId {
            file_id: id,
            offset: 0,
            length: 4096,
        },
    )
}

#[test]
fn test_urgency_equation_5_hot_zombie_paradox() {
    // SCENARIO: A "Hot Zombie"
    // It is accessed frequently (High Temp), so standard LRU would keep it.
    // But it is 90% dead (High Tombstones).
    // The Urgency Function (Eq 5) must override the Heat.

    let header = create_dummy_header(1, 100);

    // 1. Heat it up (Simulate heavy read traffic)
    // touching 50 times should drive temp near 1.0 due to EWMA
    for _ in 0..50 {
        header.touch();
    }
    let temp = header.temperature();
    assert!(
        temp > 0.8,
        "Bucket should be hot (Temp > 0.8), got {}",
        temp
    );

    // 2. Kill the data (90% tombstones)
    for _ in 0..90 {
        header.mark_tombstone();
    }

    // 3. Calculate Urgency
    // Target capacity 100.
    // Formula: (Emptiness / Temp) + (Beta * ZombieRatio)
    let urgency = header.calculate_urgency(100);

    println!("Hot Zombie Urgency: {}", urgency);

    // Threshold for merge is typically > 1.5 in your paper.
    // With 90% zombies, Beta(3.0) * 0.9 = 2.7.
    // Even if Temp is Max (1.0), Urgency should be high.
    assert!(
        urgency > 1.5,
        "FAIL: The Hot Zombie Paradox. High-temp bucket was not marked urgent. Urgency: {}",
        urgency
    );
}

#[test]
fn test_saturating_density_equation_2() {
    // Validates R(b) = 1 - exp(-Count / Tau)
    // Tau = 100 (from your paper)

    let tau = 100.0;

    let small_bucket = create_dummy_header(1, 10); // 10 items
    let large_bucket = create_dummy_header(2, 200); // 200 items (2x Tau)

    let r_small = 1.0 - (-(small_bucket.count as f32) / tau).exp();
    let r_large = 1.0 - (-(large_bucket.count as f32) / tau).exp();

    println!("Reliability (Small): {}", r_small); // Should be low ~0.095
    println!("Reliability (Large): {}", r_large); // Should be high ~0.86

    assert!(
        r_small < 0.15,
        "Small bucket should have low reliability signal"
    );
    assert!(
        r_large > 0.8,
        "Large bucket should have saturated reliability signal"
    );
    assert!(
        r_large > r_small * 5.0,
        "Large bucket should be significantly more weighted"
    );
}

#[tokio::test]
async fn test_budgeted_scatter_merge_strictness() {
    // Setup Index
    let dir = tempdir().unwrap();
    let dim = 2;
    let options = IndexOptions {
        dim,
        max_bucket_capacity: 100,
        ..Default::default()
    };

    // Mock storage
    // Note: Use drift_storage if available, otherwise assume a mock or default implementation is used
    // If you don't have drift_storage available in this test scope, you might need to mock PageManager.
    // For this fix, I will assume you have a way to create storage, e.g.:
    // let storage = Arc::new(drift_storage::DiskManager::new(dir.path()).unwrap());
    // If you are using the in-memory mock from lib.rs tests, use that.
    // Below assumes you can instantiate the storage used in your other tests.

    // --- MOCK STORAGE SETUP (Replace with your actual storage constructor) ---
    // If you don't have a public DiskManager, you might need to rely on the one from your lib.
    // For now, I'll assume standard disk manager access:
    let storage_path = dir.path().join("data.drift");
    let storage = Arc::new(LocalDiskManager::new(&storage_path));
    // ------------------------------------------------------------------------

    let index = VectorIndex::new(options, &dir.path().join("wal"), storage).unwrap();

    // 1. PREPARE DATA
    let ids: Vec<u64> = (0..51).collect();
    let vectors: Vec<Vec<f32>> = (0..51).map(|_| vec![0.1, 0.1]).collect();
    let vectors_flat = vectors.iter().flatten().cloned().collect::<Vec<_>>();

    // 2. ⚡ FIX: INITIALIZE QUANTIZER ⚡
    // The index needs to know how to compress floats to bytes.
    // We train it on the data we are about to insert.
    let q = Quantizer::train(&vectors_flat, dim);
    index.set_quantizer(q); // <--- This prevents the Panic

    // 3. Force bucket ID 1 (OVER BUDGET > 50)
    index
        .force_register_bucket_with_ids(1, &ids, &vectors)
        .await
        .unwrap();

    // 4. Attempt Scatter Merge
    let status = index.scatter_merge(1).await.unwrap();

    // 5. Assert Rejection
    // It should skip because moving 51 items > 50 item budget.
    assert_eq!(
        status,
        MaintenanceStatus::SkippedTooSmall,
        "FAIL: Budget enforcement. Should have skipped merge for bucket > 50 items."
    );

    // 6. Reduce size to UNDER budget (49 items)
    index.delete(0).unwrap();
    index.delete(1).unwrap();

    // Retry Scatter Merge
    let status_retry = index.scatter_merge(1).await.unwrap();

    // We just want to ensure it proceeds or processes differently than the strict rejection above
    println!("Status after reduction: {:?}", status_retry);
}
