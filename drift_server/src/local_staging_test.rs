#[cfg(test)]
mod tests {
    use super::super::local_staging::LocalStagingManager;
    use drift_core::partitioner::PartitionGroup;
    use drift_storage::format::{DriftFooter, FOOTER_SIZE, MAGIC_V2};
    use std::fs::File;
    use std::io::{Read, Seek, SeekFrom};
    use std::sync::Arc;
    use tempfile::tempdir;
    use zerocopy::FromBytes;

    // --- HELPERS ---

    fn create_batch(
        start_id: u64,
        count: usize,
        dim: usize,
        centroid: Option<Vec<f32>>,
    ) -> PartitionGroup {
        let mut ids = Vec::with_capacity(count);
        let mut flat_vectors = Vec::with_capacity(count * dim);

        for i in 0..count {
            ids.push(start_id + i as u64);
            // Generate dummy vector data
            flat_vectors.extend(std::iter::repeat((start_id + i as u64) as f32).take(dim));
        }

        PartitionGroup {
            ids,
            flat_vectors,
            count,
            centroid,
        }
    }

    /// specific helper to read the footer of a drift file and return the total row group count
    fn read_row_group_count(path: &std::path::Path) -> u32 {
        let mut file = File::open(path).expect("failed to open file");
        let len = file.metadata().unwrap().len();
        file.seek(SeekFrom::Start(len - FOOTER_SIZE as u64))
            .unwrap();
        let mut buf = [0u8; FOOTER_SIZE];
        file.read_exact(&mut buf).unwrap();

        let footer = DriftFooter::read_from_bytes(&buf).unwrap();
        assert_eq!(footer.magic, MAGIC_V2);
        footer.row_group_count
    }

    // --- TESTS ---

    #[tokio::test]
    async fn test_lifecycle_create_append_recover() {
        let dir = tempdir().unwrap();
        let bucket_id = 1;
        let dim = 8;

        // 1. First Manager (Day 0)
        {
            let manager = LocalStagingManager::new(dir.path()).unwrap();
            let batch = create_batch(0, 100, dim, None);
            manager.append_batch(bucket_id, &batch).await.unwrap();
        }

        let file_path = dir.path().join(format!("bucket_{}.drift", bucket_id));
        assert!(file_path.exists());
        assert_eq!(read_row_group_count(&file_path), 1);

        // 2. Second Manager (Day 1 - Restart)
        // This tests that we can recover the Quantizer and Append safely
        {
            let manager = LocalStagingManager::new(dir.path()).unwrap();
            let batch = create_batch(100, 50, dim, None); // Different IDs
            manager.append_batch(bucket_id, &batch).await.unwrap();
        }

        // Verify Growth
        assert_eq!(read_row_group_count(&file_path), 2);
    }

    #[tokio::test]
    async fn test_bucket_isolation() {
        // Ensure writing to Bucket A doesn't lock or corrupt Bucket B
        let dir = tempdir().unwrap();
        let manager = LocalStagingManager::new(dir.path()).unwrap();
        let dim = 4;

        let b1 = create_batch(0, 10, dim, None);
        let b2 = create_batch(100, 10, dim, None);

        // Interleaved writes
        manager.append_batch(1, &b1).await.unwrap();
        manager.append_batch(2, &b2).await.unwrap();
        manager.append_batch(1, &b1).await.unwrap();

        let p1 = dir.path().join("bucket_1.drift");
        let p2 = dir.path().join("bucket_2.drift");

        assert_eq!(read_row_group_count(&p1), 2);
        assert_eq!(read_row_group_count(&p2), 1);
    }

    #[tokio::test]
    async fn test_concurrent_appends_single_bucket() {
        // ⚡ STRESS TEST: 10 threads slamming the same bucket.
        // The Mutex in LocalStagingManager MUST serialize this.
        let dir = tempdir().unwrap();
        let manager = Arc::new(LocalStagingManager::new(dir.path()).unwrap());
        let bucket_id = 99;
        let dim = 4;
        let num_tasks = 10;
        let vectors_per_task = 100;

        let mut handles = Vec::new();

        for i in 0..num_tasks {
            let m = manager.clone();
            handles.push(tokio::spawn(async move {
                let start_id = (i * vectors_per_task) as u64;
                let batch = create_batch(start_id, vectors_per_task, dim, None);
                m.append_batch(bucket_id, &batch).await.unwrap();
            }));
        }

        // Await all
        for h in handles {
            h.await.unwrap();
        }

        // Validation
        let path = dir.path().join(format!("bucket_{}.drift", bucket_id));
        let rg_count = read_row_group_count(&path);

        // We expect exactly 10 RowGroups (one per task)
        assert_eq!(
            rg_count, num_tasks as u32,
            "Race condition detected! RowGroups missing."
        );

        // Optional: verify file size is substantial
        let meta = std::fs::metadata(&path).unwrap();
        assert!(meta.len() > 1000);
    }

    #[tokio::test]
    async fn test_empty_batch_ignored() {
        let dir = tempdir().unwrap();
        let manager = LocalStagingManager::new(dir.path()).unwrap();

        let empty_batch = PartitionGroup::new(0, None);
        manager.append_batch(1, &empty_batch).await.unwrap();

        let path = dir.path().join("bucket_1.drift");
        // Should NOT create a file for empty batch
        assert!(!path.exists());
    }
}

#[cfg(test)]
mod race_cond_tests {
    use crate::local_staging::LocalStagingManager;
    use drift_core::partitioner::PartitionGroup;
    use std::sync::Arc;
    use std::time::Duration;
    use tempfile::tempdir;

    fn create_batch(start_id: u64, count: usize, dim: usize) -> PartitionGroup {
        let mut ids = Vec::new();
        let mut vecs = Vec::new();
        for i in 0..count {
            ids.push(start_id + i as u64);
            vecs.extend(vec![0.1; dim]);
        }
        PartitionGroup {
            ids,
            flat_vectors: vecs,
            count,
            centroid: None,
        }
    }

    #[tokio::test]
    async fn test_reproduce_corruption_during_append() {
        let dir = tempdir().unwrap();
        let manager = Arc::new(LocalStagingManager::new(dir.path()).unwrap());
        let bucket_id = 1;
        let dim = 8;

        // 1. Initialize File
        let initial = create_batch(0, 100, dim);
        manager.append_batch(bucket_id, &initial).await.unwrap();

        let running = Arc::new(std::sync::atomic::AtomicBool::new(true));

        // --- WRITER TASK ---
        // Appends data rapidly to force footer overwrites
        let m_writer = manager.clone();
        let r_writer = running.clone();
        let writer_handle = tokio::spawn(async move {
            let mut id = 100;
            while r_writer.load(std::sync::atomic::Ordering::Relaxed) {
                let batch = create_batch(id, 10, dim);
                m_writer.append_batch(bucket_id, &batch).await.unwrap();
                id += 10;
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        });

        // --- READER TASK ---
        // Tries to read the file repeatedly.
        // IF OUR THEORY IS RIGHT: This will fail with "Invalid Data" or IO Error
        let m_reader = manager.clone();
        let reader_handle = tokio::spawn(async move {
            let mut successes = 0;
            let mut failures = 0;

            for _ in 0..50 {
                // We read the full bucket (opens file, parses footer, reads data)
                match m_reader.read_full_bucket(bucket_id).await {
                    Ok((ids, _)) => {
                        assert!(!ids.is_empty());
                        successes += 1;
                    }
                    Err(e) => {
                        println!("💥 CRASH: Reader failed due to race condition: {}", e);
                        failures += 1;
                    }
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            (successes, failures)
        });

        // Run for 1 second
        tokio::time::sleep(Duration::from_secs(1)).await;
        running.store(false, std::sync::atomic::Ordering::Relaxed);

        let _ = writer_handle.await;
        let (ok, err) = reader_handle.await.unwrap();

        println!("Test Results: {} Successes, {} Failures", ok, err);

        // ⚡ ASSERTION: If this is "In-Place Append", we expect ERRORS.
        // If we switch to "Copy-Append-Rename", we expect ZERO ERRORS.

        // Uncomment this to prove the bug exists (Expect failures)
        if err > 0 {
            panic!(
                "Reproduced: Reader crashed {} times due to footer corruption!",
                err
            );
        }
    }
}

#[cfg(test)]
mod race_cond_test_2 {
    use crate::local_staging::LocalStagingManager;
    use drift_core::partitioner::PartitionGroup;
    use drift_storage::bucket_file_reader::BucketFileReader; // ⚡ Direct Access
    use opendal::{Operator, services};
    use std::sync::Arc;
    use std::time::Duration;
    use tempfile::tempdir;

    fn create_batch(start_id: u64, count: usize, dim: usize) -> PartitionGroup {
        let mut ids = Vec::new();
        let mut vecs = Vec::new();
        for i in 0..count {
            ids.push(start_id + i as u64);
            vecs.extend(vec![0.1; dim]);
        }
        PartitionGroup {
            ids,
            flat_vectors: vecs,
            count,
            centroid: None,
        }
    }

    fn create_local_op(path: &std::path::Path) -> Operator {
        let builder = services::Fs::default().root(path.to_str().unwrap());
        Operator::new(builder).unwrap().finish()
    }

    #[tokio::test]
    async fn test_reproduce_corruption_during_append() {
        let dir = tempdir().unwrap();
        let manager = Arc::new(LocalStagingManager::new(dir.path()).unwrap());
        let bucket_id = 1;
        let dim = 8;
        let filename = format!("bucket_{}.drift", bucket_id);

        // 1. Initialize File
        let initial = create_batch(0, 100, dim);
        manager.append_batch(bucket_id, &initial).await.unwrap();

        let running = Arc::new(std::sync::atomic::AtomicBool::new(true));

        // --- WRITER TASK (Janitor) ---
        // Uses LocalStagingManager (Locked)
        let m_writer = manager.clone();
        let r_writer = running.clone();
        let writer_handle = tokio::spawn(async move {
            let mut id = 100;
            while r_writer.load(std::sync::atomic::Ordering::Relaxed) {
                let batch = create_batch(id, 50, dim); // Smaller batches, more frequent writes
                m_writer.append_batch(bucket_id, &batch).await.unwrap();
                id += 50;
                // Tiny sleep to allow reader to jump in
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        });

        // --- READER TASK (Search Query) ---
        // ⚡ USES BucketFileReader DIRECTLY (Unlocked!)
        // This mimics how BucketManager searches.
        let op = create_local_op(dir.path());
        let reader_handle = tokio::spawn(async move {
            let mut successes = 0;
            let mut failures = 0;

            // Try to read rapidly while writer is hammering the file
            for _ in 0..100 {
                match BucketFileReader::open(op.clone(), &filename).await {
                    Ok(mut reader) => {
                        // Try to parse the index
                        // This often fails if footer/index offsets are being overwritten
                        if reader
                            .scan(&[0.0; 8], 1, &drift_traits::mock::NoTombstones)
                            .await
                            .is_ok()
                        {
                            successes += 1;
                        } else {
                            failures += 1;
                        }
                    }
                    Err(e) => {
                        println!("💥 CRASH: Reader failed open/parse: {}", e);
                        failures += 1;
                    }
                }
                // No sleep, hammer it
            }
            (successes, failures)
        });

        // Run for 2 seconds
        tokio::time::sleep(Duration::from_secs(2)).await;
        running.store(false, std::sync::atomic::Ordering::Relaxed);

        let _ = writer_handle.await;
        let (ok, err) = reader_handle.await.unwrap();

        println!("Test Results: {} Successes, {} Failures", ok, err);

        // ⚡ EXPECT FAILURES
        if err > 0 {
            panic!(
                "Reproduced: Reader crashed {} times due to footer corruption!",
                err
            );
        }
    }
}

#[cfg(test)]
mod more_tests {
    use crate::local_staging::LocalStagingManager;
    use drift_core::partitioner::PartitionGroup;
    use drift_storage::format::{DriftFooter, FOOTER_SIZE, MAGIC_V2};
    use std::fs::File;
    use std::io::{Read, Seek, SeekFrom};
    use tempfile::tempdir;
    use zerocopy::FromBytes;

    fn create_batch(start_id: u64, count: usize, dim: usize, val_offset: f32) -> PartitionGroup {
        let mut ids = Vec::with_capacity(count);
        let mut flat_vectors = Vec::with_capacity(count * dim);

        for i in 0..count {
            ids.push(start_id + i as u64);
            flat_vectors
                .extend(std::iter::repeat((start_id + i as u64) as f32 + val_offset).take(dim));
        }

        PartitionGroup {
            ids,
            flat_vectors,
            count,
            centroid: None,
        }
    }

    /// Helper to read the physical footer from disk to verify total counts
    fn read_physical_footer_count(path: &std::path::Path) -> u32 {
        let mut file = File::open(path).expect("failed to open file");
        let len = file.metadata().unwrap().len();
        file.seek(SeekFrom::Start(len - FOOTER_SIZE as u64))
            .unwrap();

        let mut buf = [0u8; FOOTER_SIZE];
        file.read_exact(&mut buf).unwrap();

        let footer = DriftFooter::read_from_bytes(&buf).unwrap();
        assert_eq!(footer.magic, MAGIC_V2);
        footer.total_vector_count as u32
    }

    // --- TESTS ---

    #[tokio::test]
    async fn test_staging_append_lifecycle() {
        let dir = tempdir().unwrap();
        let manager = LocalStagingManager::new(dir.path()).unwrap();
        let bucket_id = 1;
        let dim = 8;

        // 1. First Write (Streaming Mode)
        let batch1 = create_batch(0, 10, dim, 0.0);
        let size1 = manager.append_batch(bucket_id, &batch1).await.unwrap();
        assert_eq!(size1, 10);

        let file_path = dir.path().join(format!("bucket_{}.drift", bucket_id));
        assert!(file_path.exists());

        // Verify Footer Update 1
        assert_eq!(read_physical_footer_count(&file_path), 10);

        // 2. Second Write (Append Mode)
        let batch2 = create_batch(10, 5, dim, 0.0);
        let size2 = manager.append_batch(bucket_id, &batch2).await.unwrap();

        // Total should be 15
        assert_eq!(size2, 15);

        // Verify Footer Update 2
        assert_eq!(read_physical_footer_count(&file_path), 15);

        // 3. Read Back Content (Using Reader Logic)
        let (ids, vecs) = manager.read_full_bucket(bucket_id).await.unwrap();
        assert_eq!(ids.len(), 15);
        assert_eq!(vecs.len(), 15);

        // Check order
        assert_eq!(ids[0], 0);
        assert_eq!(ids[14], 14);
    }

    #[tokio::test]
    async fn test_rotation_creates_clean_slate() {
        let dir = tempdir().unwrap();
        let manager = LocalStagingManager::new(dir.path()).unwrap();
        let bucket_id = 2;
        let dim = 2;

        // 1. Write to Active
        let batch = create_batch(0, 100, dim, 0.0);
        manager.append_batch(bucket_id, &batch).await.unwrap();

        let active_filename = manager.get_active_filename(bucket_id);
        let staging_filename = format!("bucket_{}_staging_TEST.drift", bucket_id);
        let new_active_filename = format!("bucket_{}_v2.drift", bucket_id);

        // 2. ROTATE
        // This simulates the Janitor promoting the bucket
        let rotated = manager
            .rotate_bucket_for_promotion(bucket_id, &staging_filename, &new_active_filename)
            .await
            .unwrap();

        assert!(rotated);

        // 3. Verify Files on Disk
        let staging_path = dir.path().join(&staging_filename);
        let new_active_path = dir.path().join(&new_active_filename);
        let _old_active_path = dir.path().join(&active_filename); // Should be gone if names match, or just renamed

        assert!(staging_path.exists(), "Staging file missing");
        assert!(new_active_path.exists(), "New active file missing");

        // Verify internal map updated
        assert_eq!(manager.get_active_filename(bucket_id), new_active_filename);

        // 4. Write to NEW Active (Should start fresh)
        let batch_new = create_batch(200, 1, dim, 0.0);
        let total = manager.append_batch(bucket_id, &batch_new).await.unwrap();

        // Should be 1 (Fresh start), NOT 101
        assert_eq!(total, 1);

        // 5. Verify Staging Content (Should still have 100)
        let (ids, _) = manager.read_file_content(&staging_filename).await.unwrap();
        assert_eq!(ids.len(), 100);
    }

    #[tokio::test]
    async fn test_threshold_detection() {
        let dir = tempdir().unwrap();
        let manager = LocalStagingManager::new(dir.path()).unwrap();
        let dim = 128;

        // 1. Write Small Batch (Bucket 1)
        let small_batch = create_batch(0, 10, dim, 0.0);
        manager.append_batch(1, &small_batch).await.unwrap();

        // Measure actual size (Includes overhead like Bloom Filter ~1MB)
        let path_1 = dir.path().join("bucket_1.drift");
        let size_small = std::fs::metadata(&path_1).unwrap().len();

        // 2. Write Large Batch (Bucket 2) -> 10x more vectors
        let large_batch = create_batch(0, 100, dim, 0.0);
        manager.append_batch(2, &large_batch).await.unwrap();

        let path_2 = dir.path().join("bucket_2.drift");
        let size_large = std::fs::metadata(&path_2).unwrap().len();

        println!(
            "Sizes -> Small: {} bytes, Large: {} bytes",
            size_small, size_large
        );
        assert!(
            size_large > size_small,
            "Large bucket must be larger than small bucket"
        );

        // 3. Dynamic Threshold (Halfway between)
        let threshold = (size_small + size_large) / 2;

        // Test Strict Threshold
        let candidates = manager.list_large_buckets(threshold).unwrap();
        assert_eq!(candidates.len(), 1, "Should detect exactly 1 large bucket");
        assert_eq!(candidates[0], 2, "Should be Bucket 2");

        // Test Loose Threshold (Smaller than both)
        let candidates_all = manager.list_large_buckets(100).unwrap();
        assert!(candidates_all.contains(&1));
        assert!(candidates_all.contains(&2));
    }

    #[tokio::test]
    async fn test_recovery_finds_existing_files() {
        let dir = tempdir().unwrap();
        let bucket_id = 99;
        let dim = 4;

        // 1. Create a Manager, write data, then drop it.
        {
            let manager = LocalStagingManager::new(dir.path()).unwrap();
            let batch = create_batch(0, 10, dim, 0.0);
            manager.append_batch(bucket_id, &batch).await.unwrap();
        }

        // 2. Create NEW Manager pointing to same dir
        let manager_2 = LocalStagingManager::new(dir.path()).unwrap();

        // 3. list_large_buckets triggers a scan of the directory
        // This populates the internal knowledge or just reads FS.
        let candidates = manager_2.list_large_buckets(0).unwrap();
        assert!(candidates.contains(&bucket_id));

        // 4. Append to the RECOVERED bucket
        // This ensures we don't overwrite/truncate the existing file accidentally
        let batch_new = create_batch(10, 10, dim, 0.0);
        let total = manager_2.append_batch(bucket_id, &batch_new).await.unwrap();

        // Should be 20 (10 existing + 10 new)
        assert_eq!(total, 20);

        // 5. Verify Reads
        let (ids, _) = manager_2.read_full_bucket(bucket_id).await.unwrap();
        assert_eq!(ids.len(), 20);
    }

    #[tokio::test]
    async fn test_empty_batch_no_op() {
        let dir = tempdir().unwrap();
        let manager = LocalStagingManager::new(dir.path()).unwrap();
        let bucket_id = 5;

        let empty_batch = create_batch(0, 0, 4, 0.0);
        let count = manager.append_batch(bucket_id, &empty_batch).await.unwrap();

        assert_eq!(count, 0);

        // File should NOT be created
        let path = dir.path().join(format!("bucket_{}.drift", bucket_id));
        assert!(!path.exists());
    }
}
