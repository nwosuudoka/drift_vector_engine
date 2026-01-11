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

    fn create_batch(start_id: u64, count: usize, dim: usize) -> PartitionGroup {
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
            let batch = create_batch(0, 100, dim);
            manager.append_batch(bucket_id, &batch).await.unwrap();
        }

        let file_path = dir.path().join(format!("bucket_{}.drift", bucket_id));
        assert!(file_path.exists());
        assert_eq!(read_row_group_count(&file_path), 1);

        // 2. Second Manager (Day 1 - Restart)
        // This tests that we can recover the Quantizer and Append safely
        {
            let manager = LocalStagingManager::new(dir.path()).unwrap();
            let batch = create_batch(100, 50, dim); // Different IDs
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

        let b1 = create_batch(0, 10, dim);
        let b2 = create_batch(100, 10, dim);

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
                let batch = create_batch(start_id, vectors_per_task, dim);
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

        let empty_batch = PartitionGroup {
            ids: vec![],
            flat_vectors: vec![],
            count: 0,
        };

        manager.append_batch(1, &empty_batch).await.unwrap();

        let path = dir.path().join("bucket_1.drift");
        // Should NOT create a file for empty batch
        assert!(!path.exists());
    }
}
