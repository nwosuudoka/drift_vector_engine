#[cfg(test)]
mod tests {
    use super::super::local_staging::LocalStagingManager;
    use drift_core::lock_manager::BucketCoordinator; // ⚡ Import Coordinator
    use drift_core::partitioner::PartitionGroup;
    use drift_storage::unified_format::{
        UnifiedBlockType, UnifiedFieldSchema, UnifiedLogicalType, UnifiedPayloadSchema,
    };
    use drift_storage::unified_reader::UnifiedReader;
    use opendal::{Operator, services};
    use std::collections::HashSet;
    use std::sync::Arc;
    use tempfile::tempdir;

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
            flat_vectors.extend(std::iter::repeat((start_id + i as u64) as f32).take(dim));
        }

        PartitionGroup {
            ids,
            flat_vectors,
            count,
            centroid,
        }
    }

    async fn read_row_group_count(path: &std::path::Path) -> u32 {
        let root = path.parent().unwrap().to_str().unwrap().to_string();
        let filename = path.file_name().unwrap().to_str().unwrap().to_string();
        let op = Operator::new(services::Fs::default().root(&root))
            .unwrap()
            .finish();
        let reader = UnifiedReader::open(op, &filename).await.unwrap();
        let mut groups = HashSet::new();
        for block in reader.blocks {
            if block.block_type == UnifiedBlockType::Ids {
                groups.insert((block.row_start, block.row_count));
            }
        }
        groups.len() as u32
    }

    async fn read_payload_schema(path: &std::path::Path) -> Option<UnifiedPayloadSchema> {
        let root = path.parent().unwrap().to_str().unwrap().to_string();
        let filename = path.file_name().unwrap().to_str().unwrap().to_string();
        let op = Operator::new(services::Fs::default().root(&root))
            .unwrap()
            .finish();
        let reader = UnifiedReader::open(op, &filename).await.unwrap();
        reader.read_payload_schema().await.unwrap()
    }

    // --- TESTS ---

    #[tokio::test]
    async fn test_staging_append_lifecycle() {
        let dir = tempdir().unwrap();
        let manager = LocalStagingManager::new(dir.path()).unwrap();
        let bucket_id = 1;
        let dim = 8;

        // 1. First Write
        let batch1 = create_batch(0, 10, dim, None);
        let size1 = manager.append_batch(bucket_id, &batch1).await.unwrap();
        assert_eq!(size1, 10);

        let file_path = dir.path().join(format!("bucket_{}.driftu", bucket_id));
        assert!(file_path.exists());
        assert_eq!(read_row_group_count(&file_path).await, 1);

        // 2. Second Write
        let batch2 = create_batch(10, 5, dim, None);
        let size2 = manager.append_batch(bucket_id, &batch2).await.unwrap();
        assert_eq!(size2, 15);
        assert_eq!(read_row_group_count(&file_path).await, 2);
    }

    #[tokio::test]
    async fn test_staging_append_with_schema_persists_schema_block() {
        let dir = tempdir().unwrap();
        let manager = LocalStagingManager::new(dir.path()).unwrap();
        let bucket_id = 2;
        let dim = 4;
        let schema = UnifiedPayloadSchema::new(vec![UnifiedFieldSchema {
            field_id: 1,
            name: "tenant".to_string(),
            logical_type: UnifiedLogicalType::Keyword,
            nullable: false,
            indexed: true,
        }]);

        let batch1 = create_batch(100, 4, dim, None);
        let size1 = manager
            .append_batch_with_schema(bucket_id, &batch1, Some(&schema))
            .await
            .unwrap();
        assert_eq!(size1, 4);

        let batch2 = create_batch(104, 2, dim, None);
        let size2 = manager.append_batch(bucket_id, &batch2).await.unwrap();
        assert_eq!(size2, 6);

        let file_path = dir.path().join(format!("bucket_{}.driftu", bucket_id));
        let decoded_schema = read_payload_schema(&file_path).await;
        assert_eq!(decoded_schema, Some(schema));
    }

    #[tokio::test]
    async fn test_concurrent_appends_single_bucket() {
        // ⚡ STRESS TEST: 10 threads slamming the same bucket.
        // We MUST use BucketCoordinator to protect the file, matching system design.

        let dir = tempdir().unwrap();
        let manager = Arc::new(LocalStagingManager::new(dir.path()).unwrap());
        let coordinator = Arc::new(BucketCoordinator::new()); // ⚡ Add Lock

        let bucket_id = 99;
        let dim = 4;
        let num_tasks = 10;
        let vectors_per_task = 100;

        let mut handles = Vec::new();

        for i in 0..num_tasks {
            let m = manager.clone();
            let c = coordinator.clone();

            handles.push(tokio::spawn(async move {
                // 🔒 Acquire WRITE lock before appending
                let _guard = c.write(bucket_id).await;

                let start_id = (i * vectors_per_task) as u64;
                let batch = create_batch(start_id, vectors_per_task, dim, None);
                m.append_batch(bucket_id, &batch).await.unwrap();

                // _guard dropped here
            }));
        }

        for h in handles {
            h.await.unwrap();
        }

        let path = dir.path().join(format!("bucket_{}.driftu", bucket_id));
        let rg_count = read_row_group_count(&path).await;

        assert_eq!(
            rg_count, num_tasks as u32,
            "Race condition protected! All RowGroups present."
        );
    }
}

#[cfg(test)]
mod race_cond_tests {
    use crate::local_staging::LocalStagingManager;
    use drift_core::lock_manager::BucketCoordinator; // ⚡ Import
    use drift_core::partitioner::PartitionGroup;
    use drift_storage::unified_reader::UnifiedReader;
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
    async fn test_race_condition_fixed_with_locks() {
        let dir = tempdir().unwrap();
        let manager = Arc::new(LocalStagingManager::new(dir.path()).unwrap());
        let coordinator = Arc::new(BucketCoordinator::new()); // ⚡ The Fix

        let bucket_id = 1;
        let dim = 8;
        let filename = format!("bucket_{}.driftu", bucket_id);

        // 1. Initialize
        let initial = create_batch(0, 100, dim);
        manager.append_batch(bucket_id, &initial).await.unwrap();

        let running = Arc::new(std::sync::atomic::AtomicBool::new(true));

        // --- WRITER TASK ---
        let m_writer = manager.clone();
        let c_writer = coordinator.clone();
        let r_writer = running.clone();

        let writer_handle = tokio::spawn(async move {
            let mut id = 100;
            while r_writer.load(std::sync::atomic::Ordering::Relaxed) {
                // 🔒 Write Lock
                let _guard = c_writer.write(bucket_id).await;

                let batch = create_batch(id, 50, dim);
                m_writer.append_batch(bucket_id, &batch).await.unwrap();
                id += 50;

                // Hold lock slightly to ensure overlapping attempts by reader wait
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        });

        // --- READER TASK ---
        let op = create_local_op(dir.path());
        let c_reader = coordinator.clone();

        let reader_handle = tokio::spawn(async move {
            let mut successes = 0;
            let mut failures = 0;

            for _ in 0..50 {
                // 🔒 Read Lock
                let _guard = c_reader.read(bucket_id).await;

                // Now it's safe to open and read
                match UnifiedReader::open(op.clone(), &filename).await {
                    Ok(mut reader) => {
                        if reader.read_all_vectors_flat().await.is_ok() {
                            successes += 1;
                        } else {
                            failures += 1;
                        }
                    }
                    Err(_) => failures += 1,
                }
                // Lock drops here
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
            (successes, failures)
        });

        tokio::time::sleep(Duration::from_secs(1)).await;
        running.store(false, std::sync::atomic::Ordering::Relaxed);

        let _ = writer_handle.await;
        let (ok, err) = reader_handle.await.unwrap();

        println!("With Locks: {} Successes, {} Failures", ok, err);

        // ⚡ ASSERTION: With locks, failures should be ZERO.
        assert_eq!(err, 0, "Locks failed to prevent corruption!");
    }
}
