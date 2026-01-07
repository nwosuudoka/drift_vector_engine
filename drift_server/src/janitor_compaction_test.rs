#[cfg(test)]
mod tests {
    use crate::compactor::SegmentCompactor;
    use crate::janitor::Janitor;
    use crate::persistence::PersistenceManager;
    use drift_core::index::{IndexOptions, VectorIndex};
    use drift_storage::disk_manager::DriftPageManager;
    use opendal::{Operator, services};
    use std::sync::Arc;
    use std::time::Duration;
    use tempfile::tempdir;
    use tokio::time::sleep;

    fn create_local_operator(path: &std::path::Path) -> Operator {
        let mut builder = services::Fs::default();
        builder = builder.root(path.to_str().unwrap());
        Operator::new(builder).unwrap().finish()
    }

    /// Helper to flush the active MemTable to a disk segment.
    /// Returns (RunID, BucketID_of_created_bucket).
    async fn flush_to_segment(
        index: &Arc<VectorIndex>,
        persistence: &PersistenceManager,
    ) -> (String, u32) {
        // 1. Rotate & Freeze
        let memtable = index.rotate_and_freeze().unwrap().unwrap();

        // 2. Partition (Sync CPU work)
        let partitions = index.partition_memtable(&memtable).unwrap();
        let bucket_id = partitions[0].bucket_id;

        // 3. Write to Disk
        let (run_id, locs) = persistence
            .write_partitioned_segment(&partitions, index)
            .await
            .unwrap();

        // 4. Register
        let offsets = locs
            .iter()
            .map(|(k, v)| (*k, (v.index_offset, v.index_length as u32)))
            .collect();
        index
            .register_partitions(&partitions, &run_id, &offsets)
            .await
            .unwrap();

        // 5. Cleanup
        index.confirm_flush().unwrap();

        (run_id, bucket_id)
    }

    #[tokio::test]
    async fn test_janitor_automatically_cleans_garbage_segments() {
        let _ = tracing_subscriber::fmt()
            .with_env_filter("drift_server=info")
            .with_test_writer()
            .try_init();

        let dir = tempdir().unwrap();
        let op = create_local_operator(dir.path());
        let persistence = PersistenceManager::new(op.clone(), dir.path());

        let wal_path = dir.path().join("test.wal");
        let storage = Arc::new(DriftPageManager::new(op.clone()));

        let options = IndexOptions {
            dim: 2,
            num_centroids: 1,
            training_sample_size: 10,

            // ⚡ FIX: Set capacity to 1 so our small test buckets are considered "100% Full".
            // This prevents the Janitor from seeing them as "Urgent" and merging them away.
            max_bucket_capacity: 1,

            ..Default::default()
        };
        let index = Arc::new(VectorIndex::new(options, &wal_path, storage).unwrap());

        // Train
        let train_data = vec![vec![0.0; 2]; 10];
        index.train(&train_data).await.unwrap();

        // --- STEP 1: Create "Segment A" (Target Garbage) ---
        index.insert(1, &vec![0.0, 0.0]).unwrap();
        let (run_id_a, bucket_id_a) = flush_to_segment(&index, &persistence).await;
        let file_a = format!("segment_{}.drift", run_id_a);

        // --- STEP 2: Make Segment A Garbage ---
        // To orphan the file, we must remove the bucket pointing to it.
        // 1. Delete the data.
        index.delete(1).unwrap();
        // 2. Compact the bucket. Since it is empty, it will be removed from the Index.
        let compacted = index.compact_bucket(bucket_id_a).await.unwrap();
        assert!(
            compacted.is_some(),
            "Bucket A should have been compacted away (Empty)"
        );

        // --- STEP 3: Create "Segment B" (Live Data) ---
        index.insert(2, &vec![1.0, 1.0]).unwrap();
        let (run_id_b, _bucket_id_b) = flush_to_segment(&index, &persistence).await;
        let file_b = format!("segment_{}.drift", run_id_b);

        // --- STEP 4: Start Janitor ---
        let compactor = SegmentCompactor::new(index.clone(), op.clone());
        let janitor = Janitor::new(
            index.clone(),
            persistence.clone(),
            1000,
            Duration::from_millis(10), // Fast ticks
            Some(compactor),
        );
        let janitor_handle = tokio::spawn(async move { janitor.run().await });

        // --- STEP 5: Wait for Vacuum ---
        // Janitor runs compaction cycle every 100 ticks.
        // 100 * 10ms = 1 second. We wait 2s to be safe.
        sleep(Duration::from_millis(2000)).await;

        // --- STEP 6: Verify ---
        let exists_a = op.exists(&file_a).await.unwrap();
        let exists_b = op.exists(&file_b).await.unwrap();

        janitor_handle.abort();

        if exists_a {
            panic!(
                "TEST FAIL: Segment A (Garbage) was NOT deleted! Run ID: {}",
                run_id_a
            );
        }
        if !exists_b {
            panic!(
                "TEST FAIL: Segment B (Live) was deleted! Run ID: {}",
                run_id_b
            );
        }
    }
}
