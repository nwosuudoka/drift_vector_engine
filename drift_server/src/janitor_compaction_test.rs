#[cfg(test)]
mod tests {
    use crate::compactor::SegmentCompactor;
    use crate::janitor::Janitor;
    use crate::persistence::PersistenceManager;
    use drift_core::index::{IndexOptions, VectorIndex};
    use drift_storage::disk_manager::DriftPageManager;
    use opendal::{Operator, services};
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Duration;
    use tempfile::tempdir;
    use tokio::time::sleep;

    fn create_local_operator(path: &std::path::Path) -> Operator {
        let mut builder = services::Fs::default();
        builder = builder.root(path.to_str().unwrap());
        Operator::new(builder).unwrap().finish()
    }

    #[tokio::test]
    async fn test_janitor_automatically_cleans_garbage_segments() {
        // Enable logging
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

            // Set Capacity to 1.
            // Bucket 1 (with 1 item) is now 100% full.
            // Urgency = 0. It will NEVER be merged.
            max_bucket_capacity: 1,

            ..Default::default()
        };
        let index = Arc::new(VectorIndex::new(options, &wal_path, storage).unwrap());

        // Train
        let train_data = vec![vec![0.0; 2]; 10];
        index.train(&train_data).await.unwrap();

        // 2. Create "Segment A" (Initial State)
        let ids = vec![1];
        let vectors = vec![vec![0.0; 2]];

        let partitions = index.calculate_partitions(&ids, &vectors).await.unwrap();
        let (run_id_a, locs_a) = persistence
            .write_partitioned_segment(&partitions, &index)
            .await
            .unwrap();
        let offsets_a: HashMap<u32, (u64, u32)> = locs_a
            .into_iter()
            .map(|(id, loc)| (id, (loc.index_offset, loc.index_length as u32)))
            .collect();

        // Register A
        index
            .register_partitions(&partitions, &run_id_a, &offsets_a)
            .await
            .unwrap();
        let file_a = format!("segment_{}.drift", run_id_a);

        assert!(op.exists(&file_a).await.unwrap(), "File A should exist");

        // 3. Create "Segment B" (Overwrite -> Makes A Garbage)
        let (run_id_b, locs_b) = persistence
            .write_partitioned_segment(&partitions, &index)
            .await
            .unwrap();
        let offsets_b: HashMap<u32, (u64, u32)> = locs_b
            .into_iter()
            .map(|(id, loc)| (id, (loc.index_offset, loc.index_length as u32)))
            .collect();

        // Register B (Updates index to point to B)
        index
            .register_partitions(&partitions, &run_id_b, &offsets_b)
            .await
            .unwrap();
        let file_b = format!("segment_{}.drift", run_id_b);

        // 4. Start Janitor
        let compactor = SegmentCompactor::new(index.clone(), op.clone());
        let janitor = Janitor::new(
            index.clone(),
            persistence.clone(),
            1000,
            Duration::from_millis(10),
            Some(compactor),
        );
        let janitor_handle = tokio::spawn(async move { janitor.run().await });

        // 5. Wait for GC
        sleep(Duration::from_millis(500)).await;

        // 6. Verify Cleanup
        let exists_a = op.exists(&file_a).await.unwrap();
        let exists_b = op.exists(&file_b).await.unwrap();

        janitor_handle.abort();

        if !exists_b {
            panic!(
                "TEST FAIL: Segment B (Live) was deleted! Run ID: {}",
                run_id_b
            );
        }
        if exists_a {
            panic!(
                "TEST FAIL: Segment A (Garbage) was NOT deleted! Run ID: {}",
                run_id_a
            );
        }
    }
}
