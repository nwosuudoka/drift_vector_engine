#[cfg(test)]
mod tests {
    use crate::compactor::SegmentCompactor;
    use crate::persistence::PersistenceManager;
    use drift_core::index::{IndexOptions, VectorIndex};
    use drift_storage::disk_manager::DriftPageManager;
    use opendal::{Operator, services};
    use std::collections::HashMap;
    use std::sync::Arc;
    use tempfile::tempdir;

    fn create_local_operator(path: &std::path::Path) -> Operator {
        let mut builder = services::Fs::default();
        builder = builder.root(path.to_str().unwrap());
        Operator::new(builder).unwrap().finish()
    }

    // Helper: Returns (RunID, BucketID)
    async fn create_segment_via_memtable(
        index: &Arc<VectorIndex>,
        persistence: &PersistenceManager,
        ids: &[u64],
        vectors: &[Vec<f32>],
    ) -> (String, u32) {
        // 1. Fill
        for (i, id) in ids.iter().enumerate() {
            index.insert(*id, &vectors[i]).unwrap();
        }

        // 2. Snapshot
        let memtable = index.memtable.read().clone();

        // 3. Partition (Allocates NEW Bucket ID)
        let partitions = index.partition_memtable(&memtable).unwrap();
        let bucket_id = partitions[0].bucket_id;

        // 4. Write
        {
            let mut guard = index.frozen_memtable.write();
            *guard = Some(memtable.clone());
        }

        let (run_id, locs) = persistence
            .write_partitioned_segment(&partitions, index)
            .await
            .unwrap();

        let offsets: HashMap<u32, (u64, u32)> = locs
            .into_iter()
            .map(|(id, loc)| (id, (loc.index_offset, loc.index_length as u32)))
            .collect();

        // 5. Register
        index
            .register_partitions(&partitions, &run_id, &offsets)
            .await
            .unwrap();

        // 6. Cleanup
        index.confirm_flush().unwrap();
        index.rotate_and_freeze().unwrap(); // Clear memtable
        index.confirm_flush().unwrap(); // Clear frozen

        (run_id, bucket_id)
    }

    #[tokio::test]
    async fn test_segment_vacuum_cleans_obsolete_files() {
        let dir = tempdir().unwrap();
        let op = create_local_operator(dir.path());
        let persistence = PersistenceManager::new(op.clone(), dir.path());

        let wal_path = dir.path().join("test.wal");
        let storage = Arc::new(DriftPageManager::new(op.clone()));

        let options = IndexOptions {
            dim: 2,
            num_centroids: 1,
            training_sample_size: 10,
            max_bucket_capacity: 100,
            ..Default::default()
        };

        let index = Arc::new(VectorIndex::new(options, &wal_path, storage).unwrap());

        // 1. Train
        let train_data = vec![vec![0.0; 2]; 10];
        index.train(&train_data).await.unwrap();

        // 2. Create "Segment A"
        let ids = vec![1];
        let vectors = vec![vec![0.0; 2]];

        let (run_id_a, bucket_id_a) =
            create_segment_via_memtable(&index, &persistence, &ids, &vectors).await;
        let file_a = format!("segment_{}.drift", run_id_a);

        // 3. Make Segment A Garbage
        // We delete the data, then compact the bucket.
        // This removes bucket_id_a from the Index.
        index.delete(1).unwrap();

        let compacted = index.compact_bucket(bucket_id_a).await.unwrap();
        assert!(
            compacted.is_some(),
            "Bucket A should have been compacted away"
        );

        // 4. Create "Segment B" (Live Data)
        // We re-insert ID 1 (Resurrect it in a new bucket/segment)
        let (run_id_b, _bucket_id_b) =
            create_segment_via_memtable(&index, &persistence, &ids, &vectors).await;
        let file_b = format!("segment_{}.drift", run_id_b);

        // 5. Initialize Compactor & Vacuum
        let compactor = SegmentCompactor::new(index.clone(), op.clone());
        compactor.vacuum_segments().await.unwrap();

        // 6. Verify
        // A should be gone (Bucket A removed from Index -> File A not in live set)
        // B should exist (Bucket B is in Index -> File B in live set)
        let exists_a = op.exists(&file_a).await.unwrap();
        let exists_b = op.exists(&file_b).await.unwrap();

        assert!(!exists_a, "Segment A should have been deleted (Garbage)");
        assert!(exists_b, "Segment B should exist (Live)");
    }
}
