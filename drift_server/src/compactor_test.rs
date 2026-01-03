#[cfg(test)]
mod tests {
    use crate::compactor::SegmentCompactor;
    use crate::persistence::PersistenceManager;
    use drift_core::index::{IndexOptions, VectorIndex};
    use drift_storage::disk_manager::DriftPageManager; // ⚡ Use DriftPageManager
    use opendal::{Operator, services};
    use std::collections::HashMap;
    use std::sync::Arc;
    use tempfile::tempdir;

    fn create_local_operator(path: &std::path::Path) -> Operator {
        let mut builder = services::Fs::default();
        builder = builder.root(path.to_str().unwrap());
        Operator::new(builder).unwrap().finish()
    }

    #[tokio::test]
    async fn test_segment_vacuum_cleans_obsolete_files() {
        let dir = tempdir().unwrap();
        let op = create_local_operator(dir.path());
        let persistence = PersistenceManager::new(op.clone(), dir.path());

        // 1. Setup Index using DriftPageManager (Supports File Mapping)
        let wal_path = dir.path().join("test.wal");

        // ⚡ FIX: Use DriftPageManager so register_partitions works correctly
        // and get_physical_path returns the registered segment filenames.
        let storage = Arc::new(DriftPageManager::new(op.clone()));

        let options = IndexOptions {
            dim: 2,
            num_centroids: 1,
            training_sample_size: 10,
            max_bucket_capacity: 100,
            ..Default::default()
        };

        let index = Arc::new(VectorIndex::new(options, &wal_path, storage).unwrap());

        // 2. Train (Required for Quantizer)
        let train_data = vec![vec![0.0; 2]; 10];
        index.train(&train_data).await.unwrap();

        // 3. Prepare Data
        let ids = vec![1];
        let vectors = vec![vec![0.0; 2]];

        // 4. Create "Segment A" (Simulate Run 1)
        let partitions = index.calculate_partitions(&ids, &vectors).await.unwrap();
        let (run_id_a, locs_a) = persistence
            .write_partitioned_segment(&partitions, &index)
            .await
            .unwrap();

        let offsets_a: HashMap<u32, (u64, u32)> = locs_a
            .into_iter()
            .map(|(id, loc)| (id, (loc.index_offset, loc.index_length as u32)))
            .collect();

        // Register A: Index points to Segment A
        index
            .register_partitions(&partitions, &run_id_a, &offsets_a)
            .await
            .unwrap();

        let file_a = format!("segment_{}.drift", run_id_a);

        // 5. Create "Segment B" (Simulate Run 2 / Compaction)
        // Overwrite the SAME bucket ID.
        let (run_id_b, locs_b) = persistence
            .write_partitioned_segment(&partitions, &index)
            .await
            .unwrap();

        let offsets_b: HashMap<u32, (u64, u32)> = locs_b
            .into_iter()
            .map(|(id, loc)| (id, (loc.index_offset, loc.index_length as u32)))
            .collect();

        // Register B: Index updates to point to Segment B. Segment A becomes garbage.
        index
            .register_partitions(&partitions, &run_id_b, &offsets_b)
            .await
            .unwrap();

        let file_b = format!("segment_{}.drift", run_id_b);

        // 6. Initialize Compactor
        let compactor = SegmentCompactor::new(index.clone(), op.clone());

        // 7. Run Vacuum
        compactor.vacuum_segments().await.unwrap();

        // 8. Verify
        let exists_a = op.exists(&file_a).await.unwrap();
        let exists_b = op.exists(&file_b).await.unwrap();

        assert!(!exists_a, "Segment A should have been deleted (Garbage)");
        assert!(exists_b, "Segment B should exist (Live)");
    }
}
