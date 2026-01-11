#[cfg(test)]
mod tests {
    use crate::bucket_file_reader::BucketFileReader;
    use crate::bucket_file_writer::BucketFileWriter;
    use crate::bucket_manager::BucketManager;
    use crate::disk_manager::DriftPageManager; // Use the mapping-aware manager
    use drift_core::quantizer::Quantizer;
    use drift_core::tombstone_v2::HashSetView;
    use drift_traits::{DiskSearcher, PageManager};
    use opendal::{Operator, services};
    use std::{path::PathBuf, sync::Arc};
    use tempfile::tempdir;

    // --- Helper: Create Local Operator ---
    fn create_local_operator(path: &std::path::Path) -> Operator {
        let mut builder = services::Fs::default();
        builder = builder.root(path.to_str().unwrap());
        Operator::new(builder).unwrap().finish()
    }

    // --- Helper: Mock Data ---
    fn mock_data(start_id: u64, count: usize, dim: usize) -> (Vec<u64>, Vec<f32>) {
        let ids: Vec<u64> = (start_id..start_id + count as u64).collect();
        let mut vecs = Vec::with_capacity(count * dim);
        for i in 0..count {
            for d in 0..dim {
                vecs.push((ids[i] + d as u64) as f32);
            }
        }
        (ids, vecs)
    }

    #[tokio::test]
    async fn test_writer_persists_quantizer_in_footer() {
        let dir = tempdir().unwrap();
        let filename = "segment_test_q.drift";
        let file_path = dir.path().join(filename);
        let file = std::fs::File::create(&file_path).unwrap();

        let dim = 8;
        let (ids, vecs) = mock_data(0, 100, dim);
        let q = Quantizer::train(&vecs, dim);
        let run_id = [1u8; 16];

        // 1. Write File (Standard IO)
        let mut writer = BucketFileWriter::new_streaming(file, run_id, q.clone(), dim).unwrap();
        writer.write_batch(&ids, &vecs).unwrap();
        writer.finalize().unwrap();

        // 2. Verify Footer Read (via PageManager)
        let op = create_local_operator(dir.path());
        let storage = Arc::new(DriftPageManager::new(op));

        // Register the specific filename to ID 99
        storage.register_file(99, std::path::PathBuf::from(filename));

        // Initialize Reader
        let mut reader = BucketFileReader::new(storage, 99);

        // This should now succeed because DriftPageManager maps 99 -> "segment_test_q.drift"
        reader
            .load_quantizer()
            .await
            .expect("Failed to load quantizer from footer");
    }

    #[tokio::test]
    async fn test_reader_scan_correctness() {
        let dir = tempdir().unwrap();
        let op = create_local_operator(dir.path());
        let storage = Arc::new(DriftPageManager::new(op));

        let dim = 4;
        let (ids, vecs) = mock_data(0, 10, dim);
        let q = Quantizer::train(&vecs, dim);

        let filename = "segment_run_abc.drift";
        let file_path = dir.path().join(filename);
        let file = std::fs::File::create(&file_path).unwrap();

        // 1. Write
        let mut writer = BucketFileWriter::new_streaming(file, [0u8; 16], q.clone(), dim).unwrap();
        writer.write_batch(&ids, &vecs).unwrap();
        writer.finalize().unwrap();

        // 2. Register
        storage.register_file(5, std::path::PathBuf::from(filename));

        // 3. Scan
        let mut reader = BucketFileReader::new(storage, 5);
        let query = vec![0.0; dim];

        let results = reader
            .scan(&query, 10, &HashSetView::default())
            .await
            .expect("Scan failed");

        assert_eq!(results.len(), 10);
        let found_ids: Vec<u64> = results.iter().map(|c| c.id).collect();
        assert_eq!(found_ids, ids);
    }

    #[tokio::test]
    async fn test_bucket_manager_orchestration() {
        let dir = tempdir().unwrap();

        // 1. Setup Data
        let bucket_id = 100;
        let file_id = 123;
        let dim = 8;

        // 2. Create File FIRST
        let filename = format!("segment_{}.drift", file_id);
        let file_path = dir.path().join(&filename);
        let file = std::fs::File::create(&file_path).unwrap();

        let (ids, vecs) = mock_data(1000, 50, dim);
        let q = Quantizer::train(&vecs, dim);

        let mut writer = BucketFileWriter::new_streaming(file, [0u8; 16], q, dim).unwrap();
        writer.write_batch(&ids, &vecs).unwrap();
        writer.finalize().unwrap();

        // 3. Initialize Storage & Manager AFTER file exists
        // Now DriftPageManager will see "segment_123.drift" on startup
        let op = create_local_operator(dir.path());
        let storage = Arc::new(DriftPageManager::new(op));

        storage.register_file(file_id, PathBuf::from(&filename));

        let manager = BucketManager::new(storage, 1);
        manager.update_mapping(bucket_id, file_id);

        // 4. Search
        let query = vec![0.0; 8];
        let results = manager
            .search(&[bucket_id], &query, 10, Arc::new(HashSetView::default()))
            .await;

        assert!(!results.is_empty(), "Manager failed to find results");
        assert_eq!(results.len(), 10);
    }
}
