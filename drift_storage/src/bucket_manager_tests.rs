#[cfg(test)]
mod tests {
    use crate::bucket_file_reader::BucketFileReader;
    use crate::bucket_file_writer::BucketFileWriter;
    use crate::bucket_manager::BucketManager;
    use crate::disk_manager::DriftPageManager; // Use the mapping-aware manager
    use drift_core::manifest::ManifestWrapper;
    use drift_core::quantizer::Quantizer;
    use drift_traits::{DiskSearcher, PageManager};
    use opendal::{Operator, services};
    use std::sync::{Arc, RwLock};
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
        let mut writer = BucketFileWriter::new(file, run_id, q.clone(), dim).unwrap();
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
        let mut writer = BucketFileWriter::new(file, [0u8; 16], q.clone(), dim).unwrap();
        writer.write_batch(&ids, &vecs).unwrap();
        writer.finalize().unwrap();

        // 2. Register
        storage.register_file(5, std::path::PathBuf::from(filename));

        // 3. Scan
        let mut reader = BucketFileReader::new(storage, 5);
        let query = vec![0.0; dim];

        let results = reader.scan(&query, 10).await.expect("Scan failed");

        assert_eq!(results.len(), 10);
        let found_ids: Vec<u64> = results.iter().map(|(id, _)| *id).collect();
        assert_eq!(found_ids, ids);
    }

    #[tokio::test]
    async fn test_bucket_manager_orchestration() {
        let dir = tempdir().unwrap();
        let op = create_local_operator(dir.path());
        let storage = Arc::new(DriftPageManager::new(op));

        // 1. Setup Manifest
        let mut manifest_inner = ManifestWrapper::new(8, "L2");
        let bucket_id = 100;
        let run_id = "run_uuid_123";

        manifest_inner.add_bucket(bucket_id, run_id.to_string(), vec![0.0; 8]);
        let manifest = Arc::new(RwLock::new(manifest_inner));

        // 2. Create File
        let filename = format!("segment_{}.drift", run_id);
        let file_path = dir.path().join(&filename);
        let file = std::fs::File::create(&file_path).unwrap();

        let dim = 8;
        let (ids, vecs) = mock_data(1000, 50, dim);
        let q = Quantizer::train(&vecs, dim);

        let mut writer = BucketFileWriter::new(file, [0u8; 16], q, dim).unwrap();
        writer.write_batch(&ids, &vecs).unwrap();
        writer.finalize().unwrap();

        // 3. Initialize Manager
        let manager = BucketManager::new(storage, manifest);

        // 4. Search
        // The Manager will lookup RunID -> Construct Path -> Register -> Scan
        let query = vec![0.0; 8];
        let results = manager.search(&[bucket_id], &query, 10).await;

        assert!(!results.is_empty(), "Manager failed to find results");
        assert_eq!(results.len(), 10);
        assert_eq!(results[0].0, 1000);
    }
}
