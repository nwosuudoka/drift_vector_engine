#[cfg(test)]
mod tests {
    use crate::{manifest::ManifestWrapper, math::Metric};

    #[test]
    fn test_manifest_lifecycle() {
        // 1. Create
        let metric = Metric::L2;
        let mut manifest = ManifestWrapper::new(128, metric);
        assert_eq!(manifest.version(), 1);

        // 2. Mutate (Add Bucket)
        let centroid = vec![0.1; 128];
        manifest.add_bucket(100, "run_abc_123".to_string(), Some(centroid.clone()));

        // 3. Serialize
        let bytes = manifest.to_bytes();
        assert!(bytes.len() > 0);

        // 4. Deserialize
        let loaded = ManifestWrapper::from_bytes(&bytes).unwrap();

        assert_eq!(loaded.version(), 1);
        assert_eq!(loaded.get_buckets().len(), 1);
        assert_eq!(loaded.get_buckets()[0].run_id, "run_abc_123");
        assert_eq!(
            loaded.get_buckets()[0].object_path,
            "bucket_100_run_abc_123.driftu"
        );
        assert_eq!(loaded.get_centroids().len(), 1);
    }

    #[test]
    fn test_manifest_initialization() {
        let metric = Metric::L2;
        let manifest = ManifestWrapper::new(128, metric);
        assert_eq!(manifest.version(), 1);
        assert_eq!(manifest.inner.dim, 128);
        assert_eq!(manifest.inner.metric, "L2");
        assert!(manifest.get_buckets().is_empty());
        assert!(manifest.get_centroids().is_empty());
    }

    #[test]
    fn test_bucket_management() {
        let metric = Metric::L2;
        let mut manifest = ManifestWrapper::new(128, metric);

        // 1. Add Bucket
        let centroid = vec![0.1; 128];
        manifest.add_bucket(100, "run_A".to_string(), Some(centroid.clone()));

        let buckets = manifest.get_buckets();
        assert_eq!(buckets.len(), 1);
        assert_eq!(buckets[0].id, 100);
        assert_eq!(buckets[0].run_id, "run_A");
        assert_eq!(buckets[0].object_path, "bucket_100_run_A.driftu");
        assert!(buckets[0].object_fingerprint.is_empty());

        let centroids = manifest.get_centroids();
        assert_eq!(centroids.len(), 1);
        assert_eq!(centroids[0].id, 100);
        assert_eq!(centroids[0].vector, centroid);

        // 2. Update Bucket Stats
        manifest.update_bucket_stats(100, 5000, 10);
        let updated_b = &manifest.get_buckets()[0];
        assert_eq!(updated_b.vector_count, 5000);
        assert_eq!(updated_b.tombstone_count, 10);

        // 3. Replace Bucket (Simulating Split/Update)
        // Adding the same ID should overwrite the old entry
        manifest.add_bucket(100, "run_B".to_string(), Some(centroid.clone()));
        let buckets_after = manifest.get_buckets();
        assert_eq!(buckets_after.len(), 1); // Still 1
        assert_eq!(buckets_after[0].run_id, "run_B"); // Updated run_id
        assert_eq!(buckets_after[0].object_path, "bucket_100_run_B.driftu");
        assert_eq!(buckets_after[0].vector_count, 0); // Reset count (as per add_bucket logic)
    }

    #[test]
    fn test_bucket_removal() {
        let metric = Metric::L2;
        let mut manifest = ManifestWrapper::new(128, metric);
        manifest.add_bucket(1, "run_1".to_string(), Some(vec![0.0; 128]));
        manifest.add_bucket(2, "run_2".to_string(), Some(vec![1.0; 128]));

        assert_eq!(manifest.get_buckets().len(), 2);

        // Remove ID 1
        manifest.remove_bucket(1);

        let buckets = manifest.get_buckets();
        assert_eq!(buckets.len(), 1);
        assert_eq!(buckets[0].id, 2);

        let centroids = manifest.get_centroids();
        assert_eq!(centroids.len(), 1);
        assert_eq!(centroids[0].id, 2);
    }

    #[test]
    fn test_versioning() {
        let metric = Metric::L2;
        let mut manifest = ManifestWrapper::new(128, metric);
        assert_eq!(manifest.version(), 1);

        manifest.bump_version();
        assert_eq!(manifest.version(), 2);

        manifest.bump_version();
        assert_eq!(manifest.version(), 3);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let metric = Metric::COSINE;
        let mut manifest = ManifestWrapper::new(64, metric);

        // Populate with some complex state
        for i in 0..10 {
            manifest.add_bucket(i, format!("run_{}", i), Some(vec![i as f32; 64]));
            manifest.update_bucket_stats(i, i as u64 * 100, i);
        }
        manifest.bump_version();

        // Serialize
        let bytes = manifest.to_bytes();
        assert!(!bytes.is_empty());

        // Deserialize
        let loaded = ManifestWrapper::from_bytes(&bytes).expect("Failed to deserialize");

        // Verify
        assert_eq!(loaded.version(), 2);
        assert_eq!(loaded.inner.dim, 64);
        assert_eq!(loaded.inner.metric, "COSINE");
        assert_eq!(loaded.get_buckets().len(), 10);

        // Verify specific item
        let b5 = loaded.get_buckets().iter().find(|b| b.id == 5).unwrap();
        assert_eq!(b5.run_id, "run_5");
        assert_eq!(b5.vector_count, 500);
        assert_eq!(b5.tombstone_count, 5);
    }

    #[test]
    fn test_corrupt_data_handling() {
        let garbage = vec![0u8, 1, 2, 3, 255];
        let result = ManifestWrapper::from_bytes(&garbage);
        assert!(result.is_err(), "Should fail on corrupt bytes");
    }

    #[test]
    fn test_remote_meta_update() {
        let metric = Metric::L2;
        let mut manifest = ManifestWrapper::new(8, metric);
        manifest.add_bucket(42, "run_old".to_string(), Some(vec![0.0; 8]));

        manifest.update_bucket_remote_meta(
            42,
            "run_new".to_string(),
            "remote/custom/path_42.driftu".to_string(),
            "len=123|etag=abc".to_string(),
        );

        let bucket = manifest.get_buckets().iter().find(|b| b.id == 42).unwrap();
        assert_eq!(bucket.run_id, "run_new");
        assert_eq!(bucket.object_path, "remote/custom/path_42.driftu");
        assert_eq!(bucket.object_fingerprint, "len=123|etag=abc");
    }

    #[test]
    fn test_manifest_metric_parser() {
        let mut manifest = ManifestWrapper::new(16, Metric::L2);
        assert_eq!(manifest.metric().unwrap(), Metric::L2);

        manifest.inner.metric = "COSINE".to_string();
        assert_eq!(manifest.metric().unwrap(), Metric::COSINE);

        // Backward compatibility for older manifests where metric was not set.
        manifest.inner.metric.clear();
        assert_eq!(manifest.metric().unwrap(), Metric::L2);

        manifest.inner.metric = "invalid".to_string();
        assert!(manifest.metric().is_err());
    }
}
