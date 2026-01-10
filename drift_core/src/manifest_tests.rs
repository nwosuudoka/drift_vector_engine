#[cfg(test)]
mod tests {
    use crate::manifest::ManifestWrapper;

    #[test]
    fn test_manifest_lifecycle() {
        // 1. Create
        let mut manifest = ManifestWrapper::new(128, "L2");
        assert_eq!(manifest.version(), 1);

        // 2. Mutate (Add Bucket)
        let centroid = vec![0.1; 128];
        manifest.add_bucket(100, "run_abc_123".to_string(), centroid.clone());

        // 3. Serialize
        let bytes = manifest.to_bytes();
        assert!(bytes.len() > 0);

        // 4. Deserialize
        let loaded = ManifestWrapper::from_bytes(&bytes).unwrap();

        assert_eq!(loaded.version(), 1);
        assert_eq!(loaded.get_buckets().len(), 1);
        assert_eq!(loaded.get_buckets()[0].run_id, "run_abc_123");
        assert_eq!(loaded.get_centroids().len(), 1);
    }

    #[test]
    fn test_manifest_initialization() {
        let manifest = ManifestWrapper::new(128, "L2");
        assert_eq!(manifest.version(), 1);
        assert_eq!(manifest.inner.dim, 128);
        assert_eq!(manifest.inner.metric, "L2");
        assert!(manifest.get_buckets().is_empty());
        assert!(manifest.get_centroids().is_empty());
    }

    #[test]
    fn test_bucket_management() {
        let mut manifest = ManifestWrapper::new(128, "L2");

        // 1. Add Bucket
        let centroid = vec![0.1; 128];
        manifest.add_bucket(100, "run_A".to_string(), centroid.clone());

        let buckets = manifest.get_buckets();
        assert_eq!(buckets.len(), 1);
        assert_eq!(buckets[0].id, 100);
        assert_eq!(buckets[0].run_id, "run_A");

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
        manifest.add_bucket(100, "run_B".to_string(), centroid.clone());
        let buckets_after = manifest.get_buckets();
        assert_eq!(buckets_after.len(), 1); // Still 1
        assert_eq!(buckets_after[0].run_id, "run_B"); // Updated run_id
        assert_eq!(buckets_after[0].vector_count, 0); // Reset count (as per add_bucket logic)
    }

    #[test]
    fn test_bucket_removal() {
        let mut manifest = ManifestWrapper::new(128, "L2");
        manifest.add_bucket(1, "run_1".to_string(), vec![0.0; 128]);
        manifest.add_bucket(2, "run_2".to_string(), vec![1.0; 128]);

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
        let mut manifest = ManifestWrapper::new(128, "L2");
        assert_eq!(manifest.version(), 1);

        manifest.bump_version();
        assert_eq!(manifest.version(), 2);

        manifest.bump_version();
        assert_eq!(manifest.version(), 3);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mut manifest = ManifestWrapper::new(64, "COSINE");

        // Populate with some complex state
        for i in 0..10 {
            manifest.add_bucket(i, format!("run_{}", i), vec![i as f32; 64]);
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
}
