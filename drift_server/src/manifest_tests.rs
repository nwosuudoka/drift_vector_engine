#[cfg(test)]
mod tests {
    use crate::manifest::ServerManifestManager;
    use tempfile::tempdir;

    #[test]
    fn test_server_manifest_atomic_split_transaction() {
        let dir = tempdir().unwrap();
        let manager = ServerManifestManager::new(dir.path(), 128).unwrap();

        // 1. Initial State: Bucket 1 exists
        manager
            .apply_atomic(|m| {
                m.add_bucket(1, "run_A".into(), vec![0.0; 128]);
                m.update_bucket_stats(1, 100, 0);
            })
            .unwrap();

        // 2. ATOMIC SPLIT (The Transaction)
        // We modify Bucket 1 AND create Bucket 2 in the same commit.
        manager
            .apply_atomic(|m| {
                // Update Bucket 1 (Left Child)
                m.add_bucket(1, "run_B_Left".into(), vec![-1.0; 128]);
                m.update_bucket_stats(1, 50, 0);

                // Create Bucket 2 (Right Child)
                m.add_bucket(2, "run_B_Right".into(), vec![1.0; 128]);
                m.update_bucket_stats(2, 50, 0);
            })
            .unwrap();

        // 3. Verify Disk State
        // Simulating a crash/restart by reloading from disk
        let manager2 = ServerManifestManager::new(dir.path(), 128).unwrap();
        let state = manager2.get_state();

        assert_eq!(state.version(), 3); // Init(1) + Add(2) + Split(3)
        let buckets = state.get_buckets();

        assert_eq!(buckets.len(), 2, "Split should have persisted 2 buckets");

        let b1 = buckets.iter().find(|b| b.id == 1).unwrap();
        assert_eq!(b1.run_id, "run_B_Left");

        let b2 = buckets.iter().find(|b| b.id == 2).unwrap();
        assert_eq!(b2.run_id, "run_B_Right");
    }

    #[test]
    fn test_manifest_corruption_safety() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("manifest.pb");

        // 1. Create a corrupt file (random garbage)
        std::fs::write(&path, b"not_a_valid_protobuf").unwrap();

        // 2. Attempt to open
        let result = ServerManifestManager::new(dir.path(), 128);

        // 3. Assert Failure (Do NOT silently overwrite)
        assert!(result.is_err(), "Should fail on corrupt manifest");
        // Ensure we didn't nuke the file
        assert!(path.exists());
        assert_eq!(std::fs::read(&path).unwrap(), b"not_a_valid_protobuf");
    }
}
