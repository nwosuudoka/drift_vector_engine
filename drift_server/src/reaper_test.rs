#[cfg(test)]
mod tests {
    use crate::local_staging::LocalStagingManager;
    use crate::persistence_v2::PersistenceManager;
    use crate::reaper::Reaper;
    use drift_storage::bucket_manager::{BucketVersion, StorageClass};
    use opendal::{Operator, services};
    use std::sync::Arc;
    use tempfile::tempdir;

    // --- Helpers ---
    fn create_fs_operator(path: &std::path::Path) -> Operator {
        let builder = services::Fs::default().root(path.to_str().unwrap());
        Operator::new(builder).unwrap().finish()
    }

    async fn create_dummy_file(op: &Operator, name: &str) {
        op.write(name, b"dummy data".as_slice()).await.unwrap();
    }

    #[tokio::test]
    async fn test_reaper_deletes_physical_files_after_promotion() {
        let dir = tempdir().unwrap();
        let staging_dir = dir.path().join("staging");
        let remote_dir = dir.path().join("remote");

        std::fs::create_dir(&staging_dir).unwrap();
        std::fs::create_dir(&remote_dir).unwrap();

        // 1. Setup Components
        let staging = Arc::new(LocalStagingManager::new(&staging_dir).unwrap());
        let remote_op = create_fs_operator(&remote_dir);
        let persistence = Arc::new(PersistenceManager::new(remote_op.clone()));

        let mut reaper = Reaper::new(staging.clone(), persistence);

        // 2. Create Dummy Files
        // "Old" Local File (e.g. frozen staging file)
        let local_filename = "bucket_1_frozen.drift";
        std::fs::write(staging_dir.join(local_filename), b"local data").unwrap();

        // "Old" Remote File (e.g. previous S3 segment)
        let remote_filename = "bucket_1_base_v1.drift";
        create_dummy_file(&remote_op, remote_filename).await;

        // 3. Create a BucketVersion representing the "Old State" we want to cleanup
        // This simulates a bucket that just finished promoting to a NEW S3 file,
        // so the old local frozen file and old remote base file are now garbage.
        let version = Arc::new(BucketVersion {
            bucket_id: 1,
            path: "ignored_active.drift".to_string(), // The reaper ignores this field for promoting class
            class: StorageClass::Promoting {
                local_active: "bucket_1_new_active.drift".to_string(), // Kept
                local_frozen: local_filename.to_string(),              // To Delete
                remote_path: Some(remote_filename.to_string()),        // To Delete
            },
        });

        // 4. Schedule Deletion
        reaper.schedule_deletion(version.clone());

        // 5. Run Cycle - SAFETY CHECK
        // We still hold `version` in this test scope (Arc count = 2).
        // The Reaper should REFUSE to delete it because it thinks a searcher is using it.
        reaper.run_cycle().await;

        assert!(
            staging_dir.join(local_filename).exists(),
            "Reaper deleted local file while Arc was held!"
        );
        assert!(
            remote_op.exists(remote_filename).await.unwrap(),
            "Reaper deleted remote file while Arc was held!"
        );

        // 6. Drop Reference and Run Cycle - REAL CLEANUP
        drop(version); // Arc count drops to 1 (held by Reaper queue)

        reaper.run_cycle().await;

        // 7. Verify physical deletion
        assert!(
            !staging_dir.join(local_filename).exists(),
            "Reaper failed to delete local frozen file"
        );
        assert!(
            !remote_op.exists(remote_filename).await.unwrap(),
            "Reaper failed to delete remote S3 file"
        );
    }

    #[tokio::test]
    async fn test_reaper_cleans_simple_local_files() {
        let dir = tempdir().unwrap();
        let staging_dir = dir.path().join("staging");
        std::fs::create_dir(&staging_dir).unwrap();

        let staging = Arc::new(LocalStagingManager::new(&staging_dir).unwrap());
        // Mock remote op (unused here)
        let remote_op = create_fs_operator(dir.path());
        let persistence = Arc::new(PersistenceManager::new(remote_op));

        let mut reaper = Reaper::new(staging, persistence);

        // 1. Create Local File
        let filename = "bucket_99.drift";
        std::fs::write(staging_dir.join(filename), b"trash").unwrap();

        // 2. Schedule
        let version = Arc::new(BucketVersion {
            bucket_id: 99,
            path: filename.to_string(),
            class: StorageClass::Local,
        });

        reaper.schedule_deletion(version); // Pass ownership, we don't keep a clone

        // 3. Run
        reaper.run_cycle().await;

        // 4. Verify
        assert!(
            !staging_dir.join(filename).exists(),
            "Reaper failed to delete simple local file"
        );
    }
}
