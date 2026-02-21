#[cfg(test)]
mod tests {
    use crate::janitor::{Janitor, JanitorConfig, JanitorVars};
    use crate::local_staging::LocalStagingManager;
    use crate::manifest::ServerManifestManager;
    use crate::persistence::PersistenceManager;
    use drift_core::index::VectorIndex;
    use drift_core::lock_manager::BucketCoordinator;
    use drift_core::math::Metric;
    use drift_core::router::Router;
    use drift_core::wal::WalManager;
    use drift_kv::bitstore::BitStore;
    use drift_storage::bucket_manager::BucketManager;
    use opendal::{Operator, services};
    use parking_lot::{Mutex, RwLock};
    use std::sync::Arc;
    use std::time::Duration;
    use tempfile::tempdir;

    fn create_fs_operator(path: &std::path::Path) -> Operator {
        let builder = services::Fs::default().root(path.to_str().unwrap());
        Operator::new(builder).unwrap().finish()
    }

    #[tokio::test]
    async fn test_janitor_flush_lifecycle() {
        let dir = tempdir().unwrap();
        let data_dir = dir.path().join("data");
        std::fs::create_dir(&data_dir).unwrap();
        let dim = 2;

        let manifest = Arc::new(ServerManifestManager::new(dir.path(), dim as u32).unwrap());
        let staging = Arc::new(LocalStagingManager::new(&data_dir).unwrap());
        let op = create_fs_operator(&data_dir);
        let persistence = Arc::new(PersistenceManager::new(op.clone()));
        let coordinator = Arc::new(BucketCoordinator::new());

        let bucket_manager = Arc::new(BucketManager::new(
            op.clone(),
            op.clone(),
            8,
            coordinator.clone(),
            Metric::L2,
        ));

        manifest
            .apply_atomic(|m| {
                m.add_bucket(0, "run_init".to_string(), Some(vec![0.0; dim]));
            })
            .unwrap();

        let wal_dir = dir.path().join("wal");
        let wal_mgr = Arc::new(Mutex::new(WalManager::new(&wal_dir).unwrap()));

        let centroids = vec![drift_core::manifest::pb::Centroid {
            id: 0,
            vector: vec![0.0; dim],
        }];
        let router = Arc::new(RwLock::new(
            Router::new(&centroids, &[0], dim, Metric::L2).unwrap(),
        ));

        let kv = Arc::new(BitStore::new(dir.path().join("kv")).unwrap());

        let index = Arc::new(VectorIndex::new(
            dim,
            10,
            router,
            wal_mgr,
            bucket_manager.clone(),
            kv,
        ));

        let janitor = Janitor::new(JanitorConfig {
            index: index.clone(),
            manifest: manifest.clone(),
            staging: staging.clone(),
            persistence,
            bucket_manager: bucket_manager.clone(),
            coordinator: coordinator.clone(),
            vars: JanitorVars {
                promotion_threshold_bytes: 100,
                check_interval: Duration::from_millis(10),
                ..Default::default()
            },
        });

        // 4. Trigger Flush Condition (Split Batches)
        // Batch A: Fill Capacity (10 items) -> Triggers Rotation
        let batch_a: Vec<(u64, Vec<f32>)> = (0..10).map(|i| (i, vec![0.0; dim])).collect();
        let rotated = index.insert_batch(&batch_a).unwrap();
        assert!(rotated, "First batch should fill capacity and rotate");

        // Batch B: New Active Data (5 items) -> No Rotation
        let batch_b: Vec<(u64, Vec<f32>)> = (10..15).map(|i| (i, vec![0.0; dim])).collect();
        let rotated_b = index.insert_batch(&batch_b).unwrap();
        assert!(!rotated_b, "Second batch should fit in new active table");

        // 5. Run Janitor (Flushes Frozen)
        let handle = tokio::spawn(async move { janitor.run().await });
        tokio::time::sleep(Duration::from_millis(200)).await;
        handle.abort();

        // 6. Verify Persistence
        let state = manifest.get_state();
        let b0 = state.get_buckets().iter().find(|b| b.id == 0).unwrap();

        // Assert: Only the 10 frozen items were flushed.
        assert_eq!(
            b0.vector_count, 10,
            "Manifest should reflect ONLY flushed frozen items"
        );

        //  Expect None, because Janitor should have cleared it!
        assert!(
            index.flush_frozen().is_none(),
            "Frozen slot should be cleared after flush"
        );

        // Verify WAL Rotation:
        // 1 Frozen WAL was deleted. 1 Active WAL should remain.
        let files = std::fs::read_dir(&wal_dir).unwrap().count();
        assert_eq!(files, 1, "Should have exactly 1 active WAL file remaining");
    }
}

#[cfg(test)]
mod stress_tests {
    use crate::janitor::{Janitor, JanitorConfig, JanitorVars};
    use crate::local_staging::LocalStagingManager;
    use crate::manifest::ServerManifestManager;
    use crate::persistence::PersistenceManager;
    use drift_core::index::VectorIndex;
    use drift_core::lock_manager::BucketCoordinator;
    use drift_core::math::Metric;
    use drift_core::router::Router;
    use drift_core::wal::WalManager;
    use drift_kv::bitstore::BitStore;
    use drift_storage::bucket_manager::BucketManager;
    use opendal::{Operator, services};
    use parking_lot::{Mutex, RwLock};
    use std::sync::Arc;
    use std::time::Duration;
    use tempfile::tempdir;

    fn create_fs_operator(path: &std::path::Path) -> Operator {
        let mut builder = services::Fs::default();
        builder = builder.root(path.to_str().unwrap());
        Operator::new(builder).unwrap().finish()
    }

    async fn setup_env(
        dir: &std::path::Path,
        capacity: usize,
        dim: usize,
    ) -> (
        Arc<VectorIndex>,
        Arc<BucketManager>,
        Janitor,
        Arc<ServerManifestManager>,
    ) {
        let _ = tracing_subscriber::fmt()
            .with_env_filter("drift_storage=info,drift_server=info") // Filter for relevant crates
            .with_test_writer() // Important: Prints to test output
            .try_init();

        let data_dir = dir.join("data");
        std::fs::create_dir(&data_dir).unwrap();

        let manifest = Arc::new(ServerManifestManager::new(dir, dim as u32).unwrap());
        let staging = Arc::new(LocalStagingManager::new(&data_dir).unwrap());
        let op = create_fs_operator(&data_dir);
        let persistence = Arc::new(PersistenceManager::new(op.clone()));

        let coordinator = Arc::new(BucketCoordinator::new());

        let bucket_manager = Arc::new(BucketManager::new(
            op.clone(),
            op.clone(),
            8,
            coordinator.clone(),
            Metric::L2,
        ));

        manifest
            .apply_atomic(|m| {
                m.add_bucket(0, "".to_string(), Some(vec![0.0; dim]));
            })
            .unwrap();

        let wal_dir = dir.join("wal");
        let wal_mgr = Arc::new(Mutex::new(WalManager::new(&wal_dir).unwrap()));

        let centroids = vec![drift_core::manifest::pb::Centroid {
            id: 0,
            vector: vec![0.0; dim],
        }];
        let router = Arc::new(RwLock::new(
            Router::new(&centroids, &[0], dim, Metric::L2).unwrap(),
        ));

        let bit_store = Arc::new(BitStore::new(dir.join("kv")).unwrap());

        let index = Arc::new(VectorIndex::new(
            dim,
            capacity,
            router,
            wal_mgr,
            bucket_manager.clone(),
            bit_store,
        ));

        // Low threshold to force S3 promotion
        let promotion_threshold = 1024;

        let janitor = Janitor::new(JanitorConfig {
            index: index.clone(),
            manifest: manifest.clone(),
            staging: staging,
            persistence,
            bucket_manager: bucket_manager.clone(),
            coordinator: coordinator.clone(),
            vars: JanitorVars {
                promotion_threshold_bytes: promotion_threshold,
                check_interval: Duration::from_millis(5),
                ..Default::default()
            },
        });

        (index, bucket_manager, janitor, manifest)
    }

    #[tokio::test]
    async fn test_consistent_reads_during_tiering() {
        let dir = tempdir().unwrap();
        let dim = 8;
        let capacity = 10;
        let (index, bucket_manager, janitor, _) = setup_env(dir.path(), capacity, dim).await;

        let janitor_task = tokio::spawn(async move { janitor.run().await });

        // 1. Insert Target Data (ID 999)
        let target_vec = vec![100.0; dim];
        index.insert(999, &target_vec).unwrap();

        // 2. Spawn Reader Thread
        let index_ref = index.clone();
        let bucket_mgr_ref = bucket_manager.clone();

        let reader_handle = tokio::spawn(async move {
            for i in 0..50 {
                let results = index_ref
                    .search(&target_vec, 5, 0.9, 1.0, 10.0)
                    .await
                    .unwrap();

                let found = results.iter().any(|(id, _)| *id == 999);

                if !found {
                    // Debugging
                    println!("❌ Read #{}: MISSING ID 999.", i);
                    if let Some(v) = bucket_mgr_ref.get_version(0) {
                        println!("   -> Bucket 0 is at: {:?} (Path: {})", v.class, v.path);
                    } else {
                        println!("   -> Bucket 0 NOT FOUND in Registry!");
                    }
                    return false;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            true
        });

        // 3. Trigger Tiering Moves (Flush -> Promote)
        for i in 0..100 {
            let vec = vec![i as f32; dim];
            index.insert(i, &vec).unwrap();
            tokio::time::sleep(Duration::from_millis(5)).await;
        }

        // Wait for reader
        let success = reader_handle.await.unwrap();
        janitor_task.abort();

        assert!(
            success,
            "Read Consistency Violation! Lost data during tiering transition."
        );
    }
}

#[cfg(test)]
mod janitor_split_test {
    use crate::janitor::{Janitor, JanitorConfig, JanitorVars};
    use crate::local_staging::LocalStagingManager;
    use crate::manifest::ServerManifestManager;
    use crate::persistence::PersistenceManager;
    use drift_core::index::VectorIndex;
    use drift_core::lock_manager::BucketCoordinator;
    use drift_core::math::Metric;
    use drift_core::quantizer::Quantizer;
    use drift_core::router::Router;
    use drift_core::wal::WalManager;
    use drift_kv::bitstore::BitStore;
    use drift_storage::bucket_file_writer::BucketFileWriter;
    use drift_storage::bucket_manager::{BucketManager, StorageClass};
    use opendal::{Operator, services};
    use parking_lot::{Mutex, RwLock};
    use std::sync::Arc;
    use tempfile::tempdir;

    fn create_fs_operator(path: &std::path::Path) -> Operator {
        let builder = services::Fs::default().root(path.to_str().unwrap());
        Operator::new(builder).unwrap().finish()
    }

    async fn create_bucket_file(
        dir: &std::path::Path,
        filename: &str,
        ids: &[u64],
        vecs: &[Vec<f32>],
        dim: usize,
    ) {
        let path = dir.join(filename);
        let file = std::fs::File::create(&path).unwrap();
        let flat: Vec<f32> = vecs.iter().flatten().copied().collect();
        let q = Quantizer::train(&flat, dim);
        let mut writer = BucketFileWriter::new_streaming(file, [0u8; 16], q, dim).unwrap();
        writer.write_batch(ids, &flat).unwrap();
        writer.finalize().unwrap();
    }

    #[tokio::test]
    async fn test_janitor_performs_split_and_updates_manifest() {
        let dir = tempdir().unwrap();
        let data_dir = dir.path().join("data");
        std::fs::create_dir(&data_dir).unwrap();
        let dim = 2;

        // 1. Setup Components
        let manifest = Arc::new(ServerManifestManager::new(dir.path(), dim as u32).unwrap());
        let staging = Arc::new(LocalStagingManager::new(&data_dir).unwrap());
        let op = create_fs_operator(&data_dir);
        let persistence = Arc::new(PersistenceManager::new(op.clone()));
        let coordinator = Arc::new(BucketCoordinator::new());
        let bucket_manager = Arc::new(BucketManager::new(
            op.clone(),
            op.clone(),
            4,
            coordinator.clone(),
            Metric::L2,
        ));

        // 2. Setup Index & Router
        let wal_dir = dir.path().join("wal");
        let wal_mgr = Arc::new(Mutex::new(WalManager::new(&wal_dir).unwrap()));
        let kv = Arc::new(BitStore::new(dir.path().join("kv")).unwrap());

        // Create initial bucket (ID 1)
        let centroids = vec![drift_core::manifest::pb::Centroid {
            id: 1,
            vector: vec![0.0; dim],
        }];
        let router = Arc::new(RwLock::new(
            Router::new(&centroids, &[100], dim, Metric::L2).unwrap(),
        ));

        let index = Arc::new(VectorIndex::new(
            dim,
            1000,
            router.clone(),
            wal_mgr,
            bucket_manager.clone(),
            kv,
        ));

        // 3. Populate Bucket 1 (Needs Splitting)
        // Cluster Left: [0,0] .. [0.1, 0.1]
        // Cluster Right: [10,10] .. [10.1, 10.1]
        // Total 100 items (50 Left, 50 Right)
        let mut ids = Vec::new();
        let mut vecs = Vec::new();
        for i in 0..50 {
            ids.push(i as u64);
            vecs.push(vec![0.1 * (i as f32), 0.1 * (i as f32)]);
        }
        for i in 50..100 {
            ids.push(i as u64);
            vecs.push(vec![10.0 + 0.1 * (i as f32), 10.0 + 0.1 * (i as f32)]);
        }

        let filename = "bucket_1.drift";
        create_bucket_file(&data_dir, filename, &ids, &vecs, dim).await;

        // Register in Manager & Manifest
        bucket_manager.register_bucket(1, filename.to_string(), StorageClass::Local);
        manifest
            .apply_atomic(|m| {
                m.add_bucket(1, "run1".into(), Some(vec![5.0, 5.0])); // Centroid in middle
                m.update_bucket_stats(1, 100, 0);
            })
            .unwrap();

        // 4. Initialize Janitor
        let janitor = Janitor::new(JanitorConfig {
            index: index.clone(),
            manifest: manifest.clone(),
            staging: staging.clone(),
            persistence,
            bucket_manager: bucket_manager.clone(),
            coordinator: coordinator.clone(),
            vars: JanitorVars::default(),
        });

        // 5. EXECUTE SPLIT
        // We call the private method directly? Or invoke via run loop?
        // Since `perform_split` is private, we can't call it directly in integration test unless we make it pub(crate).
        // Let's assume we made it pub(crate) or we use `run()` with a trigger.
        // For reliability, let's assume we expose `perform_split` as `pub(crate)`.

        // Note: You need to make `perform_split` `pub(crate)` in `janitor.rs` for this test to compile.
        janitor.perform_split(1).await.expect("Split failed");

        // 6. Verify Results
        let state = manifest.get_state();
        let buckets = state.get_buckets();

        // A. Old bucket gone?
        assert!(
            !buckets.iter().any(|b| b.id == 1),
            "Bucket 1 should be removed"
        );

        // B. New buckets present?
        // IDs should be 2 and 3 (since next_id started at 2)
        let b2 = buckets
            .iter()
            .find(|b| b.id == 2)
            .expect("Bucket 2 missing");
        let b3 = buckets
            .iter()
            .find(|b| b.id == 3)
            .expect("Bucket 3 missing");

        // C. Counts Check
        assert_eq!(b2.vector_count + b3.vector_count, 100);
        assert!(
            b2.vector_count > 40 && b3.vector_count > 40,
            "Split should be roughly balanced"
        );

        // D. Router Update Check
        let router_snap = router.read().get_snapshot();
        assert!(router_snap.1.contains(&2));
        assert!(router_snap.1.contains(&3));
        assert!(!router_snap.1.contains(&1));
    }
}

#[cfg(test)]
mod edge_case_tests {
    use drift_core::index::{MaintenanceStatus, VectorIndex};
    // Note: MaintenanceStatus is in index.rs
    use drift_core::lock_manager::BucketCoordinator;
    use drift_core::manifest::pb::Centroid;
    use drift_core::math::Metric;
    use drift_core::quantizer::Quantizer;
    use drift_core::router::Router;
    use drift_core::wal::WalManager;
    use drift_kv::bitstore::BitStore;
    use drift_storage::bucket_file_writer::BucketFileWriter;
    use drift_storage::bucket_manager::{BucketManager, StorageClass};
    use opendal::{Operator, services};
    use parking_lot::{Mutex, RwLock};
    use std::sync::Arc;
    use tempfile::tempdir;

    // --- SETUP HELPER ---
    fn create_fs_operator(path: &std::path::Path) -> Operator {
        let builder = services::Fs::default().root(path.to_str().unwrap());
        Operator::new(builder).unwrap().finish()
    }

    async fn setup_env(
        dir: &std::path::Path,
        dim: usize,
    ) -> (Arc<VectorIndex>, Arc<BucketManager>) {
        let data_dir = dir.join("data");
        std::fs::create_dir_all(&data_dir).unwrap();

        let op = create_fs_operator(&data_dir);
        let bucket_manager = Arc::new(BucketManager::new(
            op.clone(),
            op.clone(),
            4,
            Arc::new(BucketCoordinator::new()),
            Metric::L2,
        ));

        let wal_dir = dir.join("wal");
        let wal = Arc::new(Mutex::new(WalManager::new(&wal_dir).unwrap()));
        let kv = Arc::new(BitStore::new(dir.join("kv")).unwrap());

        // Router with 2 buckets initially to allow neighbor checking
        let centroids = vec![
            Centroid {
                id: 1,
                vector: vec![0.0; dim],
            }, // Target
            Centroid {
                id: 2,
                vector: vec![100.0; dim],
            }, // Neighbor
        ];
        let router = Arc::new(RwLock::new(
            Router::new(&centroids, &[0, 0], dim, Metric::L2).unwrap(),
        ));

        let index = Arc::new(VectorIndex::new(
            dim,
            1000,
            router,
            wal,
            bucket_manager.clone(),
            kv,
        ));

        (index, bucket_manager)
    }

    async fn create_bucket_file(
        dir: &std::path::Path,
        name: &str,
        ids: &[u64],
        vecs: &[Vec<f32>],
        dim: usize,
    ) {
        let path = dir.join("data").join(name);
        let file = std::fs::File::create(&path).unwrap();
        let flat: Vec<f32> = vecs.iter().flatten().copied().collect();
        let q = Quantizer::train(&flat, dim);
        let mut w = BucketFileWriter::new_streaming(file, [0u8; 16], q, dim).unwrap();
        w.write_batch(ids, &flat).unwrap();
        w.finalize().unwrap();
    }

    // --- TEST 1: SINGULARITY DETECTION ---
    #[tokio::test]
    async fn test_split_aborts_on_singularity() {
        let dir = tempdir().unwrap();
        let dim = 2;
        let (index, bucket_mgr) = setup_env(dir.path(), dim).await;

        // 1. Create a "Singularity" Bucket
        // 100 identical vectors at [10.0, 10.0]
        let count = 100;
        let ids: Vec<u64> = (0..count).collect();
        let vecs = vec![vec![10.0, 10.0]; count as usize];

        create_bucket_file(dir.path(), "singularity.drift", &ids, &vecs, dim).await;
        bucket_mgr.register_bucket(1, "singularity.drift".into(), StorageClass::Local);

        // 2. Attempt Split
        let result = index.calculate_split(1).await.unwrap();

        // 3. Assert Failure
        match result {
            Err(MaintenanceStatus::SkippedSingularity { variance }) => {
                println!("✅ Correctly skipped singularity. Variance: {}", variance);
                assert!(variance < 0.001);
            }
            _ => panic!(
                "Should have skipped singularity! Got: {:?}",
                // Debug print hack if enum isn't Debug, but it is
                "Success/Other"
            ),
        }
    }

    #[tokio::test]
    async fn test_split_identifies_defectors() {
        let dir = tempdir().unwrap();
        let dim = 2;
        let (index, bucket_mgr) = setup_env(dir.path(), dim).await;

        // 1. Setup Environment
        // Anchor A: [0.0] (40 items)
        // Anchor B: [100.0] (40 items)
        // Defectors: [130.0] (10 items) -> Attached to B, but closer to Neighbor [140.0]

        let mut ids = Vec::new();
        let mut vecs = Vec::new();

        // A. Anchor Cluster 1 (0.0)
        for i in 0..40 {
            ids.push(i);
            vecs.push(vec![0.0; dim]);
        }

        // B. Anchor Cluster 2 (100.0)
        for i in 40..80 {
            ids.push(i);
            vecs.push(vec![100.0; dim]);
        }

        // C. Defectors (130.0)
        // K-Means will calculate centroids at approx 0.0 and 100.0 (weighted by mass).
        // Defector Local Dist: |130 - 100| = 30.
        // Defector Neighbor Dist: |130 - 140| = 10.
        // Hysteresis: 10 < (30 * 0.9). -> TRUE.
        for i in 80..90 {
            ids.push(i);
            vecs.push(vec![130.0; dim]);
        }

        create_bucket_file(dir.path(), "mixed.drift", &ids, &vecs, dim).await;
        bucket_mgr.register_bucket(1, "mixed.drift".into(), StorageClass::Local);

        // 2. Setup Neighbor Bucket (ID 2 at [140.0])
        {
            let mut w = index.get_router().write();
            // Ensure ID 2 exists at the correct location
            if w.get_centroid(2).is_some() {
                w.remove_bucket(2);
            }
            w.add_bucket(2, vec![140.0; dim]);
        }

        // 3. Calculate Split
        // This is deterministic now because the variance of merging [0, 100] is huge,
        // so K-Means will always split them.
        let proposal = index
            .calculate_split(1)
            .await
            .unwrap()
            .expect("Split calculation failed");

        // 4. Verify Defectors
        // We expect exactly 10 defectors
        assert_eq!(
            proposal.loopback.len(),
            10,
            "Should identify exactly 10 defectors"
        );

        // Verify we caught the high IDs (80..90)
        let defector_id = proposal.loopback[0].0;
        assert!(defector_id >= 80, "Loopback items should be the high IDs");

        // 5. Verify Remaining Split (Anchors A and B)
        // 40 items in Left, 40 items in Right
        let total_kept = proposal.left.count + proposal.right.count;
        assert_eq!(total_kept, 80);

        // Optional: Verify balance
        assert_eq!(proposal.left.count, 40);
        assert_eq!(proposal.right.count, 40);
    }

    // --- TEST 3: TIERED STORAGE MERGE ---
    #[tokio::test]
    async fn test_split_merges_tiered_storage() {
        let dir = tempdir().unwrap();
        let dim = 2;
        let (index, bucket_mgr) = setup_env(dir.path(), dim).await;

        // 1. Remote Base (IDs 0-49)
        let ids_base: Vec<u64> = (0..50).collect();
        let vecs_base = vec![vec![0.0, 0.0]; 50];
        create_bucket_file(dir.path(), "base.drift", &ids_base, &vecs_base, dim).await;

        // 2. Local Delta (IDs 50-99)
        let ids_delta: Vec<u64> = (50..100).collect();
        let vecs_delta = vec![vec![10.0, 10.0]; 50];
        create_bucket_file(dir.path(), "delta.drift", &ids_delta, &vecs_delta, dim).await;

        // 3. Register Tiered
        bucket_mgr.register_bucket_with_count(
            1,
            "ignored".into(),
            StorageClass::Tiered {
                remote_path: "base.drift".into(),
                local_path: "delta.drift".into(),
            },
            100,
        );

        // 4. Calculate Split
        // Should fetch BOTH files, merge them (100 items), and split into 2 clusters [0,0] and [10,10]
        let proposal = index
            .calculate_split(1)
            .await
            .unwrap()
            .expect("Split failed");

        // 5. Verify
        let total = proposal.left.count + proposal.right.count;
        assert_eq!(total, 100, "Split should include data from both tiers");

        // Verify separation (K-Means should separate 0.0 from 10.0 perfectly)
        // One bucket should have IDs < 50, the other > 50
        let left_is_base = proposal.left.ids.iter().all(|&id| id < 50);
        let right_is_delta = proposal.right.ids.iter().all(|&id| id >= 50);

        // Since K-Means labels are arbitrary (0 or 1), we check if they are cleanly separated
        assert!(
            (left_is_base && right_is_delta)
                || (proposal.left.ids.iter().all(|&id| id >= 50)
                    && proposal.right.ids.iter().all(|&id| id < 50)),
            "K-Means failed to separate the two distinct clusters from Base/Delta"
        );
    }
}

#[cfg(test)]
mod janitor_scatter_merge_test {
    use crate::janitor::{Janitor, JanitorConfig, JanitorVars};
    use crate::local_staging::LocalStagingManager;
    use crate::manifest::ServerManifestManager;
    use crate::persistence::PersistenceManager;
    use drift_core::index::VectorIndex;
    use drift_core::lock_manager::BucketCoordinator;
    use drift_core::math::Metric;
    use drift_core::quantizer::Quantizer;
    use drift_core::router::Router;
    use drift_core::wal::WalManager;
    use drift_kv::bitstore::BitStore;
    use drift_storage::bucket_file_writer::BucketFileWriter;
    use drift_storage::bucket_manager::{BucketManager, StorageClass};
    use drift_traits::StorageEngine;
    use opendal::{Operator, services};
    use parking_lot::{Mutex, RwLock};
    use std::sync::Arc;
    use std::time::Duration;
    use tempfile::tempdir;

    fn create_fs_operator(path: &std::path::Path) -> Operator {
        let builder = services::Fs::default().root(path.to_str().unwrap());
        Operator::new(builder).unwrap().finish()
    }

    async fn create_bucket_file(
        dir: &std::path::Path,
        filename: &str,
        ids: &[u64],
        vecs: &[Vec<f32>],
        dim: usize,
    ) {
        let path = dir.join(filename);
        let file = std::fs::File::create(&path).unwrap();
        // Flatten vectors
        let flat: Vec<f32> = vecs.iter().flatten().copied().collect();
        // Handle empty case for quantization
        let q = if !flat.is_empty() {
            Quantizer::train(&flat, dim)
        } else {
            Quantizer::train(&vec![0.0; dim], dim)
        };

        let mut writer = BucketFileWriter::new_streaming(file, [0u8; 16], q, dim).unwrap();
        if !ids.is_empty() {
            writer.write_batch(ids, &flat).unwrap();
        }
        writer.finalize().unwrap();
    }

    async fn setup_env(
        dir: &std::path::Path,
        dim: usize,
        max_bucket_cap: usize,
    ) -> (
        Arc<VectorIndex>,
        Arc<BucketManager>,
        Janitor,
        Arc<ServerManifestManager>,
        Arc<LocalStagingManager>,
    ) {
        let data_dir = dir.join("data");
        std::fs::create_dir_all(&data_dir).unwrap();

        let manifest = Arc::new(ServerManifestManager::new(dir, dim as u32).unwrap());
        let staging = Arc::new(LocalStagingManager::new(&data_dir).unwrap());
        let op = create_fs_operator(&data_dir);
        let persistence = Arc::new(PersistenceManager::new(op.clone()));
        let coordinator = Arc::new(BucketCoordinator::new());
        let bucket_manager = Arc::new(BucketManager::new(
            op.clone(),
            op.clone(),
            4,
            coordinator.clone(),
            Metric::L2,
        ));

        let wal_dir = dir.join("wal");
        let wal_mgr = Arc::new(Mutex::new(WalManager::new(&wal_dir).unwrap()));
        let kv = Arc::new(BitStore::new(dir.join("kv")).unwrap());

        let centroids = vec![];
        let router = Arc::new(RwLock::new(
            Router::new(&centroids, &[], dim, Metric::L2).unwrap_or(Router::empty(dim, Metric::L2)),
        ));

        let index = Arc::new(VectorIndex::new(
            dim,
            max_bucket_cap,
            router,
            wal_mgr,
            bucket_manager.clone(),
            kv,
        ));

        let janitor = Janitor::new(JanitorConfig {
            index: index.clone(),
            manifest: manifest.clone(),
            staging: staging.clone(),
            persistence,
            bucket_manager: bucket_manager.clone(),
            coordinator,
            vars: JanitorVars {
                check_interval: Duration::from_millis(10),
                promotion_threshold_bytes: 1024 * 1024,
                max_bucket_capacity: max_bucket_cap,
                split_threshold: 0.8,
                drift_threshold: 0.15,
                ..Default::default()
            },
        });

        (index, bucket_manager, janitor, manifest, staging)
    }

    #[tokio::test]
    async fn test_scatter_merge_e2e_small_bucket() {
        let dir = tempdir().unwrap();
        let dim = 2;
        let max_cap = 100;
        let (index, bucket_mgr, janitor, manifest, staging) =
            setup_env(dir.path(), dim, max_cap).await;

        let c1 = vec![0.0; dim];
        let c2 = vec![10.0; dim];

        {
            let mut r = index.get_router().write();
            r.add_bucket(1, c1.clone());
            r.add_bucket(2, c2.clone());
            r.update_bucket(1, 5, c1.clone());
            r.update_bucket(2, 50, c2.clone());
        }

        // Bucket 1: 5 items
        let b1_ids: Vec<u64> = (0..5).collect();
        let b1_vecs = vec![c1.clone(); 5];
        create_bucket_file(
            staging.get_base_path(),
            "bucket_1.drift",
            &b1_ids,
            &b1_vecs,
            dim,
        )
        .await;

        // Bucket 2: 50 items
        let b2_ids: Vec<u64> = (100..150).collect();
        let b2_vecs = vec![c2.clone(); 50];
        create_bucket_file(
            staging.get_base_path(),
            "bucket_2.drift",
            &b2_ids,
            &b2_vecs,
            dim,
        )
        .await;

        staging.set_active_filename(1, "bucket_1.drift".into());
        staging.set_active_filename(2, "bucket_2.drift".into());

        bucket_mgr.register_bucket(1, "bucket_1.drift".into(), StorageClass::Local);
        bucket_mgr.register_bucket(2, "bucket_2.drift".into(), StorageClass::Local);

        bucket_mgr
            .update_bucket_drift(1, &vec![0.0; dim], 5)
            .unwrap();
        bucket_mgr
            .update_bucket_drift(2, &vec![500.0; dim], 50)
            .unwrap();

        manifest
            .apply_atomic(|m| {
                m.add_bucket(1, "".into(), Some(c1));
                m.update_bucket_stats(1, 5, 0);
                m.add_bucket(2, "".into(), Some(c2));
                m.update_bucket_stats(2, 50, 0);
            })
            .unwrap();

        let handle = tokio::spawn(async move { janitor.run().await });

        // Wait for Merge
        let mut merged = false;
        for _ in 0..50 {
            tokio::time::sleep(Duration::from_millis(100)).await;
            let state = manifest.get_state();
            if !state.get_buckets().iter().any(|b| b.id == 1) {
                merged = true;
                break;
            }
        }
        handle.abort();

        assert!(merged, "Janitor failed to merge zombie bucket 1!");

        // Verify Bucket 2 Grew
        let state = manifest.get_state();
        let b2 = state.get_buckets().iter().find(|b| b.id == 2).unwrap();
        assert_eq!(b2.vector_count, 55, "Neighbor bucket count incorrect");
    }

    #[tokio::test]
    async fn test_scatter_merge_budget_constraint() {
        let dir = tempdir().unwrap();
        let dim = 2;
        let max_cap = 100;
        let (index, bucket_mgr, janitor, manifest, staging) =
            setup_env(dir.path(), dim, max_cap).await;

        let c1 = vec![0.0; dim];
        let c2 = vec![10.0; dim];

        {
            let mut r = index.get_router().write();
            r.add_bucket(1, c1.clone());
            r.add_bucket(2, c2.clone());
        }

        // ⚡ FIX: Use 110 items to exceed the hardcoded 100 limit in drift_core
        let large_count = 110;
        let b1_vecs = vec![c1.clone(); large_count];
        create_bucket_file(
            staging.get_base_path(),
            "bucket_1.drift",
            &vec![0; large_count],
            &b1_vecs,
            dim,
        )
        .await;

        create_bucket_file(
            staging.get_base_path(),
            "bucket_2.drift",
            &vec![0; 10],
            &vec![c2.clone(); 10],
            dim,
        )
        .await;

        staging.set_active_filename(1, "bucket_1.drift".into());
        bucket_mgr.register_bucket(1, "bucket_1.drift".into(), StorageClass::Local);
        bucket_mgr
            .update_bucket_drift(1, &vec![0.0; dim], large_count as u32)
            .unwrap();

        // Force High Urgency to simulate "Hot Zombie" state
        manifest
            .apply_atomic(|m| {
                m.add_bucket(1, "".into(), Some(c1));
                m.update_bucket_stats(1, large_count as u64, 0);
            })
            .unwrap();

        let handle = tokio::spawn(async move { janitor.run().await });
        tokio::time::sleep(Duration::from_millis(500)).await;
        handle.abort();

        // 4. Verify Bucket 1 Still Exists (Budget Protected)
        let state = manifest.get_state();
        assert!(
            state.get_buckets().iter().any(|b| b.id == 1),
            "Bucket 1 should NOT be merged (over budget)"
        );
    }

    #[tokio::test]
    async fn test_scatter_merge_no_neighbors_empty() {
        let dir = tempdir().unwrap();
        let dim = 2;
        let (index, bucket_mgr, janitor, manifest, staging) = setup_env(dir.path(), dim, 100).await;

        let c1 = vec![0.0; dim];
        {
            let mut r = index.get_router().write();
            r.add_bucket(1, c1.clone());
        }

        // If it has data but no neighbors, index returns SkippedTooSmall to prevent data loss.
        // If it is EMPTY, index returns Empty Proposal, allowing deletion.
        create_bucket_file(staging.get_base_path(), "bucket_1.drift", &[], &[], dim).await;

        staging.set_active_filename(1, "bucket_1.drift".into());
        bucket_mgr.register_bucket(1, "bucket_1.drift".into(), StorageClass::Local);
        // Zero drift stats
        bucket_mgr.update_bucket_drift(1, &c1, 0).unwrap();

        manifest
            .apply_atomic(|m| {
                m.add_bucket(1, "".into(), Some(c1));
                m.update_bucket_stats(1, 0, 0); // 0 items -> Emptiness = 1.0 -> Urgency high
            })
            .unwrap();

        let handle = tokio::spawn(async move { janitor.run().await });
        tokio::time::sleep(Duration::from_millis(500)).await;
        handle.abort();

        // 3. Verify Bucket 1 is Deleted
        let state = manifest.get_state();
        assert!(
            !state.get_buckets().iter().any(|b| b.id == 1),
            "Empty zombie bucket should be deleted"
        );

        let path = staging.get_base_path().join("bucket_1.drift");
        assert!(!path.exists(), "File should be deleted");
    }
}

#[cfg(test)]
mod janitor_promotion_test {
    use crate::janitor::{Janitor, JanitorConfig, JanitorVars};
    use crate::local_staging::LocalStagingManager;
    use crate::manifest::ServerManifestManager;
    use crate::persistence::PersistenceManager;
    use drift_core::index::VectorIndex;
    use drift_core::lock_manager::BucketCoordinator;
    use drift_core::math::Metric;
    use drift_core::partitioner::PartitionGroup;
    use drift_core::quantizer::Quantizer;
    use drift_core::router::Router;
    use drift_core::wal::WalManager;
    use drift_kv::bitstore::BitStore;
    use drift_storage::bucket_file_writer::BucketFileWriter;
    use drift_storage::bucket_manager::{BucketManager, StorageClass};
    use drift_traits::StorageEngine;
    use opendal::{Operator, services};
    use parking_lot::{Mutex, RwLock};
    use std::sync::Arc;
    use std::time::Duration;
    use tempfile::tempdir;

    fn create_fs_operator(path: &std::path::Path) -> Operator {
        let builder = services::Fs::default().root(path.to_str().unwrap());
        Operator::new(builder).unwrap().finish()
    }

    /// Helper to write a valid .drift segment file manually (simulating an old S3 flush)
    async fn create_s3_segment(
        dir: &std::path::Path,
        filename: &str,
        ids: &[u64],
        vecs: &[Vec<f32>],
        dim: usize,
    ) {
        let path = dir.join(filename);
        let file = std::fs::File::create(&path).unwrap();

        let flat: Vec<f32> = vecs.iter().flatten().copied().collect();
        let q = Quantizer::train(&flat, dim);

        // We use new_streaming to create a proper segment with header/footer
        let mut writer = BucketFileWriter::new_streaming(file, [1u8; 16], q, dim).unwrap();
        writer.write_batch(ids, &flat).unwrap();
        writer.finalize().unwrap();
    }

    #[tokio::test]
    async fn test_janitor_promotion_merges_tiers_and_purges_deletes() {
        let dir = tempdir().unwrap();
        let data_dir = dir.path().join("data");
        std::fs::create_dir(&data_dir).unwrap();
        let dim = 2;
        let bucket_id = 1;

        // --- SETUP COMPONENTS ---
        let manifest = Arc::new(ServerManifestManager::new(dir.path(), dim as u32).unwrap());
        let staging = Arc::new(LocalStagingManager::new(&data_dir).unwrap());
        let op = create_fs_operator(&data_dir); // Shared Operator for S3 simulation
        let persistence = Arc::new(PersistenceManager::new(op.clone()));
        let coordinator = Arc::new(BucketCoordinator::new());
        let bucket_manager = Arc::new(BucketManager::new(
            op.clone(), // Remote
            op.clone(), // Local (reused for test simplicity, usually staging)
            4,
            coordinator.clone(),
            Metric::L2,
        ));

        let wal = Arc::new(Mutex::new(WalManager::new(dir.path().join("wal")).unwrap()));
        let kv = Arc::new(BitStore::new(dir.path().join("kv")).unwrap());
        let router = Arc::new(RwLock::new(Router::empty(dim, Metric::L2)));
        let index = Arc::new(VectorIndex::new(
            dim,
            100,
            router,
            wal,
            bucket_manager.clone(),
            kv,
        ));

        // --- PREPARE DATA ---

        // 1. Remote Base (Old S3 Segment) - IDs 0..10
        // We simulate a previous run "run_OLD"
        let remote_ids: Vec<u64> = (0..10).collect();
        let remote_vecs = vec![vec![1.0, 1.0]; 10];
        create_s3_segment(
            &data_dir,
            "bucket_1_run_OLD.drift",
            &remote_ids,
            &remote_vecs,
            dim,
        )
        .await;

        // 2. Local Staging (New Delta) - IDs 10..20
        let local_ids: Vec<u64> = (10..20).collect();
        let mut group = PartitionGroup::new(dim, None);
        group.ids = local_ids.clone();
        for _ in 0..10 {
            group.flat_vectors.extend_from_slice(&[2.0, 2.0]);
        }
        group.count = 10;

        // Write to staging and register as Active
        staging.append_batch(bucket_id, &group).await.unwrap();

        // Register the Tiered State (simulating startup or previous operations)
        // Note: Janitor expects a file in staging to exist for promotion logic
        let staging_file = staging.get_active_filename(bucket_id);

        bucket_manager.register_bucket(
            bucket_id,
            "bucket_1_run_OLD.drift".to_string(), // Currently pointing to remote
            StorageClass::Tiered {
                remote_path: "bucket_1_run_OLD.drift".to_string(),
                local_path: staging_file.clone(), // But also has local data
            },
        );

        // [FIX START]: Register the bucket in the manifest so the Janitor knows to check it
        manifest
            .apply_atomic(|m| {
                // 1. Create the bucket entry
                m.add_bucket(
                    bucket_id,
                    "run_OLD".to_string(),
                    Some(vec![0.0; dim]), // Dummy centroid
                );
                // 2. Update its statistics
                m.update_bucket_stats(bucket_id, 10, 0); // Initial count before merge
            })
            .unwrap();
        // [FIX END]

        // 3. Mark Deletes (One from Remote, One from Local)
        // ID 5 (Remote) and ID 15 (Local)
        bucket_manager.mark_delete(bucket_id, 5).unwrap();
        bucket_manager.mark_delete(bucket_id, 15).unwrap();

        assert_eq!(
            bucket_manager.get_tombstones(bucket_id).len(),
            2,
            "Tombstones should be registered"
        );

        // --- EXECUTE PROMOTION ---

        let janitor = Janitor::new(JanitorConfig {
            index: index.clone(),
            manifest: manifest.clone(),
            staging: staging.clone(),
            persistence: persistence.clone(),
            bucket_manager: bucket_manager.clone(),
            coordinator: coordinator.clone(),
            vars: JanitorVars {
                promotion_threshold_bytes: 1, // Force promotion (file > 1 byte)
                check_interval: Duration::from_millis(100),
                ..Default::default()
            },
        });

        // Trigger explicitly
        janitor.promote_segments().await.expect("Promotion failed");

        // --- VERIFY RESULTS ---

        // 1. Check Manifest
        let state = manifest.get_state();
        let b1 = state
            .get_buckets()
            .iter()
            .find(|b| b.id == bucket_id)
            .unwrap(); // This should now succeed

        // Count should be 18 (20 total - 2 deleted)
        assert_eq!(
            b1.vector_count, 18,
            "Manifest count should match merged & purged data"
        );
        assert_ne!(b1.run_id, "run_OLD", "Run ID should update");
        assert!(
            !b1.object_path.is_empty(),
            "Object path should be recorded in manifest"
        );
        assert!(
            !b1.object_fingerprint.is_empty(),
            "Object fingerprint should be recorded in manifest"
        );
        assert_eq!(
            b1.object_path,
            format!("bucket_{}_{}.drift", bucket_id, b1.run_id)
        );

        // 2. Check S3 File Content
        let new_key = format!("bucket_1_{}.drift", b1.run_id);
        assert!(
            op.exists(&new_key).await.unwrap(),
            "New S3 segment should exist"
        );

        let (stored_ids, _) = persistence
            .read_remote_bucket(bucket_id, &b1.run_id)
            .await
            .unwrap();
        assert_eq!(stored_ids.len(), 18);
        assert!(!stored_ids.contains(&5), "ID 5 should be purged");
        assert!(!stored_ids.contains(&15), "ID 15 should be purged");
        assert!(stored_ids.contains(&0), "ID 0 should remain");
        assert!(stored_ids.contains(&19), "ID 19 should remain");

        // 3. Check Memory Reconcile (Pruning)
        // The bucket's specific tombstones should now be empty (since we flushed them)
        let remaining_tombstones = bucket_manager.get_tombstones(bucket_id);
        assert!(
            remaining_tombstones.is_empty(),
            "Memory tombstones should be pruned after flush"
        );

        // 4. Check Cleanup
        // Staging file should be gone
        let staging_path = staging.get_base_path().join(staging_file);
        assert!(!staging_path.exists(), "Staging file should be deleted");

        // Old S3 file should be gone
        let old_path = data_dir.join("bucket_1_run_OLD.drift");
        assert!(!old_path.exists(), "Old S3 segment should be deleted");
    }
}
