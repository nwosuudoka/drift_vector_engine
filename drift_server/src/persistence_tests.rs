#[cfg(test)]
mod tests {
    use crate::janitor::{Janitor, JanitorConfig, JanitorVars};
    use crate::local_staging::LocalStagingManager;
    use crate::manifest::ServerManifestManager;
    use crate::persistence::PersistenceManager;
    use crate::recovery::RecoveryManager;
    use drift_core::index::VectorIndex;
    use drift_core::lock_manager::BucketCoordinator;
    use drift_core::manifest::pb::Centroid;
    use drift_core::math::Metric;
    use drift_core::router::Router;
    use drift_core::wal::WalManager;
    use drift_kv::bitstore::BitStore;
    use drift_storage::bucket_manager::BucketManager;
    use opendal::Operator;
    use opendal::services::Fs;
    use parking_lot::{Mutex, RwLock};
    use std::path::Path;
    use std::sync::Arc;
    use std::time::Duration;
    use tempfile::tempdir;

    fn create_local_operator(path: &Path) -> std::io::Result<Operator> {
        let builder = Fs::default().root(path.to_str().unwrap());
        Ok(Operator::new(builder)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?
            .finish())
    }

    // --- SETUP HELPER ---
    async fn setup_env(
        path: &std::path::Path,
        capacity: usize,
        dim: usize,
    ) -> (
        Arc<VectorIndex>,
        Arc<BucketManager>,
        Janitor,
        Arc<ServerManifestManager>,
        Arc<PersistenceManager>,
    ) {
        let data_dir = path.join("data");
        let wal_dir = path.join("wal");
        let kv_dir = data_dir.join("kv");
        let staging_dir = data_dir.join("staging");

        std::fs::create_dir_all(&data_dir).unwrap();
        std::fs::create_dir_all(&wal_dir).unwrap();
        std::fs::create_dir_all(&kv_dir).unwrap();
        std::fs::create_dir_all(&staging_dir).unwrap();

        let op = create_local_operator(&staging_dir).unwrap();
        let persistence = Arc::new(PersistenceManager::new(op.clone()));
        let staging = Arc::new(LocalStagingManager::new(&staging_dir).unwrap());
        let coordinator = Arc::new(BucketCoordinator::new());
        let bucket_manager = Arc::new(BucketManager::new(
            op.clone(),
            op.clone(),
            4,
            coordinator.clone(),
            Metric::L2,
        ));

        let manifest = Arc::new(ServerManifestManager::new(&data_dir, dim as u32).unwrap());

        let centroids = vec![Centroid {
            id: 1,
            vector: vec![0.0; dim],
        }];
        let router = Arc::new(RwLock::new(
            Router::new(&centroids, &[0], dim, Metric::L2).unwrap(),
        ));

        let wal = Arc::new(Mutex::new(WalManager::new(&wal_dir).unwrap()));
        let kv = Arc::new(BitStore::new(&kv_dir).unwrap());

        let index = Arc::new(VectorIndex::new(
            dim,
            capacity,
            router,
            wal,
            bucket_manager.clone(),
            kv,
        ));

        let janitor = Janitor::new(JanitorConfig {
            index: index.clone(),
            manifest: manifest.clone(),
            staging,
            persistence: persistence.clone(),
            bucket_manager: bucket_manager.clone(),
            coordinator,
            vars: JanitorVars {
                check_interval: Duration::from_millis(100),
                promotion_threshold_bytes: 1024,
                max_bucket_capacity: 2000,
                split_threshold: 0.8,
                drift_threshold: 0.15,
                ..Default::default()
            },
        });

        (index, bucket_manager, janitor, manifest, persistence)
    }

    // --- TEST 1: WAL Recovery (Crash before Flush) ---
    #[tokio::test]
    async fn test_wal_delete_recovery_logic() {
        let dir = tempdir().unwrap();
        let root = dir.path();

        // 1. Setup Components
        let (index, bucket_manager, _, manifest, _) = setup_env(root, 100, 2).await;

        // 2. Insert & Delete
        index.insert(100, &[1.0, 1.0]).unwrap();
        index.delete(100).unwrap(); // Writes Delete to WAL

        // Force WAL flush by writing another entry
        index.delete(101).unwrap();

        // 3. Drop to flush/close
        drop(index);

        // 4. Recover
        let wal_path = root.join("wal");

        let recovery_mgr = RecoveryManager::new(&root.join("data"), manifest.clone());

        // ⚡ NEW API: Pass WAL Path directly
        let (_, replay_data) = recovery_mgr
            .recover(&bucket_manager, 2, &wal_path)
            .await
            .unwrap();

        assert!(
            replay_data.deletes.contains(&100),
            "WAL Scan missed the delete of ID 100"
        );
        assert!(replay_data.deletes.contains(&101));
    }

    // --- TEST 2: S3 Tombstone Hydration (Crash after Flush) ---
    #[tokio::test]
    async fn test_s3_tombstone_hydration_lifecycle() {
        let dir = tempdir().unwrap();
        let (index, _, janitor, _, persistence) = setup_env(dir.path(), 100, 2).await;

        // 1. Insert & Delete
        index.insert(200, &[0.0, 0.0]).unwrap();
        index.delete(200).unwrap();

        // 2. FORCE FLUSH
        persistence
            .flush_tombstones(&[200], "run_simulation")
            .await
            .unwrap();

        // 3. Restart
        drop(index);
        drop(janitor);

        // 4. Hydrate
        let recovered_deletes = persistence.load_all_tombstones().await.unwrap();

        assert!(
            recovered_deletes.contains(&200),
            "Failed to hydrate deletions from disk (S3)"
        );
    }
}

#[cfg(test)]
mod persistence_integration_tests {
    use crate::persistence::PersistenceManager;
    use drift_storage::unified_format::{
        UNIFIED_FLAG_HAS_EXACT_INDEX, UNIFIED_FLAG_HAS_PAYLOAD_COLUMNS,
        UNIFIED_FLAG_HAS_PAYLOAD_STATS, UnifiedFieldSchema, UnifiedLogicalType, UnifiedPayloadRow,
        UnifiedPayloadSchema, UnifiedPayloadValue,
    };
    use drift_storage::unified_reader::UnifiedReader;
    use opendal::{Operator, services};
    use tempfile::tempdir;

    // --- Helpers ---

    fn create_local_operator(path: &std::path::Path) -> Operator {
        let builder = services::Fs::default().root(path.to_str().unwrap());
        Operator::new(builder).unwrap().finish()
    }

    fn mock_data(start_id: u64, count: usize, dim: usize, val: f32) -> (Vec<u64>, Vec<Vec<f32>>) {
        let ids: Vec<u64> = (0..count as u64).map(|i| start_id + i).collect();
        let vecs: Vec<Vec<f32>> = (0..count).map(|_| vec![val; dim]).collect();
        (ids, vecs)
    }

    #[tokio::test]
    async fn test_persistence_promotion_and_merge_flow() {
        // Setup
        let dir = tempdir().unwrap();
        let op = create_local_operator(dir.path());
        let persistence = PersistenceManager::new(op.clone());
        let bucket_id = 1;
        let dim = 4;

        // ==========================================================
        // PHASE 1: Initial Promotion (Local Staging -> S3)
        // ==========================================================
        println!("Phase 1: Initial Promotion...");
        let (ids_1, vecs_1) = mock_data(0, 10, dim, 1.0); // IDs 0-9, Value 1.0

        // ⚡ CHANGE: Use atomic write_remote_bucket
        let (run_id_1, count_1) = persistence
            .write_remote_bucket(bucket_id, &ids_1, &vecs_1, dim)
            .await
            .expect("Initial promotion failed");

        assert_eq!(count_1, 10);
        let key_1 = format!("bucket_{}_{}.driftu", bucket_id, run_id_1);

        // Verify file exists
        assert!(op.exists(&key_1).await.unwrap());

        // ==========================================================
        // PHASE 2: Merge Promotion (New Local + Old S3 -> New S3)
        // ==========================================================
        println!("Phase 2: Merge Promotion...");
        let (ids_2, vecs_2) = mock_data(10, 10, dim, 2.0); // IDs 10-19, Value 2.0

        // ⚡ CHANGE: Explicitly Fetch + Merge + Write

        // A. Read Remote (Base)
        let (remote_ids, remote_vecs) = persistence
            .read_remote_bucket(bucket_id, &run_id_1)
            .await
            .expect("Failed to read remote bucket");

        // B. Merge in Memory (Local Delta + Remote Base)
        // Note: We simulate the Janitor's append logic (Local first, then Remote)
        let mut merged_ids = ids_2.clone();
        let mut merged_vecs = vecs_2.clone();
        merged_ids.extend(remote_ids);
        merged_vecs.extend(remote_vecs);

        // C. Write New Segment
        let (run_id_2, count_2) = persistence
            .write_remote_bucket(bucket_id, &merged_ids, &merged_vecs, dim)
            .await
            .expect("Merge promotion failed");

        assert_eq!(count_2, 20, "Should contain 10 old + 10 new items");
        let key_2 = format!("bucket_{}_{}.driftu", bucket_id, run_id_2);

        // ==========================================================
        // PHASE 3: Verification (Read Back)
        // ==========================================================
        println!("Phase 3: Verify Merged Data...");

        let mut reader = UnifiedReader::open(op.clone(), &key_2)
            .await
            .expect("Failed to open merged file");

        let (read_ids, read_vecs) = reader
            .read_all_vectors()
            .await
            .expect("Failed to read vectors");

        // Check Counts
        assert_eq!(read_ids.len(), 20);
        assert_eq!(read_vecs.len(), 20);

        // Check Content
        // We merged [10..19] (Local) then [0..9] (Remote).

        // Verify ID 10 (Local) is present and has value 2.0
        let idx_10 = read_ids.iter().position(|&x| x == 10).unwrap();
        assert_eq!(read_vecs[idx_10][0], 2.0);

        // Verify ID 0 (Remote) is present and has value 1.0
        let idx_0 = read_ids.iter().position(|&x| x == 0).unwrap();
        assert_eq!(read_vecs[idx_0][0], 1.0);

        println!("✅ Persistence Integration Test Passed!");
    }

    #[tokio::test]
    async fn test_persistence_writes_and_reads_payload_schema() {
        let dir = tempdir().unwrap();
        let op = create_local_operator(dir.path());
        let persistence = PersistenceManager::new(op.clone());
        let bucket_id = 7;
        let dim = 3;

        let ids = vec![10, 11];
        let flat = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let schema = UnifiedPayloadSchema::new(vec![
            UnifiedFieldSchema {
                field_id: 1,
                name: "tenant".to_string(),
                logical_type: UnifiedLogicalType::Keyword,
                nullable: false,
                indexed: true,
            },
            UnifiedFieldSchema {
                field_id: 2,
                name: "ts".to_string(),
                logical_type: UnifiedLogicalType::TimestampMicros,
                nullable: false,
                indexed: true,
            },
        ]);
        let rows: Vec<UnifiedPayloadRow> = vec![
            std::collections::BTreeMap::from([
                (1, UnifiedPayloadValue::Keyword("tenant_a".to_string())),
                (2, UnifiedPayloadValue::TimestampMicros(1_700_000)),
            ]),
            std::collections::BTreeMap::from([
                (1, UnifiedPayloadValue::Keyword("tenant_b".to_string())),
                (2, UnifiedPayloadValue::TimestampMicros(1_700_100)),
            ]),
        ];

        let (run_id, count) = persistence
            .write_remote_bucket_unified_flat_with_payload(
                bucket_id,
                &ids,
                &flat,
                dim,
                Some(&schema),
                Some(&rows),
            )
            .await
            .unwrap();
        assert_eq!(count, ids.len() as u64);

        let key = format!("bucket_{}_{}.driftu", bucket_id, run_id);
        let reader = UnifiedReader::open(op.clone(), &key).await.unwrap();
        let decoded = reader.read_payload_schema().await.unwrap();
        assert_eq!(decoded, Some(schema.clone()));

        let via_persistence = persistence
            .read_remote_bucket_payload_schema(bucket_id, &run_id)
            .await
            .unwrap();
        assert_eq!(via_persistence, Some(schema));

        let payload_rows = persistence
            .read_remote_bucket_payload_rows(bucket_id, &run_id)
            .await
            .unwrap();
        assert_eq!(payload_rows, rows);
    }

    #[tokio::test]
    async fn test_persistence_write_result_exposes_payload_index_meta() {
        let dir = tempdir().unwrap();
        let op = create_local_operator(dir.path());
        let persistence = PersistenceManager::new(op.clone());
        let bucket_id = 9;
        let dim = 2;

        let ids = vec![101, 102];
        let flat = vec![0.1, 0.2, 0.3, 0.4];
        let schema = UnifiedPayloadSchema::new(vec![UnifiedFieldSchema {
            field_id: 1,
            name: "tenant".to_string(),
            logical_type: UnifiedLogicalType::Keyword,
            nullable: false,
            indexed: true,
        }]);
        let rows: Vec<UnifiedPayloadRow> = vec![
            std::collections::BTreeMap::from([(1, UnifiedPayloadValue::Keyword("a".to_string()))]),
            std::collections::BTreeMap::from([(1, UnifiedPayloadValue::Keyword("b".to_string()))]),
        ];

        let result = persistence
            .write_remote_bucket_unified_flat_with_payload_result(
                bucket_id,
                &ids,
                &flat,
                dim,
                Some(&schema),
                Some(&rows),
            )
            .await
            .unwrap();

        assert_eq!(result.row_count, ids.len() as u64);
        assert!(result.payload_index_meta.has_payload_columns);
        assert!(result.payload_index_meta.has_exact_index);
        assert!(result.payload_index_meta.has_payload_stats);
        assert!(result.payload_index_meta.payload_schema_hash != 0);

        let reader = UnifiedReader::open(op.clone(), &result.object_path)
            .await
            .unwrap();
        assert!(
            (reader.header.flags & UNIFIED_FLAG_HAS_PAYLOAD_COLUMNS) != 0,
            "header should advertise payload columns"
        );
        assert!(
            (reader.header.flags & UNIFIED_FLAG_HAS_EXACT_INDEX) != 0,
            "header should advertise exact index"
        );
        assert!(
            (reader.header.flags & UNIFIED_FLAG_HAS_PAYLOAD_STATS) != 0,
            "header should advertise payload stats"
        );
        assert_eq!(
            result.payload_index_meta.payload_schema_hash,
            reader.header.payload_schema_hash
        );
    }
}

#[cfg(test)]
mod cleanup_invalidation_tests {
    use crate::cleanup::CleanupApi;
    use crate::local_staging::LocalStagingManager;
    use crate::persistence::PersistenceManager;
    use drift_storage::disk_manager::DiskManager;
    use opendal::{Operator, services};
    use std::sync::{Arc, OnceLock};
    use tempfile::tempdir;
    use walkdir::WalkDir;

    static ENV_LOCK: OnceLock<std::sync::Mutex<()>> = OnceLock::new();

    struct EnvVarGuard {
        key: &'static str,
        prev: Option<String>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let prev = std::env::var(key).ok();
            // SAFETY: This test serializes access via ENV_LOCK so environment mutation
            // does not race with other env-mutating tests in this crate.
            unsafe { std::env::set_var(key, value) };
            Self { key, prev }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            if let Some(prev) = &self.prev {
                // SAFETY: Guarded by ENV_LOCK in this test module.
                unsafe { std::env::set_var(self.key, prev) };
            } else {
                // SAFETY: Guarded by ENV_LOCK in this test module.
                unsafe { std::env::remove_var(self.key) };
            }
        }
    }

    fn memory_op() -> Operator {
        Operator::new(services::Memory::default()).unwrap().finish()
    }

    fn count_cache_object_files(root: &std::path::Path) -> usize {
        WalkDir::new(root)
            .into_iter()
            .flatten()
            .filter(|e| e.file_type().is_file() && e.file_name() == "object.cache")
            .count()
    }

    #[tokio::test]
    async fn test_cleanup_api_delete_remote_invalidates_nvme_cache() {
        let _env_guard = ENV_LOCK
            .get_or_init(|| std::sync::Mutex::new(()))
            .lock()
            .unwrap();

        let cache_root = tempdir().unwrap();
        let _cache_dir =
            EnvVarGuard::set("DRIFT_NVME_CACHE_DIR", cache_root.path().to_str().unwrap());

        let op = memory_op();
        let path = "cleanup/test_object.bin";
        let payload = vec![9u8; 64];
        op.write(path, payload.clone()).await.unwrap();

        let mgr = DiskManager::new(op.clone(), path.to_string());
        let first = mgr.read_at(0, 16).await.unwrap();
        assert_eq!(first, payload[..16].to_vec());
        assert_eq!(count_cache_object_files(cache_root.path()), 1);

        let staging_root = tempdir().unwrap();
        let staging = Arc::new(LocalStagingManager::new(staging_root.path()).unwrap());
        let persistence = Arc::new(PersistenceManager::new(op.clone()));
        let cleanup = CleanupApi::new(staging, persistence);

        cleanup.delete_remote(path).await.unwrap();

        assert_eq!(count_cache_object_files(cache_root.path()), 0);
        let err = mgr
            .read_at(0, 16)
            .await
            .expect_err("cache should be invalidated and remote object deleted");
        assert!(
            err.kind() == std::io::ErrorKind::NotFound || err.kind() == std::io::ErrorKind::Other
        );
    }
}
