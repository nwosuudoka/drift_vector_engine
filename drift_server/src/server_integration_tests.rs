#[cfg(test)]
mod tests {
    use crate::config::{Config, FileConfig, StorageCommand};
    use crate::drift_proto::{
        CreateCollectionRequest, FieldFilter, HealthRequest, InsertBatchRequest, MetricType,
        PayloadRow, PayloadValue, PayloadValueList, RangeFilter, SearchRequest, Vector,
        drift_server::Drift,
    };
    use crate::local_staging::LocalStagingManager;
    use crate::manager::CollectionManager;
    use crate::manifest::ServerManifestManager;
    use crate::persistence::PersistenceManager;
    use crate::recovery::RecoveryManager;
    use crate::server::DriftService;
    use drift_core::lock_manager::BucketCoordinator;
    use drift_core::math::Metric;
    use drift_core::partitioner::PartitionGroup;
    use drift_core::wal::WalWriter;
    use drift_storage::bucket_manager::{BucketManager, StorageClass};
    use drift_storage::unified_writer::UnifiedLocalWriter;
    use drift_traits::StorageEngine;
    use opendal::{Operator, services};
    use std::collections::HashMap;
    use std::sync::Arc;
    use tempfile::tempdir;
    use tonic::Request;

    fn create_fs_operator(path: &std::path::Path) -> Operator {
        let builder = services::Fs::default().root(path.to_str().unwrap());
        Operator::new(builder).unwrap().finish()
    }

    fn mock_batch(start_id: u64, count: usize, dim: usize, val_offset: f32) -> PartitionGroup {
        let mut ids = Vec::new();
        let mut vecs = Vec::new();
        for i in 0..count {
            let id = start_id + i as u64;
            ids.push(id);
            vecs.extend(vec![id as f32 + val_offset; dim]);
        }
        PartitionGroup {
            ids,
            flat_vectors: vecs,
            count,
            centroid: Some(vec![0.0; dim]),
            payload_schema: None,
            payload_rows: None,
        }
    }

    #[tokio::test]
    async fn test_health_endpoint_reports_ready_and_version() {
        let dir = tempdir().unwrap();
        let config = Config {
            port: 50051,
            wal_dir: dir.path().join("wal"),
            data_dir: dir.path().join("data"),
            default_dim: 8,
            max_bucket_capacity: 100,
            ef_construction: 32,
            ef_search: 32,
            storage: StorageCommand::File(FileConfig {
                path: dir.path().join("storage"),
            }),
        };

        let service = DriftService {
            manager: Arc::new(CollectionManager::new(config)),
        };

        let health = service
            .health(Request::new(HealthRequest {}))
            .await
            .expect("health call should succeed")
            .into_inner();

        assert!(health.ready, "health endpoint should report ready=true");
        assert!(
            !health.version.trim().is_empty(),
            "health endpoint should return a non-empty version"
        );
        assert!(
            health.nvme_cache.is_some(),
            "health endpoint should always return nvme cache metrics payload"
        );
        assert!(
            health.recovery_guard.is_some(),
            "health endpoint should always return recovery guard metrics payload"
        );
    }

    fn payload_keyword(value: &str) -> PayloadValue {
        PayloadValue {
            kind: Some(crate::drift_proto::payload_value::Kind::KeywordValue(
                value.to_string(),
            )),
        }
    }

    fn payload_int64(value: i64) -> PayloadValue {
        PayloadValue {
            kind: Some(crate::drift_proto::payload_value::Kind::Int64Value(value)),
        }
    }

    fn payload_row(entries: Vec<(u32, PayloadValue)>) -> PayloadRow {
        PayloadRow {
            fields: entries.into_iter().collect::<HashMap<_, _>>(),
        }
    }

    #[tokio::test]
    async fn test_search_field_filters_exact_anyof_range_and_projection() {
        let dir = tempdir().unwrap();
        let config = Config {
            port: 50057,
            wal_dir: dir.path().join("wal"),
            data_dir: dir.path().join("data"),
            default_dim: 2,
            max_bucket_capacity: 100,
            ef_construction: 32,
            ef_search: 32,
            storage: StorageCommand::File(FileConfig {
                path: dir.path().join("storage"),
            }),
        };

        let service = DriftService {
            manager: Arc::new(CollectionManager::new(config)),
        };
        let collection = "filters_demo";

        service
            .create_collection(Request::new(CreateCollectionRequest {
                collection_name: collection.to_string(),
                dim: 2,
                metric: MetricType::L2 as i32,
                max_bucket_capacity: 0,
            }))
            .await
            .expect("create_collection should succeed");

        let vectors = vec![
            Vector {
                id: 1,
                values: vec![0.0, 0.0],
            },
            Vector {
                id: 2,
                values: vec![0.1, 0.1],
            },
            Vector {
                id: 3,
                values: vec![0.2, 0.2],
            },
            Vector {
                id: 4,
                values: vec![0.3, 0.3],
            },
        ];
        let payload_rows = vec![
            payload_row(vec![(1, payload_keyword("tenant_a")), (2, payload_int64(10))]),
            payload_row(vec![(1, payload_keyword("tenant_b")), (2, payload_int64(20))]),
            payload_row(vec![(1, payload_keyword("tenant_a")), (2, payload_int64(30))]),
            payload_row(vec![(1, payload_keyword("tenant_c")), (2, payload_int64(40))]),
        ];

        service
            .insert_batch(Request::new(InsertBatchRequest {
                collection_name: collection.to_string(),
                vectors,
                payload_rows,
            }))
            .await
            .expect("insert_batch should succeed");

        let exact = service
            .search(Request::new(SearchRequest {
                collection_name: collection.to_string(),
                vector: vec![0.0, 0.0],
                k: 4,
                target_confidence: 0.99,
                lambda: 0.1,
                tau: 10.0,
                filters: vec![FieldFilter {
                    field_id: 1,
                    condition: Some(crate::drift_proto::field_filter::Condition::Exact(
                        payload_keyword("tenant_a"),
                    )),
                }],
                payload_projection_fields: vec![1, 2],
            }))
            .await
            .expect("exact-filter search should succeed")
            .into_inner();
        let exact_ids: Vec<u64> = exact.results.iter().map(|r| r.id).collect();
        assert_eq!(exact_ids, vec![1, 3]);
        assert!(
            exact
                .results
                .iter()
                .all(|r| r.payload.is_some() && r.payload.as_ref().unwrap().fields.contains_key(&1))
        );

        let any_of = service
            .search(Request::new(SearchRequest {
                collection_name: collection.to_string(),
                vector: vec![0.0, 0.0],
                k: 4,
                target_confidence: 0.99,
                lambda: 0.1,
                tau: 10.0,
                filters: vec![FieldFilter {
                    field_id: 1,
                    condition: Some(crate::drift_proto::field_filter::Condition::AnyOf(
                        PayloadValueList {
                            values: vec![payload_keyword("tenant_b"), payload_keyword("tenant_c")],
                        },
                    )),
                }],
                payload_projection_fields: vec![],
            }))
            .await
            .expect("any_of-filter search should succeed")
            .into_inner();
        let any_ids: Vec<u64> = any_of.results.iter().map(|r| r.id).collect();
        assert_eq!(any_ids, vec![2, 4]);

        let range = service
            .search(Request::new(SearchRequest {
                collection_name: collection.to_string(),
                vector: vec![0.0, 0.0],
                k: 4,
                target_confidence: 0.99,
                lambda: 0.1,
                tau: 10.0,
                filters: vec![FieldFilter {
                    field_id: 2,
                    condition: Some(crate::drift_proto::field_filter::Condition::Range(
                        RangeFilter {
                            lower: Some(payload_int64(15)),
                            lower_inclusive: Some(true),
                            upper: Some(payload_int64(35)),
                            upper_inclusive: Some(true),
                        },
                    )),
                }],
                payload_projection_fields: vec![],
            }))
            .await
            .expect("range-filter search should succeed")
            .into_inner();
        let range_ids: Vec<u64> = range.results.iter().map(|r| r.id).collect();
        assert_eq!(range_ids, vec![2, 3]);
    }

    #[tokio::test]
    async fn test_full_lifecycle_flush_promote_recover() {
        let dir = tempdir().unwrap();
        let wal_dir = dir.path().join("wal");
        let data_dir = dir.path().join("data");
        std::fs::create_dir(&data_dir).unwrap();

        let dim = 8;
        let bucket_id = 1;

        let manifest = Arc::new(ServerManifestManager::new(dir.path(), dim as u32).unwrap());
        let staging = Arc::new(LocalStagingManager::new(&data_dir).unwrap());
        let op = create_fs_operator(&data_dir);
        let persistence = PersistenceManager::new(op.clone());

        // A. Flush to Local Staging
        let group = mock_batch(0, 10, dim, 0.0);
        staging.append_batch(bucket_id, &group).await.unwrap();

        manifest
            .apply_atomic(|m| {
                m.add_bucket(bucket_id, String::new(), group.centroid.clone());
                m.update_bucket_stats(bucket_id, 10, 0);
            })
            .unwrap();

        // B. Promote to S3 (Simulating Janitor Logic)
        // 1. Read Local
        let (local_ids, local_vecs) = staging.read_full_bucket(bucket_id).await.unwrap();

        // 2. Read Remote (None here)
        // 3. Write New S3 Segment using the new primitive
        let (new_run_id, _) = persistence
            .write_remote_bucket(bucket_id, &local_ids, &local_vecs, dim)
            .await
            .unwrap();

        manifest
            .apply_atomic(|m| {
                m.update_bucket_run_id(bucket_id, new_run_id.clone());
            })
            .unwrap();
        staging.delete_bucket(bucket_id).await.unwrap();

        let coordinator = Arc::new(BucketCoordinator::new());

        // C. Recover
        let bucket_manager =
            BucketManager::new(op.clone(), op.clone(), 4, coordinator.clone(), Metric::L2);
        let recovery = RecoveryManager::new(&data_dir, manifest.clone());

        let (router_lock, _replay) = recovery
            .recover(&bucket_manager, dim, &wal_dir)
            .await
            .unwrap();

        // D. Verify
        let router = router_lock.read();
        assert!(router.get_centroid(bucket_id).is_some());

        let (reg_path, class) = bucket_manager
            .get_location(bucket_id)
            .expect("Bucket registered");
        assert!(reg_path.contains(&new_run_id));
        assert_eq!(class, StorageClass::Remote);

        let query = vec![5.0; dim];
        drop(router);

        let results = bucket_manager
            .search_and_refine(&[bucket_id], &query, 5, 15)
            .await;

        assert!(!results.is_empty(), "Should find results");
        assert_eq!(results[0].0, 5);
    }

    #[tokio::test]
    async fn test_recovery_local_priority() {
        let dir = tempdir().unwrap();
        let data_dir = dir.path().join("data");
        let staging_dir = data_dir.join("staging");
        std::fs::create_dir_all(&staging_dir).unwrap();

        let dim = 8;
        let bucket_id = 1;

        let manifest = Arc::new(ServerManifestManager::new(dir.path(), dim as u32).unwrap());

        // ⚡ Operators
        let remote_op = create_fs_operator(&data_dir);
        let local_op = create_fs_operator(&staging_dir); // Rooted at staging

        // 1. Create "Old" S3 File in data_dir (Remote)
        let run_id = "run_OLD";
        let s3_path = data_dir.join(format!("bucket_{}_{}.driftu", bucket_id, run_id));
        {
            UnifiedLocalWriter::write_vector_only_flat_to_path(
                &s3_path,
                &[100],
                &vec![100.0; dim],
                dim,
            )
            .unwrap();
        }

        // 2. Create "New" Local File in staging_dir
        let local_path = staging_dir.join(format!("bucket_{}.driftu", bucket_id));
        {
            UnifiedLocalWriter::write_vector_only_flat_to_path(
                &local_path,
                &[200],
                &vec![200.0; dim],
                dim,
            )
            .unwrap();
        }

        manifest
            .apply_atomic(|m| {
                m.add_bucket(bucket_id, run_id.to_string(), Some(vec![0.0; dim]));
            })
            .unwrap();

        // 4. Recover
        let coordinator = Arc::new(BucketCoordinator::new());
        let bucket_manager =
            BucketManager::new(local_op, remote_op, 1, coordinator.clone(), Metric::L2);

        let recovery = RecoveryManager::new(&data_dir, manifest.clone());
        let wal_dir = data_dir.join("wal");
        let _ = recovery
            .recover(&bucket_manager, dim, &wal_dir)
            .await
            .unwrap();

        // 5. Verify Priority
        let (path, class) = bucket_manager.get_location(bucket_id).unwrap();

        assert!(
            path.contains("bucket_1.driftu"),
            "Recovery failed to prefer Local Staging. Got: {}",
            path
        );
        assert_eq!(class, StorageClass::Local);

        let query = vec![200.0; dim];
        let results = bucket_manager
            .search_and_refine(&[bucket_id], &query, 1, 3)
            .await;

        assert!(!results.is_empty(), "Search should find data in local file");
        assert_eq!(results[0].0, 200, "Should find ID 200 from Local Staging");
    }

    #[tokio::test]
    async fn test_wal_replay() {
        let dir = tempdir().unwrap();
        let wal_dir = dir.path().join("wal").join("test_col");
        std::fs::create_dir_all(&wal_dir).unwrap();

        let wal_path = wal_dir.join("wal_1.log");

        let dim = 8;
        let manifest = Arc::new(ServerManifestManager::new(dir.path(), dim as u32).unwrap());
        let op = create_fs_operator(dir.path());
        let coordinator = Arc::new(BucketCoordinator::new());
        let bucket_manager =
            BucketManager::new(op.clone(), op.clone(), 1, coordinator.clone(), Metric::L2);

        // 1. Write to WAL
        {
            let mut wal = WalWriter::new(&wal_path).unwrap();
            wal.write_insert(999, &vec![1.0; dim]).unwrap();
            wal.write_insert(888, &vec![2.0; dim]).unwrap();
            wal.sync().unwrap();
        }

        // 2. Recover
        let recovery = RecoveryManager::new(dir.path(), manifest.clone());

        let (router, replay_data) = recovery
            .recover(&bucket_manager, dim, &wal_dir)
            .await
            .unwrap();

        assert!(
            router.read().get_centroid(0).is_none(),
            "Router should be empty"
        );

        assert_eq!(replay_data.inserts.len(), 2);
        assert_eq!(replay_data.inserts[0].0, 999);
        assert_eq!(replay_data.inserts[1].0, 888);
    }
}

#[cfg(test)]
mod janitor_reaper_integration_tests {
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
    use drift_storage::unified_writer::UnifiedLocalWriter;
    use opendal::{Operator, services};
    use parking_lot::Mutex;
    use parking_lot::RwLock;
    use std::sync::Arc;
    use std::time::Duration;
    use tempfile::tempdir;
    use tokio::time::sleep;

    fn create_fs_operator(path: &std::path::Path) -> Operator {
        let mut builder = services::Fs::default();
        builder = builder.root(path.to_str().unwrap());
        Operator::new(builder).unwrap().finish()
    }

    async fn create_bucket_file(dir: &std::path::Path, filename: &str, count: usize, dim: usize) {
        let path = dir.join(filename);
        let ids: Vec<u64> = (0..count as u64).collect();
        let vecs: Vec<f32> = (0..count * dim).map(|i| i as f32).collect();
        UnifiedLocalWriter::write_vector_only_flat_to_path(path, &ids, &vecs, dim).unwrap();
    }

    #[tokio::test]
    async fn test_janitor_automatically_reaps_promoted_files() {
        let _ = tracing_subscriber::fmt()
            .with_env_filter("drift_server=info,drift_storage=info")
            .with_test_writer()
            .try_init();

        let dir = tempdir().unwrap();
        let data_dir = dir.path().join("data");
        let staging_dir = data_dir.join("staging");
        std::fs::create_dir_all(&data_dir).unwrap();
        std::fs::create_dir_all(&staging_dir).unwrap();
        let dim = 2;

        let manifest = Arc::new(ServerManifestManager::new(dir.path(), dim as u32).unwrap());
        let staging = Arc::new(LocalStagingManager::new(&staging_dir).unwrap());
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

        let wal_mgr = Arc::new(Mutex::new(WalManager::new(dir.path().join("wal")).unwrap()));
        let kv = Arc::new(BitStore::new(dir.path().join("kv")).unwrap());
        let router = Arc::new(RwLock::new(Router::empty(dim, Metric::L2)));
        let index = Arc::new(VectorIndex::new(
            dim,
            100,
            router,
            wal_mgr,
            bucket_manager.clone(),
            kv,
        ));

        // 3. Create Local File
        let bucket_id = 1;
        let filename = format!("bucket_{}.driftu", bucket_id);
        let item_count = 50;

        create_bucket_file(&staging_dir, &filename, item_count, dim).await;

        staging.set_active_filename(bucket_id, filename.clone());

        // ⚡ FIX 1: Register with COUNT so Janitor knows it's not empty
        bucket_manager.register_bucket_with_count(
            bucket_id,
            filename.clone(),
            drift_storage::bucket_manager::StorageClass::Local,
            item_count as u32,
        );

        manifest
            .apply_atomic(|m| {
                m.add_bucket(bucket_id, "run1".into(), Some(vec![0.0; dim]));
                m.update_bucket_stats(bucket_id, item_count as u64, 0);
            })
            .unwrap();

        let janitor = Janitor::new(JanitorConfig {
            index: index.clone(),
            manifest: manifest.clone(),
            staging: staging.clone(),
            persistence: persistence.clone(),
            bucket_manager: bucket_manager.clone(),
            coordinator: coordinator.clone(),
            vars: JanitorVars {
                promotion_threshold_bytes: 100,
                check_interval: Duration::from_millis(10),
                // ⚡ FIX 2: Set capacity to match data size.
                // 50 items / 50 capacity = 100% full. Urgency = 0.
                // This prevents the "Zombie Merge" logic from deleting it.
                max_bucket_capacity: item_count,
                ..Default::default()
            },
        });

        let handle = tokio::spawn(async move { janitor.run().await });

        sleep(Duration::from_millis(1000)).await;
        handle.abort();

        // 5. VERIFY
        let local_path = staging_dir.join(&filename);
        assert!(
            !local_path.exists(),
            "Reaper failed! Local file {:?} still exists.",
            local_path
        );

        let mut found_remote = false;
        let mut remote_files = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&data_dir) {
            for entry in entries {
                let path = entry.unwrap().path();
                if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
                    remote_files.push(name.to_string());
                    if name.starts_with("bucket_1_") && name.ends_with(".driftu") {
                        found_remote = true;
                    }
                }
            }
        }

        if !found_remote {
            panic!(
                "Promotion failed! No remote file found in {:?}.\nFound files: {:?}",
                data_dir, remote_files
            );
        }
    }
}
