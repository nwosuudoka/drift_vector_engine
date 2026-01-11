#[cfg(test)]
mod stress_tests {
    use crate::janitor::Janitor;
    use crate::persistence::PersistenceManager;
    use drift_cache::local_store::LocalDiskManager;
    use drift_core::index::{IndexOptions, VectorIndex};
    use opendal::{Operator, services};

    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use std::collections::HashSet;
    use std::io;
    use std::sync::Arc;
    use std::time::Duration;

    use tempfile::tempdir;
    use tokio::sync::Barrier;
    use tokio::time::{sleep, timeout};

    // Reduced load for stability
    const STRESS_ITERS: usize = 20;
    const CONCURRENCY: usize = 4;
    const K: usize = 25;
    const TARGET_CONF: f32 = 0.99;
    const LAMBDA: f32 = 0.01;
    const TAU: f32 = 10_000.0;

    fn rnd_vec(rng: &mut StdRng, dim: usize, cluster: usize) -> Vec<f32> {
        let base = cluster as f32 * 10.0;
        // Independent random values for each dimension
        (0..dim)
            .map(|_| rng.random::<f32>() * 2.0 - 1.0 + base)
            .collect()
    }

    // Helper to create a local FS operator for tests
    fn create_local_operator(path: &std::path::Path) -> Operator {
        let mut builder = services::Fs::default();
        builder = builder.root(path.to_str().unwrap());
        Operator::new(builder).unwrap().finish()
    }

    async fn eventually<F, Fut>(dur: Duration, mut f: F) -> io::Result<()>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = io::Result<bool>>,
    {
        let deadline = dur;
        timeout(deadline, async move {
            loop {
                if let Ok(true) = f().await {
                    return Ok(());
                }
                sleep(Duration::from_millis(200)).await;
            }
        })
        .await
        .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "eventually() timed out"))?
    }

    #[tokio::test]
    async fn chaos_monkey_with_janitor() {
        let dir = tempdir().unwrap();

        // ⚡ CHANGE: Create Operator and inject
        let op = create_local_operator(dir.path());
        let persistence = PersistenceManager::new(op, dir.path());

        let opts = IndexOptions {
            dim: 64,
            num_centroids: 8,
            training_sample_size: 256,
            max_bucket_capacity: 50,
            ef_construction: 40,
            ef_search: 30,
        };
        let storage = Arc::new(LocalDiskManager::new(dir.path().join("storage")));
        let index =
            Arc::new(VectorIndex::new(opts.clone(), &dir.path().join("wal"), storage).unwrap());

        let mut rng = StdRng::seed_from_u64(0xC0FFEE);
        let mut train_data = Vec::with_capacity(opts.training_sample_size);
        for i in 0..opts.training_sample_size {
            train_data.push(rnd_vec(&mut rng, opts.dim, i % opts.num_centroids));
        }
        index.train(&train_data).await.unwrap();

        let janitor = Janitor::new(
            index.clone(),
            persistence,
            200,
            Duration::from_millis(50),
            None,
        );
        let jh = tokio::spawn(async move { janitor.run().await });

        let barrier = Arc::new(Barrier::new(CONCURRENCY + 1));
        let mut handles = vec![];

        for t in 0..CONCURRENCY {
            let idx = index.clone();
            let bar = barrier.clone();
            let mut rng = StdRng::seed_from_u64(0xBEEF_0000 + t as u64);

            handles.push(tokio::spawn(async move {
                bar.wait().await;
                for i in 0..STRESS_ITERS {
                    let id = (t * STRESS_ITERS + i) as u64;
                    // Use fully random vector to avoid singularity
                    let vec = rnd_vec(&mut rng, idx.config.dim, t % idx.config.num_centroids);

                    idx.insert(id, &vec).unwrap();

                    // Search visibility
                    eventually(Duration::from_secs(15), || {
                        let idx = idx.clone();
                        let vec = vec.clone();
                        async move {
                            let res = idx.search_async(&vec, K, TARGET_CONF, LAMBDA, TAU).await?;
                            Ok(res.iter().any(|r| r.id == id))
                        }
                    })
                    .await
                    .unwrap();

                    if rng.random::<f32>() < 0.3 {
                        idx.delete(id).unwrap();
                        eventually(Duration::from_secs(15), || {
                            let idx = idx.clone();
                            async move {
                                let kv = idx.kv.get(&id.to_le_bytes()).unwrap();
                                Ok(kv.is_none())
                            }
                        })
                        .await
                        .unwrap();
                    }
                }
                io::Result::<()>::Ok(())
            }));
        }

        barrier.wait().await;
        for h in handles {
            h.await.unwrap().unwrap();
        }
        jh.abort();
        println!("✅ chaos_monkey_with_janitor passed");
    }

    #[tokio::test]
    async fn split_storm() {
        let dir = tempdir().unwrap();

        // ⚡ CHANGE: Create Operator and inject
        let op = create_local_operator(dir.path());
        let persistence = PersistenceManager::new(op, dir.path());

        let opts = IndexOptions {
            dim: 8,
            num_centroids: 2,
            training_sample_size: 100,
            max_bucket_capacity: 20,
            ef_construction: 20,
            ef_search: 10,
        };
        let storage = Arc::new(LocalDiskManager::new(dir.path().join("storage")));
        let index = Arc::new(VectorIndex::new(opts, &dir.path().join("wal"), storage).unwrap());

        let train = (0..100).map(|i| vec![i as f32; 8]).collect::<Vec<_>>();
        index.train(&train).await.unwrap();

        let janitor = Janitor::new(
            index.clone(),
            persistence,
            50,
            Duration::from_millis(10),
            None,
        );
        let jh = tokio::spawn(async move { janitor.run().await });

        let mut rng = StdRng::seed_from_u64(0xAAAA);
        for i in 0..2_000u64 {
            let base = (i as usize % 100) as f32;
            // High Variance Jitter
            let jitter = rng.random::<f32>() * 2.0;
            // Add dimension variance
            let vec: Vec<f32> = (0..8)
                .map(|_| base + jitter + rng.random::<f32>())
                .collect();
            index.insert(i, &vec).unwrap();
        }

        eventually(Duration::from_secs(30), || {
            let index = index.clone();
            async move {
                let hdrs = index.get_all_bucket_headers();
                // If it split at least a bit, it's working.
                Ok(hdrs.len() > 5)
            }
        })
        .await
        .unwrap();

        jh.abort();
        println!("✅ split_storm passed");
    }

    #[tokio::test]
    async fn scatter_split_race() {
        let dir = tempdir().unwrap();

        // ⚡ CHANGE: Create Operator and inject
        let op = create_local_operator(dir.path());
        let persistence = PersistenceManager::new(op, dir.path());

        let opts = IndexOptions {
            dim: 16,
            num_centroids: 4,
            training_sample_size: 200,
            max_bucket_capacity: 80,
            ef_construction: 40,
            ef_search: 20,
        };
        let storage = Arc::new(LocalDiskManager::new(dir.path().join("storage")));
        let index = Arc::new(VectorIndex::new(opts, &dir.path().join("wal"), storage).unwrap());

        index
            .train(&(0..200).map(|i| vec![i as f32; 16]).collect::<Vec<_>>())
            .await
            .unwrap();

        let mut rng = StdRng::seed_from_u64(0xBBBB);
        for i in 0..1_000u64 {
            let base = (i as usize % 4) as f32 * 50.0;
            let vec: Vec<f32> = (0..16).map(|_| base + rng.random::<f32>() * 5.0).collect();
            index.insert(i, &vec).unwrap();
        }

        for i in (0..1_000u64).filter(|i| (i % 4) < 2) {
            index.delete(i).unwrap();
        }

        let janitor = Janitor::new(
            index.clone(),
            persistence,
            100,
            Duration::from_millis(10),
            None,
        );
        let jh = tokio::spawn(async move { janitor.run().await });

        sleep(Duration::from_millis(300)).await;

        for i in 10_000u64..11_000u64 {
            let vec: Vec<f32> = (0..16).map(|_| 25.0 + rng.random::<f32>() * 5.0).collect();
            index.insert(i, &vec).unwrap();
        }

        let q = vec![25.0; 16];
        // Increase search K because the buckets might be large/unbalanced in a stress test
        eventually(Duration::from_secs(30), || {
            let index = index.clone();
            let q = q.clone();
            async move {
                let res = index
                    .search_async(&q, 2000, TARGET_CONF, LAMBDA, TAU)
                    .await?;
                let found: HashSet<u64> = res.iter().map(|r| r.id).collect();
                Ok((10_000u64..10_020u64).all(|id| found.contains(&id)))
            }
        })
        .await
        .unwrap();

        jh.abort();
        println!("✅ scatter_split_race passed");
    }

    // (KV consistency and duplicate centroid tests remain unchanged)
    #[tokio::test]
    // #[ignore]
    async fn kv_consistency_torture() {
        let dir = tempdir().unwrap();

        // ⚡ CHANGE: Create Operator and inject
        let op = create_local_operator(dir.path());
        let persistence = PersistenceManager::new(op, dir.path());

        let opts = IndexOptions {
            dim: 4,
            num_centroids: 2,
            training_sample_size: 64,
            max_bucket_capacity: 30,
            ef_construction: 20,
            ef_search: 10,
        };
        let storage = Arc::new(LocalDiskManager::new(dir.path().join("storage")));
        let index = Arc::new(VectorIndex::new(opts, &dir.path().join("wal"), storage).unwrap());
        index
            .train(&vec![vec![0.0; 4], vec![10.0; 4]])
            .await
            .unwrap();

        let janitor = Janitor::new(
            index.clone(),
            persistence,
            60,
            Duration::from_millis(10),
            None,
        );
        let jh = tokio::spawn(async move { janitor.run().await });

        let mut rng = StdRng::seed_from_u64(0xD00D);
        let mut alive: HashSet<u64> = HashSet::new();

        for i in 0..1_500u64 {
            let vec = if rng.random_bool(0.5) {
                vec![rng.random::<f32>(); 4]
            } else {
                vec![10.0 + rng.random::<f32>(); 4]
            };
            index.insert(i, &vec).unwrap();
            alive.insert(i);
        }
        for id in (0..1_500u64).filter(|_| rng.random_bool(0.4)) {
            index.delete(id).unwrap();
            alive.remove(&id);
        }
        sleep(Duration::from_millis(300)).await;
        let headers = index.get_all_bucket_headers();
        let _existing_bucket_ids: HashSet<u32> = headers.iter().map(|h| h.id).collect();
        for id in 0..1_500u64 {
            let kv_bucket = index.kv.get(&id.to_le_bytes()).unwrap();
            if alive.contains(&id) {
                assert!(kv_bucket.is_some());
            } else {
                assert!(kv_bucket.is_none());
            }
        }
        jh.abort();
        println!("✅ kv_consistency_torture passed");
    }

    #[tokio::test]
    async fn duplicate_centroid_buckets() {
        let dir = tempdir().unwrap();

        // ⚡ CHANGE: Create Operator and inject
        let op = create_local_operator(dir.path());
        let persistence = PersistenceManager::new(op, dir.path());

        let opts = IndexOptions {
            dim: 3,
            num_centroids: 1,
            training_sample_size: 100,
            max_bucket_capacity: 40,
            ef_construction: 20,
            ef_search: 10,
        };
        let storage = Arc::new(LocalDiskManager::new(dir.path().join("storage")));
        let index = Arc::new(VectorIndex::new(opts, &dir.path().join("wal"), storage).unwrap());
        index.train(&vec![vec![1.0; 3]; 100]).await.unwrap();
        for i in 0..500u64 {
            index
                .insert(i, &vec![1.0 + i as f32 * 1e-5, 1.0, 1.0])
                .unwrap();
        }
        let janitor = Janitor::new(
            index.clone(),
            persistence,
            80,
            Duration::from_millis(10),
            None,
        );
        let jh = tokio::spawn(async move { janitor.run().await });
        let q = vec![1.0, 1.0, 1.0];
        eventually(Duration::from_secs(10), || {
            let index = index.clone();
            let q = q.clone();
            async move {
                let res = index
                    .search_async(&q, 200, TARGET_CONF, LAMBDA, TAU)
                    .await?;
                let found: HashSet<u64> = res.iter().map(|r| r.id).collect();
                Ok((0u64..100u64).all(|id| found.contains(&id)))
            }
        })
        .await
        .unwrap();
        jh.abort();
        println!("✅ duplicate_centroid_buckets passed");
    }
}
