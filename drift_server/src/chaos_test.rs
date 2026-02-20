#[cfg(test)]
mod tests {
    use crate::drift_proto::{
        CreateCollectionRequest, HealthRequest, InsertRequest, MetricType, SearchRequest, Vector,
        drift_client::DriftClient,
    };
    use std::collections::HashSet;
    use std::io::ErrorKind;
    use std::io::Write;
    use std::path::PathBuf;
    use std::process::{Child, Command, Stdio};
    use std::time::{Duration, Instant};
    use tempfile::TempDir;
    use tokio::time::sleep;

    fn get_server_bin() -> PathBuf {
        let mut path = std::env::current_exe().expect("Failed to get current exe path");
        path.pop();
        if path.ends_with("deps") {
            path.pop();
        }
        path.push("drift_server");

        if !path.exists() {
            let cwd = std::env::current_dir().unwrap();
            let candidates = vec![
                cwd.join("target/debug/drift_server"),
                cwd.join("../target/debug/drift_server"),
            ];
            for candidate in candidates {
                if candidate.exists() {
                    return candidate;
                }
            }
            panic!(
                "❌ Could not find 'drift_server' binary. Run `cargo build -p drift_server --bin drift_server`"
            );
        }
        path
    }

    fn can_bind_listeners() -> bool {
        match std::net::TcpListener::bind("0.0.0.0:0") {
            Ok(listener) => {
                drop(listener);
                true
            }
            Err(err) if err.kind() == ErrorKind::PermissionDenied => {
                eprintln!(
                    "Skipping chaos test: environment disallows binding listener sockets ({err})"
                );
                false
            }
            Err(err) => {
                eprintln!("Warning: socket bind preflight failed unexpectedly ({err}); continuing");
                true
            }
        }
    }

    fn find_free_port() -> u16 {
        let listener =
            std::net::TcpListener::bind("127.0.0.1:0").expect("failed to reserve test port");
        let port = listener
            .local_addr()
            .expect("failed to read local socket address")
            .port();
        drop(listener);
        port
    }

    async fn spawn_server(port: u16, data_dir: &std::path::Path) -> Child {
        let bin_path = get_server_bin();
        print!("   🚀 Spawning server (Port {})... ", port);
        std::io::stdout().flush().unwrap();

        let mut child = Command::new(&bin_path)
            .arg("--port")
            .arg(port.to_string())
            .arg("--wal-dir")
            .arg(data_dir.join("wal"))
            .arg("--data-dir")
            .arg(data_dir.join("data"))
            .arg("file")
            .arg("--path")
            .arg(data_dir.join("storage"))
            // Keep test output clean; rely on explicit panic messages on failure.
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .env("RUST_LOG", "error")
            .spawn()
            .expect("Failed to spawn drift_server");

        let addr = format!("http://127.0.0.1:{}", port);

        // Wait for port to be open
        for _ in 0..50 {
            if let Ok(Some(status)) = child.try_wait() {
                // If we get here, the child exited before accepting connections.
                panic!(
                    "❌ Server died immediately (Exit Code: {:?})",
                    status.code()
                );
            }

            if DriftClient::connect(addr.clone()).await.is_ok() {
                println!("Connected.");
                return child;
            }
            sleep(Duration::from_millis(100)).await;
        }
        let _ = child.kill();
        panic!("❌ Server timed out connecting to {}", addr);
    }

    #[tokio::test]
    async fn test_chaos_wal_durability_kill_9() {
        if !can_bind_listeners() {
            return;
        }

        let dir = TempDir::new().unwrap();
        let port = find_free_port();
        let addr = format!("http://127.0.0.1:{}", port);
        let dim = 4;

        let mut confirmed_ids = HashSet::new();
        let mut next_id = 0;

        for epoch in 1..=3 {
            println!("\n⚡ EPOCH {} STARTING...", epoch);

            // 1. Start
            let mut process = spawn_server(port, dir.path()).await;

            // Warmup: wait for gRPC and health readiness.
            print!("   ⏳ Warming up with health check...");
            std::io::stdout().flush().unwrap();

            let warmup_start = Instant::now();
            let mut last_err = String::new();

            let mut client: DriftClient<tonic::transport::Channel> = loop {
                // Check if process died
                if let Ok(Some(status)) = process.try_wait() {
                    panic!("❌ Server process died during warmup! Exit: {:?}", status);
                }

                if warmup_start.elapsed() > Duration::from_secs(30) {
                    let _ = process.kill();
                    panic!("\n❌ Warmup timed out! Last Client Error: {}", last_err);
                }

                match DriftClient::connect(addr.clone()).await {
                    Ok(mut connected_client) => {
                        match connected_client
                            .health(tonic::Request::new(HealthRequest {}))
                            .await
                        {
                            Ok(_) => {
                                println!(" Healthy.");
                                break connected_client;
                            }
                            Err(e) if e.code() == tonic::Code::Unimplemented => {
                                // Backward compatibility for stale local binaries.
                                println!(" Health endpoint unavailable; continuing.");
                                break connected_client;
                            }
                            Err(e) => {
                                last_err = format!("Health failed: {e}");
                            }
                        }
                    }
                    Err(e) => {
                        last_err = e.to_string();
                    }
                }

                sleep(Duration::from_millis(200)).await;
            };

            let create_req = tonic::Request::new(CreateCollectionRequest {
                collection_name: "chaos_test".to_string(),
                dim: dim as u32,
                metric: MetricType::L2 as i32,
                max_bucket_capacity: 0,
            });
            if let Err(e) = client.create_collection(create_req).await
                && e.code() != tonic::Code::Unimplemented
            {
                let _ = process.kill();
                panic!("❌ create_collection failed during warmup: {}", e);
            }

            // Verify insert path is healthy after collection creation.
            let warmup_insert = tonic::Request::new(InsertRequest {
                collection_name: "chaos_test".to_string(),
                vector: Some(Vector {
                    id: 999999,
                    values: vec![0.0; dim],
                }),
            });
            match client.insert(warmup_insert).await {
                Ok(_) => {
                    println!("   ✅ Warmup insert accepted");
                }
                Err(e) => {
                    let _ = process.kill();
                    panic!("❌ Warmup insert failed after collection create: {}", e);
                }
            }

            // 2. Ingest
            println!("   💣 Bombarding...");
            let start = Instant::now();
            let run_duration = Duration::from_secs(3);

            while start.elapsed() < run_duration {
                let id = next_id;
                next_id += 1;

                let req = tonic::Request::new(InsertRequest {
                    collection_name: "chaos_test".to_string(),
                    vector: Some(Vector {
                        id,
                        values: vec![0.1; dim],
                    }),
                });

                let f = client.insert(req);
                if let Ok(Ok(_)) = tokio::time::timeout(Duration::from_millis(1000), f).await {
                    confirmed_ids.insert(id);
                    if id % 100 == 0 {
                        print!(".");
                        std::io::stdout().flush().unwrap();
                    }
                }
            }
            println!("");

            // 3. KILL
            println!("   💀 KILLING SERVER...");
            process.kill().expect("Failed to kill");
            let _ = process.wait();

            println!("   🛑 Stopped. Confirmed writes: {}", confirmed_ids.len());

            // ⚡ COOLDOWN
            sleep(Duration::from_millis(1000)).await;

            if confirmed_ids.is_empty() {
                panic!("❌ TEST FAILED: Server was too slow to accept ANY writes.");
            }

            // 4. RESTART
            println!("   ♻️  RESTARTING...");
            let mut process_2 = spawn_server(port, dir.path()).await;
            let mut client_2 = DriftClient::connect(addr.clone()).await.unwrap();

            // 5. VERIFY
            sleep(Duration::from_millis(500)).await;

            let res = client_2
                .search(tonic::Request::new(SearchRequest {
                    collection_name: "chaos_test".to_string(),
                    vector: vec![0.1; dim],
                    k: 10,
                    target_confidence: 0.9,
                    lambda: 1.0,
                    tau: 100.0,
                }))
                .await;

            match res {
                Ok(resp) => {
                    let hits = resp.into_inner().results;
                    if !hits.is_empty() {
                        println!("   ✅ RECOVERY SUCCESS: Found {} results.", hits.len());
                    } else {
                        panic!(
                            "❌ DATA LOSS: Index is empty but we confirmed {} writes!",
                            confirmed_ids.len()
                        );
                    }
                }
                Err(e) => panic!("❌ Search failed: {}", e),
            }

            process_2.kill().unwrap();
            let _ = process_2.wait();
        }
        println!("\n✅ CHAOS TEST PASSED");
    }
}
