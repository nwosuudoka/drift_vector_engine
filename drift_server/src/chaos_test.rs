#[cfg(test)]
mod tests {
    use crate::drift_proto::{
        CreateCollectionRequest, HealthRequest, InsertRequest, MetricType, SearchRequest, Vector,
        drift_client::DriftClient,
    };
    use std::collections::HashSet;
    use std::io::ErrorKind;
    use std::io::Write;
    use std::path::Path;
    use std::path::PathBuf;
    use std::process::{Child, Command, Stdio};
    use std::time::{Duration, Instant};
    use tempfile::TempDir;
    use tokio::time::sleep;

    const LOG_TAIL_LINES: usize = 60;

    struct SpawnedServer {
        child: Child,
        stdout_log: PathBuf,
        stderr_log: PathBuf,
    }

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

    fn read_log_tail(path: &Path, line_count: usize) -> String {
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => return format!("<unavailable: {}>", e),
        };

        let mut lines: Vec<&str> = content.lines().rev().take(line_count).collect();
        lines.reverse();
        if lines.is_empty() {
            "<empty>".to_string()
        } else {
            lines.join("\n")
        }
    }

    fn panic_with_server_logs(message: &str, stdout_log: &Path, stderr_log: &Path) -> ! {
        let stdout_tail = read_log_tail(stdout_log, LOG_TAIL_LINES);
        let stderr_tail = read_log_tail(stderr_log, LOG_TAIL_LINES);
        panic!(
            "{message}\n\
             📄 stdout log: {}\n\
             📄 stderr log: {}\n\
             --- stdout tail ---\n{}\n\
             --- stderr tail ---\n{}",
            stdout_log.display(),
            stderr_log.display(),
            stdout_tail,
            stderr_tail
        );
    }

    async fn spawn_server(
        port: u16,
        data_dir: &std::path::Path,
        epoch: u32,
        phase: &str,
    ) -> SpawnedServer {
        let bin_path = get_server_bin();
        print!("   🚀 Spawning server (Port {})... ", port);
        std::io::stdout().flush().unwrap();

        let log_dir = data_dir.join("chaos_logs");
        std::fs::create_dir_all(&log_dir).expect("failed to create chaos log dir");
        let stdout_log = log_dir.join(format!("epoch_{epoch}_{phase}_stdout.log"));
        let stderr_log = log_dir.join(format!("epoch_{epoch}_{phase}_stderr.log"));
        let stdout_file = std::fs::File::create(&stdout_log).expect("failed to create stdout log");
        let stderr_file = std::fs::File::create(&stderr_log).expect("failed to create stderr log");

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
            .stdout(Stdio::from(stdout_file))
            .stderr(Stdio::from(stderr_file))
            .env("RUST_LOG", "error")
            .spawn()
            .expect("Failed to spawn drift_server");

        let addr = format!("http://127.0.0.1:{}", port);

        // Wait for port to be open
        for _ in 0..50 {
            if let Ok(Some(status)) = child.try_wait() {
                // If we get here, the child exited before accepting connections.
                panic_with_server_logs(
                    &format!(
                        "❌ Server died immediately (Exit Code: {:?})",
                        status.code()
                    ),
                    &stdout_log,
                    &stderr_log,
                );
            }

            if DriftClient::connect(addr.clone()).await.is_ok() {
                println!("Connected.");
                return SpawnedServer {
                    child,
                    stdout_log,
                    stderr_log,
                };
            }
            sleep(Duration::from_millis(100)).await;
        }
        let _ = child.kill();
        panic_with_server_logs(
            &format!("❌ Server timed out connecting to {}", addr),
            &stdout_log,
            &stderr_log,
        );
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
            let mut process = spawn_server(port, dir.path(), epoch, "primary").await;

            // Warmup: wait for gRPC and health readiness.
            print!("   ⏳ Warming up with health check...");
            std::io::stdout().flush().unwrap();

            let warmup_start = Instant::now();
            let mut last_err = String::new();

            let mut client: DriftClient<tonic::transport::Channel> = loop {
                // Check if process died
                if let Ok(Some(status)) = process.child.try_wait() {
                    panic_with_server_logs(
                        &format!("❌ Server process died during warmup! Exit: {:?}", status),
                        &process.stdout_log,
                        &process.stderr_log,
                    );
                }

                if warmup_start.elapsed() > Duration::from_secs(30) {
                    let _ = process.child.kill();
                    panic_with_server_logs(
                        &format!("\n❌ Warmup timed out! Last Client Error: {}", last_err),
                        &process.stdout_log,
                        &process.stderr_log,
                    );
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
                                let _ = process.child.kill();
                                panic_with_server_logs(
                                    "❌ Health RPC is unimplemented. This usually means a stale \
                                     drift_server binary is running with older proto/server code. \
                                     Rebuild and restart with `cargo build -p drift_server --bin \
                                     drift_server`.",
                                    &process.stdout_log,
                                    &process.stderr_log,
                                );
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
            if let Err(e) = client.create_collection(create_req).await {
                let _ = process.child.kill();
                if e.code() == tonic::Code::Unimplemented {
                    panic_with_server_logs(
                        "❌ CreateCollection RPC is unimplemented. This usually means a stale \
                         drift_server binary is running with older proto/server code. Rebuild and \
                         restart with `cargo build -p drift_server --bin drift_server`.",
                        &process.stdout_log,
                        &process.stderr_log,
                    );
                }
                panic_with_server_logs(
                    &format!("❌ create_collection failed during warmup: {}", e),
                    &process.stdout_log,
                    &process.stderr_log,
                );
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
                    let _ = process.child.kill();
                    panic_with_server_logs(
                        &format!("❌ Warmup insert failed after collection create: {}", e),
                        &process.stdout_log,
                        &process.stderr_log,
                    );
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
            process.child.kill().expect("Failed to kill");
            let _ = process.child.wait();

            println!("   🛑 Stopped. Confirmed writes: {}", confirmed_ids.len());

            // ⚡ COOLDOWN
            sleep(Duration::from_millis(1000)).await;

            if confirmed_ids.is_empty() {
                panic!("❌ TEST FAILED: Server was too slow to accept ANY writes.");
            }

            // 4. RESTART
            println!("   ♻️  RESTARTING...");
            let mut process_2 = spawn_server(port, dir.path(), epoch, "restart").await;
            let mut client_2 = match DriftClient::connect(addr.clone()).await {
                Ok(c) => c,
                Err(e) => {
                    panic_with_server_logs(
                        &format!("❌ Failed to reconnect after restart: {}", e),
                        &process_2.stdout_log,
                        &process_2.stderr_log,
                    );
                }
            };

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
                        panic_with_server_logs(
                            &format!(
                                "❌ DATA LOSS: Index is empty but we confirmed {} writes!",
                                confirmed_ids.len()
                            ),
                            &process_2.stdout_log,
                            &process_2.stderr_log,
                        );
                    }
                }
                Err(e) => panic_with_server_logs(
                    &format!("❌ Search failed after restart: {}", e),
                    &process_2.stdout_log,
                    &process_2.stderr_log,
                ),
            }

            process_2.child.kill().unwrap();
            let _ = process_2.child.wait();
        }
        println!("\n✅ CHAOS TEST PASSED");
    }
}
