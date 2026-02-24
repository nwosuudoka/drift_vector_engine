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
    use std::sync::Once;
    use std::time::{Duration, Instant};
    use tempfile::TempDir;
    use tokio::time::sleep;

    const LOG_TAIL_LINES: usize = 60;
    const STARTUP_CONNECT_ATTEMPTS: usize = 50;
    const HEALTH_WAIT_TIMEOUT: Duration = Duration::from_secs(30);
    const HEALTH_POLL_INTERVAL: Duration = Duration::from_millis(200);
    const RECOVERY_VERIFY_TIMEOUT: Duration = Duration::from_secs(10);
    const COLLECTION_NAME: &str = "chaos_test";

    struct SpawnedServer {
        child: Child,
        stdout_log: PathBuf,
        stderr_log: PathBuf,
    }

    fn ensure_server_bin_is_current() {
        static BUILD_ONCE: Once = Once::new();
        BUILD_ONCE.call_once(|| {
            let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            let workspace_root = manifest_dir.parent().unwrap_or(&manifest_dir);

            let status = Command::new("cargo")
                .arg("build")
                .arg("-p")
                .arg("drift_server")
                .arg("--bin")
                .arg("drift_server")
                .current_dir(workspace_root)
                .status()
                .expect("Failed to run `cargo build -p drift_server --bin drift_server`");

            assert!(
                status.success(),
                "❌ Unable to build drift_server binary for chaos test"
            );
        });
    }

    fn get_server_bin() -> PathBuf {
        if let Ok(bin) = std::env::var("CARGO_BIN_EXE_drift_server") {
            let path = PathBuf::from(bin);
            if path.exists() {
                return path;
            }
        }

        ensure_server_bin_is_current();

        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let workspace_root = manifest_dir.parent().unwrap_or(&manifest_dir);
        let mut path = std::env::current_exe().expect("Failed to get current exe path");
        path.pop();
        if path.ends_with("deps") {
            path.pop();
        }
        path.push("drift_server");

        if path.exists() {
            return path;
        }

        let candidates = vec![
            workspace_root.join("target/debug/drift_server"),
            std::env::current_dir()
                .unwrap_or_else(|_| workspace_root.to_path_buf())
                .join("target/debug/drift_server"),
            std::env::current_dir()
                .unwrap_or_else(|_| workspace_root.to_path_buf())
                .join("../target/debug/drift_server"),
        ];
        for candidate in candidates {
            if candidate.exists() {
                return candidate;
            }
        }

        panic!(
            "❌ Could not find 'drift_server' binary after build. Expected target/debug/drift_server"
        );
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

    fn stop_process(process: &mut SpawnedServer) {
        let _ = process.child.kill();
        let _ = process.child.wait();
    }

    async fn connect_healthy_client(
        addr: &str,
        process: &mut SpawnedServer,
        phase: &str,
    ) -> DriftClient<tonic::transport::Channel> {
        print!("   ⏳ Waiting for healthy server ({phase})...");
        std::io::stdout().flush().unwrap();

        let start = Instant::now();
        let mut last_err = String::new();

        loop {
            if let Ok(Some(status)) = process.child.try_wait() {
                panic_with_server_logs(
                    &format!(
                        "❌ Server process died while waiting for health during {phase}! Exit: {:?}",
                        status
                    ),
                    &process.stdout_log,
                    &process.stderr_log,
                );
            }

            if start.elapsed() > HEALTH_WAIT_TIMEOUT {
                stop_process(process);
                panic_with_server_logs(
                    &format!(
                        "❌ {phase} timed out waiting for health. Last client error: {}",
                        last_err
                    ),
                    &process.stdout_log,
                    &process.stderr_log,
                );
            }

            match DriftClient::connect(addr.to_string()).await {
                Ok(mut client) => {
                    match client.health(tonic::Request::new(HealthRequest {})).await {
                        Ok(resp) => {
                            let health = resp.into_inner();
                            if health.ready {
                                println!(" Healthy.");
                                return client;
                            }
                            last_err = format!("health.ready=false (version={})", health.version);
                        }
                        Err(e) if e.code() == tonic::Code::Unimplemented => {
                            stop_process(process);
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
                            last_err = format!("health failed: {e}");
                        }
                    }
                }
                Err(e) => {
                    last_err = e.to_string();
                }
            }

            sleep(HEALTH_POLL_INTERVAL).await;
        }
    }

    async fn verify_epoch_recovery(
        client: &mut DriftClient<tonic::transport::Channel>,
        process: &mut SpawnedServer,
        dim: usize,
        epoch_value: f32,
        epoch_confirmed_ids: &HashSet<u64>,
    ) {
        let start = Instant::now();
        let mut last_detail = String::new();

        loop {
            if let Ok(Some(status)) = process.child.try_wait() {
                panic_with_server_logs(
                    &format!(
                        "❌ Restarted server died during recovery verification! Exit: {:?}",
                        status
                    ),
                    &process.stdout_log,
                    &process.stderr_log,
                );
            }

            if start.elapsed() > RECOVERY_VERIFY_TIMEOUT {
                panic_with_server_logs(
                    &format!(
                        "❌ Recovery verification timed out. Last detail: {}",
                        last_detail
                    ),
                    &process.stdout_log,
                    &process.stderr_log,
                );
            }

            let res = client
                .search(tonic::Request::new(SearchRequest {
                    collection_name: COLLECTION_NAME.to_string(),
                    vector: vec![epoch_value; dim],
                    k: 20,
                    target_confidence: 0.9,
                    lambda: 1.0,
                    tau: 100.0,
                    filters: vec![],
                    payload_projection_fields: vec![],
                }))
                .await;

            match res {
                Ok(resp) => {
                    let hits = resp.into_inner().results;
                    let recovered = hits
                        .iter()
                        .filter(|hit| epoch_confirmed_ids.contains(&hit.id))
                        .count();
                    if recovered > 0 {
                        println!(
                            "   ✅ RECOVERY SUCCESS: matched {} confirmed ids for this epoch.",
                            recovered
                        );
                        return;
                    }

                    last_detail = format!(
                        "0 matching hits. returned={} confirmed_this_epoch={}",
                        hits.len(),
                        epoch_confirmed_ids.len()
                    );
                }
                Err(e) => {
                    last_detail = format!("search failed: {e}");
                }
            }

            sleep(Duration::from_millis(250)).await;
        }
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
        for _ in 0..STARTUP_CONNECT_ATTEMPTS {
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

        let mut confirmed_ids: HashSet<u64> = HashSet::new();
        let mut next_id: u64 = 0;

        for epoch in 1..=3 {
            println!("\n⚡ EPOCH {} STARTING...", epoch);
            let mut epoch_confirmed_ids: HashSet<u64> = HashSet::new();
            let epoch_value = epoch as f32 * 10.0 + 0.1;

            // 1. Start
            let mut process = spawn_server(port, dir.path(), epoch, "primary").await;
            let mut client = connect_healthy_client(&addr, &mut process, "warmup").await;

            let create_req = tonic::Request::new(CreateCollectionRequest {
                collection_name: COLLECTION_NAME.to_string(),
                dim: dim as u32,
                metric: MetricType::L2 as i32,
                max_bucket_capacity: 0,
            });
            if let Err(e) = client.create_collection(create_req).await {
                stop_process(&mut process);
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
                collection_name: COLLECTION_NAME.to_string(),
                vector: Some(Vector {
                    id: 999999,
                    values: vec![0.0; dim],
                }),
                payload: None,
            });
            match client.insert(warmup_insert).await {
                Ok(_) => {
                    println!("   ✅ Warmup insert accepted");
                }
                Err(e) => {
                    stop_process(&mut process);
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
                    collection_name: COLLECTION_NAME.to_string(),
                    vector: Some(Vector {
                        id,
                        values: vec![epoch_value; dim],
                    }),
                    payload: None,
                });

                let f = client.insert(req);
                if let Ok(Ok(_)) = tokio::time::timeout(Duration::from_millis(1000), f).await {
                    confirmed_ids.insert(id);
                    epoch_confirmed_ids.insert(id);
                    if id % 100 == 0 {
                        print!(".");
                        std::io::stdout().flush().unwrap();
                    }
                }
            }
            println!("");

            // 3. KILL
            println!("   💀 KILLING SERVER...");
            stop_process(&mut process);

            println!(
                "   🛑 Stopped. Confirmed writes: total={}, epoch={}",
                confirmed_ids.len(),
                epoch_confirmed_ids.len()
            );

            // ⚡ COOLDOWN
            sleep(Duration::from_millis(1000)).await;

            if epoch_confirmed_ids.is_empty() {
                panic!(
                    "❌ TEST FAILED: Server was too slow to accept any writes in epoch {}.",
                    epoch
                );
            }

            // 4. RESTART
            println!("   ♻️  RESTARTING...");
            let mut process_2 = spawn_server(port, dir.path(), epoch, "restart").await;
            let mut client_2 = connect_healthy_client(&addr, &mut process_2, "restart").await;

            // 5. VERIFY
            verify_epoch_recovery(
                &mut client_2,
                &mut process_2,
                dim,
                epoch_value,
                &epoch_confirmed_ids,
            )
            .await;

            stop_process(&mut process_2);
        }
        println!("\n✅ CHAOS TEST PASSED");
    }
}
