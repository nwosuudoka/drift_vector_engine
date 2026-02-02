#[cfg(test)]
mod tests {
    use crate::drift_proto::{InsertRequest, SearchRequest, Vector, drift_client::DriftClient};
    use std::collections::HashSet;
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
            // ⚡ FIX: Pipe logs directly to terminal so we see them LIVE
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .env("RUST_LOG", "info")
            .spawn()
            .expect("Failed to spawn drift_server");

        let addr = format!("http://127.0.0.1:{}", port);

        // Wait for port to be open
        for _ in 0..50 {
            if let Ok(Some(status)) = child.try_wait() {
                // If we get here, it exited, so logs are already on screen due to inherit()
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
        let dir = TempDir::new().unwrap();
        let port = 50099;
        let addr = format!("http://127.0.0.1:{}", port);
        let dim = 4;

        let mut confirmed_ids = HashSet::new();
        let mut next_id = 0;

        for epoch in 1..=3 {
            println!("\n⚡ EPOCH {} STARTING...", epoch);

            // 1. Start
            let mut process = spawn_server(port, dir.path()).await;
            let mut client = DriftClient::connect(addr.clone()).await.unwrap();

            // ⚡ WARMUP
            print!("   ⏳ Warming up...");
            std::io::stdout().flush().unwrap();

            let warmup_start = Instant::now();
            let mut last_err = String::new();

            loop {
                // Check if process died
                if let Ok(Some(status)) = process.try_wait() {
                    panic!("❌ Server process died during warmup! Exit: {:?}", status);
                }

                if warmup_start.elapsed() > Duration::from_secs(30) {
                    let _ = process.kill();
                    panic!("\n❌ Warmup timed out! Last Client Error: {}", last_err);
                }

                let req = tonic::Request::new(InsertRequest {
                    collection_name: "chaos_test".to_string(),
                    vector: Some(Vector {
                        id: 999999,
                        values: vec![0.0; dim],
                    }),
                });

                match client.insert(req).await {
                    Ok(_) => {
                        println!(" Ready!");
                        break;
                    }
                    Err(e) => {
                        last_err = e.to_string();
                        // ⚡ Print retries so we know it's trying
                        // print!("x");
                        // std::io::stdout().flush().unwrap();
                    }
                }
                sleep(Duration::from_millis(200)).await;
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
