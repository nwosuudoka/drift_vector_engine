#[cfg(test)]
mod tests {
    use crate::local_store::LocalDiskManager;
    use crate::store::{PageId, PageManager};
    use std::sync::Arc;
    use tempfile::tempdir;
    use tokio::task;

    /// Test 1: Basic Persistence (Write -> Close -> Reopen -> Read)
    #[tokio::test]
    async fn test_persistence_across_restarts() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("data.bin");
        let file_id = 1;

        // Phase 1: Write
        {
            let manager = LocalDiskManager::new(dir.path());
            manager.register_file(file_id, file_path.clone());
            manager.write_page(file_id, 0, b"PersistMe").await.unwrap();
        } // Manager dropped here, file handles closed.

        // Phase 2: Reopen
        {
            let manager = LocalDiskManager::new(dir.path());
            manager.register_file(file_id, file_path.clone());

            let page = PageId {
                file_id,
                offset: 0,
                length: 9,
            };
            let data = manager.read_page(page).await.unwrap();
            assert_eq!(data, b"PersistMe");
        }
    }

    /// Test 2: Concurrency Hammer (The "Stress Test")
    /// 50 threads reading/writing to different offsets of the SAME file.
    /// This proves `pread` / `pwrite` are working without global locks blocking everything.
    #[tokio::test]
    async fn test_concurrent_io_same_file() {
        let dir = tempdir().unwrap();
        let manager = Arc::new(LocalDiskManager::new(dir.path()));

        let file_id = 1;
        let file_path = dir.path().join("concurrent.bin");
        manager.register_file(file_id, file_path);

        // Pre-allocate file size (avoid OS-level file growth lock contention for this specific test)
        // 50 threads * 100 bytes = 5000 bytes.
        manager.write_page(file_id, 5000, b"EOF").await.unwrap();

        let mut handles = vec![];

        for i in 0..50 {
            let m = manager.clone();
            handles.push(task::spawn(async move {
                let offset = i as u64 * 100;
                let payload = format!("Thread-{}", i).into_bytes();

                // Write
                m.write_page(file_id, offset, &payload).await.unwrap();

                // Read Back immediately
                let page = PageId {
                    file_id,
                    offset,
                    length: payload.len() as u32,
                };
                let result = m.read_page(page).await.unwrap();
                assert_eq!(result, payload);
            }));
        }

        for h in handles {
            h.await.unwrap();
        }
    }

    /// Test 3: Multi-File Management
    /// Ensures ID mapping works correctly for different files.
    #[tokio::test]
    async fn test_multi_file_isolation() {
        let dir = tempdir().unwrap();
        let manager = LocalDiskManager::new(dir.path());

        manager.register_file(1, dir.path().join("file_1.bin"));
        manager.register_file(2, dir.path().join("file_2.bin"));

        manager.write_page(1, 0, b"Data1").await.unwrap();
        manager.write_page(2, 0, b"Data2").await.unwrap();

        let p1 = PageId {
            file_id: 1,
            offset: 0,
            length: 5,
        };
        let p2 = PageId {
            file_id: 2,
            offset: 0,
            length: 5,
        };

        assert_eq!(manager.read_page(p1).await.unwrap(), b"Data1");
        assert_eq!(manager.read_page(p2).await.unwrap(), b"Data2");
    }

    /// Test 4: Error Handling (EOF / Missing File)
    #[tokio::test]
    async fn test_error_handling() {
        let dir = tempdir().unwrap();
        let manager = LocalDiskManager::new(dir.path());

        // A. Read Unregistered File
        let bad_page = PageId {
            file_id: 999,
            offset: 0,
            length: 10,
        };
        assert!(manager.read_page(bad_page).await.is_err());

        // B. Read Past EOF
        manager.register_file(1, dir.path().join("short.bin"));
        manager.write_page(1, 0, b"Tiny").await.unwrap();

        let eof_page = PageId {
            file_id: 1,
            offset: 100,
            length: 10,
        };
        // std::fs::read_at usually returns 0 bytes at EOF, or an error if we enforce exact length.
        // Our impl uses `read_exact_at`, so it MUST error on EOF.
        assert!(manager.read_page(eof_page).await.is_err());
    }
}
