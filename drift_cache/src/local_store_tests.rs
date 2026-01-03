#[cfg(test)]
mod tests {
    use crate::local_store::LocalDiskManager;
    use drift_traits::{PageId, PageManager};
    use std::sync::Arc;
    use tempfile::tempdir;
    use tokio::task;

    #[tokio::test]
    async fn test_persistence_across_restarts() {
        let dir = tempdir().unwrap();
        let file_id = 1;

        // Phase 1: Write
        {
            let manager = LocalDiskManager::new(dir.path());
            manager.write_page(file_id, 0, b"PersistMe").await.unwrap();
        }

        // Phase 2: Reopen
        {
            let manager = LocalDiskManager::new(dir.path());
            let page = PageId {
                file_id,
                offset: 0,
                length: 9,
            };
            let data = manager.read_page(page).await.unwrap();
            assert_eq!(data, b"PersistMe");
        }
    }

    #[tokio::test]
    async fn test_concurrent_io_same_file() {
        let dir = tempdir().unwrap();
        let manager = Arc::new(LocalDiskManager::new(dir.path()));
        let file_id = 1;

        manager.write_page(file_id, 5000, b"EOF").await.unwrap();

        let mut handles = vec![];
        for i in 0..50 {
            let m = manager.clone();
            handles.push(task::spawn(async move {
                let offset = i as u64 * 100;
                let payload = format!("Thread-{}", i).into_bytes();
                m.write_page(file_id, offset, &payload).await.unwrap();
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

    #[tokio::test]
    async fn test_multi_file_isolation() {
        let dir = tempdir().unwrap();
        let manager = LocalDiskManager::new(dir.path());

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

    #[tokio::test]
    async fn test_error_handling() {
        let dir = tempdir().unwrap();
        let manager = LocalDiskManager::new(dir.path());

        // A. Read Unknown File
        // ⚡ FIX: We now expect NotFound, because we stopped auto-creating empty files on read.
        let bad_page = PageId {
            file_id: 999,
            offset: 0,
            length: 10,
        };
        let res = manager.read_page(bad_page).await;
        assert!(res.is_err(), "Should fail reading missing file");
        assert_eq!(res.unwrap_err().kind(), std::io::ErrorKind::NotFound);

        // B. Read Past EOF (Valid file)
        manager.write_page(1, 0, b"Tiny").await.unwrap();
        let eof_page = PageId {
            file_id: 1,
            offset: 100,
            length: 10,
        };
        assert!(manager.read_page(eof_page).await.is_err());
    }

    #[tokio::test]
    async fn test_auto_registration_on_write() {
        let dir = tempdir().unwrap();
        let manager = LocalDiskManager::new(dir.path());

        // 1. Write to a new ID (33)
        manager.write_page(33, 0, b"Magic").await.unwrap();

        // 2. Verify file exists on disk
        // ⚡ FIX: Check for .bin (matches LocalDiskManager impl)
        let expected_path = dir.path().join("33.bin");
        assert!(expected_path.exists(), "File 33.bin was not auto-created");

        // 3. Read back
        let page = PageId {
            file_id: 33,
            offset: 0,
            length: 5,
        };
        let data = manager.read_page(page).await.unwrap();
        assert_eq!(data, b"Magic");
    }
}
