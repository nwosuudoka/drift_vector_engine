#[cfg(test)]
mod tests {
    use crate::{local_store::LocalDiskManager, tiered_store::TieredPageManager};
    use drift_traits::{PageId, PageManager};
    use std::{sync::Arc, time::Duration};
    use tempfile::tempdir;
    use tokio::time::sleep;

    #[tokio::test]
    async fn test_tiered_cache_population_exact() {
        let dir = tempdir().unwrap();
        let local_dir = dir.path().join("local");
        let remote_dir = dir.path().join("remote");

        std::fs::create_dir_all(&local_dir).unwrap();
        std::fs::create_dir_all(&remote_dir).unwrap();

        let local = Arc::new(LocalDiskManager::new(&local_dir));
        let remote = Arc::new(LocalDiskManager::new(&remote_dir));

        let file_id = 1;
        let data = vec![0xAAu8; 100]; // 100 bytes of data

        // Write "Remote Segment" data (simulating a bucket)
        remote.write_page(file_id, 0, &data).await.unwrap();

        let tiered = TieredPageManager::new(local.clone(), remote.clone());

        // We request a slice from the middle
        let page = PageId {
            file_id,
            offset: 10,
            length: 20,
        };

        // 1. First Read (Cache Miss)
        // Should fetch exactly 20 bytes from Remote (offset 10) and return them.
        // It will async populate local cache with these 20 bytes.
        let res = tiered.read_page(page.clone()).await.unwrap();
        assert_eq!(res, &data[10..30]);

        // Wait for background population
        sleep(Duration::from_millis(200)).await;

        // 2. Verify Local Cache
        // ⚡ CHANGE: The local file contains JUST the 20 bytes we fetched.
        // It is stored at offset 0 of `1.bin`.
        let local_data = local
            .read_page(PageId {
                file_id,
                offset: 0,
                length: 20,
            })
            .await
            .unwrap();

        assert_eq!(local_data, &data[10..30]);

        // 3. Serve from Local
        // Corrupt remote to prove we hit local
        remote.write_page(file_id, 10, &[0x99]).await.unwrap(); // Corrupt index 10

        let cached_res = tiered.read_page(page).await.unwrap();
        assert_eq!(cached_res[0], 0xAA, "Did not serve from Local Cache!");
    }
}

#[cfg(test)]
mod edge_cases {
    use crate::{local_store::LocalDiskManager, tiered_store::TieredPageManager};
    use drift_traits::{PageId, PageManager};
    use std::{sync::Arc, time::Duration};
    use tempfile::{TempDir, tempdir};
    use tokio::time::sleep;

    fn setup() -> (
        TempDir,
        Arc<TieredPageManager>,
        Arc<LocalDiskManager>,
        Arc<LocalDiskManager>,
    ) {
        let dir = tempdir().unwrap();
        let local_path = dir.path().join("local");
        let remote_path = dir.path().join("remote");
        std::fs::create_dir_all(&local_path).unwrap();
        std::fs::create_dir_all(&remote_path).unwrap();

        let local = Arc::new(LocalDiskManager::new(local_path));
        let remote = Arc::new(LocalDiskManager::new(remote_path));
        let tiered = Arc::new(TieredPageManager::new(local.clone(), remote.clone()));

        (dir, tiered, local, remote)
    }

    #[tokio::test]
    async fn test_small_file_exact_cache() {
        let (_guard, tiered, local, remote) = setup();
        let file_id = 10;
        let data = vec![0xFFu8; 100];
        remote.write_page(file_id, 0, &data).await.unwrap();

        let page = PageId {
            file_id,
            offset: 0,
            length: 50,
        };
        let res = tiered.read_page(page).await.unwrap();

        assert_eq!(res.len(), 50);
        assert_eq!(res, &data[0..50]);

        sleep(Duration::from_millis(100)).await;

        // Verify local has exactly what was requested
        let local_data = local
            .read_page(PageId {
                file_id,
                offset: 0,
                length: 50,
            })
            .await
            .unwrap();
        assert_eq!(local_data, &data[0..50]);
    }

    #[tokio::test]
    async fn test_offset_read_dense_storage() {
        let (_guard, tiered, local, remote) = setup();
        let file_id = 20;

        // Remote file has data at a large offset
        let target_offset = 5000;
        let data = vec![0xAA, 0xBB];
        remote
            .write_page(file_id, target_offset, &data)
            .await
            .unwrap();

        // Request specific offset
        let page = PageId {
            file_id,
            offset: target_offset,
            length: 2,
        };
        let res = tiered.read_page(page).await.unwrap();
        assert_eq!(res, vec![0xAA, 0xBB]);

        sleep(Duration::from_millis(100)).await;

        // ⚡ CHANGE: Local file `20.bin` should contain the data at offset 0.
        // It does NOT replicate the 5000 byte offset (no sparse files).
        let local_check = local
            .read_page(PageId {
                file_id,
                offset: 0,
                length: 2,
            })
            .await
            .unwrap();
        assert_eq!(local_check, vec![0xAA, 0xBB]);
    }
}
