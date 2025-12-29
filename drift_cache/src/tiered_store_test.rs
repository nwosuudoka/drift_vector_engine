#[cfg(test)]
mod tests {
    use crate::{
        local_store::LocalDiskManager,
        tiered_store::{REMOTE_CHUNK_SIZE, TieredPageManager},
    };
    use drift_traits::{PageId, PageManager};
    use std::{sync::Arc, time::Duration};
    use tempfile::tempdir;
    use tokio::time::sleep;

    #[tokio::test]
    async fn test_tiered_cache_population_chunked() {
        let dir = tempdir().unwrap();
        let local_dir = dir.path().join("local");
        let remote_dir = dir.path().join("remote");

        std::fs::create_dir_all(&local_dir).unwrap();
        std::fs::create_dir_all(&remote_dir).unwrap();

        let local = Arc::new(LocalDiskManager::new(&local_dir));
        let remote = Arc::new(LocalDiskManager::new(&remote_dir));

        // Setup: Write a full CHUNK size to Remote to ensure the prefetch logic succeeds.
        // If we write less than 4MB, the LocalDiskManager (acting as S3) will throw EOF
        // on the chunk read, triggering the fallback path which skips caching.
        let file_id = 1;

        // Create 4MB of dummy data
        let chunk_size = REMOTE_CHUNK_SIZE as usize;
        let data = vec![0xAAu8; chunk_size];

        remote.write_page(file_id, 0, &data).await.unwrap();

        // Setup Tiered Manager
        let tiered = TieredPageManager::new(local.clone(), remote.clone());

        // We only request a small slice (first 10 bytes)
        let page = PageId {
            file_id,
            offset: 0,
            length: 10,
        };

        // 1. First Read (Should hit Remote Chunk Read -> trigger populate)
        let res = tiered.read_page(page.clone()).await.unwrap();
        assert_eq!(res, &data[0..10]);

        // Wait for background population (async)
        sleep(Duration::from_millis(200)).await;

        // 2. Verify Local Cache now has the FULL CHUNK
        // We read the *chunk* from local storage to verify the prefetch worked.
        let local_chunk = local
            .read_page(PageId {
                file_id,
                offset: 0,
                length: chunk_size as u32,
            })
            .await
            .unwrap();

        assert_eq!(
            local_chunk.len(),
            chunk_size,
            "Local cache did not store the full chunk"
        );
        assert_eq!(local_chunk[0], 0xAA);

        // 3. Corrupt Remote (to prove we are reading from Local now)
        // We overwrite the first byte on remote
        remote.write_page(file_id, 0, &[0xBB]).await.unwrap();

        // Read again from Tiered (should serve 0xAA from Local, not 0xBB)
        let cached_res = tiered.read_page(page).await.unwrap();
        assert_eq!(cached_res[0], 0xAA, "Did not serve from Local Cache!");
    }
}

#[cfg(test)]
mod edge_cases {
    use crate::{
        local_store::LocalDiskManager,
        tiered_store::{REMOTE_CHUNK_SIZE, TieredPageManager},
    };
    use drift_traits::{PageId, PageManager};
    use std::{sync::Arc, time::Duration};
    use tempfile::{TempDir, tempdir};
    use tokio::time::sleep;

    // Helper to setup tiered environment
    // MUST return TempDir to prevent early cleanup!
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
    async fn test_small_file_handling() {
        let (_guard, tiered, local, remote) = setup();
        let file_id = 10;

        // 1. Create a tiny file on Remote (100 bytes)
        let data = vec![0xFFu8; 100];
        remote.write_page(file_id, 0, &data).await.unwrap();

        // 2. Read first 50 bytes via Tiered
        // The manager will try 4MB -> Fail (Mock is strict) -> Fallback to 50 bytes.
        let page = PageId {
            file_id,
            offset: 0,
            length: 50,
        };
        let res = tiered.read_page(page).await.unwrap();

        assert_eq!(res.len(), 50);
        assert_eq!(res, &data[0..50]);

        // Wait for async population
        sleep(Duration::from_millis(100)).await;

        // 3. Verify Local Cache has the *requested* slice (50 bytes)
        // (Since prefetch failed, we only cached what we saw)
        let local_data = local
            .read_page(PageId {
                file_id,
                offset: 0,
                length: 50,
            })
            .await
            .unwrap();
        assert_eq!(local_data.len(), 50);
        assert_eq!(local_data, &data[0..50]);
    }

    #[tokio::test]
    async fn test_offset_read_within_chunk() {
        let (_guard, tiered, local, remote) = setup();
        let file_id = 20;

        // 1. Create a full 4MB chunk on Remote
        let chunk_size = REMOTE_CHUNK_SIZE as usize;
        let mut data = vec![0x00u8; chunk_size];
        let target_offset = 1024 * 1024; // 1MB
        data[target_offset] = 0xAA;
        data[target_offset + 1] = 0xBB;

        remote.write_page(file_id, 0, &data).await.unwrap();

        // 2. Read tiny slice from middle
        let page = PageId {
            file_id,
            offset: target_offset as u64,
            length: 2,
        };
        let res = tiered.read_page(page).await.unwrap();

        assert_eq!(res, vec![0xAA, 0xBB]);

        // Wait for async population
        sleep(Duration::from_millis(100)).await;

        // 3. Verify Local Cache has the chunk head
        let local_head = local
            .read_page(PageId {
                file_id,
                offset: 0,
                length: 10,
            })
            .await
            .unwrap();
        assert_eq!(local_head, vec![0x00; 10]);
    }

    #[tokio::test]
    async fn test_straddle_boundary_bypass() {
        let (_guard, tiered, local, remote) = setup();
        let file_id = 30;

        // 1. Create Data spanning the 4MB boundary
        let boundary = REMOTE_CHUNK_SIZE;
        let start_write = boundary - 10;
        let data_len = 20;
        let data = vec![0x77u8; data_len];

        remote
            .write_page(file_id, start_write, &data)
            .await
            .unwrap();

        // 2. Request Straddling Data
        let page = PageId {
            file_id,
            offset: boundary - 5,
            length: 10,
        };

        let res = tiered.read_page(page).await.unwrap();
        assert_eq!(res, vec![0x77u8; 10]);

        sleep(Duration::from_millis(100)).await;

        // 3. Verify Local Cache is EMPTY (Bypassed)
        // Trying to read from local should fail (FileNotFound) because we never populated it.
        let local_res = local
            .read_page(PageId {
                file_id,
                offset: 0,
                length: 10,
            })
            .await;

        // If LocalDiskManager auto-creates files on write but not read,
        // and we never wrote to local, the file won't exist.
        // If it returns UnexpectedEof (empty file created?), that's also a "miss".
        // But strictly it should be NotFound.
        assert!(
            local_res.is_err(),
            "Straddling request should NOT populate local cache"
        );
    }
}
