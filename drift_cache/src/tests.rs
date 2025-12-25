#[cfg(test)]
mod storage_tests {
    use crate::local_store::LocalDiskManager;
    use crate::store::{PageId, PageManager};
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_local_disk_manager_lifecycle() {
        let dir = tempdir().unwrap();
        let manager = LocalDiskManager::new(dir.path());

        let file_id = 1;
        let file_path = dir.path().join("segment_1.bin");

        // 1. Register
        manager.register_file(file_id, file_path.clone());

        // 2. Write Data (Offset 100)
        let data = b"Hello, NVMe!";
        manager.write_page(file_id, 100, data).await.unwrap();

        // 3. Read Data (Exact)
        let page = PageId {
            file_id,
            offset: 100,
            length: data.len() as u32,
        };
        let read_back = manager.read_page(page).await.unwrap();
        assert_eq!(read_back, data);

        // 4. Read Data (Overlap/Different Offset)
        // Write "World" at 107 ("Hello, " is 7 chars)
        manager.write_page(file_id, 107, b"World").await.unwrap();

        let full_page = PageId {
            file_id,
            offset: 100,
            length: 12, // "Hello, World"
        };
        let full_read = manager.read_page(full_page).await.unwrap();
        assert_eq!(full_read, b"Hello, World");
    }
}
