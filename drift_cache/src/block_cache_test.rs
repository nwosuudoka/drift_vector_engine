#[cfg(test)]
mod tests {
    use crate::block_cache::BlockCache;
    use async_trait::async_trait;
    use drift_traits::{Cacheable, PageId, PageManager};
    use std::io::Result;
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};

    // --- Mocks ---
    struct MockStorage {
        reads: Mutex<usize>,
    }

    #[async_trait]
    impl PageManager for MockStorage {
        fn register_file(&self, _id: u32, _path: PathBuf) {}
        async fn read_page(&self, _id: PageId) -> Result<Vec<u8>> {
            *self.reads.lock().unwrap() += 1;
            // Return "serialized" data (just a byte string)
            Ok(b"Hello Cache".to_vec())
        }
        async fn write_page(&self, _id: u32, _off: u64, _d: &[u8]) -> Result<()> {
            Ok(())
        }

        async fn len(&self, _file_id: u32) -> Result<u64> {
            let k = b"Hello Cache".len() as u64;
            Ok(k)
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    struct MyData {
        content: String,
    }

    impl Cacheable for MyData {
        fn from_bytes(data: &[u8]) -> Result<Self> {
            let s = String::from_utf8(data.to_vec()).unwrap();
            Ok(MyData { content: s })
        }
    }

    #[tokio::test]
    async fn test_block_cache_hit_miss() {
        let storage = Arc::new(MockStorage {
            reads: Mutex::new(0),
        });
        let cache = BlockCache::<MyData>::new(storage.clone(), 10, 1);
        let page = PageId {
            file_id: 1,
            offset: 0,
            length: 10,
        };

        // 1. First Access (Miss -> Disk)
        let val1 = cache.get_optimized(&page).await.unwrap();
        assert_eq!(val1.content, "Hello Cache");
        assert_eq!(*storage.reads.lock().unwrap(), 1);

        // 2. Second Access (Hit -> RAM)
        let val2 = cache.get_optimized(&page).await.unwrap();
        assert_eq!(val2.content, "Hello Cache");

        // Reads should STILL be 1 (Cache Hit)
        assert_eq!(*storage.reads.lock().unwrap(), 1);
    }
}
