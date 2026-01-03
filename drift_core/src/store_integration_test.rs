#[cfg(test)]
mod tests {
    use crate::aligned::AlignedBytes;
    use crate::bucket::Bucket;
    use crate::quantizer::Quantizer;
    use async_trait::async_trait;
    use bit_set::BitSet;
    use drift_traits::{PageId, PageManager};
    use std::path::PathBuf;
    use std::sync::Mutex;

    // --- Mock Storage (In-Memory PageManager) ---
    struct MockStorage {
        data: Mutex<std::collections::HashMap<u32, Vec<u8>>>,
    }

    #[async_trait]
    impl PageManager for MockStorage {
        fn register_file(&self, _id: u32, _path: PathBuf) {}
        async fn read_page(&self, id: PageId) -> std::io::Result<Vec<u8>> {
            let map = self.data.lock().unwrap();
            match map.get(&id.file_id) {
                Some(bytes) => Ok(bytes.clone()), // In real life we'd read offset/len
                None => Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "Page not found",
                )),
            }
        }
        async fn write_page(&self, id: u32, _off: u64, data: &[u8]) -> std::io::Result<()> {
            let mut map = self.data.lock().unwrap();
            map.insert(id, data.to_vec());
            Ok(())
        }
    }
}
