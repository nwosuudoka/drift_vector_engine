#[cfg(test)]
mod tests {
    use crate::aligned::AlignedBytes;
    use crate::bucket::Bucket; // for scan_static
    use crate::bucket::{BucketData, BucketHeader};
    use crate::quantizer::Quantizer;
    use async_trait::async_trait;
    use bit_set::BitSet;
    use drift_cache::block_cache::BlockCache;
    use drift_traits::{PageId, PageManager};
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};

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

    #[tokio::test]
    async fn test_quantization_storage_roundtrip() {
        let dim = 8;
        let count = 100;

        // 1. Generate Training Data
        let mut samples = Vec::new();
        for i in 0..count {
            let val = i as f32;
            samples.push(vec![val; dim]); // Vectors like [0,0..], [1,1..]
        }

        // 2. Train Quantizer
        let quantizer = Quantizer::train(&samples);
        let q_arc = Arc::new(quantizer.clone());

        // 3. Create BucketData
        let mut bucket_data = BucketData {
            codes: AlignedBytes::new(count * dim),
            vids: Vec::with_capacity(count),
            tombstones: BitSet::with_capacity(count),
        };

        for (i, vec) in samples.iter().enumerate() {
            let code = quantizer.encode(vec);
            bucket_data.vids.push(i as u64);
            for b in code {
                bucket_data.codes.push(b);
            }
        }

        // 4. Setup Cache
        let storage = Arc::new(MockStorage {
            data: Mutex::new(std::collections::HashMap::new()),
        });
        let cache = BlockCache::<BucketData>::new(storage.clone(), 10, 1);

        // 5. Write to Disk (Mock) via Storage directly (simulating Index logic)
        let file_id = 99;
        let bytes = bucket_data.to_bytes(dim).unwrap();
        storage.write_page(file_id, 0, &bytes).await.unwrap();

        let page_id = PageId {
            file_id,
            offset: 0,
            length: bytes.len() as u32,
        };

        // 6. Read back via Cache (Async)
        let loaded_arc = cache.get(&page_id).await.expect("Cache load failed");

        // 7. Verify Data Integrity
        assert_eq!(loaded_arc.vids.len(), count);
        assert_eq!(loaded_arc.codes.len(), count * dim);

        // 8. Verify Reconstruction
        let (rec_vecs, rec_ids) = loaded_arc.reconstruct(&quantizer);
        assert_eq!(rec_vecs.len(), count);

        // Check accuracy (SQ8 is lossy, but [1.0, 1.0...] should be close)
        let original = &samples[50]; // [50.0, 50.0...]
        let reconstructed = &rec_vecs[50];

        let diff: f32 = original
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        println!("Reconstruction Error for vector 50: {}", diff);
        assert!(diff < 1.0, "Quantization error too high!");

        // 9. Verify ADC Scan
        let query = vec![50.2; dim]; // Close to vector 50
        let results = Bucket::scan_static(&loaded_arc, &quantizer, &query);

        // Vector 50 should be very close (distance ~0)
        // Sort results by distance
        let mut sorted = results;
        sorted.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

        assert_eq!(sorted[0].id, 50);
        println!(
            "Nearest Neighbor: ID={} Dist={}",
            sorted[0].id, sorted[0].distance
        );
    }
}
