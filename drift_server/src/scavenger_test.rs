#[cfg(test)]
mod tests {
    use crate::janitor::Janitor;
    use crate::persistence::PersistenceManager;
    use drift_cache::local_store::LocalDiskManager;
    use drift_core::index::{IndexOptions, VectorIndex};
    use opendal::{Operator, services};
    use std::sync::Arc;
    use std::time::Duration;
    use tempfile::tempdir;

    // --- Helpers ---
    fn create_local_operator(path: &std::path::Path) -> Operator {
        let mut builder = services::Fs::default();
        builder = builder.root(path.to_str().unwrap());
        Operator::new(builder).unwrap().finish()
    }

    async fn setup_index(dir: &std::path::Path) -> Arc<VectorIndex> {
        let wal_path = dir.join("test.wal");
        let storage_path = dir.join("storage");
        std::fs::create_dir_all(&storage_path).unwrap();

        let storage = Arc::new(LocalDiskManager::new(storage_path));
        let options = IndexOptions {
            dim: 2,
            num_centroids: 1,
            training_sample_size: 10,
            max_bucket_capacity: 100,
            ef_construction: 10,
            ef_search: 10,
        };

        let index = Arc::new(VectorIndex::new(options, &wal_path, storage).unwrap());
        let train_data = vec![vec![0.0, 0.0]; 10];
        index.train(&train_data).await.unwrap();
        index
    }

    // ⚡ HELPER: Flushes memtable to L1 and returns the created bucket ID
    async fn flush_to_bucket(index: &Arc<VectorIndex>) -> u32 {
        // 1. Rotate
        let memtable = index
            .rotate_and_freeze()
            .unwrap()
            .expect("Failed to rotate");

        if memtable.len() == 0 {
            panic!("flush_to_bucket called with empty memtable");
        }

        // 2. Partition (Sync)
        let partitions = index.partition_memtable(&memtable).unwrap();

        // For testing scavenger, we don't strictly need to write to disk if we use force_register,
        // BUT force_register requires raw vectors.
        // Let's extract them from the partitions.

        // Note: PartitionResult now contains INDICES. We need to look up vectors.
        // But force_register expects vectors.
        // We can cheat and read from memtable using the indices.

        let p = &partitions[0]; // Assuming 1 bucket for this test
        let (_ids_guard, data_guard, _) = memtable.get_data_guards();

        let mut vecs = Vec::new();
        for &idx in &p.indices {
            let start = idx * 2; // dim 2
            vecs.push(data_guard[start..start + 2].to_vec());
        }

        // Register
        let bucket_id = p.bucket_id; // Use the ID allocated by partitioner
        index
            .force_register_bucket_with_ids(bucket_id, &p.ids, &vecs)
            .await
            .unwrap();

        // Clean up frozen slot
        index.confirm_flush().unwrap();

        bucket_id
    }

    #[tokio::test]
    async fn test_scavenger_liberates_memory() {
        let dir = tempdir().unwrap();
        let index = setup_index(dir.path()).await;

        // 1. Fill Bucket with 50 items
        for i in 0..50 {
            index.insert(i, &vec![1.0, 1.0]).unwrap();
        }

        // ⚡ FIX: Capture the specific Bucket ID containing our data
        let target_bucket = flush_to_bucket(&index).await;

        // 2. Delete 20 items (40%)
        for i in 0..20 {
            index.delete(i).unwrap();
        }

        assert_eq!(
            index.deleted_ids.read().len(),
            20,
            "Memory should hold deletes before scavenge"
        );

        // 3. Verify Stats
        let headers = index.get_all_bucket_headers();
        let header = headers
            .iter()
            .find(|h| h.id == target_bucket)
            .expect("Target bucket not found");

        let dead = header
            .stats
            .tombstone_count
            .load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(dead, 20, "Bucket stats should reflect deletions");

        // 4. EXECUTE COMPACTION
        let purged_ids = index.compact_bucket(target_bucket).await.unwrap().unwrap();
        assert_eq!(
            purged_ids.len(),
            20,
            "Scavenger should have found all 20 deleted items"
        );

        // 5. Purge Memory
        {
            let mut guard = index.deleted_ids.write();
            for id in purged_ids {
                guard.remove(&id);
            }
        }

        assert_eq!(
            index.deleted_ids.read().len(),
            0,
            "Scavenger failed to free RAM!"
        );
    }

    #[tokio::test]
    async fn test_scavenger_updates_kv_pointers() {
        let dir = tempdir().unwrap();
        let index = setup_index(dir.path()).await;

        // 1. Insert ID 100 AND Trash together
        index.insert(100, &vec![0.5, 0.5]).unwrap();
        for i in 200..210 {
            index.insert(i, &vec![0.5, 0.5]).unwrap();
        }

        // ⚡ FIX: Capture ID
        let old_bucket_id = flush_to_bucket(&index).await;

        // Verify KV points to this bucket
        let id_bytes = 100u64.to_le_bytes();
        let bucket_bytes = index.kv.get(&id_bytes).unwrap().unwrap();
        let stored_bucket_id = u32::from_le_bytes(bucket_bytes[..4].try_into().unwrap());
        assert_eq!(stored_bucket_id, old_bucket_id);

        // 2. Delete the trash
        for i in 200..210 {
            index.delete(i).unwrap();
        }

        // 3. Compact
        index.compact_bucket(old_bucket_id).await.unwrap().unwrap();

        // 4. Verify KV Update
        let new_bucket_bytes = index.kv.get(&id_bytes).unwrap().unwrap();
        let new_bucket_id = u32::from_le_bytes(new_bucket_bytes[..4].try_into().unwrap());

        assert_ne!(new_bucket_id, old_bucket_id, "KV store was NOT updated!");

        let headers = index.get_all_bucket_headers();
        assert!(
            headers.iter().any(|h| h.id == new_bucket_id),
            "New bucket ID missing from Index!"
        );
    }

    #[tokio::test]
    async fn test_scavenger_ignores_clean_buckets() {
        let dir = tempdir().unwrap();
        let index = setup_index(dir.path()).await;

        // ⚡ CHANGE: Create Operator and inject into PersistenceManager
        let op = create_local_operator(dir.path());
        let persistence = PersistenceManager::new(op, dir.path());

        // 1. Insert 10 items
        for i in 0..10 {
            index.insert(i, &vec![0.0, 0.0]).unwrap();
        }

        // ⚡ FIX: Capture ID
        let target_bucket_id = flush_to_bucket(&index).await;

        // 2. Delete only 1 item (10% - below 20% threshold)
        index.delete(0).unwrap();

        // 3. Run Janitor Logic
        let janitor = Janitor::new(
            index.clone(),
            persistence,
            100,
            Duration::from_secs(1),
            None,
        );

        // Should ignore clean buckets
        janitor.scavenge().await.unwrap();

        // 4. Verify the bucket ID is UNCHANGED
        let headers = index.get_all_bucket_headers();
        let bucket = headers
            .iter()
            .find(|h| h.id == target_bucket_id)
            .expect("Target bucket vanished!");

        assert_eq!(
            bucket.id, target_bucket_id,
            "Janitor aggressively compacted a clean bucket!"
        );
    }
}
