use std::hash::Hash;
use std::hash::Hasher;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use twox_hash::XxHash64;

use crate::s3fifo::{CacheMetrics, CacheStats, FastS3Fifo};

/// A thread-safe wrapper that shards the cache into `num_shards` independent segments.
/// This drastically reduces lock contention.
pub struct ShardedFastS3Fifo<K, V> {
    shards: Vec<Mutex<FastS3Fifo<K, V>>>,
    shard_mask: usize, // Used for fast modulo (num_shards - 1)
    metrics: Arc<CacheMetrics>,
}

impl<K, V> ShardedFastS3Fifo<K, V>
where
    K: Hash + Eq + Clone + Send + 'static,
    V: Send + 'static, // V needs Clone to return values safely out of the lock
{
    pub fn new(total_capacity: usize, concurrency: usize) -> Self {
        // 1. Round concurrency up to nearest power of 2 for fast bitwise masking
        let num_shards = if concurrency.is_power_of_two() {
            concurrency
        } else {
            concurrency.next_power_of_two()
        };

        // 2. Distribute capacity evenly
        let capacity_per_shard = (total_capacity + num_shards - 1) / num_shards;

        let metrics = Arc::new(CacheMetrics::default());

        // 3. Create Shards
        let mut shards = Vec::with_capacity(num_shards);
        for _ in 0..num_shards {
            shards.push(Mutex::new(FastS3Fifo::new(
                capacity_per_shard,
                metrics.clone(),
            )));
        }

        Self {
            shards,
            shard_mask: num_shards - 1,
            metrics,
        }
    }

    /// Helper to find which shard a key belongs to
    fn get_shard_index(&self, key: &K) -> usize {
        let mut s = XxHash64::default();
        key.hash(&mut s);
        (s.finish() as usize) & self.shard_mask
    }

    pub fn put(&self, key: K, value: V) {
        let idx = self.get_shard_index(&key);
        // We only lock ONE shard. Other threads can access other shards freely.
        let mut shard = self.shards[idx].lock().unwrap();
        shard.put(key, value);
    }

    /// Returns a Clone of the value.
    /// Why Clone? Because we cannot return a reference `&V` once the Mutex lock is dropped.
    pub fn get(&self, key: &K) -> Option<Arc<V>> {
        let idx = self.get_shard_index(key);
        let mut shard = self.shards[idx].lock().unwrap();

        // We use the inner logic, but we must clone the result to return it
        shard.get(key)
    }

    /// Optional: Peek without promoting (if your base impl supports it)
    /// or simple length check.
    pub fn len(&self) -> usize {
        // Summing length requires iterating all shards (approximate in concurrent settings)
        self.shards.iter().map(|s| s.lock().unwrap().len()).sum()
    }

    /// Export a snapshot of the current stats
    pub fn stats(&self) -> CacheStats {
        let hits = self.metrics.hits.load(Ordering::Relaxed);
        let misses = self.metrics.misses.load(Ordering::Relaxed);
        let write_ops = self.metrics.write_ops.load(Ordering::Relaxed);
        let evictions = self.metrics.evictions.load(Ordering::Relaxed);
        let ghost_hits = self.metrics.ghost_hits.load(Ordering::Relaxed);
        let rejected_writes = self.metrics.rejected_writes.load(Ordering::Relaxed);

        let total_ops = hits + misses;
        let hit_rate = if total_ops > 0 {
            hits as f64 / total_ops as f64
        } else {
            0.0
        };

        CacheStats {
            hits,
            misses,
            write_ops,
            evictions,
            ghost_hits,
            rejected_writes,
            hit_rate,
        }
    }
}

#[cfg(test)]
mod concurrent_tests {
    use super::ShardedFastS3Fifo;
    use std::sync::{Arc, Barrier};
    use std::thread; // Add `rand` to dev-dependencies in Cargo.toml if needed, or use simple math

    /// Test 1: High Concurrency Throughput
    /// Ensures 50 threads can hammer the cache without deadlocking or panicking.
    #[test]
    fn test_high_concurrency_hammer() {
        let cache = Arc::new(ShardedFastS3Fifo::<u32, String>::new(1000, 16));
        let num_threads = 50;
        let ops_per_thread = 2000;

        // Barrier ensures all threads start slamming the cache at the exact same moment
        let barrier = Arc::new(Barrier::new(num_threads));
        let mut handles = vec![];

        for t_id in 0..num_threads {
            let c = cache.clone();
            let b = barrier.clone();

            handles.push(thread::spawn(move || {
                b.wait(); // Wait for everyone

                for i in 0..ops_per_thread {
                    let key = (t_id * 1000 + i) as u32; // Unique keys per thread

                    // Mix of Puts and Gets
                    c.put(key, format!("val-{}", key));
                    let _ = c.get(&key);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        // Check internal consistency
        // Note: len() might be slightly less than capacity if eviction is lazy,
        // but it should definitely not exceed capacity significantly.
        println!("Final Cache Size: {}", cache.len());
        assert!(cache.len() <= 1000 + 16); // Allow small margin for shard-local variances
    }

    /// Test 2: The "Bank Account" Consistency Test
    /// This detects "Lost Updates".
    /// We have a fixed set of Keys. Threads repeatedly read a value and overwrite it.
    /// In a naive cache, this is race-condition city. In a Sharded cache,
    /// the locks should protect individual keys (atomic access per shard).
    #[test]
    fn test_data_consistency_hot_keys() {
        // Small cache, High contention on few keys
        let cache = Arc::new(ShardedFastS3Fifo::<u32, usize>::new(100, 4));

        // Pre-fill keys 0..10 with value 0
        for i in 0..10 {
            cache.put(i, 0);
        }

        let num_threads = 20;
        let barrier = Arc::new(Barrier::new(num_threads));
        let mut handles = vec![];

        for _ in 0..num_threads {
            let c = cache.clone();
            let b = barrier.clone();

            handles.push(thread::spawn(move || {
                b.wait();
                // Everyone fights over Key #5
                for _ in 0..100 {
                    // NOTE: This test proves thread-safety of the Internal Mutex,
                    // NOT atomic read-modify-write of the value itself.
                    // Since `put` overwrites, the last write wins.
                    // We just want to ensure no internal panics or corrupt state occurs.
                    c.put(5, 999);
                    let val = c.get(&5);
                    assert!(val.is_some());
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        // After chaos, the key should still be readable
        assert_eq!(cache.get(&5).as_deref(), Some(&999_usize));
    }

    /// Test 3: Ghost Resurrection under Sharding
    /// Ensures that the complex "Ghost -> Main" logic works when wrapped in Sharded Mutexes.
    #[test]
    fn test_sharded_ghost_resurrection() {
        let cache = Arc::new(ShardedFastS3Fifo::<String, String>::new(10, 2));

        // 1. Insert "A" and flood it out to Ghost
        cache.put("A".to_string(), "val_a".to_string());

        for i in 0..20 {
            cache.put(format!("flood_{}", i), "flood".to_string());
        }

        // "A" should be a Ghost now (None) or Evicted completely.
        // If it's a Ghost, putting it again should resurrect it.

        // 2. Resurrect "A"
        cache.put("A".to_string(), "resurrected".to_string());

        // 3. Verify
        assert_eq!(
            cache.get(&"A".to_string()).as_deref(),
            Some(&"resurrected".to_string())
        );
    }

    /// Test 4: Heavy Load & Capacity Enforcement
    /// Pushes millions of items to ensure memory stays flat.
    #[test]
    fn test_memory_stability() {
        let capacity = 2000;
        let cache = Arc::new(ShardedFastS3Fifo::<u32, u32>::new(capacity, 8));

        let num_threads = 8;
        let ops = 50_000; // 400,000 total items

        let mut handles = vec![];
        for t in 0..num_threads {
            let c = cache.clone();
            handles.push(thread::spawn(move || {
                for i in 0..ops {
                    let k = (t * ops + i) as u32;
                    c.put(k, k);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        // The cache should have capped at roughly capacity
        let final_len = cache.len();
        println!("Target Capacity: {}, Final Len: {}", capacity, final_len);

        // Allow slight overshoot due to sharding division math
        assert!(final_len <= capacity + 100);
    }

    // ===== ./tests.rs =====

    #[test]
    fn test_zero_copy_semantics() {
        // 1. Define a Non-Cloneable struct (simulating a unique DB resource)
        #[derive(Debug, PartialEq)]
        struct HeavyPage {
            data: [u8; 1024],
            id: u32,
        }
        // No Clone impl!

        let cache = Arc::new(ShardedFastS3Fifo::<u32, HeavyPage>::new(10, 4));

        let page = HeavyPage {
            data: [0; 1024],
            id: 1,
        };

        // 2. Put it in. (Takes ownership)
        cache.put(1, page);

        // 3. Get it out.
        let retrieved_1 = cache.get(&1).expect("Should exist");
        let retrieved_2 = cache.get(&1).expect("Should exist");

        // 4. Verify Identity
        // The pointers should point to the EXACT same memory address.
        let ptr1 = Arc::as_ptr(&retrieved_1);
        let ptr2 = Arc::as_ptr(&retrieved_2);

        assert_eq!(ptr1, ptr2, "Pointers should match (Zero Copy)");

        // 5. Verify Data
        assert_eq!(retrieved_1.id, 1);
    }

    #[test]
    fn test_concurrency_no_clone_deadlock() {
        // Ensure that holding an Arc outside the lock doesn't cause issues
        let cache = Arc::new(ShardedFastS3Fifo::<String, String>::new(100, 4));

        cache.put("key".to_string(), "long_value".to_string());

        let val_ref = cache.get(&"key".to_string()).unwrap();

        // We are holding a reference to the data here...
        // While a writer tries to update it.

        let c2 = cache.clone();
        let handler = std::thread::spawn(move || {
            // This will replace the Arc inside the cache.
            // But 'val_ref' in the main thread should still point to the OLD data safely.
            c2.put("key".to_string(), "new_value".to_string());
        });

        handler.join().unwrap();

        // Main thread still sees old value (Snapshot Isolation effectively)
        assert_eq!(*val_ref, "long_value");

        // New get sees new value
        assert_eq!(*cache.get(&"key".to_string()).unwrap(), "new_value");
    }

    // ===== ./tests.rs =====

    #[test]
    fn test_telemetry_counters() {
        // 1. Setup
        let cache = Arc::new(ShardedFastS3Fifo::<u32, String>::new(10, 2));

        // 2. Generate Hits and Misses
        cache.put(1, "A".to_string()); // Write +1

        cache.get(&1); // Hit +1
        cache.get(&1); // Hit +2
        cache.get(&99); // Miss +1

        // 3. Check Snapshot
        let stats = cache.stats();

        assert_eq!(stats.write_ops, 1);
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
        assert!(stats.hit_rate > 0.6 && stats.hit_rate < 0.7); // 2/3 = 0.66
    }

    #[test]
    fn test_telemetry_rejection_tracking() {
        // Test that Doorkeeper rejections show up in stats
        let cache = Arc::new(ShardedFastS3Fifo::<u32, u32>::new(10, 1));

        // Fill cache
        for i in 0..10 {
            cache.put(i, i);
        }

        // New item -> Rejected by Doorkeeper
        cache.put(999, 999);

        let stats = cache.stats();
        assert_eq!(stats.rejected_writes, 1);
    }

    #[test]
    fn test_xxhash_integration() {
        // ERROR WAS HERE: You need both Hash (the trait for the key)
        // and Hasher (the trait for the algorithm)
        use std::hash::{Hash, Hasher};
        use twox_hash::XxHash64;

        // 1. Verify we can hash a key manually using the same algo
        let key = 12345_u32;
        let mut s = XxHash64::default();

        // Now this works because the 'Hash' trait is in scope
        key.hash(&mut s);

        let h1 = s.finish();

        // 2. Verify determinism (Same key = Same hash)
        let mut s2 = XxHash64::default();
        key.hash(&mut s2);
        let h2 = s2.finish();

        assert_eq!(h1, h2);
        assert!(h1 != 0);
    }

    #[test]
    fn test_sharding_distribution() {
        // Indirectly tests that xxHash provides decent distribution for sharding
        let cache = Arc::new(ShardedFastS3Fifo::<u32, u32>::new(1000, 4));

        for i in 0..1000 {
            cache.put(i, i);
        }

        // Check internal shard lengths (using our new telemetry!)
        // They should be roughly balanced (250 items each).
        // Since we can't access shards directly, we rely on the fact that
        // if distribution was broken (e.g. everything went to Shard 0),
        // we would see massive evictions despite having total capacity.

        let stats = cache.stats();
        // If sharding was broken (all to one), we'd have 750 evictions (1000 items into 250 cap shard).
        // If sharding works, evictions should be low/zero.
        assert!(
            stats.evictions < 50,
            "Hash distribution is poor; too many evictions!"
        );
    }
}
