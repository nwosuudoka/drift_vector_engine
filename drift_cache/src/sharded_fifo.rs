use std::hash::Hash;
use std::hash::Hasher;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use twox_hash::XxHash64;

use crate::s3fifo::{CacheMetrics, CacheStats, FastS3Fifo};

pub struct ShardedFastS3Fifo<K, V> {
    shards: Vec<Mutex<FastS3Fifo<K, V>>>,
    shard_mask: usize,
    metrics: Arc<CacheMetrics>,
}

impl<K, V> ShardedFastS3Fifo<K, V>
where
    K: Hash + Eq + Clone + Send + 'static,
    V: Send + 'static,
{
    pub fn new(total_capacity: usize, concurrency: usize) -> Self {
        let num_shards = if concurrency.is_power_of_two() {
            concurrency
        } else {
            concurrency.next_power_of_two()
        };
        let capacity_per_shard = (total_capacity + num_shards - 1) / num_shards;
        let metrics = Arc::new(CacheMetrics::default());

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

    fn get_shard_index(&self, key: &K) -> usize {
        let mut s = XxHash64::default();
        key.hash(&mut s);
        (s.finish() as usize) & self.shard_mask
    }

    // Accept Arc<V>
    pub fn put(&self, key: K, value: Arc<V>) {
        let idx = self.get_shard_index(&key);
        let mut shard = self.shards[idx].lock().unwrap();
        shard.put(key, value);
    }

    pub fn get(&self, key: &K) -> Option<Arc<V>> {
        let idx = self.get_shard_index(key);
        let mut shard = self.shards[idx].lock().unwrap();
        shard.get(key)
    }

    pub fn len(&self) -> usize {
        self.shards.iter().map(|s| s.lock().unwrap().len()).sum()
    }

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
    use std::thread;

    #[test]
    fn test_high_concurrency_hammer() {
        let cache = Arc::new(ShardedFastS3Fifo::<u32, String>::new(1000, 16));
        let num_threads = 50;
        let ops_per_thread = 2000;
        let barrier = Arc::new(Barrier::new(num_threads));
        let mut handles = vec![];

        for t_id in 0..num_threads {
            let c = cache.clone();
            let b = barrier.clone();
            handles.push(thread::spawn(move || {
                b.wait();
                for i in 0..ops_per_thread {
                    let key = (t_id * 1000 + i) as u32;
                    c.put(key, Arc::new(format!("val-{}", key)));
                    let _ = c.get(&key);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert!(cache.len() <= 1000 + 16);
    }

    #[test]
    fn test_data_consistency_hot_keys() {
        let cache = Arc::new(ShardedFastS3Fifo::<u32, usize>::new(100, 4));
        for i in 0..10 {
            cache.put(i, Arc::new(0));
        }

        let num_threads = 20;
        let barrier = Arc::new(Barrier::new(num_threads));
        let mut handles = vec![];

        for _ in 0..num_threads {
            let c = cache.clone();
            let b = barrier.clone();
            handles.push(thread::spawn(move || {
                b.wait();
                for _ in 0..100 {
                    c.put(5, Arc::new(999));
                    let val = c.get(&5);
                    assert!(val.is_some());
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(cache.get(&5).as_deref(), Some(&999_usize));
    }

    #[test]
    fn test_sharded_ghost_resurrection() {
        let cache = Arc::new(ShardedFastS3Fifo::<String, String>::new(10, 2));
        cache.put("A".to_string(), Arc::new("val_a".to_string()));
        for i in 0..20 {
            cache.put(format!("flood_{}", i), Arc::new("flood".to_string()));
        }

        cache.put("A".to_string(), Arc::new("resurrected".to_string()));
        assert_eq!(
            cache.get(&"A".to_string()).as_deref(),
            Some(&"resurrected".to_string())
        );
    }

    #[test]
    fn test_memory_stability() {
        let capacity = 2000;
        let cache = Arc::new(ShardedFastS3Fifo::<u32, u32>::new(capacity, 8));

        let num_threads = 8;
        let ops = 50_000;
        let mut handles = vec![];
        for t in 0..num_threads {
            let c = cache.clone();
            handles.push(thread::spawn(move || {
                for i in 0..ops {
                    let k = (t * ops + i) as u32;
                    c.put(k, Arc::new(k));
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let final_len = cache.len();
        assert!(final_len <= capacity + 100);
    }
}
