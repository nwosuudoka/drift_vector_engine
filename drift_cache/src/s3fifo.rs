use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use twox_hash::XxHash64;

type FastHasherBuilder = BuildHasherDefault<XxHash64>;
type Index = u32;
const NULL: Index = u32::MAX;

#[derive(Debug, Clone, Copy, PartialEq)]
enum QueueType {
    None,
    Small,
    Main,
    Ghost,
}

struct Entry<K, V> {
    key: K,
    value: Option<Arc<V>>,
    freq: u8,
    next: Index,
    prev: Index,
    queue: QueueType,
}

#[derive(Debug, Default)]
pub struct CacheMetrics {
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    pub write_ops: AtomicU64,
    pub evictions: AtomicU64,
    pub ghost_hits: AtomicU64,
    pub rejected_writes: AtomicU64,
}

#[derive(Debug, Clone, Copy)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub write_ops: u64,
    pub evictions: u64,
    pub ghost_hits: u64,
    pub rejected_writes: u64,
    pub hit_rate: f64,
}

pub struct FastS3Fifo<K, V> {
    capacity: usize,
    s_target: usize,
    doorkeeper: Vec<u64>,
    doorkeeper_mask: usize,
    write_count: usize,
    entries: Vec<Entry<K, V>>,
    free_head: Index,
    index_map: HashMap<K, Index, FastHasherBuilder>,
    s_head: Index,
    s_tail: Index,
    s_len: usize,
    m_head: Index,
    m_tail: Index,
    m_len: usize,
    g_head: Index,
    g_tail: Index,
    g_len: usize,
    metrics: Arc<CacheMetrics>,
}

impl<K, V> FastS3Fifo<K, V>
where
    K: Hash + Eq + Clone,
{
    pub fn new(capacity: usize, metrics: Arc<CacheMetrics>) -> Self {
        let bits = (capacity * 4).next_power_of_two().max(64);
        let u64_count = bits / 64;

        Self {
            capacity,
            s_target: (capacity as f64 * 0.1).ceil() as usize,
            doorkeeper: vec![0; u64_count],
            doorkeeper_mask: bits - 1,
            write_count: 0,
            entries: Vec::with_capacity(capacity + (capacity / 10)),
            free_head: NULL,
            index_map: HashMap::with_capacity_and_hasher(capacity, FastHasherBuilder::default()),
            s_head: NULL,
            s_tail: NULL,
            s_len: 0,
            m_head: NULL,
            m_tail: NULL,
            m_len: 0,
            g_head: NULL,
            g_tail: NULL,
            g_len: 0,
            metrics,
        }
    }

    pub fn get(&mut self, key: &K) -> Option<Arc<V>> {
        if let Some(&idx) = self.index_map.get(key) {
            let entry = &mut self.entries[idx as usize];
            if entry.value.is_some() {
                entry.freq = std::cmp::min(3, entry.freq + 1);
                self.metrics.hits.fetch_add(1, Ordering::Relaxed);
                return entry.value.clone();
            }
        }
        self.metrics.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    fn check_doorkeeper(&mut self, key_hash: u64) -> bool {
        let bit_idx = (key_hash as usize) & self.doorkeeper_mask;
        let block_idx = bit_idx / 64;
        let bit_offset = bit_idx % 64;
        let mask = 1u64 << bit_offset;
        let seen = (self.doorkeeper[block_idx] & mask) != 0;
        if !seen {
            self.doorkeeper[block_idx] |= mask;
        }
        seen
    }

    fn reset_doorkeeper(&mut self) {
        self.doorkeeper.fill(0);
    }

    // Accept Arc<V> directly to allow caller to retain ownership on rejection
    pub fn put(&mut self, key: K, value: Arc<V>) {
        self.metrics.write_ops.fetch_add(1, Ordering::Relaxed);
        let shared_value = value; // Already Arc

        if let Some(&idx) = self.index_map.get(&key) {
            let is_ghost = self.entries[idx as usize].value.is_none();
            if is_ghost {
                self.metrics.ghost_hits.fetch_add(1, Ordering::Relaxed);
                if self.s_target < (self.capacity as f64 * 0.9) as usize {
                    self.s_target += 1;
                }
                self.detach(idx);
                while self.len() >= self.capacity {
                    self.evict();
                }
                {
                    let entry = &mut self.entries[idx as usize];
                    entry.value = Some(shared_value);
                    entry.freq = 0;
                }
                self.attach_main(idx);
            } else {
                let entry = &mut self.entries[idx as usize];
                entry.value = Some(shared_value);
                entry.freq = std::cmp::min(3, entry.freq + 1);
            }
            return;
        }

        if self.len() >= self.capacity {
            let mut s = XxHash64::default();
            key.hash(&mut s);
            let h = s.finish();

            let seen = self.check_doorkeeper(h);
            self.write_count += 1;
            if self.write_count >= self.capacity * 2 {
                self.reset_doorkeeper();
                self.write_count = 0;
            }

            if !seen {
                self.metrics.rejected_writes.fetch_add(1, Ordering::Relaxed);
                return;
            }
        }

        while self.len() >= self.capacity {
            self.evict();
        }

        let idx = self.alloc_slot(key.clone(), Some(shared_value));
        self.index_map.insert(key, idx);
        self.attach_small(idx);
    }

    fn evict(&mut self) {
        if self.s_len > self.s_target {
            self.evict_from_small();
        } else {
            self.evict_from_main();
        }
    }

    fn evict_from_small(&mut self) {
        let mut evicted = false;
        while !evicted && self.s_len > 0 {
            let idx = self.s_head;
            let freq = self.entries[idx as usize].freq;
            if freq > 0 {
                self.detach(idx);
                self.entries[idx as usize].freq = 0;
                self.attach_main(idx);
            } else {
                self.detach(idx);
                self.entries[idx as usize].value = None;
                self.attach_ghost(idx);
                evicted = true;
            }
        }
        if !evicted && self.len() >= self.capacity {
            self.evict_from_main();
        }
    }

    fn evict_from_main(&mut self) {
        if self.s_target > 1 {
            self.s_target -= 1;
        }

        let mut evicted = false;
        while !evicted && self.m_len > 0 {
            let idx = self.m_head;
            let freq = self.entries[idx as usize].freq;

            if freq > 0 {
                self.detach(idx);
                self.entries[idx as usize].freq -= 1;
                self.attach_main(idx);
            } else {
                self.detach(idx);
                let key = &self.entries[idx as usize].key;
                self.index_map.remove(key);
                self.free_slot(idx);
                evicted = true;
            }
        }
    }

    fn attach_small(&mut self, idx: Index) {
        self.entries[idx as usize].queue = QueueType::Small;
        self.entries[idx as usize].next = NULL;
        self.entries[idx as usize].prev = self.s_tail;
        if self.s_tail != NULL {
            self.entries[self.s_tail as usize].next = idx;
        } else {
            self.s_head = idx;
        }
        self.s_tail = idx;
        self.s_len += 1;
    }

    fn attach_main(&mut self, idx: Index) {
        self.entries[idx as usize].queue = QueueType::Main;
        self.entries[idx as usize].next = NULL;
        self.entries[idx as usize].prev = self.m_tail;
        if self.m_tail != NULL {
            self.entries[self.m_tail as usize].next = idx;
        } else {
            self.m_head = idx;
        }
        self.m_tail = idx;
        self.m_len += 1;
    }

    fn attach_ghost(&mut self, idx: Index) {
        if self.g_len >= self.capacity {
            let old_ghost = self.g_head;
            self.detach(old_ghost);
            let key = &self.entries[old_ghost as usize].key;
            self.index_map.remove(key);
            self.free_slot(old_ghost);
        }

        self.entries[idx as usize].queue = QueueType::Ghost;
        self.entries[idx as usize].next = NULL;
        self.entries[idx as usize].prev = self.g_tail;
        if self.g_tail != NULL {
            self.entries[self.g_tail as usize].next = idx;
        } else {
            self.g_head = idx;
        }
        self.g_tail = idx;
        self.g_len += 1;
    }

    fn detach(&mut self, idx: Index) {
        let (prev, next, queue) = {
            let e = &self.entries[idx as usize];
            (e.prev, e.next, e.queue)
        };
        if prev != NULL {
            self.entries[prev as usize].next = next;
        } else {
            match queue {
                QueueType::Small => self.s_head = next,
                QueueType::Main => self.m_head = next,
                QueueType::Ghost => self.g_head = next,
                _ => {}
            }
        }
        if next != NULL {
            self.entries[next as usize].prev = prev;
        } else {
            match queue {
                QueueType::Small => self.s_tail = prev,
                QueueType::Main => self.m_tail = prev,
                QueueType::Ghost => self.g_tail = prev,
                _ => {}
            }
        }
        match queue {
            QueueType::Small => self.s_len -= 1,
            QueueType::Main => self.m_len -= 1,
            QueueType::Ghost => self.g_len -= 1,
            _ => {}
        }
        self.entries[idx as usize].queue = QueueType::None;
        self.entries[idx as usize].next = NULL;
        self.entries[idx as usize].prev = NULL;
    }

    fn alloc_slot(&mut self, key: K, value: Option<Arc<V>>) -> Index {
        if self.free_head != NULL {
            let idx = self.free_head;
            let next_free = self.entries[idx as usize].next;
            self.free_head = next_free;
            let e = &mut self.entries[idx as usize];
            e.key = key;
            e.value = value;
            e.freq = 0;
            e.queue = QueueType::None;
            e.next = NULL;
            e.prev = NULL;
            return idx;
        }

        let max_allowed_slots = self.capacity * 2;
        if self.entries.len() >= max_allowed_slots {
            if self.g_len > 0 {
                let victim = self.g_head;
                self.detach(victim);
                self.index_map.remove(&self.entries[victim as usize].key);
                let e = &mut self.entries[victim as usize];
                e.key = key;
                e.value = value;
                e.freq = 0;
                e.queue = QueueType::None;
                e.next = NULL;
                e.prev = NULL;
                return victim;
            }
        }

        let idx = self.entries.len() as Index;
        self.entries.push(Entry {
            key,
            value,
            freq: 0,
            next: NULL,
            prev: NULL,
            queue: QueueType::None,
        });
        idx
    }

    fn free_slot(&mut self, idx: Index) {
        self.metrics.evictions.fetch_add(1, Ordering::Relaxed);
        self.entries[idx as usize].value = None;
        self.entries[idx as usize].next = self.free_head;
        self.entries[idx as usize].queue = QueueType::None;
        self.free_head = idx;
    }

    pub fn len(&self) -> usize {
        self.s_len + self.m_len
    }

    #[cfg(test)]
    pub fn get_small_target(&self) -> usize {
        self.s_target
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn assert_entry_state<K, V>(cache: &FastS3Fifo<K, V>, key: K, expected_queue: QueueType)
    where
        K: Hash + Eq + Clone + std::fmt::Debug,
    {
        if let Some(&idx) = cache.index_map.get(&key) {
            let entry = &cache.entries[idx as usize];
            assert_eq!(
                entry.queue, expected_queue,
                "Key {:?} is in wrong queue!",
                key
            );
        } else {
            panic!("Key {:?} not found in index map", key);
        }
    }

    #[test]
    fn test_basic_put_get() {
        let mut cache = FastS3Fifo::new(10, Arc::new(CacheMetrics::default()));
        cache.put(1, Arc::new("A".to_string()));
        cache.put(2, Arc::new("B".to_string()));

        assert_eq!(cache.get(&1).as_deref(), Some(&"A".to_string()));
        assert_eq!(cache.get(&2).as_deref(), Some(&"B".to_string()));
        assert_eq!(cache.get(&3), None);
    }

    #[test]
    fn test_overwrite_existing() {
        let mut cache = FastS3Fifo::new(10, Arc::new(CacheMetrics::default()));
        cache.put(1, Arc::new("Old".to_string()));
        assert_eq!(cache.get(&1).as_deref(), Some(&"Old".to_string()));

        cache.put(1, Arc::new("New".to_string()));
        assert_eq!(cache.get(&1).as_deref(), Some(&"New".to_string()));
        let idx = cache.index_map.get(&1).unwrap();
        assert!(cache.entries[*idx as usize].freq > 0);
    }

    #[test]
    fn test_len_tracking() {
        let mut cache = FastS3Fifo::new(10, Arc::new(CacheMetrics::default()));
        cache.put(1, Arc::new(1));
        assert_eq!(cache.len(), 1);
        cache.put(2, Arc::new(2));
        assert_eq!(cache.len(), 2);
        cache.put(1, Arc::new(99));
        assert_eq!(cache.len(), 2);
        for i in 3..11 {
            cache.put(i, Arc::new(i));
        }
        assert_eq!(cache.len(), 10);
        cache.put(100, Arc::new(100));
        assert_eq!(cache.len(), 10);
    }

    #[test]
    fn test_admission_policy_scan_resistance() {
        let mut cache = FastS3Fifo::new(10, Arc::new(CacheMetrics::default()));
        for i in 0..10 {
            cache.put(i, Arc::new(format!("val-{}", i)));
        }
        assert_eq!(cache.len(), 10);
        let scan_key = 999;
        cache.put(scan_key, Arc::new("scan-val".to_string()));
        assert_eq!(cache.get(&scan_key), None);
        assert_eq!(cache.len(), 10);

        cache.put(scan_key, Arc::new("scan-val-2".to_string()));
        assert_eq!(
            cache.get(&scan_key).as_deref(),
            Some(&"scan-val-2".to_string())
        );
        assert!(cache.len() <= 10);
    }

    #[test]
    fn test_admission_policy_allow_when_empty() {
        let mut cache = FastS3Fifo::new(10, Arc::new(CacheMetrics::default()));
        cache.put(1, Arc::new("A".to_string()));
        assert_eq!(cache.get(&1).as_deref(), Some(&"A".to_string()));
    }

    #[test]
    fn test_ghost_resurrection_logic() {
        let capacity = 10;
        let mut cache = FastS3Fifo::new(capacity, Arc::new(CacheMetrics::default()));
        let key = 999;
        cache.put(key, Arc::new("Original".to_string()));

        for i in 0..15 {
            if i != key {
                let k = i;
                let v = format!("flood-{}", i);
                cache.put(k, Arc::new(v.clone()));
                cache.put(k, Arc::new(v));
            }
        }

        assert!(cache.index_map.contains_key(&key));
        assert_eq!(cache.get(&key), None);
        assert_entry_state(&cache, key, QueueType::Ghost);
        cache.put(key, Arc::new("Resurrected".to_string()));

        assert_eq!(cache.get(&key).as_deref(), Some(&"Resurrected".to_string()));
        assert_entry_state(&cache, key, QueueType::Main);
    }

    #[test]
    fn test_eviction_one_hit_wonder() {
        let mut cache = FastS3Fifo::new(10, Arc::new(CacheMetrics::default()));
        cache.put(0, Arc::new("hot".to_string()));
        cache.get(&0);
        cache.get(&0);

        for i in 1..=10 {
            let v = format!("cold-{}", i);
            if cache.len() < 10 {
                cache.put(i, Arc::new(v));
            } else {
                cache.put(i, Arc::new(v.clone()));
                cache.put(i, Arc::new(v));
            }
        }
        assert_eq!(cache.get(&0).as_deref(), Some(&"hot".to_string()));
        assert_eq!(cache.get(&1), None);
    }

    #[test]
    fn test_arena_slot_cycling_and_safety() {
        let capacity = 100;
        let mut cache = FastS3Fifo::new(capacity, Arc::new(CacheMetrics::default()));

        for i in 0..100 {
            cache.put(i, Arc::new(i));
        }

        for i in 100..200 {
            cache.put(i, Arc::new(i));
            cache.put(i, Arc::new(i));
        }

        assert_eq!(cache.get(&0), None);
        assert_eq!(cache.get(&199), Some(Arc::new(199)));

        for i in 0..500 {
            let k = i % 150;
            cache.put(k, Arc::new(k * 10));
            cache.put(k, Arc::new(k * 10));
        }
        assert!(cache.len() <= capacity);
    }

    #[test]
    fn test_small_to_main_promotion() {
        let mut cache = FastS3Fifo::new(10, Arc::new(CacheMetrics::default()));
        cache.put(1, Arc::new("A".to_string()));
        cache.get(&1);
        cache.put(2, Arc::new("B".to_string()));

        for i in 3..12 {
            cache.put(i, Arc::new("fill".to_string()));
            cache.put(i, Arc::new("fill".to_string()));
        }

        assert_eq!(cache.get(&1).as_deref(), Some(&"A".to_string()));
        assert_entry_state(&cache, 1, QueueType::Main);
    }

    #[test]
    fn test_adaptive_resizing_internals() {
        let mut cache = FastS3Fifo::new(100, Arc::new(CacheMetrics::default()));
        let key = 1;
        cache.put(key, Arc::new("A".to_string()));

        for i in 2..102 {
            cache.put(i, Arc::new("flood".to_string()));
            cache.put(i, Arc::new("flood".to_string()));
        }

        assert_eq!(cache.get(&key), None);
        cache.put(key, Arc::new("A-resurrected".to_string()));
        assert_eq!(cache.get_small_target(), 11);
    }
}
