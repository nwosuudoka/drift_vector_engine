use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use twox_hash::XxHash64;

// Define our standard high-performance hasher builder
type FastHasherBuilder = BuildHasherDefault<XxHash64>;

// Use u32 for indices to save cache space (4 bytes vs 8 bytes for usize).
// 4 billion items is enough for a cache.
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
    // Option allows us to drop the value (creating a Ghost) while keeping the Key/Entry alive.
    value: Option<Arc<V>>,
    freq: u8,

    // Intrusive Linked List Pointers
    next: Index,
    prev: Index,
    queue: QueueType,
}

/// Internal high-performance counters
#[derive(Debug, Default)]
pub struct CacheMetrics {
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    pub write_ops: AtomicU64,
    pub evictions: AtomicU64,
    pub ghost_hits: AtomicU64,
    pub rejected_writes: AtomicU64,
}

/// Public snapshot of metrics (safe to print/serialize)
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

    // We store this as a state to allow dynamic resizing.
    s_target: usize,

    // Admission policy
    // A simple bitset size = Capacity * 10 bits usually
    // We use u64 buckets
    doorkeeper: Vec<u64>,
    doorkeeper_mask: usize, // to map hash -> bit index
    write_count: usize,     // to trigger doorkeeper reset

    // THE ARENA: Contiguous memory for all nodes.
    // Cache Access: Accessing entries[i] usually prefetches entries[i+1].
    entries: Vec<Entry<K, V>>,

    // Free List: Head of the linked list of empty slots in 'entries'
    free_head: Index,

    // The Lookup: Maps Key -> Arena Index.
    // We ONLY use this for Get/Put. Never for Eviction loops.
    index_map: HashMap<K, Index, FastHasherBuilder>,

    // Queue Metadata (Head/Tail indices and sizes)
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
        // Doorkeeper size: 4x capacity bits (heuristics)
        // Ensure it is a power of 2 for fast masking
        let bits = (capacity * 4).next_power_of_two().max(64);
        let u64_count = bits / 64;

        Self {
            capacity,
            // Initialize target to 10% of capacity (Standard S3FIFO baseline)
            s_target: (capacity as f64 * 0.1).ceil() as usize,

            doorkeeper: vec![0; u64_count],
            doorkeeper_mask: bits - 1,
            write_count: 0,

            // Pre-allocate to avoid re-allocations during runtime
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

    /// Read path: O(1) Lookup + O(1) Arithmetic
    /// Read path: Returns a CLONE of the Arc (cheap), not the data.
    pub fn get(&mut self, key: &K) -> Option<Arc<V>> {
        if let Some(&idx) = self.index_map.get(key) {
            let entry = &mut self.entries[idx as usize];
            if entry.value.is_some() {
                entry.freq = std::cmp::min(3, entry.freq + 1);
                // Return a clone of the Arc smart pointer
                // TRACK HIT
                self.metrics.hits.fetch_add(1, Ordering::Relaxed);
                return entry.value.clone();
            }
        }
        // - The original code tracked hits but returned None without tracking misses.
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
        // Efficiently zero out the vector
        self.doorkeeper.fill(0);
    }

    pub fn put(&mut self, key: K, value: V) {
        // TRACK WRITE OP
        self.metrics.write_ops.fetch_add(1, Ordering::Relaxed);

        let shared_value = Arc::new(value);

        // 1. Update Existing (Always Allowed)
        if let Some(&idx) = self.index_map.get(&key) {
            let is_ghost = self.entries[idx as usize].value.is_none();
            if is_ghost {
                // TRACK GHOST HIT (Successful Resurrection)
                self.metrics.ghost_hits.fetch_add(1, Ordering::Relaxed);

                // --- GHOST RESURRECTION (Adaptive Trigger) ---

                // Signal: We accessed a Ghost.
                // Meaning: It was evicted from Small too early.
                // Action: Increase Small Target to hold items longer in probation.
                // Limit: Don't let Small swallow the whole cache (cap at 90%).

                // NOTE: Remember to reset freq/value etc.
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

        // 2. Admission Policy for NEW items
        // Only apply if cache is already under pressure (full).
        if self.len() >= self.capacity {
            // Hash the key to check doorkeeper
            // We use DefaultHasher here for simplicity, or reuse if K is Hash
            let mut s = XxHash64::default();
            key.hash(&mut s);
            let h = s.finish();

            let seen = self.check_doorkeeper(h);

            // Periodically reset doorkeeper to prevent saturation
            self.write_count += 1;
            if self.write_count >= self.capacity * 2 {
                self.reset_doorkeeper();
                self.write_count = 0;
            }

            if !seen {
                // TRACK REJECTION
                self.metrics.rejected_writes.fetch_add(1, Ordering::Relaxed);
                // REJECT!
                // We've never seen this before, and the cache is full.
                // We marked it in the doorkeeper. If it comes back, we'll admit it.
                return;
            }
        }

        // 3. Insert New (Admitted)
        while self.len() >= self.capacity {
            self.evict();
        }

        let idx = self.alloc_slot(key.clone(), Some(shared_value));
        self.index_map.insert(key, idx);
        self.attach_small(idx);
    }

    // pub fn put(&mut self, key: K, value: V) {
    //     let shared_value = Arc::new(value);
    //     if let Some(&idx) = self.index_map.get(&key) {
    //         let is_ghost = self.entries[idx as usize].value.is_none();
    //         if is_ghost {
    //             // --- GHOST RESURRECTION (Adaptive Trigger) ---

    //             // Signal: We accessed a Ghost.
    //             // Meaning: It was evicted from Small too early.
    //             // Action: Increase Small Target to hold items longer in probation.
    //             // Limit: Don't let Small swallow the whole cache (cap at 90%).
    //             let max_small = (self.capacity as f64 * 0.9) as usize;
    //             if self.s_target < max_small {
    //                 self.s_target += 1;
    //             }

    //             // 1. Detach from Ghost Queue immediately.
    //             self.detach(idx);

    //             // 2. CHECK CAPACITY
    //             while self.len() >= self.capacity {
    //                 self.evict();
    //             }

    //             // 3. Update the Data
    //             {
    //                 let entry = &mut self.entries[idx as usize];
    //                 entry.value = Some(shared_value);
    //                 entry.freq = 0;
    //             }

    //             // 4. Attach to Main Queue
    //             self.attach_main(idx);
    //         } else {
    //             // --- STANDARD UPDATE ---
    //             let entry = &mut self.entries[idx as usize];
    //             entry.value = Some(shared_value);
    //             entry.freq = std::cmp::min(3, entry.freq + 1);
    //         }
    //         return;
    //     }

    //     // 2. Insert New
    //     while self.len() >= self.capacity {
    //         self.evict();
    //     }

    //     let idx = self.alloc_slot(key.clone(), Some(shared_value));
    //     self.index_map.insert(key, idx);
    //     self.attach_small(idx);
    // }

    /// The Hot Path: Logic operates entirely on u32 indices.
    /// No Hashing. No Key Cloning.
    /// The Hot Path: Logic operates entirely on u32 indices.
    fn evict(&mut self) {
        // CHANGED: Use dynamic `self.s_target` instead of calculating 10% every time.
        if self.s_len > self.s_target {
            self.evict_from_small();
        } else {
            self.evict_from_main();
        }
    }

    fn evict_from_small(&mut self) {
        let mut evicted = false;

        // We iterate purely by following integer indices in the vector.
        // Extremely cache friendly compared to chasing pointers.
        while !evicted && self.s_len > 0 {
            let idx = self.s_head; // Head of Small Queue

            // Access frequency without a Hash Lookup
            let freq = self.entries[idx as usize].freq;

            if freq > 0 {
                // Promote to Main
                self.detach(idx);
                self.entries[idx as usize].freq = 0; // Reset
                self.attach_main(idx);
            } else {
                // Evict (Demote to Ghost)
                self.detach(idx);
                self.entries[idx as usize].value = None; // Drop Value, keep Key
                self.attach_ghost(idx);
                evicted = true;
            }
        }

        // Safety valve if Small was full of hot items
        if !evicted && self.len() >= self.capacity {
            self.evict_from_main();
        }
    }

    fn evict_from_main(&mut self) {
        // Signal: We are forced to evict from Main.
        // Meaning: Main is under pressure.
        // Action: Decrease Small Target to give Main more room.
        // Limit: Don't let Small disappear completely (min 1 slot).
        if self.s_target > 1 {
            self.s_target -= 1;
        }

        let mut evicted = false;
        while !evicted && self.m_len > 0 {
            let idx = self.m_head;
            let freq = self.entries[idx as usize].freq;

            if freq > 0 {
                // Second Chance: Move to Tail, Decr Freq
                self.detach(idx);
                self.entries[idx as usize].freq -= 1;
                self.attach_main(idx); // Back to tail
            } else {
                // Hard Evict: Remove from Cache entirely
                self.detach(idx);
                let key = &self.entries[idx as usize].key;
                self.index_map.remove(key);
                self.free_slot(idx);
                evicted = true;
            }
        }
    }

    // --- Intrusive List Helpers (Pointer Arithmetic) ---

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
        // If Ghost is full, we must free the oldest Ghost node completely
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

        // Unlink prev
        if prev != NULL {
            self.entries[prev as usize].next = next;
        } else {
            // We were head, update head pointer
            match queue {
                QueueType::Small => self.s_head = next,
                QueueType::Main => self.m_head = next,
                QueueType::Ghost => self.g_head = next,
                _ => {}
            }
        }

        // Unlink next
        if next != NULL {
            self.entries[next as usize].prev = prev;
        } else {
            // We were tail, update tail pointer
            match queue {
                QueueType::Small => self.s_tail = prev,
                QueueType::Main => self.m_tail = prev,
                QueueType::Ghost => self.g_tail = prev,
                _ => {}
            }
        }

        // Update counts
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

    // --- Arena Memory Management ---

    fn alloc_slot(&mut self, key: K, value: Option<Arc<V>>) -> Index {
        // 1. Try to reuse a slot from the free list (Fastest)
        if self.free_head != NULL {
            let idx = self.free_head;
            let next_free = self.entries[idx as usize].next;
            self.free_head = next_free;

            let e = &mut self.entries[idx as usize];
            e.key = key;
            e.value = value; // Assign Arc
            e.freq = 0;
            e.queue = QueueType::None;
            e.next = NULL;
            e.prev = NULL;
            return idx;
        }

        // 2. Safety Valve: Check if we are leaking memory.
        // In a healthy S3-FIFO, the total slots (Ghosts + Main + Small) shouldn't
        // massively exceed capacity. We allow some buffer (e.g., 2x) for heavy Ghost traffic.
        let max_allowed_slots = self.capacity * 2;

        if self.entries.len() >= max_allowed_slots {
            // CRITICAL: We are growing unboundedly. This implies a logic bug
            // where evictions are not freeing slots.
            // Option A: Panic (Good for debugging)
            // panic!("Memory Leak Detected: Cache logical len is {}, but vector size is {}", self.len(), self.entries.len());

            // Option B: Hard Reset (Good for production resilience)
            // If we hit this, we force-reclaim the oldest Ghost or just fail the insert.
            // Here, we'll try to force a hard reclaim from the Ghost tail.
            if self.g_len > 0 {
                let victim = self.g_head; // Oldest ghost
                self.detach(victim);
                self.index_map.remove(&self.entries[victim as usize].key);

                // We reuse this victim slot immediately instead of growing
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

        // 3. Grow vector (Standard path if within limits)
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
        // Clear data to drop refcounts if necessary
        self.entries[idx as usize].value = None;
        // We usually don't drop the Key here to save time, unless K owns heap memory
        // that is massive. In this impl, we overwrite key on alloc_slot.

        // Add to free list (stack style)
        self.entries[idx as usize].next = self.free_head;
        self.entries[idx as usize].queue = QueueType::None;
        self.free_head = idx;
    }

    pub fn len(&self) -> usize {
        self.s_len + self.m_len
    }
}

impl<K, V> FastS3Fifo<K, V>
where
    K: Hash + Eq + Clone,
{
    #[cfg(test)]
    pub fn get_small_target(&self) -> usize {
        self.s_target
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc; // <--- NEW: Required for Zero-Copy tests

    /// Helper to visualize what is happening in the tests
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
        // [Visual]
        // Empty Cache -> Put(1, "A") -> [Small: {1: "A"}]
        let mut cache = FastS3Fifo::new(10, Arc::new(CacheMetrics::default()));

        cache.put(1, "A".to_string());
        cache.put(2, "B".to_string());

        // Update: Use .as_deref() to compare Option<Arc<String>> with Option<&str>
        assert_eq!(cache.get(&1).as_deref(), Some(&"A".to_string()));
        assert_eq!(cache.get(&2).as_deref(), Some(&"B".to_string()));
        assert_eq!(cache.get(&3), None);
    }

    #[test]
    fn test_overwrite_existing() {
        // [Visual]
        // [Small: {1: "Old"}] -> Put(1, "New") -> [Small: {1: "New"}]
        // Frequency should increase.
        let mut cache = FastS3Fifo::new(10, Arc::new(CacheMetrics::default()));

        cache.put(1, "Old".to_string());
        assert_eq!(cache.get(&1).as_deref(), Some(&"Old".to_string()));

        cache.put(1, "New".to_string());
        assert_eq!(cache.get(&1).as_deref(), Some(&"New".to_string()));

        // Verify frequency increased (internal check)
        let idx = cache.index_map.get(&1).unwrap();
        assert!(cache.entries[*idx as usize].freq > 0);
    }

    #[test]
    fn test_len_tracking() {
        let mut cache = FastS3Fifo::new(10, Arc::new(CacheMetrics::default()));
        assert_eq!(cache.len(), 0);

        cache.put(1, 1);
        assert_eq!(cache.len(), 1);

        cache.put(2, 2);
        assert_eq!(cache.len(), 2);

        // Overwrite shouldn't change len
        cache.put(1, 99);
        assert_eq!(cache.len(), 2);

        // Fill up
        for i in 3..11 {
            cache.put(i, i);
        }
        assert_eq!(cache.len(), 10);

        // Overflow (evicts one, adds one) -> Len stays 10
        cache.put(100, 100);
        assert_eq!(cache.len(), 10);
    }

    #[test]
    fn test_admission_policy_scan_resistance() {
        // [Visual]
        // Cap = 10. Fill it.
        // Insert "Scan1" -> Doorkeeper says "New" -> Rejected.
        // Insert "Scan1" again -> Doorkeeper says "Seen" -> Admitted.

        let mut cache = FastS3Fifo::new(10, Arc::new(CacheMetrics::default()));

        // 1. Fill Cache
        for i in 0..10 {
            cache.put(i, format!("val-{}", i));
        }
        assert_eq!(cache.len(), 10);

        // 2. Insert New Item (Should be rejected)
        let scan_key = 999;
        cache.put(scan_key, "scan-val".to_string());

        // VERIFY REJECTION
        // It should NOT be in the map (completely ignored)
        assert_eq!(cache.get(&scan_key), None);
        assert_eq!(cache.len(), 10); // Length shouldn't change (no eviction happened)

        // 3. Insert Same Item Again (Should be admitted)
        cache.put(scan_key, "scan-val-2".to_string());

        // VERIFY ADMISSION
        // Now it should exist.
        // Note: checking deref because of Zero-Copy Arc change
        assert_eq!(
            cache.get(&scan_key).as_deref(),
            Some(&"scan-val-2".to_string())
        );

        // Since we admitted one, we must have evicted one.
        // Capacity is still 10 (or 11 if lazy, but logic enforces < cap before insert)
        assert!(cache.len() <= 10);
    }

    #[test]
    fn test_admission_policy_allow_when_empty() {
        // If cache is NOT full, doorkeeper should be skipped.
        // We want to fill empty RAM quickly.

        let mut cache = FastS3Fifo::new(10, Arc::new(CacheMetrics::default()));

        // Insert 1 item (Cap is 10)
        cache.put(1, "A".to_string());

        // Should be there immediately (First hit)
        assert_eq!(cache.get(&1).as_deref(), Some(&"A".to_string()));
    }

    #[test]
    fn test_ghost_resurrection_logic() {
        // [Visual: Ghost Resurrection]
        let capacity = 10;
        let mut cache = FastS3Fifo::new(capacity, Arc::new(CacheMetrics::default()));
        let key = 999;

        // 1. Insert target key
        cache.put(key, "Original".to_string());

        // 2. Flood to force eviction.
        // FIX: We must insert TWICE to bypass Doorkeeper and force entry/eviction.
        for i in 0..15 {
            if i != key {
                let k = i;
                let v = format!("flood-{}", i);
                cache.put(k, v.clone()); // 1st: Doorkeeper rejection
                cache.put(k, v); // 2nd: Admission -> Evicts older items
            }
        }

        // Verify it is a Ghost now:
        assert!(cache.index_map.contains_key(&key));
        assert_eq!(cache.get(&key), None);

        // Internal Check: Ensure it is actually in Ghost Queue
        assert_entry_state(&cache, key, QueueType::Ghost);

        // 3. Resurrect it!
        cache.put(key, "Resurrected".to_string());

        // Verify it is back with new value
        assert_eq!(cache.get(&key).as_deref(), Some(&"Resurrected".to_string()));
        assert_entry_state(&cache, key, QueueType::Main);
    }

    #[test]
    fn test_eviction_one_hit_wonder() {
        // [Visual: S3-FIFO Standard Behavior]
        // Items accessed once (freq 0) should be evicted before items accessed frequently.
        let mut cache = FastS3Fifo::new(10, Arc::new(CacheMetrics::default()));

        // 1. Insert Hot Item and access it
        cache.put(0, "hot".to_string());
        cache.get(&0);
        cache.get(&0); // Freq = 2. Should survive.

        // 2. Insert 10 Cold Items (Capacity is 10)
        // For the first few items (filling the cache), single put keeps them "Cold" (freq=0).
        for i in 1..=10 {
            let v = format!("cold-{}", i);
            if cache.len() < 10 {
                // Cache not full: Insert once (Freq=0)
                cache.put(i, v);
            } else {
                // Cache full: Doorkeeper is active.
                // 1st put: Rejected. 2nd put: Admitted (Freq=0).
                cache.put(i, v.clone());
                cache.put(i, v);
            }
        }

        // 3. Check results
        // Item 0 (Hot) should still be present (Promoted to Main).
        assert_eq!(cache.get(&0).as_deref(), Some(&"hot".to_string()));

        // Item 1 (First Cold item) should be evicted.
        // It entered Small, had 0 freq, and was pushed out by the loop.
        assert_eq!(cache.get(&1), None);
    }

    #[test]
    fn test_arena_slot_cycling_and_safety() {
        // [Visual: Memory Corruption Test]
        let capacity = 100;
        let mut cache = FastS3Fifo::new(capacity, Arc::new(CacheMetrics::default()));

        // Cycle 1: Fill 0..100
        for i in 0..100 {
            cache.put(i, i);
        }

        // Cycle 2: Overwrite with 100..200
        // FIX: Insert twice to ensure we actually overwrite/evict
        for i in 100..200 {
            cache.put(i, i); // Rejected
            cache.put(i, i); // Admitted
        }

        // Verify: Old items gone, New items present
        assert_eq!(cache.get(&0), None);
        assert_eq!(cache.get(&199), Some(Arc::new(199)));

        // Cycle 3: Random Chaos
        for i in 0..500 {
            let k = i % 150;
            cache.put(k, k * 10);
            cache.put(k, k * 10); // Ensure admission for chaos
        }

        assert!(cache.len() <= capacity);
    }

    #[test]
    fn test_small_to_main_promotion() {
        let mut cache = FastS3Fifo::new(10, Arc::new(CacheMetrics::default()));

        cache.put(1, "A".to_string()); // In Small
        cache.get(&1); // Freq > 0

        cache.put(2, "B".to_string()); // In Small

        // Trigger eviction by filling the rest.
        // FIX: Double put to bypass doorkeeper
        for i in 3..12 {
            cache.put(i, "fill".to_string());
            cache.put(i, "fill".to_string());
        }

        // 1 should be alive in Main
        assert_eq!(cache.get(&1).as_deref(), Some(&"A".to_string()));
        assert_entry_state(&cache, 1, QueueType::Main);
    }

    #[test]
    fn test_adaptive_resizing_internals() {
        let mut cache = FastS3Fifo::new(100, Arc::new(CacheMetrics::default()));

        assert_eq!(cache.get_small_target(), 10);

        // 1. Force a Ghost Hit
        let key = 1;
        cache.put(key, "A".to_string());

        // Evict it to ghost (flood small queue)
        for i in 2..102 {
            cache.put(i, "flood".to_string());
            cache.put(i, "flood".to_string()); // FIX: Double put
        }

        // Confirmed Ghost (value is None)
        assert_eq!(cache.get(&key), None);

        // 2. Resurrect -> Should increment target
        cache.put(key, "A-resurrected".to_string());

        // Target should have adapted from 10 -> 11
        assert_eq!(cache.get_small_target(), 11);
    }
}
