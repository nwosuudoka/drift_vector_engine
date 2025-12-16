use crate::aligned::AlignedBytes;
use atomic_float::AtomicF32;
use bit_set::BitSet;
use parking_lot::RwLock;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// The structure of a "Drift-Aware" Bucket.
/// optimized for high-velocity churn.
pub struct Bucket {
    pub id: u32,

    // --- HOT PATH (Read-Heavy) ---
    // Protected by RwLock.
    // Searchers grab Read lock (Nano-second latency).
    // Splitters grab Write lock.
    pub data: RwLock<BucketData>,

    // --- METRICS (Lock-Free) ---
    // Searchers read these to calculate "Urgency" without locking.
    pub centroid: RwLock<Vec<f32>>, // Centroid changes rarely (only on split/merge)
    pub count: AtomicU32,
    pub tombstone_count: AtomicU32,
    pub temperature: AtomicF32,

    // Hysteresis: Timestamp of last modification
    pub last_maintenance: AtomicU64,
}

pub struct BucketData {
    // 64-byte aligned codes for SIMD
    pub codes: AlignedBytes,

    // Parallel array of IDs (u64)
    pub vids: Vec<u64>,

    // Tombstones (Bitmask is faster than Vec<bool>)
    pub tombstones: BitSet,
}

impl Bucket {
    pub fn new(id: u32, capacity: usize, dim: usize) -> Self {
        Self {
            id,
            data: RwLock::new(BucketData {
                // Reserve space: Capacity * Dim bytes
                codes: AlignedBytes::new(capacity * dim),
                vids: Vec::with_capacity(capacity),
                tombstones: BitSet::with_capacity(capacity),
            }),
            centroid: RwLock::new(vec![0.0; dim]),
            count: AtomicU32::new(0),
            tombstone_count: AtomicU32::new(0),
            temperature: AtomicF32::new(1.0),
            last_maintenance: AtomicU64::new(0),
        }
    }

    // The "Heat" Update
    pub fn touch(&self) {
        // Simple cooling/heating logic
        // In prod, this would use a decay function
        self.temperature.fetch_add(0.1, Ordering::Relaxed);
    }
}

impl Bucket {
    /// Helper to insert a vector (Thread-Safe)
    /// In a real system, this would handle splitting logic.
    pub fn insert(&self, vid: u64, code: &[u8]) {
        let mut data = self.data.write(); // Acquire Write Lock

        // Push ID
        data.vids.push(vid);

        // Push Code Bytes (Critical: SIMD Alignment check happens here)
        for &b in code {
            data.codes.push(b);
        }

        let n = data.vids.len() - 1;
        // Init Tombstone (Alive)
        data.tombstones.insert(n);

        // Atomic Update
        self.count.fetch_add(1, Ordering::Relaxed);
    }
}
