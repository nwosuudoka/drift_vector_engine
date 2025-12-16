use crate::aligned::AlignedBytes;
use crate::index::SearchResult;
use crate::quantizer::Quantizer;
use atomic_float::AtomicF32;
use bit_set::BitSet;
use parking_lot::RwLock;
use std::collections::BinaryHeap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

pub struct BucketData {
    pub codes: AlignedBytes,
    pub vids: Vec<u64>,
    pub tombstones: BitSet,
}

pub struct Bucket {
    pub id: u32,
    pub quantizer: Arc<Quantizer>, // NEW: Buckets know how to read themselves
    pub data: RwLock<BucketData>,

    // Stats
    pub centroid: RwLock<Vec<f32>>,
    pub count: AtomicU32,
    pub tombstone_count: AtomicU32,
    pub temperature: AtomicF32,
    pub last_maintenance: AtomicU64,
}

impl Bucket {
    pub fn new(id: u32, capacity: usize, dim: usize, quantizer: Arc<Quantizer>) -> Self {
        Self {
            id,
            quantizer,
            data: RwLock::new(BucketData {
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

    pub fn insert(&self, vid: u64, code: &[u8]) {
        let mut data = self.data.write();
        data.vids.push(vid);
        for &b in code {
            data.codes.push(b);
        }

        // Ensure tombstone bitset stays in sync
        if data.tombstones.contains(data.vids.len() - 1) {
            let n = data.vids.len() - 1;
            data.tombstones.remove(n);
        }

        self.count.fetch_add(1, Ordering::Relaxed);
    }

    /// The Heating Operation (Hot Path)
    /// T_new = T_old + alpha * (1.0 - T_old)
    pub fn touch(&self) {
        const ALPHA: f32 = 0.05;
        // Optimization: Stop atomic writes if already max heat
        if self.temperature.load(Ordering::Relaxed) > 0.99 {
            return;
        }
        let _ = self
            .temperature
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |t| {
                Some(t + ALPHA * (1.0 - t))
            });
    }

    /// The Cooling Operation
    pub fn decay_temperature(&self) {
        const LAMBDA: f32 = 0.98;
        let _ = self
            .temperature
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |t| Some(t * LAMBDA));
    }

    /// Calculate Urgency based on Equation (5)
    /// Urgency = (Emptiness / (T + epsilon)) + (Beta * ZombieRatio)
    pub fn calculate_urgency(&self, target_capacity: usize) -> f32 {
        let count = self.count.load(Ordering::Relaxed) as f32;
        let dead = self.tombstone_count.load(Ordering::Relaxed) as f32;
        let temp = self.temperature.load(Ordering::Relaxed);

        // 1. Calculate Live Count (Saturating sub to be safe)
        let live = (count - dead).max(0.0);

        // 2. Calculate Emptiness (0.0 to 1.0)
        // If target is 100 and live is 10, Emptiness is 0.9.
        let emptiness = if live < target_capacity as f32 {
            (target_capacity as f32 - live) / target_capacity as f32
        } else {
            0.0
        };

        // 3. Calculate Zombie Ratio (0.0 to 1.0)
        let zombie_ratio = if count > 0.0 { dead / count } else { 0.0 };

        // 4. Constants
        const EPSILON: f32 = 0.001; // Avoid div by zero
        const BETA: f32 = 3.0; // Weight for dead data

        // 5. The Formula
        // High Temp (1.0) -> Denominator ~1.0 -> Low Score (Protected)
        // Low Temp (0.0) -> Denominator ~0.001 -> High Score (Urgent)
        (emptiness / (temp + EPSILON)) + (BETA * zombie_ratio)
    }

    /// Helper: Reconstruct all VALID vectors (f32) for splitting.
    /// Skips tombstones.
    pub fn extract_reconstructed(&self) -> (Vec<Vec<f32>>, Vec<u64>) {
        let data = self.data.read();
        let dim = self.quantizer.min.len();
        let count = data.vids.len();

        let mut vecs = Vec::with_capacity(count);
        let mut ids = Vec::with_capacity(count);

        for i in 0..count {
            if data.tombstones.contains(i) {
                continue;
            }

            let start = i * dim;
            let code = &data.codes.as_slice()[start..start + dim];
            let vec = self.quantizer.reconstruct(code);

            vecs.push(vec);
            ids.push(data.vids[i]);
        }
        (vecs, ids)
    }

    /// Optimized ADC Scan (Asymmetric Distance Calculation).
    /// Dispatches to AVX2 or NEON kernels based on hardware.
    pub fn scan_adc(&self, lut: &[f32], k: usize) -> Vec<SearchResult> {
        self.temperature.fetch_add(1.0, Ordering::Relaxed);

        let data = self.data.read();
        let n_codes = data.vids.len();
        // LUT is flattened: [dim * 256] floats
        let dim = lut.len() / 256;

        // Prepare Heap
        let mut local_heap = BinaryHeap::with_capacity(k);

        // Access raw pointers for kernels
        let codes_base = data.codes.as_ptr();

        for i in 0..n_codes {
            // Logical Delete Check
            if data.tombstones.contains(i) {
                continue;
            }

            // Calculate Distance (Hot Loop)
            // Offset in the codes buffer: vector_index * dimension
            let offset = i * dim;

            let dist = unsafe { compute_distance_simd(codes_base.add(offset), lut, dim) };

            // Heap Management (Keep Top-K Smallest Distances)
            // Note: BinaryHeap is Max-Heap. We want smallest distance.
            // So we store SearchResult which implements Ord based on distance.
            // If we want smallest K, we push. If heap > K, we pop the MAX element.
            let result = SearchResult {
                id: data.vids[i],
                distance: dist,
            };

            if local_heap.len() < k {
                local_heap.push(result);
            } else if dist < local_heap.peek().unwrap().distance {
                local_heap.pop();
                local_heap.push(result);
            }
        }

        // Return sorted vectors
        let mut results = local_heap.into_vec();
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        results
    }

    /// Calculate Urgency based on Equation (5)
    /// Urgency = (Emptiness / (T + epsilon)) + (Beta * ZombieRatio)
    // pub fn calculate_urgency(&self, target_capacity: usize) -> f32 {
    //     let count = self.count.load(Ordering::Relaxed) as f32;
    //     let dead = self.tombstone_count.load(Ordering::Relaxed) as f32;
    //     let temp = self.temperature.load(Ordering::Relaxed);

    //     let live = count - dead;

    //     // 1. Calculate Emptiness
    //     // "How much space is unused relative to target?"
    //     // If we have 10 live items and target is 100, Emptiness is 0.9.
    //     let emptiness = if live < target_capacity as f32 {
    //         (target_capacity as f32 - live) / target_capacity as f32
    //     } else {
    //         0.0
    //     };

    //     // 2. Calculate Zombie Ratio (Tombstones / Total)
    //     let zombie_ratio = if count > 0.0 { dead / count } else { 0.0 };

    //     // 3. Constants from Spec
    //     const EPSILON: f32 = 0.001; // Avoid div by zero if temp is 0
    //     const BETA: f32 = 3.0;

    //     // 4. The Formula
    //     // High Temp -> Large Denominator -> Low Urgency (Protected)
    //     (emptiness / (temp + EPSILON)) + (BETA * zombie_ratio)
    // }

    /// Helper: Removes specific VIDs for Neighbor Stealing.
    /// Returns the reconstructed vectors for the removed IDs.
    pub fn steal_vectors(&self, target_ids: &[u64]) -> Vec<Vec<f32>> {
        let mut data = self.data.write();
        let mut stolen_vecs = Vec::new();
        let dim = self.quantizer.min.len();

        // Optimization: Create a HashSet/Bitmap for O(1) lookup if target_ids is large.
        // For strict budget of 50, linear scan is acceptable (50 * N).
        // However, we must physically remove them from `vids` and `codes`.
        // This is O(N) memory move. In production, we usually mark Tombstone instead
        // and copy the data out.

        for &id in target_ids {
            if let Some(pos) = data.vids.iter().position(|&x| x == id) {
                // 1. Reconstruct
                let start = pos * dim;
                let code = &data.codes.as_slice()[start..start + dim];
                let vec = self.quantizer.reconstruct(code);
                stolen_vecs.push(vec);

                // 2. Mark as Tombstone (Logical Move)
                // Physical removal happens during next compaction/split.
                data.tombstones.insert(pos);
                self.tombstone_count.fetch_add(1, Ordering::Relaxed);
            }
        }
        stolen_vecs
    }
}

// =================================================================================
//  SIMD KERNELS
// =================================================================================

/// Dispatcher: Chooses best kernel at compile time
#[inline(always)]
unsafe fn compute_distance_simd(code_ptr: *const u8, lut: &[f32], dim: usize) -> f32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return compute_distance_avx2(code_ptr, lut.as_ptr(), dim);
        }
    }

    // Fallback or ARM
    compute_distance_scalar_unrolled(code_ptr, lut, dim)
}

/// AVX2 Kernel: Uses Gather for 8-way parallel lookup.
/// Processes 8 dimensions per cycle.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn compute_distance_avx2(mut code_ptr: *const u8, lut_ptr: *const f32, dim: usize) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut sum_vec = _mm256_setzero_ps();
    let mut i = 0;

    // We process 8 dimensions at a time.
    // Base offsets for LUT: [0*256, 1*256, ..., 7*256]
    // These align the gathering to the correct dimension block in the flattened LUT.
    let mut offsets = _mm256_set_epi32(
        7 * 256,
        6 * 256,
        5 * 256,
        4 * 256,
        3 * 256,
        2 * 256,
        1 * 256,
        0 * 256,
    );
    let offset_step = _mm256_set1_epi32(8 * 256);

    while i + 8 <= dim {
        // 1. Load 8 bytes of codes (8 dimensions)
        // We load as i64 (8 bytes) then expand to 8 x 32-bit integers
        let raw_codes_64 = (code_ptr as *const i64).read_unaligned();
        let codes_128 = _mm_cvtsi64_si128(raw_codes_64);

        // Expand u8 -> i32 (Zero extend)
        let codes_idx = _mm256_cvtepu8_epi32(codes_128);

        // 2. Calculate exact indices: LUT_Index = Base_Offset + Code_Byte
        let gather_indices = _mm256_add_epi32(offsets, codes_idx);

        // 3. VGATHER: Load 8 floats from LUT using the calculated indices
        // scale=4 (sizeof f32)
        let values = _mm256_i32gather_ps(lut_ptr, gather_indices, 4);

        // 4. Accumulate
        sum_vec = _mm256_add_ps(sum_vec, values);

        // Advance
        code_ptr = code_ptr.add(8);
        offsets = _mm256_add_epi32(offsets, offset_step);
        i += 8;
    }

    // Horizontal Sum of the 8 accumulators
    // AVX2 H-Sum trick
    let t1 = _mm256_hadd_ps(sum_vec, sum_vec);
    let t2 = _mm256_hadd_ps(t1, t1);
    let t3 = _mm256_extractf128_ps(t2, 1);
    let t4 = _mm_add_ss(_mm256_castps256_ps128(t2), t3);
    let mut sum = _mm_cvtss_f32(t4);

    // Handle Remainder (Scalar)
    let lut_slice = std::slice::from_raw_parts(lut_ptr, dim * 256);
    while i < dim {
        let code = *code_ptr as usize;
        // lut index = dim_index * 256 + code
        let lut_idx = (i << 8) + code;
        sum += lut_slice[lut_idx];

        code_ptr = code_ptr.add(1);
        i += 1;
    }

    sum
}

/// Fallback / NEON Kernel
/// Uses 4x Unrolling to saturate instruction pipeline.
unsafe fn compute_distance_scalar_unrolled(
    mut code_ptr: *const u8,
    lut: &[f32],
    dim: usize,
) -> f32 {
    let mut sum = 0.0;
    let mut i = 0;

    // Unroll 4x
    while i + 4 <= dim {
        let c0 = *code_ptr.add(0) as usize;
        let c1 = *code_ptr.add(1) as usize;
        let c2 = *code_ptr.add(2) as usize;
        let c3 = *code_ptr.add(3) as usize;

        // LUT Access Pattern:
        // i * 256 + c
        let idx0 = ((i + 0) << 8) + c0;
        let idx1 = ((i + 1) << 8) + c1;
        let idx2 = ((i + 2) << 8) + c2;
        let idx3 = ((i + 3) << 8) + c3;

        let v0 = *lut.get_unchecked(idx0);
        let v1 = *lut.get_unchecked(idx1);
        let v2 = *lut.get_unchecked(idx2);
        let v3 = *lut.get_unchecked(idx3);

        sum += v0 + v1 + v2 + v3;

        code_ptr = code_ptr.add(4);
        i += 4;
    }

    // Remainder
    while i < dim {
        let c = *code_ptr as usize;
        let idx = (i << 8) + c;
        sum += *lut.get_unchecked(idx);
        code_ptr = code_ptr.add(1);
        i += 1;
    }

    sum
}
