use crate::aligned::AlignedBytes;
use crate::bitpack::{pack_u32_dynamic, unpack_u32_dynamic};
use crate::index::SearchResult;
use crate::quantizer::Quantizer;
use atomic_float::AtomicF32;
use bit_set::BitSet;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use drift_traits::{Cacheable, PageId};
use parking_lot::RwLock;
use std::io::{Cursor, Error, ErrorKind, Read, Result, Write};
use std::ops::{Index, IndexMut, Range};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

// --- SHARED STATS ---
#[derive(Debug)]
pub struct BucketStats {
    pub vector_sum: RwLock<Vec<f32>>,
    pub tombstone_count: AtomicU32,
    pub temperature: AtomicF32,
}

impl Default for BucketStats {
    fn default() -> Self {
        Self {
            tombstone_count: AtomicU32::new(0),
            temperature: AtomicF32::new(0.5),
            vector_sum: RwLock::new(Vec::new()),
        }
    }
}

// --- HEADER ---
#[derive(Debug, Clone)]
pub struct BucketHeader {
    pub id: u32,
    pub centroid: Vec<f32>,
    pub count: u32,
    pub page_id: PageId,
    pub stats: Arc<BucketStats>,
}

impl BucketHeader {
    pub fn new(id: u32, centroid: Vec<f32>, count: u32, page_id: PageId) -> Self {
        Self {
            id,
            centroid,
            count,
            page_id,
            stats: Arc::new(BucketStats::default()),
        }
    }

    pub fn temperature(&self) -> f32 {
        self.stats.temperature.load(Ordering::Relaxed)
    }

    /// Updates temperature using EWMA.
    /// High temperature indicates frequent access ("Heat").
    pub fn touch(&self) {
        const ALPHA: f32 = 0.05;
        let _ =
            self.stats
                .temperature
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                    if current >= 1.0 {
                        None
                    } else {
                        Some(current + ALPHA * (1.0 - current))
                    }
                });
    }

    /// ⚡ NEW: Decay temperature (Cooling).
    /// Called by the Janitor to simulate heat dissipation over time.
    pub fn cool(&self, decay_rate: f32) {
        let _ =
            self.stats
                .temperature
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                    Some(current * decay_rate)
                });
    }

    pub fn mark_tombstone(&self) {
        self.stats.tombstone_count.fetch_add(1, Ordering::Relaxed);
    }

    /// ⚡ THE URGENCY FUNCTION (Equation 5 from Research Paper)
    /// Urgency = (Emptiness / (T + epsilon)) + (Beta * ZombieRatio)
    ///
    /// This allows the "Death" signal (Zombie Ratio) to override the
    /// "Heat" signal (Temperature), solving the "Hot Zombie" paradox.
    pub fn calculate_urgency(&self, target_capacity: usize) -> f32 {
        let total = self.count as f32;
        let dead = self.stats.tombstone_count.load(Ordering::Relaxed) as f32;
        let temp = self.stats.temperature.load(Ordering::Relaxed);

        let live = (total - dead).max(0.0);

        // 1. Emptiness: How empty is the bucket relative to target?
        let emptiness = if live < target_capacity as f32 {
            (target_capacity as f32 - live) / target_capacity as f32
        } else {
            0.0
        };

        // 2. Zombie Ratio: What % of data is dead?
        let zombie_ratio = if total > 0.0 { dead / total } else { 0.0 };

        // Constants from Technical Paper Section 5 & Eq 5
        const EPSILON: f32 = 0.001;
        const BETA: f32 = 3.0;

        (emptiness / (temp + EPSILON)) + (BETA * zombie_ratio)
    }

    /// ⚡ NEW: Calculate Drift O(1) (Section 3.C)
    /// Returns distance between Initial Centroid and Current Mean.
    pub fn calculate_drift(&self) -> f32 {
        let sum_guard = self.stats.vector_sum.read();
        if sum_guard.is_empty() || self.count == 0 {
            return 0.0;
        }

        // Current Mean = Sum / Count
        // Drift = Dist(Mean, InitialCentroid)

        // Use squared distance first to avoid sqrt if possible, but the spec says norm.
        let mut dist_sq = 0.0;
        let n = self.count as f32;

        for (i, &sum_val) in sum_guard.iter().enumerate() {
            let mean_val = sum_val / n;
            let diff = mean_val - self.centroid[i];
            dist_sq += diff * diff;
        }

        dist_sq.sqrt()
    }
}

// --- DATA ---
pub struct BucketData {
    pub codes: AlignedBytes,
    pub vids: Vec<u64>,
    pub tombstones: BitSet,
}

impl BucketData {
    pub fn from(codes: AlignedBytes, vids: Vec<u64>, tombstones: BitSet) -> Self {
        Self {
            codes,
            vids,
            tombstones,
        }
    }
}

impl Cacheable for BucketData {
    fn from_bytes(data: &[u8]) -> Result<Self> {
        let mut cursor = Cursor::new(data);
        // [Implementation identical to your provided code...]
        // Validation Magic
        let magic = cursor.read_u32::<LittleEndian>()?;
        if magic != 0xBD47001 {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("Invalid Bucket Magic {} != {}", magic, 0xBD47001),
            ));
        }

        // Read Metadata
        let count = cursor.read_u32::<LittleEndian>()? as usize;
        let dim = cursor.read_u32::<LittleEndian>()? as usize;

        // Read Codes (Aligned)
        let codes_len = count * dim;
        let mut codes = AlignedBytes::new(codes_len);
        unsafe {
            codes.set_len(codes_len);
        }
        cursor.read_exact(codes.as_mut_slice())?;

        // Read IDs
        let mut vids = Vec::with_capacity(count);
        for _ in 0..count {
            vids.push(cursor.read_u64::<LittleEndian>()?);
        }

        // Read Compressed Tombstones
        let num_blocks = cursor.read_u32::<LittleEndian>()? as usize;
        let bit_width = cursor.read_u8()? as usize;
        let packed_len = cursor.read_u32::<LittleEndian>()? as usize;

        let mut packed_data = vec![0u32; packed_len];
        for val in &mut packed_data {
            *val = cursor.read_u32::<LittleEndian>()?;
        }

        let mut blocks = vec![0u32; num_blocks];
        if num_blocks > 0 {
            unpack_u32_dynamic(&packed_data, num_blocks, bit_width, &mut blocks);
        }

        let mut tombstones = BitSet::with_capacity(count);
        for (i, block) in blocks.iter().enumerate() {
            if *block == 0 {
                continue;
            }
            for bit in 0..32 {
                if (block & (1 << bit)) != 0 {
                    tombstones.insert(i * 32 + bit);
                }
            }
        }

        Ok(BucketData {
            codes,
            vids,
            tombstones,
        })
    }
}

impl BucketData {
    pub fn to_bytes(&self, dim: usize) -> Result<Vec<u8>> {
        let mut buf = Vec::new();
        // Magic
        buf.write_u32::<LittleEndian>(0xBD47001)?;
        buf.write_u32::<LittleEndian>(self.vids.len() as u32)?;
        buf.write_u32::<LittleEndian>(dim as u32)?;
        // Codes
        buf.write_all(self.codes.as_slice())?;
        // IDs
        for vid in &self.vids {
            buf.write_u64::<LittleEndian>(*vid)?;
        }
        // Tombstones (Bitpacked)
        let blocks = self.tombstones.get_ref().storage();
        let mut packed_buf = vec![0u32; blocks.len()]; // Worst case
        let (width, packed_count) = if blocks.is_empty() {
            (1, 0)
        } else {
            pack_u32_dynamic(blocks, &mut packed_buf)
        };

        buf.write_u32::<LittleEndian>(blocks.len() as u32)?;
        buf.write_u8(width as u8)?;
        buf.write_u32::<LittleEndian>(packed_count as u32)?;
        for &value in &packed_buf[0..packed_count] {
            buf.write_u32::<LittleEndian>(value)?;
        }
        Ok(buf)
    }

    pub fn reconstruct(&self, quantizer: &Quantizer) -> (Vec<Vec<f32>>, Vec<u64>) {
        let dim = quantizer.min.len();
        let count = self.vids.len();
        let mut vecs = Vec::with_capacity(count);
        let mut ids = Vec::with_capacity(count);

        for i in 0..count {
            if self.tombstones.contains(i) {
                continue;
            }
            let start = i * dim;
            let code = &self.codes[start..start + dim];
            vecs.push(quantizer.reconstruct(code));
            ids.push(self.vids[i]);
        }
        (vecs, ids)
    }
}

// --- BUCKET OPERATIONS ---
pub struct Bucket;

impl Bucket {
    /// ⚡ ADC KERNEL (Equation 4)
    /// Uses precomputed LUT to calculate L2 distance in high fidelity.
    pub fn scan_with_lut(data: &BucketData, lut: &[f32], dim: usize) -> Vec<SearchResult> {
        let mut results = Vec::with_capacity(data.vids.len());
        debug_assert_eq!(lut.len(), dim * 256);

        let codes_ptr = data.codes.as_ptr();
        let lut_ptr = lut.as_ptr();

        for i in 0..data.vids.len() {
            if data.tombstones.contains(i) {
                continue;
            }

            let code_offset = i * dim;
            // Unsafe SIMD-friendly kernel
            let dist = unsafe { compute_distance_lut(codes_ptr.add(code_offset), lut_ptr, dim) };

            results.push(SearchResult {
                id: data.vids[i],
                distance: dist,
            });
        }
        results
    }
}

/// ⚡ CORE ADC KERNEL
#[inline(always)]
unsafe fn compute_distance_lut(mut code_ptr: *const u8, lut_ptr: *const f32, dim: usize) -> f32 {
    let mut sum = 0.0;
    let mut i = 0;
    // 4x Loop Unrolling
    while i + 4 <= dim {
        let c0 = *code_ptr.add(0) as usize;
        let c1 = *code_ptr.add(1) as usize;
        let c2 = *code_ptr.add(2) as usize;
        let c3 = *code_ptr.add(3) as usize;

        let v0 = *lut_ptr.add((i + 0) * 256 + c0);
        let v1 = *lut_ptr.add((i + 1) * 256 + c1);
        let v2 = *lut_ptr.add((i + 2) * 256 + c2);
        let v3 = *lut_ptr.add((i + 3) * 256 + c3);

        sum += v0 + v1 + v2 + v3;
        code_ptr = code_ptr.add(4);
        i += 4;
    }
    while i < dim {
        let c = *code_ptr as usize;
        let val = *lut_ptr.add(i * 256 + c);
        sum += val;
        code_ptr = code_ptr.add(1);
        i += 1;
    }
    sum
}

// Boilerplate for AlignedBytes indexing
impl Index<Range<usize>> for AlignedBytes {
    type Output = [u8];
    #[inline]
    fn index(&self, range: Range<usize>) -> &Self::Output {
        &self.as_slice()[range]
    }
}
impl IndexMut<Range<usize>> for AlignedBytes {
    #[inline]
    fn index_mut(&mut self, range: Range<usize>) -> &mut Self::Output {
        &mut self.as_mut_slice()[range]
    }
}
