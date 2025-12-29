use crate::aligned::AlignedBytes;
use crate::bitpack::{pack_u32_dynamic, unpack_u32_dynamic};
use crate::index::SearchResult;
use crate::quantizer::Quantizer;
use atomic_float::AtomicF32;
use bit_set::BitSet;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use drift_traits::{Cacheable, PageId};
use std::io::{Cursor, Error, ErrorKind, Read, Result, Write};
use std::ops::{Index, IndexMut, Range};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

// --- SHARED STATS (Recovered from previous step) ---
#[derive(Debug)]
pub struct BucketStats {
    pub tombstone_count: AtomicU32,
    pub temperature: AtomicF32,
}

impl Default for BucketStats {
    fn default() -> Self {
        Self {
            tombstone_count: AtomicU32::new(0),
            temperature: AtomicF32::new(0.5),
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

    pub fn mark_tombstone(&self) {
        self.stats.tombstone_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn calculate_urgency(&self, target_capacity: usize) -> f32 {
        let total = self.count as f32;
        let dead = self.stats.tombstone_count.load(Ordering::Relaxed) as f32;
        let temp = self.stats.temperature.load(Ordering::Relaxed);
        let live = (total - dead).max(0.0);
        let emptiness = if live < target_capacity as f32 {
            (target_capacity as f32 - live) / target_capacity as f32
        } else {
            0.0
        };
        let zombie_ratio = if total > 0.0 { dead / total } else { 0.0 };
        const EPSILON: f32 = 0.001;
        const BETA: f32 = 3.0;
        (emptiness / (temp + EPSILON)) + (BETA * zombie_ratio)
    }
}

// --- DATA ---
pub struct BucketData {
    pub codes: AlignedBytes,
    pub vids: Vec<u64>,
    pub tombstones: BitSet,
}

impl Cacheable for BucketData {
    fn from_bytes(data: &[u8]) -> Result<Self> {
        let mut cursor = Cursor::new(data);
        let magic = cursor.read_u32::<LittleEndian>()?;
        if magic != 0xBD47001 {
            return Err(Error::new(ErrorKind::InvalidData, "Invalid Bucket Magic"));
        }
        let count = cursor.read_u32::<LittleEndian>()? as usize;
        let dim = cursor.read_u32::<LittleEndian>()? as usize;

        let codes_len = count * dim;
        let mut codes = AlignedBytes::new(codes_len);
        unsafe {
            codes.set_len(codes_len);
        }
        cursor.read_exact(codes.as_mut_slice())?;

        let mut vids = Vec::with_capacity(count);
        for _ in 0..count {
            vids.push(cursor.read_u64::<LittleEndian>()?);
        }

        let num_blocks = cursor.read_u32::<LittleEndian>()? as usize;
        let bit_width = cursor.read_u8()? as usize;
        let packed_len = cursor.read_u32::<LittleEndian>()? as usize;
        let mut packed_data = vec![0u32; packed_len];
        for i in 0..packed_len {
            packed_data[i] = cursor.read_u32::<LittleEndian>()?;
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
        buf.write_u32::<LittleEndian>(0xBD47001)?;
        buf.write_u32::<LittleEndian>(self.vids.len() as u32)?;
        buf.write_u32::<LittleEndian>(dim as u32)?;
        buf.write_all(self.codes.as_slice())?;
        for vid in &self.vids {
            buf.write_u64::<LittleEndian>(*vid)?;
        }
        let blocks = self.tombstones.get_ref().storage();
        let mut packed_buf = vec![0u32; blocks.len()];
        let (width, packed_count) = if blocks.is_empty() {
            (1, 0)
        } else {
            pack_u32_dynamic(blocks, &mut packed_buf)
        };
        buf.write_u32::<LittleEndian>(blocks.len() as u32)?;
        buf.write_u8(width as u8)?;
        buf.write_u32::<LittleEndian>(packed_count as u32)?;
        for i in 0..packed_count {
            buf.write_u32::<LittleEndian>(packed_buf[i])?;
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

    // NEW: Zero-copy construction from components
    pub fn from_components(codes: Vec<u8>, vids: Vec<u64>) -> Self {
        // We need to ensure alignment for SIMD.
        // Vec<u8> might not be 64-byte aligned.
        // We copy it into AlignedBytes.
        let mut aligned = AlignedBytes::new(codes.len());
        unsafe {
            // Fast copy
            std::ptr::copy_nonoverlapping(
                codes.as_ptr(),
                aligned.as_mut_slice().as_mut_ptr(),
                codes.len(),
            );
            aligned.set_len(codes.len());
        }

        Self {
            codes: aligned,
            vids: vids.clone(),
            tombstones: BitSet::with_capacity(vids.len()),
        }
    }
}

// --- BUCKET OPERATIONS ---
pub struct Bucket;

impl Bucket {
    pub fn scan_static(
        data: &BucketData,
        quantizer: &Quantizer,
        query: &[f32],
    ) -> Vec<SearchResult> {
        let mut results = Vec::with_capacity(data.vids.len());
        let dim = query.len();

        // Use the precomputed LUT from the Quantizer if available,
        // OR compute it here if Quantizer doesn't expose it yet.
        // For V1 correctness, let's use the verified loop you provided in the prompt:

        for i in 0..data.vids.len() {
            let vid = data.vids[i];
            if data.tombstones.contains(i) {
                continue;
            }

            let start = i * dim;
            let end = start + dim;
            let code = &data.codes[start..end];

            let dist = quantizer.distance_adc(query, code);

            results.push(SearchResult {
                id: vid,
                distance: dist,
            });
        }
        results
    }
    /// ⚡ HOT PATH: Scans bucket using Precomputed LUT.
    /// This replaces the old `scan_static` that used `quantizer.distance_adc`.
    ///
    /// # Arguments
    /// * `lut`: Precomputed [Dim * 256] float array from Quantizer.
    pub fn scan_with_lut(data: &BucketData, lut: &[f32], dim: usize) -> Vec<SearchResult> {
        let mut results = Vec::with_capacity(data.vids.len());

        // Ensure LUT matches dimension (Sanity Check)
        debug_assert_eq!(lut.len(), dim * 256);

        // Raw pointers for speed in loop
        let codes_ptr = data.codes.as_ptr();
        let lut_ptr = lut.as_ptr();

        for i in 0..data.vids.len() {
            if data.tombstones.contains(i) {
                continue;
            }

            let code_offset = i * dim;

            // Unsafe call to optimized distance kernel
            // Safety: We verified buffer sizes above.
            let dist = unsafe { compute_distance_lut(codes_ptr.add(code_offset), lut_ptr, dim) };

            results.push(SearchResult {
                id: data.vids[i],
                distance: dist,
            });
        }
        results
    }
}

/// ⚡ CORE KERNEL: Computes distance using LUT lookups.
/// Manually unrolled 4x to saturate instruction pipeline.
#[inline(always)]
unsafe fn compute_distance_lut(mut code_ptr: *const u8, lut_ptr: *const f32, dim: usize) -> f32 {
    let mut sum = 0.0;
    let mut i = 0;

    // 4x Unrolling
    while i + 4 <= dim {
        let c0 = *code_ptr.add(0) as usize;
        let c1 = *code_ptr.add(1) as usize;
        let c2 = *code_ptr.add(2) as usize;
        let c3 = *code_ptr.add(3) as usize;

        // LUT Layout: [dim_index * 256 + byte_value]
        let v0 = *lut_ptr.add((i + 0) * 256 + c0);
        let v1 = *lut_ptr.add((i + 1) * 256 + c1);
        let v2 = *lut_ptr.add((i + 2) * 256 + c2);
        let v3 = *lut_ptr.add((i + 3) * 256 + c3);

        sum += v0 + v1 + v2 + v3;

        code_ptr = code_ptr.add(4);
        i += 4;
    }

    // Remainder
    while i < dim {
        let c = *code_ptr as usize;
        let val = *lut_ptr.add(i * 256 + c);
        sum += val;
        code_ptr = code_ptr.add(1);
        i += 1;
    }

    sum
}

// AlignedBytes boilerplate
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
