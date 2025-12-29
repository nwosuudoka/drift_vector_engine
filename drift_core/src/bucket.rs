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

pub struct BucketData {
    pub codes: AlignedBytes,
    pub vids: Vec<u64>,
    pub tombstones: BitSet,
}

impl Cacheable for BucketData {
    fn from_bytes(data: &[u8]) -> Result<Self> {
        let mut cursor = Cursor::new(data);

        // 1. Read Header
        // Valid Hex Literal (BDAT001 approx)
        let magic = cursor.read_u32::<LittleEndian>()?;
        if magic != 0xBD47001 {
            return Err(Error::new(ErrorKind::InvalidData, "Invalid Bucket Magic"));
        }
        let count = cursor.read_u32::<LittleEndian>()? as usize;
        let dim = cursor.read_u32::<LittleEndian>()? as usize;

        // 2. Read Codes
        let codes_len = count * dim;
        let mut codes = AlignedBytes::new(codes_len);

        unsafe {
            codes.set_len(codes_len);
        }

        cursor.read_exact(codes.as_mut_slice())?;

        // 3. Read VIDs
        let mut vids = Vec::with_capacity(count);
        for _ in 0..count {
            vids.push(cursor.read_u64::<LittleEndian>()?);
        }

        // 4. Read Tombstones (BitPacked)
        // [Num_Blocks(u32)] [Bit_Width(u8)] [Packed_Len(u32)] [Packed_Data...]
        let num_blocks = cursor.read_u32::<LittleEndian>()? as usize;
        let bit_width = cursor.read_u8()? as usize;
        let packed_len = cursor.read_u32::<LittleEndian>()? as usize;

        let mut packed_data = vec![0u32; packed_len];
        #[allow(clippy::needless_range_loop)]
        for i in 0..packed_len {
            packed_data[i] = cursor.read_u32::<LittleEndian>()?;
        }

        // Unpack
        let mut blocks = vec![0u32; num_blocks];
        if num_blocks > 0 {
            unpack_u32_dynamic(&packed_data, num_blocks, bit_width, &mut blocks);
        }

        // Reconstruct BitSet from blocks
        // BitSet uses BitVec, which can be constructed from bytes.
        // We cast &[u32] -> &[u8] safely.
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

        // Header
        buf.write_u32::<LittleEndian>(0xBD47001)?; // Magic Fixed
        buf.write_u32::<LittleEndian>(self.vids.len() as u32)?;
        buf.write_u32::<LittleEndian>(dim as u32)?;

        // Codes
        buf.write_all(self.codes.as_slice())?;

        // VIDs
        for vid in &self.vids {
            buf.write_u64::<LittleEndian>(*vid)?;
        }

        // Tombstones (BitPacked)
        // Access inner storage: BitVec storage is usually Vec<u32>
        let blocks = self.tombstones.get_ref().storage();

        // Prepare output buffer for packing
        // Max size is same as input (width=32)
        let mut packed_buf = vec![0u32; blocks.len()];

        let (width, packed_count) = if blocks.is_empty() {
            (1, 0)
        } else {
            pack_u32_dynamic(blocks, &mut packed_buf)
        };

        // Write Metadata
        buf.write_u32::<LittleEndian>(blocks.len() as u32)?; // Num Blocks
        buf.write_u8(width as u8)?; // Bit Width
        buf.write_u32::<LittleEndian>(packed_count as u32)?; // Packed Word Count

        // Write Packed Data
        #[allow(clippy::needless_range_loop)]
        for i in 0..packed_count {
            buf.write_u32::<LittleEndian>(packed_buf[i])?;
        }

        Ok(buf)
    }

    /// Reconstructs all valid vectors (skipping tombstones) from this data block.
    /// Used during Splitting/Merging to recover the training data.
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
            let end = start + dim;
            // Safe slicing with new AlignedBytes index trait
            let code = &self.codes[start..end];

            let vec = quantizer.reconstruct(code);
            vecs.push(vec);
            ids.push(self.vids[i]);
        }
        (vecs, ids)
    }
}

#[derive(Debug)]
pub struct BucketStats {
    pub tombstone_count: AtomicU32,
    pub temperature: AtomicF32,
}

impl Default for BucketStats {
    fn default() -> Self {
        Self {
            tombstone_count: AtomicU32::new(0),
            // Start neutral/warm so we don't merge immediately on creation
            temperature: AtomicF32::new(0.5),
        }
    }
}

/// The RAM-resident metadata for a bucket.
/// This allows us to find the "Right Bucket" without loading the "Heavy Data".
#[derive(Debug, Clone)]
pub struct BucketHeader {
    pub id: u32,
    pub centroid: Vec<f32>, // For routing (ADC/L2 distance)
    pub count: u32,         // For density weighting
    pub page_id: PageId,    // Pointer to Disk/Cache
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

    /// Update Heat: Called during search
    pub fn touch(&self) {
        // Simple Decay-based update: New = Old + Alpha * (1 - Old)
        // This asymptotically approaches 1.0 on frequent hits.
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

    /// Mark Dead: Called during delete
    pub fn mark_tombstone(&self) {
        self.stats.tombstone_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn calculate_urgency(&self, target_capacity: usize) -> f32 {
        let total = self.count as f32;
        let dead = self.stats.tombstone_count.load(Ordering::Relaxed) as f32;
        let temp = self.stats.temperature.load(Ordering::Relaxed);

        let live = (total - dead).max(0.0);

        // Emptiness (Inverse Density)
        let emptiness = if live < target_capacity as f32 {
            (target_capacity as f32 - live) / target_capacity as f32
        } else {
            0.0
        };

        // Zombie Ratio
        let zombie_ratio = if total > 0.0 { dead / total } else { 0.0 };

        const EPSILON: f32 = 0.001;
        const BETA: f32 = 3.0;

        (emptiness / (temp + EPSILON)) + (BETA * zombie_ratio)
    }
}

pub struct Bucket {
    pub id: u32,
}

impl Bucket {
    /// Stateless Scan: Performs search on detached BucketData.
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
}

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
