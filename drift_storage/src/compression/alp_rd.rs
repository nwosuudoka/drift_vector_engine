use std::mem::size_of;

// ============================================================
//  Traits and Types
// ============================================================

mod private {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

pub trait ALPRDFloat: private::Sealed + Copy + Default + PartialEq + std::fmt::Debug {
    type UInt: Copy
        + Into<u64>
        + From<u8>
        + std::ops::Shl<usize, Output = Self::UInt>
        + std::ops::Shr<usize, Output = Self::UInt>
        + std::ops::BitAnd<Output = Self::UInt>
        + std::ops::BitOr<Output = Self::UInt>
        + std::ops::Not<Output = Self::UInt>;

    const BITS: usize;
    const BYTES: usize;

    fn to_bits(v: Self) -> Self::UInt;
    fn from_bits(u: Self::UInt) -> Self;
    fn to_u64(u: Self::UInt) -> u64;
    fn from_u64(x: u64) -> Self::UInt;
}

impl ALPRDFloat for f64 {
    type UInt = u64;
    const BITS: usize = 64;
    const BYTES: usize = 8;
    #[inline(always)]
    fn to_bits(v: Self) -> Self::UInt {
        v.to_bits()
    }
    #[inline(always)]
    fn from_bits(u: Self::UInt) -> Self {
        f64::from_bits(u)
    }
    #[inline(always)]
    fn to_u64(u: Self::UInt) -> u64 {
        u
    }
    #[inline(always)]
    fn from_u64(x: u64) -> Self::UInt {
        x
    }
}

impl ALPRDFloat for f32 {
    type UInt = u32;
    const BITS: usize = 32;
    const BYTES: usize = 4;
    #[inline(always)]
    fn to_bits(v: Self) -> Self::UInt {
        v.to_bits()
    }
    #[inline(always)]
    fn from_bits(u: Self::UInt) -> Self {
        f32::from_bits(u)
    }
    #[inline(always)]
    fn to_u64(u: Self::UInt) -> u64 {
        u as u64
    }
    #[inline(always)]
    fn from_u64(x: u64) -> Self::UInt {
        x as u32
    }
}

// ============================================================
//  High-Performance Bit IO (The Engine)
// ============================================================

pub struct BitReader<'a> {
    ptr: *const u8,
    end: *const u8,
    buffer: u128,
    bits_count: u32,
    _marker: std::marker::PhantomData<&'a u8>,
}

impl<'a> BitReader<'a> {
    #[inline(always)]
    fn new(data: &'a [u8]) -> Self {
        Self {
            ptr: data.as_ptr(),
            end: unsafe { data.as_ptr().add(data.len()) },
            buffer: 0,
            bits_count: 0,
            _marker: std::marker::PhantomData,
        }
    }

    #[inline(always)]
    fn refill(&mut self) {
        if self.bits_count <= 64 {
            if unsafe { self.ptr.add(8) } <= self.end {
                let chunk = unsafe { (self.ptr as *const u64).read_unaligned().to_le() };
                self.buffer |= (chunk as u128) << self.bits_count;
                self.ptr = unsafe { self.ptr.add(8) };
                self.bits_count += 64;
            } else {
                while self.bits_count < 64 && self.ptr < self.end {
                    let byte = unsafe { *self.ptr };
                    self.buffer |= (byte as u128) << self.bits_count;
                    self.bits_count += 8;
                    self.ptr = unsafe { self.ptr.add(1) };
                }
            }
        }
    }

    #[inline(always)]
    fn read(&mut self, width: u8) -> u64 {
        if width == 0 {
            return 0;
        }
        self.refill();

        let mask = if width == 64 {
            u128::MAX
        } else {
            (1u128 << width) - 1
        };
        let val = (self.buffer & mask) as u64;

        self.buffer >>= width;
        self.bits_count -= width as u32;
        val
    }
}

pub(crate) struct BitWriter {
    bytes: Vec<u8>,
    buffer: u64,
    bits_count: u32,
}

impl BitWriter {
    pub fn new_with_capacity(cap: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(cap),
            buffer: 0,
            bits_count: 0,
        }
    }

    #[inline(always)]
    pub fn write(&mut self, val: u64, width: u8) {
        if width == 0 {
            return;
        }
        // Mask val to width to be safe
        let val = if width == 64 {
            val
        } else {
            val & ((1 << width) - 1)
        };

        self.buffer |= val << self.bits_count;
        self.bits_count += width as u32;

        if self.bits_count >= 64 {
            self.bytes.extend_from_slice(&self.buffer.to_le_bytes());
            let remaining = self.bits_count - 64;
            if remaining == 0 {
                self.buffer = 0;
            } else {
                // Safe shift: width >= remaining because we just added width
                self.buffer = val >> (width as u32 - remaining);
            }
            self.bits_count = remaining;
        }
    }

    pub fn flush(&mut self) {
        while self.bits_count > 0 {
            self.bytes.push(self.buffer as u8);
            self.buffer >>= 8;
            self.bits_count = self.bits_count.saturating_sub(8);
        }
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

fn bit_width(v: u64) -> u8 {
    if v == 0 {
        1
    } else {
        64 - v.leading_zeros() as u8
    }
}

// ============================================================
//  Dictionary Logic
// ============================================================

#[derive(Clone)]
pub(crate) struct Dictionary {
    pub left_bit_width: u8,
    pub right_bit_width: u8,
    pub dict: Vec<u64>,
}

// Replaces `choose_dictionary`
fn train_dictionary<T: ALPRDFloat>(values: &[T]) -> Dictionary {
    let n = values.len();

    // 1. Sampling Strategy
    // We aim for ~1024 samples. If N < 1024, use everything.
    let sample_target = 1024;
    let step = if n <= sample_target {
        1
    } else {
        n / sample_target
    };

    // We collect samples into a small vector to avoid repeated striding in the hot loop
    let samples: Vec<u64> = values
        .iter()
        .step_by(step)
        .take(sample_target * 2) // safety margin
        .map(|&v| T::to_bits(v).into())
        .collect();

    let sample_count = samples.len();

    // 2. Optimization Loop (Run ONLY on samples)
    let mut best_est_bits = f64::MAX;
    let mut best_dict = Dictionary {
        left_bit_width: 0,
        right_bit_width: 0,
        dict: vec![],
    };

    let range = if T::BITS == 64 {
        (48..=64).chain(0..1)
    } else {
        (0..=32).chain(0..0)
    };
    let mut counts = std::collections::HashMap::with_capacity(16);

    for left_bits_count in range {
        if left_bits_count > T::BITS {
            continue;
        }
        let right_bw = (T::BITS - left_bits_count) as u8;

        counts.clear();

        // Histogram on samples
        for &val in &samples {
            let left = if right_bw >= 64 { 0 } else { val >> right_bw };
            *counts.entry(left).or_insert(0) += 1;
        }

        let mut sorted: Vec<(u64, usize)> = counts.iter().map(|(&k, &v)| (k, v)).collect();
        sorted.sort_unstable_by(|a, b| b.1.cmp(&a.1));

        let dict_len = sorted.len().min(8);
        let dict_vals: Vec<u64> = sorted.iter().take(dict_len).map(|x| x.0).collect();

        let left_bw = bit_width((dict_len.saturating_sub(1)) as u64);

        // Estimate size based on sample frequencies
        let exceptions: usize = sorted.iter().skip(dict_len).map(|x| x.1).sum();
        let raw_left_w = T::BITS as u8 - right_bw;

        // Calculate Cost Function
        // We project the sample cost to the full array size for comparison
        let total_bits = (sample_count as f64 * (left_bw as f64 + right_bw as f64))
            + (exceptions as f64 * raw_left_w as f64)
            + (exceptions as f64 * 16.0); // Penalty for exception position overhead

        if total_bits < best_est_bits {
            best_est_bits = total_bits;
            best_dict = Dictionary {
                left_bit_width: left_bw,
                right_bit_width: right_bw,
                dict: dict_vals,
            };
        }
    }

    best_dict
}

// ============================================================
//  Encoder
// ============================================================

pub fn alp_rd_encode<T: ALPRDFloat>(values: &[T]) -> Vec<u8> {
    if values.is_empty() {
        return vec![];
    }

    // 1. FAST TRAINING (Sampling)
    // Only looks at ~1000 items, regardless of array size.
    let dict = train_dictionary(values);

    let left_bw = dict.left_bit_width;
    let right_bw = dict.right_bit_width;
    let raw_left_width = (T::BITS - right_bw as usize) as u8;

    // Estimate size
    let est_size = (values.len() * (left_bw as usize + right_bw as usize)) / 8 + 1024;
    let mut writer = BitWriter::new_with_capacity(est_size);

    // --- Header ---
    writer.bytes.push(if size_of::<T>() == 4 { 32 } else { 64 });
    writer.bytes.push(left_bw);
    writer.bytes.push(right_bw);
    writer.bytes.push(dict.dict.len() as u8);
    for &v in &dict.dict {
        writer.bytes.extend_from_slice(&v.to_le_bytes());
    }
    writer
        .bytes
        .extend_from_slice(&(values.len() as u32).to_le_bytes());

    // Placeholders for exception counts/sizes
    // We don't know them yet, so we remember the position to patch later.
    let exc_pos_count_offset = writer.bytes.len();
    writer.bytes.extend_from_slice(&0u32.to_le_bytes()); // exc_pos len placeholder

    let exc_pos_bw_offset = writer.bytes.len();
    writer.bytes.push(0); // exc_pos_bw placeholder

    // Body BitWriters
    let mut left_writer = BitWriter::new_with_capacity(values.len() * left_bw as usize / 8);
    let mut right_writer = BitWriter::new_with_capacity(values.len() * right_bw as usize / 8);

    let mut exc_pos = Vec::new();
    let mut exc_val = Vec::new();

    // Optimization: Pre-compute dictionary lookup map if dict is large,
    // but for size 8, a simple array scan is faster than a HashMap.

    // 2. SINGLE PASS ENCODING
    for (i, &v) in values.iter().enumerate() {
        let val: u64 = T::to_bits(v).into();

        let right = if right_bw == 64 {
            val
        } else {
            val & ((1 << right_bw) - 1)
        };
        let left = if right_bw >= 64 { 0 } else { val >> right_bw };

        // Find in dictionary
        if let Some(pos) = dict.dict.iter().position(|&x| x == left) {
            left_writer.write(pos as u64, left_bw);
        } else {
            // Exception: Write 0 (or any valid placeholder) to left stream
            left_writer.write(0, left_bw);
            exc_pos.push(i as u64);
            exc_val.push(left);
        }

        right_writer.write(right, right_bw);
    }

    left_writer.flush();
    right_writer.flush();

    // 3. Write Exceptions
    let exc_pos_bw = bit_width(*exc_pos.iter().max().unwrap_or(&0));

    let mut ep_writer = BitWriter::new_with_capacity(exc_pos.len() * 4);
    for &p in &exc_pos {
        ep_writer.write(p, exc_pos_bw);
    }
    ep_writer.flush();

    let mut ev_writer = BitWriter::new_with_capacity(exc_val.len() * 4);
    for &v in &exc_val {
        ev_writer.write(v, raw_left_width);
    }
    ev_writer.flush();

    // 4. Final Stitching & Header Patching

    // Write Stream Sizes
    writer
        .bytes
        .extend_from_slice(&(left_writer.bytes.len() as u32).to_le_bytes());
    writer
        .bytes
        .extend_from_slice(&(right_writer.bytes.len() as u32).to_le_bytes());
    writer
        .bytes
        .extend_from_slice(&(ep_writer.bytes.len() as u32).to_le_bytes());
    writer
        .bytes
        .extend_from_slice(&(ev_writer.bytes.len() as u32).to_le_bytes());

    // Append Bodies
    writer.bytes.extend_from_slice(&left_writer.bytes);
    writer.bytes.extend_from_slice(&right_writer.bytes);
    writer.bytes.extend_from_slice(&ep_writer.bytes);
    writer.bytes.extend_from_slice(&ev_writer.bytes);

    // Patch Header (Exception Count and Bit Width)
    let count_bytes = (exc_pos.len() as u32).to_le_bytes();
    writer.bytes[exc_pos_count_offset..exc_pos_count_offset + 4].copy_from_slice(&count_bytes);
    writer.bytes[exc_pos_bw_offset] = exc_pos_bw;

    writer.bytes
}

// ============================================================
//  Decoder (Fixed Panic on 64-bit shift and Index Out of Bounds)
// ============================================================

pub fn alp_rd_decode<T: ALPRDFloat>(bytes: &[u8]) -> Vec<T> {
    if bytes.is_empty() {
        return Vec::new();
    }

    let mut p = 0;

    macro_rules! read_u8 {
        () => {{
            let v = bytes[p];
            p += 1;
            v
        }};
    }
    macro_rules! read_u32 {
        () => {{
            let s = bytes[p..p + 4].try_into().unwrap();
            p += 4;
            u32::from_le_bytes(s)
        }};
    }

    let type_marker = read_u8!();
    assert_eq!(type_marker as usize, T::BITS, "Type mismatch");

    let left_bw = read_u8!();
    let right_bw = read_u8!();
    let dict_len = read_u8!() as usize;

    let mut dict = [0u64; 8];
    for di in dict.iter_mut().take(dict_len) {
        let s = bytes[p..p + 8].try_into().unwrap();
        *di = u64::from_le_bytes(s);
        p += 8;
    }

    let total_count = read_u32!() as usize;
    let exc_count = read_u32!() as usize;
    let exc_pos_bw = read_u8!();

    let len_left = read_u32!() as usize;
    let len_right = read_u32!() as usize;
    let len_exc_pos = read_u32!() as usize;
    let _len_exc_val = read_u32!() as usize;

    // Use unsafe pointer arithmetic relative to base to avoid bounds checks
    // panicking when calculating start of empty sections at end of file.
    let base_ptr = bytes.as_ptr();
    let left_ptr = unsafe { base_ptr.add(p) };
    let right_ptr = unsafe { base_ptr.add(p + len_left) };
    let exc_pos_ptr = unsafe { base_ptr.add(p + len_left + len_right) };
    let exc_val_ptr = unsafe { base_ptr.add(p + len_left + len_right + len_exc_pos) };

    let mut out = Vec::with_capacity(total_count);

    let mut left_r = BitReader::new(unsafe { std::slice::from_raw_parts(left_ptr, len_left) });
    let mut right_r = BitReader::new(unsafe { std::slice::from_raw_parts(right_ptr, len_right) });

    let rbw_usize = right_bw as usize;

    // --- HOT LOOP ---
    for _ in 0..total_count {
        let idx = left_r.read(left_bw) as usize;
        let high_bits = unsafe { *dict.get_unchecked(idx) };
        let low_bits = right_r.read(right_bw);

        // Guard against shift overflow if right_bw == 64
        let full_bits_u64 = if right_bw >= 64 {
            low_bits
        } else {
            (high_bits << rbw_usize) | low_bits
        };

        out.push(T::from_bits(T::from_u64(full_bits_u64)));
    }

    // --- EXCEPTION PATCHING ---
    if exc_count > 0 {
        let raw_left_width = (T::BITS as u8) - right_bw;
        let mut ep_r =
            BitReader::new(unsafe { std::slice::from_raw_parts(exc_pos_ptr, len_exc_pos) });
        // Note: for exc_val, we can just give it the remainder of the file or expected length
        let val_len = bytes.len() - (p + len_left + len_right + len_exc_pos);
        let mut ev_r = BitReader::new(unsafe { std::slice::from_raw_parts(exc_val_ptr, val_len) });

        for _ in 0..exc_count {
            let pos = ep_r.read(exc_pos_bw) as usize;
            let val_left = ev_r.read(raw_left_width);

            if pos < out.len() {
                let existing_bits_t = T::to_bits(out[pos]);
                let existing_u64 = T::to_u64(existing_bits_t);

                // Guard against shift overflow if right_bw == 64
                let new_full_u64 = if right_bw >= 64 {
                    existing_u64
                } else {
                    let mask = (1u64 << right_bw) - 1;
                    let right_part = existing_u64 & mask;
                    (val_left << right_bw) | right_part
                };

                out[pos] = T::from_bits(T::from_u64(new_full_u64));
            }
        }
    }

    out
}
