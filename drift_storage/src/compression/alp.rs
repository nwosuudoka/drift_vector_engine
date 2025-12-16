// ============================================================
//  Constants & Lookup Tables
// ============================================================

use crate::compression::alp_rd::BitWriter;

const MAX_EXPONENT: usize = 18; // Practical limit for IEEE754 doubles integer precision
const EXCEPTION_SIZE_BITS: usize = 64 + 16; // 64-bit value + 16-bit position
#[allow(dead_code)]
const SAMPLE_SIZE: usize = 32; // Defined in Section 4 "Sampling Parameters"

// Precomputed powers of 10 for fast access
static POW10: [f64; 20] = [
    1.0e0, 1.0e1, 1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6, 1.0e7, 1.0e8, 1.0e9, 1.0e10, 1.0e11, 1.0e12,
    1.0e13, 1.0e14, 1.0e15, 1.0e16, 1.0e17, 1.0e18, 1.0e19,
];

static INV_POW10: [f64; 20] = [
    1.0e0, 1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9, 1.0e-10,
    1.0e-11, 1.0e-12, 1.0e-13, 1.0e-14, 1.0e-15, 1.0e-16, 1.0e-17, 1.0e-18, 1.0e-19,
];

// ============================================================
//  Helper Functions
// ============================================================

/// The "Fast Rounding" trick described in Section 3.1.
/// Adds a "magic number" to push the value into the range where IEEE754
/// doubles represent integers exactly, effectively rounding it.
#[inline(always)]
#[allow(dead_code)]
fn fast_round(val: f64) -> i64 {
    // The magic number for standard doubles to round to nearest integer.
    // Works for values < 2^51.
    const MAGIC: f64 = 6755399441055744.0; // 2^51 + 2^52
    let biased = val + MAGIC;
    let raw = biased.to_bits();
    // Mask out the exponent part of the magic number implies a cast to integer
    // This is a Rust equivalent of the C cast trick
    (raw as i64) - (MAGIC.to_bits() as i64)
}

/// Calculate bit width of a number (zigzag encoded or raw unsigned).
#[allow(dead_code)]
fn signed_bit_width(v: i64) -> u8 {
    if v == 0 {
        return 1;
    }
    // Zigzag encode to handle negative numbers in bit packing
    let zigzag = ((v >> 63) ^ (v << 1)) as u64;
    64 - zigzag.leading_zeros() as u8
}

fn unsigned_bit_width(v: u64) -> u8 {
    if v == 0 {
        1
    } else {
        64 - v.leading_zeros() as u8
    }
}

// ============================================================
//  ALP Core Logic
// ============================================================

#[derive(Debug, Clone, Copy)]
struct CompressionConfig {
    exp: u8,
    factor: u8,
}

/// Estimates the number of bits needed to compress the vector with specific e and f.
/// This corresponds to the adaptive sampling described in Section 3.2.
fn estimate_compression_cost(values: &[f64], exp: u8, factor: u8) -> usize {
    let mut exceptions = 0;
    let mut max_encoded = 0;
    let mut min_encoded = i64::MAX;

    let exp_pow = POW10[exp as usize];
    let fact_inv_pow = INV_POW10[factor as usize];
    let fact_pow = POW10[factor as usize];
    let exp_inv_pow = INV_POW10[exp as usize];

    for &val in values {
        // Formula (1): ALPenc = round(n * 10^e * 10^-f)
        let encoded_val = (val * exp_pow * fact_inv_pow).round() as i64;

        // Formula (2): ALPdec = d * 10^f * 10^-e
        let recovered = (encoded_val as f64) * fact_pow * exp_inv_pow;

        // Check if lossless (exact bit match)
        if recovered.to_bits() != val.to_bits() {
            exceptions += 1;
        } else {
            if encoded_val > max_encoded {
                max_encoded = encoded_val;
            }
            if encoded_val < min_encoded {
                min_encoded = encoded_val;
            }
        }
    }

    if min_encoded == i64::MAX {
        return usize::MAX;
    } // All exceptions

    let delta = (max_encoded as i128 - min_encoded as i128) as u64;
    let bit_width = unsigned_bit_width(delta) as usize;

    // Cost = (bits for packed integers) + (overhead for exceptions)
    let packed_cost = values.len() * bit_width;
    let exception_cost = exceptions * EXCEPTION_SIZE_BITS;

    packed_cost + exception_cost
}

/// Adaptive Sampling: Finds the best e and f for the vector.
/// (Simplified greedy version of Section 3.2).
fn find_best_parameters(values: &[f64]) -> CompressionConfig {
    let mut best_cost = usize::MAX;
    let mut best_config = CompressionConfig { exp: 0, factor: 0 };

    // We check a sparse set of exponents to be fast, or check all if vector is small.
    // The paper suggests max exponent around 16 is usually best.
    for exp in 0..=MAX_EXPONENT as u8 {
        let cost = estimate_compression_cost(values, exp, 0);
        if cost < best_cost {
            best_cost = cost;
            best_config = CompressionConfig { exp, factor: 0 };
        }

        // Optimization: Try to cut trailing zeros (Factor f)
        // We only try f if e is high, as described in Section 2.6
        if exp > 0 {
            // Try a few factors f <= exp
            for factor in 1..=exp {
                let cost_f = estimate_compression_cost(values, exp, factor);
                if cost_f < best_cost {
                    best_cost = cost_f;
                    best_config = CompressionConfig { exp, factor };
                }
            }
        }
    }
    best_config
}

// ============================================================
//  ALP Encoder
// ============================================================

pub fn alp_encode(values: &[f64]) -> Vec<u8> {
    if values.is_empty() {
        return vec![];
    }

    // 1. Find best exponent (e) and factor (f)
    // In a full implementation, you might sample; here we scan the vector (safe for small vec sizes like 1024)
    let config = find_best_parameters(values);
    let exp = config.exp;
    let factor = config.factor;

    // 2. Encoding Pass
    let exp_pow = POW10[exp as usize];
    let fact_inv_pow = INV_POW10[factor as usize];
    let fact_pow = POW10[factor as usize];
    let exp_inv_pow = INV_POW10[exp as usize];

    let mut encoded_integers = Vec::with_capacity(values.len());
    let mut exceptions = Vec::new();
    let mut exception_positions = Vec::new();

    // First pass: Encode and detect exceptions
    for (i, &val) in values.iter().enumerate() {
        let encoded_val = (val * exp_pow * fact_inv_pow).round() as i64;
        let recovered = (encoded_val as f64) * fact_pow * exp_inv_pow;

        if recovered.to_bits() != val.to_bits() {
            // Exception: we will store a placeholder in the stream
            // and the real value in the exception list.
            // Placeholder can be anything, ideally similar to neighbors for delta compression,
            // but for simplicity here we assume the previous valid value or 0.
            let placeholder = if let Some(&last) = encoded_integers.last() {
                last
            } else {
                0
            };
            encoded_integers.push(placeholder);
            exceptions.push(val);
            exception_positions.push(i as u16);
        } else {
            encoded_integers.push(encoded_val);
        }
    }

    // 3. FFOR (Fused Frame-Of-Reference)
    // Find min value to subtract (Frame of Reference)
    let min_val = encoded_integers.iter().min().copied().unwrap_or(0);

    // Subtract min and calculate bit width required
    let mut deltas = Vec::with_capacity(encoded_integers.len());
    let mut max_delta = 0;

    for &val in &encoded_integers {
        let delta = (val as i128 - min_val as i128) as u64; // Handle potential overflow if using raw i64 subtraction
        if delta > max_delta {
            max_delta = delta;
        }
        deltas.push(delta);
    }

    let bit_width = unsigned_bit_width(max_delta);

    // 4. Bit Packing
    let est_size = (values.len() * bit_width as usize) / 8 + exceptions.len() * 10 + 128;
    let mut writer = BitWriter::new_with_capacity(est_size);

    // --- Header ---
    // [Exp: 8] [Factor: 8] [BitWidth: 8] [Exception Count: 16] [FOR Min: 64]
    writer.write(exp as u64, 8);
    writer.write(factor as u64, 8);
    writer.write(bit_width as u64, 8);
    writer.write(exceptions.len() as u64, 16);
    writer.write(min_val as u64, 64); // Store FOR base as raw u64 (bit cast)

    // --- Body (Packed Deltas) ---
    for delta in deltas {
        writer.write(delta, bit_width);
    }
    writer.flush();

    // --- Exceptions ---
    // Positions (16 bits each)
    for &pos in &exception_positions {
        writer.write(pos as u64, 16);
    }
    // Values (64 bits each)
    for &exc in &exceptions {
        writer.write(exc.to_bits(), 64);
    }
    writer.flush();

    writer.into_bytes()
}

// ============================================================
//  ALP Decoder
// ============================================================

// ============================================================
//  Enhanced Bit IO (With Alignment)
// ============================================================

struct BitReader<'a> {
    ptr: *const u8,
    end: *const u8,
    buffer: u128,
    bits_count: u32,
    // NEW: Track total bits read to handle byte-alignment
    total_bits_read: usize,
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
            total_bits_read: 0,
            _marker: std::marker::PhantomData,
        }
    }

    #[inline(always)]
    fn refill(&mut self) {
        // (Existing refill logic remains the same)
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
        self.total_bits_read += width as usize; // Track usage
        val
    }

    /// Skips bits until the next byte boundary.
    /// Call this if the Writer called `flush()`.
    fn align_to_byte(&mut self) {
        let remainder = self.total_bits_read % 8;
        if remainder > 0 {
            let bits_to_skip = 8 - remainder;
            self.read(bits_to_skip as u8);
        }
    }
}

// ============================================================
//  Corrected ALP Decoder
// ============================================================

pub fn alp_decode(bytes: &[u8], count: usize) -> Vec<f64> {
    if bytes.is_empty() || count == 0 {
        return vec![];
    }

    let mut reader = BitReader::new(bytes);
    let mut out = Vec::with_capacity(count);

    // 1. Read Header
    let exp = reader.read(8) as u8;
    let factor = reader.read(8) as u8;
    let bit_width = reader.read(8) as u8;
    let exc_count = reader.read(16) as usize;
    let min_val_u64 = reader.read(64);
    let min_val = min_val_u64 as i64;

    let fact_pow = POW10[factor as usize];
    let exp_inv_pow = INV_POW10[exp as usize];

    // 2. Decode Main Stream (Deltas)
    // The paper's FFOR layout packs these sequentially[cite: 423].
    for _ in 0..count {
        let delta = reader.read(bit_width);
        let encoded_val = min_val.wrapping_add(delta as i64);
        // Formula (2): d * 10^f * 10^-e [cite: 254]
        let val = (encoded_val as f64) * fact_pow * exp_inv_pow;
        out.push(val);
    }

    // ALIGNMENT: The encoder flushed after writing deltas.
    // We must skip any padding bits to reach the next section.
    reader.align_to_byte();

    // 3. Read Exception Positions
    // Exceptions are stored in a separate segment[cite: 412].
    // We must read ALL positions first because that is how they were written.
    let mut positions = Vec::with_capacity(exc_count);
    for _ in 0..exc_count {
        positions.push(reader.read(16) as usize);
    }

    // ALIGNMENT: The encoder flushed after writing positions.
    reader.align_to_byte();

    // 4. Read Exception Values and Patch
    // Values are stored uncompressed (64 bits)[cite: 407].
    for &pos in &positions {
        let val_bits = reader.read(64);
        let val = f64::from_bits(val_bits);

        // Patch the value into the output vector [cite: 505]
        if pos < out.len() {
            out[pos] = val;
        }
    }

    out
}
