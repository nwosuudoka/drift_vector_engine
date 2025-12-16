use bincode::{Decode, Encode};

// --- Constants ---
const BLOCK_SIZE: usize = 1024;
const TILE_COUNT: usize = 8;
const TILE_SIZE: usize = BLOCK_SIZE / TILE_COUNT; // 128 items per tile

// The "Unified" Order (Figure 6d in the FastLanes paper)
const UNIFIED_TILE_ORDER: [usize; 8] = [0, 4, 2, 6, 1, 5, 3, 7];

#[derive(Debug, Clone, PartialEq, Encode, Decode)]
pub struct FastLanesBlock<T> {
    pub dict: Vec<T>,
    pub lane_seeds: Vec<u32>,
    pub packed_deltas: Vec<u32>, // These are raw positive deltas (no ZigZag)
}

// ============================================================================
//  SECTION 1: THE ENCODER
// ============================================================================

pub fn encode<T: Clone + PartialEq + Eq + std::hash::Hash>(data: &[T]) -> FastLanesBlock<T> {
    assert_eq!(
        data.len(),
        BLOCK_SIZE,
        "FastLanes requires exactly 1024 items."
    );

    // 1. Run-Length Encoding
    // Result: A Dictionary (values) and an Index Vector (Run IDs).
    // Note: The Index Vector is strictly monotonic (0, 0, 1, 1, 1, 2...).
    let (run_values, index_vector) = perform_rle(data);

    // 2. Extract Seeds
    // We grab the starting value for each tile (Indices 0, 512, 256...)
    let mut seeds = Vec::with_capacity(TILE_COUNT);
    for &tile_idx in UNIFIED_TILE_ORDER.iter() {
        let global_start_index = tile_idx * TILE_SIZE;
        seeds.push(index_vector[global_start_index]);
    }

    // 3. Unified Transpose & Interleaved Delta Encoding
    // We iterate Row-by-Row.
    // Since input is monotonic, Current >= Prev, so Delta is always >= 0.
    let mut packed_deltas = Vec::with_capacity(BLOCK_SIZE);

    // We need to track the "previous value" for each lane independently.
    // Initialize with the seeds.
    let mut lane_cursors = seeds.clone();

    for row in 0..TILE_SIZE {
        for (lane_idx, &tile_id) in UNIFIED_TILE_ORDER.iter().enumerate() {
            // Find the actual value in the source data
            let global_index = (tile_id * TILE_SIZE) + row;
            let current_val = index_vector[global_index];

            if row == 0 {
                // First element of the lane is the seed.
                // Delta is implicitly 0 because Decoder starts at Seed.
                packed_deltas.push(0);
            } else {
                // Standard Delta: Current - Prev
                // Since data is monotonic, this never underflows/wraps negatively.
                let delta = current_val.wrapping_sub(lane_cursors[lane_idx]);
                packed_deltas.push(delta);

                // Update cursor for this lane
                lane_cursors[lane_idx] = current_val;
            }
        }
    }

    FastLanesBlock {
        dict: run_values,
        lane_seeds: seeds,
        packed_deltas,
    }
}

fn perform_rle<T: Clone + PartialEq + Eq + std::hash::Hash>(data: &[T]) -> (Vec<T>, Vec<u32>) {
    let mut run_values = Vec::new();
    let mut index_vector = Vec::with_capacity(data.len());

    if data.is_empty() {
        return (run_values, index_vector);
    }

    let mut current_val = &data[0];
    let mut current_run_id = 0;
    run_values.push(current_val.clone());

    for item in data {
        if item != current_val {
            current_val = item;
            current_run_id += 1;
            run_values.push(item.clone());
        }
        index_vector.push(current_run_id);
    }

    (run_values, index_vector)
}

// ============================================================================
//  SECTION 2: THE DECODER
// ============================================================================

pub fn decode<T: Clone>(block: &FastLanesBlock<T>) -> Vec<T> {
    let indices = decode_indices_auto(block);

    // Safety check: Ensure we don't access invalid dictionary entries
    indices
        .into_iter()
        .map(|idx| {
            if (idx as usize) < block.dict.len() {
                block.dict[idx as usize].clone()
            } else {
                block.dict[0].clone() // Fallback
            }
        })
        .collect()
}

fn decode_indices_auto<T>(block: &FastLanesBlock<T>) -> Vec<u32> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { decode_indices_avx2(block) };
        }
    }
    decode_indices_scalar(block)
}

fn decode_indices_scalar<T>(block: &FastLanesBlock<T>) -> Vec<u32> {
    let mut reordered_buffer = vec![0u32; BLOCK_SIZE];

    // Initialize accumulators with the seeds (Row 0 values)
    let mut accumulators = block.lane_seeds.clone();

    // chunks(8) corresponds to one row of interleaved deltas (one for each lane)
    let chunks = block.packed_deltas.chunks(TILE_COUNT);

    for (i, chunk) in chunks.enumerate() {
        for lane in 0..TILE_COUNT {
            let delta = chunk[lane];
            // Apply delta
            accumulators[lane] = accumulators[lane].wrapping_add(delta);

            // Store in interleaved buffer
            reordered_buffer[i * TILE_COUNT + lane] = accumulators[lane];
        }
    }

    restore_unified_order(&reordered_buffer)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn decode_indices_avx2<T>(block: &FastLanesBlock<T>) -> Vec<u32> {
    use std::arch::x86_64::*;

    let mut reordered_buffer = vec![0u32; BLOCK_SIZE];

    // Load Seeds
    let mut accumulator = _mm256_loadu_si256(block.lane_seeds.as_ptr() as *const __m256i);

    // Process Payload (256 bits at a time)
    // There are 1024 items.
    // packed_deltas has 1024 items.
    // _mm256 loads 8 x u32 items.
    // So we iterate 1024 / 8 = 128 times.
    let iterations = block.packed_deltas.len() / 8;

    for i in 0..iterations {
        let deltas_ptr = block.packed_deltas.as_ptr().add(i * 8);
        let deltas = _mm256_loadu_si256(deltas_ptr as *const __m256i);

        accumulator = _mm256_add_epi32(accumulator, deltas);

        _mm256_storeu_si256(
            reordered_buffer.as_mut_ptr().add(i * 8) as *mut __m256i,
            accumulator,
        );
    }

    restore_unified_order(&reordered_buffer)
}

fn restore_unified_order(interleaved: &[u32]) -> Vec<u32> {
    let mut original = vec![0u32; BLOCK_SIZE];

    for row in 0..TILE_SIZE {
        for (lane_id, &tile_idx) in UNIFIED_TILE_ORDER.iter().enumerate() {
            let src_idx = (row * TILE_COUNT) + lane_id;
            let dest_idx = (tile_idx * TILE_SIZE) + row;
            original[dest_idx] = interleaved[src_idx];
        }
    }
    original
}

// ============================================================================
//  SECTION 3: SERIALIZATION (Bit Packing)
// ============================================================================

pub fn serialize_to_writer<W, T>(block: &FastLanesBlock<T>, writer: &mut W) -> std::io::Result<()>
where
    W: std::io::Write,
    T: Encode,
{
    // 1. Write Dictionary (Bincode)
    let dict_bytes = bincode::encode_to_vec(&block.dict, bincode::config::standard())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    writer.write_all(&(dict_bytes.len() as u64).to_le_bytes())?;
    writer.write_all(&dict_bytes)?;

    // 2. Write Seeds (standard u32s)
    for seed in &block.lane_seeds {
        writer.write_all(&seed.to_le_bytes())?;
    }

    // 3. Bit-Packing Logic
    // Find max delta to determine bit-width
    let max_delta = block.packed_deltas.iter().max().copied().unwrap_or(0);

    // We don't need ZigZag logic here because deltas are strictly positive.
    let bit_width = if max_delta == 0 {
        0
    } else {
        (32 - max_delta.leading_zeros()) as u8
    };

    writer.write_all(&[bit_width])?;

    if bit_width > 0 {
        let packed_bytes = pack_interleaved(&block.packed_deltas, bit_width);
        writer.write_all(&packed_bytes)?;
    }

    Ok(())
}

pub fn deserialize_from_reader<R, T>(reader: &mut R) -> std::io::Result<FastLanesBlock<T>>
where
    R: std::io::Read,
    T: Decode<()>,
{
    // 1. Read Dictionary
    let mut len_buf = [0u8; 8];
    reader.read_exact(&mut len_buf)?;
    let dict_byte_len = u64::from_le_bytes(len_buf) as usize;

    let mut dict_raw = vec![0u8; dict_byte_len];
    reader.read_exact(&mut dict_raw)?;

    let (dict, _): (Vec<T>, usize) =
        bincode::decode_from_slice(&dict_raw, bincode::config::standard())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    // 2. Read Seeds
    let mut seeds = Vec::with_capacity(TILE_COUNT);
    let mut seed_buf = [0u8; 4];
    for _ in 0..TILE_COUNT {
        reader.read_exact(&mut seed_buf)?;
        seeds.push(u32::from_le_bytes(seed_buf));
    }

    // 3. Read Bit Width & Payload
    let mut width_buf = [0u8; 1];
    reader.read_exact(&mut width_buf)?;
    let bit_width = width_buf[0];

    let packed_deltas = if bit_width == 0 {
        vec![0u32; BLOCK_SIZE]
    } else {
        let total_bits = BLOCK_SIZE * (bit_width as usize);
        let total_bytes = (total_bits + 7) / 8;
        let mut packed_buf = vec![0u8; total_bytes];
        reader.read_exact(&mut packed_buf)?;

        unpack_interleaved(&packed_buf, bit_width)
    };

    Ok(FastLanesBlock {
        dict,
        lane_seeds: seeds,
        packed_deltas,
    })
}

// --- Bit Packing Implementation ---

/// Packs a slice of u32s into a Vec<u8> using `bit_width` bits per item.
fn pack_interleaved(values: &[u32], bit_width: u8) -> Vec<u8> {
    let mut buffer = Vec::new();
    let mut pending_bits: u64 = 0;
    let mut pending_count = 0;

    for &v in values {
        // Mask value to ensure it fits (safety)
        let masked_v = (v as u64) & ((1 << bit_width) - 1);

        pending_bits |= masked_v << pending_count;
        pending_count += bit_width;

        // Flush bytes
        while pending_count >= 8 {
            buffer.push(pending_bits as u8);
            pending_bits >>= 8;
            pending_count -= 8;
        }
    }

    // Flush remaining bits
    if pending_count > 0 {
        buffer.push(pending_bits as u8);
    }

    buffer
}

/// Unpacks a byte slice into Vec<u32> given a `bit_width`.
fn unpack_interleaved(packed: &[u8], bit_width: u8) -> Vec<u32> {
    let mut result = Vec::with_capacity(BLOCK_SIZE);
    let mut bit_cursor = 0;

    // Mask to extract exactly `bit_width` bits
    let value_mask = (1u64 << bit_width) - 1;

    for _ in 0..BLOCK_SIZE {
        let byte_idx = bit_cursor / 8;
        let bit_offset = bit_cursor % 8;

        // Read enough bytes to cover the bit width (max 32 bits, so 5 bytes is safe cover)
        let mut raw_u64 = 0u64;
        for i in 0..5 {
            if byte_idx + i < packed.len() {
                raw_u64 |= (packed[byte_idx + i] as u64) << (i * 8);
            }
        }

        // Shift down to align the target value to bit 0
        let val = (raw_u64 >> bit_offset) & value_mask;
        result.push(val as u32);

        bit_cursor += bit_width as usize;
    }

    result
}

// ============================================================================
//  SECTION 4: TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn test_encoding_decoding_roundtrip() {
        let mut data = Vec::new();
        for i in 0..BLOCK_SIZE {
            let char_code = (b'A' + (i / 100) as u8) as char;
            data.push(char_code.to_string());
        }

        let block = encode(&data);
        let decoded = decode(&block);
        assert_eq!(data, decoded);
    }

    #[test]
    fn test_serialization() {
        let data: Vec<String> = (0..BLOCK_SIZE).map(|i| i.to_string()).collect();
        let block = encode(&data);

        let mut buffer = Vec::new();
        serialize_to_writer(&block, &mut buffer).unwrap();

        let mut reader = &buffer[..];
        let loaded_block: FastLanesBlock<String> = deserialize_from_reader(&mut reader).unwrap();
        let decoded = decode(&loaded_block);

        assert_eq!(data, decoded);
    }

    #[test]
    fn fuzz_test_random_integers() {
        let mut rng = rand::rng();
        for _ in 0..20 {
            let mut data = Vec::new();
            for _ in 0..BLOCK_SIZE {
                let val: u32 = rng.random_range(0..50);
                data.push(val);
            }
            let block = encode(&data);

            // Check serialization logic
            let mut buffer = Vec::new();
            serialize_to_writer(&block, &mut buffer).unwrap();
            let mut reader = &buffer[..];
            let loaded: FastLanesBlock<u32> = deserialize_from_reader(&mut reader).unwrap();

            let decoded = decode(&loaded);
            assert_eq!(data, decoded);
        }
    }
}
