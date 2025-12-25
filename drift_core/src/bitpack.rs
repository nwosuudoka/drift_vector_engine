#![allow(dead_code)]

#[inline]
pub fn required_bits_u32(input: &[u32]) -> usize {
    let mut max = 0u32;
    for &v in input {
        max |= v;
    }

    if max == 0 {
        1
    } else {
        32 - max.leading_zeros() as usize
    }
}

pub fn pack_u32_dynamic(input: &[u32], out: &mut [u32]) -> (usize, usize) {
    let w = required_bits_u32(input);
    let words = match w {
        1 => pack_u32_scalar::<1>(input, out),
        2 => pack_u32_scalar::<2>(input, out),
        3 => pack_u32_scalar::<3>(input, out),
        4 => pack_u32_scalar::<4>(input, out),
        5 => pack_u32_scalar::<5>(input, out),
        6 => pack_u32_scalar::<6>(input, out),
        7 => pack_u32_scalar::<7>(input, out),
        8 => pack_u32_scalar::<8>(input, out),
        9 => pack_u32_scalar::<9>(input, out),
        10 => pack_u32_scalar::<10>(input, out),
        11 => pack_u32_scalar::<11>(input, out),
        12 => pack_u32_scalar::<12>(input, out),
        13 => pack_u32_scalar::<13>(input, out),
        14 => pack_u32_scalar::<14>(input, out),
        15 => pack_u32_scalar::<15>(input, out),
        16 => pack_u32_scalar::<16>(input, out),
        17 => pack_u32_scalar::<17>(input, out),
        18 => pack_u32_scalar::<18>(input, out),
        19 => pack_u32_scalar::<19>(input, out),
        20 => pack_u32_scalar::<20>(input, out),
        21 => pack_u32_scalar::<21>(input, out),
        22 => pack_u32_scalar::<22>(input, out),
        23 => pack_u32_scalar::<23>(input, out),
        24 => pack_u32_scalar::<24>(input, out),
        25 => pack_u32_scalar::<25>(input, out),
        26 => pack_u32_scalar::<26>(input, out),
        27 => pack_u32_scalar::<27>(input, out),
        28 => pack_u32_scalar::<28>(input, out),
        29 => pack_u32_scalar::<29>(input, out),
        30 => pack_u32_scalar::<30>(input, out),
        31 => pack_u32_scalar::<31>(input, out),
        32 => pack_u32_scalar::<32>(input, out),
        _ => unreachable!(),
    };

    (w, words)
}

pub fn unpack_u32_dynamic(packed: &[u32], n: usize, w: usize, out: &mut [u32]) {
    match w {
        1 => unpack_u32_scalar::<1>(packed, n, out),
        2 => unpack_u32_scalar::<2>(packed, n, out),
        3 => unpack_u32_scalar::<3>(packed, n, out),
        4 => unpack_u32_scalar::<4>(packed, n, out),
        5 => unpack_u32_scalar::<5>(packed, n, out),
        6 => unpack_u32_scalar::<6>(packed, n, out),
        7 => unpack_u32_scalar::<7>(packed, n, out),
        8 => unpack_u32_scalar::<8>(packed, n, out),
        9 => unpack_u32_scalar::<9>(packed, n, out),
        10 => unpack_u32_scalar::<10>(packed, n, out),
        11 => unpack_u32_scalar::<11>(packed, n, out),
        12 => unpack_u32_scalar::<12>(packed, n, out),
        13 => unpack_u32_scalar::<13>(packed, n, out),
        14 => unpack_u32_scalar::<14>(packed, n, out),
        15 => unpack_u32_scalar::<15>(packed, n, out),
        16 => unpack_u32_scalar::<16>(packed, n, out),
        17 => unpack_u32_scalar::<17>(packed, n, out),
        18 => unpack_u32_scalar::<18>(packed, n, out),
        19 => unpack_u32_scalar::<19>(packed, n, out),
        20 => unpack_u32_scalar::<20>(packed, n, out),
        21 => unpack_u32_scalar::<21>(packed, n, out),
        22 => unpack_u32_scalar::<22>(packed, n, out),
        23 => unpack_u32_scalar::<23>(packed, n, out),
        24 => unpack_u32_scalar::<24>(packed, n, out),
        25 => unpack_u32_scalar::<25>(packed, n, out),
        26 => unpack_u32_scalar::<26>(packed, n, out),
        27 => unpack_u32_scalar::<27>(packed, n, out),
        28 => unpack_u32_scalar::<28>(packed, n, out),
        29 => unpack_u32_scalar::<29>(packed, n, out),
        30 => unpack_u32_scalar::<30>(packed, n, out),
        31 => unpack_u32_scalar::<31>(packed, n, out),
        32 => unpack_u32_scalar::<32>(packed, n, out),
        _ => panic!("invalid bit width"),
    }
}

/*
    Scalar bit-packing / unpacking for u32 values.

    Design goals:
    - Very fast scalar implementation
    - Supports arbitrary N
    - Constant-bit-width per block
    - Suitable as a tail path next to SIMD / FastLanes

    Public API:
        pack_u32_scalar::<W>(input, out) -> words_written
        unpack_u32_scalar::<W>(input, n, out)
*/

/// Packs `input.len()` u32 values into a linear bitstream.
/// Each value uses exactly `W` bits.
///
/// Output is written as u32 words into `out`.
/// Returns the number of u32 words written.
///
/// SAFETY / CONTRACT:
/// - `W` must be in 1..=32
/// - All input values must fit in `W` bits
/// - `out` must be large enough: ceil(input.len() * W / 32)
pub fn pack_u32_scalar<const W: usize>(input: &[u32], out: &mut [u32]) -> usize {
    assert!(W > 0 && W <= 32);

    let mask: u64 = if W == 32 { u64::MAX } else { (1u64 << W) - 1 };

    let mut bitbuf: u64 = 0;
    let mut bits: usize = 0;
    let mut out_idx: usize = 0;

    for &v in input {
        bitbuf |= ((v as u64) & mask) << bits;
        bits += W;

        if bits >= 32 {
            out[out_idx] = bitbuf as u32;
            out_idx += 1;

            bitbuf >>= 32;
            bits -= 32;
        }
    }

    if bits > 0 {
        out[out_idx] = bitbuf as u32;
        out_idx += 1;
    }

    out_idx
}

/// Unpacks `n` u32 values from a linear bitstream.
/// Each value uses exactly `W` bits.
///
/// SAFETY / CONTRACT:
/// - `W` must be in 1..=32
/// - `input` must contain enough words
/// - `out.len() >= n`
pub fn unpack_u32_scalar<const W: usize>(input: &[u32], n: usize, out: &mut [u32]) {
    assert!(W > 0 && W <= 32);
    assert!(out.len() >= n);

    let mask: u64 = if W == 32 { u64::MAX } else { (1u64 << W) - 1 };

    let mut bitbuf: u64 = 0;
    let mut bits: usize = 0;
    let mut in_idx: usize = 0;

    for i in 0..n {
        while bits < W {
            bitbuf |= (input[in_idx] as u64) << bits;
            bits += 32;
            in_idx += 1;
        }

        out[i] = (bitbuf & mask) as u32;
        bitbuf >>= W;
        bits -= W;
    }
}

/* -----------------------------------------------------------
   Helper utilities (used by tests)
----------------------------------------------------------- */

fn max_value_for_width(w: usize) -> u32 {
    if w == 32 { u32::MAX } else { (1u32 << w) - 1 }
}

fn packed_len(n: usize, w: usize) -> usize {
    (n * w + 31) / 32
}

/* -----------------------------------------------------------
   Tests
----------------------------------------------------------- */

#[cfg(test)]
mod tests {
    use super::*;

    /* -------------------------------------------------------
       1) Basic round-trip tests (all widths, small sizes)
    ------------------------------------------------------- */

    #[test]
    fn roundtrip_small_all_widths() {
        for n in 0..=64 {
            // try multiple patterns so required_bits_u32 varies
            for pattern in 0..4 {
                let input: Vec<u32> = (0..n)
                    .map(|i| match pattern {
                        0 => i as u32,
                        1 => (i * 3) as u32,
                        2 => (i * 17) as u32,
                        _ => (i * 7919) as u32,
                    })
                    .collect();

                let mut packed = vec![0u32; packed_len(n, 32)];

                let (w, words) = pack_u32_dynamic(&input, &mut packed);

                let mut output = vec![0u32; n];
                unpack_u32_dynamic(&packed[..words], n, w, &mut output);

                assert_eq!(input, output, "n={n}, pattern={pattern}, w={w}");
            }
        }
    }

    /* -------------------------------------------------------
       2) Boundary values (all zeros / all max)
    ------------------------------------------------------- */

    #[test]
    fn boundary_values() {
        for &value in &[0u32, u32::MAX] {
            for n in [0, 1, 2, 7, 32, 100] {
                let input = vec![value; n];
                let mut packed = vec![0u32; packed_len(n, 32)];

                let (w, words) = pack_u32_dynamic(&input, &mut packed);

                let mut output = vec![0u32; n];
                unpack_u32_dynamic(&packed[..words], n, w, &mut output);

                assert_eq!(input, output, "value={value}, n={n}, w={w}");
            }
        }
    }

    /* -------------------------------------------------------
       3) Word-boundary stress tests
    ------------------------------------------------------- */

    #[test]
    fn word_boundary_cases() {
        for n in [1, 2, 3, 7, 31, 32, 33, 63, 64, 65, 127, 128] {
            let input: Vec<u32> = (0..n).map(|i| ((i * 7919) ^ (i << 5)) as u32).collect();

            let mut packed = vec![0u32; packed_len(n, 32)];
            let (w, words) = pack_u32_dynamic(&input, &mut packed);

            let mut output = vec![0u32; n];
            unpack_u32_dynamic(&packed[..words], n, w, &mut output);

            assert_eq!(input, output, "n={n}, w={w}");
        }
    }

    /* -------------------------------------------------------
       4) Randomized fuzz test (deterministic RNG)
    ------------------------------------------------------- */

    #[test]
    fn randomized_fuzz() {
        let mut seed: u64 = 0x1234_5678_9ABC_DEF0;

        fn next_u32(seed: &mut u64) -> u32 {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            *seed as u32
        }

        for _ in 0..500 {
            let n = (next_u32(&mut seed) % 500) as usize;

            let mut input = vec![0u32; n];
            for v in &mut input {
                *v = next_u32(&mut seed);
            }

            let mut packed = vec![0u32; packed_len(n, 32)];
            let (w, words) = pack_u32_dynamic(&input, &mut packed);

            let mut output = vec![0u32; n];
            unpack_u32_dynamic(&packed[..words], n, w, &mut output);

            assert_eq!(input, output, "fuzz n={n}, w={w}");
        }
    }

    /* -------------------------------------------------------
       5) Packed-size invariant
    ------------------------------------------------------- */

    #[test]
    fn packed_size_is_correct() {
        for n in 0..=1000 {
            let input: Vec<u32> = (0..n).map(|i| (i * 13) as u32).collect();

            let mut packed = vec![0u32; packed_len(n, 32)];
            let (w, words) = pack_u32_dynamic(&input, &mut packed);

            let expected = packed_len(n, w);
            assert_eq!(words, expected, "n={n}, w={w}");
        }
    }
}
