use super::byte_shuffle;
use crate::buf_pool::BufferPool;
use crate::codec::filter::FilterImpl;
use crate::dtype::Dtype;
use core::ops::{BitAnd, BitXor, Shl, Shr};
use fearless_simd::{Bytes, Level as SimdLevel, Simd, SimdBase};

// Bitshuffle filter, derived from Bitshuffle by Kiyoshi Masui (MIT,
// https://github.com/kiyo-masui/bitshuffle) via its adaptation in
// C-Blosc2 (BSD-3-Clause, https://github.com/Blosc/c-blosc2).
// See the top-level NOTICE file for full attribution and license text.

/// Bit-shuffle filter.
///
/// Rearranges the bits of a numeric array so that bits sharing the same
/// byte-position *and* bit-within-byte across all elements are grouped into
/// contiguous runs. For data whose high bytes tend to repeat (the common case:
/// small-valued integers, floats with similar magnitudes, differenced signals)
/// this creates long runs of constant or near-constant bits that downstream
/// entropy coders (typically LZ/zstd) can compress far better than the
/// original element-interleaved layout.
///
/// # Layout
///
/// For `N` elements of `B` bytes each, view the input as an `(N, B, 8)` array
/// of bits `bit[n, b, i]` (element, byte within the element, bit within the
/// byte). Splitting `n = 8 * g + k` into 8-element groups `g in 0..G`, `G = N/8`,
/// the output is the same bits in `(B, 8, G, 8)` order:
///
/// ```text
/// out_bit[b, i, g, k] = bit[8 * g + k, b, i]
/// ```
///
/// That is, `B * 8` consecutive *bit-planes* of `G` bytes each, byte-plane-major:
/// byte `g` of bit-plane `(b, i)` packs bit `i` of byte `b` of the elements
/// `8 * g .. 8 * g + 8`, element `k`'s bit at bit position `k`. This is the
/// wire format of the reference `bitshuffle.c` (as used by c-blosc2).
///
/// # Algorithm
///
/// Encoding is two byte shuffles and a bit transpose, all out-of-place:
///
/// ```text
///   (N, B)          --byte shuffle, itemsize B-->  (B, N) = (B, G, 8)   byte-planes
///   (G, 8) per b    --byte shuffle, itemsize 8-->  (8, G)               8 rows per byte-plane
///   (8, G) per b    --transpose_bit_rows------->  (8, G)               bit-planes
/// ```
///
/// After the second byte shuffle, row `k` of byte-plane `b` holds byte `b` of
/// the elements `8 * g + k`. [`transpose_bit_rows`] then transposes
/// every column of 8 bytes (one per row) as an 8x8 bit matrix, so row `i`
/// packs bit `i` of those 8 bytes: bit-plane `(b, i)`.
///
/// Decoding runs the inverse steps in reverse order. The bit transpose is
/// self-inverse.
///
/// # Tail handling
///
/// Bit-shuffle groups 8 elements at a time. Any trailing `N mod 8` elements
/// that don't fill a group are copied verbatim at the end of the buffer,
/// exactly as in the reference C implementation.
#[derive(Default)]
pub(super) struct BitShuffleFilter;

impl FilterImpl for BitShuffleFilter {
    fn encode(&self, src: &[u8], dst: &mut [u8], dtype: &Dtype, buf_pool: &BufferPool) {
        assert_eq!(src.len(), dst.len());
        let typesize = dtype.itemsize() as usize;
        let n_full = src.len() / typesize / 8 * 8;
        let full_bytes = n_full * typesize;

        let mut tmp = buf_pool.get(n_full, 16.try_into().unwrap());
        let tmp = &mut tmp.as_mut_slice()[..n_full];

        byte_shuffle::encode(&src[..full_bytes], &mut dst[..full_bytes], typesize);
        // `max(1)`: no byte-plane at all when there is no full group.
        for plane in dst[..full_bytes].chunks_exact_mut(n_full.max(1)) {
            byte_shuffle::encode(plane, tmp, 8);
            transpose_bit_rows(tmp, plane);
        }

        // Tail: the final `N mod 8` elements are copied through verbatim.
        dst[full_bytes..].copy_from_slice(&src[full_bytes..]);
    }

    fn decode(&self, src: &[u8], dst: &mut [u8], dtype: &Dtype, buf_pool: &BufferPool) {
        assert_eq!(src.len(), dst.len());
        let typesize = dtype.itemsize() as usize;
        let n_full = src.len() / typesize / 8 * 8;
        let full_bytes = n_full * typesize;

        let mut tmp = buf_pool.get(full_bytes, 16.try_into().unwrap());
        let tmp = &mut tmp.as_mut_slice()[..full_bytes];

        let plane = n_full.max(1); // no byte-plane at all when there is no full group
        for ((src, dst), tmp) in src[..full_bytes]
            .chunks_exact(plane)
            .zip(dst[..full_bytes].chunks_exact_mut(plane))
            .zip(tmp.chunks_exact_mut(plane))
        {
            transpose_bit_rows(src, dst);
            byte_shuffle::decode(dst, tmp, 8);
        }
        byte_shuffle::decode(tmp, &mut dst[..full_bytes], typesize);

        dst[full_bytes..].copy_from_slice(&src[full_bytes..]);
    }
}

/// Bit-transpose the 8 rows of `src` into the 8 rows of `dst`, column by column: bit `i` of
/// `src` row `k` goes to bit `k` of `dst` row `i`. Both are 8 rows of `src.len() / 8` bytes.
///
/// Self-inverse, so it serves both directions of the filter.
fn transpose_bit_rows(src: &[u8], dst: &mut [u8]) {
    let done =
        fearless_simd::dispatch!(SimdLevel::new(), simd => transpose_bit_rows_simd(simd, src, dst));
    // Tail of the remaining columns, one at a time.
    let g = src.len() / 8;
    for j in done..g {
        let mut r = [0u64; 8];
        for (k, x) in r.iter_mut().enumerate() {
            *x = src[k * g + j].into();
        }
        transpose8x8_rows(&mut r);
        for (k, x) in r.iter().enumerate() {
            dst[k * g + j] = *x as u8;
        }
    }
}

/// Main loop of [`transpose_bit_rows`]: transposes whole vectors of columns, and returns the
/// number of columns transposed.
///
/// Optimized by static analysis of the generated asm (cargo-asm + llvm-mca, steady-state cycles
/// of the main loop), for x86-64 SSE4.2 (Sandy Bridge, Jaguar), AVX2 (Skylake, Alder Lake, Zen 3)
/// and AVX-512 (Ice Lake, Sapphire Rapids, Zen 4), i686 SSE2 (Skylake), and aarch64 NEON
/// (Cortex-A72, Neoverse N1 / V2, Apple M1).
#[inline(always)]
fn transpose_bit_rows_simd<S: Simd>(simd: S, src: &[u8], dst: &mut [u8]) -> usize {
    let g = src.len() / 8;
    assert!(dst.len() >= 8 * g);
    let lanes = S::u8s::LEN;
    let (src, dst) = (src.as_ptr(), dst.as_mut_ptr());
    for c in 0..g / lanes {
        let j = c * lanes;
        // SAFETY (all pointer accesses below): `k < 8` and `j + lanes <= g`, so each vector is in
        // row `k` of `src` / `dst`. `S::u8s::Array` is `[u8; lanes]`, with alignment 1.
        // Loading through `load_array_ref` rather than `from_slice` avoids a length `unwrap`
        // that LLVM does not always fold away.
        let mut r = [S::u64s::splat(simd, 0); 8];
        for (k, x) in r.iter_mut().enumerate() {
            *x = S::u8s::load_array_ref(simd, unsafe { &*src.add(k * g + j).cast() }).bitcast();
        }
        transpose8x8_rows(&mut r);
        for (k, x) in r.iter().enumerate() {
            x.bitcast::<S::u8s>()
                .store_array(unsafe { &mut *dst.add(k * g + j).cast() });
        }
    }
    g / lanes * lanes
}

/// Bit-transpose 8 rows, byte-wise: in every byte position, bit `i` of row `k` goes to bit `k` of
/// row `i`. A row is a `u64` or a vector of `u64`s, i.e. a run of bytes of one row.
///
/// The classic recursive transpose (Hacker's Delight 7-3), across rows: swap the off-diagonal
/// 4x4 blocks of every 8x8 bit matrix, then the 2x2 blocks within the 4x4s, then single bits.
/// Swapping blocks of size `s` exchanges the bits in the upper half (mask `!m`) of each `2s`-bit
/// group of row `k` with the bits in the lower half (mask `m`) of row `k + s`. Self-inverse.
#[inline(always)]
fn transpose8x8_rows<T: Rows>(r: &mut [T; 8]) {
    swap_blocks(r, 4, 0x0F0F_0F0F_0F0F_0F0F);
    swap_blocks(r, 2, 0x3333_3333_3333_3333);
    swap_blocks(r, 1, 0x5555_5555_5555_5555);
}

#[inline(always)]
fn swap_blocks<T: Rows>(r: &mut [T; 8], s: usize, m: u64) {
    for k in 0..8 {
        if k & s == 0 {
            let t = ((r[k] >> s as u32) ^ r[k + s]) & m;
            r[k + s] = r[k + s] ^ t;
            r[k] = r[k] ^ (t << s as u32);
        }
    }
}

/// The operations [`transpose8x8_rows`] needs: `u64` and fearless_simd's `u64` vectors.
pub trait Rows:
    Copy
    + Shr<u32, Output = Self>
    + Shl<u32, Output = Self>
    + BitXor<Output = Self>
    + BitAnd<u64, Output = Self>
{
}
impl<T> Rows for T where
    T: Copy
        + Shr<u32, Output = T>
        + Shl<u32, Output = T>
        + BitXor<Output = T>
        + BitAnd<u64, Output = T>
{
}

#[cfg(test)]
mod tests {
    use super::BitShuffleFilter;
    use crate::buf_pool::BufferPool;
    #[cfg(feature = "num-complex")]
    use crate::scalar::Complex;

    macro_rules! test_roundtrip {
        ($ty:ty, $fn_name:ident) => {
            #[test]
            fn $fn_name() {
                crate::codec::filter::tests::run_bytes_proptest::<$ty>(|data| {
                    crate::codec::filter::tests::test_roundtrip::<BitShuffleFilter, $ty>(data);
                });
            }
        };
    }

    // This filter operates on raw bytes keyed only by itemsize, so dtypes that
    // share a byte width run byte-identical code (same principle as the
    // `copy_tests!` dedup at `jix/src/util/nd_copy.rs:916`, e.g. i32/f32 both
    // hit the same 4-byte path as u32). Keep one representative dtype per
    // distinct itemsize actually exercised here: 1 (u8, also covers i8/bool),
    // 2 (u16, also covers i16/f16), 4 (u32, also covers i32/f32), 8 (u64, also
    // covers i64/f64/Complex<f32>), and 16 (Complex<f64>, not covered by any
    // narrower width).
    test_roundtrip!(u8, u8_roundtrip);
    test_roundtrip!(u16, u16_roundtrip);
    test_roundtrip!(u32, u32_roundtrip);
    test_roundtrip!(u64, u64_roundtrip);
    #[cfg(feature = "num-complex")]
    test_roundtrip!(Complex<f64>, complex_f64_roundtrip);

    /// Reference of `transpose_bit_rows`: bit `i` of row `k` <-> bit `k` of row `i`, per column.
    fn transpose_bit_rows_reference(src: &[u8]) -> Vec<u8> {
        let g = src.len() / 8;
        let mut dst = vec![0u8; src.len()];
        for j in 0..g {
            for i in 0..8 {
                for k in 0..8 {
                    let bit = (src[k * g + j] >> i) & 1;
                    dst[i * g + j] |= bit << k;
                }
            }
        }
        dst
    }

    proptest::proptest! {
        #[test]
        fn transpose8x8_rows(rows: [u64; 8]) {
            let src: Vec<u8> = rows.iter().flat_map(|r| r.to_le_bytes()).collect();
            let mut out = rows;
            super::transpose8x8_rows(&mut out);
            let out: Vec<u8> = out.iter().flat_map(|r| r.to_le_bytes()).collect();
            proptest::prop_assert_eq!(out, transpose_bit_rows_reference(&src));
            let mut twice = rows;
            super::transpose8x8_rows(&mut twice);
            super::transpose8x8_rows(&mut twice);
            proptest::prop_assert_eq!(twice, rows);
        }
    }

    /// `transpose_bit_rows_simd` at every SIMD level this CPU supports (not only the one
    /// `fearless_simd::dispatch!` picks) plus the dispatched function with its tail, against the reference.
    #[test]
    fn transpose_bit_rows_all_levels() {
        use super::{transpose_bit_rows, transpose_bit_rows_simd};
        use fearless_simd::{Level as SimdLevel, Simd};

        let lengths = [
            0, 1, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 300,
        ];
        let input = |g: usize| -> Vec<u8> { (0..8 * g).map(|x| (x * 37 + 11) as u8).collect() };
        for g in lengths {
            let src = input(g);
            let mut dst = vec![0u8; src.len()];
            transpose_bit_rows(&src, &mut dst);
            assert_eq!(dst, transpose_bit_rows_reference(&src), "g {g}");
        }

        fn check<S: Simd>(simd: S, lengths: &[usize], input: &dyn Fn(usize) -> Vec<u8>) {
            for &g in lengths {
                let src = input(g);
                let expected = transpose_bit_rows_reference(&src);
                let mut dst = expected.clone();
                let done = simd.vectorize(|| transpose_bit_rows_simd(simd, &src, &mut dst));
                assert_eq!(done, g / S::u8s::LEN * S::u8s::LEN);
                assert_eq!(dst, expected, "g {g}");
            }
        }
        use fearless_simd::SimdBase;
        let level = SimdLevel::new();
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if let Some(s) = level.as_sse2() {
                check(s, &lengths, &input);
            }
            if let Some(s) = level.as_sse4_2() {
                check(s, &lengths, &input);
            }
            if let Some(s) = level.as_avx2() {
                check(s, &lengths, &input);
            }
            if let Some(s) = level.as_avx512() {
                check(s, &lengths, &input);
            }
        }
        #[cfg(target_arch = "aarch64")]
        if let Some(s) = level.as_neon() {
            check(s, &lengths, &input);
        }
    }

    /// Trivial reference implementation of bitshuffle, for tests.
    ///
    /// This file intentionally has no passes, no intermediate buffers, and no
    /// bit-matrix tricks. It works directly from the mathematical definition
    ///
    /// ```text
    ///   out_bit[b, i, g, k] = in_bit[g*8 + k, b, i]
    /// ```
    ///
    /// where `b in 0..B` is the byte-plane, `i in 0..8` is the bit-plane, `g in 0..G`
    /// is the 8-element group, `k in 0..8` is the element within the group, and
    /// `G = N/8`. Concretely, the encoded byte at offset `b*8*G + i*G + g` packs
    /// bit `i` of byte `b` of the 8 elements `g*8 .. g*8+8`, with element `k`'s
    /// bit landing in bit-position `k` of the packed byte.
    ///
    /// Tail handling matches the three-pass implementation: the last `N mod 8`
    /// elements don't fill a group and are copied through verbatim.
    ///
    /// It is O(N * B * 8) with poor constants and no SIMD - strictly a test
    /// oracle, never called on the hot path.
    fn bit_shuffle_trivial(src: &[u8], dst: &mut [u8], typesize: usize) {
        assert_eq!(src.len(), dst.len());
        assert_eq!(src.len() % typesize, 0);

        let n = src.len() / typesize;
        let n_full = (n / 8) * 8;
        let g = n_full / 8;
        let full_bytes = n_full * typesize;

        // We'll OR bits into the destination, so start from zero.
        dst[..full_bytes].fill(0);

        // For every input bit in the "full" region, compute its destination byte
        // and bit-position and OR it in. This is the encoding definition,
        // transcribed.
        for element in 0..n_full {
            let group = element / 8;
            let k = element % 8;
            for b in 0..typesize {
                let byte = src[element * typesize + b];
                for i in 0..8 {
                    let bit = (byte >> i) & 1;
                    let dst_idx = b * 8 * g + i * g + group;
                    dst[dst_idx] |= bit << k;
                }
            }
        }

        // Tail: copy the trailing `N mod 8` elements verbatim.
        dst[full_bytes..].copy_from_slice(&src[full_bytes..]);
    }

    fn test_agrees_with_trivial<T: crate::dtype::Dtyped>(items: &[T]) {
        use crate::codec::filter::FilterImpl;
        use crate::util::gen_data_bytes_from_slice;

        let data = gen_data_bytes_from_slice::<T>(items);
        let src = data.as_slice();
        let typesize = T::DTYPE.itemsize() as usize;
        let dtype = T::DTYPE;
        let buf_pool = BufferPool::new();

        let mut optimized_out = vec![0u8; src.len()];
        BitShuffleFilter.encode(src, &mut optimized_out, &dtype, &buf_pool);

        let mut trivial_out = vec![0u8; src.len()];
        bit_shuffle_trivial(src, &mut trivial_out, typesize);

        assert_eq!(optimized_out, trivial_out);
    }

    macro_rules! test_agrees_with_trivial {
        ($ty:ty, $fn_name:ident) => {
            #[test]
            fn $fn_name() {
                crate::codec::filter::tests::run_bytes_proptest::<$ty>(|data| {
                    test_agrees_with_trivial::<$ty>(data);
                });
            }
        };
    }

    // Same itemsize-only dedup as the roundtrip macros above: one dtype per
    // distinct byte width (1/2/4/8/16), see `jix/src/util/nd_copy.rs:916`.
    test_agrees_with_trivial!(u8, u8_agrees_with_trivial);
    test_agrees_with_trivial!(u16, u16_agrees_with_trivial);
    test_agrees_with_trivial!(u32, u32_agrees_with_trivial);
    test_agrees_with_trivial!(u64, u64_agrees_with_trivial);
    #[cfg(feature = "num-complex")]
    test_agrees_with_trivial!(Complex<f64>, complex_f64_agrees_with_trivial);
}
