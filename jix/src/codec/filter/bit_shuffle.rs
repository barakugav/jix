use super::byte_shuffle::ByteShuffleFilter;
use crate::array_from_fn_inline;
use crate::buf_pool::BufferPool;
use crate::codec::filter::FilterImpl;
use crate::dtype::Dtype;

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
/// # Layout notation
///
/// For `N` elements of `B` bytes each, view the input as an `(N, B, 8)` array
/// of bits `bit[n, b, i]` where
///
/// * `n in 0..N` indexes the element,
/// * `b in 0..B` indexes the byte within an element (byte-plane),
/// * `i in 0..8` indexes the bit within a byte (bit-plane).
///
/// The encoder produces the same bits permuted into `(B, 8, N)` order:
///
/// ```text
/// out_bit[b, i, n] = bit[n, b, i]
/// ```
///
/// Concretely, the output is `B*8` consecutive *bit-planes* of `N/8` bytes
/// each, laid out byte-plane-major, bit-plane-minor, element-minor-most.
///
/// # Three-pass algorithm
///
/// The transposition is performed in three out-of-place passes, exactly
/// mirroring the reference `bitshuffle.c` (as used by c-blosc2). Letting
/// `G = N/8` denote the number of 8-element groups:
///
/// ```text
///            pass 1: byte-shuffle                  (AoS -> SoA)
///   src ----------------------------------------->  P1
///   (N, B, 8) bits                                  (B, N, 8) bits
///                                                 = (B, G, 8, 8) bits
///
///            pass 2: trans_bit_byte                 (TRANS_BIT_8X8 + scatter)
///   P1  ----------------------------------------->  P2
///   (B, G, 8, 8) bits                               (8, B, G, 8) bits
///   rows = elements-in-group                        rows = bit-within-byte
///   cols = bit-within-byte                          cols = elements-in-group
///
///            pass 3: trans_bitrow_eight             (outer-axis swap)
///   P2  ----------------------------------------->  dst
///   (8, B, G)   length-G byte runs                  (B, 8, G) length-G byte runs
/// ```
///
/// Pass 1 is handled entirely by [`ByteShuffleFilter`]: it transposes the
/// `(N, B)` AoS byte matrix to the `(B, N)` SoA byte matrix.
///
/// Pass 2 is where the actual bit-level work happens. It reads each contiguous
/// 8-byte group from the byte-shuffled buffer as a `u64`, interprets it as an
/// 8*8 bit matrix with **rows = elements** (0..8 within the group) and
/// **columns = bit-within-byte** (0..8), applies a constant-time bit-matrix
/// transpose (see [`transpose8x8`]), and scatters the 8 resulting bytes into
/// 8 separate bit-plane regions of the destination - so byte `k` of the
/// transposed group goes to the `k`-th bit-plane.
///
/// Pass 3 is pure data movement: it swaps the outer `(8, B)` axes of the
/// `(8, B, G)` byte array while keeping each length-`G` inner run intact, so
/// that the final layout is byte-plane-major with bit-planes nested inside
/// (the `(B, 8, G)` layout expected by the bitshuffle wire format).
///
/// # Decoding
///
/// Decoding is the exact inverse of encoding in reverse pass order. Pass 3
/// and pass 1 have trivial byte-level inverses (the reverse outer-axis swap
/// and [`ByteShuffleFilter::decode`] respectively). Pass 2's inverse gathers
/// 8 bytes from 8 different bit-plane regions and applies [`transpose8x8`]
/// again - because [`transpose8x8`] is self-inverse, a single primitive serves
/// both directions; only the surrounding data-movement pattern changes.
///
/// # Tail handling
///
/// Bit-shuffle groups 8 elements at a time (the pass-2 `u64` is exactly 8
/// bytes of a single byte-plane, i.e. 8 elements). Any trailing `N mod 8`
/// elements that don't fill a group are copied verbatim at the end of the
/// buffer, exactly as in the reference C implementation.
#[derive(Default)]
pub(super) struct BitShuffleFilter {
    byte_shuffle: ByteShuffleFilter,
}

impl FilterImpl for BitShuffleFilter {
    /// Encode: `(N, B)` element-major bytes -> `(B, 8, G)` bit-plane-major
    /// bytes, where `G = N/8`.
    ///
    /// Buffer routing mirrors `bitshuffle.c` exactly - three distinct buffers,
    /// ping-ponging so the final result lands back in `dst`:
    ///
    /// | pass | function                   | in    | out   |
    /// |------|----------------------------|-------|-------|
    /// | 1    | [`ByteShuffleFilter::encode`] (AoS->SoA) | `src` | `dst` |
    /// | 2    | [`trans_bit_byte`]         | `dst` | `tmp` |
    /// | 3    | [`trans_bitrow_eight`]     | `tmp` | `dst` |
    ///
    /// After pass 3 the final output is in `dst`; `tmp` is scratch and is
    /// discarded on return.
    fn encode(&self, src: &[u8], dst: &mut [u8], dtype: &Dtype, tmp_buffers: &BufferPool) {
        assert_eq!(src.len(), dst.len());
        let typesize = dtype.itemsize() as usize;
        let n = src.len() / typesize;
        let n_full = (n / 8) * 8;
        let full_bytes = n_full * typesize;

        let mut tmp = tmp_buffers.get(full_bytes, 16.try_into().unwrap());
        let tmp = tmp.as_mut_slice();

        // Pass 1: byte shuffle, `(N, B) -> (B, N)`. After this, `dst` holds per
        // byte-plane a contiguous run of N bytes (one byte per element).
        self.byte_shuffle.encode(
            &src[..full_bytes],
            &mut dst[..full_bytes],
            dtype,
            tmp_buffers,
        );

        // Pass 2: TRANS_BIT_8X8 + scatter, `(B, G, 8, 8) bits -> (8, B, G, 8) bits`.
        // For each 8-byte group within each byte-plane, bit-transpose the
        // group and distribute its 8 output bytes to 8 separate bit-plane
        // regions of `tmp`.
        trans_bit_byte(&dst[..full_bytes], tmp, n_full, typesize);

        // Pass 3: outer-axis swap, `(8, B, G) bytes -> (B, 8, G) bytes`.
        // Just moves length-`G` byte runs around to produce the final
        // byte-plane-major, bit-plane-minor layout.
        trans_bitrow_eight(tmp, &mut dst[..full_bytes], n_full, typesize);

        // Tail: the final `N mod 8` elements weren't processed; copy them
        // through verbatim so the decoder can recover them the same way.
        dst[full_bytes..].copy_from_slice(&src[full_bytes..]);
    }

    /// Decode: `(B, 8, G)` bit-plane-major bytes -> `(N, B)` element-major
    /// bytes. Each pass is the exact inverse of its encode counterpart,
    /// applied in reverse order.
    ///
    /// | pass | function                   | in    | out   | inverts        |
    /// |------|----------------------------|-------|-------|----------------|
    /// | 1    | [`untrans_bitrow_eight`]   | `src` | `dst` | encode pass 3  |
    /// | 2    | [`untrans_bit_byte`]       | `dst` | `tmp` | encode pass 2  |
    /// | 3    | [`ByteShuffleFilter::decode`] (SoA->AoS) | `tmp` | `dst` | encode pass 1 |
    fn decode(&self, src: &[u8], dst: &mut [u8], dtype: &Dtype, tmp_buffers: &BufferPool) {
        assert_eq!(src.len(), dst.len());
        let typesize = dtype.itemsize() as usize;
        let n = src.len() / typesize;
        let n_full = (n / 8) * 8;
        let full_bytes = n_full * typesize;

        let mut tmp = tmp_buffers.get(full_bytes, 16.try_into().unwrap());
        let tmp = tmp.as_mut_slice();

        // Pass 1: invert encode pass 3. `(B, 8, G) -> (8, B, G)`. Length-`G`
        // runs are moved; no bit-level work.
        untrans_bitrow_eight(&src[..full_bytes], &mut dst[..full_bytes], n_full, typesize);

        // Pass 2: invert encode pass 2. For each `(b, g)`, gather the 8 bytes
        // that live one-per-bit-plane, apply `transpose8x8` (self-inverse) to
        // undo the encode-time bit transpose, and write the resulting 8 bytes
        // contiguously as an 8-element group of byte-plane `b`.
        untrans_bit_byte(&dst[..full_bytes], tmp, n_full, typesize);

        // Pass 3: invert encode pass 1 via byte_shuffle's own decode (SoA -> AoS).
        self.byte_shuffle
            .decode(tmp, &mut dst[..full_bytes], dtype, tmp_buffers);

        // Tail was copied verbatim by the encoder; copy it back.
        dst[full_bytes..].copy_from_slice(&src[full_bytes..]);
    }
}

/// Encode pass 2 - combined bit-transpose and scatter.
///
/// Reads input in `(B, G, 8)`-byte layout and writes output in `(8, B, G)`-byte
/// layout, with a bit-level transpose applied in between.
///
/// **Indexing.**
///
/// * Input:  `src[b * N + g * 8 + k]` - byte-plane `b`, group `g`, byte-in-group `k`.
/// * Output: `dst[i * B*G + b * G + g]` - bit-plane `i`, byte-plane `b`, group `g`,
///   where the output stride between consecutive bit-planes is
///   `bit_row_skip = B * G = typesize * n_per_plane`.
///
/// **What actually happens in one iteration.** For each `(b, g)` we pull the
/// 8 consecutive input bytes `src[b*N + g*8 + 0..8]` into a `u64` via
/// [`transpose8x8`]. These 8 bytes are the 8 consecutive elements
/// `g*8 .. g*8+8` of byte-plane `b`. Viewed as an 8*8 bit matrix, rows index
/// elements within the group and columns index bit-position within a byte.
/// After [`transpose8x8`], rows index bit-position and columns index
/// element-within-group, so output byte `k` now contains bit `k` of those 8
/// elements - exactly what belongs in bit-plane `k` at position `(b, g)` of
/// the bit-plane-major output.
///
/// Equivalent to `bshuf_trans_bit_byte_scal` from the reference C
/// implementation on little-endian targets (the `u64` read + `TRANS_BIT_8X8`
/// + strided scatter pattern).
#[cfg_attr(feature = "multiversion", multiversion::multiversion(targets(
    // x86-64-v4
    "x86_64+sse3+ssse3+sse4.1+sse4.2+popcnt+cmpxchg16b+avx+avx2+bmi1+bmi2+f16c+fma+lzcnt+movbe+xsave+avx512f+avx512bw+avx512cd+avx512dq+avx512vl",
    // x86-64-v3
    "x86_64+sse3+ssse3+sse4.1+sse4.2+popcnt+cmpxchg16b+avx+avx2+bmi1+bmi2+f16c+fma+lzcnt+movbe+xsave",
    // x86-64-v2
    "x86_64+sse3+ssse3+sse4.1+sse4.2+popcnt+cmpxchg16b",
)))]
fn trans_bit_byte(src: &[u8], dst: &mut [u8], n_full: usize, typesize: usize) {
    let n_per_plane = n_full / 8;
    let bit_row_skip = typesize * n_per_plane; // = B * G

    for b in 0..typesize {
        for g in 0..n_per_plane {
            let src_off = b * n_full + g * 8;
            let group: [u8; 8] = array_from_fn_inline(|k| src[src_off + k]);

            // Bit-matrix transpose of the 8 bytes viewed as an 8*8 bit square.
            let transposed = transpose8x8(group);

            // Scatter: byte `k` of the transposed group goes to bit-plane `k`
            // at position (byte-plane `b`, group `g`). The 8 writes land in
            // 8 distant regions of the output, `bit_row_skip` bytes apart.
            for k in 0..8 {
                dst[k * bit_row_skip + b * n_per_plane + g] = transposed[k];
            }
        }
    }
}

/// Encode pass 3 - outer-axis swap.
///
/// Swaps the `(8, B)` outer axes of an `(8, B, G)` byte array, keeping each
/// length-`G` innermost run intact. No bits are permuted within a byte; this
/// pass is pure data movement via `copy_from_slice` on length-`G` runs.
///
/// * Input:  `src[i * B*G + b * G + g]` - bit-plane `i`, byte-plane `b`, group `g`.
/// * Output: `dst[b * 8*G + i * G + g]` - byte-plane `b`, bit-plane `i`, group `g`.
///
/// The final layout `(B, 8, G)` is the bitshuffle wire format: for each
/// byte-plane (outermost), the 8 bit-planes in order, each a contiguous run
/// of `G = N/8` bytes.
///
/// Equivalent to `bshuf_trans_bitrow_eight` in the reference, itself a
/// specialisation of `bshuf_trans_elem(lda=8, ldb=B, elem_size=G)`.
#[cfg_attr(feature = "multiversion", multiversion::multiversion(targets(
    // x86-64-v4
    "x86_64+sse3+ssse3+sse4.1+sse4.2+popcnt+cmpxchg16b+avx+avx2+bmi1+bmi2+f16c+fma+lzcnt+movbe+xsave+avx512f+avx512bw+avx512cd+avx512dq+avx512vl",
    // x86-64-v3
    "x86_64+sse3+ssse3+sse4.1+sse4.2+popcnt+cmpxchg16b+avx+avx2+bmi1+bmi2+f16c+fma+lzcnt+movbe+xsave",
    // x86-64-v2
    "x86_64+sse3+ssse3+sse4.1+sse4.2+popcnt+cmpxchg16b",
)))]
fn trans_bitrow_eight(src: &[u8], dst: &mut [u8], n_full: usize, typesize: usize) {
    let n_per_plane = n_full / 8;
    for i in 0..8 {
        for b in 0..typesize {
            let src_off = i * typesize * n_per_plane + b * n_per_plane;
            let dst_off = b * 8 * n_per_plane + i * n_per_plane;
            dst[dst_off..dst_off + n_per_plane]
                .copy_from_slice(&src[src_off..src_off + n_per_plane]);
        }
    }
}

/// Decode pass 1 - inverse of [`trans_bitrow_eight`].
///
/// `(B, 8, G) -> (8, B, G)` byte-level outer-axis swap. Pure data movement in
/// length-`G` runs; reads and writes are just the encode-side roles flipped.
#[cfg_attr(feature = "multiversion", multiversion::multiversion(targets(
    // x86-64-v4
    "x86_64+sse3+ssse3+sse4.1+sse4.2+popcnt+cmpxchg16b+avx+avx2+bmi1+bmi2+f16c+fma+lzcnt+movbe+xsave+avx512f+avx512bw+avx512cd+avx512dq+avx512vl",
    // x86-64-v3
    "x86_64+sse3+ssse3+sse4.1+sse4.2+popcnt+cmpxchg16b+avx+avx2+bmi1+bmi2+f16c+fma+lzcnt+movbe+xsave",
    // x86-64-v2
    "x86_64+sse3+ssse3+sse4.1+sse4.2+popcnt+cmpxchg16b",
)))]
fn untrans_bitrow_eight(src: &[u8], dst: &mut [u8], n_full: usize, typesize: usize) {
    let n_per_plane = n_full / 8;
    for b in 0..typesize {
        for i in 0..8 {
            let src_off = b * 8 * n_per_plane + i * n_per_plane;
            let dst_off = i * typesize * n_per_plane + b * n_per_plane;
            dst[dst_off..dst_off + n_per_plane]
                .copy_from_slice(&src[src_off..src_off + n_per_plane]);
        }
    }
}

/// Decode pass 2 - inverse of [`trans_bit_byte`]; gather + bit-transpose.
///
/// `(8, B, G) -> (B, G, 8)` bytes. For each `(b, g)` we read 8 bytes, one from
/// each of the 8 bit-plane regions of the input (`src[k * bit_row_skip + b * G + g]`
/// for `k in 0..8`). These 8 bytes are the encode-pass-2 output for that
/// `(b, g)`, in bit-transposed form. Applying [`transpose8x8`] again - which
/// is self-inverse - restores the original element-major 8-byte group, which
/// we then write contiguously at `dst[b * N + g * 8 .. b * N + g * 8 + 8]`.
#[cfg_attr(feature = "multiversion", multiversion::multiversion(targets(
    // x86-64-v4
    "x86_64+sse3+ssse3+sse4.1+sse4.2+popcnt+cmpxchg16b+avx+avx2+bmi1+bmi2+f16c+fma+lzcnt+movbe+xsave+avx512f+avx512bw+avx512cd+avx512dq+avx512vl",
    // x86-64-v3
    "x86_64+sse3+ssse3+sse4.1+sse4.2+popcnt+cmpxchg16b+avx+avx2+bmi1+bmi2+f16c+fma+lzcnt+movbe+xsave",
    // x86-64-v2
    "x86_64+sse3+ssse3+sse4.1+sse4.2+popcnt+cmpxchg16b",
)))]
fn untrans_bit_byte(src: &[u8], dst: &mut [u8], n_full: usize, typesize: usize) {
    let n_per_plane = n_full / 8;
    let bit_row_skip = typesize * n_per_plane;

    for b in 0..typesize {
        for g in 0..n_per_plane {
            // Gather: one byte from each of 8 bit-plane regions.
            let transposed: [u8; 8] =
                array_from_fn_inline(|k| src[k * bit_row_skip + b * n_per_plane + g]);

            // Self-inverse bit transpose: applying it a second time recovers
            // the original element-major 8-byte group.
            let group = transpose8x8(transposed);

            let dst_off = b * n_full + g * 8;
            dst[dst_off..dst_off + 8].copy_from_slice(&group);
        }
    }
}

/// Transpose an 8*8 bit matrix packed into 8 bytes, using Warren's delta-swap
/// (Hacker's Delight 7-3). Branchless, constant-time, self-inverse.
///
/// **Input/output convention (little-endian).** The 8 bytes are loaded as a
/// `u64` with byte 0 at the least-significant position, and within each byte
/// bit 0 is the LSB. We view this `u64` as an 8*8 bit matrix with
/// **rows = byte index** and **columns = bit-within-byte**; the return value
/// is the same `u64` with rows and columns swapped, written back as 8 bytes.
///
/// **Delta-swap structure.** The three stages swap progressively larger
/// blocks across the anti-diagonal of the matrix, following the classic
/// recursive 8*8 -> two 4*4s -> four 2*2s -> sixteen 1*1s decomposition:
///
/// | stage | delta | mask                     | what it swaps              |
/// |-------|-------|--------------------------|----------------------------|
/// | 1     | 7     | `0x00AA00AA00AA00AA`     | 1*1 blocks within 2*2s     |
/// | 2     | 14    | `0x0000CCCC0000CCCC`     | 2*2 blocks within 4*4s     |
/// | 3     | 28    | `0x00000000F0F0F0F0`     | 4*4 blocks within the 8*8  |
///
/// Each stage is the classic XOR-swap `t = (x ^ (x >> k)) & mask;
/// x ^= t ^ (t << k)` which simultaneously exchanges bits at distance k
/// wherever the mask is set. Applying the three stages in order performs the
/// full 8*8 bit transpose; applying them a second time performs the inverse
/// (and, since transpose is an involution, returns the original value).
///
/// Only little-endian targets are supported - a big-endian transpose would
/// require a mirrored mask schedule. A compile-time assert enforces this.
#[inline(always)]
fn transpose8x8(x: [u8; 8]) -> [u8; 8] {
    const _: () = const {
        assert!(
            cfg!(target_endian = "little"),
            "Only little-endian is supported"
        );
    };

    let mut x = u64::from_le_bytes(x);
    let mut t;
    t = (x ^ (x >> 7)) & 0x00AA_00AA_00AA_00AAu64;
    x = x ^ t ^ (t << 7);
    t = (x ^ (x >> 14)) & 0x0000_CCCC_0000_CCCCu64;
    x = x ^ t ^ (t << 14);
    t = (x ^ (x >> 28)) & 0x0000_0000_F0F0_F0F0u64;
    x = x ^ t ^ (t << 28);
    x.to_le_bytes()
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

    // Reference: bit i (LSB) of byte k <-> bit k (LSB) of byte i.
    // This matches the TRANS_BIT_8X8 / little-endian u64 convention used by Blosc.
    fn transpose8x8_reference(x: [u8; 8]) -> [u8; 8] {
        let mut y = [0u8; 8];
        for i in 0..8u32 {
            for k in 0..8u32 {
                let bit = (x[i as usize] >> k) & 1;
                y[k as usize] |= bit << i;
            }
        }
        y
    }

    proptest::proptest! {
        #[test]
        fn transpose8x8(x: [u8; 8]) {
            proptest::prop_assert_eq!(super::transpose8x8(x), transpose8x8_reference(x));
        }

        #[test]
        fn transpose8x8_is_self_inverse(x: [u8; 8]) {
            proptest::prop_assert_eq!(super::transpose8x8(super::transpose8x8(x)), x);
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
        let tmp_buffers = BufferPool::new();

        let mut optimized_out = vec![0u8; src.len()];
        BitShuffleFilter::default().encode(src, &mut optimized_out, &dtype, &tmp_buffers);

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
