//! Bit-shuffle kernels: the bit-level passes of [`super::BitShuffleFilter`] (the byte-shuffle
//! pass is the byte-shuffle filter's own).
//!
//! This file is intentionally self-contained (only `core`/`std`, plus the optional `multiversion`
//! attribute) so that `jix/probe` can `#[path]`-include it and compile it for any target without
//! pulling in the C dependencies of `jix` (zstd). Keep it that way.

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
pub fn trans_bit_byte(src: &[u8], dst: &mut [u8], n_full: usize, typesize: usize) {
    let n_per_plane = n_full / 8;
    let bit_row_skip = typesize * n_per_plane; // = B * G

    for b in 0..typesize {
        for g in 0..n_per_plane {
            let src_off = b * n_full + g * 8;
            let mut group = [0u8; 8];
            for k in 0..8 {
                group[k] = src[src_off + k];
            }

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
/// specialization of `bshuf_trans_elem(lda=8, ldb=B, elem_size=G)`.
#[cfg_attr(feature = "multiversion", multiversion::multiversion(targets(
    // x86-64-v4
    "x86_64+sse3+ssse3+sse4.1+sse4.2+popcnt+cmpxchg16b+avx+avx2+bmi1+bmi2+f16c+fma+lzcnt+movbe+xsave+avx512f+avx512bw+avx512cd+avx512dq+avx512vl",
    // x86-64-v3
    "x86_64+sse3+ssse3+sse4.1+sse4.2+popcnt+cmpxchg16b+avx+avx2+bmi1+bmi2+f16c+fma+lzcnt+movbe+xsave",
    // x86-64-v2
    "x86_64+sse3+ssse3+sse4.1+sse4.2+popcnt+cmpxchg16b",
)))]
pub fn trans_bitrow_eight(src: &[u8], dst: &mut [u8], n_full: usize, typesize: usize) {
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
pub fn untrans_bitrow_eight(src: &[u8], dst: &mut [u8], n_full: usize, typesize: usize) {
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
pub fn untrans_bit_byte(src: &[u8], dst: &mut [u8], n_full: usize, typesize: usize) {
    let n_per_plane = n_full / 8;
    let bit_row_skip = typesize * n_per_plane;

    for b in 0..typesize {
        for g in 0..n_per_plane {
            // Gather: one byte from each of 8 bit-plane regions.
            let mut transposed = [0u8; 8];
            for k in 0..8 {
                transposed[k] = src[k * bit_row_skip + b * n_per_plane + g];
            }

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
pub fn transpose8x8(x: [u8; 8]) -> [u8; 8] {
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
