//! Bit-shuffle kernels: the bit transpose of [`super::BitShuffleFilter`] (its other passes are
//! byte shuffles).
//!
//! This file is intentionally self-contained (only `core`/`std` and `fearless_simd`) so that
//! `jix/probe` can `#[path]`-include it and compile it for any target without pulling in the C
//! dependencies of `jix` (zstd). Keep it that way.

use core::ops::{BitAnd, BitXor, Shl, Shr};
use fearless_simd::{dispatch, Bytes, Level, Simd, SimdBase};

/// Bit-transpose the 8 rows of `src` into the 8 rows of `dst`, column by column: bit `i` of
/// `src` row `k` goes to bit `k` of `dst` row `i`. Both are 8 rows of `src.len() / 8` bytes.
///
/// Self-inverse, so it serves both directions of the filter.
pub fn transpose_bit_rows(src: &[u8], dst: &mut [u8]) {
    let done = dispatch!(Level::new(), simd => transpose_bit_rows_simd(simd, src, dst));
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
pub fn transpose_bit_rows_simd<S: Simd>(simd: S, src: &[u8], dst: &mut [u8]) -> usize {
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
pub fn transpose8x8_rows<T: Rows>(r: &mut [T; 8]) {
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
