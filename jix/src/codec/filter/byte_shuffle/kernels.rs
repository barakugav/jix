//! Byte-shuffle kernels.
//!
//! This file is intentionally self-contained (only `core`/`std` and `fearless_simd`) so that
//! `jix/probe` can `#[path]`-include it and compile it for any target without pulling in the C
//! dependencies of `jix` (zstd). Keep it that way.

use fearless_simd::{dispatch, Level, Simd, SimdBase};

/// Byte-shuffle `src` into `dst` for elements of `itemsize` bytes.
pub fn encode(src: &[u8], dst: &mut [u8], itemsize: usize) {
    assert_eq!(src.len(), dst.len());
    debug_assert!(src.len().is_multiple_of(itemsize));
    match itemsize {
        1 => dst.copy_from_slice(src), // identity permutation
        2 => encode_dispatch::<2>(src, dst),
        4 => encode_dispatch::<4>(src, dst),
        8 => encode_dispatch::<8>(src, dst),
        16 => encode_dispatch::<16>(src, dst),
        _ => encode_impl_generic(src, dst, itemsize, 0),
    }
}

/// Inverse of [`encode`].
pub fn decode(src: &[u8], dst: &mut [u8], itemsize: usize) {
    assert_eq!(src.len(), dst.len());
    debug_assert!(src.len().is_multiple_of(itemsize));
    match itemsize {
        1 => dst.copy_from_slice(src), // identity permutation
        2 => decode_dispatch::<2>(src, dst),
        4 => decode_dispatch::<4>(src, dst),
        8 => decode_dispatch::<8>(src, dst),
        16 => decode_dispatch::<16>(src, dst),
        _ => decode_impl_generic(src, dst, itemsize, 0),
    }
}

#[inline(always)]
fn is_sse2<S: Simd>(simd: S) -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if let Level::Sse2(_) = simd.level() {
        return true;
    }
    let _ = simd;
    false
}

fn encode_dispatch<const ITEMSIZE: usize>(src: &[u8], dst: &mut [u8]) {
    let done = dispatch!(Level::new(), simd => encode_simd::<_, ITEMSIZE>(simd, src, dst));
    // Tail of the remaining items
    encode_impl_generic(src, dst, ITEMSIZE, done);
}

/// Encode main loop: shuffles whole chunks of `S::u8s::LEN` items, and returns the number of
/// items encoded. `ITEMSIZE` must be a power of two (checked at compile time).
///
/// Each chunk is a transpose of ITEMSIZE vectors of whole items into one vector per byte plane:
/// log2(ITEMSIZE) rounds of the inverse perfect shuffle, the exact inverse of [`decode_simd`].
///
/// Optimized by static analysis of the generated asm (cargo-asm + llvm-mca, steady-state cycles
/// of the main loop), for x86-64 SSE4.2 (Sandy Bridge, Jaguar), AVX2 (Skylake, Alder Lake, Zen 3)
/// and AVX-512 (Ice Lake, Sapphire Rapids, Zen 4), i686 SSE2 (Skylake), and aarch64 NEON
/// (Cortex-A72, Neoverse N1 / V2, Apple M1).
#[inline(always)]
pub fn encode_simd<S: Simd, const ITEMSIZE: usize>(simd: S, src: &[u8], dst: &mut [u8]) -> usize {
    let shuffle_steps = const {
        assert!(ITEMSIZE.is_power_of_two());
        ITEMSIZE.ilog2()
    };
    let lanes = S::u8s::LEN;
    // Chunks per loop iteration, so an iteration covers at least MIN_BYTES_PER_ITER bytes: amortizes
    // the loop overhead for narrow vectors and small itemsizes.
    let unroll = (MIN_BYTES_PER_ITER / (ITEMSIZE * lanes)).max(1);
    // A decode round is a perfect shuffle of the chunk's ITEMSIZE * lanes bytes (a rotation of the
    // byte index bits by one), and decode rotates by log2(ITEMSIZE). The inverse is either
    // log2(ITEMSIZE) inverse rounds (`deinterleave`), or log2(lanes) more decode rounds
    // (`interleave`). `interleave` is never more expensive than `deinterleave`, so use it when it
    // needs no more rounds. On SSE2, which has no byte shuffle, fearless_simd's `deinterleave` is
    // scalar code: always use `interleave` there.
    let interleave_steps = lanes.ilog2();
    let use_deinterleave = shuffle_steps < interleave_steps && !is_sse2(simd);
    let nitems = src.len() / ITEMSIZE;
    assert!(dst.len() >= nitems * ITEMSIZE);
    let nchunks = nitems / (lanes * unroll) * unroll;
    for c in (0..nchunks).step_by(unroll) {
        for u in 0..unroll {
            let i = (c + u) * lanes;
            // SAFETY (all `get_unchecked*` below): `b < ITEMSIZE` and `i + lanes <= nitems`, so
            // the ranges are within the `nitems * ITEMSIZE` bytes of `src` and `dst`.
            let mut v = [S::u8s::splat(simd, 0); ITEMSIZE];
            for (b, x) in v.iter_mut().enumerate() {
                let start = i * ITEMSIZE + b * lanes;
                *x = S::u8s::from_slice(simd, unsafe { src.get_unchecked(start..start + lanes) });
            }
            if use_deinterleave {
                for _ in 0..shuffle_steps {
                    let mut w = v;
                    for j in 0..ITEMSIZE / 2 {
                        (w[j], w[j + ITEMSIZE / 2]) = v[2 * j].deinterleave(v[2 * j + 1]);
                    }
                    v = w;
                }
            } else {
                for _ in 0..interleave_steps {
                    let mut w = v;
                    for j in 0..ITEMSIZE / 2 {
                        (w[2 * j], w[2 * j + 1]) = v[j].interleave(v[j + ITEMSIZE / 2]);
                    }
                    v = w;
                }
            }
            for (b, x) in v.iter().enumerate() {
                let start = b * nitems + i;
                x.store_slice(unsafe { dst.get_unchecked_mut(start..start + lanes) });
            }
        }
    }
    nchunks * lanes
}

/// Scalar encode of the items from `start` on, for any itemsize. Compiled for each SIMD level
/// (auto-vectorized).
#[inline(never)]
pub fn encode_impl_generic(src: &[u8], dst: &mut [u8], itemsize: usize, start: usize) {
    debug_assert!(src.len().is_multiple_of(itemsize));
    let nitems = src.len() / itemsize;
    assert!(dst.len() >= nitems * itemsize);
    let src = src.as_ptr();
    let dst = dst.as_mut_ptr();
    dispatch!(Level::new(), _ => {
        for b in 0..itemsize {
            for i in start..nitems {
                unsafe {
                    let elm = src.add(i * itemsize + b).read();
                    dst.add(b * nitems + i).write(elm);
                }
            }
        }
    })
}

fn decode_dispatch<const ITEMSIZE: usize>(src: &[u8], dst: &mut [u8]) {
    let done = dispatch!(Level::new(), simd => decode_simd::<_, ITEMSIZE>(simd, src, dst));
    // Tail of the remaining items
    decode_impl_generic(src, dst, ITEMSIZE, done);
}

const MIN_BYTES_PER_ITER: usize = 256;

/// Decode main loop: un-shuffles whole chunks of `S::u8s::LEN` items, and returns the number of
/// items decoded. `ITEMSIZE` must be a power of two (checked at compile time).
///
/// Each chunk is a transpose of ITEMSIZE vectors (one per byte plane): log2(ITEMSIZE) rounds of
/// a perfect shuffle, pairing plane `j` with plane `j + ITEMSIZE / 2`.
///
/// Optimized by static analysis of the generated asm (cargo-asm + llvm-mca, steady-state cycles
/// of the main loop), for x86-64 SSE4.2 (Sandy Bridge, Jaguar), AVX2 (Skylake, Alder Lake, Zen 3)
/// and AVX-512 (Ice Lake, Sapphire Rapids, Zen 4), i686 SSE2 (Skylake), and aarch64 NEON
/// (Cortex-A72, Neoverse N1 / V2, Apple M1).
#[inline(always)]
pub fn decode_simd<S: Simd, const ITEMSIZE: usize>(simd: S, src: &[u8], dst: &mut [u8]) -> usize {
    let shuffle_steps = const {
        assert!(ITEMSIZE.is_power_of_two());
        ITEMSIZE.ilog2()
    };
    let lanes = S::u8s::LEN;
    // Chunks per loop iteration, so an iteration covers at least MIN_BYTES_PER_ITER bytes: amortizes
    // the loop overhead for narrow vectors and small itemsizes.
    let unroll = (MIN_BYTES_PER_ITER / (ITEMSIZE * lanes)).max(1);
    let nitems = src.len() / ITEMSIZE;
    assert!(dst.len() >= nitems * ITEMSIZE);
    let nchunks = nitems / (lanes * unroll) * unroll;
    for c in (0..nchunks).step_by(unroll) {
        for u in 0..unroll {
            let i = (c + u) * lanes;
            // SAFETY (all `get_unchecked*` below): `b < ITEMSIZE` and `i + lanes <= nitems`, so
            // the ranges are within the `nitems * ITEMSIZE` bytes of `src` and `dst`.
            let mut v = [S::u8s::splat(simd, 0); ITEMSIZE];
            for (b, x) in v.iter_mut().enumerate() {
                let start = b * nitems + i;
                *x = S::u8s::from_slice(simd, unsafe { src.get_unchecked(start..start + lanes) });
            }
            for _ in 0..shuffle_steps {
                let mut w = v;
                for j in 0..ITEMSIZE / 2 {
                    (w[2 * j], w[2 * j + 1]) = v[j].interleave(v[j + ITEMSIZE / 2]);
                }
                v = w;
            }
            for (b, x) in v.iter().enumerate() {
                let start = i * ITEMSIZE + b * lanes;
                x.store_slice(unsafe { dst.get_unchecked_mut(start..start + lanes) });
            }
        }
    }
    nchunks * lanes
}

/// Scalar decode of the items from `start` on, for any itemsize. Compiled for each SIMD level
/// (auto-vectorized).
#[inline(never)]
pub fn decode_impl_generic(src: &[u8], dst: &mut [u8], itemsize: usize, start: usize) {
    debug_assert!(src.len().is_multiple_of(itemsize));
    let nitems = src.len() / itemsize;
    assert!(dst.len() >= nitems * itemsize);
    let src = src.as_ptr();
    let dst = dst.as_mut_ptr();
    dispatch!(Level::new(), _ => {
        for i in start..nitems {
            for b in 0..itemsize {
                unsafe {
                    let elm = src.add(b * nitems + i).read();
                    dst.add(i * itemsize + b).write(elm);
                }
            }
        }
    })
}
