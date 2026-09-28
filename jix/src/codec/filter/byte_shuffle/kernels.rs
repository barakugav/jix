//! Byte-shuffle kernels.
//!
//! This file is intentionally self-contained (only `core`/`std`, `fearless_simd`, plus the optional
//! `multiversion` attribute) so that `jix/probe` can `#[path]`-include it and compile it for any
//! target without pulling in the C dependencies of `jix` (zstd). Keep it that way.

use fearless_simd::{dispatch, Level, Simd, SimdBase};

/// Byte-shuffle `src` into `dst` for elements of `itemsize` bytes.
pub fn encode(src: &[u8], dst: &mut [u8], itemsize: usize) {
    assert_eq!(src.len(), dst.len());
    debug_assert!(src.len().is_multiple_of(itemsize));
    match itemsize {
        1 => dst.copy_from_slice(src), // identity permutation
        2 => encode_impl::<2, 64>(src, dst),
        4 => encode_impl::<4, 32>(src, dst),
        8 => encode_impl::<8, 16>(src, dst),
        16 => encode_impl::<16, 8>(src, dst),
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

#[inline(never)]
#[cfg_attr(feature = "multiversion", multiversion::multiversion(targets(
    // x86-64-v4
    "x86_64+sse3+ssse3+sse4.1+sse4.2+popcnt+cmpxchg16b+avx+avx2+bmi1+bmi2+f16c+fma+lzcnt+movbe+xsave+avx512f+avx512bw+avx512cd+avx512dq+avx512vl",
    // x86-64-v3
    "x86_64+sse3+ssse3+sse4.1+sse4.2+popcnt+cmpxchg16b+avx+avx2+bmi1+bmi2+f16c+fma+lzcnt+movbe+xsave",
    // x86-64-v2
    "x86_64+sse3+ssse3+sse4.1+sse4.2+popcnt+cmpxchg16b",
)))]
pub fn encode_impl<const ITEMSIZE: usize, const LANES: usize>(src: &[u8], dst: &mut [u8]) {
    let nitems = src.len() / ITEMSIZE;

    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();
    let mut i = 0;
    let body_limit = nitems - nitems % LANES;
    while i < body_limit {
        let elms = unsafe {
            src_ptr
                .cast::<[u8; ITEMSIZE]>()
                .add(i)
                .cast::<[[u8; ITEMSIZE]; LANES]>()
                .read()
        };
        #[allow(clippy::needless_range_loop)]
        for b in 0..ITEMSIZE {
            let byte_elms = std::array::from_fn(|j| elms[j][b]);
            unsafe {
                dst_ptr
                    .add(b * nitems + i)
                    .cast::<[u8; LANES]>()
                    .write(byte_elms);
            }
        }
        i += LANES;
    }
    // Tail of the remaining <LANES items
    encode_impl_generic(src, dst, ITEMSIZE, i);
}

#[inline(never)]
#[cfg_attr(feature = "multiversion", multiversion::multiversion(targets(
    // x86-64-v4
    "x86_64+sse3+ssse3+sse4.1+sse4.2+popcnt+cmpxchg16b+avx+avx2+bmi1+bmi2+f16c+fma+lzcnt+movbe+xsave+avx512f+avx512bw+avx512cd+avx512dq+avx512vl",
    // x86-64-v3
    "x86_64+sse3+ssse3+sse4.1+sse4.2+popcnt+cmpxchg16b+avx+avx2+bmi1+bmi2+f16c+fma+lzcnt+movbe+xsave",
    // x86-64-v2
    "x86_64+sse3+ssse3+sse4.1+sse4.2+popcnt+cmpxchg16b",
)))]
pub fn encode_impl_generic(src: &[u8], dst: &mut [u8], itemsize: usize, start: usize) {
    debug_assert!(src.len().is_multiple_of(itemsize));
    let nitems = src.len() / itemsize;
    let src = src.as_ptr();
    let dst = dst.as_mut_ptr();
    for b in 0..itemsize {
        for i in start..nitems {
            unsafe {
                let elm = src.add(i * itemsize + b).read();
                dst.add(b * nitems + i).write(elm);
            }
        }
    }
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

#[cfg_attr(feature = "multiversion", multiversion::multiversion(targets(
    // x86-64-v4
    "x86_64+sse3+ssse3+sse4.1+sse4.2+popcnt+cmpxchg16b+avx+avx2+bmi1+bmi2+f16c+fma+lzcnt+movbe+xsave+avx512f+avx512bw+avx512cd+avx512dq+avx512vl",
    // x86-64-v3
    "x86_64+sse3+ssse3+sse4.1+sse4.2+popcnt+cmpxchg16b+avx+avx2+bmi1+bmi2+f16c+fma+lzcnt+movbe+xsave",
    // x86-64-v2
    "x86_64+sse3+ssse3+sse4.1+sse4.2+popcnt+cmpxchg16b",
)))]
#[inline(never)]
pub fn decode_impl_generic(src: &[u8], dst: &mut [u8], itemsize: usize, start: usize) {
    debug_assert!(src.len().is_multiple_of(itemsize));
    let nitems = src.len() / itemsize;
    let src = src.as_ptr();
    let dst = dst.as_mut_ptr();
    for i in start..nitems {
        for b in 0..itemsize {
            unsafe {
                let elm = src.add(b * nitems + i).read();
                dst.add(i * itemsize + b).write(elm);
            }
        }
    }
}
