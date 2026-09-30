use crate::buf_pool::BufferPool;
use crate::codec::filter::FilterImpl;
use crate::dtype::Dtype;
use fearless_simd::{dispatch, Level, Simd, SimdBase};

#[derive(Default)]
pub(in crate::codec::filter) struct ByteShuffleFilter;
impl FilterImpl for ByteShuffleFilter {
    fn encode(&self, src: &[u8], dst: &mut [u8], dtype: &Dtype, _tmp_buffers: &BufferPool) {
        encode(src, dst, dtype.itemsize() as usize);
    }

    fn decode(&self, src: &[u8], dst: &mut [u8], dtype: &Dtype, _tmp_buffers: &BufferPool) {
        decode(src, dst, dtype.itemsize() as usize);
    }
}

/// Byte-shuffle `src` into `dst` for elements of `itemsize` bytes.
pub(super) fn encode(src: &[u8], dst: &mut [u8], itemsize: usize) {
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
pub(super) fn decode(src: &[u8], dst: &mut [u8], itemsize: usize) {
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
fn encode_simd<S: Simd, const ITEMSIZE: usize>(simd: S, src: &[u8], dst: &mut [u8]) -> usize {
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
fn encode_impl_generic(src: &[u8], dst: &mut [u8], itemsize: usize, start: usize) {
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
fn decode_simd<S: Simd, const ITEMSIZE: usize>(simd: S, src: &[u8], dst: &mut [u8]) -> usize {
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
fn decode_impl_generic(src: &[u8], dst: &mut [u8], itemsize: usize, start: usize) {
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

#[cfg(test)]
mod tests {
    use super::ByteShuffleFilter;
    use crate::buf_pool::BufferPool;
    #[cfg(feature = "num-complex")]
    use crate::scalar::Complex;

    fn byte_shuffle_encode_reference(src: &[u8], dst: &mut [u8], itemsize: usize) {
        debug_assert!(src.len().is_multiple_of(itemsize));
        let nitems = src.len() / itemsize;
        for i in 0..nitems {
            for b in 0..itemsize {
                dst[b * nitems + i] = src[i * itemsize + b];
            }
        }
    }

    fn byte_shuffle_decode_reference(src: &[u8], dst: &mut [u8], itemsize: usize) {
        debug_assert!(src.len().is_multiple_of(itemsize));
        let nitems = src.len() / itemsize;
        for i in 0..nitems {
            for b in 0..itemsize {
                dst[i * itemsize + b] = src[b * nitems + i];
            }
        }
    }

    /// Assert the optimized `encode`/`decode` match the trivial reference
    /// implementations on the same input, in both directions.
    fn test_agrees_with_reference<T: crate::dtype::Dtyped>(items: &[T]) {
        use crate::codec::filter::FilterImpl;
        use crate::util::gen_data_bytes_from_slice;

        let data = gen_data_bytes_from_slice::<T>(items);
        let src = data.as_slice();
        let itemsize = T::DTYPE.itemsize() as usize;
        let dtype = T::DTYPE;
        let tmp_buffers = BufferPool::new();

        // Encode: optimized vs reference.
        let mut optimized_encoded = vec![0u8; src.len()];
        ByteShuffleFilter.encode(src, &mut optimized_encoded, &dtype, &tmp_buffers);
        let mut reference_encoded = vec![0u8; src.len()];
        byte_shuffle_encode_reference(src, &mut reference_encoded, itemsize);
        assert_eq!(optimized_encoded, reference_encoded);

        // Decode: optimized vs reference, applied to the shuffled bytes.
        let shuffled = reference_encoded.as_slice();
        let mut optimized_decoded = vec![0u8; src.len()];
        ByteShuffleFilter.decode(shuffled, &mut optimized_decoded, &dtype, &tmp_buffers);
        let mut reference_decoded = vec![0u8; src.len()];
        byte_shuffle_decode_reference(shuffled, &mut reference_decoded, itemsize);
        assert_eq!(optimized_decoded, reference_decoded);
    }

    /// `decode_simd` at every SIMD level this CPU supports (not only the one `dispatch!` picks),
    /// against the reference, for every power-of-two itemsize and lengths around the chunk sizes.
    #[test]
    fn decode_simd_all_levels() {
        use super::{decode_impl_generic, decode_simd};
        use fearless_simd::{Level, Simd};

        fn check<S: Simd>(simd: S) {
            fn one<S: Simd, const ITEMSIZE: usize>(simd: S) {
                for nitems in [0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 300] {
                    let src: Vec<u8> = (0..nitems * ITEMSIZE).map(|x| (x * 7 + 3) as u8).collect();
                    let mut expected = vec![0u8; src.len()];
                    byte_shuffle_decode_reference(&src, &mut expected, ITEMSIZE);
                    let mut dst = vec![0u8; src.len()];
                    let done = simd.vectorize(|| decode_simd::<S, ITEMSIZE>(simd, &src, &mut dst));
                    decode_impl_generic(&src, &mut dst, ITEMSIZE, done);
                    assert_eq!(dst, expected, "itemsize {ITEMSIZE}, nitems {nitems}");
                }
            }
            one::<S, 2>(simd);
            one::<S, 4>(simd);
            one::<S, 8>(simd);
            one::<S, 16>(simd);
        }

        let level = Level::new();
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if let Some(s) = level.as_sse2() {
                check(s);
            }
            if let Some(s) = level.as_sse4_2() {
                check(s);
            }
            if let Some(s) = level.as_avx2() {
                check(s);
            }
            if let Some(s) = level.as_avx512() {
                check(s);
            }
        }
        #[cfg(target_arch = "aarch64")]
        if let Some(s) = level.as_neon() {
            check(s);
        }
    }

    /// `encode_simd` at every SIMD level this CPU supports (not only the one `dispatch!` picks),
    /// against the reference, for every power-of-two itemsize and lengths around the chunk sizes.
    #[test]
    fn encode_simd_all_levels() {
        use super::{encode_impl_generic, encode_simd};
        use fearless_simd::{Level, Simd};

        fn check<S: Simd>(simd: S) {
            fn one<S: Simd, const ITEMSIZE: usize>(simd: S) {
                for nitems in [0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 300] {
                    let src: Vec<u8> = (0..nitems * ITEMSIZE).map(|x| (x * 7 + 3) as u8).collect();
                    let mut expected = vec![0u8; src.len()];
                    byte_shuffle_encode_reference(&src, &mut expected, ITEMSIZE);
                    let mut dst = vec![0u8; src.len()];
                    let done = simd.vectorize(|| encode_simd::<S, ITEMSIZE>(simd, &src, &mut dst));
                    encode_impl_generic(&src, &mut dst, ITEMSIZE, done);
                    assert_eq!(dst, expected, "itemsize {ITEMSIZE}, nitems {nitems}");
                }
            }
            one::<S, 2>(simd);
            one::<S, 4>(simd);
            one::<S, 8>(simd);
            one::<S, 16>(simd);
        }

        let level = Level::new();
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if let Some(s) = level.as_sse2() {
                check(s);
            }
            if let Some(s) = level.as_sse4_2() {
                check(s);
            }
            if let Some(s) = level.as_avx2() {
                check(s);
            }
            if let Some(s) = level.as_avx512() {
                check(s);
            }
        }
        #[cfg(target_arch = "aarch64")]
        if let Some(s) = level.as_neon() {
            check(s);
        }
    }

    macro_rules! test_roundtrip {
        ($ty:ty, $fn_name:ident) => {
            #[test]
            fn $fn_name() {
                crate::codec::filter::tests::run_bytes_proptest::<$ty>(|data| {
                    crate::codec::filter::tests::test_roundtrip::<ByteShuffleFilter, $ty>(data);
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

    macro_rules! test_agrees_with_reference {
        ($ty:ty, $fn_name:ident) => {
            #[test]
            fn $fn_name() {
                crate::codec::filter::tests::run_bytes_proptest::<$ty>(|data| {
                    test_agrees_with_reference::<$ty>(data);
                });
            }
        };
    }

    // Same itemsize-only dedup as the roundtrip macros above: one dtype per
    // distinct byte width (1/2/4/8/16), see `jix/src/util/nd_copy.rs:916`.
    test_agrees_with_reference!(u8, u8_agrees_with_reference);
    test_agrees_with_reference!(u16, u16_agrees_with_reference);
    test_agrees_with_reference!(u32, u32_agrees_with_reference);
    test_agrees_with_reference!(u64, u64_agrees_with_reference);
    #[cfg(feature = "num-complex")]
    test_agrees_with_reference!(Complex<f64>, complex_f64_agrees_with_reference);
}
