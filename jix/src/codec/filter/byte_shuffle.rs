use crate::buf_pool::BufferPool;
use crate::codec::filter::FilterImpl;
use crate::dtype::Dtype;
use fearless_simd::{Level as SimdLevel, Simd, SimdBase};

// This code has been optimized by static analysis using cargo-asm and llvm-mca, for
// - x86-64 SSE4.2 (Sandy Bridge, Jaguar)
// - AVX2 (Skylake, Alder Lake, Zen 3)
// - AVX-512 (Ice Lake, Sapphire Rapids, Zen 4)
// - i686 SSE2 (Skylake)
// - aarch64 NEON (Cortex-A72, Neoverse N1 / V2, Apple M1)

const MIN_BYTES_PER_ITERATION: usize = 256;

#[derive(Default)]
pub(in crate::codec::filter) struct ByteShuffleFilter;
impl FilterImpl for ByteShuffleFilter {
    fn encode(&self, src: &[u8], dst: &mut [u8], dtype: &Dtype, _buf_pool: &BufferPool) {
        encode(src, dst, dtype.itemsize() as usize);
    }

    fn decode(&self, src: &[u8], dst: &mut [u8], dtype: &Dtype, _buf_pool: &BufferPool) {
        decode(src, dst, dtype.itemsize() as usize);
    }
}

pub(super) fn encode(src: &[u8], dst: &mut [u8], itemsize: usize) {
    assert!(src.len() == dst.len() && src.len().is_multiple_of(itemsize));
    match itemsize {
        1 => dst.copy_from_slice(src), // no-op
        2 => encode_impl::<2>(src, dst),
        4 => encode_impl::<4>(src, dst),
        8 => encode_impl::<8>(src, dst),
        16 => encode_impl::<16>(src, dst),
        _ => encode_impl_generic(src, dst, itemsize, 0),
    }
}

#[inline(never)]
fn encode_impl<const ITEMSIZE: usize>(src: &[u8], dst: &mut [u8]) {
    let n_elements_done = fearless_simd::dispatch!(SimdLevel::new(), simd => encode_simd::<ITEMSIZE, _>(src, dst, simd));
    // Tail of the remaining items
    encode_impl_generic_inline(src, dst, ITEMSIZE, n_elements_done);
}

/// Encode main loop.
///
/// # Returns
///
/// The number of items encoded.
#[inline(always)]
fn encode_simd<const ITEMSIZE: usize, S: Simd>(src: &[u8], dst: &mut [u8], simd: S) -> usize {
    let nitems = src.len() / ITEMSIZE;
    assert!(dst.len() >= nitems * ITEMSIZE);

    let lanes = S::u8s::LEN;
    let unroll = (MIN_BYTES_PER_ITERATION / (ITEMSIZE * lanes)).max(1);
    let nchunks = nitems / (lanes * unroll) * unroll;

    let deinterleave_steps = const {
        assert!(ITEMSIZE.is_power_of_two());
        ITEMSIZE.ilog2()
    };
    let interleave_steps = const {
        let lanes = S::u8s::LEN;
        assert!(lanes.is_power_of_two());
        lanes.ilog2()
    };
    // `interleave` is never more expensive than `deinterleave`, so use it when it
    // needs no more rounds. SSE2 has no byte shuffle, fearless_simd's `deinterleave` is
    // scalar code, so always use `interleave` there.
    let use_deinterleave = deinterleave_steps < interleave_steps && !is_sse2(simd);

    for chunk in (0..nchunks).step_by(unroll) {
        for u in 0..unroll {
            let i = (chunk + u) * lanes;

            // load from src into registers
            let mut v = [S::u8s::splat(simd, 0); ITEMSIZE];
            for (b, x) in v.iter_mut().enumerate() {
                let start = i * ITEMSIZE + b * lanes;
                *x = S::u8s::from_slice(simd, unsafe { src.get_unchecked(start..start + lanes) });
            }

            // shuffle
            if use_deinterleave {
                for _ in 0..deinterleave_steps {
                    let mut w = [S::u8s::splat(simd, 0); ITEMSIZE];
                    for j in 0..ITEMSIZE / 2 {
                        (w[j], w[j + ITEMSIZE / 2]) = v[2 * j].deinterleave(v[2 * j + 1]);
                    }
                    v = w;
                }
            } else {
                for _ in 0..interleave_steps {
                    let mut w = [S::u8s::splat(simd, 0); ITEMSIZE];
                    for j in 0..ITEMSIZE / 2 {
                        (w[2 * j], w[2 * j + 1]) = v[j].interleave(v[j + ITEMSIZE / 2]);
                    }
                    v = w;
                }
            }

            // store into dst
            for (b, x) in v.iter().enumerate() {
                let start = b * nitems + i;
                x.store_slice(unsafe { dst.get_unchecked_mut(start..start + lanes) });
            }
        }
    }
    nchunks * lanes
}

#[inline(never)]
fn encode_impl_generic(src: &[u8], dst: &mut [u8], itemsize: usize, start: usize) {
    encode_impl_generic_inline(src, dst, itemsize, start);
}
#[inline(always)]
fn encode_impl_generic_inline(src: &[u8], dst: &mut [u8], itemsize: usize, start: usize) {
    debug_assert!(src.len().is_multiple_of(itemsize));

    let nitems = src.len() / itemsize;
    assert!(dst.len() >= nitems * itemsize);

    fearless_simd::dispatch!(SimdLevel::new(), _ => {
        for b in 0..itemsize {
            for i in start..nitems {
                let elm = unsafe { *src.get_unchecked(i * itemsize + b) };
                unsafe { *dst.get_unchecked_mut(b * nitems + i) = elm };
            }
        }
    })
}

pub(super) fn decode(src: &[u8], dst: &mut [u8], itemsize: usize) {
    assert!(src.len() == dst.len() && src.len().is_multiple_of(itemsize));
    match itemsize {
        1 => dst.copy_from_slice(src), // no-op
        2 => decode_impl::<2>(src, dst),
        4 => decode_impl::<4>(src, dst),
        8 => decode_impl::<8>(src, dst),
        16 => decode_impl::<16>(src, dst),
        _ => decode_impl_generic(src, dst, itemsize, 0),
    }
}

#[inline(never)]
fn decode_impl<const ITEMSIZE: usize>(src: &[u8], dst: &mut [u8]) {
    let n_elements_done = fearless_simd::dispatch!(SimdLevel::new(), simd => decode_simd::<ITEMSIZE, _>(src, dst, simd));
    // Tail of the remaining items
    decode_impl_generic_inline(src, dst, ITEMSIZE, n_elements_done);
}

/// Decode main loop.
///
/// # Returns
///
/// The number of items decoded.
#[inline(always)]
fn decode_simd<const ITEMSIZE: usize, S: Simd>(src: &[u8], dst: &mut [u8], simd: S) -> usize {
    let nitems = src.len() / ITEMSIZE;
    assert!(dst.len() >= nitems * ITEMSIZE);

    let lanes = S::u8s::LEN;
    let unroll = (MIN_BYTES_PER_ITERATION / (ITEMSIZE * lanes)).max(1);
    let n_chunks = nitems / (lanes * unroll) * unroll;

    for chunk in (0..n_chunks).step_by(unroll) {
        for u in 0..unroll {
            let i = (chunk + u) * lanes;

            // load from src into registers
            let mut v = [S::u8s::splat(simd, 0); ITEMSIZE];
            for (b, x) in v.iter_mut().enumerate() {
                let start = b * nitems + i;
                *x = S::u8s::from_slice(simd, unsafe { src.get_unchecked(start..start + lanes) });
            }

            // shuffle
            let interleave_steps = const {
                assert!(ITEMSIZE.is_power_of_two());
                ITEMSIZE.ilog2()
            };
            for _ in 0..interleave_steps {
                let mut w = [S::u8s::splat(simd, 0); ITEMSIZE];
                for j in 0..ITEMSIZE / 2 {
                    (w[2 * j], w[2 * j + 1]) = v[j].interleave(v[j + ITEMSIZE / 2]);
                }
                v = w;
            }

            // write to dst
            for (b, x) in v.iter().enumerate() {
                let start = i * ITEMSIZE + b * lanes;
                x.store_slice(unsafe { dst.get_unchecked_mut(start..start + lanes) });
            }
        }
    }
    n_chunks * lanes
}

#[inline(never)]
fn decode_impl_generic(src: &[u8], dst: &mut [u8], itemsize: usize, start: usize) {
    decode_impl_generic_inline(src, dst, itemsize, start);
}
#[inline(always)]
fn decode_impl_generic_inline(src: &[u8], dst: &mut [u8], itemsize: usize, start: usize) {
    debug_assert!(src.len().is_multiple_of(itemsize));

    let nitems = src.len() / itemsize;
    assert!(dst.len() >= nitems * itemsize);

    fearless_simd::dispatch!(SimdLevel::new(), _ => {
        for i in start..nitems {
            for b in 0..itemsize {
                let elm = *unsafe { src.get_unchecked(b * nitems + i) };
                unsafe { *dst.get_unchecked_mut(i * itemsize + b) = elm };
            }
        }
    })
}

#[inline(always)]
fn is_sse2<S: Simd>(simd: S) -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if matches!(simd.level(), SimdLevel::Sse2(_)) {
        return true;
    }
    let _ = simd;
    false
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
        let buf_pool = BufferPool::new();

        // Encode: optimized vs reference.
        let mut optimized_encoded = vec![0u8; src.len()];
        ByteShuffleFilter.encode(src, &mut optimized_encoded, &dtype, &buf_pool);
        let mut reference_encoded = vec![0u8; src.len()];
        byte_shuffle_encode_reference(src, &mut reference_encoded, itemsize);
        assert_eq!(optimized_encoded, reference_encoded);

        // Decode: optimized vs reference, applied to the shuffled bytes.
        let shuffled = reference_encoded.as_slice();
        let mut optimized_decoded = vec![0u8; src.len()];
        ByteShuffleFilter.decode(shuffled, &mut optimized_decoded, &dtype, &buf_pool);
        let mut reference_decoded = vec![0u8; src.len()];
        byte_shuffle_decode_reference(shuffled, &mut reference_decoded, itemsize);
        assert_eq!(optimized_decoded, reference_decoded);
    }

    /// `decode_simd` at every SIMD level this CPU supports (not only the one `fearless_simd::dispatch!` picks),
    /// against the reference, for every power-of-two itemsize and lengths around the chunk sizes.
    #[test]
    fn decode_simd_all_levels() {
        use super::{decode_impl_generic, decode_simd};
        use fearless_simd::{Level as SimdLevel, Simd};

        fn check<S: Simd>(simd: S) {
            fn one<S: Simd, const ITEMSIZE: usize>(simd: S) {
                for nitems in [0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 300] {
                    let src: Vec<u8> = (0..nitems * ITEMSIZE).map(|x| (x * 7 + 3) as u8).collect();
                    let mut expected = vec![0u8; src.len()];
                    byte_shuffle_decode_reference(&src, &mut expected, ITEMSIZE);
                    let mut dst = vec![0u8; src.len()];
                    let done = simd.vectorize(|| decode_simd::<ITEMSIZE, S>(&src, &mut dst, simd));
                    decode_impl_generic(&src, &mut dst, ITEMSIZE, done);
                    assert_eq!(dst, expected, "itemsize {ITEMSIZE}, nitems {nitems}");
                }
            }
            one::<S, 2>(simd);
            one::<S, 4>(simd);
            one::<S, 8>(simd);
            one::<S, 16>(simd);
        }

        let level = SimdLevel::new();
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

    /// `encode_simd` at every SIMD level this CPU supports (not only the one `fearless_simd::dispatch!` picks),
    /// against the reference, for every power-of-two itemsize and lengths around the chunk sizes.
    #[test]
    fn encode_simd_all_levels() {
        use super::{encode_impl_generic, encode_simd};
        use fearless_simd::{Level as SimdLevel, Simd};

        fn check<S: Simd>(simd: S) {
            fn one<S: Simd, const ITEMSIZE: usize>(simd: S) {
                for nitems in [0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 300] {
                    let src: Vec<u8> = (0..nitems * ITEMSIZE).map(|x| (x * 7 + 3) as u8).collect();
                    let mut expected = vec![0u8; src.len()];
                    byte_shuffle_encode_reference(&src, &mut expected, ITEMSIZE);
                    let mut dst = vec![0u8; src.len()];
                    let done = simd.vectorize(|| encode_simd::<ITEMSIZE, S>(&src, &mut dst, simd));
                    encode_impl_generic(&src, &mut dst, ITEMSIZE, done);
                    assert_eq!(dst, expected, "itemsize {ITEMSIZE}, nitems {nitems}");
                }
            }
            one::<S, 2>(simd);
            one::<S, 4>(simd);
            one::<S, 8>(simd);
            one::<S, 16>(simd);
        }

        let level = SimdLevel::new();
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
