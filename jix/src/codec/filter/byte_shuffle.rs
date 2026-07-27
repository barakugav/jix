use crate::buf_pool::BufferPool;
use crate::codec::filter::FilterImpl;
use crate::dtype::Dtype;

#[derive(Default)]
pub(in crate::codec::filter) struct ByteShuffleFilter;
impl FilterImpl for ByteShuffleFilter {
    fn encode(&self, src: &[u8], dst: &mut [u8], dtype: &Dtype, _tmp_buffers: &BufferPool) {
        assert_eq!(src.len(), dst.len());
        let itemsize = dtype.itemsize() as usize;
        debug_assert!(src.len().is_multiple_of(itemsize));

        #[inline(never)]
        #[cfg_attr(feature = "multiversion", multiversion::multiversion(targets(
            // x86-64-v4
            "x86_64+sse3+ssse3+sse4.1+sse4.2+popcnt+cmpxchg16b+avx+avx2+bmi1+bmi2+f16c+fma+lzcnt+movbe+xsave+avx512f+avx512bw+avx512cd+avx512dq+avx512vl",
            // x86-64-v3
            "x86_64+sse3+ssse3+sse4.1+sse4.2+popcnt+cmpxchg16b+avx+avx2+bmi1+bmi2+f16c+fma+lzcnt+movbe+xsave",
            // x86-64-v2
            "x86_64+sse3+ssse3+sse4.1+sse4.2+popcnt+cmpxchg16b",
        )))]
        fn encode_impl<const ITEMSIZE: usize, const LANES: usize>(src: &[u8], dst: &mut [u8]) {
            let nitems = src.len() / ITEMSIZE;

            let src_ptr = src.as_ptr();
            let dst_ptr = dst.as_mut_ptr();
            let mut i = 0;
            while i + LANES <= nitems {
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
        fn encode_impl_generic(src: &[u8], dst: &mut [u8], itemsize: usize, start: usize) {
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

        match itemsize {
            1 => dst.copy_from_slice(src), // identity permutation
            2 => encode_impl::<2, 64>(src, dst),
            4 => encode_impl::<4, 32>(src, dst),
            8 => encode_impl::<8, 16>(src, dst),
            16 => encode_impl::<16, 8>(src, dst),
            _ => encode_impl_generic(src, dst, itemsize, 0),
        }
    }

    fn decode(&self, src: &[u8], dst: &mut [u8], dtype: &Dtype, _tmp_buffers: &BufferPool) {
        assert_eq!(src.len(), dst.len());
        let itemsize = dtype.itemsize() as usize;
        debug_assert!(src.len().is_multiple_of(itemsize));

        #[inline(never)]
        #[cfg_attr(feature = "multiversion", multiversion::multiversion(targets(
            // x86-64-v4
            "x86_64+sse3+ssse3+sse4.1+sse4.2+popcnt+cmpxchg16b+avx+avx2+bmi1+bmi2+f16c+fma+lzcnt+movbe+xsave+avx512f+avx512bw+avx512cd+avx512dq+avx512vl",
            // x86-64-v3
            "x86_64+sse3+ssse3+sse4.1+sse4.2+popcnt+cmpxchg16b+avx+avx2+bmi1+bmi2+f16c+fma+lzcnt+movbe+xsave",
            // x86-64-v2
            "x86_64+sse3+ssse3+sse4.1+sse4.2+popcnt+cmpxchg16b",
        )))]
        fn decode_impl<const ITEMSIZE: usize, const LANES: usize>(src: &[u8], dst: &mut [u8]) {
            let nitems = src.len() / ITEMSIZE;

            let src_ptr = src.as_ptr();
            let dst_ptr = dst.as_mut_ptr();
            let mut i = 0;
            while i + LANES <= nitems {
                let mut elms = [[std::mem::MaybeUninit::<u8>::uninit(); ITEMSIZE]; LANES];
                #[allow(clippy::needless_range_loop)]
                for b in 0..ITEMSIZE {
                    let byte_elms =
                        unsafe { src_ptr.add(b * nitems + i).cast::<[u8; LANES]>().read() };
                    for j in 0..LANES {
                        elms[j][b].write(byte_elms[j]);
                    }
                }
                // SAFETY: every one of the ITEMSIZE * LANES entries was written above.
                let elms = unsafe { std::mem::transmute_copy::<_, [[u8; ITEMSIZE]; LANES]>(&elms) };
                unsafe {
                    dst_ptr
                        .cast::<[u8; ITEMSIZE]>()
                        .add(i)
                        .cast::<[[u8; ITEMSIZE]; LANES]>()
                        .write(elms);
                }
                i += LANES;
            }
            // Tail of the remaining <LANES items
            decode_impl_generic(src, dst, ITEMSIZE, i);
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
        fn decode_impl_generic(src: &[u8], dst: &mut [u8], itemsize: usize, start: usize) {
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

        match itemsize {
            1 => dst.copy_from_slice(src), // identity permutation
            2 => decode_impl::<2, 64>(src, dst),
            4 => decode_impl::<4, 32>(src, dst),
            8 => decode_impl::<8, 16>(src, dst),
            16 => decode_impl::<16, 8>(src, dst),
            _ => decode_impl_generic(src, dst, itemsize, 0),
        }
    }
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
