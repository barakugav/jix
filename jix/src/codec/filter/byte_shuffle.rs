use crate::buf_pool::BufferPool;
use crate::codec::filter::FilterImpl;
use crate::dtype::Dtype;

pub mod kernels;

#[derive(Default)]
pub(in crate::codec::filter) struct ByteShuffleFilter;
impl FilterImpl for ByteShuffleFilter {
    fn encode(&self, src: &[u8], dst: &mut [u8], dtype: &Dtype, _tmp_buffers: &BufferPool) {
        kernels::encode(src, dst, dtype.itemsize() as usize);
    }

    fn decode(&self, src: &[u8], dst: &mut [u8], dtype: &Dtype, _tmp_buffers: &BufferPool) {
        kernels::decode(src, dst, dtype.itemsize() as usize);
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
