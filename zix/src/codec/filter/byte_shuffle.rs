use crate::codec::filter::FilterImpl;
use crate::codec::TmpBufferPool;
use crate::dtype::Dtype;

#[derive(Default)]
pub(in crate::codec::filter) struct ByteShuffleFilter;
impl FilterImpl for ByteShuffleFilter {
    fn encode(&self, src: &[u8], dst: &mut [u8], dtype: &Dtype, _tmp_buffers: &TmpBufferPool) {
        // TODO: optimize using [u8; 32] (SIMD)
        let itemsize = dtype.itemsize() as usize;
        debug_assert!(src.len().is_multiple_of(itemsize));
        let nitems = src.len() / itemsize;
        for i in 0..nitems {
            for b in 0..itemsize {
                dst[b * nitems + i] = src[i * itemsize + b];
            }
        }
    }

    fn decode(&self, src: &[u8], dst: &mut [u8], dtype: &Dtype, _tmp_buffers: &TmpBufferPool) {
        // TODO: optimize using [u8; 32] (SIMD)
        let itemsize = dtype.itemsize() as usize;
        debug_assert!(src.len().is_multiple_of(itemsize));
        let nitems = src.len() / itemsize;
        for i in 0..nitems {
            for b in 0..itemsize {
                dst[i * itemsize + b] = src[b * nitems + i];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ByteShuffleFilter;
    use crate::dtype::Complex;

    macro_rules! test_roundtrip {
        ($ty:ty, $fn_name:ident) => {
            #[test]
            fn $fn_name() {
                let mut rng = crate::util::test_rng(file!(), stringify!($fn_name));
                for len in crate::util::test_lengths(&mut rng) {
                    crate::codec::filter::tests::test_roundtrip::<ByteShuffleFilter, $ty>(
                        len, &mut rng,
                    );
                }
            }
        };
    }

    test_roundtrip!(u8, u8_roundtrip);
    test_roundtrip!(u16, u16_roundtrip);
    test_roundtrip!(u32, u32_roundtrip);
    test_roundtrip!(u64, u64_roundtrip);
    test_roundtrip!(i8, i8_roundtrip);
    test_roundtrip!(i16, i16_roundtrip);
    test_roundtrip!(i32, i32_roundtrip);
    test_roundtrip!(i64, i64_roundtrip);
    #[cfg(feature = "half")]
    test_roundtrip!(crate::dtype::f16, f16_roundtrip);
    test_roundtrip!(f32, f32_roundtrip);
    test_roundtrip!(f64, f64_roundtrip);
    test_roundtrip!(Complex<f32>, complex_f32_roundtrip);
    test_roundtrip!(Complex<f64>, complex_f64_roundtrip);
    test_roundtrip!(bool, bool_roundtrip);
}
