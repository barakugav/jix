use std::io;
use std::ops::Range;

use crate::array::Array;
use crate::codec::{DecoderParams, EncoderParams, ReadContext};
#[cfg(feature = "half")]
use crate::dtype::f16;
use crate::dtype::{Complex, Dtype, DtypeScalarKind};
use crate::storage::{ArrayStorage, BlocksLayout, Ref};
use crate::util::DimArray;

impl<S> Array<S>
where
    S: ArrayStorage,
{
    #[track_caller]
    pub fn astype(&self, dtype: Dtype) -> Array<AsType<Ref<'_, S>>> {
        self.try_astype(dtype).unwrap()
    }

    pub fn try_astype(&self, dtype: Dtype) -> io::Result<Array<AsType<Ref<'_, S>>>> {
        let a = Array::from_storage(Ref(&self.storage));
        Ok(Array::from_storage(AsType::new(a, dtype)?))
    }
}
pub struct AsType<S> {
    a: Array<S>,

    dtype: Dtype,
    shape: DimArray<u64>,
    blocks_layout: BlocksLayout,
}
impl<S> AsType<S> {
    pub(crate) fn new(a: Array<S>, dtype: Dtype) -> io::Result<Self>
    where
        S: ArrayStorage,
    {
        let src_dtype = a.dtype();
        if !Self::is_cast_supported(src_dtype, &dtype) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unsupported cast from {src_dtype:?} to {dtype:?}"),
            ));
        }

        Ok(Self {
            dtype,
            shape: a.shape().try_into().unwrap(),
            blocks_layout: a.blocks_layout().clone(),
            a,
        })
    }

    fn is_cast_supported(src_dtype: &Dtype, dst_dtype: &Dtype) -> bool {
        if src_dtype == dst_dtype {
            return true; // same dtype, always supported
        }

        let (Some(src_scalar), Some(dst_scalar)) =
            (src_dtype.try_to_scalar(), dst_dtype.try_to_scalar())
        else {
            return false; // non scalar
        };

        if !cfg!(feature = "half")
            && (src_scalar == DtypeScalarKind::F16 || dst_scalar == DtypeScalarKind::F16)
        {
            return false; // f16 not supported without "half" feature
        }

        let is_complex = |kind| {
            matches!(
                kind,
                DtypeScalarKind::ComplexF32 | DtypeScalarKind::ComplexF64
            )
        };
        if is_complex(src_scalar) != is_complex(dst_scalar) {
            return false; // cannot cast between complex and non-complex
        }

        true
    }
}
impl<S> ArrayStorage for AsType<S>
where
    S: ArrayStorage,
{
    fn shape(&self) -> &[u64] {
        &self.shape
    }

    fn dtype(&self) -> &Dtype {
        &self.dtype
    }

    fn read_data(
        &self,
        index: &[Range<u64>],
        buf: &mut [u8],
        context: &ReadContext,
    ) -> io::Result<()> {
        let (src_dtype, dst_dtype) = (self.a.dtype(), &self.dtype);
        let (src_itemsize, dst_itemsize) =
            (src_dtype.itemsize() as usize, dst_dtype.itemsize() as usize);
        let nitems = buf.len() / dst_itemsize;

        let in_place = src_itemsize == dst_itemsize
            && (buf.as_ptr() as usize).is_multiple_of(src_dtype.alignment() as usize);
        let mut tmp_buf;
        let (read_buf, dst) = if in_place {
            let ptr = buf.as_mut_ptr();
            ((ptr, buf.len()), ptr)
        } else {
            tmp_buf = context.tmp_buf(nitems * src_itemsize, src_dtype.alignment());
            let tmp_buf = tmp_buf.as_mut_slice();
            ((tmp_buf.as_mut_ptr(), tmp_buf.len()), buf.as_mut_ptr())
        };
        let read_buf = unsafe { std::slice::from_raw_parts_mut(read_buf.0, read_buf.1) };
        self.a.storage.read_data(index, read_buf, context)?;
        let src = read_buf.as_ptr();

        if src_dtype == dst_dtype {
            debug_assert!(in_place);
            return Ok(());
        }
        let (src_scalar, dst_scalar) = (
            src_dtype.try_to_scalar().unwrap(),
            dst_dtype.try_to_scalar().unwrap(),
        );

        let mut supported = true;
        macro_rules! cast_from {
            ($value:expr, bool) => {
                ($value) as i8
            };
            ($value:expr, f16) => {
                ($value).to_f32()
            };
            ($value:expr, $type:ident) => {
                $value
            };
        }
        macro_rules! cast_to {
            ($value:expr, bool) => {
                ($value) != (0 as _)
            };
            ($value:expr, f16) => {
                f16::from_f32(($value) as f32)
            };
            ($value:expr, $type:ident) => {
                ($value) as $type
            };
        }
        macro_rules! cast_loop {
            ($src_type:ident, $dst_type:ident) => {
                for i in 0..nitems {
                    unsafe {
                        #![allow(clippy::redundant_locals)]
                        let value = src.cast::<$src_type>().add(i).read();
                        let value = cast_from!(value, $src_type);
                        let value = cast_to!(value, $dst_type);
                        dst.cast::<$dst_type>().add(i).write(value);
                    }
                }
            };
        }
        macro_rules! cast_num {
            ($src_type:ident) => {
                match dst_scalar {
                    DtypeScalarKind::I8 => cast_loop!($src_type, i8),
                    DtypeScalarKind::I16 => cast_loop!($src_type, i16),
                    DtypeScalarKind::I32 => cast_loop!($src_type, i32),
                    DtypeScalarKind::I64 => cast_loop!($src_type, i64),
                    DtypeScalarKind::U8 => cast_loop!($src_type, u8),
                    DtypeScalarKind::U16 => cast_loop!($src_type, u16),
                    DtypeScalarKind::U32 => cast_loop!($src_type, u32),
                    DtypeScalarKind::U64 => cast_loop!($src_type, u64),
                    DtypeScalarKind::F16 => {
                        cfg_if::cfg_if! { if #[cfg(feature = "half")] {
                           cast_loop!($src_type, f16);
                        } else {
                            supported = false;
                        } }
                    }
                    DtypeScalarKind::F32 => cast_loop!($src_type, f32),
                    DtypeScalarKind::F64 => cast_loop!($src_type, f64),
                    DtypeScalarKind::Bool => cast_loop!($src_type, bool),
                    DtypeScalarKind::ComplexF32 | DtypeScalarKind::ComplexF64 => supported = false,
                }
            };
        }
        match src_scalar {
            DtypeScalarKind::I8 => cast_num!(i8),
            DtypeScalarKind::I16 => cast_num!(i16),
            DtypeScalarKind::I32 => cast_num!(i32),
            DtypeScalarKind::I64 => cast_num!(i64),
            DtypeScalarKind::U8 => cast_num!(u8),
            DtypeScalarKind::U16 => cast_num!(u16),
            DtypeScalarKind::U32 => cast_num!(u32),
            DtypeScalarKind::U64 => cast_num!(u64),
            DtypeScalarKind::F16 => {
                cfg_if::cfg_if! { if #[cfg(feature = "half")] {
                   cast_num!(f16);
                } else {
                    supported = false;
                } }
            }
            DtypeScalarKind::F32 => cast_num!(f32),
            DtypeScalarKind::F64 => cast_num!(f64),
            DtypeScalarKind::ComplexF32 => match dst_scalar {
                DtypeScalarKind::ComplexF32 => {}
                DtypeScalarKind::ComplexF64 => {
                    for i in 0..nitems {
                        unsafe {
                            let value = src.cast::<Complex<f32>>().add(i).read();
                            dst.cast::<Complex<f64>>().add(i).write(Complex {
                                re: value.re as f64,
                                im: value.im as f64,
                            });
                        }
                    }
                }
                _ => supported = false,
            },
            DtypeScalarKind::ComplexF64 => match dst_scalar {
                DtypeScalarKind::ComplexF32 => {
                    for i in 0..nitems {
                        unsafe {
                            let value = src.cast::<Complex<f64>>().add(i).read();
                            dst.cast::<Complex<f32>>().add(i).write(Complex {
                                re: value.re as f32,
                                im: value.im as f32,
                            });
                        }
                    }
                }
                DtypeScalarKind::ComplexF64 => {}
                _ => supported = false,
            },
            DtypeScalarKind::Bool => cast_num!(bool),
        };

        if !supported {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unsupported cast from {src_scalar:?} to {dst_scalar:?}"),
            ));
        }

        Ok(())
    }

    fn blocks_layout(&self) -> &BlocksLayout {
        &self.blocks_layout
    }
    fn codec_params(&self) -> (&EncoderParams, &DecoderParams) {
        self.a.storage.codec_params()
    }
}

#[cfg(test)]
mod tests {
    use crate::array::{Array, ArrayParams};
    use crate::block::BlockSize;
    #[allow(unused_imports)]
    use crate::dtype::f16;
    use crate::dtype::{Complex, Dtyped};

    fn arr_params(block_shape: &[usize]) -> ArrayParams {
        ArrayParams {
            block_shape: Some(block_shape.iter().map(|&x| x as BlockSize).collect()),
            ..ArrayParams::default()
        }
    }

    // --- scalar sampling ---
    trait Scalar: Sized + Copy + 'static {
        fn sample(rng: &mut fastrand::Rng) -> Self;
    }
    macro_rules! impl_scalar {
        ($($t:ty),+) => {
            $(impl Scalar for $t {
                fn sample(rng: &mut fastrand::Rng) -> Self {
                    (rng.f64() * 9.0 + 1.0) as Self
                }
            })+
        };
    }
    impl_scalar!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);
    impl Scalar for bool {
        fn sample(rng: &mut fastrand::Rng) -> Self {
            rng.bool()
        }
    }

    fn rand_array<T: Scalar>(rng: &mut fastrand::Rng, shape: &[usize]) -> ndarray::ArrayD<T> {
        ndarray::Array::from_shape_fn(ndarray::IxDyn(shape), |_| T::sample(rng))
    }

    // --- element-wise conversion for expected values ---
    trait CastTo<D>: Copy {
        fn cast_to(self) -> D;
    }
    macro_rules! impl_cast_as_single {
        (bool => bool) => {
            impl CastTo<bool> for bool {
                fn cast_to(self) -> bool {
                    self
                }
            }
        };
        (bool => $dst:ty) => {
            impl CastTo<$dst> for bool {
                fn cast_to(self) -> $dst {
                    self as i8 as $dst
                }
            }
        };
        ($src:ty => bool) => {
            impl CastTo<bool> for $src {
                fn cast_to(self) -> bool {
                    self != (0 as $src)
                }
            }
        };
        ($src:ty => $dst:ty) => {
            impl CastTo<$dst> for $src {
                fn cast_to(self) -> $dst {
                    self as $dst
                }
            }
        };
    }
    macro_rules! impl_cast_as {
        ($src:tt => $($dst:tt),+) => {
            $(impl_cast_as_single!($src => $dst);)+
        };
    }
    impl_cast_as!(i8   => i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool);
    impl_cast_as!(i16  => i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool);
    impl_cast_as!(i32  => i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool);
    impl_cast_as!(i64  => i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool);
    impl_cast_as!(u8   => i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool);
    impl_cast_as!(u16  => i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool);
    impl_cast_as!(u32  => i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool);
    impl_cast_as!(u64  => i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool);
    impl_cast_as!(f32  => i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool);
    impl_cast_as!(f64  => i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool);
    impl_cast_as!(bool => i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool);

    #[cfg(feature = "half")]
    mod f16_impls {
        use super::*;
        impl Scalar for f16 {
            fn sample(rng: &mut fastrand::Rng) -> Self {
                f16::from_f32(rng.f32() * 9.0 + 1.0)
            }
        }
        // f16 cannot use `as` for conversions; go through f32
        macro_rules! impl_cast_to_f16 {
        ($($src:ty),+) => {
            $(impl CastTo<f16> for $src {
                fn cast_to(self) -> f16 { f16::from_f32(self as f32) }
            })+
        }
    }
        impl_cast_to_f16!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);
        impl CastTo<f16> for f16 {
            fn cast_to(self) -> f16 {
                self
            }
        }
        impl CastTo<f16> for bool {
            fn cast_to(self) -> f16 {
                f16::from_f32(self as u8 as f32)
            }
        }
        macro_rules! impl_cast_from_f16 {
        ($($dst:ty),+) => {
            $(impl CastTo<$dst> for f16 {
                fn cast_to(self) -> $dst { self.to_f32() as $dst }
            })+
        }
    }
        impl_cast_from_f16!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);
        impl CastTo<bool> for f16 {
            fn cast_to(self) -> bool {
                self.to_f32() != 0.0
            }
        }
    }

    impl Scalar for Complex<f32> {
        fn sample(rng: &mut fastrand::Rng) -> Self {
            Self {
                re: rng.f32() * 9.0 + 1.0,
                im: rng.f32() * 9.0 + 1.0,
            }
        }
    }
    impl Scalar for Complex<f64> {
        fn sample(rng: &mut fastrand::Rng) -> Self {
            Self {
                re: rng.f64() * 9.0 + 1.0,
                im: rng.f64() * 9.0 + 1.0,
            }
        }
    }
    impl CastTo<Complex<f64>> for Complex<f32> {
        fn cast_to(self) -> Complex<f64> {
            Complex {
                re: self.re as f64,
                im: self.im as f64,
            }
        }
    }
    impl CastTo<Complex<f32>> for Complex<f64> {
        fn cast_to(self) -> Complex<f32> {
            Complex {
                re: self.re as f32,
                im: self.im as f32,
            }
        }
    }
    impl CastTo<Complex<f32>> for Complex<f32> {
        fn cast_to(self) -> Complex<f32> {
            self
        }
    }
    impl CastTo<Complex<f64>> for Complex<f64> {
        fn cast_to(self) -> Complex<f64> {
            self
        }
    }

    // --- test generation macro ---
    // Generates 4 test functions for a (src → dst) cast pair.
    macro_rules! test_cast_pair {
        ($mod_name:ident, $src:ty, $dst:ty) => {
            mod $mod_name {
                use super::*;
                use crate::dtype::Dtyped;

                fn seed() -> u64 {
                    stringify!($mod_name)
                        .as_bytes()
                        .iter()
                        .fold(0xdeadbeef_cafe1234u64, |acc, b| acc.wrapping_add(*b as u64))
                        .swap_bytes()
                }

                #[test]
                fn cast_1d() {
                    let mut rng = fastrand::Rng::with_seed(seed());
                    let a = rand_array::<$src>(&mut rng, &[8]);
                    let za = Array::from_ndarray(&a, arr_params(&[8])).unwrap();
                    let actual = za
                        .astype(<$dst>::dtype())
                        .data()
                        .to_ndarray::<$dst>()
                        .unwrap();
                    let expected = a.mapv(|x| CastTo::<$dst>::cast_to(x));
                    assert_eq!(actual, expected);
                }

                #[test]
                fn cast_1d_multi_block() {
                    let mut rng = fastrand::Rng::with_seed(seed() ^ 1);
                    let a = rand_array::<$src>(&mut rng, &[6]);
                    let za = Array::from_ndarray(&a, arr_params(&[2])).unwrap();
                    let actual = za
                        .astype(<$dst>::dtype())
                        .data()
                        .to_ndarray::<$dst>()
                        .unwrap();
                    let expected = a.mapv(|x| CastTo::<$dst>::cast_to(x));
                    assert_eq!(actual, expected);
                }

                #[test]
                fn cast_2d() {
                    let mut rng = fastrand::Rng::with_seed(seed() ^ 2);
                    let a = rand_array::<$src>(&mut rng, &[3, 4]);
                    let za = Array::from_ndarray(&a, arr_params(&[3, 4])).unwrap();
                    let actual = za
                        .astype(<$dst>::dtype())
                        .data()
                        .to_ndarray::<$dst>()
                        .unwrap();
                    let expected = a.mapv(|x| CastTo::<$dst>::cast_to(x));
                    assert_eq!(actual, expected);
                }

                #[test]
                fn cast_2d_multi_block() {
                    let mut rng = fastrand::Rng::with_seed(seed() ^ 3);
                    let a = rand_array::<$src>(&mut rng, &[4, 4]);
                    let za = Array::from_ndarray(&a, arr_params(&[2, 2])).unwrap();
                    let actual = za
                        .astype(<$dst>::dtype())
                        .data()
                        .to_ndarray::<$dst>()
                        .unwrap();
                    let expected = a.mapv(|x| CastTo::<$dst>::cast_to(x));
                    assert_eq!(actual, expected);
                }
            }
        };
    }

    // numeric widening / narrowing
    test_cast_pair!(u8_to_f32, u8, f32); // smaller → larger itemsize
    test_cast_pair!(f32_to_u8, f32, u8); // larger → smaller itemsize
    test_cast_pair!(i32_to_f64, i32, f64);
    test_cast_pair!(f64_to_i32, f64, i32);
    test_cast_pair!(f32_to_f64, f32, f64);
    test_cast_pair!(f64_to_f32, f64, f32);
    test_cast_pair!(i8_to_u8, i8, u8);
    test_cast_pair!(u8_to_i8, u8, i8);
    // identity cast (same src and dst dtype)
    test_cast_pair!(i32_to_i32, i32, i32);
    test_cast_pair!(f64_to_f64, f64, f64);
    // bool
    test_cast_pair!(i8_to_bool, i8, bool);
    test_cast_pair!(bool_to_i8, bool, i8);
    test_cast_pair!(u8_to_bool, u8, bool);
    test_cast_pair!(bool_to_u8, bool, u8);
    test_cast_pair!(i32_to_bool, i32, bool);
    test_cast_pair!(bool_to_i32, bool, i32);
    test_cast_pair!(f32_to_bool, f32, bool);
    test_cast_pair!(bool_to_f32, bool, f32);
    // f16 (feature-gated)
    #[cfg(feature = "half")]
    test_cast_pair!(i32_to_f16, i32, f16);
    #[cfg(feature = "half")]
    test_cast_pair!(f16_to_i32, f16, i32);
    #[cfg(feature = "half")]
    test_cast_pair!(f32_to_f16, f32, f16);
    #[cfg(feature = "half")]
    test_cast_pair!(f16_to_f32, f16, f32);
    #[cfg(feature = "half")]
    test_cast_pair!(f64_to_f16, f64, f16);
    #[cfg(feature = "half")]
    test_cast_pair!(f16_to_f64, f16, f64);
    #[cfg(feature = "half")]
    test_cast_pair!(f16_to_f16, f16, f16);
    // complex (feature-gated)
    test_cast_pair!(cf32_to_cf64, Complex<f32>, Complex<f64>);
    #[cfg(feature = "num-complex")]
    test_cast_pair!(cf64_to_cf32, Complex<f64>, Complex<f32>);
    #[cfg(feature = "num-complex")]
    test_cast_pair!(cf32_to_cf32, Complex<f32>, Complex<f32>);
    #[cfg(feature = "num-complex")]
    test_cast_pair!(cf64_to_cf64, Complex<f64>, Complex<f64>);

    // --- error cases ---
    #[test]
    fn cast_complex_to_real_fails() {
        let a = Array::from_ndarray(
            &ndarray::array![Complex { re: 1.0, im: 2.0 }].into_dyn(),
            arr_params(&[1]),
        )
        .unwrap();
        assert!(a.try_astype(f32::dtype()).is_err());
    }

    #[test]
    fn cast_real_to_complex_fails() {
        let a = Array::from_ndarray(&ndarray::array![1.0f32].into_dyn(), arr_params(&[1])).unwrap();
        assert!(a.try_astype(Complex::<f32>::dtype()).is_err());
    }

    #[cfg(not(feature = "half"))]
    #[test]
    fn cast_f16_without_feature_fails() {
        let a = Array::from_ndarray(&ndarray::array![1.0f32].into_dyn(), arr_params(&[1])).unwrap();
        assert!(a.try_astype(f16::dtype()).is_err());
        let a = Array::from_ndarray(
            &ndarray::array![f16::from_bits(17)].into_dyn(),
            arr_params(&[1]),
        )
        .unwrap();
        assert!(a.try_astype(f32::dtype()).is_err());
        let a = Array::from_ndarray(
            &ndarray::array![f16::from_bits(17)].into_dyn(),
            arr_params(&[1]),
        )
        .unwrap();
        assert!(a.try_astype(f16::dtype()).is_ok());
    }

    #[test]
    fn cast_f16_to_f16() {
        // must work even without the "half" feature, since it's a no-op cast
        let a = Array::from_ndarray(
            &ndarray::array![f16::from_bits(17)].into_dyn(),
            arr_params(&[1]),
        )
        .unwrap();
        assert!(a.try_astype(f16::dtype()).is_ok());
    }
}
