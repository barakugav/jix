use std::ops::Range;

use crate::array::Array;
use crate::codec::ReadContext;
use crate::dtype::{f16, Complex, Dtype, DtypeScalarKind, Dtyped};
use crate::error::{bail, check_get_buffer_size, check_get_range, ensure, Result};
use crate::storage::{ArrayStorage, ArrayStorageSpec};
use crate::util::DimArray;

impl<S> Array<S>
where
    S: ArrayStorage,
{
    /// Casts the element type of the array to `T`. See [`AsType`] for details and examples.
    ///
    /// # Panics
    ///
    /// Panics if the cast is unsupported.
    #[track_caller]
    pub fn astype<T>(self) -> Array<AsType<S>>
    where
        T: Dtyped,
    {
        self.astype_dyn(T::DTYPE)
    }

    /// Casts the element type of the array to a runtime `dtype`. See [`AsType`] for details and examples.
    ///
    /// Prefer `astype::<T>()` when the target dtype is known at compile time.
    ///
    /// # Panics
    ///
    /// Panics if the cast is unsupported.
    #[track_caller]
    pub fn astype_dyn(self, dtype: Dtype) -> Array<AsType<S>> {
        Array::from_storage(AsType::new(self, dtype).unwrap())
    }
}
/// Casts each element to a new dtype, returned by [`Array::astype`].
///
/// Supported casts:
/// - Between any two scalar types: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
///   `f16` (requires the `half` feature), `f32`, `f64`, `bool`.
/// - Between the two complex types: `Complex<f32>` <-> `Complex<f64>`.
///
/// `bool` conversions follow C semantics: zero -> `false`, any non-zero value -> `true`.
/// Casting between complex and non-complex types, or involving struct dtypes, is not supported.
///
/// Output dtype is the target dtype. Output shape equals the input shape.
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// This struct is the bare storage implementation, but the operation is also available as
/// [`Array::astype()`](crate::Array::astype) and [`Array::astype_dyn()`](crate::Array::astype_dyn).
///
/// # Examples
/// ```
/// use zix::{Array, ArrayParams};
/// use zix::dtype::Dtyped;
/// use ndarray::array;
///
/// let za = Array::compact_array(&array![1i32, 2, 3, 4])?;
/// let result = za.astype::<f64>().to_ndarray::<f64>()?;
/// assert_eq!(result.as_slice().unwrap(), &[1.0f64, 2.0, 3.0, 4.0]);
///
/// // Zero -> false, non-zero -> true
/// let zb = Array::compact_array(&array![0i32, 1, -2, 0])?;
/// let result = zb.astype::<bool>().to_ndarray::<bool>()?;
/// assert_eq!(result.as_slice().unwrap(), &[false, true, true, false]);
/// # Ok::<(), zix::Error>(())
/// ```
pub struct AsType<S> {
    array: Array<S>,

    dst_dtype: Dtype,
    shape: DimArray<u64>,
}
impl<S> AsType<S> {
    /// Constructs an `AsType` storage. See [`AsType`] for semantics, supported casts, and examples.
    pub fn new(array: Array<S>, dtype: Dtype) -> Result<Self>
    where
        S: ArrayStorage,
    {
        let src_dtype = array.dtype();
        ensure!(
            Self::is_cast_supported(src_dtype, &dtype),
            UnsupportedDtype,
            "unsupported cast from {src_dtype:?} to {dtype:?}"
        );

        Ok(Self {
            dst_dtype: dtype,
            shape: array.shape().try_into().unwrap(),
            array,
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
    fn read_data(&self, index: &[Range<u64>], buf: &mut [u8], context: &ReadContext) -> Result<()> {
        check_get_range(&self.shape, index)?;
        let nitems = check_get_buffer_size(index, &self.dst_dtype, buf)?;

        let (src_dtype, dst_dtype) = (self.array.dtype(), &self.dst_dtype);
        let (src_itemsize, dst_itemsize) =
            (src_dtype.itemsize() as usize, dst_dtype.itemsize() as usize);

        let in_place = src_itemsize == dst_itemsize
            && (buf.as_ptr() as usize).is_multiple_of(src_dtype.alignment().as_usize());
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
        self.array.storage.read_data(index, read_buf, context)?;
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
        macro_rules! cast_loop {
            ($src_type:ident, $dst_type:ident) => {
                for i in 0..nitems {
                    unsafe {
                        let value = src.cast::<$src_type>().add(i).read();
                        let value = crate::ops::astype::cast::<$src_type, $dst_type>(value);
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
                            let value =
                                crate::ops::astype::cast::<Complex<f32>, Complex<f64>>(value);
                            dst.cast::<Complex<f64>>().add(i).write(value);
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
                            let value =
                                crate::ops::astype::cast::<Complex<f64>, Complex<f32>>(value);
                            dst.cast::<Complex<f32>>().add(i).write(value);
                        }
                    }
                }
                DtypeScalarKind::ComplexF64 => {}
                _ => supported = false,
            },
            DtypeScalarKind::Bool => cast_num!(bool),
        };

        if !supported {
            bail!(
                UnsupportedDtype,
                "unsupported cast from {src_scalar:?} to {dst_scalar:?}"
            );
        }
        Ok(())
    }

    fn shape(&self) -> &[u64] {
        &self.shape
    }
    fn dtype(&self) -> &Dtype {
        &self.dst_dtype
    }
    fn _spec(&self) -> ArrayStorageSpec<'_> {
        self.array.storage._spec()
    }
}

pub(crate) fn cast<S, D>(value: S) -> D
where
    S: Cast<D>,
{
    value.cast()
}
pub(crate) fn cast_as<S, D>(value: S, #[allow(unused_variables)] like: &D) -> D
where
    S: Cast<D>,
{
    value.cast()
}
pub(crate) trait Cast<D> {
    fn cast(self) -> D;
}

macro_rules! impl_cast {
    ($src_type:ident => $dst_type:ident) => {
        impl Cast<$dst_type> for $src_type {
            fn cast(self) -> $dst_type {
                #![allow(clippy::redundant_locals)]
                let value = self;
                let value = impl_cast!(@from $src_type, value);
                let value = impl_cast!(@to $dst_type, value);
                value
            }
        }
    };
    (@from bool, $value:expr) => {
        ($value) as i8
    };
    (@from f16, $value:expr) => {
        ($value).to_f32()
    };
    (@from $type:ident, $value:expr) => {
        $value
    };

    (@to bool, $value:expr) => {
        ($value) != (0 as _)
    };
    (@to f16, $value:expr) => {
        f16::from_f32(($value) as f32)
    };
    (@to $type:ident, $value:expr) => {
        ($value) as $type
    };
}

macro_rules! impl_cast_num {
    ($src_type:ident) => {
        impl_cast!($src_type => i8);
        impl_cast!($src_type => i16);
        impl_cast!($src_type => i32);
        impl_cast!($src_type => i64);
        impl_cast!($src_type => u8);
        impl_cast!($src_type => u16);
        impl_cast!($src_type => u32);
        impl_cast!($src_type => u64);
        #[cfg(feature = "half")]
        impl_cast!($src_type => f16);
        impl_cast!($src_type => f32);
        impl_cast!($src_type => f64);
        impl_cast!($src_type => bool);
        #[cfg(feature = "half")]
        impl_cast_num!(@impl_to_complex, $src_type, Complex<f16>);
        impl_cast_num!(@impl_to_complex, $src_type, Complex<f32>);
        impl_cast_num!(@impl_to_complex, $src_type, Complex<f64>);
    };

    (@impl_to_complex, $src_type:ident, Complex<$dst_type:ident>) => {
        impl Cast<Complex<$dst_type>> for $src_type {
            fn cast(self) -> Complex<$dst_type> {
                Complex {
                    re: crate::ops::astype::cast(self),
                    im: crate::ops::astype::cast(0.0),
                }
            }
        }
    };
}
impl_cast_num!(i8);
impl_cast_num!(i16);
impl_cast_num!(i32);
impl_cast_num!(i64);
impl_cast_num!(u8);
impl_cast_num!(u16);
impl_cast_num!(u32);
impl_cast_num!(u64);
#[cfg(feature = "half")]
impl_cast_num!(f16);
impl_cast_num!(f32);
impl_cast_num!(f64);
impl_cast_num!(bool);

#[cfg(not(feature = "half"))]
impl Cast<f16> for f16 {
    fn cast(self) -> f16 {
        self
    }
}

macro_rules! impl_cast_complex_to_complex {
    ($src_type:ident, $dst_type:ident) => {
        impl Cast<Complex<$dst_type>> for Complex<$src_type> {
            fn cast(self) -> Complex<$dst_type> {
                Complex {
                    re: crate::ops::astype::cast::<$src_type, $dst_type>(self.re),
                    im: crate::ops::astype::cast::<$src_type, $dst_type>(self.im),
                }
            }
        }
    };
}
macro_rules! impl_cast_complex {
    ($src_type:ident) => {
        #[cfg(feature = "half")]
        impl_cast_complex_to_complex!($src_type, f16);
        impl_cast_complex_to_complex!($src_type, f32);
        impl_cast_complex_to_complex!($src_type, f64);

        impl Cast<bool> for Complex<$src_type> {
            fn cast(self) -> bool {
                self != (cast::<bool, Self>(false))
            }
        }
    };
}
#[cfg(feature = "half")]
impl_cast_complex!(f16);
impl_cast_complex!(f32);
impl_cast_complex!(f64);

#[cfg(test)]
mod tests {
    use ndarray::array;

    use crate::array::Array;
    #[allow(unused_imports)]
    use crate::dtype::f16;
    use crate::dtype::Complex;
    use crate::util::arr_params;
    #[allow(non_camel_case_types)]
    type complex_f32 = crate::dtype::Complex<f32>;
    #[allow(non_camel_case_types)]
    type complex_f64 = crate::dtype::Complex<f64>;

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
    // One proptest function per (src -> dst) cast pair: random shape, random block shape,
    // full read + random sub-range reads via assert_array_matches.
    macro_rules! test_cast_pair {
        ($src:ty, $dst:ty) => {
            paste::paste! {
                proptest::proptest! {
                    #[test]
                    fn [<cast_ $src:lower _to_ $dst:lower>](
                        (nd, za) in crate::util::carray_strategy_from_shape::<$src>(
                            crate::util::shape_strategy(),
                            <$src as crate::util::ScalarStrategy>::any_strategy(),
                        )
                    ) {
                        let result = za.astype::<$dst>();
                        let expected = nd.mapv(|x| CastTo::<$dst>::cast_to(x));
                        crate::util::assert_array_matches(&result, &expected);
                    }
                }
            }
        };
    }

    // numeric widening / narrowing
    test_cast_pair!(u8, f32); // smaller -> larger itemsize
    test_cast_pair!(f32, u8); // larger -> smaller itemsize
    test_cast_pair!(i32, f64);
    test_cast_pair!(f64, i32);
    test_cast_pair!(f32, f64);
    test_cast_pair!(f64, f32);
    test_cast_pair!(i8, u8);
    test_cast_pair!(u8, i8);
    // identity cast (same src and dst dtype)
    test_cast_pair!(i32, i32);
    test_cast_pair!(f64, f64);
    // bool
    test_cast_pair!(i8, bool);
    test_cast_pair!(bool, i8);
    test_cast_pair!(u8, bool);
    test_cast_pair!(bool, u8);
    test_cast_pair!(i32, bool);
    test_cast_pair!(bool, i32);
    test_cast_pair!(f32, bool);
    test_cast_pair!(bool, f32);
    // f16 (feature-gated)
    #[cfg(feature = "half")]
    test_cast_pair!(i32, f16);
    #[cfg(feature = "half")]
    test_cast_pair!(f16, i32);
    #[cfg(feature = "half")]
    test_cast_pair!(f32, f16);
    #[cfg(feature = "half")]
    test_cast_pair!(f16, f32);
    #[cfg(feature = "half")]
    test_cast_pair!(f64, f16);
    #[cfg(feature = "half")]
    test_cast_pair!(f16, f64);
    #[cfg(feature = "half")]
    test_cast_pair!(f16, f16);
    // complex (feature-gated)
    test_cast_pair!(complex_f32, complex_f64);
    #[cfg(feature = "num-complex")]
    test_cast_pair!(complex_f64, complex_f32);
    #[cfg(feature = "num-complex")]
    test_cast_pair!(complex_f32, complex_f32);
    #[cfg(feature = "num-complex")]
    test_cast_pair!(complex_f64, complex_f64);

    // --- error cases ---
    #[test]
    #[should_panic]
    fn cast_complex_to_real_fails() {
        let a = Array::compact_array_with(&array![Complex { re: 1.0, im: 2.0 }], arr_params(&[1]))
            .unwrap();
        let _ = a.astype::<f32>();
    }

    #[test]
    #[should_panic]
    fn cast_real_to_complex_fails() {
        let a = Array::compact_array_with(&array![1.0f32], arr_params(&[1])).unwrap();
        let _ = a.astype::<Complex<f32>>();
    }

    #[cfg(not(feature = "half"))]
    #[test]
    fn cast_f16_without_feature_fails() {
        let a = Array::compact_array_with(&array![1.0f32], arr_params(&[1])).unwrap();
        assert!(std::panic::catch_unwind(|| a.astype::<f16>()).is_err());
        let a = Array::compact_array_with(&array![f16::from_bits(17)], arr_params(&[1])).unwrap();
        assert!(std::panic::catch_unwind(|| a.astype::<f32>()).is_err());
        let a = Array::compact_array_with(&array![f16::from_bits(17)], arr_params(&[1])).unwrap();
        let _ = a.astype::<f16>(); // no-op cast must not panic
    }

    #[cfg(not(feature = "half"))]
    #[test]
    fn cast_f16_to_f16() {
        // must work even without the "half" feature, since it's a no-op cast
        let a = Array::compact_array_with(&array![f16::from_bits(17)], arr_params(&[1])).unwrap();
        let _ = a.astype::<f16>();
    }
}
