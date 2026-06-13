use crate::dtype::Dtyped;
use crate::storage::ArrayStorageTyped;
use crate::{Array, ArrayStorage};

pub(crate) mod _traits {
    #[allow(unused_imports)]
    use crate::scalar::{f16, Complex};

    /// Cast a scalar value to another scalar type.
    ///
    /// Supported casts:
    /// - Between any two integer types: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`.
    /// - Between any two floating-point types: `f16` (requires the `half` feature), `f32`, `f64`.
    /// - Between any two complex types: `Complex<f32>`, `Complex<f64>`.
    /// - Between any integer to floating-point type, or vice versa.
    /// - Between any integer or floating-point type to complex type, but NOT from complex to integer.
    /// - Between `bool` and any other scalar, and visa versa: zero -> `false`, any non-zero value -> `true`.
    pub trait Cast<D> {
        /// Casts `self` to `D`.
        fn cast(self) -> D;
    }
    macro_rules! impl_cast {
        ($src_type:ident => $dst_type:ident) => {
            impl Cast<$dst_type> for $src_type {
                #[inline(always)]
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
                #[inline(always)]
                fn cast(self) -> Complex<$dst_type> {
                    Complex {
                        re: <_ as crate::scalar::Cast<$dst_type>>::cast(self),
                        im: <_ as crate::scalar::Cast<$dst_type>>::cast(0.0),
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
        #[inline(always)]
        fn cast(self) -> f16 {
            self
        }
    }

    macro_rules! impl_cast_complex_to_complex {
        ($src_type:ident, $dst_type:ident) => {
            impl Cast<Complex<$dst_type>> for Complex<$src_type> {
                #[inline(always)]
                fn cast(self) -> Complex<$dst_type> {
                    Complex {
                        re: <_ as crate::scalar::Cast<$dst_type>>::cast(self.re),
                        im: <_ as crate::scalar::Cast<$dst_type>>::cast(self.im),
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
                #[inline(always)]
                fn cast(self) -> bool {
                    self != (<_ as crate::scalar::Cast<Self>>::cast(false))
                }
            }
        };
    }
    #[cfg(feature = "half")]
    impl_cast_complex!(f16);
    impl_cast_complex!(f32);
    impl_cast_complex!(f64);
}

/// Casts each element to a new element type, returned by [`Array::cast`].
///
/// Supported casts:
/// - Between any two integer types: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`.
/// - Between any two floating-point types: `f16` (requires the `half` feature), `f32`, `f64`.
/// - Between any two complex types: `Complex<f32>`, `Complex<f64>`.
/// - Between any integer to floating-point type, or vice versa.
/// - Between any integer or floating-point type to complex type, but NOT from complex to integer.
/// - Between `bool` and any other scalar, and visa versa: zero -> `false`, any non-zero value -> `true`.
///
/// The cast is checked via the [`Cast<T>`](crate::scalar::Cast) trait bound on the source element
/// type; unsupported source-to-target type pairs are rejected at compile time.
///
/// Output dtype is the target dtype; output element type is `Ty<T>`. Output shape equals the
/// input shape.
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// This struct is the bare storage implementation, the operation is also available as
/// [`Array::cast()`](crate::Array::cast).
///
/// # Examples
/// ```
/// use jix::dtype::Dtyped;
/// use jix::Array;
/// use ndarray::array;
///
/// let za = Array::compact_ndarray(&array![1i32, 2, 3, 4])?;
/// let result = za.cast::<f64>().to_ndarray()?;
/// assert_eq!(result.as_slice().unwrap(), &[1.0f64, 2.0, 3.0, 4.0]);
///
/// // Zero -> false, non-zero -> true
/// let zb = Array::compact_ndarray(&array![0i32, 1, -2, 0])?;
/// let result = zb.cast::<bool>().to_ndarray()?;
/// assert_eq!(result.as_slice().unwrap(), &[false, true, true, false]);
/// # Ok::<(), jix::Error>(())
/// ```
pub struct Cast<S, T>(crate::ops::op1::Op1<S, CastKernel<T>>);
struct CastKernel<T>(std::marker::PhantomData<T>);
impl<T1, T2> crate::ops::op1::Op1Kernel<T1> for CastKernel<T2>
where
    T1: crate::scalar::Cast<T2>,
{
    type Output = T2;
    #[inline(always)]
    fn apply(&self, x: T1) -> Self::Output {
        x.cast()
    }
}
impl<S, T> Cast<S, T>
where
    S: ArrayStorageTyped,
    S::Item: crate::scalar::Cast<T>,
    T: Dtyped,
{
    /// Constructs a [`Cast`] storage. See the struct docs for semantics and examples.
    pub fn new(array: S) -> crate::error::Result<Self> {
        let kernel = CastKernel(std::marker::PhantomData);
        Ok(Self(crate::ops::op1::Op1::new(array, kernel)?))
    }

    /// Constructs an array with [`Cast`] storage. See the storage struct docs for semantics and examples.
    pub fn new_array(array: Array<S>) -> crate::error::Result<Array<Self>> {
        Self::new(array.into_storage()).map(Array::from_storage)
    }
}
impl<S, T> ArrayStorage for Cast<S, T>
where
    S: ArrayStorageTyped,
    S::Item: crate::scalar::Cast<T>,
    T: Dtyped,
{
    type ElementType = crate::Ty<T>;
    type Dimension = S::Dimension;
    crate::storage::impl_array_storage_forward!('a, T2, <S, T>);

    type DimensionChange<NewD: crate::Dimension> = Cast<S::DimensionChange<NewD>, T>;
    #[inline]
    fn dimension_change<NewD: crate::Dimension>(
        self,
    ) -> crate::error::Result<Self::DimensionChange<NewD>> {
        Ok(Cast(self.0.dimension_change()?))
    }
}

impl<S> Array<S>
where
    S: ArrayStorage,
{
    /// Casts each element to scalar type `T`. See [`Cast`] for details and examples.
    #[track_caller]
    pub fn cast<T>(self) -> Array<Cast<S, T>>
    where
        S: ArrayStorageTyped,
        S::Item: crate::scalar::Cast<T>,
        T: Dtyped,
    {
        Cast::new_array(self).unwrap()
    }
}

#[cfg(test)]
mod tests {

    #[allow(unused_imports)]
    use crate::scalar::f16;
    use crate::scalar::Complex;
    #[allow(non_camel_case_types)]
    type complex_f32 = crate::scalar::Complex<f32>;
    #[allow(non_camel_case_types)]
    type complex_f64 = crate::scalar::Complex<f64>;

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
    impl CastTo<f16> for f16 {
        fn cast_to(self) -> f16 {
            self
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
                        let result = za.cast::<$dst>();
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

    #[cfg(not(feature = "half"))]
    #[test]
    fn cast_f16_to_f16() {
        // must work even without the "half" feature, since it's a no-op cast

        use ndarray::array;

        use crate::util::arr_params;
        use crate::Array;

        let a = Array::compact_ndarray_with(&array![f16::from_bits(17)], arr_params(&[1])).unwrap();
        let _ = a.cast::<f16>();
    }
}
