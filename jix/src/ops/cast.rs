use crate::dtype::Dtyped;
use crate::storage::{ArrayStorageInfo, ArrayStorageTyped};
use crate::{Array, ArrayStorage};

pub(crate) mod _traits {
    #[cfg(feature = "half")]
    use crate::scalar::f16;
    #[cfg(feature = "num-complex")]
    use crate::scalar::Complex;

    /// Cast a scalar value to another scalar type.
    ///
    /// Supported casts:
    /// - Between any two integer types: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`.
    /// - Between any two floating-point types: `f16` (requires the `half` feature), `f32`, `f64`.
    /// - Between any two complex types: `Complex<f16>` (requires the `half` feature), `Complex<f32>`, `Complex<f64>`.
    /// - Between any integer to floating-point type, or vice versa.
    /// - From any integer or floating-point type to a complex type. A complex value can only be
    ///   cast to another complex type or to `bool` - not to a real integer or floating-point type.
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
            #[cfg(all(feature = "half", feature = "num-complex"))]
            impl_cast_num!(@impl_to_complex, $src_type, Complex<f16>);
            #[cfg(feature = "num-complex")]
            impl_cast_num!(@impl_to_complex, $src_type, Complex<f32>);
            #[cfg(feature = "num-complex")]
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

    #[cfg(feature = "num-complex")]
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
    #[cfg(feature = "num-complex")]
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
    #[cfg(all(feature = "half", feature = "num-complex"))]
    impl_cast_complex!(f16);
    #[cfg(feature = "num-complex")]
    impl_cast_complex!(f32);
    #[cfg(feature = "num-complex")]
    impl_cast_complex!(f64);
}

/// Casts each element to a new element type, returned by [`Array::cast`].
///
/// Supported casts:
/// - Between any two integer types: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`.
/// - Between any two floating-point types: `f16` (requires the `half` feature), `f32`, `f64`.
/// - Between any two complex types: `Complex<f32>`, `Complex<f64>`.
/// - Between any integer to floating-point type, or vice versa.
/// - From any integer or floating-point type to a complex type. A complex value can only be
///   cast to another complex type or to `bool` - not to a real integer or floating-point type.
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

    fn info(&self) -> ArrayStorageInfo<'_> {
        ArrayStorageInfo::new_deps("Cast", [&self.0.array])
    }

    type DimensionChange<NewD: crate::Dimension> = Cast<S::DimensionChange<NewD>, T>;
    #[inline]
    fn dimension_change<NewD: crate::Dimension>(
        self,
    ) -> crate::error::Result<Self::DimensionChange<NewD>> {
        Ok(Cast(self.0.dimension_change()?))
    }

    crate::ops::impl_element_type_change_default!();
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

    #[cfg(feature = "half")]
    use crate::scalar::f16;
    #[cfg(feature = "num-complex")]
    use crate::scalar::Complex;
    #[cfg(feature = "num-complex")]
    #[allow(non_camel_case_types)]
    type complex_f32 = crate::scalar::Complex<f32>;
    #[cfg(feature = "num-complex")]
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
        impl CastTo<f16> for f16 {
            fn cast_to(self) -> f16 {
                self
            }
        }
    }

    #[cfg(feature = "num-complex")]
    impl CastTo<Complex<f64>> for Complex<f32> {
        fn cast_to(self) -> Complex<f64> {
            Complex {
                re: self.re as f64,
                im: self.im as f64,
            }
        }
    }
    #[cfg(feature = "num-complex")]
    impl CastTo<Complex<f32>> for Complex<f64> {
        fn cast_to(self) -> Complex<f32> {
            Complex {
                re: self.re as f32,
                im: self.im as f32,
            }
        }
    }
    #[cfg(feature = "num-complex")]
    impl CastTo<Complex<f32>> for Complex<f32> {
        fn cast_to(self) -> Complex<f32> {
            self
        }
    }
    #[cfg(feature = "num-complex")]
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

    // --- kept property tests: one representative pair per cast category (see
    // cast_int_float_concrete / cast_float_widen_narrow_concrete /
    // cast_int_sign_reinterpret_concrete / cast_identity_concrete / cast_bool_concrete /
    // cast_f16_concrete / cast_complex_concrete below for the remaining pairs, converted to
    // fixed-input tests) ---

    // int-widen: smaller signed int -> larger signed int (sign-extension path).
    test_cast_pair!(i8, i32);
    // int-narrow: larger signed int -> smaller signed int (truncation/wraparound path).
    test_cast_pair!(i64, i16);
    // int<->float: both directions of the int/float boundary.
    test_cast_pair!(i32, f64);
    test_cast_pair!(f64, i32);
    // to-bool: zero -> false, any nonzero -> true.
    test_cast_pair!(i32, bool);
    // identity cast (same src and dst dtype).
    test_cast_pair!(f32, f32);

    // int <-> float at a different byte width than the kept i32/f64 pair: smaller -> larger
    // itemsize (u8 -> f32, exact) and larger -> smaller itemsize (f32 -> u8, truncation +
    // saturation).
    #[test]
    fn cast_int_float_concrete() {
        use crate::Array;

        // u8 -> f32: 0, a mid value, and u8::MAX; exact int-to-float widening.
        let nd_u8 = ndarray::array![[0u8, 1, 128], [200, 254, u8::MAX]];
        let za_u8 = Array::compact_ndarray(&nd_u8).unwrap();
        let expected_u8_f32 = nd_u8.mapv(CastTo::<f32>::cast_to);
        crate::util::assert_array_matches(&za_u8.cast::<f32>(), &expected_u8_f32);

        // f32 -> u8: negative saturates to 0, a fraction truncates toward zero, the exact
        // dtype max, an out-of-range value that saturates to u8::MAX, and NaN saturates to 0.
        let nd_f32 = ndarray::array![[-5.5f32, 0.0, 128.9], [255.0, 300.5, f32::NAN]];
        let za_f32 = Array::compact_ndarray(&nd_f32).unwrap();
        let expected_f32_u8 = nd_f32.mapv(CastTo::<u8>::cast_to);
        crate::util::assert_array_matches(&za_f32.cast::<u8>(), &expected_f32_u8);
    }

    // float widen (f32 -> f64, exact) / narrow (f64 -> f32, precision loss and
    // out-of-range saturation to +/-infinity).
    #[test]
    fn cast_float_widen_narrow_concrete() {
        use crate::Array;

        let nd_f32 = ndarray::array![[f32::MIN, -1.5, 0.0], [1.5, 0.1, f32::MAX]];
        let za_f32 = Array::compact_ndarray(&nd_f32).unwrap();
        let expected_f32_f64 = nd_f32.mapv(CastTo::<f64>::cast_to);
        crate::util::assert_array_matches(&za_f32.cast::<f64>(), &expected_f32_f64);

        // f64::MIN / f64::MAX are far outside the f32 range, so they saturate to +/-infinity
        // (same-sign infinity compares equal, so this is safe under exact equality).
        let nd_f64 = ndarray::array![[f64::MIN, -1.5, 0.0], [1.5, 0.1, f64::MAX]];
        let za_f64 =
            Array::compact_ndarray_with(&nd_f64, crate::util::arr_params(&[1, 2])).unwrap();
        let expected_f64_f32 = nd_f64.mapv(CastTo::<f32>::cast_to);
        crate::util::assert_array_matches(&za_f64.cast::<f32>(), &expected_f64_f32);
    }

    // same-size signed/unsigned bit reinterpretation, both directions.
    #[test]
    fn cast_int_sign_reinterpret_concrete() {
        use crate::Array;

        let nd_i8 = ndarray::array![[i8::MIN, -1, 0], [1, 100, i8::MAX]];
        let za_i8 = Array::compact_ndarray(&nd_i8).unwrap();
        let expected_i8_u8 = nd_i8.mapv(CastTo::<u8>::cast_to);
        crate::util::assert_array_matches(&za_i8.cast::<u8>(), &expected_i8_u8);

        // 128 wraps to a negative i8; u8::MAX wraps to -1.
        let nd_u8 = ndarray::array![[0u8, 127, 128], [200, 254, u8::MAX]];
        let za_u8 = Array::compact_ndarray(&nd_u8).unwrap();
        let expected_u8_i8 = nd_u8.mapv(CastTo::<i8>::cast_to);
        crate::util::assert_array_matches(&za_u8.cast::<i8>(), &expected_u8_i8);
    }

    // identity casts at other dtypes than the kept f32 -> f32 representative.
    #[test]
    fn cast_identity_concrete() {
        use crate::Array;

        let nd_i32 = ndarray::array![[i32::MIN, -1, 0], [1, 100, i32::MAX]];
        let za_i32 = Array::compact_ndarray(&nd_i32).unwrap();
        let expected_i32 = nd_i32.mapv(CastTo::<i32>::cast_to);
        crate::util::assert_array_matches(&za_i32.cast::<i32>(), &expected_i32);

        let nd_f64 = ndarray::array![[f64::MIN, -1.5, 0.0], [1.5, 100.25, f64::MAX]];
        let za_f64 =
            Array::compact_ndarray_with(&nd_f64, crate::util::arr_params(&[1, 2])).unwrap();
        let expected_f64 = nd_f64.mapv(CastTo::<f64>::cast_to);
        crate::util::assert_array_matches(&za_f64.cast::<f64>(), &expected_f64);
    }

    // to-bool / from-bool at dtypes other than the kept i32 <-> bool representative.
    #[test]
    fn cast_bool_concrete() {
        use crate::Array;

        // i8 -> bool: zero is false; the negative and positive extremes are both true.
        let nd_i8 = ndarray::array![[i8::MIN, -1, 0], [1, 100, i8::MAX]];
        let za_i8 = Array::compact_ndarray(&nd_i8).unwrap();
        let expected_i8_bool = nd_i8.mapv(CastTo::<bool>::cast_to);
        crate::util::assert_array_matches(&za_i8.cast::<bool>(), &expected_i8_bool);

        // u8 -> bool: zero is false; any nonzero, including the dtype max, is true.
        let nd_u8 = ndarray::array![[0u8, 1, 128], [200, 254, u8::MAX]];
        let za_u8 = Array::compact_ndarray(&nd_u8).unwrap();
        let expected_u8_bool = nd_u8.mapv(CastTo::<bool>::cast_to);
        crate::util::assert_array_matches(&za_u8.cast::<bool>(), &expected_u8_bool);

        // f32 -> bool: 0.0 and -0.0 are both false (IEEE -0.0 == 0.0); any nonzero,
        // including the extremes, is true.
        let nd_f32 = ndarray::array![[0.0f32, -0.0, 1.5], [-1.5, f32::MIN, f32::MAX]];
        let za_f32 = Array::compact_ndarray(&nd_f32).unwrap();
        let expected_f32_bool = nd_f32.mapv(CastTo::<bool>::cast_to);
        crate::util::assert_array_matches(&za_f32.cast::<bool>(), &expected_f32_bool);

        // bool -> i8 / u8 / i32 / f32: false -> 0, true -> 1 (shared `bool => i8 => dst` path).
        let nd_bool = ndarray::array![[false, true], [true, false]];

        let expected_bool_i8 = nd_bool.mapv(CastTo::<i8>::cast_to);
        crate::util::assert_array_matches(
            &Array::compact_ndarray(&nd_bool).unwrap().cast::<i8>(),
            &expected_bool_i8,
        );

        let expected_bool_u8 = nd_bool.mapv(CastTo::<u8>::cast_to);
        crate::util::assert_array_matches(
            &Array::compact_ndarray(&nd_bool).unwrap().cast::<u8>(),
            &expected_bool_u8,
        );

        let expected_bool_i32 = nd_bool.mapv(CastTo::<i32>::cast_to);
        crate::util::assert_array_matches(
            &Array::compact_ndarray(&nd_bool).unwrap().cast::<i32>(),
            &expected_bool_i32,
        );

        let expected_bool_f32 = nd_bool.mapv(CastTo::<f32>::cast_to);
        crate::util::assert_array_matches(
            &Array::compact_ndarray(&nd_bool).unwrap().cast::<f32>(),
            &expected_bool_f32,
        );
    }

    // f16 (feature-gated): both directions with i32/f32/f64, plus the f16 -> f16 identity.
    #[cfg(feature = "half")]
    #[test]
    fn cast_f16_concrete() {
        use crate::Array;

        // i32 -> f16: 0, +/-1, a mid value, +/-2048 (an exactly-representable f16 integer),
        // through the int -> f32 -> f16 widen-then-round path, and i32::MIN/MAX, which far
        // exceed f16's ~65504 range and so saturate to +/-infinity (same-sign infinity
        // compares equal, so this is safe under exact equality).
        let nd_i32 = ndarray::array![[i32::MIN, -2048, -1, 0], [1, 100, 2048, i32::MAX]];
        let za_i32 = Array::compact_ndarray(&nd_i32).unwrap();
        let expected_i32_f16 = nd_i32.mapv(CastTo::<f16>::cast_to);
        crate::util::assert_array_matches(&za_i32.cast::<f16>(), &expected_i32_f16);

        // f16 -> i32: 0, +/-1, a fraction that truncates toward zero, and f16::MIN/MAX.
        let nd_f16 = ndarray::array![
            [f16::from_f32(-1.5), f16::from_f32(0.0), f16::from_f32(1.5)],
            [f16::from_f32(100.0), f16::MIN, f16::MAX],
        ];
        let expected_f16_i32 = nd_f16.mapv(CastTo::<i32>::cast_to);
        crate::util::assert_array_matches(
            &Array::compact_ndarray(&nd_f16).unwrap().cast::<i32>(),
            &expected_f16_i32,
        );

        // f32 -> f16 (narrowing, with rounding) and f16 -> f32 (exact widening). f32::MIN/MAX
        // are far outside f16's ~65504 range, so they saturate to +/-infinity (same-sign
        // infinity compares equal, so this is safe under exact equality).
        let nd_f32 = ndarray::array![
            [f32::MIN, -100.0, -1.5, 0.0],
            [1.5, 100.0, f16::MAX.to_f32(), f32::MAX]
        ];
        let expected_f32_f16 = nd_f32.mapv(CastTo::<f16>::cast_to);
        crate::util::assert_array_matches(
            &Array::compact_ndarray(&nd_f32).unwrap().cast::<f16>(),
            &expected_f32_f16,
        );
        let expected_f16_f32 = nd_f16.mapv(CastTo::<f32>::cast_to);
        crate::util::assert_array_matches(
            &Array::compact_ndarray(&nd_f16).unwrap().cast::<f32>(),
            &expected_f16_f32,
        );

        // f64 -> f16 (narrowing, with rounding) and f16 -> f64 (exact widening). f64::MIN/MAX
        // are far outside f16's ~65504 range, so they saturate to +/-infinity (same-sign
        // infinity compares equal, so this is safe under exact equality).
        let nd_f64 = ndarray::array![
            [f64::MIN, -100.0, -1.5, 0.0],
            [1.5, 100.0, f16::MAX.to_f32() as f64, f64::MAX]
        ];
        let expected_f64_f16 = nd_f64.mapv(CastTo::<f16>::cast_to);
        crate::util::assert_array_matches(
            &Array::compact_ndarray(&nd_f64).unwrap().cast::<f16>(),
            &expected_f64_f16,
        );
        let expected_f16_f64 = nd_f16.mapv(CastTo::<f64>::cast_to);
        crate::util::assert_array_matches(
            &Array::compact_ndarray(&nd_f16).unwrap().cast::<f64>(),
            &expected_f16_f64,
        );

        // f16 -> f16: identity.
        let expected_f16_f16 = nd_f16.mapv(CastTo::<f16>::cast_to);
        crate::util::assert_array_matches(
            &Array::compact_ndarray(&nd_f16).unwrap().cast::<f16>(),
            &expected_f16_f16,
        );
    }

    // complex (feature-gated): widen/narrow between complex_f32/complex_f64 plus both
    // identity casts.
    #[cfg(feature = "num-complex")]
    #[test]
    fn cast_complex_concrete() {
        use crate::Array;

        // complex_f32 -> complex_f64 (exact widening of both parts) and
        // complex_f32 -> complex_f32 (identity): zero, negative, fraction, and the f32
        // extremes on the real/imaginary parts.
        let nd_c32 = ndarray::array![
            [
                Complex {
                    re: 0.0f32,
                    im: 0.0
                },
                Complex { re: -1.5, im: 2.5 }
            ],
            [
                Complex {
                    re: f32::MIN,
                    im: f32::MAX
                },
                Complex {
                    re: 100.0,
                    im: -100.0
                }
            ],
        ];
        let expected_c32_c64 = nd_c32.mapv(CastTo::<complex_f64>::cast_to);
        crate::util::assert_array_matches(
            &Array::compact_ndarray(&nd_c32)
                .unwrap()
                .cast::<complex_f64>(),
            &expected_c32_c64,
        );
        let expected_c32_c32 = nd_c32.mapv(CastTo::<complex_f32>::cast_to);
        crate::util::assert_array_matches(
            &Array::compact_ndarray(&nd_c32)
                .unwrap()
                .cast::<complex_f32>(),
            &expected_c32_c32,
        );

        // complex_f64 -> complex_f32 (narrowing, including an out-of-range magnitude that
        // saturates to infinity) and complex_f64 -> complex_f64 (identity).
        let nd_c64 = ndarray::array![
            [
                Complex {
                    re: 0.0f64,
                    im: 0.0
                },
                Complex { re: -1.5, im: 2.5 }
            ],
            [
                Complex {
                    re: f64::MIN,
                    im: f64::MAX
                },
                Complex {
                    re: 100.0,
                    im: -100.0
                }
            ],
        ];
        let expected_c64_c32 = nd_c64.mapv(CastTo::<complex_f32>::cast_to);
        crate::util::assert_array_matches(
            &Array::compact_ndarray(&nd_c64)
                .unwrap()
                .cast::<complex_f32>(),
            &expected_c64_c32,
        );
        let expected_c64_c64 = nd_c64.mapv(CastTo::<complex_f64>::cast_to);
        crate::util::assert_array_matches(
            &Array::compact_ndarray(&nd_c64)
                .unwrap()
                .cast::<complex_f64>(),
            &expected_c64_c64,
        );
    }
}
