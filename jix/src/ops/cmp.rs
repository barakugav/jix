use crate::ops::common::define_array_op2_method;
use crate::ops::op2::define_op2;
use crate::ops::prelude::*;
use crate::ops::{Op2, Op2Kernel};

pub(crate) mod _traits {
    #[cfg(feature = "half")]
    use crate::scalar::f16;
    use crate::scalar::traits_util::{define_scalar_op2_trait, impl_scalar_op2};
    #[cfg(feature = "num-complex")]
    use crate::scalar::Complex;

    define_scalar_op2_trait!(
        /// Maximum operation with NaN-propagating semantics for floating-point types.
        ///
        /// This trait exists because neither of the standard alternatives covers all supported dtypes:
        ///
        /// - [`std::cmp::max`] requires [`Ord`], which floating-point types do not implement due to
        ///   the unordered nature of `NaN`. It cannot be used for `f32`, `f64`, or `f16`.
        /// - [`f32::max`] / [`f64::max`] use **NaN-ignoring** semantics: when exactly one operand is
        ///   `NaN`, they return the non-`NaN` value. This matches `numpy.fmax` / `numpy.nanmax`, not
        ///   `numpy.maximum`.
        ///
        /// `Maximum` instead uses **NaN-propagating** semantics: if *either* operand is `NaN`,
        /// the result is `NaN`. This matches `numpy.maximum` and makes NaN visible rather than
        /// silently discarding it. The result is always the `NaN` of [`f32::NAN`] / [`f64::NAN`],
        /// not the operand's (`numpy.maximum` keeps the operand's sign and payload).
        ///
        /// Of `+0.0` and `-0.0` either may be returned, as with [`f32::max`].
        ///
        /// For integer and `bool` types the implementation delegates to [`std::cmp::max`], which is
        /// equivalent. The trait therefore provides a single uniform interface usable across all
        /// supported numeric dtypes.
        Maximum,
        /// Compute the maximum with another value.
        maximum,
        bulk_fn: maximum_bulk,
    );
    impl_scalar_op2!(
        impl Maximum for [i8, i16, i32, i64, u8, u16, u32, u64] {
            fn maximum = |a, b| std::cmp::max(a, b),
            fn maximum_bulk = |a, b| a.max(b),
        }
    );
    impl_scalar_op2!(
        impl Maximum for [bool] {
            fn maximum = |a, b| std::cmp::max(a, b),
        }
    );
    // NaN-propagating, to the one NaN `Self::NAN` (not an operand's, which SIMD and scalar code
    // would keep differently).
    impl_scalar_op2!(
        impl Maximum for [f32, f64] {
            // Not `f32::max`: past the NaN check, its NaN handling is dead code LLVM still emits.
            fn maximum = |a, b| if a.is_nan() | b.is_nan() { Self::NAN } else if a > b { a } else { b },
            fn maximum_bulk = |a, b| (/* NaN check */ a.simd_ne(a) | b.simd_ne(b))
                .select(SimdBase::splat(fearless_simd::ExtractToken::token(&a), Self::NAN), a.max(b)),
        }
    );
    #[cfg(feature = "half")]
    impl_scalar_op2!(
        impl Maximum for [f16] {
            fn maximum = |a, b| if a.is_nan() | b.is_nan() { Self::NAN } else { a.max(b) },
        }
    );

    define_scalar_op2_trait!(
        /// Minimum operation with NaN-propagating semantics for floating-point types.
        ///
        /// This trait exists because neither of the standard alternatives covers all supported dtypes:
        ///
        /// - [`std::cmp::min`] requires [`Ord`], which floating-point types do not implement due to
        ///   the unordered nature of `NaN`. It cannot be used for `f32`, `f64`, or `f16`.
        /// - [`f32::min`] / [`f64::min`] use **NaN-ignoring** semantics: when exactly one operand is
        ///   `NaN`, they return the non-`NaN` value. This matches `numpy.fmin` / `numpy.nanmin`, not
        ///   `numpy.minimum`.
        ///
        /// `Minimum` instead uses **NaN-propagating** semantics: if *either* operand is `NaN`,
        /// the result is `NaN`. This matches `numpy.minimum` and makes NaN visible rather than
        /// silently discarding it. The result is always the `NaN` of [`f32::NAN`] / [`f64::NAN`],
        /// not the operand's (`numpy.minimum` keeps the operand's sign and payload).
        ///
        /// Of `+0.0` and `-0.0` either may be returned, as with [`f32::min`].
        ///
        /// For integer and `bool` types the implementation delegates to [`std::cmp::min`], which is
        /// equivalent. The trait therefore provides a single uniform interface usable across all
        /// supported numeric dtypes.
        Minimum,
        /// Compute the minimum with another value.
        minimum,
        bulk_fn: minimum_bulk,
    );
    impl_scalar_op2!(
        impl Minimum for [i8, i16, i32, i64, u8, u16, u32, u64] {
            fn minimum = |a, b| std::cmp::min(a, b),
            fn minimum_bulk = |a, b| a.min(b),
        }
    );
    impl_scalar_op2!(
        impl Minimum for [bool] {
            fn minimum = |a, b| std::cmp::min(a, b),
        }
    );
    // NaN-propagating, to the one NaN `Self::NAN` (not an operand's, which SIMD and scalar code
    // would keep differently).
    impl_scalar_op2!(
        impl Minimum for [f32, f64] {
            // Not `f32::min`: past the NaN check, its NaN handling is dead code LLVM still emits.
            fn minimum = |a, b| if a.is_nan() | b.is_nan() { Self::NAN } else if a < b { a } else { b },
            fn minimum_bulk = |a, b| (/* NaN check */ a.simd_ne(a) | b.simd_ne(b))
                .select(SimdBase::splat(fearless_simd::ExtractToken::token(&a), Self::NAN), a.min(b)),
        }
    );
    #[cfg(feature = "half")]
    impl_scalar_op2!(
        impl Minimum for [f16] {
            fn minimum = |a, b| if a.is_nan() | b.is_nan() { Self::NAN } else { a.min(b) },
        }
    );

    macro_rules! cmp_op2 {
        (
            $(#[$trait_meta:meta])*
            impl $Trait:ident {
                $(#[$fn_meta:meta])*
                fn $f:ident = $op:tt,
                fn $f_bulk:ident = $simd_f:ident,
                bool: |$a:ident, $b:ident| $bool:expr,
                u64_i64: |$ua:ident, $ib:ident| $u64_i64:expr,
                i64_u64: |$ia:ident, $ub:ident| $i64_u64:expr,
            }
        ) => {
            define_scalar_op2_trait!(
                $(#[$trait_meta])* $Trait,
                $(#[$fn_meta])* $f,
                bulk_fn: $f_bulk,
            );
            impl_scalar_op2!(
                impl $Trait for [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64] {
                    type Output = bool;
                    fn $f = |a, b| a $op b,
                    fn $f_bulk = |a, b| a.$simd_f(b),
                }
            );
            impl_scalar_op2!(
                impl $Trait for [bool] {
                    type Output = bool;
                    fn $f = |$a, $b| $bool,
                }
            );
            impl_scalar_op2!(
                impl $Trait for [u64] x [i64] {
                    type Output = bool;
                    fn $f = |$ua, $ib| $u64_i64,
                }
            );
            impl_scalar_op2!(
                impl $Trait for [i64] x [u64] {
                    type Output = bool;
                    fn $f = |$ia, $ub| $i64_u64,
                }
            );
            #[cfg(feature = "half")]
            impl_scalar_op2!(
                impl $Trait for [f16] {
                    type Output = bool;
                    fn $f = |a, b| a $op b,
                }
            );
        };
        (
            @complex
            impl $Trait:ident {
                fn $f:ident = $op:tt,
            }
        ) => {
            #[cfg(feature = "num-complex")]
            impl_scalar_op2!(
                impl $Trait for [Complex<f32>, Complex<f64>] {
                    type Output = bool;
                    fn $f = |a, b| a $op b,
                }
            );
            #[cfg(all(feature = "half", feature = "num-complex"))]
            impl_scalar_op2!(
                impl $Trait for [Complex<f16>] {
                    type Output = bool;
                    fn $f = |a, b| a $op b,
                }
            );
        };
    }
    cmp_op2!(
        /// Equality operation, as [`PartialEq::eq`].
        ///
        /// Output dtype is `bool`.
        ///
        /// For **float** types, `NaN != NaN` per IEEE 754: comparing two `NaN` values
        /// returns `false`.
        /// For **complex** types, both the real and imaginary components must be equal.
        impl Equal {
            /// Test whether the value equals another value.
            fn equal = ==,
            fn equal_bulk = simd_eq,
            bool: |a, b| a == b,
            u64_i64: |a, b| b >= 0 && a == b as u64,
            i64_u64: |a, b| a >= 0 && a as u64 == b,
        }
    );
    cmp_op2!(@complex
        impl Equal {
            fn equal = ==,
        }
    );
    cmp_op2!(
        /// Inequality operation, as [`PartialEq::ne`].
        ///
        /// Output dtype is `bool`.
        ///
        /// For **float** types, `NaN != NaN` returns `true` per IEEE 754.
        /// For **complex** types, returns `true` if either the real or imaginary component differs.
        impl NotEqual {
            /// Test whether the value differs from another value.
            fn not_equal = !=,
            fn not_equal_bulk = simd_ne,
            bool: |a, b| a != b,
            u64_i64: |a, b| b < 0 || a != b as u64,
            i64_u64: |a, b| a < 0 || a as u64 != b,
        }
    );
    cmp_op2!(@complex
        impl NotEqual {
            fn not_equal = !=,
        }
    );
    cmp_op2!(
        /// Less-than operation, as [`PartialOrd::lt`].
        ///
        /// Complex types are not supported as they have no total ordering. Output dtype is `bool`.
        ///
        /// For **float** types, any comparison involving `NaN` returns `false` (IEEE 754).
        /// For **bool**: `false < true`.
        impl Less {
            /// Test whether the value is less than another value.
            fn less = <,
            fn less_bulk = simd_lt,
            bool: |a, b| !a & b,
            u64_i64: |a, b| b >= 0 && a < b as u64,
            i64_u64: |a, b| a < 0 || (a as u64) < b,
        }
    );
    cmp_op2!(
        /// Less-than-or-equal operation, as [`PartialOrd::le`].
        ///
        /// Complex types are not supported as they have no total ordering. Output dtype is `bool`.
        ///
        /// For **float** types, any comparison involving `NaN` returns `false` (IEEE 754).
        /// For **bool**: `false <= true`, and both `false <= false` and `true <= true` hold.
        impl LessEqual {
            /// Test whether the value is less than or equal to another value.
            fn less_equal = <=,
            fn less_equal_bulk = simd_le,
            bool: |a, b| !a | b,
            u64_i64: |a, b| b >= 0 && a <= b as u64,
            i64_u64: |a, b| a < 0 || a as u64 <= b,
        }
    );

    /// Approximate equality check of two scalar values.
    ///
    /// Returns `true` when two values are *close enough* under absolute and/or relative
    /// tolerance. Namely if `|self - other| <= max(atol, rtol * max(|self|, |other|))`
    /// returns `true`, otherwise `false`.
    ///
    /// `NaN` inputs always return `false` because `NaN != NaN`.
    ///
    /// Implemented for `f32`, `f64`, `f16` (feature `half`), and
    /// [`Complex<F>`](crate::scalar::Complex) (feature `num-complex`).
    /// For [`Complex<F>`](crate::scalar::Complex) the check is applied independently to the
    /// real and imaginary parts: `atol` is `Complex<F>` (separate per-component absolute
    /// tolerances) and `rtol` is `F` (a shared relative tolerance applied per component).
    pub trait ApproxEq {
        /// The type used for the absolute tolerance parameter (`atol`).
        type AbsoluteTolerance;
        /// The type used for the relative tolerance parameter (`rtol`).
        type RelativeTolerance;

        /// Returns `true` if `self` and `other` are approximately equal under the given
        /// tolerances, `|self - other| <= max(atol, rtol * max(|self|, |other|))`.
        fn approx_eq(
            &self,
            other: &Self,
            rtol: &Self::RelativeTolerance,
            atol: &Self::AbsoluteTolerance,
        ) -> bool;

        /// Apply [`approx_eq`](Self::approx_eq) on each pair of `xs` and `ys`, vectorized with
        /// `simd` where the types have SIMD support.
        #[inline(always)]
        fn approx_eq_bulk<const N: usize, S: fearless_simd::Simd>(
            xs: [Self; N],
            ys: [Self; N],
            rtol: &Self::RelativeTolerance,
            atol: &Self::AbsoluteTolerance,
            simd: S,
        ) -> [bool; N]
        where
            Self: Sized,
        {
            let _ = simd;
            crate::util::array2_map_inline(
                xs,
                ys,
                #[inline(always)]
                |_, x, y| x.approx_eq(&y, rtol, atol),
            )
        }
    }
    macro_rules! impl_approx_eq {
        ($T:ty) => {
            impl ApproxEq for $T {
                type AbsoluteTolerance = $T;
                type RelativeTolerance = $T;

                #[inline(always)]
                fn approx_eq(
                    &self,
                    other: &Self,
                    rtol: &Self::RelativeTolerance,
                    atol: &Self::AbsoluteTolerance,
                ) -> bool {
                    // credit to https://github.com/brendanzab/approx
                    //
                    // The code was re-written as branchless. enabled auto-vectorization.

                    let (a, b) = (*self, *other);
                    let diff = crate::scalar::Abs::abs(a - b);
                    let (abs_a, abs_b) = (crate::scalar::Abs::abs(a), crate::scalar::Abs::abs(b));
                    let largest = if abs_b > abs_a { abs_b } else { abs_a };
                    let close = (diff <= *atol) | (diff <= largest * *rtol);
                    let no_inf = !<$T>::is_infinite(a) & !<$T>::is_infinite(b);
                    (a == b) | (close & no_inf)
                }
            }
        };
    }
    impl_approx_eq!(f32);
    impl_approx_eq!(f64);
    #[cfg(feature = "half")]
    impl_approx_eq!(f16);

    #[cfg(feature = "num-complex")]
    impl<F> ApproxEq for Complex<F>
    where
        F: ApproxEq<AbsoluteTolerance = F, RelativeTolerance = F>,
    {
        type AbsoluteTolerance = Complex<F>;
        type RelativeTolerance = F;

        #[inline(always)]
        fn approx_eq(
            &self,
            other: &Self,
            rtol: &Self::RelativeTolerance,
            atol: &Self::AbsoluteTolerance,
        ) -> bool {
            self.re.approx_eq(&other.re, rtol, &atol.re)
                & self.im.approx_eq(&other.im, rtol, &atol.im)
        }
    }
}

define_op2!(
    /// Element-wise equality test (`a == b`).
    ///
    /// See [`jix::scalar::Equal`](crate::scalar::Equal) scalar trait for the per-element semantics.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::equal()`](crate::Array::equal).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![1i32, 2, 3])?;
    /// let b = Array::compact_ndarray(&array![1i32, 0, 3])?;
    /// let result = a.equal(b).to_ndarray()?;
    /// assert_eq!(result, array![true, false, true]);
    ///
    /// // NaN != NaN per IEEE 754.
    /// let c = Array::compact_ndarray(&array![f32::NAN, 1.0f32])?;
    /// let d = Array::compact_ndarray(&array![f32::NAN, 1.0f32])?;
    /// let result = c.equal(d).to_ndarray()?;
    /// assert_eq!(result, array![false, true]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Equal,
    EqualKernel,
    crate::scalar::Equal::equal,
    simd: equal_bulk,
    inner_loops: InnerLoops::CONTIGUOUS,
);
define_op2!(
    /// Element-wise inequality test (`a != b`).
    ///
    /// See [`jix::scalar::NotEqual`](crate::scalar::NotEqual) scalar trait for the
    /// per-element semantics.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::not_equal()`](crate::Array::not_equal).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![1i32, 2, 3])?;
    /// let b = Array::compact_ndarray(&array![1i32, 0, 3])?;
    /// let result = a.not_equal(b).to_ndarray()?;
    /// assert_eq!(result, array![false, true, false]);
    ///
    /// // NaN != NaN is true per IEEE 754.
    /// let c = Array::compact_ndarray(&array![f32::NAN, 1.0f32])?;
    /// let d = Array::compact_ndarray(&array![f32::NAN, 2.0f32])?;
    /// let result = c.not_equal(d).to_ndarray()?;
    /// assert_eq!(result, array![true, true]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    NotEqual,
    NotEqualKernel,
    crate::scalar::NotEqual::not_equal,
    simd: not_equal_bulk,
    inner_loops: InnerLoops::CONTIGUOUS,
);

// Defines `$Op<S1, S2>`, a comparison evaluated as the mirrored comparison `$Inner` with the
// operands swapped (`a > b` as `b < a`).
macro_rules! define_swapped_cmp {
    (
        $(#[$meta:meta])*
        $Op:ident = $Inner:ident($InnerKernel:ident, crate::scalar::$InnerTrait:ident),
    ) => {
        $(#[$meta])*
        pub struct $Op<S1, S2>($Inner<S2, S1>);

        impl<S1, S2> $Op<S1, S2>
        where
            S1: ArrayStorageTyped,
            S2: ArrayStorageTyped,
            S2::Item: crate::scalar::$InnerTrait<S1::Item, Output: Dtyped>,
        {
            #[doc = concat!("Constructs a [`", stringify!($Op), "`] storage. See the struct docs for semantics and examples.")]
            pub fn new(a: S1, b: S2) -> Result<Self> {
                Ok(Self($Inner(Op2::new(b, a, $InnerKernel, true)?)))
            }

            #[doc = concat!("Constructs an array with [`", stringify!($Op), "`] storage. See the storage struct docs for semantics and examples.")]
            pub fn new_array(a: Array<S1>, b: Array<S2>) -> Result<Array<Self>> {
                Self::new(a.into_storage(), b.into_storage()).map(Array::from_storage)
            }
        }

        impl<S1, S2> ArrayStorage for $Op<S1, S2>
        where
            S1: ArrayStorageTyped,
            S2: ArrayStorageTyped,
            S2::Item: crate::scalar::$InnerTrait<S1::Item, Output: Dtyped>,
        {
            type ElementType = Ty<<S2::Item as crate::scalar::$InnerTrait<S1::Item>>::Output>;

            #[inline(always)]
            fn read_data<'a>(
                &'a self,
                index: &[Range<u64>],
                context: &'a ReadContext,
                out: Option<&'a mut StridedBuf<'_>>,
            ) -> Result<StridedBuf<'a>> {
                self.0.read_data(index, context, out)
            }

            #[allow(refining_impl_trait)]
            #[inline(always)]
            fn read_as_elementwise_pipeline<'a, T>(
                &'a self,
                index: &[Range<u64>],
                context: &'a ReadContext,
            ) -> Result<impl ElementwisePipeline<T> + use<'a, T, S1, S2>>
            where
                T: Dtyped,
            {
                self.0.read_as_elementwise_pipeline(index, context)
            }

            #[inline(always)]
            fn shape(&self) -> &[u64] {
                self.0.shape()
            }

            #[inline(always)]
            fn dtype(&self) -> &Dtype {
                self.0.dtype()
            }

            #[inline]
            fn spec(&self) -> ArraySpec<'_> {
                // The inner op bases its spec on its first operand, which is our `b`.
                let op = &self.0 .0;
                op.b.spec()
                    .with_dynamic_spec(op.dynamic_spec())
                    .with_cleared_flags()
            }

            fn info(&self) -> ArrayStorageInfo<'_> {
                let op = &self.0 .0;
                ArrayStorageInfo::new_deps(stringify!($Op), [&op.b, &op.a])
            }

            crate::ops::impl_element_type_change_default!();
        }
    };
}

define_swapped_cmp!(
    /// Element-wise greater-than test (`a > b`).
    ///
    /// Evaluated as `b < a`: this storage wraps a [`Less`] with the operands swapped,
    /// see it for more detailed semantics.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::greater()`](crate::Array::greater).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![3i32, 1, 2])?;
    /// let b = Array::compact_ndarray(&array![1i32, 1, 3])?;
    /// let result = a.greater(b).to_ndarray()?;
    /// assert_eq!(result, array![true, false, false]);
    ///
    /// // true > false for bool dtype.
    /// let c = Array::compact_ndarray(&array![true, false, true])?;
    /// let d = Array::compact_ndarray(&array![false, false, true])?;
    /// let result = c.greater(d).to_ndarray()?;
    /// assert_eq!(result, array![true, false, false]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Greater = Less(LessKernel, crate::scalar::Less),
);
define_swapped_cmp!(
    /// Element-wise greater-than-or-equal test (`a >= b`).
    ///
    /// Evaluated as `b <= a`: this storage wraps a [`LessEqual`] with the operands swapped,
    /// see it for more detailed semantics.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::greater_equal()`](crate::Array::greater_equal).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![3i32, 1, 2])?;
    /// let b = Array::compact_ndarray(&array![1i32, 1, 3])?;
    /// let result = a.greater_equal(b).to_ndarray()?;
    /// assert_eq!(result, array![true, true, false]);
    ///
    /// // true >= false and false >= false hold.
    /// let c = Array::compact_ndarray(&array![true, false, true])?;
    /// let d = Array::compact_ndarray(&array![false, false, true])?;
    /// let result = c.greater_equal(d).to_ndarray()?;
    /// assert_eq!(result, array![true, true, true]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    GreaterEqual = LessEqual(LessEqualKernel, crate::scalar::LessEqual),
);
define_op2!(
    /// Element-wise less-than test (`a < b`).
    ///
    /// See [`jix::scalar::Less`](crate::scalar::Less) scalar trait for the per-element semantics.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::less()`](crate::Array::less).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![1i32, 1, 3])?;
    /// let b = Array::compact_ndarray(&array![3i32, 1, 2])?;
    /// let result = a.less(b).to_ndarray()?;
    /// assert_eq!(result, array![true, false, false]);
    ///
    /// // false < true for bool dtype.
    /// let c = Array::compact_ndarray(&array![false, false, true])?;
    /// let d = Array::compact_ndarray(&array![true, false, true])?;
    /// let result = c.less(d).to_ndarray()?;
    /// assert_eq!(result, array![true, false, false]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Less,
    LessKernel,
    crate::scalar::Less::less,
    simd: less_bulk,
    inner_loops: InnerLoops::CONTIGUOUS,
);
define_op2!(
    /// Element-wise less-than-or-equal test (`a <= b`).
    ///
    /// See [`jix::scalar::LessEqual`](crate::scalar::LessEqual) scalar trait for the
    /// per-element semantics.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::less_equal()`](crate::Array::less_equal).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![1i32, 1, 3])?;
    /// let b = Array::compact_ndarray(&array![3i32, 1, 2])?;
    /// let result = a.less_equal(b).to_ndarray()?;
    /// assert_eq!(result, array![true, true, false]);
    ///
    /// // false <= true and true <= true hold.
    /// let c = Array::compact_ndarray(&array![false, false, true])?;
    /// let d = Array::compact_ndarray(&array![true, false, true])?;
    /// let result = c.less_equal(d).to_ndarray()?;
    /// assert_eq!(result, array![true, true, true]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    LessEqual,
    LessEqualKernel,
    crate::scalar::LessEqual::less_equal,
    simd: less_equal_bulk,
    inner_loops: InnerLoops::CONTIGUOUS,
);

define_op2!(
    /// Element-wise maximum of two arrays.
    ///
    /// See [`jix::scalar::Maximum`](crate::scalar::Maximum) scalar trait for the
    /// per-element semantics.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::maximum()`](crate::Array::maximum).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::{array, Ix1};
    ///
    /// let a = Array::compact_ndarray(&array![1i32, 5, 3])?;
    /// let b = Array::compact_ndarray(&array![4i32, 2, 3])?;
    /// let result = a.maximum(b).to_ndarray()?;
    /// assert_eq!(result, array![4, 5, 3]);
    ///
    /// // NaN is propagated: if either operand is NaN the result is NaN.
    /// let c = Array::compact_ndarray(&array![f32::NAN, 1.0f32])?;
    /// let d = Array::compact_ndarray(&array![2.0f32, 3.0f32])?;
    /// let result = c.maximum(d).to_ndarray::<Ix1>()?;
    /// assert!(result[[0]].is_nan());
    /// assert_eq!(result[[1]], 3.0);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Maximum,
    MaximumKernel,
    crate::scalar::Maximum::maximum,
    simd: maximum_bulk,
    inner_loops: InnerLoops::CONTIGUOUS,
);
define_op2!(
    /// Element-wise minimum of two arrays.
    ///
    /// See [`jix::scalar::Minimum`](crate::scalar::Minimum) scalar trait for the
    /// per-element semantics.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::minimum()`](crate::Array::minimum).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::{array, Ix1};
    ///
    /// let a = Array::compact_ndarray(&array![1i32, 5, 3])?;
    /// let b = Array::compact_ndarray(&array![4i32, 2, 3])?;
    /// let result = a.minimum(b).to_ndarray()?;
    /// assert_eq!(result, array![1, 2, 3]);
    ///
    /// // NaN is propagated: if either operand is NaN the result is NaN.
    /// let c = Array::compact_ndarray(&array![f32::NAN, 1.0f32])?;
    /// let d = Array::compact_ndarray(&array![2.0f32, 3.0f32])?;
    /// let result = c.minimum(d).to_ndarray::<Ix1>()?;
    /// assert!(result[[0]].is_nan());
    /// assert_eq!(result[[1]], 1.0);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Minimum,
    MinimumKernel,
    crate::scalar::Minimum::minimum,
    simd: minimum_bulk,
    inner_loops: InnerLoops::CONTIGUOUS,
);

/// Element-wise clamp of `a` to the range `[min, max]`, where `min` and `max` are arrays.
///
/// Computed as `minimum(maximum(a, min), max)`, the order used by `numpy.clip` and
/// `torch.clamp`: where `min > max` the result is `max`. For **float** types a `NaN` in any
/// operand produces `NaN` (see the [`jix::scalar::Maximum`](crate::scalar::Maximum) and
/// [`jix::scalar::Minimum`](crate::scalar::Minimum) scalar traits). All three arrays must have
/// the same dtype and shape.
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// This struct is the bare storage implementation, the operation is also available as
/// [`Array::clamp()`](crate::Array::clamp).
///
/// # Examples
/// ```
/// use jix::Array;
/// use ndarray::array;
///
/// let a = Array::compact_ndarray(&array![-3i32, 0, 5, 10])?;
/// let min = Array::compact_ndarray(&array![0i32, 1, 0, 0])?;
/// let max = Array::compact_ndarray(&array![7i32, 7, 4, 7])?;
/// let result = a.clamp(min, max).to_ndarray()?;
/// assert_eq!(result, array![0, 1, 4, 7]);
/// # Ok::<(), jix::Error>(())
/// ```
pub struct Clamp<S, SMin, SMax> {
    a: S,
    min: SMin,
    max: SMax,
    spec: ArraySpecDynamic,
}
impl<S, SMin, SMax> Clamp<S, SMin, SMax>
where
    S: ArrayStorageTyped,
    SMin: ArrayStorageTyped<Item = S::Item>,
    SMax: ArrayStorageTyped<Item = S::Item>,
    S::Item: crate::scalar::Maximum<Output = S::Item> + crate::scalar::Minimum<Output = S::Item>,
{
    /// Constructs a [`Clamp`] storage. See the struct docs for semantics and examples.
    pub fn new(a: S, min: SMin, max: SMax) -> Result<Self> {
        ensure!(
            a.shape() == min.shape() && a.shape() == max.shape(),
            InvalidArgument,
            "Clamp shape mismatch between `a` {:?}, `min` {:?} and `max` {:?}",
            a.shape(),
            min.shape(),
            max.shape()
        );
        let specs = [a.spec(), min.spec(), max.spec()];
        let (element_cost, read_shape_scale_weight, read_layout_order) =
            combine_elementwise_hints(&specs.each_ref().map(|spec| {
                (
                    spec.element_cost(),
                    spec.read_shape_scale_weight(),
                    spec.read_layout_order(),
                )
            }));
        let (block_shape, block_shape_fixed_dims) = combine_block_layout(
            &specs
                .each_ref()
                .map(|spec| (spec.block_shape().as_slice(), spec.block_shape_fixed_dims())),
        );
        let spec = ArraySpecDynamic::new(
            block_shape,
            block_shape_fixed_dims,
            element_cost,
            read_shape_scale_weight,
            read_layout_order,
        );
        Ok(Self { a, min, max, spec })
    }

    /// Constructs an array with [`Clamp`] storage. See the storage struct docs for semantics and examples.
    pub fn new_array(a: Array<S>, min: Array<SMin>, max: Array<SMax>) -> Result<Array<Self>> {
        Self::new(a.into_storage(), min.into_storage(), max.into_storage()).map(Array::from_storage)
    }
}
impl<S, SMin, SMax> ArrayStorage for Clamp<S, SMin, SMax>
where
    S: ArrayStorageTyped,
    SMin: ArrayStorageTyped<Item = S::Item>,
    SMax: ArrayStorageTyped<Item = S::Item>,
    S::Item: crate::scalar::Maximum<Output = S::Item> + crate::scalar::Minimum<Output = S::Item>,
{
    type ElementType = Ty<S::Item>;

    #[inline]
    fn read_data<'a>(
        &'a self,
        index: &[Range<u64>],
        context: &'a ReadContext,
        out: Option<&'a mut StridedBuf<'_>>,
    ) -> Result<StridedBuf<'a>> {
        check_out_buf(
            out.as_deref(),
            self.shape(),
            index,
            self.dtype().itemsize() as usize,
        )?;
        self.read_as_elementwise_pipeline::<S::Item>(index, context)?
            .to_buf(index, context, out)
    }

    #[inline]
    fn read_as_elementwise_pipeline<'a, T>(
        &'a self,
        index: &[Range<u64>],
        context: &'a ReadContext,
    ) -> Result<impl ElementwisePipeline<T> + use<'a, T, S, SMin, SMax>>
    where
        T: Dtyped,
    {
        check_dtype(Dtype::new_ref::<T>(), Dtype::new_ref::<S::Item>())?;
        let a = self
            .a
            .read_as_elementwise_pipeline::<S::Item>(index, context)?;
        let min = self
            .min
            .read_as_elementwise_pipeline::<S::Item>(index, context)?;
        let max = self
            .max
            .read_as_elementwise_pipeline::<S::Item>(index, context)?;

        struct ClampPipeline<P, PMin, PMax, U> {
            a: P,
            min: PMin,
            max: PMax,
            phantom: std::marker::PhantomData<U>,
        }
        impl<T, U, P, PMin, PMax> ElementwisePipelineImpl<T> for ClampPipeline<P, PMin, PMax, U>
        where
            P: ElementwisePipelineImpl<U>,
            PMin: ElementwisePipelineImpl<U>,
            PMax: ElementwisePipelineImpl<U>,
            U: crate::scalar::Maximum<Output = U> + crate::scalar::Minimum<Output = U> + Copy,
            T: Dtyped,
        {
            const N_OPERANDS: Option<usize> =
                n_operands_sum(&[P::N_OPERANDS, PMin::N_OPERANDS, PMax::N_OPERANDS]);
            const MAX_ITEMSIZE: usize = max_itemsize(&[
                size_of::<T>(),
                P::MAX_ITEMSIZE,
                PMin::MAX_ITEMSIZE,
                PMax::MAX_ITEMSIZE,
            ]);
            // Like the binary kernels, the max and min of `clamp` leave out the mixed loops.
            const INNER_LOOPS: InnerLoops = InnerLoops::and_all(&[
                InnerLoops::CONTIGUOUS,
                P::INNER_LOOPS,
                PMin::INNER_LOOPS,
                PMax::INNER_LOOPS,
            ]);

            #[inline]
            fn operands<'s>(&'s self) -> impl Iterator<Item = &'s Operand<'s>> + 's {
                self.a
                    .operands()
                    .chain(self.min.operands())
                    .chain(self.max.operands())
            }

            #[inline(always)]
            unsafe fn read_bulk<const N: usize, const CONTIGUOUS: bool, S: fearless_simd::Simd>(
                &self,
                offset: usize,
                status: &mut ArithmeticOpStatus,
                simd: S,
            ) -> [T; N] {
                let a = unsafe { self.a.read_bulk::<N, CONTIGUOUS, S>(offset, status, simd) };
                let min = unsafe { self.min.read_bulk::<N, CONTIGUOUS, S>(offset, status, simd) };
                let max = unsafe { self.max.read_bulk::<N, CONTIGUOUS, S>(offset, status, simd) };
                let ys = U::minimum_bulk(U::maximum_bulk(a, min, simd), max, simd);

                const { assert!(size_of::<U>() == size_of::<T>()) };
                // SAFETY: we checked `T` and `U` are the same dtype in the outer func
                unsafe { std::mem::transmute_copy::<[U; N], [T; N]>(&ys) }
            }
        }

        Ok(ClampPipeline {
            a,
            min,
            max,
            phantom: std::marker::PhantomData,
        })
    }

    #[inline(always)]
    fn shape(&self) -> &[u64] {
        self.a.shape()
    }

    #[inline(always)]
    fn dtype(&self) -> &Dtype {
        Dtype::new_ref::<S::Item>()
    }

    #[inline]
    fn spec(&self) -> ArraySpec<'_> {
        self.a
            .spec()
            .with_dynamic_spec(&self.spec)
            .with_cleared_flags()
    }

    fn info(&self) -> ArrayStorageInfo<'_> {
        ArrayStorageInfo::new_deps("Clamp", [&self.a, &self.min, &self.max])
    }

    crate::ops::impl_element_type_change_default!();
}

/// Element-wise approximate equality test.
///
/// Produces a `bool` array that is `true` at each position where the corresponding elements of
/// the two input arrays are within the specified tolerances:
/// `|a - b| <= max(atol, rtol * max(|a|, |b|))`.
///
/// **Not equivalent to `numpy.isclose`.** NumPy uses `|a - b| <= atol + rtol * |b|`: the
/// tolerances are combined additively and the relative term is asymmetric (scaled by `|b|` only).
/// Jix treats `atol` and `rtol` as independent thresholds and uses the symmetric
/// `max(|a|, |b|)` as the scale.
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// This struct is the bare storage implementation; the operation is also available as
/// [`Array::approx_equal()`](crate::Array::approx_equal).
///
/// # Examples
///
/// ```
/// use jix::Array;
/// use ndarray::array;
///
/// // atol=0.02 covers the 0.01 gap; the 1.0 difference in the third element does not fit.
/// let a = Array::compact_ndarray(&array![1.0f32, 2.0, 3.0])?;
/// let b = Array::compact_ndarray(&array![1.0f32, 2.01, 4.0])?;
/// let result = a.approx_equal(b, 0.0, 0.02).to_ndarray()?;
/// assert_eq!(result, array![true, true, false]);
///
/// // rtol=0.1 accepts a ~9 % relative difference but rejects an ~11 % one.
/// let c = Array::compact_ndarray(&array![100.0f32, 1.0])?;
/// let d = Array::compact_ndarray(&array![109.0f32, 1.12])?;
/// let result = c.approx_equal(d, 0.1, 0.0).to_ndarray()?;
/// assert_eq!(result, array![true, false]);
///
/// // NaN inputs always produce false.
/// let e = Array::compact_ndarray(&array![f32::NAN, 1.0f32])?;
/// let f = Array::compact_ndarray(&array![f32::NAN, 1.0f32])?;
/// let result = e.approx_equal(f, 0.0, 0.0).to_ndarray()?;
/// assert_eq!(result, array![false, true]);
///
/// // Same infinities are equal; opposite signs are not.
/// let g = Array::compact_ndarray(&array![f32::INFINITY, f32::NEG_INFINITY])?;
/// let h = Array::compact_ndarray(&array![f32::INFINITY, f32::INFINITY])?;
/// let result = g.approx_equal(h, 0.0, 0.0).to_ndarray()?;
/// assert_eq!(result, array![true, false]);
/// # Ok::<(), jix::Error>(())
/// ```
pub struct ApproxEq<S1, S2>(Op2<S1, S2, ApproxEqKernel<S1::Item>>)
where
    S1: ArrayStorageTyped<Item: crate::scalar::ApproxEq>;
struct ApproxEqKernel<T: crate::scalar::ApproxEq> {
    rtol: <T as crate::scalar::ApproxEq>::RelativeTolerance,
    atol: <T as crate::scalar::ApproxEq>::AbsoluteTolerance,
}
impl<T: crate::scalar::ApproxEq> Op2Kernel<T, T> for ApproxEqKernel<T> {
    type Output = bool;

    const INNER_LOOPS: InnerLoops = InnerLoops::CONTIGUOUS;

    #[inline(always)]
    fn apply(&self, a: T, b: T, _status: &mut crate::error::ArithmeticOpStatus) -> Self::Output {
        a.approx_eq(&b, &self.rtol, &self.atol)
    }

    #[inline(always)]
    fn apply_bulk<const N: usize, S: fearless_simd::Simd>(
        &self,
        a: [T; N],
        b: [T; N],
        _status: &mut crate::error::ArithmeticOpStatus,
        simd: S,
    ) -> [Self::Output; N] {
        T::approx_eq_bulk(a, b, &self.rtol, &self.atol, simd)
    }
}
impl<S1, S2> ApproxEq<S1, S2>
where
    S1: ArrayStorageTyped<Item: crate::scalar::ApproxEq>,
{
    /// Constructs an [`ApproxEq`] storage. See the struct docs for semantics and examples.
    pub fn new(
        a: S1,
        b: S2,
        rtol: <S1::Item as crate::scalar::ApproxEq>::RelativeTolerance,
        atol: <S1::Item as crate::scalar::ApproxEq>::AbsoluteTolerance,
    ) -> Result<Self>
    where
        S1: ArrayStorageTyped<Item: crate::scalar::ApproxEq>,
        S2: ArrayStorageTyped<Item = S1::Item>,
    {
        Ok(Self(Op2::new(a, b, ApproxEqKernel { rtol, atol }, false)?))
    }

    /// Constructs an array with [`ApproxEq`] storage. See the storage struct docs for semantics and examples.
    pub fn new_array(
        a: Array<S1>,
        b: Array<S2>,
        rtol: <S1::Item as crate::scalar::ApproxEq>::RelativeTolerance,
        atol: <S1::Item as crate::scalar::ApproxEq>::AbsoluteTolerance,
    ) -> Result<Array<Self>>
    where
        S1: ArrayStorageTyped<Item: crate::scalar::ApproxEq>,
        S2: ArrayStorageTyped<Item = S1::Item>,
    {
        Self::new(a.into_storage(), b.into_storage(), rtol, atol).map(Array::from_storage)
    }
}
impl<S1, S2> ArrayStorage for ApproxEq<S1, S2>
where
    S1: ArrayStorageTyped<Item: crate::scalar::ApproxEq>,
    S2: ArrayStorageTyped<Item = S1::Item>,
{
    type ElementType = Ty<bool>;
    crate::storage::impl_array_storage_forward!(<S1, S2>);

    fn info(&self) -> ArrayStorageInfo<'_> {
        ArrayStorageInfo::new_deps("ApproxEq", [&self.0.a, &self.0.b])
    }

    crate::ops::impl_element_type_change_default!();
}

impl<S> Array<S>
where
    S: ArrayStorage,
{
    define_array_op2_method!(equal: Equal, crate::scalar::Equal);
    define_array_op2_method!(not_equal: NotEqual, crate::scalar::NotEqual);

    /// Applies the [`Greater`] operation, see the op struct docs for details.
    #[track_caller]
    pub fn greater<S2>(self, other: Array<S2>) -> Array<Greater<S, S2>>
    where
        S: ArrayStorageTyped,
        S2: ArrayStorageTyped,
        S2::Item: crate::scalar::Less<S::Item, Output: Dtyped>,
    {
        Greater::new_array(self, other).unwrap()
    }

    /// Applies the [`GreaterEqual`] operation, see the op struct docs for details.
    #[track_caller]
    pub fn greater_equal<S2>(self, other: Array<S2>) -> Array<GreaterEqual<S, S2>>
    where
        S: ArrayStorageTyped,
        S2: ArrayStorageTyped,
        S2::Item: crate::scalar::LessEqual<S::Item, Output: Dtyped>,
    {
        GreaterEqual::new_array(self, other).unwrap()
    }

    define_array_op2_method!(less: Less, crate::scalar::Less);
    define_array_op2_method!(less_equal: LessEqual, crate::scalar::LessEqual);
    define_array_op2_method!(maximum: Maximum, crate::scalar::Maximum);
    define_array_op2_method!(minimum: Minimum, crate::scalar::Minimum);

    /// Applies the [`Clamp`] operation, see the op struct docs for details.
    #[track_caller]
    pub fn clamp<SMin, SMax>(
        self,
        min: Array<SMin>,
        max: Array<SMax>,
    ) -> Array<Clamp<S, SMin, SMax>>
    where
        S: ArrayStorageTyped,
        SMin: ArrayStorageTyped<Item = S::Item>,
        SMax: ArrayStorageTyped<Item = S::Item>,
        S::Item:
            crate::scalar::Maximum<Output = S::Item> + crate::scalar::Minimum<Output = S::Item>,
    {
        Clamp::new_array(self, min, max).unwrap()
    }

    /// Applies the [`ApproxEq`] operation, see the op struct docs for details.
    #[track_caller]
    pub fn approx_equal<S2>(
        self,
        other: crate::Array<S2>,
        rtol: <S::Item as crate::scalar::ApproxEq>::RelativeTolerance,
        atol: <S::Item as crate::scalar::ApproxEq>::AbsoluteTolerance,
    ) -> crate::Array<ApproxEq<S, S2>>
    where
        S: ArrayStorageTyped<Item: crate::scalar::ApproxEq>,
        S2: ArrayStorageTyped<Item = S::Item>,
    {
        ApproxEq::new_array(self, other, rtol, atol).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Ix1;

    #[cfg(feature = "half")]
    use crate::scalar::f16;
    use crate::scalar::{Maximum, Minimum};
    #[cfg(feature = "num-complex")]
    #[allow(non_camel_case_types)]
    type complex_f32 = crate::scalar::Complex<f32>;
    #[cfg(feature = "num-complex")]
    #[allow(non_camel_case_types)]
    type complex_f64 = crate::scalar::Complex<f64>;
    use crate::ops::op2::tests::test_op2;

    #[test]
    fn u64_vs_i64_is_exact() {
        use ndarray::Array1;

        use crate::ops::{Equal, Greater, GreaterEqual, Less, LessEqual, NotEqual};
        use crate::Array;

        let us = [
            0,
            1,
            (1 << 53) + 1,
            i64::MAX as u64,
            i64::MAX as u64 + 1,
            u64::MAX,
        ];
        let is = [i64::MIN, -1, 0, 1, (1 << 53), (1 << 53) + 1, i64::MAX];
        let (mut a, mut b) = (vec![], vec![]);
        for &u in &us {
            for &i in &is {
                a.push(u);
                b.push(i);
            }
        }
        let u = || Array::compact_ndarray(&Array1::from(a.clone())).unwrap();
        let i = || Array::compact_ndarray(&Array1::from(b.clone())).unwrap();
        let pairs = || a.iter().zip(&b).map(|(&u, &i)| (u as i128, i as i128));
        macro_rules! check {
            ($($Op:ident: $op:tt),*) => {$(
                let expected = pairs().map(|(u, i)| u $op i).collect::<Vec<_>>();
                assert_eq!($Op::new_array(u(), i()).unwrap().to_ndarray::<Ix1>().unwrap().to_vec(), expected);
                let expected = pairs().map(|(u, i)| i $op u).collect::<Vec<_>>();
                assert_eq!($Op::new_array(i(), u()).unwrap().to_ndarray::<Ix1>().unwrap().to_vec(), expected);
            )*};
        }
        check!(Equal: ==, NotEqual: !=, Less: <, LessEqual: <=, Greater: >, GreaterEqual: >=);
    }

    /// The `apply_bulk` of the kernels of this module (their scalar traits' SIMD `*_bulk`) against
    /// their scalar semantics, on every SIMD level of the CPU, over edge cases (NaN included).
    #[test]
    fn simd_bodies_all_levels() {
        use super::{
            EqualKernel, LessEqualKernel, LessKernel, MaximumKernel, MinimumKernel, NotEqualKernel,
        };
        use crate::ops::op2::Op2Kernel;
        use crate::util::{assert_same_elements, for_each_simd_level, SimdTestValues};
        use fearless_simd::Simd;

        fn check<S: Simd>(simd: S) {
            macro_rules! case {
                ($kernel:ident, $ta:ty, $tb:ty, $f:expr) => {
                    let (a, b) = (<$ta>::simd_test_values(0), <$tb>::simd_test_values(5));
                    let what = concat!(
                        stringify!($kernel),
                        " ",
                        stringify!($ta),
                        " ",
                        stringify!($tb)
                    );
                    assert_same_elements(
                        $kernel.apply_bulk(a, b, &mut Default::default(), simd),
                        |i| $f(a[i], b[i]),
                        what,
                    );
                };
            }
            macro_rules! all {
                ($kernel:ident, $f:expr) => {
                    case!($kernel, f32, f32, $f);
                    case!($kernel, f64, f64, $f);
                    case!($kernel, i8, i8, $f);
                    case!($kernel, i16, i16, $f);
                    case!($kernel, i32, i32, $f);
                    case!($kernel, i64, i64, $f);
                    case!($kernel, u8, u8, $f);
                    case!($kernel, u16, u16, $f);
                    case!($kernel, u32, u32, $f);
                    case!($kernel, u64, u64, $f);
                };
            }
            simd.vectorize(|| {
                all!(EqualKernel, |x, y| x == y);
                all!(NotEqualKernel, |x, y| x != y);
                all!(LessKernel, |x, y| x < y);
                all!(LessEqualKernel, |x, y| x <= y);
                all!(MaximumKernel, Maximum::maximum);
                all!(MinimumKernel, Minimum::minimum);
                // Not just a NaN: the one NaN, whichever operand's it was (`-NaN` included).
                macro_rules! nan_bits {
                    ($kernel:ident, $t:ty) => {
                        let (a, b) = (<$t>::simd_test_values(0), <$t>::simd_test_values(5));
                        for (i, y) in $kernel
                            .apply_bulk(a, b, &mut Default::default(), simd)
                            .into_iter()
                            .enumerate()
                        {
                            if y.is_nan() {
                                let what = concat!(stringify!($kernel), " ", stringify!($t));
                                assert_eq!(y.to_bits(), <$t>::NAN.to_bits(), "{what}: element {i}");
                            }
                        }
                    };
                }
                nan_bits!(MaximumKernel, f32);
                nan_bits!(MaximumKernel, f64);
                nan_bits!(MinimumKernel, f32);
                nan_bits!(MinimumKernel, f64);
            });
        }
        for_each_simd_level!(check);
    }

    // Shared by greater_equal_concrete and less_equal_concrete: NaN operands, inf/-inf
    // ordering, and an equal pair (1.0 == 1.0).
    const CMP_FLOAT_A: [f32; 4] = [f32::NAN, 1.0, f32::INFINITY, f32::NEG_INFINITY];
    const CMP_FLOAT_B: [f32; 4] = [1.0, f32::NAN, f32::INFINITY, 1.0];

    // equal: comparable_strategy gives ~33 % equal pairs and exercises NaN != NaN.
    // Output is bool, so NaN in float inputs is safe for assert_array_matches.
    test_op2!(
        equal,
        |a, b| a == b,
        [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool],
        comparable_strategy,
        #[cfg(feature = "half")]
        [f16],
        #[cfg(feature = "num-complex")]
        [complex_f32, complex_f64]
    );

    // not_equal edge cases: NaN != NaN is true (IEEE 754) while equal ints/infinities
    // are false, plus signed negatives.
    #[test]
    fn not_equal_concrete() {
        use crate::Array;
        // int path: equal values (false), signed negatives, differing values (true).
        let a = ndarray::array![1i32, -5, 3, 0];
        let b = ndarray::array![1i32, 5, 3, 0];
        let za = Array::compact_ndarray(&a).unwrap();
        let zb = Array::compact_ndarray(&b).unwrap();
        let expected = ndarray::Zip::from(&a).and(&b).map_collect(|&a, &b| a != b);
        crate::util::assert_array_matches(&za.view().not_equal(zb.view()), &expected);

        // float path: NaN != NaN is true; same infinities are equal (so not_equal is false).
        let fa = ndarray::array![f32::NAN, f32::INFINITY, 1.0, f32::NEG_INFINITY];
        let fb = ndarray::array![f32::NAN, f32::INFINITY, 1.0, 1.0];
        let zfa = Array::compact_ndarray(&fa).unwrap();
        let zfb = Array::compact_ndarray(&fb).unwrap();
        let fexp = ndarray::Zip::from(&fa)
            .and(&fb)
            .map_collect(|&a, &b| a != b);
        crate::util::assert_array_matches(&zfa.view().not_equal(zfb.view()), &fexp);
    }

    // greater: output is bool, so NaN inputs are safe.
    // Integers: any_strategy (no overflow). Floats: maybe_non_finite covers NaN -> false paths.
    test_op2!(
        greater,
        |a, b| a > b,
        [i8, i16, i32, i64, u8, u16, u32, u64, bool],
        any_strategy
    );
    test_op2!(
        greater,
        |a, b| a > b,
        [f32, f64],
        maybe_non_finite_strategy,
        #[cfg(feature = "half")]
        [f16]
    );

    #[test]
    fn greater_equal_concrete() {
        use crate::Array;
        // int path: equal values (true), ordering, signed negatives.
        let a = ndarray::array![3i32, -1, 2, 2];
        let b = ndarray::array![1i32, 1, 2, 3];
        let za = Array::compact_ndarray(&a).unwrap();
        let zb = Array::compact_ndarray(&b).unwrap();
        let expected = ndarray::Zip::from(&a).and(&b).map_collect(|&a, &b| a >= b);
        crate::util::assert_array_matches(&za.view().greater_equal(zb.view()), &expected);

        // float path: NaN comparisons are false; inf ordering and inf >= inf (true).
        let fa = ndarray::arr1(&CMP_FLOAT_A);
        let fb = ndarray::arr1(&CMP_FLOAT_B);
        let zfa = Array::compact_ndarray(&fa).unwrap();
        let zfb = Array::compact_ndarray(&fb).unwrap();
        let fexp = ndarray::Zip::from(&fa)
            .and(&fb)
            .map_collect(|&a, &b| a >= b);
        crate::util::assert_array_matches(&zfa.view().greater_equal(zfb.view()), &fexp);

        // bool path: true>=false, false>=false, true>=true, false>=true.
        let ba = ndarray::array![true, false, true, false];
        let bb = ndarray::array![false, false, true, true];
        let zba = Array::compact_ndarray(&ba).unwrap();
        let zbb = Array::compact_ndarray(&bb).unwrap();
        let bexp = ndarray::Zip::from(&ba)
            .and(&bb)
            .map_collect(|&a, &b| a >= b);
        crate::util::assert_array_matches(&zba.view().greater_equal(zbb.view()), &bexp);
    }

    test_op2!(
        greater_equal,
        |a, b| a >= b,
        [i8, i16, i32, i64, u8, u16, u32, u64, bool],
        any_strategy
    );
    test_op2!(
        greater_equal,
        |a, b| a >= b,
        [f32, f64],
        maybe_non_finite_strategy,
        #[cfg(feature = "half")]
        [f16]
    );

    // `Greater` stores its operands swapped inside a `Less`; everything that reports them must
    // still see the caller's order.
    #[test]
    fn swapped_cmp_reports_operands_in_caller_order() {
        use crate::{Array, ArrayParams, ArrayStorage};
        use ndarray::ShapeBuilder;

        let c_order = ndarray::Array::from_shape_vec([3, 4], (0..12i32).collect()).unwrap();
        let f_order = ndarray::Array::from_shape_vec([3, 4].f(), (0..12i32).collect()).unwrap();
        let mut params = ArrayParams::default();
        params.block_size(1 << 12);
        let a = Array::plain_ndarray_with(c_order, params).unwrap();
        let b = Array::plain_ndarray(f_order).unwrap();
        assert_ne!(
            a.storage().spec().block_size(),
            b.storage().spec().block_size()
        );

        let check = |storage: &dyn ArrayStorage, name: &str| {
            let info = storage.info();
            assert_eq!(info.name(), name);
            let [dep_a, dep_b] = info.dependencies() else {
                panic!("expected two dependencies")
            };
            // `view()` wraps the storage anew, so tell the operands apart by their layout.
            assert_eq!(dep_a.spec().read_layout_order(), &[0, 1]);
            assert_eq!(dep_b.spec().read_layout_order(), &[1, 0]);
        };
        let gt = a.view().greater(b.view());
        let ge = a.view().greater_equal(b.view());
        check(gt.storage(), "Greater");
        check(ge.storage(), "GreaterEqual");

        // The base spec comes from `a`, and a tie in the read layout order goes to `a`: C order,
        // where the swapped `Less` alone would pick `b`'s F order.
        for spec in [gt.storage().spec(), ge.storage().spec()] {
            assert_eq!(spec.block_size(), a.storage().spec().block_size());
            assert_eq!(spec.read_layout_order(), &[0, 1]);
        }
        let lt = b.view().less(a.view());
        assert_eq!(lt.storage().spec().read_layout_order(), &[1, 0]);
    }

    #[test]
    fn swapped_cmp_shape_mismatch_names_operands_in_caller_order() {
        use crate::ops::Greater;
        use crate::Array;

        let a = Array::compact_ndarray(&ndarray::Array2::<i32>::zeros([2, 3])).unwrap();
        let b = Array::compact_ndarray(&ndarray::Array2::<i32>::zeros([3, 2])).unwrap();
        let Err(err) = Greater::new_array(a, b) else {
            panic!("expected a shape mismatch")
        };
        assert!(
            err.to_string().contains("`a` [2, 3] and `b` [3, 2]"),
            "{err}"
        );
    }

    #[test]
    // The bool path below intentionally exercises `a < b` on bool arrays directly (clippy
    // would rewrite this to `!a & b`, which is less clear about what's being tested).
    #[allow(clippy::bool_comparison)]
    fn less_concrete() {
        use crate::Array;
        // int path: equal values (false), ordering, signed negatives.
        let a = ndarray::array![1i32, 1, 3, -5];
        let b = ndarray::array![3i32, 1, 1, -1];
        let za = Array::compact_ndarray(&a).unwrap();
        let zb = Array::compact_ndarray(&b).unwrap();
        let expected = ndarray::Zip::from(&a).and(&b).map_collect(|&a, &b| a < b);
        crate::util::assert_array_matches(&za.view().less(zb.view()), &expected);

        // float path: NaN comparisons are false; -inf < finite, inf < inf is false.
        let fa = ndarray::array![f32::NAN, 1.0, f32::NEG_INFINITY, f32::INFINITY];
        let fb = ndarray::array![1.0f32, f32::NAN, 1.0, f32::INFINITY];
        let zfa = Array::compact_ndarray(&fa).unwrap();
        let zfb = Array::compact_ndarray(&fb).unwrap();
        let fexp = ndarray::Zip::from(&fa).and(&fb).map_collect(|&a, &b| a < b);
        crate::util::assert_array_matches(&zfa.view().less(zfb.view()), &fexp);

        // bool path: false<true (true), true<false (false), equal values are false.
        let ba = ndarray::array![false, true, true, false];
        let bb = ndarray::array![true, false, true, false];
        let zba = Array::compact_ndarray(&ba).unwrap();
        let zbb = Array::compact_ndarray(&bb).unwrap();
        let bexp = ndarray::Zip::from(&ba).and(&bb).map_collect(|&a, &b| a < b);
        crate::util::assert_array_matches(&zba.view().less(zbb.view()), &bexp);
    }

    #[test]
    fn less_equal_concrete() {
        use crate::Array;
        // int path incl. equal values (a<=a true) and ordering
        let a = ndarray::array![1i32, 2, 2, 5];
        let b = ndarray::array![2i32, 2, 1, 5];
        let za = Array::compact_ndarray(&a).unwrap();
        let zb = Array::compact_ndarray(&b).unwrap();
        let expected = ndarray::Zip::from(&a).and(&b).map_collect(|&a, &b| a <= b);
        crate::util::assert_array_matches(&za.view().less_equal(zb.view()), &expected);

        // float path: NaN <= x is false; inf ordering.
        let fa = ndarray::arr1(&CMP_FLOAT_A);
        let fb = ndarray::arr1(&CMP_FLOAT_B);
        let zfa = Array::compact_ndarray(&fa).unwrap();
        let zfb = Array::compact_ndarray(&fb).unwrap();
        let fexp = ndarray::Zip::from(&fa)
            .and(&fb)
            .map_collect(|&a, &b| a <= b);
        crate::util::assert_array_matches(&zfa.view().less_equal(zfb.view()), &fexp);

        // bool path on a 2x2 shape with a non-default 1x1 block shape, so a
        // block-boundary bug in the ordering kernel would still show up.
        let ba = ndarray::array![[false, true], [true, false]];
        let bb = ndarray::array![[true, false], [true, true]];
        let mut params_a = crate::ArrayParams::new();
        params_a.block_shape(&[1u32, 1]);
        let mut params_b = crate::ArrayParams::new();
        params_b.block_shape(&[1u32, 1]);
        let zba = Array::compact_ndarray_with(&ba, params_a).unwrap();
        let zbb = Array::compact_ndarray_with(&bb, params_b).unwrap();
        let bexp = ndarray::Zip::from(&ba)
            .and(&bb)
            .map_collect(|&a, &b| a <= b);
        crate::util::assert_array_matches(&zba.view().less_equal(zbb.view()), &bexp);
    }

    // approx_equal: output is bool so NaN inputs are safe; use dedicated proptest tests because
    // the method takes extra rtol/atol parameters that test_op2! cannot supply.
    fn ref_approx_eq<T>(a: T, b: T, rtol: T, atol: T) -> bool
    where
        T: Copy
            + PartialOrd
            + core::ops::Sub<Output = T>
            + core::ops::Mul<Output = T>
            + crate::scalar::Abs<Output = T>
            + crate::scalar::IsInfinite<Output = bool>,
    {
        if a == b {
            return true;
        }
        if a.is_infinite() || b.is_infinite() {
            return false;
        }
        let diff = (a - b).abs();
        if diff <= atol {
            return true;
        }
        let largest = if a.abs() > b.abs() { a.abs() } else { b.abs() };
        diff <= largest * rtol
    }

    /// The branch-free `approx_eq` against the reference, on every pair of edge values (infinities,
    /// NaN, +-0, subnormals, MAX) and odd tolerances.
    #[test]
    fn approx_eq_edge_cases() {
        use crate::scalar::ApproxEq;
        use crate::util::SimdTestValues;
        macro_rules! check {
            ($t:ty) => {
                let values = <$t>::edge_cases();
                let tols = [0.0, -0.0, 1e-6, 0.1, 1.0, -0.1, <$t>::INFINITY, <$t>::NAN];
                for &a in &values {
                    for &b in &values {
                        for &rtol in &tols {
                            for &atol in &tols {
                                assert_eq!(
                                    a.approx_eq(&b, &rtol, &atol),
                                    ref_approx_eq(a, b, rtol, atol),
                                    "{a:?} {b:?} rtol={rtol:?} atol={atol:?}",
                                );
                            }
                        }
                    }
                }
            };
        }
        check!(f32);
        check!(f64);
    }

    proptest::proptest! {
        #[test]
        fn approx_equal_f32(
            (arrays, rtol, atol) in (
                crate::util::arrays2_strategy_generic::<f32>(
                    crate::util::shape_strategy(),
                    <f32 as crate::util::ScalarStrategy>::maybe_non_finite_strategy(),
                ),
                0.0f32..=0.5f32,
                0.0f32..=0.5f32,
            )
        ) {
            let ((nd_a, za), (nd_b, zb)) = arrays;
            let result = za.approx_equal(zb, rtol, atol);
            let expected = ndarray::Zip::from(&nd_a)
                .and(&nd_b)
                .map_collect(|&a, &b| ref_approx_eq(a, b, rtol, atol));
            crate::util::assert_array_matches(&result, &expected);
        }

        #[test]
        fn approx_equal_f64(
            (arrays, rtol, atol) in (
                crate::util::arrays2_strategy_generic::<f64>(
                    crate::util::shape_strategy(),
                    <f64 as crate::util::ScalarStrategy>::maybe_non_finite_strategy(),
                ),
                0.0f64..=0.5f64,
                0.0f64..=0.5f64,
            )
        ) {
            let ((nd_a, za), (nd_b, zb)) = arrays;
            let result = za.approx_equal(zb, rtol, atol);
            let expected = ndarray::Zip::from(&nd_a)
                .and(&nd_b)
                .map_collect(|&a, &b| ref_approx_eq(a, b, rtol, atol));
            crate::util::assert_array_matches(&result, &expected);
        }
    }

    #[cfg(feature = "half")]
    proptest::proptest! {
        #[test]
        fn approx_equal_f16(
            (arrays, rtol_f32, atol_f32) in (
                crate::util::arrays2_strategy_generic::<f16>(
                    crate::util::shape_strategy(),
                    <f16 as crate::util::ScalarStrategy>::maybe_non_finite_strategy(),
                ),
                0.0f32..=0.5f32,
                0.0f32..=0.5f32,
            )
        ) {
            let rtol = f16::from_f32(rtol_f32);
            let atol = f16::from_f32(atol_f32);
            let ((nd_a, za), (nd_b, zb)) = arrays;
            let result = za.approx_equal(zb, rtol, atol);
            let expected = ndarray::Zip::from(&nd_a)
                .and(&nd_b)
                .map_collect(|&a, &b| ref_approx_eq(a, b, rtol, atol));
            crate::util::assert_array_matches(&result, &expected);
        }
    }

    #[cfg(feature = "num-complex")]
    proptest::proptest! {
        #[test]
        fn approx_equal_complex_f32(
            (arrays, rtol, atol_re, atol_im) in (
                crate::util::arrays2_strategy_generic::<complex_f32>(
                    crate::util::shape_strategy(),
                    <complex_f32 as crate::util::ScalarStrategy>::maybe_non_finite_strategy(),
                ),
                0.0f32..=0.5f32,
                0.0f32..=0.5f32,
                0.0f32..=0.5f32,
            )
        ) {
            let ((nd_a, za), (nd_b, zb)) = arrays;
            let atol = complex_f32 { re: atol_re, im: atol_im };
            let result = za.approx_equal(zb, rtol, atol);
            let expected = ndarray::Zip::from(&nd_a)
                .and(&nd_b)
                .map_collect(|a, b| {
                    ref_approx_eq(a.re, b.re, rtol, atol_re)
                        && ref_approx_eq(a.im, b.im, rtol, atol_im)
                });
            crate::util::assert_array_matches(&result, &expected);
        }

        #[test]
        fn approx_equal_complex_f64(
            (arrays, rtol, atol_re, atol_im) in (
                crate::util::arrays2_strategy_generic::<complex_f64>(
                    crate::util::shape_strategy(),
                    <complex_f64 as crate::util::ScalarStrategy>::maybe_non_finite_strategy(),
                ),
                0.0f64..=0.5f64,
                0.0f64..=0.5f64,
                0.0f64..=0.5f64,
            )
        ) {
            let ((nd_a, za), (nd_b, zb)) = arrays;
            let atol = complex_f64 { re: atol_re, im: atol_im };
            let result = za.approx_equal(zb, rtol, atol);
            let expected = ndarray::Zip::from(&nd_a)
                .and(&nd_b)
                .map_collect(|a, b| {
                    ref_approx_eq(a.re, b.re, rtol, atol_re)
                        && ref_approx_eq(a.im, b.im, rtol, atol_im)
                });
            crate::util::assert_array_matches(&result, &expected);
        }
    }

    // maximum: NaN propagates to the float *output*, which breaks assert_eq via PartialEq.
    // Use op_safe_strategy for floats to keep all outputs finite.
    test_op2!(
        maximum,
        |a, b| Maximum::maximum(a, b),
        [i8, i16, i32, i64, u8, u16, u32, u64, bool],
        any_strategy
    );
    test_op2!(
        maximum,
        |a, b| Maximum::maximum(a, b),
        [f32, f64],
        op_safe_strategy,
        #[cfg(feature = "half")]
        [f16]
    );

    // minimum edge cases: signed negatives, equal values, inf ordering (finite outputs),
    // and a NaN-propagation case matching the kernel (checked directly since NaN output
    // breaks the PartialEq-based assert_array_matches oracle).
    #[test]
    fn minimum_concrete() {
        use crate::Array;
        // int path: negatives, equal values, ordering.
        let a = ndarray::array![1i32, -5, 3, 3];
        let b = ndarray::array![4i32, -2, 3, -3];
        let za = Array::compact_ndarray(&a).unwrap();
        let zb = Array::compact_ndarray(&b).unwrap();
        let expected = ndarray::Zip::from(&a)
            .and(&b)
            .map_collect(|&a, &b| Minimum::minimum(a, b));
        crate::util::assert_array_matches(&za.view().minimum(zb.view()), &expected);

        // float path: inf/-inf ordering, all finite outputs (no NaN yet).
        let fa = ndarray::array![f32::NEG_INFINITY, 1.0, f32::INFINITY, -1.0];
        let fb = ndarray::array![5.0f32, -1.0, 2.0, -1.0];
        let zfa = Array::compact_ndarray(&fa).unwrap();
        let zfb = Array::compact_ndarray(&fb).unwrap();
        let fexp = ndarray::Zip::from(&fa)
            .and(&fb)
            .map_collect(|&a, &b| Minimum::minimum(a, b));
        crate::util::assert_array_matches(&zfa.view().minimum(zfb.view()), &fexp);

        // NaN-propagation edge: minimum(NaN, x) == NaN for any x, unlike f32::min which
        // discards NaN when exactly one operand is NaN.
        let na = ndarray::array![f32::NAN, 1.0f32];
        let nb = ndarray::array![2.0f32, f32::NAN];
        let zna = Array::compact_ndarray(&na).unwrap();
        let znb = Array::compact_ndarray(&nb).unwrap();
        let result = zna.view().minimum(znb.view()).to_ndarray::<Ix1>().unwrap();
        assert!(result[[0]].is_nan());
        assert!(result[[1]].is_nan());
    }

    // clamp: per-element bounds, min > max (max wins, as in numpy.clip / torch.clamp), and NaN
    // in any operand propagating to the output.
    #[test]
    fn clamp_concrete() {
        use crate::Array;
        let a = ndarray::array![[-3i32, 0, 5], [10, 4, 4]];
        let lo = ndarray::array![[0i32, 1, 0], [0, 6, 2]];
        let hi = ndarray::array![[7i32, 7, 4], [7, 3, 9]];
        let za = Array::compact_ndarray_with(&a, crate::util::arr_params(&[1, 2])).unwrap();
        let zlo = Array::compact_ndarray(&lo).unwrap();
        let zhi = Array::compact_ndarray(&hi).unwrap();
        let expected = ndarray::array![[0i32, 1, 4], [7, 3, 4]]; // [1][1]: min 6 > max 3 -> 3
        crate::util::assert_array_matches(&za.clamp(zlo, zhi), &expected);

        let f = Array::compact_ndarray(&ndarray::array![f32::NAN, 1.0, 1.0, 5.0]).unwrap();
        let flo = Array::compact_ndarray(&ndarray::array![0.0f32, f32::NAN, 0.0, 0.0]).unwrap();
        let fhi = Array::compact_ndarray(&ndarray::array![2.0f32, 2.0, f32::NAN, 2.0]).unwrap();
        let result = f.clamp(flo, fhi).to_ndarray::<Ix1>().unwrap();
        assert!(result.iter().take(3).all(|x| x.is_nan()));
        assert_eq!(result[3], 2.0);
    }

    proptest::proptest! {
        #[test]
        fn clamp_i32(
            ((nd_a, za), (nd_lo, zlo)) in crate::util::arrays2_strategy_generic::<i32>(
                crate::util::shape_strategy(),
                <i32 as crate::util::ScalarStrategy>::any_strategy(),
            ),
            hi_offset in -50i32..50,
        ) {
            let nd_hi = nd_lo.mapv(|x| x.saturating_add(hi_offset));
            let zhi = crate::Array::compact_ndarray(&nd_hi).unwrap();
            let expected = ndarray::Zip::from(&nd_a)
                .and(&nd_lo)
                .and(&nd_hi)
                .map_collect(|&a, &lo, &hi| a.max(lo).min(hi));
            crate::util::assert_array_matches(&za.clamp(zlo, zhi), &expected);
        }
    }
}
