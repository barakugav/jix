use crate::error::Result;
use crate::ops::common::define_array_op2_method;
use crate::ops::op2::define_op2;
use crate::ops::{Op2, Op2Kernel};
use crate::storage::{ArrayStorageInfo, ArrayStorageTyped};
use crate::{Array, ArrayStorage, Ty};

pub(crate) mod _traits {
    #[cfg(feature = "half")]
    use crate::scalar::f16;
    #[cfg(feature = "num-complex")]
    use crate::scalar::Complex;

    /// Element-wise maximum with NaN-propagating semantics for floating-point types.
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
    /// silently discarding it.
    ///
    /// For integer and `bool` types the implementation delegates to [`std::cmp::max`], which is
    /// equivalent. The trait therefore provides a single uniform interface usable across all
    /// supported numeric dtypes.
    pub trait Maximum<Rhs = Self> {
        /// The output element type of this maximum operation.
        type Output;
        /// Return the element-wise maximum of `self` and `other`, propagating `NaN` for floats.
        fn maximum(self, other: Rhs) -> Self::Output;
    }
    macro_rules! impl_integer_maximum {
        ($($t:ty),* $(,)?) => {
            $(impl Maximum for $t {
                type Output = Self;

                #[inline(always)]
                fn maximum(self, other: Self) -> Self {
                    std::cmp::max(self, other)
                }
            })*
        };
    }
    macro_rules! impl_float_maximum {
        ($($t:ty),* $(,)?) => {
            $(impl Maximum for $t {
                type Output = Self;

                #[inline(always)]
                fn maximum(self, other: Self) -> Self {
                    if self.is_nan() | other.is_nan() {
                        Self::NAN
                    } else {
                        self.max(other)
                    }
                }
            })*
        };
    }
    impl_integer_maximum!(i8, i16, i32, i64, u8, u16, u32, u64, bool);
    impl_float_maximum!(f32, f64);
    #[cfg(feature = "half")]
    impl_float_maximum!(f16);

    /// Element-wise minimum with NaN-propagating semantics for floating-point types.
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
    /// silently discarding it.
    ///
    /// For integer and `bool` types the implementation delegates to [`std::cmp::min`], which is
    /// equivalent. The trait therefore provides a single uniform interface usable across all
    /// supported numeric dtypes.
    pub trait Minimum<Rhs = Self> {
        // TODO: rename this to MinimumPartial

        /// The output element type of this minimum operation.
        type Output;
        /// Return the element-wise minimum of `self` and `other`, propagating `NaN` for floats.
        fn minimum(self, other: Rhs) -> Self::Output;
    }
    macro_rules! impl_integer_minimum {
        ($($t:ty),* $(,)?) => {
            $(impl Minimum for $t {
                type Output = Self;

                #[inline(always)]
                fn minimum(self, other: Self) -> Self {
                    std::cmp::min(self, other)
                }
            })*
        };
    }
    macro_rules! impl_float_minimum {
    ($($t:ty),* $(,)?) => {
        $(impl Minimum for $t {
            type Output = Self;

            #[inline(always)]
            fn minimum(self, other: Self) -> Self {
                if self.is_nan() | other.is_nan() {
                    Self::NAN
                } else {
                    self.min(other)
                }
            }
        })*
    };
}
    impl_integer_minimum!(i8, i16, i32, i64, u8, u16, u32, u64, bool);
    impl_float_minimum!(f32, f64);
    #[cfg(feature = "half")]
    impl_float_minimum!(f16);

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

                    // Handle same infinities
                    if self == other {
                        return true;
                    }

                    // Handle remaining infinities
                    if <$T>::is_infinite(*self) || <$T>::is_infinite(*other) {
                        return false;
                    }

                    let abs_diff = <$T as num_traits::Float>::abs(self - other);

                    // For when the numbers are really close together
                    if abs_diff <= *atol {
                        return true;
                    }

                    let abs_self = <$T as num_traits::Float>::abs(*self);
                    let abs_other = <$T as num_traits::Float>::abs(*other);

                    let largest = if abs_other > abs_self {
                        abs_other
                    } else {
                        abs_self
                    };

                    // Use a relative difference comparison
                    abs_diff <= largest * rtol
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
                && self.im.approx_eq(&other.im, rtol, &atol.im)
        }
    }
}

define_op2!(
    /// Element-wise equality test (`a == b`).
    ///
    /// Output dtype is `bool`.
    ///
    /// For **float** types, `NaN != NaN` per IEEE 754: comparing two `NaN` values
    /// returns `false`.
    /// For **complex** types, both the real and imaginary components must be equal.
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
    /// assert_eq!(result.as_slice().unwrap(), &[true, false, true]);
    ///
    /// // NaN != NaN per IEEE 754.
    /// let c = Array::compact_ndarray(&array![f32::NAN, 1.0f32])?;
    /// let d = Array::compact_ndarray(&array![f32::NAN, 1.0f32])?;
    /// let result = c.equal(d).to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[false, true]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Equal,
    EqualKernel,
    <PartialEq>::eq(&a, &b),
    type Output = bool,
);
define_op2!(
    /// Element-wise inequality test (`a != b`).
    ///
    /// Output dtype is `bool`.
    ///
    /// For **float** types, `NaN != NaN` returns `true` per IEEE 754.
    /// For **complex** types, returns `true` if either the real or imaginary component differs.
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
    /// assert_eq!(result.as_slice().unwrap(), &[false, true, false]);
    ///
    /// // NaN != NaN is true per IEEE 754.
    /// let c = Array::compact_ndarray(&array![f32::NAN, 1.0f32])?;
    /// let d = Array::compact_ndarray(&array![f32::NAN, 2.0f32])?;
    /// let result = c.not_equal(d).to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[true, true]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    NotEqual,
    NotEqualKernel,
    <PartialEq>::ne(&a, &b),
    type Output = bool,
);
define_op2!(
    /// Element-wise greater-than test (`a > b`).
    ///
    /// Complex types are not supported as they have no total ordering. Output dtype is `bool`.
    ///
    /// For **float** types, any comparison involving `NaN` returns `false` (IEEE 754).
    /// For **bool**: `true > false`.
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
    /// assert_eq!(result.as_slice().unwrap(), &[true, false, false]);
    ///
    /// // true > false for bool dtype.
    /// let c = Array::compact_ndarray(&array![true, false, true])?;
    /// let d = Array::compact_ndarray(&array![false, false, true])?;
    /// let result = c.greater(d).to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[true, false, false]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Greater,
    GreaterKernel,
    <PartialOrd>::gt(&a, &b),
    type Output = bool,
);
define_op2!(
    /// Element-wise greater-than-or-equal test (`a >= b`).
    ///
    /// Complex types are not supported as they have no total ordering. Output dtype is `bool`.
    ///
    /// For **float** types, any comparison involving `NaN` returns `false` (IEEE 754).
    /// For **bool**: `true >= false`, and both `true >= true` and `false >= false` hold.
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
    /// assert_eq!(result.as_slice().unwrap(), &[true, true, false]);
    ///
    /// // true >= false and false >= false hold.
    /// let c = Array::compact_ndarray(&array![true, false, true])?;
    /// let d = Array::compact_ndarray(&array![false, false, true])?;
    /// let result = c.greater_equal(d).to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[true, true, true]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    GreaterEqual,
    GreaterEqualKernel,
    <PartialOrd>::ge(&a, &b),
    type Output = bool,
);
define_op2!(
    /// Element-wise less-than test (`a < b`).
    ///
    /// Complex types are not supported as they have no total ordering. Output dtype is `bool`.
    ///
    /// For **float** types, any comparison involving `NaN` returns `false` (IEEE 754).
    /// For **bool**: `false < true`.
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
    /// assert_eq!(result.as_slice().unwrap(), &[true, false, false]);
    ///
    /// // false < true for bool dtype.
    /// let c = Array::compact_ndarray(&array![false, false, true])?;
    /// let d = Array::compact_ndarray(&array![true, false, true])?;
    /// let result = c.less(d).to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[true, false, false]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Less,
    LessKernel,
    <PartialOrd>::lt(&a, &b),
    type Output = bool,
);
define_op2!(
    /// Element-wise less-than-or-equal test (`a <= b`).
    ///
    /// Complex types are not supported as they have no total ordering. Output dtype is `bool`.
    ///
    /// For **float** types, any comparison involving `NaN` returns `false` (IEEE 754).
    /// For **bool**: `false <= true`, and both `false <= false` and `true <= true` hold.
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
    /// assert_eq!(result.as_slice().unwrap(), &[true, true, false]);
    ///
    /// // false <= true and true <= true hold.
    /// let c = Array::compact_ndarray(&array![false, false, true])?;
    /// let d = Array::compact_ndarray(&array![true, false, true])?;
    /// let result = c.less_equal(d).to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[true, true, true]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    LessEqual,
    LessEqualKernel,
    <PartialOrd>::le(&a, &b),
    type Output = bool,
);

define_op2!(
    /// Element-wise maximum of two arrays.
    ///
    /// For **integer** and **bool** types the result is `std::cmp::max(a, b)`.
    /// For **float** types this operation is NaN-propagating: if either operand is `NaN`,
    /// the result is `NaN`. This deviates from [`f32::max`], which returns the non-`NaN`
    /// operand when exactly one is `NaN`, but matches the behaviour of `numpy.maximum`.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::maximum()`](crate::Array::maximum).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![1i32, 5, 3])?;
    /// let b = Array::compact_ndarray(&array![4i32, 2, 3])?;
    /// let result = a.maximum(b).to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[4, 5, 3]);
    ///
    /// // NaN is propagated: if either operand is NaN the result is NaN.
    /// let c = Array::compact_ndarray(&array![f32::NAN, 1.0f32])?;
    /// let d = Array::compact_ndarray(&array![2.0f32, 3.0f32])?;
    /// let result = c.maximum(d).to_ndarray()?;
    /// assert!(result[[0]].is_nan());
    /// assert_eq!(result[[1]], 3.0);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Maximum,
    MaximumKernel,
    <crate::scalar::Maximum>::maximum(a, b),
);
define_op2!(
    /// Element-wise minimum of two arrays.
    ///
    /// For **integer** and **bool** types the result is `std::cmp::min(a, b)`.
    /// For **float** types this operation is NaN-propagating: if either operand is `NaN`,
    /// the result is `NaN`. This deviates from [`f32::min`], which returns the non-`NaN`
    /// operand when exactly one is `NaN`, but matches the behaviour of `numpy.minimum`.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::minimum()`](crate::Array::minimum).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![1i32, 5, 3])?;
    /// let b = Array::compact_ndarray(&array![4i32, 2, 3])?;
    /// let result = a.minimum(b).to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[1, 2, 3]);
    ///
    /// // NaN is propagated: if either operand is NaN the result is NaN.
    /// let c = Array::compact_ndarray(&array![f32::NAN, 1.0f32])?;
    /// let d = Array::compact_ndarray(&array![2.0f32, 3.0f32])?;
    /// let result = c.minimum(d).to_ndarray()?;
    /// assert!(result[[0]].is_nan());
    /// assert_eq!(result[[1]], 1.0);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Minimum,
    MinimumKernel,
    <crate::scalar::Minimum>::minimum(a, b),
);

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
/// assert_eq!(result.as_slice().unwrap(), &[true, true, false]);
///
/// // rtol=0.1 accepts a ~9 % relative difference but rejects an ~11 % one.
/// let c = Array::compact_ndarray(&array![100.0f32, 1.0])?;
/// let d = Array::compact_ndarray(&array![109.0f32, 1.12])?;
/// let result = c.approx_equal(d, 0.1, 0.0).to_ndarray()?;
/// assert_eq!(result.as_slice().unwrap(), &[true, false]);
///
/// // NaN inputs always produce false.
/// let e = Array::compact_ndarray(&array![f32::NAN, 1.0f32])?;
/// let f = Array::compact_ndarray(&array![f32::NAN, 1.0f32])?;
/// let result = e.approx_equal(f, 0.0, 0.0).to_ndarray()?;
/// assert_eq!(result.as_slice().unwrap(), &[false, true]);
///
/// // Same infinities are equal; opposite signs are not.
/// let g = Array::compact_ndarray(&array![f32::INFINITY, f32::NEG_INFINITY])?;
/// let h = Array::compact_ndarray(&array![f32::INFINITY, f32::INFINITY])?;
/// let result = g.approx_equal(h, 0.0, 0.0).to_ndarray()?;
/// assert_eq!(result.as_slice().unwrap(), &[true, false]);
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

    #[inline(always)]
    fn apply(&self, a: T, b: T) -> Self::Output {
        a.approx_eq(&b, &self.rtol, &self.atol)
    }
}
impl<S1, S2> ApproxEq<S1, S2>
where
    S1: ArrayStorageTyped<Item: crate::scalar::ApproxEq>,
{
    /// Constructs a [`ApproxEq`] storage. See the struct docs for semantics and examples.
    pub fn new(
        a: S1,
        b: S2,
        rtol: <S1::Item as crate::scalar::ApproxEq>::RelativeTolerance,
        atol: <S1::Item as crate::scalar::ApproxEq>::AbsoluteTolerance,
    ) -> Result<Self>
    where
        S1: ArrayStorageTyped<Item: crate::scalar::ApproxEq>,
        S2: ArrayStorageTyped<Item = S1::Item, Dimension = S1::Dimension>,
    {
        Ok(Self(Op2::new(a, b, ApproxEqKernel { rtol, atol })?))
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
        S2: ArrayStorageTyped<Item = S1::Item, Dimension = S1::Dimension>,
    {
        Self::new(a.into_storage(), b.into_storage(), rtol, atol).map(Array::from_storage)
    }
}
impl<S1, S2> ArrayStorage for ApproxEq<S1, S2>
where
    S1: ArrayStorageTyped<Item: crate::scalar::ApproxEq>,
    S2: ArrayStorageTyped<Item = S1::Item, Dimension = S1::Dimension>,
{
    type ElementType = Ty<bool>;
    type Dimension = S1::Dimension;
    crate::storage::impl_array_storage_forward!(<S1, S2>);

    fn info(&self) -> ArrayStorageInfo<'_> {
        ArrayStorageInfo::new_deps("ApproxEq", [&self.0.a, &self.0.b])
    }

    type DimensionChange<NewD: crate::Dimension> =
        ApproxEq<S1::DimensionChange<NewD>, S2::DimensionChange<NewD>>;
    #[inline]
    fn dimension_change<NewD: crate::Dimension>(
        self,
    ) -> crate::error::Result<Self::DimensionChange<NewD>> {
        Ok(ApproxEq(self.0.dimension_change()?))
    }

    crate::ops::impl_element_type_change_default!();
}

impl<S> Array<S>
where
    S: ArrayStorage,
{
    define_array_op2_method!(equal: Equal, PartialEq, fixed_output_type = true);
    define_array_op2_method!(not_equal: NotEqual, PartialEq, fixed_output_type = true);
    define_array_op2_method!(greater: Greater, PartialOrd, fixed_output_type = true);
    define_array_op2_method!(greater_equal: GreaterEqual, PartialOrd, fixed_output_type = true);
    define_array_op2_method!(less: Less, PartialOrd, fixed_output_type = true);
    define_array_op2_method!(less_equal: LessEqual, PartialOrd, fixed_output_type = true);
    define_array_op2_method!(maximum: Maximum, crate::scalar::Maximum);
    define_array_op2_method!(minimum: Minimum, crate::scalar::Minimum);

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
        S2: ArrayStorageTyped<Item = S::Item, Dimension = S::Dimension>,
    {
        ApproxEq::new_array(self, other, rtol, atol).unwrap()
    }
}

#[cfg(test)]
mod tests {
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

    // equal / not_equal: comparable_strategy gives ~33 % equal pairs and exercises NaN != NaN.
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
    test_op2!(
        not_equal,
        |a, b| a != b,
        [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool],
        comparable_strategy,
        #[cfg(feature = "half")]
        [f16],
        #[cfg(feature = "num-complex")]
        [complex_f32, complex_f64]
    );

    // Ordering ops: output is bool, so NaN inputs are safe.
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
    test_op2!(
        less,
        |a, b| a < b,
        [i8, i16, i32, i64, u8, u16, u32, u64, bool],
        any_strategy
    );
    test_op2!(
        less,
        |a, b| a < b,
        [f32, f64],
        maybe_non_finite_strategy,
        #[cfg(feature = "half")]
        [f16]
    );
    test_op2!(
        less_equal,
        |a, b| a <= b,
        [i8, i16, i32, i64, u8, u16, u32, u64, bool],
        any_strategy
    );
    test_op2!(
        less_equal,
        |a, b| a <= b,
        [f32, f64],
        maybe_non_finite_strategy,
        #[cfg(feature = "half")]
        [f16]
    );

    // approx_equal: output is bool so NaN inputs are safe; use dedicated proptest tests because
    // the method takes extra rtol/atol parameters that test_op2! cannot supply.
    fn ref_approx_eq<T: num_traits::Float>(a: T, b: T, rtol: T, atol: T) -> bool {
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

    proptest::proptest! {
        #[test]
        fn approx_equal_f32(
            (arrays, rtol, atol) in (
                crate::util::carrays2_strategy_generic::<f32>(
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
                crate::util::carrays2_strategy_generic::<f64>(
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
                crate::util::carrays2_strategy_generic::<f16>(
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
                crate::util::carrays2_strategy_generic::<complex_f32>(
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
                crate::util::carrays2_strategy_generic::<complex_f64>(
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

    // maximum / minimum: NaN propagates to the float *output*, which breaks assert_eq via PartialEq.
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
    test_op2!(
        minimum,
        |a, b| Minimum::minimum(a, b),
        [i8, i16, i32, i64, u8, u16, u32, u64, bool],
        any_strategy
    );
    test_op2!(
        minimum,
        |a, b| Minimum::minimum(a, b),
        [f32, f64],
        op_safe_strategy,
        #[cfg(feature = "half")]
        [f16]
    );
}
