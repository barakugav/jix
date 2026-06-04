use crate::ops::common::define_array_op2_method;
use crate::ops::op2::define_op2;
#[allow(unused_imports)]
use crate::scalar::{f16, Complex};
use crate::{Array, ArrayStorage};

pub(crate) mod _traits {
    #[allow(unused_imports)]
    use crate::scalar::{f16, Complex};

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
        /// The output element type of this minimum operation.
        type Output;
        /// Return the element-wise minimum of `self` and `other`, propagating `NaN` for floats.
        fn minimum(self, other: Rhs) -> Self::Output;
    }
    macro_rules! impl_integer_minimum {
    ($($t:ty),* $(,)?) => {
        $(impl Minimum for $t { // TODO: rename
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
    /// let a = Array::compact_array(&array![1i32, 2, 3])?;
    /// let b = Array::compact_array(&array![1i32, 0, 3])?;
    /// let result = a.equal(b).to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[true, false, true]);
    ///
    /// // NaN != NaN per IEEE 754.
    /// let c = Array::compact_array(&array![f32::NAN, 1.0f32])?;
    /// let d = Array::compact_array(&array![f32::NAN, 1.0f32])?;
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
    /// let a = Array::compact_array(&array![1i32, 2, 3])?;
    /// let b = Array::compact_array(&array![1i32, 0, 3])?;
    /// let result = a.not_equal(b).to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[false, true, false]);
    ///
    /// // NaN != NaN is true per IEEE 754.
    /// let c = Array::compact_array(&array![f32::NAN, 1.0f32])?;
    /// let d = Array::compact_array(&array![f32::NAN, 2.0f32])?;
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
    /// let a = Array::compact_array(&array![3i32, 1, 2])?;
    /// let b = Array::compact_array(&array![1i32, 1, 3])?;
    /// let result = a.greater(b).to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[true, false, false]);
    ///
    /// // true > false for bool dtype.
    /// let c = Array::compact_array(&array![true, false, true])?;
    /// let d = Array::compact_array(&array![false, false, true])?;
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
    /// let a = Array::compact_array(&array![3i32, 1, 2])?;
    /// let b = Array::compact_array(&array![1i32, 1, 3])?;
    /// let result = a.greater_equal(b).to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[true, true, false]);
    ///
    /// // true >= false and false >= false hold.
    /// let c = Array::compact_array(&array![true, false, true])?;
    /// let d = Array::compact_array(&array![false, false, true])?;
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
    /// let a = Array::compact_array(&array![1i32, 1, 3])?;
    /// let b = Array::compact_array(&array![3i32, 1, 2])?;
    /// let result = a.less(b).to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[true, false, false]);
    ///
    /// // false < true for bool dtype.
    /// let c = Array::compact_array(&array![false, false, true])?;
    /// let d = Array::compact_array(&array![true, false, true])?;
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
    /// let a = Array::compact_array(&array![1i32, 1, 3])?;
    /// let b = Array::compact_array(&array![3i32, 1, 2])?;
    /// let result = a.less_equal(b).to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[true, true, false]);
    ///
    /// // false <= true and true <= true hold.
    /// let c = Array::compact_array(&array![false, false, true])?;
    /// let d = Array::compact_array(&array![true, false, true])?;
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
    /// let a = Array::compact_array(&array![1i32, 5, 3])?;
    /// let b = Array::compact_array(&array![4i32, 2, 3])?;
    /// let result = a.maximum(b).to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[4, 5, 3]);
    ///
    /// // NaN is propagated: if either operand is NaN the result is NaN.
    /// let c = Array::compact_array(&array![f32::NAN, 1.0f32])?;
    /// let d = Array::compact_array(&array![2.0f32, 3.0f32])?;
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
    /// let a = Array::compact_array(&array![1i32, 5, 3])?;
    /// let b = Array::compact_array(&array![4i32, 2, 3])?;
    /// let result = a.minimum(b).to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[1, 2, 3]);
    ///
    /// // NaN is propagated: if either operand is NaN the result is NaN.
    /// let c = Array::compact_array(&array![f32::NAN, 1.0f32])?;
    /// let d = Array::compact_array(&array![2.0f32, 3.0f32])?;
    /// let result = c.minimum(d).to_ndarray()?;
    /// assert!(result[[0]].is_nan());
    /// assert_eq!(result[[1]], 1.0);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Minimum,
    MinimumKernel,
    <crate::scalar::Minimum>::minimum(a, b),
);

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
