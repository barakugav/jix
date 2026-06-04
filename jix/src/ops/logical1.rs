use crate::array::Array;
use crate::ops::common::define_array_op1_method;
use crate::ops::define_op1;
#[allow(unused_imports)]
use crate::scalar::f16;
use crate::ArrayStorage;

define_op1!(
    /// Tests whether each element is `NaN` (not a number).
    ///
    /// Output dtype is `bool`.
    ///
    /// Returns `true` if the element is `NaN`, `false` otherwise.
    /// Semantics follow [`f32::is_nan`].
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::is_nan()`](crate::Array::is_nan).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_array(&array![f32::NAN, 1.0f32, f32::INFINITY, -1.0f32])?;
    /// let result = a.is_nan().to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[true, false, false, false]);
    ///
    /// // Shape is preserved for 2-D input.
    /// let b = Array::compact_array(&array![[f32::NAN, 1.0f32], [2.0f32, f32::NAN]])?;
    /// let result = b.is_nan().to_ndarray()?;
    /// assert_eq!(result.shape(), &[2, 2]);
    /// assert_eq!(result[[0, 0]], true);
    /// assert_eq!(result[[1, 1]], true);
    /// # Ok::<(), jix::Error>(())
    /// ```
    IsNan,
    IsNanKernel,
    <num_traits::Float>::is_nan,
    type Output = bool,
);
define_op1!(
    /// Tests whether each element is finite (not `+/-inf` and not `NaN`).
    ///
    /// Output dtype is `bool`.
    ///
    /// Returns `true` if the element is a finite number, `false` for `+/-inf` and `NaN`.
    /// Semantics follow [`f32::is_finite`].
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::is_finite()`](crate::Array::is_finite).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_array(&array![1.0f32, f32::NAN, f32::INFINITY, f32::NEG_INFINITY])?;
    /// let result = a.is_finite().to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[true, false, false, false]);
    ///
    /// // Shape is preserved for 2-D input.
    /// let b = Array::compact_array(&array![[1.0f32, f32::INFINITY], [f32::NAN, -2.0f32]])?;
    /// let result = b.is_finite().to_ndarray()?;
    /// assert_eq!(result.shape(), &[2, 2]);
    /// assert_eq!(result[[0, 0]], true);
    /// assert_eq!(result[[0, 1]], false);
    /// # Ok::<(), jix::Error>(())
    /// ```
    IsFinite,
    IsFiniteKernel,
    <num_traits::Float>::is_finite,
    type Output = bool,
);
define_op1!(
    /// Tests whether each element is infinite (`+inf` or `-inf`).
    ///
    /// Output dtype is `bool`.
    ///
    /// Returns `true` only for `+inf` and `-inf`; returns `false` for finite values and `NaN`.
    /// Semantics follow [`f32::is_infinite`].
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::is_infinite()`](crate::Array::is_infinite).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_array(&array![f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 1.0f32])?;
    /// let result = a.is_infinite().to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[true, true, false, false]);
    ///
    /// // Shape is preserved for 2-D input.
    /// let b = Array::compact_array(&array![[f32::INFINITY, 0.0f32], [-1.0f32, f32::NEG_INFINITY]])?;
    /// let result = b.is_infinite().to_ndarray()?;
    /// assert_eq!(result.shape(), &[2, 2]);
    /// assert_eq!(result[[0, 0]], true);
    /// assert_eq!(result[[1, 1]], true);
    /// # Ok::<(), jix::Error>(())
    /// ```
    IsInfinite,
    IsInfiniteKernel,
    <num_traits::Float>::is_infinite,
    type Output = bool,
);

impl<S> Array<S>
where
    S: ArrayStorage,
{
    define_array_op1_method!(is_nan: IsNan, num_traits::Float, fixed_output_type = true);
    define_array_op1_method!(is_finite: IsFinite, num_traits::Float, fixed_output_type = true);
    define_array_op1_method!(is_infinite: IsInfinite, num_traits::Float, fixed_output_type = true);
}

#[cfg(test)]
mod tests {
    use crate::ops::op1::tests::test_op1;
    #[cfg(feature = "half")]
    use crate::scalar::f16;

    // full domain is valid; output is bool so NaN inputs produce well-defined bool results
    test_op1!(
        is_nan,
        |a| a.is_nan(),
        [f32, f64],
        maybe_non_finite_strategy,
        #[cfg(feature = "half")]
        [f16]
    );
    test_op1!(
        is_finite,
        |a| a.is_finite(),
        [f32, f64],
        maybe_non_finite_strategy,
        #[cfg(feature = "half")]
        [f16]
    );
    test_op1!(
        is_infinite,
        |a| a.is_infinite(),
        [f32, f64],
        maybe_non_finite_strategy,
        #[cfg(feature = "half")]
        [f16]
    );
}
