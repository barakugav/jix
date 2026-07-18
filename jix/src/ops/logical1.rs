use crate::array::Array;
use crate::ops::common::define_array_op1_method;
use crate::ops::define_op1;
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
    /// let a = Array::compact_ndarray(&array![f32::NAN, 1.0f32, f32::INFINITY, -1.0f32])?;
    /// let result = a.is_nan().to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[true, false, false, false]);
    ///
    /// // Shape is preserved for 2-D input.
    /// let b = Array::compact_ndarray(&array![[f32::NAN, 1.0f32], [2.0f32, f32::NAN]])?;
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
    /// let a = Array::compact_ndarray(&array![1.0f32, f32::NAN, f32::INFINITY, f32::NEG_INFINITY])?;
    /// let result = a.is_finite().to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[true, false, false, false]);
    ///
    /// // Shape is preserved for 2-D input.
    /// let b = Array::compact_ndarray(&array![[1.0f32, f32::INFINITY], [f32::NAN, -2.0f32]])?;
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
    /// let a = Array::compact_ndarray(&array![f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 1.0f32])?;
    /// let result = a.is_infinite().to_ndarray()?;
    /// assert_eq!(result.as_slice().unwrap(), &[true, true, false, false]);
    ///
    /// // Shape is preserved for 2-D input.
    /// let b = Array::compact_ndarray(&array![[f32::INFINITY, 0.0f32], [-1.0f32, f32::NEG_INFINITY]])?;
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

    #[test]
    fn is_finite_concrete() {
        use crate::Array;
        // Edge inputs: NaN, +inf, -inf, 0.0, 1.0 - only 0.0 and 1.0 are finite.
        let nd = ndarray::array![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, 1.0];
        let za = Array::compact_ndarray(&nd).unwrap();
        let expected = nd.mapv(|a: f32| a.is_finite());
        crate::util::assert_array_matches(&za.as_ref().is_finite(), &expected);

        let nd64 = ndarray::array![f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, 1.0];
        let za64 = Array::compact_ndarray(&nd64).unwrap();
        let expected64 = nd64.mapv(|a: f64| a.is_finite());
        crate::util::assert_array_matches(&za64.as_ref().is_finite(), &expected64);

        #[cfg(feature = "half")]
        {
            let ndh = ndarray::array![
                f16::NAN,
                f16::INFINITY,
                f16::NEG_INFINITY,
                f16::from_f32(0.0),
                f16::from_f32(1.0)
            ];
            let zah = Array::compact_ndarray(&ndh).unwrap();
            let expectedh = ndh.mapv(|a: f16| a.is_finite());
            crate::util::assert_array_matches(&zah.as_ref().is_finite(), &expectedh);
        }
    }

    #[test]
    fn is_infinite_concrete() {
        use crate::Array;
        // Edge inputs: NaN, +inf, -inf, 0.0, 1.0 - only +inf/-inf are infinite.
        let nd = ndarray::array![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, 1.0];
        let za = Array::compact_ndarray(&nd).unwrap();
        let expected = nd.mapv(|a: f32| a.is_infinite());
        crate::util::assert_array_matches(&za.as_ref().is_infinite(), &expected);

        let nd64 = ndarray::array![f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, 1.0];
        let za64 = Array::compact_ndarray(&nd64).unwrap();
        let expected64 = nd64.mapv(|a: f64| a.is_infinite());
        crate::util::assert_array_matches(&za64.as_ref().is_infinite(), &expected64);

        #[cfg(feature = "half")]
        {
            let ndh = ndarray::array![
                f16::NAN,
                f16::INFINITY,
                f16::NEG_INFINITY,
                f16::from_f32(0.0),
                f16::from_f32(1.0)
            ];
            let zah = Array::compact_ndarray(&ndh).unwrap();
            let expectedh = ndh.mapv(|a: f16| a.is_infinite());
            crate::util::assert_array_matches(&zah.as_ref().is_infinite(), &expectedh);
        }
    }
}
