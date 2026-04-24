use crate::array::Array;
#[allow(unused_imports)]
use crate::dtype::f16;
use crate::ops::common::define_array_op1_method;
use crate::ops::define_op1;
use crate::storage::ArrayStorage;

define_op1!(
    /// Tests whether each element is `NaN` (not a number).
    ///
    /// Supported dtypes: `f16`, `f32`, `f64`. Output dtype is `bool`.
    /// The output shape equals the input shape.
    ///
    /// Returns `true` if the element is `NaN`, `false` otherwise.
    /// Semantics follow [`f32::is_nan`].
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// let a = ndarray::array![f32::NAN, 1.0f32, f32::INFINITY, -1.0f32];
    /// let za = Array::from_ndarray(&a, ArrayParams::new())?;
    /// let result = za.is_nan().data().to_ndarray::<bool>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[true, false, false, false]);
    ///
    /// // Shape is preserved for 2-D input.
    /// let b = ndarray::array![[f32::NAN, 1.0f32], [2.0f32, f32::NAN]];
    /// let zb = Array::from_ndarray(&b, ArrayParams::new())?;
    /// let result = zb.is_nan().data().to_ndarray::<bool>()?;
    /// assert_eq!(result.shape(), &[2, 2]);
    /// assert_eq!(result[[0, 0]], true);
    /// assert_eq!(result[[1, 1]], true);
    /// # Ok::<(), zix::error::Error>(())
    /// ```
    IsNan,
    IsNanKernel,
    |a| a.is_nan(),
    [f16, f32, f64],
    output_type = bool
);
define_op1!(
    /// Tests whether each element is finite (not `±∞` and not `NaN`).
    ///
    /// Supported dtypes: `f16`, `f32`, `f64`. Output dtype is `bool`.
    /// The output shape equals the input shape.
    ///
    /// Returns `true` if the element is a finite number, `false` for `±∞` and `NaN`.
    /// Semantics follow [`f32::is_finite`].
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// let a = ndarray::array![1.0f32, f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
    /// let za = Array::from_ndarray(&a, ArrayParams::new())?;
    /// let result = za.is_finite().data().to_ndarray::<bool>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[true, false, false, false]);
    ///
    /// // Shape is preserved for 2-D input.
    /// let b = ndarray::array![[1.0f32, f32::INFINITY], [f32::NAN, -2.0f32]];
    /// let zb = Array::from_ndarray(&b, ArrayParams::new())?;
    /// let result = zb.is_finite().data().to_ndarray::<bool>()?;
    /// assert_eq!(result.shape(), &[2, 2]);
    /// assert_eq!(result[[0, 0]], true);
    /// assert_eq!(result[[0, 1]], false);
    /// # Ok::<(), zix::error::Error>(())
    /// ```
    IsFinite,
    IsFiniteKernel,
    |a| a.is_finite(),
    [f16, f32, f64],
    output_type = bool
);
define_op1!(
    /// Tests whether each element is infinite (`+∞` or `-∞`).
    ///
    /// Supported dtypes: `f16`, `f32`, `f64`. Output dtype is `bool`.
    /// The output shape equals the input shape.
    ///
    /// Returns `true` only for `+∞` and `-∞`; returns `false` for finite values and `NaN`.
    /// Semantics follow [`f32::is_infinite`].
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use zix::{Array, ArrayParams};
    /// let a = ndarray::array![f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 1.0f32];
    /// let za = Array::from_ndarray(&a, ArrayParams::new())?;
    /// let result = za.is_infinite().data().to_ndarray::<bool>()?;
    /// assert_eq!(result.as_slice().unwrap(), &[true, true, false, false]);
    ///
    /// // Shape is preserved for 2-D input.
    /// let b = ndarray::array![[f32::INFINITY, 0.0f32], [-1.0f32, f32::NEG_INFINITY]];
    /// let zb = Array::from_ndarray(&b, ArrayParams::new())?;
    /// let result = zb.is_infinite().data().to_ndarray::<bool>()?;
    /// assert_eq!(result.shape(), &[2, 2]);
    /// assert_eq!(result[[0, 0]], true);
    /// assert_eq!(result[[1, 1]], true);
    /// # Ok::<(), zix::error::Error>(())
    /// ```
    IsInfinite,
    IsInfiniteKernel,
    |a| a.is_infinite(),
    [f16, f32, f64],
    output_type = bool
);

impl<S> Array<S>
where
    S: ArrayStorage,
{
    define_array_op1_method!(is_nan: IsNan);
    define_array_op1_method!(is_finite: IsFinite);
    define_array_op1_method!(is_infinite: IsInfinite);
}
