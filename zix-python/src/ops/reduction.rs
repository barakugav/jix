macro_rules! define_reduction_op {
    ($(#[$meta:meta])* $name:ident, $core_op:ident $(, extra_args = ($($extra_arg:ident : $extra_ty:ty = $extra_default:expr),+))?) => {
        $(#[$meta])*
        #[pyo3_stub_gen::derive::gen_stub_pyfunction]
        #[pyo3::pyfunction]
        #[pyo3(signature = (
            array,
            axes=None,
            keepdims=false,
            $($($extra_arg=$extra_default,)+)?
        ))]
        pub fn $name<'py>(
            array: &pyo3::Bound<'py, pyo3::PyAny>,
            axes: Option<Vec<i32>>,
            keepdims: bool,
            $($($extra_arg: $extra_ty),+)?
        ) -> pyo3::PyResult<crate::Array> {
            let array = crate::ops::as_array::any_to_core_array(array)?;
            let axes = match axes {
                Some(axes) => crate::util::normalize_axes(axes, array.ndim())?,
                None => (0..array.ndim()).collect(),
            };
            let res = zix_core::ops::$core_op::new(array, &axes, keepdims $($(, $extra_arg)+)?);
            let ret = <_ as crate::util::IntoPyResult<_>>::into_py_result(res)?;
            Ok(crate::Array::from_core_storage(ret))
        }
    };
    ($(#[$meta:meta])* $name:ident, $core_op:ident, single_axis = true) => {
        $(#[$meta])*
        #[pyo3_stub_gen::derive::gen_stub_pyfunction]
        #[pyo3::pyfunction]
        #[pyo3(signature = (
            array,
            axis=None,
            keepdims=false,
        ))]
        pub fn $name<'py>(
            array: &pyo3::Bound<'py, pyo3::PyAny>,
            axis: Option<i32>,
            keepdims: bool,
        ) -> pyo3::PyResult<crate::Array> {
            let array = crate::ops::as_array::any_to_core_array(array)?;
            let axis = match axis {
                Some(axis) => crate::util::normalize_axis(axis, array.ndim())?,
                None => {
                    if array.ndim() != 1 {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "axis must be specified for arrays with ndim != 1",
                        ));
                    }
                    0
                },
            };
            let res = zix_core::ops::$core_op::new(array, axis, keepdims);
            let ret = <_ as crate::util::IntoPyResult<_>>::into_py_result(res)?;
            Ok(crate::Array::from_core_storage(ret))
        }
    };
}
define_reduction_op!(
    /// Reduces one or more axes by taking the maximum element.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `bool`. Output dtype equals the input dtype.
    ///
    /// For **float** types, `NaN` values are ignored: the result is the maximum of all
    /// non-`NaN` values. If all elements are `NaN`, the result is `NaN`.
    ///
    /// `axes` accepts negative values (e.g. `-1` for the last axis). `axes=None` reduces
    /// over all axes, returning a scalar.
    ///
    /// The `array` argument may be anything that `zix.asarray()` accepts.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import zix
    /// import numpy as np
    ///
    /// a = zix.compact([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    /// # Reduce all axes → scalar
    /// assert zix.max(a).numpy()[()] == 6
    /// # Reduce axis 0 → shape [3]
    /// assert np.array_equal(zix.max(a, axes=[0]).numpy(), [4, 5, 6])
    /// # Reduce axis 0, keepdims=True → shape [1, 3]
    /// assert zix.max(a, axes=[0], keepdims=True).numpy().shape == (1, 3)
    /// ```
    max,
    Max
);
define_reduction_op!(
    /// Reduces one or more axes by taking the minimum element.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `bool`. Output dtype equals the input dtype.
    ///
    /// For **float** types, `NaN` values are ignored: the result is the minimum of all
    /// non-`NaN` values. If all elements are `NaN`, the result is `NaN`.
    ///
    /// `axes` accepts negative values (e.g. `-1` for the last axis). `axes=None` reduces
    /// over all axes, returning a scalar.
    ///
    /// The `array` argument may be anything that `zix.asarray()` accepts.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import zix
    /// import numpy as np
    ///
    /// a = zix.compact([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    /// # Reduce all axes → scalar
    /// assert zix.min(a).numpy()[()] == 1
    /// # Reduce axis 0 → shape [3]
    /// assert np.array_equal(zix.min(a, axes=[0]).numpy(), [1, 2, 3])
    /// ```
    min,
    Min
);
define_reduction_op!(
    /// Returns the index of the maximum element along a single axis.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `bool`. Output dtype is `u64`.
    ///
    /// If multiple elements share the maximum value, the index of the first occurrence is
    /// returned. For **float** types, `NaN` values are treated as less than any non-`NaN`
    /// value, so they are never selected unless all elements are `NaN`.
    ///
    /// `axis` accepts negative values (e.g. `-1` for the last axis). For 1-D arrays,
    /// `axis=None` is equivalent to `axis=0`.
    ///
    /// The `array` argument may be anything that `zix.asarray()` accepts.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import zix
    /// import numpy as np
    ///
    /// a = zix.compact([[1, 5, 3], [4, 2, 6]], dtype=np.int32)
    /// # Index of max along axis 1 (per row)
    /// assert np.array_equal(zix.argmax(a, axis=1).numpy(), [1, 2])
    /// # Index of max along axis 0 (per column)
    /// assert np.array_equal(zix.argmax(a, axis=0).numpy(), [1, 0, 1])
    /// ```
    argmax,
    ArgMax,
    single_axis = true
);
define_reduction_op!(
    /// Returns the index of the minimum element along a single axis.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `bool`. Output dtype is `u64`.
    ///
    /// If multiple elements share the minimum value, the index of the first occurrence is
    /// returned. For **float** types, `NaN` values are treated as greater than any non-`NaN`
    /// value, so they are never selected unless all elements are `NaN`.
    ///
    /// `axis` accepts negative values (e.g. `-1` for the last axis). For 1-D arrays,
    /// `axis=None` is equivalent to `axis=0`.
    ///
    /// The `array` argument may be anything that `zix.asarray()` accepts.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import zix
    /// import numpy as np
    ///
    /// a = zix.compact([[1, 5, 3], [4, 2, 6]], dtype=np.int32)
    /// # Index of min along axis 1 (per row)
    /// assert np.array_equal(zix.argmin(a, axis=1).numpy(), [0, 1])
    /// ```
    argmin,
    ArgMin,
    single_axis = true
);
define_reduction_op!(
    /// Reduces one or more axes by summing all elements.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `Complex<f32>`, `Complex<f64>`. Output dtype equals the input
    /// dtype.
    ///
    /// For **integer** types, the result wraps on overflow (two's complement). For
    /// large sums, consider casting to a wider type first.
    ///
    /// `axes` accepts negative values (e.g. `-1` for the last axis). `axes=None` reduces
    /// over all axes, returning a scalar.
    ///
    /// The `array` argument may be anything that `zix.asarray()` accepts.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import zix
    /// import numpy as np
    ///
    /// a = zix.compact([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    /// assert zix.sum(a).numpy()[()] == 21
    /// assert np.array_equal(zix.sum(a, axes=[0]).numpy(), [5, 7, 9])
    /// ```
    sum,
    Sum
);
define_reduction_op!(
    /// Reduces one or more axes by multiplying all elements.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `Complex<f32>`, `Complex<f64>`. Output dtype equals the input
    /// dtype.
    ///
    /// For **integer** types, the result wraps on overflow. For large products, consider
    /// casting to a wider type first.
    ///
    /// `axes` accepts negative values (e.g. `-1` for the last axis). `axes=None` reduces
    /// over all axes, returning a scalar.
    ///
    /// The `array` argument may be anything that `zix.asarray()` accepts.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import zix
    /// import numpy as np
    ///
    /// a = zix.compact([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    /// assert zix.product(a).numpy()[()] == 720
    /// assert np.array_equal(zix.product(a, axes=[0]).numpy(), [4, 10, 18])
    /// ```
    product,
    Product
);
define_reduction_op!(
    /// Computes the arithmetic mean along one or more axes.
    ///
    /// Supported dtypes: `f32`, `f64`, `Complex<f32>`, `Complex<f64>`. Output dtype equals
    /// the input dtype.
    ///
    /// `axes` accepts negative values (e.g. `-1` for the last axis). `axes=None` reduces
    /// over all axes, returning a scalar.
    ///
    /// This function deviates from numpy in that only float and complex types are supported.
    /// For integer inputs, cast to `f64` first with `zix.astype(array, 'float64')`.
    ///
    /// The `array` argument may be anything that `zix.asarray()` accepts.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import zix
    /// import numpy as np
    ///
    /// a = zix.compact([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    /// assert zix.mean(a).numpy()[()] == 3.5
    /// assert np.allclose(zix.mean(a, axes=[0]).numpy(), [2.5, 3.5, 4.5])
    /// ```
    mean,
    Mean
);
define_reduction_op!(
    /// Computes the variance along one or more axes.
    ///
    /// Supported dtypes: `f32`, `f64`. Output dtype equals the input dtype.
    ///
    /// `ddof` (delta degrees of freedom) defaults to `0.0` (population variance). Use
    /// `ddof=1.0` for the sample (Bessel-corrected) variance.
    ///
    /// `axes` accepts negative values (e.g. `-1` for the last axis). `axes=None` reduces
    /// over all axes, returning a scalar.
    ///
    /// The `array` argument may be anything that `zix.asarray()` accepts.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import zix
    /// import numpy as np
    ///
    /// a = zix.compact([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], dtype=np.float32)
    /// assert abs(zix.var(a).numpy()[()] - 4.0) < 1e-5   # population variance
    /// assert abs(zix.var(a, ddof=1.0).numpy()[()] - np.var(a.numpy(), ddof=1)) < 1e-3
    /// ```
    var,
    Variance,
    extra_args = (ddof: f64 = 0.0)
);
define_reduction_op!(
    /// Computes the standard deviation along one or more axes.
    ///
    /// Supported dtypes: `f32`, `f64`. Output dtype equals the input dtype.
    ///
    /// `ddof` (delta degrees of freedom) defaults to `0.0` (population standard deviation).
    /// Use `ddof=1.0` for the sample (Bessel-corrected) standard deviation.
    ///
    /// `axes` accepts negative values (e.g. `-1` for the last axis). `axes=None` reduces
    /// over all axes, returning a scalar.
    ///
    /// The `array` argument may be anything that `zix.asarray()` accepts.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import zix
    /// import numpy as np
    ///
    /// a = zix.compact([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], dtype=np.float32)
    /// assert abs(zix.std(a).numpy()[()] - 2.0) < 1e-5   # population std dev
    /// ```
    std,
    StandardDeviation,
    extra_args = (ddof: f64 = 0.0)
);
define_reduction_op!(
    /// Reduces one or more axes with logical AND: returns `True` if all elements are truthy.
    ///
    /// Supported dtypes: all integer types, `f16`, `f32`, `f64`, `Complex<f32>`,
    /// `Complex<f64>`, and `bool`. Output dtype is `bool`.
    ///
    /// Each element is first cast to `bool` (zero → `False`, non-zero → `True`), then the
    /// AND reduction is applied. Returns `True` only if every element in the reduced
    /// dimensions is truthy; returns `True` for empty reductions.
    ///
    /// `axes` accepts negative values (e.g. `-1` for the last axis). `axes=None` reduces
    /// over all axes, returning a scalar.
    ///
    /// The `array` argument may be anything that `zix.asarray()` accepts.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import zix
    /// import numpy as np
    ///
    /// a = zix.compact([[True, True], [True, False]])
    /// assert zix.all(a).numpy()[()] == False
    /// assert np.array_equal(zix.all(a, axes=[1]).numpy(), [True, False])
    /// ```
    all,
    All
);
define_reduction_op!(
    /// Reduces one or more axes with logical OR: returns `True` if any element is truthy.
    ///
    /// Supported dtypes: all integer types, `f16`, `f32`, `f64`, `Complex<f32>`,
    /// `Complex<f64>`, and `bool`. Output dtype is `bool`.
    ///
    /// Each element is first cast to `bool` (zero → `False`, non-zero → `True`), then the
    /// OR reduction is applied. Returns `True` if at least one element in the reduced
    /// dimensions is truthy; returns `False` for empty reductions.
    ///
    /// `axes` accepts negative values (e.g. `-1` for the last axis). `axes=None` reduces
    /// over all axes, returning a scalar.
    ///
    /// The `array` argument may be anything that `zix.asarray()` accepts.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```python,ignore
    /// import zix
    /// import numpy as np
    ///
    /// a = zix.compact([[False, False], [False, True]])
    /// assert zix.any(a).numpy()[()] == True
    /// assert np.array_equal(zix.any(a, axes=[1]).numpy(), [False, True])
    /// ```
    any,
    Any
);
