use jix_core::scalar::{f16, Complex};
use jix_core::ArrayAny;
use pyo3::prelude::*;

fn keepdims_after_reduction(
    array: ArrayAny,
    original_reduced_axes: &[usize],
) -> PyResult<ArrayAny> {
    // keepdims=true: re-insert singleton axes via insert_axis.
    // insert_axis uses gap indices in the space of the array it receives, so we must
    // re-map: original sorted axis a_i -> result-space gap (a_i - i).
    let mut axes = original_reduced_axes.to_vec();
    axes.sort_unstable();
    let mapped_axes = axes
        .iter()
        .enumerate()
        .map(|(i, &ax)| ax - i)
        .collect::<Vec<_>>();
    let res = jix_core::ops::InsertAxis::new_array(array, &mapped_axes);
    let ret = <_ as crate::util::IntoPyResult<_>>::into_py_result(res)?;
    Ok(ret.into_any())
}

fn keepdim_after_reduction(array: ArrayAny, original_reduced_axis: usize) -> PyResult<ArrayAny> {
    // keepdims=true: for a single-axis reduction, the result-space gap equals the
    // original axis index (only one axis removed, shift = 0).
    let res = jix_core::ops::InsertAxis::new_array(array, &[original_reduced_axis]);
    let ret = <_ as crate::util::IntoPyResult<_>>::into_py_result(res)?;
    Ok(ret.into_any())
}

macro_rules! define_reduction_op {
    (
        $(#[$meta:meta])* $name:ident,
        $core_op:ident,
        dispatch = { $($dispatch:tt)* }
        $(, extra_args = ($($extra_arg:ident : $extra_ty:ty = $extra_default:expr),+))?
    ) => {
        $(#[$meta])*
        #[pyo3_stub_gen::derive::gen_stub_pyfunction]
        #[pyo3::pyfunction]
        #[pyo3(signature = (
            array,
            axis=None,
            *,
            keepdims=false,
            $($($extra_arg=$extra_default,)+)?
        ))]
        pub fn $name<'py>(
            array: &pyo3::Bound<'py, pyo3::PyAny>,
            axis: Option<crate::util::ItemOrSequence<i32>>,
            keepdims: bool,
            $($($extra_arg: $extra_ty),+)?
        ) -> pyo3::PyResult<crate::Array> {
            struct DispatchArgs {
                axes: Vec<usize>,
                $($($extra_arg: $extra_ty),*)?
            }
            static DISPATCH_TABLE: ::std::sync::LazyLock<crate::ops::common::OpDescriptor<1, DispatchArgs>> = ::std::sync::LazyLock::new(|| {
                crate::ops::common::OpDescriptor::new(
                    stringify!($name),
                    crate::ops::common::define_op1_desc!(
                        $core_op,
                        extra_args = DispatchArgs { axes $($(, $extra_arg)+)? },
                        $($dispatch)*
                    ),
                )
            });

            let array = crate::ops::as_array::any_to_core_array(array)?;
            let axes = crate::util::normalize_axes_optional(axis.map(|a| a.into_vec()), array.ndim())?;
            let mut res = DISPATCH_TABLE.dispatch_args(
                [crate::ops::common::Operand::Array(array)],
                DispatchArgs { axes: axes.clone(), $($($extra_arg),+)? }
            )?;
            if keepdims {
                res = keepdims_after_reduction(res, &axes)?;
            }
            Ok(crate::Array::from_core(res))
        }
    };

    (
        $(#[$meta:meta])* $name:ident,
        $core_op:ident,
        single_axis = true,
        dispatch = { $($dispatch:tt)* }
    ) => {
        $(#[$meta])*
        #[pyo3_stub_gen::derive::gen_stub_pyfunction]
        #[pyo3::pyfunction]
        #[pyo3(signature = (
            array,
            axis=None,
            *,
            keepdims=false,
        ))]
        pub fn $name<'py>(
            array: &pyo3::Bound<'py, pyo3::PyAny>,
            axis: Option<i32>,
            keepdims: bool,
        ) -> pyo3::PyResult<crate::Array> {
            struct DispatchArgs {
                axis: usize,
            }
            static DISPATCH_TABLE: ::std::sync::LazyLock<crate::ops::common::OpDescriptor<1, DispatchArgs>> = ::std::sync::LazyLock::new(|| {
                crate::ops::common::OpDescriptor::new(
                    stringify!($name),
                    crate::ops::common::define_op1_desc!(
                        $core_op,
                        extra_args = DispatchArgs { axis },
                        $($dispatch)*
                    ),
                )
            });

            let array = crate::ops::as_array::any_to_core_array(array)?;
            let axis = crate::util::normalize_axis_optional(axis, array.ndim())?;
            let mut res = DISPATCH_TABLE.dispatch_args(
                [crate::ops::common::Operand::Array(array)],
                DispatchArgs { axis }
            )?;
            if keepdims {
                res = keepdim_after_reduction(res, axis)?;
            }
            Ok(crate::Array::from_core(res))
        }
    };
}

define_reduction_op!(
    /// Reduces one or more axes by taking the maximum element.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `bool`.
    ///
    /// For **float** types, `NaN` values are ignored: the result is the maximum of all
    /// non-`NaN` values. If all elements are `NaN`, the result is `NaN`.
    ///
    /// Args:
    ///     array: May be anything that [`jix.asarray()`][jix.asarray] accepts.
    ///     axis: Axis or axes to reduce. Accepts negative values (e.g. `-1` for the last axis).
    ///         When `None`, reduces over all axes, returning a scalar.
    ///     keepdims: If `True`, reduced axes are kept as length-1 dimensions. Default `False`.
    ///
    /// Returns:
    ///     A [`jix.Array`][jix.Array] with the specified axes reduced. When `keepdims=True`, reduced axes
    ///     are kept with size 1.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    ///     # Reduce all axes -> scalar
    ///     assert jix.max(a).numpy()[()] == 6
    ///     # Reduce axis 0 -> shape [3]
    ///     assert np.array_equal(jix.max(a, axis=0).numpy(), [4, 5, 6])
    ///     # Reduce axis 0, keepdims=True -> shape [1, 3]
    ///     assert jix.max(a, axis=0, keepdims=True).numpy().shape == (1, 3)
    ///     ```
    max,
    Max,
    dispatch = {
        [bool, u8, i8, u16, i16, u32, i32, u64, i64, f16, f32, f64],
        None
    }
);
define_reduction_op!(
    /// Reduces one or more axes by taking the minimum element.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `bool`.
    ///
    /// For **float** types, `NaN` values are ignored: the result is the minimum of all
    /// non-`NaN` values. If all elements are `NaN`, the result is `NaN`.
    ///
    /// Args:
    ///     array: May be anything that [`jix.asarray()`][jix.asarray] accepts.
    ///     axis: Axis or axes to reduce. Accepts negative values (e.g. `-1` for the last axis).
    ///         When `None`, reduces over all axes, returning a scalar.
    ///     keepdims: If `True`, reduced axes are kept as length-1 dimensions. Default `False`.
    ///
    /// Returns:
    ///     A [`jix.Array`][jix.Array] with the specified axes reduced. When `keepdims=True`, reduced axes
    ///     are kept with size 1.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    ///     # Reduce all axes -> scalar
    ///     assert jix.min(a).numpy()[()] == 1
    ///     # Reduce axis 0 -> shape [3]
    ///     assert np.array_equal(jix.min(a, axis=0).numpy(), [1, 2, 3])
    ///     ```
    min,
    Min,
    dispatch = {
        [bool, u8, i8, u16, i16, u32, i32, u64, i64, f16, f32, f64],
        None
    }
);
define_reduction_op!(
    /// Returns the index of the maximum element along a single axis.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `bool`.
    ///
    /// If multiple elements share the maximum value, the index of the first occurrence is
    /// returned. For **float** types, `NaN` values are treated as less than any non-`NaN`
    /// value, so they are never selected unless all elements are `NaN`.
    ///
    /// Args:
    ///     array: May be anything that [`jix.asarray()`][jix.asarray] accepts.
    ///     axis: Single axis to reduce along. Accepts negative values. For 1-D arrays,
    ///         `None` is equivalent to `axis=0`.
    ///     keepdims: If `True`, the reduced axis is kept as a length-1 dimension. Default `False`.
    ///
    /// Returns:
    ///     A [`jix.Array`][jix.Array] of dtype `u64` with the index of the max/min element. When `axis` is
    ///     `None`, returns a scalar. When `keepdims=True`, the reduced axis is kept with size 1.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([[1, 5, 3], [4, 2, 6]], dtype=np.int32)
    ///     # Index of max along axis 1 (per row)
    ///     assert np.array_equal(jix.argmax(a, axis=1).numpy(), [1, 2])
    ///     # Index of max along axis 0 (per column)
    ///     assert np.array_equal(jix.argmax(a, axis=0).numpy(), [1, 0, 1])
    ///     ```
    argmax,
    ArgMax,
    single_axis = true,
    dispatch = {
        [bool, u8, i8, u16, i16, i32, u32, i64, u64, f16, f32, f64],
        None
    }
);
define_reduction_op!(
    /// Returns the index of the minimum element along a single axis.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `bool`.
    ///
    /// If multiple elements share the minimum value, the index of the first occurrence is
    /// returned. For **float** types, `NaN` values are treated as greater than any non-`NaN`
    /// value, so they are never selected unless all elements are `NaN`.
    ///
    /// Args:
    ///     array: May be anything that [`jix.asarray()`][jix.asarray] accepts.
    ///     axis: Single axis to reduce along. Accepts negative values. For 1-D arrays,
    ///         `None` is equivalent to `axis=0`.
    ///     keepdims: If `True`, the reduced axis is kept as a length-1 dimension. Default `False`.
    ///
    /// Returns:
    ///     A [`jix.Array`][jix.Array] of dtype `u64` with the index of the max/min element. When `axis` is
    ///     `None`, returns a scalar. When `keepdims=True`, the reduced axis is kept with size 1.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([[1, 5, 3], [4, 2, 6]], dtype=np.int32)
    ///     # Index of min along axis 1 (per row)
    ///     assert np.array_equal(jix.argmin(a, axis=1).numpy(), [0, 1])
    ///     ```
    argmin,
    ArgMin,
    single_axis = true,
    dispatch = {
        [bool, u8, i8, u16, i16, i32, u32, i64, u64, f16, f32, f64],
        None
    }
);
define_reduction_op!(
    /// Reduces one or more axes by summing all elements.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `Complex<f32>`, `Complex<f64>`.
    ///
    /// For **integer** types, the result wraps on overflow (two's complement). For
    /// large sums, consider casting to a wider type first.
    ///
    /// Args:
    ///     array: May be anything that [`jix.asarray()`][jix.asarray] accepts.
    ///     axis: Axis or axes to reduce. Accepts negative values (e.g. `-1` for the last axis).
    ///         When `None`, reduces over all axes, returning a scalar.
    ///     keepdims: If `True`, reduced axes are kept as length-1 dimensions. Default `False`.
    ///
    /// Returns:
    ///     A [`jix.Array`][jix.Array] with the specified axes reduced. When `keepdims=True`, reduced axes
    ///     are kept with size 1.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    ///     assert jix.sum(a).numpy()[()] == 21
    ///     assert np.array_equal(jix.sum(a, axis=0).numpy(), [5, 7, 9])
    ///     ```
    sum,
    Sum,
    dispatch = {
        [bool, u8, i8, u16, i16, u32, i32, u64, i64, f16, f32, f64, Complex<f32>, Complex<f64>],
        None
    }
);
define_reduction_op!(
    /// Reduces one or more axes by multiplying all elements.
    ///
    /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
    /// `f16`, `f32`, `f64`, `Complex<f32>`, `Complex<f64>`.
    ///
    /// For **integer** types, the result wraps on overflow. For large products, consider
    /// casting to a wider type first.
    ///
    /// Args:
    ///     array: May be anything that [`jix.asarray()`][jix.asarray] accepts.
    ///     axis: Axis or axes to reduce. Accepts negative values (e.g. `-1` for the last axis).
    ///         When `None`, reduces over all axes, returning a scalar.
    ///     keepdims: If `True`, reduced axes are kept as length-1 dimensions. Default `False`.
    ///
    /// Returns:
    ///     A [`jix.Array`][jix.Array] with the specified axes reduced. When `keepdims=True`, reduced axes
    ///     are kept with size 1.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    ///     assert jix.product(a).numpy()[()] == 720
    ///     assert np.array_equal(jix.product(a, axis=0).numpy(), [4, 10, 18])
    ///     ```
    product,
    Product,
    dispatch = {
        [u8, i8, u16, i16, u32, i32, u64, i64, f16, f32, f64, Complex<f32>, Complex<f64>],
        None
    }
);
define_reduction_op!(
    /// Computes the arithmetic mean along one or more axes.
    ///
    /// Supported dtypes: all integers, floats, complex types, and `bool`.
    ///
    /// This function deviates from numpy in that only float and complex types are supported.
    /// For integer inputs, cast to `f64` first with [`jix.astype(array, 'float64')`][jix.astype].
    ///
    /// Args:
    ///     array: May be anything that [`jix.asarray()`][jix.asarray] accepts.
    ///     axis: Axis or axes to reduce. Accepts negative values (e.g. `-1` for the last axis).
    ///         When `None`, reduces over all axes, returning a scalar.
    ///     keepdims: If `True`, reduced axes are kept as length-1 dimensions. Default `False`.
    ///
    /// Returns:
    ///     A [`jix.Array`][jix.Array] with the specified axes reduced. When `keepdims=True`, reduced axes
    ///     are kept with size 1.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    ///     assert jix.mean(a).numpy()[()] == 3.5
    ///     assert np.allclose(jix.mean(a, axis=0).numpy(), [2.5, 3.5, 4.5])
    ///     ```
    mean,
    Mean,
    dispatch = {
        [bool, u8, i8, u16, i16, u32, i32, u64, i64, f16, f32, f64, Complex<f32>, Complex<f64>],
        None
    }
);
define_reduction_op!(
    /// Computes the variance along one or more axes.
    ///
    /// Supported dtypes: all integers, floats, complex types, and `bool`.
    ///
    /// Args:
    ///     array: May be anything that [`jix.asarray()`][jix.asarray] accepts.
    ///     axis: Axis or axes to reduce. Accepts negative values. When `None`, reduces over
    ///         all axes, returning a scalar.
    ///     keepdims: If `True`, reduced axes are kept as length-1 dimensions. Default `False`.
    ///     ddof: Delta degrees of freedom. Default `0.0` (population). Use `1.0` for sample
    ///         (Bessel-corrected).
    ///
    /// Returns:
    ///     A [`jix.Array`][jix.Array] with the specified axes reduced. When `keepdims=True`, reduced axes
    ///     are kept with size 1.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], dtype=np.float32)
    ///     assert abs(jix.var(a).numpy()[()] - 4.0) < 1e-5   # population variance
    ///     assert abs(jix.var(a, ddof=1.0).numpy()[()] - np.var(a.numpy(), ddof=1)) < 1e-3
    ///     ```
    var,
    Variance,
    dispatch = {
        [bool, u8, i8, u16, i16, u32, i32, u64, i64, f16, f32, f64, Complex<f32>, Complex<f64>],
        None
    },
    extra_args = (ddof: f64 = 0.0)
);
define_reduction_op!(
    /// Computes the standard deviation along one or more axes.
    ///
    /// Supported dtypes: all integers, floats, complex types, and `bool`.
    ///
    /// Args:
    ///     array: May be anything that [`jix.asarray()`][jix.asarray] accepts.
    ///     axis: Axis or axes to reduce. Accepts negative values. When `None`, reduces over
    ///         all axes, returning a scalar.
    ///     keepdims: If `True`, reduced axes are kept as length-1 dimensions. Default `False`.
    ///     ddof: Delta degrees of freedom. Default `0.0` (population). Use `1.0` for sample
    ///         (Bessel-corrected).
    ///
    /// Returns:
    ///     A [`jix.Array`][jix.Array] with the specified axes reduced. When `keepdims=True`, reduced axes
    ///     are kept with size 1.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], dtype=np.float32)
    ///     assert abs(jix.std(a).numpy()[()] - 2.0) < 1e-5   # population std dev
    ///     ```
    std,
    StandardDeviation,
    dispatch = {
        [bool, u8, i8, u16, i16, u32, i32, u64, i64, f16, f32, f64, Complex<f32>, Complex<f64>],
        None
    },
    extra_args = (ddof: f64 = 0.0)
);
define_reduction_op!(
    /// Reduces one or more axes with logical AND: returns `True` if all elements are truthy.
    ///
    /// Supported dtypes: `bool`.
    ///
    /// Each element is first cast to `bool` (zero -> `False`, non-zero -> `True`), then the
    /// AND reduction is applied. Returns `True` only if every element in the reduced
    /// dimensions is truthy; returns `True` for empty reductions.
    ///
    /// Args:
    ///     array: May be anything that [`jix.asarray()`][jix.asarray] accepts.
    ///     axis: Axis or axes to reduce. Accepts negative values (e.g. `-1` for the last axis).
    ///         When `None`, reduces over all axes, returning a scalar.
    ///     keepdims: If `True`, reduced axes are kept as length-1 dimensions. Default `False`.
    ///
    /// Returns:
    ///     A [`jix.Array`][jix.Array] with the specified axes reduced. When `keepdims=True`, reduced axes
    ///     are kept with size 1.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([[True, True], [True, False]])
    ///     assert jix.all(a).numpy()[()] == False
    ///     assert np.array_equal(jix.all(a, axis=1).numpy(), [True, False])
    ///     ```
    all,
    All,
    dispatch = {
        [bool],
        Unsafe
    }
);
define_reduction_op!(
    /// Reduces one or more axes with logical OR: returns `True` if any element is truthy.
    ///
    /// Supported dtypes: `bool`.
    ///
    /// Each element is first cast to `bool` (zero -> `False`, non-zero -> `True`), then the
    /// OR reduction is applied. Returns `True` if at least one element in the reduced
    /// dimensions is truthy; returns `False` for empty reductions.
    ///
    /// Args:
    ///     array: May be anything that [`jix.asarray()`][jix.asarray] accepts.
    ///     axis: Axis or axes to reduce. Accepts negative values (e.g. `-1` for the last axis).
    ///         When `None`, reduces over all axes, returning a scalar.
    ///     keepdims: If `True`, reduced axes are kept as length-1 dimensions. Default `False`.
    ///
    /// Returns:
    ///     A [`jix.Array`][jix.Array] with the specified axes reduced. When `keepdims=True`, reduced axes
    ///     are kept with size 1.
    ///
    /// Examples:
    ///     ```python
    ///     import jix
    ///     import numpy as np
    ///
    ///     a = jix.compact([[False, False], [False, True]])
    ///     assert jix.any(a).numpy()[()] == True
    ///     assert np.array_equal(jix.any(a, axis=1).numpy(), [False, True])
    ///     ```
    any,
    Any,
    dispatch = {
        [bool],
        Unsafe
    }
);
