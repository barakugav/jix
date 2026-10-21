use std::sync::LazyLock;

use jix_core::scalar::{f16, Complex};
use jix_core::ArrayParams;
use pyo3::prelude::*;

use crate::ops::common::{define_op1_desc, OpDescriptor, Operand, Precision, Rank, Scalar};

/// Tests whether each element is `NaN` (not a number).
///
/// Supported dtypes: `f16`, `f32`, `f64`, `Complex<f32>`, `Complex<f64>`, and `bool` and integers,
/// which are always finite, so the result is all `False`.
///
/// Args:
///     array: Input array.
///
/// Returns:
///     A lazy [`jix.Array`][jix.Array] of dtype `bool` with the same shape as `array`. `True` where the
///         element is `NaN`, `False` otherwise.
///
/// Examples:
///     ```python
///     import jix
///     import numpy as np
///
///     a = jix.compact([float('nan'), 1.0, float('inf'), -1.0], dtype=np.float32)
///     result = jix.is_nan(a)
///     assert np.array_equal(result.numpy(), [True, False, False, False])
///     ```
#[pyo3::pyfunction]
pub fn is_nan<'py>(array: &Bound<'py, PyAny>) -> PyResult<crate::Array> {
    static DISPATCH_TABLE: LazyLock<OpDescriptor<1, ()>> = LazyLock::new(|| {
        OpDescriptor::new(
            "is_nan",
            define_op1_desc!(IsNan, [f16, f32, f64, Complex<f32>, Complex<f64>], None),
        )
    });
    classify(&DISPATCH_TABLE, array, false)
}
/// Tests whether each element is finite (not `+/-inf` and not `NaN`).
///
/// Supported dtypes: `f16`, `f32`, `f64`, `Complex<f32>`, `Complex<f64>`, and `bool` and integers,
/// which are always finite.
///
/// Args:
///     array: Input array.
///
/// Returns:
///     A lazy [`jix.Array`][jix.Array] of dtype `bool` with the same shape as `array`. `True` for finite
///         values, `False` for `+/-inf` and `NaN`.
///
/// Examples:
///     ```python
///     import jix
///     import numpy as np
///
///     a = jix.compact([1.0, float('nan'), float('inf'), float('-inf')], dtype=np.float32)
///     result = jix.is_finite(a)
///     assert np.array_equal(result.numpy(), [True, False, False, False])
///     ```
#[pyo3::pyfunction]
pub fn is_finite<'py>(array: &Bound<'py, PyAny>) -> PyResult<crate::Array> {
    static DISPATCH_TABLE: LazyLock<OpDescriptor<1, ()>> = LazyLock::new(|| {
        OpDescriptor::new(
            "is_finite",
            define_op1_desc!(IsFinite, [f16, f32, f64, Complex<f32>, Complex<f64>], None),
        )
    });
    classify(&DISPATCH_TABLE, array, true)
}
/// Tests whether each element is infinite (`+inf` or `-inf`).
///
/// Supported dtypes: `f16`, `f32`, `f64`, `Complex<f32>`, `Complex<f64>`, and `bool` and integers,
/// which are always finite, so the result is all `False`.
///
/// Args:
///     array: Input array.
///
/// Returns:
///     A lazy [`jix.Array`][jix.Array] of dtype `bool` with the same shape as `array`. `True` only for
///         `+inf` and `-inf`, `False` for finite values and `NaN`.
///
/// Examples:
///     ```python
///     import jix
///     import numpy as np
///
///     a = jix.compact([float('inf'), float('-inf'), float('nan'), 1.0], dtype=np.float32)
///     result = jix.is_infinite(a)
///     assert np.array_equal(result.numpy(), [True, True, False, False])
///     ```
#[pyo3::pyfunction]
pub fn is_infinite<'py>(array: &Bound<'py, PyAny>) -> PyResult<crate::Array> {
    static DISPATCH_TABLE: LazyLock<OpDescriptor<1, ()>> = LazyLock::new(|| {
        OpDescriptor::new(
            "is_infinite",
            define_op1_desc!(IsInfinite, [f16, f32, f64, Complex<f32>, Complex<f64>], None),
        )
    });
    classify(&DISPATCH_TABLE, array, false)
}

/// Applies `table` to `array`, or gives `int_value` for every element of a bool or integer array.
fn classify(
    table: &OpDescriptor<1, ()>,
    array: &Bound<'_, PyAny>,
    int_value: bool,
) -> PyResult<crate::Array> {
    let array = Operand::from_any(array)?;
    if !array
        .rank_precision()
        .is_some_and(|(rank, _)| matches!(rank, Rank::Bool | Rank::UInt | Rank::Int))
    {
        return Ok(crate::Array::from_core(table.dispatch1(array)?));
    }
    let shape = match &array {
        Operand::PyArray(array) => array.get().arr.shape().iter().copied().collect(),
        Operand::Array(array) => array.shape().iter().copied().collect(),
        Operand::Scalar { shape, .. } => shape.clone(),
    };
    let res = Operand::Scalar {
        value: Scalar::Bool(int_value),
        shape,
        precision: Some(Precision::P1),
        params: ArrayParams::default(),
    };
    Ok(crate::Array::from_core(res.into_array()?))
}
