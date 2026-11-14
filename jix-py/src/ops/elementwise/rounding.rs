use jix_core::scalar::f16;
use pyo3::prelude::*;

use crate::ops::common::{define_op1_table, Operand, Rank};
use crate::Array;

/// Rounds each element down to the nearest integer (towards -inf).
///
/// Supported dtypes: `f16`, `f32`, `f64`, and integers, which are returned as is.
///
/// Args:
///     array: Input array.
///
/// Returns:
///     A lazy [`jix.Array`][jix.Array] view with the same shape as `array`. No computation occurs until
///         the result is read.
///
/// Examples:
///     ```python
///     import jix
///     import numpy as np
///
///     a = jix.compact([1.1, 2.9, -1.1, -2.9], dtype=np.float32)
///     result = jix.floor(a)
///     assert np.array_equal(result.numpy(), [1.0, 2.0, -2.0, -3.0])
///     ```
#[pyo3::pyfunction]
pub fn floor<'py>(array: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Array>> {
    let py = array.py();
    let array = Operand::from_any(array)?;
    #[allow(clippy::collapsible_if)]
    if let Some((rank, _precision)) = array.rank_precision() {
        if matches!(rank, Rank::UInt | Rank::Int) {
            return array.into_py_array(py);
        }
    }

    let dispatch_table = define_op1_table!(floor, Floor, dispatch = { [f16, f32, f64], None });
    Bound::new(py, Array::from_core(dispatch_table.dispatch1(array)?))
}

/// Rounds each element up to the nearest integer (towards +inf).
///
/// Supported dtypes: `f16`, `f32`, `f64`, and integers, which are returned as is.
///
/// Args:
///     array: Input array.
///
/// Returns:
///     A lazy [`jix.Array`][jix.Array] view with the same shape as `array`. No computation occurs until
///         the result is read.
///
/// Examples:
///     ```python
///     import jix
///     import numpy as np
///
///     a = jix.compact([1.1, 2.0, -1.7, -0.1], dtype=np.float32)
///     result = jix.ceil(a)
///     assert np.array_equal(result.numpy(), [2.0, 2.0, -1.0, 0.0])
///     ```
#[pyo3::pyfunction]
pub fn ceil<'py>(array: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Array>> {
    let py = array.py();
    let array = Operand::from_any(array)?;
    #[allow(clippy::collapsible_if)]
    if let Some((rank, _precision)) = array.rank_precision() {
        if matches!(rank, Rank::UInt | Rank::Int) {
            return array.into_py_array(py);
        }
    }

    let dispatch_table = define_op1_table!(ceil, Ceil, dispatch = { [f16, f32, f64], None });
    Bound::new(py, Array::from_core(dispatch_table.dispatch1(array)?))
}

/// Rounds each element to the nearest integer.
///
/// Supported dtypes: `f16`, `f32`, `f64`, and integers, which are returned as is.
///
/// Ties round away from zero (`round(0.5) = 1.0`, `round(-0.5) = -1.0`), unlike the
/// round-half-to-even of Python's built-in `round()` and `numpy.round`.
///
/// Args:
///     array: Input array.
///
/// Returns:
///     A lazy [`jix.Array`][jix.Array] view with the same shape as `array`. No computation occurs until
///         the result is read.
///
/// Examples:
///     ```python
///     import jix
///     import numpy as np
///
///     a = jix.compact([1.4, 1.6, 2.5, -0.5], dtype=np.float32)
///     result = jix.round(a)
///     assert np.array_equal(result.numpy(), [1.0, 2.0, 3.0, -1.0])
///     ```
#[pyo3::pyfunction]
pub fn round<'py>(array: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Array>> {
    let py = array.py();
    let array = Operand::from_any(array)?;
    #[allow(clippy::collapsible_if)]
    if let Some((rank, _precision)) = array.rank_precision() {
        if matches!(rank, Rank::UInt | Rank::Int) {
            return array.into_py_array(py);
        }
    }

    let dispatch_table = define_op1_table!(round, Round, dispatch = { [f16, f32, f64], None });
    Bound::new(py, Array::from_core(dispatch_table.dispatch1(array)?))
}
