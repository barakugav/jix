use crate::ops::any_to_core_array;
use crate::util::IntoPyResult;
use crate::{asarray, Array};

/// Selects elements element-wise from `x` or `y` based on `condition`.
///
/// For each index `i`, the output is `x[i]` if `condition[i]` is `True`, otherwise `y[i]`.
///
/// `condition` must have dtype `bool`. `x` and `y` must have the same dtype. All three arrays
/// must have the same shape. Output dtype equals the dtype of `x` and `y`. Output shape equals
/// the input shape.
///
/// This function deviates from numpy in a few ways:
/// - `x` and `y` must have the same dtype (numpy will upcast if they differ)
/// - all three arrays must have the same shape (numpy will broadcast if they differ)
/// - `condition` must already have `bool` dtype; use [`jix.astype(condition, 'bool')`][jix.astype] if needed
///   (numpy implicitly casts the condition to bool)
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// Args:
///     condition: Boolean selector array.
///     x: Values to use where `condition` is True.
///     y: Values to use where `condition` is False.
///
/// Returns:
///     A lazy [`jix.Array`][jix.Array] with the same shape as the broadcast of `condition`, `x`, and `y`.
///         Elements are drawn from `x` where `condition` is `True`, and from `y` elsewhere.
///
/// Examples:
///     ```python
///     import jix
///     import numpy as np
///
///     cond = jix.compact([True, False, True, False])
///     x = jix.compact([1, 2, 3, 4], dtype=np.int32)
///     y = jix.compact([10, 20, 30, 40], dtype=np.int32)
///     result = jix.where(cond, x, y)
///     assert np.array_equal(result.numpy(), [1, 20, 3, 40])
///     ```
#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyo3::pyfunction]
#[pyo3(name = "where")]
pub fn r#where<'py>(
    condition: &pyo3::Bound<'py, pyo3::PyAny>,
    x: &pyo3::Bound<'py, pyo3::PyAny>,
    y: &pyo3::Bound<'py, pyo3::PyAny>,
) -> pyo3::PyResult<Array> {
    let py = condition.py();
    let condition = any_to_core_array(condition)?
        .into_typed::<bool>()
        .into_py_result()?;
    let x_py = asarray(x)?;
    let x = any_to_core_array(&x_py)?;
    let y = any_to_core_array(y)?;
    let ret = jix_core::ops::Where::new_array(condition, x, y).into_py_result()?;
    let np_dtype = x_py.get().dtype(py)?;
    Ok(Array::from_core_with_np_dtype(
        ret.into_any(),
        np_dtype.unbind(),
    ))
}
