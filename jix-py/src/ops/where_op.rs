use crate::ops::common::{broadcast_operands, Operand};
use crate::util::IntoPyResult;
use crate::Array;

/// Selects elements element-wise from `x` or `y` based on `condition`.
///
/// For each index `i`, the output is `x[i]` if `condition[i]` is `True`, otherwise `y[i]`.
///
/// `condition` must have dtype `bool`. `x` and `y` must have the same dtype. Output dtype equals
/// the dtype of `x` and `y`.
///
/// **Broadcasting**: the shapes of `condition`, `x`, and `y` are broadcast to a common shape
/// following numpy rules, which is the output shape.
///
/// This function deviates from numpy in a few ways:
/// - `x` and `y` must have the same dtype (numpy will upcast if they differ)
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
///     A lazy [`jix.Array`][jix.Array] with the broadcast shape of `condition`, `x`, and `y`.
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
///
///     # y is broadcast against the (2, 3) condition
///     cond = jix.compact(np.array([[True, False, True], [False, True, False]]))
///     x = jix.compact(np.zeros((2, 3), dtype=np.int32))
///     y = jix.compact(np.array([7, 8, 9], dtype=np.int32))
///     assert np.array_equal(jix.where(cond, x, y).numpy(), [[0, 8, 0], [7, 0, 9]])
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

    let condition = Operand::from_any(condition)?;
    let x = Operand::from_any(x)?;
    let y = Operand::from_any(y)?;
    let [condition, x, y] = broadcast_operands([condition, x, y])?;

    let condition = condition
        .into_array()?
        .into_typed::<bool>()
        .into_py_result()?;
    let x_py = x.into_py_array(py)?;
    let np_dtype = x_py.get().dtype(py)?;
    let x = x_py.get().to_core();
    let y = y.into_array()?;

    let ret = jix_core::ops::Where::new_array(condition, x, y).into_py_result()?;
    Ok(Array::from_core_with_np_dtype(
        ret.into_any(),
        np_dtype.unbind(),
    ))
}
