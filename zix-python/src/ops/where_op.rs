use crate::ops::as_core_array;
use crate::util::IntoPyResult;
use crate::Array;

/// Selects elements element-wise from `x` or `y` based on `condition`.
///
/// For each index `i`, the output is `x[i]` if `condition[i]` is `True`, otherwise `y[i]`.
///
/// `condition` must have dtype `bool`. `x` and `y` must have the same dtype. All three arrays
/// must have the same shape. Output dtype equals the dtype of `x` and `y`. Output shape equals
/// the input shape.
///
/// All three arguments may be anything that `zix.asarray()` accepts.
///
/// This function deviates from numpy in a few ways:
/// - `x` and `y` must have the same dtype (numpy will upcast if they differ)
/// - all three arrays must have the same shape (numpy will broadcast if they differ)
/// - `condition` must already have `bool` dtype; use `zix.astype(condition, 'bool')` if needed
///   (numpy implicitly casts the condition to bool)
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// # Examples
/// ```python,ignore
/// import zix
/// import numpy as np
///
/// cond = zix.asarray(np.array([True, False, True, False]))
/// x = zix.asarray(np.array([1, 2, 3, 4], dtype=np.int32))
/// y = zix.asarray(np.array([10, 20, 30, 40], dtype=np.int32))
/// result = zix.where(cond, x, y)
/// assert np.array_equal(result.numpy(), [1, 20, 3, 40])
/// ```
#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyo3::pyfunction]
#[pyo3(name = "where")]
pub fn r#where<'py>(
    condition: &pyo3::Bound<'py, pyo3::PyAny>,
    x: &pyo3::Bound<'py, pyo3::PyAny>,
    y: &pyo3::Bound<'py, pyo3::PyAny>,
) -> pyo3::PyResult<Array> {
    let condition = as_core_array(condition)?;
    let x = as_core_array(x)?;
    let y = as_core_array(y)?;
    let ret = zix_core::ops::Where::new(condition, x, y).into_py_result()?;
    Ok(Array::from_core_storage(ret))
}
