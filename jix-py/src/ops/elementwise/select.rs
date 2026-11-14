use pyo3::prelude::*;

use crate::ops::common::{broadcast_operands, operands_dtype_promotion, Operand, Scalar};
use crate::util::IntoPyResult;
use crate::Array;

/// Selects `x[i]` where `condition[i]` is `True` and `y[i]` elsewhere.
///
/// Unlike numpy, `condition` must already have dtype `bool`; use
/// [`jix.cast(condition, 'bool')`][jix.cast] if needed.
///
/// **Type promotion**: `x` and `y` are cast to a common dtype, which is the output dtype.
///
/// **Broadcasting**: the shapes of `condition`, `x`, and `y` are broadcast to a common shape
/// following numpy rules, which is the output shape.
///
/// If `condition` is a scalar, or `x` and `y` are the same object, the selected operand is returned
/// as is (after broadcasting and type promotion).
///
/// Args:
///     condition: Boolean selector array.
///     x: Values to use where `condition` is True.
///     y: Values to use where `condition` is False.
///
/// Returns:
///     A lazy [`jix.Array`][jix.Array] view.
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
///
///     # Mixed dtypes promote; a Python scalar takes the array's dtype
///     cond = jix.compact([True, False, True])
///     a = jix.compact([1, 2, 3], dtype=np.uint8)
///     b = jix.compact([-1, -2, -3], dtype=np.int8)
///     assert jix.where(cond, a, b).dtype == np.int16
///     assert jix.where(cond, a, 0).dtype == np.uint8
///     ```
#[pyo3::pyfunction]
#[pyo3(name = "where")]
pub fn r#where<'py>(
    condition: &Bound<'py, PyAny>,
    x: &Bound<'py, PyAny>,
    y: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, Array>> {
    let py = condition.py();
    let same_values = x.is(y);

    let condition = Operand::from_any(condition)?;
    let x = Operand::from_any(x)?;
    let y = Operand::from_any(y)?;
    let [condition, x, y] = broadcast_operands([condition, x, y])?;
    let [x, y] = operands_dtype_promotion("where", [x, y])?;

    if let Operand::Scalar {
        value: Scalar::Bool(value),
        ..
    } = condition
    {
        let selected = if value { x } else { y };
        return selected.into_py_array(py);
    }

    let condition = condition
        .into_array()?
        .into_typed::<jix_core::scalar::bool8>()
        .into_py_result()?;
    if same_values {
        return x.into_py_array(py);
    }
    let x_py = x.into_py_array(py)?;
    let np_dtype = x_py.get().numpy_dtype(py)?;
    let x = x_py.get().to_core();
    let y = y.into_array()?;

    let ret = jix_core::ops::Select::new_array(condition, x, y).into_py_result()?;
    Bound::new(
        py,
        Array::from_core_with_np_dtype(ret.into_any(), np_dtype.unbind()),
    )
}
