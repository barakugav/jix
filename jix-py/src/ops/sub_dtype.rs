use jix_core::TypeDyn;
use pyo3::prelude::*;

use crate::util::IntoPyResult;
use crate::Array;

/// Extracts one named field from a struct dtype array.
///
/// The output has the dtype of field `sub_field` and the same shape as the input.
///
/// The result is a lazy view.
///
/// Args:
///     array: Input array. Must have a struct dtype (e.g. a NumPy structured array).
///     sub_field: The name of the field to extract.
///
/// Raises:
///     TypeError: If the array dtype is not a struct dtype, has an inner shape or does not have a
///         field named `sub_field`, or if that field is zero-sized.
///
/// Returns:
///     A [`jix.Array`][jix.Array] view of the field.
///
/// Examples:
///     ```python
///     import jix
///     import numpy as np
///
///     # 1-D array of structs with two i32 fields
///     dt = np.dtype([('x', np.int32), ('y', np.int32)])
///     pts = np.array([(1, 10), (2, 20), (3, 30)], dtype=dt)
///     za = jix.asarray(pts)
///
///     xs = jix.dtype_sub_field(za, 'x')
///     ys = jix.dtype_sub_field(za, 'y')
///     assert xs.dtype == np.int32
///     assert np.array_equal(xs.numpy(), [1, 2, 3])
///     assert np.array_equal(ys.numpy(), [10, 20, 30])
///
///     # 2-D struct array: shape is preserved
///     pts2d = np.array([[(1, 2), (3, 4)], [(5, 6), (7, 8)]], dtype=dt)
///     za2d = jix.asarray(pts2d)
///     xs2d = jix.dtype_sub_field(za2d, 'x')
///     assert xs2d.shape == (2, 2)
///     assert np.array_equal(xs2d.numpy(), [[1, 3], [5, 7]])
///     ```
#[pyfunction]
pub fn dtype_sub_field<'py>(
    array: &Bound<'py, PyAny>,
    sub_field: String,
) -> PyResult<Bound<'py, Array>> {
    let py = array.py();
    let array = crate::ops::as_array::any_to_core_array(array)?;
    let res =
        jix_core::ops::SubDtype::<_, TypeDyn>::new_array(array, &sub_field).into_py_result()?;
    Bound::new(py, Array::from_core(res.into_any()))
}
