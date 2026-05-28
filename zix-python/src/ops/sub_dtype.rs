use pyo3::prelude::*;
use zix_core::TypeDyn;

use crate::util::IntoPyResult;
use crate::Array;

/// Extracts one named field from a struct dtype array.
///
/// `array` must have a struct dtype (e.g. a NumPy structured array). The output has the dtype
/// of field `sub_field` and the same shape as the input. Field bytes are sliced out of each
/// element on demand.
///
/// The `array` argument may be anything that `zix.asarray()` accepts, including NumPy
/// structured arrays.
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// # Errors
///
/// Raises `ValueError` if the array dtype is not a struct dtype or does not have a field
/// named `sub_field`.
///
/// # Examples
/// ```python,ignore
/// import zix
/// import numpy as np
///
/// # 1-D array of structs with two i32 fields
/// dt = np.dtype([('x', np.int32), ('y', np.int32)])
/// pts = np.array([(1, 10), (2, 20), (3, 30)], dtype=dt)
/// za = zix.asarray(pts)
///
/// xs = zix.dtype_sub_field(za, 'x')
/// ys = zix.dtype_sub_field(za, 'y')
/// assert xs.dtype == np.int32
/// assert np.array_equal(xs.numpy(), [1, 2, 3])
/// assert np.array_equal(ys.numpy(), [10, 20, 30])
///
/// # 2-D struct array: shape is preserved
/// pts2d = np.array([[(1, 2), (3, 4)], [(5, 6), (7, 8)]], dtype=dt)
/// za2d = zix.asarray(pts2d)
/// xs2d = zix.dtype_sub_field(za2d, 'x')
/// assert xs2d.shape == (2, 2)
/// assert np.array_equal(xs2d.numpy(), [[1, 3], [5, 7]])
/// ```
#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
pub fn dtype_sub_field<'py>(
    array: &Bound<'py, pyo3::PyAny>,
    sub_field: String,
) -> PyResult<Bound<'py, Array>> {
    let py = array.py();
    let array = crate::ops::as_array::any_to_core_array(array)?;
    let res = zix_core::ops::SubDtype::<_, TypeDyn>::new(array, &sub_field).into_py_result()?;
    Bound::new(py, crate::Array::from_core_storage(res))
}
