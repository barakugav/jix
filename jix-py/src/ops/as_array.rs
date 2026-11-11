use jix_core::ArrayAny;
use numpy::PyUntypedArray;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict};

use crate::array::{resolve_array_params, Array};
use crate::dtype::{dtype_from_numpy, numpy_descr_from_any};
use crate::ops::cast_impl;
use crate::ops::common::Operand;
use crate::util::numpy_items;

/// Convert any array-like object to a [`jix.Array`][jix.Array].
///
/// Unlike [`jix.compact()`][jix.compact], it never compresses the data: it produces a view of the
/// input, copying only when necessary (e.g. to convert a Python list to a typed buffer).
///
/// Accepts Python scalars, lists, tuples, NumPy arrays, and any other object accepted by
/// [`numpy.asarray`](https://numpy.org/doc/stable/reference/generated/numpy.asarray.html).
///
/// Args:
///     value: The array-like to convert.
///     dtype: Optional dtype to cast the array to. Accepts a numpy dtype object, a dtype string
///         (e.g. `'float32'`), or a Python type like `np.float32` - anything `numpy.dtype()`
///         accepts. When omitted, the input dtype is preserved. The cast is a lazy view, so no
///         computation happens until the result is read (see [`jix.cast()`][jix.cast]).
///     params: Block layout and codec configuration. See [`jix.compact()`][jix.compact] for
///         details. The returned array is never compressed, so these mostly affect only arrays
///         later created from it (e.g. via `compact()` or `write_array()`), which inherit them.
///         The block shape is ignored, pass it to `compact()` or `write_array()` instead.
///         Only relevant when `value` is not already a jix array; an existing jix array is
///         returned as-is.
///
/// Note:
///     - If `value` is already an `Array` and `dtype` is `None`, it is returned as-is with no copy.
///       When `dtype` is given, the result is a lazy cast view over the input instead.
///     - 0-dimensional inputs produce a scalar array backed by a single value (no buffer).
///     - All other inputs share the underlying buffer with the intermediate NumPy array (zero-copy);
///       the NumPy array is kept alive for as long as the returned `Array` is alive.
///
/// Raises:
///     ValueError: If the input cannot be converted by `numpy.asarray`, has more dimensions than
///         jix supports, or has negative strides (e.g. a reversed slice `a[::-1]`).
///     TypeError: If `dtype` requests an unsupported cast (e.g. from a complex type to an int or
///         float type, or a cast involving struct dtypes).
///
/// Returns:
///     A [`jix.Array`][jix.Array] view of the input.
#[pyfunction]
#[pyo3(signature = (value, *, dtype=None, params=None))]
#[inline]
pub fn asarray<'py>(
    value: &Bound<'py, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
    params: Option<Bound<'_, PyDict>>,
) -> PyResult<Bound<'py, Array>> {
    let py = value.py();
    let params = resolve_array_params(py, params)?;
    let np_dtype = dtype
        .map(|dtype| numpy_descr_from_any(py, dtype))
        .transpose()?;

    let value_is_array = value.is_instance_of::<Array>()
        || value.is_instance_of::<PyUntypedArray>()
        || value.is_instance(numpy_items(py)?.generic.bind(py))?;
    let value = match &np_dtype {
        Some(np_dtype) if !value_is_array => {
            let kwargs = [("dtype", np_dtype)].into_py_dict(py)?;
            let asarray = numpy_items(py)?.asarray.bind(py);
            asarray.call((value,), Some(&kwargs))?
        }
        _ => value.clone(),
    };
    let mut py_arr = Operand::from_any_with_params(&value, params, false)?.into_py_array(py)?;

    if let Some(np_dtype) = &np_dtype {
        let dtype = dtype_from_numpy(np_dtype)?;
        let array = cast_impl(py_arr.get().arr.clone(), &dtype)?;
        py_arr = Bound::new(
            py,
            Array::from_core_with_np_dtype(array, np_dtype.clone().unbind()),
        )?;
    }

    Ok(py_arr)
}

#[inline]
pub(crate) fn asarray_simple<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Array>> {
    asarray(value, None, None)
}

#[inline]
pub(crate) fn any_to_core_array<'py>(value: &Bound<'py, PyAny>) -> PyResult<ArrayAny> {
    Ok(asarray_simple(value)?.get().to_core())
}
