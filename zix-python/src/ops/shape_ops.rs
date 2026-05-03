use pyo3::prelude::*;
use zix_core::Array as ZixArray;

use crate::ops::as_core_array;
use crate::util::{normalize_axes, normalize_axis, IntoPyResult, ItemOrSequence};
use crate::Array;

/// Expands an array to a larger shape by repeating elements along length-1 dimensions.
///
/// `new_shape` must have the same number of dimensions as the input. For each dimension `d`,
/// either `new_shape[d] == input_shape[d]` (kept as-is) or `input_shape[d] == 1` (broadcast:
/// the single element is repeated `new_shape[d]` times). Any other combination raises an error.
///
/// Output dtype equals the input dtype. Output shape equals `new_shape`.
///
/// When `copy=True` (the default) the result is an eagerly materialized compact array with a
/// block layout matched to `new_shape`. When `copy=False` the result is a lazy view; reading
/// it may decompress many blocks if the original storage is block-based and the new shape
/// crosses block boundaries.
///
/// The `array` argument must already be a `zix.Array` (no implicit `asarray()` conversion).
///
/// This function deviates from `numpy.broadcast_to`:
/// - `new_shape` must have the same number of dimensions as the input (numpy pads leading
///   dimensions automatically)
///
/// # Examples
/// ```python,ignore
/// import zix
/// import numpy as np
///
/// # Row vector [1, 3] → matrix [2, 3]: every row becomes identical
/// a = zix.asarray(np.array([[1, 2, 3]], dtype=np.int32))
/// result = zix.broadcast(a, [2, 3])
/// assert result.numpy().shape == (2, 3)
/// assert np.array_equal(result.numpy()[0], result.numpy()[1])
///
/// # Column vector [3, 1] → matrix [3, 2]: every column becomes identical
/// b = zix.asarray(np.array([[10], [20], [30]], dtype=np.int32))
/// result = zix.broadcast(b, [3, 2])
/// assert result.numpy()[0, 0] == result.numpy()[0, 1] == 10
/// ```
#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (
    array,
    new_shape,
    copy=true,
))]
pub fn broadcast<'py>(
    array: &Bound<'py, Array>,
    new_shape: ItemOrSequence<u64>,
    copy: bool,
) -> PyResult<Array> {
    let py = array.py();
    let array = array.borrow().to_core_array();
    let new_shape = new_shape.into_vec();
    let ret = zix_core::ops::Broadcast::new(array, &new_shape).into_py_result()?;
    if !copy {
        return Ok(Array::from_core_storage(ret));
    }
    // release the GIL while copying the data
    py.detach(|| {
        let ret = ZixArray::from_storage(ret).copy().into_py_result()?;
        Ok(Array::from_core_storage(ret.into_storage()))
    })
}

// TODO slice

/// Inserts new length-1 dimensions at specified positions in an array's shape.
///
/// Each value in `axes` is a **gap index** that identifies a position before an existing
/// dimension: `0` inserts before dimension 0 (a new leading axis), `1` inserts between
/// dimensions 0 and 1, …, `ndim` appends after the last dimension. Negative values are
/// supported and are resolved against `ndim + 1`.
///
/// Duplicate gap indices are allowed and each adds another length-1 dimension at the same
/// position. The order of values in `axes` does not matter.
///
/// This differs from `numpy.expand_dims`, where each axis index refers to the new (larger)
/// shape rather than the original shape.
///
/// Output dtype and total number of elements equal the input.
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// # Examples
/// ```python,ignore
/// import zix
/// import numpy as np
///
/// a = zix.asarray(np.array([1, 2, 3], dtype=np.int32))   # shape [3]
/// assert zix.insert_axes(a, [0]).numpy().shape == (1, 3)  # → [1, 3]
/// assert zix.insert_axes(a, [1]).numpy().shape == (3, 1)  # → [3, 1]
/// assert zix.insert_axes(a, [-1]).numpy().shape == (3, 1) # negative: same as [1]
///
/// b = zix.asarray(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32))  # shape [2, 3]
/// assert zix.insert_axes(b, [0, 2]).numpy().shape == (1, 2, 1, 3)    # → [1, 2, 1, 3]
///
/// # duplicate axes: multiple length-1 dimensions at the same position
/// assert zix.insert_axes(b, [0, 0, 0, 2]_.shape() == (1, 1, 1, 2, 1, 3)
/// ```
#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
pub fn insert_axes<'py>(array: &Bound<'py, Array>, axes: ItemOrSequence<i32>) -> PyResult<Array> {
    // NOTE: API different than numpy: axes are specified with respect to the original ndim, not the new ndim. Same
    // axis can be specified multiple times to insert multiple axes in the same place.
    let array = array.borrow().to_core_array();
    let axes = normalize_axes(axes.into_vec(), array.ndim() + 1)?;
    let ret = zix_core::ops::InsertAxes::new(array, &axes).into_py_result()?;
    Ok(Array::from_core_storage(ret))
}

/// Removes length-1 dimensions from an array's shape.
///
/// `axes` is a set of axis indices in the *input* shape (0-based). Each named dimension must
/// have size exactly 1 and is dropped from the output. Duplicate axis indices are not allowed.
/// Negative values are supported and are resolved against `ndim`. Removed axes must have size 1.
///
/// Output dtype and total number of elements equal the input.
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// # Examples
/// ```python,ignore
/// import zix
/// import numpy as np
///
/// a = zix.asarray(np.array([[1, 2, 3]], dtype=np.int32))  # shape [1, 3]
/// assert zix.remove_axes(a, [0]).numpy().shape == (3,)     # → [3]
///
/// b = zix.asarray(np.array([[[10], [20]]], dtype=np.int32))  # shape [1, 2, 1]
/// assert zix.remove_axes(b, [0, 2]).numpy().shape == (2,)    # → [2]
/// assert zix.remove_axes(b, [0, -1]).numpy().shape == (2,)   # negative axis
/// ```
#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
pub fn remove_axes<'py>(array: &Bound<'py, Array>, axes: ItemOrSequence<i32>) -> PyResult<Array> {
    let array = array.borrow().to_core_array();
    let axes = normalize_axes(axes.into_vec(), array.ndim())?;
    let ret = zix_core::ops::RemoveAxes::new(array, &axes).into_py_result()?;
    Ok(Array::from_core_storage(ret))
}

/// Reorders the axes of an array (generalized transpose).
///
/// The `i`-th output axis corresponds to axis `axes[i]` of the input — identical to
/// `numpy.transpose`. `axes` must be a permutation of `0..ndim`: correct length, all values
/// in range, no duplicates. Integer values are interpreted as unsigned axis indices (negative
/// axes are **not** supported for `axes`).
///
/// When `axes=None` (the default), all axes are reversed: output axis `i` maps to input axis
/// `ndim - 1 - i`. For 2-D arrays this is the standard matrix transpose.
///
/// Output dtype equals the input dtype.
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// # Examples
/// ```python,ignore
/// import zix
/// import numpy as np
///
/// # 2-D transpose: [2, 3] → [3, 2]
/// a = zix.asarray(np.arange(6, dtype=np.int32).reshape(2, 3))
/// t = zix.permute_axes(a, [1, 0])
/// assert t.numpy().shape == (3, 2)
/// assert np.array_equal(t.numpy(), a.numpy().T)
///
/// # axes=None reverses all axes (same as numpy.transpose with no argument)
/// assert zix.permute_axes(a).numpy().shape == (3, 2)
/// ```
#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (
    array,
    axes=None,
))]
pub fn permute_axes<'py>(array: &Bound<'py, Array>, axes: Option<Vec<usize>>) -> PyResult<Array> {
    let array = array.borrow().to_core_array();
    let axes = axes.unwrap_or_else(|| (0..array.ndim()).rev().collect());
    let ret = zix_core::ops::PermuteAxes::new(array, &axes).into_py_result()?;
    Ok(Array::from_core_storage(ret))
}

/// Reinterprets an array with a different shape.
///
/// The total number of elements must be preserved: the product of `new_shape` must equal the
/// product of the original shape. Exactly one dimension in `new_shape` may be `-1`; that
/// dimension is inferred from the others and the total element count.
///
/// Output dtype equals the input dtype.
///
/// When `copy=True` (the default) the result is an eagerly materialized compact array with a
/// block layout matched to `new_shape`. When `copy=False` the result is a lazy view; reading
/// it may decompress many blocks if the new shape is not aligned with the original block
/// boundaries — use with care.
///
/// The `array` argument must already be a `zix.Array` (no implicit `asarray()` conversion).
///
/// # Examples
/// ```python,ignore
/// import zix
/// import numpy as np
///
/// a = zix.asarray(np.arange(6, dtype=np.int32).reshape(2, 3))  # shape [2, 3]
///
/// # Flatten
/// flat = zix.reshape(a, [6])
/// assert np.array_equal(flat.numpy(), [0, 1, 2, 3, 4, 5])
///
/// # Infer one dimension with -1
/// r = zix.reshape(a, [-1, 2])
/// assert r.numpy().shape == (3, 2)
/// ```
#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (
    array,
    new_shape,
    copy=true,
))]
pub fn reshape<'py>(
    array: &Bound<'py, Array>,
    new_shape: ItemOrSequence<i64>,
    copy: bool,
) -> PyResult<Array> {
    let py = array.py();
    let array = array.borrow().to_core_array();

    // handle -1 in new_shape
    let new_shape = {
        let mut new_shape = new_shape.into_vec();
        let mut inferred_dim = None;
        let mut known_size = 1;
        for (i, &dim) in new_shape.iter().enumerate() {
            if dim >= 0 {
                known_size *= dim as u64;
            } else if dim == -1 {
                if inferred_dim.is_some() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Only one dimension can be inferred (-1)",
                    ));
                }
                inferred_dim = Some(i);
            } else {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "new_shape must be non negative or -1",
                ));
            }
        }
        if let Some(inferred_dim) = inferred_dim {
            let array_size = array.shape().iter().product::<u64>();
            if array_size == 0 || known_size == 0 || !array_size.is_multiple_of(known_size) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Cannot infer dimension size",
                ));
            }
            new_shape[inferred_dim] = (array_size / known_size) as i64;
        }
        new_shape.iter().map(|&dim| dim as u64).collect::<Vec<_>>()
    };

    let ret = zix_core::ops::Reshape::new(array, &new_shape).into_py_result()?;
    if !copy {
        return Ok(Array::from_core_storage(ret));
    }
    // release the GIL while copying the data
    py.detach(|| {
        let ret = ZixArray::from_storage(ret).copy().into_py_result()?;
        Ok(Array::from_core_storage(ret.into_storage()))
    })
}

/// Joins a sequence of arrays along an existing axis.
///
/// All input arrays must have the same number of dimensions, the same dtype, and identical
/// sizes on every axis *except* the concatenation axis, along which their sizes may differ.
/// The output has the same number of dimensions as the inputs.
///
/// `axis` supports negative values (e.g. `-1` for the last axis). Each element of `arrays`
/// may be anything that `zix.asarray()` accepts.
///
/// This function deviates from numpy in a few ways:
/// - all arrays must have the same dtype (numpy will upcast if they differ)
/// - all arrays must have the same number of dimensions (numpy will expand dims if they differ)
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// # Examples
/// ```python,ignore
/// import zix
/// import numpy as np
///
/// # 1-D: join end-to-end
/// a = zix.asarray(np.array([1, 2, 3], dtype=np.int32))
/// b = zix.asarray(np.array([4, 5], dtype=np.int32))
/// c = zix.concatenate([a, b])
/// assert np.array_equal(c.numpy(), [1, 2, 3, 4, 5])
///
/// # 2-D: append rows (axis 0) or columns (axis 1 / axis -1)
/// a = zix.asarray(np.array([[1, 2], [3, 4]], dtype=np.int32))
/// b = zix.asarray(np.array([[5, 6]], dtype=np.int32))
/// assert zix.concatenate([a, b], axis=0).numpy().shape == (3, 2)
/// assert zix.concatenate([a, b.T], axis=1).numpy().shape == (2, 3)
/// ```
#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (
    arrays,
    axis=0,
))]
pub fn concatenate<'py>(arrays: Vec<Bound<'py, PyAny>>, axis: i32) -> PyResult<Array> {
    let arrays = arrays
        .into_iter()
        .map(|arr| as_core_array(&arr))
        .collect::<Result<Vec<_>, _>>()?;
    if arrays.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "arrays must contain at least one array",
        ));
    }
    let ndim = arrays[0].ndim();
    for arr in &arrays {
        if arr.ndim() != ndim {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "All arrays must have the same number of dimensions",
            ));
        }
    }
    let axis = normalize_axis(axis, ndim)?;
    let ret = zix_core::ops::Concatenate::new(arrays, axis).into_py_result()?;
    Ok(Array::from_core_storage(ret))
}

/// Joins a sequence of arrays along a **new** axis.
///
/// All input arrays must have identical shapes and the same dtype. A new axis of size equal
/// to the number of arrays is inserted at position `axis` in the output. The output has one
/// more dimension than the inputs — unlike `zix.concatenate`, which joins along an existing
/// axis.
///
/// `axis` supports negative values and is resolved against `ndim + 1` (the number of valid
/// insertion points for the new axis). Each element of `arrays` may be anything that
/// `zix.asarray()` accepts.
///
/// This function deviates from numpy in a few ways:
/// - all arrays must have the same dtype (numpy will upcast if they differ)
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// # Examples
/// ```python,ignore
/// import zix
/// import numpy as np
///
/// a = zix.asarray(np.array([1, 2, 3], dtype=np.int32))
/// b = zix.asarray(np.array([4, 5, 6], dtype=np.int32))
///
/// # Stack along a new leading axis → shape [2, 3]
/// c = zix.stack([a, b], axis=0)
/// assert c.numpy().shape == (2, 3)
/// assert np.array_equal(c.numpy()[0], [1, 2, 3])
///
/// # Stack along a new trailing axis → shape [3, 2]
/// d = zix.stack([a, b], axis=1)
/// assert d.numpy().shape == (3, 2)
/// assert np.array_equal(d.numpy()[:, 0], [1, 2, 3])
/// ```
#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (
    arrays,
    axis=0,
))]
pub fn stack<'py>(arrays: Vec<Bound<'py, PyAny>>, axis: i32) -> PyResult<Array> {
    let arrays = arrays
        .into_iter()
        .map(|arr| as_core_array(&arr))
        .collect::<Result<Vec<_>, _>>()?;
    if arrays.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "arrays must contain at least one array",
        ));
    }
    let ndim = arrays[0].ndim();
    for arr in &arrays {
        if arr.ndim() != ndim {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "All arrays must have the same number of dimensions",
            ));
        }
    }
    let axis = normalize_axis(axis, ndim + 1)?;
    let ret = zix_core::ops::Stack::new(arrays, axis).into_py_result()?;
    Ok(Array::from_core_storage(ret))
}
