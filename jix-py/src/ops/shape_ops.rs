use pyo3::prelude::*;

use crate::ops::{any_to_core_array, asarray, copy_impl};
use crate::util::{normalize_axes, normalize_axis, IntoPyResult, ItemOrSequence};
use crate::Array;

/// Expands an array to a larger shape by repeating elements along length-1 dimensions.
///
/// `shape` must have the same number of dimensions as the input. For each dimension `d`,
/// either `shape[d] == input_shape[d]` (kept as-is) or `input_shape[d] == 1` (broadcast:
/// the single element is repeated `shape[d]` times). Any other combination raises an error.
/// `shape[d]` may be `-1` as a shorthand for `input_shape[d]` (keeps the dimension size
/// unchanged regardless of whether that dimension is 1 or larger).
///
/// Output dtype equals the input dtype. Output shape equals `shape`.
///
/// When `copy=True` (the default) the result is an eagerly materialized compact array with a
/// block layout matched to `shape`. When `copy=False` the result is a lazy view; reading
/// it may decompress many blocks if the original storage is block-based and the new shape
/// crosses block boundaries.
///
/// The `array` argument must already be a `jix.Array` (no implicit `asarray()` conversion).
///
/// This function deviates from `numpy.broadcast_to`:
/// - `shape` must have the same number of dimensions as the input (numpy pads leading
///   dimensions automatically)
///
/// # Examples
/// ```python,ignore
/// import jix
/// import numpy as np
///
/// # Row vector [1, 3] -> matrix [2, 3]: every row becomes identical
/// a = jix.compact([[1, 2, 3]], dtype=np.int32)
/// result = jix.broadcast(a, [2, 3])
/// assert result.numpy().shape == (2, 3)
/// assert np.array_equal(result.numpy()[0], result.numpy()[1])
///
/// # Column vector [3, 1] -> matrix [3, 2]: every column becomes identical
/// b = jix.compact([[10], [20], [30]], dtype=np.int32)
/// result = jix.broadcast(b, [3, 2])
/// assert result.numpy()[0, 0] == result.numpy()[0, 1] == 10
/// ```
#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (
    array,
    shape,
))]
pub fn broadcast<'py>(
    array: &Bound<'py, Array>,
    shape: ItemOrSequence<i64>,
) -> PyResult<Bound<'py, Array>> {
    let py_arr = array;
    let array = &py_arr.get().arr;
    let new_shape = shape.into_vec();
    let old_shape = array.shape();
    if new_shape.len() != old_shape.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Cannot broadcast array of shape {:?} to shape {:?}: different number of dimensions",
            old_shape, new_shape
        )));
    }
    let new_shape = new_shape
        .into_iter()
        .zip(old_shape)
        .map(|(new_len, old_len)| {
            if new_len >= 0 {
                Ok(new_len as u64)
            } else if new_len == -1 {
                Ok(*old_len)
            } else {
                Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid broadcast shape dimension: expected non-negative or -1, got {}",
                    new_len
                )))
            }
        })
        .collect::<PyResult<Vec<_>>>()?;

    if new_shape == array.shape() {
        // no-op if already the right shape
        return Ok(py_arr.clone());
    }

    let ret = jix_core::ops::Broadcast::new_array(array.clone(), &new_shape).into_py_result()?;
    Bound::new(py_arr.py(), Array::from_core(ret.into_any()))
}

// TODO slice

/// Inserts new length-1 dimensions at specified positions in an array's shape.
///
/// Each value in `axis` is a **gap index** that identifies a position before an existing
/// dimension: `0` inserts before dimension 0 (a new leading axis), `1` inserts between
/// dimensions 0 and 1, ..., `ndim` appends after the last dimension. Negative values are
/// supported and are resolved against `ndim + 1`.
///
/// Duplicate gap indices are allowed and each adds another length-1 dimension at the same
/// position. The order of values in `axis` does not matter.
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
/// import jix
/// import numpy as np
///
/// a = jix.compact([1, 2, 3], dtype=np.int32)   # shape [3]
/// assert jix.insert_axis(a, 0).numpy().shape == (1, 3)  # -> [1, 3]
/// assert jix.insert_axis(a, 1).numpy().shape == (3, 1)  # -> [3, 1]
/// assert jix.insert_axis(a, -1).numpy().shape == (3, 1) # negative: same as [1]
///
/// b = jix.compact([[1, 2, 3], [4, 5, 6]], dtype=np.int32)  # shape [2, 3]
/// assert jix.insert_axis(b, [0, 2]).numpy().shape == (1, 2, 1, 3)    # -> [1, 2, 1, 3]
///
/// # duplicate axes: multiple length-1 dimensions at the same position
/// assert jix.insert_axis(b, [0, 0, 0, 2]).shape() == (1, 1, 1, 2, 1, 3)
/// ```
#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
pub fn insert_axis<'py>(
    array: &Bound<'py, Array>,
    axis: ItemOrSequence<i32>,
) -> PyResult<Bound<'py, Array>> {
    // NOTE: API different than numpy: axes are specified with respect to the original ndim, not the new ndim. Same
    // axis can be specified multiple times to insert multiple axes in the same place.
    let py_arr = array;
    let array = py_arr.get().to_core();
    let axes = normalize_axes(axis.into_vec(), array.ndim() + 1)?;
    if axes.is_empty() {
        return Ok(py_arr.clone()); // no-op if no axes to insert
    }
    let ret = jix_core::ops::InsertAxis::new_array(array, &axes).into_py_result()?;
    Bound::new(py_arr.py(), Array::from_core(ret.into_any()))
}
/// Inserts new length-1 dimensions at specified positions in an array's shape. Alias for :func:`jix.insert_axis()`.
#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
pub fn unsqueeze<'py>(
    array: &Bound<'py, Array>,
    axis: ItemOrSequence<i32>,
) -> PyResult<Bound<'py, Array>> {
    insert_axis(array, axis)
}

/// Removes length-1 dimensions from an array's shape.
///
/// `axis` is a set of axis indices in the *input* shape (0-based). Each named dimension must
/// have size exactly 1 and is dropped from the output. Duplicate axis indices are not allowed.
/// Negative values are supported and are resolved against `ndim`. Removed axes must have size 1.
///
/// Output dtype and total number of elements equal the input.
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// # Examples
/// ```python,ignore
/// import jix
/// import numpy as np
///
/// a = jix.compact([[1, 2, 3]], dtype=np.int32)  # shape [1, 3]
/// assert jix.remove_axis(a, 0).numpy().shape == (3,)     # -> [3]
///
/// b = jix.compact([[[10], [20]]], dtype=np.int32)  # shape [1, 2, 1]
/// assert jix.remove_axis(b, [0, 2]).numpy().shape == (2,)    # -> [2]
/// assert jix.remove_axis(b, [0, -1]).numpy().shape == (2,)   # negative axis
/// ```
#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
pub fn remove_axis<'py>(
    array: &Bound<'py, Array>,
    axis: ItemOrSequence<i32>,
) -> PyResult<Bound<'py, Array>> {
    let py_arr = array;
    let array = py_arr.get().to_core();
    let axes = normalize_axes(axis.into_vec(), array.ndim())?;
    if axes.is_empty() {
        return Ok(py_arr.clone()); // no-op if no axes to remove
    }
    let ret = jix_core::ops::RemoveAxis::new_array(array, &axes).into_py_result()?;
    Bound::new(py_arr.py(), Array::from_core(ret.into_any()))
}

/// Removes length-1 dimensions from an array's shape.
///
/// When `axis=None` (the default), all size-1 dimensions are removed. When `axis` is given,
/// only the specified axes are removed; each named dimension must have size exactly 1.
/// Negative axis values are supported and are resolved against `ndim`.
///
/// Output dtype and total number of elements equal the input. The result is a lazy view; no
/// computation occurs until the array is read.
///
/// # Examples
/// ```python,ignore
/// import jix
/// import numpy as np
///
/// a = jix.compact([[[1, 2, 3]]], dtype=np.int32)  # shape [1, 1, 3]
/// assert jix.squeeze(a).numpy().shape == (3,)              # remove all size-1 dims
/// assert jix.squeeze(a, axis=0).numpy().shape == (1, 3)    # remove only axis 0
/// assert jix.squeeze(a, axis=[0, 1]).numpy().shape == (3,) # remove axes 0 and 1
/// ```
#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (array, axis=None))]
pub fn squeeze<'py>(
    array: &Bound<'py, Array>,
    axis: Option<ItemOrSequence<i32>>,
) -> PyResult<Bound<'py, Array>> {
    let axis = axis.unwrap_or_else(|| {
        ItemOrSequence::Sequence(
            array
                .get()
                .arr
                .shape()
                .iter()
                .enumerate()
                .filter_map(|(d, len)| (*len == 1).then_some(d as i32))
                .collect(),
        )
    });
    remove_axis(array, axis)
}

/// Reorders the axes of an array (generalized transpose).
///
/// The `i`-th output axis corresponds to axis `axes[i]` of the input - identical to
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
/// import jix
/// import numpy as np
///
/// # 2-D transpose: [2, 3] -> [3, 2]
/// a = jix.asarray(np.arange(6, dtype=np.int32).reshape(2, 3))
/// t = jix.permute_axes(a, [1, 0])
/// assert t.numpy().shape == (3, 2)
/// assert np.array_equal(t.numpy(), a.numpy().T)
///
/// # axes=None reverses all axes (same as numpy.transpose with no argument)
/// assert jix.permute_axes(a).numpy().shape == (3, 2)
/// ```
#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (
    array,
    axes=None,
))]
pub fn permute_axes<'py>(
    array: &Bound<'py, Array>,
    axes: Option<Vec<usize>>,
) -> PyResult<Bound<'py, Array>> {
    let py_arr = array;
    let array = py_arr.get().to_core();
    let axes = axes.unwrap_or_else(|| (0..array.ndim()).rev().collect());
    if axes.len() == array.ndim() && axes.iter().enumerate().all(|(i, &ax)| i == ax) {
        return Ok(py_arr.clone()); // no-op permutation
    }
    let ret = jix_core::ops::PermuteAxes::new_array(array, &axes).into_py_result()?;
    Bound::new(py_arr.py(), Array::from_core(ret.into_any()))
}

/// Reinterprets an array with a different shape.
///
/// The total number of elements must be preserved: the product of `shape` must equal the
/// product of the original shape. Exactly one dimension in `shape` may be `-1`; that
/// dimension is inferred from the others and the total element count.
///
/// Output dtype equals the input dtype.
///
/// When `copy=True` (the default) the result is an eagerly materialized compact array with a
/// block layout matched to `shape`. When `copy=False` the result is a lazy view; reading
/// it may decompress many blocks if the new shape is not aligned with the original block
/// boundaries - use with care.
///
/// The `array` argument must already be a `jix.Array` (no implicit `asarray()` conversion).
///
/// # Examples
/// ```python,ignore
/// import jix
/// import numpy as np
///
/// a = jix.asarray(np.arange(6, dtype=np.int32).reshape(2, 3))  # shape [2, 3]
///
/// # Flatten
/// flat = jix.reshape(a, [6])
/// assert np.array_equal(flat.numpy(), [0, 1, 2, 3, 4, 5])
///
/// # Infer one dimension with -1
/// r = jix.reshape(a, [-1, 2])
/// assert r.numpy().shape == (3, 2)
/// ```
#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (
    array,
    shape,
    *,
    copy=true,
))]
pub fn reshape<'py>(
    array: &Bound<'py, Array>,
    shape: ItemOrSequence<i64>,
    copy: bool,
) -> PyResult<Bound<'py, Array>> {
    let new_shape = shape;
    let py_arr = array;
    let array = &py_arr.get().arr;

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
                    "shape must be non negative or -1",
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

    if new_shape == array.shape() {
        // no-op if already the right shape
        return if !copy {
            Ok(py_arr.clone())
        } else {
            copy_impl(py_arr.py(), array)
        };
    }

    let ret = jix_core::ops::Reshape::new_array(array.clone(), &new_shape).into_py_result()?;
    if !copy {
        Bound::new(py_arr.py(), Array::from_core(ret.into_any()))
    } else {
        copy_impl(py_arr.py(), &ret)
    }
}

/// Collapses an array into a single dimension.
///
/// Equivalent to `jix.reshape(array, [n], copy=copy)` where `n` is the total number of
/// elements. Output dtype equals the input dtype. Output shape is `[n]`.
///
/// When `copy=True` (the default) the result is an eagerly materialized compact array.
/// When `copy=False` the result is a lazy view; reading it may decompress many blocks if the
/// original storage is block-based and the shape is not aligned with block boundaries.
///
/// # Examples
/// ```python,ignore
/// import jix
/// import numpy as np
///
/// a = jix.compact([[1, 2, 3], [4, 5, 6]], dtype=np.int32)  # shape [2, 3]
/// f = jix.flatten(a)
/// assert f.numpy().shape == (6,)
/// assert np.array_equal(f.numpy(), [1, 2, 3, 4, 5, 6])
/// ```
#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (array, *, copy=true))]
pub fn flatten<'py>(array: &Bound<'py, Array>, copy: bool) -> PyResult<Bound<'py, Array>> {
    let size = array.get().arr.shape().iter().product::<u64>();
    reshape(array, ItemOrSequence::Item(size as i64), copy)
}

/// Joins a sequence of arrays along an existing axis.
///
/// All input arrays must have the same number of dimensions, the same dtype, and identical
/// sizes on every axis *except* the concatenation axis, along which their sizes may differ.
/// The output has the same number of dimensions as the inputs.
///
/// `axis` supports negative values (e.g. `-1` for the last axis). Each element of `arrays`
/// may be anything that `jix.asarray()` accepts.
///
/// This function deviates from numpy in a few ways:
/// - all arrays must have the same dtype (numpy will upcast if they differ)
/// - all arrays must have the same number of dimensions (numpy will expand dims if they differ)
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// # Examples
/// ```python,ignore
/// import jix
/// import numpy as np
///
/// # 1-D: join end-to-end
/// a = jix.compact([1, 2, 3], dtype=np.int32)
/// b = jix.compact([4, 5], dtype=np.int32)
/// c = jix.concatenate([a, b])
/// assert np.array_equal(c.numpy(), [1, 2, 3, 4, 5])
///
/// # 2-D: append rows (axis 0) or columns (axis 1 / axis -1)
/// a = jix.compact([[1, 2], [3, 4]], dtype=np.int32)
/// b = jix.compact([[5, 6]], dtype=np.int32)
/// assert jix.concatenate([a, b], axis=0).numpy().shape == (3, 2)
/// assert jix.concatenate([a, b.T], axis=1).numpy().shape == (2, 3)
/// ```
#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (
    arrays,
    axis=0,
))]
pub fn concatenate<'py>(arrays: Vec<Bound<'py, PyAny>>, axis: i32) -> PyResult<Bound<'py, Array>> {
    let py_arrays = arrays
        .iter()
        .map(|arr| asarray(arr))
        .collect::<Result<Vec<_>, _>>()?;
    let arrays = py_arrays
        .iter()
        .map(|arr| any_to_core_array(arr))
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

    let py = py_arrays.first().unwrap().py();
    match arrays.len() {
        1 if axis < ndim => {
            // no-op if only one array
            let [array] = py_arrays.try_into().unwrap();
            Ok(array)
        }
        2 => {
            let [arr1, arr2] = arrays.try_into().unwrap();
            let ret = jix_core::ops::Concatenate::new_array([arr1, arr2], axis).into_py_result()?;
            Bound::new(py, Array::from_core(ret.into_any()))
        }
        3 => {
            let [arr1, arr2, arr3] = arrays.try_into().unwrap();
            let ret =
                jix_core::ops::Concatenate::new_array([arr1, arr2, arr3], axis).into_py_result()?;
            Bound::new(py, Array::from_core(ret.into_any()))
        }
        4 => {
            let [arr1, arr2, arr3, arr4] = arrays.try_into().unwrap();
            let ret = jix_core::ops::Concatenate::new_array([arr1, arr2, arr3, arr4], axis)
                .into_py_result()?;
            Bound::new(py, Array::from_core(ret.into_any()))
        }
        _ => {
            let ret = jix_core::ops::Concatenate::new_array(arrays, axis).into_py_result()?;
            Bound::new(py, Array::from_core(ret.into_any()))
        }
    }
}

/// Joins a sequence of arrays along a **new** axis.
///
/// All input arrays must have identical shapes and the same dtype. A new axis of size equal
/// to the number of arrays is inserted at position `axis` in the output. The output has one
/// more dimension than the inputs - unlike `jix.concatenate`, which joins along an existing
/// axis.
///
/// `axis` supports negative values and is resolved against `ndim + 1` (the number of valid
/// insertion points for the new axis). Each element of `arrays` may be anything that
/// `jix.asarray()` accepts.
///
/// This function deviates from numpy in a few ways:
/// - all arrays must have the same dtype (numpy will upcast if they differ)
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// # Examples
/// ```python,ignore
/// import jix
/// import numpy as np
///
/// a = jix.compact([1, 2, 3], dtype=np.int32)
/// b = jix.compact([4, 5, 6], dtype=np.int32)
///
/// # Stack along a new leading axis -> shape [2, 3]
/// c = jix.stack([a, b], axis=0)
/// assert c.numpy().shape == (2, 3)
/// assert np.array_equal(c.numpy()[0], [1, 2, 3])
///
/// # Stack along a new trailing axis -> shape [3, 2]
/// d = jix.stack([a, b], axis=1)
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
        .map(|arr| any_to_core_array(&arr))
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
    match arrays.len() {
        2 => {
            let [arr1, arr2] = arrays.try_into().unwrap();
            let ret = jix_core::ops::Stack::new_array([arr1, arr2], axis).into_py_result()?;
            Ok(Array::from_core(ret.into_any()))
        }
        3 => {
            let [arr1, arr2, arr3] = arrays.try_into().unwrap();
            let ret = jix_core::ops::Stack::new_array([arr1, arr2, arr3], axis).into_py_result()?;
            Ok(Array::from_core(ret.into_any()))
        }
        4 => {
            let [arr1, arr2, arr3, arr4] = arrays.try_into().unwrap();
            let ret =
                jix_core::ops::Stack::new_array([arr1, arr2, arr3, arr4], axis).into_py_result()?;
            Ok(Array::from_core(ret.into_any()))
        }
        _ => {
            let ret = jix_core::ops::Stack::new_array(arrays, axis).into_py_result()?;
            Ok(Array::from_core(ret.into_any()))
        }
    }
}
