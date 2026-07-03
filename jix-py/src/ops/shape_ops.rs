use jix_core::ops::{SliceItem, SliceSpec};
use jix_core::NDIM_MAX;
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyEllipsis, PySlice, PyTuple};

use crate::ops::{any_to_core_array, asarray};
use crate::util::{
    normalize_axes, normalize_axes_optional, normalize_axis, normalize_axis_optional, slice_unpack,
    DimArray, IntoPyResult, ItemOrSequence,
};
use crate::Array;

/// Expands an array to a larger shape by repeating elements along length-1 dimensions.
///
/// `shape` must have the same number of dimensions as the input. For each dimension `d`,
/// either `shape[d] == input_shape[d]` (kept as-is) or `input_shape[d] == 1` (broadcast:
/// the single element is repeated `shape[d]` times). Any other combination raises an error.
/// `shape[d]` may be `-1` as a shorthand for `input_shape[d]` (keeps the dimension size
/// unchanged regardless of whether that dimension is 1 or larger).
///
/// `broadcast` is the lazy zero-cost case of replication restricted to length-1 axes. For
/// general element replication along an axis of any length use [`jix.repeat()`][jix.repeat]
/// (each element duplicated in place) or [`jix.tile()`][jix.tile] (the whole sequence
/// duplicated).
///
/// Output dtype equals the input dtype. Output shape equals `shape`.
///
/// The result is a lazy view; no data is copied until the array is read.
///
/// This function deviates from `numpy.broadcast_to`:
/// - `shape` must have the same number of dimensions as the input (numpy pads leading
///   dimensions automatically)
///
/// Args:
///     array: Array to broadcast.
///     shape: Target shape. Must have the same number of dimensions as the input. Use `-1`
///         to keep a dimension unchanged.
///
/// Returns:
///     A [`jix.Array`][jix.Array] with the requested broadcast shape.
///
/// Examples:
///     ```python
///     import jix
///     import numpy as np
///
///     # Row vector [1, 3] -> matrix [2, 3]: every row becomes identical
///     a = jix.compact([[1, 2, 3]], dtype=np.int32)
///     result = jix.broadcast(a, [2, 3])
///     assert result.numpy().shape == (2, 3)
///     assert np.array_equal(result.numpy()[0], result.numpy()[1])
///
///     # Column vector [3, 1] -> matrix [3, 2]: every column becomes identical
///     b = jix.compact([[10], [20], [30]], dtype=np.int32)
///     result = jix.broadcast(b, [3, 2])
///     assert result.numpy()[0, 0] == result.numpy()[0, 1] == 10
///     ```
#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
pub fn broadcast<'py>(
    array: &Bound<'py, PyAny>,
    shape: ItemOrSequence<i64>,
) -> PyResult<Bound<'py, Array>> {
    let py_arr = asarray(array)?;
    let py = py_arr.py();
    let array = &py_arr.get().arr;
    let old_shape = array.shape();
    let shape = shape.into_dim_array()?;
    if shape.len() != old_shape.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Cannot broadcast array of shape {:?} to shape {:?}: different number of dimensions",
            old_shape, shape
        )));
    }
    let new_shape = shape
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
        .collect::<PyResult<DimArray<_>>>()?;

    if &new_shape == array.shape() {
        // no-op if already the right shape
        return Ok(py_arr);
    }

    let ret = jix_core::ops::Broadcast::new_array(array.clone(), new_shape.as_slice())
        .into_py_result()?;
    let np_dtype = py_arr.get().dtype(py)?;
    Bound::new(
        py,
        Array::from_core_with_np_dtype(ret.into_any(), np_dtype.unbind()),
    )
}

/// Selects a sub-region of an array as a lazy view.
///
/// `index` accepts the same forms as [`Array.numpy()`][jix.Array.numpy] / `__getitem__`:
///
/// | Form | Example | Effect |
/// |---|---|---|
/// | integer | `jix.slice(a, 2)` | select a single position along axis 0 (drops the axis) |
/// | slice | `jix.slice(a, slice(1, 4))` | select a range along axis 0 (keeps the axis) |
/// | `...` | `jix.slice(a, ...)` | fill all remaining axes with full slices |
/// | tuple | `jix.slice(a, (0, slice(1, 3), ...))` | index each axis independently |
///
/// **Integers** select one position and drop the corresponding axis. Negative indices
/// are supported. **Slices** select a contiguous range and keep the axis. The
/// **step must be 1**; non-unit steps raise `ValueError`. Bounds are checked
/// strictly (no numpy-style clamping). At most one ellipsis is allowed; missing
/// trailing axes receive implicit full-range slices.
///
/// Output dtype equals the input dtype.
///
/// The result is a lazy [`jix.Array`][jix.Array] view; no decompression occurs until the result is
/// read. Unlike `arr[...]` / `arr.numpy(...)`, this does **not** materialize the
/// selection into a NumPy array - use `.numpy()` afterward when you want the data.
///
/// Args:
///     array: Array to slice.
///     index: See the table above. Same syntax as `__getitem__` / `numpy(index=...)`.
///
/// Returns:
///     A lazy [`jix.Array`][jix.Array] view of the selected sub-region.
///
/// Raises:
///     IndexError: Integer index out of bounds, slice `start` or `stop` out of bounds,
///         more index items than array dimensions, or more than one ellipsis.
///     ValueError: Slice step other than 1.
///     TypeError: Unsupported index item type.
///
/// Examples:
///     ```python
///     import jix
///     import numpy as np
///
///     a = jix.compact(np.arange(12, dtype=np.int32).reshape(3, 4))
///
///     # Single row -- axis 0 is dropped, shape goes (3, 4) -> (4,).
///     row = jix.slice(a, 1)
///     assert row.shape == (4,)
///     assert np.array_equal(row.numpy(), [4, 5, 6, 7])
///
///     # Slice on each axis keeps both axes.
///     sub = jix.slice(a, (slice(0, 2), slice(1, 3)))
///     assert sub.shape == (2, 2)
///     assert np.array_equal(sub.numpy(), [[1, 2], [5, 6]])
///
///     # Ellipsis fills remaining axes.
///     col = jix.slice(a, (..., 2))
///     assert col.shape == (3,)
///     assert np.array_equal(col.numpy(), [2, 6, 10])
///     ```
#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
pub fn slice<'py>(
    array: &Bound<'py, PyAny>,
    index: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, Array>> {
    let py_arr = asarray(array)?;
    let py = py_arr.py();
    let arr = py_arr.get().to_core();
    let parsed = parse_basic_index(arr.shape(), Some(index))?;

    let spec = SliceSpec::new(parsed.items.as_slice());
    let sliced = jix_core::ops::Slice::new_array(arr, spec)
        .into_py_result()?
        .into_any();
    let result = if parsed.drop_axes.is_empty() {
        sliced
    } else {
        jix_core::ops::RemoveAxis::new_array(sliced, parsed.drop_axes.as_slice())
            .into_py_result()?
            .into_any()
    };
    let np_dtype = py_arr.get().dtype(py)?;
    Bound::new(
        py,
        Array::from_core_with_np_dtype(result, np_dtype.unbind()),
    )
}

/// Parse a Python `__getitem__`-style index into a per-axis [`SliceItem`] list.
///
/// Accepts an integer, a `slice`, `Ellipsis`, or a tuple of these. Slice steps other
/// than 1 are rejected. Bounds are validated strictly (no numpy-style clamping).
/// Returns one `SliceItem` per array dimension plus the list of integer-indexed axes.
#[inline]
pub(crate) fn parse_basic_index<'py>(
    shape: &[u64],
    index: Option<&Bound<'py, PyAny>>,
) -> PyResult<ParsedBasicIndex> {
    let ndim = shape.len();

    enum RawIdxItem<'a> {
        Int(i64),
        Slice(&'a Bound<'a, PySlice>),
        Ellipsis,
    }

    let raw = match index {
        Some(index) => {
            if let Ok(tup) = index.cast::<PyTuple>() {
                let tup = tup.as_slice();
                if tup.len() > NDIM_MAX {
                    return Err(PyIndexError::new_err(format!(
                        "too many indices for array: array is {ndim}-dimensional, \
                         but {len} were indexed",
                        len = tup.len()
                    )));
                }
                tup.iter().collect::<DimArray<_>>()
            } else {
                [index].into_iter().collect::<DimArray<_>>()
            }
        }
        None => DimArray::new(),
    };
    let raw = raw
        .into_iter()
        .map(|item| {
            if item.is_instance_of::<PyEllipsis>() {
                return Ok(RawIdxItem::Ellipsis);
            }
            if let Ok(slice) = item.cast::<PySlice>() {
                return Ok(RawIdxItem::Slice(slice));
            }
            if let Ok(i) = item.extract::<i64>() {
                return Ok(RawIdxItem::Int(i));
            }
            Err(PyTypeError::new_err(
                "only integers, slices (`:`), and ellipsis (`...`) are valid indices",
            ))
        })
        .collect::<PyResult<DimArray<_>>>()?;

    let ellipsis_count = raw
        .iter()
        .filter(|r| matches!(r, RawIdxItem::Ellipsis))
        .count();
    if ellipsis_count > 1 {
        return Err(PyIndexError::new_err(
            "an index can only have a single ellipsis ('...')",
        ));
    }
    let consumers = raw.len() - ellipsis_count;
    if consumers > ndim {
        return Err(PyIndexError::new_err(format!(
            "too many indices for array: array is {ndim}-dimensional, \
             but {consumers} were indexed"
        )));
    }

    let fill = ndim - consumers;
    let mut items = DimArray::new();
    let mut drop_axes = DimArray::new();
    let mut axis_cursor = 0usize;
    for r in raw {
        match r {
            RawIdxItem::Ellipsis => {
                for _ in 0..fill {
                    items.push(SliceItem {
                        start: Some(0),
                        end: Some(shape[axis_cursor] as i64),
                        step: 1,
                    });
                    axis_cursor += 1;
                }
            }
            RawIdxItem::Int(i) => {
                let len = shape[axis_cursor] as i64;
                let i_resolved = if i < 0 { i + len } else { i };
                if i_resolved < 0 || i_resolved >= len {
                    return Err(PyIndexError::new_err(format!(
                        "index {i} is out of bounds for axis {axis_cursor} with size {len}"
                    )));
                }
                items.push(SliceItem {
                    start: Some(i_resolved),
                    end: Some(i_resolved + 1),
                    step: 1,
                });
                drop_axes.push(axis_cursor);
                axis_cursor += 1;
            }
            RawIdxItem::Slice(s) => {
                let len = shape[axis_cursor] as i64;
                let s = slice_unpack(s, len)?;
                if s.step != 1 {
                    return Err(PyValueError::new_err("slice step must be 1"));
                }
                let start = s.start as i64;
                let stop = s.stop as i64;
                let start_norm = if start < 0 { start + len } else { start };
                let stop_norm = if stop < 0 { stop + len } else { stop };
                if start_norm < 0 || start_norm >= len {
                    return Err(PyIndexError::new_err(format!(
                        "slice start {start} is out of bounds for axis {axis_cursor} with size {len}"
                    )));
                }
                if stop_norm < 0 || stop_norm > len {
                    return Err(PyIndexError::new_err(format!(
                        "slice stop {stop} is out of bounds for axis {axis_cursor} with size {len}"
                    )));
                }
                if start_norm > stop_norm {
                    return Err(PyIndexError::new_err(format!(
                        "slice start {start} must be <= stop {stop} for axis {axis_cursor}"
                    )));
                }
                items.push(SliceItem {
                    start: Some(start_norm),
                    end: Some(stop_norm),
                    step: 1,
                });
                axis_cursor += 1;
            }
        }
    }
    while axis_cursor < ndim {
        items.push(SliceItem {
            start: Some(0),
            end: Some(shape[axis_cursor] as i64),
            step: 1,
        });
        axis_cursor += 1;
    }

    Ok(ParsedBasicIndex { items, drop_axes })
}

/// Parsed result of a basic-indexing expression (`int`, `slice`, `...`, or tuple of these).
///
/// Integer-indexed axes are kept in `items` as single-element slices (`start..start+1`)
/// and listed in `drop_axes`; callers can remove those axes after slicing to recover
/// numpy `arr[i]` semantics.
pub(crate) struct ParsedBasicIndex {
    /// One resolved `SliceItem` per array dimension. `start` and `end` are absolute
    /// (non-negative) indices in the input shape; `step` is always 1.
    pub items: DimArray<SliceItem>,
    /// Axes (0-based, increasing) that came from integer indices.
    pub drop_axes: DimArray<usize>,
}

/// Inserts new length-1 dimensions at specified positions in an array's shape.
///
/// This matches `numpy.expand_dims`: each value in `axis` refers to a position in the
/// **output** (larger) shape, not the input shape. With `n = len(axis)` insertions the output
/// has `ndim + n` dimensions, and the new length-1 axes end up at exactly the requested output
/// positions while the original dimensions fill the rest in order. Negative values are
/// supported and are resolved against the output ndim (`ndim + n`).
///
/// Repeated output axes are rejected, exactly as in `numpy.expand_dims`. The inverse operation
/// is [`jix.remove_axis()`][jix.remove_axis] (which is also exposed as
/// [`jix.squeeze()`][jix.squeeze]).
///
/// Output dtype and total number of elements equal the input.
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// Args:
///     array: Input array.
///     axis: Output-shape index or sequence of indices at which to place new length-1
///         dimensions. Negative values are resolved against the output ndim (`ndim + len(axis)`).
///
/// Returns:
///     A [`jix.Array`][jix.Array] with new length-1 axes inserted at the specified positions.
///
/// Examples:
///     ```python
///     import jix
///     import numpy as np
///
///     a = jix.compact([1, 2, 3], dtype=np.int32)   # shape [3]
///     assert jix.insert_axis(a, 0).numpy().shape == (1, 3)  # -> [1, 3]
///     assert jix.insert_axis(a, 1).numpy().shape == (3, 1)  # -> [3, 1]
///     assert jix.insert_axis(a, -1).numpy().shape == (3, 1) # last axis of the output
///
///     b = jix.compact([[1, 2, 3], [4, 5, 6]], dtype=np.int32)  # shape [2, 3]
///     # axes index the output shape: positions 0 and 2 become new length-1 axes
///     assert jix.insert_axis(b, [0, 2]).numpy().shape == (1, 2, 1, 3)    # -> [1, 2, 1, 3]
///     ```
#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
pub fn insert_axis<'py>(
    array: &Bound<'py, PyAny>,
    axis: ItemOrSequence<i32>,
) -> PyResult<Bound<'py, Array>> {
    let py_arr = asarray(array)?;
    let py = py_arr.py();
    let array = py_arr.get().to_core();
    if axis.is_empty() {
        return Ok(py_arr); // no-op if no axes to insert
    }
    // numpy.expand_dims semantics: each axis indexes the output (larger) shape.
    let out_ndim = array.ndim() + axis.len();
    if out_ndim > NDIM_MAX {
        return Err(PyValueError::new_err(format!(
            "Cannot insert axes: output ndim {out_ndim} exceeds maximum supported {NDIM_MAX}"
        )));
    }
    let axis = axis.into_dim_array().unwrap();
    let mut out_positions = normalize_axes(&axis, out_ndim)?;
    out_positions.sort_unstable();
    // Repeated output positions are ambiguous; numpy.expand_dims rejects them too.
    if let Some(w) = out_positions.windows(2).find(|w| w[0] == w[1]) {
        return Err(PyValueError::new_err(format!(
            "repeated axis {} in `axis` argument to insert_axis",
            w[0]
        )));
    }
    // Translate each output position into the core "gap index" (a position in the *input*
    // shape): the i-th smallest output position has exactly i new axes before it, so the
    // number of original dimensions preceding it - the gap index - is `position - i`.
    let gaps = out_positions
        .iter()
        .enumerate()
        .map(|(i, &pos)| pos - i)
        .collect::<DimArray<_>>();
    let ret = jix_core::ops::InsertAxis::new_array(array, gaps.as_slice()).into_py_result()?;
    let np_dtype = py_arr.get().dtype(py)?;
    Bound::new(
        py,
        Array::from_core_with_np_dtype(ret.into_any(), np_dtype.unbind()),
    )
}
/// Inserts new length-1 dimensions at specified positions in an array's shape. Alias for [`jix.insert_axis()`][jix.insert_axis].
///
/// Like `numpy.expand_dims`, each value in `axis` indexes the output (larger) shape.
///
/// Args:
///     array: Input array.
///     axis: Output-shape index or sequence of indices at which to place new length-1
///         dimensions. Negative values are resolved against the output ndim (`ndim + len(axis)`).
///
/// Returns:
///     A [`jix.Array`][jix.Array] with new length-1 axes inserted at the specified positions.
#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
pub fn unsqueeze<'py>(
    array: &Bound<'py, PyAny>,
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
/// The inverse operation is [`jix.insert_axis()`][jix.insert_axis] (also available as
/// [`jix.unsqueeze()`][jix.unsqueeze]). [`jix.squeeze()`][jix.squeeze] is a related variant
/// whose `axis` defaults to "every length-1 dimension".
///
/// Output dtype and total number of elements equal the input.
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// Args:
///     array: Input array.
///     axis: Axis index or sequence of axis indices to remove. Each must have size 1.
///         Negative values are supported.
///
/// Returns:
///     A [`jix.Array`][jix.Array] with the specified length-1 axes removed.
///
/// Examples:
///     ```python
///     import jix
///     import numpy as np
///
///     a = jix.compact([[1, 2, 3]], dtype=np.int32)  # shape [1, 3]
///     assert jix.remove_axis(a, 0).numpy().shape == (3,)     # -> [3]
///
///     b = jix.compact([[[10], [20]]], dtype=np.int32)  # shape [1, 2, 1]
///     assert jix.remove_axis(b, [0, 2]).numpy().shape == (2,)    # -> [2]
///     assert jix.remove_axis(b, [0, -1]).numpy().shape == (2,)   # negative axis
///     ```
#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
pub fn remove_axis<'py>(
    array: &Bound<'py, PyAny>,
    axis: ItemOrSequence<i32>,
) -> PyResult<Bound<'py, Array>> {
    let py_arr = asarray(array)?;
    let py = py_arr.py();
    let array = py_arr.get().to_core();
    let axes = normalize_axes(&axis.into_dim_array()?, array.ndim())?;
    if axes.is_empty() {
        return Ok(py_arr); // no-op if no axes to remove
    }
    let ret = jix_core::ops::RemoveAxis::new_array(array, axes.as_slice()).into_py_result()?;
    let np_dtype = py_arr.get().dtype(py)?;
    Bound::new(
        py,
        Array::from_core_with_np_dtype(ret.into_any(), np_dtype.unbind()),
    )
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
/// Args:
///     array: Input array.
///     axis: Axis or axes to remove. When `None` (default), all size-1 dimensions are
///         removed. Each named dimension must have size exactly 1.
///
/// Returns:
///     A [`jix.Array`][jix.Array] with length-1 axes removed.
///
/// Examples:
///     ```python
///     import jix
///     import numpy as np
///
///     a = jix.compact([[[1, 2, 3]]], dtype=np.int32)  # shape [1, 1, 3]
///     assert jix.squeeze(a).numpy().shape == (3,)              # remove all size-1 dims
///     assert jix.squeeze(a, axis=0).numpy().shape == (1, 3)    # remove only axis 0
///     assert jix.squeeze(a, axis=[0, 1]).numpy().shape == (3,) # remove axes 0 and 1
///     ```
#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (array, axis=None))]
pub fn squeeze<'py>(
    array: &Bound<'py, PyAny>,
    axis: Option<ItemOrSequence<i32>>,
) -> PyResult<Bound<'py, Array>> {
    let py_arr = asarray(array)?;
    let axis = axis.unwrap_or_else(|| {
        ItemOrSequence::Sequence(
            py_arr
                .get()
                .arr
                .shape()
                .iter()
                .enumerate()
                .filter_map(|(d, len)| (*len == 1).then_some(d as i32))
                .collect(),
        )
    });
    remove_axis(&py_arr, axis)
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
/// Args:
///     array: Input array.
///     axes: Permutation of axis indices. When `None` (default), reverses all axes.
///         Integer values must be unsigned (negative axes are not supported).
///
/// Returns:
///     A [`jix.Array`][jix.Array] with axes reordered as specified.
///
/// Examples:
///     ```python
///     import jix
///     import numpy as np
///
///     # 2-D transpose: [2, 3] -> [3, 2]
///     a = jix.asarray(np.arange(6, dtype=np.int32).reshape(2, 3))
///     t = jix.permute_axes(a, [1, 0])
///     assert t.numpy().shape == (3, 2)
///     assert np.array_equal(t.numpy(), a.numpy().T)
///
///     # axes=None reverses all axes (same as numpy.transpose with no argument)
///     assert jix.permute_axes(a).numpy().shape == (3, 2)
///     ```
#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (
    array,
    axes=None,
))]
pub fn permute_axes<'py>(
    array: &Bound<'py, PyAny>,
    axes: Option<Vec<usize>>,
) -> PyResult<Bound<'py, Array>> {
    let py_arr = asarray(array)?;
    let py = py_arr.py();
    let array = py_arr.get().to_core();
    let axes = axes.unwrap_or_else(|| (0..array.ndim()).rev().collect());
    if axes.len() == array.ndim() && axes.iter().enumerate().all(|(i, &ax)| i == ax) {
        return Ok(py_arr); // no-op permutation
    }
    let ret = jix_core::ops::PermuteAxes::new_array(array, &axes).into_py_result()?;
    let np_dtype = py_arr.get().dtype(py)?;
    Bound::new(
        py,
        Array::from_core_with_np_dtype(ret.into_any(), np_dtype.unbind()),
    )
}

/// Reinterprets an array with a different shape.
///
/// The total number of elements must be preserved: the product of `shape` must equal the
/// product of the original shape. Exactly one dimension in `shape` may be `-1`; that
/// dimension is inferred from the others and the total element count.
///
/// Output dtype equals the input dtype.
///
/// Like the other shape operations, the result is a lazy view; no data is copied until the array
/// is read. Reshape is uniquely prone to read-amplification, though: when the new shape is not
/// aligned with the original block boundaries, reading the view may decompress many more blocks
/// than the request appears to touch. When the result will be read more than once, materialize it
/// with [`jix.compact()`][jix.compact] to realign the block layout to `shape`.
///
/// Args:
///     array: Array to reshape.
///     shape: New shape. Exactly one dimension may be `-1` (inferred). Product must equal
///         the total element count.
///
/// Returns:
///     A lazy [`jix.Array`][jix.Array] view with the new shape.
///
/// Examples:
///     ```python
///     import jix
///     import numpy as np
///
///     a = jix.asarray(np.arange(6, dtype=np.int32).reshape(2, 3))  # shape [2, 3]
///
///     # Flatten
///     flat = jix.reshape(a, [6])
///     assert np.array_equal(flat.numpy(), [0, 1, 2, 3, 4, 5])
///
///     # Infer one dimension with -1
///     r = jix.reshape(a, [-1, 2])
///     assert r.numpy().shape == (3, 2)
///     ```
#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (
    array,
    shape,
))]
pub fn reshape<'py>(
    array: &Bound<'py, PyAny>,
    shape: ItemOrSequence<i64>,
) -> PyResult<Bound<'py, Array>> {
    let new_shape = shape;
    let py_arr = asarray(array)?;
    let py = py_arr.py();
    let array = &py_arr.get().arr;

    // handle -1 in new_shape
    let new_shape = {
        let mut new_shape = new_shape.into_dim_array()?;
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
        new_shape
            .iter()
            .map(|&dim| dim as u64)
            .collect::<DimArray<_>>()
    };

    if &new_shape == array.shape() {
        // no-op if already the right shape
        return Ok(py_arr);
    }

    let array = jix_core::ops::Reshape::new_array(array.clone(), new_shape.as_slice())
        .into_py_result()?
        .into_any();
    let np_dtype = py_arr.get().dtype(py)?;
    Bound::new(py, Array::from_core_with_np_dtype(array, np_dtype.unbind()))
}

/// Collapses an array into a single dimension.
///
/// Equivalent to [`jix.reshape(array, [n])`][jix.reshape] where `n` is the total number of
/// elements. Output dtype equals the input dtype. Output shape is `[n]`.
///
/// Like the other shape operations, the result is a lazy view; no data is copied until the array
/// is read. Reading the view may decompress many blocks if the original storage is block-based
/// and the flattened layout crosses block boundaries. When the result will be read more than
/// once, materialize it with [`jix.compact()`][jix.compact].
///
/// Args:
///     array: Array to flatten.
///
/// Returns:
///     A lazy 1-D [`jix.Array`][jix.Array] view containing all elements.
///
/// Examples:
///     ```python
///     import jix
///     import numpy as np
///
///     a = jix.compact([[1, 2, 3], [4, 5, 6]], dtype=np.int32)  # shape [2, 3]
///     f = jix.flatten(a)
///     assert f.numpy().shape == (6,)
///     assert np.array_equal(f.numpy(), [1, 2, 3, 4, 5, 6])
///     ```
#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
pub fn flatten<'py>(array: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Array>> {
    let py_arr = asarray(array)?;
    let size = py_arr.get().arr.shape().iter().product::<u64>();
    reshape(&py_arr, ItemOrSequence::Item(size as i64))
}

/// Joins a sequence of arrays along an existing axis.
///
/// All input arrays must have the same number of dimensions, the same dtype, and identical
/// sizes on every axis *except* the concatenation axis, along which their sizes may differ.
/// The output has the same number of dimensions as the inputs.
///
/// This function deviates from numpy in a few ways:
/// - all arrays must have the same dtype (numpy will upcast if they differ)
/// - all arrays must have the same number of dimensions (numpy will expand dims if they differ)
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// Args:
///     arrays: Sequence of arrays to concatenate. All must have the same dtype and number
///         of dimensions.
///     axis: Axis along which to concatenate. Supports negative values.
///
/// Returns:
///     A [`jix.Array`][jix.Array] formed by concatenating all inputs along the specified axis.
///
/// Examples:
///     ```python
///     import jix
///     import numpy as np
///
///     # 1-D: join end-to-end
///     a = jix.compact([1, 2, 3], dtype=np.int32)
///     b = jix.compact([4, 5], dtype=np.int32)
///     c = jix.concatenate([a, b])
///     assert np.array_equal(c.numpy(), [1, 2, 3, 4, 5])
///
///     # 2-D: append rows (axis 0) or columns (axis 1 / axis -1)
///     a = jix.compact([[1, 2], [3, 4]], dtype=np.int32)
///     b = jix.compact([[5, 6]], dtype=np.int32)
///     assert jix.concatenate([a, b], axis=0).numpy().shape == (3, 2)
///     assert jix.concatenate([a, b.T], axis=1).numpy().shape == (2, 3)
///     ```
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
    let py = py_arrays.first().unwrap().py();
    let ndim = arrays[0].ndim();
    for arr in &arrays {
        if arr.ndim() != ndim {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "All arrays must have the same number of dimensions",
            ));
        }
    }
    let axis = normalize_axis(axis, ndim)?;

    if arrays.len() == 1 && axis < ndim {
        // no-op if only one array
        let [array] = py_arrays.try_into().unwrap();
        return Ok(array);
    }
    let ret = jix_core::ops::Concatenate::new_array(arrays, axis).into_py_result()?;
    let np_dtype = py_arrays.first().unwrap().get().dtype(py)?;
    Bound::new(
        py,
        Array::from_core_with_np_dtype(ret.into_any(), np_dtype.unbind()),
    )
}

/// Joins a sequence of arrays along a **new** axis.
///
/// All input arrays must have identical shapes and the same dtype. A new axis of size equal
/// to the number of arrays is inserted at position `axis` in the output. The output has one
/// more dimension than the inputs - unlike [`jix.concatenate()`][jix.concatenate], which joins along an existing
/// axis.
///
/// This function deviates from numpy in a few ways:
/// - all arrays must have the same dtype (numpy will upcast if they differ)
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// Args:
///     arrays: Sequence of arrays to stack. All must have identical shapes and the same dtype.
///     axis: Position of the new axis in the output. Supports negative values.
///
/// Returns:
///     A [`jix.Array`][jix.Array] with a new axis inserted and the inputs stacked along it.
///
/// Examples:
///     ```python
///     import jix
///     import numpy as np
///
///     a = jix.compact([1, 2, 3], dtype=np.int32)
///     b = jix.compact([4, 5, 6], dtype=np.int32)
///
///     # Stack along a new leading axis -> shape [2, 3]
///     c = jix.stack([a, b], axis=0)
///     assert c.numpy().shape == (2, 3)
///     assert np.array_equal(c.numpy()[0], [1, 2, 3])
///
///     # Stack along a new trailing axis -> shape [3, 2]
///     d = jix.stack([a, b], axis=1)
///     assert d.numpy().shape == (3, 2)
///     assert np.array_equal(d.numpy()[:, 0], [1, 2, 3])
///     ```
#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (
    arrays,
    axis=0,
))]
pub fn stack<'py>(arrays: Vec<Bound<'py, PyAny>>, axis: i32) -> PyResult<Array> {
    let py_arrays = arrays
        .into_iter()
        .map(|arr| asarray(&arr))
        .collect::<Result<Vec<_>, _>>()?;
    let arrays = py_arrays
        .iter()
        .map(|arr| any_to_core_array(&arr))
        .collect::<Result<Vec<_>, _>>()?;
    if arrays.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "arrays must contain at least one array",
        ));
    }
    let py = py_arrays.first().unwrap().py();
    let ndim = arrays[0].ndim();
    for arr in &arrays {
        if arr.ndim() != ndim {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "All arrays must have the same number of dimensions",
            ));
        }
    }
    let axis = normalize_axis(axis, ndim + 1)?;
    let res = if arrays.len() == 1 && axis < ndim {
        // if only one array, equivalent to insert_axis along that axis
        let [array] = arrays.try_into().unwrap();
        let ret = jix_core::ops::InsertAxis::new_array(array, axis).into_py_result()?;
        ret.into_any()
    } else {
        let ret = jix_core::ops::Stack::new_array(arrays, axis).into_py_result()?;
        ret.into_any()
    };
    let np_dtype = py_arrays.first().unwrap().get().dtype(py)?;
    Ok(Array::from_core_with_np_dtype(res, np_dtype.unbind()))
}

/// Repeats each element along the given axis.
///
/// Every element along `axis` is replicated `repeats` times. The output has the
/// same number of dimensions as the input; only `shape[axis]` changes, from `n` to
/// `n * repeats`. `repeats == 0` produces an empty output (matches numpy). `repeats == 1`
/// is the identity.
///
/// This differs from [`jix.tile()`][jix.tile]: `repeat` duplicates each element in place
/// `(a, b, c) -> (a, a, b, b, c, c)`, whereas `tile` duplicates the whole sequence
/// `(a, b, c) -> (a, b, c, a, b, c)`.
///
/// Output dtype equals the input dtype. The result is a lazy view; no computation occurs
/// until the array is read.
///
/// Args:
///     array: Input array.
///     repeats: Number of times to repeat each element along `axis`. Must be non-negative.
///     axis: Axis along which to repeat. Supports negative values. `None` (default) is equivalent
///         to `axis=0` for 1-D arrays, unsupported for higher dimensions (deviates from numpy).
///
/// Returns:
///     A [`jix.Array`][jix.Array] with `shape[axis]` multiplied by `repeats`.
///
/// Examples:
///     ```python
///     import jix
///     import numpy as np
///
///     # 1-D: each element appears `repeats` times in a row
///     a = jix.compact([1, 2, 3], dtype=np.int32)
///     r = jix.repeat(a, 2)
///     assert np.array_equal(r.numpy(), [1, 1, 2, 2, 3, 3])
///
///     # 2-D along rows (axis=0): each row appears `repeats` times
///     b = jix.compact([[1, 2], [3, 4]], dtype=np.int32)
///     assert np.array_equal(
///         jix.repeat(b, 2, axis=0).numpy(),
///         [[1, 2], [1, 2], [3, 4], [3, 4]],
///     )
///
///     # 2-D along columns (axis=1 / axis=-1): each column appears `repeats` times
///     assert np.array_equal(
///         jix.repeat(b, 2, axis=-1).numpy(),
///         [[1, 1, 2, 2], [3, 3, 4, 4]],
///     )
///
///     # repeats=0 yields an empty array
///     assert jix.repeat(a, 0, axis=0).numpy().shape == (0,)
///     ```
#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
pub fn repeat<'py>(
    array: &Bound<'py, PyAny>,
    repeats: u64,
    axis: Option<i32>,
) -> PyResult<Bound<'py, Array>> {
    let py_arr = asarray(array)?;
    let py = py_arr.py();
    let array = py_arr.get().to_core();
    let axis = normalize_axis_optional(axis, array.ndim())?;
    if repeats == 1 {
        return Ok(py_arr); // no-op
    }
    let ret = jix_core::ops::Repeat::new_array(array, repeats, axis).into_py_result()?;
    let np_dtype = py_arr.get().dtype(py)?;
    Bound::new(
        py,
        Array::from_core_with_np_dtype(ret.into_any(), np_dtype.unbind()),
    )
}

/// Reverses the order of elements along the given axis.
///
/// Each named axis is independently reversed; non-named axes are left untouched. The shape
/// and dtype of the output equal the input. `axis` accepts an integer, a sequence of
/// integers, or `None` (the default) which reverses every axis. Negative indices are
/// supported. Duplicate axes are not allowed.
///
/// See also [`jix.roll()`][jix.roll], which cyclically shifts elements along an axis
/// without reversing them.
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// Args:
///     array: Input array.
///     axis: Axis or axes to reverse. Negative indices are supported. When `None` (the
///         default), all axes are reversed.
///
/// Returns:
///     A [`jix.Array`][jix.Array] with the specified axes reversed.
///
/// Examples:
///     ```python
///     import jix
///     import numpy as np
///
///     # 1-D: reverse the array
///     a = jix.compact([1, 2, 3, 4], dtype=np.int32)
///     assert np.array_equal(jix.flip(a).numpy(), [4, 3, 2, 1])
///
///     # 2-D: flip rows (axis=0)
///     b = jix.compact([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
///     assert np.array_equal(jix.flip(b, axis=0).numpy(), [[4, 5, 6], [1, 2, 3]])
///
///     # 2-D: flip columns (axis=1)
///     assert np.array_equal(jix.flip(b, axis=1).numpy(), [[3, 2, 1], [6, 5, 4]])
///
///     # 2-D: flip both axes (default behaviour with axis=None)
///     assert np.array_equal(jix.flip(b).numpy(), [[6, 5, 4], [3, 2, 1]])
///
///     # Sequence of axes (negative indices supported)
///     assert np.array_equal(jix.flip(b, axis=[-1]).numpy(), [[3, 2, 1], [6, 5, 4]])
///     ```
#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (array, axis=None))]
pub fn flip<'py>(
    array: &Bound<'py, PyAny>,
    axis: Option<ItemOrSequence<i32>>,
) -> PyResult<Bound<'py, Array>> {
    let py_arr = asarray(array)?;
    let py = py_arr.py();
    let array = py_arr.get().to_core();
    let ndim = array.ndim();
    let axis = axis.map(|a| a.into_dim_array()).transpose()?;
    let axis = axis.as_ref().map(|a| a.as_slice());
    let axes = normalize_axes_optional(axis, ndim)?;
    if axes.is_empty() {
        return Ok(py_arr); // no-op if no axes to flip
    }
    let ret = jix_core::ops::Flip::new_array(array, axes.as_slice()).into_py_result()?;
    let np_dtype = py_arr.get().dtype(py)?;
    Bound::new(
        py,
        Array::from_core_with_np_dtype(ret.into_any(), np_dtype.unbind()),
    )
}

/// Rolls elements along an axis, wrapping around at the boundary.
///
/// Elements pushed off the end of an axis re-enter at the beginning, so the shape and dtype
/// of the output equal the input. `shift` is taken modulo `shape[axis]`; positive shifts
/// move elements toward larger indices, negative shifts toward smaller indices.
///
/// See also [`jix.flip()`][jix.flip], which reverses element order along an axis without
/// wrapping.
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// Args:
///     array: Input array.
///     shift: Number of places to shift.
///     axis: Axis to roll. Negative indices are supported. When `None` (the default), the
///         input must be 1-D and the only axis is rolled. Unlike `numpy.roll`, this
///         function does not flatten higher-dimensional inputs when `axis` is omitted.
///
/// Returns:
///     A [`jix.Array`][jix.Array] with the same shape and dtype as the input.
///
/// Examples:
///     ```python
///     import jix
///     import numpy as np
///
///     # 1-D positive shift wraps the tail to the front.
///     a = jix.compact([0, 1, 2, 3, 4], dtype=np.int32)
///     assert np.array_equal(jix.roll(a, 2).numpy(), [3, 4, 0, 1, 2])
///
///     # Negative shift wraps the head to the back.
///     assert np.array_equal(jix.roll(a, -1).numpy(), [1, 2, 3, 4, 0])
///
///     # 2-D roll along an explicit axis (axis must be given when ndim > 1).
///     b = jix.compact([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
///     assert np.array_equal(jix.roll(b, 1, axis=0).numpy(), [[3, 4, 5], [0, 1, 2]])
///     assert np.array_equal(jix.roll(b, 1, axis=1).numpy(), [[2, 0, 1], [5, 3, 4]])
///     ```
#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (array, shift, axis=None))]
pub fn roll<'py>(
    array: &Bound<'py, PyAny>,
    shift: i64,
    axis: Option<i32>,
) -> PyResult<Bound<'py, Array>> {
    let py_arr = asarray(array)?;
    let py = py_arr.py();
    let core = py_arr.get().to_core();
    let axis = normalize_axis_optional(axis, core.ndim())?;
    if shift == 0 {
        return Ok(py_arr); // no-op if no shift
    }
    let ret = jix_core::ops::Roll::new_array(core, shift, axis).into_py_result()?;
    let np_dtype = py_arr.get().dtype(py)?;
    Bound::new(
        py,
        Array::from_core_with_np_dtype(ret.into_any(), np_dtype.unbind()),
    )
}

/// Replicates the array along a single axis.
///
/// The output shape matches the input except `shape[axis]` is multiplied by `repeats`. Element
/// `i` along the output axis comes from input element `i mod shape[axis]`, so the whole
/// sequence is repeated rather than each element in place. This differs from
/// [`jix.repeat()`][jix.repeat], which repeats each element. When the input axis already
/// has length 1, [`jix.broadcast()`][jix.broadcast] is a zero-cost alternative.
///
/// Unlike `numpy.tile`, this function only accepts a single integer `repeats` along a single
/// `axis`, and does not extend the array with new leading axes. `axis` must satisfy
/// `-ndim <= axis < ndim`.
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// Args:
///     array: Input array.
///     repeats: Number of times to repeat the array along `axis`. Must be non-negative.
///     axis: Axis to tile along. Negative indices are supported. When `None` (the default),
///         the input must be 1-D and the only axis is tiled.
///
/// Returns:
///     A [`jix.Array`][jix.Array] with `shape[axis]` multiplied by `repeats`.
///
/// Examples:
///     ```python
///     import jix
///     import numpy as np
///
///     # 1-D: the whole sequence is repeated `repeats` times
///     a = jix.compact([1, 2, 3], dtype=np.int32)
///     assert np.array_equal(jix.tile(a, 2).numpy(), [1, 2, 3, 1, 2, 3])
///
///     # 2-D along rows (axis=0): the matrix is stacked on top of itself
///     b = jix.compact([[1, 2], [3, 4]], dtype=np.int32)
///     assert np.array_equal(
///         jix.tile(b, 2, axis=0).numpy(),
///         [[1, 2], [3, 4], [1, 2], [3, 4]],
///     )
///
///     # 2-D along columns (axis=1): each row is repeated horizontally
///     assert np.array_equal(
///         jix.tile(b, 2, axis=1).numpy(),
///         [[1, 2, 1, 2], [3, 4, 3, 4]],
///     )
///
///     # repeats=0 yields an empty array along that axis
///     assert jix.tile(a, 0).numpy().shape == (0,)
///     ```
#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (array, repeats, axis=None))]
pub fn tile<'py>(
    array: &Bound<'py, PyAny>,
    repeats: u64,
    axis: Option<i32>,
) -> PyResult<Bound<'py, Array>> {
    let py_arr = asarray(array)?;
    let py = py_arr.py();
    let core = py_arr.get().to_core();
    let axis = normalize_axis_optional(axis, core.ndim())?;
    if repeats == 1 {
        return Ok(py_arr); // no-op
    }
    let ret = jix_core::ops::Tile::new_array(core, repeats, axis).into_py_result()?;
    let np_dtype = py_arr.get().dtype(py)?;
    Bound::new(
        py,
        Array::from_core_with_np_dtype(ret.into_any(), np_dtype.unbind()),
    )
}
