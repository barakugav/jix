use std::sync::{Arc, Mutex};

use numpy::{PyArrayDescr, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyEllipsis, PyTuple};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};
use zix_core::ops::SliceItem;
use zix_core::storage::ArrayStorage;
use zix_core::Array as ZixArray;

use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::types::{PyAnyMethods, PySlice};
use std::ops::Range;

use crate::codec::ReadContext;
use crate::dtype::dtype_to_numpy;
use crate::ops::NumpyAsArray;
use crate::storage::DynStorage;
use crate::util::{dim_arr, numpy_empty, DimArray, IntoPyResult, ItemOrSequence, OrKwargs};
use crate::ArrayParams;

/// A multi-dimensional compressed array.
///
/// `Array` is the central type in zix. It stores n-dimensional numeric data in a
/// block-compressed format — the array is divided into nd-blocks, each compressed
/// independently with Zstd. Data is decoded on demand: constructing an array or
/// chaining operations does no I/O. Actual decompression happens only when you
/// materialize the result, for example by indexing with `[]`, calling `.numpy()`,
/// `.copy()`, or `.write_to()`.
///
/// # Creating arrays
///
/// | Function | Description |
/// |---|---|
/// | `zix.compact(data)` | Compress any array-like (NumPy array, list, scalar) into a new zix array. |
/// | `zix.asarray(data)` | Create a zix array *view* of any array-like. Useful for mixing plain data with zix arrays in operations. |
/// | `zix.read_array(path)` | Load an array from a `.zix` file. |
///
/// ```python,ignore
/// import zix
/// import numpy as np
///
/// # From a NumPy array
/// a = zix.compact(np.arange(100, dtype=np.float32).reshape(10, 10))
///
/// # From a list
/// b = zix.compact([[1, 2, 3], [4, 5, 6]])
///
/// # From a file
/// c = zix.read_array("data.zix")
/// ```
///
/// # Reading data
///
/// Call `.numpy()` (or `[]`) to decode the array — or a sub-region of it — into a NumPy
/// array. The result is always a fresh allocation; mutations to it do not affect the
/// source array. Both forms accept the same index syntax:
///
/// ```python,ignore
/// a.numpy()           # full array
/// a.numpy(0)          # row 0 (integer drops that axis)
/// a.numpy(slice(1,4)) # rows 1–3 (slice keeps the axis)
/// a[0, 1:3]           # shorthand via __getitem__
/// a[..., -1]          # last column of any-rank array
/// ```
///
/// For tight loops that read many slices, pass an explicit [`zix.ReadContext`](crate::codec::ReadContext)
/// to amortize decompressor initialization:
///
/// ```python,ignore
/// ctx = a.read_ctx()
/// rows = [a.numpy(i, context=ctx) for i in range(len(a))]
/// ```
///
/// # Operations
///
/// Operations return a new `Array` that wraps the input and records the transformation.
/// No data is copied or computed at call time — the computation runs in a single pass when
/// you read the result. Chains compose without intermediate allocations:
///
/// ```python,ignore
/// result = (a.astype('float64') - mean).abs().sum(axis=0).numpy()
/// ```
///
/// ## Shape operations
///
/// Shape operations remap the array's indices without copying data. Most accept a `copy`
/// keyword (default `True`) that immediately re-encodes with a block layout suited to the
/// new shape. Pass `copy=False` for a zero-copy view — but be aware that if the new layout
/// crosses block boundaries that the original layout respected, reads may decompress more
/// data than necessary.
///
/// # Persistence
///
/// Save an array with [`.write_to(path)`](Array::write_to) or [`zix.write_array()`](crate::archive::write_array),
/// and reload it with [`zix.read_array()`](crate::archive::read_array). Multiple arrays can
/// be written back-to-back into a single file and read back by supplying `offset` and `len`.
///
/// ```python,ignore
/// a.write_to("data.zix")
/// b = zix.read_array("data.zix")
/// ```
///
/// # Copying and re-encoding
///
/// [`copy()`](Array::copy) materializes the current array (including any pending lazy
/// operations) into a new compressed array. This is also the way to tune the block layout
/// after shape-changing operations:
///
/// ```python,ignore
/// # Transpose and re-encode with a block layout suited for column access
/// b = zix.copy(a.T, params={"block_shape": [1024, 1]})
/// ```
///
/// # Architecture overview
///
/// Internally each `Array` holds a type-erased storage object that provides three things:
/// the array's shape, its element dtype, and the ability to read any rectangular sub-region
/// into a raw byte buffer. The primary concrete storage is a heap-allocated block-compressed
/// backend; a memory-mapped variant is also available. Every operation constructs a new
/// storage that wraps the input(s) and applies its transformation on each read request.
///
/// Because the entire operation chain is resolved at read time, there is no intermediate
/// allocation or data copy until you ask for output. The read releases the GIL while
/// decompressing, so Python threads can run concurrently.
#[gen_stub_pyclass]
#[pyclass(module = "zix", frozen)]
pub struct Array {
    pub(crate) arr: ZixArray<DynStorage>,
    cache: Mutex<ArrayCache>,
}
struct ArrayCache {
    numpy_dtype: Option<Py<PyArrayDescr>>,
}
impl Array {
    pub(crate) fn from_storage(storage: DynStorage) -> Self {
        Self {
            arr: ZixArray::from_storage(storage),
            cache: Mutex::new(ArrayCache { numpy_dtype: None }),
        }
    }

    pub(crate) fn from_core_storage(storage: impl ArrayStorage + Send + Sync + 'static) -> Self {
        Self::from_storage(DynStorage(Arc::new(storage)))
    }

    pub(crate) fn to_core_array(&self) -> ZixArray<DynStorage> {
        ZixArray::from_storage(self.arr.storage().clone())
    }

    fn to_numpy<'py>(
        &self,
        py: Python<'py>,
        index: &[Range<u64>],
        context: Option<&Bound<'py, ReadContext>>,
    ) -> PyResult<Bound<'py, PyUntypedArray>> {
        let arr_shape = self.arr.shape();
        let ndim = arr_shape.len();
        if index.len() != ndim {
            return Err(PyIndexError::new_err(format!(
                "index has {} dimensions, but array has {ndim}",
                index.len()
            )));
        }
        for (dim, r) in index.iter().enumerate() {
            if r.start > r.end || r.end > arr_shape[dim] {
                return Err(PyIndexError::new_err(format!(
                    "index {r:?} is out of bounds for axis {dim} with size {}",
                    arr_shape[dim]
                )));
            }
        }
        let read_shape = dim_arr(ndim, |dim| index[dim].end - index[dim].start);
        let itemsize = self.arr.dtype().itemsize() as usize;

        let np_arr = numpy_empty(self.dtype(py)?, &read_shape)?;
        let np_arr_data_ptr = unsafe { (*np_arr.as_array_ptr()).data.cast::<u8>() };
        let np_arr_data_size = itemsize * read_shape.iter().product::<u64>() as usize;
        let np_arr_data =
            unsafe { std::slice::from_raw_parts_mut(np_arr_data_ptr, np_arr_data_size) };

        let context = context.map(|ctx| ctx.get());

        if np_arr_data_size > 0 {
            py.detach(|| {
                let context_guard;
                let context = match context {
                    Some(ctx) => {
                        context_guard = ctx.lock();
                        &*context_guard
                    }
                    None => &self.arr.read_ctx(),
                };

                self.arr
                    .to_ndarray_buf(index, np_arr_data, context)
                    .into_py_result()
            })?;
        }

        let np_arr: Bound<'_, PyUntypedArray> = np_arr
            .call_method1("reshape", (read_shape.as_slice(),))?
            .cast_into()?;
        Ok(np_arr)
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl Array {
    /// The shape of the array: a tuple of axis lengths.
    ///
    /// ```python,ignore
    /// import zix
    ///
    /// a = zix.compact([1, 2, 3, 4])
    /// assert a.shape == (4,)
    ///
    /// b = zix.compact([[1, 2], [3, 4], [5, 6]])
    /// assert b.shape == (3, 2)
    /// ```
    #[getter]
    pub fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.arr.shape().iter().copied())
    }

    /// The number of dimensions (axes) of the array.
    ///
    /// ```python,ignore
    /// import zix
    ///
    /// a = zix.compact([1, 2, 3, 4])
    /// assert a.ndim == 1
    ///
    /// b = zix.compact([[1, 2], [3, 4], [5, 6]])
    /// assert b.ndim == 2
    /// ```
    #[getter]
    pub fn ndim(&self) -> usize {
        self.arr.shape().len()
    }

    /// The total number of elements in the array (the product of the axis lengths).
    ///
    /// ```python,ignore
    /// import zix
    ///
    /// a = zix.compact([1, 2, 3, 4])
    /// assert a.size == 4
    ///
    /// b = zix.compact([[1, 2], [3, 4], [5, 6]])
    /// assert b.size == 6
    /// ```
    #[getter]
    pub fn size(&self) -> PyResult<u64> {
        Ok(self.arr.shape().iter().product::<u64>())
    }

    /// The length of the array along the first axis (axis 0).
    ///
    /// ```python,ignore
    /// import zix
    ///
    /// a = zix.compact([1, 2, 3, 4])
    /// assert len(a) == 4
    ///
    /// b = zix.compact([[1, 2], [3, 4], [5, 6]])
    /// assert len(b) == 3
    /// ```
    pub fn __len__(&self) -> PyResult<usize> {
        let len = self
            .arr
            .shape()
            .first()
            .ok_or_else(|| PyValueError::new_err("zero-dimensional array has no length"))?;
        Ok(*len as usize)
    }

    /// The data type of the array elements, as a NumPy dtype object.
    ///
    /// ```python,ignore
    /// import zix
    /// import numpy as np
    ///
    /// a = zix.compact([1, 2, 3, 4], dtype='int32')
    /// assert a.dtype == np.dtype('int32')
    ///
    /// b = zix.compact([[1.0, 2.0], [3.0, 4.0]], dtype='float64')
    /// assert b.dtype == np.dtype('float64')
    /// ```
    #[getter]
    pub fn dtype<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDescr>> {
        let mut cache = self
            .cache
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Failed to acquire cache lock"))?;

        if cache.numpy_dtype.is_none() {
            cache.numpy_dtype = Some(dtype_to_numpy(py, self.arr.dtype())?.unbind());
        }
        Ok(cache.numpy_dtype.as_ref().unwrap().bind(py).clone())
    }

    /// Decode the array (or a sub-region of it) into a NumPy array.
    ///
    /// This is the primary way to materialize a `zix.Array` into a form that ordinary Python
    /// and NumPy code can consume. It decodes the compressed block data, copies it into a
    /// freshly-allocated NumPy array, and returns that array. The returned array is fully
    /// independent: mutations to one do not affect the other.
    ///
    /// # Return value
    ///
    /// * **dtype** — identical to `self.dtype`. No casting is performed.
    /// * **shape** — determined by the `index` argument (see below). When no index is
    ///   supplied the shape equals `self.shape`.
    /// * **memory layout** — always C-contiguous (row-major).
    /// * **ownership** — a brand-new allocation; the caller owns it outright.
    ///
    /// # The `index` argument
    ///
    /// `index` selects a sub-region to read. It accepts the same syntax Python uses inside `[…]`:
    ///
    /// | Form | Example | Effect |
    /// |---|---|---|
    /// | omitted / `None` argument | `arr.numpy()` | read the whole array |
    /// | integer | `arr[2]` or `arr.numpy(2)` | select a single position along axis 0 |
    /// | slice | `arr[1:4]` or `arr.numpy(slice(1, 4))` | select a range along axis 0 |
    /// | `...` (Ellipsis) | `arr[...]` or `arr.numpy(...)` | fill all remaining axes with full slices |
    /// | tuple of the above | `arr[0, 1:3, ..., :-2]` or `arr.numpy((0, slice(1,3)))` | index each axis independently |
    ///
    /// Most callers use `__getitem__` (`arr[…]`) instead of calling `numpy` directly; the two
    /// are equivalent.
    ///
    /// ## Integers
    ///
    /// An integer selects one position along the corresponding axis and **removes** that axis
    /// from the output shape (just like NumPy).
    ///
    /// * **Negative indices** are supported: `-1` means the last element, `-n` means the
    ///   first.
    /// * **Out-of-bounds** raises `IndexError`. The valid range is `[-len, len-1]` where
    ///   `len` is the size of that axis.
    ///
    /// ## Slices
    ///
    /// A slice selects a contiguous range along the corresponding axis and **keeps** that axis
    /// in the output (possibly with a smaller size).
    ///
    /// * `start` defaults to `0`; `stop` defaults to the axis length.
    /// * Both `start` and `stop` may be negative (counted from the end).
    /// * The **step must be 1** (explicit or omitted). Any other step raises `ValueError`.
    /// * **Bounds**: after normalizing negative values, `start` must satisfy
    ///   `0 ≤ start < len` and `stop` must satisfy `0 ≤ stop ≤ len`. This is stricter than
    ///   NumPy, which silently clamps out-of-range slice endpoints.
    /// * An empty slice (where `start == stop`) is valid and produces an axis of length 0.
    ///
    /// ## Ellipsis (`...`)
    ///
    /// Expands to as many full-range slices as needed to account for all axes not covered by
    /// the rest of the index. At most one ellipsis is allowed; a second one raises
    /// `IndexError`.
    ///
    /// ## Omitted trailing axes
    ///
    /// If the index covers fewer axes than the array has dimensions, the remaining axes are
    /// implicitly given full-range slices (equivalent to appending `...`).
    ///
    /// ## Too many indices
    ///
    /// If the number of integer/slice items exceeds the array's number of dimensions,
    /// `IndexError` is raised.
    ///
    /// ## Unsupported index types
    ///
    /// Anything other than an integer, slice, `...`, or tuple of these raises `TypeError`.
    ///
    /// # Errors
    ///
    /// | Exception | Condition |
    /// |---|---|
    /// | `IndexError` | integer out of bounds |
    /// | `IndexError` | slice `start` or `stop` out of bounds |
    /// | `IndexError` | more index items than array dimensions |
    /// | `IndexError` | more than one ellipsis |
    /// | `ValueError` | slice step other than 1 |
    /// | `TypeError` | unsupported index item type |
    ///
    /// # The `context` argument
    ///
    /// An optional `zix.ReadContext` to reuse across multiple reads. When omitted, a context
    /// is created internally for each call. Pass one explicitly when calling `numpy()` many
    /// times in a loop to avoid repeated decompressor initialization:
    ///
    /// ```python,ignore
    /// ctx = zix.ReadContext()
    /// rows = [a.numpy(i, context=ctx) for i in range(len(a))]
    /// ```
    #[pyo3(signature = (index=None, *, context=None))]
    pub fn numpy<'py>(
        &self,
        py: Python<'py>,
        index: Option<&Bound<'py, PyAny>>,
        context: Option<&Bound<'py, ReadContext>>,
    ) -> PyResult<Bound<'py, PyUntypedArray>> {
        let shape = self.arr.shape();
        let ndim = shape.len();

        enum RawIdxItem {
            Int(i64),
            Slice(SliceItem),
            Ellipsis,
        }

        enum IdxItem {
            Int(u64),        // already resolved, consumes a real axis, drops it
            Slice(u64, u64), // already resolved, consumes a real axis, keeps it
        }

        // 1. Normalize index into a tuple of items.
        let raw = match index {
            Some(index) => {
                if let Ok(tup) = index.cast::<PyTuple>() {
                    tup.iter().collect::<Vec<_>>()
                } else {
                    vec![index.clone()]
                }
            }
            None => vec![],
        };
        let raw = raw
            .into_iter()
            .map(|item| {
                if item.is_instance_of::<PyEllipsis>() {
                    return Ok(RawIdxItem::Ellipsis);
                }
                if let Ok(slice) = item.cast::<PySlice>() {
                    // Pull start/stop/step as Option<i64>. PySlice exposes them as attrs
                    // that are either int or None.
                    let start = slice.getattr("start")?.extract::<Option<i64>>()?;
                    let stop = slice.getattr("stop")?.extract::<Option<i64>>()?;
                    let step = slice
                        .getattr("step")?
                        .extract::<Option<i64>>()?
                        .unwrap_or(1);
                    return Ok(RawIdxItem::Slice(SliceItem {
                        start,
                        end: stop,
                        step,
                    }));
                }
                if let Ok(i) = item.extract::<i64>() {
                    return Ok(RawIdxItem::Int(i));
                }
                Err(PyTypeError::new_err(
                    "only integers, slices (`:`), and ellipsis (`...`) are valid indices",
                ))
            })
            .collect::<PyResult<Vec<_>>>()?;

        // 2. Validate ellipsis count and real-axis-consumer count.
        let mut ellipsis_count = 0usize;
        let mut consumers = 0usize; // Int + Slice + Ellipsis-as-one-slot
        for r in &raw {
            match r {
                RawIdxItem::Ellipsis => ellipsis_count += 1,
                RawIdxItem::Int(_) | RawIdxItem::Slice(_) => consumers += 1,
            }
        }
        if ellipsis_count > 1 {
            return Err(PyIndexError::new_err(
                "an index can only have a single ellipsis ('...')",
            ));
        }
        if consumers > ndim {
            return Err(PyIndexError::new_err(format!(
                "too many indices for array: array is {ndim}-dimensional, \
                 but {consumers} were indexed"
            )));
        }

        // 3. Expand ellipsis (or pad at the end if absent) so we get exactly
        //    ndim real-axis consumers. NewAxis entries pass through.
        let fill = ndim - consumers;
        let mut axes: Vec<IdxItem> = Vec::with_capacity(ndim);
        let mut axis_cursor = 0usize; // which real axis we're on
        for r in raw {
            match r {
                RawIdxItem::Ellipsis => {
                    for _ in 0..fill {
                        axes.push(IdxItem::Slice(0, shape[axis_cursor]));
                        axis_cursor += 1;
                    }
                }
                RawIdxItem::Int(i) => {
                    let len = shape[axis_cursor] as i64;
                    let i = if i < 0 { i + len } else { i };
                    if i < 0 || i >= len {
                        return Err(PyIndexError::new_err(format!(
                            "index {i} is out of bounds for axis {axis_cursor} with size {len}"
                        )));
                    }
                    axes.push(IdxItem::Int(i as u64));
                    axis_cursor += 1;
                }
                RawIdxItem::Slice(s) => {
                    if s.step != 1 {
                        return Err(PyValueError::new_err("slice step must be 1"));
                    }
                    let len = shape[axis_cursor] as i64;
                    let start = s.start.unwrap_or(0);
                    let stop = s.end.unwrap_or(len);
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
                    axes.push(IdxItem::Slice(start_norm as u64, stop_norm as u64));
                    axis_cursor += 1;
                }
            }
        }
        // fewer consumers than ndim: pad at the end.
        while axis_cursor < ndim {
            axes.push(IdxItem::Slice(0, shape[axis_cursor]));
            axis_cursor += 1;
        }

        // 4. Build the ranges for get_data (length == ndim) and the output
        //    shape (length == number of kept axes + NewAxis entries).
        let mut ranges = DimArray::new();
        let mut out_shape: Vec<usize> = Vec::with_capacity(axes.len());
        for ax in &axes {
            match ax {
                IdxItem::Int(i) => ranges.push(*i..*i + 1),
                IdxItem::Slice(s, e) => {
                    ranges.push(*s..*e);
                    out_shape.push((e - s) as usize);
                }
            }
        }

        // 5. Read data
        let np_arr = self.to_numpy(py, &ranges, context)?;

        let np_arr: Bound<'_, PyUntypedArray> =
            np_arr.call_method1("reshape", (out_shape,))?.cast_into()?;
        Ok(np_arr)
    }

    /// Read elements from the array (or a sub-region of it) and return them as a NumPy array.
    ///
    /// This function is identical to `numpy()`, see that method for details.
    fn __getitem__<'py>(&self, key: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyUntypedArray>> {
        self.numpy(key.py(), Some(key), None)
    }

    /// Creates a `zix.ReadContext` with decoder parameters derived from this array's storage.
    ///
    /// The returned context inherits the decoder configuration stored alongside the array data,
    /// ensuring that reads use the same settings the array was written with. Prefer this over
    /// constructing `zix.ReadContext()` directly when reading a specific array.
    ///
    /// Pass the returned context to `Array.numpy()` or `zix.copy()` to amortize decompressor
    /// initialization across many successive reads. See `zix.ReadContext` for details.
    ///
    /// ```python,ignore
    /// import zix
    /// import numpy as np
    ///
    /// a = zix.compact(np.arange(30, dtype=np.int32).reshape(10, 3))
    /// ctx = a.read_ctx()
    /// rows = [a.numpy(i, context=ctx) for i in range(len(a))]
    /// ```
    fn read_ctx<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, ReadContext>> {
        Bound::new(py, ReadContext::from_core(self.arr.read_ctx()))
    }

    /// Copies the data of an array into a new compact array by compressing it into new blocks. See :func:`zix.copy()`.
    #[pyo3(signature = (*, params=None, context=None))]
    fn copy<'py>(
        slf: &Bound<'py, Self>,
        params: Option<OrKwargs<Bound<'_, ArrayParams>>>,
        context: Option<&Bound<'_, ReadContext>>,
    ) -> PyResult<Bound<'py, Array>> {
        crate::ops::copy(slf, params, context)
    }

    // == archive I/O ==

    /// Write the array to a file or a file-like object. See :func:`zix.write_array`.
    #[pyo3(signature = (path_or_writer, *, append=false, params=None, context=None))]
    pub fn write_to(
        slf: &Bound<'_, Array>,
        path_or_writer: &Bound<'_, PyAny>,
        append: bool,
        params: Option<OrKwargs<Bound<'_, ArrayParams>>>,
        context: Option<&Bound<'_, ReadContext>>,
    ) -> PyResult<()> {
        crate::archive::write_array(slf, path_or_writer, append, params, context)
    }

    // == arithmetic ops ==

    /// Element-wise addition of two arrays. See :func:`zix.add()`.
    pub fn add(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        crate::ops::add(slf, other)
    }

    /// Element-wise addition of two arrays. See :func:`zix.add()`.
    pub fn __add__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        crate::ops::add(slf, other)
    }

    /// Element-wise addition of two arrays. See :func:`zix.add()`.
    pub fn __radd__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        crate::ops::add(other, slf)
    }

    /// Element-wise subtraction of two arrays. See :func:`zix.subtract()`.
    pub fn subtract(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        crate::ops::subtract(slf, other)
    }

    /// Element-wise subtraction of two arrays. See :func:`zix.subtract()`.
    pub fn __sub__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        crate::ops::subtract(slf, other)
    }

    /// Element-wise subtraction of two arrays. See :func:`zix.subtract()`.
    pub fn __rsub__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        crate::ops::subtract(other, slf)
    }

    /// Element-wise multiplication of two arrays. See :func:`zix.multiply()`.
    pub fn multiply(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        crate::ops::multiply(slf, other)
    }

    /// Element-wise multiplication of two arrays. See :func:`zix.multiply()`.
    pub fn __mul__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        crate::ops::multiply(slf, other)
    }

    /// Element-wise multiplication of two arrays. See :func:`zix.multiply()`.
    pub fn __rmul__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        crate::ops::multiply(other, slf)
    }

    /// Element-wise division of two arrays. See :func:`zix.divide()`.
    pub fn divide(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        crate::ops::divide(slf, other)
    }

    /// Element-wise division of two arrays. See :func:`zix.divide()`.
    pub fn __truediv__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        crate::ops::divide(slf, other)
    }

    /// Element-wise division of two arrays. See :func:`zix.divide()`.
    pub fn __rtruediv__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        crate::ops::divide(other, slf)
    }

    // TODO: __pow__

    /// Arithmetic negation applied element-wise. See :func:`zix.negative()`.
    pub fn negative(slf: &Bound<'_, Self>) -> PyResult<Self> {
        crate::ops::negative(slf)
    }

    /// Arithmetic negation applied element-wise. See :func:`zix.negative()`.
    pub fn __neg__<'py>(slf: &Bound<'py, Self>) -> PyResult<Self> {
        crate::ops::negative(slf)
    }

    /// Computes the absolute value of each element. See :func:`zix.absolute()`.
    pub fn abs(slf: &Bound<'_, Self>) -> PyResult<Self> {
        crate::ops::absolute(slf)
    }

    /// Computes the absolute value of each element. See :func:`zix.absolute()`.
    pub fn __abs__<'py>(slf: &Bound<'py, Self>) -> PyResult<Self> {
        crate::ops::absolute(slf)
    }

    /// Computes the natural exponential (`e^x`) of each element. See :func:`zix.exp()`.
    pub fn exp(slf: &Bound<'_, Self>) -> PyResult<Self> {
        crate::ops::exp(slf)
    }

    /// Computes the square root of each element. See :func:`zix.sqrt()`.
    pub fn sqrt(slf: &Bound<'_, Self>) -> PyResult<Self> {
        crate::ops::sqrt(slf)
    }

    /// Rounds each element up to the nearest integer (towards +∞). See :func:`zix.ceil()`.
    pub fn ceil(slf: &Bound<'_, Self>) -> PyResult<Self> {
        crate::ops::ceil(slf)
    }

    /// Rounds each element down to the nearest integer (towards −∞). See :func:`zix.floor()`.
    pub fn floor(slf: &Bound<'_, Self>) -> PyResult<Self> {
        crate::ops::floor(slf)
    }

    /// Rounds each element to the nearest integer. See :func:`zix.round()`.
    pub fn round(slf: &Bound<'_, Self>) -> PyResult<Self> {
        crate::ops::round(slf)
    }

    // == bitwise ops ==

    /// Element-wise bitwise AND of two arrays. See :func:`zix.bitwise_and()`.
    pub fn __and__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        crate::ops::bitwise_and(slf, other)
    }

    /// Element-wise bitwise AND of two arrays. See :func:`zix.bitwise_and()`.
    pub fn __rand__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        crate::ops::bitwise_and(other, slf)
    }

    /// Element-wise bitwise OR of two arrays. See :func:`zix.bitwise_or()`.
    pub fn __or__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        crate::ops::bitwise_or(slf, other)
    }

    /// Element-wise bitwise OR of two arrays. See :func:`zix.bitwise_or()`.
    pub fn __ror__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        crate::ops::bitwise_or(other, slf)
    }

    /// Element-wise bitwise XOR of two arrays. See :func:`zix.bitwise_xor()`.
    pub fn __xor__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        crate::ops::bitwise_xor(slf, other)
    }

    /// Element-wise bitwise XOR of two arrays. See :func:`zix.bitwise_xor()`.
    pub fn __rxor__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        crate::ops::bitwise_xor(other, slf)
    }

    /// Element-wise bitwise NOT (one's complement). See :func:`zix.bitwise_not()`.
    pub fn __invert__<'py>(slf: &Bound<'py, Self>) -> PyResult<Self> {
        crate::ops::bitwise_not(slf)
    }

    /// Element-wise left shift (`a << b`). See :func:`zix.bitwise_shift_left()`.
    pub fn __lshift__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        crate::ops::bitwise_shift_left(slf, other)
    }

    /// Element-wise left shift (`a << b`). See :func:`zix.bitwise_shift_left()`.
    pub fn __rlshift__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        crate::ops::bitwise_shift_left(other, slf)
    }

    // == comparison ops ==

    /// Element-wise less-than test (`a < b`). See :func:`zix.less()`.
    pub fn __lt__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        crate::ops::less(slf, other)
    }

    /// Element-wise less-than-or-equal test (`a <= b`). See :func:`zix.less_equal()`.
    pub fn __le__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        crate::ops::less_equal(slf, other)
    }

    /// Element-wise greater-than test (`a > b`). See :func:`zix.greater()`.
    pub fn __gt__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        crate::ops::greater(slf, other)
    }

    /// Element-wise greater-than-or-equal test (`a >= b`). See :func:`zix.greater_equal()`.
    pub fn __ge__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        crate::ops::greater_equal(slf, other)
    }

    /// Element-wise equality test (`a == b`). See :func:`zix.equal()`.
    pub fn __eq__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        crate::ops::equal(slf, other)
    }

    /// Element-wise inequality test (`a != b`). See :func:`zix.not_equal()`.
    pub fn __ne__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        crate::ops::not_equal(slf, other)
    }

    // == reduction ops ==

    /// Reduces one or more axes with logical AND: returns `True` if all elements are truthy. See :func:`zix.all()`.
    #[pyo3(signature = (axis=None, keepdims=false))]
    pub fn all(
        slf: &Bound<'_, Self>,
        axis: Option<ItemOrSequence<i32>>,
        keepdims: bool,
    ) -> PyResult<Self> {
        crate::ops::all(slf, axis, keepdims)
    }

    /// Reduces one or more axes with logical OR: returns `True` if any element is truthy. See :func:`zix.any()`.
    #[pyo3(signature = (axis=None, keepdims=false))]
    pub fn any(
        slf: &Bound<'_, Self>,
        axis: Option<ItemOrSequence<i32>>,
        keepdims: bool,
    ) -> PyResult<Self> {
        crate::ops::any(slf, axis, keepdims)
    }

    /// Reduces one or more axes by taking the maximum element. See :func:`zix.max()`.
    #[pyo3(signature = (axis=None, keepdims=false))]
    pub fn max(
        slf: &Bound<'_, Self>,
        axis: Option<ItemOrSequence<i32>>,
        keepdims: bool,
    ) -> PyResult<Self> {
        crate::ops::max(slf, axis, keepdims)
    }

    /// Reduces one or more axes by taking the minimum element. See :func:`zix.min()`.
    #[pyo3(signature = (axis=None, keepdims=false))]
    pub fn min(
        slf: &Bound<'_, Self>,
        axis: Option<ItemOrSequence<i32>>,
        keepdims: bool,
    ) -> PyResult<Self> {
        crate::ops::min(slf, axis, keepdims)
    }

    /// Returns the index of the maximum element along a single axis. See :func:`zix.argmax()`.
    #[pyo3(signature = (axis=None, keepdims=false))]
    pub fn argmax(slf: &Bound<'_, Self>, axis: Option<i32>, keepdims: bool) -> PyResult<Self> {
        crate::ops::argmax(slf, axis, keepdims)
    }

    /// Returns the index of the minimum element along a single axis. See :func:`zix.argmin()`.
    #[pyo3(signature = (axis=None, keepdims=false))]
    pub fn argmin(slf: &Bound<'_, Self>, axis: Option<i32>, keepdims: bool) -> PyResult<Self> {
        crate::ops::argmin(slf, axis, keepdims)
    }

    /// Reduces one or more axes by summing all elements. See :func:`zix.sum()`.
    #[pyo3(signature = (axis=None, keepdims=false))]
    pub fn sum(
        slf: &Bound<'_, Self>,
        axis: Option<ItemOrSequence<i32>>,
        keepdims: bool,
    ) -> PyResult<Self> {
        crate::ops::sum(slf, axis, keepdims)
    }

    /// Computes the arithmetic mean along one or more axes. See :func:`zix.mean()`.
    #[pyo3(signature = (axis=None, keepdims=false))]
    pub fn mean(
        slf: &Bound<'_, Self>,
        axis: Option<ItemOrSequence<i32>>,
        keepdims: bool,
    ) -> PyResult<Self> {
        crate::ops::mean(slf, axis, keepdims)
    }

    /// Reduces one or more axes by multiplying all elements. See :func:`zix.product()`.
    #[pyo3(signature = (axis=None, keepdims=false))]
    pub fn prod(
        slf: &Bound<'_, Self>,
        axis: Option<ItemOrSequence<i32>>,
        keepdims: bool,
    ) -> PyResult<Self> {
        crate::ops::product(slf, axis, keepdims)
    }

    /// Computes the standard deviation along one or more axes. See :func:`zix.std()`.
    #[pyo3(signature = (axis=None, keepdims=false, ddof=0.0))]
    pub fn std(
        slf: &Bound<'_, Self>,
        axis: Option<ItemOrSequence<i32>>,
        keepdims: bool,
        ddof: f64,
    ) -> PyResult<Self> {
        crate::ops::std(slf, axis, keepdims, ddof)
    }

    /// Computes the variance along one or more axes. See :func:`zix.var()`.
    #[pyo3(signature = (axis=None, keepdims=false, ddof=0.0))]
    pub fn var(
        slf: &Bound<'_, Self>,
        axis: Option<ItemOrSequence<i32>>,
        keepdims: bool,
        ddof: f64,
    ) -> PyResult<Self> {
        crate::ops::var(slf, axis, keepdims, ddof)
    }

    /// Casts each element of the array to a new dtype. See :func:`zix.astype()`.
    pub fn astype<'py>(
        slf: &Bound<'py, Self>,
        dtype: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        crate::ops::astype(slf, dtype)
    }

    // == shape ops ==

    /// Reinterprets an array with a different shape. See :func:`zix.reshape()`.
    #[pyo3(signature = (shape, *, copy=true))]
    pub fn reshape<'py>(
        slf: &Bound<'py, Self>,
        shape: ItemOrSequence<i64>,
        copy: bool,
    ) -> PyResult<Bound<'py, Self>> {
        crate::ops::reshape(slf, shape, copy)
    }

    /// Collapses the array into a single dimension. See :func:`zix.flatten()`.
    #[pyo3(signature = (*, copy=true))]
    pub fn flatten<'py>(slf: &Bound<'py, Self>, copy: bool) -> PyResult<Bound<'py, Self>> {
        crate::ops::flatten(slf, copy)
    }

    /// Reorders the axes of an array (generalized transpose). See :func:`zix.permute_axes()`.
    #[pyo3(signature = (axes=None))]
    pub fn permute_axes<'py>(
        slf: &Bound<'py, Self>,
        axes: Option<Vec<usize>>,
    ) -> PyResult<Bound<'py, Self>> {
        crate::ops::permute_axes(slf, axes)
    }

    /// Reverses all axes; shorthand for `permute_axes()` with no arguments. See :func:`zix.permute_axes()`.
    #[allow(non_snake_case)]
    #[getter]
    pub fn T<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, Array>> {
        crate::ops::permute_axes(slf, None)
    }

    /// Expands the array to a larger shape by repeating elements along length-1 dimensions. See :func:`zix.broadcast()`.
    #[pyo3(signature = (shape, *, copy=true))]
    pub fn broadcast<'py>(
        slf: &Bound<'py, Array>,
        shape: ItemOrSequence<i64>,
        copy: bool,
    ) -> PyResult<Bound<'py, Array>> {
        crate::ops::broadcast(slf, shape, copy)
    }

    /// Removes length-1 dimensions from the array's shape. See :func:`zix.squeeze()`.
    #[pyo3(signature = (axis=None))]
    pub fn squeeze<'py>(
        slf: &Bound<'py, Array>,
        axis: Option<ItemOrSequence<i32>>,
    ) -> PyResult<Bound<'py, Array>> {
        crate::ops::squeeze(slf, axis)
    }

    /// Inserts new length-1 dimensions at specified positions in the array's shape. See :func:`zix.unsqueeze()`.
    pub fn unsqueeze<'py>(
        slf: &Bound<'py, Array>,
        axis: ItemOrSequence<i32>,
    ) -> PyResult<Bound<'py, Array>> {
        crate::ops::unsqueeze(slf, axis)
    }

    // == trigonometric ops ==

    /// Computes the sine of each element (input in radians). See :func:`zix.sin()`.
    pub fn sin(slf: &Bound<'_, Self>) -> PyResult<Self> {
        crate::ops::sin(slf)
    }

    /// Computes the cosine of each element (input in radians). See :func:`zix.cos()`.
    pub fn cos(slf: &Bound<'_, Self>) -> PyResult<Self> {
        crate::ops::cos(slf)
    }

    /// Computes the tangent of each element (input in radians). See :func:`zix.tan()`.
    pub fn tan(slf: &Bound<'_, Self>) -> PyResult<Self> {
        crate::ops::tan(slf)
    }

    /// Computes the arcsine of each element; output is in radians in `[-π/2, π/2]`. See :func:`zix.asin()`.
    pub fn asin(slf: &Bound<'_, Self>) -> PyResult<Self> {
        crate::ops::asin(slf)
    }

    /// Computes the arccosine of each element; output is in radians in `[0, π]`. See :func:`zix.acos()`.
    pub fn acos(slf: &Bound<'_, Self>) -> PyResult<Self> {
        crate::ops::acos(slf)
    }

    /// Computes the arctangent of each element; output is in radians in `(-π/2, π/2)`. See :func:`zix.atan()`.
    pub fn atan(slf: &Bound<'_, Self>) -> PyResult<Self> {
        crate::ops::atan(slf)
    }
}

/// Compact any array-like object to a new zix [`Array`].
///
/// Accepts Python scalars, lists, tuples, NumPy arrays, and any other object accepted by
/// [`numpy.asarray`](https://numpy.org/doc/stable/reference/generated/numpy.asarray.html).
///
/// A new zix compact array is created, with all the input data compressed into blocks. The data
/// is compressed even if the input is already a zix array.
///
/// `params` controls the block layout and codec configuration. It accepts either a
/// `zix.ArrayParams` instance or a plain `dict` (e.g. `{"block_shape": [64, 64]}`). When
/// omitted, defaults are chosen automatically. See `zix.ArrayParams` for details.
///
/// # Errors
///
/// - If the input is not a zix array and it cannot be converted by `numpy.asarray`.
/// - If the array has more dimensions than zix supports.
/// - If the array has negative strides (e.g. a reversed slice `a[::-1]`).
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (array, dtype=None, *, params=None))]
pub fn compact(
    array: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
    params: Option<OrKwargs<Bound<'_, ArrayParams>>>,
) -> PyResult<Array> {
    let py = array.py();
    let params = ArrayParams::resolve(py, params)?;

    // already a zix array
    if let Ok(array) = array.cast::<Array>() {
        let mut array = array.clone();
        if let Some(dtype) = dtype {
            array = crate::ops::astype(&array, dtype)?;
        }
        let array = &array.get().arr;
        let array = py.detach(|| array.copy_with(params, &array.read_ctx()).into_py_result())?;
        return Ok(Array::from_core_storage(array.into_storage()));
    }

    // convert to numpy array
    let py = array.py();
    let numpy_asarray = numpy::get_array_module(py)?.getattr("asarray")?;
    let array = numpy_asarray.call1((array, dtype))?;
    let array = array.cast::<PyUntypedArray>()?;
    let array = NumpyAsArray::new(array)?;

    let array = py.detach({
        || {
            let array = match array {
                NumpyAsArray::Numpy(array) => {
                    let context = array.read_ctx();
                    array.copy_with(params, &context)
                }
                // scalar
                _ => {
                    let array = array.into_py_array(None)?;
                    let context = array.arr.read_ctx();
                    array.arr.copy_with(params, &context)
                }
            };
            array.into_py_result()
        }
    })?;
    return Ok(Array::from_core_storage(array.into_storage()));
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ndarray::{array, ArrayD, IxDyn};
    use numpy::{PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
    use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
    use pyo3::types::{PyAny, PyEllipsis, PySlice, PyTuple};
    use pyo3::{Bound, IntoPyObject, Python};
    use zix_core::dtype::Dtyped;
    use zix_core::Array as ZixArray;

    use super::{Array, DynStorage};

    fn make_py_array<'py, T: Dtyped, D>(
        py: Python<'py>,
        ndarray: &ndarray::Array<T, D>,
    ) -> Bound<'py, Array>
    where
        D: ndarray::Dimension,
    {
        let core = ZixArray::compact_array(ndarray).unwrap();
        let dyn_storage = DynStorage(Arc::new(core.into_storage()));
        Bound::new(py, Array::from_storage(dyn_storage)).unwrap()
    }

    fn roundtrip<T, D>(original: &ndarray::Array<T, D>) -> ArrayD<T>
    where
        T: Dtyped + numpy::Element + Copy,
        D: ndarray::Dimension,
    {
        // ndarray::Array -> zix_core::Array -> zix_python::Array -> numpy::PyArray -> ndarray::Array
        Python::attach(|py| {
            let py_arr = make_py_array(py, &original);
            let np = py_arr.get().numpy(py, None, None).unwrap();
            let typed = np.cast_into::<PyArrayDyn<T>>().unwrap();
            typed.to_owned_array()
        })
    }

    #[test]
    fn test_numpy_f32_1d() {
        let original = array![1.0f32, 2.0, 3.0, 4.0];
        assert_eq!(roundtrip(&original), original.into_dyn());
    }

    #[test]
    fn test_numpy_f32_2d() {
        let original = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        assert_eq!(roundtrip(&original), original.into_dyn());
    }

    #[test]
    fn test_numpy_f32_3d() {
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let original = ndarray::Array::from_shape_vec(IxDyn(&[2, 3, 4]), data).unwrap();
        assert_eq!(roundtrip(&original), original);
    }

    #[test]
    fn test_numpy_f64_2d() {
        let original: ArrayD<f64> =
            ndarray::Array::from_shape_vec(IxDyn(&[3, 4]), (0..12).map(|x| x as f64).collect())
                .unwrap();
        assert_eq!(roundtrip(&original), original);
    }

    #[test]
    fn test_numpy_i32_2d() {
        let original = ndarray::Array::from_shape_vec(IxDyn(&[4, 5]), (0..20).collect()).unwrap();
        assert_eq!(roundtrip(&original), original);
    }

    #[test]
    fn test_numpy_i64_1d() {
        let original: ArrayD<i64> =
            ndarray::Array::from_shape_vec(IxDyn(&[8]), (100..108).collect()).unwrap();
        assert_eq!(roundtrip(&original), original);
    }

    #[test]
    fn test_numpy_u8_2d() {
        let original: ArrayD<u8> =
            ndarray::Array::from_shape_vec(IxDyn(&[3, 3]), (0u8..9).collect()).unwrap();
        assert_eq!(roundtrip(&original), original);
    }

    #[test]
    fn test_numpy_u32_3d() {
        let data: Vec<u32> = (0..60).collect();
        let original: ArrayD<u32> =
            ndarray::Array::from_shape_vec(IxDyn(&[3, 4, 5]), data).unwrap();
        assert_eq!(roundtrip(&original), original);
    }

    #[test]
    fn test_numpy_bool_1d() {
        let original = array![true, false, true, true, false, true];
        assert_eq!(roundtrip(&original), original.into_dyn());
    }

    #[test]
    fn test_numpy_large_values_f64() {
        // Verify large/negative values are transferred without corruption.
        let original = array![[f64::MAX, f64::MIN, -1.0], [0.0, 1.0, f64::INFINITY]];
        assert_eq!(roundtrip(&original), original.into_dyn());
    }

    #[test]
    fn test_numpy_shape_preserved() {
        let original =
            ndarray::Array::from_shape_vec(IxDyn(&[2, 3, 4]), (0..24).map(|x| x as f32).collect())
                .unwrap();
        Python::attach(|py| {
            let py_arr = make_py_array(py, &original);
            let np = py_arr.get().numpy(py, None, None).unwrap();
            assert_eq!(np.shape(), &[2usize, 3, 4]);
        });
    }

    #[test]
    fn test_numpy_dtype_preserved_f32() {
        use numpy::PyArrayDescrMethods;
        let original = array![1.0f32, 2.0];
        Python::attach(|py| {
            let py_arr = make_py_array(py, &original);
            let np = py_arr.get().numpy(py, None, None).unwrap();
            assert_eq!(np.dtype().itemsize(), 4);
            assert_eq!(np.dtype().kind() as char, 'f');
        });
    }

    #[test]
    fn test_numpy_dtype_preserved_i32() {
        use numpy::PyArrayDescrMethods;
        let original = array![1i32, 2, 3];
        Python::attach(|py| {
            let py_arr = make_py_array(py, &original);
            let np = py_arr.get().numpy(py, None, None).unwrap();
            assert_eq!(np.dtype().itemsize(), 4);
            assert_eq!(np.dtype().kind() as char, 'i');
        });
    }

    #[test]
    fn test_numpy_single_element() {
        let original = array![42.0f64];
        assert_eq!(roundtrip(&original), original.into_dyn());
    }

    #[test]
    fn test_numpy_non_square_2d() {
        let data: Vec<f32> = (0..100).map(|x| x as f32).collect();
        let original = ndarray::Array::from_shape_vec(IxDyn(&[10, 10]), data).unwrap();
        assert_eq!(roundtrip(&original), original);
    }

    fn eval<T>(py_arr: Array) -> ArrayD<T>
    where
        T: Dtyped + numpy::Element + Copy,
    {
        Python::attach(|py| {
            py_arr
                .numpy(py, None, None)
                .unwrap()
                .cast_into::<PyArrayDyn<T>>()
                .unwrap()
                .to_owned_array()
        })
    }

    #[test]
    fn test_add_f32() {
        let a = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b = array![[10.0f32, 20.0, 30.0], [40.0, 50.0, 60.0]];
        let result = Python::attach(|py| {
            let a = make_py_array(py, &a);
            let b = make_py_array(py, &b);
            eval::<f32>(Array::__add__(&a, &b).unwrap())
        });
        assert_eq!(result, (a + b).into_dyn());
    }

    #[test]
    fn test_sub_f32() {
        let a = array![[10.0f32, 20.0, 30.0], [40.0, 50.0, 60.0]];
        let b = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let result = Python::attach(|py| {
            let a = make_py_array(py, &a);
            let b = make_py_array(py, &b);
            eval::<f32>(Array::__sub__(&a, &b).unwrap())
        });
        assert_eq!(result, (a - b).into_dyn());
    }

    #[test]
    fn test_mul_f32() {
        let a = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b = array![[2.0f32, 3.0, 4.0], [5.0, 6.0, 7.0]];
        let result = Python::attach(|py| {
            let a = make_py_array(py, &a);
            let b = make_py_array(py, &b);
            eval::<f32>(Array::__mul__(&a, &b).unwrap())
        });
        assert_eq!(result, (a * b).into_dyn());
    }

    #[test]
    fn test_div_f32() {
        let a = array![[2.0f32, 6.0, 12.0], [20.0, 30.0, 42.0]];
        let b = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let result = Python::attach(|py| {
            let a = make_py_array(py, &a);
            let b = make_py_array(py, &b);
            eval::<f32>(Array::__truediv__(&a, &b).unwrap())
        });
        assert_eq!(result, (a / b).into_dyn());
    }

    #[test]
    fn test_add_f64() {
        let a = array![1.0f64, 2.0, 3.0];
        let b = array![0.5f64, 1.5, 2.5];
        let result = Python::attach(|py| {
            let a = make_py_array(py, &a);
            let b = make_py_array(py, &b);
            eval::<f64>(Array::__add__(&a, &b).unwrap())
        });
        assert_eq!(result, (a + b).into_dyn());
    }

    #[test]
    fn test_add_i32() {
        let a = array![[1i32, 2], [3, 4]];
        let b = array![[10i32, 20], [30, 40]];
        let result = Python::attach(|py| {
            let a = make_py_array(py, &a);
            let b = make_py_array(py, &b);
            eval::<i32>(Array::__add__(&a, &b).unwrap())
        });
        assert_eq!(result, (a + b).into_dyn());
    }

    #[test]
    fn test_sub_i32() {
        let a = array![[10i32, 20], [30, 40]];
        let b = array![[1i32, 2], [3, 4]];
        let result = Python::attach(|py| {
            let a = make_py_array(py, &a);
            let b = make_py_array(py, &b);
            eval::<i32>(Array::__sub__(&a, &b).unwrap())
        });
        assert_eq!(result, (a - b).into_dyn());
    }

    #[test]
    fn test_mul_i32() {
        let a = array![[1i32, 2], [3, 4]];
        let b = array![[5i32, 6], [7, 8]];
        let result = Python::attach(|py| {
            let a = make_py_array(py, &a);
            let b = make_py_array(py, &b);
            eval::<i32>(Array::__mul__(&a, &b).unwrap())
        });
        assert_eq!(result, (a * b).into_dyn());
    }

    fn getitem<T>(py_arr: &Bound<'_, Array>, key: &Bound<'_, PyAny>) -> ArrayD<T>
    where
        T: Dtyped + numpy::Element + Copy,
    {
        Array::__getitem__(py_arr.get(), key)
            .unwrap()
            .cast_into::<PyArrayDyn<T>>()
            .unwrap()
            .to_owned_array()
    }

    // --- __getitem__: integer indexing ---

    #[test]
    fn test_getitem_int_positive() {
        let data = ndarray::Array::from_shape_vec(IxDyn(&[5]), (0..5).collect()).unwrap();
        Python::attach(|py| {
            let py_arr = make_py_array(py, &data);
            let key = 2i64.into_pyobject(py).unwrap().into_any();
            let result = getitem::<i32>(&py_arr, key.as_any());
            assert_eq!(result.shape(), &[] as &[usize]);
            assert_eq!(result.first().copied().unwrap(), 2);
        });
    }

    #[test]
    fn test_getitem_int_negative() {
        let data = ndarray::Array::from_shape_vec(IxDyn(&[5]), (0..5).collect()).unwrap();
        Python::attach(|py| {
            let py_arr = make_py_array(py, &data);
            let key = (-1i64).into_pyobject(py).unwrap().into_any();
            let result = getitem::<i32>(&py_arr, key.as_any());
            assert_eq!(result.shape(), &[] as &[usize]);
            assert_eq!(result.first().copied().unwrap(), 4);
        });
    }

    #[test]
    fn test_getitem_int_out_of_bounds() {
        let data = ndarray::Array::from_shape_vec(IxDyn(&[5]), (0..5).collect()).unwrap();
        Python::attach(|py| {
            let py_arr = make_py_array(py, &data);
            let key = 5i64.into_pyobject(py).unwrap().into_any();
            let err = Array::__getitem__(py_arr.get(), key.as_any()).unwrap_err();
            assert!(err.is_instance_of::<PyIndexError>(py));
        });
    }

    #[test]
    fn test_getitem_int_negative_out_of_bounds() {
        let data = ndarray::Array::from_shape_vec(IxDyn(&[5]), (0..5).collect()).unwrap();
        Python::attach(|py| {
            let py_arr = make_py_array(py, &data);
            let key = (-6i64).into_pyobject(py).unwrap().into_any();
            let err = Array::__getitem__(py_arr.get(), key.as_any()).unwrap_err();
            assert!(err.is_instance_of::<PyIndexError>(py));
        });
    }

    // --- __getitem__: slice indexing ---

    #[test]
    fn test_getitem_slice_start_stop() {
        let data = ndarray::Array::from_shape_vec(IxDyn(&[5]), (0..5).map(|x| x as f32).collect())
            .unwrap();
        Python::attach(|py| {
            let py_arr = make_py_array(py, &data);
            let key = PySlice::new(py, 1, 4, 1).into_any();
            let result = getitem::<f32>(&py_arr, key.as_any());
            assert_eq!(result.shape(), &[3usize]);
            assert_eq!(result.as_slice().unwrap(), &[1.0f32, 2.0, 3.0]);
        });
    }

    #[test]
    fn test_getitem_slice_no_start() {
        let data = ndarray::Array::from_shape_vec(IxDyn(&[5]), (0..5).map(|x| x as f32).collect())
            .unwrap();
        Python::attach(|py| {
            let py_arr = make_py_array(py, &data);
            let key = py.eval(c"slice(None, 3, None)", None, None).unwrap();
            let result = getitem::<f32>(&py_arr, &key);
            assert_eq!(result.shape(), &[3usize]);
            assert_eq!(result.as_slice().unwrap(), &[0.0f32, 1.0, 2.0]);
        });
    }

    #[test]
    fn test_getitem_slice_no_stop() {
        let data = ndarray::Array::from_shape_vec(IxDyn(&[5]), (0..5).map(|x| x as f32).collect())
            .unwrap();
        Python::attach(|py| {
            let py_arr = make_py_array(py, &data);
            let key = py.eval(c"slice(2, None, None)", None, None).unwrap();
            let result = getitem::<f32>(&py_arr, &key);
            assert_eq!(result.shape(), &[3usize]);
            assert_eq!(result.as_slice().unwrap(), &[2.0f32, 3.0, 4.0]);
        });
    }

    #[test]
    fn test_getitem_slice_full() {
        let data = ndarray::Array::from_shape_vec(IxDyn(&[5]), (0..5).collect()).unwrap();
        Python::attach(|py| {
            let py_arr = make_py_array(py, &data);
            let key = PySlice::full(py).into_any();
            let result = getitem::<i32>(&py_arr, key.as_any());
            assert_eq!(result.shape(), &[5usize]);
            assert_eq!(result.as_slice().unwrap(), &[0i32, 1, 2, 3, 4]);
        });
    }

    #[test]
    fn test_getitem_slice_negative_start() {
        let data = ndarray::Array::from_shape_vec(IxDyn(&[5]), (0..5).collect()).unwrap();
        Python::attach(|py| {
            let py_arr = make_py_array(py, &data);
            let key = py.eval(c"slice(-3, None, None)", None, None).unwrap();
            let result = getitem::<i32>(&py_arr, &key);
            assert_eq!(result.shape(), &[3usize]);
            assert_eq!(result.as_slice().unwrap(), &[2i32, 3, 4]);
        });
    }

    #[test]
    fn test_getitem_slice_empty() {
        let data = ndarray::Array::from_shape_vec(IxDyn(&[5]), (0..5).collect()).unwrap();
        Python::attach(|py| {
            let py_arr = make_py_array(py, &data);
            let key = PySlice::new(py, 2, 2, 1).into_any();
            let result = getitem::<i32>(&py_arr, key.as_any());
            assert_eq!(result.shape(), &[0usize]);
        });
    }

    #[test]
    fn test_getitem_slice_step_not_one_err() {
        let data = ndarray::Array::from_shape_vec(IxDyn(&[5]), (0..5).collect()).unwrap();
        Python::attach(|py| {
            let py_arr = make_py_array(py, &data);
            let key = py.eval(c"slice(None, None, 2)", None, None).unwrap();
            let err = Array::__getitem__(py_arr.get(), &key).unwrap_err();
            assert!(err.is_instance_of::<PyValueError>(py));
        });
    }

    // --- __getitem__: 2-D indexing ---

    #[test]
    fn test_getitem_2d_row_int() {
        // arr[0] on shape [2, 3] → first row, shape [3]
        let data =
            ndarray::Array::from_shape_vec(IxDyn(&[2, 3]), (0..6).map(|x| x as f32).collect())
                .unwrap();
        Python::attach(|py| {
            let py_arr = make_py_array(py, &data);
            let key = 0i64.into_pyobject(py).unwrap().into_any();
            let result = getitem::<f32>(&py_arr, key.as_any());
            assert_eq!(result.shape(), &[3usize]);
            assert_eq!(result.as_slice().unwrap(), &[0.0f32, 1.0, 2.0]);
        });
    }

    #[test]
    fn test_getitem_2d_element() {
        // arr[1, 2] on shape [2, 3] → scalar, shape []
        let data =
            ndarray::Array::from_shape_vec(IxDyn(&[2, 3]), (0..6).map(|x| x as f32).collect())
                .unwrap();
        Python::attach(|py| {
            let py_arr = make_py_array(py, &data);
            let items: Vec<Bound<'_, PyAny>> = vec![
                1i64.into_pyobject(py).unwrap().into_any(),
                2i64.into_pyobject(py).unwrap().into_any(),
            ];
            let key = PyTuple::new(py, items).unwrap().into_any();
            let result = getitem::<f32>(&py_arr, key.as_any());
            assert_eq!(result.shape(), &[] as &[usize]);
            assert_eq!(result.first().copied().unwrap(), 5.0f32);
        });
    }

    #[test]
    fn test_getitem_2d_col() {
        // arr[:, 1] on shape [2, 3] → column 1, shape [2]
        let data =
            ndarray::Array::from_shape_vec(IxDyn(&[2, 3]), (0..6).map(|x| x as f32).collect())
                .unwrap();
        Python::attach(|py| {
            let py_arr = make_py_array(py, &data);
            let items: Vec<Bound<'_, PyAny>> = vec![
                PySlice::full(py).into_any(),
                1i64.into_pyobject(py).unwrap().into_any(),
            ];
            let key = PyTuple::new(py, items).unwrap().into_any();
            let result = getitem::<f32>(&py_arr, key.as_any());
            assert_eq!(result.shape(), &[2usize]);
            assert_eq!(result.as_slice().unwrap(), &[1.0f32, 4.0]);
        });
    }

    #[test]
    fn test_getitem_2d_subarray() {
        // arr[0:2, 1:3] on shape [3, 3] → shape [2, 2]
        let data =
            ndarray::Array::from_shape_vec(IxDyn(&[3, 3]), (0..9).map(|x| x as f32).collect())
                .unwrap();
        Python::attach(|py| {
            let py_arr = make_py_array(py, &data);
            let items: Vec<Bound<'_, PyAny>> = vec![
                PySlice::new(py, 0, 2, 1).into_any(),
                PySlice::new(py, 1, 3, 1).into_any(),
            ];
            let key = PyTuple::new(py, items).unwrap().into_any();
            let result = getitem::<f32>(&py_arr, key.as_any());
            // [[0,1,2],[3,4,5],[6,7,8]] → rows 0-1, cols 1-2 → [[1,2],[4,5]]
            let expected =
                ndarray::Array::from_shape_vec(IxDyn(&[2, 2]), vec![1.0f32, 2.0, 4.0, 5.0])
                    .unwrap();
            assert_eq!(result, expected);
        });
    }

    // --- __getitem__: ellipsis ---

    #[test]
    fn test_getitem_ellipsis_full() {
        // arr[...] → entire array unchanged
        let data = ndarray::Array::from_shape_vec(IxDyn(&[2, 3]), (0..6).collect()).unwrap();
        Python::attach(|py| {
            let py_arr = make_py_array(py, &data);
            let key = PyEllipsis::get(py).to_owned().into_any();
            let result = getitem::<i32>(&py_arr, key.as_any());
            assert_eq!(result.shape(), &[2usize, 3]);
            assert_eq!(result, data);
        });
    }

    #[test]
    fn test_getitem_ellipsis_prefix() {
        // arr[0, ...] on shape [2, 3] → row 0, shape [3]
        let data =
            ndarray::Array::from_shape_vec(IxDyn(&[2, 3]), (0..6).map(|x| x as f32).collect())
                .unwrap();
        Python::attach(|py| {
            let py_arr = make_py_array(py, &data);
            let items: Vec<Bound<'_, PyAny>> = vec![
                0i64.into_pyobject(py).unwrap().into_any(),
                PyEllipsis::get(py).to_owned().into_any(),
            ];
            let key = PyTuple::new(py, items).unwrap().into_any();
            let result = getitem::<f32>(&py_arr, key.as_any());
            assert_eq!(result.shape(), &[3usize]);
            assert_eq!(result.as_slice().unwrap(), &[0.0f32, 1.0, 2.0]);
        });
    }

    #[test]
    fn test_getitem_ellipsis_suffix() {
        // arr[..., 0] on shape [2, 3] → column 0, shape [2]
        let data =
            ndarray::Array::from_shape_vec(IxDyn(&[2, 3]), (0..6).map(|x| x as f32).collect())
                .unwrap();
        Python::attach(|py| {
            let py_arr = make_py_array(py, &data);
            let items: Vec<Bound<'_, PyAny>> = vec![
                PyEllipsis::get(py).to_owned().into_any(),
                0i64.into_pyobject(py).unwrap().into_any(),
            ];
            let key = PyTuple::new(py, items).unwrap().into_any();
            let result = getitem::<f32>(&py_arr, key.as_any());
            assert_eq!(result.shape(), &[2usize]);
            assert_eq!(result.as_slice().unwrap(), &[0.0f32, 3.0]);
        });
    }

    // --- __getitem__: error cases ---

    #[test]
    fn test_getitem_too_many_indices_err() {
        let data = ndarray::Array::from_shape_vec(IxDyn(&[2, 3]), (0..6).collect()).unwrap();
        Python::attach(|py| {
            let py_arr = make_py_array(py, &data);
            let items: Vec<Bound<'_, PyAny>> = vec![
                0i64.into_pyobject(py).unwrap().into_any(),
                1i64.into_pyobject(py).unwrap().into_any(),
                2i64.into_pyobject(py).unwrap().into_any(),
            ];
            let key = PyTuple::new(py, items).unwrap().into_any();
            let err = Array::__getitem__(py_arr.get(), key.as_any()).unwrap_err();
            assert!(err.is_instance_of::<PyIndexError>(py));
        });
    }

    #[test]
    fn test_getitem_multiple_ellipses_err() {
        let data = ndarray::Array::from_shape_vec(IxDyn(&[2, 3]), (0..6).collect()).unwrap();
        Python::attach(|py| {
            let py_arr = make_py_array(py, &data);
            let items: Vec<Bound<'_, PyAny>> = vec![
                PyEllipsis::get(py).to_owned().into_any(),
                PyEllipsis::get(py).to_owned().into_any(),
            ];
            let key = PyTuple::new(py, items).unwrap().into_any();
            let err = Array::__getitem__(py_arr.get(), key.as_any()).unwrap_err();
            assert!(err.is_instance_of::<PyIndexError>(py));
        });
    }

    #[test]
    fn test_getitem_invalid_type_err() {
        let data = ndarray::Array::from_shape_vec(IxDyn(&[5]), (0..5).collect()).unwrap();
        Python::attach(|py| {
            let py_arr = make_py_array(py, &data);
            let key = "bad".into_pyobject(py).unwrap().into_any();
            let err = Array::__getitem__(py_arr.get(), key.as_any()).unwrap_err();
            assert!(err.is_instance_of::<PyTypeError>(py));
        });
    }

    #[test]
    fn test_ops_chained() {
        // (a + b) * c  computed both in zix and ndarray
        let a = array![1.0f32, 2.0, 3.0, 4.0];
        let b = array![4.0f32, 3.0, 2.0, 1.0];
        let c = array![2.0f32, 2.0, 2.0, 2.0];
        let zix_result = Python::attach(|py| {
            let a = make_py_array(py, &a);
            let b = make_py_array(py, &b);
            let c = make_py_array(py, &c);
            eval::<f32>(
                Array::__mul__(
                    &Bound::new(py, Array::__add__(&a, &b).unwrap()).unwrap(),
                    &c,
                )
                .unwrap(),
            )
        });
        assert_eq!(zix_result, ((a + b) * c).into_dyn());
    }
}
