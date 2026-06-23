use std::collections::BTreeMap;
use std::sync::Mutex;

use jix_core::{Array as CoreArray, ArrayAny};
use jix_core::{Codec, Filter};
use numpy::{PyArrayDescr, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::types::PyAnyMethods;
use std::ops::Range;

use crate::codec::ReadContext;
use crate::dtype::dtype_to_numpy;
use crate::util::{dim_arr, numpy_empty, DimArray, IntoPyResult, ItemOrSequence};

/// A multi-dimensional compressed array.
///
/// `Array` is the central type in jix. It stores n-dimensional numeric data in a
/// block-compressed format - the array is divided into nd-blocks, each compressed
/// independently with Zstd. Data is decoded on demand: constructing an array or
/// chaining operations does no I/O. Actual decompression happens only when you
/// materialize the result, for example by indexing with `[]`, calling `.numpy()`,
/// `.compact()`, or `.write_to()`.
///
/// # Creating arrays
///
/// | Function | Description |
/// |---|---|
/// | [`jix.compact(data)`][jix.compact] | Compress any array-like (NumPy array, list, scalar) into a new jix array. |
/// | [`jix.asarray(data)`][jix.asarray] | Create a jix array *view* of any array-like. Useful for mixing plain data with jix arrays in operations. |
/// | [`jix.read_array(path)`][jix.read_array] | Load an array from a `.jix` file. |
///
/// ```python
/// import jix
/// import numpy as np
///
/// # From a NumPy array
/// a = jix.compact(np.arange(100, dtype=np.float32).reshape(10, 10))
///
/// # From a list
/// b = jix.compact([[1, 2, 3], [4, 5, 6]])
///
/// # From a file
/// c = jix.read_array("data.jix")
/// ```
///
/// # Reading data
///
/// Call `.numpy()` (or `[]`) to decode the array - or a sub-region of it - into a NumPy
/// array. The result is always a fresh allocation; mutations to it do not affect the
/// source array. Both forms accept the same index syntax:
///
/// ```python
/// a.numpy()           # full array
/// a.numpy(0)          # row 0 (integer drops that axis)
/// a.numpy(slice(1,4)) # rows 1-3 (slice keeps the axis)
/// a[0, 1:3]           # shorthand via __getitem__
/// a[..., -1]          # last column of any-rank array
/// ```
///
/// For tight loops that read many slices, pass an explicit [`jix.ReadContext`][jix.ReadContext]
/// to amortize decompressor initialization:
///
/// ```python
/// ctx = a.read_ctx()
/// rows = [a.numpy(i, context=ctx) for i in range(len(a))]
/// ```
///
/// # Operations
///
/// Operations return a new `Array` that wraps the input and records the transformation.
/// No data is copied or computed at call time - the computation runs in a single pass when
/// you read the result. Chains compose without intermediate allocations:
///
/// ```python
/// result = (a.astype('float64') - a.mean()).abs().sum(axis=0).numpy()
/// ```
///
/// ## Shape operations
///
/// Shape operations remap the array's indices without copying data, returning lazy views.
/// Reshape (and its `flatten` shorthand) are uniquely prone to read-amplification: when the
/// new layout crosses block boundaries that the original layout respected, reading the view
/// may decompress more data than the request appears to touch. Materialize the result with
/// [`jix.compact()`][jix.Array.compact] to re-encode with a block layout suited to the new
/// shape when you intend to read it more than once.
///
/// # Persistence
///
/// Save an array with [`.write_to(path)`][jix.Array.write_to] or [`jix.write_array()`][jix.write_array],
/// and reload it with [`jix.read_array()`][jix.read_array]. Multiple arrays can
/// be written back-to-back into a single file and read back by supplying `offset` and `len`.
///
/// ```python
/// a.write_to("data.jix")
/// b = jix.read_array("data.jix")
/// ```
///
/// # Copying and re-encoding
///
/// [`.compact()`][jix.Array.compact] materializes the current array (including any pending lazy
/// operations) into a new compressed array. This is also the way to tune the block layout
/// after shape-changing operations:
///
/// ```python
/// # Transpose and re-encode with a block layout suited for column access
/// b = jix.compact(a.T, params={"block_shape": [1024, 1]})
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
#[pyclass(module = "jix", frozen)]
pub struct Array {
    pub(crate) arr: ArrayAny,
    cache: Mutex<ArrayCache>,
}
struct ArrayCache {
    numpy_dtype: Option<Py<PyArrayDescr>>,
}
impl Array {
    pub(crate) fn from_core(array: ArrayAny) -> Self {
        Self {
            arr: array,
            cache: Mutex::new(ArrayCache { numpy_dtype: None }),
        }
    }

    pub(crate) fn to_core(&self) -> ArrayAny {
        CoreArray::from_storage(self.arr.storage().clone())
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
    /// Returns:
    ///     The shape as a tuple of integers.
    ///
    /// ```python
    /// import jix
    ///
    /// a = jix.compact([1, 2, 3, 4])
    /// assert a.shape == (4,)
    ///
    /// b = jix.compact([[1, 2], [3, 4], [5, 6]])
    /// assert b.shape == (3, 2)
    /// ```
    #[getter]
    pub fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.arr.shape().iter().copied())
    }

    /// The number of dimensions (axes) of the array.
    ///
    /// Returns:
    ///     The number of dimensions as an integer.
    ///
    /// ```python
    /// import jix
    ///
    /// a = jix.compact([1, 2, 3, 4])
    /// assert a.ndim == 1
    ///
    /// b = jix.compact([[1, 2], [3, 4], [5, 6]])
    /// assert b.ndim == 2
    /// ```
    #[getter]
    pub fn ndim(&self) -> usize {
        self.arr.shape().len()
    }

    /// The total number of elements in the array (the product of the axis lengths).
    ///
    /// Returns:
    ///     The total element count as an integer.
    ///
    /// ```python
    /// import jix
    ///
    /// a = jix.compact([1, 2, 3, 4])
    /// assert a.size == 4
    ///
    /// b = jix.compact([[1, 2], [3, 4], [5, 6]])
    /// assert b.size == 6
    /// ```
    #[getter]
    pub fn size(&self) -> PyResult<u64> {
        Ok(self.arr.shape().iter().product::<u64>())
    }

    /// The length of the array along the first axis (axis 0).
    ///
    /// Returns:
    ///     The length of axis 0 as an integer.
    ///
    /// ```python
    /// import jix
    ///
    /// a = jix.compact([1, 2, 3, 4])
    /// assert len(a) == 4
    ///
    /// b = jix.compact([[1, 2], [3, 4], [5, 6]])
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
    /// Returns:
    ///     The element dtype as a `numpy.dtype` object.
    ///
    /// ```python
    /// import jix
    /// import numpy as np
    ///
    /// a = jix.compact([1, 2, 3, 4], dtype='int32')
    /// assert a.dtype == np.dtype('int32')
    ///
    /// b = jix.compact([[1.0, 2.0], [3.0, 4.0]], dtype='float64')
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
    /// This is the primary way to materialize a [`jix.Array`][jix.Array] into a form that ordinary Python
    /// and NumPy code can consume. It decodes the compressed block data, copies it into a
    /// freshly-allocated NumPy array, and returns that array. The returned array is fully
    /// independent: mutations to one do not affect the other.
    ///
    /// Args:
    ///     index: Selects a sub-region to read. Accepts the same syntax Python uses inside
    ///         `[...]`:
    ///
    ///         | Form | Example | Effect |
    ///         |---|---|---|
    ///         | omitted / `None` | `arr.numpy()` | read the whole array |
    ///         | integer | `arr[2]` or `arr.numpy(2)` | select a single position along axis 0 |
    ///         | slice | `arr[1:4]` or `arr.numpy(slice(1, 4))` | select a range along axis 0 |
    ///         | `...` (Ellipsis) | `arr[...]` or `arr.numpy(...)` | fill all remaining axes with full slices |
    ///         | tuple of the above | `arr[0, 1:3, ..., :-2]` | index each axis independently |
    ///
    ///         Most callers use `__getitem__` (`arr[...]`) instead; the two are equivalent.
    ///
    ///         **Integers:** Select one position along the corresponding axis and **remove** that
    ///         axis from the output shape (just like NumPy). Negative indices are supported
    ///         (`-1` means the last element). Out-of-bounds raises `IndexError`.
    ///
    ///         **Slices:** Select a contiguous range and **keep** that axis in the output.
    ///         `start` defaults to `0`; `stop` defaults to the axis length. Both may be
    ///         negative. The **step must be 1**. An empty slice (`start == stop`) is valid and
    ///         produces an axis of length 0. Bounds are checked strictly (unlike NumPy, which
    ///         silently clamps out-of-range endpoints).
    ///
    ///         **Ellipsis (`...`):** Expands to as many full-range slices as needed to cover all
    ///         remaining axes. At most one ellipsis is allowed.
    ///
    ///         **Omitted trailing axes:** If the index covers fewer axes than the array has
    ///         dimensions, the remaining axes receive implicit full-range slices.
    ///
    ///     context: An optional [`jix.ReadContext`][jix.ReadContext] to reuse across multiple reads. When omitted,
    ///         a context is created internally for each call. Pass one explicitly when calling
    ///         `numpy()` many times in a loop to avoid repeated decompressor initialization:
    ///
    ///         ```python
    ///         ctx = jix.ReadContext()
    ///         rows = [a.numpy(i, context=ctx) for i in range(len(a))]
    ///         ```
    ///
    /// Returns:
    ///     A NumPy array with the same dtype as `self`, shape determined by the `index`
    ///         argument (or `self.shape` when no index is supplied), in C-contiguous (row-major)
    ///         memory order. A brand-new allocation; the caller owns it outright.
    ///
    /// Raises:
    ///     IndexError: Integer index out of bounds, slice `start` or `stop` out of bounds,
    ///         more index items than array dimensions, or more than one ellipsis.
    ///     ValueError: Slice step other than 1.
    ///     TypeError: Unsupported index item type (anything other than an integer, slice, `...`,
    ///         or tuple of these).
    #[pyo3(signature = (index=None, *, context=None))]
    pub fn numpy<'py>(
        &self,
        py: Python<'py>,
        index: Option<&Bound<'py, PyAny>>,
        context: Option<&Bound<'py, ReadContext>>,
    ) -> PyResult<Bound<'py, PyUntypedArray>> {
        let shape = self.arr.shape();
        let parsed = crate::ops::parse_basic_index(py, shape, index)?;

        let mut ranges: DimArray<Range<u64>> = DimArray::new();
        let mut out_shape: Vec<usize> = Vec::with_capacity(parsed.items.len());
        for (axis, item) in parsed.items.iter().enumerate() {
            let start = item.start.unwrap() as u64;
            let end = item.end.unwrap() as u64;
            ranges.push(start..end);
            if !parsed.drop_axes.contains(&axis) {
                out_shape.push((end - start) as usize);
            }
        }

        let np_arr = self.to_numpy(py, &ranges, context)?;
        let np_arr: Bound<'_, PyUntypedArray> =
            np_arr.call_method1("reshape", (out_shape,))?.cast_into()?;
        Ok(np_arr)
    }

    /// Read elements from the array (or a sub-region of it) and return them as a NumPy array.
    ///
    /// This function is identical to `numpy()`, see that method for details.
    ///
    /// Args:
    ///     index: See `numpy()` for accepted index types and behavior.
    ///
    /// Returns:
    ///     A NumPy array. See `numpy()` for details.
    pub fn __getitem__<'py>(
        &self,
        index: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyUntypedArray>> {
        self.numpy(index.py(), Some(index), None)
    }

    /// Creates a [`jix.ReadContext`][jix.ReadContext] with decoder parameters derived from this array's storage.
    ///
    /// The returned context inherits the decoder configuration stored alongside the array data,
    /// ensuring that reads use the same settings the array was written with. Prefer this over
    /// constructing [`jix.ReadContext`][jix.ReadContext] directly when reading a specific array.
    ///
    /// Pass the returned context to [`Array.numpy()`][jix.Array.numpy] or [`jix.compact()`][jix.compact] to amortize decompressor
    /// initialization across many successive reads. See [`jix.ReadContext`][jix.ReadContext] for details.
    ///
    /// ```python
    /// import jix
    /// import numpy as np
    ///
    /// a = jix.compact(np.arange(30, dtype=np.int32).reshape(10, 3))
    /// ctx = a.read_ctx()
    /// rows = [a.numpy(i, context=ctx) for i in range(len(a))]
    /// ```
    ///
    /// Returns:
    ///     A [`jix.ReadContext`][jix.ReadContext] configured for this array's codec settings.
    pub fn read_ctx<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, ReadContext>> {
        Bound::new(py, ReadContext::from_core(self.arr.read_ctx()))
    }

    /// Copies the data of an array into a new compact array by compressing it into new blocks. See [`jix.compact()`][jix.compact].
    #[pyo3(signature = (*, params=None, context=None))]
    pub fn compact<'py>(
        slf: &Bound<'py, Self>,
        params: Option<Bound<'_, PyDict>>,
        context: Option<&Bound<'_, ReadContext>>,
    ) -> PyResult<Bound<'py, Array>> {
        crate::compact(slf, None, params, context)
    }

    /// Return a string representation of the array as `Array(shape=..., dtype=...)`.
    ///
    /// Returns:
    ///     A string of the form `Array(shape=..., dtype=...)`.
    pub fn __str__(&self) -> String {
        let arr = &self.arr;
        let shape_str = arr
            .shape()
            .iter()
            .map(|d| d.to_string())
            .collect::<DimArray<_>>()
            .join(", ");
        format!("Array(shape=({}), dtype={})", shape_str, arr.dtype())
    }

    /// Return a string representation of the array as `Array(shape=..., dtype=...)`.
    ///
    /// Returns:
    ///     A string of the form `Array(shape=..., dtype=...)`.
    pub fn __repr__(&self) -> String {
        self.__str__()
    }

    // == archive I/O ==

    /// Write the array to a file or a file-like object. See [`jix.write_array()`][jix.write_array].
    #[pyo3(signature = (path_or_writer, *, append=false, params=None, context=None))]
    pub fn write_to(
        slf: &Bound<'_, Array>,
        path_or_writer: &Bound<'_, PyAny>,
        append: bool,
        params: Option<Bound<'_, PyDict>>,
        context: Option<&Bound<'_, ReadContext>>,
    ) -> PyResult<()> {
        crate::archive::write_array(slf, path_or_writer, append, params, context)
    }

    // == arithmetic ops ==

    /// Element-wise addition of two arrays. See [`jix.add()`][jix.add].
    pub fn add(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        crate::ops::add(slf, other)
    }

    /// Element-wise addition of two arrays. See [`jix.add()`][jix.add].
    pub fn __add__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        crate::ops::add(slf, other)
    }

    /// Element-wise addition of two arrays. See [`jix.add()`][jix.add].
    pub fn __radd__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        crate::ops::add(other, slf)
    }

    /// Element-wise subtraction of two arrays. See [`jix.subtract()`][jix.subtract].
    pub fn subtract(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        crate::ops::subtract(slf, other)
    }

    /// Element-wise subtraction of two arrays. See [`jix.subtract()`][jix.subtract].
    pub fn __sub__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        crate::ops::subtract(slf, other)
    }

    /// Element-wise subtraction of two arrays. See [`jix.subtract()`][jix.subtract].
    pub fn __rsub__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        crate::ops::subtract(other, slf)
    }

    /// Element-wise multiplication of two arrays. See [`jix.multiply()`][jix.multiply].
    pub fn multiply(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        crate::ops::multiply(slf, other)
    }

    /// Element-wise multiplication of two arrays. See [`jix.multiply()`][jix.multiply].
    pub fn __mul__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        crate::ops::multiply(slf, other)
    }

    /// Element-wise multiplication of two arrays. See [`jix.multiply()`][jix.multiply].
    pub fn __rmul__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        crate::ops::multiply(other, slf)
    }

    /// Element-wise division of two arrays. See [`jix.divide()`][jix.divide].
    pub fn divide(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        crate::ops::divide(slf, other)
    }

    /// Element-wise division of two arrays. See [`jix.divide()`][jix.divide].
    pub fn __truediv__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        crate::ops::divide(slf, other)
    }

    /// Element-wise division of two arrays. See [`jix.divide()`][jix.divide].
    pub fn __rtruediv__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        crate::ops::divide(other, slf)
    }

    /// Element-wise floor division of two arrays. See [`jix.floor_divide()`][jix.floor_divide].
    pub fn __floordiv__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        crate::ops::floor_divide(slf, other)
    }

    /// Element-wise floor division of two arrays. See [`jix.floor_divide()`][jix.floor_divide].
    pub fn __rfloordiv__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        crate::ops::floor_divide(other, slf)
    }

    /// Element-wise exponentiation (`a ** b`). See [`jix.power()`][jix.power].
    pub fn pow(slf: &Bound<'_, Self>, exponent: &Bound<'_, PyAny>) -> PyResult<Self> {
        crate::ops::power(slf, exponent)
    }

    /// Element-wise exponentiation (`a ** b`). See [`jix.power()`][jix.power].
    pub fn __pow__<'py>(
        slf: &Bound<'py, Self>,
        exponent: &Bound<'py, PyAny>,
        modulo: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Self> {
        if modulo.is_some() {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "modulo argument to pow() is not supported",
            ));
        }
        crate::ops::power(slf, exponent)
    }

    /// Element-wise exponentiation (`b ** a`). See [`jix.power()`][jix.power].
    pub fn __rpow__<'py>(
        slf: &Bound<'py, Self>,
        base: &Bound<'py, PyAny>,
        modulo: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Self> {
        if modulo.is_some() {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "modulo argument to pow() is not supported",
            ));
        }
        crate::ops::power(base, slf)
    }

    /// Arithmetic negation applied element-wise. See [`jix.negative()`][jix.negative].
    pub fn negative(slf: &Bound<'_, Self>) -> PyResult<Self> {
        crate::ops::negative(slf)
    }

    /// Arithmetic negation applied element-wise. See [`jix.negative()`][jix.negative].
    pub fn __neg__<'py>(slf: &Bound<'py, Self>) -> PyResult<Self> {
        crate::ops::negative(slf)
    }

    /// Computes the absolute value of each element. See [`jix.absolute()`][jix.absolute].
    pub fn abs(slf: &Bound<'_, Self>) -> PyResult<Self> {
        crate::ops::absolute(slf)
    }

    /// Computes the absolute value of each element. See [`jix.absolute()`][jix.absolute].
    pub fn __abs__<'py>(slf: &Bound<'py, Self>) -> PyResult<Self> {
        crate::ops::absolute(slf)
    }

    /// Computes the natural exponential (`e^x`) of each element. See [`jix.exp()`][jix.exp].
    pub fn exp(slf: &Bound<'_, Self>) -> PyResult<Self> {
        crate::ops::exp(slf)
    }

    /// Computes the square root of each element. See [`jix.sqrt()`][jix.sqrt].
    pub fn sqrt(slf: &Bound<'_, Self>) -> PyResult<Self> {
        crate::ops::sqrt(slf)
    }

    /// Squares each element (`x * x`). See [`jix.square()`][jix.square].
    pub fn square(slf: &Bound<'_, Self>) -> PyResult<Self> {
        crate::ops::square(slf)
    }

    /// Rounds each element up to the nearest integer (towards +inf). See [`jix.ceil()`][jix.ceil].
    pub fn ceil(slf: &Bound<'_, Self>) -> PyResult<Self> {
        crate::ops::ceil(slf)
    }

    /// Rounds each element down to the nearest integer (towards -inf). See [`jix.floor()`][jix.floor].
    pub fn floor(slf: &Bound<'_, Self>) -> PyResult<Self> {
        crate::ops::floor(slf)
    }

    /// Rounds each element to the nearest integer. See [`jix.round()`][jix.round].
    pub fn round(slf: &Bound<'_, Self>) -> PyResult<Self> {
        crate::ops::round(slf)
    }

    // == bitwise ops ==

    /// Element-wise bitwise AND of two arrays. See [`jix.bitwise_and()`][jix.bitwise_and].
    pub fn __and__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        crate::ops::bitwise_and(slf, other)
    }

    /// Element-wise bitwise AND of two arrays. See [`jix.bitwise_and()`][jix.bitwise_and].
    pub fn __rand__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        crate::ops::bitwise_and(other, slf)
    }

    /// Element-wise bitwise OR of two arrays. See [`jix.bitwise_or()`][jix.bitwise_or].
    pub fn __or__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        crate::ops::bitwise_or(slf, other)
    }

    /// Element-wise bitwise OR of two arrays. See [`jix.bitwise_or()`][jix.bitwise_or].
    pub fn __ror__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        crate::ops::bitwise_or(other, slf)
    }

    /// Element-wise bitwise XOR of two arrays. See [`jix.bitwise_xor()`][jix.bitwise_xor].
    pub fn __xor__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        crate::ops::bitwise_xor(slf, other)
    }

    /// Element-wise bitwise XOR of two arrays. See [`jix.bitwise_xor()`][jix.bitwise_xor].
    pub fn __rxor__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        crate::ops::bitwise_xor(other, slf)
    }

    /// Element-wise bitwise NOT (one's complement). See [`jix.bitwise_not()`][jix.bitwise_not].
    pub fn __invert__<'py>(slf: &Bound<'py, Self>) -> PyResult<Self> {
        crate::ops::bitwise_not(slf)
    }

    /// Element-wise left shift (`a << b`). See [`jix.bitwise_left_shift()`][jix.bitwise_left_shift].
    pub fn __lshift__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        crate::ops::bitwise_left_shift(slf, other)
    }

    /// Element-wise left shift (`a << b`). See [`jix.bitwise_left_shift()`][jix.bitwise_left_shift].
    pub fn __rlshift__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        crate::ops::bitwise_left_shift(other, slf)
    }

    // == comparison ops ==

    /// Element-wise less-than test (`a < b`). See [`jix.less()`][jix.less].
    pub fn __lt__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        crate::ops::less(slf, other)
    }

    /// Element-wise less-than-or-equal test (`a <= b`). See [`jix.less_equal()`][jix.less_equal].
    pub fn __le__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        crate::ops::less_equal(slf, other)
    }

    /// Element-wise greater-than test (`a > b`). See [`jix.greater()`][jix.greater].
    pub fn __gt__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        crate::ops::greater(slf, other)
    }

    /// Element-wise greater-than-or-equal test (`a >= b`). See [`jix.greater_equal()`][jix.greater_equal].
    pub fn __ge__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        crate::ops::greater_equal(slf, other)
    }

    /// Element-wise equality test (`a == b`). See [`jix.equal()`][jix.equal].
    pub fn __eq__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        crate::ops::equal(slf, other)
    }

    /// Element-wise inequality test (`a != b`). See [`jix.not_equal()`][jix.not_equal].
    pub fn __ne__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        crate::ops::not_equal(slf, other)
    }

    // == reduction ops ==

    /// Reduces one or more axes with logical AND: returns `True` if all elements are truthy. See [`jix.all()`][jix.all].
    #[pyo3(signature = (axis=None, *, keepdims=false))]
    pub fn all(
        slf: &Bound<'_, Self>,
        axis: Option<ItemOrSequence<i32>>,
        keepdims: bool,
    ) -> PyResult<Self> {
        crate::ops::all(slf, axis, keepdims)
    }

    /// Reduces one or more axes with logical OR: returns `True` if any element is truthy. See [`jix.any()`][jix.any].
    #[pyo3(signature = (axis=None, *, keepdims=false))]
    pub fn any(
        slf: &Bound<'_, Self>,
        axis: Option<ItemOrSequence<i32>>,
        keepdims: bool,
    ) -> PyResult<Self> {
        crate::ops::any(slf, axis, keepdims)
    }

    /// Reduces one or more axes by taking the maximum element. See [`jix.max()`][jix.max].
    #[pyo3(signature = (axis=None, *, keepdims=false))]
    pub fn max(
        slf: &Bound<'_, Self>,
        axis: Option<ItemOrSequence<i32>>,
        keepdims: bool,
    ) -> PyResult<Self> {
        crate::ops::max(slf, axis, keepdims)
    }

    /// Reduces one or more axes by taking the minimum element. See [`jix.min()`][jix.min].
    #[pyo3(signature = (axis=None, *, keepdims=false))]
    pub fn min(
        slf: &Bound<'_, Self>,
        axis: Option<ItemOrSequence<i32>>,
        keepdims: bool,
    ) -> PyResult<Self> {
        crate::ops::min(slf, axis, keepdims)
    }

    /// Returns the index of the maximum element along a single axis. See [`jix.argmax()`][jix.argmax].
    #[pyo3(signature = (axis=None, *, keepdims=false))]
    pub fn argmax(slf: &Bound<'_, Self>, axis: Option<i32>, keepdims: bool) -> PyResult<Self> {
        crate::ops::argmax(slf, axis, keepdims)
    }

    /// Returns the index of the minimum element along a single axis. See [`jix.argmin()`][jix.argmin].
    #[pyo3(signature = (axis=None, *, keepdims=false))]
    pub fn argmin(slf: &Bound<'_, Self>, axis: Option<i32>, keepdims: bool) -> PyResult<Self> {
        crate::ops::argmin(slf, axis, keepdims)
    }

    /// Reduces one or more axes by summing all elements. See [`jix.sum()`][jix.sum].
    #[pyo3(signature = (axis=None, *, keepdims=false))]
    pub fn sum(
        slf: &Bound<'_, Self>,
        axis: Option<ItemOrSequence<i32>>,
        keepdims: bool,
    ) -> PyResult<Self> {
        crate::ops::sum(slf, axis, keepdims)
    }

    /// Computes the arithmetic mean along one or more axes. See [`jix.mean()`][jix.mean].
    #[pyo3(signature = (axis=None, *, keepdims=false))]
    pub fn mean(
        slf: &Bound<'_, Self>,
        axis: Option<ItemOrSequence<i32>>,
        keepdims: bool,
    ) -> PyResult<Self> {
        crate::ops::mean(slf, axis, keepdims)
    }

    /// Reduces one or more axes by multiplying all elements. See [`jix.product()`][jix.product].
    #[pyo3(signature = (axis=None, *, keepdims=false))]
    pub fn prod(
        slf: &Bound<'_, Self>,
        axis: Option<ItemOrSequence<i32>>,
        keepdims: bool,
    ) -> PyResult<Self> {
        crate::ops::product(slf, axis, keepdims)
    }

    /// Computes the standard deviation along one or more axes. See [`jix.std()`][jix.std].
    #[pyo3(signature = (axis=None, *, keepdims=false, ddof=0.0))]
    pub fn std(
        slf: &Bound<'_, Self>,
        axis: Option<ItemOrSequence<i32>>,
        keepdims: bool,
        ddof: f64,
    ) -> PyResult<Self> {
        crate::ops::std(slf, axis, keepdims, ddof)
    }

    /// Computes the variance along one or more axes. See [`jix.var()`][jix.var].
    #[pyo3(signature = (axis=None, *, keepdims=false, ddof=0.0))]
    pub fn var(
        slf: &Bound<'_, Self>,
        axis: Option<ItemOrSequence<i32>>,
        keepdims: bool,
        ddof: f64,
    ) -> PyResult<Self> {
        crate::ops::var(slf, axis, keepdims, ddof)
    }

    /// Casts each element of the array to a new dtype. See [`jix.astype()`][jix.astype].
    pub fn astype<'py>(
        slf: &Bound<'py, Self>,
        dtype: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        crate::ops::astype(slf, dtype)
    }

    // == shape ops ==

    /// Reinterprets an array with a different shape. See [`jix.reshape()`][jix.reshape].
    pub fn reshape<'py>(
        slf: &Bound<'py, Self>,
        shape: ItemOrSequence<i64>,
    ) -> PyResult<Bound<'py, Self>> {
        crate::ops::reshape(slf, shape)
    }

    /// Collapses the array into a single dimension. See [`jix.flatten()`][jix.flatten].
    pub fn flatten<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, Self>> {
        crate::ops::flatten(slf)
    }

    /// Reorders the axes of an array (generalized transpose). See [`jix.permute_axes()`][jix.permute_axes].
    #[pyo3(signature = (axes=None))]
    pub fn permute_axes<'py>(
        slf: &Bound<'py, Self>,
        axes: Option<Vec<usize>>,
    ) -> PyResult<Bound<'py, Self>> {
        crate::ops::permute_axes(slf, axes)
    }

    /// Reverses all axes; shorthand for `permute_axes()` with no arguments. See [`jix.permute_axes()`][jix.permute_axes].
    #[allow(non_snake_case)]
    #[getter]
    pub fn T<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, Array>> {
        crate::ops::permute_axes(slf, None)
    }

    /// Expands the array to a larger shape by repeating elements along length-1 dimensions. See [`jix.broadcast()`][jix.broadcast].
    #[pyo3(signature = (shape))]
    pub fn broadcast<'py>(
        slf: &Bound<'py, Array>,
        shape: ItemOrSequence<i64>,
    ) -> PyResult<Bound<'py, Array>> {
        crate::ops::broadcast(slf, shape)
    }

    /// Selects a sub-region of the array as a lazy view. See [`jix.slice()`][jix.slice].
    pub fn slice<'py>(
        slf: &Bound<'py, Array>,
        index: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Array>> {
        crate::ops::slice(slf, index)
    }

    /// Removes length-1 dimensions from the array's shape. See [`jix.squeeze()`][jix.squeeze].
    #[pyo3(signature = (axis=None))]
    pub fn squeeze<'py>(
        slf: &Bound<'py, Array>,
        axis: Option<ItemOrSequence<i32>>,
    ) -> PyResult<Bound<'py, Array>> {
        crate::ops::squeeze(slf, axis)
    }

    /// Inserts new length-1 dimensions at specified positions in the array's shape. See [`jix.unsqueeze()`][jix.unsqueeze].
    pub fn unsqueeze<'py>(
        slf: &Bound<'py, Array>,
        axis: ItemOrSequence<i32>,
    ) -> PyResult<Bound<'py, Array>> {
        crate::ops::unsqueeze(slf, axis)
    }

    /// Repeats each element along the given axis. See [`jix.repeat()`][jix.repeat].
    pub fn repeat<'py>(
        slf: &Bound<'py, Array>,
        repeats: u64,
        axis: Option<i32>,
    ) -> PyResult<Bound<'py, Array>> {
        crate::ops::repeat(slf, repeats, axis)
    }

    /// Reverses the order of elements along the given axis. See [`jix.flip()`][jix.flip].
    #[pyo3(signature = (axis=None))]
    pub fn flip<'py>(
        slf: &Bound<'py, Array>,
        axis: Option<ItemOrSequence<i32>>,
    ) -> PyResult<Bound<'py, Array>> {
        crate::ops::flip(slf, axis)
    }

    /// Rolls elements along an axis, wrapping at the boundary. See [`jix.roll()`][jix.roll].
    #[pyo3(signature = (shift, axis=None))]
    pub fn roll<'py>(
        slf: &Bound<'py, Array>,
        shift: i64,
        axis: Option<i32>,
    ) -> PyResult<Bound<'py, Array>> {
        crate::ops::roll(slf, shift, axis)
    }

    /// Replicates the array along a single axis. See [`jix.tile()`][jix.tile].
    #[pyo3(signature = (repeats, axis=None))]
    pub fn tile<'py>(
        slf: &Bound<'py, Array>,
        repeats: u64,
        axis: Option<i32>,
    ) -> PyResult<Bound<'py, Array>> {
        crate::ops::tile(slf, repeats, axis)
    }

    // == complex ops ==

    /// Extracts the real part of each complex element. See [`jix.real()`][jix.real].
    #[getter]
    pub fn real(slf: &Bound<'_, Array>) -> PyResult<Array> {
        crate::ops::real(slf)
    }

    /// Extracts the imaginary part of each complex element. See [`jix.imag()`][jix.imag].
    #[getter]
    pub fn imag(slf: &Bound<'_, Array>) -> PyResult<Array> {
        crate::ops::imag(slf)
    }

    // == trigonometric ops ==

    /// Computes the sine of each element (input in radians). See [`jix.sin()`][jix.sin].
    pub fn sin(slf: &Bound<'_, Self>) -> PyResult<Self> {
        crate::ops::sin(slf)
    }

    /// Computes the cosine of each element (input in radians). See [`jix.cos()`][jix.cos].
    pub fn cos(slf: &Bound<'_, Self>) -> PyResult<Self> {
        crate::ops::cos(slf)
    }

    /// Computes the tangent of each element (input in radians). See [`jix.tan()`][jix.tan].
    pub fn tan(slf: &Bound<'_, Self>) -> PyResult<Self> {
        crate::ops::tan(slf)
    }

    /// Computes the arcsine of each element; output is in radians in `[-pi/2, pi/2]`. See [`jix.asin()`][jix.asin].
    pub fn asin(slf: &Bound<'_, Self>) -> PyResult<Self> {
        crate::ops::asin(slf)
    }

    /// Computes the arccosine of each element; output is in radians in `[0, pi]`. See [`jix.acos()`][jix.acos].
    pub fn acos(slf: &Bound<'_, Self>) -> PyResult<Self> {
        crate::ops::acos(slf)
    }

    /// Computes the arctangent of each element; output is in radians in `(-pi/2, pi/2)`. See [`jix.atan()`][jix.atan].
    pub fn atan(slf: &Bound<'_, Self>) -> PyResult<Self> {
        crate::ops::atan(slf)
    }

    /// Computes the logarithm of each element; defaults to the natural logarithm. See [`jix.log()`][jix.log].
    #[pyo3(signature = (base=None))]
    pub fn log(slf: &Bound<'_, Self>, base: Option<f64>) -> PyResult<Self> {
        crate::ops::log(slf, base)
    }

    /// Clamps each element to `[min, max]`. See [`jix.clamp()`][jix.clamp].
    #[pyo3(signature = (min=None, max=None))]
    pub fn clamp<'py>(
        slf: &Bound<'py, Self>,
        min: Option<&Bound<'py, PyAny>>,
        max: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, Self>> {
        crate::ops::clamp(slf, min, max)
    }

    /// Returns the sign of each element as a floating-point value. See [`jix.sign()`][jix.sign].
    pub fn sign(slf: &Bound<'_, Self>) -> PyResult<Self> {
        crate::ops::sign(slf)
    }

    // == float predicates ==

    /// Tests whether each element is finite. See [`jix.is_finite()`][jix.is_finite].
    pub fn is_finite(slf: &Bound<'_, Self>) -> PyResult<Self> {
        crate::ops::is_finite(slf)
    }

    /// Tests whether each element is infinite. See [`jix.is_infinite()`][jix.is_infinite].
    pub fn is_infinite(slf: &Bound<'_, Self>) -> PyResult<Self> {
        crate::ops::is_infinite(slf)
    }

    /// Tests whether each element is `NaN`. See [`jix.is_nan()`][jix.is_nan].
    pub fn is_nan(slf: &Bound<'_, Self>) -> PyResult<Self> {
        crate::ops::is_nan(slf)
    }

    // == axis ops ==

    /// Inserts new length-1 dimensions at specified positions. See [`jix.insert_axis()`][jix.insert_axis].
    pub fn insert_axis<'py>(
        slf: &Bound<'py, Array>,
        axis: ItemOrSequence<i32>,
    ) -> PyResult<Bound<'py, Array>> {
        crate::ops::insert_axis(slf, axis)
    }

    /// Removes length-1 dimensions from the array's shape. See [`jix.remove_axis()`][jix.remove_axis].
    pub fn remove_axis<'py>(
        slf: &Bound<'py, Array>,
        axis: ItemOrSequence<i32>,
    ) -> PyResult<Bound<'py, Array>> {
        crate::ops::remove_axis(slf, axis)
    }
}

/// Compact any array-like object to a new [`jix.Array`][jix.Array] by compressing it into new blocks.
///
/// Accepts Python scalars, lists, tuples, NumPy arrays, and any other object accepted by
/// [`numpy.asarray`](https://numpy.org/doc/stable/reference/generated/numpy.asarray.html).
///
/// A new jix compact array is created, with all the input data compressed into blocks. The data
/// is compressed even if the input is already a jix array.
///
/// The primary use of `compact`, other than compacting non-jix objects, is to materialize a
/// lazy operation chain. A [`jix.Array`][jix.Array] can wrap an arbitrary lazy computation - for
/// example the result of `a * 2.0 + b`. Reads to such lazy arrays always perform the whole
/// computation pipeline on the fly, which is very flexible but can be inefficient for repeated
/// access. Calling `array_view.compact()` breaks the lazy chain and materializes the result as a
/// standalone compressed array.
///
/// In contrast to "simple" views such as unary element-wise operations, lazy ops that change
/// the shape of the array (e.g. `reshape`, `broadcast`, `permute_axes`) can cause block
/// boundaries to no longer align with the logical layout of the array, causing reads to
/// decompress excess data. Calling `compact` on the result of such an operation re-encodes the
/// data with a freshly derived block shape that matches the new layout. The block shape is
/// automatically derived using a heuristic that aims to preserve user choices, but it is not
/// perfect - pass explicit `params` after shape-changing ops when you know the access pattern.
///
/// Codec settings (compression level, filters, etc.) are inherited from the source storage if the
/// source is a jix array. Otherwise, they are either passed explicitly via `params` or chosen
/// automatically based on the input dtype and size. See `params` for details.
///
/// Note:
///     **On copy** (e.g. [`jix.compact()`][jix.Array.compact]): a new compressed array is created,
///     inheriting any unset fields from the source array's storage. After shape-changing operations
///     (`reshape`, `permute_axes`, etc.) the inherited block layout may not suit the new
///     shape - consider passing explicit params to `.compact()` after such ops.
///
/// Args:
///     array: The input data to compress. May be a Python scalar, list, tuple, NumPy array,
///         jix array, or any other object accepted by `jix.asarray`.
///     dtype: Optional dtype to cast the array to before compressing. Accepts anything
///         `numpy.dtype()` accepts. When omitted the input dtype is preserved.
///     params: Controls the block layout and codec configuration:
///
///         - **Block layout** - the nd-block shape used to divide the array into independently
///           compressed blocks. A good block layout is critical for performance and should match the
///           access pattern of your workload.
///         - **Codec** - compression settings used when writing and reading blocks. The defaults
///           (Zstd level 3 with byte shuffling, block sized to fit in the L1 data cache) are
///           suitable for most workloads.
///
///         When omitted, defaults are chosen automatically.
///         If the source array is a jix array, unset fields are inherited from the source storage.
///         A dictionary with the following optional keys:
///
///         - `block_shape`: Explicit storage block shape, as a list of integers (one per
///             dimension). When set, array data is divided into nd-blocks of exactly this shape
///             (each dimension is clamped to the array boundary). Choosing a block shape that
///             matches your access pattern is the most important tuning knob: if you always read
///             row slices, a block shape of `[1, <row_length>]` avoids decompressing neighboring
///             rows. When not set, the shape is auto-computed to fit approximately
///             `block_size` bytes.
///         - `block_shape_tag`: Per-dimension constraint on how `block_shape` is scaled when a
///             downstream operation auto-computes a new block shape. One string per dimension:
///             `"fixed"` pins the block size exactly (the default when `block_shape` is set by
///             the user); `"multiple-of"` allows scaling up while keeping it a multiple of the
///             given value; `"any"` allows free choice (used when an op makes the original size
///             irrelevant, e.g. a broadcast dimension). Requires `block_shape` to also be set.
///             Length must equal the number of dimensions.
///         - `block_size`: Target block size in bytes, used when auto-computing or scaling the
///             block shape for dimensions that are not `"fixed"`. Ignored when all dimensions
///             are `"fixed"`. Defaults to the L1 data cache size.
///         - `read_size`: Target size in bytes for the preferred read region.
///             Defaults to the L1 cache size.
///         - `codec`: Compression algorithm applied to each block. Currently the only accepted
///             value is `"zstd"`. Defaults to `"zstd"` when left unset.
///         - `compression_level`: Compression level passed to the codec. For Zstd the valid range
///             is 1-22; higher values compress more but are slower to encode. Defaults to 3.
///         - `filters`: List of filters applied to the raw block bytes *before* compression.
///             Filters improve the compression ratio for typed numeric data: `"byte-shuffle"`
///             groups bytes by significance (e.g. all high bytes together, then all low bytes);
///             `"bit-shuffle"` groups bits across elements. Defaults to `["byte-shuffle"]`.
///     context: An optional [`jix.ReadContext`][jix.ReadContext] to reuse when decoding the source
///         array. When omitted, a context is created internally. Unused if the source array is not
///         a jix array.
///
/// Returns:
///     A new compact [`jix.Array`][jix.Array] with all data compressed into blocks.
///
/// Raises:
///     ValueError: If the input cannot be converted by `jix.asarray`, the array has more
///         dimensions than jix supports, or the array has negative strides (e.g. a reversed
///         slice `a[::-1]`).
///
/// Examples:
///     ```python
///     import jix
///     import numpy as np
///
///     # Create new compact arrays from various inputs
///     a = jix.compact([1, 2, 3]) # from python list
///     assert a.numpy().tolist() == [1, 2, 3]
///     b = jix.compact(np.array([66, 8])) # from numpy
///     assert b.numpy().tolist() == [66, 8]
///     c = jix.compact(a, params={"block_shape": [2]}) # from another jix array (recompress)
///     assert c.numpy().tolist() == [1, 2, 3]
///
///     # Compacting a lazy computation pipeline
///     d = jix.compact([[1.5, 2.0], [3.14, 6.17]], dtype=np.float32)
///     e = (d * 7.399) \    # Array<Mul<Compact, Scalar<f32>>> (lazy views, rust internal types
///         .floor() \       # Array<Floor<Mul<Compact, Scalar<f32>>>>
///         .compact()       # Array<Compact> - materialize the pipeline
///
///     # After a shape-changing op, pin the block shape explicitly
///     f = jix.compact(d.T, params={"block_shape": [2, 1]})
///     ```
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (array, *, dtype=None, params=None, context=None))]
pub fn compact<'py>(
    array: &Bound<'py, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
    params: Option<Bound<'_, PyDict>>,
    context: Option<&Bound<'_, ReadContext>>,
) -> PyResult<Bound<'py, Array>> {
    let py = array.py();
    let mut array = crate::asarray(array)?;
    let params = resolve_array_params(py, params)?;
    let context = context.map(|ctx| ctx.get());

    if let Some(dtype) = dtype {
        array = crate::ops::astype(&array, dtype)?;
    }
    let array = &array.get().arr;
    let array = py.detach(|| {
        let context_guard;
        let context = match context {
            Some(ctx) => {
                context_guard = ctx.lock();
                &*context_guard
            }
            None => &array.read_ctx(),
        };

        array.compact_with(params, context).into_py_result()
    })?;

    Bound::new(py, Array::from_core(array.into_any()))
}
pub(crate) fn resolve_array_params(
    py: Python<'_>,
    params: Option<Bound<'_, PyDict>>,
) -> PyResult<jix_core::ArrayParams> {
    match params {
        None => Ok(jix_core::ArrayParams::default()),
        Some(kwargs) => {
            let mut kwargs = kwargs.extract::<BTreeMap<String, Py<PyAny>>>()?;
            macro_rules! extract_arg {
                ($key:expr, $ty:ty) => {
                    kwargs
                        .remove($key)
                        .map(|v| {
                            v.bind(py).extract::<$ty>().map_err(|e| {
                                PyTypeError::new_err(format!(
                                    "{} must be of type {}: {e}",
                                    $key,
                                    stringify!($ty)
                                ))
                            })
                        })
                        .transpose()
                };
            }
            let block_shape = extract_arg!("block_shape", Vec<u32>)?;
            let block_shape_tag = extract_arg!("block_shape_tag", Vec<String>)?;
            let block_size = extract_arg!("block_size", u64)?;
            let read_size = extract_arg!("read_size", u64)?;
            let codec = extract_arg!("codec", String)?;
            let compression_level = extract_arg!("compression_level", u32)?;
            let filters = extract_arg!("filters", Vec<String>)?;
            if !kwargs.is_empty() {
                return Err(PyTypeError::new_err(format!(
                    "Unexpected array params kwargs: {}",
                    kwargs.into_keys().collect::<Vec<_>>().join(", ")
                )));
            }

            let mut params = jix_core::ArrayParams::default();
            if let Some(block_shape) = block_shape {
                params.block_shape(&block_shape);
            }
            if let Some(block_shape_tag) = block_shape_tag {
                let block_shape_tag = block_shape_tag
                    .iter()
                    .map(|s| match s.as_str() {
                        "fixed" => Ok(jix_core::storage::BlockShapeTag::Fixed),
                        "multiple-of" => Ok(jix_core::storage::BlockShapeTag::MultipleOf),
                        "any" => Ok(jix_core::storage::BlockShapeTag::Any),
                        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "Invalid block_shape_tag: {s}"
                        ))),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                params.block_shape_tag(&block_shape_tag);
            }
            if let Some(block_size) = block_size {
                params.block_size(block_size);
            }
            if let Some(read_size) = read_size {
                params.read_size(read_size);
            }

            if let Some(codec) = codec {
                match codec.as_str() {
                    "zstd" => {
                        params.codec(Codec::Zstd);
                    }
                    _ => {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "Unsupported codec: {codec}"
                        )));
                    }
                }
            }
            if let Some(compression_level) = compression_level {
                params.level(compression_level).into_py_result()?;
            }
            if let Some(filters) = filters {
                let filters = filters
                    .into_iter()
                    .map(|filter| match filter.as_str() {
                        "byte-shuffle" => Ok(Filter::ByteShuffle),
                        "bit-shuffle" => Ok(Filter::BitShuffle),
                        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "Unsupported filter: {filter}"
                        ))),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                params.filters(&filters).into_py_result()?;
            }

            Ok(params)
        }
    }
}

#[cfg(test)]
mod tests {
    use jix_core::dtype::Dtyped;
    use jix_core::{Array as CoreArray, IntoDimension};
    use ndarray::{array, ArrayD};
    use numpy::{PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
    use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
    use pyo3::types::{PyAny, PyEllipsis, PySlice, PyTuple};
    use pyo3::{Bound, IntoPyObject, Python};

    use super::Array;

    fn make_py_array<'py, T: Dtyped, D>(
        py: Python<'py>,
        ndarray: &ndarray::Array<T, D>,
    ) -> Bound<'py, Array>
    where
        D: ndarray::Dimension + IntoDimension<Dimension: 'static>,
    {
        let core = CoreArray::compact_ndarray(ndarray).unwrap();
        let core = core.into_type_dyn().into_dim_dyn();
        Bound::new(py, Array::from_core(core.into_any())).unwrap()
    }

    fn roundtrip<T, D>(original: &ndarray::Array<T, D>) -> ndarray::Array<T, D>
    where
        T: Dtyped + numpy::Element + Copy,
        D: ndarray::Dimension + IntoDimension<Dimension: 'static>,
    {
        // ndarray::Array -> jix_core::Array -> jix_python::Array -> numpy::PyArray -> ndarray::Array
        Python::attach(|py| {
            let py_arr = make_py_array(py, &original);
            let np = py_arr.get().numpy(py, None, None).unwrap();
            let typed = np.cast_into::<PyArrayDyn<T>>().unwrap();
            typed.to_owned_array().into_dimensionality().unwrap()
        })
    }

    #[test]
    fn test_numpy_f32_1d() {
        let original = array![1.0f32, 2.0, 3.0, 4.0];
        assert_eq!(roundtrip(&original), original);
    }

    #[test]
    fn test_numpy_f32_2d() {
        let original = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        assert_eq!(roundtrip(&original), original);
    }

    #[test]
    fn test_numpy_f32_3d() {
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let original = ndarray::Array::from_shape_vec(vec![2, 3, 4], data).unwrap();
        assert_eq!(roundtrip(&original), original);
    }

    #[test]
    fn test_numpy_f64_2d() {
        let original =
            ndarray::Array::from_shape_vec(vec![3, 4], (0..12).map(|x| x as f64).collect())
                .unwrap();
        assert_eq!(roundtrip(&original), original);
    }

    #[test]
    fn test_numpy_i32_2d() {
        let original = ndarray::Array::from_shape_vec(vec![4, 5], (0..20).collect()).unwrap();
        assert_eq!(roundtrip(&original), original);
    }

    #[test]
    fn test_numpy_i64_1d() {
        let original = ndarray::Array::from_shape_vec(vec![8], (100..108).collect()).unwrap();
        assert_eq!(roundtrip(&original), original);
    }

    #[test]
    fn test_numpy_u8_2d() {
        let original = ndarray::Array::from_shape_vec(vec![3, 3], (0u8..9).collect()).unwrap();
        assert_eq!(roundtrip(&original), original);
    }

    #[test]
    fn test_numpy_u32_3d() {
        let data: Vec<u32> = (0..60).collect();
        let original = ndarray::Array::from_shape_vec(vec![3, 4, 5], data).unwrap();
        assert_eq!(roundtrip(&original), original);
    }

    #[test]
    fn test_numpy_bool_1d() {
        let original = array![true, false, true, true, false, true];
        assert_eq!(roundtrip(&original), original);
    }

    #[test]
    fn test_numpy_large_values_f64() {
        // Verify large/negative values are transferred without corruption.
        let original = array![[f64::MAX, f64::MIN, -1.0], [0.0, 1.0, f64::INFINITY]];
        assert_eq!(roundtrip(&original), original);
    }

    #[test]
    fn test_numpy_shape_preserved() {
        let original =
            ndarray::Array::from_shape_vec(vec![2, 3, 4], (0..24).map(|x| x as f32).collect())
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
        assert_eq!(roundtrip(&original), original);
    }

    #[test]
    fn test_numpy_non_square_2d() {
        let data: Vec<f32> = (0..100).map(|x| x as f32).collect();
        let original = ndarray::Array::from_shape_vec(vec![10, 10], data).unwrap();
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
        let data = ndarray::Array::from_shape_vec(vec![5], (0..5).collect()).unwrap();
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
        let data = ndarray::Array::from_shape_vec(vec![5], (0..5).collect()).unwrap();
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
        let data = ndarray::Array::from_shape_vec(vec![5], (0..5).collect()).unwrap();
        Python::attach(|py| {
            let py_arr = make_py_array(py, &data);
            let key = 5i64.into_pyobject(py).unwrap().into_any();
            let err = Array::__getitem__(py_arr.get(), key.as_any()).unwrap_err();
            assert!(err.is_instance_of::<PyIndexError>(py));
        });
    }

    #[test]
    fn test_getitem_int_negative_out_of_bounds() {
        let data = ndarray::Array::from_shape_vec(vec![5], (0..5).collect()).unwrap();
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
        let data =
            ndarray::Array::from_shape_vec(vec![5], (0..5).map(|x| x as f32).collect()).unwrap();
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
        let data =
            ndarray::Array::from_shape_vec(vec![5], (0..5).map(|x| x as f32).collect()).unwrap();
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
        let data =
            ndarray::Array::from_shape_vec(vec![5], (0..5).map(|x| x as f32).collect()).unwrap();
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
        let data = ndarray::Array::from_shape_vec(vec![5], (0..5).collect()).unwrap();
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
        let data = ndarray::Array::from_shape_vec(vec![5], (0..5).collect()).unwrap();
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
        let data = ndarray::Array::from_shape_vec(vec![5], (0..5).collect()).unwrap();
        Python::attach(|py| {
            let py_arr = make_py_array(py, &data);
            let key = PySlice::new(py, 2, 2, 1).into_any();
            let result = getitem::<i32>(&py_arr, key.as_any());
            assert_eq!(result.shape(), &[0usize]);
        });
    }

    #[test]
    fn test_getitem_slice_step_not_one_err() {
        let data = ndarray::Array::from_shape_vec(vec![5], (0..5).collect()).unwrap();
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
        // arr[0] on shape [2, 3] -> first row, shape [3]
        let data =
            ndarray::Array::from_shape_vec(vec![2, 3], (0..6).map(|x| x as f32).collect()).unwrap();
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
        // arr[1, 2] on shape [2, 3] -> scalar, shape []
        let data =
            ndarray::Array::from_shape_vec(vec![2, 3], (0..6).map(|x| x as f32).collect()).unwrap();
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
        // arr[:, 1] on shape [2, 3] -> column 1, shape [2]
        let data =
            ndarray::Array::from_shape_vec(vec![2, 3], (0..6).map(|x| x as f32).collect()).unwrap();
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
        // arr[0:2, 1:3] on shape [3, 3] -> shape [2, 2]
        let data =
            ndarray::Array::from_shape_vec(vec![3, 3], (0..9).map(|x| x as f32).collect()).unwrap();
        Python::attach(|py| {
            let py_arr = make_py_array(py, &data);
            let items: Vec<Bound<'_, PyAny>> = vec![
                PySlice::new(py, 0, 2, 1).into_any(),
                PySlice::new(py, 1, 3, 1).into_any(),
            ];
            let key = PyTuple::new(py, items).unwrap().into_any();
            let result = getitem::<f32>(&py_arr, key.as_any());
            // [[0,1,2],[3,4,5],[6,7,8]] -> rows 0-1, cols 1-2 -> [[1,2],[4,5]]
            let expected =
                ndarray::Array::from_shape_vec(vec![2, 2], vec![1.0f32, 2.0, 4.0, 5.0]).unwrap();
            assert_eq!(result, expected);
        });
    }

    // --- __getitem__: ellipsis ---

    #[test]
    fn test_getitem_ellipsis_full() {
        // arr[...] -> entire array unchanged
        let data = ndarray::Array::from_shape_vec(vec![2, 3], (0..6).collect()).unwrap();
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
        // arr[0, ...] on shape [2, 3] -> row 0, shape [3]
        let data =
            ndarray::Array::from_shape_vec(vec![2, 3], (0..6).map(|x| x as f32).collect()).unwrap();
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
        // arr[..., 0] on shape [2, 3] -> column 0, shape [2]
        let data =
            ndarray::Array::from_shape_vec(vec![2, 3], (0..6).map(|x| x as f32).collect()).unwrap();
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
        let data = ndarray::Array::from_shape_vec(vec![2, 3], (0..6).collect()).unwrap();
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
        let data = ndarray::Array::from_shape_vec(vec![2, 3], (0..6).collect()).unwrap();
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
        let data = ndarray::Array::from_shape_vec(vec![5], (0..5).collect()).unwrap();
        Python::attach(|py| {
            let py_arr = make_py_array(py, &data);
            let key = "bad".into_pyobject(py).unwrap().into_any();
            let err = Array::__getitem__(py_arr.get(), key.as_any()).unwrap_err();
            assert!(err.is_instance_of::<PyTypeError>(py));
        });
    }

    #[test]
    fn test_ops_chained() {
        // (a + b) * c  computed both in jix and ndarray
        let a = array![1.0f32, 2.0, 3.0, 4.0];
        let b = array![4.0f32, 3.0, 2.0, 1.0];
        let c = array![2.0f32, 2.0, 2.0, 2.0];
        let jix_result = Python::attach(|py| {
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
        assert_eq!(jix_result, ((a + b) * c).into_dyn());
    }
}
