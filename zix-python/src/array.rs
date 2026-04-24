use std::sync::Arc;

use numpy::{PyArrayDescr, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyEllipsis, PyTuple};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use zix_core::ops::SliceItem;
use zix_core::storage::ArrayStorage;
use zix_core::Array as ZixArray;

use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::types::{PyAnyMethods, PySlice};
use std::ops::Range;

use crate::dtype::dtype_to_numpy;
use crate::storage::DynStorage;
use crate::util::{dim_arr, numpy_empty, DimArray, IntoPyResult};

#[gen_stub_pyclass]
#[pyclass]
pub struct Array {
    pub(crate) arr: ZixArray<DynStorage>,
}
impl Array {
    pub(crate) fn from_storage(storage: DynStorage) -> Self {
        Self {
            arr: ZixArray::from_storage(storage),
        }
    }

    pub(crate) fn from_core_storage(storage: impl ArrayStorage + Send + Sync + 'static) -> Self {
        Self::from_storage(DynStorage(Arc::new(storage)))
    }

    pub(crate) fn to_core_array(&self) -> ZixArray<DynStorage> {
        ZixArray::from_storage(self.arr.storage().clone())
    }

    pub fn to_numpy<'py>(
        &self,
        py: Python<'py>,
        index: &[Range<u64>],
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

        let np_arr = numpy_empty(self.dtype_numpy(py)?, &read_shape)?;
        let np_arr_data_ptr = unsafe { (*np_arr.as_array_ptr()).data.cast::<u8>() };
        let np_arr_data_size = itemsize * read_shape.iter().product::<u64>() as usize;
        let np_arr_data =
            unsafe { std::slice::from_raw_parts_mut(np_arr_data_ptr, np_arr_data_size) };

        if np_arr_data_size > 0 {
            py.detach(|| {
                self.arr
                    .data()
                    .to_ndarray_buf(index, np_arr_data)
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
    pub fn ndim(&self) -> usize {
        self.arr.shape().len()
    }

    pub fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.arr.shape().iter().copied())
    }

    #[pyo3(signature = (axis=None))]
    pub fn size(&self, axis: Option<usize>) -> PyResult<usize> {
        let shape = self.arr.shape();
        match axis {
            Some(axis) => {
                if axis >= shape.len() {
                    Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "axis {} is out of bounds for array with ndim {}",
                        axis,
                        shape.len()
                    )))
                } else {
                    Ok(shape[axis] as usize)
                }
            }
            None => Ok(shape.iter().map(|&s| s as usize).product()),
        }
    }

    pub fn dtype_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDescr>> {
        dtype_to_numpy(py, self.arr.dtype())
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
    /// * **dtype** — identical to `self.dtype_numpy()`. No casting is performed.
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
    #[pyo3(signature = (index=None))]
    pub fn numpy<'py>(
        &self,
        py: Python<'py>,
        index: Option<&Bound<'py, PyAny>>,
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
        let np_arr = self.to_numpy(py, &ranges)?;

        let np_arr: Bound<'_, PyUntypedArray> =
            np_arr.call_method1("reshape", (out_shape,))?.cast_into()?;
        Ok(np_arr)
    }

    /// Read elements from the array (or a sub-region of it) and return them as a NumPy array.
    ///
    /// This function is identical to `numpy()`, see that method for details.
    fn __getitem__<'py>(
        slf: &Bound<'py, Self>,
        key: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyUntypedArray>> {
        Self::numpy(&slf.borrow(), slf.py(), Some(key))
    }

    pub fn __add__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        crate::ops::add(slf, other)
    }

    pub fn __sub__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        crate::ops::subtract(slf, other)
    }

    pub fn __mul__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        crate::ops::multiply(slf, other)
    }

    pub fn __truediv__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Self> {
        crate::ops::divide(slf, other)
    }
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
    use zix_core::storage::Compact;
    use zix_core::{Array as ZixArray, ArrayParams};

    use super::{Array, DynStorage};

    fn make_py_array<'py, T: Dtyped>(py: Python<'py>, ndarray: &ArrayD<T>) -> Bound<'py, Array> {
        let core = ZixArray::<Compact>::from_ndarray(ndarray, ArrayParams::default()).unwrap();
        let dyn_storage = DynStorage(Arc::new(core.into_storage()));
        Bound::new(py, Array::from_storage(dyn_storage)).unwrap()
    }

    fn roundtrip<T>(original: &ArrayD<T>) -> ArrayD<T>
    where
        T: Dtyped + numpy::Element + Copy,
    {
        // ndarray::Array -> zix_core::Array -> zix_python::Array -> numpy::PyArray -> ndarray::Array
        Python::attach(|py| {
            let py_arr = make_py_array(py, &original);
            let np = py_arr.borrow().numpy(py, None).unwrap();
            let typed = np.cast_into::<PyArrayDyn<T>>().unwrap();
            typed.to_owned_array()
        })
    }

    #[test]
    fn test_numpy_f32_1d() {
        let original: ArrayD<f32> = array![1.0f32, 2.0, 3.0, 4.0].into_dyn();
        assert_eq!(roundtrip(&original), original);
    }

    #[test]
    fn test_numpy_f32_2d() {
        let original: ArrayD<f32> = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn();
        assert_eq!(roundtrip(&original), original);
    }

    #[test]
    fn test_numpy_f32_3d() {
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let original: ArrayD<f32> =
            ndarray::Array::from_shape_vec(IxDyn(&[2, 3, 4]), data).unwrap();
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
        let original: ArrayD<i32> =
            ndarray::Array::from_shape_vec(IxDyn(&[4, 5]), (0..20).collect()).unwrap();
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
        let original: ArrayD<bool> = array![true, false, true, true, false, true].into_dyn();
        assert_eq!(roundtrip(&original), original);
    }

    #[test]
    fn test_numpy_large_values_f64() {
        // Verify large/negative values are transferred without corruption.
        let original: ArrayD<f64> =
            array![[f64::MAX, f64::MIN, -1.0], [0.0, 1.0, f64::INFINITY]].into_dyn();
        assert_eq!(roundtrip(&original), original);
    }

    #[test]
    fn test_numpy_shape_preserved() {
        let original: ArrayD<f32> =
            ndarray::Array::from_shape_vec(IxDyn(&[2, 3, 4]), (0..24).map(|x| x as f32).collect())
                .unwrap();
        Python::attach(|py| {
            let py_arr = make_py_array(py, &original);
            let np = py_arr.borrow().numpy(py, None).unwrap();
            assert_eq!(np.shape(), &[2usize, 3, 4]);
        });
    }

    #[test]
    fn test_numpy_dtype_preserved_f32() {
        use numpy::PyArrayDescrMethods;
        let original: ArrayD<f32> = array![1.0f32, 2.0].into_dyn();
        Python::attach(|py| {
            let py_arr = make_py_array(py, &original);
            let np = py_arr.borrow().numpy(py, None).unwrap();
            assert_eq!(np.dtype().itemsize(), 4);
            assert_eq!(np.dtype().kind() as char, 'f');
        });
    }

    #[test]
    fn test_numpy_dtype_preserved_i32() {
        use numpy::PyArrayDescrMethods;
        let original: ArrayD<i32> = array![1i32, 2, 3].into_dyn();
        Python::attach(|py| {
            let py_arr = make_py_array(py, &original);
            let np = py_arr.borrow().numpy(py, None).unwrap();
            assert_eq!(np.dtype().itemsize(), 4);
            assert_eq!(np.dtype().kind() as char, 'i');
        });
    }

    #[test]
    fn test_numpy_single_element() {
        let original: ArrayD<f64> = array![42.0f64].into_dyn();
        assert_eq!(roundtrip(&original), original);
    }

    #[test]
    fn test_numpy_non_square_2d() {
        let data: Vec<f32> = (0..100).map(|x| x as f32).collect();
        let original: ArrayD<f32> = ndarray::Array::from_shape_vec(IxDyn(&[10, 10]), data).unwrap();
        assert_eq!(roundtrip(&original), original);
    }

    fn eval<T>(py_arr: Array) -> ArrayD<T>
    where
        T: Dtyped + numpy::Element + Copy,
    {
        Python::attach(|py| {
            py_arr
                .numpy(py, None)
                .unwrap()
                .cast_into::<PyArrayDyn<T>>()
                .unwrap()
                .to_owned_array()
        })
    }

    #[test]
    fn test_add_f32() {
        let a: ArrayD<f32> = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn();
        let b: ArrayD<f32> = array![[10.0f32, 20.0, 30.0], [40.0, 50.0, 60.0]].into_dyn();
        let result = Python::attach(|py| {
            let a = make_py_array(py, &a);
            let b = make_py_array(py, &b);
            eval::<f32>(Array::__add__(&a, &b).unwrap())
        });
        assert_eq!(result, a + b);
    }

    #[test]
    fn test_sub_f32() {
        let a: ArrayD<f32> = array![[10.0f32, 20.0, 30.0], [40.0, 50.0, 60.0]].into_dyn();
        let b: ArrayD<f32> = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn();
        let result = Python::attach(|py| {
            let a = make_py_array(py, &a);
            let b = make_py_array(py, &b);
            eval::<f32>(Array::__sub__(&a, &b).unwrap())
        });
        assert_eq!(result, a - b);
    }

    #[test]
    fn test_mul_f32() {
        let a: ArrayD<f32> = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn();
        let b: ArrayD<f32> = array![[2.0f32, 3.0, 4.0], [5.0, 6.0, 7.0]].into_dyn();
        let result = Python::attach(|py| {
            let a = make_py_array(py, &a);
            let b = make_py_array(py, &b);
            eval::<f32>(Array::__mul__(&a, &b).unwrap())
        });
        assert_eq!(result, a * b);
    }

    #[test]
    fn test_div_f32() {
        let a: ArrayD<f32> = array![[2.0f32, 6.0, 12.0], [20.0, 30.0, 42.0]].into_dyn();
        let b: ArrayD<f32> = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn();
        let result = Python::attach(|py| {
            let a = make_py_array(py, &a);
            let b = make_py_array(py, &b);
            eval::<f32>(Array::__truediv__(&a, &b).unwrap())
        });
        assert_eq!(result, a / b);
    }

    #[test]
    fn test_add_f64() {
        let a: ArrayD<f64> = array![1.0f64, 2.0, 3.0].into_dyn();
        let b: ArrayD<f64> = array![0.5f64, 1.5, 2.5].into_dyn();
        let result = Python::attach(|py| {
            let a = make_py_array(py, &a);
            let b = make_py_array(py, &b);
            eval::<f64>(Array::__add__(&a, &b).unwrap())
        });
        assert_eq!(result, a + b);
    }

    #[test]
    fn test_add_i32() {
        let a: ArrayD<i32> = array![[1i32, 2], [3, 4]].into_dyn();
        let b: ArrayD<i32> = array![[10i32, 20], [30, 40]].into_dyn();
        let result = Python::attach(|py| {
            let a = make_py_array(py, &a);
            let b = make_py_array(py, &b);
            eval::<i32>(Array::__add__(&a, &b).unwrap())
        });
        assert_eq!(result, a + b);
    }

    #[test]
    fn test_sub_i32() {
        let a: ArrayD<i32> = array![[10i32, 20], [30, 40]].into_dyn();
        let b: ArrayD<i32> = array![[1i32, 2], [3, 4]].into_dyn();
        let result = Python::attach(|py| {
            let a = make_py_array(py, &a);
            let b = make_py_array(py, &b);
            eval::<i32>(Array::__sub__(&a, &b).unwrap())
        });
        assert_eq!(result, a - b);
    }

    #[test]
    fn test_mul_i32() {
        let a: ArrayD<i32> = array![[1i32, 2], [3, 4]].into_dyn();
        let b: ArrayD<i32> = array![[5i32, 6], [7, 8]].into_dyn();
        let result = Python::attach(|py| {
            let a = make_py_array(py, &a);
            let b = make_py_array(py, &b);
            eval::<i32>(Array::__mul__(&a, &b).unwrap())
        });
        assert_eq!(result, a * b);
    }

    fn getitem<T>(py_arr: &Bound<'_, Array>, key: &Bound<'_, PyAny>) -> ArrayD<T>
    where
        T: Dtyped + numpy::Element + Copy,
    {
        Array::__getitem__(py_arr, key)
            .unwrap()
            .cast_into::<PyArrayDyn<T>>()
            .unwrap()
            .to_owned_array()
    }

    // --- __getitem__: integer indexing ---

    #[test]
    fn test_getitem_int_positive() {
        let data: ArrayD<i32> =
            ndarray::Array::from_shape_vec(IxDyn(&[5]), (0..5).collect()).unwrap();
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
        let data: ArrayD<i32> =
            ndarray::Array::from_shape_vec(IxDyn(&[5]), (0..5).collect()).unwrap();
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
        let data: ArrayD<i32> =
            ndarray::Array::from_shape_vec(IxDyn(&[5]), (0..5).collect()).unwrap();
        Python::attach(|py| {
            let py_arr = make_py_array(py, &data);
            let key = 5i64.into_pyobject(py).unwrap().into_any();
            let err = Array::__getitem__(&py_arr, key.as_any()).unwrap_err();
            assert!(err.is_instance_of::<PyIndexError>(py));
        });
    }

    #[test]
    fn test_getitem_int_negative_out_of_bounds() {
        let data: ArrayD<i32> =
            ndarray::Array::from_shape_vec(IxDyn(&[5]), (0..5).collect()).unwrap();
        Python::attach(|py| {
            let py_arr = make_py_array(py, &data);
            let key = (-6i64).into_pyobject(py).unwrap().into_any();
            let err = Array::__getitem__(&py_arr, key.as_any()).unwrap_err();
            assert!(err.is_instance_of::<PyIndexError>(py));
        });
    }

    // --- __getitem__: slice indexing ---

    #[test]
    fn test_getitem_slice_start_stop() {
        let data: ArrayD<f32> =
            ndarray::Array::from_shape_vec(IxDyn(&[5]), (0..5).map(|x| x as f32).collect())
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
        let data: ArrayD<f32> =
            ndarray::Array::from_shape_vec(IxDyn(&[5]), (0..5).map(|x| x as f32).collect())
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
        let data: ArrayD<f32> =
            ndarray::Array::from_shape_vec(IxDyn(&[5]), (0..5).map(|x| x as f32).collect())
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
        let data: ArrayD<i32> =
            ndarray::Array::from_shape_vec(IxDyn(&[5]), (0..5).collect()).unwrap();
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
        let data: ArrayD<i32> =
            ndarray::Array::from_shape_vec(IxDyn(&[5]), (0..5).collect()).unwrap();
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
        let data: ArrayD<i32> =
            ndarray::Array::from_shape_vec(IxDyn(&[5]), (0..5).collect()).unwrap();
        Python::attach(|py| {
            let py_arr = make_py_array(py, &data);
            let key = PySlice::new(py, 2, 2, 1).into_any();
            let result = getitem::<i32>(&py_arr, key.as_any());
            assert_eq!(result.shape(), &[0usize]);
        });
    }

    #[test]
    fn test_getitem_slice_step_not_one_err() {
        let data: ArrayD<i32> =
            ndarray::Array::from_shape_vec(IxDyn(&[5]), (0..5).collect()).unwrap();
        Python::attach(|py| {
            let py_arr = make_py_array(py, &data);
            let key = py.eval(c"slice(None, None, 2)", None, None).unwrap();
            let err = Array::__getitem__(&py_arr, &key).unwrap_err();
            assert!(err.is_instance_of::<PyValueError>(py));
        });
    }

    // --- __getitem__: 2-D indexing ---

    #[test]
    fn test_getitem_2d_row_int() {
        // arr[0] on shape [2, 3] → first row, shape [3]
        let data: ArrayD<f32> =
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
        let data: ArrayD<f32> =
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
        let data: ArrayD<f32> =
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
        let data: ArrayD<f32> =
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
        let data: ArrayD<i32> =
            ndarray::Array::from_shape_vec(IxDyn(&[2, 3]), (0..6).collect()).unwrap();
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
        let data: ArrayD<f32> =
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
        let data: ArrayD<f32> =
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
        let data: ArrayD<i32> =
            ndarray::Array::from_shape_vec(IxDyn(&[2, 3]), (0..6).collect()).unwrap();
        Python::attach(|py| {
            let py_arr = make_py_array(py, &data);
            let items: Vec<Bound<'_, PyAny>> = vec![
                0i64.into_pyobject(py).unwrap().into_any(),
                1i64.into_pyobject(py).unwrap().into_any(),
                2i64.into_pyobject(py).unwrap().into_any(),
            ];
            let key = PyTuple::new(py, items).unwrap().into_any();
            let err = Array::__getitem__(&py_arr, key.as_any()).unwrap_err();
            assert!(err.is_instance_of::<PyIndexError>(py));
        });
    }

    #[test]
    fn test_getitem_multiple_ellipses_err() {
        let data: ArrayD<i32> =
            ndarray::Array::from_shape_vec(IxDyn(&[2, 3]), (0..6).collect()).unwrap();
        Python::attach(|py| {
            let py_arr = make_py_array(py, &data);
            let items: Vec<Bound<'_, PyAny>> = vec![
                PyEllipsis::get(py).to_owned().into_any(),
                PyEllipsis::get(py).to_owned().into_any(),
            ];
            let key = PyTuple::new(py, items).unwrap().into_any();
            let err = Array::__getitem__(&py_arr, key.as_any()).unwrap_err();
            assert!(err.is_instance_of::<PyIndexError>(py));
        });
    }

    #[test]
    fn test_getitem_invalid_type_err() {
        let data: ArrayD<i32> =
            ndarray::Array::from_shape_vec(IxDyn(&[5]), (0..5).collect()).unwrap();
        Python::attach(|py| {
            let py_arr = make_py_array(py, &data);
            let key = "bad".into_pyobject(py).unwrap().into_any();
            let err = Array::__getitem__(&py_arr, key.as_any()).unwrap_err();
            assert!(err.is_instance_of::<PyTypeError>(py));
        });
    }

    #[test]
    fn test_ops_chained() {
        // (a + b) * c  computed both in zix and ndarray
        let a: ArrayD<f32> = array![1.0f32, 2.0, 3.0, 4.0].into_dyn();
        let b: ArrayD<f32> = array![4.0f32, 3.0, 2.0, 1.0].into_dyn();
        let c: ArrayD<f32> = array![2.0f32, 2.0, 2.0, 2.0].into_dyn();
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
        assert_eq!(zix_result, (a + b) * c);
    }
}
