use std::sync::Arc;

use numpy::npyffi::npy_intp;
use numpy::{PyArrayDescrMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use zix_core::array::Array as ZixArray;

use crate::dtype::dtype_to_numpy;
use crate::storage::DynStorage;
use crate::util::dim_arr;

#[gen_stub_pyclass]
#[pyclass]
pub struct Array(ZixArray<DynStorage>);

#[gen_stub_pymethods]
#[pymethods]
impl Array {
    /// Export the array as a NumPy array.
    ///
    /// Allocates a new C-contiguous NumPy array with the same shape and dtype, decodes all blocks
    /// into it, and returns it. The returned array is independent of this array — mutations to one
    /// do not affect the other.
    pub fn numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyUntypedArray>> {
        let dtype = self.0.dtype();
        let dtype_np = dtype_to_numpy(py, dtype)?;
        let shape = self.0.shape();
        let ndim = shape.len();
        let shape = dim_arr(ndim, |dim| shape[dim] as npy_intp);
        let is_fortran = false;
        let np_arr = unsafe {
            numpy::PY_ARRAY_API.PyArray_Empty(
                py,
                shape.len() as _,
                shape.as_ptr().cast_mut(),
                dtype_np.into_dtype_ptr(),
                if is_fortran { -1 } else { 0 },
            )
        };
        let np_arr = unsafe { Bound::from_owned_ptr(py, np_arr).cast_into_unchecked() };
        let np_arr_data_ptr = unsafe { (*np_arr.as_array_ptr()).data.cast::<u8>() };
        let np_arr_data_size =
            dtype.itemsize() as usize * shape.iter().map(|s| *s as usize).product::<usize>();
        let np_arr_data =
            unsafe { std::slice::from_raw_parts_mut(np_arr_data_ptr, np_arr_data_size) };

        py.detach(|| {
            let range = dim_arr(ndim, |dim| 0..(shape[dim] as usize));
            self.0.data().to_ndarray_buf(&range, np_arr_data)
        })?;

        Ok(np_arr)
    }

    pub fn __add__(&self, _other: &Self) -> PyResult<Self> {
        let a = ZixArray::from_storage(self.0.storage().clone());
        let b = ZixArray::from_storage(self.0.storage().clone());
        let storage = DynStorage(Arc::new(zix_core::ops::Add::new(a, b)?));
        Ok(Self(ZixArray::from_storage(storage)))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ndarray::{ArrayD, IxDyn, array};
    use numpy::{PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
    use pyo3::Python;
    use zix_core::array::Array as ZixArray;
    use zix_core::dtype::Dtyped;
    use zix_core::storage::Owned;

    use super::{Array, DynStorage};

    fn make_py_array<T: Dtyped>(ndarray: &ArrayD<T>) -> Array {
        let block_shape: Vec<usize> = ndarray.shape().to_vec();
        let core = ZixArray::<Owned>::from_ndarray(ndarray, &block_shape).unwrap();
        let dyn_storage = DynStorage(Arc::new(core.into_storage()));
        Array(ZixArray::from_storage(dyn_storage))
    }

    fn roundtrip<T>(original: &ArrayD<T>) -> ArrayD<T>
    where
        T: Dtyped + numpy::Element + Copy,
    {
        // ndarray::Array -> zix_core::Array -> zix_python::Array -> numpy::PyArray -> ndarray::Array
        Python::attach(|py| {
            let py_arr = make_py_array(&original);
            let np = py_arr.numpy(py).unwrap();
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
            let py_arr = make_py_array(&original);
            let np = py_arr.numpy(py).unwrap();
            assert_eq!(np.shape(), &[2usize, 3, 4]);
        });
    }

    #[test]
    fn test_numpy_dtype_preserved_f32() {
        use numpy::PyArrayDescrMethods;
        let original: ArrayD<f32> = array![1.0f32, 2.0].into_dyn();
        Python::attach(|py| {
            let py_arr = make_py_array(&original);
            let np = py_arr.numpy(py).unwrap();
            assert_eq!(np.dtype().itemsize(), 4);
            assert_eq!(np.dtype().kind() as char, 'f');
        });
    }

    #[test]
    fn test_numpy_dtype_preserved_i32() {
        use numpy::PyArrayDescrMethods;
        let original: ArrayD<i32> = array![1i32, 2, 3].into_dyn();
        Python::attach(|py| {
            let py_arr = make_py_array(&original);
            let np = py_arr.numpy(py).unwrap();
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
}
