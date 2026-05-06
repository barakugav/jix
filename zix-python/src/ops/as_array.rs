use std::sync::Arc;

use numpy::{PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::PyOverflowError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;
use zix_core::dtype::{f16, Complex, DtypeScalarKind, Dtyped};
use zix_core::storage::{Plain, Scalar as ScalarStorage};
use zix_core::Array as ZixArray;

use crate::array::Array;
use crate::dtype::dtype_from_numpy;
use crate::storage::DynStorage;
use crate::util::{check_ndim, DimArray, IntoPyResult};

/// Convert any array-like object to a zix [`Array`].
///
/// Accepts Python scalars, lists, tuples, NumPy arrays, and any other object accepted by
/// [`numpy.asarray`](https://numpy.org/doc/stable/reference/generated/numpy.asarray.html).
///
/// # Storage
///
/// - If `value` is already an `Array`, it is returned as-is with no copy.
/// - 0-dimensional inputs produce a scalar array backed by a single value (no buffer).
/// - All other inputs share the underlying buffer with the intermediate NumPy array (zero-copy);
///   the NumPy array is kept alive for as long as the returned `Array` is alive.
///
/// # Errors
///
/// - If the input cannot be converted by `numpy.asarray`.
/// - If the array has more dimensions than zix supports.
/// - If the array has negative strides (e.g. a reversed slice `a[::-1]`).
#[gen_stub_pyfunction]
#[pyfunction]
pub fn asarray<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Array>> {
    asarray_impl(value)?.into_py_array(value.py())
}

pub(crate) enum AsArray<'py> {
    Array(Bound<'py, Array>),
    Numpy(ZixArray<Plain<Py<PyUntypedArray>>>),
    Scalar(ScalarAsArray),
}
impl<'py> AsArray<'py> {
    pub(crate) fn into_py_array(self, py: Python<'py>) -> PyResult<Bound<'py, Array>> {
        let numpy = match self {
            AsArray::Array(array) => return Ok(array),
            AsArray::Numpy(numpy) => NumpyAsArray::Numpy(numpy),
            AsArray::Scalar(scalar) => NumpyAsArray::Scalar(scalar),
        };
        Bound::new(py, numpy.into_py_array(None)?)
    }
}

pub(crate) fn asarray_impl<'py>(value: &Bound<'py, PyAny>) -> PyResult<AsArray<'py>> {
    Ok(if let Ok(value) = value.cast::<Array>() {
        // already a zix array
        AsArray::Array(value.clone())
    } else {
        // convert to numpy array
        let array = NumpyAsArray::from_any(value)?;
        match array {
            NumpyAsArray::Numpy(array) => AsArray::Numpy(array),
            NumpyAsArray::Scalar(scalar) => AsArray::Scalar(scalar),
        }
    })
}

pub(crate) fn asarray2<'py>(
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, Array>, Bound<'py, Array>)> {
    let py = a.py();
    let mut a = asarray_impl(a)?;
    let mut b = asarray_impl(b)?;

    fn get_shape(asarray: &AsArray) -> Option<Vec<u64>> {
        match asarray {
            AsArray::Array(array) => Some(array.get().arr.shape().to_vec()),
            AsArray::Numpy(array) => Some(array.shape().to_vec()),
            AsArray::Scalar(_) => None,
        }
    }
    let shape = if let Some(a) = get_shape(&a) {
        Some(a)
    } else if let Some(b) = get_shape(&b) {
        Some(b)
    } else {
        None
    };

    fn asarray_broadcast_if_scalar<'py>(
        value: &mut AsArray<'py>,
        broadcast: &[u64],
    ) -> Result<(), zix_core::Error> {
        match value {
            AsArray::Scalar(scalar_array) => match scalar_array {
                ScalarAsArray::I8(scalar) => {
                    *scalar = ZixArray::from_storage(ScalarStorage::new(
                        *scalar.storage().data(),
                        broadcast,
                    )?)
                }
                ScalarAsArray::I16(scalar) => {
                    *scalar = ZixArray::from_storage(ScalarStorage::new(
                        *scalar.storage().data(),
                        broadcast,
                    )?)
                }
                ScalarAsArray::I32(scalar) => {
                    *scalar = ZixArray::from_storage(ScalarStorage::new(
                        *scalar.storage().data(),
                        broadcast,
                    )?)
                }
                ScalarAsArray::I64(scalar) => {
                    *scalar = ZixArray::from_storage(ScalarStorage::new(
                        *scalar.storage().data(),
                        broadcast,
                    )?)
                }
                ScalarAsArray::U8(scalar) => {
                    *scalar = ZixArray::from_storage(ScalarStorage::new(
                        *scalar.storage().data(),
                        broadcast,
                    )?)
                }
                ScalarAsArray::U16(scalar) => {
                    *scalar = ZixArray::from_storage(ScalarStorage::new(
                        *scalar.storage().data(),
                        broadcast,
                    )?)
                }
                ScalarAsArray::U32(scalar) => {
                    *scalar = ZixArray::from_storage(ScalarStorage::new(
                        *scalar.storage().data(),
                        broadcast,
                    )?)
                }
                ScalarAsArray::U64(scalar) => {
                    *scalar = ZixArray::from_storage(ScalarStorage::new(
                        *scalar.storage().data(),
                        broadcast,
                    )?)
                }
                ScalarAsArray::F16(scalar) => {
                    *scalar = ZixArray::from_storage(ScalarStorage::new(
                        *scalar.storage().data(),
                        broadcast,
                    )?)
                }
                ScalarAsArray::F32(scalar) => {
                    *scalar = ZixArray::from_storage(ScalarStorage::new(
                        *scalar.storage().data(),
                        broadcast,
                    )?)
                }
                ScalarAsArray::F64(scalar) => {
                    *scalar = ZixArray::from_storage(ScalarStorage::new(
                        *scalar.storage().data(),
                        broadcast,
                    )?)
                }
                ScalarAsArray::ComplexF32(scalar) => {
                    *scalar = ZixArray::from_storage(ScalarStorage::new(
                        *scalar.storage().data(),
                        broadcast,
                    )?)
                }
                ScalarAsArray::ComplexF64(scalar) => {
                    *scalar = ZixArray::from_storage(ScalarStorage::new(
                        *scalar.storage().data(),
                        broadcast,
                    )?)
                }
                ScalarAsArray::Bool(scalar) => {
                    *scalar = ZixArray::from_storage(ScalarStorage::new(
                        *scalar.storage().data(),
                        broadcast,
                    )?)
                }
            },
            _ => {}
        };
        Ok(())
    }

    if let Some(shape) = shape {
        asarray_broadcast_if_scalar(&mut a, &shape).into_py_result()?;
        asarray_broadcast_if_scalar(&mut b, &shape).into_py_result()?;
    }

    let a = a.into_py_array(py)?;
    let b = b.into_py_array(py)?;
    Ok((a, b))
}

pub(crate) enum ScalarAsArray {
    I8(ZixArray<ScalarStorage<i8>>),
    I16(ZixArray<ScalarStorage<i16>>),
    I32(ZixArray<ScalarStorage<i32>>),
    I64(ZixArray<ScalarStorage<i64>>),
    U8(ZixArray<ScalarStorage<u8>>),
    U16(ZixArray<ScalarStorage<u16>>),
    U32(ZixArray<ScalarStorage<u32>>),
    U64(ZixArray<ScalarStorage<u64>>),
    F16(ZixArray<ScalarStorage<f16>>),
    F32(ZixArray<ScalarStorage<f32>>),
    F64(ZixArray<ScalarStorage<f64>>),
    ComplexF32(ZixArray<ScalarStorage<Complex<f32>>>),
    ComplexF64(ZixArray<ScalarStorage<Complex<f64>>>),
    Bool(ZixArray<ScalarStorage<bool>>),
}

pub(crate) enum NumpyAsArray {
    Numpy(ZixArray<Plain<Py<PyUntypedArray>>>),
    Scalar(ScalarAsArray),
}
impl NumpyAsArray {
    pub(crate) fn from_any(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        let py = value.py();
        let numpy_asarray = numpy::get_array_module(py)?.getattr("asarray")?;
        let array = numpy_asarray.call1((value,))?;
        let array = array.cast::<PyUntypedArray>()?;
        NumpyAsArray::new(array)
    }
    pub(crate) fn new(array: &Bound<numpy::PyUntypedArray>) -> PyResult<Self> {
        // scalar
        let dtype = dtype_from_numpy(&array.dtype())?;
        if array.ndim() == 0 {
            if let Some(scalar) = dtype.try_to_scalar() {
                let item = array.call_method0("item")?;

                fn scalar_array<T>(item: T) -> PyResult<ZixArray<ScalarStorage<T>>>
                where
                    T: Dtyped,
                {
                    let storage = ScalarStorage::new(item, &[]).into_py_result()?;
                    Ok(ZixArray::from_storage(storage))
                }

                let scalar = match scalar {
                    DtypeScalarKind::I8 => ScalarAsArray::I8(scalar_array(item.extract::<i8>()?)?),
                    DtypeScalarKind::I16 => {
                        ScalarAsArray::I16(scalar_array(item.extract::<i16>()?)?)
                    }
                    DtypeScalarKind::I32 => {
                        ScalarAsArray::I32(scalar_array(item.extract::<i32>()?)?)
                    }
                    DtypeScalarKind::I64 => {
                        ScalarAsArray::I64(scalar_array(item.extract::<i64>()?)?)
                    }
                    DtypeScalarKind::U8 => ScalarAsArray::U8(scalar_array(item.extract::<u8>()?)?),
                    DtypeScalarKind::U16 => {
                        ScalarAsArray::U16(scalar_array(item.extract::<u16>()?)?)
                    }
                    DtypeScalarKind::U32 => {
                        ScalarAsArray::U32(scalar_array(item.extract::<u32>()?)?)
                    }
                    DtypeScalarKind::U64 => {
                        ScalarAsArray::U64(scalar_array(item.extract::<u64>()?)?)
                    }
                    DtypeScalarKind::F16 => {
                        ScalarAsArray::F16(scalar_array(f16::from_f32(item.extract::<f32>()?))?)
                    }
                    DtypeScalarKind::F32 => {
                        ScalarAsArray::F32(scalar_array(item.extract::<f32>()?)?)
                    }
                    DtypeScalarKind::F64 => {
                        ScalarAsArray::F64(scalar_array(item.extract::<f64>()?)?)
                    }
                    DtypeScalarKind::ComplexF32 => {
                        let re = item.getattr("real")?.extract::<f32>()?;
                        let im = item.getattr("imag")?.extract::<f32>()?;
                        ScalarAsArray::ComplexF32(scalar_array(Complex::new(re, im))?)
                    }
                    DtypeScalarKind::ComplexF64 => {
                        let re = item.getattr("real")?.extract::<f64>()?;
                        let im = item.getattr("imag")?.extract::<f64>()?;
                        ScalarAsArray::ComplexF64(scalar_array(Complex::new(re, im))?)
                    }
                    DtypeScalarKind::Bool => {
                        ScalarAsArray::Bool(scalar_array(item.extract::<bool>()?)?)
                    }
                };
                return Ok(Self::Scalar(scalar));
            }
        }

        // array
        check_ndim(array.ndim())?;
        let shape = array
            .shape()
            .iter()
            .map(|&d| d as u64)
            .collect::<DimArray<_>>();
        let strides = array
            .strides()
            .iter()
            .map(|&s| {
                usize::try_from(s)
                    .map_err(|_| PyOverflowError::new_err("Negative strides are not supported"))
            })
            .collect::<PyResult<DimArray<_>>>()?;
        let data_ptr = {
            let arr_ptr = unsafe { &*array.as_array_ptr() };
            arr_ptr.data.cast_const().cast::<u8>()
        };
        let array = array.clone().unbind();
        let storage =
            unsafe { zix_core::storage::Plain::new(array, data_ptr, &shape, &strides, dtype) };
        let storage = storage.into_py_result()?;

        Ok(NumpyAsArray::Numpy(ZixArray::from_storage(storage)))
    }

    pub(crate) fn into_py_array(self, optional_broadcast: Option<&[u64]>) -> PyResult<Array> {
        fn broadcast_scalar<T>(
            scalar_array: ZixArray<ScalarStorage<T>>,
            scalar_shape: Option<&[u64]>,
        ) -> PyResult<DynStorage>
        where
            T: Dtyped,
        {
            Ok(match scalar_shape {
                None => DynStorage(Arc::new(scalar_array.into_storage())),
                Some(scalar_shape) => {
                    assert!(scalar_array.shape().is_empty());
                    let scalar = scalar_array.storage().data();
                    let storage = ScalarStorage::new(*scalar, scalar_shape).into_py_result()?;
                    DynStorage(Arc::new(storage))
                }
            })
        }
        Ok(Array::from_storage(match self {
            NumpyAsArray::Numpy(array) => DynStorage(Arc::new(array.into_storage())),
            NumpyAsArray::Scalar(scalar) => match scalar {
                ScalarAsArray::I8(value) => broadcast_scalar(value, optional_broadcast)?,
                ScalarAsArray::I16(value) => broadcast_scalar(value, optional_broadcast)?,
                ScalarAsArray::I32(value) => broadcast_scalar(value, optional_broadcast)?,
                ScalarAsArray::I64(value) => broadcast_scalar(value, optional_broadcast)?,
                ScalarAsArray::U8(value) => broadcast_scalar(value, optional_broadcast)?,
                ScalarAsArray::U16(value) => broadcast_scalar(value, optional_broadcast)?,
                ScalarAsArray::U32(value) => broadcast_scalar(value, optional_broadcast)?,
                ScalarAsArray::U64(value) => broadcast_scalar(value, optional_broadcast)?,
                ScalarAsArray::F16(value) => broadcast_scalar(value, optional_broadcast)?,
                ScalarAsArray::F32(value) => broadcast_scalar(value, optional_broadcast)?,
                ScalarAsArray::F64(value) => broadcast_scalar(value, optional_broadcast)?,
                ScalarAsArray::ComplexF32(value) => broadcast_scalar(value, optional_broadcast)?,
                ScalarAsArray::ComplexF64(value) => broadcast_scalar(value, optional_broadcast)?,
                ScalarAsArray::Bool(value) => broadcast_scalar(value, optional_broadcast)?,
            },
        }))
    }
}

pub(crate) fn any_to_core_array<'py>(
    value: &Bound<'py, PyAny>,
) -> PyResult<zix_core::Array<DynStorage>> {
    Ok(asarray(value)?.get().to_core_array())
}
#[cfg(test)]
mod tests {
    use ndarray::{array, Array0, ArrayD, IxDyn};
    use numpy::PyArray;
    use pyo3::prelude::*;
    use zix_core::dtype::{Complex, Dtyped};

    use super::asarray;

    /// Call `asarray` and read back the data as an ndarray.
    fn collect<T: Dtyped>(val: &Bound<'_, PyAny>) -> ArrayD<T> {
        asarray(val).unwrap().get().arr.to_ndarray::<T>().unwrap()
    }

    // -- 0-D: Python scalars --------------------------------------------------

    #[test]
    fn test_python_int() {
        Python::attach(|py| {
            let val = 42i64.into_pyobject(py).unwrap();
            let data = collect::<i64>(val.as_any());
            assert_eq!(data.ndim(), 0);
            assert_eq!(*data.first().unwrap(), 42);
        });
    }

    #[test]
    fn test_python_float() {
        Python::attach(|py| {
            let val = 2.5f64.into_pyobject(py).unwrap();
            let data = collect::<f64>(val.as_any());
            assert_eq!(data.ndim(), 0);
            assert_eq!(*data.first().unwrap(), 2.5);
        });
    }

    #[test]
    fn test_python_bool() {
        Python::attach(|py| {
            let val = true.into_pyobject(py).unwrap();
            let data = collect::<bool>(val.as_any());
            assert_eq!(data.ndim(), 0);
            assert!(*data.first().unwrap());
        });
    }

    // -- 0-D: typed numpy scalars ---------------------------------------------

    fn np0<T>(py: Python<'_>, v: T) -> Bound<'_, PyAny>
    where
        T: numpy::Element + Copy,
    {
        PyArray::<T, _>::from_array(py, &Array0::from_elem((), v)).into_any()
    }

    #[test]
    fn test_0d_i8() {
        Python::attach(|py| {
            let data = collect::<i8>(&np0(py, -5i8));
            assert_eq!(data.ndim(), 0);
            assert_eq!(*data.first().unwrap(), -5i8);
        });
    }

    #[test]
    fn test_0d_i32() {
        Python::attach(|py| {
            let data = collect::<i32>(&np0(py, -7i32));
            assert_eq!(data.ndim(), 0);
            assert_eq!(*data.first().unwrap(), -7i32);
        });
    }

    #[test]
    fn test_0d_u8() {
        Python::attach(|py| {
            let data = collect::<u8>(&np0(py, 255u8));
            assert_eq!(data.ndim(), 0);
            assert_eq!(*data.first().unwrap(), 255u8);
        });
    }

    #[test]
    fn test_0d_f32() {
        Python::attach(|py| {
            let data = collect::<f32>(&np0(py, 1.5f32));
            assert_eq!(data.ndim(), 0);
            assert_eq!(*data.first().unwrap(), 1.5f32);
        });
    }

    #[test]
    fn test_0d_f64() {
        Python::attach(|py| {
            let data = collect::<f64>(&np0(py, -3.14f64));
            assert_eq!(data.ndim(), 0);
            assert_eq!(*data.first().unwrap(), -3.14f64);
        });
    }

    #[test]
    fn test_0d_bool() {
        Python::attach(|py| {
            let data = collect::<bool>(&np0(py, false));
            assert_eq!(data.ndim(), 0);
            assert!(!*data.first().unwrap());
        });
    }

    // -- 0-D: complex scalars -------------------------------------------------

    #[test]
    fn test_0d_complex_f32() {
        Python::attach(|py| {
            let val = py
                .eval(
                    cr#"__import__('numpy').array(complex(1, -2), dtype='complex64')"#,
                    None,
                    None,
                )
                .unwrap();
            let data = collect::<Complex<f32>>(&val);
            assert_eq!(data.ndim(), 0);
            assert_eq!(*data.first().unwrap(), Complex::new(1.0f32, -2.0f32));
        });
    }

    #[test]
    fn test_0d_complex_f64() {
        Python::attach(|py| {
            let val = py
                .eval(cr#"__import__('numpy').array(complex(3, 4))"#, None, None)
                .unwrap();
            let data = collect::<Complex<f64>>(&val);
            assert_eq!(data.ndim(), 0);
            assert_eq!(*data.first().unwrap(), Complex::new(3.0f64, 4.0f64));
        });
    }

    // -- multi-dimensional arrays ---------------------------------------------

    fn npd<T, D>(py: Python<'_>, arr: ndarray::Array<T, D>) -> Bound<'_, PyAny>
    where
        T: numpy::Element,
        D: ndarray::Dimension,
    {
        PyArray::<T, D>::from_array(py, &arr).into_any()
    }

    #[test]
    fn test_1d_f32() {
        Python::attach(|py| {
            let orig = array![1.0f32, 2.0, 3.0, 4.0];
            let data = collect::<f32>(&npd(py, orig.clone()));
            assert_eq!(data, orig.into_dyn());
        });
    }

    #[test]
    fn test_1d_i64() {
        Python::attach(|py| {
            let orig = array![10i64, 20, 30];
            let data = collect::<i64>(&npd(py, orig.clone()));
            assert_eq!(data, orig.into_dyn());
        });
    }

    #[test]
    fn test_2d_f64() {
        Python::attach(|py| {
            let orig = array![[1.0f64, 2.0], [3.0, 4.0], [5.0, 6.0]];
            let data = collect::<f64>(&npd(py, orig.clone()));
            assert_eq!(data, orig.into_dyn());
        });
    }

    #[test]
    fn test_2d_u8() {
        Python::attach(|py| {
            let orig: ArrayD<u8> =
                ArrayD::from_shape_vec(IxDyn(&[3, 4]), (0u8..12).collect()).unwrap();
            let data = collect::<u8>(&npd(py, orig.clone()));
            assert_eq!(data, orig);
        });
    }

    #[test]
    fn test_3d_i32() {
        Python::attach(|py| {
            let orig: ArrayD<i32> =
                ArrayD::from_shape_vec(IxDyn(&[2, 3, 4]), (0..24).collect()).unwrap();
            let data = collect::<i32>(&npd(py, orig.clone()));
            assert_eq!(data, orig);
        });
    }

    #[test]
    fn test_shape_preserved() {
        Python::attach(|py| {
            let orig: ArrayD<f32> =
                ArrayD::from_shape_vec(IxDyn(&[2, 3, 4]), (0..24).map(|x| x as f32).collect())
                    .unwrap();
            let arr = asarray(&npd(py, orig)).unwrap();
            assert_eq!(arr.get().arr.shape(), &[2u64, 3, 4]);
        });
    }

    // -- passthrough ----------------------------------------------------------

    #[test]
    fn test_passthrough() {
        Python::attach(|py| {
            let orig = array![1.0f32, 2.0];
            let arr1 = asarray(&npd(py, orig)).unwrap();
            let ptr1 = arr1.as_ptr();
            let arr2 = asarray(arr1.as_any()).unwrap();
            assert_eq!(arr2.as_ptr(), ptr1);
        });
    }

    // -- error cases ----------------------------------------------------------

    #[test]
    fn test_negative_stride_error() {
        Python::attach(|py| {
            let val = py
                .eval(cr#"__import__('numpy').array([1, 2, 3])[::-1]"#, None, None)
                .unwrap();
            assert!(asarray(&val)
                .unwrap_err()
                .is_instance_of::<pyo3::exceptions::PyOverflowError>(py));
        });
    }
}
