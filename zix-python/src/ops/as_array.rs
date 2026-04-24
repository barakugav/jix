use std::sync::Arc;

use numpy::{PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::PyOverflowError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;
use zix_core::dtype::{f16, Complex, DtypeScalarKind, Dtyped};
use zix_core::storage::Scalar;

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
    // already a zix array
    if let Ok(s) = value.cast::<Array>() {
        return Ok(s.clone());
    };

    // convert to numpy array
    let py = value.py();
    let numpy_asarray = numpy::get_array_module(py)?.getattr("asarray")?;
    let array = numpy_asarray.call1((value,))?;
    let array = array.cast::<PyUntypedArray>()?;

    // scalar
    let dtype = dtype_from_numpy(array.dtype())?;
    if array.ndim() == 0 {
        if let Some(scalar) = dtype.try_to_scalar() {
            let item = array.call_method0("item")?;

            fn scalar_array<'py, T>(py: Python<'py>, item: T) -> PyResult<Bound<'py, Array>>
            where
                T: Dtyped,
            {
                let storage = Scalar::new(item, &[]).into_py_result()?;
                Bound::new(py, Array::from_storage(DynStorage(Arc::new(storage))))
            }

            return match scalar {
                DtypeScalarKind::I8 => scalar_array(py, item.extract::<i8>()?),
                DtypeScalarKind::I16 => scalar_array(py, item.extract::<i16>()?),
                DtypeScalarKind::I32 => scalar_array(py, item.extract::<i32>()?),
                DtypeScalarKind::I64 => scalar_array(py, item.extract::<i64>()?),
                DtypeScalarKind::U8 => scalar_array(py, item.extract::<u8>()?),
                DtypeScalarKind::U16 => scalar_array(py, item.extract::<u16>()?),
                DtypeScalarKind::U32 => scalar_array(py, item.extract::<u32>()?),
                DtypeScalarKind::U64 => scalar_array(py, item.extract::<u64>()?),
                DtypeScalarKind::F16 => scalar_array(py, f16::from_f32(item.extract::<f32>()?)),
                DtypeScalarKind::F32 => scalar_array(py, item.extract::<f32>()?),
                DtypeScalarKind::F64 => scalar_array(py, item.extract::<f64>()?),
                DtypeScalarKind::ComplexF32 => {
                    let re = item.getattr("real")?.extract::<f32>()?;
                    let im = item.getattr("imag")?.extract::<f32>()?;
                    scalar_array(py, Complex::new(re, im))
                }
                DtypeScalarKind::ComplexF64 => {
                    let re = item.getattr("real")?.extract::<f64>()?;
                    let im = item.getattr("imag")?.extract::<f64>()?;
                    scalar_array(py, Complex::new(re, im))
                }
                DtypeScalarKind::Bool => scalar_array(py, item.extract::<bool>()?),
            };
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

    Bound::new(py, Array::from_storage(DynStorage(Arc::new(storage))))
}

pub(crate) fn as_core_array<'py>(
    value: &Bound<'py, PyAny>,
) -> PyResult<zix_core::Array<DynStorage>> {
    Ok(asarray(value)?.borrow().to_core_array())
}
#[cfg(test)]
mod tests {
    use ndarray::{array, Array0, ArrayD, IxDyn};
    use numpy::{PyArray, PyArrayDyn};
    use pyo3::prelude::*;
    use zix_core::dtype::{Complex, Dtyped};

    use super::asarray;

    /// Call `asarray` and read back the data as an ndarray.
    fn collect<T: Dtyped>(val: &Bound<'_, PyAny>) -> ArrayD<T> {
        asarray(val)
            .unwrap()
            .borrow()
            .arr
            .to_ndarray::<T>()
            .unwrap()
    }

    // ── 0-D: Python scalars ──────────────────────────────────────────────────

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

    // ── 0-D: typed numpy scalars ─────────────────────────────────────────────

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

    // ── 0-D: complex scalars ─────────────────────────────────────────────────

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

    // ── multi-dimensional arrays ─────────────────────────────────────────────

    fn npd<T>(py: Python<'_>, arr: ArrayD<T>) -> Bound<'_, PyAny>
    where
        T: numpy::Element,
    {
        PyArrayDyn::<T>::from_array(py, &arr).into_any()
    }

    #[test]
    fn test_1d_f32() {
        Python::attach(|py| {
            let orig = array![1.0f32, 2.0, 3.0, 4.0].into_dyn();
            let data = collect::<f32>(&npd(py, orig.clone()));
            assert_eq!(data, orig);
        });
    }

    #[test]
    fn test_1d_i64() {
        Python::attach(|py| {
            let orig = array![10i64, 20, 30].into_dyn();
            let data = collect::<i64>(&npd(py, orig.clone()));
            assert_eq!(data, orig);
        });
    }

    #[test]
    fn test_2d_f64() {
        Python::attach(|py| {
            let orig = array![[1.0f64, 2.0], [3.0, 4.0], [5.0, 6.0]].into_dyn();
            let data = collect::<f64>(&npd(py, orig.clone()));
            assert_eq!(data, orig);
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
            assert_eq!(arr.borrow().arr.shape(), &[2u64, 3, 4]);
        });
    }

    // ── passthrough ──────────────────────────────────────────────────────────

    #[test]
    fn test_passthrough() {
        Python::attach(|py| {
            let orig = array![1.0f32, 2.0].into_dyn();
            let arr1 = asarray(&npd(py, orig)).unwrap();
            let ptr1 = arr1.as_ptr();
            let arr2 = asarray(arr1.as_any()).unwrap();
            assert_eq!(arr2.as_ptr(), ptr1);
        });
    }

    // ── error cases ──────────────────────────────────────────────────────────

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
