use std::sync::Arc;

use numpy::{PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::PyOverflowError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;
use zix_core::dtype::{f16, Complex, Dtype, DtypeScalarKind, Dtyped};
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

    pub(crate) fn dtype(asarray: &AsArray) -> Dtype {
        match asarray {
            AsArray::Array(array) => array.get().arr.dtype().clone(),
            AsArray::Numpy(array) => array.dtype().clone(),
            AsArray::Scalar(scalar) => match scalar {
                ScalarAsArray::I8(_) => i8::DTYPE,
                ScalarAsArray::I16(_) => i16::DTYPE,
                ScalarAsArray::I32(_) => i32::DTYPE,
                ScalarAsArray::I64(_) => i64::DTYPE,
                ScalarAsArray::U8(_) => u8::DTYPE,
                ScalarAsArray::U16(_) => u16::DTYPE,
                ScalarAsArray::U32(_) => u32::DTYPE,
                ScalarAsArray::U64(_) => u64::DTYPE,
                ScalarAsArray::F16(_) => f16::DTYPE,
                ScalarAsArray::F32(_) => f32::DTYPE,
                ScalarAsArray::F64(_) => f64::DTYPE,
                ScalarAsArray::ComplexF32(_) => Complex::<f32>::DTYPE,
                ScalarAsArray::ComplexF64(_) => Complex::<f64>::DTYPE,
                ScalarAsArray::Bool(_) => bool::DTYPE,
            },
        }
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

    // returns None for scalars
    fn extract_shape<'a>(asarray: &'a AsArray) -> Option<&'a [u64]> {
        match asarray {
            AsArray::Array(array) => Some(array.get().arr.shape()),
            AsArray::Numpy(array) => Some(array.shape()),
            AsArray::Scalar(_) => None,
        }
    }
    let shape = if let Some(a) = extract_shape(&a) {
        Some(a.to_vec())
    } else if let Some(b) = extract_shape(&b) {
        Some(b.to_vec())
    } else {
        None
    };

    fn extract_dtype(asarray: &AsArray) -> Option<DtypeScalarKind> {
        match asarray {
            AsArray::Array(array) => array.get().arr.dtype().try_to_scalar(),
            AsArray::Numpy(array) => array.dtype().try_to_scalar(),
            AsArray::Scalar(_) => Some(AsArray::dtype(asarray).try_to_scalar().unwrap()),
        }
    }

    let operands_scalar_dtypes = extract_dtype(&a).zip(extract_dtype(&b));
    let dtype = operands_scalar_dtypes.map(|(a_dtype, b_dtype)| promote(a_dtype, b_dtype));

    fn asarray_cast_if_scalar<'py>(
        value: &mut AsArray<'py>,
        target_dtype: DtypeScalarKind,
    ) -> Result<(), zix_core::Error> {
        enum Scalar {
            Bool(bool),
            Unsigned(u64),
            Signed(i64),
            Float(f64),
            Complex(Complex<f64>),
        }
        let AsArray::Scalar(scalar_array) = value else {
            return Ok(());
        };
        let value = match scalar_array {
            ScalarAsArray::Bool(array) => Scalar::Bool(*array.storage().data()),
            ScalarAsArray::U8(array) => Scalar::Unsigned(*array.storage().data() as u64),
            ScalarAsArray::U16(array) => Scalar::Unsigned(*array.storage().data() as u64),
            ScalarAsArray::U32(array) => Scalar::Unsigned(*array.storage().data() as u64),
            ScalarAsArray::U64(array) => Scalar::Unsigned(*array.storage().data()),
            ScalarAsArray::I8(array) => Scalar::Signed(*array.storage().data() as i64),
            ScalarAsArray::I16(array) => Scalar::Signed(*array.storage().data() as i64),
            ScalarAsArray::I32(array) => Scalar::Signed(*array.storage().data() as i64),
            ScalarAsArray::I64(array) => Scalar::Signed(*array.storage().data()),
            ScalarAsArray::F16(array) => {
                Scalar::Float(zix_core::ops::__private::cast(*array.storage().data()))
            }
            ScalarAsArray::F32(array) => Scalar::Float(*array.storage().data() as f64),
            ScalarAsArray::F64(array) => Scalar::Float(*array.storage().data()),
            ScalarAsArray::ComplexF32(array) => {
                Scalar::Complex(zix_core::ops::__private::cast(*array.storage().data()))
            }
            ScalarAsArray::ComplexF64(array) => Scalar::Complex(*array.storage().data()),
        };

        macro_rules! do_cast {
            ($scalar_array:ident, $value:ident, $ty:ty, $variant:ident) => {
                *$scalar_array = ScalarAsArray::$variant(ZixArray::from_storage(
                    ScalarStorage::new(zix_core::ops::__private::cast::<_, $ty>($value), &[])?,
                ))
            };
        }
        match (value, target_dtype) {
            (Scalar::Bool(value), DtypeScalarKind::Bool) => {
                do_cast!(scalar_array, value, bool, Bool)
            }
            (Scalar::Bool(value), DtypeScalarKind::U8) => {
                do_cast!(scalar_array, value, u8, U8)
            }
            (Scalar::Bool(value), DtypeScalarKind::U16) => {
                do_cast!(scalar_array, value, u16, U16)
            }
            (Scalar::Bool(value), DtypeScalarKind::U32) => {
                do_cast!(scalar_array, value, u32, U32)
            }
            (Scalar::Bool(value), DtypeScalarKind::U64) => {
                do_cast!(scalar_array, value, u64, U64)
            }
            (Scalar::Bool(value), DtypeScalarKind::I8) => {
                do_cast!(scalar_array, value, i8, I8)
            }
            (Scalar::Bool(value), DtypeScalarKind::I16) => {
                do_cast!(scalar_array, value, i16, I16)
            }
            (Scalar::Bool(value), DtypeScalarKind::I32) => {
                do_cast!(scalar_array, value, i32, I32)
            }
            (Scalar::Bool(value), DtypeScalarKind::I64) => {
                do_cast!(scalar_array, value, i64, I64)
            }
            (Scalar::Bool(value), DtypeScalarKind::F16) => {
                do_cast!(scalar_array, value, f16, F16)
            }
            (Scalar::Bool(value), DtypeScalarKind::F32) => {
                do_cast!(scalar_array, value, f32, F32)
            }
            (Scalar::Bool(value), DtypeScalarKind::F64) => {
                do_cast!(scalar_array, value, f64, F64)
            }
            (Scalar::Bool(value), DtypeScalarKind::ComplexF32) => {
                do_cast!(scalar_array, value, Complex<f32>, ComplexF32)
            }
            (Scalar::Bool(value), DtypeScalarKind::ComplexF64) => {
                do_cast!(scalar_array, value, Complex<f64>, ComplexF64)
            }
            (Scalar::Unsigned(_), DtypeScalarKind::Bool) => {}
            (Scalar::Unsigned(value), DtypeScalarKind::U8) => {
                do_cast!(scalar_array, value, u8, U8)
            }
            (Scalar::Unsigned(value), DtypeScalarKind::U16) => {
                do_cast!(scalar_array, value, u16, U16)
            }
            (Scalar::Unsigned(value), DtypeScalarKind::U32) => {
                do_cast!(scalar_array, value, u32, U32)
            }
            (Scalar::Unsigned(value), DtypeScalarKind::U64) => {
                do_cast!(scalar_array, value, u64, U64)
            }
            (Scalar::Unsigned(value), DtypeScalarKind::I8) => {
                do_cast!(scalar_array, value, i8, I8)
            }
            (Scalar::Unsigned(value), DtypeScalarKind::I16) => {
                do_cast!(scalar_array, value, i16, I16)
            }
            (Scalar::Unsigned(value), DtypeScalarKind::I32) => {
                do_cast!(scalar_array, value, i32, I32)
            }
            (Scalar::Unsigned(value), DtypeScalarKind::I64) => {
                do_cast!(scalar_array, value, i64, I64)
            }
            (Scalar::Unsigned(value), DtypeScalarKind::F16) => {
                do_cast!(scalar_array, value, f16, F16)
            }
            (Scalar::Unsigned(value), DtypeScalarKind::F32) => {
                do_cast!(scalar_array, value, f32, F32)
            }
            (Scalar::Unsigned(value), DtypeScalarKind::F64) => {
                do_cast!(scalar_array, value, f64, F64)
            }
            (Scalar::Unsigned(value), DtypeScalarKind::ComplexF32) => {
                do_cast!(scalar_array, value, Complex<f32>, ComplexF32)
            }
            (Scalar::Unsigned(value), DtypeScalarKind::ComplexF64) => {
                do_cast!(scalar_array, value, Complex<f64>, ComplexF64)
            }
            (Scalar::Signed(_), DtypeScalarKind::Bool) => {}
            (Scalar::Signed(_), DtypeScalarKind::U8) => {}
            (Scalar::Signed(_), DtypeScalarKind::U16) => {}
            (Scalar::Signed(_), DtypeScalarKind::U32) => {}
            (Scalar::Signed(_), DtypeScalarKind::U64) => {}
            (Scalar::Signed(value), DtypeScalarKind::I8) => {
                do_cast!(scalar_array, value, i8, I8)
            }
            (Scalar::Signed(value), DtypeScalarKind::I16) => {
                do_cast!(scalar_array, value, i16, I16)
            }
            (Scalar::Signed(value), DtypeScalarKind::I32) => {
                do_cast!(scalar_array, value, i32, I32)
            }
            (Scalar::Signed(value), DtypeScalarKind::I64) => {
                do_cast!(scalar_array, value, i64, I64)
            }
            (Scalar::Signed(value), DtypeScalarKind::F16) => {
                do_cast!(scalar_array, value, f16, F16)
            }
            (Scalar::Signed(value), DtypeScalarKind::F32) => {
                do_cast!(scalar_array, value, f32, F32)
            }
            (Scalar::Signed(value), DtypeScalarKind::F64) => {
                do_cast!(scalar_array, value, f64, F64)
            }
            (Scalar::Signed(value), DtypeScalarKind::ComplexF32) => {
                do_cast!(scalar_array, value, Complex<f32>, ComplexF32)
            }
            (Scalar::Signed(value), DtypeScalarKind::ComplexF64) => {
                do_cast!(scalar_array, value, Complex<f64>, ComplexF64)
            }
            (Scalar::Float(_), DtypeScalarKind::Bool) => {}
            (Scalar::Float(_), DtypeScalarKind::U8) => {}
            (Scalar::Float(_), DtypeScalarKind::U16) => {}
            (Scalar::Float(_), DtypeScalarKind::U32) => {}
            (Scalar::Float(_), DtypeScalarKind::U64) => {}
            (Scalar::Float(_), DtypeScalarKind::I8) => {}
            (Scalar::Float(_), DtypeScalarKind::I16) => {}
            (Scalar::Float(_), DtypeScalarKind::I32) => {}
            (Scalar::Float(_), DtypeScalarKind::I64) => {}
            (Scalar::Float(value), DtypeScalarKind::F16) => {
                do_cast!(scalar_array, value, f16, F16)
            }
            (Scalar::Float(value), DtypeScalarKind::F32) => {
                do_cast!(scalar_array, value, f32, F32)
            }
            (Scalar::Float(value), DtypeScalarKind::F64) => {
                do_cast!(scalar_array, value, f64, F64)
            }
            (Scalar::Float(value), DtypeScalarKind::ComplexF32) => {
                do_cast!(scalar_array, value, Complex<f32>, ComplexF32)
            }
            (Scalar::Float(value), DtypeScalarKind::ComplexF64) => {
                do_cast!(scalar_array, value, Complex<f64>, ComplexF64)
            }
            (Scalar::Complex(_), DtypeScalarKind::Bool) => {}
            (Scalar::Complex(_), DtypeScalarKind::I8) => {}
            (Scalar::Complex(_), DtypeScalarKind::I16) => {}
            (Scalar::Complex(_), DtypeScalarKind::I32) => {}
            (Scalar::Complex(_), DtypeScalarKind::I64) => {}
            (Scalar::Complex(_), DtypeScalarKind::U8) => {}
            (Scalar::Complex(_), DtypeScalarKind::U16) => {}
            (Scalar::Complex(_), DtypeScalarKind::U32) => {}
            (Scalar::Complex(_), DtypeScalarKind::U64) => {}
            (Scalar::Complex(_), DtypeScalarKind::F16) => {}
            (Scalar::Complex(_), DtypeScalarKind::F32) => {}
            (Scalar::Complex(_), DtypeScalarKind::F64) => {}
            (Scalar::Complex(value), DtypeScalarKind::ComplexF32) => {
                do_cast!(scalar_array, value, Complex<f32>, ComplexF32)
            }
            (Scalar::Complex(value), DtypeScalarKind::ComplexF64) => {
                do_cast!(scalar_array, value, Complex<f64>, ComplexF64)
            }
        }

        Ok(())
    }

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

    if let Some(dtype) = dtype {
        asarray_cast_if_scalar(&mut a, dtype).into_py_result()?;
        asarray_cast_if_scalar(&mut b, dtype).into_py_result()?;
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

fn promote(a: DtypeScalarKind, b: DtypeScalarKind) -> DtypeScalarKind {
    #[derive(PartialEq, Eq, PartialOrd, Ord)]
    enum Rank {
        Bool = 0,
        UnsignedInteger = 1,
        SignedInteger = 2,
        Float = 3,
        Complex = 4,
    }
    let rank = |kind: DtypeScalarKind| match kind {
        _ if kind.is_bool() => Rank::Bool,
        _ if kind.is_unsigned_integer() => Rank::UnsignedInteger,
        _ if kind.is_integer() => Rank::SignedInteger,
        _ if kind.is_float() => Rank::Float,
        _ if kind.is_complex() => Rank::Complex,
        _ => unreachable!(),
    };
    match (
        std::cmp::max(rank(a), rank(b)),
        std::cmp::max(a.alignment().as_usize(), b.alignment().as_usize()),
    ) {
        (Rank::SignedInteger, 1) => DtypeScalarKind::I8,
        (Rank::SignedInteger, 2) => DtypeScalarKind::I16,
        (Rank::SignedInteger, 4) => DtypeScalarKind::I32,
        (Rank::SignedInteger, 8) => DtypeScalarKind::I64,
        (Rank::UnsignedInteger, 1) => DtypeScalarKind::U8,
        (Rank::UnsignedInteger, 2) => DtypeScalarKind::U16,
        (Rank::UnsignedInteger, 4) => DtypeScalarKind::U32,
        (Rank::UnsignedInteger, 8) => DtypeScalarKind::U64,
        (Rank::Float, 2) => DtypeScalarKind::F16,
        (Rank::Float, 4) => DtypeScalarKind::F32,
        (Rank::Float, 8) => DtypeScalarKind::F64,
        (Rank::Complex, 4) => DtypeScalarKind::ComplexF32,
        (Rank::Complex, 8) => DtypeScalarKind::ComplexF64,
        (Rank::Bool, 1) => DtypeScalarKind::Bool,
        _ => unreachable!(),
    }
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
