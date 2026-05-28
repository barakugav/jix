use numpy::{PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::PyOverflowError;
use pyo3::prelude::*;
use pyo3::types::{PyComplex, PyFloat, PyInt};

use zix_core::dtype::{DtypeScalarKind, Dtyped, Itemsize};
use zix_core::scalar::{f16, Complex};
use zix_core::storage::{Plain, TypeDyn};
use zix_core::{Array as ZixArray, DimDyn};

use crate::dtype::dtype_from_numpy;
use crate::util::{check_ndim, DimArray, IntoPyResult};
use crate::Array;

pub(crate) enum Operand {
    Zix(Py<Array>),
    Numpy(ZixArray<Plain<Py<PyUntypedArray>, TypeDyn, DimDyn>>),
    Scalar {
        value: Scalar,
        shape: Vec<u64>,
        precision: Option<Precision>,
    },
}
pub(crate) enum Scalar {
    Bool(bool),
    UInt(u64),
    Int(i64),
    Float(f64),
    Complex(Complex<f64>),
}
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Precision {
    P1 = 0,
    P2 = 1,
    P4 = 2,
    P8 = 3,
}

impl Precision {
    pub(crate) fn from_itemsize(itemsize: Itemsize) -> Self {
        match itemsize {
            1 => Self::P1,
            2 => Self::P2,
            4 => Self::P4,
            8 => Self::P8,
            _ => unreachable!(),
        }
    }

    pub(crate) fn higher(self) -> Option<Self> {
        match self {
            Self::P1 => Some(Self::P2),
            Self::P2 => Some(Self::P4),
            Self::P4 => Some(Self::P8),
            Self::P8 => None,
        }
    }
}

impl Operand {
    pub(crate) fn from_any(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(array) = value.cast::<Array>() {
            return Ok(Self::Zix(array.clone().unbind()));
        };

        let py = value.py();
        let np = numpy::get_array_module(py)?;
        if !value.is_instance(&np.getattr("generic")?)? {
            if let Ok(value) = value.cast::<PyInt>() {
                return Ok(Self::Scalar {
                    value: Scalar::Int(value.extract()?),
                    precision: None,
                    shape: Vec::new(),
                });
            }
            if let Ok(value) = value.cast::<PyFloat>() {
                return Ok(Self::Scalar {
                    value: Scalar::Float(value.extract()?),
                    precision: None,
                    shape: Vec::new(),
                });
            }
            if let Ok(value) = value.cast::<PyComplex>() {
                return Ok(Self::Scalar {
                    value: Scalar::Complex(Complex {
                        re: value.real() as f64,
                        im: value.imag() as f64,
                    }),
                    precision: None,
                    shape: Vec::new(),
                });
            }
        }

        let array = np
            .getattr("asarray")?
            .call1((value,))?
            .cast_into::<PyUntypedArray>()?;
        let dtype = dtype_from_numpy(&array.dtype())?;
        if array.ndim() == 0 {
            if let Some(scalar) = dtype.try_to_scalar() {
                let item = array.call_method0("item")?;

                let (value, precision) = match scalar {
                    DtypeScalarKind::I8 => (
                        Scalar::Int(item.extract::<i8>()? as i64),
                        Some(Precision::P1),
                    ),
                    DtypeScalarKind::I16 => (
                        Scalar::Int(item.extract::<i16>()? as i64),
                        Some(Precision::P2),
                    ),
                    DtypeScalarKind::I32 => (
                        Scalar::Int(item.extract::<i32>()? as i64),
                        Some(Precision::P4),
                    ),
                    DtypeScalarKind::I64 => (
                        Scalar::Int(item.extract::<i64>()? as i64),
                        Some(Precision::P4),
                    ),
                    DtypeScalarKind::U8 => (
                        Scalar::UInt(item.extract::<u8>()? as u64),
                        Some(Precision::P1),
                    ),
                    DtypeScalarKind::U16 => (
                        Scalar::UInt(item.extract::<u16>()? as u64),
                        Some(Precision::P2),
                    ),
                    DtypeScalarKind::U32 => (
                        Scalar::UInt(item.extract::<u32>()? as u64),
                        Some(Precision::P4),
                    ),
                    DtypeScalarKind::U64 => (
                        Scalar::UInt(item.extract::<u64>()? as u64),
                        Some(Precision::P8),
                    ),
                    DtypeScalarKind::F16 => (
                        Scalar::Float(item.extract::<f32>()? as f64),
                        Some(Precision::P2),
                    ),
                    DtypeScalarKind::F32 => (
                        Scalar::Float(item.extract::<f32>()? as f64),
                        Some(Precision::P4),
                    ),
                    DtypeScalarKind::F64 => (
                        Scalar::Float(item.extract::<f64>()? as f64),
                        Some(Precision::P8),
                    ),
                    DtypeScalarKind::ComplexF32 => {
                        let re = item.getattr("real")?.extract::<f32>()?;
                        let im = item.getattr("imag")?.extract::<f32>()?;
                        (
                            Scalar::Complex(Complex::new(re as f64, im as f64)),
                            Some(Precision::P4),
                        )
                    }
                    DtypeScalarKind::ComplexF64 => {
                        let re = item.getattr("real")?.extract::<f64>()?;
                        let im = item.getattr("imag")?.extract::<f64>()?;
                        (Scalar::Complex(Complex::new(re, im)), Some(Precision::P8))
                    }
                    DtypeScalarKind::Bool => (Scalar::Bool(item.extract::<bool>()?), None),
                };
                return Ok(Self::Scalar {
                    value,
                    precision,
                    shape: Vec::new(),
                });
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
        let storage = unsafe {
            zix_core::storage::Plain::new(array, data_ptr, shape.as_slice(), &strides, dtype)
        };
        let storage = storage.into_py_result()?;

        Ok(Self::Numpy(ZixArray::from_storage(storage)))
    }

    pub(crate) fn into_py_array<'py>(self, py: Python<'py>) -> PyResult<Bound<'py, Array>> {
        match self {
            Operand::Zix(array) => Ok(array.into_bound(py)),
            Operand::Numpy(array) => Bound::new(py, Array::from_core_storage(array.into_storage())),
            Operand::Scalar {
                value,
                precision,
                shape,
            } => {
                fn create_scalar_array<T>(value: T, shape: &[u64]) -> PyResult<Array>
                where
                    T: Dtyped,
                {
                    let scalar_storage =
                        zix_core::storage::Scalar::new(value, shape).into_py_result()?;
                    Ok(Array::from_core_storage(scalar_storage))
                }
                let array = match value {
                    Scalar::Bool(value) => match precision {
                        None | Some(Precision::P1) => create_scalar_array(value, &shape),
                        Some(_) => unreachable!(),
                    },
                    Scalar::UInt(value) => match precision {
                        None | Some(Precision::P8) => create_scalar_array(value as u64, &shape),
                        Some(Precision::P4) => create_scalar_array(value as u32, &shape),
                        Some(Precision::P2) => create_scalar_array(value as u16, &shape),
                        Some(Precision::P1) => create_scalar_array(value as u8, &shape),
                    },
                    Scalar::Int(value) => match precision {
                        None | Some(Precision::P8) => create_scalar_array(value as i64, &shape),
                        Some(Precision::P4) => create_scalar_array(value as i32, &shape),
                        Some(Precision::P2) => create_scalar_array(value as i16, &shape),
                        Some(Precision::P1) => create_scalar_array(value as i8, &shape),
                    },
                    Scalar::Float(value) => match precision {
                        None | Some(Precision::P8) => create_scalar_array(value as f64, &shape),
                        Some(Precision::P4) => create_scalar_array(value as f32, &shape),
                        Some(Precision::P2) => {
                            create_scalar_array(f16::from_f32(value as f32), &shape)
                        }
                        Some(_) => unreachable!(),
                    },
                    Scalar::Complex(value) => match precision {
                        None | Some(Precision::P8) => create_scalar_array(value, &shape),
                        Some(Precision::P4) => create_scalar_array::<Complex<f32>>(
                            <_ as zix_core::scalar::Cast<_>>::cast(value),
                            &shape,
                        ),
                        Some(_) => unreachable!(),
                    },
                };
                Bound::new(py, array?)
            }
        }
    }
}
