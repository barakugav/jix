use numpy::{PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::PyOverflowError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyComplex, PyFloat, PyInt};

use zix_core::dtype::{DtypeScalarKind, Dtyped};
use zix_core::scalar::{f16, Complex};
use zix_core::{Array as ZixArray, ArrayAny};

use crate::dtype::dtype_from_numpy;
use crate::ops::common::{Precision, Rank};
use crate::util::{check_ndim, DimArray, IntoPyResult};
use crate::Array;

pub(crate) enum Operand {
    Array(ArrayAny),
    Scalar {
        value: Scalar,
        shape: DimArray<u64>,
        precision: Option<Precision>,
    },
}
impl Operand {
    pub(crate) fn from_any(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(array) = value.cast::<Array>() {
            return Ok(Self::Array(array.get().arr.clone()));
        };

        let py = value.py();
        let np = numpy::get_array_module(py)?;
        if !value.is_instance(&np.getattr("generic")?)? {
            let mut scalar = None;
            if let Ok(value) = value.cast::<PyBool>() {
                scalar = Some(Scalar::Bool(value.extract()?));
            } else if let Ok(value) = value.cast::<PyInt>() {
                scalar = Some(Scalar::Int(value.extract()?));
            } else if let Ok(value) = value.cast::<PyFloat>() {
                scalar = Some(Scalar::Float(value.extract()?));
            } else if let Ok(value) = value.cast::<PyComplex>() {
                scalar = Some(Scalar::Complex(Complex {
                    re: value.real(),
                    im: value.imag(),
                }));
            }
            if let Some(scalar) = scalar {
                return Ok(Self::Scalar {
                    value: scalar,
                    precision: None,
                    shape: DimArray::new(),
                });
            }
        }

        let array = np
            .getattr("asarray")?
            .call1((value,))?
            .cast_into::<PyUntypedArray>()?;
        let dtype = dtype_from_numpy(&array.dtype())?;
        if array.ndim() == 0
            && let Some(scalar) = dtype.try_to_scalar()
        {
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
                shape: DimArray::new(),
            });
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

        Ok(Self::Array(ZixArray::from_storage(storage).into_any()))
    }

    pub(crate) fn into_array(self) -> PyResult<ArrayAny> {
        match self {
            Operand::Array(array) => Ok(array),
            Operand::Scalar {
                value,
                precision,
                shape,
            } => {
                fn create_scalar_array<T>(value: T, shape: &[u64]) -> PyResult<ArrayAny>
                where
                    T: Dtyped,
                {
                    let array = zix_core::Array::from_storage(
                        zix_core::storage::Scalar::new(value, shape).into_py_result()?,
                    );
                    Ok(array.to_type_dyn().into_any())
                }
                #[allow(clippy::unnecessary_cast)]
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
                array
            }
        }
    }

    pub(crate) fn rank_precision(&self) -> Option<(Rank, Option<Precision>)> {
        match self {
            Operand::Array(arr) => arr
                .dtype()
                .try_to_scalar()
                .map(scalar_kind_to_rank_precision),
            Operand::Scalar {
                value,
                precision,
                shape: _,
            } => {
                let kind = match value {
                    Scalar::Bool(_) => Rank::Bool,
                    Scalar::UInt(_) => Rank::UInt,
                    Scalar::Int(_) => Rank::Int,
                    Scalar::Float(_) => Rank::Float,
                    Scalar::Complex(_) => Rank::Complex,
                };
                Some((kind, *precision))
            }
        }
    }
}

pub(crate) fn scalar_kind_to_rank_precision(kind: DtypeScalarKind) -> (Rank, Option<Precision>) {
    let (rank, precision) = match kind {
        _ if kind.is_bool() => (Rank::Bool, 1),
        _ if kind.is_unsigned_integer() => (Rank::UInt, kind.itemsize()),
        _ if kind.is_integer() => (Rank::Int, kind.itemsize()),
        _ if kind.is_float() => (Rank::Float, kind.itemsize()),
        _ if kind.is_complex() => (Rank::Complex, kind.itemsize() / 2),
        _ => unreachable!(),
    };
    (rank, Some(Precision::from_itemsize(precision)))
}

#[derive(Debug)]
pub(crate) enum Scalar {
    Bool(bool),
    UInt(u64),
    Int(i64),
    Float(f64),
    Complex(Complex<f64>),
}
