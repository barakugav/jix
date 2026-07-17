use numpy::{PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::{PyOverflowError, PyTypeError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBool, PyComplex, PyFloat, PyInt};

use jix_core::dtype::{Dtyped, ScalarKind};
use jix_core::scalar::{f16, Complex};
use jix_core::{Array as CoreArray, ArrayAny, ArrayParams};

use crate::dtype::dtype_from_numpy;
use crate::ops::common::{Precision, Rank};
use crate::util::{check_ndim, DimArray, IntoPyResult};
use crate::Array;

pub(crate) enum Operand {
    PyArray(Py<Array>),
    Array(ArrayAny),
    Scalar {
        value: Scalar,
        shape: DimArray<u64>,
        precision: Option<Precision>,
        params: ArrayParams,
    },
}
impl Operand {
    #[inline]
    pub(crate) fn from_any(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        Self::from_any_with_params(value, ArrayParams::default(), false)
    }

    pub(crate) fn from_any_with_params(
        value: &Bound<'_, PyAny>,
        params: ArrayParams,
        only_scalar: bool,
    ) -> PyResult<Self> {
        if !only_scalar && let Ok(array) = value.cast::<Array>() {
            return Ok(Self::PyArray(array.clone().unbind()));
        };
        let py = value.py();

        let (np_generic, np_asarray) = {
            struct NumpyItems {
                generic: Py<PyAny>,
                asarray: Py<PyAny>,
            }
            static NUMPY_ITEMS: PyOnceLock<NumpyItems> = PyOnceLock::new();
            let np_items = NUMPY_ITEMS.get_or_try_init::<_, PyErr>(py, || {
                let np = numpy::get_array_module(py)?;
                Ok(NumpyItems {
                    generic: np.getattr("generic")?.unbind(),
                    asarray: np.getattr("asarray")?.unbind(),
                })
            })?;
            (np_items.generic.bind(py), np_items.asarray.bind(py))
        };

        let is_np_generic = value.is_instance(np_generic)?;
        if !is_np_generic {
            let mut scalar = None;
            if let Ok(value) = value.cast::<PyBool>() {
                scalar = Some(Scalar::Bool(value.extract()?));
            } else if let Ok(value) = value.cast::<PyInt>() {
                if let Ok(value) = value.extract::<i64>() {
                    scalar = Some(Scalar::Int(value));
                } else if let Ok(value) = value.extract::<u64>() {
                    scalar = Some(Scalar::UInt(value));
                } else {
                    return Err(PyErr::new::<PyOverflowError, _>(
                        "Integer value is too large to fit in 64 bits",
                    ));
                }
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
                    params,
                });
            }
        }
        if only_scalar && !is_np_generic {
            return Err(PyErr::new::<PyTypeError, _>("expected a scalar value"));
        }

        let array = if let Ok(array) = value.cast::<PyUntypedArray>() {
            array.clone() // already a NumPy array
        } else {
            np_asarray.call1((value,))?.cast_into::<PyUntypedArray>()?
        };
        let dtype = dtype_from_numpy(&array.dtype())?;
        if array.ndim() == 0
            && let Some(scalar) = dtype.try_to_scalar()
        {
            let item = array.call_method0("item")?;

            let (value, precision) = match scalar {
                ScalarKind::I8 => (
                    Scalar::Int(item.extract::<i8>()? as i64),
                    Some(Precision::P1),
                ),
                ScalarKind::I16 => (
                    Scalar::Int(item.extract::<i16>()? as i64),
                    Some(Precision::P2),
                ),
                ScalarKind::I32 => (
                    Scalar::Int(item.extract::<i32>()? as i64),
                    Some(Precision::P4),
                ),
                ScalarKind::I64 => (Scalar::Int(item.extract::<i64>()?), Some(Precision::P8)),
                ScalarKind::U8 => (
                    Scalar::UInt(item.extract::<u8>()? as u64),
                    Some(Precision::P1),
                ),
                ScalarKind::U16 => (
                    Scalar::UInt(item.extract::<u16>()? as u64),
                    Some(Precision::P2),
                ),
                ScalarKind::U32 => (
                    Scalar::UInt(item.extract::<u32>()? as u64),
                    Some(Precision::P4),
                ),
                ScalarKind::U64 => (Scalar::UInt(item.extract::<u64>()?), Some(Precision::P8)),
                ScalarKind::F16 => (
                    Scalar::Float(item.extract::<f32>()? as f64),
                    Some(Precision::P2),
                ),
                ScalarKind::F32 => (
                    Scalar::Float(item.extract::<f32>()? as f64),
                    Some(Precision::P4),
                ),
                ScalarKind::F64 => (Scalar::Float(item.extract::<f64>()?), Some(Precision::P8)),
                ScalarKind::ComplexF32 => {
                    let re = item.getattr("real")?.extract::<f32>()?;
                    let im = item.getattr("imag")?.extract::<f32>()?;
                    (
                        Scalar::Complex(Complex::new(re as f64, im as f64)),
                        Some(Precision::P4),
                    )
                }
                ScalarKind::ComplexF64 => {
                    let re = item.getattr("real")?.extract::<f64>()?;
                    let im = item.getattr("imag")?.extract::<f64>()?;
                    (Scalar::Complex(Complex::new(re, im)), Some(Precision::P8))
                }
                ScalarKind::Bool => (Scalar::Bool(item.extract::<bool>()?), None),
            };
            return Ok(Self::Scalar {
                value,
                precision,
                shape: DimArray::new(),
                params,
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
        let array = array.unbind();
        let storage = unsafe {
            jix_core::storage::Plain::new(
                array,
                data_ptr,
                shape.as_slice(),
                &strides,
                dtype,
                params,
            )
        };
        let storage = storage.into_py_result()?;

        Ok(Self::Array(CoreArray::from_storage(storage).into_any()))
    }

    pub(crate) fn into_array(self) -> PyResult<ArrayAny> {
        match self {
            Operand::PyArray(array) => Ok(array.get().arr.clone()),
            Operand::Array(array) => Ok(array),
            Operand::Scalar {
                value,
                precision,
                shape,
                params,
            } => {
                fn create_scalar_array<T>(
                    value: T,
                    shape: &[u64],
                    params: ArrayParams,
                ) -> PyResult<ArrayAny>
                where
                    T: Dtyped,
                {
                    let array = jix_core::Array::from_storage(
                        jix_core::__private::Scalar::new(value, shape, params).into_py_result()?,
                    );
                    Ok(array.into_any())
                }
                #[allow(clippy::unnecessary_cast)]
                let array = match value {
                    Scalar::Bool(value) => match precision {
                        None | Some(Precision::P1) => create_scalar_array(value, &shape, params),
                        Some(_) => unreachable!(),
                    },
                    Scalar::UInt(value) => match precision {
                        None | Some(Precision::P8) => {
                            create_scalar_array(value as u64, &shape, params)
                        }
                        Some(Precision::P4) => create_scalar_array(value as u32, &shape, params),
                        Some(Precision::P2) => create_scalar_array(value as u16, &shape, params),
                        Some(Precision::P1) => create_scalar_array(value as u8, &shape, params),
                    },
                    Scalar::Int(value) => match precision {
                        None | Some(Precision::P8) => {
                            create_scalar_array(value as i64, &shape, params)
                        }
                        Some(Precision::P4) => create_scalar_array(value as i32, &shape, params),
                        Some(Precision::P2) => create_scalar_array(value as i16, &shape, params),
                        Some(Precision::P1) => create_scalar_array(value as i8, &shape, params),
                    },
                    Scalar::Float(value) => match precision {
                        None | Some(Precision::P8) => {
                            create_scalar_array(value as f64, &shape, params)
                        }
                        Some(Precision::P4) => create_scalar_array(value as f32, &shape, params),
                        Some(Precision::P2) => {
                            create_scalar_array(f16::from_f32(value as f32), &shape, params)
                        }
                        Some(_) => unreachable!(),
                    },
                    Scalar::Complex(value) => match precision {
                        None | Some(Precision::P8) => create_scalar_array(value, &shape, params),
                        Some(Precision::P4) => create_scalar_array::<Complex<f32>>(
                            <_ as jix_core::scalar::Cast<_>>::cast(value),
                            &shape,
                            params,
                        ),
                        Some(_) => unreachable!(),
                    },
                };
                array
            }
        }
    }

    pub(crate) fn into_py_array<'py>(self, py: Python<'py>) -> PyResult<Bound<'py, Array>> {
        match self {
            Operand::PyArray(array) => Ok(array.into_bound(py)),
            _ => Bound::new(py, Array::from_core(self.into_array()?)),
        }
    }

    pub(crate) fn rank_precision(&self) -> Option<(Rank, Option<Precision>)> {
        match self {
            Operand::PyArray(arr) => arr
                .get()
                .arr
                .dtype()
                .try_to_scalar()
                .map(scalar_kind_to_rank_precision),
            Operand::Array(arr) => arr
                .dtype()
                .try_to_scalar()
                .map(scalar_kind_to_rank_precision),
            Operand::Scalar {
                value,
                precision,
                shape: _,
                params: _,
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

pub(crate) fn scalar_kind_to_rank_precision(kind: ScalarKind) -> (Rank, Option<Precision>) {
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
impl Scalar {
    #[inline]
    pub(crate) fn from_any(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        match Operand::from_any_with_params(value, ArrayParams::default(), true)? {
            Operand::Scalar { value, .. } => Ok(value),
            Operand::PyArray(_) | Operand::Array(_) => {
                Err(PyErr::new::<PyTypeError, _>("expected a scalar value"))
            }
        }
    }
}
