use numpy::{PyArrayDescrMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::{PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyComplex, PyFloat, PyInt};

use jix_core::dtype::{Dtype, Dtyped, ScalarKind};
use jix_core::scalar::bool8;
use jix_core::scalar::{f16, Complex};
use jix_core::{Array as CoreArray, ArrayAny, ArrayParams};

use crate::dtype::dtype_from_numpy;
use crate::ops::cast_impl;
use crate::ops::common::{Precision, Rank};
use crate::util::{check_ndim, numpy_items, DimArray, IntoPyResult};
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
            let items = numpy_items(py)?;
            (items.generic.bind(py), items.asarray.bind(py))
        };

        let is_np_generic = value.is_instance(np_generic)?;
        if !is_np_generic {
            let mut scalar = None;
            if let Ok(value) = value.cast::<PyBool>() {
                scalar = Some((Scalar::Bool(value.extract::<bool>()?), Some(Precision::P1)));
            } else if let Ok(value) = value.cast::<PyInt>() {
                let Ok(int) = value.extract::<i64>() else {
                    return Err(PyOverflowError::new_err(format!(
                        "Python integer {value} out of bounds for int64"
                    )));
                };
                scalar = Some((Scalar::Int(int), None));
            } else if let Ok(value) = value.cast::<PyFloat>() {
                scalar = Some((Scalar::Float(value.extract()?), None));
            } else if let Ok(value) = value.cast::<PyComplex>() {
                let value = Complex {
                    re: value.real(),
                    im: value.imag(),
                };
                scalar = Some((Scalar::Complex(value), None));
            }
            if let Some((scalar, precision)) = scalar {
                return Ok(Self::Scalar {
                    value: scalar,
                    precision,
                    shape: DimArray::new(),
                    params,
                });
            }
        }
        if only_scalar && !is_np_generic {
            return Err(PyTypeError::new_err("expected a scalar value"));
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
                ScalarKind::Bool8 => (Scalar::Bool(item.extract::<bool>()?), Some(Precision::P1)),
            };
            return Ok(Self::Scalar {
                value,
                precision,
                shape: DimArray::new(),
                params,
            });
        }

        Ok(Self::Array(plain_from_numpy(array, dtype, params)?))
    }

    pub(crate) fn cast(self, kind: ScalarKind) -> PyResult<Self> {
        if let Self::Scalar {
            value,
            precision,
            shape,
            params,
        } = self
        {
            // a python int will be cast to the target kind, check its wont overflow
            if precision.is_none() {
                value.check_int_in_bounds(kind)?;
            }
            return Ok(Self::Scalar {
                value: value.cast(kind)?,
                precision: scalar_kind_to_rank_precision(kind).1,
                shape,
                params,
            });
        }

        if let Self::PyArray(array) = &self {
            let arr = array.get();
            if arr.arr.dtype() == &Dtype::new_scalar(kind) {
                let Self::PyArray(array) = self else {
                    unreachable!()
                };
                return Ok(Self::PyArray(array));
            }
        }

        let array = self.into_array()?;
        Ok(Self::Array(cast_impl(array, &Dtype::new_scalar(kind))?))
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
                        None | Some(Precision::P1) => {
                            create_scalar_array(bool8::from(value), &shape, params)
                        }
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
                            create_scalar_array(f16::from_f64(value), &shape, params)
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
                Err(PyTypeError::new_err("expected a scalar value"))
            }
        }
    }

    /// Raise an `OverflowError` if this is an integer value that does not fit in the integer type
    /// `kind`. Non-integer values and non-integer targets always pass.
    fn check_int_in_bounds(&self, kind: ScalarKind) -> PyResult<()> {
        let value = match *self {
            Self::Int(value) => i128::from(value),
            Self::UInt(value) => i128::from(value),
            Self::Bool(_) | Self::Float(_) | Self::Complex(_) => return Ok(()),
        };
        let in_bounds = match kind {
            ScalarKind::I8 => i8::try_from(value).is_ok(),
            ScalarKind::I16 => i16::try_from(value).is_ok(),
            ScalarKind::I32 => i32::try_from(value).is_ok(),
            ScalarKind::I64 => i64::try_from(value).is_ok(),
            ScalarKind::U8 => u8::try_from(value).is_ok(),
            ScalarKind::U16 => u16::try_from(value).is_ok(),
            ScalarKind::U32 => u32::try_from(value).is_ok(),
            ScalarKind::U64 => u64::try_from(value).is_ok(),
            ScalarKind::F16
            | ScalarKind::F32
            | ScalarKind::F64
            | ScalarKind::ComplexF32
            | ScalarKind::ComplexF64
            | ScalarKind::Bool8 => true,
        };
        if in_bounds {
            Ok(())
        } else {
            Err(PyOverflowError::new_err(format!(
                "Python integer {value} out of bounds for {}",
                Dtype::new_scalar(kind)
            )))
        }
    }

    fn cast(self, kind: ScalarKind) -> PyResult<Self> {
        use jix_core::scalar::Cast;

        fn cast_to<D>(value: impl Cast<D>) -> D {
            value.cast()
        }

        macro_rules! cast_num {
            ($value:expr) => {{
                let value = $value;
                match kind {
                    ScalarKind::Bool8 => Self::Bool(cast_to::<bool8>(value).into()),
                    ScalarKind::I8 => Self::Int(cast_to::<i8>(value) as i64),
                    ScalarKind::I16 => Self::Int(cast_to::<i16>(value) as i64),
                    ScalarKind::I32 => Self::Int(cast_to::<i32>(value) as i64),
                    ScalarKind::I64 => Self::Int(cast_to::<i64>(value)),
                    ScalarKind::U8 => Self::UInt(cast_to::<u8>(value) as u64),
                    ScalarKind::U16 => Self::UInt(cast_to::<u16>(value) as u64),
                    ScalarKind::U32 => Self::UInt(cast_to::<u32>(value) as u64),
                    ScalarKind::U64 => Self::UInt(cast_to::<u64>(value)),
                    ScalarKind::F16 => Self::Float(cast_to::<f16>(value).to_f64()),
                    ScalarKind::F32 => Self::Float(cast_to::<f32>(value) as f64),
                    ScalarKind::F64 => Self::Float(cast_to::<f64>(value)),
                    ScalarKind::ComplexF32 => {
                        Self::Complex(cast_to::<Complex<f64>>(cast_to::<Complex<f32>>(value)))
                    }
                    ScalarKind::ComplexF64 => Self::Complex(cast_to::<Complex<f64>>(value)),
                }
            }};
        }

        Ok(match self {
            Self::Bool(value) => cast_num!(bool8::from(value)),
            Self::UInt(value) => cast_num!(value),
            Self::Int(value) => cast_num!(value),
            Self::Float(value) => cast_num!(value),
            // A complex value can only be cast to another complex type or to `bool`.
            Self::Complex(value) => match kind {
                ScalarKind::Bool8 => Self::Bool(cast_to::<bool8>(value).into()),
                ScalarKind::ComplexF32 => {
                    Self::Complex(cast_to::<Complex<f64>>(cast_to::<Complex<f32>>(value)))
                }
                ScalarKind::ComplexF64 => Self::Complex(value),
                _ => {
                    return Err(PyTypeError::new_err(format!(
                        "Unsupported cast from {} to {}",
                        Dtype::new_scalar(ScalarKind::ComplexF64),
                        Dtype::new_scalar(kind)
                    )))
                }
            },
        })
    }
}

pub(crate) fn plain_from_numpy(
    array: Bound<'_, PyUntypedArray>,
    dtype: Dtype,
    params: ArrayParams,
) -> PyResult<ArrayAny> {
    let mut ndim = array.ndim();

    let dtype_shape_ndim = dtype.shape().len();
    if dtype_shape_ndim > 0 {
        ndim = array.ndim().checked_sub(dtype_shape_ndim).ok_or_else(|| {
            PyValueError::new_err("Array has fewer dimensions than the dtype inner shape")
        })?;

        // make sure the trailing dims of the array match the dtype inner shape
        let inner_shape = &array.shape()[ndim..];
        if inner_shape
            .iter()
            .copied()
            .ne(dtype.shape().iter().map(|&d| d as usize))
        {
            return Err(PyValueError::new_err(format!(
                "Array shape {:?} is not compatible with dtype inner shape {:?}",
                array.shape(),
                dtype.shape()
            )));
        }
        // make sure the trailing dims (corresponding to the dtype inner shape) of the array are C-contiguous
        if !array.is_c_contiguous() {
            let inner_strides = &array.strides()[ndim..];
            let mut expected_stride = array.dtype().itemsize();
            for (&stride, &d_len) in inner_strides.iter().rev().zip(inner_shape.iter().rev()) {
                if d_len <= 1 {
                    continue;
                }
                if usize::try_from(stride) != Ok(expected_stride) {
                    return Err(PyValueError::new_err(
                        "Array is not C-contiguous in the inner shape",
                    ));
                }
                expected_stride *= d_len;
            }
        }
    }

    check_ndim(ndim)?;

    let shape = array.shape()[..ndim]
        .iter()
        .map(|&d| d as u64)
        .collect::<DimArray<_>>();
    let strides = shape
        .iter()
        .zip(array.strides())
        .map(|(&d_len, &stride)| {
            if d_len <= 1 {
                return Ok(dtype.itemsize() as usize); // allow any stride on 0 or 1 len dim
            }
            usize::try_from(stride)
                .map_err(|_| PyValueError::new_err("Negative strides are not supported"))
        })
        .collect::<PyResult<DimArray<_>>>()?;
    let data_ptr = {
        let arr_ptr = unsafe { &*array.as_array_ptr() };
        arr_ptr.fields.data.cast_const().cast::<u8>()
    };
    let array = array.unbind();
    let storage = unsafe {
        jix_core::storage::Plain::new(array, data_ptr, shape.as_slice(), &strides, dtype, params)
    };
    Ok(CoreArray::from_storage(storage.into_py_result()?).into_any())
}

#[cfg(test)]
mod tests {
    use std::ffi::CStr;

    use jix_core::dtype::{Dtype, Dtyped, Itemsize};
    use jix_core::ArrayParams;
    use numpy::PyUntypedArray;
    use pyo3::types::{PyAnyMethods, PyDict};
    use pyo3::{Bound, Python};

    use super::plain_from_numpy;

    /// `i32` with the inner shape `shape`.
    fn i32_dtype(shape: &[Itemsize]) -> Dtype {
        let mut dtype = i32::DTYPE;
        dtype.set_shape(shape).unwrap();
        dtype
    }

    /// The NumPy array `x` set by `code`.
    fn np_array<'py>(py: Python<'py>, code: &CStr) -> Bound<'py, PyUntypedArray> {
        let locals = PyDict::new(py);
        py.run(code, None, Some(&locals)).unwrap();
        locals.get_item("x").unwrap().cast_into().unwrap()
    }

    #[test]
    fn plain_from_numpy_inner_shape() {
        Python::attach(|py| {
            let plain = |code: &CStr, shape: &[Itemsize]| {
                plain_from_numpy(np_array(py, code), i32_dtype(shape), ArrayParams::default())
            };
            let base = "import numpy as np; a = np.arange(72, dtype=np.int32).reshape(3, 4, 2, 3)";
            let code = |x: &str| std::ffi::CString::new(format!("{base}; x = {x}")).unwrap();

            let ok = plain(&code("a"), &[2, 3]).unwrap();
            assert_eq!(ok.shape(), [3, 4]);
            // outer dims in any order, inner dims C-contiguous
            let ok = plain(&code("a.transpose(1, 0, 2, 3)"), &[2, 3]).unwrap();
            assert_eq!(ok.shape(), [4, 3]);
            // inner dims not C-contiguous
            assert!(plain(&code("a.transpose(0, 1, 3, 2)"), &[3, 2]).is_err());
            assert!(plain(&code("a[..., ::2]"), &[2, 2]).is_err());
            // trailing dims do not match the inner shape
            assert!(plain(&code("a"), &[3, 2]).is_err());
            // fewer dims than the inner shape - must error, not underflow
            assert!(plain(&code("a[0, 0, 0]"), &[2, 3]).is_err());
            // a length-1 inner dim may have any stride, as NumPy allows
            let x = "np.lib.stride_tricks.as_strided(a, (4, 3, 1, 3), (12, 96, 1000, 4))";
            let ok = plain(&code(x), &[1, 3]).unwrap();
            assert_eq!(ok.shape(), [4, 3]);
        });
    }
}
