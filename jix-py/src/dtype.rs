#![allow(clippy::assertions_on_constants)]

use jix_core::dtype::{Alignment, Dtype, Itemsize, ScalarKind, DTYPE_NDIM_MAX};
use numpy::{PyArrayDescr, PyArrayDescrMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::util::DimArray;

pub(crate) fn dtype_to_numpy<'py>(
    py: Python<'py>,
    dtype: &Dtype,
) -> PyResult<Bound<'py, PyArrayDescr>> {
    let itemsize = dtype.itemsize();
    let shape = dtype.shape();

    let numpy_dtype = if let Some(scalar) = dtype.scalar_kind() {
        assert!(cfg!(target_endian = "little"));
        match scalar {
            ScalarKind::Bool8 => PyArrayDescr::of::<bool>(py),
            ScalarKind::I8 => PyArrayDescr::of::<i8>(py),
            ScalarKind::I16 => PyArrayDescr::of::<i16>(py),
            ScalarKind::I32 => PyArrayDescr::of::<i32>(py),
            ScalarKind::I64 => PyArrayDescr::of::<i64>(py),
            ScalarKind::U8 => PyArrayDescr::of::<u8>(py),
            ScalarKind::U16 => PyArrayDescr::of::<u16>(py),
            ScalarKind::U32 => PyArrayDescr::of::<u32>(py),
            ScalarKind::U64 => PyArrayDescr::of::<u64>(py),
            ScalarKind::F16 => PyArrayDescr::of::<jix_core::scalar::f16>(py),
            ScalarKind::F32 => PyArrayDescr::of::<f32>(py),
            ScalarKind::F64 => PyArrayDescr::of::<f64>(py),
            ScalarKind::ComplexF32 => PyArrayDescr::of::<jix_core::scalar::Complex<f32>>(py),
            ScalarKind::ComplexF64 => PyArrayDescr::of::<jix_core::scalar::Complex<f64>>(py),
        }
    } else {
        let fields = dtype.fields().unwrap();

        let mut names = Vec::with_capacity(fields.len());
        let mut offsets = Vec::with_capacity(fields.len());
        let mut dtypes = Vec::with_capacity(fields.len());
        for (name, offset, sub_dtype) in fields {
            names.push(name.clone());
            offsets.push(*offset as usize);
            dtypes.push(dtype_to_numpy(py, sub_dtype)?);
        }

        let dtype_dict = PyDict::new(py);
        dtype_dict.set_item("names", names)?;
        dtype_dict.set_item("offsets", offsets)?;
        dtype_dict.set_item("formats", dtypes)?;
        dtype_dict.set_item(
            "itemsize",
            itemsize as u64 / shape.iter().map(|&s| s as u64).product::<u64>(),
        )?;
        let align = dtype.is_aligned();
        if !align {
            // No need to pass 'align=False', its the default
            PyArrayDescr::new(py, dtype_dict)?
        } else {
            let dtype_kwargs = PyDict::new(py);
            dtype_kwargs.set_item("align", align)?;

            // Can't pass kwargs to PyArrayDescr::new, so we use the Python API instead of numpy C API.
            py.import("numpy")?
                .getattr("dtype")?
                .call((dtype_dict,), Some(&dtype_kwargs))?
                .cast_into::<PyArrayDescr>()?
        }
    };

    let shape = PyTuple::new(py, shape)?;
    let numpy_dtype = PyArrayDescr::new(py, (numpy_dtype, shape))?;

    let dtype_layout = (itemsize as usize, dtype.alignment().as_usize());
    let numpy_layout = (numpy_dtype.itemsize(), numpy_dtype.alignment());
    if dtype_layout != numpy_layout {
        return Err(PyValueError::new_err(format!(
            "numpy dtype itemsize or alignment mismatch: {:?} != {:?}",
            numpy_layout, dtype_layout
        )));
    }
    Ok(numpy_dtype)
}

pub(crate) fn numpy_descr_from_any<'py>(
    py: Python<'py>,
    dtype: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArrayDescr>> {
    if dtype.is_none() {
        return Err(PyTypeError::new_err(
            "dtype must be a dtype-like object, not None",
        ));
    }
    PyArrayDescr::new(py, dtype)
}

pub(crate) fn dtype_from_numpy(numpy_dtype: &Bound<PyArrayDescr>) -> PyResult<Dtype> {
    dtype_from_numpy_nested(numpy_dtype, 0)
}

const DTYPE_NESTING_MAX: usize = 32;

fn dtype_from_numpy_nested(numpy_dtype: &Bound<PyArrayDescr>, depth: usize) -> PyResult<Dtype> {
    let shape = numpy_dtype.shape();
    if shape.len() > DTYPE_NDIM_MAX {
        return Err(PyValueError::new_err(format!(
            "Unsupported dtype: too many dimensions: {}",
            shape.len()
        )));
    }
    let shape = shape
        .iter()
        .map(|&s| {
            Itemsize::try_from(s).map_err(|_| {
                PyValueError::new_err(format!(
                    "Unsupported dtype: sub-array dimension {s} exceeds the maximum supported {}",
                    Itemsize::MAX
                ))
            })
        })
        .collect::<PyResult<DimArray<_>>>()?;
    let itemsize = Itemsize::try_from(numpy_dtype.itemsize()).map_err(|_| {
        PyValueError::new_err(format!(
            "Unsupported dtype: itemsize {} exceeds the maximum supported {}",
            numpy_dtype.itemsize(),
            Itemsize::MAX
        ))
    })?;
    let alignment = Alignment::new(numpy_dtype.alignment()).ok_or_else(|| {
        PyValueError::new_err(format!(
            "Unsupported dtype: invalid alignment {}",
            numpy_dtype.alignment()
        ))
    })?;
    let numpy_base = numpy_dtype.base();
    let base_itemsize = numpy_base.itemsize();

    let dtype = if !numpy_base.has_fields() {
        // scalar
        if numpy_base.byteorder() == b'>' {
            return Err(PyValueError::new_err(format!(
                "Unsupported dtype: big-endian {numpy_base:?}; convert it to native byte order \
                 first, e.g. with `x.astype(x.dtype.newbyteorder('='))`"
            )));
        }
        let scalar_kind = match (numpy_base.kind() as char, base_itemsize) {
            ('b', 1) => Ok(ScalarKind::Bool8),
            ('b', itemsize) => Err(PyValueError::new_err(format!(
                "Unsupported bool itemsize: {itemsize}"
            ))),
            ('i', 1) => Ok(ScalarKind::I8),
            ('i', 2) => Ok(ScalarKind::I16),
            ('i', 4) => Ok(ScalarKind::I32),
            ('i', 8) => Ok(ScalarKind::I64),
            ('i', itemsize) => Err(PyValueError::new_err(format!(
                "Unsupported signed integer itemsize: {itemsize}"
            ))),
            ('u', 1) => Ok(ScalarKind::U8),
            ('u', 2) => Ok(ScalarKind::U16),
            ('u', 4) => Ok(ScalarKind::U32),
            ('u', 8) => Ok(ScalarKind::U64),
            ('u', itemsize) => Err(PyValueError::new_err(format!(
                "Unsupported unsigned integer itemsize: {itemsize}"
            ))),
            ('f', 2) => Ok(ScalarKind::F16),
            ('f', 4) => Ok(ScalarKind::F32),
            ('f', 8) => Ok(ScalarKind::F64),
            ('f', itemsize) => Err(PyValueError::new_err(format!(
                "Unsupported float itemsize: {itemsize}"
            ))),
            ('c', 8) => Ok(ScalarKind::ComplexF32),
            ('c', 16) => Ok(ScalarKind::ComplexF64),
            ('c', itemsize) => Err(PyValueError::new_err(format!(
                "Unsupported complex itemsize: {itemsize}"
            ))),
            ('m' | 'M' | 'O' | 'S' | 'U' | 'V', _) => Err(PyValueError::new_err(format!(
                "Unsupported dtype: {numpy_base:?}"
            ))),
            (_, _) => {
                return Err(PyValueError::new_err(format!(
                    "Unsupported dtype: {numpy_base:?}"
                )));
            }
        }?;
        let mut dtype = Dtype::new_scalar(scalar_kind);
        dtype
            .set_shape(&shape)
            .map_err(|e| PyValueError::new_err(format!("Unsupported dtype shape: {e}")))?;
        dtype
    } else {
        // struct
        if depth >= DTYPE_NESTING_MAX {
            return Err(PyValueError::new_err(format!(
                "Unsupported struct dtype: nested more than {DTYPE_NESTING_MAX} levels"
            )));
        }
        let names = numpy_base.names().ok_or_else(|| {
            PyValueError::new_err(format!("Unsupported struct dtype: {numpy_base:?}"))
        })?;
        let mut fields = names
            .into_iter()
            .map::<PyResult<_>, _>(|field_name| {
                let (field_dtype, field_offset) = numpy_base.get_field(&field_name)?;
                let field_offset = Itemsize::try_from(field_offset).map_err(|_| {
                    PyValueError::new_err(format!(
                        "Unsupported struct dtype: field '{field_name}' offset {field_offset} exceeds the maximum supported {}",
                        Itemsize::MAX
                    ))
                })?;
                let field_dtype = dtype_from_numpy_nested(&field_dtype, depth + 1)?;
                Ok((field_name, field_offset, field_dtype))
            })
            .collect::<Result<Vec<_>, _>>()?;
        fields.sort_by_key(|(_, offset, _)| *offset);

        Dtype::new_struct(fields, &shape, itemsize, alignment)
            .map_err(|e| PyValueError::new_err(format!("Unsupported struct dtype: {e}")))?
    };

    debug_assert_eq!(dtype.itemsize() as usize, numpy_dtype.itemsize());
    debug_assert_eq!(dtype.alignment().as_usize(), numpy_dtype.alignment());
    debug_assert!(!numpy_dtype.is_aligned_struct() || dtype.is_aligned());
    debug_assert_eq!(dtype.shape().len(), shape.len());
    debug_assert!(dtype.shape().iter().zip(shape.iter()).all(|(a, b)| a == b));
    Ok(dtype)
}
