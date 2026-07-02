#![allow(clippy::assertions_on_constants)]

use jix_core::dtype::{Alignment, Dtype, Itemsize, ScalarKind, DTYPE_MAX_NDIM};
use numpy::{PyArrayDescr, PyArrayDescrMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::util::DimArray;

pub(crate) fn dtype_to_numpy<'py>(
    py: pyo3::Python<'py>,
    dtype: &Dtype,
) -> PyResult<Bound<'py, PyArrayDescr>> {
    let itemsize = dtype.itemsize();
    let shape = dtype.shape();

    let numpy_dtype = if let Some(scalar) = dtype.scalar_kind() {
        assert!(cfg!(target_endian = "little"));
        match scalar {
            ScalarKind::Bool => PyArrayDescr::of::<bool>(py),
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
        dtype_dict.set_item("names", names).unwrap();
        dtype_dict.set_item("offsets", offsets).unwrap();
        dtype_dict.set_item("formats", dtypes).unwrap();
        dtype_dict
            .set_item(
                "itemsize",
                itemsize as u64 / shape.iter().map(|&s| s as u64).product::<u64>(),
            )
            .unwrap();
        let align = dtype.is_aligned();
        if !align {
            // No need to pass 'align=False', its the default
            PyArrayDescr::new(py, dtype_dict)?
        } else {
            let dtype_kwargs = PyDict::new(py);
            dtype_kwargs.set_item("align", align).unwrap();

            // Can't pass kwargs to PyArrayDescr::new, so we use the Python API instead of numpy C API.
            py.import("numpy")?
                .getattr("dtype")?
                .call((dtype_dict,), Some(&dtype_kwargs))?
                .cast_into::<PyArrayDescr>()
                .unwrap()
        }
    };

    let shape = PyTuple::new(py, shape).unwrap();
    let numpy_dtype = PyArrayDescr::new(py, (numpy_dtype, shape))?;

    let dtype_layout = (itemsize as usize, dtype.alignment().as_usize());
    let numpy_layout = (numpy_dtype.itemsize(), numpy_dtype.alignment());
    if dtype_layout != numpy_layout {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "numpy dtype itemsize or alignment mismatch: {:?} != {:?}",
            numpy_layout, dtype_layout
        )));
    }
    Ok(numpy_dtype)
}

pub(crate) fn dtype_from_numpy(numpy_dtype: &Bound<PyArrayDescr>) -> PyResult<Dtype> {
    let shape = numpy_dtype.shape();
    if shape.len() > DTYPE_MAX_NDIM {
        return Err(PyValueError::new_err(format!(
            "Unsupported dtype: too many dimensions: {}",
            shape.len()
        )));
    }
    let shape: DimArray<Itemsize> = shape
        .iter()
        .map(|&s| s.try_into().expect("dtype shape should fit into u16"))
        .collect();
    let itemsize: Itemsize = numpy_dtype
        .itemsize()
        .try_into()
        .expect("itemsize should fit into u16");
    let alignment: Alignment = numpy_dtype
        .alignment()
        .try_into()
        .expect("alignment should fit into u16");
    let numpy_base = numpy_dtype.base();
    let base_itemsize = numpy_base.itemsize();

    let dtype = if !numpy_base.has_fields() {
        // scalar
        let scalar_kind = match (numpy_base.kind() as char, base_itemsize) {
            ('b', 1) => Ok(ScalarKind::Bool),
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
        let fields = numpy_base
            .names()
            .unwrap()
            .into_iter()
            .map::<PyResult<_>, _>(|field_name| {
                let (field_dtype, field_offset) = numpy_base.get_field(&field_name).unwrap();
                let field_offset: Itemsize = field_offset.try_into().unwrap();
                let field_dtype = dtype_from_numpy(&field_dtype)?;
                Ok((field_name, field_offset, field_dtype))
            })
            .collect::<Result<Vec<_>, _>>()?;

        Dtype::new_struct(fields, &shape, itemsize, alignment)
            .map_err(|e| PyValueError::new_err(format!("Unsupported struct dtype: {e}")))?
    };

    assert_eq!(dtype.itemsize() as usize, numpy_dtype.itemsize());
    assert_eq!(dtype.alignment().as_usize(), numpy_dtype.alignment());
    assert_eq!(
        dtype.is_aligned() && dtype.fields().is_some(),
        numpy_dtype.is_aligned_struct()
    );
    assert_eq!(dtype.shape().len(), shape.len());
    assert!(dtype.shape().iter().zip(shape.iter()).all(|(a, b)| a == b));
    Ok(dtype)
}

#[inline]
pub(crate) fn dtype_from_any(dtype: &Bound<PyAny>) -> PyResult<Dtype> {
    dtype_from_numpy(&PyArrayDescr::new(dtype.py(), dtype)?)
}

#[cfg(test)]
mod tests {
    use jix_core::dtype::{Dtype, ScalarKind};
    use numpy::PyArrayDescrMethods;
    use pyo3::Python;

    use super::*;

    fn from_str(py: Python<'_>, s: &str) -> PyResult<Dtype> {
        dtype_from_numpy(&PyArrayDescr::new(py, s)?)
    }

    fn roundtrip(py: Python<'_>, dtype: &Dtype) -> Dtype {
        let np = dtype_to_numpy(py, dtype).expect("dtype_to_numpy failed");
        dtype_from_numpy(&np).expect("dtype_from_numpy failed")
    }

    // ===== dtype_from_numpy: scalar dtypes =====

    #[test]
    fn test_from_numpy_bool() {
        Python::attach(|py| {
            let dtype = from_str(py, "bool").unwrap();
            assert_eq!(dtype, Dtype::new_scalar(ScalarKind::Bool));
        });
    }

    #[test]
    fn test_from_numpy_i8() {
        Python::attach(|py| {
            let dtype = from_str(py, "<i1").unwrap();
            assert_eq!(dtype, Dtype::new_scalar(ScalarKind::I8));
        });
    }

    #[test]
    fn test_from_numpy_i16() {
        Python::attach(|py| {
            let dtype = from_str(py, "<i2").unwrap();
            assert_eq!(dtype, Dtype::new_scalar(ScalarKind::I16));
        });
    }

    #[test]
    fn test_from_numpy_i32() {
        Python::attach(|py| {
            let dtype = from_str(py, "<i4").unwrap();
            assert_eq!(dtype, Dtype::new_scalar(ScalarKind::I32));
        });
    }

    #[test]
    fn test_from_numpy_i64() {
        Python::attach(|py| {
            let dtype = from_str(py, "<i8").unwrap();
            assert_eq!(dtype, Dtype::new_scalar(ScalarKind::I64));
        });
    }

    #[test]
    fn test_from_numpy_u8() {
        Python::attach(|py| {
            let dtype = from_str(py, "<u1").unwrap();
            assert_eq!(dtype, Dtype::new_scalar(ScalarKind::U8));
        });
    }

    #[test]
    fn test_from_numpy_u16() {
        Python::attach(|py| {
            let dtype = from_str(py, "<u2").unwrap();
            assert_eq!(dtype, Dtype::new_scalar(ScalarKind::U16));
        });
    }

    #[test]
    fn test_from_numpy_u32() {
        Python::attach(|py| {
            let dtype = from_str(py, "<u4").unwrap();
            assert_eq!(dtype, Dtype::new_scalar(ScalarKind::U32));
        });
    }

    #[test]
    fn test_from_numpy_u64() {
        Python::attach(|py| {
            let dtype = from_str(py, "<u8").unwrap();
            assert_eq!(dtype, Dtype::new_scalar(ScalarKind::U64));
        });
    }

    #[test]
    fn test_from_numpy_f16() {
        Python::attach(|py| {
            let dtype = from_str(py, "<f2").unwrap();
            assert_eq!(dtype, Dtype::new_scalar(ScalarKind::F16));
        });
    }

    #[test]
    fn test_from_numpy_f32() {
        Python::attach(|py| {
            let dtype = from_str(py, "<f4").unwrap();
            assert_eq!(dtype, Dtype::new_scalar(ScalarKind::F32));
        });
    }

    #[test]
    fn test_from_numpy_f64() {
        Python::attach(|py| {
            let dtype = from_str(py, "<f8").unwrap();
            assert_eq!(dtype, Dtype::new_scalar(ScalarKind::F64));
        });
    }

    #[test]
    fn test_from_numpy_complex_f32() {
        Python::attach(|py| {
            let dtype = from_str(py, "<c8").unwrap();
            assert_eq!(dtype, Dtype::new_scalar(ScalarKind::ComplexF32));
        });
    }

    #[test]
    fn test_from_numpy_complex_f64() {
        Python::attach(|py| {
            let dtype = from_str(py, "<c16").unwrap();
            assert_eq!(dtype, Dtype::new_scalar(ScalarKind::ComplexF64));
        });
    }

    // ===== dtype_from_numpy: scalar with shape =====

    #[test]
    fn test_from_numpy_scalar_shape_1d() {
        Python::attach(|py| {
            let np_dtype = PyArrayDescr::new(py, ("<f4", (4,))).unwrap();
            let dtype = dtype_from_numpy(&np_dtype).unwrap();
            assert_eq!(dtype.scalar_kind(), Some(ScalarKind::F32));
            assert_eq!(dtype.shape(), &[4]);
            assert_eq!(dtype.itemsize(), 16);
        });
    }

    #[test]
    fn test_from_numpy_scalar_shape_2d() {
        Python::attach(|py| {
            let np_dtype = PyArrayDescr::new(py, ("<i4", (3, 4))).unwrap();
            let dtype = dtype_from_numpy(&np_dtype).unwrap();
            assert_eq!(dtype.scalar_kind(), Some(ScalarKind::I32));
            assert_eq!(dtype.shape(), &[3, 4]);
            assert_eq!(dtype.itemsize(), 48);
        });
    }

    #[test]
    fn test_from_numpy_scalar_shape_4d() {
        Python::attach(|py| {
            let np_dtype = PyArrayDescr::new(py, ("<u1", (2, 3, 4, 5))).unwrap();
            let dtype = dtype_from_numpy(&np_dtype).unwrap();
            assert_eq!(dtype.scalar_kind(), Some(ScalarKind::U8));
            assert_eq!(dtype.shape(), &[2, 3, 4, 5]);
            assert_eq!(dtype.itemsize(), 120);
        });
    }

    // Too many dimensions (> DTYPE_MAX_NDIM = 4)
    #[test]
    fn test_from_numpy_scalar_shape_too_many_dims() {
        Python::attach(|py| {
            let np_dtype = PyArrayDescr::new(py, ("<u1", (2, 3, 4, 5, 6))).unwrap();
            let result = dtype_from_numpy(&np_dtype);
            assert!(result.is_err());
        });
    }

    // ===== dtype_from_numpy: struct dtypes =====

    #[test]
    fn test_from_numpy_struct_packed() {
        Python::attach(|py| {
            // packed: u8 at 0, u16 at 1, u32 at 3 - no padding
            let dict = PyDict::new(py);
            dict.set_item("names", vec!["a", "b", "c"]).unwrap();
            dict.set_item("formats", vec!["<u1", "<u2", "<u4"]).unwrap();
            dict.set_item("offsets", vec![0usize, 1, 3]).unwrap();
            dict.set_item("itemsize", 7usize).unwrap();
            let np_dtype = PyArrayDescr::new(py, dict).unwrap();
            let dtype = dtype_from_numpy(&np_dtype).unwrap();

            assert!(!dtype.is_aligned());
            assert_eq!(dtype.alignment().as_usize(), 1);
            assert_eq!(dtype.itemsize(), 7);
            let fields = dtype.fields().unwrap();
            assert_eq!(fields.len(), 3);
            assert_eq!(fields[0].0, "a");
            assert_eq!(fields[0].1, 0);
            assert_eq!(fields[0].2, Dtype::new_scalar(ScalarKind::U8));
            assert_eq!(fields[1].0, "b");
            assert_eq!(fields[1].1, 1);
            assert_eq!(fields[1].2, Dtype::new_scalar(ScalarKind::U16));
            assert_eq!(fields[2].0, "c");
            assert_eq!(fields[2].1, 3);
            assert_eq!(fields[2].2, Dtype::new_scalar(ScalarKind::U32));
        });
    }

    #[test]
    fn test_from_numpy_struct_aligned() {
        Python::attach(|py| {
            // aligned: u8 at 0, pad 3 bytes, f32 at 4 - alignment=4
            let np = py
                .import("numpy")
                .unwrap()
                .getattr("dtype")
                .unwrap()
                .call(
                    (vec![("x", "<u1"), ("y", "<f4")],),
                    Some(&{
                        let kw = PyDict::new(py);
                        kw.set_item("align", true).unwrap();
                        kw
                    }),
                )
                .unwrap()
                .cast_into::<PyArrayDescr>()
                .unwrap();
            let dtype = dtype_from_numpy(&np).unwrap();

            assert!(dtype.is_aligned());
            assert_eq!(dtype.alignment().as_usize(), 4);
            assert_eq!(dtype.itemsize(), 8);
            let fields = dtype.fields().unwrap();
            assert_eq!(fields.len(), 2);
            assert_eq!(fields[0].0, "x");
            assert_eq!(fields[0].1, 0);
            assert_eq!(fields[1].0, "y");
            assert_eq!(fields[1].1, 4);
        });
    }

    #[test]
    fn test_from_numpy_struct_aligned_complex() {
        Python::attach(|py| {
            // aligned struct with multiple field sizes
            let np = py
                .import("numpy")
                .unwrap()
                .getattr("dtype")
                .unwrap()
                .call(
                    (vec![("a", "<u1"), ("b", "<i2"), ("c", "<f8")],),
                    Some(&{
                        let kw = PyDict::new(py);
                        kw.set_item("align", true).unwrap();
                        kw
                    }),
                )
                .unwrap()
                .cast_into::<PyArrayDescr>()
                .unwrap();
            let dtype = dtype_from_numpy(&np).unwrap();

            assert!(dtype.is_aligned());
            assert_eq!(dtype.alignment().as_usize(), 8);
            // u8@0, pad1, i16@2, pad4, f64@8 -> itemsize=16
            assert_eq!(dtype.itemsize(), 16);
            let fields = dtype.fields().unwrap();
            assert_eq!(fields[0].1, 0);
            assert_eq!(fields[1].1, 2);
            assert_eq!(fields[2].1, 8);
        });
    }

    // ===== dtype_from_numpy: struct with shape =====

    #[test]
    fn test_from_numpy_struct_with_shape() {
        Python::attach(|py| {
            // struct dtype with shape (2,)
            let struct_dt = PyArrayDescr::new(py, vec![("x", "<f4"), ("y", "<f4")]).unwrap();
            let np_dtype = PyArrayDescr::new(py, (struct_dt, (2,))).unwrap();
            let dtype = dtype_from_numpy(&np_dtype).unwrap();
            assert_eq!(dtype.shape(), &[2]);
            assert_eq!(dtype.itemsize(), 16); // 2 * (4+4)
            let fields = dtype.fields().unwrap();
            assert_eq!(fields.len(), 2);
        });
    }

    // ===== dtype_from_numpy: unsupported types (should error) =====

    #[test]
    fn test_from_numpy_object_errors() {
        Python::attach(|py| {
            let np_dtype = PyArrayDescr::new(py, "O").unwrap();
            assert!(dtype_from_numpy(&np_dtype).is_err());
        });
    }

    #[test]
    fn test_from_numpy_unicode_errors() {
        Python::attach(|py| {
            let np_dtype = PyArrayDescr::new(py, "U10").unwrap();
            assert!(dtype_from_numpy(&np_dtype).is_err());
        });
    }

    #[test]
    fn test_from_numpy_bytes_string_errors() {
        Python::attach(|py| {
            let np_dtype = PyArrayDescr::new(py, "S4").unwrap();
            assert!(dtype_from_numpy(&np_dtype).is_err());
        });
    }

    #[test]
    fn test_from_numpy_datetime_errors() {
        Python::attach(|py| {
            let np_dtype = PyArrayDescr::new(py, "datetime64").unwrap();
            assert!(dtype_from_numpy(&np_dtype).is_err());
        });
    }

    #[test]
    fn test_from_numpy_timedelta_errors() {
        Python::attach(|py| {
            let np_dtype = PyArrayDescr::new(py, "timedelta64").unwrap();
            assert!(dtype_from_numpy(&np_dtype).is_err());
        });
    }

    // Zero-size void dtype (no fields, itemsize=0): V0
    #[test]
    fn test_from_numpy_void_zero_size_errors() {
        Python::attach(|py| {
            // V0 is a void dtype with 0 bytes - zero-sized, unsupported
            let np_dtype = PyArrayDescr::new(py, "V0").unwrap();
            // V0 has no fields, kind 'V' - should error as unsupported
            assert!(dtype_from_numpy(&np_dtype).is_err());
        });
    }

    // Shape containing a zero dimension
    #[test]
    fn test_from_numpy_zero_dim_shape_errors() {
        // numpy doesn't normally create dtypes with shape containing 0,
        // but we can try to trigger the error path via set_shape
        let mut dtype = Dtype::new_scalar(ScalarKind::F32);
        // set_shape rejects zero dimensions
        assert!(dtype.set_shape(&[0]).is_err());
        assert!(dtype.set_shape(&[3, 0, 2]).is_err());
    }

    // ===== dtype_to_numpy: scalar dtypes =====

    #[test]
    fn test_to_numpy_all_scalars() {
        Python::attach(|py| {
            let cases: &[(ScalarKind, usize, usize)] = &[
                (ScalarKind::Bool, 1, 1),
                (ScalarKind::I8, 1, 1),
                (ScalarKind::I16, 2, 2),
                (ScalarKind::I32, 4, 4),
                (ScalarKind::I64, 8, 8),
                (ScalarKind::U8, 1, 1),
                (ScalarKind::U16, 2, 2),
                (ScalarKind::U32, 4, 4),
                (ScalarKind::U64, 8, 8),
                (ScalarKind::F16, 2, 2),
                (ScalarKind::F32, 4, 4),
                (ScalarKind::F64, 8, 8),
                (ScalarKind::ComplexF32, 8, 4),
                (ScalarKind::ComplexF64, 16, 8),
            ];
            for &(scalar, expected_itemsize, expected_alignment) in cases {
                let dtype = Dtype::new_scalar(scalar);
                let np = dtype_to_numpy(py, &dtype).unwrap();
                assert_eq!(
                    np.itemsize(),
                    expected_itemsize,
                    "itemsize mismatch for {scalar:?}"
                );
                assert_eq!(
                    np.alignment(),
                    expected_alignment,
                    "alignment mismatch for {scalar:?}"
                );
                assert!(!np.has_fields());
            }
        });
    }

    // ===== dtype_to_numpy: scalar with shape =====

    #[test]
    fn test_to_numpy_scalar_with_shape_1d() {
        Python::attach(|py| {
            let mut dtype = Dtype::new_scalar(ScalarKind::F32);
            dtype.set_shape(&[4]).unwrap();
            let np = dtype_to_numpy(py, &dtype).unwrap();
            assert_eq!(np.itemsize(), 16);
            assert_eq!(np.shape(), vec![4]);
        });
    }

    #[test]
    fn test_to_numpy_scalar_with_shape_2d() {
        Python::attach(|py| {
            let mut dtype = Dtype::new_scalar(ScalarKind::I16);
            dtype.set_shape(&[3, 4]).unwrap();
            let np = dtype_to_numpy(py, &dtype).unwrap();
            assert_eq!(np.itemsize(), 24);
            assert_eq!(np.shape(), vec![3, 4]);
        });
    }

    #[test]
    fn test_to_numpy_scalar_with_shape_4d() {
        Python::attach(|py| {
            let mut dtype = Dtype::new_scalar(ScalarKind::U8);
            dtype.set_shape(&[2, 3, 4, 5]).unwrap();
            let np = dtype_to_numpy(py, &dtype).unwrap();
            assert_eq!(np.itemsize(), 120);
            assert_eq!(np.shape(), vec![2, 3, 4, 5]);
        });
    }

    // ===== dtype_to_numpy: struct dtypes =====

    #[test]
    fn test_to_numpy_struct_packed() {
        Python::attach(|py| {
            let fields = vec![
                ("a".to_string(), 0u16, Dtype::new_scalar(ScalarKind::U8)),
                ("b".to_string(), 1u16, Dtype::new_scalar(ScalarKind::U16)),
                ("c".to_string(), 3u16, Dtype::new_scalar(ScalarKind::U32)),
            ];
            let dtype = Dtype::new_struct(fields, &[], 7, 1.try_into().unwrap()).unwrap();
            let np = dtype_to_numpy(py, &dtype).unwrap();
            assert_eq!(np.itemsize(), 7);
            assert_eq!(np.alignment(), 1);
            assert!(np.has_fields());
            assert!(!np.is_aligned_struct());
            let names = np.names().unwrap();
            assert_eq!(
                names,
                vec!["a".to_string(), "b".to_string(), "c".to_string()]
            );
            let (_, offset_a) = np.get_field("a").unwrap();
            let (_, offset_b) = np.get_field("b").unwrap();
            let (_, offset_c) = np.get_field("c").unwrap();
            assert_eq!(offset_a, 0);
            assert_eq!(offset_b, 1);
            assert_eq!(offset_c, 3);
        });
    }

    #[test]
    fn test_to_numpy_struct_aligned() {
        Python::attach(|py| {
            // u8@0, pad3, f32@4 -> aligned, itemsize=8, alignment=4
            let fields = vec![
                ("x".to_string(), 0u16, Dtype::new_scalar(ScalarKind::U8)),
                ("y".to_string(), 4u16, Dtype::new_scalar(ScalarKind::F32)),
            ];
            let dtype = Dtype::new_struct(fields, &[], 8, 4.try_into().unwrap()).unwrap();
            let np = dtype_to_numpy(py, &dtype).unwrap();
            assert_eq!(np.itemsize(), 8);
            assert_eq!(np.alignment(), 4);
            assert!(np.has_fields());
            assert!(np.is_aligned_struct());
            let (_, offset_x) = np.get_field("x").unwrap();
            let (_, offset_y) = np.get_field("y").unwrap();
            assert_eq!(offset_x, 0);
            assert_eq!(offset_y, 4);
        });
    }

    #[test]
    fn test_to_numpy_struct_with_shape() {
        Python::attach(|py| {
            // packed struct (f32, f32) with shape (3,)
            let fields = vec![
                ("x".to_string(), 0u16, Dtype::new_scalar(ScalarKind::F32)),
                ("y".to_string(), 4u16, Dtype::new_scalar(ScalarKind::F32)),
            ];
            let dtype = Dtype::new_struct(fields, &[3], 24, 4.try_into().unwrap()).unwrap();
            let np = dtype_to_numpy(py, &dtype).unwrap();
            assert_eq!(np.itemsize(), 24);
            assert_eq!(np.shape(), vec![3]);
        });
    }

    // ===== Round-trip tests =====

    #[test]
    fn test_roundtrip_all_scalars() {
        Python::attach(|py| {
            for scalar in [
                ScalarKind::Bool,
                ScalarKind::I8,
                ScalarKind::I16,
                ScalarKind::I32,
                ScalarKind::I64,
                ScalarKind::U8,
                ScalarKind::U16,
                ScalarKind::U32,
                ScalarKind::U64,
                ScalarKind::F16,
                ScalarKind::F32,
                ScalarKind::F64,
                ScalarKind::ComplexF32,
                ScalarKind::ComplexF64,
            ] {
                let dtype = Dtype::new_scalar(scalar);
                assert_eq!(
                    roundtrip(py, &dtype),
                    dtype,
                    "round-trip failed for {scalar:?}"
                );
            }
        });
    }

    #[test]
    fn test_roundtrip_scalar_with_shape() {
        Python::attach(|py| {
            let mut dtype = Dtype::new_scalar(ScalarKind::F64);
            dtype.set_shape(&[2, 3]).unwrap();
            assert_eq!(roundtrip(py, &dtype), dtype);
        });
    }

    #[test]
    fn test_roundtrip_struct_packed() {
        Python::attach(|py| {
            let fields = vec![
                ("a".to_string(), 0u16, Dtype::new_scalar(ScalarKind::U8)),
                ("b".to_string(), 1u16, Dtype::new_scalar(ScalarKind::I16)),
                ("c".to_string(), 3u16, Dtype::new_scalar(ScalarKind::F32)),
            ];
            let dtype = Dtype::new_struct(fields, &[], 7, 1.try_into().unwrap()).unwrap();
            assert_eq!(roundtrip(py, &dtype), dtype);
        });
    }

    #[test]
    fn test_roundtrip_struct_aligned() {
        Python::attach(|py| {
            let fields = vec![
                ("x".to_string(), 0u16, Dtype::new_scalar(ScalarKind::U8)),
                ("y".to_string(), 4u16, Dtype::new_scalar(ScalarKind::F32)),
            ];
            let dtype = Dtype::new_struct(fields, &[], 8, 4.try_into().unwrap()).unwrap();
            assert_eq!(roundtrip(py, &dtype), dtype);
        });
    }

    #[test]
    fn test_roundtrip_struct_with_shape() {
        Python::attach(|py| {
            let fields = vec![
                ("r".to_string(), 0u16, Dtype::new_scalar(ScalarKind::F32)),
                ("g".to_string(), 4u16, Dtype::new_scalar(ScalarKind::F32)),
                ("b".to_string(), 8u16, Dtype::new_scalar(ScalarKind::F32)),
            ];
            let dtype = Dtype::new_struct(fields, &[4], 48, 4.try_into().unwrap()).unwrap();
            assert_eq!(roundtrip(py, &dtype), dtype);
        });
    }

    #[test]
    fn test_roundtrip_nested_struct() {
        Python::attach(|py| {
            let inner_fields = vec![
                ("x".to_string(), 0u16, Dtype::new_scalar(ScalarKind::F32)),
                ("y".to_string(), 4u16, Dtype::new_scalar(ScalarKind::F32)),
            ];
            let inner = Dtype::new_struct(inner_fields, &[], 8, 4.try_into().unwrap()).unwrap();
            let outer_fields = vec![
                ("pos".to_string(), 0u16, inner),
                ("w".to_string(), 8u16, Dtype::new_scalar(ScalarKind::F32)),
            ];
            let dtype = Dtype::new_struct(outer_fields, &[], 12, 4.try_into().unwrap()).unwrap();
            assert_eq!(roundtrip(py, &dtype), dtype);
        });
    }

    #[test]
    fn test_roundtrip_from_numpy_struct_packed() {
        Python::attach(|py| {
            let dict = PyDict::new(py);
            dict.set_item("names", vec!["a", "b"]).unwrap();
            dict.set_item("formats", vec!["<u1", "<u4"]).unwrap();
            dict.set_item("offsets", vec![0usize, 1]).unwrap();
            dict.set_item("itemsize", 5usize).unwrap();
            let np_dtype = PyArrayDescr::new(py, dict).unwrap();
            let dtype = dtype_from_numpy(&np_dtype).unwrap();
            assert_eq!(roundtrip(py, &dtype), dtype);
        });
    }

    #[test]
    fn test_roundtrip_from_numpy_struct_aligned() {
        Python::attach(|py| {
            let np = py
                .import("numpy")
                .unwrap()
                .getattr("dtype")
                .unwrap()
                .call(
                    (vec![("a", "<u1"), ("b", "<i8")],),
                    Some(&{
                        let kw = PyDict::new(py);
                        kw.set_item("align", true).unwrap();
                        kw
                    }),
                )
                .unwrap()
                .cast_into::<PyArrayDescr>()
                .unwrap();
            let dtype = dtype_from_numpy(&np).unwrap();
            assert_eq!(roundtrip(py, &dtype), dtype);
        });
    }
}
