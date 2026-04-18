#![allow(clippy::assertions_on_constants)]

use crate::archive::schema;
use crate::dtype::{Alignment, Itemsize};
use crate::error::{bail, check_ndim, ensure, Error, ErrorKind, Result};
use crate::util::{DimArray, IxIterExt};

impl crate::dtype::Dtype {
    pub(crate) fn from_proto(dtype: &schema::Dtype) -> Result<Self> {
        let alignment: Alignment = dtype.alignment.try_into().map_err(|_| {
            Error::new(
                ErrorKind::InvalidArchive,
                format!(
                    "dtype alignment exceeds maximum supported alignment: {}",
                    dtype.alignment
                ),
            )
        })?;
        let itemsize: Itemsize = dtype.itemsize.try_into().map_err(|_| {
            Error::new(
                ErrorKind::InvalidArchive,
                format!(
                    "dtype itemsize exceeds maximum supported itemsize: {}",
                    dtype.itemsize
                ),
            )
        })?;
        check_ndim(dtype.shape.len())?;
        let shape = dtype
            .shape
            .iter()
            .map(|&d| {
                d.try_into().map_err(|_| {
                    Error::new(
                        ErrorKind::InvalidArchive,
                        "dtype shape has too many elements (size overflow)",
                    )
                })
            })
            .collect::<Result<DimArray<_>>>()?;
        let shape_size = shape.iter().cloned().try_product().ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidArchive,
                "dtype shape has too many elements (size overflow)",
            )
        })?;

        let mut dtype = match dtype.kind.as_ref() {
            Some(schema::dtype::Kind::Scalar(scalar)) => {
                let scalar_kind = match scalar.kind() {
                    schema::DtypeScalarKind::I8 => crate::dtype::DtypeScalarKind::I8,
                    schema::DtypeScalarKind::I16 => crate::dtype::DtypeScalarKind::I16,
                    schema::DtypeScalarKind::I32 => crate::dtype::DtypeScalarKind::I32,
                    schema::DtypeScalarKind::I64 => crate::dtype::DtypeScalarKind::I64,
                    schema::DtypeScalarKind::U8 => crate::dtype::DtypeScalarKind::U8,
                    schema::DtypeScalarKind::U16 => crate::dtype::DtypeScalarKind::U16,
                    schema::DtypeScalarKind::U32 => crate::dtype::DtypeScalarKind::U32,
                    schema::DtypeScalarKind::U64 => crate::dtype::DtypeScalarKind::U64,
                    schema::DtypeScalarKind::F16 => crate::dtype::DtypeScalarKind::F16,
                    schema::DtypeScalarKind::F32 => crate::dtype::DtypeScalarKind::F32,
                    schema::DtypeScalarKind::F64 => crate::dtype::DtypeScalarKind::F64,
                    schema::DtypeScalarKind::ComplexF32 => {
                        crate::dtype::DtypeScalarKind::ComplexF32
                    }
                    schema::DtypeScalarKind::ComplexF64 => {
                        crate::dtype::DtypeScalarKind::ComplexF64
                    }
                    schema::DtypeScalarKind::Bool => crate::dtype::DtypeScalarKind::Bool,
                    schema::DtypeScalarKind::Unspecified => {
                        bail!(InvalidArchive, "unknown dtype scalar kind (unspecified)");
                    }
                };

                match scalar.endianness() {
                    schema::Endianness::Little => {}
                    schema::Endianness::Big => {
                        bail!(InvalidArchive, "big-endian dtypes are not supported");
                    }
                    schema::Endianness::Unspecified => {
                        bail!(
                            InvalidArchive,
                            "unknown dtype endianness (not little or big)"
                        );
                    }
                }

                ensure!(
                    alignment == scalar_kind.alignment(),
                    InvalidArchive,
                    "dtype alignment {alignment} does not match expected alignment {} for scalar kind {scalar_kind:?}",
                    scalar_kind.alignment()
                );
                ensure!(
                    dtype.itemsize as u64 == scalar_kind.itemsize() as u64 * shape_size as u64,
                    InvalidArchive,
                    "dtype itemsize {} does not match expected itemsize {} for scalar kind {scalar_kind:?} with shape size {shape_size}",
                    dtype.itemsize,
                    scalar_kind.itemsize() as u64 * shape_size as u64
                );

                Self::of_scalar(scalar_kind)
            }
            Some(schema::dtype::Kind::Struct(schema::DtypeStruct { fields })) => {
                let fields = fields
                    .iter()
                    .map::<Result<_>, _>(|f| {
                        let offset: Itemsize = f.offset.try_into().map_err(|_| {
                            Error::new(
                                ErrorKind::InvalidArchive,
                                "dtype struct field offset exceeds maximum supported offset",
                            )
                        })?;
                        let f_dtype = f
                            .dtype
                            .as_ref()
                            .ok_or_else(|| {
                                Error::new(
                                    ErrorKind::InvalidArchive,
                                    "dtype struct field is missing dtype",
                                )
                            })
                            .and_then(Self::from_proto)?;
                        Ok((f.name.clone(), offset, f_dtype))
                    })
                    .collect::<Result<Vec<_>>>()?;
                Self::new_struct(fields, &shape, itemsize, alignment).map_err(|e| {
                    Error::new(ErrorKind::InvalidArchive, format!("invalid dtype: {e}"))
                })?
            }
            None => {
                bail!(
                    InvalidArchive,
                    "dtype kind is missing (not a scalar or struct)"
                );
            }
        };

        dtype.set_shape(&shape).map_err(|e| {
            Error::new(
                ErrorKind::InvalidArchive,
                format!("invalid dtype shape: {e}"),
            )
        })?;
        Ok(dtype)
    }
    pub(crate) fn to_proto(&self) -> schema::Dtype {
        let kind = if let Some(scalar) = self.scalar_kind() {
            schema::dtype::Kind::Scalar(schema::DtypeScalar {
                kind: match scalar {
                    crate::dtype::DtypeScalarKind::I8 => schema::DtypeScalarKind::I8,
                    crate::dtype::DtypeScalarKind::I16 => schema::DtypeScalarKind::I16,
                    crate::dtype::DtypeScalarKind::I32 => schema::DtypeScalarKind::I32,
                    crate::dtype::DtypeScalarKind::I64 => schema::DtypeScalarKind::I64,
                    crate::dtype::DtypeScalarKind::U8 => schema::DtypeScalarKind::U8,
                    crate::dtype::DtypeScalarKind::U16 => schema::DtypeScalarKind::U16,
                    crate::dtype::DtypeScalarKind::U32 => schema::DtypeScalarKind::U32,
                    crate::dtype::DtypeScalarKind::U64 => schema::DtypeScalarKind::U64,
                    crate::dtype::DtypeScalarKind::F16 => schema::DtypeScalarKind::F16,
                    crate::dtype::DtypeScalarKind::F32 => schema::DtypeScalarKind::F32,
                    crate::dtype::DtypeScalarKind::F64 => schema::DtypeScalarKind::F64,
                    crate::dtype::DtypeScalarKind::ComplexF32 => {
                        schema::DtypeScalarKind::ComplexF32
                    }
                    crate::dtype::DtypeScalarKind::ComplexF64 => {
                        schema::DtypeScalarKind::ComplexF64
                    }
                    crate::dtype::DtypeScalarKind::Bool => schema::DtypeScalarKind::Bool,
                } as i32,
                endianness: {
                    assert!(cfg!(target_endian = "little"));
                    schema::Endianness::Little as i32
                },
            })
        } else {
            let fields = self.fields().unwrap();
            let fields = fields
                .iter()
                .map(|(name, offset, dtype)| schema::dtype_struct::Field {
                    name: name.to_string(),
                    offset: *offset as u32,
                    dtype: Some(dtype.to_proto()),
                })
                .collect::<Vec<_>>();
            schema::dtype::Kind::Struct(schema::DtypeStruct { fields })
        };

        schema::Dtype {
            shape: self.shape().iter().map(|&d| d as u64).collect(),
            itemsize: self.itemsize() as u32,
            alignment: self.alignment() as u32,
            kind: Some(kind),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::{Dtype, DtypeScalarKind, Dtyped};

    fn make_scalar_proto(
        kind: schema::DtypeScalarKind,
        itemsize: u32,
        alignment: u32,
        shape: Vec<u64>,
    ) -> schema::Dtype {
        schema::Dtype {
            shape,
            itemsize,
            alignment,
            kind: Some(schema::dtype::Kind::Scalar(schema::DtypeScalar {
                kind: kind as i32,
                endianness: schema::Endianness::Little as i32,
            })),
        }
    }

    // ---- round-trip tests ----

    #[test]
    fn roundtrip_scalar_i32() {
        let original = i32::DTYPE;
        let roundtripped = Dtype::from_proto(&original.to_proto()).unwrap();
        assert_eq!(original, roundtripped);
    }

    #[test]
    fn roundtrip_scalar_f64() {
        let original = f64::DTYPE;
        let roundtripped = Dtype::from_proto(&original.to_proto()).unwrap();
        assert_eq!(original, roundtripped);
    }

    #[test]
    fn roundtrip_scalar_with_shape() {
        let mut original = f32::DTYPE;
        original.set_shape(&[3, 2]).unwrap();
        let roundtripped = Dtype::from_proto(&original.to_proto()).unwrap();
        assert_eq!(original, roundtripped);
        assert_eq!(roundtripped.shape(), &[3, 2]);
        assert_eq!(roundtripped.itemsize(), 3 * 2 * 4);
    }

    #[test]
    fn roundtrip_struct() {
        let original = Dtype::from_fields(vec![
            ("x".to_string(), 0, u8::DTYPE),
            ("y".to_string(), 8, f64::DTYPE),
        ])
        .unwrap();
        let roundtripped = Dtype::from_proto(&original.to_proto()).unwrap();
        assert_eq!(original, roundtripped);
    }

    #[test]
    fn roundtrip_nested_struct() {
        let inner = Dtype::from_fields(vec![
            ("a".to_string(), 0, i32::DTYPE),
            ("b".to_string(), 4, i32::DTYPE),
        ])
        .unwrap();
        let outer = Dtype::from_fields(vec![
            ("inner".to_string(), 0, inner),
            ("c".to_string(), 8, u8::DTYPE),
        ])
        .unwrap();
        let roundtripped = Dtype::from_proto(&outer.to_proto()).unwrap();
        assert_eq!(outer, roundtripped);
    }

    // ---- from_proto error cases ----

    #[test]
    fn from_proto_big_endian_errors() {
        let mut proto = make_scalar_proto(schema::DtypeScalarKind::I32, 4, 4, vec![]);
        if let Some(schema::dtype::Kind::Scalar(ref mut s)) = proto.kind {
            s.endianness = schema::Endianness::Big as i32;
        }
        assert!(Dtype::from_proto(&proto).is_err());
    }

    #[test]
    fn from_proto_unspecified_endianness_errors() {
        let mut proto = make_scalar_proto(schema::DtypeScalarKind::I32, 4, 4, vec![]);
        if let Some(schema::dtype::Kind::Scalar(ref mut s)) = proto.kind {
            s.endianness = schema::Endianness::Unspecified as i32;
        }
        assert!(Dtype::from_proto(&proto).is_err());
    }

    #[test]
    fn from_proto_unspecified_scalar_kind_errors() {
        let proto = make_scalar_proto(schema::DtypeScalarKind::Unspecified, 0, 1, vec![]);
        assert!(Dtype::from_proto(&proto).is_err());
    }

    #[test]
    fn from_proto_missing_kind_errors() {
        let proto = schema::Dtype {
            shape: vec![],
            itemsize: 4,
            alignment: 4,
            kind: None,
        };
        assert!(Dtype::from_proto(&proto).is_err());
    }

    #[test]
    fn from_proto_wrong_alignment_errors() {
        // i32 expects alignment=4, give it 1
        let proto = make_scalar_proto(schema::DtypeScalarKind::I32, 4, 1, vec![]);
        assert!(Dtype::from_proto(&proto).is_err());
    }

    #[test]
    fn from_proto_wrong_itemsize_errors() {
        // i32 expects itemsize=4, give it 8
        let proto = make_scalar_proto(schema::DtypeScalarKind::I32, 8, 4, vec![]);
        assert!(Dtype::from_proto(&proto).is_err());
    }

    #[test]
    fn from_proto_too_many_shape_dims_errors() {
        let proto = make_scalar_proto(
            schema::DtypeScalarKind::U8,
            1,
            1,
            vec![1, 1, 1, 1, 1], // 5 dims > DTYPE_MAX_NDIM=4
        );
        assert!(Dtype::from_proto(&proto).is_err());
    }

    #[test]
    fn from_proto_struct_missing_field_dtype_errors() {
        let proto = schema::Dtype {
            shape: vec![],
            itemsize: 4,
            alignment: 4,
            kind: Some(schema::dtype::Kind::Struct(schema::DtypeStruct {
                fields: vec![schema::dtype_struct::Field {
                    name: "x".to_string(),
                    offset: 0,
                    dtype: None, // missing
                }],
            })),
        };
        assert!(Dtype::from_proto(&proto).is_err());
    }

    // ---- to_proto spot checks ----

    #[test]
    fn to_proto_scalar_kind_and_endianness() {
        let proto = i32::DTYPE.to_proto();
        match proto.kind.unwrap() {
            schema::dtype::Kind::Scalar(s) => {
                assert_eq!(s.kind(), schema::DtypeScalarKind::I32);
                assert_eq!(s.endianness(), schema::Endianness::Little);
            }
            _ => panic!("expected scalar"),
        }
        assert_eq!(proto.itemsize, 4);
        assert_eq!(proto.alignment, 4);
        assert_eq!(proto.shape, vec![]);
    }

    #[test]
    fn to_proto_shape_is_serialized() {
        let mut d = u8::DTYPE;
        d.set_shape(&[5, 3]).unwrap();
        let proto = d.to_proto();
        assert_eq!(proto.shape, vec![5, 3]);
        assert_eq!(proto.itemsize, 15);
    }

    #[test]
    fn to_proto_struct_fields_are_serialized() {
        let dtype = Dtype::from_fields(vec![
            ("a".to_string(), 0, u8::DTYPE),
            ("b".to_string(), 1, u8::DTYPE),
        ])
        .unwrap();
        let proto = dtype.to_proto();
        match proto.kind.unwrap() {
            schema::dtype::Kind::Struct(s) => {
                assert_eq!(s.fields.len(), 2);
                assert_eq!(s.fields[0].name, "a");
                assert_eq!(s.fields[0].offset, 0);
                assert_eq!(s.fields[1].name, "b");
                assert_eq!(s.fields[1].offset, 1);
            }
            _ => panic!("expected struct"),
        }
    }

    #[test]
    fn all_scalar_kinds_roundtrip() {
        let cases: &[DtypeScalarKind] = &[
            DtypeScalarKind::I8,
            DtypeScalarKind::I16,
            DtypeScalarKind::I32,
            DtypeScalarKind::I64,
            DtypeScalarKind::U8,
            DtypeScalarKind::U16,
            DtypeScalarKind::U32,
            DtypeScalarKind::U64,
            DtypeScalarKind::F16,
            DtypeScalarKind::F32,
            DtypeScalarKind::F64,
            DtypeScalarKind::ComplexF32,
            DtypeScalarKind::ComplexF64,
            DtypeScalarKind::Bool,
        ];
        for &kind in cases {
            let original = Dtype::of_scalar(kind);
            let roundtripped = Dtype::from_proto(&original.to_proto()).unwrap();
            assert_eq!(original, roundtripped, "{kind:?} failed roundtrip");
        }
    }
}
