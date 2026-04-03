use crate::dtype::{Complex, Dtype, DtypeScalarKind, Itemsize, f16};
use crate::iter::NdIter;
use crate::iter::strides::NdIterExtensionStridesPtrMut;

pub(crate) trait Op1 {
    fn i8(&mut self, ptr: *mut i8, idx: &[usize]);
    fn i16(&mut self, ptr: *mut i16, idx: &[usize]);
    fn i32(&mut self, ptr: *mut i32, idx: &[usize]);
    fn i64(&mut self, ptr: *mut i64, idx: &[usize]);
    fn u8(&mut self, ptr: *mut u8, idx: &[usize]);
    fn u16(&mut self, ptr: *mut u16, idx: &[usize]);
    fn u32(&mut self, ptr: *mut u32, idx: &[usize]);
    fn u64(&mut self, ptr: *mut u64, idx: &[usize]);
    fn f16(&mut self, ptr: *mut f16, idx: &[usize]);
    fn f32(&mut self, ptr: *mut f32, idx: &[usize]);
    fn f64(&mut self, ptr: *mut f64, idx: &[usize]);
    fn complex_f32(&mut self, ptr: *mut Complex<f32>, idx: &[usize]);
    fn complex_f64(&mut self, ptr: *mut Complex<f64>, idx: &[usize]);
    fn bool(&mut self, ptr: *mut bool, idx: &[usize]);
}
pub(crate) unsafe fn apply_op1(
    data_ptr: *mut u8,
    shape: &[usize],
    strides: &[usize],
    dtype: &Dtype,
    op: &mut impl Op1,
) {
    let scalar_fields = extract_inner_scalar_fields(dtype);
    for (scalar_kind, field_offset) in scalar_fields {
        let base_ptr = unsafe { data_ptr.add(field_offset as usize) };
        let mut iter = NdIter::new(shape, NdIterExtensionStridesPtrMut::new(strides, base_ptr));

        macro_rules! handle_scalar_kind {
            ($method:ident, $type:ty) => {
                while let Some((index, ptr)) = iter.next() {
                    op.$method(ptr.cast::<$type>(), &index);
                }
            };
        }

        match scalar_kind {
            DtypeScalarKind::I8 => handle_scalar_kind!(i8, i8),
            DtypeScalarKind::I16 => handle_scalar_kind!(i16, i16),
            DtypeScalarKind::I32 => handle_scalar_kind!(i32, i32),
            DtypeScalarKind::I64 => handle_scalar_kind!(i64, i64),
            DtypeScalarKind::U8 => handle_scalar_kind!(u8, u8),
            DtypeScalarKind::U16 => handle_scalar_kind!(u16, u16),
            DtypeScalarKind::U32 => handle_scalar_kind!(u32, u32),
            DtypeScalarKind::U64 => handle_scalar_kind!(u64, u64),
            DtypeScalarKind::F16 => handle_scalar_kind!(f16, f16),
            DtypeScalarKind::F32 => handle_scalar_kind!(f32, f32),
            DtypeScalarKind::F64 => handle_scalar_kind!(f64, f64),
            DtypeScalarKind::ComplexF32 => handle_scalar_kind!(complex_f32, Complex<f32>),
            DtypeScalarKind::ComplexF64 => handle_scalar_kind!(complex_f64, Complex<f64>),
            DtypeScalarKind::Bool => handle_scalar_kind!(bool, bool),
        }
    }
}

pub(crate) trait Op2 {
    fn i8(&mut self, ptr1: *mut i8, ptr2: *mut i8, idx: &[usize]);
    fn i16(&mut self, ptr1: *mut i16, ptr2: *mut i16, idx: &[usize]);
    fn i32(&mut self, ptr1: *mut i32, ptr2: *mut i32, idx: &[usize]);
    fn i64(&mut self, ptr1: *mut i64, ptr2: *mut i64, idx: &[usize]);
    fn u8(&mut self, ptr1: *mut u8, ptr2: *mut u8, idx: &[usize]);
    fn u16(&mut self, ptr1: *mut u16, ptr2: *mut u16, idx: &[usize]);
    fn u32(&mut self, ptr1: *mut u32, ptr2: *mut u32, idx: &[usize]);
    fn u64(&mut self, ptr1: *mut u64, ptr2: *mut u64, idx: &[usize]);
    fn f16(&mut self, ptr1: *mut f16, ptr2: *mut f16, idx: &[usize]);
    fn f32(&mut self, ptr1: *mut f32, ptr2: *mut f32, idx: &[usize]);
    fn f64(&mut self, ptr1: *mut f64, ptr2: *mut f64, idx: &[usize]);
    fn complex_f32(&mut self, ptr1: *mut Complex<f32>, ptr2: *mut Complex<f32>, idx: &[usize]);
    fn complex_f64(&mut self, ptr1: *mut Complex<f64>, ptr2: *mut Complex<f64>, idx: &[usize]);
    fn bool(&mut self, ptr1: *mut bool, ptr2: *mut bool, idx: &[usize]);
}

pub(crate) unsafe fn apply_op2(
    data_ptr1: *mut u8,
    data_ptr2: *mut u8,
    shape: &[usize],
    strides1: &[usize],
    strides2: &[usize],
    dtype: &Dtype,
    op: &mut impl Op2,
) {
    let scalar_fields = extract_inner_scalar_fields(dtype);
    for (scalar_kind, field_offset) in scalar_fields {
        let base_ptr1 = unsafe { data_ptr1.add(field_offset as usize) };
        let base_ptr2 = unsafe { data_ptr2.add(field_offset as usize) };
        let mut iter = NdIter::new(
            shape,
            (
                NdIterExtensionStridesPtrMut::new(strides1, base_ptr1),
                NdIterExtensionStridesPtrMut::new(strides2, base_ptr2),
            ),
        );

        macro_rules! handle_scalar_kind {
            ($method:ident, $type:ty) => {
                loop {
                    match iter.next() {
                        Some((index, (ptr1, ptr2))) => {
                            op.$method(ptr1.cast::<$type>(), ptr2.cast::<$type>(), &index);
                        }
                        None => break,
                    }
                }
            };
        }

        match scalar_kind {
            DtypeScalarKind::I8 => handle_scalar_kind!(i8, i8),
            DtypeScalarKind::I16 => handle_scalar_kind!(i16, i16),
            DtypeScalarKind::I32 => handle_scalar_kind!(i32, i32),
            DtypeScalarKind::I64 => handle_scalar_kind!(i64, i64),
            DtypeScalarKind::U8 => handle_scalar_kind!(u8, u8),
            DtypeScalarKind::U16 => handle_scalar_kind!(u16, u16),
            DtypeScalarKind::U32 => handle_scalar_kind!(u32, u32),
            DtypeScalarKind::U64 => handle_scalar_kind!(u64, u64),
            DtypeScalarKind::F16 => handle_scalar_kind!(f16, f16),
            DtypeScalarKind::F32 => handle_scalar_kind!(f32, f32),
            DtypeScalarKind::F64 => handle_scalar_kind!(f64, f64),
            DtypeScalarKind::ComplexF32 => handle_scalar_kind!(complex_f32, Complex<f32>),
            DtypeScalarKind::ComplexF64 => handle_scalar_kind!(complex_f64, Complex<f64>),
            DtypeScalarKind::Bool => handle_scalar_kind!(bool, bool),
        }
    }
}

fn extract_inner_scalar_fields(dtype: &Dtype) -> Vec<(DtypeScalarKind, Itemsize)> {
    // TODO: implement this function as iterator, to avoid the vec allocation

    let mut inner_fields = Vec::new();
    if let Some(scalar_kind) = dtype.scalar_kind() {
        inner_fields.push((scalar_kind, 0));
    } else {
        let fields = dtype.fields().unwrap();
        for (_f_name, offset, field) in fields {
            let mut subtype_scalars = extract_inner_scalar_fields(field);
            for (_, subtype_scalar_offset) in subtype_scalars.iter_mut() {
                *subtype_scalar_offset += offset;
            }
            inner_fields.extend(subtype_scalars);
        }
    }
    if !dtype.shape().is_empty() {
        let repeated = dtype.shape().iter().product::<Itemsize>();
        if repeated == 0 {
            return Vec::new();
        }
        assert_eq!(dtype.itemsize() % repeated, 0);
        let base_itemsize = dtype.itemsize() / repeated;
        assert!(
            inner_fields
                .iter()
                .all(|(_kind, offset)| (0..base_itemsize).contains(offset))
        );
        inner_fields = (0..repeated)
            .flat_map(|r| {
                let base_offset = r * base_itemsize;
                inner_fields
                    .iter()
                    .map(|(kind, offset)| (*kind, base_offset + offset))
                    .collect::<Vec<_>>()
            })
            .collect();
    }
    inner_fields
}
