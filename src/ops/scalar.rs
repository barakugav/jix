use std::io;

use crate::array::{Array, BlocksLayout};
use crate::dtype::{Complex, Dtype, DtypeScalarKind, f16};
use crate::storage::ArrayStorage;
use crate::util::{DimArray, cast_slice, cast_slice_mut};

pub(crate) trait ScalarOp2Kernel {
    fn apply_i8(&self, a: i8, b: i8) -> i8;
    fn apply_i16(&self, a: i16, b: i16) -> i16;
    fn apply_i32(&self, a: i32, b: i32) -> i32;
    fn apply_i64(&self, a: i64, b: i64) -> i64;
    fn apply_u8(&self, a: u8, b: u8) -> u8;
    fn apply_u16(&self, a: u16, b: u16) -> u16;
    fn apply_u32(&self, a: u32, b: u32) -> u32;
    fn apply_u64(&self, a: u64, b: u64) -> u64;
    fn apply_f16(&self, a: f16, b: f16) -> f16;
    fn apply_f32(&self, a: f32, b: f32) -> f32;
    fn apply_f64(&self, a: f64, b: f64) -> f64;
    fn apply_complex_f32(&self, a: Complex<f32>, b: Complex<f32>) -> Complex<f32>;
    fn apply_complex_f64(&self, a: Complex<f64>, b: Complex<f64>) -> Complex<f64>;
    fn apply_bool(&self, a: bool, b: bool) -> bool;

    fn is_support_dtype(&self, dtype: &Dtype) -> bool;
}

pub(crate) struct ScalarOp2<Op, S1, S2> {
    op: Op,
    a: Array<S1>,
    b: Array<S2>,
    dtype: Dtype,
    shape: DimArray<usize>,
    blocks_layout: BlocksLayout,
}
impl<Op, S1, S2> ScalarOp2<Op, S1, S2> {
    pub(crate) fn new(op: Op, a: Array<S1>, b: Array<S2>) -> io::Result<Self>
    where
        Op: ScalarOp2Kernel,
        S1: ArrayStorage,
        S2: ArrayStorage,
    {
        if a.dtype() != b.dtype() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "dtype mismatch",
            ));
        }
        if a.shape() != b.shape() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "shape mismatch",
            ));
        }
        if !op.is_support_dtype(a.dtype()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unsupported dtype for operation: {:#?}", a.dtype()),
            ));
        }
        Ok(Self {
            op,
            dtype: a.dtype().clone(),
            shape: a.shape().try_into().unwrap(),
            blocks_layout: a.blocks_layout().clone(),
            a,
            b,
        })
    }
}
impl<Op, S1, S2> ArrayStorage for ScalarOp2<Op, S1, S2>
where
    Op: ScalarOp2Kernel,
    S1: ArrayStorage,
    S2: ArrayStorage,
{
    fn dtype(&self) -> &Dtype {
        &self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn blocks_layout(&self) -> &BlocksLayout {
        &self.blocks_layout
    }

    fn read_data(
        &self,
        index: &[std::ops::Range<usize>],
        buf: &mut [u8],
        context: &crate::codec::ReadContext,
    ) -> std::io::Result<()> {
        let mut buf2 = vec![0u8; buf.len()];
        self.a.storage.read_data(index, buf, context)?;
        self.b.storage.read_data(index, &mut buf2, context)?;

        macro_rules! apply_loop {
            ($ty:ty, $apply:ident) => {
                let buf1 = unsafe { cast_slice_mut::<u8, $ty>(buf) };
                let buf2 = unsafe { cast_slice::<u8, $ty>(&buf2) };
                for (a, b) in buf1.iter_mut().zip(buf2) {
                    *a = self.op.$apply(*a, *b);
                }
            };
        }

        Ok(match self.dtype.try_to_scalar() {
            Some(DtypeScalarKind::I8) => {
                apply_loop!(i8, apply_i8);
            }
            Some(DtypeScalarKind::I16) => {
                apply_loop!(i16, apply_i16);
            }
            Some(DtypeScalarKind::I32) => {
                apply_loop!(i32, apply_i32);
            }
            Some(DtypeScalarKind::I64) => {
                apply_loop!(i64, apply_i64);
            }
            Some(DtypeScalarKind::U8) => {
                apply_loop!(u8, apply_u8);
            }
            Some(DtypeScalarKind::U16) => {
                apply_loop!(u16, apply_u16);
            }
            Some(DtypeScalarKind::U32) => {
                apply_loop!(u32, apply_u32);
            }
            Some(DtypeScalarKind::U64) => {
                apply_loop!(u64, apply_u64);
            }
            Some(DtypeScalarKind::F16) => {
                apply_loop!(f16, apply_f16);
            }
            Some(DtypeScalarKind::F32) => {
                apply_loop!(f32, apply_f32);
            }
            Some(DtypeScalarKind::F64) => {
                apply_loop!(f64, apply_f64);
            }
            Some(DtypeScalarKind::ComplexF32) => {
                apply_loop!(Complex<f32>, apply_complex_f32);
            }
            Some(DtypeScalarKind::ComplexF64) => {
                apply_loop!(Complex<f64>, apply_complex_f64);
            }
            Some(DtypeScalarKind::Bool) => {
                apply_loop!(bool, apply_bool);
            }
            None => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Unsupported,
                    "only scalar dtypes are supported for ScalarOp2",
                ));
            }
        })
    }
}
