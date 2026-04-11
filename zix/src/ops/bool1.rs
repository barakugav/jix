use std::io;
use std::ops::Range;

use crate::array::Array;
use crate::codec::{DecoderCodecConfig, DecoderParams, EncoderParams, ReadContext};
use crate::dtype::{Complex, Dtype, DtypeScalarKind, Dtyped, f16};
use crate::storage::{ArrayStorage, BlocksLayout};
use crate::util::{DimArray, cast_slice, cast_slice_mut};

#[allow(unused_variables)]
pub(crate) trait BoolOp1Kernel {
    fn apply_i8(&self, a: i8) -> bool {
        unimplemented!()
    }
    fn apply_i16(&self, a: i16) -> bool {
        unimplemented!()
    }
    fn apply_i32(&self, a: i32) -> bool {
        unimplemented!()
    }
    fn apply_i64(&self, a: i64) -> bool {
        unimplemented!()
    }
    fn apply_u8(&self, a: u8) -> bool {
        unimplemented!()
    }
    fn apply_u16(&self, a: u16) -> bool {
        unimplemented!()
    }
    fn apply_u32(&self, a: u32) -> bool {
        unimplemented!()
    }
    fn apply_u64(&self, a: u64) -> bool {
        unimplemented!()
    }
    fn apply_f16(&self, a: f16) -> bool {
        unimplemented!()
    }
    fn apply_f32(&self, a: f32) -> bool {
        unimplemented!()
    }
    fn apply_f64(&self, a: f64) -> bool {
        unimplemented!()
    }
    fn apply_complex_f32(&self, a: Complex<f32>) -> bool {
        unimplemented!()
    }
    fn apply_complex_f64(&self, a: Complex<f64>) -> bool {
        unimplemented!()
    }
    fn apply_bool(&self, a: bool) -> bool {
        unimplemented!()
    }

    fn is_support_dtype(&self, dtype: &Dtype) -> bool;
}

pub(crate) struct BoolOp1<Op, S> {
    op: Op,

    a: Array<S>,

    dtype: Dtype,
    shape: DimArray<u64>,
    blocks_layout: BlocksLayout,
}
impl<Op, S> BoolOp1<Op, S> {
    pub(crate) fn new(op: Op, a: Array<S>) -> io::Result<Self>
    where
        Op: BoolOp1Kernel,
        S: ArrayStorage,
    {
        if !op.is_support_dtype(a.dtype()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unsupported dtype for operation: {:#?}", a.dtype()),
            ));
        }
        Ok(Self {
            op,
            dtype: bool::dtype(),
            shape: a.shape().try_into().unwrap(),
            blocks_layout: a.blocks_layout().clone(),
            a,
        })
    }
}
impl<Op, S> ArrayStorage for BoolOp1<Op, S>
where
    Op: BoolOp1Kernel,
    S: ArrayStorage,
{
    fn shape(&self) -> &[u64] {
        &self.shape
    }

    fn dtype(&self) -> &Dtype {
        &self.dtype
    }

    fn read_data(
        &self,
        index: &[Range<u64>],
        buf: &mut [u8],
        context: &ReadContext,
    ) -> std::io::Result<()> {
        let inner_dtype = self.a.dtype();
        let inner_buf_size = buf.len() * inner_dtype.itemsize() as usize;
        let mut tmp_buf = context.tmp_buf(inner_buf_size, inner_dtype.alignment());
        let tmp_buf = tmp_buf.as_mut_slice();
        self.a.storage.read_data(index, tmp_buf, context)?;

        macro_rules! apply_loop {
            ($ty:ty, $apply:ident) => {
                let src = unsafe { cast_slice::<u8, $ty>(tmp_buf) };
                let dst = unsafe { cast_slice_mut::<u8, bool>(buf) };
                for (a, b) in dst.iter_mut().zip(src) {
                    *a = self.op.$apply(*b);
                }
            };
        }

        match inner_dtype.try_to_scalar() {
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
                    "only scalar dtypes are supported for BoolOp1",
                ));
            }
        }
        Ok(())
    }

    fn blocks_layout(&self) -> &BlocksLayout {
        &self.blocks_layout
    }
    fn codec_params(&self) -> (&EncoderParams, &DecoderParams, &DecoderCodecConfig) {
        self.a.storage.codec_params()
    }
}

macro_rules! define_op {
    ($Name:ident, $NameKernel:ident, $op:ident, [$($scalar:tt),* $(,)?]) => {
        pub struct $Name<S>(BoolOp1<$NameKernel, S>);
        impl<S> $Name<S> {
            pub(crate) fn new(a: Array<S>) -> io::Result<Self>
            where
                S: ArrayStorage,
            {
                Ok(Self(BoolOp1::new($NameKernel, a)?))
            }
        }
        crate::storage::impl_array_storage_forward!($Name<S> where S: ArrayStorage);

        define_op_kernel!($NameKernel, $op, [$($scalar),*]);
    };
}
macro_rules! define_op_kernel {
    ($NameKernel:ident, $op:ident, [$($scalar:tt),* $(,)?]) => {
        struct $NameKernel;
        impl BoolOp1Kernel for $NameKernel {
            $(define_op_kernel!(@apply $op, $scalar);)*

            fn is_support_dtype(&self, dtype: &crate::dtype::Dtype) -> bool {
                use crate::dtype::DtypeScalarKind;
                let Some(scalar_kind) = dtype.try_to_scalar() else {
                    return false;
                };
                false $(|| define_op_kernel!(@dtype_match scalar_kind, $scalar))*
            }
        }
    };

    // --- apply arms ---
    (@apply $op:tt, i8)  => { fn apply_i8(&self, a: i8) -> bool { a.$op() } };
    (@apply $op:tt, i16) => { fn apply_i16(&self, a: i16) -> bool { a.$op() } };
    (@apply $op:tt, i32) => { fn apply_i32(&self, a: i32) -> bool { a.$op() } };
    (@apply $op:tt, i64) => { fn apply_i64(&self, a: i64) -> bool { a.$op() } };
    (@apply $op:tt, u8)  => { fn apply_u8(&self, a: u8) -> bool { a.$op() } };
    (@apply $op:tt, u16) => { fn apply_u16(&self, a: u16) -> bool { a.$op() } };
    (@apply $op:tt, u32) => { fn apply_u32(&self, a: u32) -> bool { a.$op() } };
    (@apply $op:tt, u64) => { fn apply_u64(&self, a: u64) -> bool { a.$op() } };
    (@apply $op:tt, f32) => { fn apply_f32(&self, a: f32) -> bool { a.$op() } };
    (@apply $op:tt, f64) => { fn apply_f64(&self, a: f64) -> bool { a.$op() } };
    (@apply $op:tt, f16) => {
        fn apply_f16(&self, #[allow(unused_variables)] a: f16) -> bool {
            cfg_if::cfg_if! { if #[cfg(feature = "half")] {
                a.$op()
            } else {
                unimplemented!()
            } }
        }
    };
    (@apply $op:tt, (Complex<f32>)) => {
        fn apply_complex_f32(&self, #[allow(unused_variables)] a: Complex<f32>) -> bool {
            cfg_if::cfg_if! { if #[cfg(feature = "num-complex")] {
                a.$op()
            } else {
                unimplemented!()
            } }
        }
    };
    (@apply $op:tt, (Complex<f64>)) => {
        fn apply_complex_f64(&self, #[allow(unused_variables)] a: Complex<f64>) -> bool {
            cfg_if::cfg_if! { if #[cfg(feature = "num-complex")] {
                a.$op()
            } else {
                unimplemented!()
            } }
        }
    };
    (@apply $op:tt, bool) => { fn apply_bool(&self, a: bool) -> bool { a.$op() } };

    // --- dtype match arms ---
    (@dtype_match $sk:ident, i8)  => { $sk == DtypeScalarKind::I8 };
    (@dtype_match $sk:ident, i16) => { $sk == DtypeScalarKind::I16 };
    (@dtype_match $sk:ident, i32) => { $sk == DtypeScalarKind::I32 };
    (@dtype_match $sk:ident, i64) => { $sk == DtypeScalarKind::I64 };
    (@dtype_match $sk:ident, u8)  => { $sk == DtypeScalarKind::U8 };
    (@dtype_match $sk:ident, u16) => { $sk == DtypeScalarKind::U16 };
    (@dtype_match $sk:ident, u32) => { $sk == DtypeScalarKind::U32 };
    (@dtype_match $sk:ident, u64) => { $sk == DtypeScalarKind::U64 };
    (@dtype_match $sk:ident, f32) => { $sk == DtypeScalarKind::F32 };
    (@dtype_match $sk:ident, f64) => { $sk == DtypeScalarKind::F64 };
    (@dtype_match $sk:ident, f16) => {
        (cfg!(feature = "half") && $sk == DtypeScalarKind::F16)
    };
    (@dtype_match $sk:ident, (Complex<f32>)) => {
        (cfg!(feature = "num-complex") && $sk == DtypeScalarKind::ComplexF32)
    };
    (@dtype_match $sk:ident, (Complex<f64>)) => {
        (cfg!(feature = "num-complex") && $sk == DtypeScalarKind::ComplexF64)
    };
    (@dtype_match $sk:ident, bool) => { $sk == DtypeScalarKind::Bool };
}

define_op!(IsNan, IsNanKernel, is_nan, [f16, f32, f64]);
define_op!(IsFinite, IsFiniteKernel, is_finite, [f16, f32, f64]);
define_op!(IsInfinite, IsInfiniteKernel, is_infinite, [f16, f32, f64]);

macro_rules! define_array_op_methods {
    ($($op:ident : $Name:ident),+ $(,)?) => {
        impl<S> Array<S>
        where
            S: ArrayStorage,
        {
            $(
                #[track_caller]
                pub fn $op(self) -> Array<$Name<S>> {
                    let op = $Name::new(self).unwrap();
                    Array::from_storage(op)
                }
            )*
        }
    };
}
define_array_op_methods!(
    is_nan: IsNan,
    is_finite: IsFinite,
    is_infinite: IsInfinite,
);
