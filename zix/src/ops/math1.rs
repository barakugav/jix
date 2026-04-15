use std::io;
use std::ops::Range;

use crate::array::Array;
use crate::codec::{DecoderCodecConfig, DecoderParams, EncoderParams, ReadContext};
use crate::dtype::{Complex, Dtype, DtypeScalarKind, f16};
use crate::ops::common::define_array_op1_method;
use crate::storage::{ArrayStorage, BlocksLayout};
use crate::util::{DimArray, cast_slice_mut};

#[allow(unused_variables)]
pub(crate) trait MathOp1Kernel {
    fn apply_i8(&self, a: i8) -> i8 {
        unimplemented!()
    }
    fn apply_i16(&self, a: i16) -> i16 {
        unimplemented!()
    }
    fn apply_i32(&self, a: i32) -> i32 {
        unimplemented!()
    }
    fn apply_i64(&self, a: i64) -> i64 {
        unimplemented!()
    }
    fn apply_u8(&self, a: u8) -> u8 {
        unimplemented!()
    }
    fn apply_u16(&self, a: u16) -> u16 {
        unimplemented!()
    }
    fn apply_u32(&self, a: u32) -> u32 {
        unimplemented!()
    }
    fn apply_u64(&self, a: u64) -> u64 {
        unimplemented!()
    }
    fn apply_f16(&self, a: f16) -> f16 {
        unimplemented!()
    }
    fn apply_f32(&self, a: f32) -> f32 {
        unimplemented!()
    }
    fn apply_f64(&self, a: f64) -> f64 {
        unimplemented!()
    }
    fn apply_complex_f32(&self, a: Complex<f32>) -> Complex<f32> {
        unimplemented!()
    }
    fn apply_complex_f64(&self, a: Complex<f64>) -> Complex<f64> {
        unimplemented!()
    }
    fn apply_bool(&self, a: bool) -> bool {
        unimplemented!()
    }

    fn is_support_dtype(&self, dtype: &Dtype) -> bool;
}

pub(crate) struct MathOp1<Op, S> {
    op: Op,

    a: Array<S>,

    dtype: Dtype,
    shape: DimArray<u64>,
    blocks_layout: BlocksLayout,
}
impl<Op, S> MathOp1<Op, S> {
    pub(crate) fn new(op: Op, a: Array<S>) -> io::Result<Self>
    where
        Op: MathOp1Kernel,
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
            dtype: a.dtype().clone(),
            shape: a.shape().try_into().unwrap(),
            blocks_layout: a.blocks_layout().clone(),
            a,
        })
    }
}
impl<Op, S> ArrayStorage for MathOp1<Op, S>
where
    Op: MathOp1Kernel,
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
        self.a.storage.read_data(index, buf, context)?;

        macro_rules! apply_loop {
            ($ty:ty, $apply_fn:ident) => {{
                let buf = unsafe { cast_slice_mut::<u8, $ty>(buf) };
                for a in buf.iter_mut() {
                    *a = self.op.$apply_fn(*a);
                }
            }};
        }

        match self.dtype.try_to_scalar() {
            Some(DtypeScalarKind::I8) => apply_loop!(i8, apply_i8),
            Some(DtypeScalarKind::I16) => apply_loop!(i16, apply_i16),
            Some(DtypeScalarKind::I32) => apply_loop!(i32, apply_i32),
            Some(DtypeScalarKind::I64) => apply_loop!(i64, apply_i64),
            Some(DtypeScalarKind::U8) => apply_loop!(u8, apply_u8),
            Some(DtypeScalarKind::U16) => apply_loop!(u16, apply_u16),
            Some(DtypeScalarKind::U32) => apply_loop!(u32, apply_u32),
            Some(DtypeScalarKind::U64) => apply_loop!(u64, apply_u64),
            Some(DtypeScalarKind::F16) => apply_loop!(f16, apply_f16),
            Some(DtypeScalarKind::F32) => apply_loop!(f32, apply_f32),
            Some(DtypeScalarKind::F64) => apply_loop!(f64, apply_f64),
            Some(DtypeScalarKind::ComplexF32) => apply_loop!(Complex<f32>, apply_complex_f32),
            Some(DtypeScalarKind::ComplexF64) => apply_loop!(Complex<f64>, apply_complex_f64),
            Some(DtypeScalarKind::Bool) => apply_loop!(bool, apply_bool),
            None => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Unsupported,
                    "only scalar dtypes are supported for MathOp1",
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

macro_rules! define_math1_op {
    ($Name:ident, $NameKernel:ident, |$arg:ident| $body:expr, [$($scalar:tt),* $(,)?]) => {
        pub struct $Name<S>(crate::ops::math1::MathOp1<$NameKernel, S>);
        impl<S> $Name<S> {
            pub fn new(a: crate::Array<S>) -> std::io::Result<Self>
            where
                S: crate::storage::ArrayStorage,
            {
                Ok(Self(crate::ops::math1::MathOp1::new($NameKernel, a)?))
            }
        }
        crate::storage::impl_array_storage_forward!($Name<S> where S: crate::storage::ArrayStorage);

        define_math1_op_kernel!($NameKernel, |$arg| $body, [$($scalar),*]);
    };
}
macro_rules! define_math1_core_op {
    ($Name:ident, $NameKernel:ident, $op_trait:ident, $op_fn:ident, |$arg:ident| $body:expr, [$($scalar:tt),* $(,)?]) => {
        define_math1_op!($Name, $NameKernel, |$arg| $body, [$($scalar),*]);
        impl<S> core::ops::$op_trait for crate::Array<S>
        where
            S: crate::storage::ArrayStorage,
        {
            type Output = crate::Array<$Name<S>>;
            #[track_caller]
            fn $op_fn(self) -> crate::Array<$Name<S>> {
                let op = $Name::new(self).unwrap();
                crate::Array::from_storage(op)
            }
        }
    };
}
macro_rules! define_math1_op_kernel {
    ($NameKernel:ident, |$arg:ident| $body:expr, [$($scalar:tt),* $(,)?]) => {
        struct $NameKernel;
        impl crate::ops::math1::MathOp1Kernel for $NameKernel {
            $(define_math1_op_kernel!(@apply |$arg| $body, $scalar);)*

            fn is_support_dtype(&self, dtype: &crate::dtype::Dtype) -> bool {
                use crate::dtype::DtypeScalarKind;
                let Some(scalar_kind) = dtype.try_to_scalar() else {
                    return false;
                };
                false $(|| define_math1_op_kernel!(@dtype_match scalar_kind, $scalar))*
            }
        }
    };

    // --- apply arms ---
    (@apply |$arg:ident| $body:expr, i8)  => { fn apply_i8(&self, $arg: i8) -> i8 { $body } };
    (@apply |$arg:ident| $body:expr, i16) => { fn apply_i16(&self, $arg: i16) -> i16 { $body } };
    (@apply |$arg:ident| $body:expr, i32) => { fn apply_i32(&self, $arg: i32) -> i32 { $body } };
    (@apply |$arg:ident| $body:expr, i64) => { fn apply_i64(&self, $arg: i64) -> i64 { $body } };
    (@apply |$arg:ident| $body:expr, u8)  => { fn apply_u8(&self, $arg: u8) -> u8 { $body } };
    (@apply |$arg:ident| $body:expr, u16) => { fn apply_u16(&self, $arg: u16) -> u16 { $body } };
    (@apply |$arg:ident| $body:expr, u32) => { fn apply_u32(&self, $arg: u32) -> u32 { $body } };
    (@apply |$arg:ident| $body:expr, u64) => { fn apply_u64(&self, $arg: u64) -> u64 { $body } };
    (@apply |$arg:ident| $body:expr, f32) => { fn apply_f32(&self, $arg: f32) -> f32 { $body } };
    (@apply |$arg:ident| $body:expr, f64) => { fn apply_f64(&self, $arg: f64) -> f64 { $body } };
    (@apply |$arg:ident| $body:expr, f16) => {
        fn apply_f16(&self, #[allow(unused_variables)] $arg: f16) -> f16 {
            cfg_if::cfg_if! { if #[cfg(feature = "half")] {
                $body
            } else {
                unimplemented!()
            } }
        }
    };
    (@apply |$arg:ident| $body:expr, (Complex<f32>)) => {
        fn apply_complex_f32(&self, #[allow(unused_variables)] $arg: Complex<f32>) -> Complex<f32> {
            cfg_if::cfg_if! { if #[cfg(feature = "num-complex")] {
                $body
            } else {
                unimplemented!()
            } }
        }
    };
    (@apply |$arg:ident| $body:expr, (Complex<f64>)) => {
        fn apply_complex_f64(&self, #[allow(unused_variables)] $arg: Complex<f64>) -> Complex<f64> {
            cfg_if::cfg_if! { if #[cfg(feature = "num-complex")] {
                $body
            } else {
                unimplemented!()
            } }
        }
    };
    (@apply |$arg:ident| $body:expr, bool) => { fn apply_bool(&self, $arg: bool) -> bool { $body } };

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
pub(crate) use {define_math1_op, define_math1_op_kernel};

define_math1_core_op!(Neg, NegKernel, Neg, neg, |a| -a, [i8, i16, i32, i64, f16, f32, f64, (Complex<f32>), (Complex<f64>)]);
define_math1_op!(Floor, FloorKernel, |a| a.floor(), [f32, f64]);
define_math1_op!(Ceil, CeilKernel, |a| a.ceil(), [f32, f64]);
define_math1_op!(Round, RoundKernel, |a| a.round(), [f32, f64]);
define_math1_op!(Sqrt, SqrtKernel, |a| a.sqrt(), [f32, f64]);
define_math1_op!(Exp, ExpKernel, |a| a.exp(), [f32, f64]);
define_math1_op!(Log, LogKernel, |a| a.ln(), [f32, f64]);
define_math1_op!(Sin, SinKernel, |a| a.sin(), [f32, f64]);
define_math1_op!(Cos, CosKernel, |a| a.cos(), [f32, f64]);
define_math1_op!(Tan, TanKernel, |a| a.tan(), [f32, f64]);
define_math1_op!(Asin, AsinKernel, |a| a.asin(), [f32, f64]);
define_math1_op!(Acos, AcosKernel, |a| a.acos(), [f32, f64]);
define_math1_op!(Atan, AtanKernel, |a| a.atan(), [f32, f64]);
define_math1_op!(Signum, SignumKernel, |a| a.signum(), [f16, f32, f64]);

impl<S> Array<S>
where
    S: ArrayStorage,
{
    define_array_op1_method!(floor: Floor);
    define_array_op1_method!(ceil: Ceil);
    define_array_op1_method!(round: Round);
    define_array_op1_method!(sqrt: Sqrt);
    define_array_op1_method!(exp: Exp);
    define_array_op1_method!(ln: Log);
    define_array_op1_method!(sin: Sin);
    define_array_op1_method!(cos: Cos);
    define_array_op1_method!(tan: Tan);
    define_array_op1_method!(asin: Asin);
    define_array_op1_method!(acos: Acos);
    define_array_op1_method!(atan: Atan);
    define_array_op1_method!(signum: Signum);
}
