use std::hint::assert_unchecked;
use std::io;
use std::ops::Range;

use crate::array::Array;
use crate::codec::{DecoderCodecConfig, DecoderParams, EncoderParams, ReadContext};
use crate::dtype::{Complex, Dtype, DtypeScalarKind, Dtyped, f16};
use crate::ops::common::define_array_op2_method;
use crate::storage::{ArrayStorage, BlocksLayout};
use crate::util::{DimArray, cast_slice, cast_slice_mut};

#[allow(unused_variables)]
pub(crate) trait BoolOp2Kernel {
    fn apply_i8(&self, a: i8, b: i8) -> bool {
        unimplemented!()
    }
    fn apply_i16(&self, a: i16, b: i16) -> bool {
        unimplemented!()
    }
    fn apply_i32(&self, a: i32, b: i32) -> bool {
        unimplemented!()
    }
    fn apply_i64(&self, a: i64, b: i64) -> bool {
        unimplemented!()
    }
    fn apply_u8(&self, a: u8, b: u8) -> bool {
        unimplemented!()
    }
    fn apply_u16(&self, a: u16, b: u16) -> bool {
        unimplemented!()
    }
    fn apply_u32(&self, a: u32, b: u32) -> bool {
        unimplemented!()
    }
    fn apply_u64(&self, a: u64, b: u64) -> bool {
        unimplemented!()
    }
    fn apply_f16(&self, a: f16, b: f16) -> bool {
        unimplemented!()
    }
    fn apply_f32(&self, a: f32, b: f32) -> bool {
        unimplemented!()
    }
    fn apply_f64(&self, a: f64, b: f64) -> bool {
        unimplemented!()
    }
    fn apply_complex_f32(&self, a: Complex<f32>, b: Complex<f32>) -> bool {
        unimplemented!()
    }
    fn apply_complex_f64(&self, a: Complex<f64>, b: Complex<f64>) -> bool {
        unimplemented!()
    }
    fn apply_bool(&self, a: bool, b: bool) -> bool {
        unimplemented!()
    }

    fn is_support_dtype(&self, dtype: &Dtype) -> bool;
}

pub(crate) struct BoolOp2<Op, S1, S2> {
    op: Op,

    a: Array<S1>,
    b: Array<S2>,

    dtype: Dtype,
    shape: DimArray<u64>,
    blocks_layout: BlocksLayout,
}
impl<Op, S1, S2> BoolOp2<Op, S1, S2> {
    pub(crate) fn new(op: Op, a: Array<S1>, b: Array<S2>) -> io::Result<Self>
    where
        Op: BoolOp2Kernel,
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
            dtype: bool::DTYPE,
            shape: a.shape().try_into().unwrap(),
            blocks_layout: a.blocks_layout().clone(),
            a,
            b,
        })
    }
}
impl<Op, S1, S2> ArrayStorage for BoolOp2<Op, S1, S2>
where
    Op: BoolOp2Kernel,
    S1: ArrayStorage,
    S2: ArrayStorage,
{
    fn shape(&self) -> &[u64] {
        &self.shape
    }

    fn dtype(&self) -> &Dtype {
        unsafe { assert_unchecked(self.dtype == bool::DTYPE) };
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
        let mut buf1 = context.tmp_buf(inner_buf_size, inner_dtype.alignment());
        let mut buf2 = context.tmp_buf(inner_buf_size, inner_dtype.alignment());
        let buf1 = buf1.as_mut_slice();
        let buf2 = buf2.as_mut_slice();

        self.a.storage.read_data(index, buf1, context)?;
        self.b.storage.read_data(index, buf2, context)?;

        macro_rules! apply_loop {
            ($ty:ty, $apply:ident) => {
                let buf1 = unsafe { cast_slice::<u8, $ty>(buf1) };
                let buf2 = unsafe { cast_slice::<u8, $ty>(buf2) };
                let buf = unsafe { cast_slice_mut::<u8, bool>(buf) };
                for (o, (a, b)) in buf.iter_mut().zip(buf1.iter().zip(buf2)) {
                    *o = self.op.$apply(*a, *b);
                }
            };
        }

        match self.dtype.try_to_scalar() {
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
                    "only scalar dtypes are supported for BoolOp2",
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
    ($Name:ident, $NameKernel:ident, |$a:ident, $b:ident| $body:expr, [$($scalar:tt),* $(,)?]) => {
        pub struct $Name<S1, S2>(BoolOp2<$NameKernel, S1, S2>);
        impl<S1, S2> $Name<S1, S2> {
            pub fn new(a: Array<S1>, b: Array<S2>) -> io::Result<Self>
            where
                S1: ArrayStorage,
                S2: ArrayStorage,
            {
                Ok(Self(BoolOp2::new($NameKernel, a, b)?))
            }
        }
        crate::storage::impl_array_storage_forward!($Name<S1, S2> where S1: ArrayStorage, S2: ArrayStorage);

        define_op_kernel!($NameKernel, |$a, $b| $body, [$($scalar),*]);
    };
}
macro_rules! define_op_kernel {
    ($NameKernel:ident, |$a:ident, $b:ident| $body:expr, [$($scalar:tt),* $(,)?]) => {
        struct $NameKernel;
        impl BoolOp2Kernel for $NameKernel {
            $(define_op_kernel!(@apply |$a, $b| $body, $scalar);)*

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
    (@apply |$a:ident, $b:ident| $body:expr, i8)  => { fn apply_i8(&self, $a: i8, $b: i8) -> bool { $body } };
    (@apply |$a:ident, $b:ident| $body:expr, i16) => { fn apply_i16(&self, $a: i16, $b: i16) -> bool { $body } };
    (@apply |$a:ident, $b:ident| $body:expr, i32) => { fn apply_i32(&self, $a: i32, $b: i32) -> bool { $body } };
    (@apply |$a:ident, $b:ident| $body:expr, i64) => { fn apply_i64(&self, $a: i64, $b: i64) -> bool { $body } };
    (@apply |$a:ident, $b:ident| $body:expr, u8)  => { fn apply_u8(&self, $a: u8, $b: u8) -> bool { $body } };
    (@apply |$a:ident, $b:ident| $body:expr, u16) => { fn apply_u16(&self, $a: u16, $b: u16) -> bool { $body } };
    (@apply |$a:ident, $b:ident| $body:expr, u32) => { fn apply_u32(&self, $a: u32, $b: u32) -> bool { $body } };
    (@apply |$a:ident, $b:ident| $body:expr, u64) => { fn apply_u64(&self, $a: u64, $b: u64) -> bool { $body } };
    (@apply |$a:ident, $b:ident| $body:expr, f32) => { fn apply_f32(&self, $a: f32, $b: f32) -> bool { $body } };
    (@apply |$a:ident, $b:ident| $body:expr, f64) => { fn apply_f64(&self, $a: f64, $b: f64) -> bool { $body } };
    (@apply |$a:ident, $b:ident| $body:expr, f16) => {
        fn apply_f16(&self, #[allow(unused_variables)] $a: f16, #[allow(unused_variables)] $b: f16) -> bool {
            cfg_if::cfg_if! { if #[cfg(feature = "half")] {
                $body
            } else {
                unimplemented!()
            } }
        }
    };
    (@apply |$a:ident, $b:ident| $body:expr, (Complex<f32>)) => {
        fn apply_complex_f32(&self, #[allow(unused_variables)] $a: Complex<f32>, #[allow(unused_variables)] $b: Complex<f32>) -> bool {
            cfg_if::cfg_if! { if #[cfg(feature = "num-complex")] {
                $body
            } else {
                unimplemented!()
            } }
        }
    };
    (@apply |$a:ident, $b:ident| $body:expr, (Complex<f64>)) => {
        fn apply_complex_f64(&self, #[allow(unused_variables)] $a: Complex<f64>, #[allow(unused_variables)] $b: Complex<f64>) -> bool {
            cfg_if::cfg_if! { if #[cfg(feature = "num-complex")] {
                $body
            } else {
                unimplemented!()
            } }
        }
    };
    (@apply |$a:ident, $b:ident| $body:expr, bool) => { fn apply_bool(&self, $a: bool, $b: bool) -> bool { $body } };

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

define_op!(Equal,        EqualKernel,        |a, b| a == b, [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, f16, (Complex<f32>), (Complex<f64>)]);
define_op!(NotEqual,     NotEqualKernel,     |a, b| a != b, [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, f16, (Complex<f32>), (Complex<f64>)]);
define_op!(
    Greater,
    GreaterKernel,
    |a, b| a > b,
    [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, f16]
);
define_op!(
    GreaterEqual,
    GreaterEqualKernel,
    |a, b| a >= b,
    [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, f16]
);
define_op!(
    Less,
    LessKernel,
    |a, b| a < b,
    [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, f16]
);
define_op!(
    LessEqual,
    LessEqualKernel,
    |a, b| a <= b,
    [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, f16]
);

impl<S> crate::Array<S>
where
    S: crate::storage::ArrayStorage,
{
    define_array_op2_method!(equal: Equal);
    define_array_op2_method!(not_equal: NotEqual);
    define_array_op2_method!(greater: Greater);
    define_array_op2_method!(greater_equal: GreaterEqual);
    define_array_op2_method!(less: Less);
    define_array_op2_method!(less_equal: LessEqual);
}
