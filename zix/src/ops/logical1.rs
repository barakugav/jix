use std::io;
use std::ops::Range;

use crate::array::Array;
use crate::codec::{DecoderCodecConfig, DecoderParams, EncoderParams, ReadContext};
use crate::dtype::{Complex, Dtype, DtypeScalarKind, Dtyped, f16};
use crate::ops::common::define_array_op1_method;
use crate::storage::{ArrayStorage, BlocksLayout};
use crate::util::DimArray;

#[allow(unused_variables)]
pub(crate) trait LogicalOp1Kernel<O> {
    fn apply_i8(&self, a: i8) -> O {
        unimplemented!()
    }
    fn apply_i16(&self, a: i16) -> O {
        unimplemented!()
    }
    fn apply_i32(&self, a: i32) -> O {
        unimplemented!()
    }
    fn apply_i64(&self, a: i64) -> O {
        unimplemented!()
    }
    fn apply_u8(&self, a: u8) -> O {
        unimplemented!()
    }
    fn apply_u16(&self, a: u16) -> O {
        unimplemented!()
    }
    fn apply_u32(&self, a: u32) -> O {
        unimplemented!()
    }
    fn apply_u64(&self, a: u64) -> O {
        unimplemented!()
    }
    fn apply_f16(&self, a: f16) -> O {
        unimplemented!()
    }
    fn apply_f32(&self, a: f32) -> O {
        unimplemented!()
    }
    fn apply_f64(&self, a: f64) -> O {
        unimplemented!()
    }
    fn apply_complex_f32(&self, a: Complex<f32>) -> O {
        unimplemented!()
    }
    fn apply_complex_f64(&self, a: Complex<f64>) -> O {
        unimplemented!()
    }
    fn apply_bool(&self, a: bool) -> O {
        unimplemented!()
    }

    fn is_support_dtype(&self, dtype: &Dtype) -> bool;
}

pub(crate) struct LogicalOp1<Op, S, O> {
    op: Op,
    a: Array<S>,
    dtype: Dtype,
    shape: DimArray<u64>,
    blocks_layout: BlocksLayout,
    _phantom: std::marker::PhantomData<O>,
}

impl<Op, S, O> LogicalOp1<Op, S, O> {
    pub(crate) fn new(op: Op, a: Array<S>) -> io::Result<Self>
    where
        Op: LogicalOp1Kernel<O>,
        S: ArrayStorage,
        O: Dtyped,
    {
        if !op.is_support_dtype(a.dtype()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unsupported dtype for operation: {:#?}", a.dtype()),
            ));
        }
        Ok(Self {
            op,
            dtype: O::DTYPE,
            shape: a.shape().try_into().unwrap(),
            blocks_layout: a.blocks_layout().clone(),
            _phantom: std::marker::PhantomData,
            a,
        })
    }
}

impl<Op, S, O> ArrayStorage for LogicalOp1<Op, S, O>
where
    Op: LogicalOp1Kernel<O>,
    S: ArrayStorage,
    O: Dtyped,
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
        let (src_dtype, dst_dtype) = (self.a.dtype(), O::DTYPE);
        let (src_itemsize, dst_itemsize) =
            (src_dtype.itemsize() as usize, dst_dtype.itemsize() as usize);
        let nitems = buf.len() / dst_itemsize;

        let in_place = src_itemsize == dst_itemsize
            && (buf.as_ptr() as usize).is_multiple_of(src_dtype.alignment() as usize);
        let mut tmp_buf;
        let (read_buf, dst) = if in_place {
            let ptr = buf.as_mut_ptr();
            ((ptr, buf.len()), ptr)
        } else {
            tmp_buf = context.tmp_buf(nitems * src_itemsize, src_dtype.alignment());
            let tmp_buf = tmp_buf.as_mut_slice();
            ((tmp_buf.as_mut_ptr(), tmp_buf.len()), buf.as_mut_ptr())
        };
        let read_buf = unsafe { std::slice::from_raw_parts_mut(read_buf.0, read_buf.1) };
        self.a.storage.read_data(index, read_buf, context)?;
        let src = read_buf.as_ptr();

        macro_rules! apply_loop {
            ($src_ty:ty, $apply_fn:ident) => {{
                for i in 0..nitems {
                    unsafe {
                        let value = src.cast::<$src_ty>().add(i).read();
                        let value = self.op.$apply_fn(value);
                        dst.cast::<O>().add(i).write(value);
                    }
                }
            }};
        }

        match src_dtype.try_to_scalar() {
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
                    "only scalar dtypes are supported for LogicalOp1",
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

macro_rules! define_logical1_op {
    ($Name:ident, $NameKernel:ident, |$arg:ident| -> $O:ty $body:block, [$($scalar:tt),* $(,)?]) => {
        pub struct $Name<S>(crate::ops::logical1::LogicalOp1<$NameKernel, S, $O>);
        impl<S> $Name<S> {
            pub(crate) fn new(a: crate::Array<S>) -> std::io::Result<Self>
            where
                S: crate::storage::ArrayStorage,
            {
                Ok(Self(crate::ops::logical1::LogicalOp1::new($NameKernel, a)?))
            }
        }
        crate::storage::impl_array_storage_forward!($Name<S> where S: crate::storage::ArrayStorage);

        crate::ops::logical1::define_logical1_op_kernel!($NameKernel, |$arg| -> $O $body, [$($scalar),*]);
    };
}
macro_rules! define_logical1_op_kernel {
    ($NameKernel:ident, |$arg:ident| -> $O:ty $body:block, [$($scalar:tt),* $(,)?]) => {
        struct $NameKernel;
        impl crate::ops::logical1::LogicalOp1Kernel<$O> for $NameKernel {
            $(crate::ops::logical1::define_logical1_op_kernel!(@apply |$arg| -> $O $body, $scalar);)*

            fn is_support_dtype(&self, dtype: &crate::dtype::Dtype) -> bool {
                use crate::dtype::DtypeScalarKind;
                let Some(scalar_kind) = dtype.try_to_scalar() else {
                    return false;
                };
                false $(|| crate::ops::logical1::define_logical1_op_kernel!(@dtype_match scalar_kind, $scalar))*
            }
        }
    };

    // --- apply arms ---
    (@apply |$arg:ident| -> $O:ty $body:block, i8)  => { fn apply_i8(&self, $arg: i8) -> $O $body };
    (@apply |$arg:ident| -> $O:ty $body:block, i16) => { fn apply_i16(&self, $arg: i16) -> $O $body };
    (@apply |$arg:ident| -> $O:ty $body:block, i32) => { fn apply_i32(&self, $arg: i32) -> $O $body };
    (@apply |$arg:ident| -> $O:ty $body:block, i64) => { fn apply_i64(&self, $arg: i64) -> $O $body };
    (@apply |$arg:ident| -> $O:ty $body:block, u8)  => { fn apply_u8(&self, $arg: u8) -> $O $body };
    (@apply |$arg:ident| -> $O:ty $body:block, u16) => { fn apply_u16(&self, $arg: u16) -> $O $body };
    (@apply |$arg:ident| -> $O:ty $body:block, u32) => { fn apply_u32(&self, $arg: u32) -> $O $body };
    (@apply |$arg:ident| -> $O:ty $body:block, u64) => { fn apply_u64(&self, $arg: u64) -> $O $body };
    (@apply |$arg:ident| -> $O:ty $body:block, f32) => { fn apply_f32(&self, $arg: f32) -> $O $body };
    (@apply |$arg:ident| -> $O:ty $body:block, f64) => { fn apply_f64(&self, $arg: f64) -> $O $body };
    (@apply |$arg:ident| -> $O:ty $body:block, f16) => {
        fn apply_f16(&self, #[allow(unused_variables)] $arg: f16) -> $O {
            cfg_if::cfg_if! { if #[cfg(feature = "half")] {
                $body
            } else {
                unimplemented!()
            } }
        }
    };
    (@apply |$arg:ident| -> $O:ty $body:block, (Complex<f32>)) => {
        fn apply_complex_f32(&self, #[allow(unused_variables)] $arg: Complex<f32>) -> $O {
            cfg_if::cfg_if! { if #[cfg(feature = "num-complex")] {
                $body
            } else {
                unimplemented!()
            } }
        }
    };
    (@apply |$arg:ident| -> $O:ty $body:block, (Complex<f64>)) => {
        fn apply_complex_f64(&self, #[allow(unused_variables)] $arg: Complex<f64>) -> $O {
            cfg_if::cfg_if! { if #[cfg(feature = "num-complex")] {
                $body
            } else {
                unimplemented!()
            } }
        }
    };
    (@apply |$arg:ident| -> $O:ty $body:block, bool) => { fn apply_bool(&self, $arg: bool) -> $O $body };

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
pub(crate) use {define_logical1_op, define_logical1_op_kernel};

define_logical1_op!(
    IsNan,
    IsNanKernel,
    |a| -> bool { a.is_nan() },
    [f16, f32, f64]
);
define_logical1_op!(
    IsFinite,
    IsFiniteKernel,
    |a| -> bool { a.is_finite() },
    [f16, f32, f64]
);
define_logical1_op!(
    IsInfinite,
    IsInfiniteKernel,
    |a| -> bool { a.is_infinite() },
    [f16, f32, f64]
);

impl<S> Array<S>
where
    S: ArrayStorage,
{
    define_array_op1_method!(is_nan: IsNan);
    define_array_op1_method!(is_finite: IsFinite);
    define_array_op1_method!(is_infinite: IsInfinite);
}
