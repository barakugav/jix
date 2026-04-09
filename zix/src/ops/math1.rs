use std::io;
use std::ops::Range;

use crate::array::{Array, BlocksLayout};
use crate::codec::ReadContext;
use crate::dtype::{Complex, Dtype, DtypeScalarKind, f16};
use crate::storage::{ArrayStorage, Ref};
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
    shape: DimArray<usize>,
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
        index: &[Range<usize>],
        buf: &mut [u8],
        context: &ReadContext,
    ) -> std::io::Result<()> {
        self.a.storage.read_data(index, buf, context)?;

        macro_rules! apply_loop {
            ($ty:ty, $apply:ident) => {
                let buf = unsafe { cast_slice_mut::<u8, $ty>(buf) };
                for a in buf.iter_mut() {
                    *a = self.op.$apply(*a);
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
                    "only scalar dtypes are supported for MathOp1",
                ));
            }
        })
    }
}

macro_rules! define_core_op {
    ($Name:ident, $NameKernel:ident, $op_trait:ident, $op:ident, [$($scalar:tt),* $(,)?]) => {
        pub struct $Name<S>(MathOp1<$NameKernel, S>);
        impl<S> $Name<S> {
            pub(crate) fn new(a: Array<S>) -> io::Result<Self>
            where
                S: ArrayStorage,
            {
                Ok(Self(MathOp1::new($NameKernel, a)?))
            }
        }
        impl<'a, S> core::ops::$op_trait for &'a Array<S>
        where
            S: ArrayStorage,
        {
            type Output = Array<$Name<Ref<'a, S>>>;
            #[track_caller]
            fn $op(self) -> Array<$Name<Ref<'a, S>>> {
                let a = Array::from_storage(Ref(&self.storage));
                let op = $Name::new(a).unwrap();
                Array::from_storage(op)
            }
        }
        crate::storage::impl_array_storage_forward!($Name<S> where S: ArrayStorage);

        define_op_kernel!($NameKernel, $op, [$($scalar),*]);
    };
}
macro_rules! define_op {
    ($Name:ident, $NameKernel:ident, $op_trait:ident, $op:ident, [$($scalar:tt),* $(,)?]) => {
        pub struct $Name<S>(MathOp1<$NameKernel, S>);
        impl<S> $Name<S> {
            pub(crate) fn new(a: Array<S>) -> io::Result<Self>
            where
                S: ArrayStorage,
            {
                Ok(Self(MathOp1::new($NameKernel, a)?))
            }
        }
        impl<S> Array<S>
            where
                S: ArrayStorage,
            {
            #[track_caller]
            pub fn $op(&self) -> Array<$Name<Ref<'_, S>>> {
                let a = Array::from_storage(Ref(&self.storage));
                let op = $Name::new(a).unwrap();
                Array::from_storage(op)
            }
        }
        crate::storage::impl_array_storage_forward!($Name<S> where S: ArrayStorage);

        define_op_kernel!($NameKernel, $op, [$($scalar),*]);
    };
}
macro_rules! define_op_kernel {
    ($NameKernel:ident, $op:ident, [$($scalar:tt),* $(,)?]) => {
        struct $NameKernel;
        impl MathOp1Kernel for $NameKernel {
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
    (@apply $op:tt, i8)  => { fn apply_i8(&self, a: i8) -> i8 { a.$op() } };
    (@apply $op:tt, i16) => { fn apply_i16(&self, a: i16) -> i16 { a.$op() } };
    (@apply $op:tt, i32) => { fn apply_i32(&self, a: i32) -> i32 { a.$op() } };
    (@apply $op:tt, i64) => { fn apply_i64(&self, a: i64) -> i64 { a.$op() } };
    (@apply $op:tt, u8)  => { fn apply_u8(&self, a: u8) -> u8 { a.$op() } };
    (@apply $op:tt, u16) => { fn apply_u16(&self, a: u16) -> u16 { a.$op() } };
    (@apply $op:tt, u32) => { fn apply_u32(&self, a: u32) -> u32 { a.$op() } };
    (@apply $op:tt, u64) => { fn apply_u64(&self, a: u64) -> u64 { a.$op() } };
    (@apply $op:tt, f32) => { fn apply_f32(&self, a: f32) -> f32 { a.$op() } };
    (@apply $op:tt, f64) => { fn apply_f64(&self, a: f64) -> f64 { a.$op() } };
    (@apply $op:tt, f16) => {
        fn apply_f16(&self, #[allow(unused_variables)] a: f16) -> f16 {
            cfg_if::cfg_if! { if #[cfg(feature = "half")] {
                a.$op()
            } else {
                unimplemented!()
            } }
        }
    };
    (@apply $op:tt, (Complex<f32>)) => {
        fn apply_complex_f32(&self, #[allow(unused_variables)] a: Complex<f32>) -> Complex<f32> {
            cfg_if::cfg_if! { if #[cfg(feature = "num-complex")] {
                a.$op()
            } else {
                unimplemented!()
            } }
        }
    };
    (@apply $op:tt, (Complex<f64>)) => {
        fn apply_complex_f64(&self, #[allow(unused_variables)] a: Complex<f64>) -> Complex<f64> {
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

use core::ops::Neg as _;
define_core_op!(Neg, NegKernel, Neg, neg, [i8, i16, i32, i64, f16, f32, f64, (Complex<f32>), (Complex<f64>)]);
define_op!(Floor, FloorKernel, Floor, floor, [f32, f64]);
define_op!(Ceil, CeilKernel, Ceil, ceil, [f32, f64]);
