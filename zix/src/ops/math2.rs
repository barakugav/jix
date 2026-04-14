use std::io;
use std::ops::Range;

use crate::array::Array;
use crate::codec::{DecoderCodecConfig, DecoderParams, EncoderParams, ReadContext};
use crate::dtype::{Complex, Dtype, DtypeScalarKind, f16};
use crate::ops::common::define_array_op2_method;
use crate::storage::{ArrayStorage, BlocksLayout};
use crate::util::{DimArray, cast_slice, cast_slice_mut};

#[allow(unused_variables)]
pub(crate) trait MathOp2Kernel {
    fn apply_i8(&self, a: i8, b: i8) -> i8 {
        unimplemented!()
    }
    fn apply_i16(&self, a: i16, b: i16) -> i16 {
        unimplemented!()
    }
    fn apply_i32(&self, a: i32, b: i32) -> i32 {
        unimplemented!()
    }
    fn apply_i64(&self, a: i64, b: i64) -> i64 {
        unimplemented!()
    }
    fn apply_u8(&self, a: u8, b: u8) -> u8 {
        unimplemented!()
    }
    fn apply_u16(&self, a: u16, b: u16) -> u16 {
        unimplemented!()
    }
    fn apply_u32(&self, a: u32, b: u32) -> u32 {
        unimplemented!()
    }
    fn apply_u64(&self, a: u64, b: u64) -> u64 {
        unimplemented!()
    }
    fn apply_f16(&self, a: f16, b: f16) -> f16 {
        unimplemented!()
    }
    fn apply_f32(&self, a: f32, b: f32) -> f32 {
        unimplemented!()
    }
    fn apply_f64(&self, a: f64, b: f64) -> f64 {
        unimplemented!()
    }
    fn apply_complex_f32(&self, a: Complex<f32>, b: Complex<f32>) -> Complex<f32> {
        unimplemented!()
    }
    fn apply_complex_f64(&self, a: Complex<f64>, b: Complex<f64>) -> Complex<f64> {
        unimplemented!()
    }
    fn apply_bool(&self, a: bool, b: bool) -> bool {
        unimplemented!()
    }

    fn is_support_dtype(&self, dtype: &Dtype) -> bool;
}

pub(crate) struct MathOp2<Op, S1, S2> {
    op: Op,

    a: Array<S1>,
    b: Array<S2>,

    dtype: Dtype,
    shape: DimArray<u64>,
    blocks_layout: BlocksLayout,
}
impl<Op, S1, S2> MathOp2<Op, S1, S2> {
    pub(crate) fn new(op: Op, a: Array<S1>, b: Array<S2>) -> io::Result<Self>
    where
        Op: MathOp2Kernel,
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
impl<Op, S1, S2> ArrayStorage for MathOp2<Op, S1, S2>
where
    Op: MathOp2Kernel,
    S1: ArrayStorage,
    S2: ArrayStorage,
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
        let mut buf2 = context.tmp_buf(buf.len(), self.dtype.alignment());
        let buf2 = buf2.as_mut_slice();

        self.a.storage.read_data(index, buf, context)?;
        self.b.storage.read_data(index, buf2, context)?;

        macro_rules! apply_loop {
            ($ty:ty, $apply:ident) => {
                let buf1 = unsafe { cast_slice_mut::<u8, $ty>(buf) };
                let buf2 = unsafe { cast_slice::<u8, $ty>(&buf2) };
                for (a, b) in buf1.iter_mut().zip(buf2) {
                    *a = self.op.$apply(*a, *b);
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
                    "only scalar dtypes are supported for MathOp2",
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

macro_rules! define_math2_op {
    ($Name:ident, $NameKernel:ident, |$a:ident, $b:ident| $body:expr, [$($scalar:tt),* $(,)?]) => {
        pub struct $Name<S1, S2>(crate::ops::math2::MathOp2<$NameKernel, S1, S2>);
        impl<S1, S2> $Name<S1, S2> {
            pub fn new(a: crate::Array<S1>, b: crate::Array<S2>) -> std::io::Result<Self>
            where
                S1: crate::storage::ArrayStorage,
                S2: crate::storage::ArrayStorage,
            {
                Ok(Self(crate::ops::math2::MathOp2::new($NameKernel, a, b)?))
            }
        }
        crate::storage::impl_array_storage_forward!($Name<S1, S2> where S1: crate::storage::ArrayStorage, S2: crate::storage::ArrayStorage);

        crate::ops::math2::define_math2_op_kernel!($NameKernel, |$a, $b| $body, [$($scalar),*]);
    };
}
macro_rules! define_math2_core_op {
    ($Name:ident, $NameKernel:ident, $op_trait:ident, $op_fn:ident, |$a:ident, $b:ident| $body:expr, [$($scalar:tt),* $(,)?]) => {
        define_math2_op!($Name, $NameKernel, |$a, $b| $body, [$($scalar),*]);
        impl<S1, S2> core::ops::$op_trait<Array<S2>> for Array<S1>
        where
            S1: ArrayStorage,
            S2: ArrayStorage,
        {
            type Output = Array<$Name<S1, S2>>;
            #[track_caller]
            fn $op_fn(self, b: Array<S2>) -> Array<$Name<S1, S2>> {
                let op = $Name::new(self, b).unwrap();
                Array::from_storage(op)
            }
        }
    };
}
macro_rules! define_math2_op_kernel {
    ($NameKernel:ident, |$a:ident, $b:ident| $body:expr, [$($scalar:tt),* $(,)?]) => {
        struct $NameKernel;
        impl crate::ops::math2::MathOp2Kernel for $NameKernel {
            $(crate::ops::math2::define_math2_op_kernel!(@apply |$a, $b| $body, $scalar);)*

            fn is_support_dtype(&self, dtype: &crate::dtype::Dtype) -> bool {
                use crate::dtype::DtypeScalarKind;
                let Some(scalar_kind) = dtype.try_to_scalar() else {
                    return false;
                };
                false $(|| crate::ops::math2::define_math2_op_kernel!(@dtype_match scalar_kind, $scalar))*
            }
        }
    };

    // --- apply arms ---
    (@apply |$a:ident, $b:ident| $body:expr, i8)  => { fn apply_i8(&self, $a: i8, $b: i8) -> i8 { $body } };
    (@apply |$a:ident, $b:ident| $body:expr, i16) => { fn apply_i16(&self, $a: i16, $b: i16) -> i16 { $body } };
    (@apply |$a:ident, $b:ident| $body:expr, i32) => { fn apply_i32(&self, $a: i32, $b: i32) -> i32 { $body } };
    (@apply |$a:ident, $b:ident| $body:expr, i64) => { fn apply_i64(&self, $a: i64, $b: i64) -> i64 { $body } };
    (@apply |$a:ident, $b:ident| $body:expr, u8)  => { fn apply_u8(&self, $a: u8, $b: u8) -> u8 { $body } };
    (@apply |$a:ident, $b:ident| $body:expr, u16) => { fn apply_u16(&self, $a: u16, $b: u16) -> u16 { $body } };
    (@apply |$a:ident, $b:ident| $body:expr, u32) => { fn apply_u32(&self, $a: u32, $b: u32) -> u32 { $body } };
    (@apply |$a:ident, $b:ident| $body:expr, u64) => { fn apply_u64(&self, $a: u64, $b: u64) -> u64 { $body } };
    (@apply |$a:ident, $b:ident| $body:expr, f32) => { fn apply_f32(&self, $a: f32, $b: f32) -> f32 { $body } };
    (@apply |$a:ident, $b:ident| $body:expr, f64) => { fn apply_f64(&self, $a: f64, $b: f64) -> f64 { $body } };
    (@apply |$a:ident, $b:ident| $body:expr, f16) => {
        fn apply_f16(&self, #[allow(unused_variables)] $a: f16, #[allow(unused_variables)] $b: f16) -> f16 {
            cfg_if::cfg_if! { if #[cfg(feature = "half")] {
                $body
            } else {
                unimplemented!()
            } }
        }
    };
    (@apply |$a:ident, $b:ident| $body:expr, (Complex<f32>)) => {
        fn apply_complex_f32(&self, #[allow(unused_variables)] $a: Complex<f32>, #[allow(unused_variables)] $b: Complex<f32>) -> Complex<f32> {
            cfg_if::cfg_if! { if #[cfg(feature = "num-complex")] {
                $body
            } else {
                unimplemented!()
            } }
        }
    };
    (@apply |$a:ident, $b:ident| $body:expr, (Complex<f64>)) => {
        fn apply_complex_f64(&self, #[allow(unused_variables)] $a: Complex<f64>, #[allow(unused_variables)] $b: Complex<f64>) -> Complex<f64> {
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

pub(crate) use {define_math2_op, define_math2_op_kernel};

define_math2_core_op!(Add, AddKernel, Add, add, |a, b| a + b, [i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64, (Complex<f32>), (Complex<f64>)]);
define_math2_core_op!(Sub, SubKernel, Sub, sub, |a, b| a - b, [i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64, (Complex<f32>), (Complex<f64>)]);
define_math2_core_op!(Mul, MulKernel, Mul, mul, |a, b| a * b, [i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64, (Complex<f32>), (Complex<f64>)]);
define_math2_core_op!(Div, DivKernel, Div, div, |a, b| a / b, [i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64, (Complex<f32>), (Complex<f64>)]);
define_math2_op!(
    Maximum,
    MaximumKernel,
    |a, b| MaximumTrait::maximum(a, b),
    [i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64, bool]
);
define_math2_op!(
    Minimum,
    MinimumKernel,
    |a, b| MinimumTrait::minimum(a, b),
    [i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64, bool]
);

trait MaximumTrait {
    fn maximum(self, other: Self) -> Self;
}
macro_rules! impl_integer_maximum {
    ($($t:ty),* $(,)?) => {
        $(impl MaximumTrait for $t {
            fn maximum(self, other: Self) -> Self {
                std::cmp::max(self, other)
            }
        })*
    };
}
macro_rules! impl_float_maximum {
    ($($t:ty),* $(,)?) => {
        $(impl MaximumTrait for $t {
            fn maximum(self, other: Self) -> Self {
                if self.is_nan() | other.is_nan() {
                    Self::NAN
                } else {
                    self.max(other)
                }
            }
        })*
    };
}
impl_integer_maximum!(i8, i16, i32, i64, u8, u16, u32, u64, bool);
impl_float_maximum!(f32, f64);
#[cfg(feature = "half")]
impl_float_maximum!(f16);

trait MinimumTrait {
    fn minimum(self, other: Self) -> Self;
}
macro_rules! impl_integer_minimum {
    ($($t:ty),* $(,)?) => {
        $(impl MinimumTrait for $t {
            fn minimum(self, other: Self) -> Self {
                std::cmp::min(self, other)
            }
        })*
    };
}
macro_rules! impl_float_minimum {
    ($($t:ty),* $(,)?) => {
        $(impl MinimumTrait for $t {
            fn minimum(self, other: Self) -> Self {
                if self.is_nan() | other.is_nan() {
                    Self::NAN
                } else {
                    self.min(other)
                }
            }
        })*
    };
}
impl_integer_minimum!(i8, i16, i32, i64, u8, u16, u32, u64, bool);
impl_float_minimum!(f32, f64);
#[cfg(feature = "half")]
impl_float_minimum!(f16);

impl<S> crate::Array<S>
where
    S: crate::storage::ArrayStorage,
{
    define_array_op2_method!(maximum: Maximum);
    define_array_op2_method!(minimum: Minimum);
}

#[cfg(test)]
mod tests {
    use crate::{array::ArrayParams, block::BlockSize};

    fn arr_params(block_shape: &[usize]) -> ArrayParams {
        ArrayParams {
            block_shape: Some(block_shape.iter().map(|&x| x as BlockSize).collect()),
            ..ArrayParams::default()
        }
    }

    // Generates 5 test functions per (op, dtype).
    // Each Scalar impl controls the sampling range for its type.
    macro_rules! test_op_dtype {
        ($op:tt, $dtype:ident) => {
            paste::paste! {
                #[test]
                fn [<test_ $dtype _1d>]() {
                    use crate::array::Array;
                    let seed = stringify!($dtype).as_bytes().iter().chain(stringify!($op).as_bytes()).fold(0xdeadbeef_cafe1234u64, |acc, b| acc + *b as u64);
                    let mut rng = fastrand::Rng::with_seed(seed);
                    let b = super::rand_array::<$dtype>(&mut rng, &[4]);
                    let a = super::rand_array::<$dtype>(&mut rng, &[4]) + &b;
                    let za = Array::from_ndarray(&a, arr_params(&[4])).unwrap();
                    let zb = Array::from_ndarray(&b, arr_params(&[4])).unwrap();
                    let actual = (za $op zb).data().to_ndarray::<$dtype>().unwrap();
                    let expected = &a $op &b;
                    assert_eq!(actual, expected);
                }

                #[test]
                fn [<test_ $dtype _1d_multi_block>]() {
                    use crate::array::Array;
                    let seed = stringify!($dtype).as_bytes().iter().chain(stringify!($op).as_bytes()).fold(0xdeadbeef_cafe1234u64, |acc, b| acc + *b as u64);
                    let mut rng = fastrand::Rng::with_seed(seed);
                    let b = super::rand_array::<$dtype>(&mut rng, &[6]);
                    let a = super::rand_array::<$dtype>(&mut rng, &[6]) + &b;
                    let za = Array::from_ndarray(&a, arr_params(&[2])).unwrap();
                    let zb = Array::from_ndarray(&b, arr_params(&[2])).unwrap();
                    let actual = (za $op zb).data().to_ndarray::<$dtype>().unwrap();
                    let expected = &a $op &b;
                    assert_eq!(actual, expected);
                }

                #[test]
                fn [<test_ $dtype _2d>]() {
                    use crate::array::Array;
                    let seed = stringify!($dtype).as_bytes().iter().chain(stringify!($op).as_bytes()).fold(0xdeadbeef_cafe1234u64, |acc, b| acc + *b as u64);
                    let mut rng = fastrand::Rng::with_seed(seed);
                    let b = super::rand_array::<$dtype>(&mut rng, &[2, 3]);
                    let a = super::rand_array::<$dtype>(&mut rng, &[2, 3]) + &b;
                    let za = Array::from_ndarray(&a, arr_params(&[2, 3])).unwrap();
                    let zb = Array::from_ndarray(&b, arr_params(&[2, 3])).unwrap();
                    let actual = (za $op zb).data().to_ndarray::<$dtype>().unwrap();
                    let expected = &a $op &b;
                    assert_eq!(actual, expected);
                }

                #[test]
                fn [<test_ $dtype _2d_multi_block>]() {
                    use crate::array::Array;
                    let seed = stringify!($dtype).as_bytes().iter().chain(stringify!($op).as_bytes()).fold(0xdeadbeef_cafe1234u64, |acc, b| acc + *b as u64);
                    let mut rng = fastrand::Rng::with_seed(seed);
                    let b = super::rand_array::<$dtype>(&mut rng, &[4, 4]);
                    let a = super::rand_array::<$dtype>(&mut rng, &[4, 4]) + &b;
                    let za = Array::from_ndarray(&a, arr_params(&[2, 2])).unwrap();
                    let zb = Array::from_ndarray(&b, arr_params(&[2, 2])).unwrap();
                    let actual = (za $op zb).data().to_ndarray::<$dtype>().unwrap();
                    let expected = &a $op &b;
                    assert_eq!(actual, expected);
                }

                #[test]
                fn [<test_ $dtype _three_arrays>]() {
                    if size_of::<$dtype>() < 2 {
                        // Skip this test for small types to avoid overflow in ops.
                        return;
                    }
                    use crate::array::Array;
                    let seed = stringify!($dtype).as_bytes().iter().chain(stringify!($op).as_bytes()).fold(0xdeadbeef_cafe1234u64, |acc, b| acc + *b as u64);
                    let mut rng = fastrand::Rng::with_seed(seed);
                    let c = super::rand_array::<$dtype>(&mut rng, &[4]);
                    let b = super::rand_array::<$dtype>(&mut rng, &[4]);
                    let a = super::rand_array::<$dtype>(&mut rng, &[4]) + &b + &c;
                    let za = Array::from_ndarray(&a, arr_params(&[4])).unwrap();
                    let zb = Array::from_ndarray(&b, arr_params(&[4])).unwrap();
                    let zc = Array::from_ndarray(&c, arr_params(&[4])).unwrap();
                    let zab = za $op zb.as_ref();
                    let actual = (zab $op zc).data().to_ndarray::<$dtype>().unwrap();
                    let expected = &(&a $op &b) $op &c;
                    assert_eq!(actual, expected);
                }
            }
        };
    }

    // Creates a module named $mod_name with one test set per dtype, all using $op.
    // Optional trailing groups add feature-gated dtypes: #[cfg(feature = "...")] [dtype, ...]
    macro_rules! test_op {
        ($mod_name:ident, $op:tt, [$($dtype:ident),+] $(, #[cfg($cfg:meta)] [$($cfg_dtype:ident),+])*) => {
            mod $mod_name {
                // Import feature-gated type aliases defined in the parent tests module.
                use super::arr_params;
                $(#[cfg($cfg)] use super::{$($cfg_dtype),+};)*
                $(test_op_dtype!($op, $dtype);)+
                $($(
                    #[cfg($cfg)]
                    test_op_dtype!($op, $cfg_dtype);
                )+)*
            }
        };
    }

    test_op!(add, +,
        [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64],
        #[cfg(feature = "half")] [f16],
        #[cfg(feature = "num-complex")] [complex_f32, complex_f64]
    );
    test_op!(sub, -,
        [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64],
        #[cfg(feature = "half")] [f16],
        #[cfg(feature = "num-complex")] [complex_f32, complex_f64]
    );
    test_op!(mul, *,
        [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64],
        #[cfg(feature = "half")] [f16],
        #[cfg(feature = "num-complex")] [complex_f32, complex_f64]
    );
    test_op!(div, /,
        [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64],
        #[cfg(feature = "half")] [f16],
        #[cfg(feature = "num-complex")] [complex_f32, complex_f64]
    );

    // Bring half::f16 into scope under the name `f16` so the macro ident resolves correctly.
    #[cfg(feature = "half")]
    use crate::dtype::f16;
    #[cfg(feature = "num-complex")]
    #[allow(non_camel_case_types)]
    type complex_f32 = crate::dtype::Complex<f32>;
    #[cfg(feature = "num-complex")]
    #[allow(non_camel_case_types)]
    type complex_f64 = crate::dtype::Complex<f64>;

    trait Scalar: Sized {
        fn sample(rng: &mut fastrand::Rng) -> Self;
    }
    macro_rules! impl_test_val {
        ($range:expr, $($t:ty),+) => {
            $(impl Scalar for $t {
                fn sample(rng: &mut fastrand::Rng) -> Self {
                    rng.u8($range) as Self
                }
            })+
        };
    }
    // [1,4]:  max cube 4³=64  < i8::MAX (127)
    // [1,30]: max cube 30³=27k < i16::MAX (32767)
    // [1,100]: safe for i32/i64/f32/f64
    impl_test_val!(1..=4, i8, u8);
    impl_test_val!(1..=30, i16, u16, u32, u64);
    impl_test_val!(1..=100, i32, i64, f32, f64);
    #[cfg(feature = "half")]
    impl Scalar for f16 {
        fn sample(rng: &mut fastrand::Rng) -> Self {
            Self::from_f32(rng.u8(1..=15) as f32)
        }
    }
    #[cfg(feature = "num-complex")]
    impl Scalar for complex_f32 {
        fn sample(rng: &mut fastrand::Rng) -> Self {
            Self::new(rng.u8(1..=15) as f32, 0.0)
        }
    }
    #[cfg(feature = "num-complex")]
    impl Scalar for complex_f64 {
        fn sample(rng: &mut fastrand::Rng) -> Self {
            Self::new(rng.u8(1..=15) as f64, 0.0)
        }
    }

    fn rand_array<T: Scalar>(rng: &mut fastrand::Rng, shape: &[usize]) -> ndarray::ArrayD<T> {
        ndarray::Array::from_shape_fn(ndarray::IxDyn(shape), |_| T::sample(rng))
    }
}
