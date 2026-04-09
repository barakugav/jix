use std::io;
use std::ops::Range;

use crate::array::{Array, BlocksLayout};
use crate::codec::ReadContext;
use crate::dtype::{Complex, Dtype, DtypeScalarKind, f16};
use crate::storage::{ArrayStorage, Ref};
use crate::util::{DimArray, cast_slice, cast_slice_mut};

pub(crate) trait MathOp2Kernel {
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

pub(crate) struct MathOp2<Op, S1, S2> {
    op: Op,

    a: Array<S1>,
    b: Array<S2>,

    dtype: Dtype,
    shape: DimArray<usize>,
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
                    "only scalar dtypes are supported for MathOp2",
                ));
            }
        })
    }
}

macro_rules! define_op {
    ($Name:ident, $NameKernel:ident, $op_trait:ident, $op_fn:ident, $op:tt) => {
        pub struct $Name<S1, S2>(MathOp2<$NameKernel, S1, S2>);
        impl<S1, S2> $Name<S1, S2> {
            pub fn new(a: Array<S1>, b: Array<S2>) -> io::Result<Self>
            where
                S1: ArrayStorage,
                S2: ArrayStorage,
            {
                Ok(Self(MathOp2::new($NameKernel, a, b)?))
            }
        }
        impl<'a, 'b, S1, S2> core::ops::$op_trait<&'b Array<S2>> for &'a Array<S1>
        where
            S1: ArrayStorage,
            S2: ArrayStorage,
        {
            type Output = Array<$Name<Ref<'a, S1>, Ref<'b, S2>>>;
            #[track_caller]
            fn $op_fn(self, b: &'b Array<S2>) -> Array<$Name<Ref<'a, S1>, Ref<'b, S2>>> {
                let a = Array::from_storage(Ref(&self.storage));
                let b = Array::from_storage(Ref(&b.storage));
                let op = $Name::new(a, b).unwrap();
                Array::from_storage(op)
            }
        }
        crate::storage::impl_array_storage_forward!($Name<S1, S2> where S1: ArrayStorage, S2: ArrayStorage);

        struct $NameKernel;
        impl MathOp2Kernel for $NameKernel {
            fn apply_i8(&self, a: i8, b: i8) -> i8 { a $op b }
            fn apply_i16(&self, a: i16, b: i16) -> i16 { a $op b }
            fn apply_i32(&self, a: i32, b: i32) -> i32 { a $op b }
            fn apply_i64(&self, a: i64, b: i64) -> i64 { a $op b }
            fn apply_u8(&self, a: u8, b: u8) -> u8 { a $op b }
            fn apply_u16(&self, a: u16, b: u16) -> u16 { a $op b }
            fn apply_u32(&self, a: u32, b: u32) -> u32 { a $op b }
            fn apply_u64(&self, a: u64, b: u64) -> u64 { a $op b }

            #[allow(unused_variables)]
            fn apply_f16(&self, a: f16, b: f16) -> f16 {
                cfg_if::cfg_if! { if #[cfg(feature = "half")] {
                    a $op b
                } else {
                    unimplemented!()
                } }
            }

            fn apply_f32(&self, a: f32, b: f32) -> f32 { a $op b }
            fn apply_f64(&self, a: f64, b: f64) -> f64 { a $op b }

            #[allow(unused_variables)]
            fn apply_complex_f32(&self, a: Complex<f32>, b: Complex<f32>) -> Complex<f32> {
                cfg_if::cfg_if! { if #[cfg(feature = "num-complex")] {
                    a $op b
                } else {
                    unimplemented!()
                } }
            }

            #[allow(unused_variables)]
            fn apply_complex_f64(&self, a: Complex<f64>, b: Complex<f64>) -> Complex<f64> {
                cfg_if::cfg_if! { if #[cfg(feature = "num-complex")] {
                    a $op b
                } else {
                    unimplemented!()
                } }
            }

            #[allow(unused_variables)]
            fn apply_bool(&self, a: bool, b: bool) -> bool {
                unimplemented!()
            }

            fn is_support_dtype(&self, dtype: &crate::dtype::Dtype) -> bool {
                use crate::dtype::DtypeScalarKind;
                let Some(scalar_kind) = dtype.try_to_scalar() else {
                    return false;
                };
                matches!(scalar_kind,
                    DtypeScalarKind::I8
                    | DtypeScalarKind::I16
                    | DtypeScalarKind::I32
                    | DtypeScalarKind::I64
                    | DtypeScalarKind::U8
                    | DtypeScalarKind::U16
                    | DtypeScalarKind::U32
                    | DtypeScalarKind::U64
                    | DtypeScalarKind::F32
                    | DtypeScalarKind::F64
                )
                || (cfg!(feature = "half") && matches!(scalar_kind,
                    DtypeScalarKind::F16
                ))
                || (cfg!(feature = "num-complex") && matches!(scalar_kind,
                    DtypeScalarKind::ComplexF32
                    | DtypeScalarKind::ComplexF64
                ))
            }
        }
    };
}

define_op!(Add, AddKernel, Add, add, +);
define_op!(Sub, SubKernel, Sub, sub, -);
define_op!(Mul, MulKernel, Mul, mul, *);
define_op!(Div, DivKernel, Div, div, /);

#[cfg(test)]
mod tests {

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
                    let za = Array::from_ndarray(&a, &[4]).unwrap();
                    let zb = Array::from_ndarray(&b, &[4]).unwrap();
                    let actual = (&za $op &zb).data().to_ndarray::<$dtype>().unwrap();
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
                    let za = Array::from_ndarray(&a, &[2]).unwrap();
                    let zb = Array::from_ndarray(&b, &[2]).unwrap();
                    let actual = (&za $op &zb).data().to_ndarray::<$dtype>().unwrap();
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
                    let za = Array::from_ndarray(&a, &[2, 3]).unwrap();
                    let zb = Array::from_ndarray(&b, &[2, 3]).unwrap();
                    let actual = (&za $op &zb).data().to_ndarray::<$dtype>().unwrap();
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
                    let za = Array::from_ndarray(&a, &[2, 2]).unwrap();
                    let zb = Array::from_ndarray(&b, &[2, 2]).unwrap();
                    let actual = (&za $op &zb).data().to_ndarray::<$dtype>().unwrap();
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
                    let za = Array::from_ndarray(&a, &[4]).unwrap();
                    let zb = Array::from_ndarray(&b, &[4]).unwrap();
                    let zc = Array::from_ndarray(&c, &[4]).unwrap();
                    let zab = &za $op &zb;
                    let actual = (&zab $op &zc).data().to_ndarray::<$dtype>().unwrap();
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
