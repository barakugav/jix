use std::io;

use crate::array::{Array, ArrayStorage, Ref};
use crate::dtype::{Complex, f16};
use crate::ops::scalar::{ScalarOp2, ScalarOp2Base};

macro_rules! define_op {
    ($Name:ident, $op_trait:ident, $op_fn:ident, $op:tt) => {
        pub struct $Name<S1, S2>(ScalarOp2Base<S1, S2>);
        impl<'a, 'b, S1, S2> $Name<Ref<'a, S1>, Ref<'b, S2>> {
            pub(crate) fn new(a: &'a Array<S1>, b: &'b Array<S2>) -> io::Result<Self>
            where
                S1: ArrayStorage,
                S2: ArrayStorage,
            {
                if a.shape() != b.shape() {
                    return Err(io::Error::new(io::ErrorKind::InvalidInput, "shape mismatch"));
                }
                if a.dtype() != b.dtype() {
                    return Err(io::Error::new(io::ErrorKind::InvalidInput, "dtype mismatch"));
                }
                Ok(Self(ScalarOp2Base {
                    a: Array::new(Ref::new(&a.storage)),
                    b: Array::new(Ref::new(&b.storage)),

                    dtype: a.dtype().clone(),
                    shape: a.shape().try_into().unwrap(),
                    blocks_layout: a.blocks_layout().clone(),
                }))
            }
        }
        impl<'a, 'b, S1, S2> core::ops::$op_trait<&'b Array<S2>> for &'a Array<S1>
        where
            S1: ArrayStorage,
            S2: ArrayStorage,
        {
            type Output = Array<$Name<Ref<'a, S1>, Ref<'b, S2>>>;
            fn $op_fn(self, b: &'b Array<S2>) -> Array<$Name<Ref<'a, S1>, Ref<'b, S2>>> {
                Array::new($Name::new(self, b).unwrap())
            }
        }

        impl<S1, S2> ScalarOp2 for $Name<S1, S2> {
            fn apply_i8(a: i8, b: i8) -> i8 { a $op b }
            fn apply_i16(a: i16, b: i16) -> i16 { a $op b }
            fn apply_i32(a: i32, b: i32) -> i32 { a $op b }
            fn apply_i64(a: i64, b: i64) -> i64 { a $op b }
            fn apply_u8(a: u8, b: u8) -> u8 { a $op b }
            fn apply_u16(a: u16, b: u16) -> u16 { a $op b }
            fn apply_u32(a: u32, b: u32) -> u32 { a $op b }
            fn apply_u64(a: u64, b: u64) -> u64 { a $op b }

            #[allow(unused_variables)]
            fn apply_f16(a: f16, b: f16) -> f16 {
                cfg_if::cfg_if! { if #[cfg(feature = "half")] {
                    a $op b
                } else {
                    unimplemented!()
                } }
            }

            fn apply_f32(a: f32, b: f32) -> f32 { a $op b }
            fn apply_f64(a: f64, b: f64) -> f64 { a $op b }

            #[allow(unused_variables)]
            fn apply_complex_f32(a: Complex<f32>, b: Complex<f32>) -> Complex<f32> {
                cfg_if::cfg_if! { if #[cfg(feature = "num-complex")] {
                    a $op b
                } else {
                    unimplemented!()
                } }
            }

            #[allow(unused_variables)]
            fn apply_complex_f64(a: Complex<f64>, b: Complex<f64>) -> Complex<f64> {
                cfg_if::cfg_if! { if #[cfg(feature = "num-complex")] {
                    a $op b
                } else {
                    unimplemented!()
                } }
            }

            #[allow(unused_variables)]
            fn apply_bool(a: bool, b: bool) -> bool {
                unimplemented!()
            }

            type S1 = S1;
            type S2 = S2;
            fn base(&self) -> &ScalarOp2Base<S1, S2> {
                &self.0
            }
        }
    };
}

define_op!(Add, Add, add, +);
define_op!(Sub, Sub, sub, -);
define_op!(Mul, Mul, mul, *);
define_op!(Div, Div, div, /);

#[cfg(test)]
mod tests {
    // Bring half::f16 into scope under the name `f16` so the macro ident resolves correctly.
    #[cfg(feature = "half")]
    use crate::dtype::f16;
    #[cfg(feature = "num-complex")]
    type complex_f32 = crate::dtype::Complex<f32>;
    #[cfg(feature = "num-complex")]
    type complex_f64 = crate::dtype::Complex<f64>;

    trait TestVal: Sized {
        fn sample(rng: &mut rand::rngs::StdRng) -> Self;
    }
    macro_rules! impl_test_val {
        ($range:expr, $($t:ty),+) => {
            $(impl TestVal for $t {
                fn sample(rng: &mut rand::rngs::StdRng) -> Self {
                    use rand::RngExt;
                    rng.random_range($range) as Self
                }
            })+
        };
    }
    // [1,4]:  max cube 4³=64  < i8::MAX (127)
    // [1,30]: max cube 30³=27k < i16::MAX (32767)
    // [1,100]: safe for i32/i64/f32/f64
    impl_test_val!(1u8..=4, i8, u8);
    impl_test_val!(1u8..=30, i16, u16, u32, u64);
    impl_test_val!(1u8..=100, i32, i64, f32, f64);
    #[cfg(feature = "half")]
    impl TestVal for f16 {
        fn sample(rng: &mut rand::rngs::StdRng) -> Self {
            use rand::RngExt;
            Self::from_f32(rng.random_range(1u8..=15u8) as f32)
        }
    }
    #[cfg(feature = "num-complex")]
    impl TestVal for complex_f32 {
        fn sample(rng: &mut rand::rngs::StdRng) -> Self {
            use rand::RngExt;
            Self::new(rng.random_range(1u8..=15u8) as f32, 0.0)
        }
    }
    #[cfg(feature = "num-complex")]
    impl TestVal for complex_f64 {
        fn sample(rng: &mut rand::rngs::StdRng) -> Self {
            use rand::RngExt;
            Self::new(rng.random_range(1u8..=15u8) as f64, 0.0)
        }
    }

    fn rand_array<T: TestVal>(rng: &mut rand::rngs::StdRng, shape: &[usize]) -> ndarray::ArrayD<T> {
        ndarray::Array::from_shape_fn(ndarray::IxDyn(shape), |_| T::sample(rng))
    }

    // Generates 5 test functions per (op, dtype).
    // Each TestVal impl controls the sampling range for its type.
    macro_rules! test_op_dtype {
        ($op:tt, $dtype:ident) => {
            paste::paste! {
                #[test]
                fn [<test_ $dtype _1d>]() {
                    use rand::{SeedableRng, rngs::StdRng};
                    use crate::array::Array;
                    let seed = stringify!($dtype).as_bytes().iter().chain(stringify!($op).as_bytes()).fold(0xdeadbeef_cafe1234u64, |acc, b| acc + *b as u64);
                    let mut rng = StdRng::seed_from_u64(seed);
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
                    use rand::{SeedableRng, rngs::StdRng};
                    use crate::array::Array;
                    let seed = stringify!($dtype).as_bytes().iter().chain(stringify!($op).as_bytes()).fold(0xdeadbeef_cafe1234u64, |acc, b| acc + *b as u64);
                    let mut rng = StdRng::seed_from_u64(seed);
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
                    use rand::{SeedableRng, rngs::StdRng};
                    use crate::array::Array;
                    let seed = stringify!($dtype).as_bytes().iter().chain(stringify!($op).as_bytes()).fold(0xdeadbeef_cafe1234u64, |acc, b| acc + *b as u64);
                    let mut rng = StdRng::seed_from_u64(seed);
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
                    use rand::{SeedableRng, rngs::StdRng};
                    use crate::array::Array;
                    let seed = stringify!($dtype).as_bytes().iter().chain(stringify!($op).as_bytes()).fold(0xdeadbeef_cafe1234u64, |acc, b| acc + *b as u64);
                    let mut rng = StdRng::seed_from_u64(seed);
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
                    use rand::{SeedableRng, rngs::StdRng};
                    use crate::array::Array;
                    let seed = stringify!($dtype).as_bytes().iter().chain(stringify!($op).as_bytes()).fold(0xdeadbeef_cafe1234u64, |acc, b| acc + *b as u64);
                    let mut rng = StdRng::seed_from_u64(seed);
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
}
