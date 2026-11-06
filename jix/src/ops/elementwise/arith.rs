use crate::ops::elementwise::op1::{define_array_op1_method, define_op1};
use crate::ops::elementwise::op2::{define_array_op2_method, define_op2};
use crate::ops::prelude::*;

pub(crate) mod traits {
    use crate::error::ArithmeticErrorKind;
    #[cfg(feature = "half")]
    use crate::scalar::f16;
    use crate::scalar::traits_util::{
        define_scalar_op1_trait, define_scalar_op2_trait, impl_scalar_op1, impl_scalar_op2,
    };
    #[cfg(feature = "num-complex")]
    use crate::scalar::Complex;

    define_scalar_op1_trait!(
        /// Sign operation: as [`i32::signum`] for signed integers, `0` or `1` for unsigned
        /// integers, and as `numpy.sign` for floats.
        ///
        /// Supported dtypes: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
        /// `f16`, `f32`, `f64`.
        ///
        /// For **signed integer** types: returns `-1`, `0`, or `+1` of the same type.
        ///
        /// For **unsigned integer** types: returns `0` or `1` of the same type (since
        /// unsigned values cannot be negative).
        ///
        /// For **float** types: returns `+1.0` for positive values, `-1.0` for negative values,
        /// `0.0` for `+0.0` and `-0.0`, and `NaN` for `NaN`, like `numpy.sign`.
        Sign,
        /// Compute the sign of the value.
        sign,
        bulk_fn: sign_bulk,
    );
    impl_scalar_op1!(
        impl Sign for [i8, i16, i32, i64] {
            fn sign = |x| x.signum(),
        }
    );
    impl_scalar_op1!(
        impl Sign for [u8, u16, u32, u64] {
            fn sign = |x| (x != 0) as Self,
        }
    );
    // Unlike `signum`, which maps `+-0.0` to `+-1.0`, zero maps to `0.0` (as in numpy and torch).
    // `NaN` stays `NaN`.
    impl_scalar_op1!(
        impl Sign for [f32, f64] {
            fn sign = |x| if x == 0.0 { 0.0 } else { x.signum() },
        }
    );
    #[cfg(feature = "half")]
    impl_scalar_op1!(
        impl Sign for [f16] {
            fn sign = |x| if x == f16::ZERO { f16::ZERO } else { f16::from_f32(x.to_f32().signum()) },
        }
    );
    define_scalar_op1_trait!(
        /// Absolute value operation: the magnitude for complex types.
        ///
        /// Supported dtypes and output dtype:
        ///
        /// | Input dtype | Output dtype |
        /// |-------------|--------------|
        /// | `i8`, `i16`, `i32`, `i64` | same |
        /// | `f16`, `f32`, `f64` | same |
        /// | `Complex<f32>` | `f32` |
        /// | `Complex<f64>` | `f64` |
        ///
        /// For **complex** types the result is the modulus `sqrt(re^2 + im^2)`, computed
        /// via `hypot` for numerical stability. The output dtype is the real component type
        /// (`f32` for `Complex<f32>`, `f64` for `Complex<f64>`).
        ///
        /// For **signed integer** types `MIN.abs()` wraps back to `MIN`: `(-128i8).abs()` is
        /// `-128`.
        ///
        /// Floating-point semantics follow [`f32::abs`].
        Abs,
        /// Compute the absolute value.
        abs,
        bulk_fn: abs_bulk,
    );
    impl_scalar_op1!(
        impl Abs for [f32, f64] {
            fn abs = |x| x.abs(),
            fn abs_bulk = |x| x.abs(),
        }
    );
    impl_scalar_op1!(
        impl Abs for [i8, i16, i32, i64] {
            fn abs = |x| x.wrapping_abs(),
            fn abs_bulk = |x| x.abs(),
        }
    );
    #[cfg(feature = "half")]
    impl_scalar_op1!(
        impl Abs for [f16] {
            fn abs = |x| f16::from_f32(x.to_f32().abs()),
        }
    );
    #[cfg(feature = "num-complex")]
    impl_scalar_op1!(
        impl Abs for [Complex<f32>] {
            type Output = f32;
            fn abs = |x| x.re.hypot(x.im),
        }
    );
    #[cfg(feature = "num-complex")]
    impl_scalar_op1!(
        impl Abs for [Complex<f64>] {
            type Output = f64;
            fn abs = |x| x.re.hypot(x.im),
        }
    );

    define_scalar_op1_trait!(
        /// Squaring operation: `x * x`.
        ///
        /// The output dtype is the input dtype.
        ///
        /// For **integer** types the result wraps on overflow (two's complement).
        Square,
        /// Compute the square of the value.
        square,
        bulk_fn: square_bulk,
    );
    // SIMD for the types with SIMD mul
    impl_scalar_op1!(
        impl Square for [f32, f64] {
            fn square = |x| x * x,
            fn square_bulk = |x| x * x,
        }
    );
    impl_scalar_op1!(
        impl Square for [i16, i32, i64, u8, u16, u32, u64] {
            fn square = |x| x.wrapping_mul(x),
            fn square_bulk = |x| x * x,
        }
    );
    impl_scalar_op1!(
        impl Square for [i8] {
            fn square = |x| x.wrapping_mul(x),
        }
    );
    // In f32 without FP16
    #[cfg(all(
        feature = "half",
        not(all(target_arch = "aarch64", target_feature = "fp16"))
    ))]
    impl_scalar_op1!(
        impl Square for [f16] {
            fn square = |x| f16::from_f32(x.to_f32() * x.to_f32()),
        }
    );
    #[cfg(all(feature = "half", target_arch = "aarch64", target_feature = "fp16"))]
    impl_scalar_op1!(
        impl Square for [f16] {
            fn square = |x| x * x,
        }
    );
    #[cfg(feature = "num-complex")]
    impl_scalar_op1!(
        impl Square for [Complex<f32>, Complex<f64>] {
            fn square = |x| x * x,
        }
    );
    #[cfg(all(feature = "half", feature = "num-complex"))]
    impl_scalar_op1!(
        impl Square for [Complex<f16>] {
            fn square = |x| x * x,
        }
    );

    define_scalar_op1_trait!(
        /// Negation operation, like [`core::ops::Neg`].
        ///
        /// For **integer** types the result is the two's-complement negation, which wraps:
        /// `-i32::MIN` is `i32::MIN`.
        ///
        /// For **complex** types both components are negated independently:
        /// `-(a + bi) = -a - bi`.
        ///
        /// Floating-point semantics follow `f32::neg`.
        Neg,
        /// Negate the value.
        neg,
        bulk_fn: neg_bulk,
    );
    impl_scalar_op1!(
        impl Neg for [f32, f64] {
            fn neg = |x| -x,
            fn neg_bulk = |x| -x,
        }
    );
    impl_scalar_op1!(
        impl Neg for [i8, i16, i32, i64] {
            fn neg = |x| x.wrapping_neg(),
            fn neg_bulk = |x| -x,
        }
    );
    #[cfg(feature = "half")]
    impl_scalar_op1!(
        impl Neg for [f16] {
            fn neg = |x| -x,
        }
    );
    #[cfg(feature = "num-complex")]
    impl_scalar_op1!(
        impl Neg for [Complex<f32>, Complex<f64>] {
            fn neg = |x| -x,
        }
    );
    #[cfg(all(feature = "half", feature = "num-complex"))]
    impl_scalar_op1!(
        impl Neg for [Complex<f16>] {
            fn neg = |x| -x,
        }
    );

    define_scalar_op2_trait!(
        /// Addition operation, like [`core::ops::Add`].
        ///
        /// For **integer** types the result wraps on overflow (two's complement).
        /// For **complex** types each component is added independently:
        /// `(a + bi) + (c + di) = (a+c) + (b+d)i`.
        Add,
        /// Add another value.
        add,
        bulk_fn: add_bulk,
    );
    define_scalar_op2_trait!(
        /// Subtraction operation, like [`core::ops::Sub`].
        ///
        /// For **integer** types the result wraps on underflow (two's complement).
        /// For **complex** types each component is subtracted independently:
        /// `(a + bi) - (c + di) = (a-c) + (b-d)i`.
        Sub,
        /// Subtract another value.
        sub,
        bulk_fn: sub_bulk,
    );
    define_scalar_op2_trait!(
        /// Multiplication operation, like [`core::ops::Mul`].
        ///
        /// For **integer** types the result wraps on overflow (two's complement).
        /// For **complex** types this is full complex multiplication:
        /// `(a + bi) * (c + di) = (ac - bd) + (ad + bc)i`.
        Mul,
        /// Multiply with another value.
        mul,
        bulk_fn: mul_bulk,
    );

    macro_rules! impl_arith {
        (
            $Trait:ident::$f:ident, $op:tt, $wrapping_f:ident, [$($scalar:ty),*],
            simd: $f_bulk:ident, [$($simd:ty),*]
        ) => {
            impl_scalar_op2!(
                impl $Trait for [f32, f64] {
                    fn $f = |a, b| a $op b,
                    fn $f_bulk = |a, b| a $op b,
                }
            );
            impl_scalar_op2!(
                impl $Trait for [$($simd),*] {
                    fn $f = |a, b| a.$wrapping_f(b),
                    fn $f_bulk = |a, b| a $op b,
                }
            );
            impl_scalar_op2!(
                impl $Trait for [$($scalar),*] {
                    fn $f = |a, b| a.$wrapping_f(b),
                }
            );
            // In f32 without FP16, as half's operators do, but inlined: theirs call a software
            // fallback function.
            #[cfg(all(feature = "half", not(all(target_arch = "aarch64", target_feature = "fp16"))))]
            impl_scalar_op2!(
                impl $Trait for [f16] {
                    fn $f = |a, b| f16::from_f32(a.to_f32() $op b.to_f32()),
                }
            );
            #[cfg(all(feature = "half", target_arch = "aarch64", target_feature = "fp16"))]
            impl_scalar_op2!(
                impl $Trait for [f16] {
                    fn $f = |a, b| a $op b,
                }
            );
            #[cfg(feature = "num-complex")]
            impl_scalar_op2!(
                impl $Trait for [Complex<f32>, Complex<f64>] {
                    fn $f = |a, b| a $op b,
                }
            );
            #[cfg(all(feature = "half", feature = "num-complex"))]
            impl_scalar_op2!(
                impl $Trait for [Complex<f16>] {
                    fn $f = |a, b| a $op b,
                }
            );
        };
    }
    impl_arith!(
        Add::add, +, wrapping_add, [],
        simd: add_bulk, [i8, i16, i32, i64, u8, u16, u32, u64]
    );
    impl_arith!(
        Sub::sub, -, wrapping_sub, [],
        simd: sub_bulk, [i8, i16, i32, i64, u8, u16, u32, u64]
    );
    impl_arith!(
        Mul::mul, *, wrapping_mul, [i8], // i8 auto-vectorized scalar is faster
        simd: mul_bulk, [i16, i32, i64, u8, u16, u32, u64]
    );
    define_scalar_op2_trait!(
        /// Exponentiation operation: as [`i32::wrapping_pow`] for integers, [`f32::powf`] for
        /// floats.
        ///
        /// Integer results wrap on overflow. A negative exponent of an integer base gives `0`
        /// and reports an arithmetic error to `status`.
        ///
        /// Float values follow [`f32::powf`] (`+-inf` and `NaN` included), and no error is
        /// reported.
        Pow,
        /// Raise to the power of another value, reporting an arithmetic error to `status`.
        pow(status),
        bulk_fn: pow_bulk,
    );

    impl_scalar_op2!(
        impl Pow for [i8, i16, i32, i64, u8, u16, u32, u64] x [u8, u16, u32] {
            fn pow = |a, b, _status| a.wrapping_pow(b as u32),
        }
    );
    impl_scalar_op2!(
        impl Pow for [i8, i16, i32, i64, u8, u16, u32, u64] x [i8, i16, i32, i64] {
            fn pow = |a, b, status| {
                status.set_if(ArithmeticErrorKind::NegativePower, b < 0);
                if let Ok(b) = u32::try_from(b) {
                    a.wrapping_pow(b)
                } else if b < 0 {
                    0
                } else {
                    let (mut acc, mut base, mut exp): (Self, Self, u64) = (1, a, b as u64);
                    while exp > 0 {
                        if exp & 1 == 1 {
                            acc = acc.wrapping_mul(base);
                        }
                        base = base.wrapping_mul(base);
                        exp >>= 1;
                    }
                    acc
                }
            },
        }
    );
    impl_scalar_op2!(
        impl Pow for [f32, f64] x [i8, u8, i16, u16, i32] {
            fn pow = |a, b, _status| match b {
                0 => 1.0,
                1 => a,
                2 => a * a,
                // compute a**i32 in f64
                _ if size_of_val(&b) >= 4 => (a as f64).powf(b as f64) as Self,
                _ => a.powf(b as Self),
            },
        }
    );
    impl_scalar_op2!(
        impl Pow for [f32, f64] x [f32] {
            fn pow = |a, b, _status| a.powf(b as Self),
        }
    );
    impl_scalar_op2!(
        impl Pow for [f64] x [f64] {
            fn pow = |a, b, _status| a.powf(b),
        }
    );
    #[cfg(feature = "half")]
    impl_scalar_op2!(
        impl Pow for [f16] x [i8, u8, i16, u16, i32] {
            fn pow = |a, b, _status| {
                let a = a.to_f32();
                let p = match b {
                    0 => 1.0,
                    1 => a,
                    2 => a * a,
                    // compute a**i32 in f64
                    _ if size_of_val(&b) >= 4 => (a as f64).powf(b as f64) as f32,
                    _ => a.powf(b as f32),
                };
                f16::from_f32(p)
            },
        }
    );
    #[cfg(feature = "half")]
    impl_scalar_op2!(
        impl Pow for [f16] {
            fn pow = |a, b, _status| f16::from_f32(a.to_f32().powf(b.to_f32())),
        }
    );
}

define_op1!(
    /// Arithmetic negation applied element-wise.
    ///
    /// See [`jix::scalar::Neg`](crate::scalar::Neg) scalar trait for the per-element semantics.
    ///
    /// Available via the unary `-` operator on [`Array`](crate::Array): `-arr`.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::neg()`](core::ops::Neg::neg).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![1.0f32, -2.5, 3.0])?;
    /// let result = (-a).to_ndarray()?;
    /// assert_eq!(result, array![-1.0, 2.5, -3.0]);
    ///
    /// // Integer negation wraps (two's complement overflow): -i8::MIN is i8::MIN.
    /// let b = Array::compact_ndarray(&array![0i8, 1, -1, i8::MIN])?;
    /// let result = (-b).to_ndarray()?;
    /// assert_eq!(result, array![0, -1, 1, i8::MIN]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Neg,
    NegKernel,
    crate::scalar::Neg::neg,
    simd: neg_bulk,
    core_op: core::ops::Neg::neg,
);
define_op1!(
    /// Returns the sign of each element.
    ///
    /// See [`jix::scalar::Sign`](crate::scalar::Sign) scalar trait for the per-element semantics.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::sign()`](crate::Array::sign).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::{array, Ix1};
    ///
    /// let a = Array::compact_ndarray(&array![3i32, -5, 0])?;
    /// let result = a.sign().to_ndarray()?;
    /// assert_eq!(result, array![1, -1, 0]);
    ///
    /// let b = Array::compact_ndarray(&array![3.0f32, -5.0, -0.1])?;
    /// let result = b.sign().to_ndarray()?;
    /// assert_eq!(result, array![1.0, -1.0, -1.0]);
    ///
    /// // Float: both zeros return 0.0.
    /// let c = Array::compact_ndarray(&array![0.0f32, -0.0])?;
    /// let result = c.sign().to_ndarray()?;
    /// assert_eq!(result, array![0.0, 0.0]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Sign,
    SignKernel,
    crate::scalar::Sign::sign,
    simd: sign_bulk,
);
define_op1!(
    /// Computes the absolute value of each element.
    ///
    /// See [`jix::scalar::Abs`](crate::scalar::Abs) scalar trait for the per-element semantics.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::abs()`](crate::Array::abs).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![-3i32, 0, 5, -7])?;
    /// let result = a.abs().to_ndarray()?;
    /// assert_eq!(result, array![3, 0, 5, 7]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    ///
    /// ```
    /// # #[cfg(feature = "num-complex")]
    /// # {
    /// use jix::Array;
    /// use ndarray::{array, Ix1};
    ///
    /// // For complex input the result is the modulus sqrt(re^2 + im^2).
    /// use jix::scalar::Complex;
    /// let b = Array::compact_ndarray(&array![Complex { re: 3.0f32, im: 4.0 }])?;
    /// let result = b.abs().to_ndarray::<Ix1>()?;
    /// assert!((result[[0]] - 5.0).abs() < 1e-5);
    /// # }
    /// # Ok::<(), jix::Error>(())
    /// ```
    Abs,
    AbsKernel,
    crate::scalar::Abs::abs,
    simd: abs_bulk,
);

define_op1!(
    /// Squares each element (`x * x`).
    ///
    /// See [`jix::scalar::Square`](crate::scalar::Square) scalar trait for the
    /// per-element semantics.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::square()`](crate::Array::square).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![1.0f32, -2.0, 3.0])?;
    /// let result = a.square().to_ndarray()?;
    /// assert_eq!(result, array![1.0, 4.0, 9.0]);
    ///
    /// // Works on integer types too.
    /// let b = Array::compact_ndarray(&array![2i32, -3, 4])?;
    /// let result = b.square().to_ndarray()?;
    /// assert_eq!(result, array![4, 9, 16]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Square,
    SquareKernel,
    crate::scalar::Square::square,
    simd: square_bulk,
);
define_op2!(
    /// Element-wise addition of two arrays.
    ///
    /// See [`jix::scalar::Add`](crate::scalar::Add) scalar trait for the per-element semantics.
    ///
    /// Available via the `+` operator on arrays. For adding a constant to every element, use
    /// [`Array::map`]: `a.map(|x| x + 1i32)`.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![1i32, 2, 3])?;
    /// let b = Array::compact_ndarray(&array![10i32, 20, 30])?;
    /// let result = (a + b).to_ndarray()?;
    /// assert_eq!(result, array![11, 22, 33]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Add,
    AddKernel,
    crate::scalar::Add::add,
    simd: add_bulk,
    core_op: core::ops::Add::add,
);
define_op2!(
    /// Element-wise subtraction of two arrays (`a - b`).
    ///
    /// See [`jix::scalar::Sub`](crate::scalar::Sub) scalar trait for the per-element semantics.
    ///
    /// Available via the `-` operator on arrays. For subtracting a constant from every element,
    /// use [`Array::map`]: `a.map(|x| x - 1i32)`.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![10i32, 20, 30])?;
    /// let b = Array::compact_ndarray(&array![1i32, 2, 3])?;
    /// let result = (a - b).to_ndarray()?;
    /// assert_eq!(result, array![9, 18, 27]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Sub,
    SubKernel,
    crate::scalar::Sub::sub,
    simd: sub_bulk,
    core_op: core::ops::Sub::sub,
);
define_op2!(
    /// Element-wise multiplication of two arrays.
    ///
    /// See [`jix::scalar::Mul`](crate::scalar::Mul) scalar trait for the per-element semantics.
    ///
    /// Available via the `*` operator on arrays. For scaling every element by a constant, use
    /// [`Array::map`]: `a.map(|x| x * 2i32)`.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![1i32, 2, 3])?;
    /// let b = Array::compact_ndarray(&array![4i32, 5, 6])?;
    /// let result = (a * b).to_ndarray()?;
    /// assert_eq!(result, array![4, 10, 18]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Mul,
    MulKernel,
    crate::scalar::Mul::mul,
    simd: mul_bulk,
    core_op: core::ops::Mul::mul,
);

define_op2!(
    /// Element-wise exponentiation (`a` raised to the power `b`).
    ///
    /// See [`jix::scalar::Pow`](crate::scalar::Pow) scalar trait for the per-element semantics.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::pow()`](crate::Array::pow).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![2.0f32, 3.0, 4.0])?;
    /// let b = Array::compact_ndarray(&array![3.0f32, 2.0, 0.5])?;
    /// let result = a.pow(b).to_ndarray()?;
    /// assert_eq!(result, array![8.0, 9.0, 2.0]);
    ///
    /// // Raise each element to a constant exponent.
    /// let a = Array::compact_ndarray(&array![2.0f32, 3.0, 4.0])?;
    /// let result = a.map(|x| x.powf(2.0)).to_ndarray()?;
    /// assert_eq!(result, array![4.0, 9.0, 16.0]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Pow,
    PowKernel,
    crate::scalar::Pow::pow(status),
    simd: pow_bulk,
);

impl<S> Array<S>
where
    S: ArrayStorage,
{
    define_array_op1_method!(square: Square, crate::scalar::Square);
    define_array_op1_method!(sign: Sign, crate::scalar::Sign);
    define_array_op1_method!(abs: Abs, crate::scalar::Abs);
    define_array_op2_method!(pow: Pow, crate::scalar::Pow);
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
mod tests {
    #[cfg(feature = "half")]
    use crate::scalar::f16;
    #[cfg(feature = "num-complex")]
    #[allow(non_camel_case_types)]
    type complex_f32 = crate::scalar::Complex<f32>;
    #[cfg(feature = "num-complex")]
    #[allow(non_camel_case_types)]
    type complex_f64 = crate::scalar::Complex<f64>;

    use ndarray::Ix1;

    use crate::ops::elementwise::op1::tests::test_op1;
    use crate::ops::elementwise::op2::tests::test_op2;
    #[cfg(feature = "half")]
    use crate::ops::elementwise::op2::tests::test_op2_dtype;

    /// The `apply_bulk` of the kernels of this module (their scalar traits' SIMD `*_bulk`) against
    /// their scalar semantics (release builds: wrapping), on every SIMD level of the CPU, over edge
    /// cases.
    #[test]
    fn simd_bodies_all_levels() {
        use super::{AbsKernel, NegKernel, SquareKernel};
        use crate::ops::elementwise::op1::Op1Kernel;
        use crate::util::{assert_same_elements, for_each_simd_level, SimdTestValues};
        use fearless_simd::Simd;

        fn check<S: Simd>(simd: S) {
            macro_rules! case {
                ($kernel:ident, $t:ty, $f:expr) => {
                    let xs = <$t>::simd_test_values(0);
                    let what = concat!(stringify!($kernel), " ", stringify!($t));
                    assert_same_elements(
                        $kernel.apply_bulk(xs, &mut Default::default(), simd),
                        |i| $f(xs[i]),
                        what,
                    );
                };
            }
            simd.vectorize(|| {
                case!(NegKernel, f32, |x: f32| -x);
                case!(NegKernel, f64, |x: f64| -x);
                case!(NegKernel, i8, i8::wrapping_neg);
                case!(NegKernel, i16, i16::wrapping_neg);
                case!(NegKernel, i32, i32::wrapping_neg);
                case!(NegKernel, i64, i64::wrapping_neg);
                case!(AbsKernel, f32, f32::abs);
                case!(AbsKernel, f64, f64::abs);
                case!(AbsKernel, i8, i8::wrapping_abs);
                case!(AbsKernel, i16, i16::wrapping_abs);
                case!(AbsKernel, i32, i32::wrapping_abs);
                case!(AbsKernel, i64, i64::wrapping_abs);
                case!(SquareKernel, f32, |x: f32| x * x);
                case!(SquareKernel, f64, |x: f64| x * x);
                case!(SquareKernel, i16, |x: i16| x.wrapping_mul(x));
                case!(SquareKernel, i32, |x: i32| x.wrapping_mul(x));
                case!(SquareKernel, i64, |x: i64| x.wrapping_mul(x));
                case!(SquareKernel, u8, |x: u8| x.wrapping_mul(x));
                case!(SquareKernel, u16, |x: u16| x.wrapping_mul(x));
                case!(SquareKernel, u32, |x: u32| x.wrapping_mul(x));
                case!(SquareKernel, u64, |x: u64| x.wrapping_mul(x));
                // Not a multiple of the vector length, and a type without a SIMD body: scalar.
                assert_eq!(
                    NegKernel.apply_bulk([1.5f32], &mut Default::default(), simd),
                    [-1.5]
                );
            });
        }
        for_each_simd_level!(check);
    }

    test_op1!(neg, |a| a.wrapping_neg(), [i8, i16, i32, i64], any_strategy);
    test_op1!(abs, |a| a.wrapping_abs(), [i8, i16, i32, i64], any_strategy);
    test_op1!(
        neg,
        |a| -a,
        [f32, f64],
        op_safe_strategy,
        #[cfg(feature = "half")]
        [f16],
        #[cfg(feature = "num-complex")]
        [complex_f32, complex_f64]
    );
    test_op1!(
        square,
        |a| a.wrapping_mul(a),
        [i8, i16, i32, i64, u8, u16, u32, u64],
        any_strategy
    );
    #[test]
    fn square_concrete() {
        use crate::Array;
        // i32: negative, zero, positive, and values at the op_safe_strategy bound (+/-100)
        // so the squared result (10000) still fits comfortably in i32.
        let nd = ndarray::array![[-100i32, -1, 0], [1, 5, 100]];
        let za = Array::compact_ndarray(&nd).unwrap();
        let expected = nd.mapv(|a: i32| a * a);
        crate::util::assert_array_matches(&za.view().square(), &expected);

        // f32/f64: same edge values (negative, zero, positive, near the +/-100.0 bound), plus
        // a non-default block shape on the f64 arm to cross a block boundary.
        let ndf = ndarray::array![[-100.0f32, -0.5, 0.0], [0.5, 5.0, 100.0]];
        let zaf = Array::compact_ndarray(&ndf).unwrap();
        let expectedf = ndf.mapv(|a: f32| a * a);
        crate::util::assert_array_matches(&zaf.view().square(), &expectedf);

        let ndd = ndarray::array![[-100.0f64, -0.5, 0.0], [0.5, 5.0, 100.0]];
        let zad = Array::compact_ndarray_with(&ndd, crate::util::arr_params(&[1, 2])).unwrap();
        let expectedd = ndd.mapv(|a: f64| a * a);
        crate::util::assert_array_matches(&zad.view().square(), &expectedd);
    }
    #[test]
    fn sign_concrete() {
        use crate::Array;
        let nd = ndarray::array![-5i32, 0, 7];
        let za = Array::compact_ndarray(&nd).unwrap();
        let expected = nd.mapv(|a: i32| a.signum());
        crate::util::assert_array_matches(&za.view().sign(), &expected);

        // Unsigned int: zero maps to 0, any positive value maps to 1 (`a - a` is 0,
        // `a - a + 1` is 1).
        let ndu = ndarray::array![0u32, 1, 7];
        let zau = Array::compact_ndarray(&ndu).unwrap();
        let expectedu = ndu.mapv(|a: u32| if a == 0 { a - a } else { a - a + 1 });
        crate::util::assert_array_matches(&zau.view().sign(), &expectedu);

        // f32: both zeros map to +0.0 (not signum's +-1.0), and NaN stays NaN.
        let ndf = ndarray::array![-3.0f32, -0.0, 0.0, 5.0];
        let zaf = Array::compact_ndarray(&ndf).unwrap();
        let resf = zaf.view().sign().to_ndarray::<ndarray::Ix1>().unwrap();
        assert_eq!(resf.as_slice().unwrap(), &[-1.0, 0.0, 0.0, 1.0]);
        assert!(resf.iter().all(|x| !x.is_sign_negative() || *x == -1.0));
        let nan = Array::compact_ndarray(&ndarray::array![f32::NAN]).unwrap();
        assert!(nan.sign().to_ndarray::<ndarray::Ix1>().unwrap()[0].is_nan());

        // f64: same, plus a non-default block shape.
        let ndd = ndarray::array![-3.0f64, -0.0, 0.0, 5.0];
        let zad = Array::compact_ndarray_with(&ndd, crate::util::arr_params(&[2])).unwrap();
        let resd = zad.view().sign().to_ndarray::<ndarray::Ix1>().unwrap();
        assert_eq!(resd.as_slice().unwrap(), &[-1.0, 0.0, 0.0, 1.0]);
    }
    // abs: same dtype for scalar types; complex types have a different output dtype (see below).
    #[test]
    fn abs_concrete() {
        use crate::Array;
        // i32: negative, zero, positive, and values at the op_safe_strategy bound (+/-100) -
        // within range so no MIN-overflow wraparound occurs.
        let nd = ndarray::array![-5i32, 0, 7, -100, 100];
        let za = Array::compact_ndarray(&nd).unwrap();
        let expected = nd.mapv(|a: i32| a.abs());
        crate::util::assert_array_matches(&za.view().abs(), &expected);

        let ndf = ndarray::array![-5.0f32, 0.0, 7.5, -100.0];
        let zaf = Array::compact_ndarray(&ndf).unwrap();
        let expectedf = ndf.mapv(|a: f32| a.abs());
        crate::util::assert_array_matches(&zaf.view().abs(), &expectedf);

        let ndd = ndarray::array![-5.0f64, 0.0, 7.5, -100.0];
        let zad = Array::compact_ndarray_with(&ndd, crate::util::arr_params(&[2])).unwrap();
        let expectedd = ndd.mapv(|a: f64| a.abs());
        crate::util::assert_array_matches(&zad.view().abs(), &expectedd);
    }
    // TODO
    // #[cfg(feature = "half")]
    // [f16]

    #[cfg(feature = "num-complex")]
    mod complex {
        use super::{complex_f32, complex_f64};
        use crate::ops::elementwise::op1::tests::test_op1_dtype;

        // abs on complex types: output dtype is the real component type, not the input dtype.
        // Reference uses hypot to match the Abs kernel exactly.
        test_op1_dtype!(abs, |a| a.re.hypot(a.im), complex_f32, op_safe_strategy);
        test_op1_dtype!(abs, |a| a.re.hypot(a.im), complex_f64, op_safe_strategy);
    }

    /// The `apply_bulk` of the kernels of this module (their scalar traits' SIMD `*_bulk`) against
    /// their scalar semantics (release builds: wrapping), on every SIMD level of the CPU, over edge
    /// cases.
    #[test]
    fn simd_bodies_all_levels_binary() {
        use super::{AddKernel, MulKernel, SubKernel};
        use crate::ops::elementwise::op2::Op2Kernel;
        use crate::util::{assert_same_elements, for_each_simd_level, SimdTestValues};
        use fearless_simd::Simd;

        fn check<S: Simd>(simd: S) {
            macro_rules! case {
                ($kernel:ident, $t:ty, $f:expr) => {
                    let (a, b) = (<$t>::simd_test_values(0), <$t>::simd_test_values(5));
                    let what = concat!(stringify!($kernel), " ", stringify!($t));
                    assert_same_elements(
                        $kernel.apply_bulk(a, b, &mut Default::default(), simd),
                        |i| $f(a[i], b[i]),
                        what,
                    );
                };
            }
            macro_rules! ints {
                ($kernel:ident, $f:ident) => {
                    case!($kernel, i8, i8::$f);
                    case!($kernel, i16, i16::$f);
                    case!($kernel, i32, i32::$f);
                    case!($kernel, i64, i64::$f);
                    case!($kernel, u8, u8::$f);
                    case!($kernel, u16, u16::$f);
                    case!($kernel, u32, u32::$f);
                    case!($kernel, u64, u64::$f);
                };
            }
            simd.vectorize(|| {
                case!(AddKernel, f32, |x, y| x + y);
                case!(AddKernel, f64, |x, y| x + y);
                ints!(AddKernel, wrapping_add);
                case!(SubKernel, f32, |x, y| x - y);
                case!(SubKernel, f64, |x, y| x - y);
                ints!(SubKernel, wrapping_sub);
                case!(MulKernel, f32, |x, y| x * y);
                case!(MulKernel, f64, |x, y| x * y);
                case!(MulKernel, i16, i16::wrapping_mul);
                case!(MulKernel, i32, i32::wrapping_mul);
                case!(MulKernel, i64, i64::wrapping_mul);
                case!(MulKernel, u8, u8::wrapping_mul);
                case!(MulKernel, u16, u16::wrapping_mul);
                case!(MulKernel, u32, u32::wrapping_mul);
                case!(MulKernel, u64, u64::wrapping_mul);
                // Not a multiple of the vector length: scalar.
                assert_eq!(
                    AddKernel.apply_bulk([1.5f32], [2.0], &mut Default::default(), simd),
                    [3.5]
                );
            });
        }
        for_each_simd_level!(check);
    }

    test_op2!(
        add,
        |a, b| a.wrapping_add(b),
        [i8, i16, i32, i64, u8, u16, u32, u64],
        any_strategy
    );
    test_op2!(
        sub,
        |a, b| a.wrapping_sub(b),
        [i8, i16, i32, i64, u8, u16, u32, u64],
        any_strategy
    );
    test_op2!(
        mul,
        |a, b| a.wrapping_mul(b),
        [i8, i16, i32, i64, u8, u16, u32, u64],
        any_strategy
    );
    test_op2!(
        add,
        |a, b| a + b,
        [f32, f64],
        op_safe_strategy,
        #[cfg(feature = "half")]
        [f16],
        #[cfg(feature = "num-complex")]
        [complex_f32, complex_f64]
    );

    #[test]
    fn sub_concrete() {
        use crate::Array;

        // i32: negatives, zero, positive, at the op_safe_strategy bound (+/-100); differences
        // stay well within i32 range so there is no overflow panic in the reference either.
        let nd_a = ndarray::array![[-100i32, -1, 0], [1, 50, 100]];
        let nd_b = ndarray::array![[100i32, 1, 0], [-1, -50, -100]];
        let za = Array::compact_ndarray(&nd_a).unwrap();
        let zb = Array::compact_ndarray(&nd_b).unwrap();
        let expected = ndarray::Zip::from(&nd_a)
            .and(&nd_b)
            .map_collect(|&a, &b| a - b);
        crate::util::assert_array_matches(&(za - zb), &expected);

        // f64: negatives, zero, fractions, and the +/-100.0 bound, with a non-default block
        // shape so this arm crosses a block boundary.
        let nd_a64 = ndarray::array![[-100.0f64, -0.5, 0.0], [0.5, 50.0, 100.0]];
        let nd_b64 = ndarray::array![[100.0f64, 0.5, 0.0], [-0.5, -50.0, -100.0]];
        let za64 = Array::compact_ndarray_with(&nd_a64, crate::util::arr_params(&[1, 2])).unwrap();
        let zb64 = Array::compact_ndarray_with(&nd_b64, crate::util::arr_params(&[1, 2])).unwrap();
        let expected64 = ndarray::Zip::from(&nd_a64)
            .and(&nd_b64)
            .map_collect(|&a, &b| a - b);
        crate::util::assert_array_matches(&(za64 - zb64), &expected64);
    }

    #[test]
    fn mul_concrete() {
        use crate::Array;

        // i32: negatives, zero, positive, at the op_safe_strategy bound (+/-100); products
        // stay well within i32 range so there is no overflow panic in the reference either.
        let nd_a = ndarray::array![[-100i32, -1, 0], [1, 10, 100]];
        let nd_b = ndarray::array![[100i32, -1, 0], [-1, 10, -100]];
        let za = Array::compact_ndarray(&nd_a).unwrap();
        let zb = Array::compact_ndarray(&nd_b).unwrap();
        let expected = ndarray::Zip::from(&nd_a)
            .and(&nd_b)
            .map_collect(|&a, &b| a * b);
        crate::util::assert_array_matches(&(za * zb), &expected);

        // f64: negatives, zero, fractions, and the +/-100.0 bound, with a non-default block
        // shape so this arm crosses a block boundary.
        let nd_a64 = ndarray::array![[-100.0f64, -0.5, 0.0], [0.5, 10.0, 100.0]];
        let nd_b64 = ndarray::array![[100.0f64, -0.5, 0.0], [-0.5, 10.0, -100.0]];
        let za64 = Array::compact_ndarray_with(&nd_a64, crate::util::arr_params(&[1, 2])).unwrap();
        let zb64 = Array::compact_ndarray_with(&nd_b64, crate::util::arr_params(&[1, 2])).unwrap();
        let expected64 = ndarray::Zip::from(&nd_a64)
            .and(&nd_b64)
            .map_collect(|&a, &b| a * b);
        crate::util::assert_array_matches(&(za64 * zb64), &expected64);
    }

    // pow, f16 arm: computed in f32 and rounded back to f16. Non-negative bases only, a negative
    // base with a non-integer exponent is NaN, which never compares equal.
    #[cfg(feature = "half")]
    test_op2_dtype!(
        pow,
        |a, b| f16::from_f32(a.to_f32().powf(b.to_f32())),
        f16,
        op_safe_non_negative_strategy
    );

    #[cfg(feature = "half")]
    #[test]
    fn pow_f16_concrete() {
        use crate::Array;

        let f16s = |xs: &[f32]| ndarray::Array1::from_iter(xs.iter().map(|&x| f16::from_f32(x)));

        // Integer exponents, including negative ones and a result past the f16 max (65504).
        let za = Array::compact_ndarray(&f16s(&[0.5, 2.0, -3.0, 10.0])).unwrap();
        let zb = Array::compact_ndarray(&ndarray::array![3i32, -2, 3, 5]).unwrap();
        let expected = f16s(&[0.125, 0.25, -27.0, f32::INFINITY]);
        crate::util::assert_array_matches(&za.pow(zb), &expected);

        let za = Array::compact_ndarray(&f16s(&[1.5, 2.0])).unwrap();
        let zb = Array::compact_ndarray(&ndarray::array![2u8, 4]).unwrap();
        crate::util::assert_array_matches(&za.pow(zb), &f16s(&[2.25, 16.0]));

        // Float exponents.
        let za = Array::compact_ndarray(&f16s(&[4.0, 2.0, 0.0, 8.0])).unwrap();
        let zb = Array::compact_ndarray(&f16s(&[0.5, 10.0, 0.0, -1.0])).unwrap();
        crate::util::assert_array_matches(&za.pow(zb), &f16s(&[2.0, 1024.0, 1.0, 0.125]));
    }

    #[test]
    fn pow_float_by_int_matches_powf() {
        use crate::Array;

        // powi would flush 2^-1074 to 0 (2^1074 overflows) and drift for large exponents.
        let za = Array::compact_ndarray(&ndarray::array![2.0f64, 1.0001, 1.5, 3.0]).unwrap();
        let zb = Array::compact_ndarray(&ndarray::array![-1074i32, 10000, 100, 2]).unwrap();
        let expected = ndarray::array![
            2.0f64.powf(-1074.0),
            1.0001f64.powf(10000.0),
            1.5f64.powf(100.0),
            9.0
        ];
        assert_eq!(za.pow(zb).to_ndarray::<ndarray::Ix1>().unwrap(), expected);
        assert!(expected[0] > 0.0);

        // Exponents 0..=2 take the multiplication shortcut, 3 goes through powf: a * a * a in f32
        // would overflow on the first value, powf gives the rounded exact cube.
        let a = ndarray::array![6.9814636e12f32, -1.5, 0.1, -1.5, 0.0];
        let b = ndarray::array![3i32, 3, 2, 1, 0];
        let expected = ndarray::Zip::from(&a)
            .and(&b)
            .map_collect(|&a, &b| a.powf(b as f32));
        let za = Array::compact_ndarray(&a).unwrap();
        let zb = Array::compact_ndarray(&b).unwrap();
        assert_eq!(za.pow(zb).to_ndarray::<ndarray::Ix1>().unwrap(), expected);
        assert!(expected[0].is_finite());
    }

    // mul, complex arm: full complex multiplication `(a+bi)*(c+di) = (ac-bd) + (ad+bc)i`,
    // with negative and zero components on both operands.
    #[cfg(feature = "num-complex")]
    #[test]
    fn mul_concrete_complex() {
        use crate::scalar::Complex;
        use crate::Array;

        let nd_a = ndarray::array![
            [
                Complex {
                    re: -3.0f32,
                    im: 4.0
                },
                Complex { re: 0.0, im: 0.0 }
            ],
            [
                Complex { re: 2.5, im: -1.5 },
                Complex {
                    re: 100.0,
                    im: -100.0
                }
            ],
        ];
        let nd_b = ndarray::array![
            [
                Complex {
                    re: 1.0f32,
                    im: -2.0
                },
                Complex { re: 5.0, im: 5.0 }
            ],
            [
                Complex { re: -2.5, im: 1.5 },
                Complex {
                    re: -100.0,
                    im: 100.0
                }
            ],
        ];
        let za = Array::compact_ndarray(&nd_a).unwrap();
        let zb = Array::compact_ndarray(&nd_b).unwrap();
        let expected = ndarray::Zip::from(&nd_a)
            .and(&nd_b)
            .map_collect(|&a, &b| a * b);
        crate::util::assert_array_matches(&(za * zb), &expected);
    }

    // three_arrays_f32: (za + zb) * zc lazy chain equals (&a + &b) * &c.
    // Fixed 2*3 shape with two different block shapes to exercise block-boundary handling.
    proptest::proptest! {
        #[test]
        fn three_arrays_f32(
            a_vals in proptest::collection::vec(<f32 as crate::util::ScalarStrategy>::op_safe_strategy(), 6usize),
            b_vals in proptest::collection::vec(<f32 as crate::util::ScalarStrategy>::op_safe_strategy(), 6usize),
            c_vals in proptest::collection::vec(<f32 as crate::util::ScalarStrategy>::op_safe_strategy(), 6usize),
        ) {
            use crate::array::Array;
            let nd_a = ndarray::Array::from_shape_vec([2, 3], a_vals).unwrap();
            let nd_b = ndarray::Array::from_shape_vec([2, 3], b_vals).unwrap();
            let nd_c = ndarray::Array::from_shape_vec([2, 3], c_vals).unwrap();
            let za = Array::compact_ndarray_with(&nd_a, crate::util::arr_params(&[2, 3])).unwrap();
            let zb = Array::compact_ndarray_with(&nd_b, crate::util::arr_params(&[2, 3])).unwrap();
            let zc = Array::compact_ndarray_with(&nd_c, crate::util::arr_params(&[1, 2])).unwrap();
            let result = (za + zb) * zc;
            let expected = (&nd_a + &nd_b) * &nd_c;
            crate::util::assert_array_matches(&result, &expected);
        }
    }

    #[test]
    fn int_pow() {
        use crate::error::{ArithmeticErrorKind, ArithmeticStatus};
        use crate::scalar::Pow;

        let mut status = ArithmeticStatus::default();
        // Wraps on overflow.
        assert_eq!(Pow::pow(3i32, 40u32, &mut status), 689956897);
        assert_eq!(Pow::pow(3u8, 9u8, &mut status), 227);
        assert_eq!(Pow::pow(-2i8, 7i8, &mut status), -128);
        assert_eq!(Pow::pow(-3i64, 0i64, &mut status), 1);
        assert_eq!(Pow::pow(0u16, 0i16, &mut status), 1);
        // Exponents beyond `u32`.
        assert_eq!(Pow::pow(1i32, i64::MAX, &mut status), 1);
        assert_eq!(Pow::pow(-1i32, (1i64 << 40) + 1, &mut status), -1);
        assert_eq!(Pow::pow(2u64, 1i64 << 33, &mut status), 0);
        let mut squared = 3u32;
        for _ in 0..33 {
            squared = squared.wrapping_mul(squared);
        }
        assert_eq!(Pow::pow(3u32, 1i64 << 33, &mut status), squared);
        assert!(status.is_ok());

        // A negative exponent: `0`, and an error.
        assert_eq!(Pow::pow(2i32, -1i32, &mut status), 0);
        assert!(status.contains(ArithmeticErrorKind::NegativePower));
        let mut status = ArithmeticStatus::default();
        assert_eq!(Pow::pow(1u8, i64::MIN, &mut status), 0);
        assert!(status.contains(ArithmeticErrorKind::NegativePower));
    }

    #[test]
    fn float_pow_odd_i32_exponent_keeps_sign() {
        use crate::error::ArithmeticStatus;
        use crate::scalar::Pow;

        let mut status = ArithmeticStatus::default();
        let odd = (1i32 << 24) + 1;
        assert_eq!(Pow::pow(-1.0f32, odd, &mut status), -1.0);
        assert_eq!(Pow::pow(-1.0f32, -odd, &mut status), -1.0);
        assert_eq!(Pow::pow(-2.0f32, odd, &mut status), f32::NEG_INFINITY);
        #[cfg(feature = "half")]
        assert_eq!(
            Pow::pow(f16::from_f32(-1.0), odd, &mut status),
            f16::from_f32(-1.0)
        );
    }

    #[test]
    fn int_pow_negative_exponent_read() {
        use crate::error::ErrorKind;
        use crate::Array;

        let a = Array::compact_ndarray(&ndarray::array![2i32, 3, 4, -5]).unwrap();
        let b = Array::compact_ndarray(&ndarray::array![3i32, -1, 0, 3]).unwrap();
        let q = a.pow(b);
        let err = q.to_ndarray::<Ix1>().unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Arithmetic);
        assert_eq!(
            err.message(),
            "integers to negative integer powers are not allowed"
        );
        // Only the elements read count.
        let ctx = q.read_ctx();
        assert_eq!(
            q.to_ndarray_sub(&[2..4], &ctx).unwrap(),
            ndarray::array![1, -125]
        );
    }

    // Integer powers against repeated wrapping multiplication.
    proptest::proptest! {
        #[test]
        fn int_pow_random(a in proptest::num::i32::ANY, b in -3i64..100) {
            use crate::error::{ArithmeticErrorKind, ArithmeticStatus};
            use crate::scalar::Pow;

            let mut status = ArithmeticStatus::default();
            let res = Pow::pow(a, b, &mut status);
            if b < 0 {
                proptest::prop_assert_eq!(res, 0);
                proptest::prop_assert!(status.contains(ArithmeticErrorKind::NegativePower));
            } else {
                proptest::prop_assert_eq!(res, (0..b).fold(1i32, |acc, _| acc.wrapping_mul(a)));
                proptest::prop_assert!(status.is_ok());
                let mut status = ArithmeticStatus::default();
                proptest::prop_assert_eq!(res, Pow::pow(a, b as i32, &mut status));
                proptest::prop_assert_eq!(res, Pow::pow(a, b as u32, &mut status));
            }
        }
    }
}
