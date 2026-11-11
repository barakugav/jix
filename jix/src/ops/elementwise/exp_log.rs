use crate::ops::elementwise::op1::{define_array_op1_method, define_op1};
use crate::ops::prelude::*;

pub(crate) mod traits {
    use crate::scalar::traits_util::float_op1;

    float_op1!(
        /// Square root operation, as [`f32::sqrt`].
        ///
        /// A negative input produces `NaN`.
        Sqrt,
        /// Compute the square root.
        sqrt,
        sqrt_bulk,
        simd = "enabled"
    );
    float_op1!(
        /// Natural exponential operation (`e^x`), as [`f32::exp`].
        ///
        /// A result too large for the type is `inf`.
        Exp,
        /// Compute the natural exponential.
        exp,
        exp_bulk
    );
    float_op1!(
        /// Natural logarithm operation, as [`f32::ln`].
        ///
        /// Zero produces `-inf`, and a negative input `NaN`.
        Ln,
        /// Compute the natural logarithm.
        ln,
        ln_bulk
    );
    float_op1!(
        /// Base-2 logarithm operation, as [`f32::log2`].
        ///
        /// Zero produces `-inf`, and a negative input `NaN`.
        Log2,
        /// Compute the base-2 logarithm.
        log2,
        log2_bulk
    );
    float_op1!(
        /// Base-10 logarithm operation, as [`f32::log10`].
        ///
        /// Zero produces `-inf`, and a negative input `NaN`.
        Log10,
        /// Compute the base-10 logarithm.
        log10,
        log10_bulk
    );
}

define_op1!(
    /// Computes the square root of each element.
    ///
    /// See [`jix::scalar::Sqrt`](crate::scalar::Sqrt) scalar trait for the per-element semantics.
    ///
    /// The result is a lazy view.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::sqrt()`](crate::Array::sqrt).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::{array, Ix1};
    ///
    /// let a = Array::compact_ndarray(&array![4.0f32, 9.0, 16.0])?;
    /// let result = a.sqrt().to_ndarray()?;
    /// assert_eq!(result, array![2.0, 3.0, 4.0]);
    ///
    /// // A negative input produces NaN.
    /// let b = Array::compact_ndarray(&array![-1.0f32])?;
    /// assert!(b.sqrt().to_ndarray::<Ix1>()?[[0]].is_nan());
    /// # Ok::<(), jix::Error>(())
    /// ```
    Sqrt,
    SqrtKernel,
    crate::scalar::Sqrt::sqrt,
    simd: sqrt_bulk,
);
define_op1!(
    /// Computes the natural exponential (`e^x`) of each element.
    ///
    /// See [`jix::scalar::Exp`](crate::scalar::Exp) scalar trait for the per-element semantics.
    ///
    /// The result is a lazy view.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::exp()`](crate::Array::exp).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::{array, Ix1};
    ///
    /// let a = Array::compact_ndarray(&array![1.0f32, 2.0, 3.0])?;
    /// let result = a.exp().to_ndarray::<Ix1>()?;
    /// assert!((result[[0]] - std::f32::consts::E).abs() < 1e-5);
    ///
    /// // exp(0.0) = 1.0 and exp(1.0) = e.
    /// let b = Array::compact_ndarray(&array![0.0f32, 1.0])?;
    /// let result = b.exp().to_ndarray::<Ix1>()?;
    /// assert_eq!(result[[0]], 1.0);
    /// assert!((result[[1]] - std::f32::consts::E).abs() < 1e-5);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Exp,
    ExpKernel,
    crate::scalar::Exp::exp,
    simd: exp_bulk,
);
define_op1!(
    /// Computes the natural logarithm (`ln x`) of each element.
    ///
    /// See [`jix::scalar::Ln`](crate::scalar::Ln) scalar trait for the per-element semantics.
    ///
    /// The result is a lazy view.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::ln()`](crate::Array::ln).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::{array, Ix1};
    ///
    /// let e = std::f32::consts::E;
    /// let a = Array::compact_ndarray(&array![1.0f32, e, e * e])?;
    /// let result = a.ln().to_ndarray::<Ix1>()?;
    /// assert!((result[[0]] - 0.0).abs() < 1e-5);
    /// assert!((result[[1]] - 1.0).abs() < 1e-5);
    ///
    /// // Zero produces -inf, and a negative input NaN.
    /// let b = Array::compact_ndarray(&array![0.0f32, -1.0])?;
    /// let result = b.ln().to_ndarray::<Ix1>()?;
    /// assert_eq!(result[[0]], f32::NEG_INFINITY);
    /// assert!(result[[1]].is_nan());
    /// # Ok::<(), jix::Error>(())
    /// ```
    Ln,
    LnKernel,
    crate::scalar::Ln::ln,
    simd: ln_bulk,
);
define_op1!(
    /// Computes the base-2 logarithm of each element.
    ///
    /// See [`jix::scalar::Log2`](crate::scalar::Log2) scalar trait for the per-element semantics.
    ///
    /// The result is a lazy view.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::log2()`](crate::Array::log2).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![1.0f64, 8.0, 0.5])?;
    /// assert_eq!(a.log2().to_ndarray()?, array![0.0, 3.0, -1.0]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Log2,
    Log2Kernel,
    crate::scalar::Log2::log2,
    simd: log2_bulk,
);
define_op1!(
    /// Computes the base-10 logarithm of each element.
    ///
    /// See [`jix::scalar::Log10`](crate::scalar::Log10) scalar trait for the per-element semantics.
    ///
    /// The result is a lazy view.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::log10()`](crate::Array::log10).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![1.0f64, 1000.0, 0.01])?;
    /// assert_eq!(a.log10().to_ndarray()?, array![0.0, 3.0, -2.0]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Log10,
    Log10Kernel,
    crate::scalar::Log10::log10,
    simd: log10_bulk,
);

impl<S> Array<S>
where
    S: ArrayStorage,
{
    define_array_op1_method!(sqrt: Sqrt, crate::scalar::Sqrt);
    define_array_op1_method!(exp: Exp, crate::scalar::Exp);
    define_array_op1_method!(ln: Ln, crate::scalar::Ln);
    define_array_op1_method!(log2: Log2, crate::scalar::Log2);
    define_array_op1_method!(log10: Log10, crate::scalar::Log10);
}

#[cfg(test)]
mod tests {
    use crate::ops::elementwise::op1::tests::test_op1;

    /// The `apply_bulk` of the kernels of this module (their scalar traits' SIMD `*_bulk`) against
    /// their scalar semantics (release builds: wrapping), on every SIMD level of the CPU, over edge
    /// cases.
    #[test]
    fn simd_bodies_all_levels() {
        use super::SqrtKernel;
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
                case!(SqrtKernel, f32, f32::sqrt);
                case!(SqrtKernel, f64, f64::sqrt);
            });
        }
        for_each_simd_level!(check);
    }
    #[test]
    fn sqrt_concrete() {
        use crate::Array;
        // Domain is non-negative (op_safe_non_negative_strategy): 0.0, a perfect square, a
        // non-perfect square, and the strategy's upper bound (100.0).
        let nd = ndarray::array![[0.0f32, 4.0, 2.0], [9.0, 0.25, 100.0]];
        let za = Array::compact_ndarray(&nd).unwrap();
        let expected = nd.mapv(|a: f32| a.sqrt());
        crate::util::assert_array_matches_approx(&za.view().sqrt(), &expected, 1e-6, 1e-6);

        let nd64 = ndarray::array![[0.0f64, 4.0, 2.0], [9.0, 0.25, 100.0]];
        let za64 = Array::compact_ndarray_with(&nd64, crate::util::arr_params(&[1, 2])).unwrap();
        let expected64 = nd64.mapv(|a: f64| a.sqrt());
        crate::util::assert_array_matches_approx(&za64.view().sqrt(), &expected64, 1e-12, 1e-12);
    }
    test_op1!(exp, |a| a.exp(), [f32, f64], op_safe_strategy);
    test_op1!(ln, |a| a.ln(), [f32, f64], op_safe_positive_strategy);

    #[test]
    fn log2_log10_of_exact_powers_are_exact() {
        use crate::scalar::{Log10, Log2};
        for k in -60..=60 {
            assert_eq!(Log2::log2(2f64.powi(k)), k as f64);
            assert_eq!(Log2::log2(2f32.powi(k / 3)), (k / 3) as f32);
        }
        for k in 0..=22 {
            assert_eq!(Log10::log10(10f64.powi(k)), k as f64);
        }
        #[cfg(feature = "half")]
        for k in -14..=15 {
            let x = crate::scalar::f16::from_f32(2f32.powi(k));
            assert_eq!(Log2::log2(x).to_f32(), k as f32);
        }
    }

    macro_rules! test_float_special_results {
        ($($t:ident),*) => {$(
            paste::paste! {
                /// Out of their domains or ranges, the float functions give `NaN` and `inf`, which
                /// are not errors: the read succeeds, in the bulk part of the loop and in its
                /// scalar tail.
                #[test]
                fn [<float_special_results_ $t>]() {
                    #[cfg(feature = "half")]
                    #[allow(unused_imports)]
                    use crate::scalar::f16;

                    let c = |x: f64| <f64 as crate::scalar::Cast<$t>>::cast(x);
                    let n = 1000;
                    let (inf, nan) = (f64::INFINITY, f64::NAN);
                    for (name, x, expected) in [
                        ("sqrt", -1.0, nan),
                        ("ln", 0.0, -inf),
                        ("ln", -1.0, nan),
                        ("log2", 0.0, -inf),
                        ("log10", -1.0, nan),
                        ("exp", 1000.0, inf),
                    ] {
                        for i in [300, n - 1] {
                            let mut nd = ndarray::Array1::from_shape_fn(n, |i| c((i % 10) as f64 * 0.1));
                            nd[i] = c(x);
                            let a = crate::Array::compact_ndarray(&nd).unwrap();
                            let res = match name {
                                "sqrt" => a.sqrt().to_ndarray::<ndarray::Ix1>(),
                                "ln" => a.ln().to_ndarray::<ndarray::Ix1>(),
                                "log2" => a.log2().to_ndarray::<ndarray::Ix1>(),
                                "log10" => a.log10().to_ndarray::<ndarray::Ix1>(),
                                "exp" => a.exp().to_ndarray::<ndarray::Ix1>(),
                                _ => unreachable!(),
                            };
                            let what = format!("{name}({x}) at {i}");
                            let y: f64 = res.expect(&what)[i].into();
                            assert!(y == expected || y.is_nan() && expected.is_nan(), "{what}: {y}");
                        }
                    }
                }
            }
        )*};
    }
    test_float_special_results!(f32, f64);
    #[cfg(feature = "half")]
    test_float_special_results!(f16);
}
