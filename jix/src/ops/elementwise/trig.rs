use crate::ops::elementwise::op1::{define_array_op1_method, define_op1};
use crate::ops::prelude::*;

pub(crate) mod traits {
    use crate::scalar::traits_util::float_op1;

    float_op1!(
        /// Sine operation, as [`f32::sin`].
        ///
        /// An infinite input produces `NaN`.
        Sin,
        /// Compute the sine (input in radians).
        sin,
        sin_bulk
    );
    float_op1!(
        /// Cosine operation, as [`f32::cos`].
        ///
        /// An infinite input produces `NaN`.
        Cos,
        /// Compute the cosine (input in radians).
        cos,
        cos_bulk
    );
    float_op1!(
        /// Tangent operation, as [`f32::tan`].
        ///
        /// An infinite input produces `NaN`.
        Tan,
        /// Compute the tangent (input in radians).
        tan,
        tan_bulk
    );
    float_op1!(
        /// Arcsine operation, as [`f32::asin`].
        ///
        /// An input outside `[-1, 1]` produces `NaN`.
        Asin,
        /// Compute the arcsine; the output is in radians in `[-pi/2, pi/2]`.
        asin,
        asin_bulk
    );
    float_op1!(
        /// Arccosine operation, as [`f32::acos`].
        ///
        /// An input outside `[-1, 1]` produces `NaN`.
        Acos,
        /// Compute the arccosine; the output is in radians in `[0, pi]`.
        acos,
        acos_bulk
    );
    float_op1!(
        /// Arctangent operation, as [`f32::atan`].
        Atan,
        /// Compute the arctangent; the output is in radians in `(-pi/2, pi/2)`.
        atan,
        atan_bulk
    );
}

define_op1!(
    /// Computes the sine of each element (input in radians).
    ///
    /// See [`jix::scalar::Sin`](crate::scalar::Sin) scalar trait for the per-element semantics.
    ///
    /// The result is a lazy view.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::sin()`](crate::Array::sin).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::{array, Ix1};
    ///
    /// let a = Array::compact_ndarray(&array![std::f32::consts::FRAC_PI_2, std::f32::consts::PI])?;
    /// let result = a.sin().to_ndarray::<Ix1>()?;
    /// assert!((result[[0]] - 1.0).abs() < 1e-5);
    ///
    /// // sin(0.0) = 0.0.
    /// let b = Array::compact_ndarray(&array![0.0f32])?;
    /// let result = b.sin().to_ndarray::<Ix1>()?;
    /// assert_eq!(result[[0]], 0.0);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Sin,
    SinKernel,
    crate::scalar::Sin::sin,
    simd: sin_bulk,
);
define_op1!(
    /// Computes the cosine of each element (input in radians).
    ///
    /// See [`jix::scalar::Cos`](crate::scalar::Cos) scalar trait for the per-element semantics.
    ///
    /// The result is a lazy view.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::cos()`](crate::Array::cos).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::{array, Ix1};
    ///
    /// let a = Array::compact_ndarray(&array![0.0f32, std::f32::consts::PI])?;
    /// let result = a.cos().to_ndarray::<Ix1>()?;
    /// assert!((result[[0]] - 1.0).abs() < 1e-5);
    /// assert!((result[[1]] - (-1.0)).abs() < 1e-5);
    ///
    /// // cos(0.0) = 1.0.
    /// let b = Array::compact_ndarray(&array![0.0f32])?;
    /// let result = b.cos().to_ndarray::<Ix1>()?;
    /// assert_eq!(result[[0]], 1.0);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Cos,
    CosKernel,
    crate::scalar::Cos::cos,
    simd: cos_bulk,
);
define_op1!(
    /// Computes the tangent of each element (input in radians).
    ///
    /// See [`jix::scalar::Tan`](crate::scalar::Tan) scalar trait for the per-element semantics.
    ///
    /// The result is a lazy view.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::tan()`](crate::Array::tan).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::{array, Ix1};
    ///
    /// let a = Array::compact_ndarray(&array![std::f32::consts::FRAC_PI_4, std::f32::consts::FRAC_PI_2 * 0.5])?;
    /// let result = a.tan().to_ndarray::<Ix1>()?;
    /// assert!((result[[0]] - 1.0).abs() < 1e-5);
    ///
    /// // tan(0.0) = 0.0.
    /// let b = Array::compact_ndarray(&array![0.0f32])?;
    /// let result = b.tan().to_ndarray::<Ix1>()?;
    /// assert_eq!(result[[0]], 0.0);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Tan,
    TanKernel,
    crate::scalar::Tan::tan,
    simd: tan_bulk,
);
define_op1!(
    /// Computes the arcsine of each element; output is in radians in `[-pi/2, pi/2]`.
    ///
    /// See [`jix::scalar::Asin`](crate::scalar::Asin) scalar trait for the per-element semantics.
    ///
    /// The result is a lazy view.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::asin()`](crate::Array::asin).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::{array, Ix1};
    ///
    /// let a = Array::compact_ndarray(&array![0.0f32, 1.0, -1.0])?;
    /// let result = a.asin().to_ndarray::<Ix1>()?;
    /// assert_eq!(result[[0]], 0.0);
    /// assert!((result[[1]] - std::f32::consts::FRAC_PI_2).abs() < 1e-5);
    ///
    /// // An input outside [-1, 1] produces NaN.
    /// let b = Array::compact_ndarray(&array![2.0f32])?;
    /// assert!(b.asin().to_ndarray::<Ix1>()?[[0]].is_nan());
    /// # Ok::<(), jix::Error>(())
    /// ```
    Asin,
    AsinKernel,
    crate::scalar::Asin::asin,
    simd: asin_bulk,
);
define_op1!(
    /// Computes the arccosine of each element; output is in radians in `[0, pi]`.
    ///
    /// See [`jix::scalar::Acos`](crate::scalar::Acos) scalar trait for the per-element semantics.
    ///
    /// The result is a lazy view.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::acos()`](crate::Array::acos).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::{array, Ix1};
    ///
    /// let a = Array::compact_ndarray(&array![1.0f32, 0.0, -1.0])?;
    /// let result = a.acos().to_ndarray::<Ix1>()?;
    /// assert_eq!(result[[0]], 0.0);
    /// assert!((result[[1]] - std::f32::consts::FRAC_PI_2).abs() < 1e-5);
    ///
    /// // An input outside [-1, 1] produces NaN.
    /// let b = Array::compact_ndarray(&array![2.0f32])?;
    /// assert!(b.acos().to_ndarray::<Ix1>()?[[0]].is_nan());
    /// # Ok::<(), jix::Error>(())
    /// ```
    Acos,
    AcosKernel,
    crate::scalar::Acos::acos,
    simd: acos_bulk,
);
define_op1!(
    /// Computes the arctangent of each element; output is in radians in `(-pi/2, pi/2)`.
    ///
    /// See [`jix::scalar::Atan`](crate::scalar::Atan) scalar trait for the per-element semantics.
    ///
    /// The result is a lazy view.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::atan()`](crate::Array::atan).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::{array, Ix1};
    ///
    /// let a = Array::compact_ndarray(&array![0.0f32, -1.0, 1.0])?;
    /// let result = a.atan().to_ndarray::<Ix1>()?;
    /// assert_eq!(result[[0]], 0.0);
    ///
    /// // atan(1.0) = pi/4.
    /// let b = Array::compact_ndarray(&array![1.0f32])?;
    /// let result = b.atan().to_ndarray::<Ix1>()?;
    /// assert!((result[[0]] - std::f32::consts::FRAC_PI_4).abs() < 1e-5);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Atan,
    AtanKernel,
    crate::scalar::Atan::atan,
    simd: atan_bulk,
);

impl<S> Array<S>
where
    S: ArrayStorage,
{
    define_array_op1_method!(sin: Sin, crate::scalar::Sin);
    define_array_op1_method!(cos: Cos, crate::scalar::Cos);
    define_array_op1_method!(tan: Tan, crate::scalar::Tan);
    define_array_op1_method!(asin: Asin, crate::scalar::Asin);
    define_array_op1_method!(acos: Acos, crate::scalar::Acos);
    define_array_op1_method!(atan: Atan, crate::scalar::Atan);
}

#[cfg(test)]
mod tests {
    #[test]
    fn sin_concrete() {
        use crate::Array;
        // 0, +/-pi/2, pi, and a couple of interior op_safe_strategy values.
        let nd = ndarray::array![
            0.0f32,
            std::f32::consts::FRAC_PI_2,
            std::f32::consts::PI,
            -std::f32::consts::FRAC_PI_2,
            1.0,
            -1.0
        ];
        let za = Array::compact_ndarray(&nd).unwrap();
        let expected = nd.mapv(|a: f32| a.sin());
        crate::util::assert_array_matches_approx(&za.view().sin(), &expected, 1e-6, 1e-6);

        let nd64 = nd.mapv(f64::from);
        let za64 = Array::compact_ndarray(&nd64).unwrap();
        let expected64 = nd64.mapv(|a: f64| a.sin());
        crate::util::assert_array_matches_approx(&za64.view().sin(), &expected64, 1e-12, 1e-12);
    }
    #[test]
    fn cos_concrete() {
        use crate::Array;
        let nd = ndarray::array![
            0.0f32,
            std::f32::consts::FRAC_PI_2,
            std::f32::consts::PI,
            -std::f32::consts::FRAC_PI_2,
            1.0,
            -1.0
        ];
        let za = Array::compact_ndarray(&nd).unwrap();
        let expected = nd.mapv(|a: f32| a.cos());
        crate::util::assert_array_matches_approx(&za.view().cos(), &expected, 1e-6, 1e-6);

        let nd64 = nd.mapv(f64::from);
        let za64 = Array::compact_ndarray(&nd64).unwrap();
        let expected64 = nd64.mapv(|a: f64| a.cos());
        crate::util::assert_array_matches_approx(&za64.view().cos(), &expected64, 1e-12, 1e-12);
    }
    #[test]
    fn tan_concrete() {
        use crate::Array;
        // Avoid drawing exactly at the +/-pi/2 asymptote; op_safe_strategy never hits it
        // either (it draws x/100.0 for integer x, never an exact multiple of pi/2).
        let nd = ndarray::array![
            0.0f32,
            std::f32::consts::FRAC_PI_4,
            -std::f32::consts::FRAC_PI_4,
            1.0,
            -1.0,
            10.0
        ];
        let za = Array::compact_ndarray(&nd).unwrap();
        let expected = nd.mapv(|a: f32| a.tan());
        crate::util::assert_array_matches_approx(&za.view().tan(), &expected, 1e-6, 1e-6);

        let nd64 = nd.mapv(f64::from);
        let za64 = Array::compact_ndarray(&nd64).unwrap();
        let expected64 = nd64.mapv(|a: f64| a.tan());
        crate::util::assert_array_matches_approx(&za64.view().tan(), &expected64, 1e-12, 1e-12);
    }
    #[test]
    fn asin_concrete() {
        use crate::Array;
        // Domain is [-1, 1] (unit_strategy): both endpoints plus interior points.
        let nd = ndarray::array![-1.0f32, -0.5, 0.0, 0.5, 1.0];
        let za = Array::compact_ndarray(&nd).unwrap();
        let expected = nd.mapv(|a: f32| a.asin());
        crate::util::assert_array_matches_approx(&za.view().asin(), &expected, 1e-6, 1e-6);

        let nd64 = nd.mapv(f64::from);
        let za64 = Array::compact_ndarray(&nd64).unwrap();
        let expected64 = nd64.mapv(|a: f64| a.asin());
        crate::util::assert_array_matches_approx(&za64.view().asin(), &expected64, 1e-12, 1e-12);
    }
    #[test]
    fn acos_concrete() {
        use crate::Array;
        // Domain is [-1, 1] (unit_strategy): both endpoints plus interior points.
        let nd = ndarray::array![-1.0f32, -0.5, 0.0, 0.5, 1.0];
        let za = Array::compact_ndarray(&nd).unwrap();
        let expected = nd.mapv(|a: f32| a.acos());
        crate::util::assert_array_matches_approx(&za.view().acos(), &expected, 1e-6, 1e-6);

        let nd64 = nd.mapv(f64::from);
        let za64 = Array::compact_ndarray(&nd64).unwrap();
        let expected64 = nd64.mapv(|a: f64| a.acos());
        crate::util::assert_array_matches_approx(&za64.view().acos(), &expected64, 1e-12, 1e-12);
    }
    #[test]
    fn atan_concrete() {
        use crate::Array;
        let nd = ndarray::array![0.0f32, 1.0, -1.0, 100.0, -100.0];
        let za = Array::compact_ndarray(&nd).unwrap();
        let expected = nd.mapv(|a: f32| a.atan());
        crate::util::assert_array_matches_approx(&za.view().atan(), &expected, 1e-6, 1e-6);

        let nd64 = nd.mapv(f64::from);
        let za64 = Array::compact_ndarray(&nd64).unwrap();
        let expected64 = nd64.mapv(|a: f64| a.atan());
        crate::util::assert_array_matches_approx(&za64.view().atan(), &expected64, 1e-12, 1e-12);
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
                        ("sin", inf, nan),
                        ("cos", -inf, nan),
                        ("tan", inf, nan),
                        ("asin", 2.0, nan),
                        ("acos", -2.0, nan),
                    ] {
                        for i in [300, n - 1] {
                            let mut nd = ndarray::Array1::from_shape_fn(n, |i| c((i % 10) as f64 * 0.1));
                            nd[i] = c(x);
                            let a = crate::Array::compact_ndarray(&nd).unwrap();
                            let res = match name {
                                "sin" => a.sin().to_ndarray::<ndarray::Ix1>(),
                                "cos" => a.cos().to_ndarray::<ndarray::Ix1>(),
                                "tan" => a.tan().to_ndarray::<ndarray::Ix1>(),
                                "asin" => a.asin().to_ndarray::<ndarray::Ix1>(),
                                "acos" => a.acos().to_ndarray::<ndarray::Ix1>(),
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
