use crate::ops::elementwise::op1::{define_array_op1_method, define_op1};
use crate::ops::prelude::*;

pub(crate) mod traits {
    use crate::scalar::traits_util::float_op1;

    float_op1!(
        /// Flooring operation, as [`f32::floor`].
        Floor,
        /// Round down to the nearest integer (towards -inf).
        floor,
        floor_bulk,
        simd = "enabled"
    );
    float_op1!(
        /// Ceiling operation, as [`f32::ceil`].
        Ceil,
        /// Round up to the nearest integer (towards +inf).
        ceil,
        ceil_bulk,
        simd = "enabled"
    );
    float_op1!(
        /// Rounding operation, as [`f32::round`].
        ///
        /// Ties (values exactly halfway between two integers) are broken by rounding
        /// away from zero: `round(0.5) = 1.0`, `round(-0.5) = -1.0`. This differs from
        /// "round-half-to-even" (banker's rounding) used in some other libraries.
        Round,
        /// Round to the nearest integer.
        round,
        round_bulk
    );
}

define_op1!(
    /// Rounds each element down to the nearest integer (towards -inf).
    ///
    /// See [`jix::scalar::Floor`](crate::scalar::Floor) scalar trait for the per-element semantics.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::floor()`](crate::Array::floor).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![1.1f32, 2.9, 3.0])?;
    /// let result = a.floor().to_ndarray()?;
    /// assert_eq!(result, array![1.0, 2.0, 3.0]);
    ///
    /// // Floor rounds towards -inf, so negative values floor down.
    /// let b = Array::compact_ndarray(&array![-1.1f32, -2.9, -3.0])?;
    /// let result = b.floor().to_ndarray()?;
    /// assert_eq!(result, array![-2.0, -3.0, -3.0]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Floor,
    FloorKernel,
    crate::scalar::Floor::floor,
    simd: floor_bulk,
);
define_op1!(
    /// Rounds each element up to the nearest integer (towards +inf).
    ///
    /// See [`jix::scalar::Ceil`](crate::scalar::Ceil) scalar trait for the per-element semantics.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::ceil()`](crate::Array::ceil).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![1.1f32, 2.0, 3.9])?;
    /// let result = a.ceil().to_ndarray()?;
    /// assert_eq!(result, array![2.0, 2.0, 4.0]);
    ///
    /// // Ceil rounds towards +inf, so negative values ceil up.
    /// let b = Array::compact_ndarray(&array![-1.7f32, -2.0, -0.1])?;
    /// let result = b.ceil().to_ndarray()?;
    /// assert_eq!(result, array![-1.0, -2.0, 0.0]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Ceil,
    CeilKernel,
    crate::scalar::Ceil::ceil,
    simd: ceil_bulk,
);
define_op1!(
    /// Rounds each element to the nearest integer.
    ///
    /// See [`jix::scalar::Round`](crate::scalar::Round) scalar trait for the per-element semantics.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::round()`](crate::Array::round).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_ndarray(&array![1.4f32, 1.6, 2.0])?;
    /// let result = a.round().to_ndarray()?;
    /// assert_eq!(result, array![1.0, 2.0, 2.0]);
    ///
    /// // Ties are broken away from zero: 0.5 -> 1.0, -0.5 -> -1.0.
    /// let b = Array::compact_ndarray(&array![0.5f32, -0.5])?;
    /// let result = b.round().to_ndarray()?;
    /// assert_eq!(result, array![1.0, -1.0]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Round,
    RoundKernel,
    crate::scalar::Round::round,
    simd: round_bulk,
);

impl<S> Array<S>
where
    S: ArrayStorage,
{
    define_array_op1_method!(floor: Floor, crate::scalar::Floor);
    define_array_op1_method!(ceil: Ceil, crate::scalar::Ceil);
    define_array_op1_method!(round: Round, crate::scalar::Round);
}

#[cfg(test)]
mod tests {
    /// The `apply_bulk` of the kernels of this module (their scalar traits' SIMD `*_bulk`) against
    /// their scalar semantics (release builds: wrapping), on every SIMD level of the CPU, over edge
    /// cases.
    #[test]
    fn simd_bodies_all_levels() {
        use super::{CeilKernel, FloorKernel, RoundKernel};
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
                case!(FloorKernel, f32, f32::floor);
                case!(FloorKernel, f64, f64::floor);
                case!(CeilKernel, f32, f32::ceil);
                case!(CeilKernel, f64, f64::ceil);
                case!(RoundKernel, f32, f32::round);
                case!(RoundKernel, f64, f64::round);
                // Not a multiple of the vector length, and a type without a SIMD body: scalar.
                assert_eq!(
                    FloorKernel.apply_bulk([1.5f32, 2.5], &mut Default::default(), simd),
                    [1.0, 2.0]
                );
            });
        }
        for_each_simd_level!(check);
    }
    #[test]
    fn floor_concrete() {
        use crate::Array;
        // Edge inputs: exact integers, positive/negative fractions, .5 tie, and a
        // multi-block shape so a block-boundary bug still shows up.
        let nd = ndarray::array![[-2.5f32, -0.5, 0.0], [0.5, 2.5, 3.9]];
        let za = Array::compact_ndarray(&nd).unwrap();
        let expected = nd.mapv(|a: f32| a.floor());
        crate::util::assert_array_matches(&za.view().floor(), &expected);

        // Second dtype (f64) and a non-default block shape to cross block boundaries.
        let nd64 = ndarray::array![[-2.5f64, -0.5, 0.0], [0.5, 2.5, 3.9]];
        let za64 = Array::compact_ndarray_with(&nd64, crate::util::arr_params(&[1, 2])).unwrap();
        let expected64 = nd64.mapv(|a: f64| a.floor());
        crate::util::assert_array_matches(&za64.view().floor(), &expected64);
    }
    #[test]
    fn ceil_concrete() {
        use crate::Array;
        // Same edge inputs as floor: exact integers, positive/negative fractions, .5 tie.
        let nd = ndarray::array![[-2.5f32, -0.5, 0.0], [0.5, 2.5, 3.9]];
        let za = Array::compact_ndarray(&nd).unwrap();
        let expected = nd.mapv(|a: f32| a.ceil());
        crate::util::assert_array_matches(&za.view().ceil(), &expected);

        let nd64 = ndarray::array![[-2.5f64, -0.5, 0.0], [0.5, 2.5, 3.9]];
        let za64 = Array::compact_ndarray(&nd64).unwrap();
        let expected64 = nd64.mapv(|a: f64| a.ceil());
        crate::util::assert_array_matches(&za64.view().ceil(), &expected64);
    }
    #[test]
    fn round_concrete() {
        use crate::Array;
        // .5 ties round away from zero (not banker's rounding): 0.5 -> 1.0, -0.5 -> -1.0,
        // 2.5 -> 3.0, -2.5 -> -3.0.
        let nd = ndarray::array![[-2.5f32, -0.5, 0.0], [0.5, 2.5, 1.4]];
        let za = Array::compact_ndarray(&nd).unwrap();
        let expected = nd.mapv(|a: f32| a.round());
        crate::util::assert_array_matches(&za.view().round(), &expected);

        let nd64 = ndarray::array![[-2.5f64, -0.5, 0.0], [0.5, 2.5, 1.4]];
        let za64 = Array::compact_ndarray(&nd64).unwrap();
        let expected64 = nd64.mapv(|a: f64| a.round());
        crate::util::assert_array_matches(&za64.view().round(), &expected64);
    }
}
