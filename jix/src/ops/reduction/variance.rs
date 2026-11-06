use fearless_simd::Simd;

use crate::ops::prelude::*;
use crate::ops::reduction::common::{
    define_array_reduction_method, define_reduction_op, ReductionOpKernel,
};

pub(crate) mod traits {
    use crate::scalar::bool8;
    #[cfg(feature = "half")]
    use crate::scalar::f16;
    #[cfg(feature = "num-complex")]
    use crate::scalar::Complex;
    use crate::util::{array2_map_inline, ArrayExt};

    /// Welford accumulator used by [`Variance`]
    #[derive(Clone, Copy)]
    pub struct VarianceState<M> {
        /// Running mean in the type-specific accumulator domain.
        mean: M,
        /// Running sum of squared deviations from the mean.
        m2: f64,
        /// Number of items folded into this accumulator.
        count: u64,
    }

    /// Scalar kernel trait for the `var` (variance) and `std` (standard deviation) reductions.
    ///
    /// The degree-of-freedom correction is controlled by `ddof` passed to [`finalize`]:
    /// use `0.0` for population variance (`N` denominator) and `1.0` for sample variance
    /// (`N-1` denominator).
    ///
    /// Uses Welford's online algorithm for numerical stability.
    ///
    /// [`finalize`]: Variance::finalize
    pub trait Variance {
        /// The output element type - always a real `Float`: `f64` for integer and `bool8`
        /// inputs, the input width for real floats (`f16`/`f32`/`f64`), and the real component
        /// type for complex inputs (`Complex<f32> -> f32`, `Complex<f64> -> f64`).
        type Output;
        /// Welford accumulator state.
        type State;
        /// Return the initial (empty) accumulator.
        fn init() -> Self::State;
        /// Fold `item` into the running Welford accumulator.
        fn update(state: Self::State, item: Self) -> Self::State;
        /// [`update`](Self::update) on each lane, vectorized with `simd` where the types have
        /// SIMD support.
        #[inline(always)]
        fn update_bulk<const N: usize, S: fearless_simd::Simd>(
            states: [Self::State; N],
            items: [Self; N],
            simd: S,
        ) -> [Self::State; N]
        where
            Self: Sized,
        {
            let _ = simd;
            array2_map_inline(
                states,
                items,
                #[inline(always)]
                |_, state, item| Self::update(state, item),
            )
        }
        /// Combine two Welford accumulators computed over disjoint subsets (Chan's parallel
        /// algorithm). Associative + commutative up to float error.
        fn merge_states(a: Self::State, b: Self::State) -> Self::State;
        /// Finalize `state` into the variance using `ddof` degrees-of-freedom correction, or into
        /// its square root, the standard deviation, if `sqrt`. `nitems` is the total number of
        /// elements folded in.
        ///
        /// Returns `NaN` if the effective denominator (`nitems - ddof`) is non-positive.
        fn finalize(state: Self::State, ddof: f64, nitems: u64, sqrt: bool) -> Self::Output;
    }
    macro_rules! impl_variance {
        (
            $item_ty:ty => $output_ty:ty,
            MeanT = $mean_ty:ty,
            is_complex = $is_complex:ident,
        ) => {
            impl Variance for $item_ty {
                type Output = $output_ty;
                type State = VarianceState<$mean_ty>;

                #[inline(always)]
                fn init() -> Self::State {
                    VarianceState {
                        mean: <i32 as crate::scalar::Cast<$mean_ty>>::cast(0),
                        m2: 0.0,
                        count: 0,
                    }
                }
                #[inline(always)]
                fn update(mut state: Self::State, item: Self) -> Self::State {
                    state.count += 1;
                    let x = <_ as crate::scalar::Cast<$mean_ty>>::cast(item);
                    let delta = x - state.mean;
                    state.mean += delta / state.count as f64;
                    let delta2 = x - state.mean;
                    state.m2 += impl_variance!(@mul_dev is_complex=$is_complex, delta, delta2);
                    state
                }
                #[inline(always)]
                fn update_bulk<const N: usize, S: fearless_simd::Simd>(
                    states: [Self::State; N],
                    items: [Self; N],
                    simd: S,
                ) -> [Self::State; N] {
                    use crate::util::array_from_fn_inline;
                    let x = items.map_inline(#[inline(always)] |x| <_ as crate::scalar::Cast<$mean_ty>>::cast(x));
                    // let x = <_ as crate::scalar::Cast<$mean_ty>>::cast_bulk(items, simd); // TODO: is this faster?
                    let count = states.map_inline(#[inline(always)] |s| s.count + 1);
                    let count_f = count.map_inline(#[inline(always)] |c| c as f64);
                    let old_mean = states.map_inline(#[inline(always)] |s| s.mean);
                    let old_m2 = states.map_inline(#[inline(always)] |s| s.m2);
                    let (mean, m2) = impl_variance!(@update_bulk is_complex=$is_complex, old_mean, old_m2, x, count_f, simd);
                    array_from_fn_inline(
                        #[inline(always)]
                        |i| VarianceState {
                            mean: mean[i],
                            m2: m2[i],
                            count: count[i],
                        },
                    )
                }
                #[inline(always)]
                fn merge_states(a: Self::State, b: Self::State) -> Self::State {
                    if a.count == 0 {
                        return b;
                    }
                    if b.count == 0 {
                        return a;
                    }
                    let na = a.count;
                    let nb = b.count;
                    let n = na + nb;
                    // Chan's parallel Welford combine. The m2 cross-term is the squared
                    // deviation of the means (real: delta^2; complex: |delta|^2).
                    let delta = b.mean - a.mean;
                    let mean = a.mean
                        + delta
                            * <f64 as crate::scalar::Cast<$mean_ty>>::cast(nb as f64 / n as f64);
                    let m2 = a.m2
                        + b.m2
                        + impl_variance!(@mul_dev is_complex=$is_complex, delta, delta)
                            * (na as f64 * nb as f64 / n as f64);
                    VarianceState {
                        mean,
                        m2,
                        count: a.count + b.count,
                    }
                }
                #[inline(always)]
                fn finalize(state: Self::State, ddof: f64, nitems: u64, sqrt: bool) -> Self::Output {
                    let m2 = if state.m2 < 0.0 { f64::INFINITY } else { state.m2 };
                    let denom = nitems as f64 - ddof;
                    let res = if denom > 0.0 { m2 / denom } else { f64::NAN };
                    let res = if sqrt { res.sqrt() } else { res };
                    <_ as crate::scalar::Cast<Self::Output>>::cast(res)
                }
            }
        };
        (@mul_dev is_complex=false, $a:ident, $b:ident) => { $a * $b };
        (@mul_dev is_complex=true, $a:ident, $b:ident) => { $a.re * $b.re + $a.im * $b.im };
        (@square_dev is_complex=false, $dev:ident) => { $dev.square() };
        (@square_dev is_complex=true, $dev:ident) => { $dev.map(|d| d.re * d.re + d.im * d.im) };
        (@update_bulk is_complex=false,
            $mean:ident, $m2:ident, $x:ident, $count:ident, $simd:ident
        ) => {
            welford_update_bulk($mean, $m2, $x, $count, $simd)
        };
        (@update_bulk is_complex=true,
            $mean:ident, $m2:ident, $x:ident, $count:ident, $simd:ident
        ) => {{
            let re = |c: Complex<_>| c.re;
            let im = |c: Complex<_>| c.im;
            let (mean_re, m2) =
                welford_update_bulk($mean.map_inline(re), $m2, $x.map_inline(re), $count, $simd);
            let (mean_im, m2) =
                welford_update_bulk($mean.map_inline(im), m2, $x.map_inline(im), $count, $simd);
            let mean: [_; N] = crate::util::array_from_fn_inline(
                #[inline(always)]
                |i| Complex::new(mean_re[i], mean_im[i]),
            );
            (mean, m2)
        }};
    }

    #[inline(always)]
    fn welford_update_bulk<const N: usize, S: fearless_simd::Simd>(
        mean: [f64; N],
        m2: [f64; N],
        x: [f64; N],
        count: [f64; N],
        simd: S,
    ) -> ([f64; N], [f64; N]) {
        use crate::scalar::simd::map_vectors2;
        let delta = map_vectors2(x, mean, |x, m| x - m, |x, m| x - m, simd);

        let same_count = count.len() > 1 && count.iter().all(|&c| c == count[0]);
        let step = if same_count {
            let inv_count = [1.0 / count[0]; N]; // only one divide
            map_vectors2(delta, inv_count, |d, r| d * r, |d, r| d * r, simd)
        } else {
            map_vectors2(delta, count, |d, n| d / n, |d, n| d / n, simd)
        };

        let mean = map_vectors2(mean, step, |m, s| m + s, |m, s| m + s, simd);
        let delta2 = map_vectors2(x, mean, |x, m| x - m, |x, m| x - m, simd);
        let sq = map_vectors2(delta, delta2, |a, b| a * b, |a, b| a * b, simd);
        let m2 = map_vectors2(m2, sq, |a, b| a + b, |a, b| a + b, simd);
        (mean, m2)
    }

    impl_variance!(
        i8 => f64,
        MeanT = f64,
        is_complex = false,
    );
    impl_variance!(
        i16 => f64,
        MeanT = f64,
        is_complex = false,
    );
    impl_variance!(
        i32 => f64,
        MeanT = f64,
        is_complex = false,
    );
    impl_variance!(
        i64 => f64,
        MeanT = f64,
        is_complex = false,
    );
    impl_variance!(
        u8 => f64,
        MeanT = f64,
        is_complex = false,
    );
    impl_variance!(
        u16 => f64,
        MeanT = f64,
        is_complex = false,
    );
    impl_variance!(
        u32 => f64,
        MeanT = f64,
        is_complex = false,
    );
    impl_variance!(
        u64 => f64,
        MeanT = f64,
        is_complex = false,
    );
    #[cfg(feature = "half")]
    impl_variance!(
        f16 => f16,
        MeanT = f64,
        is_complex = false,
    );
    impl_variance!(
        f32 => f32,
        MeanT = f64,
        is_complex = false,
    );
    impl_variance!(
        f64 => f64,
        MeanT = f64,
        is_complex = false,
    );
    #[cfg(feature = "num-complex")]
    impl_variance!(
        Complex<f32> => f32,
        MeanT = Complex<f64>,
        is_complex = true,
    );
    #[cfg(feature = "num-complex")]
    impl_variance!(
        Complex<f64> => f64,
        MeanT = Complex<f64>,
        is_complex = true,
    );
    impl_variance!(
        bool8 => f64,
        MeanT = f64,
        is_complex = false,
    );
}

define_reduction_op!(
    /// Reduces one or more dims by computing the variance.
    ///
    /// The variance is real-valued and its dtype matches NumPy: integer and `bool8` inputs
    /// promote to `f64`, real floats keep their width (`f16 -> f16`, `f32 -> f32`, `f64 ->
    /// f64`), and complex inputs reduce to their real component type (`Complex<f32> -> f32`,
    /// `Complex<f64> -> f64`), computing `E[|x - mean|^2]`.
    ///
    /// The `ddof` parameter (delta degrees of freedom) adjusts the divisor: the variance
    /// is computed as `sum((x - mean)^2) / (n - ddof)`. Use `ddof=0` for the population
    /// variance and `ddof=1` for the sample variance (Bessel's correction). If
    /// `n - ddof <= 0`, the result is `NaN`.
    ///
    /// Uses Welford's online algorithm for numerical stability.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::var()`](crate::Array::var).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::{array, Ix0};
    ///
    /// let nd = array![[1i32, 2, 3], [4, 5, 6]];
    ///
    /// // Population variance (ddof=0) of all elements -> f64
    /// let var_all = Array::compact_ndarray(&nd)?
    ///     .var((0, 1), 0.0).to_ndarray::<Ix0>()?;
    /// assert!((var_all[[]] - 2.9167).abs() < 0.001);
    ///
    /// // Sample variance (ddof=1) along dim 0 -> shape [3]
    /// let col_vars = Array::compact_ndarray(&nd)?
    ///     .var(0, 1.0).to_ndarray()?;
    /// assert_eq!(col_vars, array![4.5, 4.5, 4.5]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Variance,
    VarianceKernel { ddof: f64 },
    where {
        S: ArrayStorageTyped,
        S::Item: crate::scalar::Variance<Output: Dtyped>,
    }
    output = <S::Item as crate::scalar::Variance>::Output,
);
impl<T> ReductionOpKernel<T> for VarianceKernel
where
    T: crate::scalar::Variance,
{
    type Output = <T as crate::scalar::Variance>::Output;
    type State = <T as crate::scalar::Variance>::State;

    #[inline(always)]
    fn init_state(&self, init_item: Option<(T, Option<u64>)>) -> Self::State {
        let mut state = <T as crate::scalar::Variance>::init();
        if let Some((item, _idx)) = init_item {
            state = self.update_state(state, item, None);
        }
        state
    }
    #[inline(always)]
    fn init_state_bulk<const N: usize, S: fearless_simd::Simd>(
        &self,
        items: [T; N],
        _indices: Option<[u64; N]>,
        simd: S,
    ) -> [Self::State; N] {
        // somehow this helps the compiler optimize the loop better, avoid lane shuffles
        let states = array_from_fn_inline(
            #[inline(always)]
            |_| <T as crate::scalar::Variance>::init(),
        );
        <T as crate::scalar::Variance>::update_bulk(states, items, simd)
    }
    #[inline(always)]
    fn update_state(&self, state: Self::State, item: T, _idx: Option<u64>) -> Self::State {
        <T as crate::scalar::Variance>::update(state, item)
    }
    #[inline(always)]
    fn update_state_bulk<const N: usize, S: Simd>(
        &self,
        states: [Self::State; N],
        items: [T; N],
        _indices: Option<[u64; N]>,
        simd: S,
    ) -> [Self::State; N] {
        <T as crate::scalar::Variance>::update_bulk(states, items, simd)
    }
    #[inline(always)]
    fn merge_states(&self, a: Self::State, b: Self::State) -> Self::State {
        <T as crate::scalar::Variance>::merge_states(a, b)
    }
    #[inline(always)]
    fn finalize_state(&self, state: Self::State, nitems: u64) -> Self::Output {
        <T as crate::scalar::Variance>::finalize(state, self.ddof, nitems, false)
    }
    const SUPPORTS_EMPTY: bool = false;
    const PREFER_TREE_MERGE: bool = true;
    const NEEDS_INDICES: bool = false;
}

define_reduction_op!(
    /// Reduces one or more dims by computing the standard deviation.
    ///
    /// The standard deviation is real-valued and follows the same dtype rules as [`Variance`]:
    /// integer and `bool8` inputs promote to `f64`, real floats keep their width, and complex
    /// inputs reduce to their real component type (`Complex<f32> -> f32`, `Complex<f64> ->
    /// f64`), computing `sqrt(E[|x - mean|^2])`.
    ///
    /// Equivalent to `sqrt(variance)`. The `ddof` parameter has the same meaning as in
    /// [`Variance`]: use `ddof=0` for population std and `ddof=1` for sample std.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::std()`](crate::Array::std).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::{array, Ix0, Ix1};
    ///
    /// let nd = array![[1i32, 2, 3], [4, 5, 6]];
    ///
    /// // Population std (ddof=0) of all elements -> f64
    /// let std_all = Array::compact_ndarray(&nd)?
    ///     .std((0, 1), 0.0).to_ndarray::<Ix0>()?;
    /// assert!((std_all[[]] - 1.7078).abs() < 0.001);
    ///
    /// // Sample std (ddof=1) along dim 0 -> shape [3]
    /// let col_stds = Array::compact_ndarray(&nd)?
    ///     .std(0, 1.0).to_ndarray::<Ix1>()?;
    /// assert!((col_stds[[0]] - 2.1213).abs() < 0.001);
    /// # Ok::<(), jix::Error>(())
    /// ```
    StandardDeviation,
    StandardDeviationKernel { ddof: f64 },
    where {
        S: ArrayStorageTyped,
        S::Item: crate::scalar::Variance<Output: Dtyped>,
    }
    output = <S::Item as crate::scalar::Variance>::Output,
);
impl<T> ReductionOpKernel<T> for StandardDeviationKernel
where
    T: crate::scalar::Variance,
{
    type Output = <T as crate::scalar::Variance>::Output;
    type State = <T as crate::scalar::Variance>::State;

    #[inline(always)]
    fn init_state(&self, init_item: Option<(T, Option<u64>)>) -> Self::State {
        let mut state = <T as crate::scalar::Variance>::init();
        if let Some((item, _idx)) = init_item {
            state = self.update_state(state, item, None);
        }
        state
    }
    #[inline(always)]
    fn init_state_bulk<const N: usize, S: fearless_simd::Simd>(
        &self,
        items: [T; N],
        _indices: Option<[u64; N]>,
        simd: S,
    ) -> [Self::State; N] {
        // somehow this helps the compiler optimize the loop better, avoid lane shuffles
        let states = array_from_fn_inline(
            #[inline(always)]
            |_| <T as crate::scalar::Variance>::init(),
        );
        <T as crate::scalar::Variance>::update_bulk(states, items, simd)
    }
    #[inline(always)]
    fn update_state(&self, state: Self::State, item: T, _idx: Option<u64>) -> Self::State {
        <T as crate::scalar::Variance>::update(state, item)
    }
    #[inline(always)]
    fn update_state_bulk<const N: usize, S: Simd>(
        &self,
        states: [Self::State; N],
        items: [T; N],
        _indices: Option<[u64; N]>,
        simd: S,
    ) -> [Self::State; N] {
        <T as crate::scalar::Variance>::update_bulk(states, items, simd)
    }
    #[inline(always)]
    fn merge_states(&self, a: Self::State, b: Self::State) -> Self::State {
        <T as crate::scalar::Variance>::merge_states(a, b)
    }
    #[inline(always)]
    fn finalize_state(&self, state: Self::State, nitems: u64) -> Self::Output {
        <T as crate::scalar::Variance>::finalize(state, self.ddof, nitems, true)
    }
    const SUPPORTS_EMPTY: bool = false;
    const PREFER_TREE_MERGE: bool = true;
    const NEEDS_INDICES: bool = false;
}

impl<S> Array<S>
where
    S: ArrayStorage,
{
    define_array_reduction_method!(
        var: Variance,
        where {
            S: ArrayStorageTyped,
            S::Item: crate::scalar::Variance<Output: Dtyped>,
        },
        extra_args = (ddof: f64)
    );
    define_array_reduction_method!(
        std: StandardDeviation,
        where {
            S: ArrayStorageTyped,
            S::Item: crate::scalar::Variance<Output: Dtyped>,
        },
        extra_args = (ddof: f64)
    );
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
mod tests {

    use ndarray::{array, Ix0, Ix1};

    #[cfg(feature = "half")]
    use crate::scalar::f16;
    #[cfg(feature = "num-complex")]
    #[allow(non_camel_case_types)]
    type complex_f64 = crate::scalar::Complex<f64>;

    use crate::array::Array;

    #[test]
    fn variance() {
        let a = Array::compact_ndarray(&array![[1i32, 2, 3], [4, 5, 6]]).unwrap();
        let var_all = a.view().var((0, 1), 0.0).to_ndarray::<Ix0>().unwrap();
        assert!((var_all[[]] - 2.9166).abs() < 0.001);
        let var_col = a.view().var(0, 0.0).to_ndarray::<Ix1>().unwrap();
        assert!((var_col[[0]] - 2.25).abs() < 0.001);
        let var_row = a.view().var(1, 0.0).to_ndarray::<Ix1>().unwrap();
        assert!((var_row[[0]] - 0.6666).abs() < 0.001);
    }
    #[test]
    fn std() {
        let a = Array::compact_ndarray(&array![[7i32, 8, 9], [4, 5, 6]]).unwrap();
        let std_all = a.view().std((0, 1), 0.0).to_ndarray::<Ix0>().unwrap();
        assert!((std_all[[]] - 1.7078).abs() < 0.001);
        let std_col = a.view().std(0, 0.0).to_ndarray::<Ix1>().unwrap();
        assert!((std_col[[0]] - 1.5).abs() < 0.001);
        let std_row = a.view().std(1, 0.0).to_ndarray::<Ix1>().unwrap();
        assert!((std_row[[0]] - 0.8164).abs() < 0.001);
    }

    #[test]
    fn std_of_narrow_floats_where_only_the_variance_overflows_or_underflows() {
        let std = |a: [f32; 2]| {
            let a = Array::compact_ndarray(&ndarray::arr1(&a)).unwrap();
            a.std(0usize, 0.0).to_ndarray::<Ix0>().unwrap()[[]]
        };
        assert_eq!(std([-2e19, 2e19]), 2e19);
        assert_eq!(std([1e-25, 2e-25]), 5e-26);
        #[cfg(feature = "half")]
        {
            let a = array![f16::from_f32(-300.0), f16::from_f32(300.0)];
            let a = Array::compact_ndarray(&a).unwrap();
            let std = a.std(0usize, 0.0).to_ndarray::<Ix0>().unwrap()[[]];
            assert_eq!(std, f16::from_f32(300.0));
        }
    }

    #[test]
    fn var_std_of_nan_or_inf_is_nan() {
        let var = |a: ndarray::Array1<f64>| {
            let a = Array::compact_ndarray(&a).unwrap();
            a.var(0usize, 0.0).to_ndarray::<Ix0>().unwrap()[[]]
        };
        assert!(var(array![f64::NAN, 1.0]).is_nan());
        assert!(var(array![1.0, f64::NAN, 2.0]).is_nan());
        assert!(var(array![f64::INFINITY, 1.0]).is_nan());
        assert!(var(array![1.0, f64::NEG_INFINITY]).is_nan());
        let a = Array::compact_ndarray(&array![1.0f64, f64::NAN, 2.0]).unwrap();
        assert!(a.std(0usize, 0.0).to_ndarray::<Ix0>().unwrap()[[]].is_nan());
    }

    #[test]
    fn var_std_overflowing_deviation_is_inf() {
        let a = Array::compact_ndarray(&array![1e308f64, -1e308]).unwrap();
        assert_eq!(
            a.view().var(0usize, 0.0).to_ndarray::<Ix0>().unwrap()[[]],
            f64::INFINITY
        );
        assert_eq!(
            a.std(0usize, 0.0).to_ndarray::<Ix0>().unwrap()[[]],
            f64::INFINITY
        );
        #[cfg(feature = "num-complex")]
        {
            let a = Array::compact_ndarray(&array![
                complex_f64::new(0.0, 1e308),
                complex_f64::new(0.0, -1e308)
            ])
            .unwrap();
            assert_eq!(
                a.std(0usize, 0.0).to_ndarray::<Ix0>().unwrap()[[]],
                f64::INFINITY
            );
        }
    }
}
