use fearless_simd::Simd;

use crate::ops::prelude::*;
use crate::ops::reduction::common::{
    define_array_reduction_method, define_reduction_op, ReductionOpKernel,
};

pub(crate) mod traits {
    use crate::scalar::bool8;
    #[cfg(feature = "half")]
    use crate::scalar::f16;

    use crate::util::array2_map_inline;

    macro_rules! define_arg_trait {
        ($Trait:ident, $what:literal) => {
            #[doc = concat!("Scalar kernel trait for the `arg", $what, "` reduction: the position of the ", $what, "imum.")]
            ///
            /// The state is `(index, value)` of the running best. `NaN` propagates: the first
            /// `NaN` is the best, beating every number and every later `NaN`. On ties the earlier
            /// index wins. Both as in NumPy.
            pub trait $Trait {
                /// Fold `item`, at stream position `idx`, into `state`.
                fn update(state: (u64, Self), item: Self, idx: u64) -> (u64, Self);
                /// [`update`](Self::update) on each lane, vectorized with `simd` where the types
                /// have SIMD support.
                #[inline(always)]
                fn update_bulk<const N: usize, S: fearless_simd::Simd>(
                    states: [(u64, Self); N],
                    items: [Self; N],
                    indices: [u64; N],
                    simd: S,
                ) -> [(u64, Self); N]
                where
                    Self: Sized
                {
                    let _ = simd;
                    array2_map_inline(
                        states,
                        items,
                        #[inline(always)]
                        |i, state, item| Self::update(state, item, indices[i]),
                    )
                }
                /// Combine two states folded over disjoint subsets of the stream.
                fn merge_states(a: (u64, Self), b: (u64, Self)) -> (u64, Self);
            }
        };
    }
    define_arg_trait!(ArgMax, "max");
    define_arg_trait!(ArgMin, "min");

    // `$not_better` is `>=` for the max (`<=` for the min): `x` beats `y` if `y` is not `NaN` and
    // `!(y >= x)`, that is if `x > y` or `x` is `NaN`. Two comparisons, as many as a `NaN`-blind
    // `x > y || x != x`; integers fold to `x > y`.
    macro_rules! impl_arg {
        ($Trait:ident, $not_better:tt, [$($t:ty),*]) => {$(
            #[allow(clippy::eq_op, clippy::neg_cmp_op_on_partial_ord)]
            impl $Trait for $t {
                #[inline(always)]
                fn update(state: (u64, Self), item: Self, idx: u64) -> (u64, Self) {
                    // Items come in stream order: a tie keeps the earlier one.
                    if state.1 == state.1 && !(state.1 $not_better item) {
                        (idx, item)
                    } else {
                        state
                    }
                }
                #[inline(always)]
                fn merge_states(a: (u64, Self), b: (u64, Self)) -> (u64, Self) {
                    let ((ai, av), (bi, bv)) = (a, b);
                    let b_better = av == av && !(av $not_better bv);
                    let a_better = bv == bv && !(bv $not_better av);
                    // Neither is better on equal values, or two `NaN`s: the lower index.
                    if b_better || (!a_better && bi < ai) { b } else { a }
                }
            }
        )*};
    }
    impl_arg!(ArgMax, >=, [bool8, i8, i16, i32, i64, u8, u16, u32, u64, f32, f64]);
    impl_arg!(ArgMin, <=, [bool8, i8, i16, i32, i64, u8, u16, u32, u64, f32, f64]);
    #[cfg(feature = "half")]
    impl_arg!(ArgMax, >=, [f16]);
    #[cfg(feature = "half")]
    impl_arg!(ArgMin, <=, [f16]);
}

define_reduction_op!(
    /// Reduces one or more dims by taking the maximum element.
    ///
    /// For **float** types, `NaN` is propagated: if any element is `NaN`, the result
    /// is `NaN`. This matches the element-wise [`Maximum`](crate::ops::Maximum) op and
    /// `numpy.max` (not `numpy.nanmax`, which would ignore `NaN`).
    ///
    /// The result is a lazy view.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::max()`](crate::Array::max).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::{array, Ix0};
    ///
    /// let nd = array![[1i32, 2, 3], [4, 5, 6]];
    ///
    /// // Reduce all dims -> scalar
    /// let scalar = Array::compact_ndarray(&nd)?
    ///     .max((0, 1)).to_ndarray::<Ix0>()?;
    /// assert_eq!(scalar[[]], 6);
    ///
    /// // Reduce dim 0 -> shape [3]
    /// let col_max = Array::compact_ndarray(&nd)?
    ///     .max(0).to_ndarray()?;
    /// assert_eq!(col_max, array![4, 5, 6]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Max,
    MaxKernel,
    where {
        S: ArrayStorageTyped,
        S::Item: crate::scalar::Maximum<Output = S::Item> + Dtyped,
    }
    output = S::Item,
);
impl<T> ReductionOpKernel<T> for MaxKernel
where
    T: crate::scalar::Maximum<Output = T>,
{
    type Output = T;
    type State = T;

    #[inline(always)]
    fn init_state(&self, init_item: Option<(T, Option<u64>)>) -> Self::State {
        init_item.unwrap().0
    }
    #[inline(always)]
    fn update_state(&self, state: Self::State, item: T, _idx: Option<u64>) -> Self::State {
        state.maximum(item)
    }
    #[inline(always)]
    fn update_state_bulk<const N: usize, S: Simd>(
        &self,
        states: [Self::State; N],
        items: [T; N],
        _indices: Option<[u64; N]>,
        simd: S,
    ) -> [Self::State; N] {
        <T as crate::scalar::Maximum>::maximum_bulk(states, items, simd)
    }
    #[inline(always)]
    fn merge_states(&self, a: Self::State, b: Self::State) -> Self::State {
        a.maximum(b)
    }
    #[inline(always)]
    fn finalize_state(&self, state: Self::State, _nitems: u64) -> Self::Output {
        state
    }
    const SUPPORTS_EMPTY: bool = false;
    const NEEDS_FINALIZE: bool = false;
    const PREFER_TREE_MERGE: bool = false;
    const NEEDS_INDICES: bool = false;
}

define_reduction_op!(
    /// Reduces one or more dims by taking the minimum element.
    ///
    /// For **float** types, `NaN` is propagated: if any element is `NaN`, the result
    /// is `NaN`. This matches the element-wise [`Minimum`](crate::ops::Minimum) op and
    /// `numpy.min` (not `numpy.nanmin`, which would ignore `NaN`).
    ///
    /// The result is a lazy view.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::min()`](crate::Array::min).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::{array, Ix0};
    ///
    /// let nd = array![[1i32, 2, 3], [4, 5, 6]];
    ///
    /// // Reduce all dims -> scalar
    /// let scalar = Array::compact_ndarray(&nd)?
    ///     .min((0, 1)).to_ndarray::<Ix0>()?;
    /// assert_eq!(scalar[[]], 1);
    ///
    /// // Reduce dim 0 -> shape [3]
    /// let col_min = Array::compact_ndarray(&nd)?
    ///     .min(0).to_ndarray()?;
    /// assert_eq!(col_min, array![1, 2, 3]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Min,
    MinKernel,
    where {
        S: ArrayStorageTyped,
        S::Item: crate::scalar::Minimum<Output = S::Item> + Dtyped,
    }
    output = S::Item,
);
impl<T> ReductionOpKernel<T> for MinKernel
where
    T: crate::scalar::Minimum<Output = T>,
{
    type Output = T;
    type State = T;

    #[inline(always)]
    fn init_state(&self, init_item: Option<(T, Option<u64>)>) -> Self::State {
        init_item.unwrap().0
    }
    #[inline(always)]
    fn update_state(&self, state: Self::State, item: T, _idx: Option<u64>) -> Self::State {
        state.minimum(item)
    }
    #[inline(always)]
    fn update_state_bulk<const N: usize, S: Simd>(
        &self,
        states: [Self::State; N],
        items: [T; N],
        _indices: Option<[u64; N]>,
        simd: S,
    ) -> [Self::State; N] {
        <T as crate::scalar::Minimum>::minimum_bulk(states, items, simd)
    }
    #[inline(always)]
    fn merge_states(&self, a: Self::State, b: Self::State) -> Self::State {
        a.minimum(b)
    }
    #[inline(always)]
    fn finalize_state(&self, state: Self::State, _nitems: u64) -> Self::Output {
        state
    }
    const SUPPORTS_EMPTY: bool = false;
    const NEEDS_FINALIZE: bool = false;
    const PREFER_TREE_MERGE: bool = false;
    const NEEDS_INDICES: bool = false;
}

define_reduction_op!(
    /// Reduces a single dim by returning the index of the maximum element.
    ///
    /// Output dtype is `u64`. As in `numpy.argmax`, ties return the *first* index, and for
    /// **float** types `NaN` propagates: if any element along the reduced dim is `NaN`, the
    /// index of the first `NaN` is returned.
    ///
    /// The result is a lazy view.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::argmax()`](crate::Array::argmax).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let nd = array![[1i32, 5, 3], [4, 2, 6]];
    ///
    /// // Index of max along dim 1 (per row) -> shape [2]
    /// let idx = Array::compact_ndarray(&nd)?
    ///     .argmax(1).to_ndarray()?;
    /// assert_eq!(idx, array![1, 2]); // max of row 0 at col 1, row 1 at col 2
    ///
    /// // Index of max along dim 0 (per column) -> shape [3]
    /// let col_idx = Array::compact_ndarray(&nd)?
    ///     .argmax(0).to_ndarray()?;
    /// assert_eq!(col_idx, array![1, 0, 1]); // max of col 0 at row 1, col 1 at row 0, col 2 at row 1
    /// # Ok::<(), jix::Error>(())
    /// ```
    ArgMax,
    ArgMaxKernel,
    where {
        S: ArrayStorageTyped,
        S::Item: crate::scalar::ArgMax,
    }
    output = u64,
    single_dim,
);
impl<T> ReductionOpKernel<T> for ArgMaxKernel
where
    T: crate::scalar::ArgMax,
{
    type Output = u64;
    /// `(best_idx, best_val)`.
    type State = (u64, T);

    #[inline(always)]
    fn init_state(&self, init_item: Option<(T, Option<u64>)>) -> Self::State {
        let (item, idx) = init_item.unwrap();
        (idx.unwrap(), item)
    }
    #[inline(always)]
    fn update_state(&self, state: Self::State, item: T, idx: Option<u64>) -> Self::State {
        <T as crate::scalar::ArgMax>::update(state, item, idx.unwrap())
    }
    #[inline(always)]
    fn update_state_bulk<const N: usize, S: Simd>(
        &self,
        states: [Self::State; N],
        items: [T; N],
        indices: Option<[u64; N]>,
        simd: S,
    ) -> [Self::State; N] {
        <T as crate::scalar::ArgMax>::update_bulk(states, items, indices.unwrap(), simd)
    }
    #[inline(always)]
    fn merge_states(&self, a: Self::State, b: Self::State) -> Self::State {
        <T as crate::scalar::ArgMax>::merge_states(a, b)
    }
    #[inline(always)]
    fn finalize_state(&self, state: Self::State, _nitems: u64) -> Self::Output {
        state.0
    }
    const SUPPORTS_EMPTY: bool = false;
    const PREFER_TREE_MERGE: bool = false;
    const ONE_CELL_LANES: Option<usize> = Some(if cfg!(target_arch = "x86") { 8 } else { 16 });
}

define_reduction_op!(
    /// Reduces a single dim by returning the index of the minimum element.
    ///
    /// Output dtype is `u64`. As in `numpy.argmin`, ties return the *first* index, and for
    /// **float** types `NaN` propagates: if any element along the reduced dim is `NaN`, the
    /// index of the first `NaN` is returned.
    ///
    /// The result is a lazy view.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::argmin()`](crate::Array::argmin).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let nd = array![[1i32, 5, 3], [4, 2, 6]];
    ///
    /// // Index of min along dim 1 (per row) -> shape [2]
    /// let idx = Array::compact_ndarray(&nd)?
    ///     .argmin(1).to_ndarray()?;
    /// assert_eq!(idx, array![0, 1]); // min of row 0 at col 0, row 1 at col 1
    ///
    /// // Index of min along dim 0 (per column) -> shape [3]
    /// let col_idx = Array::compact_ndarray(&nd)?
    ///     .argmin(0).to_ndarray()?;
    /// assert_eq!(col_idx, array![0, 1, 0]); // min of col 0 at row 0, col 1 at row 1, col 2 at row 0
    /// # Ok::<(), jix::Error>(())
    /// ```
    ArgMin,
    ArgMinKernel,
    where {
        S: ArrayStorageTyped,
        S::Item: crate::scalar::ArgMin,
    }
    output = u64,
    single_dim,
);
impl<T> ReductionOpKernel<T> for ArgMinKernel
where
    T: crate::scalar::ArgMin,
{
    type Output = u64;
    /// `(best_idx, best_val)`.
    type State = (u64, T);

    #[inline(always)]
    fn init_state(&self, init_item: Option<(T, Option<u64>)>) -> Self::State {
        let (item, idx) = init_item.unwrap();
        (idx.unwrap(), item)
    }
    #[inline(always)]
    fn update_state(&self, state: Self::State, item: T, idx: Option<u64>) -> Self::State {
        <T as crate::scalar::ArgMin>::update(state, item, idx.unwrap())
    }
    #[inline(always)]
    fn update_state_bulk<const N: usize, S: Simd>(
        &self,
        states: [Self::State; N],
        items: [T; N],
        indices: Option<[u64; N]>,
        simd: S,
    ) -> [Self::State; N] {
        <T as crate::scalar::ArgMin>::update_bulk(states, items, indices.unwrap(), simd)
    }
    #[inline(always)]
    fn merge_states(&self, a: Self::State, b: Self::State) -> Self::State {
        <T as crate::scalar::ArgMin>::merge_states(a, b)
    }
    #[inline(always)]
    fn finalize_state(&self, state: Self::State, _nitems: u64) -> Self::Output {
        state.0
    }
    const SUPPORTS_EMPTY: bool = false;
    const PREFER_TREE_MERGE: bool = false;
    const ONE_CELL_LANES: Option<usize> = Some(if cfg!(target_arch = "x86") { 8 } else { 16 });
}

impl<S> Array<S>
where
    S: ArrayStorage,
{
    define_array_reduction_method!(
        max: Max,
        where {
            S: ArrayStorageTyped,
            S::Item: crate::scalar::Maximum<Output = S::Item> + Dtyped,
        }
    );
    define_array_reduction_method!(
        min: Min,
        where {
            S: ArrayStorageTyped,
            S::Item: crate::scalar::Minimum<Output = S::Item> + Dtyped,
        }
    );
    define_array_reduction_method!(
        argmax: ArgMax,
        where {
            S: ArrayStorageTyped,
            S::Item: crate::scalar::ArgMax,
        },
        single_dim
    );
    define_array_reduction_method!(
        argmin: ArgMin,
        where {
            S: ArrayStorageTyped,
            S::Item: crate::scalar::ArgMin,
        },
        single_dim
    );
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
mod tests {

    use crate::scalar::bool8;

    use ndarray::{array, IxDyn};

    #[cfg(feature = "half")]
    use crate::scalar::f16;

    use crate::array::Array;

    use crate::ops::reduction::common::tests::*;

    test_reduction!(
        max,
        |items| { items.reduce(|m, x| if x > m { x } else { m }).unwrap() },
        [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool8],
        any_strategy,
        #[cfg(feature = "half")]
        [f16]
    );
    #[test]
    fn min_concrete() {
        // i32: negatives and the signed dtype MIN/MAX edges; dim 0, dim 1, and all dims.
        // A non-default block shape ([2, 2]) also crosses a block boundary on both the
        // reduced and the kept dim.
        let nd = array![[i32::MIN, 3, -5], [7, i32::MAX, -9]];
        let za = Array::compact_ndarray_with(&nd, crate::util::arr_params(&[2, 2])).unwrap();
        // dim 0 (per column): min(MIN,7)=MIN; min(3,MAX)=3; min(-5,-9)=-9.
        crate::util::assert_array_matches(&za.view().min(0usize), &array![i32::MIN, 3, -9]);
        // dim 1 (per row): min(MIN,3,-5)=MIN; min(7,MAX,-9)=-9.
        crate::util::assert_array_matches(&za.view().min(1usize), &array![i32::MIN, -9]);
        // all dims: overall minimum is i32::MIN.
        crate::util::assert_array_matches(&za.view().min((0, 1)), &ndarray::arr0(i32::MIN));

        // u8: the unsigned dtype MIN (0) / MAX (255) edges - distinct from the signed case
        // above since unsigned types have no negatives.
        let ndu = array![[0u8, 200, 255], [10, 255, 5]];
        let zau = Array::compact_ndarray(&ndu).unwrap();
        crate::util::assert_array_matches(&zau.view().min(0usize), &array![0u8, 200, 5]);
        crate::util::assert_array_matches(&zau.view().min(1usize), &array![0u8, 5]);
        crate::util::assert_array_matches(&zau.view().min((0, 1)), &ndarray::arr0(0u8));

        // f32: negatives, zero, +/-infinity. `NaN` is deliberately not used here: for float
        // outputs `ReductionCompare` compares via `assert_array_matches_approx`, whose
        // `ApproxEq` - like IEEE 754 - never treats `NaN` as approximately equal to `NaN`, so
        // a `NaN` result can't be asserted through that helper.
        let ndf = array![
            [1.5f32, -2.0, 0.0],
            [f32::INFINITY, f32::NEG_INFINITY, -3.5]
        ];
        let zaf = Array::compact_ndarray(&ndf).unwrap();
        crate::util::assert_array_matches_approx(
            &zaf.view().min(0usize),
            &array![1.5f32, f32::NEG_INFINITY, -3.5],
            1e-3,
            1e-1,
        );
        crate::util::assert_array_matches_approx(
            &zaf.view().min(1usize),
            &array![-2.0f32, f32::NEG_INFINITY],
            1e-3,
            1e-1,
        );
        crate::util::assert_array_matches_approx(
            &zaf.view().min((0, 1)),
            &ndarray::arr0(f32::NEG_INFINITY),
            1e-3,
            1e-1,
        );
    }

    #[test]
    fn argmax_concrete() {
        // i32 with negatives and a deliberate 2-way tie on both dims: the kernel must return
        // the FIRST index among tied maxima, not an arbitrary one.
        let nd = array![[5i32, 5, 2], [5, -1, -8]];
        let za = Array::compact_ndarray(&nd).unwrap();
        // dim 0 (per column): col 0 ties row 0/row 1 at 5 -> first row (0) wins.
        crate::util::assert_array_matches(&za.view().argmax(0usize), &array![0u64, 0, 0]);
        // dim 1 (per row): row 0 ties col 0/col 1 at 5 -> first col (0) wins.
        crate::util::assert_array_matches(&za.view().argmax(1usize), &array![0u64, 0]);

        // f32 path, same tie-break rule, negatives included. (`NaN`: `argmax_argmin_first_nan`.)
        let ndf = array![[2.5f32, 2.5, -1.0], [3.0, 3.0, 0.0]];
        let zaf = Array::compact_ndarray(&ndf).unwrap();
        crate::util::assert_array_matches(&zaf.view().argmax(0usize), &array![1u64, 1, 1]);
        // row 0 ties col 0/col 1 at 2.5, row 1 ties col 0/col 1 at 3.0 -> first col (0) wins.
        crate::util::assert_array_matches(&zaf.view().argmax(1usize), &array![0u64, 0]);
    }

    #[test]
    fn argmin_concrete() {
        // Same arrays as `argmax_concrete`, exercising the minimum side of the tie-break rule.
        let nd = array![[5i32, 5, 2], [5, -1, -8]];
        let za = Array::compact_ndarray(&nd).unwrap();
        // dim 0 (per column): col 0 ties row 0/row 1 at 5 -> first row (0) wins.
        crate::util::assert_array_matches(&za.view().argmin(0usize), &array![0u64, 1, 1]);
        // dim 1 (per row): both rows have a unique minimum (2, then -8).
        crate::util::assert_array_matches(&za.view().argmin(1usize), &array![2u64, 2]);

        let ndf = array![[2.5f32, 2.5, -1.0], [3.0, 3.0, 0.0]];
        let zaf = Array::compact_ndarray(&ndf).unwrap();
        // col 0/col 1 tie at 2.5 vs. 3.0 -> row 0 (index 0) wins both.
        crate::util::assert_array_matches(&zaf.view().argmin(0usize), &array![0u64, 0, 0]);
        crate::util::assert_array_matches(&zaf.view().argmin(1usize), &array![2u64, 2]);
    }

    /// The first `NaN` wins, as in NumPy: whatever lane, run or tile each `NaN` lands in.
    #[test]
    fn argmax_argmin_first_nan() {
        /// The index of the first `NaN`, else of the first item no other `beats`.
        #[allow(clippy::eq_op)]
        fn expected<T: PartialOrd>(row: ndarray::ArrayView1<T>, beats: fn(&T, &T) -> bool) -> u64 {
            let best = (1..row.len()).fold(0, |b, i| if beats(&row[i], &row[b]) { i } else { b });
            row.iter().position(|x| x != x).unwrap_or(best) as u64
        }
        macro_rules! check {
            ($t:ty) => {{
                for n in [1usize, 2, 3, 7, 8, 9, 16, 31, 33, 64, 65, 100, 513, 2049] {
                    let nan_sets = [
                        vec![],
                        vec![0],
                        vec![n - 1],
                        vec![n / 2],
                        vec![n / 3, n / 2, n - 1],
                        vec![5, 33],
                        vec![1, 39],
                        vec![2, 3],
                        vec![40, 9],
                    ];
                    // 1, 3 and 37 cells: fewer and more than a vector of states.
                    for (nans, ncols) in nan_sets.iter().flat_map(|s| [(s, 1), (s, 3), (s, 37)]) {
                        // Row `c`: values with ties, and the NaNs shifted by `c`.
                        let rows = ndarray::Array2::from_shape_fn((ncols, n), |(c, i)| {
                            let is_nan = nans.iter().any(|&j| j < n && (j + c) % n == i);
                            if is_nan {
                                <$t>::NAN
                            } else {
                                ((i * 7919 + c) % 13) as $t - 6.0
                            }
                        });
                        let expected = |beats: fn(&$t, &$t) -> bool| -> Vec<u64> {
                            rows.rows()
                                .into_iter()
                                .map(|r| expected(r, beats))
                                .collect()
                        };
                        let (exp_max, exp_min) = (expected(<$t>::gt), expected(<$t>::lt));
                        // The rows contiguous (one cell at a time), and as columns (across cells).
                        for (nd, dim) in [(rows.clone(), 1), (rows.t().to_owned(), 0)] {
                            for block in [[4096, 4096], [7, 5]] {
                                let block = [block[0].min(nd.nrows()), block[1].min(nd.ncols())];
                                let params = crate::util::arr_params(&block);
                                let za = Array::compact_ndarray_with(&nd, params).unwrap();
                                let what = format!(
                                    "{} n={n} nans={nans:?} dim={dim} block={block:?}",
                                    stringify!($t)
                                );
                                let got_max = za.view().argmax(dim).to_ndarray::<IxDyn>().unwrap();
                                let got_min = za.view().argmin(dim).to_ndarray::<IxDyn>().unwrap();
                                assert_eq!(got_max.as_slice().unwrap(), exp_max, "argmax {what}");
                                assert_eq!(got_min.as_slice().unwrap(), exp_min, "argmin {what}");
                            }
                        }
                    }
                }
            }};
        }
        check!(f32);
        check!(f64);
    }
}
