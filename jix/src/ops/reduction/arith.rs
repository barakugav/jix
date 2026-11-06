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

    macro_rules! update_bulk_simd {
        ($Trait:ident, $Op:ident::$op_bulk:ident) => {
            update_bulk_simd!($Trait, $Op::$op_bulk, disable_on_x86_64 = []);
        };
        ($Trait:ident, $Op:ident::$op_bulk:ident, disable_on_x86_64 = [$($disable_on_x86_64:ident),*]) => {
            #[inline(always)]
            fn update_bulk<const N: usize, S: fearless_simd::Simd>(
                states: [Self::Output; N],
                items: [Self; N],
                simd: S,
            ) -> [Self::Output; N] {
                use crate::scalar::simd::{simd_level, SimdLevel};
                let large_cast = size_of::<Self::Output>() >= 4 * size_of::<Self>();
                let has_simd_cast = !matches!(simd_level(simd), SimdLevel::Sse2 | SimdLevel::Neon);
                let inefficient_simd_cast = large_cast && !has_simd_cast;
                let disable_on_x86_64 = (cfg!(target_arch = "x86_64")
                    && (false $(|| simd_level(simd) == SimdLevel::$disable_on_x86_64)*));
                if inefficient_simd_cast || disable_on_x86_64 {
                    return states.map_enumerate(
                        #[inline(always)]
                        |i, state| <Self as $Trait>::update(state, items[i]),
                    );
                }
                let items = items.map_inline(
                    #[inline(always)]
                    |x| <_ as crate::scalar::Cast<Self::Output>>::cast(x),
                );
                <Self::Output as crate::scalar::$Op>::$op_bulk(states, items, simd)
            }
        };
    }

    /// Scalar kernel trait for the element-wise `sum` reduction.
    ///
    /// Integer types accumulate into a wider output to reduce overflow risk (`i8..=i64 ->
    /// i64`; `u8..=u64` and `bool8 -> u64`), which wraps on overflow. Floating-point and complex
    /// types keep their input width (`f16 -> f16`, `f32 -> f32`, `Complex<f32> -> Complex<f32>`,
    /// ...), matching NumPy.
    pub trait Sum {
        /// The sum element type: `i64`/`u64` for integers and `bool8`, otherwise the input type.
        type Output;
        /// Whether summing this type is exact: `true` for the integer accumulators, whose
        /// result does not depend on how the stream is grouped, `false` for the float and
        /// complex ones, where every partial sum rounds.
        const IS_PRECISE: bool;
        /// Return the initial accumulator (zero).
        fn init() -> Self::Output;
        /// Return the sum of an empty stream (zero).
        fn empty() -> Self::Output;
        /// Fold `item` into the running sum.
        fn update(state: Self::Output, item: Self) -> Self::Output;
        /// [`update`](Self::update) on each lane, vectorized with `simd` where the types have
        /// SIMD support.
        #[inline(always)]
        fn update_bulk<const N: usize, S: fearless_simd::Simd>(
            states: [Self::Output; N],
            items: [Self; N],
            simd: S,
        ) -> [Self::Output; N]
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
        /// Combine two partial sums (used to merge interleaved lane accumulators).
        fn merge_states(a: Self::Output, b: Self::Output) -> Self::Output;
    }

    macro_rules! impl_sum {
        ($item_ty:ty, $output_ty:ty, is_precise = $is_precise:expr, simd="enabled") => {
            impl_sum!($item_ty, $output_ty, is_precise = $is_precise, { update_bulk_simd!(Sum, Add::add_bulk); });
        };
        ($item_ty:ty, $output_ty:ty, is_precise = $is_precise:expr) => {
            impl_sum!($item_ty, $output_ty, is_precise = $is_precise, {});
        };
        ($item_ty:ty, $output_ty:ty, is_precise = $is_precise:expr, { $($bulk:tt)* }) => {
            impl Sum for $item_ty {
                type Output = $output_ty;

                const IS_PRECISE: bool = $is_precise;

                #[inline(always)]
                fn init() -> Self::Output {
                    <f32 as crate::scalar::Cast<Self::Output>>::cast(-0.0)
                }
                #[inline(always)]
                fn empty() -> Self::Output {
                    <f32 as crate::scalar::Cast<Self::Output>>::cast(0.0)
                }
                #[inline(always)]
                fn update(state: Self::Output, item: Self) -> Self::Output {
                    let item = <_ as crate::scalar::Cast<Self::Output>>::cast(item);
                    crate::scalar::Add::add(state, item)
                }
                $($bulk)*
                #[inline(always)]
                fn merge_states(a: Self::Output, b: Self::Output) -> Self::Output {
                    crate::scalar::Add::add(a, b)
                }
            }
        };
    }
    impl_sum!(i8, i64, is_precise = true, simd = "enabled");
    impl_sum!(i16, i64, is_precise = true, simd = "enabled");
    impl_sum!(i32, i64, is_precise = true, simd = "enabled");
    impl_sum!(i64, i64, is_precise = true, simd = "enabled");
    impl_sum!(u8, u64, is_precise = true, simd = "enabled");
    impl_sum!(u16, u64, is_precise = true, simd = "enabled");
    impl_sum!(u32, u64, is_precise = true, simd = "enabled");
    impl_sum!(u64, u64, is_precise = true, simd = "enabled");
    #[cfg(feature = "half")]
    impl_sum!(f16, f16, is_precise = false);
    impl_sum!(f32, f32, is_precise = false, simd = "enabled");
    impl_sum!(f64, f64, is_precise = false, simd = "enabled");
    #[cfg(feature = "num-complex")]
    impl_sum!(Complex<f32>, Complex<f32>, is_precise = false);
    #[cfg(feature = "num-complex")]
    impl_sum!(Complex<f64>, Complex<f64>, is_precise = false);
    impl_sum!(bool8, u64, is_precise = true, simd = "enabled");

    /// Scalar kernel trait for the element-wise `product` reduction.
    ///
    /// Integer types accumulate into a wider output to reduce overflow risk (`i8..=i64 ->
    /// i64`; `u8..=u64 -> u64`), which wraps on overflow. Floating-point and complex types keep
    /// their input width (`f16 -> f16`, `f32 -> f32`, `Complex<f32> -> Complex<f32>`, ...),
    /// matching NumPy.
    pub trait Product {
        /// The product element type: `i64`/`u64` for integers, otherwise the input type.
        type Output;
        /// Return the initial accumulator (one).
        fn init() -> Self::Output;
        /// Fold `item` into the running product.
        fn update(state: Self::Output, item: Self) -> Self::Output;
        /// [`update`](Self::update) on each lane, vectorized with `simd` where the types have
        /// SIMD support.
        #[inline(always)]
        fn update_bulk<const N: usize, S: fearless_simd::Simd>(
            states: [Self::Output; N],
            items: [Self; N],
            simd: S,
        ) -> [Self::Output; N]
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
        /// Combine two partial products (used to merge interleaved lane accumulators).
        fn merge_states(a: Self::Output, b: Self::Output) -> Self::Output;
    }
    macro_rules! impl_product {
        ($item_ty:ty, $output_ty:ty, simd="enabled") => {
            impl_product!($item_ty, $output_ty, { update_bulk_simd!(Product, Mul::mul_bulk); });
        };
        ($item_ty:ty, $output_ty:ty, simd=disable_on_x86_64($($disable_on_x86_64:ident),*)) => {
            impl_product!($item_ty, $output_ty, {
                update_bulk_simd!(Product, Mul::mul_bulk, disable_on_x86_64 = [$($disable_on_x86_64),*]);
            });
        };
        ($item_ty:ty, $output_ty:ty) => {
            impl_product!($item_ty, $output_ty, {});
        };
        ($item_ty:ty, $output_ty:ty, { $($bulk:tt)* }) => {
            impl Product for $item_ty {
                type Output = $output_ty;

                #[inline(always)]
                fn init() -> Self::Output {
                    <i32 as crate::scalar::Cast<Self::Output>>::cast(1)
                }
                #[inline(always)]
                fn update(state: Self::Output, item: Self) -> Self::Output {
                    let item = <_ as crate::scalar::Cast<Self::Output>>::cast(item);
                    crate::scalar::Mul::mul(state, item)
                }
                $($bulk)*
                #[inline(always)]
                fn merge_states(a: Self::Output, b: Self::Output) -> Self::Output {
                    crate::scalar::Mul::mul(a, b)
                }
            }
        };
    }
    // static analysis showed 64-bit integer products are slower with SIMD up to AVX2
    impl_product!(i8, i64, simd = disable_on_x86_64(Sse2, Sse4_2));
    impl_product!(i16, i64, simd = disable_on_x86_64(Sse2, Sse4_2));
    impl_product!(i32, i64, simd = disable_on_x86_64(Sse2, Sse4_2));
    impl_product!(i64, i64, simd = disable_on_x86_64(Sse2, Sse4_2));
    impl_product!(u8, u64, simd = disable_on_x86_64(Sse2, Sse4_2));
    impl_product!(u16, u64, simd = disable_on_x86_64(Sse2, Sse4_2));
    impl_product!(u32, u64, simd = disable_on_x86_64(Sse2, Sse4_2));
    impl_product!(u64, u64, simd = disable_on_x86_64(Sse2, Sse4_2));
    #[cfg(feature = "half")]
    impl_product!(f16, f16);
    impl_product!(f32, f32, simd = "enabled");
    impl_product!(f64, f64, simd = "enabled");
    #[cfg(feature = "num-complex")]
    impl_product!(Complex<f32>, Complex<f32>);
    #[cfg(feature = "num-complex")]
    impl_product!(Complex<f64>, Complex<f64>);

    /// Scalar kernel trait for the element-wise `mean` reduction.
    ///
    /// The mean is computed as the sum divided by the count. Integer and `bool8` inputs promote
    /// to `f64`; floating-point and complex inputs keep their input width (`f16 -> f16`,
    /// `f32 -> f32`, `Complex<f32> -> Complex<f32>`, ...), matching NumPy.
    ///
    /// The count is tracked **outside** the accumulator: callers thread the number of
    /// folded items into [`finalize`](Self::finalize) themselves.
    pub trait Mean {
        /// The output element type: `f64` for integer and `bool8` inputs, otherwise the input type.
        type Output;
        /// Accumulator state - the running sum.
        type State;
        /// Whether the sum this mean accumulates is exact.
        const IS_PRECISE_SUM: bool;
        /// Return the initial (empty) accumulator.
        fn init() -> Self::State;
        /// Fold `item` into the running sum.
        fn update(state: Self::State, item: Self) -> Self::State;
        /// [`update`](Self::update) on each lane, vectorized with `simd` where the types have
        /// SIMD support.
        fn update_bulk<const N: usize, S: fearless_simd::Simd>(
            states: [Self::State; N],
            items: [Self; N],
            simd: S,
        ) -> [Self::State; N]
        where
            Self: Sized;
        /// Combine two partial sums (used to merge interleaved lane accumulators).
        fn merge_states(a: Self::State, b: Self::State) -> Self::State;
        /// Finalize `state` into the mean. Returns `None` if `nitems == 0`; otherwise
        /// returns `state / nitems` (cast to the output domain).
        fn finalize(state: Self::State, nitems: u64) -> Option<Self::Output>;
    }
    macro_rules! impl_mean {
        ($ItemT:ty, $MeanT:ty, $FinalizeT:ty, $OutputT:ty) => {
            impl Mean for $ItemT {
                type Output = $OutputT;
                type State = $MeanT;

                const IS_PRECISE_SUM: bool = <Self::State as Sum>::IS_PRECISE;

                #[inline(always)]
                fn init() -> Self::State {
                    <Self::State as Sum>::init()
                }
                #[inline(always)]
                fn update(state: Self::State, item: Self) -> Self::State {
                    let item = <Self as crate::scalar::Cast<Self::State>>::cast(item);
                    <Self::State as Sum>::update(state, item)
                }
                #[inline(always)]
                fn update_bulk<const N: usize, S: fearless_simd::Simd>(
                    states: [Self::State; N],
                    items: [Self; N],
                    simd: S,
                ) -> [Self::State; N] {
                    let items = <Self as crate::scalar::Cast<Self::State>>::cast_bulk(items, simd);
                    <Self::State as Sum>::update_bulk(states, items, simd)
                }
                #[inline(always)]
                fn merge_states(a: Self::State, b: Self::State) -> Self::State {
                    <Self::State as Sum>::merge_states(a, b)
                }
                #[inline(always)]
                fn finalize(state: Self::State, nitems: u64) -> Option<Self::Output> {
                    if nitems == 0 {
                        return None;
                    }
                    let state = <_ as crate::scalar::Cast<$FinalizeT>>::cast(state);
                    let mean = state / nitems as f64;
                    Some(<_ as crate::scalar::Cast<Self::Output>>::cast(mean))
                }
            }
        };
    }
    impl_mean!(i8, i64, f64, f64);
    impl_mean!(i16, i64, f64, f64);
    impl_mean!(i32, i64, f64, f64);
    impl_mean!(i64, f64, f64, f64);
    impl_mean!(u8, u64, f64, f64);
    impl_mean!(u16, u64, f64, f64);
    impl_mean!(u32, u64, f64, f64);
    impl_mean!(u64, f64, f64, f64);
    #[cfg(feature = "half")]
    impl_mean!(f16, f32, f64, f16);
    impl_mean!(f32, f64, f64, f32);
    impl_mean!(f64, f64, f64, f64);
    #[cfg(feature = "num-complex")]
    impl_mean!(Complex<f32>, Complex<f64>, Complex<f64>, Complex<f32>);
    #[cfg(feature = "num-complex")]
    impl_mean!(Complex<f64>, Complex<f64>, Complex<f64>, Complex<f64>);
    impl_mean!(bool8, u64, f64, f64);
}

define_reduction_op!(
    /// Reduces one or more dims by summing all elements along those dims.
    ///
    /// Supported dtypes and output dtype:
    ///
    /// | Input dtype | Output dtype |
    /// |-------------|--------------|
    /// | `i8`, `i16`, `i32`, `i64` | `i64` |
    /// | `u8`, `u16`, `u32`, `u64`, `bool8` | `u64` |
    /// | `f16` | `f16` |
    /// | `f32` | `f32` |
    /// | `f64` | `f64` |
    /// | `Complex<f32>` | `Complex<f32>` |
    /// | `Complex<f64>` | `Complex<f64>` |
    ///
    /// Integer inputs are widened to a 64-bit accumulator to reduce overflow on large
    /// reductions; floating-point and complex inputs keep their width, matching NumPy (except
    /// `bool8`, which jix sums into `u64` whereas NumPy uses `int64`).
    /// An empty reduction (zero elements along the reduced dims) returns `0`.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::sum()`](crate::Array::sum).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::{array, Ix0};
    ///
    /// let nd = array![[1i32, 2, 3], [4, 5, 6]];
    ///
    /// // Sum all elements -> i64
    /// let total = Array::compact_ndarray(&nd)?
    ///     .sum((0, 1)).to_ndarray::<Ix0>()?;
    /// assert_eq!(total[[]], 21);
    ///
    /// // Sum along dim 0 -> shape [3]
    /// let col_sums = Array::compact_ndarray(&nd)?
    ///     .sum(0).to_ndarray()?;
    /// assert_eq!(col_sums, array![5, 7, 9]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Sum,
    SumKernel,
    where {
        S: ArrayStorageTyped,
        S::Item: crate::scalar::Sum<Output: Dtyped>,
    }
    output = <S::Item as crate::scalar::Sum>::Output,
);
impl<T> ReductionOpKernel<T> for SumKernel
where
    T: crate::scalar::Sum,
{
    type Output = <T as crate::scalar::Sum>::Output;
    type State = <T as crate::scalar::Sum>::Output;

    #[inline(always)]
    fn init_state(&self, init_item: Option<(T, Option<u64>)>) -> Self::State {
        match init_item {
            Some((item, _idx)) => self.update_state(<T as crate::scalar::Sum>::init(), item, None),
            None => <T as crate::scalar::Sum>::empty(), // +0.0 and not init()'s -0.0
        }
    }
    #[inline(always)]
    fn update_state(&self, state: Self::State, item: T, _idx: Option<u64>) -> Self::State {
        <T as crate::scalar::Sum>::update(state, item)
    }
    #[inline(always)]
    fn update_state_bulk<const N: usize, S: Simd>(
        &self,
        states: [Self::State; N],
        items: [T; N],
        _indices: Option<[u64; N]>,
        simd: S,
    ) -> [Self::State; N] {
        <T as crate::scalar::Sum>::update_bulk(states, items, simd)
    }
    #[inline(always)]
    fn merge_states(&self, a: Self::State, b: Self::State) -> Self::State {
        <T as crate::scalar::Sum>::merge_states(a, b)
    }
    #[inline(always)]
    fn finalize_state(&self, state: Self::State, _nitems: u64) -> Self::Output {
        state
    }
    const SUPPORTS_EMPTY: bool = true;
    const NEEDS_FINALIZE: bool = false;
    const PREFER_TREE_MERGE: bool = !<T as crate::scalar::Sum>::IS_PRECISE;
    const NEEDS_INDICES: bool = false;
}

define_reduction_op!(
    /// Reduces one or more dims by multiplying all elements along those dims.
    ///
    /// Supported dtypes and output dtype:
    ///
    /// | Input dtype | Output dtype |
    /// |-------------|--------------|
    /// | `i8`, `i16`, `i32`, `i64` | `i64` |
    /// | `u8`, `u16`, `u32`, `u64` | `u64` |
    /// | `f16` | `f16` |
    /// | `f32` | `f32` |
    /// | `f64` | `f64` |
    /// | `Complex<f32>` | `Complex<f32>` |
    /// | `Complex<f64>` | `Complex<f64>` |
    ///
    /// Integer inputs are widened to a 64-bit accumulator to reduce overflow on large
    /// reductions; floating-point and complex inputs keep their width, matching NumPy. `bool8`
    /// is not supported (NumPy would promote it to `int64`).
    /// An empty reduction (zero elements along the reduced dims) returns `1`.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::product()`](crate::Array::product).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::{array, Ix0};
    ///
    /// let nd = array![[1i32, 2, 3], [4, 5, 6]];
    ///
    /// // Product of all elements -> i64
    /// let total = Array::compact_ndarray(&nd)?
    ///     .product((0, 1)).to_ndarray::<Ix0>()?;
    /// assert_eq!(total[[]], 720);
    ///
    /// // Product along dim 0 -> shape [3]
    /// let col_products = Array::compact_ndarray(&nd)?
    ///     .product(0).to_ndarray()?;
    /// assert_eq!(col_products, array![4, 10, 18]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Product,
    ProductKernel,
    where {
        S: ArrayStorageTyped,
        S::Item: crate::scalar::Product<Output: Dtyped>,
    }
    output = <S::Item as crate::scalar::Product>::Output,
);
impl<T> ReductionOpKernel<T> for ProductKernel
where
    T: crate::scalar::Product,
{
    type Output = <T as crate::scalar::Product>::Output;
    type State = <T as crate::scalar::Product>::Output;

    #[inline(always)]
    fn init_state(&self, init_item: Option<(T, Option<u64>)>) -> Self::State {
        let mut state = <T as crate::scalar::Product>::init();
        if let Some((item, _idx)) = init_item {
            state = self.update_state(state, item, None);
        }
        state
    }
    #[inline(always)]
    fn update_state(&self, state: Self::State, item: T, _idx: Option<u64>) -> Self::State {
        <T as crate::scalar::Product>::update(state, item)
    }
    #[inline(always)]
    fn update_state_bulk<const N: usize, S: Simd>(
        &self,
        states: [Self::State; N],
        items: [T; N],
        _indices: Option<[u64; N]>,
        simd: S,
    ) -> [Self::State; N] {
        <T as crate::scalar::Product>::update_bulk(states, items, simd)
    }
    #[inline(always)]
    fn merge_states(&self, a: Self::State, b: Self::State) -> Self::State {
        <T as crate::scalar::Product>::merge_states(a, b)
    }
    #[inline(always)]
    fn finalize_state(&self, state: Self::State, _nitems: u64) -> Self::Output {
        state
    }
    const SUPPORTS_EMPTY: bool = true;
    const NEEDS_FINALIZE: bool = false;
    const PREFER_TREE_MERGE: bool = false;
    const NEEDS_INDICES: bool = false;
}

define_reduction_op!(
    /// Reduces one or more dims by computing the arithmetic mean.
    ///
    /// Integer and `bool8` inputs promote to `f64`; floating-point and complex inputs keep
    /// their input width (`f16 -> f16`, `f32 -> f32`, `f64 -> f64`, `Complex<f32> ->
    /// Complex<f32>`, `Complex<f64> -> Complex<f64>`), matching NumPy.
    ///
    /// Reducing an empty slice (zero elements) panics.
    ///
    /// The result is a lazy view; no computation occurs until the array is read.
    ///
    /// This struct is the bare storage implementation, the operation is also available as
    /// [`Array::mean()`](crate::Array::mean).
    ///
    /// # Examples
    /// ```
    /// use jix::Array;
    /// use ndarray::{array, Ix0};
    ///
    /// let nd = array![[1i32, 2, 3], [4, 5, 6]];
    ///
    /// // Mean of all elements -> f64
    /// let total = Array::compact_ndarray(&nd)?
    ///     .mean((0, 1)).to_ndarray::<Ix0>()?;
    /// assert_eq!(total[[]], 3.5);
    ///
    /// // Mean along dim 0 -> shape [3]
    /// let col_means = Array::compact_ndarray(&nd)?
    ///     .mean(0).to_ndarray()?;
    /// assert_eq!(col_means, array![2.5, 3.5, 4.5]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    Mean,
    MeanKernel,
    where {
        S: ArrayStorageTyped,
        S::Item: crate::scalar::Mean<Output: Dtyped>,
    }
    output = <S::Item as crate::scalar::Mean>::Output,
);
impl<T> ReductionOpKernel<T> for MeanKernel
where
    T: crate::scalar::Mean,
{
    type Output = <T as crate::scalar::Mean>::Output;
    type State = <T as crate::scalar::Mean>::State;

    #[inline(always)]
    fn init_state(&self, init_item: Option<(T, Option<u64>)>) -> Self::State {
        let mut state = <T as crate::scalar::Mean>::init();
        if let Some((item, _idx)) = init_item {
            state = self.update_state(state, item, None);
        }
        state
    }
    #[inline(always)]
    fn update_state(&self, state: Self::State, item: T, _idx: Option<u64>) -> Self::State {
        <T as crate::scalar::Mean>::update(state, item)
    }
    #[inline(always)]
    fn update_state_bulk<const N: usize, S: Simd>(
        &self,
        states: [Self::State; N],
        items: [T; N],
        _indices: Option<[u64; N]>,
        simd: S,
    ) -> [Self::State; N] {
        <T as crate::scalar::Mean>::update_bulk(states, items, simd)
    }
    #[inline(always)]
    fn merge_states(&self, a: Self::State, b: Self::State) -> Self::State {
        <T as crate::scalar::Mean>::merge_states(a, b)
    }
    #[inline(always)]
    fn finalize_state(&self, state: Self::State, nitems: u64) -> Self::Output {
        <T as crate::scalar::Mean>::finalize(state, nitems).unwrap()
    }
    const SUPPORTS_EMPTY: bool = false;
    const PREFER_TREE_MERGE: bool = !<T as crate::scalar::Mean>::IS_PRECISE_SUM;
    const NEEDS_INDICES: bool = false;
}

impl<S> Array<S>
where
    S: ArrayStorage,
{
    define_array_reduction_method!(
        sum: Sum,
        where {
            S: ArrayStorageTyped,
            S::Item: crate::scalar::Sum<Output: Dtyped>,
        }
    );
    define_array_reduction_method!(
        product: Product,
        where {
            S: ArrayStorageTyped,
            S::Item: crate::scalar::Product<Output: Dtyped>,
        }
    );
    define_array_reduction_method!(
        mean: Mean,
        where {
            S: ArrayStorageTyped,
            S::Item: crate::scalar::Mean<Output: Dtyped>,
        }
    );
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
mod tests {

    use crate::scalar::bool8;

    use ndarray::{array, Ix0, Ix1};

    #[cfg(feature = "num-complex")]
    use crate::scalar::Complex;
    #[cfg(feature = "num-complex")]
    #[allow(non_camel_case_types)]
    type complex_f32 = crate::scalar::Complex<f32>;
    #[cfg(feature = "num-complex")]
    #[allow(non_camel_case_types)]
    type complex_f64 = crate::scalar::Complex<f64>;

    use crate::array::Array;

    use crate::ops::reduction::common::tests::*;

    test_reduction!(
        sum,
        |items| { items.fold(0u64, |m, x| m.wrapping_add(x as u64)) },
        [u8, u16, u32, u64],
        any_strategy
    );
    test_reduction!(
        sum,
        |items| { items.fold(0u64, |m, x| m + bool::from(x) as u64) },
        [bool8],
        any_strategy
    );
    test_reduction!(
        sum,
        |items| { items.fold(0i64, |m, x| m.wrapping_add(x as i64)) },
        [i8, i16, i32, i64],
        any_strategy
    );
    // Float/complex sum preserves the input width and accumulates in it, so the block fold
    // reassociates - compare approximately, and skip f16 (its native accumulation drifts too
    // far to pin with a tolerance; covered by max/min instead).
    test_reduction!(
        sum,
        approx,
        |items| { items.fold(0.0f64, |m, x| m + <_ as crate::scalar::Cast<f64>>::cast(x)) },
        [f32, f64],
        op_safe_strategy
    );
    #[cfg(feature = "num-complex")]
    test_reduction!(
        sum,
        approx,
        |items| {
            items.fold(Complex::<f64>::default(), |m, x| {
                m + <_ as crate::scalar::Cast<Complex<f64>>>::cast(x)
            })
        },
        [complex_f32, complex_f64],
        op_safe_strategy
    );
    test_reduction!(
        product,
        |items| { items.fold(1u64, |m, x| m.wrapping_mul(x as u64)) },
        [u8, u16, u32, u64],
        any_strategy
    );
    test_reduction!(
        product,
        |items| { items.fold(1i64, |m, x| m.wrapping_mul(x as i64)) },
        [i8, i16, i32, i64],
        any_strategy
    );
    #[test]
    fn product_concrete() {
        // i32 -> i64 (integer product widens; see `traits::Product`). Includes negatives and
        // a `0` (zero-absorption edge); values stay small so the widened i64 accumulator can't
        // overflow.
        let nd = array![[2i32, -3, 0], [-1, 5, -2]];
        let za = Array::compact_ndarray(&nd).unwrap();
        // dim 0 (per column): 2*-1=-2; -3*5=-15; 0*-2=0.
        crate::util::assert_array_matches(&za.view().product(0usize), &array![-2i64, -15, 0]);
        // dim 1 (per row): 2*-3*0=0; -1*5*-2=10.
        crate::util::assert_array_matches(&za.view().product(1usize), &array![0i64, 10]);
        // all dims: the `0` term collapses the whole product to 0.
        crate::util::assert_array_matches(&za.view().product((0, 1)), &ndarray::arr0(0i64));

        // f32 -> f32 (float product keeps its input width). Negatives, no zero this time so
        // the full multiplication chain is exercised (the i32 case above covers the zero edge).
        let ndf = array![[1.5f32, -2.0, 0.5], [2.0, -1.0, 3.0]];
        let zaf = Array::compact_ndarray(&ndf).unwrap();
        // dim 0 (per column): 1.5*2.0=3.0; -2.0*-1.0=2.0; 0.5*3.0=1.5.
        crate::util::assert_array_matches_approx(
            &zaf.view().product(0usize),
            &array![3.0f32, 2.0, 1.5],
            1e-3,
            1e-1,
        );
        // dim 1 (per row): 1.5*-2.0*0.5=-1.5; 2.0*-1.0*3.0=-6.0.
        crate::util::assert_array_matches_approx(
            &zaf.view().product(1usize),
            &array![-1.5f32, -6.0],
            1e-3,
            1e-1,
        );
        crate::util::assert_array_matches_approx(
            &zaf.view().product((0, 1)),
            &ndarray::arr0(9.0f32),
            1e-3,
            1e-1,
        );
    }

    #[test]
    fn sum_empty_float_is_positive_zero() {
        let a = Array::compact_ndarray(&ndarray::Array2::<f64>::zeros((3, 0))).unwrap();
        let s = a.sum(1usize).to_ndarray::<Ix1>().unwrap();
        assert!(s.iter().all(|v| v.to_bits() == 0.0f64.to_bits()), "{s:?}");
        let a = Array::compact_ndarray(&ndarray::Array1::<f32>::zeros(0)).unwrap();
        let s = a.sum(0usize).to_ndarray::<Ix0>().unwrap()[[]];
        assert_eq!(s.to_bits(), 0.0f32.to_bits());
        #[cfg(feature = "num-complex")]
        {
            let a = Array::compact_ndarray(&ndarray::Array1::<complex_f64>::zeros(0)).unwrap();
            let s = a.sum(0usize).to_ndarray::<Ix0>().unwrap()[[]];
            assert_eq!((s.re.to_bits(), s.im.to_bits()), (0, 0));
        }
        // A non-empty sum of -0.0 keeps its sign.
        let a = Array::compact_ndarray(&array![-0.0f64, -0.0]).unwrap();
        let s = a.sum(0usize).to_ndarray::<Ix0>().unwrap()[[]];
        assert_eq!(s.to_bits(), (-0.0f64).to_bits());
    }

    // mean. Integer/bool inputs widen to f64 (exact-in-f64 sum, then divide) - compared via the
    // f64 `ReductionCompare` (approx). Float/complex inputs preserve their width and reassociate,
    // so they take the approx path; f16 is skipped (see `sum`).
    test_reduction!(
        mean,
        |items| {
            {
                let mut sum: f64 = 0.0;
                let mut count: usize = 0;
                for x in items {
                    sum += <_ as crate::scalar::Cast<f64>>::cast(x);
                    count += 1;
                }
                sum / count as f64
            }
        },
        [i8, i16, i32, i64, u8, u16, u32, u64],
        op_safe_strategy
    );
    test_reduction!(
        mean,
        approx,
        |items| {
            {
                let mut sum: f64 = 0.0;
                let mut count: usize = 0;
                for x in items {
                    sum += <_ as crate::scalar::Cast<f64>>::cast(x);
                    count += 1;
                }
                sum / count as f64
            }
        },
        [f32, f64],
        op_safe_strategy
    );
    #[cfg(feature = "num-complex")]
    test_reduction!(
        mean,
        approx,
        |items| {
            {
                let mut sum: Complex<f64> = Complex::default();
                let mut count: usize = 0;
                for x in items {
                    sum += <_ as crate::scalar::Cast<Complex<f64>>>::cast(x);
                    count += 1;
                }
                sum / count as f64
            }
        },
        [complex_f32, complex_f64],
        op_safe_strategy
    );

    #[cfg(feature = "num-complex")]
    #[test]
    fn mean_complex_near_max() {
        let a = Array::compact_ndarray(&array![
            complex_f64::new(1.7e308, -1.7e308),
            complex_f64::new(0.0, 0.0)
        ])
        .unwrap();
        let m = a.mean(0usize).to_ndarray::<Ix0>().unwrap()[[]];
        assert_eq!(m, complex_f64::new(0.85e308, -0.85e308));
    }
}
