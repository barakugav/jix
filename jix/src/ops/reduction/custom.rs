use crate::ops::prelude::*;
use crate::ops::reduction::common::{ReductionOp, ReductionOpKernel};

/// Reduces one or more dims by combining the elements along those dims with a user-supplied
/// binary closure, in an **unspecified order**.
///
/// The output dtype is the same as the input dtype (`S::Item`), and the closure has signature
/// `Fn(S::Item, S::Item) -> S::Item`. There is no separate seed value: the first element each
/// accumulator sees becomes its initial value. `f` is used for two distinct jobs, and cannot
/// tell them apart:
///
/// - folding one element into a running accumulator - `f(acc, x)`;
/// - combining two accumulators built from *disjoint* subsets of the same output cell's
///   elements - `f(acc_a, acc_b)`.
///
/// # The closure MUST be associative and commutative
///
/// Despite the name, this is **not** [`Iterator::reduce`]: it is not a left fold, and that
/// holds even when a single dim is reduced. One output cell's elements are spread over multiple
/// interleaved lane accumulators that are then collapsed by a pairwise tree, and when the
/// reduced extent is larger than one read tile a cell's accumulator is merged across tiles.
/// Which elements land in which accumulator, and in which order the accumulators are combined,
/// is an implementation detail that shifts with the array shape, the block shape, the read
/// window size, and the library version.
///
/// So `f` must be associative *and* commutative for the result to be well-defined - `min`,
/// `max`, `+`, `*`, `&`, `|`, `^`, `gcd` all qualify. A non-commutative closure such as
/// `|a, b| a - b` compiles and runs but yields an arbitrary value.
///
/// Float `+` / `*` are only associative up to rounding. That is fine and expected: the
/// built-in [`Sum`](crate::ops::Sum) / [`Product`](crate::ops::Product) reductions
/// reassociate exactly the same way.
///
/// # Empty reductions
///
/// Empty reductions (any reduced dimension has length `0`) are **not supported**: with no seed
/// value and no first element there is nothing to return. [`Reduce::new`] returns an
/// `Err`, and [`Array::reduce_unordered`] panics at construction time.
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// This struct is the bare storage implementation; the operation is also available as
/// [`Array::reduce_unordered()`](crate::Array::reduce_unordered).
///
/// # Examples
///
/// Custom maximum over a single dim:
/// ```
/// use jix::Array;
/// use ndarray::array;
///
/// let nd = array![[1i32, 5, 3], [4, 2, 6]];
/// let row_max = Array::compact_ndarray(&nd)?
///     .reduce_unordered(1, |a, b| if a > b { a } else { b })
///     .to_ndarray()?;
/// assert_eq!(row_max, array![5, 6]);
/// # Ok::<(), jix::Error>(())
/// ```
///
/// Multi-dim reduction with a closure that has no built-in equivalent:
/// ```
/// use jix::Array;
/// use ndarray::{array, Ix0};
///
/// let nd = array![[0b0011u8, 0b0101], [0b1001, 0b0001]];
/// let xor = Array::compact_ndarray(&nd)?
///     .reduce_unordered((0, 1), |a, b| a ^ b)
///     .to_ndarray::<Ix0>()?;
/// assert_eq!(xor[[]], 0b1110);
/// # Ok::<(), jix::Error>(())
/// ```
pub struct Reduce<S: ArrayStorage, F>(ReductionOp<S, ReduceKernel<F>>);
impl<S: ArrayStorage, F> Reduce<S, F> {
    /// Constructs a [`Reduce`] storage. See the struct docs for semantics and examples.
    pub fn new(array: S, dims: impl DimsArg, f: F) -> Result<Self>
    where
        S: ArrayStorageTyped,
        F: Fn(S::Item, S::Item) -> S::Item,
    {
        Ok(Self(ReductionOp::new(array, ReduceKernel(f), dims)?))
    }

    /// Constructs an array with [`Reduce`] storage. See the storage struct docs for semantics and examples.
    pub fn new_array(array: Array<S>, dims: impl DimsArg, f: F) -> Result<Array<Reduce<S, F>>>
    where
        S: ArrayStorageTyped,
        F: Fn(S::Item, S::Item) -> S::Item,
    {
        Self::new(array.into_storage(), dims, f).map(Array::from_storage)
    }
}
struct ReduceKernel<F>(F);
impl<T, F> ReductionOpKernel<T> for ReduceKernel<F>
where
    F: Fn(T, T) -> T,
{
    type Output = T;
    type State = T;

    #[inline(always)]
    fn init_state(&self, init_item: Option<(T, Option<u64>)>) -> Self::State {
        init_item.unwrap().0
    }
    #[inline(always)]
    fn update_state(&self, state: Self::State, item: T, _idx: Option<u64>) -> Self::State {
        (self.0)(state, item)
    }
    #[inline(always)]
    fn merge_states(&self, a: Self::State, b: Self::State) -> Self::State {
        (self.0)(a, b)
    }
    #[inline(always)]
    fn finalize_state(&self, state: Self::State, _nitems: u64) -> Self::Output {
        state
    }
    const SUPPORTS_EMPTY: bool = false;
    const NEEDS_FINALIZE: bool = false;
    const PREFER_TREE_MERGE: bool = true;
    const NEEDS_INDICES: bool = false;
}
impl<S, F> crate::storage::ArrayStorageImpl for Reduce<S, F>
where
    S: ArrayStorageTyped,
    F: Fn(S::Item, S::Item) -> S::Item,
{
}
impl<S, F> ArrayStorage for Reduce<S, F>
where
    S: ArrayStorageTyped,
    F: Fn(S::Item, S::Item) -> S::Item,
{
    type ElementType = Ty<S::Item>;
    crate::storage::impl_array_storage_forward!(<S, F>);

    fn info(&self) -> ArrayStorageInfo<'_> {
        ArrayStorageInfo::new_deps("Reduce", [&self.0.array])
    }

    crate::ops::impl_element_type_change_default!();
}

/*
/// Reduces one or more dims by folding the elements along those dims through a
/// user-supplied closure, starting from an explicit initial accumulator.
///
/// The output dtype is the accumulator type `B` (which can differ from the input element
/// type `S::Item`). The closure has signature `Fn(B, S::Item) -> B` and is applied with
/// the running accumulator on the left, mirroring [`Iterator::fold`]: for a stream
/// `x0, x1, ..., xn`, the result is `f(f(f(init, x0), x1), ..., xn)`.
///
/// `B` must implement [`Dtyped`](crate::dtype::Dtyped) (i.e. `Copy + Send + Sync + 'static`
/// plus the jix dtype contract). The initial value is stored inside the storage and cloned
/// (via `Copy`) once per output cell.
///
/// # Traversal order
///
/// - **Single reduced dim**: elements along that dim are visited in logical order (index
///   `0` upward). The result is well-defined for non-commutative / non-associative
///   closures.
/// - **Multiple reduced dims**: elements are visited in an *implementation-defined* order
///   driven by the storage's internal tiling. The order is not stable across array shapes,
///   block sizes, or library versions. Closures used here MUST be both associative and
///   commutative for the result to be well-defined.
///
/// # Empty reductions
///
/// Unlike [`Reduce`], `Fold` supports empty reductions: when a reduced dim has length
/// `0`, every output cell receives `init` unchanged (the closure is never invoked).
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// This struct is the bare storage implementation; the operation is also available as
/// [`Array::fold()`](crate::Array::fold).
///
/// # Examples
///
/// Sum with a wider accumulator (the input is `u8`, the result is `i64`):
/// ```
/// use jix::Array;
/// use ndarray::array;
///
/// let nd = array![[1u8, 2, 3], [4, 5, 6]];
/// let total = Array::compact_ndarray(&nd)?
///     .fold((0, 1), 0i64, |a, x| a + x as i64)
///     .to_ndarray()?;
/// assert_eq!(total[[]], 21);
/// # Ok::<(), jix::Error>(())
/// ```
///
/// Single-dim fold: order is guaranteed to be logical, so the non-commutative closure
/// produces a deterministic result:
/// ```
/// use jix::Array;
/// use ndarray::array;
///
/// // ((((100 - 10) - 1) - 2) - 3) = 84
/// let nd = array![10i32, 1, 2, 3];
/// let result = Array::compact_ndarray(&nd)?
///     .fold(0, 100i32, |a, x| a - x)
///     .to_ndarray()?;
/// assert_eq!(result[[]], 84);
/// # Ok::<(), jix::Error>(())
/// ```
///
/// Counting elements satisfying a predicate (per row):
/// ```
/// use jix::Array;
/// use ndarray::array;
///
/// let nd = array![[1i32, 5, 3, 7], [4, 2, 8, 1]];
/// // Count elements > 3 along each row.
/// let counts = Array::compact_ndarray(&nd)?
///     .fold(1, 0u64, |c, x| c + (x > 3) as u64)
///     .to_ndarray()?;
/// assert_eq!(counts.as_slice().unwrap(), &[2, 2]);
/// # Ok::<(), jix::Error>(())
/// ```
pub struct Fold<S: ArrayStorage, B, F>(ReductionOp<S, VanillaFoldKernel<B, F>>);
impl<S: ArrayStorage, B, F> Fold<S, B, F> {
    /// Constructs a [`Fold`] storage. See the struct docs for semantics and examples.
    pub fn new(array: S, dims: impl DimsArg, init: B, f: F) -> Result<Self>
    where
        S: ArrayStorageTyped,
        B: Dtyped,
        F: Fn(B, S::Item) -> B,
    {
        Ok(Self(ReductionOp::new(
            array,
            VanillaFoldKernel { init, f },
            dims,
        )?))
    }

    /// Constructs an array with [`Fold`] storage. See the storage struct docs for semantics and examples.
    pub fn new_array(
        array: Array<S>,
        dims: impl DimsArg,
        init: B,
        f: F,
    ) -> Result<Array<Fold<S, B, F>>>
    where
        S: ArrayStorageTyped,
        B: Dtyped,
        F: Fn(B, S::Item) -> B,
    {
        Self::new(array.into_storage(), dims, init, f).map(Array::from_storage)
    }
}
struct VanillaFoldKernel<B, F> {
    init: B,
    f: F,
}
impl<T, B, F> ReductionOpKernel<T> for VanillaFoldKernel<B, F>
where
    B: Dtyped,
    F: Fn(B, T) -> B,
{
    type Output = B;
    type State = B;

    #[inline(always)]
    fn init_state(&self, init_item: Option<(T, Option<u64>)>) -> Self::State {
        let mut state = self.init;
        if let Some((item, idx)) = init_item {
            state = self.update_state(state, item, idx);
        }
        state
    }
    #[inline(always)]
    fn update_state(&self, state: Self::State, item: T, _idx: Option<u64>) -> Self::State {
        (self.f)(state, item)
    }
    #[inline(always)]
    fn finalize_state(&self, state: Self::State, _nitems: u64) -> Self::Output {
        state
    }
    const SUPPORTS_EMPTY: bool = true;
    const NEEDS_FINALIZE: bool = false;
    const PREFER_TREE_MERGE: bool = true;
    const NEEDS_INDICES: bool = false;
}
impl<S, B, F> crate::storage::ArrayStorageImpl for Fold<S, B, F> {}
impl<S, B, F> ArrayStorage for Fold<S, B, F>
where
    S: ArrayStorageTyped,
    B: Dtyped,
    F: Fn(B, S::Item) -> B,
{
    type ElementType = Ty<B>;
    crate::storage::impl_array_storage_forward!(<S, B, F>);

    fn info(&self) -> ArrayStorageInfo<'_> {
        ArrayStorageInfo::new_deps("Fold", [&self.0.array])
    }

    crate::ops::impl_element_type_change_default!();
}
*/

impl<S> Array<S>
where
    S: ArrayStorage,
{
    /// Applies the [`Reduce`] operation, see the op struct docs for details.
    ///
    /// `f` MUST be associative and commutative: the elements are visited in an unspecified
    /// order, and `f` combines partial accumulators as well as single elements.
    #[track_caller]
    pub fn reduce_unordered<F>(self, dims: impl DimsArg, f: F) -> Array<Reduce<S, F>>
    where
        S: ArrayStorageTyped,
        F: Fn(S::Item, S::Item) -> S::Item,
    {
        Reduce::new_array(self, dims, f).unwrap()
    }

    /* TODO
    /// Applies the [`Fold`] operation, see the op struct docs for details.
    #[track_caller]
    #[allow(clippy::type_complexity)]
    pub fn fold<F, B>(
        self,
        dims: impl DimsArg,
        init: B,
        f: F,
    ) -> Array<Fold<S, B, F>>
    where
        S: ArrayStorageTyped,
        B: Dtyped,
        F: Fn(B, S::Item) -> B,
    {
        Fold::new_array(self, dims, init, f).unwrap()
    }
    */
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
mod tests {

    use crate::scalar::bool8;
    use crate::{ArrayParams, ArrayStorage};

    use ndarray::{array, Ix0, Ix1};

    #[cfg(feature = "half")]
    use crate::scalar::f16;

    #[cfg(feature = "num-complex")]
    #[allow(non_camel_case_types)]
    type complex_f32 = crate::scalar::Complex<f32>;
    #[cfg(feature = "num-complex")]
    #[allow(non_camel_case_types)]
    type complex_f64 = crate::scalar::Complex<f64>;

    use super::Reduce;
    use crate::array::Array;

    use crate::ops::reduction::common::tests::*;

    // --- Reduce ------------------------------------------------------------------

    // Combiners for the `reduce_unordered` property tests. Each one is associative AND
    // commutative, as the op requires - which is what lets a plain sequential fold serve as the
    // reference for the op's unspecified order.
    fn xor<T: std::ops::BitXor<Output = T>>(a: T, b: T) -> T {
        a ^ b
    }
    fn maximum<T: PartialOrd>(a: T, b: T) -> T {
        if a > b {
            a
        } else {
            b
        }
    }
    #[cfg(feature = "num-complex")]
    fn add<T: std::ops::Add<Output = T>>(a: T, b: T) -> T {
        a + b
    }

    /// Proptest driver for `reduce_unordered` over one dtype: the op must agree with a plain
    /// sequential fold of the same combiner `$f`, for every shape/dims combination the shared
    /// reduction strategy generates.
    macro_rules! test_reduce_unordered_dtype {
        ($f:path, $dtype:ident, $strategy:ident) => {
            paste::paste! {
                #[test]
                fn [<reduce_unordered_ $dtype:lower>]() {
                    check_reduction::<$dtype>(
                        proptest::strategy::Strategy::boxed(
                            array_strategy_for_reduction::<$dtype>(
                                <$dtype as crate::util::ScalarStrategy>::$strategy()
                            )
                        ),
                        |nd, za, dims| {
                            let result = za.view().reduce_unordered(dims, $f);
                            let expected = ndarray_reduce(nd, dims, |arr| {
                                arr.iter().cloned().reduce($f).unwrap()
                            });
                            assert_reduction_matches(&result, &expected);
                        },
                    );
                }
            }
        };
        ($f:path, [$($dtype:ident),+ $(,)?], $strategy:ident) => {
            $(test_reduce_unordered_dtype!($f, $dtype, $strategy);)+
        };
    }

    // `xor` is exact and has no built-in op counterpart, so it pins the whole custom-closure
    // path (lane seeding, lane tree merge, cross-tile merge) without needing a tolerance.
    test_reduce_unordered_dtype!(xor, [i8, i16, i32, i64], any_strategy);
    test_reduce_unordered_dtype!(xor, [u8, u16, u32, u64, bool8], any_strategy);
    // Floats have no `xor`; `maximum` is the exact order-independent stand-in. `any_strategy`
    // never yields `NaN` for floats, so `>` is a total order here and the reference agrees
    // exactly (this is also the only group `f16` can join - see `ReductionCompare`).
    test_reduce_unordered_dtype!(maximum, [f32, f64], any_strategy);
    #[cfg(feature = "half")]
    test_reduce_unordered_dtype!(maximum, [f16], any_strategy);
    // Complex has no ordering, so use `add` and let the per-dtype approximate comparison absorb
    // the reassociation (same deal as the built-in `sum` on floats).
    #[cfg(feature = "num-complex")]
    test_reduce_unordered_dtype!(add, [complex_f32, complex_f64], op_safe_strategy);

    #[test]
    fn reduce_unordered_concrete() {
        // Dim 0, dim 1 and all-dims, with a non-default block shape ([2, 2]) so both the
        // reduced and the kept dim cross a block boundary.
        let nd = array![[1i32, 5, 3], [4, 2, 6]];
        let za = Array::compact_ndarray_with(&nd, crate::util::arr_params(&[2, 2])).unwrap();
        // dim 0 (per column): max(1,4)=4; max(5,2)=5; max(3,6)=6.
        crate::util::assert_array_matches(
            &za.view().reduce_unordered(0usize, maximum::<i32>),
            &array![4, 5, 6],
        );
        // dim 1 (per row): max(1,5,3)=5; max(4,2,6)=6.
        crate::util::assert_array_matches(
            &za.view().reduce_unordered(1usize, maximum::<i32>),
            &array![5, 6],
        );
        // all dims -> scalar.
        crate::util::assert_array_matches(
            &za.view().reduce_unordered((0, 1), maximum::<i32>),
            &ndarray::arr0(6),
        );
        // A reduction over zero dims is a no-op copy: every cell is its own accumulator.
        crate::util::assert_array_matches(
            &za.view()
                .reduce_unordered([0usize; 0].as_slice(), maximum::<i32>),
            &nd,
        );
    }

    #[test]
    fn reduce_unordered_preserves_dtype() {
        // The output dtype must equal the input dtype (no widening, unlike `sum`/`product`).
        use crate::dtype::Dtyped;
        let a = Array::compact_ndarray(&array![1i8, 2, 3]).unwrap();
        let r = a.view().reduce_unordered(0usize, |a, b| a.wrapping_add(b));
        assert_eq!(r.dtype(), &<i8 as Dtyped>::DTYPE);
        assert_eq!(r.to_ndarray::<Ix0>().unwrap()[[]], 6i8);
    }

    #[test]
    fn reduce_unordered_wide_elements_fit_the_stack() {
        const W: usize = 16000;
        let combine = |a: [u8; W], b: [u8; W]| std::array::from_fn(|i| a[i].wrapping_add(b[i]));
        let run = move || {
            let data = ndarray::Array1::from(vec![[1u8; W]; 64]);
            // Strided one-cell fold.
            let a = Array::plain_ndarray_view(data.slice(ndarray::s![..;2])).unwrap();
            let r = a
                .reduce_unordered(0usize, combine)
                .to_ndarray::<Ix0>()
                .unwrap();
            assert!(r[[]].iter().all(|&b| b == 32));
            // Contiguous one-cell fold.
            let a = Array::plain_ndarray_view(data.view()).unwrap();
            let r = a
                .reduce_unordered(0usize, combine)
                .to_ndarray::<Ix0>()
                .unwrap();
            assert!(r[[]].iter().all(|&b| b == 64));
        };
        // Without the lanes cap this needs more than 8 MiB in debug builds.
        let thread = std::thread::Builder::new()
            .stack_size(8 << 20)
            .spawn(run)
            .unwrap();
        thread.join().unwrap();
    }

    #[test]
    fn reduce_unordered_errs_on_empty_dim() {
        // With no seed value and no first element there is nothing to return, so construction
        // fails (`supports_empty() == false`).
        use ndarray::Array2;
        let empty: Array2<i32> = Array2::from_shape_vec((2, 0), vec![]).unwrap();
        let a = Array::compact_ndarray(&empty).unwrap();
        let err = Reduce::new_array(a.view(), 1usize, |a: i32, b: i32| a + b)
            .expect_err("empty reduced dim must be rejected");
        assert!(
            err.to_string().contains("empty dimension"),
            "unexpected error: {err}"
        );
        // An empty *kept* dim is fine - the reduced dim is non-empty and the output is empty.
        let empty: Array2<i32> = Array2::from_shape_vec((0, 2), vec![]).unwrap();
        let a = Array::compact_ndarray(&empty).unwrap();
        let r = a
            .view()
            .reduce_unordered(1usize, |a, b| a + b)
            .to_ndarray::<Ix1>()
            .unwrap();
        assert_eq!(r.shape(), &[0]);
    }

    #[test]
    #[should_panic(expected = "empty dimension")]
    fn reduce_unordered_panics_on_empty_dim() {
        // `Array::reduce_unordered` unwraps the construction error.
        use ndarray::Array2;
        let empty: Array2<i32> = Array2::from_shape_vec((2, 0), vec![]).unwrap();
        let a = Array::compact_ndarray(&empty).unwrap();
        let _ = a.view().reduce_unordered(1usize, |a, b| a + b);
    }

    #[test]
    fn reduce_unordered_multi_bulk_multi_tile_2d_reduce_outer() {
        // Same chunking scenarios as the `sum` bulk/tile tests, driven through the custom
        // closure instead. Reduce dim 0: bulks split the reduced dim, tiles split the kept one.
        check_reduce_unordered_divided(&[8, 6], &[2, 3], (32, 64), &[0]);
    }

    #[test]
    fn reduce_unordered_multi_bulk_multi_tile_2d_reduce_inner() {
        check_reduce_unordered_divided(&[6, 8], &[3, 2], (32, 64), &[1]);
    }

    #[test]
    fn reduce_unordered_multi_bulk_multi_tile_3d_reduce_middle() {
        check_reduce_unordered_divided(&[4, 4, 4], &[2, 2, 2], (32, 64), &[1]);
    }

    #[test]
    fn reduce_unordered_multi_bulk_long_reduced_dim() {
        // A long, prime-length reduced dim with a tiny read window: the dim is chopped into
        // many bulks whose partial accumulators all have to be merged back together, and 137 is
        // not a multiple of the block extent so the last bulk is a short one.
        //
        // (The *lane* interleaving inside a single tile - which needs a per-tile reduced extent
        // of at least LANES - is covered by the 1D property-test shapes of 100..=1000 elements,
        // which are read in one tile with the default read window.)
        check_reduce_unordered_divided(&[137, 3], &[8, 1], (32, 64), &[0]);
    }

    /// Drives `reduce_unordered` through the two-level bulk/tile chunking (see
    /// [`check_sum_divided`] for why `read_data` is called directly with the whole output range)
    /// and checks two independent things:
    ///
    /// 1. **Value**: `xor` is associative, commutative and exact, so the result must equal a
    ///    sequential reference fold no matter how the stream got partitioned.
    /// 2. **Call count**: combining `n` elements into one accumulator is an `n`-leaf tree, which
    ///    has exactly `n - 1` internal nodes - so the closure must be invoked exactly
    ///    `n_out * (reduction_size - 1)` times. Counting them catches a dropped, duplicated or
    ///    re-seeded element even when `xor` would happen to mask it (e.g. an element folded in
    ///    twice cancels out).
    fn check_reduce_unordered_divided(
        shape: &[usize],
        block_shape: &[u32],
        read_size: (u64, u64),
        dims: &[usize],
    ) {
        use std::cell::Cell;

        let n: usize = shape.iter().product();
        let nd = ndarray::ArrayD::from_shape_vec(
            shape.to_vec(),
            (0..n as i32).map(|x| (x % 251) as u8).collect(),
        )
        .unwrap();

        let params = ArrayParams::new()
            .block_shape(block_shape)
            .unwrap()
            .read_size(read_size);
        let za = Array::compact_ndarray_with(&nd, params).unwrap();

        let calls = Cell::new(0u64);
        let reduced = za.view().reduce_unordered(dims, |a: u8, b: u8| {
            calls.set(calls.get() + 1);
            xor(a, b)
        });
        let out_shape: Vec<u64> = reduced.shape().to_vec();
        let full_index: Vec<std::ops::Range<u64>> = out_shape.iter().map(|&s| 0..s).collect();
        let n_out: usize = out_shape.iter().product::<u64>() as usize;
        let ctx = reduced.read_ctx();
        let storage = reduced.into_storage();

        let mut buf = vec![0u8; n_out.max(1)];
        {
            let sh = full_index
                .iter()
                .map(|r| (r.end - r.start) as usize)
                .collect::<Vec<_>>();
            let c = crate::util::default_strides_slice(&sh, size_of::<u8>());
            let mut out = unsafe {
                crate::storage::StridedBuf::from_slice_mut(&mut buf[..n_out], c.as_ref())
            };
            storage
                .read_data(&full_index, &ctx, Some(&mut out))
                .unwrap();
        }

        let expected = ndarray_reduce(&nd, dims, |v| v.iter().cloned().reduce(xor).unwrap());
        assert_eq!(&buf[..n_out], expected.as_slice().unwrap());

        let reduction_size = n / n_out;
        assert_eq!(
            calls.get(),
            (n_out * (reduction_size - 1)) as u64,
            "each output cell must combine its {reduction_size} elements with exactly \
             {} closure calls",
            reduction_size - 1
        );
    }

    /* TODO(reduction-merge): fold tests removed along with the op; restore later.
    #[test]
    fn fold_single_dim_in_logical_order() {
        // Single-dim fold with a non-commutative closure (subtraction). Logical-order
        // traversal makes the result deterministic:
        // ((((100 - 10) - 1) - 2) - 3) = 84.
        let a = Array::compact_ndarray(&array![10i32, 1, 2, 3]).unwrap();
        let r = a
            .view()
            .fold(0, 100i32, |a, x| a - x)
            .to_ndarray()
            .unwrap();
        assert_eq!(r[[]], 84);
    }

    #[test]
    fn fold_widens_output_dtype() {
        // The accumulator type `B` is independent of the input element type. Here we sum
        // a u8 array into an i64.
        let a = Array::compact_ndarray(&array![[1u8, 2, 3], [4, 5, 6]]).unwrap();
        let r = a
            .view()
            .fold((0, 1), 0i64, |a, x| a + x as i64)
            .to_ndarray()
            .unwrap();
        assert_eq!(r[[]], 21);
        // dtype is the accumulator type, not the input type.
        use crate::dtype::Dtyped;
        let r = a.view().fold((0, 1), 0i64, |a, x| a + x as i64);
        assert_eq!(r.dtype(), &<i64 as Dtyped>::DTYPE);
    }

    #[test]
    fn fold_multi_dim_count_predicate() {
        // Per-row count via fold along dim 1. The closure is commutative + associative
        // (addition), so single-dim order doesn't matter here - but dim 1 is a single
        // dim, so order is still guaranteed.
        let a = Array::compact_ndarray(&array![[1i32, 5, 3, 7], [4, 2, 8, 1]]).unwrap();
        let r = a
            .view()
            .fold(1, 0u64, |c, x| c + (x > 3) as u64)
            .to_ndarray()
            .unwrap();
        assert_eq!(r.as_slice().unwrap(), &[2, 2]);
    }

    #[test]
    fn fold_empty_dim_returns_init() {
        // Folding over an empty dim must produce the init accumulator at every output
        // cell - the closure is never invoked. Reducing dim 1 of a `(2, 0)` array
        // yields shape `[2]` with both cells equal to `init`.
        use ndarray::Array2;
        let empty: Array2<i32> = Array2::from_shape_vec((2, 0), vec![]).unwrap();
        let a = Array::compact_ndarray(&empty).unwrap();
        let r = a
            .view()
            .fold(1, 42i64, |a, x| a + x as i64)
            .to_ndarray()
            .unwrap();
        assert_eq!(r.as_slice().unwrap(), &[42, 42]);
    }

    #[test]
    fn fold_init_passed_through_when_closure_never_runs() {
        // A scalar reduction (reduce all dims) over a 0-element input collapses to a
        // single output cell whose value is exactly `init`.
        use ndarray::Array1;
        let empty: Array1<i32> = Array1::from_shape_vec(0, vec![]).unwrap();
        let a = Array::compact_ndarray(&empty).unwrap();
        let r = a
            .view()
            .fold(0, 999i64, |a, x| a + x as i64)
            .to_ndarray()
            .unwrap();
        assert_eq!(r[[]], 999);
    }
    */
}
