use std::hint::unreachable_unchecked;

use crate::util::iter::block::NdIterExtBlockOffsetSize;
use crate::util::iter::strides::{
    nd_iter_ext_logical_global_index, NdIterExtStridesOffset, NdIterExtStridesOffsetMulti,
    NdIterExtStridesOffsetMultiDyn, NdIterExtStridesPtr, NdIterExtStridesPtrMut,
};
use crate::util::{Idx, OperandsArray};
use crate::{DimVec, Dimension};

/// A multi-dimensional iterator that advances indices in row-major (C) order.
///
/// Extensions are supported via the generic parameter `E`, see [`NdIterExtension`].
/// The iterator notifies the extension on each index change, allowing extensions to track derived
/// state (e.g. a pointer into a strided buffer) without recomputing it from scratch.
#[derive(Clone)]
pub(crate) struct NdIter<D: Dimension, E> {
    begin: D::Vec<u64>,
    end: D::Vec<u64>,
    current_idx: D::Vec<u64>,
    status: IterStatus,
    pub(crate) extensions: E,
}

/// The iteration status is encoded in a single `i64` field:
/// - zero: exhausted or no items to begin with
/// - negative: not started yet, with `-status` items remaining
/// - positive: in progress, with `status` items remaining
#[derive(Clone)]
struct IterStatus(i64);
impl IterStatus {
    #[inline(always)]
    fn new(nitems: u64) -> Self {
        Self(-(nitems as i64))
    }

    #[inline(always)]
    fn is_not_started(&self) -> bool {
        self.0 < 0
    }
    #[inline(always)]
    fn is_in_progress(&self) -> bool {
        self.0 > 0
    }
    #[inline(always)]
    fn is_exhausted(&self) -> bool {
        self.0 == 0
    }
    #[inline(always)]
    fn len(&self) -> u64 {
        if self.is_not_started() {
            (-self.0) as u64
        } else {
            self.0 as u64
        }
    }

    #[inline(always)]
    fn start(&mut self) {
        debug_assert!(self.is_not_started());
        self.0 = -self.0;
    }
    #[inline(always)]
    fn advance(&mut self) {
        debug_assert!(self.is_in_progress());
        self.0 -= 1;
    }
}

impl<D, E> NdIter<D, E>
where
    D: Dimension,
    E: NdIterExtension,
{
    /// Creates an iterator over `[0, shape)` in every dimension.
    #[inline]
    pub(crate) fn new<V>(shape: V, extensions: E) -> Self
    where
        D: Dimension<Vec<u64> = V>,
        V: DimVec<u64, Dimension = D>,
    {
        let begin = V::Dimension::vec(shape.as_ref().len(), |_| 0u64);
        Self::new_with_begin(begin, shape, extensions)
    }

    /// Creates an iterator over `[begin, end)` in every dimension.
    #[inline]
    pub(crate) fn new_with_begin<V>(begin: V, end: V, extensions: E) -> Self
    where
        D: Dimension<Vec<u64> = V>,
        V: DimVec<u64, Dimension = D>,
    {
        let begin_slice = begin.as_ref();
        let end_slice = end.as_ref();
        let ndim = begin_slice.len();
        assert!(
            begin_slice.len() == ndim
                && end_slice.len() == ndim
                && extensions.check_ndim(ndim)
                && begin_slice
                    .iter()
                    .zip(end_slice.iter())
                    .all(|(&b, &e)| b <= e)
        );
        let current_idx = begin.clone();

        let nitems = begin_slice
            .iter()
            .zip(end_slice)
            .map(|(&b, &e)| e - b)
            .product::<u64>();
        let status = IterStatus::new(nitems);

        NdIter {
            end,
            begin,
            current_idx,
            status,
            extensions,
        }
    }

    /// Advances to the next index in row-major order and returns `(current_index, extension_item)`,
    /// or `None` once every index has been yielded.
    ///
    /// On each step the rightmost dimension that has not yet reached its bound is incremented,
    /// and all dimensions to its right are reset to `begin`.
    ///
    /// This is the *lending* `next`: the item may borrow from the extension (see
    /// [`NdIterExtension::Item`]). Extensions whose item does not borrow also get a plain
    /// [`Iterator`] impl, so those can be driven with `for .. in iter` instead.
    #[inline(always)]
    #[allow(clippy::should_implement_trait)]
    pub(crate) fn next(&mut self) -> Option<(D::Vec<u64>, E::Item<'_>)> {
        // Advancing is split out so the borrow of `self` taken for the returned item starts after
        // all mutation is done - borrowck rejects returning it from inside the branches below.
        if !self.advance() {
            return None;
        }
        Some((self.current_idx.clone(), self.extensions.value()))
    }

    /// Move to the next index, returning `false` once exhausted. On `true`, `current_idx` and the
    /// extensions hold the values to yield.
    #[inline(always)]
    fn advance(&mut self) -> bool {
        if self.status.is_exhausted() {
            return false;
        }

        if self.status.is_not_started() {
            self.status.start();
            self.status.advance();
            return true;
        }

        debug_assert!(self.status.is_in_progress());
        let end = self.end.as_ref();
        let ndim = end.len();
        for dim in (0..ndim).rev() {
            let advanced_idx = self.current_idx[dim] + 1;
            if advanced_idx < end[dim] {
                self.extensions
                    .on_increase(dim, self.current_idx[dim], advanced_idx, 1);
                self.current_idx[dim] = advanced_idx;
                for smaller_dim in dim + 1..ndim {
                    let begin = self.begin[smaller_dim];
                    self.extensions.on_decrease(
                        smaller_dim,
                        self.current_idx[smaller_dim],
                        begin,
                        self.current_idx[smaller_dim] - begin,
                    );
                    self.current_idx[smaller_dim] = begin;
                }
                self.status.advance();
                return true;
            }
        }
        unsafe { unreachable_unchecked() }
    }

    #[allow(unused)]
    #[inline(always)]
    pub(crate) fn len(&self) -> u64 {
        self.status.len()
    }
}

/// `NdIter` is a real [`Iterator`] whenever its extension's item does not borrow from it.
///
/// `for<'a> NdIterExtension<Item<'a> = I>` says exactly that: one `I` serves every lifetime, so the
/// item can outlive the `&self` that produced it and fit [`Iterator::Item`], which has no lifetime
/// of its own. An extension that does borrow (e.g. [`NdIterExtStridesOffsetMultiDyn`], which lends
/// a slice of per-operand offsets) has an `Item<'a>` that genuinely varies with `'a`, no single `I`
/// exists, and this impl simply does not apply - those must use the lending
/// [`next`](NdIter::next).
///
/// The `E: 'static` is forced by the `where Self: 'a` that Rust requires on any GAT returned from
/// `&self`: without it the `for<'a>` above would quantify over lifetimes for which `E` is not even
/// valid. Every extension owns its state (dimension vectors, raw pointers, indices), so this costs
/// nothing in practice.
impl<D, E, I> Iterator for NdIter<D, E>
where
    D: Dimension,
    E: 'static + for<'a> NdIterExtension<Item<'a> = I>,
{
    type Item = (D::Vec<u64>, I);

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        NdIter::next(self)
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.status.len() as usize;
        (len, Some(len))
    }
}

/// An extension trait for [`NdIter`] that tracks derived state alongside the current index.
///
/// Instead of recomputing derived state (e.g. a raw pointer offset) from scratch at every step,
/// implementors receive incremental [`on_increase`](NdIterExtension::on_increase) and
/// [`on_decrease`](NdIterExtension::on_decrease) notifications and
/// return the current derived value via [`value`](NdIterExtension::value).
pub(crate) trait NdIterExtension {
    /// The derived value produced at each iteration step.
    ///
    /// Borrows from the extension, so an extension holding a runtime-sized amount of state (e.g.
    /// [`NdIterExtStridesOffsetMultiDyn`], one offset per operand) can hand out a slice instead of
    /// copying into a fixed-size array. Extensions whose value is a plain `Copy` scalar just ignore
    /// the lifetime.
    type Item<'a>
    where
        Self: 'a;

    /// Called when dimension `dim` changes from `before` to `after`.
    ///
    /// All dimension changes for a single step are delivered before
    /// [`value`](NdIterExtension::value) is called.
    fn on_increase(&mut self, dim: usize, before: u64, after: u64, diff: u64);
    fn on_decrease(&mut self, dim: usize, before: u64, after: u64, diff: u64);

    /// Returns the current derived value after all index changes have been applied.
    fn value(&self) -> Self::Item<'_>;

    fn check_ndim(&self, ndim: usize) -> bool;

    /// The merged extension type of `Self` and `E`.
    ///
    /// A merge is performed like appending to a list:
    /// - `()` is the identity of append: `() + E = E`
    /// - A concrete extension `X` behaves as a one-element list: `X + E = (X, E)`
    /// - An `n`-tuple appends to become an `(n + 1)`-tuple: `(X1, X2, ..., Xn) + E = (X1, X2, ..., Xn, E)`
    type MergeExtension<E: NdIterExtension>: NdIterExtension;

    /// Merge `self` with another extension `other`, producing a new extension that tracks both.
    fn merge_extension<E: NdIterExtension>(self, other: E) -> Self::MergeExtension<E>
    where
        Self: Sized;
}

/// A plain index-only iterator; a thin wrapper around [`NdIter`] with a `()` extension.
#[allow(unused)]
pub(crate) struct IdxIter<D: Dimension>(NdIter<D, ()>);

#[allow(unused)]
impl<D> IdxIter<D>
where
    D: Dimension,
{
    #[inline]
    pub(crate) fn new<V>(shape: V) -> Self
    where
        D: Dimension<Vec<u64> = V>,
        V: DimVec<u64, Dimension = D>,
    {
        Self(NdIter::new(shape, ()))
    }

    /// Returns the next multi-dimensional index, or `None` when exhausted.
    #[inline(always)]
    pub(crate) fn next(&mut self) -> Option<D::Vec<u64>> {
        Some(self.0.next()?.0)
    }
}

// ---------------------------------------------------------------------------
// Tuple blanket impls - compose multiple extensions so that a single NdIter
// can maintain several pieces of derived state simultaneously.
// ---------------------------------------------------------------------------

impl NdIterExtension for () {
    type Item<'a> = ();
    #[inline(always)]
    fn on_increase(&mut self, _dim: usize, _before: u64, _after: u64, _diff: u64) {}
    #[inline(always)]
    fn on_decrease(&mut self, _dim: usize, _before: u64, _after: u64, _diff: u64) {}
    #[inline(always)]
    fn value(&self) {}
    #[inline(always)]
    fn check_ndim(&self, _ndim: usize) -> bool {
        true
    }

    /// The merged extension type of `()` and `E` is just `E`.
    type MergeExtension<E: NdIterExtension> = E;

    #[inline(always)]
    fn merge_extension<E: NdIterExtension>(self, ext: E) -> E {
        ext
    }
}
impl<T1> NdIterExtension for (T1,)
where
    T1: NdIterExtension,
{
    type Item<'a>
        = (T1::Item<'a>,)
    where
        Self: 'a;
    #[inline(always)]
    fn on_increase(&mut self, dim: usize, before: u64, after: u64, diff: u64) {
        self.0.on_increase(dim, before, after, diff);
    }
    #[inline(always)]
    fn on_decrease(&mut self, dim: usize, before: u64, after: u64, diff: u64) {
        self.0.on_decrease(dim, before, after, diff);
    }
    #[inline(always)]
    fn value(&self) -> Self::Item<'_> {
        (self.0.value(),)
    }
    #[inline(always)]
    fn check_ndim(&self, ndim: usize) -> bool {
        self.0.check_ndim(ndim)
    }

    type MergeExtension<E: NdIterExtension> = (T1, E);
    #[inline(always)]
    fn merge_extension<E: NdIterExtension>(self, ext: E) -> (T1, E) {
        (self.0, ext)
    }
}
impl<T1, T2> NdIterExtension for (T1, T2)
where
    T1: NdIterExtension,
    T2: NdIterExtension,
{
    type Item<'a>
        = (T1::Item<'a>, T2::Item<'a>)
    where
        Self: 'a;
    #[inline(always)]
    fn on_increase(&mut self, dim: usize, before: u64, after: u64, diff: u64) {
        self.0.on_increase(dim, before, after, diff);
        self.1.on_increase(dim, before, after, diff);
    }
    #[inline(always)]
    fn on_decrease(&mut self, dim: usize, before: u64, after: u64, diff: u64) {
        self.0.on_decrease(dim, before, after, diff);
        self.1.on_decrease(dim, before, after, diff);
    }
    #[inline(always)]
    fn value(&self) -> Self::Item<'_> {
        (self.0.value(), self.1.value())
    }
    #[inline(always)]
    fn check_ndim(&self, ndim: usize) -> bool {
        self.0.check_ndim(ndim) && self.1.check_ndim(ndim)
    }

    type MergeExtension<E: NdIterExtension> = (T1, T2, E);
    #[inline(always)]
    fn merge_extension<E: NdIterExtension>(self, ext: E) -> (T1, T2, E) {
        (self.0, self.1, ext)
    }
}
impl<T1, T2, T3> NdIterExtension for (T1, T2, T3)
where
    T1: NdIterExtension,
    T2: NdIterExtension,
    T3: NdIterExtension,
{
    type Item<'a>
        = (T1::Item<'a>, T2::Item<'a>, T3::Item<'a>)
    where
        Self: 'a;

    #[inline(always)]
    fn on_increase(&mut self, dim: usize, before: u64, after: u64, diff: u64) {
        self.0.on_increase(dim, before, after, diff);
        self.1.on_increase(dim, before, after, diff);
        self.2.on_increase(dim, before, after, diff);
    }
    #[inline(always)]
    fn on_decrease(&mut self, dim: usize, before: u64, after: u64, diff: u64) {
        self.0.on_decrease(dim, before, after, diff);
        self.1.on_decrease(dim, before, after, diff);
        self.2.on_decrease(dim, before, after, diff);
    }
    #[inline(always)]
    fn value(&self) -> Self::Item<'_> {
        (self.0.value(), self.1.value(), self.2.value())
    }
    #[inline(always)]
    fn check_ndim(&self, ndim: usize) -> bool {
        self.0.check_ndim(ndim) && self.1.check_ndim(ndim) && self.2.check_ndim(ndim)
    }

    // Three extensions is the terminal arity of the extension list: merging a fourth is unsupported,
    // so `MergeExtension` resolves to `()` and `merge_extension` panics. No call site merges past three (real
    // usage never exceeds two), so this is unreachable in practice.
    type MergeExtension<E: NdIterExtension> = ();
    #[inline(always)]
    fn merge_extension<E: NdIterExtension>(self, _ext: E) {
        unimplemented!(
            "NdIter extension list is capped at three extensions; cannot append a fourth"
        )
    }
}

macro_rules! impl_merge_extension {
    () => {
        type MergeExtension<E: $crate::util::iter::NdIterExtension> = (Self, E);
        #[inline(always)]
        fn merge_extension<E: $crate::util::iter::NdIterExtension>(self, ext: E) -> (Self, E) {
            (self, ext)
        }
    };
}
pub(crate) use impl_merge_extension;

/// A builder for [`NdIter`] that accumulates extensions one at a time.
///
/// Start from [`NdIter::builder`] (or [`NdIter::builder_with_begin`]) with an empty extension list
/// (`E = ()`), chain any number of `with_*_ext` calls - each appends one extension and advances the
/// `E` type via [`NdIterExtension::MergeExtension`] - then call [`build`](NdIterBuilder::build).
pub(crate) struct NdIterBuilder<D: Dimension, E: NdIterExtension> {
    begin: D::Vec<u64>,
    end: D::Vec<u64>,
    extensions: E,
}

impl<D: Dimension> NdIter<D, ()> {
    /// Starts a builder iterating `[0, shape)` in every dimension, with no extensions.
    #[inline]
    pub(crate) fn builder<V>(shape: V) -> NdIterBuilder<D, ()>
    where
        D: Dimension<Vec<u64> = V>,
        V: DimVec<u64, Dimension = D>,
    {
        let begin = V::Dimension::vec(shape.as_ref().len(), |_| 0u64);
        NdIterBuilder {
            begin,
            end: shape,
            extensions: (),
        }
    }

    /// Starts a builder iterating `[begin, end)` in every dimension, with no extensions.
    #[inline]
    pub(crate) fn builder_with_begin<V>(begin: V, end: V) -> NdIterBuilder<D, ()>
    where
        D: Dimension<Vec<u64> = V>,
        V: DimVec<u64, Dimension = D>,
    {
        NdIterBuilder {
            begin,
            end,
            extensions: (),
        }
    }
}

impl<D: Dimension, E: NdIterExtension> NdIterBuilder<D, E> {
    /// Adds a [`NdIterExtStridesPtr`]  extension.
    #[inline]
    pub(crate) fn with_strides_ptr_ext<T, S>(
        self,
        strides: D::Vec<S>,
        initial_ptr: *const T,
    ) -> NdIterBuilder<D, E::MergeExtension<NdIterExtStridesPtr<D, T, S>>>
    where
        S: Idx,
    {
        let ext = NdIterExtStridesPtr::<D, T, S>::new(strides, initial_ptr);
        NdIterBuilder {
            begin: self.begin,
            end: self.end,
            extensions: self.extensions.merge_extension(ext),
        }
    }

    /// Adds a [`NdIterExtStridesPtrMut`] extension.
    #[inline]
    pub(crate) fn with_strides_ptr_mut_ext<T, S>(
        self,
        strides: D::Vec<S>,
        initial_ptr: *mut T,
    ) -> NdIterBuilder<D, E::MergeExtension<NdIterExtStridesPtrMut<D, T, S>>>
    where
        S: Idx,
    {
        let ext = NdIterExtStridesPtrMut::<D, T, S>::new(strides, initial_ptr);
        NdIterBuilder {
            begin: self.begin,
            end: self.end,
            extensions: self.extensions.merge_extension(ext),
        }
    }

    /// Adds a [`NdIterExtStridesOffset`] extension.
    #[inline]
    pub(crate) fn with_strides_offset_ext<S: Idx>(
        self,
        strides: D::Vec<S>,
        initial_offset: S,
    ) -> NdIterBuilder<D, E::MergeExtension<NdIterExtStridesOffset<D, S>>> {
        let ext = NdIterExtStridesOffset::<D, S>::new(strides, initial_offset);
        NdIterBuilder {
            begin: self.begin,
            end: self.end,
            extensions: self.extensions.merge_extension(ext),
        }
    }

    /// Adds a [`NdIterExtStridesOffsetMulti`] extension tracking `N` offsets at once.
    #[inline]
    pub(crate) fn with_strides_offset_multi_ext<S: Idx, const N: usize>(
        self,
        strides: [D::Vec<S>; N],
        initial_offsets: [S; N],
    ) -> NdIterBuilder<D, E::MergeExtension<NdIterExtStridesOffsetMulti<D, S, N>>> {
        let ext = NdIterExtStridesOffsetMulti::<D, S, N>::new(strides, initial_offsets);
        NdIterBuilder {
            begin: self.begin,
            end: self.end,
            extensions: self.extensions.merge_extension(ext),
        }
    }

    /// Adds a [`NdIterExtStridesOffsetMultiDyn`] extension tracking one offset per operand.
    #[inline]
    pub(crate) fn with_strides_offset_multi_dyn_ext<S: Idx>(
        self,
        strides: OperandsArray<D::Vec<S>>,
        initial_offsets: OperandsArray<S>,
    ) -> NdIterBuilder<D, E::MergeExtension<NdIterExtStridesOffsetMultiDyn<D, S>>> {
        let ext = NdIterExtStridesOffsetMultiDyn::<D, S>::new(strides, initial_offsets);
        NdIterBuilder {
            begin: self.begin,
            end: self.end,
            extensions: self.extensions.merge_extension(ext),
        }
    }

    /// Adds a [`NdIterExtStridesOffset`] extension.
    #[inline]
    pub(crate) fn with_logical_global_index_ext(
        self,
        shape: &[u64],
    ) -> NdIterBuilder<D, E::MergeExtension<NdIterExtStridesOffset<D, u64>>> {
        let ext = nd_iter_ext_logical_global_index::<D>(shape, self.begin.as_ref());
        NdIterBuilder {
            begin: self.begin,
            end: self.end,
            extensions: self.extensions.merge_extension(ext),
        }
    }

    /// Adds a [`NdIterExtBlockOffsetSize`] extension.
    #[inline]
    pub(crate) fn with_block_offset_size_ext<V>(
        self,
        begin: &V,
        end: &V,
        block_shape: V,
    ) -> NdIterBuilder<D, E::MergeExtension<NdIterExtBlockOffsetSize<D>>>
    where
        D: Dimension<Vec<u64> = V>,
        V: DimVec<u64, Dimension = D>,
    {
        let ext = NdIterExtBlockOffsetSize::<D>::new(begin, end, block_shape);
        NdIterBuilder {
            begin: self.begin,
            end: self.end,
            extensions: self.extensions.merge_extension(ext),
        }
    }

    #[inline]
    pub(crate) fn build(self) -> NdIter<D, E> {
        NdIter::new_with_begin(self.begin, self.end, self.extensions)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::util::iter::strides::NdIterExtStridesPtrMut;
    use crate::util::DimArray;
    use crate::{Dim, DimDyn, SliceExt};

    /// Build a [`DimArray`] (the `DimDyn` vec container) from a slice, for constructing test
    /// extensions whose `new` takes an owned `D::Vec<S>`.
    fn dv<T: Copy>(s: &[T]) -> DimArray<T> {
        s.to_dim_vec::<DimDyn>()
    }

    // ---------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------

    fn collect_idx<D>(mut iter: IdxIter<D>) -> Vec<Vec<u64>>
    where
        D: Dimension,
    {
        let mut out = Vec::new();
        while let Some(idx) = iter.next() {
            out.push(idx.as_ref().to_vec());
        }
        out
    }

    /// Records every `on_increase/decrease` notification it receives.
    struct ChangeLog {
        log: Vec<(usize, u64, u64)>,
    }
    impl ChangeLog {
        fn new() -> Self {
            Self { log: Vec::new() }
        }
    }
    impl NdIterExtension for ChangeLog {
        // number of on_increase/decrease calls so far when value() is called
        type Item<'a> = usize;
        fn on_increase(&mut self, dim: usize, before: u64, after: u64, _diff: u64) {
            self.log.push((dim, before, after));
        }
        fn on_decrease(&mut self, dim: usize, before: u64, after: u64, _diff: u64) {
            self.log.push((dim, before, after));
        }
        fn value(&self) -> usize {
            self.log.len()
        }
        fn check_ndim(&self, _ndim: usize) -> bool {
            true
        }
        impl_merge_extension!();
    }

    // ---------------------------------------------------------------------------
    // IdxIter - basic traversal
    // ---------------------------------------------------------------------------

    #[test]
    fn idx_iter_0d_yields_one_empty_index() {
        // A 0-D iterator has no dimensions; it should yield exactly one empty index.
        assert_eq!(collect_idx(IdxIter::<Dim<0>>::new([])), vec![vec![]]);
    }

    #[test]
    fn idx_iter_1d() {
        assert_eq!(
            collect_idx(IdxIter::<Dim<1>>::new([4u64])),
            vec![vec![0], vec![1], vec![2], vec![3]],
        );
    }

    #[test]
    fn idx_iter_1d_size_1() {
        assert_eq!(collect_idx(IdxIter::<Dim<1>>::new([1u64])), vec![vec![0]]);
    }

    #[test]
    fn idx_iter_1d_size_0() {
        assert!(collect_idx(IdxIter::<Dim<1>>::new([0u64])).is_empty());
    }

    #[test]
    fn idx_iter_2d_row_major_order() {
        assert_eq!(
            collect_idx(IdxIter::<Dim<2>>::new([2u64, 3])),
            vec![
                vec![0, 0],
                vec![0, 1],
                vec![0, 2],
                vec![1, 0],
                vec![1, 1],
                vec![1, 2],
            ],
        );
    }

    #[test]
    fn idx_iter_3d_row_major_order() {
        let got = collect_idx(IdxIter::<Dim<3>>::new([2u64, 3, 2]));
        let expected: Vec<Vec<u64>> = (0..2)
            .flat_map(|i| (0..3).flat_map(move |j| (0..2).map(move |k| vec![i, j, k])))
            .collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn idx_iter_total_count_equals_shape_product() {
        let shape: [u64; 4] = [2, 3, 4, 5];
        assert_eq!(
            collect_idx(IdxIter::<Dim<4>>::new(shape)).len(),
            2 * 3 * 4 * 5
        );
    }

    // ---------------------------------------------------------------------------
    // IdxIter - empty / exhaustion
    // ---------------------------------------------------------------------------

    #[test]
    fn idx_iter_zero_in_first_dim_is_empty() {
        assert!(collect_idx(IdxIter::<Dim<2>>::new([0u64, 3])).is_empty());
    }

    #[test]
    fn idx_iter_zero_in_last_dim_is_empty() {
        assert!(collect_idx(IdxIter::<Dim<2>>::new([3u64, 0])).is_empty());
    }

    #[test]
    fn idx_iter_zero_in_middle_dim_is_empty() {
        assert!(collect_idx(IdxIter::<Dim<3>>::new([3u64, 0, 4])).is_empty());
    }

    #[test]
    fn idx_iter_zero_1d_is_empty() {
        assert!(collect_idx(IdxIter::<Dim<1>>::new([0u64])).is_empty());
    }

    #[test]
    fn idx_iter_returns_none_repeatedly_after_exhaustion() {
        let mut iter = IdxIter::<Dim<1>>::new([2u64]);
        iter.next();
        iter.next();
        assert!(iter.next().is_none());
        assert!(iter.next().is_none());
        assert!(iter.next().is_none());
    }

    // ---------------------------------------------------------------------------
    // NdIter with new_with_begin
    // ---------------------------------------------------------------------------

    /// Generic over an abstract `E`, so the `Iterator` impl (which needs a single concrete item
    /// type for every lifetime) does not apply here - drive the lending `next` instead.
    fn collect_indices<D: Dimension, E: NdIterExtension>(mut iter: NdIter<D, E>) -> Vec<Vec<u64>> {
        let mut out = Vec::new();
        while let Some((idx, _)) = iter.next() {
            out.push(idx.as_ref().to_vec());
        }
        out
    }

    #[test]
    fn new_with_begin_1d_offset() {
        assert_eq!(
            collect_indices(NdIter::<Dim<1>, _>::new_with_begin([2u64], [5u64], ())),
            vec![vec![2], vec![3], vec![4]],
        );
    }

    #[test]
    fn new_with_begin_2d_offset() {
        let got = collect_indices(NdIter::<Dim<2>, _>::new_with_begin(
            [1u64, 2],
            [3u64, 4],
            (),
        ));
        let expected: Vec<Vec<u64>> = (1..3)
            .flat_map(|r| (2..4).map(move |c| vec![r, c]))
            .collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn new_with_begin_count_matches_range_product() {
        let got = collect_indices(NdIter::<Dim<3>, _>::new_with_begin(
            [1u64, 2, 0],
            [4u64, 5, 3],
            (),
        ));
        assert_eq!(got.len(), (4 - 1) * (5 - 2) * 3);
    }

    #[test]
    fn new_with_begin_empty_when_one_dim_degenerate() {
        // begin[0] == end[0] -> no elements even though other dims are non-empty
        assert!(collect_indices(NdIter::<Dim<2>, _>::new_with_begin(
            [2u64, 0],
            [2u64, 5],
            ()
        ))
        .is_empty());
    }

    #[test]
    fn new_with_begin_empty_when_all_dims_degenerate() {
        assert!(
            collect_indices(NdIter::<Dim<1>, _>::new_with_begin([3u64], [3u64], ())).is_empty()
        );
    }

    #[test]
    fn new_with_begin_begin_equals_zero_matches_new() {
        let via_new = collect_indices(NdIter::<Dim<2>, _>::new([3u64, 4], ()));
        let via_begin = collect_indices(NdIter::<Dim<2>, _>::new_with_begin(
            [0u64, 0],
            [3u64, 4],
            (),
        ));
        assert_eq!(via_new, via_begin);
    }

    #[test]
    fn new_with_begin_1d_offset_dyn() {
        assert_eq!(
            collect_indices(NdIter::new_with_begin(dv(&[2u64]), dv(&[5u64]), ())),
            vec![vec![2], vec![3], vec![4]],
        );
    }

    #[test]
    fn new_with_begin_2d_offset_dyn() {
        let got = collect_indices(NdIter::new_with_begin(dv(&[1u64, 2]), dv(&[3u64, 4]), ()));
        let expected: Vec<Vec<u64>> = (1..3)
            .flat_map(|r| (2..4).map(move |c| vec![r, c]))
            .collect();
        assert_eq!(got, expected);
    }

    // ---------------------------------------------------------------------------
    // on_increase / on_decrease notifications
    // ---------------------------------------------------------------------------

    #[test]
    fn on_change_not_called_on_first_step() {
        let mut iter = NdIter::<Dim<2>, _>::new([3u64, 4], ChangeLog::new());
        let (_, n_changes) = iter.next().unwrap();
        assert_eq!(n_changes, 0, "no changes on the very first step");
    }

    #[test]
    fn on_change_called_once_for_innermost_advance() {
        // [0,0] -> [0,1]: only dim 1 changes
        let mut iter = NdIter::<Dim<2>, _>::new([3u64, 4], ChangeLog::new());
        iter.next(); // emit [0,0] - no changes
        let before = iter.extensions.log.len();
        iter.next(); // emit [0,1]
        let new: Vec<_> = iter.extensions.log[before..].to_vec();
        assert_eq!(new, vec![(1usize, 0, 1)]);
    }

    #[test]
    fn on_change_called_twice_on_row_wrap() {
        // At [0,3] -> [1,0]: dim 0 increments then dim 1 resets.
        let mut iter = NdIter::<Dim<2>, _>::new([3u64, 4], ChangeLog::new());
        for _ in 0..4 {
            iter.next();
        } // reach end of first row
        let before = iter.extensions.log.len();
        iter.next(); // [0,3] -> [1,0]
        let new: Vec<_> = iter.extensions.log[before..].to_vec();
        assert_eq!(new, vec![(0, 0, 1), (1, 3, 0)]);
    }

    #[test]
    fn on_change_all_smaller_dims_reset_in_order() {
        // Shape [2,3,4]: when dim 0 increments, dims 1 and 2 both reset, in order.
        let mut iter = NdIter::<Dim<3>, _>::new([2u64, 3, 4], ChangeLog::new());
        for _ in 0..(3 * 4) {
            iter.next();
        } // reach [0,2,3]
        let before = iter.extensions.log.len();
        iter.next(); // [0,2,3] -> [1,0,0]
        let new: Vec<_> = iter.extensions.log[before..].to_vec();
        assert_eq!(new, vec![(0, 0, 1), (1, 2, 0), (2, 3, 0)]);
    }

    #[test]
    fn on_change_reset_targets_begin_not_zero() {
        // begin=[1,2], end=[3,5]: when dim 0 wraps, dim 1 should reset to 2 (begin), not 0.
        let mut iter = NdIter::<Dim<2>, _>::new_with_begin([1u64, 2], [3u64, 5], ChangeLog::new());
        // advance to [1,4] - last element of first row
        for _ in 0..3 {
            iter.next();
        }
        let before = iter.extensions.log.len();
        iter.next(); // [1,4] -> [2,2]
        let new: Vec<_> = iter.extensions.log[before..].to_vec();
        assert_eq!(new[0], (0, 1, 2), "dim 0: 1->2");
        assert_eq!(new[1], (1, 4, 2), "dim 1 resets to begin=2, not 0");
    }

    #[test]
    fn on_change_total_call_count_for_full_2d_traversal() {
        // For a [R, C] iterator the number of on_change calls is:
        //   (R*C - 1) innermost increments of dim 1,
        //   minus the (R-1) times dim 1 wraps back (each wrap fires 2 instead of 1),
        //   so total = (R*C - 1) + (R - 1) = R*C + R - 2.
        // More simply: each step after the first fires 1 change for a plain inner advance,
        // or 2 changes for a row-carry. There are (R-1) row-carries and (R*(C-1)) plain advances.
        // Total = (R-1)*2 + R*(C-1)*1.
        let (r, c): (u64, u64) = (4, 5);
        let mut iter = NdIter::<Dim<2>, _>::new([r, c], ChangeLog::new());
        while iter.next().is_some() {}
        let expected = ((r - 1) * 2 + r * (c - 1)) as usize;
        assert_eq!(iter.extensions.log.len(), expected);
    }

    // ---------------------------------------------------------------------------
    // Tuple extensions
    // ---------------------------------------------------------------------------

    #[test]
    fn tuple_1_extension_delegates() {
        let ext = ChangeLog::new();
        let mut iter = NdIter::new(dv(&[3u64, 3]), (ext,));
        while iter.next().is_some() {}
        // Behaviour should be identical to a bare ChangeLog
        assert!(!iter.extensions.0.log.is_empty());
    }

    #[test]
    fn tuple_2_two_ptr_extensions_track_independently() {
        let mut a = [0u8; 6];
        let mut b = [0u8; 12]; // b has stride 2
        let base_a = a.as_mut_ptr();
        let base_b = b.as_mut_ptr();
        let ext = (
            NdIterExtStridesPtrMut::new(dv(&[3usize, 1]), base_a),
            NdIterExtStridesPtrMut::new(dv(&[6usize, 2]), base_b),
        );
        let iter = NdIter::new(dv(&[2u64, 3]), ext);
        let mut flat = 0usize;
        for (_, (pa, pb)) in iter {
            assert_eq!(pa, unsafe { base_a.add(flat) }, "a step {flat}");
            assert_eq!(pb, unsafe { base_b.add(flat * 2) }, "b step {flat}");
            flat += 1;
        }
        assert_eq!(flat, 6);
    }

    #[test]
    fn tuple_2_ptr_and_change_log() {
        let mut data = [0u8; 4];
        let base = data.as_mut_ptr();
        let ext = (
            NdIterExtStridesPtrMut::new(dv(&[1usize]), base),
            ChangeLog::new(),
        );
        let mut iter = NdIter::new(dv(&[4u64]), ext);
        let mut ptrs: Vec<*mut u8> = Vec::new();
        for (_, (ptr, _)) in iter.by_ref() {
            ptrs.push(ptr);
        }
        // 3 on_change calls: steps 2, 3, 4 each fire once for dim 0
        assert_eq!(iter.extensions.1.log.len(), 3);
        for (i, &ptr) in ptrs.iter().enumerate() {
            assert_eq!(ptr, unsafe { base.add(i) }, "ptr {i}");
        }
    }

    #[test]
    fn tuple_3_all_three_extensions_receive_changes() {
        let mut a = [0u8; 4];
        let mut b = [0u8; 4];
        let mut c = [0u8; 4];
        let ext = (
            NdIterExtStridesPtrMut::new(dv(&[1usize]), a.as_mut_ptr()),
            NdIterExtStridesPtrMut::new(dv(&[1usize]), b.as_mut_ptr()),
            NdIterExtStridesPtrMut::new(dv(&[1usize]), c.as_mut_ptr()),
        );
        let iter = NdIter::new(dv(&[4u64]), ext);
        let mut count = 0usize;
        for (_, (pa, pb, pc)) in iter {
            let off_a = unsafe { pa.offset_from(a.as_ptr()) };
            let off_b = unsafe { pb.offset_from(b.as_ptr()) };
            let off_c = unsafe { pc.offset_from(c.as_ptr()) };
            assert_eq!(off_a, off_b, "step {count}: a vs b");
            assert_eq!(off_a, off_c, "step {count}: a vs c");
            count += 1;
        }
        assert_eq!(count, 4);
    }

    // ---------------------------------------------------------------------------
    // () extension (no-op)
    // ---------------------------------------------------------------------------

    #[test]
    fn unit_extension_yields_unit_items() {
        let iter = NdIter::<Dim<2>, _>::new([2u64, 2], ());
        for (_, item) in iter {
            let _: () = item;
        }
    }

    // ---------------------------------------------------------------------------
    // NdIterBuilder
    // ---------------------------------------------------------------------------

    #[test]
    fn builder_no_ext_matches_new() {
        let via_builder = collect_indices(NdIter::builder(dv(&[2u64, 3])).build());
        let via_new = collect_indices(NdIter::new(dv(&[2u64, 3]), ()));
        assert_eq!(via_builder, via_new);
    }

    #[test]
    fn builder_with_begin_no_ext_matches_new_with_begin() {
        let via_builder =
            collect_indices(NdIter::builder_with_begin(dv(&[1u64, 2]), dv(&[3u64, 4])).build());
        let via_new = collect_indices(NdIter::new_with_begin(dv(&[1u64, 2]), dv(&[3u64, 4]), ()));
        assert_eq!(via_builder, via_new);
    }

    #[test]
    fn builder_single_ext_yields_bare_item() {
        // A single extension appends onto `()`, so the item is the bare pointer (not a 1-tuple):
        // `ptr` below binds directly to `*mut u8` with no tuple destructuring.
        let mut data = [0u8; 6];
        let base = data.as_mut_ptr();
        let iter = NdIter::builder(dv(&[2u64, 3]))
            .with_strides_ptr_mut_ext(dv(&[3usize, 1]), base)
            .build();
        let mut flat = 0usize;
        for (_, ptr) in iter {
            let _: *mut u8 = ptr;
            assert_eq!(ptr, unsafe { base.add(flat) }, "step {flat}");
            flat += 1;
        }
        assert_eq!(flat, 6);
    }

    #[test]
    fn builder_two_exts_yield_flat_pair_in_call_order() {
        let mut src = [0u8; 6];
        let mut dst = [0u8; 12]; // dst has stride 2
        let base_src = src.as_mut_ptr();
        let base_dst = dst.as_mut_ptr();
        let iter = NdIter::builder(dv(&[2u64, 3]))
            .with_strides_ptr_ext(dv(&[3usize, 1]), base_src.cast_const())
            .with_strides_ptr_mut_ext(dv(&[6usize, 2]), base_dst)
            .build();
        let mut flat = 0usize;
        for (_, (sp, dp)) in iter {
            let _: *const u8 = sp;
            let _: *mut u8 = dp;
            assert_eq!(sp, unsafe { base_src.add(flat) }, "src step {flat}");
            assert_eq!(dp, unsafe { base_dst.add(flat * 2) }, "dst step {flat}");
            flat += 1;
        }
        assert_eq!(flat, 6);
    }

    #[test]
    fn builder_two_exts_match_manual_tuple() {
        let a = [0u8; 6];
        let mut b = [0u8; 6];
        let base_a = a.as_ptr();
        let base_b = b.as_mut_ptr();
        // The same two extensions, once via the builder and once via a manual tuple constructor.
        let built: Vec<(*const u8, *mut u8)> = NdIter::builder(dv(&[2u64, 3]))
            .with_strides_ptr_ext(dv(&[3usize, 1]), base_a)
            .with_strides_ptr_mut_ext(dv(&[3usize, 1]), base_b)
            .build()
            .map(|(_, pair)| pair)
            .collect();
        let manual: Vec<(*const u8, *mut u8)> = NdIter::new(
            dv(&[2u64, 3]),
            (
                NdIterExtStridesPtr::new(dv(&[3usize, 1]), base_a),
                NdIterExtStridesPtrMut::new(dv(&[3usize, 1]), base_b),
            ),
        )
        .map(|(_, pair)| pair)
        .collect();
        assert_eq!(built, manual);
    }

    #[test]
    fn builder_three_exts_yield_flat_triple() {
        let mut a = [0u8; 4];
        let mut b = [0u8; 4];
        let mut c = [0u8; 4];
        let iter = NdIter::builder(dv(&[4u64]))
            .with_strides_ptr_mut_ext(dv(&[1usize]), a.as_mut_ptr())
            .with_strides_ptr_mut_ext(dv(&[1usize]), b.as_mut_ptr())
            .with_strides_ptr_mut_ext(dv(&[1usize]), c.as_mut_ptr())
            .build();
        let mut count = 0usize;
        for (_, (pa, pb, pc)) in iter {
            let off_a = unsafe { pa.offset_from(a.as_ptr()) };
            let off_b = unsafe { pb.offset_from(b.as_ptr()) };
            let off_c = unsafe { pc.offset_from(c.as_ptr()) };
            assert_eq!(off_a, off_b, "step {count}: a vs b");
            assert_eq!(off_a, off_c, "step {count}: a vs c");
            count += 1;
        }
        assert_eq!(count, 4);
    }
}
