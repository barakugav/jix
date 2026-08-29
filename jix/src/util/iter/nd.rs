use crate::util::iter::block::NdIterExtBlockOffsetSize;
use crate::util::iter::strides::{
    nd_iter_ext_logical_global_index, NdIterExtStridesOffset, NdIterExtStridesOffsetMulti,
    NdIterExtStridesOffsetMultiDyn,
};
use crate::util::Idx;
use crate::{DimVec, Dimension};

/// A multi-dimensional iterator that advances indices in row-major (C) order.
///
/// Extensions are supported via the generic parameter `E`, see [`NdIterExtension`].
/// The iterator notifies the extension on each index change, allowing extensions to track derived
/// state (e.g. a pointer into a strided buffer) without recomputing it from scratch.
#[derive(Clone)]
pub(crate) struct NdIter<D: Dimension, E> {
    shape: D::Vec<u64>,
    /// Positions left on each axis
    counters: D::Vec<u64>,
    /// The index of the item to be yielded next
    current_idx: D::Vec<u64>,
    exhausted: bool,
    pub(crate) extensions: E,
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
        let shape = V::Dimension::vec(ndim, |dim| end_slice[dim] - begin_slice[dim]);
        let nitems = shape.as_ref().iter().product::<u64>();
        let counters = DimVec::clone(&shape);

        NdIter {
            shape,
            counters,
            current_idx: begin,
            exhausted: nitems == 0,
            extensions,
        }
    }

    /// Advance the iterator to the next index, returning `false` if the iterator is exhausted.
    #[inline(always)]
    fn advance(&mut self) -> bool {
        if let Some(ndim) = D::NDIM {
            macro_rules! carry_ladder {
                ($this:ident, $ndim:ident $(, $dim:literal)*) => {
                    $(
                        if $dim < $ndim && $this.advance_axis($dim) {
                            return true;
                        }
                    )*
                };
            }
            carry_ladder!(self, ndim, 7, 6, 5, 4, 3, 2, 1, 0);
        } else {
            let ndim = self.counters.as_ref().len();
            for dim in (0..ndim).rev() {
                if self.advance_axis(dim) {
                    return true;
                }
            }
        }
        false
    }

    /// Advance the iterator along a single axis, returning `false` if the axis wrapped back to `begin`.
    #[inline(always)]
    fn advance_axis(&mut self, dim: usize) -> bool {
        // `counters[dim]` counts the current position too, so it is >= 1 here and cannot underflow.
        debug_assert!(self.counters[dim] >= 1);
        let remaining = self.counters[dim] - 1;
        self.counters[dim] = remaining;

        if remaining != 0 {
            let after = self.current_idx[dim] + 1;
            self.current_idx[dim] = after;
            self.extensions.on_increase(dim, after, 1);
            return true;
        }

        // axis overflow, reset to begin
        let axis_len = self.shape[dim];
        let diff = axis_len - 1;
        let begin = self.current_idx[dim] - diff;
        self.counters[dim] = axis_len;
        self.current_idx[dim] = begin;
        self.extensions.on_decrease(dim, begin, diff);
        false
    }

    #[inline(always)]
    pub(crate) fn for_each(mut self, mut f: impl FnMut(&D::Vec<u64>, E::Item<'_>)) {
        if self.exhausted {
            return;
        }
        loop {
            f(&self.current_idx, self.extensions.value());
            if !self.advance() {
                return;
            }
        }
    }

    #[inline]
    fn remaining(&self) -> u64 {
        if self.exhausted {
            return 0;
        }
        let (shape, counters) = (self.shape.as_ref(), self.counters.as_ref());
        let mut remaining = 1;
        let mut axis_volume = 1;
        for dim in (0..counters.len()).rev() {
            debug_assert!(self.counters[dim] >= 1);
            remaining += (counters[dim] - 1) * axis_volume;
            axis_volume *= shape[dim];
        }
        remaining
    }
}

impl<D, E, I> Iterator for NdIter<D, E>
where
    D: Dimension,
    // `NdIter` is a real [`Iterator`] whenever its extension's item does not borrow from it.
    E: 'static + for<'a> NdIterExtension<Item<'a> = I>,
{
    type Item = (D::Vec<u64>, I);

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }
        // `I` does not borrow from the extension, so the item can be read off the current
        // position before we call `advance` - the same do-while shape as `for_each`.
        let item = (DimVec::clone(&self.current_idx), self.extensions.value());
        self.exhausted = !self.advance();
        Some(item)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match usize::try_from(self.remaining()) {
            Ok(len) => (len, Some(len)),
            Err(_) => (usize::MAX, None),
        }
    }
}

/// An extension trait for [`NdIter`] that tracks derived state alongside the current index.
///
/// Instead of recomputing derived state (e.g. a raw pointer offset) from scratch at every step,
/// implementors receive incremental [`on_increase`](NdIterExtension::on_increase) and
/// [`on_decrease`](NdIterExtension::on_decrease) notifications and
/// return the current derived value via [`value`](NdIterExtension::value).
pub(crate) trait NdIterExtension {
    /// The derived value produced at each iteration step, borrows from the extension.
    type Item<'a>
    where
        Self: 'a;

    /// Called when dimension `dim` moves to `after`, a change of `diff` positions.
    ///
    /// A single step delivers its changes right to left, so the axes that wrapped back to `begin`
    /// are reported (as decreases) before the axis that was incremented. All of a step's changes
    /// arrive before [`value`](NdIterExtension::value) is called, and the last element's step also
    /// reports the wrap that runs off the end of the walk.
    fn on_increase(&mut self, dim: usize, after: u64, diff: u64);
    fn on_decrease(&mut self, dim: usize, after: u64, diff: u64);

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
    fn on_increase(&mut self, _dim: usize, _after: u64, _diff: u64) {}
    #[inline(always)]
    fn on_decrease(&mut self, _dim: usize, _after: u64, _diff: u64) {}
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
    fn on_increase(&mut self, dim: usize, after: u64, diff: u64) {
        self.0.on_increase(dim, after, diff);
    }
    #[inline(always)]
    fn on_decrease(&mut self, dim: usize, after: u64, diff: u64) {
        self.0.on_decrease(dim, after, diff);
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
    fn on_increase(&mut self, dim: usize, after: u64, diff: u64) {
        self.0.on_increase(dim, after, diff);
        self.1.on_increase(dim, after, diff);
    }
    #[inline(always)]
    fn on_decrease(&mut self, dim: usize, after: u64, diff: u64) {
        self.0.on_decrease(dim, after, diff);
        self.1.on_decrease(dim, after, diff);
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
    fn on_increase(&mut self, dim: usize, after: u64, diff: u64) {
        self.0.on_increase(dim, after, diff);
        self.1.on_increase(dim, after, diff);
        self.2.on_increase(dim, after, diff);
    }
    #[inline(always)]
    fn on_decrease(&mut self, dim: usize, after: u64, diff: u64) {
        self.0.on_decrease(dim, after, diff);
        self.1.on_decrease(dim, after, diff);
        self.2.on_decrease(dim, after, diff);
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
        strides: Vec<D::Vec<S>>,
        initial_offsets: Vec<S>,
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
    use crate::util::iter::strides::NdIterExtStridesOffset;
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
        log: Vec<(usize, u64)>,
    }
    impl ChangeLog {
        fn new() -> Self {
            Self { log: Vec::new() }
        }
    }
    impl NdIterExtension for ChangeLog {
        // number of on_increase/decrease calls so far when value() is called
        type Item<'a> = usize;
        fn on_increase(&mut self, dim: usize, after: u64, _diff: u64) {
            self.log.push((dim, after));
        }
        fn on_decrease(&mut self, dim: usize, after: u64, _diff: u64) {
            self.log.push((dim, after));
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
    // size_hint
    // ---------------------------------------------------------------------------

    /// Walks `iter` to exhaustion, checking the hint against the number of items actually left
    /// before every step - `size_hint` is derived from the counters, so a stale or off-by-one
    /// derivation would only show up at some interior step, not at the ends.
    fn assert_size_hint_exact<D: Dimension>(mut iter: NdIter<D, ()>, total: usize) {
        for left in (0..=total).rev() {
            assert_eq!(iter.size_hint(), (left, Some(left)), "{left} item(s) left");
            assert_eq!(iter.next().is_some(), left > 0, "{left} item(s) left");
        }
        assert_eq!(iter.size_hint(), (0, Some(0)), "past the end");
    }

    #[test]
    fn size_hint_is_exact_at_every_step() {
        assert_size_hint_exact(NdIter::<Dim<0>, _>::new([], ()), 1);
        assert_size_hint_exact(NdIter::<Dim<1>, _>::new([5u64], ()), 5);
        assert_size_hint_exact(NdIter::<Dim<2>, _>::new([3u64, 4], ()), 12);
        assert_size_hint_exact(NdIter::<Dim<3>, _>::new([2u64, 3, 4], ()), 24);
        assert_size_hint_exact(NdIter::<DimDyn, _>::new(dv(&[2u64, 3, 2]), ()), 12);
    }

    #[test]
    fn size_hint_counts_the_range_not_the_index() {
        // begin=[1,2], end=[4,5]: 3*3 items, however far from the origin they sit.
        assert_size_hint_exact(
            NdIter::<Dim<2>, _>::new_with_begin([1u64, 2], [4u64, 5], ()),
            9,
        );
    }

    #[test]
    fn size_hint_is_zero_for_an_empty_range() {
        assert_size_hint_exact(NdIter::<Dim<2>, _>::new([4u64, 0], ()), 0);
        assert_size_hint_exact(NdIter::<Dim<3>, _>::new([3u64, 0, 4], ()), 0);
        assert_size_hint_exact(
            NdIter::<Dim<2>, _>::new_with_begin([2u64, 2], [2u64, 5], ()),
            0,
        );
    }

    #[test]
    fn size_hint_sizes_the_collect_allocation() {
        let out: Vec<_> = NdIter::<Dim<2>, _>::new([3u64, 4], ()).collect();
        assert_eq!(out.len(), 12);
        assert_eq!(
            out.capacity(),
            12,
            "exact hint should collect without regrowing"
        );
    }

    // ---------------------------------------------------------------------------
    // NdIter with new_with_begin
    // ---------------------------------------------------------------------------

    /// Generic over an abstract `E`, so the `Iterator` impl (which needs a single concrete item
    /// type for every lifetime) does not apply here - drive the lending `for_each` instead.
    fn collect_indices<D: Dimension, E: NdIterExtension>(iter: NdIter<D, E>) -> Vec<Vec<u64>> {
        let mut out = Vec::new();
        iter.for_each(|idx, _| out.push(idx.as_ref().to_vec()));
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
        // The walk yields first and carries afterwards, so the notifications a step produces
        // describe the move *out* of the index it just yielded. Here: [0,0] -> [0,1].
        let mut iter = NdIter::<Dim<2>, _>::new([3u64, 4], ChangeLog::new());
        iter.next(); // emit [0,0], then carry to [0,1]
        assert_eq!(iter.extensions.log, vec![(1usize, 1)]);
    }

    #[test]
    fn on_change_called_twice_on_row_wrap() {
        // Carrying out of [0,3] into [1,0]: dim 1 resets, then the carry moves left and dim 0
        // increments.
        let mut iter = NdIter::<Dim<2>, _>::new([3u64, 4], ChangeLog::new());
        for _ in 0..3 {
            iter.next();
        } // emit [0,0], [0,1], [0,2]
        let before = iter.extensions.log.len();
        iter.next(); // emit [0,3], then carry to [1,0]
        let new: Vec<_> = iter.extensions.log[before..].to_vec();
        assert_eq!(new, vec![(1, 0), (0, 1)]);
    }

    #[test]
    fn on_change_all_smaller_dims_reset_in_order() {
        // Shape [2,3,4]: the carry resets dims 2 and 1 on its way left, then increments dim 0.
        let mut iter = NdIter::<Dim<3>, _>::new([2u64, 3, 4], ChangeLog::new());
        for _ in 0..(3 * 4 - 1) {
            iter.next();
        } // emit up to [0,2,2]
        let before = iter.extensions.log.len();
        iter.next(); // emit [0,2,3], then carry to [1,0,0]
        let new: Vec<_> = iter.extensions.log[before..].to_vec();
        assert_eq!(new, vec![(2, 0), (1, 0), (0, 1)]);
    }

    #[test]
    fn on_change_reset_targets_begin_not_zero() {
        // begin=[1,2], end=[3,5]: when dim 0 wraps, dim 1 should reset to 2 (begin), not 0.
        let mut iter = NdIter::<Dim<2>, _>::new_with_begin([1u64, 2], [3u64, 5], ChangeLog::new());
        // emit [1,2], [1,3] - up to the element before the last of the first row
        for _ in 0..2 {
            iter.next();
        }
        let before = iter.extensions.log.len();
        iter.next(); // emit [1,4], then carry to [2,2]
        let new: Vec<_> = iter.extensions.log[before..].to_vec();
        assert_eq!(new[0], (1, 2), "dim 1 resets to begin=2, not 0");
        assert_eq!(new[1], (0, 2), "dim 0: 1->2");
    }

    #[test]
    fn on_change_total_call_count_for_full_2d_traversal() {
        // Every one of the R*C emitted elements is followed by a carry:
        //   R*(C-1) of them advance dim 1 only                        -> 1 change each,
        //   R-1 of them end a row, wrapping dim 1 and bumping dim 0   -> 2 changes each,
        //   and the very last one wraps both dims off the end         -> 2 changes.
        let (r, c): (u64, u64) = (4, 5);
        let mut iter = NdIter::<Dim<2>, _>::new([r, c], ChangeLog::new());
        while iter.next().is_some() {}
        let expected = (r * (c - 1) + (r - 1) * 2 + 2) as usize;
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
    fn tuple_2_two_offset_extensions_track_independently() {
        // b has stride 2
        let ext = (
            NdIterExtStridesOffset::new(dv(&[3usize, 1]), 0),
            NdIterExtStridesOffset::new(dv(&[6usize, 2]), 0),
        );
        let iter = NdIter::new(dv(&[2u64, 3]), ext);
        let mut flat = 0usize;
        for (_, (oa, ob)) in iter {
            assert_eq!(oa, flat, "a step {flat}");
            assert_eq!(ob, flat * 2, "b step {flat}");
            flat += 1;
        }
        assert_eq!(flat, 6);
    }

    #[test]
    fn tuple_2_offset_and_change_log() {
        let ext = (
            NdIterExtStridesOffset::new(dv(&[1usize]), 0),
            ChangeLog::new(),
        );
        let mut iter = NdIter::new(dv(&[4u64]), ext);
        let mut offsets: Vec<usize> = Vec::new();
        for (_, (offset, _)) in iter.by_ref() {
            offsets.push(offset);
        }
        // One carry per emitted element: three that advance dim 0, plus the final one that
        // wraps it off the end.
        assert_eq!(iter.extensions.1.log.len(), 4);
        for (i, &offset) in offsets.iter().enumerate() {
            assert_eq!(offset, i, "offset {i}");
        }
    }

    #[test]
    fn tuple_3_all_three_extensions_receive_changes() {
        let ext = (
            NdIterExtStridesOffset::new(dv(&[1usize]), 0),
            NdIterExtStridesOffset::new(dv(&[1usize]), 0),
            NdIterExtStridesOffset::new(dv(&[1usize]), 0),
        );
        let iter = NdIter::new(dv(&[4u64]), ext);
        let mut count = 0usize;
        for (_, (off_a, off_b, off_c)) in iter {
            assert_eq!(off_a, count, "step {count}: a");
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
        // A single extension appends onto `()`, so the item is the bare offset (not a 1-tuple):
        // `offset` below binds directly to `usize` with no tuple destructuring.
        let iter = NdIter::builder(dv(&[2u64, 3]))
            .with_strides_offset_ext(dv(&[3usize, 1]), 0)
            .build();
        let mut flat = 0usize;
        for (_, offset) in iter {
            let _: usize = offset;
            assert_eq!(offset, flat, "step {flat}");
            flat += 1;
        }
        assert_eq!(flat, 6);
    }

    #[test]
    fn builder_two_exts_yield_flat_pair_in_call_order() {
        // dst has stride 2, and a different offset type than src.
        let iter = NdIter::builder(dv(&[2u64, 3]))
            .with_strides_offset_ext(dv(&[3usize, 1]), 0usize)
            .with_strides_offset_ext(dv(&[6u64, 2]), 0u64)
            .build();
        let mut flat = 0usize;
        for (_, (src, dst)) in iter {
            let _: usize = src;
            let _: u64 = dst;
            assert_eq!(src, flat, "src step {flat}");
            assert_eq!(dst, flat as u64 * 2, "dst step {flat}");
            flat += 1;
        }
        assert_eq!(flat, 6);
    }

    #[test]
    fn builder_two_exts_match_manual_tuple() {
        // The same two extensions, once via the builder and once via a manual tuple constructor.
        let built: Vec<(usize, usize)> = NdIter::builder(dv(&[2u64, 3]))
            .with_strides_offset_ext(dv(&[3usize, 1]), 0)
            .with_strides_offset_ext(dv(&[6usize, 2]), 0)
            .build()
            .map(|(_, pair)| pair)
            .collect();
        let manual: Vec<(usize, usize)> = NdIter::new(
            dv(&[2u64, 3]),
            (
                NdIterExtStridesOffset::new(dv(&[3usize, 1]), 0),
                NdIterExtStridesOffset::new(dv(&[6usize, 2]), 0),
            ),
        )
        .map(|(_, pair)| pair)
        .collect();
        assert_eq!(built, manual);
    }

    #[test]
    fn builder_three_exts_yield_flat_triple() {
        let iter = NdIter::builder(dv(&[4u64]))
            .with_strides_offset_ext(dv(&[1usize]), 0)
            .with_strides_offset_ext(dv(&[1usize]), 0)
            .with_strides_offset_ext(dv(&[1usize]), 0)
            .build();
        let mut count = 0usize;
        for (_, (off_a, off_b, off_c)) in iter {
            assert_eq!(off_a, count, "step {count}: a");
            assert_eq!(off_a, off_b, "step {count}: a vs b");
            assert_eq!(off_a, off_c, "step {count}: a vs c");
            count += 1;
        }
        assert_eq!(count, 4);
    }
}
