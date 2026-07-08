use std::hint::unreachable_unchecked;

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
        assert!(self.is_not_started());
        self.0 = -self.0;
    }
    #[inline(always)]
    fn advance(&mut self) {
        assert!(self.is_in_progress());
        self.0 -= 1;
    }
}

impl<D, E> NdIter<D, E>
where
    D: Dimension,
    E: NdIterExtension,
{
    /// Creates an iterator over `[0, shape)` in every dimension.
    #[inline(always)]
    pub(crate) fn new<V>(shape: V, extensions: E) -> Self
    where
        D: Dimension<Vec<u64> = V>,
        V: DimVec<u64, Dimension = D>,
    {
        let begin = V::Dimension::vec(shape.as_ref().len(), |_| 0u64);
        Self::new_with_begin(begin, shape, extensions)
    }

    /// Creates an iterator over `[begin, end)` in every dimension.
    #[inline(always)]
    pub(crate) fn new_with_begin<V>(begin: V, end: V, extensions: E) -> Self
    where
        D: Dimension<Vec<u64> = V>,
        V: DimVec<u64, Dimension = D>,
    {
        let begin_slice = begin.as_ref();
        let end_slice = end.as_ref();
        let ndim = begin_slice.len();
        assert_eq!(begin_slice.len(), ndim);
        assert_eq!(end_slice.len(), ndim);
        extensions.assert_ndim(ndim);
        assert!(begin_slice
            .iter()
            .zip(end_slice.iter())
            .all(|(&b, &e)| b <= e));
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

    #[inline(always)]
    pub(crate) fn get_current_and_advance_status(&mut self) -> (D::Vec<u64>, E::Item) {
        self.status.advance();
        (self.current_idx.clone(), self.extensions.next())
    }

    #[inline(always)]
    pub(crate) fn len(&self) -> u64 {
        self.status.len()
    }
}
impl<D, E> Iterator for NdIter<D, E>
where
    D: Dimension,
    E: NdIterExtension,
{
    type Item = (D::Vec<u64>, E::Item);

    /// Advances to the next index in row-major order and returns `(current_index, extension_item)`.
    ///
    /// On each step the rightmost dimension that has not yet reached its bound is incremented,
    /// and all dimensions to its right are reset to `begin`.
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.status.is_exhausted() {
            return None;
        }

        if self.status.is_not_started() {
            self.status.start();
            return Some(self.get_current_and_advance_status());
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
                return Some(self.get_current_and_advance_status());
            }
        }
        unsafe { unreachable_unchecked() }
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
/// return the current derived value via [`next`](NdIterExtension::next).
pub(crate) trait NdIterExtension {
    /// The derived value produced at each iteration step.
    type Item;

    /// Called when dimension `dim` changes from `before` to `after`.
    ///
    /// All dimension changes for a single step are delivered before [`next`](NdIterExtension::next)
    /// is called.
    fn on_increase(&mut self, dim: usize, before: u64, after: u64, diff: u64);
    fn on_decrease(&mut self, dim: usize, before: u64, after: u64, diff: u64);

    /// Returns the current derived value after all index changes have been applied.
    fn next(&self) -> Self::Item;

    fn assert_ndim(&self, ndim: usize);
}

/// A plain index-only iterator; a thin wrapper around [`NdIter`] with a `()` extension.
#[allow(unused)]
pub(crate) struct IdxIter<D: Dimension>(NdIter<D, ()>);

#[allow(unused)]
impl<D> IdxIter<D>
where
    D: Dimension,
{
    #[inline(always)]
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
    type Item = ();
    #[inline(always)]
    fn on_increase(&mut self, _dim: usize, _before: u64, _after: u64, _diff: u64) {}
    #[inline(always)]
    fn on_decrease(&mut self, _dim: usize, _before: u64, _after: u64, _diff: u64) {}
    #[inline(always)]
    fn next(&self) {}
    #[inline(always)]
    fn assert_ndim(&self, _ndim: usize) {}
}
impl<T1> NdIterExtension for (T1,)
where
    T1: NdIterExtension,
{
    type Item = (T1::Item,);
    #[inline(always)]
    fn on_increase(&mut self, dim: usize, before: u64, after: u64, diff: u64) {
        self.0.on_increase(dim, before, after, diff);
    }
    #[inline(always)]
    fn on_decrease(&mut self, dim: usize, before: u64, after: u64, diff: u64) {
        self.0.on_decrease(dim, before, after, diff);
    }
    #[inline(always)]
    fn next(&self) -> (T1::Item,) {
        (self.0.next(),)
    }
    #[inline(always)]
    fn assert_ndim(&self, ndim: usize) {
        self.0.assert_ndim(ndim);
    }
}
impl<T1, T2> NdIterExtension for (T1, T2)
where
    T1: NdIterExtension,
    T2: NdIterExtension,
{
    type Item = (T1::Item, T2::Item);
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
    fn next(&self) -> (T1::Item, T2::Item) {
        (self.0.next(), self.1.next())
    }
    #[inline(always)]
    fn assert_ndim(&self, ndim: usize) {
        self.0.assert_ndim(ndim);
        self.1.assert_ndim(ndim);
    }
}
impl<T1, T2, T3> NdIterExtension for (T1, T2, T3)
where
    T1: NdIterExtension,
    T2: NdIterExtension,
    T3: NdIterExtension,
{
    type Item = (T1::Item, T2::Item, T3::Item);
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
    fn next(&self) -> (T1::Item, T2::Item, T3::Item) {
        (self.0.next(), self.1.next(), self.2.next())
    }
    #[inline(always)]
    fn assert_ndim(&self, ndim: usize) {
        self.0.assert_ndim(ndim);
        self.1.assert_ndim(ndim);
        self.2.assert_ndim(ndim);
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::util::iter::strides::NdIterExtStridesPtrMut;
    use crate::util::DimArray;
    use crate::{Dim, DimDyn, SliceExt};

    /// Build a [`DimArray`] (the `DimDyn` vec container) from a slice, for constructing test
    /// extensions whose `new` now takes an owned `D::Vec<S>`.
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
        type Item = usize; // number of on_increase/decrease calls so far when next() is called
        fn on_increase(&mut self, dim: usize, before: u64, after: u64, _diff: u64) {
            self.log.push((dim, before, after));
        }
        fn on_decrease(&mut self, dim: usize, before: u64, after: u64, _diff: u64) {
            self.log.push((dim, before, after));
        }
        fn next(&self) -> usize {
            self.log.len()
        }
        fn assert_ndim(&self, _ndim: usize) {}
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
        assert_eq!(got.len(), (4 - 1) * (5 - 2) * (3 - 0));
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
        let mut iter = NdIter::new(dv(&[2u64, 3]), ext);
        let mut flat = 0usize;
        while let Some((_, (pa, pb))) = iter.next() {
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
        while let Some((_, (ptr, _))) = iter.next() {
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
        let mut iter = NdIter::new(dv(&[4u64]), ext);
        let mut count = 0usize;
        while let Some((_, (pa, pb, pc))) = iter.next() {
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
        let mut iter = NdIter::<Dim<2>, _>::new([2u64, 2], ());
        while let Some((_, item)) = iter.next() {
            let _: () = item;
        }
    }
}
