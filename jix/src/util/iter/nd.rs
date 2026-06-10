use std::hint::unreachable_unchecked;

use crate::util::{dim_arr, DimArray, Idx};

/// A multi-dimensional iterator that advances indices in row-major (C) order.
///
/// Extensions are supported via the generic parameter `E`, see [`NdIterExtension`].
/// The iterator notifies the extension on each index change, allowing extensions to track derived
/// state (e.g. a pointer into a strided buffer) without recomputing it from scratch.
#[derive(Clone)]
pub(crate) struct NdIter<Ix, E> {
    begin: DimArray<Ix>,
    end: DimArray<Ix>,
    current_idx: DimArray<Ix>,
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

impl<Ix, E> NdIter<Ix, E>
where
    Ix: Idx,
    E: NdIterExtension<Ix>,
{
    /// Creates an iterator over `[0, shape)` in every dimension.
    #[inline(always)]
    pub(crate) fn new(shape: &[Ix], extensions: E) -> Self {
        let begin = dim_arr(shape.len(), |_| Ix::ZERO);
        Self::new_with_begin(&begin, shape, extensions)
    }

    /// Creates an iterator over `[begin, end)` in every dimension.
    #[inline(always)]
    pub(crate) fn new_with_begin(begin: &[Ix], end: &[Ix], extensions: E) -> Self {
        let begin = DimArray::from_slice(begin).unwrap();
        let end = DimArray::from_slice(end).unwrap();
        let ndim = begin.len();
        assert_eq!(begin.len(), ndim);
        assert_eq!(end.len(), ndim);
        extensions.assert_ndim(ndim);
        assert!(begin.iter().zip(end.iter()).all(|(&b, &e)| b <= e));
        let current_idx = begin.clone();

        let nitems = begin
            .iter()
            .zip(&end)
            .map(|(&b, &e)| {
                let n: usize = (e - b).try_into().unwrap();
                n as u64
            })
            .product::<u64>();
        let status = IterStatus::new(nitems);

        Self {
            end,
            begin,
            current_idx,
            status,
            extensions,
        }
    }

    /// Advances to the next index in row-major order and returns `(current_index, extension_item)`.
    ///
    /// On each step the rightmost dimension that has not yet reached its bound is incremented,
    /// and all dimensions to its right are reset to `begin`.
    #[inline(always)]
    pub(crate) fn next(&mut self) -> Option<(&[Ix], E::Item<'_>)> {
        if self.status.is_exhausted() {
            return None;
        }

        if self.status.is_not_started() {
            self.status.start();
            return Some(self.get_current_and_advance_status());
        }

        debug_assert!(self.status.is_in_progress());
        let shape = self.end.as_ref();
        let ndim = shape.len();
        for dim in (0..ndim).rev() {
            let advanced_idx = self.current_idx[dim] + Ix::ONE;
            if advanced_idx < shape[dim] {
                self.extensions
                    .on_increase(dim, self.current_idx[dim], advanced_idx, Ix::ONE);
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
    pub(crate) fn get_current_and_advance_status(&mut self) -> (&[Ix], E::Item<'_>) {
        self.status.advance();
        (&self.current_idx, self.extensions.next())
    }

    #[inline(always)]
    pub(crate) fn len(&self) -> u64 {
        self.status.len()
    }

    #[inline(always)]
    pub(crate) fn map<T>(
        self,
        f: impl FnMut((&[Ix], E::Item<'_>)) -> T + Clone,
    ) -> impl Iterator<Item = T> + Clone
    where
        Self: Clone,
        E: Clone,
    {
        #[derive(Clone)]
        struct Iter<Ix, E, F> {
            iter: NdIter<Ix, E>,
            f: F,
        }
        impl<Ix, E, F, T> Iterator for Iter<Ix, E, F>
        where
            Ix: Idx,
            E: NdIterExtension<Ix> + Clone,
            F: FnMut((&[Ix], E::Item<'_>)) -> T + Clone,
        {
            type Item = T;
            #[inline(always)]
            fn next(&mut self) -> Option<Self::Item> {
                self.iter.next().map(|step| (self.f)(step))
            }

            #[inline(always)]
            fn size_hint(&self) -> (usize, Option<usize>) {
                let len = self.iter.status.len() as usize;
                (len, Some(len))
            }
        }
        Iter { iter: self, f }
    }
}

/// An extension trait for [`NdIter`] that tracks derived state alongside the current index.
///
/// Instead of recomputing derived state (e.g. a raw pointer offset) from scratch at every step,
/// implementors receive incremental [`on_increase`](NdIterExtension::on_increase) and
/// [`on_decrease`](NdIterExtension::on_decrease) notifications and
/// return the current derived value via [`next`](NdIterExtension::next).
pub(crate) trait NdIterExtension<Ix> {
    /// The derived value produced at each iteration step.
    type Item<'a>
    where
        Self: 'a;

    /// Called when dimension `dim` changes from `before` to `after`.
    ///
    /// All dimension changes for a single step are delivered before [`next`](NdIterExtension::next)
    /// is called.
    fn on_increase(&mut self, dim: usize, before: Ix, after: Ix, diff: Ix);
    fn on_decrease(&mut self, dim: usize, before: Ix, after: Ix, diff: Ix);

    /// Returns the current derived value after all index changes have been applied.
    fn next<'a>(&'a self) -> Self::Item<'a>;

    fn assert_ndim(&self, ndim: usize);
}

/// A plain index-only iterator; a thin wrapper around [`NdIter`] with a `()` extension.
#[allow(unused)]
pub(crate) struct IdxIter<Ix>(NdIter<Ix, ()>);

#[allow(unused)]
impl<Ix> IdxIter<Ix>
where
    Ix: Idx,
{
    #[inline(always)]
    pub(crate) fn new(shape: &[Ix]) -> Self {
        Self(NdIter::new(shape, ()))
    }

    /// Returns the next multi-dimensional index, or `None` when exhausted.
    #[inline(always)]
    pub(crate) fn next(&mut self) -> Option<&[Ix]> {
        Some(self.0.next()?.0)
    }
}

// ---------------------------------------------------------------------------
// Tuple blanket impls - compose multiple extensions so that a single NdIter
// can maintain several pieces of derived state simultaneously.
// ---------------------------------------------------------------------------

impl<Ix> NdIterExtension<Ix> for () {
    type Item<'a> = ();
    #[inline(always)]
    fn on_increase(&mut self, _dim: usize, _before: Ix, _after: Ix, _diff: Ix) {}
    #[inline(always)]
    fn on_decrease(&mut self, _dim: usize, _before: Ix, _after: Ix, _diff: Ix) {}
    #[inline(always)]
    fn next(&self) {}
    #[inline(always)]
    fn assert_ndim(&self, _ndim: usize) {}
}
impl<Ix, T1> NdIterExtension<Ix> for (T1,)
where
    T1: NdIterExtension<Ix>,
{
    type Item<'a>
        = (T1::Item<'a>,)
    where
        T1: 'a;
    #[inline(always)]
    fn on_increase(&mut self, dim: usize, before: Ix, after: Ix, diff: Ix) {
        self.0.on_increase(dim, before, after, diff);
    }
    #[inline(always)]
    fn on_decrease(&mut self, dim: usize, before: Ix, after: Ix, diff: Ix) {
        self.0.on_decrease(dim, before, after, diff);
    }
    #[inline(always)]
    fn next<'a>(&'a self) -> (T1::Item<'a>,) {
        (self.0.next(),)
    }
    #[inline(always)]
    fn assert_ndim(&self, ndim: usize) {
        self.0.assert_ndim(ndim);
    }
}
impl<Ix, T1, T2> NdIterExtension<Ix> for (T1, T2)
where
    Ix: Idx,
    T1: NdIterExtension<Ix>,
    T2: NdIterExtension<Ix>,
{
    type Item<'a>
        = (T1::Item<'a>, T2::Item<'a>)
    where
        T1: 'a,
        T2: 'a;
    #[inline(always)]
    fn on_increase(&mut self, dim: usize, before: Ix, after: Ix, diff: Ix) {
        self.0.on_increase(dim, before, after, diff);
        self.1.on_increase(dim, before, after, diff);
    }
    #[inline(always)]
    fn on_decrease(&mut self, dim: usize, before: Ix, after: Ix, diff: Ix) {
        self.0.on_decrease(dim, before, after, diff);
        self.1.on_decrease(dim, before, after, diff);
    }
    #[inline(always)]
    fn next<'a>(&'a self) -> (T1::Item<'a>, T2::Item<'a>) {
        (self.0.next(), self.1.next())
    }
    #[inline(always)]
    fn assert_ndim(&self, ndim: usize) {
        self.0.assert_ndim(ndim);
        self.1.assert_ndim(ndim);
    }
}
impl<Ix, T1, T2, T3> NdIterExtension<Ix> for (T1, T2, T3)
where
    Ix: Idx,
    T1: NdIterExtension<Ix>,
    T2: NdIterExtension<Ix>,
    T3: NdIterExtension<Ix>,
{
    type Item<'a>
        = (T1::Item<'a>, T2::Item<'a>, T3::Item<'a>)
    where
        T1: 'a,
        T2: 'a,
        T3: 'a;
    #[inline(always)]
    fn on_increase(&mut self, dim: usize, before: Ix, after: Ix, diff: Ix) {
        self.0.on_increase(dim, before, after, diff);
        self.1.on_increase(dim, before, after, diff);
        self.2.on_increase(dim, before, after, diff);
    }
    #[inline(always)]
    fn on_decrease(&mut self, dim: usize, before: Ix, after: Ix, diff: Ix) {
        self.0.on_decrease(dim, before, after, diff);
        self.1.on_decrease(dim, before, after, diff);
        self.2.on_decrease(dim, before, after, diff);
    }
    #[inline(always)]
    fn next<'a>(&'a self) -> (T1::Item<'a>, T2::Item<'a>, T3::Item<'a>) {
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

    // ---------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------

    fn collect_idx<Ix: Idx>(mut iter: IdxIter<Ix>) -> Vec<Vec<Ix>> {
        let mut out = Vec::new();
        while let Some(idx) = iter.next() {
            out.push(idx.to_vec());
        }
        out
    }

    /// Records every `on_increase/decrease` notification it receives.
    struct ChangeLog<Ix> {
        log: Vec<(usize, Ix, Ix)>,
    }
    impl<Ix: Copy> ChangeLog<Ix> {
        fn new() -> Self {
            Self { log: Vec::new() }
        }
    }
    impl<Ix: Copy> NdIterExtension<Ix> for ChangeLog<Ix> {
        type Item<'a>
            = usize
        where
            Ix: 'a; // number of on_increase/decrease calls so far when next() is called
        fn on_increase(&mut self, dim: usize, before: Ix, after: Ix, _diff: Ix) {
            self.log.push((dim, before, after));
        }
        fn on_decrease(&mut self, dim: usize, before: Ix, after: Ix, _diff: Ix) {
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
        assert_eq!(collect_idx(IdxIter::<usize>::new(&[])), vec![vec![]]);
    }

    #[test]
    fn idx_iter_1d() {
        assert_eq!(
            collect_idx(IdxIter::new(&[4usize])),
            vec![vec![0], vec![1], vec![2], vec![3]],
        );
    }

    #[test]
    fn idx_iter_1d_size_1() {
        assert_eq!(collect_idx(IdxIter::new(&[1usize])), vec![vec![0]]);
    }

    #[test]
    fn idx_iter_1d_size_0() {
        assert!(collect_idx(IdxIter::new(&[0usize])).is_empty());
    }

    #[test]
    fn idx_iter_2d_row_major_order() {
        assert_eq!(
            collect_idx(IdxIter::new(&[2usize, 3])),
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
        let got = collect_idx(IdxIter::new(&[2usize, 3, 2]));
        let expected: Vec<Vec<usize>> = (0..2)
            .flat_map(|i| (0..3).flat_map(move |j| (0..2).map(move |k| vec![i, j, k])))
            .collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn idx_iter_total_count_equals_shape_product() {
        let shape = [2usize, 3, 4, 5];
        assert_eq!(collect_idx(IdxIter::new(&shape)).len(), 2 * 3 * 4 * 5);
    }

    #[test]
    fn idx_iter_u32_index_type() {
        assert_eq!(
            collect_idx(IdxIter::new(&[3u32])),
            vec![vec![0u32], vec![1], vec![2]],
        );
    }

    #[test]
    fn idx_iter_u64_index_type() {
        assert_eq!(
            collect_idx(IdxIter::new(&[3u64])),
            vec![vec![0u64], vec![1], vec![2]],
        );
    }

    // ---------------------------------------------------------------------------
    // IdxIter - empty / exhaustion
    // ---------------------------------------------------------------------------

    #[test]
    fn idx_iter_zero_in_first_dim_is_empty() {
        assert!(collect_idx(IdxIter::new(&[0usize, 3])).is_empty());
    }

    #[test]
    fn idx_iter_zero_in_last_dim_is_empty() {
        assert!(collect_idx(IdxIter::new(&[3usize, 0])).is_empty());
    }

    #[test]
    fn idx_iter_zero_in_middle_dim_is_empty() {
        assert!(collect_idx(IdxIter::new(&[3usize, 0, 4])).is_empty());
    }

    #[test]
    fn idx_iter_zero_1d_is_empty() {
        assert!(collect_idx(IdxIter::new(&[0usize])).is_empty());
    }

    #[test]
    fn idx_iter_returns_none_repeatedly_after_exhaustion() {
        let mut iter = IdxIter::new(&[2usize]);
        iter.next();
        iter.next();
        assert!(iter.next().is_none());
        assert!(iter.next().is_none());
        assert!(iter.next().is_none());
    }

    // ---------------------------------------------------------------------------
    // NdIter with new_with_begin
    // ---------------------------------------------------------------------------

    fn collect_indices<E: NdIterExtension<usize>>(mut iter: NdIter<usize, E>) -> Vec<Vec<usize>> {
        let mut out = Vec::new();
        while let Some((idx, _)) = iter.next() {
            out.push(idx.to_vec());
        }
        out
    }

    #[test]
    fn new_with_begin_1d_offset() {
        assert_eq!(
            collect_indices(NdIter::new_with_begin(&[2usize], &[5], ())),
            vec![vec![2], vec![3], vec![4]],
        );
    }

    #[test]
    fn new_with_begin_2d_offset() {
        let got = collect_indices(NdIter::new_with_begin(&[1usize, 2], &[3, 4], ()));
        let expected: Vec<Vec<usize>> = (1..3)
            .flat_map(|r| (2..4).map(move |c| vec![r, c]))
            .collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn new_with_begin_count_matches_range_product() {
        let got = collect_indices(NdIter::new_with_begin(&[1usize, 2, 0], &[4, 5, 3], ()));
        assert_eq!(got.len(), (4 - 1) * (5 - 2) * (3 - 0));
    }

    #[test]
    fn new_with_begin_empty_when_one_dim_degenerate() {
        // begin[0] == end[0] -> no elements even though other dims are non-empty
        assert!(collect_indices(NdIter::new_with_begin(&[2usize, 0], &[2, 5], ())).is_empty());
    }

    #[test]
    fn new_with_begin_empty_when_all_dims_degenerate() {
        assert!(collect_indices(NdIter::new_with_begin(&[3usize], &[3], ())).is_empty());
    }

    #[test]
    fn new_with_begin_begin_equals_zero_matches_new() {
        let via_new = collect_indices(NdIter::new(&[3usize, 4], ()));
        let via_begin = collect_indices(NdIter::new_with_begin(&[0usize, 0], &[3, 4], ()));
        assert_eq!(via_new, via_begin);
    }

    // ---------------------------------------------------------------------------
    // on_increase / on_decrease notifications
    // ---------------------------------------------------------------------------

    #[test]
    fn on_change_not_called_on_first_step() {
        let mut iter = NdIter::new(&[3usize, 4], ChangeLog::new());
        let (_, n_changes) = iter.next().unwrap();
        assert_eq!(n_changes, 0, "no changes on the very first step");
    }

    #[test]
    fn on_change_called_once_for_innermost_advance() {
        // [0,0] -> [0,1]: only dim 1 changes
        let mut iter = NdIter::new(&[3usize, 4], ChangeLog::new());
        iter.next(); // emit [0,0] - no changes
        let before = iter.extensions.log.len();
        iter.next(); // emit [0,1]
        let new: Vec<_> = iter.extensions.log[before..].to_vec();
        assert_eq!(new, vec![(1usize, 0usize, 1usize)]);
    }

    #[test]
    fn on_change_called_twice_on_row_wrap() {
        // At [0,3] -> [1,0]: dim 0 increments then dim 1 resets.
        let mut iter = NdIter::new(&[3usize, 4], ChangeLog::new());
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
        let mut iter = NdIter::new(&[2usize, 3, 4], ChangeLog::new());
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
        let mut iter = NdIter::new_with_begin(&[1usize, 2], &[3, 5], ChangeLog::new());
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
        let (r, c) = (4usize, 5usize);
        let mut iter = NdIter::new(&[r, c], ChangeLog::new());
        while iter.next().is_some() {}
        let expected = (r - 1) * 2 + r * (c - 1);
        assert_eq!(iter.extensions.log.len(), expected);
    }

    // ---------------------------------------------------------------------------
    // Tuple extensions
    // ---------------------------------------------------------------------------

    #[test]
    fn tuple_1_extension_delegates() {
        let ext = ChangeLog::<usize>::new();
        let mut iter: NdIter<usize, (ChangeLog<usize>,)> = NdIter::new(&[3usize, 3], (ext,));
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
            NdIterExtStridesPtrMut::new(&[3usize, 1], base_a),
            NdIterExtStridesPtrMut::new(&[6usize, 2], base_b),
        );
        let mut iter: NdIter<usize, _> = NdIter::new(&[2usize, 3], ext);
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
            NdIterExtStridesPtrMut::new(&[1usize], base),
            ChangeLog::<usize>::new(),
        );
        let mut iter: NdIter<usize, _> = NdIter::new(&[4usize], ext);
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
            NdIterExtStridesPtrMut::new(&[1usize], a.as_mut_ptr()),
            NdIterExtStridesPtrMut::new(&[1usize], b.as_mut_ptr()),
            NdIterExtStridesPtrMut::new(&[1usize], c.as_mut_ptr()),
        );
        let mut iter: NdIter<usize, _> = NdIter::new(&[4usize], ext);
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
        let mut iter = NdIter::new(&[2usize, 2], ());
        while let Some((_, item)) = iter.next() {
            let _: () = item;
        }
    }
}
