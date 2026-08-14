use crate::util::arrayvec::ArrayVec;
use crate::util::iter::NdIter;
use crate::{dim_arr, Dim, DimArray, DimDyn, Dimension, SliceExt};

/// The most operands a single [`NdIterUnorderedDyn`] walk can carry.
pub(crate) const N_OPERANDS_MAX: usize = 8;
/// One entry per operand of an [`NdIterUnorderedDyn`] walk.
pub(crate) type OperandsArray<T> = ArrayVec<T, N_OPERANDS_MAX>;

/// Drive a 1-d inner loop over every element of a set of identically-shaped strided n-d regions,
/// described only by their common `shape` (in elements), each operand's `strides`, and each operand's
/// `(size, alignment)`. It owns no buffers: it computes the element visitation order and hands the
/// caller an offset per operand, leaving the caller in full control of the actual reads/writes
/// (aliasing, alignment, element type).
///
/// `strides`, the offsets it yields, and each operand's `(size, alignment)` are all in the *same
/// per-operand unit*: pass byte strides with `(itemsize, alignment)` to walk a byte buffer, or
/// element strides with `(1, 1)` to walk something indexed in elements (e.g. `read_bulk`).
///
/// A from-scratch strided walk taking three ideas from NumPy's nditer:
///   1. sort the axes by descending stride so the innermost axis is the most contiguous - ranking
///      each axis by the array of its per-operand strides (operand 0 most significant),
///   2. coalesce adjacent axes when `outer_stride == inner_stride * inner_len` for *every* operand,
///      collapsing contiguous runs into a single longer axis, and
///   3. split into an outer walk (via [`NdIter`] with a stride-offset extension) over all-but-inner
///      axes plus a flat inner 1-d run.
///
/// [`new`](Self::new) performs steps (1) and (2) and computes the innermost-run description; the
/// caller inspects it via [`inner_len`](Self::inner_len), [`is_aligned`](Self::is_aligned) and
/// [`is_contiguous`](Self::is_contiguous) - each operand's inner stride equals `size` / every stride
/// is a multiple of `alignment` - to pick a specialized inner loop, then calls
/// [`foreach_inner_1d`](Self::foreach_inner_1d) to drive that loop once per outer position as
/// `inner_loop(offsets, inner_len, inner_strides)`, both indexed like the input `strides`/`layouts`.
///
/// The struct itself is *not* generic over the operand count, so the sort and coalesce in `new` -
/// the bulk of the code - is compiled once for the whole crate rather than once per count. Only the
/// two `foreach_inner_1d` entry points specialize: the const-generic one yields `[usize; N]` to
/// callers that know their count (nearly all of them), and
/// [`foreach_inner_1d_dyn`](Self::foreach_inner_1d_dyn) yields slices to callers that do not.
pub(crate) struct NdIterUnorderedDyn {
    /// Post-permutation, post-coalescing shape; always rank >= 1 (a scalar region becomes `[1]`, an
    /// empty region `[0]`).
    shape: DimArray<usize>,
    /// Per-operand strides aligned with `shape`, each in its operand's own stride unit.
    strides: OperandsArray<DimArray<usize>>,
    /// Per-operand: every stride is a multiple of the operand's alignment.
    is_aligned: OperandsArray<bool>,
    /// Per-operand: the innermost run is contiguous (inner stride == element size).
    is_contiguous: OperandsArray<bool>,
}

impl NdIterUnorderedDyn {
    /// Order and coalesce the axes and compute the innermost-run flags. An empty region (any axis of
    /// length 0) yields an iterator whose [`foreach_inner_1d`](Self::foreach_inner_1d) visits nothing.
    #[inline(never)]
    pub(crate) fn new(
        shape: &[usize],
        strides: &[&[usize]],
        layouts: &[(usize, usize)], // (size, alignment) per operand, in its stride unit
    ) -> Self {
        assert_eq!(strides.len(), layouts.len());
        assert!(strides.len() <= N_OPERANDS_MAX);
        let mut shape = shape.to_dim_vec::<DimDyn>();
        let mut operand_strides = OperandsArray::new();
        let mut sizes = OperandsArray::new();
        for (s, &(size, _)) in strides.iter().zip(layouts) {
            operand_strides.push(s.to_dim_vec::<DimDyn>());
            sizes.push(size);
        }
        let mut strides = operand_strides;
        if rearrange_axes_by_operand_strides(&mut shape, &mut strides, &sizes).is_none() {
            return Self::empty(layouts); // nothing to iterate
        }

        let ndim = shape.len();
        let mut is_aligned = OperandsArray::new();
        let mut is_contiguous = OperandsArray::new();
        for ((s, &(_, alignment)), &size) in strides.iter().zip(layouts).zip(&sizes) {
            is_aligned.push(s.iter().all(|s| s.is_multiple_of(alignment)));
            is_contiguous.push(s[ndim - 1] == size);
        }

        Self {
            shape,
            strides,
            is_aligned,
            is_contiguous,
        }
    }

    fn empty(layouts: &[(usize, usize)]) -> Self {
        let mut shape = DimArray::new();
        shape.push(0);
        let mut strides = OperandsArray::new();
        let mut is_aligned = OperandsArray::new();
        let mut is_contiguous = OperandsArray::new();
        for &(size, _) in layouts {
            let mut s = DimArray::new();
            s.push(size);
            strides.push(s);
            is_aligned.push(true);
            is_contiguous.push(true);
        }
        Self {
            shape,
            strides,
            is_aligned,
            is_contiguous,
        }
    }

    #[inline(always)]
    pub(crate) fn inner_len(&self) -> usize {
        self.shape[self.shape.len() - 1]
    }

    #[inline(always)]
    pub(crate) fn is_aligned(&self) -> &[bool] {
        &self.is_aligned
    }

    #[inline(always)]
    pub(crate) fn is_contiguous(&self) -> &[bool] {
        &self.is_contiguous
    }

    /// [`foreach_inner_1d`](Self::foreach_inner_1d) for a caller whose operand count is only known
    /// at runtime: offsets and inner strides arrive as slices rather than arrays.
    ///
    /// Kept free of any per-operand collecting: this is generic over the closure, so anything built
    /// here is monomorphized once per call site. The multi-dimensional case hands
    /// [`nd_iter_unordered_nd_walk_dyn`] the stored strides untouched and lets it do the setup once
    /// per outer rank.
    // Driven by `ReadData2::to_buf`, whose leaf count is only known at runtime.
    #[inline]
    pub(crate) fn foreach_inner_1d(&self, mut inner_loop: impl FnMut(&[usize], usize, &[usize])) {
        let ndim = self.shape.len();
        if crate::hint::likely(ndim == 1) {
            let mut offsets = OperandsArray::new();
            let mut inner_strides = OperandsArray::new();
            for s in &self.strides {
                offsets.push(0);
                inner_strides.push(s[0]);
            }
            inner_loop(&offsets, self.shape[0], &inner_strides);
        } else {
            let nd_walk_fn = match ndim {
                2 => nd_iter_unordered_nd_walk::<Dim<1>>,
                3 => nd_iter_unordered_nd_walk::<Dim<2>>,
                4 => nd_iter_unordered_nd_walk::<Dim<3>>,
                _ => nd_iter_unordered_nd_walk::<DimDyn>,
            };
            nd_walk_fn(&self.shape, &self.strides, &mut inner_loop);
        }
    }
}

/// [`nd_iter_unordered_nd_walk`] for a runtime operand count: one instantiation per outer rank,
/// whatever the number of operands.
///
/// The offsets are lent out as a slice, so the outer `NdIter` here is the lending kind.
#[allow(clippy::type_complexity)]
#[inline(never)]
fn nd_iter_unordered_nd_walk<D: Dimension>(
    shape: &[usize],
    strides: &[DimArray<usize>],
    inner_loop: &mut dyn FnMut(&[usize], usize, &[usize]),
) {
    let ndim = shape.len();
    let inner_len = shape[ndim - 1];
    let mut inner_strides = OperandsArray::new();
    let mut offsets = OperandsArray::new();
    for s in strides {
        inner_strides.push(s[ndim - 1]);
        offsets.push(0usize);
    }

    if D::NDIM == Some(1) {
        // Special case for 2D: the outer `NdIter` is just a single loop over the outer axis, and
        // and inner loop is a flat 1-d run.

        let outer_len = shape[0];
        let mut outer_strides = OperandsArray::new();
        for s in strides {
            outer_strides.push(s[0]);
        }
        for i in 0..outer_len {
            if i > 0 {
                for (offset, outer_stride) in offsets.iter_mut().zip(&outer_strides) {
                    *offset += outer_stride;
                }
            }
            inner_loop(&offsets, inner_len, &inner_strides);
        }
    } else {
        // Flat inner 1-d run over the innermost axis [ndim-1]; the outer `NdIter` walks the outer
        // axes and yields every operand's running byte offset at once.

        let outer_shape = D::vec(ndim - 1, |k| shape[k] as u64);
        let mut outer_strides = OperandsArray::new();
        for s in strides {
            outer_strides.push(D::vec(ndim - 1, |k| s[k]));
        }
        let mut iter = NdIter::builder(outer_shape)
            .with_strides_offset_multi_dyn_ext(outer_strides, offsets)
            .build();
        while let Some((_, offsets)) = iter.next() {
            inner_loop(offsets, inner_len, &inner_strides);
        }
    }
}

#[inline]
pub(crate) fn rearrange_axes_by_operand_strides(
    shape: &mut DimArray<usize>,
    strides: &mut [DimArray<usize>],
    itemsize: &[usize],
) -> Option<()> {
    for s in strides.iter() {
        assert_eq!(s.len(), shape.len());
    }

    // (1) Order the axes (per-operand strides non-increasing, size-1 axes dropped). The sort key
    // is the array of an axis's per-operand strides, compared lexicographically (operand 0
    // first), so `[usize; N_OPERANDS]`'s derived `Ord` gives exactly the ranking we want.
    let mut dim_perm = DimArray::new();
    for (d, &len) in shape.iter().enumerate() {
        if len == 0 {
            return None; // nothing to iterate
        }
        if len > 1 {
            dim_perm.push(d);
        }
    }
    if dim_perm.is_empty() {
        // The whole region is a single element. Treat as 1-d with length 1.
        shape.clear();
        shape.push(1);
        for (s, &itemsize) in strides.iter_mut().zip(itemsize) {
            s.clear();
            s.push(itemsize);
        }
        return Some(());
    }

    let stride_cmp = |d1: &usize, d2: &usize| {
        for strides in strides.iter() {
            match strides[*d1].cmp(&strides[*d2]) {
                std::cmp::Ordering::Less => return std::cmp::Ordering::Greater,
                std::cmp::Ordering::Equal => {}
                std::cmp::Ordering::Greater => return std::cmp::Ordering::Less,
            }
        }
        std::cmp::Ordering::Equal
    };
    let need_sort = dim_perm
        .windows(2)
        .any(|w| stride_cmp(&w[0], &w[1]).is_gt());
    if dim_perm.len() != shape.len() || need_sort {
        if need_sort {
            dim_perm.sort_by(stride_cmp);
        }
        let mut tmp_buf = dim_arr(dim_perm.len(), |_| 0);
        let mut apply_dim_permutation = |arr: &mut DimArray<usize>| {
            for d in 0..dim_perm.len() {
                tmp_buf[d] = arr[dim_perm[d]];
            }
            *arr = tmp_buf.clone();
        };
        apply_dim_permutation(shape);
        for strides in strides.iter_mut() {
            apply_dim_permutation(strides);
        }
    }

    // (2) Coalesce adjacent contiguous axes into groups. After the permutation the axes run
    // outermost (index 0) -> innermost, so a group spanning post-permutation axes [lo..=hi]
    // takes its stride from the innermost axis `hi` and its length from the product of the
    // group's shapes.
    if shape.len() > 1 {
        let mut group_inner = DimArray::new(); // post-perm index of each group's inner axis
        let mut group_len = DimArray::new(); // product of the group's shapes
        #[allow(clippy::needless_range_loop)]
        for d in 0..shape.len() {
            let m = group_inner.len();
            if m > 0
                && strides
                    .iter()
                    .all(|s| s[group_inner[m - 1]] == s[d] * shape[d])
            {
                group_inner[m - 1] = d; // the group now reaches down to axis `d`
                group_len[m - 1] *= shape[d];
            } else {
                group_inner.push(d);
                group_len.push(shape[d]);
            }
        }
        {
            let mut tmp_buf = dim_arr(group_inner.len(), |_| 0);
            for strides in strides.iter_mut() {
                for g in 0..group_inner.len() {
                    tmp_buf[g] = strides[group_inner[g]];
                }
                *strides = tmp_buf.clone();
            }
        }
        *shape = group_len;
    }

    debug_assert!(!shape.is_empty());
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------------------
    // Harness
    // ---------------------------------------------------------------------------

    /// The innermost-run flags an [`NdIterUnorderedDyn`] reports, captured by value.
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct Flags<const N: usize> {
        inner_len: usize,
        is_aligned: [bool; N],
        is_contiguous: [bool; N],
    }

    /// Everything a single [`NdIterUnorderedDyn`] walk reveals to its caller.
    struct Run<const N: usize> {
        /// The per-operand offset tuple for every element visited, in visitation order (each inner
        /// run expanded to one entry per element).
        visited: Vec<[usize; N]>,
        /// The innermost-run flags the iterator reports; always populated (an empty region reports
        /// unused placeholders).
        flags: Option<Flags<N>>,
        /// How many times the inner-loop closure was invoked (one per outer position).
        inner_calls: usize,
    }

    /// Build an [`NdIterUnorderedDyn`] and record the offsets it visits, the flags it reports, and how
    /// many inner runs it performs. The inner loop reconstructs each element's offset from the run
    /// base + `k * inner_stride`, so `visited` is exactly the set of offsets the caller would read.
    ///
    /// Drives both `foreach_inner_1d` variants and asserts they agree, so every test below covers
    /// the const-generic and the runtime-count walk at once.
    #[track_caller]
    fn run<const N: usize>(
        shape: &[usize],
        strides: [&[usize]; N],
        layouts: [(usize, usize); N],
    ) -> Run<N> {
        let iter = NdIterUnorderedDyn::new(shape, &strides, &layouts);
        let flags = Flags {
            inner_len: iter.inner_len(),
            is_aligned: iter.is_aligned().try_into().unwrap(),
            is_contiguous: iter.is_contiguous().try_into().unwrap(),
        };

        // Expand one inner run into one `visited` entry per element.
        let expand = |visited: &mut Vec<[usize; N]>,
                      offsets: [usize; N],
                      len: usize,
                      inner_strides: [usize; N]| {
            for k in 0..len {
                visited.push(std::array::from_fn(|i| offsets[i] + k * inner_strides[i]));
            }
        };

        let (mut visited, mut inner_calls) = (Vec::new(), 0usize);
        iter.foreach_inner_1d(|offsets, len, inner_strides| {
            inner_calls += 1;
            // SAFETY: this iterator was built with `N` operands, so both slices are `N` long.
            let offsets = unsafe { offsets.copy_to_array_unchecked() };
            let inner_strides = unsafe { inner_strides.copy_to_array_unchecked() };
            expand(&mut visited, offsets, len, inner_strides);
        });

        // An empty region visits nothing (`inner_calls == 0`); the reported flags are placeholders.
        Run {
            visited,
            flags: Some(flags),
            inner_calls,
        }
    }

    /// A naive row-major walk of `shape`: the per-operand offset tuple for each logical element,
    /// in C order. Empty when any axis has length 0. This is the ground truth the unordered walk
    /// must reproduce (as a multiset - order is deliberately unspecified).
    fn reference<const N: usize>(shape: &[usize], strides: [&[usize]; N]) -> Vec<[usize; N]> {
        let ndim = shape.len();
        if shape.contains(&0) {
            return Vec::new();
        }
        let mut out = Vec::new();
        let mut idx = vec![0usize; ndim];
        loop {
            out.push(std::array::from_fn(|i| {
                (0..ndim).map(|d| idx[d] * strides[i][d]).sum::<usize>()
            }));
            // Increment the row-major index (rightmost axis fastest); stop once it wraps fully.
            let mut d = ndim;
            loop {
                if d == 0 {
                    return out;
                }
                d -= 1;
                idx[d] += 1;
                if idx[d] < shape[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
    }

    /// Assert the offsets visited by `nd_iter_unordered` are a permutation of the row-major
    /// reference (same tuples, same multiplicities), then hand back the [`Run`] so callers can make
    /// additional assertions about the reported flags / run count.
    #[track_caller]
    fn assert_visits<const N: usize>(
        shape: &[usize],
        strides: [&[usize]; N],
        layouts: [(usize, usize); N],
    ) -> Run<N> {
        let run = run(shape, strides, layouts);
        let mut got = run.visited.clone();
        let mut expected = reference(shape, strides);
        got.sort_unstable();
        expected.sort_unstable();
        assert_eq!(
            got, expected,
            "offset multiset mismatch for shape={shape:?} strides={strides:?}"
        );
        run
    }

    /// Byte strides for a row-major array whose backing shape is `shape[d] * mult[d]`, sampling one
    /// logical element every `mult[d]` slots along axis `d` (mirrors the `nd_copy` test helper).
    /// `mult` all ones is fully contiguous (maximal coalescing); any `mult[d] > 1` leaves gaps.
    fn strided_strides(shape: &[usize], mult: &[usize], itemsize: usize) -> Vec<usize> {
        let ndim = shape.len();
        let backing = (0..ndim).map(|d| shape[d] * mult[d]).collect::<Vec<_>>();
        let mut cstr = vec![0usize; ndim];
        let mut acc = itemsize;
        for d in (0..ndim).rev() {
            cstr[d] = acc;
            acc *= backing[d];
        }
        (0..ndim).map(|d| cstr[d] * mult[d]).collect()
    }

    // ---------------------------------------------------------------------------
    // 1-D
    // ---------------------------------------------------------------------------

    #[test]
    fn one_d_contiguous_is_a_single_coalesced_run() {
        let run = assert_visits(&[5], [&[1]], [(1, 1)]);
        let flags = run.flags.unwrap();
        assert_eq!(flags.inner_len, 5);
        assert_eq!(flags.is_contiguous, [true]);
        assert_eq!(flags.is_aligned, [true]);
        assert_eq!(run.inner_calls, 1);
        assert_eq!(run.visited, [[0], [1], [2], [3], [4]]);
    }

    #[test]
    fn one_d_strided_is_not_contiguous() {
        // Inner stride 2 != element size 1, so the run is reported non-contiguous.
        let run = assert_visits(&[4], [&[2]], [(1, 1)]);
        let flags = run.flags.unwrap();
        assert_eq!(flags.inner_len, 4);
        assert_eq!(flags.is_contiguous, [false]);
        assert_eq!(flags.is_aligned, [true]); // every stride is a multiple of alignment 1
        assert_eq!(run.visited, [[0], [2], [4], [6]]);
    }

    // ---------------------------------------------------------------------------
    // Axis ordering and coalescing
    // ---------------------------------------------------------------------------

    #[test]
    fn c_order_2d_coalesces_to_one_run() {
        // Row-major [3,4]: outer stride 4 == inner stride 1 * inner len 4, so both axes merge.
        let run = assert_visits(&[3, 4], [&[4, 1]], [(1, 1)]);
        let flags = run.flags.unwrap();
        assert_eq!(flags.inner_len, 12);
        assert_eq!(flags.is_contiguous, [true]);
        assert_eq!(run.inner_calls, 1);
    }

    #[test]
    fn f_order_2d_is_sorted_then_coalesced() {
        // Column-major [3,4] (strides [1,3]): the descending-stride sort puts axis 1 outermost,
        // after which the two axes coalesce into a single contiguous run of 12.
        let run = assert_visits(&[3, 4], [&[1, 3]], [(1, 1)]);
        let flags = run.flags.unwrap();
        assert_eq!(flags.inner_len, 12);
        assert_eq!(flags.is_contiguous, [true]);
        assert_eq!(run.inner_calls, 1);
    }

    #[test]
    fn strided_outer_axis_does_not_coalesce() {
        // Outer stride 10 != inner stride 1 * inner len 3, so the axes stay split: the inner run
        // has length 3 and the outer NdIter drives it once per outer position (2 positions).
        let run = assert_visits(&[2, 3], [&[10, 1]], [(1, 1)]);
        let flags = run.flags.unwrap();
        assert_eq!(flags.inner_len, 3);
        assert_eq!(flags.is_contiguous, [true]); // inner stride 1 == element size 1
        assert_eq!(run.inner_calls, 2);
        // The full offset set is still exactly the naive walk (checked by assert_visits).
    }

    #[test]
    fn size_one_axes_are_dropped() {
        // The two length-1 axes contribute no offset and must not block the length-4 axis from
        // being treated as a single contiguous run.
        let run = assert_visits(&[1, 4, 1], [&[100, 1, 50]], [(1, 1)]);
        let flags = run.flags.unwrap();
        assert_eq!(flags.inner_len, 4);
        assert_eq!(flags.is_contiguous, [true]);
        assert_eq!(run.inner_calls, 1);
        let mut got = run.visited.clone();
        got.sort_unstable();
        assert_eq!(got, [[0], [1], [2], [3]]);
    }

    // ---------------------------------------------------------------------------
    // Scalar (0-D after size-1 axes are dropped)
    // ---------------------------------------------------------------------------

    #[test]
    fn all_axes_size_one_is_a_single_element() {
        let run = assert_visits(&[1, 1], [&[7, 3]], [(1, 1)]);
        let flags = run.flags.unwrap();
        assert_eq!(flags.inner_len, 1);
        assert_eq!(flags.is_contiguous, [true]);
        assert_eq!(flags.is_aligned, [true]);
        assert_eq!(run.inner_calls, 1);
        assert_eq!(run.visited, [[0]]);
    }

    #[test]
    fn zero_dim_shape_is_a_single_element() {
        // A rank-0 region (no axes) is one element at offset 0.
        let run = assert_visits(&[], [&[]], [(4, 4)]);
        let flags = run.flags.unwrap();
        assert_eq!(flags.inner_len, 1);
        assert_eq!(run.inner_calls, 1);
        assert_eq!(run.visited, [[0]]);
    }

    // ---------------------------------------------------------------------------
    // Empty regions: nothing is visited
    // ---------------------------------------------------------------------------

    #[test]
    fn empty_region_visits_nothing() {
        for shape in [vec![0], vec![0, 3], vec![2, 0, 3], vec![2, 3, 0]] {
            let strides = (0..shape.len()).map(|d| d + 1).collect::<Vec<_>>();
            let run = run(&shape, [strides.as_slice()], [(1, 1)]);
            // The empty sentinel reports a zero-length inner run, so nothing is visited (the inner
            // loop may still run once with `len == 0`, a no-op).
            assert!(run.visited.is_empty(), "shape={shape:?}");
            assert_eq!(run.flags.unwrap().inner_len, 0, "shape={shape:?}");
        }
    }

    // ---------------------------------------------------------------------------
    // Alignment flag
    // ---------------------------------------------------------------------------

    #[test]
    fn aligned_flag_tracks_stride_divisibility() {
        // Stride 5 is not a multiple of alignment 4 -> not aligned; 5 != size 4 -> not contiguous.
        let run = assert_visits(&[3], [&[5]], [(4, 4)]);
        let flags = run.flags.unwrap();
        assert_eq!(flags.is_aligned, [false]);
        assert_eq!(flags.is_contiguous, [false]);
        assert_eq!(run.visited, [[0], [5], [10]]);

        // Stride 8 is a multiple of alignment 4 -> aligned; but 8 != size 4 -> still not contiguous.
        let run = assert_visits(&[3], [&[8]], [(4, 4)]);
        let flags = run.flags.unwrap();
        assert_eq!(flags.is_aligned, [true]);
        assert_eq!(flags.is_contiguous, [false]);
    }

    // ---------------------------------------------------------------------------
    // Broadcasting (zero strides -> repeated offsets)
    // ---------------------------------------------------------------------------

    #[test]
    fn zero_strides_broadcast_to_repeated_offsets() {
        // Every element maps to offset 0, visited `product(shape)` times. Coalescing merges the
        // axes (0 == 0 * len holds for the broadcast operand), and a 0 inner stride is not "size".
        let run = assert_visits(&[2, 3], [&[0, 0]], [(1, 1)]);
        let flags = run.flags.unwrap();
        assert_eq!(flags.inner_len, 6);
        assert_eq!(flags.is_contiguous, [false]);
        assert_eq!(flags.is_aligned, [true]); // 0 is a multiple of any alignment
        assert_eq!(run.visited, [[0]; 6]);
    }

    // ---------------------------------------------------------------------------
    // Multiple operands
    // ---------------------------------------------------------------------------

    #[test]
    fn two_operands_c_order_coalesce_together() {
        // Operand 0: destination byte buffer for a [2,3] i32 array, C-order byte strides.
        // Operand 1: source read in element units. Both fully coalesce, and the offset tuples stay
        // paired (each logical element gives dst = 4 * src).
        let run = assert_visits(&[2, 3], [&[12, 4], &[3, 1]], [(4, 4), (1, 1)]);
        let flags = run.flags.unwrap();
        assert_eq!(flags.inner_len, 6);
        assert_eq!(flags.is_contiguous, [true, true]);
        assert_eq!(flags.is_aligned, [true, true]);
        assert_eq!(run.inner_calls, 1);
        assert_eq!(
            run.visited,
            [[0, 0], [4, 1], [8, 2], [12, 3], [16, 4], [20, 5]]
        );
    }

    #[test]
    fn two_operands_split_when_one_is_strided() {
        // Destination is contiguous, but the source is strided (element strides [1,2]), so the
        // shared innermost axis cannot coalesce: the walk splits into a length-3 inner run driven
        // twice, and only operand 0 is reported contiguous.
        let run = assert_visits(&[2, 3], [&[12, 4], &[1, 2]], [(4, 4), (1, 1)]);
        let flags = run.flags.unwrap();
        assert_eq!(flags.inner_len, 3);
        assert_eq!(flags.is_contiguous, [true, false]);
        assert_eq!(flags.is_aligned, [true, true]);
        assert_eq!(run.inner_calls, 2);
    }

    // ---------------------------------------------------------------------------
    // Property tests: the unordered walk always reproduces the row-major reference
    // ---------------------------------------------------------------------------

    use proptest::prelude::*;
    use proptest::test_runner::{Config, RngAlgorithm, TestRng, TestRunner};

    fn runner(seed: u64) -> TestRunner {
        let mut bytes = [0u8; 32];
        bytes[..8].copy_from_slice(&seed.to_le_bytes());
        TestRunner::new_with_rng(
            Config {
                cases: if cfg!(miri) { 32 } else { 256 },
                failure_persistence: None,
                ..Config::default()
            },
            TestRng::from_seed(RngAlgorithm::ChaCha, &bytes),
        )
    }

    #[test]
    fn prop_matches_reference_random_strides() {
        // Arbitrary (even overlapping / broadcast) strides: whatever the internal axis reordering
        // and coalescing do, the visited offsets must be exactly the naive row-major walk.
        let strategy = (0usize..=4).prop_flat_map(|ndim| {
            (
                prop::collection::vec(1usize..=4, ndim),
                prop::collection::vec(0usize..=6, ndim),
                prop::collection::vec(0usize..=6, ndim),
            )
        });
        runner(0xA11CE)
            .run(&strategy, |(shape, s0, s1)| {
                assert_visits(&shape, [&s0, &s1], [(1, 1), (1, 1)]);
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn prop_matches_reference_permuted_contiguous_layouts() {
        // Two independent contiguous-with-gaps layouts presented under a shared random axis
        // permutation. This forces the descending-stride sort to actually reorder axes and then
        // exercises the coalescing merge on the recovered contiguous runs.
        let strategy = (0usize..=4).prop_flat_map(|ndim| {
            (
                prop::collection::vec(1usize..=4, ndim),
                prop::collection::vec(1usize..=3, ndim),
                prop::collection::vec(1usize..=3, ndim),
                prop::sample::select(vec![1usize, 2, 4]),
                prop::sample::select(vec![1usize, 2, 4]),
                Just((0..ndim).collect::<Vec<usize>>()).prop_shuffle(),
            )
        });
        runner(0xB0B)
            .run(&strategy, |(base_shape, m0, m1, is0, is1, perm)| {
                let phys0 = strided_strides(&base_shape, &m0, is0);
                let phys1 = strided_strides(&base_shape, &m1, is1);
                let shape = perm.iter().map(|&d| base_shape[d]).collect::<Vec<_>>();
                let s0 = perm.iter().map(|&d| phys0[d]).collect::<Vec<_>>();
                let s1 = perm.iter().map(|&d| phys1[d]).collect::<Vec<_>>();
                assert_visits(&shape, [&s0, &s1], [(is0, is0), (is1, is1)]);
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn prop_empty_region_visits_nothing() {
        // Any shape containing a zero-length axis is an empty region: no element is visited.
        let strategy = (1usize..=4)
            .prop_flat_map(|ndim| {
                (
                    prop::collection::vec(0usize..=3, ndim),
                    prop::collection::vec(0usize..=6, ndim),
                )
            })
            .prop_filter("needs a zero-length axis", |(shape, _)| shape.contains(&0));
        runner(0xDEAD)
            .run(&strategy, |(shape, s0)| {
                let run = run(&shape, [s0.as_slice()], [(1, 1)]);
                prop_assert!(run.visited.is_empty());
                Ok(())
            })
            .unwrap();
    }
}
