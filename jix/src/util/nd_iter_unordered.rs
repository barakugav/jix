use std::cmp::Ordering;

use crate::util::iter::NdIter;
use crate::{array_from_fn_inline, dim_arr, Dim, DimArray, DimDyn, Dimension};

/// Drive a 1-d inner loop over every element of `N_OPERANDS` identically-shaped strided n-d regions,
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
/// `inner_loop(offsets, inner_len, inner_strides)`, where `offsets`/`inner_strides` are
/// `[_; N_OPERANDS]` indexed like the input `strides`/`layouts`.
pub(crate) struct NdIterUnordered<const N_OPERANDS: usize> {
    /// Post-permutation, post-coalescing shape; always rank >= 1 (a scalar region becomes `[1]`, an
    /// empty region `[0]`).
    shape: DimArray<usize>,
    /// Per-operand strides aligned with `shape`, each in its operand's own stride unit.
    strides: [DimArray<usize>; N_OPERANDS],
    /// Per-operand: every stride is a multiple of the operand's alignment.
    is_aligned: [bool; N_OPERANDS],
    /// Per-operand: the innermost run is contiguous (inner stride == element size).
    is_contiguous: [bool; N_OPERANDS],
}

impl<const N_OPERANDS: usize> NdIterUnordered<N_OPERANDS> {
    /// Order and coalesce the axes and compute the innermost-run flags. An empty region (any axis of
    /// length 0) yields an iterator whose [`foreach_inner_1d`](Self::foreach_inner_1d) visits nothing.
    #[inline]
    pub(crate) fn new(
        shape: &[usize],
        strides: [&[usize]; N_OPERANDS],
        layouts: [(usize, usize); N_OPERANDS], // (size, alignment) per operand, in its stride unit
    ) -> Self {
        Self::new_with(shape, strides, layouts, [true; N_OPERANDS])
    }

    /// Like [`new`](Self::new), with control over which operands take part in the axis ordering.
    ///
    /// An operand whose `affects_dim_order` entry is `false` is skipped by the sort.
    /// This lets the caller add an extra bookkeeping operand along without letting it perturb
    /// the memory-access order.
    #[inline(never)]
    pub(crate) fn new_with(
        shape: &[usize],
        strides: [&[usize]; N_OPERANDS],
        layouts: [(usize, usize); N_OPERANDS], // (size, alignment) per operand, in its stride unit
        affects_dim_order: [bool; N_OPERANDS],
    ) -> Self {
        for s in strides {
            assert_eq!(s.len(), shape.len());
        }

        // (1) Order the axes (per-operand strides non-increasing, size-1 axes dropped). The sort key
        // is the array of an axis's per-operand strides, compared lexicographically (operand 0
        // first), so `[usize; N_OPERANDS]`'s derived `Ord` gives exactly the ranking we want.
        let mut dim_perm = DimArray::new();
        for (d, &len) in shape.iter().enumerate() {
            if len == 0 {
                return Self::empty(layouts); // nothing to iterate
            }
            if len > 1 {
                dim_perm.push(d);
            }
        }
        if dim_perm.is_empty() {
            // The whole region is a single element. Treat as 1-d with length 1.
            return Self {
                shape: DimArray::from_slice(&[1]).unwrap(),
                strides: array_from_fn_inline(|i| DimArray::from_slice(&[layouts[i].0]).unwrap()),
                is_aligned: [true; N_OPERANDS],
                is_contiguous: [true; N_OPERANDS],
            };
        }

        let (shape, strides) = if dim_perm.len() == 1 {
            // Only one axis remains after dropping size-1 axes: no sort or coalesce needed.
            let d = dim_perm[0];
            let shape = DimArray::from_slice(&[shape[d]]).unwrap();
            let strides = array_from_fn_inline(|i| DimArray::from_slice(&[strides[i][d]]).unwrap());
            (shape, strides)
        } else {
            axes_sort_by(&mut dim_perm, |d1: usize, d2: usize| {
                let mut compared = false;
                for (op_i, strides) in strides.iter().enumerate() {
                    if !affects_dim_order[op_i] {
                        continue;
                    }
                    // A zero stride means this operand does not move along the axis at all, so it
                    // has no opinion on where the axis belongs.
                    if strides[d1] == 0 || strides[d2] == 0 {
                        continue;
                    }
                    compared = true;
                    match strides[d1].cmp(&strides[d2]) {
                        Ordering::Less => return Some(Ordering::Greater),
                        Ordering::Equal => {}
                        Ordering::Greater => return Some(Ordering::Less),
                    }
                }
                compared.then_some(Ordering::Equal)
            });

            // (2) Coalesce adjacent contiguous axes into groups. `dim_perm` lists the axes to visit,
            // outermost first, so a group takes its stride from the innermost axis it reaches down to
            // and its length from the product of the group's shapes. Reading the caller's shape and
            // strides through `dim_perm` leaves the permutation implicit: nothing is materialized until
            // the groups are known, and then only once.
            let mut group_inner = DimArray::new(); // input axis of each group's inner axis
            let mut group_len = DimArray::new(); // product of the group's shapes
            for &d in dim_perm.iter() {
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
            let shape = group_len;
            let strides = array_from_fn_inline::<_, N_OPERANDS>(|i| {
                dim_arr(group_inner.len(), |g| strides[i][group_inner[g]])
            });
            (shape, strides)
        };

        // (3) Compute the innermost-run flags (length, and per-operand alignment / contiguity).
        debug_assert!(!shape.is_empty());
        let sizes = array_from_fn_inline::<_, N_OPERANDS>(|i| layouts[i].0);
        let ndim = shape.len();
        let is_aligned = array_from_fn_inline::<_, N_OPERANDS>(|i| {
            let alignment = layouts[i].1;
            strides[i].iter().all(|s| s.is_multiple_of(alignment))
        });
        let is_contiguous =
            array_from_fn_inline::<_, N_OPERANDS>(|i| strides[i][ndim - 1] == sizes[i]);

        Self {
            shape,
            strides,
            is_aligned,
            is_contiguous,
        }
    }

    fn empty(layouts: [(usize, usize); N_OPERANDS]) -> Self {
        let mut shape = DimArray::new();
        shape.push(0);
        let strides = array_from_fn_inline(|op_i| {
            let mut s = DimArray::new();
            s.push(layouts[op_i].0);
            s
        });
        Self {
            shape,
            strides,
            is_aligned: [true; N_OPERANDS],
            is_contiguous: [true; N_OPERANDS],
        }
    }

    #[inline(always)]
    pub(crate) fn inner_len(&self) -> usize {
        self.shape[self.shape.len() - 1]
    }

    #[inline(always)]
    pub(crate) fn is_aligned(&self) -> [bool; N_OPERANDS] {
        self.is_aligned
    }

    #[inline(always)]
    pub(crate) fn is_contiguous(&self) -> [bool; N_OPERANDS] {
        self.is_contiguous
    }

    /// Each operand's stride along the innermost run, in its own stride unit. Constant across outer
    /// positions, so a caller can pick its inner loop once instead of per run.
    #[inline(always)]
    pub(crate) fn inner_strides(&self) -> [usize; N_OPERANDS] {
        let inner = self.shape.len() - 1;
        array_from_fn_inline(|i| self.strides[i][inner])
    }

    /// Drive `inner_loop` once per outer position, as `inner_loop(offsets, inner_len, inner_strides)`
    #[inline]
    pub(crate) fn foreach_inner_1d(
        &self,
        mut inner_loop: impl FnMut([usize; N_OPERANDS], usize, [usize; N_OPERANDS]),
    ) {
        let ndim = self.shape.len();
        let inner_len = self.shape[ndim - 1];
        let inner_strides = array_from_fn_inline::<_, N_OPERANDS>(|i| self.strides[i][ndim - 1]);
        if crate::hint::likely(ndim == 1) {
            inner_loop([0; N_OPERANDS], inner_len, inner_strides);
        } else {
            let nd_walk_fn = match ndim {
                2 => nd_iter_unordered_nd_walk::<N_OPERANDS, Dim<1>>,
                3 => nd_iter_unordered_nd_walk::<N_OPERANDS, Dim<2>>,
                4 => nd_iter_unordered_nd_walk::<N_OPERANDS, Dim<3>>,
                _ => nd_iter_unordered_nd_walk::<N_OPERANDS, DimDyn>,
            };
            let strides = self.strides.each_ref().map(|s| s.as_slice());
            nd_walk_fn(&self.shape, strides, &mut inner_loop);
        }
    }
}
#[inline(never)]
fn nd_iter_unordered_nd_walk<const N_OPERANDS: usize, OuterD: Dimension>(
    shape: &[usize],
    strides: [&[usize]; N_OPERANDS],
    // TODO: accept &dyn FnMut
    mut inner_loop: impl FnMut([usize; N_OPERANDS], usize, [usize; N_OPERANDS]),
) {
    let ndim = shape.len();
    let inner_len = shape[ndim - 1];
    let inner_strides = std::array::from_fn(|i| strides[i][ndim - 1]);

    if OuterD::NDIM == Some(1) {
        // Special case for 2D: the outer `NdIter` is just a single loop over the outer axis, and
        // and inner loop is a flat 1-d run.

        let outer_len = shape[0];
        let outer_strides: [_; N_OPERANDS] = std::array::from_fn(|i| [strides[i][0]]);
        let mut offsets = [0usize; N_OPERANDS];
        for i in 0..outer_len {
            if i > 0 {
                for op_i in 0..N_OPERANDS {
                    offsets[op_i] += outer_strides[op_i][0];
                }
            }
            inner_loop(offsets, inner_len, inner_strides);
        }
    } else {
        // Flat inner 1-d run over the innermost axis [ndim-1]; the outer `NdIter` walks the outer
        // axes and yields all `N_OPERANDS` running byte offsets at once.

        let outer_shape = OuterD::vec(ndim - 1, |k| shape[k] as u64);
        let outer_strides: [_; N_OPERANDS] =
            std::array::from_fn(|i| OuterD::vec(ndim - 1, |k| strides[i][k]));
        let iter = NdIter::builder(outer_shape)
            .with_strides_offset_multi_ext(outer_strides, [0usize; N_OPERANDS])
            .build();
        for (_, offsets) in iter {
            inner_loop(offsets, inner_len, inner_strides);
        }
    }
}

/// Stable insertion sort over axis indices, with a three-valued comparator.
///
/// `compare(d0, d1)` returns `None` when the two axes cannot be ordered relative to each other -
/// no operand had anything to say about the pair.
#[inline]
pub(super) fn axes_sort_by(
    arr: &mut [usize],
    mut compare: impl FnMut(usize, usize) -> Option<Ordering>,
) {
    for i in 1..arr.len() {
        let mut insertion_idx = i;
        for i1 in (0..i).rev() {
            match compare(arr[i], arr[i1]) {
                Some(ord) if ord.is_ge() => break,
                Some(_) => insertion_idx = i1,
                None => {} // ambiguous: transparent, keep scanning outward
            }
        }
        if insertion_idx != i {
            let tmp = arr[i];
            arr.copy_within(insertion_idx..i, insertion_idx + 1);
            arr[insertion_idx] = tmp;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------------------
    // Harness
    // ---------------------------------------------------------------------------

    /// The innermost-run flags an [`NdIterUnordered`] reports, captured by value.
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct Flags<const N: usize> {
        inner_len: usize,
        is_aligned: [bool; N],
        is_contiguous: [bool; N],
    }

    /// Everything a single [`NdIterUnordered`] walk reveals to its caller.
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

    /// Build an [`NdIterUnordered`] and record the offsets it visits, the flags it reports, and how
    /// many inner runs it performs. The inner loop reconstructs each element's offset from the run
    /// base + `k * inner_stride`, so `visited` is exactly the set of offsets the caller would read.
    fn run<const N: usize>(
        shape: &[usize],
        strides: [&[usize]; N],
        layouts: [(usize, usize); N],
    ) -> Run<N> {
        run_with(shape, strides, layouts, [true; N])
    }

    /// [`run`], with control over which operands take part in the axis ordering.
    fn run_with<const N: usize>(
        shape: &[usize],
        strides: [&[usize]; N],
        layouts: [(usize, usize); N],
        affects_dim_order: [bool; N],
    ) -> Run<N> {
        let mut visited: Vec<[usize; N]> = Vec::new();
        let mut inner_calls = 0usize;

        let iter = NdIterUnordered::new_with(shape, strides, layouts, affects_dim_order);
        let flags = Flags {
            inner_len: iter.inner_len(),
            is_aligned: iter.is_aligned(),
            is_contiguous: iter.is_contiguous(),
        };
        iter.foreach_inner_1d(
            |offsets: [usize; N], len: usize, inner_strides: [usize; N]| {
                inner_calls += 1;
                for k in 0..len {
                    visited.push(std::array::from_fn(|i| offsets[i] + k * inner_strides[i]));
                }
            },
        );

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

    // ---------------------------------------------------------------------------
    // axes_sort_by
    // ---------------------------------------------------------------------------

    /// Sort by the axis indices themselves, ascending.
    fn sorted(mut axes: Vec<usize>) -> Vec<usize> {
        axes_sort_by(&mut axes, |a, b| Some(a.cmp(&b)));
        axes
    }

    #[test]
    fn axes_sort_by_orders_ascending() {
        assert_eq!(sorted(vec![]), Vec::<usize>::new());
        assert_eq!(sorted(vec![7]), vec![7]);
        assert_eq!(sorted(vec![1, 2, 3]), vec![1, 2, 3]); // already ordered
        assert_eq!(sorted(vec![3, 2, 1]), vec![1, 2, 3]); // fully reversed
        assert_eq!(sorted(vec![2, 0, 3, 1]), vec![0, 1, 2, 3]);
        assert_eq!(sorted(vec![5, 4, 5, 4]), vec![4, 4, 5, 5]); // duplicates
    }

    #[test]
    fn axes_sort_by_is_stable() {
        // Rank by `d % 3` alone, so the three axes in each rank compare `Equal` to each other and
        // only a stable sort keeps them in the order they came in.
        let mut axes = vec![0, 1, 2, 3, 4, 5, 6, 7, 8];
        axes_sort_by(&mut axes, |a, b| Some((a % 3).cmp(&(b % 3))));
        assert_eq!(axes, vec![0, 3, 6, 1, 4, 7, 2, 5, 8]);
    }

    #[test]
    fn axes_sort_by_compares_elements_not_positions() {
        // The comparator is handed the *elements* - axis indices - never their positions in `arr`.
        // Every element here is >= 10, so a comparator fed positions would hit the panic.
        let rank = |axis: usize| match axis {
            10 => 2usize,
            11 => 0,
            12 => 1,
            other => panic!("comparator got {other}, which is a position, not an element"),
        };
        let mut axes = vec![10, 11, 12];
        axes_sort_by(&mut axes, |a, b| Some(rank(a).cmp(&rank(b))));
        assert_eq!(axes, vec![11, 12, 10]);
    }

    #[test]
    fn axes_sort_by_ranks_axes_by_descending_stride() {
        // How both walks use it: rank an axis by its per-operand strides, operand 0 most
        // significant, with the comparison reversed so the largest stride ends up outermost.
        fn sort_by_strides(strides: &[&[usize]], axes: &mut [usize]) {
            axes_sort_by(axes, |d1, d2| {
                for s in strides {
                    match s[d1].cmp(&s[d2]) {
                        Ordering::Less => return Some(Ordering::Greater),
                        Ordering::Equal => {}
                        Ordering::Greater => return Some(Ordering::Less),
                    }
                }
                Some(Ordering::Equal)
            });
        }

        let mut axes = [0, 1, 2];
        sort_by_strides(&[&[4, 400, 40], &[1, 100, 10]], &mut axes);
        assert_eq!(axes, [1, 2, 0]);

        // Operand 0 has the same stride on both axes, so operand 1 breaks the tie.
        let mut axes = [0, 1];
        sort_by_strides(&[&[8, 8], &[1, 2]], &mut axes);
        assert_eq!(axes, [1, 0]);
    }

    // ---------------------------------------------------------------------------
    // `inner_strides`, `new_with` and the zero-stride sort rule
    // ---------------------------------------------------------------------------

    #[test]
    fn inner_strides_reports_the_innermost_run_stride_per_operand() {
        // Axis 0 is outermost (operand 0's stride is larger), so axis 1 forms the inner run and
        // each operand's inner stride is its own stride on axis 1. Operand 1's strides block the
        // coalesce, so the inner run really is axis 1 and not the whole region.
        let iter = NdIterUnordered::new(&[2, 3], [&[3, 1], &[1, 2]], [(1, 1); 2]);
        assert_eq!(iter.inner_strides(), [1, 2]);
        assert_eq!(iter.inner_len(), 3);
    }

    #[test]
    fn zero_stride_abstains_instead_of_sorting_innermost() {
        // Operand 0 has stride 0 on axis 0. Sorting `0` as the smallest stride would drag axis 0
        // into the inner run and leave both operands non-contiguous there; abstaining hands the
        // decision to operand 1, which puts the larger-stride axis 0 outermost and leaves axis 1 -
        // contiguous for both operands - as the inner run.
        let shape = [4, 5];
        let iter = NdIterUnordered::new(&shape, [&[0, 8], &[40, 8]], [(8, 8); 2]);
        assert_eq!(iter.inner_strides(), [8, 8]);
        assert_eq!(iter.is_contiguous(), [true, true]);
        assert_visits(&shape, [&[0, 8], &[40, 8]], [(8, 8); 2]);
    }

    #[test]
    fn new_with_can_exclude_an_operand_from_the_dim_order() {
        // Operand 0 ties on both axes, so operand 1 decides - it wants axis 1 outermost, leaving
        // axis 0 (its stride 4) as the inner run.
        let strides: [&[usize]; 2] = [&[8, 8], &[4, 100]];
        let included = NdIterUnordered::new(&[4, 5], strides, [(4, 4); 2]);
        assert_eq!(included.inner_strides(), [8, 4]);

        // Excluded from the ordering, operand 1 no longer gets to flip the axes, so the input
        // order stands and axis 1 (its stride 100) becomes the inner run instead.
        let excluded = NdIterUnordered::new_with(&[4, 5], strides, [(4, 4); 2], [true, false]);
        assert_eq!(excluded.inner_strides(), [8, 100]);
    }

    #[test]
    fn new_with_still_coalesces_against_an_excluded_operand() {
        // Operand 0 alone is fully contiguous and would coalesce both axes into one run of 6.
        // Operand 1 is excluded from the *ordering* but must still constrain the *coalesce*: its
        // stride 5 on axis 0 does not equal 1 * 3, so the axes stay separate.
        let iter =
            NdIterUnordered::new_with(&[2, 3], [&[3, 1], &[5, 1]], [(1, 1); 2], [true, false]);
        assert_eq!(iter.inner_len(), 3);
        let run = run_with(&[2, 3], [&[3, 1], &[5, 1]], [(1, 1); 2], [true, false]);
        assert_eq!(run.inner_calls, 2);
    }

    #[test]
    fn axes_sort_by_treats_an_ambiguous_comparison_as_transparent() {
        // Axis 2 cannot be compared against axis 1, but it is decisively outermost of axis 0. An
        // ambiguous neighbour must not stop the scan, or axis 2 never reaches the front.
        let mut axes = vec![0, 1, 2];
        axes_sort_by(&mut axes, |d0, d1| match (d0, d1) {
            (2, 1) => None,
            (2, 0) => Some(Ordering::Less),
            (1, 0) => Some(Ordering::Greater),
            other => panic!("unexpected comparison {other:?}"),
        });
        assert_eq!(axes, vec![2, 0, 1]);
    }
}
