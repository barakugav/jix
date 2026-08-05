use crate::dtype::Dtyped;
use crate::util::iter::NdIter;
use crate::util::PtrExt;
use crate::{dim_arr, Dim, DimArray, DimDyn, Dimension, PtrMutExt};

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
    #[inline(never)]
    pub(crate) fn new(
        shape: &[usize],
        strides: [&[usize]; N_OPERANDS],
        layouts: [(usize, usize); N_OPERANDS], // (size, alignment) per operand, in its stride unit
    ) -> Self {
        for s in strides {
            assert_eq!(s.len(), shape.len());
        }

        // (1) Order the axes (per-operand strides non-increasing, size-1 axes dropped). The sort key
        // is the array of an axis's per-operand strides, compared lexicographically (operand 0
        // first), so `[usize; N_OPERANDS]`'s derived `Ord` gives exactly the ranking we want.
        let key = |d: usize| -> [usize; N_OPERANDS] { std::array::from_fn(|i| strides[i][d]) };
        let mut dim_perm = DimArray::new();
        for (d, &len) in shape.iter().enumerate() {
            if len == 0 {
                return Self::empty(layouts); // nothing to iterate
            }
            if len > 1 {
                dim_perm.push(d);
            }
        }
        let shape_storage;
        let strides_storage: [_; N_OPERANDS];
        let need_sort = dim_perm.windows(2).any(|w| key(w[0]) < key(w[1]));
        let (shape, strides) = if dim_perm.len() != shape.len() || need_sort {
            if need_sort {
                dim_perm.sort_by_key(|&d| std::cmp::Reverse(key(d)));
            }
            let apply_dim_permutation =
                |arr: &[usize]| dim_arr(dim_perm.len(), |d| arr[dim_perm[d]]);
            shape_storage = apply_dim_permutation(shape);
            strides_storage = std::array::from_fn(|i| apply_dim_permutation(strides[i]));
            (
                shape_storage.as_slice(),
                strides_storage.each_ref().map(|s| s.as_slice()),
            )
        } else {
            (shape, strides)
        };

        // (2) Coalesce adjacent contiguous axes into groups. After the permutation the axes run
        // outermost (index 0) -> innermost, so a group spanning post-permutation axes [lo..=hi]
        // takes its stride from the innermost axis `hi` and its length from the product of the
        // group's shapes.
        let mut group_inner = DimArray::new(); // post-perm index of each group's inner axis
        let mut group_len = DimArray::new(); // product of the group's shapes
        #[allow(clippy::needless_range_loop)]
        for d in 0..shape.len() {
            let m = group_inner.len();
            if m > 0
                && (0..N_OPERANDS)
                    .all(|i| strides[i][group_inner[m - 1]] == strides[i][d] * shape[d])
            {
                group_inner[m - 1] = d; // the group now reaches down to axis `d`
                group_len[m - 1] *= shape[d];
            } else {
                group_inner.push(d);
                group_len.push(shape[d]);
            }
        }
        let mut strides: [DimArray<usize>; N_OPERANDS] =
            std::array::from_fn(|i| dim_arr(group_inner.len(), |g| strides[i][group_inner[g]]));
        let mut shape = group_len;

        // (3) Compute the innermost-run flags (length, and per-operand alignment / contiguity).
        let sizes: [_; N_OPERANDS] = std::array::from_fn(|i| layouts[i].0);
        let is_aligned;
        let is_contiguous;
        if shape.is_empty() {
            // The whole region is a single element. Treat as 1-d with length 1.
            shape.push(1);
            for i in 0..N_OPERANDS {
                strides[i].push(sizes[i]);
            }
            is_aligned = [true; N_OPERANDS];
            is_contiguous = [true; N_OPERANDS];
        } else {
            let ndim = shape.len();
            is_aligned = std::array::from_fn::<_, N_OPERANDS, _>(|i| {
                let alignment = layouts[i].1;
                strides[i].iter().all(|s| s.is_multiple_of(alignment))
            });
            is_contiguous =
                std::array::from_fn::<_, N_OPERANDS, _>(|i| strides[i][ndim - 1] == sizes[i]);
        }

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
        let strides = std::array::from_fn(|op_i| {
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

    /// Drive `inner_loop` once per outer position, as `inner_loop(offsets, inner_len, inner_strides)`
    #[inline]
    pub(crate) fn foreach_inner_1d(
        &self,
        mut inner_loop: impl FnMut([usize; N_OPERANDS], usize, [usize; N_OPERANDS]),
    ) {
        let ndim = self.shape.len();
        let inner_len = self.shape[ndim - 1];
        let inner_strides: [_; N_OPERANDS] = std::array::from_fn(|i| self.strides[i][ndim - 1]);
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
fn nd_iter_unordered_nd_walk<const N_OPERANDS: usize, D: Dimension>(
    shape: &[usize],
    strides: [&[usize]; N_OPERANDS],
    mut inner_loop: impl FnMut([usize; N_OPERANDS], usize, [usize; N_OPERANDS]),
) {
    let ndim = shape.len();
    let inner_len = shape[ndim - 1];
    let inner_strides = std::array::from_fn(|i| strides[i][ndim - 1]);

    if D::NDIM == Some(1) {
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

        let outer_shape = D::vec(ndim - 1, |k| shape[k] as u64);
        let outer_strides: [_; N_OPERANDS] =
            std::array::from_fn(|i| D::vec(ndim - 1, |k| strides[i][k]));
        let iter = NdIter::builder(outer_shape)
            .with_strides_offset_multi_ext(outer_strides, [0usize; N_OPERANDS])
            .build();
        for (_, offsets) in iter {
            inner_loop(offsets, inner_len, inner_strides);
        }
    }
}

#[allow(unused)] // TODO
pub(crate) fn nd_iter_unordered_op0<T, U>(
    shape: &[usize],
    data_ptr: *mut T,
    strides: &[usize],
    mut kernel: impl FnMut(T) -> U,
) where
    T: Dtyped,
    U: Dtyped,
{
    let (itemsize, alignment) = (size_of::<T>(), align_of::<T>());
    assert!(itemsize == size_of::<U>() && alignment <= align_of::<U>());

    let iter = NdIterUnordered::new(shape, [strides], [(itemsize, alignment)]);
    let aligned = iter.is_aligned()[0] && data_ptr.is_aligned();
    let inner_loop_fn = match (aligned, iter.is_contiguous()[0]) {
        (true, true) => inner_loop::<T, U, 1, true, true>,
        (true, false) => inner_loop::<T, U, 1, true, false>,
        (false, true) => inner_loop::<T, U, 1, false, true>,
        (false, false) => inner_loop::<T, U, 1, false, false>,
    };
    iter.foreach_inner_1d(|[offset], inner_len, [inner_stride]| {
        let data_ptr = unsafe { data_ptr.cast::<u8>().add(offset).cast::<T>() };
        inner_loop_fn(data_ptr, inner_len, inner_stride, &mut kernel)
    });

    #[inline(never)]
    fn inner_loop<T, U, const LANES: usize, const ALIGNED: bool, const CONTIGUOUS: bool>(
        data_ptr: *mut T,
        len: usize,
        inner_stride: usize,
        kernel: &mut impl FnMut(T) -> U,
    ) where
        T: Dtyped,
        U: Dtyped,
    {
        assert_eq!(size_of::<T>(), size_of::<U>());
        let data_ptr = data_ptr.cast::<T>();
        if CONTIGUOUS {
            debug_assert_eq!(inner_stride, size_of::<T>());
        }
        let write = |j: usize, val: U| {
            let dst = data_ptr.cast::<U>();
            let elm = if CONTIGUOUS {
                unsafe { dst.add(j) }
            } else {
                unsafe { dst.cast::<u8>().add(j * inner_stride).cast::<U>() }
            };
            unsafe { elm.write_maybe_aligned::<ALIGNED>(val) };
        };
        let mut i = 0;
        if CONTIGUOUS {
            while i + LANES <= len {
                let chunk = unsafe {
                    data_ptr
                        .add(i)
                        .cast::<[T; LANES]>()
                        .read_maybe_aligned::<ALIGNED>()
                };
                #[allow(clippy::needless_range_loop)]
                for k in 0..LANES {
                    write(i + k, kernel(chunk[k]));
                }
                i += LANES;
            }

            while i < len {
                let val = unsafe { data_ptr.add(i).read_maybe_aligned::<ALIGNED>() };
                write(i, kernel(val));
                i += 1;
            }
        } else {
            while i < len {
                let val = unsafe {
                    data_ptr
                        .cast::<u8>()
                        .add(i * inner_stride)
                        .cast::<T>()
                        .read_maybe_aligned::<ALIGNED>()
                };
                write(i, kernel(val));
                i += 1;
            }
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
        let mut visited: Vec<[usize; N]> = Vec::new();
        let mut inner_calls = 0usize;

        let iter = NdIterUnordered::new(shape, strides, layouts);
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
    // nd_iter_unordered_op0 - in-place strided map over operand 0
    // ---------------------------------------------------------------------------

    use crate::dtype::Dtyped;

    // Deterministic kernels used both to drive the function and to build the reference. Fn pointers
    // are `Copy`, so the same value serves both roles.
    fn kmul_u32(x: u32) -> u32 {
        x.wrapping_mul(0x9E37_79B1).wrapping_add(1)
    }
    fn kmul_u16(x: u16) -> u16 {
        x.wrapping_mul(3).wrapping_add(7)
    }
    fn k_i32_to_u32(x: i32) -> u32 {
        (x as u32) ^ 0xABCD_1234
    }

    // Bytes needed to hold every element of `shape` at the given byte strides (0 for empty regions).
    fn buf_len(shape: &[usize], strides: &[usize], itemsize: usize) -> usize {
        if shape.contains(&0) {
            return 0;
        }
        let max_off: usize = shape
            .iter()
            .zip(strides)
            .map(|(&s, &st)| (s - 1) * st)
            .sum();
        max_off + itemsize
    }

    /// Run `nd_iter_unordered_op0` as an in-place map and assert it byte-matches a naive reference
    /// that maps every in-region element and leaves the gaps (bytes no element covers) untouched.
    ///
    /// `strides` are byte strides and MUST give every element a distinct offset (no broadcast / no
    /// overlap): an in-place map over overlapping elements is order-dependent, and the walk order is
    /// deliberately unspecified. Returns the `(aligned, contiguous)` flags the walk reported (i.e.
    /// which of the four inner-loop specializations ran), or `None` for an empty region.
    #[track_caller]
    fn check_op0<T, U>(
        shape: &[usize],
        strides: &[usize],
        kernel: fn(T) -> U,
    ) -> Option<(bool, bool)>
    where
        T: Dtyped,
        U: Dtyped,
    {
        let itemsize = size_of::<T>();
        assert_eq!(itemsize, size_of::<U>());

        // Observe which specialization this layout selects, so callers can pin it down. An empty
        // region visits nothing; report `None` for it.
        let r = run(shape, [strides], [(itemsize, align_of::<T>())]);
        let observed = (!r.visited.is_empty()).then(|| {
            let f = r.flags.unwrap();
            (f.is_aligned[0], f.is_contiguous[0])
        });

        let len = buf_len(shape, strides, itemsize);
        let initial: Vec<u8> = (0..len)
            .map(|i| (i as u8).wrapping_mul(37).wrapping_add(11))
            .collect();

        // Reference: map each in-region element in place, reading from the pristine initial bytes
        // (offsets are distinct, so no element is read after another has overwritten it).
        let mut expected = initial.clone();
        for [off] in reference::<1>(shape, [strides]) {
            let t = unsafe { initial.as_ptr().add(off).cast::<T>().read_unaligned() };
            unsafe {
                expected
                    .as_mut_ptr()
                    .add(off)
                    .cast::<U>()
                    .write_unaligned(kernel(t))
            };
        }

        // Actual: seed an 8-byte-aligned buffer with the same bytes (so the aligned inner-loop path
        // is sound whenever the walk selects it), then run the in-place map over it.
        let mut backing = vec![0u64; len.div_ceil(8) + 1];
        let base = backing.as_mut_ptr().cast::<u8>();
        unsafe { base.copy_from_nonoverlapping(initial.as_ptr(), len) };
        nd_iter_unordered_op0::<T, U>(shape, base.cast::<T>(), strides, kernel);
        let actual = unsafe { std::slice::from_raw_parts(base, len) };
        assert_eq!(
            actual,
            expected.as_slice(),
            "shape={shape:?} strides={strides:?}"
        );

        observed
    }

    // The four inner-loop specializations, keyed on (aligned, contiguous). `aligned` reflects every
    // byte stride being a multiple of the element alignment; `contiguous` reflects the innermost run
    // having stride == element size. Each case is constructed to select exactly one specialization.

    #[test]
    fn op0_aligned_contiguous() {
        // Fully contiguous u32 run: strides multiple of align 4, inner stride == size 4.
        assert_eq!(
            check_op0::<u32, u32>(&[3, 4], &[16, 4], kmul_u32),
            Some((true, true))
        );
    }

    #[test]
    fn op0_aligned_not_contiguous() {
        // Stride 8 is a multiple of align 4 (aligned) but not equal to size 4 (not contiguous).
        assert_eq!(
            check_op0::<u32, u32>(&[3], &[8], kmul_u32),
            Some((true, false))
        );
    }

    #[test]
    fn op0_not_aligned_contiguous() {
        // u16 (align 2): the inner axis stays contiguous (stride 2 == size 2) while the odd outer
        // stride 7 makes the walk not-aligned - so contiguous chunk reads go through the unaligned
        // intrinsics.
        assert_eq!(
            check_op0::<u16, u16>(&[2, 3], &[7, 2], kmul_u16),
            Some((false, true))
        );
    }

    #[test]
    fn op0_not_aligned_not_contiguous() {
        // u16 with an odd inner stride 3: neither a multiple of align 2 nor equal to size 2.
        assert_eq!(
            check_op0::<u16, u16>(&[3], &[3], kmul_u16),
            Some((false, false))
        );
    }

    #[test]
    fn op0_multidim_coalesces_to_one_run() {
        // C-order [2,3,4] u32 collapses to a single contiguous run of 24 elements.
        assert_eq!(
            check_op0::<u32, u32>(&[2, 3, 4], &[48, 16, 4], kmul_u32),
            Some((true, true))
        );
    }

    #[test]
    fn op0_strided_outer_drives_the_outer_walk() {
        // Outer stride 20 != inner span 12, so the axes stay split and the outer NdIter drives the
        // inner run twice - exercising op0's per-outer-position offset add.
        assert_eq!(
            check_op0::<u32, u32>(&[2, 3], &[20, 4], kmul_u32),
            Some((true, true))
        );
    }

    #[test]
    fn op0_size_one_axes_are_dropped() {
        assert_eq!(
            check_op0::<u32, u32>(&[1, 4, 1], &[400, 4, 200], kmul_u32),
            Some((true, true))
        );
    }

    #[test]
    fn op0_scalar_single_element() {
        assert_eq!(
            check_op0::<u32, u32>(&[], &[], kmul_u32),
            Some((true, true))
        );
        assert_eq!(
            check_op0::<u32, u32>(&[1, 1], &[8, 4], kmul_u32),
            Some((true, true))
        );
    }

    #[test]
    fn op0_empty_region_is_a_noop() {
        // A zero-length axis yields an empty walk: nothing is mapped.
        assert_eq!(check_op0::<u32, u32>(&[0], &[4], kmul_u32), None);
        assert_eq!(
            check_op0::<u32, u32>(&[2, 0, 3], &[24, 8, 4], kmul_u32),
            None
        );
    }

    #[test]
    fn op0_type_changing_kernel() {
        // T and U differ but share size and alignment (i32 -> u32); the reference reads i32 bytes
        // and writes u32 bytes at the same offsets, matching the in-place transform.
        assert_eq!(
            check_op0::<i32, u32>(&[5], &[4], k_i32_to_u32),
            Some((true, true))
        );
    }

    #[test]
    fn prop_op0_matches_reference() {
        // Contiguous-with-gaps u32 layouts over ranks 0..=4: the inner axis is contiguous or strided
        // depending on the multipliers, covering the aligned contiguous/non-contiguous paths, the
        // coalescing merge, and the scalar / outer-walk shapes.
        let strategy = (0usize..=4).prop_flat_map(|ndim| {
            (
                prop::collection::vec(1usize..=4, ndim),
                prop::collection::vec(1usize..=3, ndim),
            )
        });
        runner(0x0F0)
            .run(&strategy, |(shape, mult)| {
                let strides = strided_strides(&shape, &mult, size_of::<u32>());
                check_op0::<u32, u32>(&shape, &strides, kmul_u32);
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn prop_op0_unaligned_outer_stride() {
        // u16 with a contiguous inner axis behind a forced-odd outer stride: every draw is
        // not-aligned-but-contiguous, so this fuzzes the unaligned chunk-read path across shapes.
        // Both extents are >= 2 so neither axis is dropped as size-1 (which would revive alignment).
        runner(0xF00D)
            .run(&(2usize..=4, 2usize..=4, 0usize..=3), |(a, b, k)| {
                let strides = [2 * b + (2 * k + 1), 2]; // outer odd and > inner span 2*b
                let flags = check_op0::<u16, u16>(&[a, b], &strides, kmul_u16).unwrap();
                prop_assert_eq!(flags, (false, true));
                Ok(())
            })
            .unwrap();
    }
}
