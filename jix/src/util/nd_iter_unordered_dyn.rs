use crate::util::axes_sort_by;
use crate::util::iter::NdIter;
use crate::{dim_arr, Dim, DimArray, DimDyn, Dimension};

/// [`NdIterUnordered`](crate::NdIterUnordered) with the operand count known only at runtime.
///
/// Same walk, same flags, same caller contract - see [`NdIterUnordered`](crate::NdIterUnordered) for
/// all of it. Two differences: the per-operand tables are heap-allocated rather than inline, and
/// [`foreach_inner_1d`](Self::foreach_inner_1d) lends the offsets and inner strides out as slices
/// instead of yielding `[usize; N_OPERANDS]`. In exchange any number of operands is allowed, and the
/// sort and coalesce are compiled once for the whole crate rather than once per count.
pub(crate) struct NdIterUnorderedDyn {
    /// Post-permutation, post-coalescing shape; always rank >= 1.
    shape: DimArray<usize>,
    /// Per-operand strides aligned with `shape`, each in its operand's own stride unit.
    strides: Vec<DimArray<usize>>,
    is_aligned: Vec<bool>,
    is_contiguous: Vec<bool>,
}

impl NdIterUnorderedDyn {
    #[inline(never)]
    pub(crate) fn new(
        shape: &[usize],
        strides: &[&[usize]],
        layouts: &[(usize, usize)], // (size, alignment) per operand, in its stride unit
    ) -> Self {
        assert_eq!(strides.len(), layouts.len());
        for s in strides {
            assert_eq!(s.len(), shape.len());
        }

        // (1) Order the axes (per-operand strides non-increasing, size-1 axes dropped). The sort key
        // is the array of an axis's per-operand strides, compared lexicographically (operand 0
        // first), so ranking by it gives exactly the order we want.
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
                strides: layouts
                    .iter()
                    .map(|&(size, _)| DimArray::from_slice(&[size]).unwrap())
                    .collect(),
                is_aligned: vec![true; strides.len()],
                is_contiguous: vec![true; strides.len()],
            };
        }
        let (shape, strides) = if dim_perm.len() == 1 {
            // Only one axis remains after dropping size-1 axes: no sort or coalesce needed.
            let d = dim_perm[0];
            let shape = DimArray::from_slice(&[shape[d]]).unwrap();
            let strides = strides
                .iter()
                .map(|s| DimArray::from_slice(&[s[d]]).unwrap())
                .collect::<Vec<_>>();
            (shape, strides)
        } else {
            axes_sort_by(&mut dim_perm, |d1: usize, d2: usize| {
                for strides in strides.iter() {
                    match strides[d1].cmp(&strides[d2]) {
                        std::cmp::Ordering::Less => return std::cmp::Ordering::Greater,
                        std::cmp::Ordering::Equal => {}
                        std::cmp::Ordering::Greater => return std::cmp::Ordering::Less,
                    }
                }
                std::cmp::Ordering::Equal
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
            let strides = strides
                .iter()
                .map(|s| dim_arr(group_inner.len(), |g| s[group_inner[g]]))
                .collect::<Vec<_>>();
            (shape, strides)
        };

        // (3) Compute the innermost-run flags (length, and per-operand alignment / contiguity).
        debug_assert!(!shape.is_empty());
        let ndim = shape.len();
        let is_aligned = strides
            .iter()
            .zip(layouts)
            .map(|(s, &(_, alignment))| s.iter().all(|s| s.is_multiple_of(alignment)))
            .collect();
        let is_contiguous = strides
            .iter()
            .zip(layouts)
            .map(|(s, &(size, _))| s[ndim - 1] == size)
            .collect();

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
        let strides = layouts
            .iter()
            .map(|&(size, _)| {
                let mut s = DimArray::new();
                s.push(size);
                s
            })
            .collect::<Vec<_>>();
        Self {
            is_aligned: vec![true; strides.len()],
            is_contiguous: vec![true; strides.len()],
            shape,
            strides,
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

    #[inline]
    pub(crate) fn foreach_inner_1d(&self, mut inner_loop: impl FnMut(&[usize], usize, &[usize])) {
        let ndim = self.shape.len();
        if crate::hint::likely(ndim == 1) {
            let offsets = vec![0usize; self.strides.len()];
            let inner_strides = self.strides.iter().map(|s| s[0]).collect::<Vec<_>>();
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

/// The `nd_iter_unordered_nd_walk` of [`NdIterUnordered`](crate::NdIterUnordered), instantiated per
/// outer rank only - whatever the number of operands. The offsets are lent out as a slice, so the
/// outer `NdIter` here is the lending kind.
#[allow(clippy::type_complexity)]
#[inline(never)]
fn nd_iter_unordered_nd_walk<D: Dimension>(
    shape: &[usize],
    strides: &[DimArray<usize>],
    inner_loop: &mut dyn FnMut(&[usize], usize, &[usize]),
) {
    let ndim = shape.len();
    let inner_len = shape[ndim - 1];
    let inner_strides = strides.iter().map(|s| s[ndim - 1]).collect::<Vec<_>>();
    let mut offsets = vec![0usize; strides.len()];

    if D::NDIM == Some(1) {
        // Special case for 2D: the outer `NdIter` is just a single loop over the outer axis, and
        // and inner loop is a flat 1-d run.

        let outer_len = shape[0];
        let outer_strides = strides.iter().map(|s| s[0]).collect::<Vec<_>>();
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
        let outer_strides = strides
            .iter()
            .map(|s| D::vec(ndim - 1, |k| s[k]))
            .collect::<Vec<_>>();
        let mut iter = NdIter::builder(outer_shape)
            .with_strides_offset_multi_dyn_ext(outer_strides, offsets)
            .build();
        while let Some((_, offsets)) = iter.advance_and_get() {
            inner_loop(offsets, inner_len, &inner_strides);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NdIterUnordered;

    // ---------------------------------------------------------------------------
    // Harness: what a walk reveals, reduced to a form both iterators share
    // ---------------------------------------------------------------------------

    /// Everything a single walk reveals to its caller: the innermost-run flags it reports, the
    /// per-operand offset of every element visited (in visitation order, each inner run expanded to
    /// one entry per element), and how many times the inner-loop closure ran.
    #[derive(Debug, PartialEq, Eq)]
    struct Walk {
        inner_len: usize,
        is_aligned: Vec<bool>,
        is_contiguous: Vec<bool>,
        visited: Vec<Vec<usize>>,
        inner_calls: usize,
    }

    /// Reconstruct each element's offset from its run base + `k * inner_stride`, so `visited` is
    /// exactly the set of offsets the caller would read.
    fn expand(
        visited: &mut Vec<Vec<usize>>,
        offsets: &[usize],
        len: usize,
        inner_strides: &[usize],
    ) {
        for k in 0..len {
            visited.push(
                offsets
                    .iter()
                    .zip(inner_strides)
                    .map(|(&off, &stride)| off + k * stride)
                    .collect(),
            );
        }
    }

    fn walk_dyn(shape: &[usize], strides: &[&[usize]], layouts: &[(usize, usize)]) -> Walk {
        let iter = NdIterUnorderedDyn::new(shape, strides, layouts);
        let (mut visited, mut inner_calls) = (Vec::new(), 0usize);
        iter.foreach_inner_1d(|offsets, len, inner_strides| {
            inner_calls += 1;
            expand(&mut visited, offsets, len, inner_strides);
        });
        Walk {
            inner_len: iter.inner_len(),
            is_aligned: iter.is_aligned().to_vec(),
            is_contiguous: iter.is_contiguous().to_vec(),
            visited,
            inner_calls,
        }
    }

    fn walk_const<const N: usize>(
        shape: &[usize],
        strides: [&[usize]; N],
        layouts: [(usize, usize); N],
    ) -> Walk {
        let iter = NdIterUnordered::new(shape, strides, layouts);
        let (mut visited, mut inner_calls) = (Vec::new(), 0usize);
        iter.foreach_inner_1d(
            |offsets: [usize; N], len: usize, inner_strides: [usize; N]| {
                inner_calls += 1;
                expand(&mut visited, &offsets, len, &inner_strides);
            },
        );
        Walk {
            inner_len: iter.inner_len(),
            is_aligned: iter.is_aligned().to_vec(),
            is_contiguous: iter.is_contiguous().to_vec(),
            visited,
            inner_calls,
        }
    }

    /// The whole contract of this type: it must be indistinguishable from [`NdIterUnordered`] on the
    /// same input - same flags, same offsets, same order, same number of inner runs. What the walk
    /// itself has to do (axis ordering, coalescing, the flags, the scalar and empty sentinels) is
    /// pinned down by the tests in `nd_iter_unordered.rs`, so the cases below only need to reach
    /// each structurally different path and let the comparison do the checking.
    #[track_caller]
    fn assert_same<const N: usize>(
        shape: &[usize],
        strides: [&[usize]; N],
        layouts: [(usize, usize); N],
    ) -> Walk {
        let got = walk_dyn(shape, &strides, &layouts);
        let expected = walk_const(shape, strides, layouts);
        assert_eq!(
            got, expected,
            "dyn walk differs from NdIterUnordered<{N}> for shape={shape:?} strides={strides:?} layouts={layouts:?}"
        );
        got
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
    // Every structurally different path through `new` / `foreach_inner_1d`
    // ---------------------------------------------------------------------------

    /// A two-operand walk to compare: `(shape, per-operand strides, per-operand (size, alignment))`.
    type Case<'a> = (&'a [usize], [&'a [usize]; 2], [(usize, usize); 2]);

    #[test]
    fn matches_const_generic_on_every_walk_shape() {
        let cases: &[Case] = &[
            // 1-d, taken straight to the single-axis branch of `new`.
            (&[5], [&[1], &[4]], [(1, 1), (4, 4)]),
            (&[4], [&[2], &[8]], [(1, 1), (4, 4)]),
            // Sort and coalesce: C order merges as-is, F order only after being reordered.
            (&[3, 4], [&[4, 1], &[16, 4]], [(1, 1), (4, 4)]),
            (&[3, 4], [&[1, 3], &[4, 12]], [(1, 1), (4, 4)]),
            // A gap on the outer axis blocks the merge, so an outer walk survives.
            (&[2, 3], [&[10, 1], &[40, 4]], [(1, 1), (4, 4)]),
            // Only one of the two operands is contiguous, so the flags must disagree per operand.
            (&[2, 3], [&[12, 4], &[1, 2]], [(4, 4), (1, 1)]),
            // Size-1 axes dropped, and the all-size-1 scalar sentinel.
            (&[1, 4, 1], [&[100, 1, 50], &[7, 4, 9]], [(1, 1), (4, 4)]),
            (&[1, 1], [&[7, 3], &[8, 4]], [(1, 1), (4, 4)]),
            (&[], [&[], &[]], [(4, 4), (1, 1)]),
            // The empty sentinel: nothing is visited.
            (&[2, 0, 3], [&[9, 3, 1], &[36, 12, 4]], [(1, 1), (4, 4)]),
            // Broadcast (zero strides) and a stride that breaks alignment.
            (&[2, 3], [&[0, 0], &[4, 4]], [(1, 1), (4, 4)]),
            (&[3], [&[5], &[8]], [(4, 4), (4, 4)]),
            // One case per outer-rank instantiation of `nd_iter_unordered_nd_walk`: pairing C-order
            // strides against F-order ones keeps every axis from coalescing, so the post-coalesce
            // rank is the input rank and picks Dim<1>, Dim<2>, Dim<3> and then DimDyn in turn.
            (&[2, 3], [&[3, 1], &[1, 2]], [(1, 1), (1, 1)]),
            (&[2, 3, 4], [&[12, 4, 1], &[1, 2, 6]], [(1, 1), (1, 1)]),
            (
                &[2, 3, 4, 5],
                [&[60, 20, 5, 1], &[1, 2, 6, 24]],
                [(1, 1), (1, 1)],
            ),
            (
                &[2, 3, 4, 5, 6],
                [&[360, 120, 30, 6, 1], &[1, 2, 6, 24, 120]],
                [(1, 1), (1, 1)],
            ),
        ];
        for &(shape, strides, layouts) in cases {
            assert_same(shape, strides, layouts);
        }
    }

    // ---------------------------------------------------------------------------
    // What only the runtime-count walk can do
    // ---------------------------------------------------------------------------

    #[test]
    fn operand_count_has_no_ceiling() {
        // The per-operand tables are heap-allocated, so the count is not bounded by anything the
        // callers instantiate - `elementwise_pipeline`'s const-generic dispatch table stops at 16
        // and hands everything above it to this type. Twenty operands, each with its own itemsize
        // and its own gap, still walk exactly like `NdIterUnordered<20>`.
        const N: usize = 20;
        let strides: [Vec<usize>; N] = std::array::from_fn(|i| vec![(i + 1) * 10, i + 1]);
        let refs: [&[usize]; N] = std::array::from_fn(|i| strides[i].as_slice());
        let layouts: [(usize, usize); N] = std::array::from_fn(|i| (i % 4 + 1, 1));
        let walk = assert_same(&[4, 6], refs, layouts);
        // Nothing coalesced (every operand's outer stride leaves a gap), so the inner run is one
        // row and the offsets are 20-wide.
        assert_eq!(walk.inner_len, 6);
        assert_eq!(walk.inner_calls, 4);
        assert_eq!(walk.visited.len(), 24);
        assert!(walk.visited.iter().all(|offsets| offsets.len() == N));
    }

    #[test]
    fn zero_operands_walk_the_shape_and_yield_nothing() {
        // Degenerate but reachable: with no operands the coalesce condition holds vacuously, so the
        // whole shape collapses into one run of empty offset tuples.
        let walk = assert_same::<0>(&[2, 3], [], []);
        assert_eq!(walk.inner_len, 6);
        assert_eq!(walk.inner_calls, 1);
        assert_eq!(walk.visited, vec![Vec::<usize>::new(); 6]);
    }

    // ---------------------------------------------------------------------------
    // Property tests: still indistinguishable under randomized input
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
    fn prop_matches_const_generic_random_strides() {
        // Arbitrary strides (overlapping, broadcast) over ranks up to 6, and shapes that may
        // contain a zero-length axis, so the empty sentinel and the DimDyn outer walk both come up.
        let strategy = (0usize..=6).prop_flat_map(|ndim| {
            (
                prop::collection::vec(0usize..=3, ndim),
                prop::collection::vec(0usize..=6, ndim),
                prop::collection::vec(0usize..=6, ndim),
                prop::collection::vec(0usize..=6, ndim),
            )
        });
        runner(0x51DE)
            .run(&strategy, |(shape, s0, s1, s2)| {
                assert_same(&shape, [&s0, &s1, &s2], [(4, 4), (1, 1), (2, 2)]);
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn prop_matches_const_generic_permuted_contiguous_layouts() {
        // Two independent contiguous-with-gaps layouts presented under a shared random axis
        // permutation: this is what actually forces the descending-stride sort to reorder axes and
        // then run the coalescing merge on the recovered contiguous runs.
        let strategy = (0usize..=5).prop_flat_map(|ndim| {
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
                assert_same(&shape, [&s0, &s1], [(is0, is0), (is1, is1)]);
                Ok(())
            })
            .unwrap();
    }
}
