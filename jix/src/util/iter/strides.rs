use crate::util::iter::{impl_merge_extension, NdIterExtension};
use crate::util::Idx;
use crate::Dimension;
use crate::{default_logical_strides_slice, DimVec};

/// An nd-iterator extension that tracks an offset into a strided array.
///
/// On each dimension change the offset is adjusted by the difference in element counts:
/// `offset += (after - before) * stride[dim]`.
pub(crate) struct NdIterExtStridesOffset<D: Dimension, S> {
    strides: D::Vec<S>,
    offset: S,
}
impl<D: Dimension, S: Idx> NdIterExtStridesOffset<D, S> {
    #[inline]
    pub fn new<V>(strides: V, initial_offset: S) -> Self
    where
        D: Dimension<Vec<S> = V>,
        V: DimVec<S, Dimension = D>,
    {
        Self {
            strides,
            offset: initial_offset,
        }
    }
}
impl<D: Dimension, S: Idx> NdIterExtension for NdIterExtStridesOffset<D, S> {
    type Item<'a>
        = S
    where
        Self: 'a;

    #[inline(always)]
    fn on_increase(&mut self, dim: usize, _before: u64, _after: u64, diff: u64) {
        self.offset += S::from_u64(diff) * self.strides[dim];
    }
    #[inline(always)]
    fn on_decrease(&mut self, dim: usize, _before: u64, _after: u64, diff: u64) {
        self.offset -= S::from_u64(diff) * self.strides[dim];
    }

    #[inline(always)]
    fn value(&self) -> Self::Item<'_> {
        self.offset
    }

    #[inline(always)]
    fn check_ndim(&self, ndim: usize) -> bool {
        self.strides.as_ref().len() == ndim
    }

    impl_merge_extension!();
}

/// An nd-iterator extension that tracks `N` offsets into `N` strided arrays simultaneously.
///
/// Like [`NdIterExtStridesOffset`] but for a fixed number of operands sharing one index walk: on
/// each dimension change every offset is advanced by `(after - before) * strides[operand][dim]`, and
/// [`value`](NdIterExtension::value) yields `[S; N]`.
pub(crate) struct NdIterExtStridesOffsetMulti<D: Dimension, S, const N: usize> {
    strides: [D::Vec<S>; N],
    offsets: [S; N],
}
impl<D: Dimension, S: Idx, const N: usize> NdIterExtStridesOffsetMulti<D, S, N> {
    #[inline]
    pub fn new(strides: [D::Vec<S>; N], initial_offsets: [S; N]) -> Self {
        Self {
            strides,
            offsets: initial_offsets,
        }
    }
}
impl<D: Dimension, S: Idx, const N: usize> NdIterExtension
    for NdIterExtStridesOffsetMulti<D, S, N>
{
    type Item<'a>
        = [S; N]
    where
        Self: 'a;

    #[inline(always)]
    fn on_increase(&mut self, dim: usize, _before: u64, _after: u64, diff: u64) {
        let diff = S::from_u64(diff);
        for i in 0..N {
            self.offsets[i] += diff * self.strides[i][dim];
        }
    }
    #[inline(always)]
    fn on_decrease(&mut self, dim: usize, _before: u64, _after: u64, diff: u64) {
        let diff = S::from_u64(diff);
        for i in 0..N {
            self.offsets[i] -= diff * self.strides[i][dim];
        }
    }

    #[inline(always)]
    fn value(&self) -> Self::Item<'_> {
        self.offsets
    }

    #[inline(always)]
    fn check_ndim(&self, ndim: usize) -> bool {
        self.strides.iter().all(|s| s.as_ref().len() == ndim)
    }

    impl_merge_extension!();
}

/// The runtime-length sibling of [`NdIterExtStridesOffsetMulti`], for callers whose operand count is
/// not known at compile time.
pub(crate) struct NdIterExtStridesOffsetMultiDyn<D: Dimension, S> {
    strides: Vec<D::Vec<S>>,
    offsets: Vec<S>,
}
impl<D: Dimension, S: Idx> NdIterExtStridesOffsetMultiDyn<D, S> {
    #[inline]
    pub fn new(strides: Vec<D::Vec<S>>, initial_offsets: Vec<S>) -> Self {
        assert_eq!(strides.len(), initial_offsets.len());
        Self {
            strides,
            offsets: initial_offsets,
        }
    }
}
impl<D: Dimension, S: Idx> NdIterExtension for NdIterExtStridesOffsetMultiDyn<D, S> {
    /// The current offset of each operand, in the order the strides were given.
    type Item<'a>
        = &'a [S]
    where
        Self: 'a;

    #[inline(always)]
    fn on_increase(&mut self, dim: usize, _before: u64, _after: u64, diff: u64) {
        let diff = S::from_u64(diff);
        for (offset, strides) in self.offsets.iter_mut().zip(self.strides.iter()) {
            *offset += diff * strides[dim];
        }
    }
    #[inline(always)]
    fn on_decrease(&mut self, dim: usize, _before: u64, _after: u64, diff: u64) {
        let diff = S::from_u64(diff);
        for (offset, strides) in self.offsets.iter_mut().zip(self.strides.iter()) {
            *offset -= diff * strides[dim];
        }
    }

    #[inline(always)]
    fn value(&self) -> Self::Item<'_> {
        &self.offsets
    }

    #[inline(always)]
    fn check_ndim(&self, ndim: usize) -> bool {
        self.strides.iter().all(|s| s.as_ref().len() == ndim)
    }

    impl_merge_extension!();
}

#[inline]
pub(crate) fn nd_iter_ext_logical_global_index<D: Dimension>(
    shape: &[u64],
    begin: &[u64],
) -> NdIterExtStridesOffset<D, u64> {
    let logical_strides = default_logical_strides_slice(shape);
    let initial_offset = (0..shape.len())
        .map(|dim| begin[dim] * logical_strides[dim])
        .sum();
    NdIterExtStridesOffset::new(
        D::vec(logical_strides.len(), |i| logical_strides[i]),
        initial_offset,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::iter::NdIter;
    use crate::util::DimArray;
    use crate::{Dim, DimDyn, SliceExt};

    /// Build a [`DimArray`] (the `DimDyn` vec container) from a slice, for constructing test
    /// extensions whose `new` takes an owned `D::Vec<S>`.
    fn dv<T: Copy>(s: &[T]) -> DimArray<T> {
        s.to_dim_vec::<DimDyn>()
    }

    #[test]
    fn strides_offset_1d_stride_1() {
        let ext = NdIterExtStridesOffset::new(dv(&[1usize]), 0);
        let iter = NdIter::new(dv(&[8u64]), ext);
        let mut i = 0usize;
        for (_, offset) in iter {
            assert_eq!(offset, i, "step {i}");
            i += 1;
        }
        assert_eq!(i, 8);
    }

    #[test]
    fn strides_offset_1d_larger_stride() {
        let stride = 8usize;
        let ext = NdIterExtStridesOffset::new(dv(&[stride]), 0);
        let iter = NdIter::new(dv(&[8u64]), ext);
        let mut i = 0usize;
        for (_, offset) in iter {
            assert_eq!(offset, i * stride, "step {i}");
            i += 1;
        }
        assert_eq!(i, 8);
    }

    #[test]
    fn strides_offset_2d_row_major_contiguous() {
        let rows = 3usize;
        let cols = 4usize;
        let elem = 4usize; // e.g. f32
        let ext = NdIterExtStridesOffset::new(dv(&[cols * elem, elem]), 0);
        let iter = NdIter::new(dv(&[rows as u64, cols as u64]), ext);
        let mut flat = 0usize;
        for (_, offset) in iter {
            assert_eq!(offset, flat * elem, "flat index {flat}");
            flat += 1;
        }
        assert_eq!(flat, rows * cols);
    }

    #[test]
    fn strides_offset_2d_non_contiguous_column_major() {
        // Simulate a column-major (Fortran-order) 2*3 layout.
        // Row stride = 1, column stride = nrows = 2.
        let rows = 2usize;
        let cols = 3usize;
        let ext = NdIterExtStridesOffset::new(dv(&[1, rows]), 0);
        let iter = NdIter::new(dv(&[rows as u64, cols as u64]), ext);
        // Iteration order is row-major by *index*, but the offsets follow the column-major layout:
        // [0,0]=0, [0,1]=2, [0,2]=4, [1,0]=1, [1,1]=3, [1,2]=5
        let expected_offsets: &[usize] = &[0, 2, 4, 1, 3, 5];
        let mut i = 0usize;
        for (_, offset) in iter {
            assert_eq!(offset, expected_offsets[i], "step {i}");
            i += 1;
        }
        assert_eq!(i, rows * cols);
    }

    #[test]
    fn strides_offset_with_begin_offset() {
        // begin=[1,1], end=[3,3], strides=[4,1].
        // Initial offset = 1*4 + 1*1 = 5.
        // Expected traversal offsets: [1,1]=5, [1,2]=6, [2,1]=9, [2,2]=10.
        let ext = NdIterExtStridesOffset::new(dv(&[4usize, 1]), 4 + 1);
        let iter = NdIter::new_with_begin(dv(&[1u64, 1]), dv(&[3u64, 3]), ext);
        let expected: &[usize] = &[5, 6, 9, 10];
        let mut i = 0;
        for (_, offset) in iter {
            assert_eq!(offset, expected[i], "step {i}");
            i += 1;
        }
        assert_eq!(i, 4);
    }

    #[test]
    fn strides_offset_empty_range_yields_no_offsets() {
        let ext = NdIterExtStridesOffset::new(dv(&[4usize, 1]), 0);
        let mut iter = NdIter::new_with_begin(dv(&[2u64, 2]), dv(&[2u64, 5]), ext);
        assert!(iter.next().is_none());
    }

    #[test]
    fn strides_offset_u64_offsets() {
        // The offset type is generic over `Idx`, not just `usize`.
        let ext = NdIterExtStridesOffset::new(dv(&[3u64, 1]), 0u64);
        let iter = NdIter::new(dv(&[2u64, 3]), ext);
        let offsets: Vec<u64> = iter.map(|(_, offset)| offset).collect();
        assert_eq!(offsets, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn strides_offset_2d_indexes_correct_elements() {
        // Functional check: walk a contiguous row-major buffer and write through each offset.
        let rows = 3usize;
        let cols = 4usize;
        let mut data = vec![0u32; rows * cols];
        let ext = NdIterExtStridesOffset::new(dv(&[cols, 1]), 0);
        let iter = NdIter::new(dv(&[rows as u64, cols as u64]), ext);
        let mut flat = 0u32;
        for (_, offset) in iter {
            assert_eq!(offset, flat as usize, "flat index {flat}");
            data[offset] = flat * 10 + 7;
            flat += 1;
        }
        assert_eq!(flat as usize, rows * cols);
        let expected: Vec<u32> = (0..(rows * cols) as u32).map(|f| f * 10 + 7).collect();
        assert_eq!(data, expected);
    }

    // --- Static Dim<N> variants -----------------------------------------------------------
    // These mirror key DimDyn tests using fixed-size Dim<N> to confirm both paths compile and work.

    #[test]
    fn strides_offset_dim1_static() {
        let ext = NdIterExtStridesOffset::new([1usize], 0);
        let iter = NdIter::<Dim<1>, _>::new([8u64], ext);
        let mut i = 0usize;
        for (_, offset) in iter {
            assert_eq!(offset, i, "step {i}");
            i += 1;
        }
        assert_eq!(i, 8);
    }

    #[test]
    fn strides_offset_dim2_static() {
        let rows = 3usize;
        let cols = 4usize;
        let elem = 4usize;
        let ext = NdIterExtStridesOffset::new([cols * elem, elem], 0);
        let iter = NdIter::<Dim<2>, _>::new([rows as u64, cols as u64], ext);
        let mut flat = 0usize;
        for (_, offset) in iter {
            assert_eq!(offset, flat * elem, "flat index {flat}");
            flat += 1;
        }
        assert_eq!(flat, rows * cols);
    }
}
