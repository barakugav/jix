use crate::util::iter::NdIterExtension;
use crate::util::DimArray;
use crate::Dimension;

/// [`NdIterExtension`] that tracks the per-block inner offset and active size for each dimension
/// as a block-space index advances through an N-dimensional sub-range.
///
/// # Usage
///
/// 1. Create the extension with [`NdIterExtBlockOffsetSize::new`], passing the full array `shape`,
///    the element-space `[begin, end)` range, and the [`BlocksLayout`].
/// 2. Compute the block-space iteration bounds from the same inputs:
///    - `block_begin[d] = begin[d] / block_shape[d]`
///    - `block_end[d]   = end[d].div_ceil(block_shape[d])`
/// 3. Pass both to [`NdIter::new_with_begin`].
///
/// Each call to [`NdIter::next`] returns `(block_idx, (inner_offset, block_size))` where
/// `inner_offset` and `block_size` are per-dimension slices describing which elements of the
/// current block fall inside the requested range. Interior blocks (fully covered) always have
/// `inner_offset = 0` and `block_size = block_shape`; border blocks carry their partial values.
pub(crate) struct NdIterExtBlockOffsetSize<D> {
    /// The size of a full block in each dimension.
    block_shape: D,
    /// Low and high border descriptors for each dimension.
    borders: DimArray<(BlocksIterBorder, BlocksIterBorder)>,

    /// Per-dimension element offset within the current block.
    inner_offset: D,
    /// Per-dimension number of active elements in the current block.
    current_block_size: D,
}
/// Describes the boundary block on one end of a single dimension.
struct BlocksIterBorder {
    /// The block index (in block-space) of this border block.
    index: u64,
    /// The element offset inside the block where the requested range begins.
    inner_offset: u64,
    /// The number of elements from this block that fall inside the requested range.
    length: u64,
}
impl<D> NdIterExtBlockOffsetSize<D>
where
    D: Dimension,
{
    #[inline(always)]
    pub(crate) fn new(begin: D, end: D, block_shape: D) -> Self {
        let ndim = begin.ndim();
        assert_eq!(ndim, end.ndim());
        assert_eq!(ndim, block_shape.ndim());

        let mut borders = DimArray::new();
        for dim in 0..ndim {
            assert!(begin[dim] <= end[dim]);
            let block_len = block_shape[dim];

            let low_block_inner_offset = begin[dim] % block_len;
            let low = BlocksIterBorder {
                index: begin[dim] / block_len,
                inner_offset: low_block_inner_offset,
                length: u64::min(block_len - low_block_inner_offset, end[dim] - begin[dim]),
            };
            let high_block_idx = end[dim] / block_len;
            let high_block_inner_offset = if low.index != high_block_idx {
                0
            } else {
                low_block_inner_offset
            };
            let high_block_size = end[dim] % block_len - high_block_inner_offset;

            borders.push((
                low,
                BlocksIterBorder {
                    index: high_block_idx,
                    inner_offset: high_block_inner_offset,
                    length: high_block_size,
                },
            ));
        }

        let inner_offset = D::from_fn(ndim, |dim| borders[dim].0.inner_offset);
        let current_block_size = D::from_fn(ndim, |dim| borders[dim].0.length);
        Self {
            block_shape,
            borders,
            inner_offset,
            current_block_size,
        }
    }
}
impl<D> NdIterExtension for NdIterExtBlockOffsetSize<D>
where
    D: Dimension,
{
    type Item = (D, D);
    #[inline(always)]
    fn on_increase(&mut self, dim: usize, _before: u64, after: u64, _diff: u64) {
        let (low, high) = &self.borders[dim];
        let (offset, size) = if after != high.index {
            debug_assert_ne!(after, low.index);
            (0, self.block_shape[dim])
        } else {
            (high.inner_offset, high.length)
        };
        self.inner_offset[dim] = offset;
        self.current_block_size[dim] = size;
    }

    #[inline(always)]
    fn on_decrease(&mut self, dim: usize, _before: u64, after: u64, _diff: u64) {
        let (low, high) = &self.borders[dim];
        let (offset, size) = if after != low.index {
            debug_assert_ne!(after, high.index);
            (0, self.block_shape[dim])
        } else {
            (low.inner_offset, low.length)
        };
        self.inner_offset[dim] = offset;
        self.current_block_size[dim] = size;
    }

    #[inline(always)]
    fn next(&self) -> Self::Item {
        (self.inner_offset.clone(), self.current_block_size.clone())
    }

    #[inline(always)]
    fn assert_ndim(&self, ndim: usize) {
        assert_eq!(self.block_shape.ndim(), ndim);
        assert_eq!(self.borders.len(), ndim);
        assert_eq!(self.inner_offset.ndim(), ndim);
        assert_eq!(self.current_block_size.ndim(), ndim);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::DimDyn;
    use crate::util::iter::NdIter;

    fn make_iter(
        begin: &[u64],
        end: &[u64],
        block: &[u64],
    ) -> NdIter<DimDyn, NdIterExtBlockOffsetSize<DimDyn>> {
        let ext = NdIterExtBlockOffsetSize::new(
            DimDyn::from_slice(begin),
            DimDyn::from_slice(end),
            DimDyn::from_slice(block),
        );
        let block_begin = begin
            .iter()
            .zip(block)
            .map(|(&b, &c)| b / c)
            .collect::<Vec<_>>();
        let block_end = end
            .iter()
            .zip(block)
            .map(|(&e, &c)| e.div_ceil(c))
            .collect::<Vec<_>>();
        NdIter::new_with_begin(&block_begin, &block_end, ext)
    }

    #[derive(Debug, PartialEq)]
    struct BlocksIterItemOwned {
        block_idx: Vec<u64>,
        inner_offset: Vec<u64>,
        block_size: Vec<u64>,
    }

    fn collect(
        mut iter: NdIter<DimDyn, NdIterExtBlockOffsetSize<DimDyn>>,
    ) -> Vec<BlocksIterItemOwned> {
        let mut out = Vec::new();
        while let Some((block_idx, (inner_offset, block_size))) = iter.next() {
            out.push(BlocksIterItemOwned {
                block_idx: block_idx.as_slice().to_vec(),
                inner_offset: inner_offset.as_slice().to_vec(),
                block_size: block_size.as_slice().to_vec(),
            });
        }
        out
    }

    fn item(block_idx: &[u64], inner_offset: &[u64], block_size: &[u64]) -> BlocksIterItemOwned {
        BlocksIterItemOwned {
            block_idx: block_idx.to_vec(),
            inner_offset: inner_offset.to_vec(),
            block_size: block_size.to_vec(),
        }
    }

    #[test]
    fn full_1d_one_block() {
        assert_eq!(
            collect(make_iter(&[0], &[4], &[4])),
            vec![item(&[0], &[0], &[4])],
        );
    }

    #[test]
    fn full_1d_two_blocks() {
        assert_eq!(
            collect(make_iter(&[0], &[6], &[3])),
            vec![item(&[0], &[0], &[3]), item(&[1], &[0], &[3])],
        );
    }

    #[test]
    fn full_1d_three_blocks() {
        assert_eq!(
            collect(make_iter(&[0], &[9], &[3])),
            vec![
                item(&[0], &[0], &[3]),
                item(&[1], &[0], &[3]),
                item(&[2], &[0], &[3]),
            ],
        );
    }

    #[test]
    fn single_block_full() {
        assert_eq!(
            collect(make_iter(&[0], &[3], &[3])),
            vec![item(&[0], &[0], &[3])],
        );
    }

    #[test]
    fn single_block_offset_start() {
        assert_eq!(
            collect(make_iter(&[1], &[3], &[3])),
            vec![item(&[0], &[1], &[2])],
        );
    }

    #[test]
    fn single_block_interior_slice() {
        assert_eq!(
            collect(make_iter(&[1], &[2], &[3])),
            vec![item(&[0], &[1], &[1])],
        );
    }

    #[test]
    fn single_block_in_middle_of_array() {
        assert_eq!(
            collect(make_iter(&[3], &[5], &[3])),
            vec![item(&[1], &[0], &[2])],
        );
    }

    #[test]
    fn single_block_mid_offset_in_middle_of_array() {
        assert_eq!(
            collect(make_iter(&[4], &[5], &[3])),
            vec![item(&[1], &[1], &[1])],
        );
    }

    #[test]
    fn non_aligned_start_two_blocks() {
        assert_eq!(
            collect(make_iter(&[1], &[6], &[3])),
            vec![item(&[0], &[1], &[2]), item(&[1], &[0], &[3])],
        );
    }

    #[test]
    fn non_aligned_start_three_blocks() {
        assert_eq!(
            collect(make_iter(&[2], &[9], &[3])),
            vec![
                item(&[0], &[2], &[1]),
                item(&[1], &[0], &[3]),
                item(&[2], &[0], &[3]),
            ],
        );
    }

    #[test]
    fn start_at_block_boundary() {
        assert_eq!(
            collect(make_iter(&[3], &[9], &[3])),
            vec![item(&[1], &[0], &[3]), item(&[2], &[0], &[3])],
        );
    }

    #[test]
    fn non_aligned_end_two_blocks() {
        assert_eq!(
            collect(make_iter(&[0], &[5], &[3])),
            vec![item(&[0], &[0], &[3]), item(&[1], &[0], &[2])],
        );
    }

    #[test]
    fn non_aligned_end_three_blocks() {
        assert_eq!(
            collect(make_iter(&[0], &[10], &[4])),
            vec![
                item(&[0], &[0], &[4]),
                item(&[1], &[0], &[4]),
                item(&[2], &[0], &[2]),
            ],
        );
    }

    #[test]
    fn end_aligned_to_block_boundary_within_array() {
        assert_eq!(
            collect(make_iter(&[0], &[6], &[3])),
            vec![item(&[0], &[0], &[3]), item(&[1], &[0], &[3])],
        );
    }

    #[test]
    fn non_aligned_both_two_blocks() {
        assert_eq!(
            collect(make_iter(&[1], &[5], &[3])),
            vec![item(&[0], &[1], &[2]), item(&[1], &[0], &[2])],
        );
    }

    #[test]
    fn non_aligned_both_three_blocks() {
        assert_eq!(
            collect(make_iter(&[1], &[7], &[3])),
            vec![
                item(&[0], &[1], &[2]),
                item(&[1], &[0], &[3]),
                item(&[2], &[0], &[1]),
            ],
        );
    }

    #[test]
    fn non_aligned_both_four_blocks() {
        assert_eq!(
            collect(make_iter(&[1], &[11], &[4])),
            vec![
                item(&[0], &[1], &[3]),
                item(&[1], &[0], &[4]),
                item(&[2], &[0], &[3]),
            ],
        );
    }

    #[test]
    fn block_larger_than_range() {
        assert_eq!(
            collect(make_iter(&[2], &[6], &[8])),
            vec![item(&[0], &[2], &[4])],
        );
    }

    #[test]
    fn full_2d_aligned() {
        assert_eq!(
            collect(make_iter(&[0, 0], &[6, 4], &[3, 2])),
            vec![
                item(&[0, 0], &[0, 0], &[3, 2]),
                item(&[0, 1], &[0, 0], &[3, 2]),
                item(&[1, 0], &[0, 0], &[3, 2]),
                item(&[1, 1], &[0, 0], &[3, 2]),
            ],
        );
    }

    #[test]
    fn full_2d_asymmetric_blocks() {
        assert_eq!(
            collect(make_iter(&[0, 0], &[4, 9], &[2, 3])),
            vec![
                item(&[0, 0], &[0, 0], &[2, 3]),
                item(&[0, 1], &[0, 0], &[2, 3]),
                item(&[0, 2], &[0, 0], &[2, 3]),
                item(&[1, 0], &[0, 0], &[2, 3]),
                item(&[1, 1], &[0, 0], &[2, 3]),
                item(&[1, 2], &[0, 0], &[2, 3]),
            ],
        );
    }

    #[test]
    fn non_aligned_start_2d() {
        assert_eq!(
            collect(make_iter(&[1, 1], &[6, 4], &[3, 2])),
            vec![
                item(&[0, 0], &[1, 1], &[2, 1]),
                item(&[0, 1], &[1, 0], &[2, 2]),
                item(&[1, 0], &[0, 1], &[3, 1]),
                item(&[1, 1], &[0, 0], &[3, 2]),
            ],
        );
    }

    #[test]
    fn non_aligned_start_2d_asymmetric() {
        assert_eq!(
            collect(make_iter(&[1, 2], &[9, 9], &[3, 3])),
            vec![
                item(&[0, 0], &[1, 2], &[2, 1]),
                item(&[0, 1], &[1, 0], &[2, 3]),
                item(&[0, 2], &[1, 0], &[2, 3]),
                item(&[1, 0], &[0, 2], &[3, 1]),
                item(&[1, 1], &[0, 0], &[3, 3]),
                item(&[1, 2], &[0, 0], &[3, 3]),
                item(&[2, 0], &[0, 2], &[3, 1]),
                item(&[2, 1], &[0, 0], &[3, 3]),
                item(&[2, 2], &[0, 0], &[3, 3]),
            ],
        );
    }

    #[test]
    fn non_aligned_both_2d() {
        assert_eq!(
            collect(make_iter(&[1, 1], &[7, 7], &[3, 3])),
            vec![
                item(&[0, 0], &[1, 1], &[2, 2]),
                item(&[0, 1], &[1, 0], &[2, 3]),
                item(&[0, 2], &[1, 0], &[2, 1]),
                item(&[1, 0], &[0, 1], &[3, 2]),
                item(&[1, 1], &[0, 0], &[3, 3]),
                item(&[1, 2], &[0, 0], &[3, 1]),
                item(&[2, 0], &[0, 1], &[1, 2]),
                item(&[2, 1], &[0, 0], &[1, 3]),
                item(&[2, 2], &[0, 0], &[1, 1]),
            ],
        );
    }

    #[test]
    fn full_3d_aligned() {
        let got = collect(make_iter(&[0, 0, 0], &[4, 6, 8], &[2, 3, 4]));
        let expected = (0..2)
            .flat_map(|i| {
                (0..2).flat_map(move |j| {
                    (0..2).map(move |k| item(&[i, j, k], &[0, 0, 0], &[2, 3, 4]))
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(got, expected);
    }
}
