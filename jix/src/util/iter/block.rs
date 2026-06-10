use crate::util::iter::NdIterExtension;
use crate::util::{dim_arr, DimArray, Idx};

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
pub(crate) struct NdIterExtBlockOffsetSize<Ix> {
    /// The size of a full block in each dimension.
    block_shape: DimArray<Ix>,
    /// Low and high border descriptors for each dimension.
    borders: DimArray<(BlocksIterBorder<Ix>, BlocksIterBorder<Ix>)>,

    /// Per-dimension element offset within the current block.
    inner_offset: DimArray<Ix>,
    /// Per-dimension number of active elements in the current block.
    current_block_size: DimArray<Ix>,
}
/// Describes the boundary block on one end of a single dimension.
struct BlocksIterBorder<Ix> {
    /// The block index (in block-space) of this border block.
    index: Ix,
    /// The element offset inside the block where the requested range begins.
    inner_offset: Ix,
    /// The number of elements from this block that fall inside the requested range.
    length: Ix,
}
impl<Ix> NdIterExtBlockOffsetSize<Ix>
where
    Ix: Idx,
{
    pub(crate) fn new(begin: &[Ix], end: &[Ix], block_shape: &[Ix]) -> Self {
        let ndim = begin.len();
        assert_eq!(ndim, end.len());
        assert_eq!(ndim, block_shape.len());

        let mut borders = DimArray::new();
        for dim in 0..ndim {
            assert!(begin[dim] <= end[dim]);
            let block_len = block_shape[dim];

            let low_block_inner_offset = begin[dim] % block_len;
            let low = BlocksIterBorder {
                index: begin[dim] / block_len,
                inner_offset: low_block_inner_offset,
                length: Ix::min(block_len - low_block_inner_offset, end[dim] - begin[dim]),
            };
            let high_block_idx = end[dim] / block_len;
            let high_block_inner_offset = if low.index != high_block_idx {
                Ix::ZERO
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

        let inner_offset = dim_arr(ndim, |dim| borders[dim].0.inner_offset);
        let current_block_size = dim_arr(ndim, |dim| borders[dim].0.length);
        Self {
            block_shape: DimArray::from_slice(block_shape).unwrap(),
            borders,
            inner_offset,
            current_block_size,
        }
    }
}
impl<Ix> NdIterExtension<Ix> for NdIterExtBlockOffsetSize<Ix>
where
    Ix: Idx,
{
    type Item<'a>
        = (&'a [Ix], &'a [Ix])
    where
        Self: 'a;

    #[inline(always)]
    fn on_increase(&mut self, dim: usize, _before: Ix, after: Ix, _diff: Ix) {
        let (low, high) = &self.borders[dim];
        let (offset, size) = if after != high.index {
            debug_assert_ne!(after, low.index);
            (Idx::ZERO, self.block_shape[dim])
        } else {
            (high.inner_offset, high.length)
        };
        self.inner_offset[dim] = offset;
        self.current_block_size[dim] = size;
    }

    #[inline(always)]
    fn on_decrease(&mut self, dim: usize, _before: Ix, after: Ix, _diff: Ix) {
        let (low, high) = &self.borders[dim];
        let (offset, size) = if after != low.index {
            debug_assert_ne!(after, high.index);
            (Idx::ZERO, self.block_shape[dim])
        } else {
            (low.inner_offset, low.length)
        };
        self.inner_offset[dim] = offset;
        self.current_block_size[dim] = size;
    }

    #[inline(always)]
    fn next(&self) -> Self::Item<'_> {
        (&self.inner_offset, &self.current_block_size)
    }

    fn assert_ndim(&self, ndim: usize) {
        assert_eq!(self.block_shape.len(), ndim);
        assert_eq!(self.borders.len(), ndim);
        assert_eq!(self.inner_offset.len(), ndim);
        assert_eq!(self.current_block_size.len(), ndim);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::iter::NdIter;
    use crate::util::Idx;

    fn make_iter<Ix: Idx>(
        begin: &[Ix],
        end: &[Ix],
        block: &[Ix],
    ) -> NdIter<Ix, NdIterExtBlockOffsetSize<Ix>> {
        let ext = NdIterExtBlockOffsetSize::new(begin, end, block);
        let block_begin: Vec<Ix> = begin.iter().zip(block).map(|(&b, &c)| b / c).collect();
        let block_end: Vec<Ix> = end
            .iter()
            .zip(block)
            .map(|(&e, &c)| e.div_ceil(c))
            .collect();
        NdIter::new_with_begin(&block_begin, &block_end, ext)
    }

    #[derive(Debug, PartialEq)]
    struct BlocksIterItemOwned<Ix> {
        block_idx: Vec<Ix>,
        inner_offset: Vec<Ix>,
        block_size: Vec<Ix>,
    }

    fn collect<Ix: Idx>(
        mut iter: NdIter<Ix, NdIterExtBlockOffsetSize<Ix>>,
    ) -> Vec<BlocksIterItemOwned<Ix>> {
        let mut out = Vec::new();
        while let Some((block_idx, (inner_offset, block_size))) = iter.next() {
            out.push(BlocksIterItemOwned {
                block_idx: block_idx.to_vec(),
                inner_offset: inner_offset.to_vec(),
                block_size: block_size.to_vec(),
            });
        }
        out
    }

    fn item<Ix: Idx>(
        block_idx: &[usize],
        inner_offset: &[usize],
        block_size: &[usize],
    ) -> BlocksIterItemOwned<Ix> {
        BlocksIterItemOwned {
            block_idx: block_idx.iter().map(|&x| x.try_into().unwrap()).collect(),
            inner_offset: inner_offset
                .iter()
                .map(|&x| x.try_into().unwrap())
                .collect(),
            block_size: block_size.iter().map(|&x| x.try_into().unwrap()).collect(),
        }
    }

    #[test]
    fn full_1d_one_block() {
        assert_eq!(
            collect(make_iter(&[0usize], &[4], &[4])),
            vec![item(&[0], &[0], &[4])],
        );
    }

    #[test]
    fn full_1d_two_blocks() {
        assert_eq!(
            collect(make_iter(&[0usize], &[6], &[3])),
            vec![item(&[0], &[0], &[3]), item(&[1], &[0], &[3])],
        );
    }

    #[test]
    fn full_1d_three_blocks() {
        assert_eq!(
            collect(make_iter(&[0usize], &[9], &[3])),
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
            collect(make_iter(&[0usize], &[3], &[3])),
            vec![item(&[0], &[0], &[3])],
        );
    }

    #[test]
    fn single_block_offset_start() {
        assert_eq!(
            collect(make_iter(&[1usize], &[3], &[3])),
            vec![item(&[0], &[1], &[2])],
        );
    }

    #[test]
    fn single_block_interior_slice() {
        assert_eq!(
            collect(make_iter(&[1usize], &[2], &[3])),
            vec![item(&[0], &[1], &[1])],
        );
    }

    #[test]
    fn single_block_in_middle_of_array() {
        assert_eq!(
            collect(make_iter(&[3usize], &[5], &[3])),
            vec![item(&[1], &[0], &[2])],
        );
    }

    #[test]
    fn single_block_mid_offset_in_middle_of_array() {
        assert_eq!(
            collect(make_iter(&[4usize], &[5], &[3])),
            vec![item(&[1], &[1], &[1])],
        );
    }

    #[test]
    fn non_aligned_start_two_blocks() {
        assert_eq!(
            collect(make_iter(&[1usize], &[6], &[3])),
            vec![item(&[0], &[1], &[2]), item(&[1], &[0], &[3])],
        );
    }

    #[test]
    fn non_aligned_start_three_blocks() {
        assert_eq!(
            collect(make_iter(&[2usize], &[9], &[3])),
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
            collect(make_iter(&[3usize], &[9], &[3])),
            vec![item(&[1], &[0], &[3]), item(&[2], &[0], &[3])],
        );
    }

    #[test]
    fn non_aligned_end_two_blocks() {
        assert_eq!(
            collect(make_iter(&[0usize], &[5], &[3])),
            vec![item(&[0], &[0], &[3]), item(&[1], &[0], &[2])],
        );
    }

    #[test]
    fn non_aligned_end_three_blocks() {
        assert_eq!(
            collect(make_iter(&[0usize], &[10], &[4])),
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
            collect(make_iter(&[0usize], &[6], &[3])),
            vec![item(&[0], &[0], &[3]), item(&[1], &[0], &[3])],
        );
    }

    #[test]
    fn non_aligned_both_two_blocks() {
        assert_eq!(
            collect(make_iter(&[1usize], &[5], &[3])),
            vec![item(&[0], &[1], &[2]), item(&[1], &[0], &[2])],
        );
    }

    #[test]
    fn non_aligned_both_three_blocks() {
        assert_eq!(
            collect(make_iter(&[1usize], &[7], &[3])),
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
            collect(make_iter(&[1usize], &[11], &[4])),
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
            collect(make_iter(&[2usize], &[6], &[8])),
            vec![item(&[0], &[2], &[4])],
        );
    }

    #[test]
    fn index_type_u32_full_array() {
        assert_eq!(
            collect(make_iter(&[0u32], &[6], &[3])),
            vec![item(&[0], &[0], &[3]), item(&[1], &[0], &[3])],
        );
    }

    #[test]
    fn index_type_u32_non_aligned_both() {
        assert_eq!(
            collect(make_iter(&[1u32], &[7], &[3])),
            vec![
                item(&[0], &[1], &[2]),
                item(&[1], &[0], &[3]),
                item(&[2], &[0], &[1]),
            ],
        );
    }

    #[test]
    fn index_type_u64_full_array() {
        assert_eq!(
            collect(make_iter(&[0u64], &[9], &[3])),
            vec![
                item(&[0], &[0], &[3]),
                item(&[1], &[0], &[3]),
                item(&[2], &[0], &[3]),
            ],
        );
    }

    #[test]
    fn full_2d_aligned() {
        assert_eq!(
            collect(make_iter(&[0usize, 0], &[6, 4], &[3, 2])),
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
            collect(make_iter(&[0usize, 0], &[4, 9], &[2, 3])),
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
            collect(make_iter(&[1usize, 1], &[6, 4], &[3, 2])),
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
            collect(make_iter(&[1usize, 2], &[9, 9], &[3, 3])),
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
            collect(make_iter(&[1usize, 1], &[7, 7], &[3, 3])),
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
        let got = collect(make_iter(&[0usize, 0, 0], &[4, 6, 8], &[2, 3, 4]));
        let expected: Vec<BlocksIterItemOwned<usize>> = (0..2)
            .flat_map(|i| {
                (0..2).flat_map(move |j| {
                    (0..2).map(move |k| item(&[i, j, k], &[0, 0, 0], &[2, 3, 4]))
                })
            })
            .collect();
        assert_eq!(got, expected);
    }
}
