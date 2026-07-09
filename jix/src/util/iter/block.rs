use crate::util::iter::NdIterExtension;
use crate::{DimVec, Dimension};

/// [`NdIterExtension`] that tracks the per-block inner offset and active size for each dimension
/// as a block-space index advances through an N-dimensional sub-range.
///
/// # Usage
///
/// 1. Create the extension with [`NdIterExtBlockOffsetSize::new`], passing the full array `shape`,
///    the element-space `[begin, end)` range.
/// 2. Compute the block-space iteration bounds from the same inputs:
///    - `block_begin[d] = begin[d] / block_shape[d]`
///    - `block_end[d]   = end[d].div_ceil(block_shape[d])`
/// 3. Pass both to [`NdIter::new_with_begin`].
///
/// Each call to [`NdIter::next`] returns `(block_idx, (inner_offset, block_size))` where
/// `inner_offset` and `block_size` are per-dimension slices describing which elements of the
/// current block fall inside the requested range. Interior blocks (fully covered) always have
/// `inner_offset = 0` and `block_size = block_shape`; border blocks carry their partial values.
pub(crate) struct NdIterExtBlockOffsetSize<D: Dimension> {
    /// The size of a full block in each dimension.
    block_shape: D::Vec<u64>,
    /// Low and high border descriptors for each dimension.
    borders: D::Vec<(BlocksIterBorder, BlocksIterBorder)>,

    /// Per-dimension element offset within the current block.
    inner_offset: D::Vec<u64>,
    /// Per-dimension number of active elements in the current block.
    current_block_size: D::Vec<u64>,
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
    pub(crate) fn new<V>(begin: &V, end: &V, block_shape: V) -> Self
    where
        D: Dimension<Vec<u64> = V>,
        V: DimVec<u64, Dimension = D>,
    {
        let ndim = begin.as_ref().len();
        assert_eq!(ndim, end.as_ref().len());
        assert_eq!(ndim, block_shape.as_ref().len());

        let borders = D::vec(ndim, |dim| {
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

            (
                low,
                BlocksIterBorder {
                    index: high_block_idx,
                    inner_offset: high_block_inner_offset,
                    length: high_block_size,
                },
            )
        });

        let inner_offset = D::vec(ndim, |dim| borders[dim].0.inner_offset);
        let current_block_size = D::vec(ndim, |dim| borders[dim].0.length);
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
    type Item = (D::Vec<u64>, D::Vec<u64>);
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
        assert_eq!(self.block_shape.as_ref().len(), ndim);
        assert_eq!(self.borders.as_ref().len(), ndim);
        assert_eq!(self.inner_offset.as_ref().len(), ndim);
        assert_eq!(self.current_block_size.as_ref().len(), ndim);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::DimDyn;
    use crate::util::calc_block_end;
    use crate::util::iter::NdIter;
    use crate::SliceExt;

    fn make_iter(
        begin: &[u64],
        end: &[u64],
        block: &[u64],
    ) -> NdIter<DimDyn, NdIterExtBlockOffsetSize<DimDyn>> {
        let ext = NdIterExtBlockOffsetSize::new(
            &begin.to_dim_vec::<DimDyn>(),
            &end.to_dim_vec::<DimDyn>(),
            block.to_dim_vec::<DimDyn>(),
        );
        let (block_begin, block_end) = begin
            .iter()
            .zip(end)
            .zip(block)
            .map(|((&b, &e), &c)| (b / c, calc_block_end(b, e, c)))
            .unzip::<_, _, Vec<_>, Vec<_>>();

        NdIter::new_with_begin(
            block_begin.to_dim_vec::<DimDyn>(),
            block_end.to_dim_vec::<DimDyn>(),
            ext,
        )
    }

    #[derive(Debug, PartialEq)]
    struct BlocksIterItemOwned {
        block_idx: Vec<u64>,
        inner_offset: Vec<u64>,
        block_size: Vec<u64>,
    }

    fn collect(iter: NdIter<DimDyn, NdIterExtBlockOffsetSize<DimDyn>>) -> Vec<BlocksIterItemOwned> {
        let mut out = Vec::new();
        for (block_idx, (inner_offset, block_size)) in iter {
            out.push(BlocksIterItemOwned {
                block_idx: block_idx.as_ref().to_vec(),
                inner_offset: inner_offset.as_ref().to_vec(),
                block_size: block_size.as_ref().to_vec(),
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
