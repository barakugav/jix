use crate::iter::NdIterExtension;
use crate::storage::ChunksLayout;
use crate::util::{DimArray, Idx};

/// [`NdIterExtension`] that tracks the per-chunk inner offset and active size for each dimension
/// as a chunk-space index advances through an N-dimensional sub-range.
///
/// # Usage
///
/// 1. Create the extension with [`NdIterExtChunkOffsetSize::new`], passing the full array `shape`,
///    the element-space `[begin, end)` range, and the [`ChunksLayout`].
/// 2. Compute the chunk-space iteration bounds from the same inputs:
///    - `chunk_begin[d] = begin[d] / chunk_shape[d]`
///    - `chunk_end[d]   = end[d].div_ceil(chunk_shape[d])`
/// 3. Pass both to [`NdIter::new_with_begin`].
///
/// Each call to [`NdIter::next`] returns `(chunk_idx, (inner_offset, chunk_size))` where
/// `inner_offset` and `chunk_size` are per-dimension slices describing which elements of the
/// current chunk fall inside the requested range. Interior chunks (fully covered) always have
/// `inner_offset = 0` and `chunk_size = chunk_shape`; border chunks carry their partial values.
pub(crate) struct NdIterExtChunkOffsetSize<Ix> {
    /// The size of a full chunk in each dimension.
    chunk_shape: DimArray<Ix>,
    /// Low and high border descriptors for each dimension.
    borders: DimArray<(ChunksIterBorder<Ix>, ChunksIterBorder<Ix>)>,

    /// Per-dimension element offset within the current chunk.
    inner_offset: DimArray<Ix>,
    /// Per-dimension number of active elements in the current chunk.
    current_chunk_size: DimArray<Ix>,
}
/// Describes the boundary chunk on one end of a single dimension.
///
/// Both the low (first) and high (last) chunk along a given axis can be partial — they may not
/// start at element 0 or extend to the end of the chunk. This struct captures those details so
/// [`NdIterExtChunkOffsetSize`] can quickly compute the correct offset and size whenever the iterator
/// enters or leaves a border chunk.
struct ChunksIterBorder<Ix> {
    /// The chunk index (in chunk-space) of this border chunk.
    index: Ix,
    /// The element offset inside the chunk where the requested range begins.
    inner_offset: Ix,
    /// The number of elements from this chunk that fall inside the requested range.
    length: Ix,
}
impl<Ix> NdIterExtChunkOffsetSize<Ix>
where
    Ix: Idx,
{
    /// Creates the extension for the sub-range `[begin, end)` of an array with the given `shape`
    /// and `chunks_layout`. The initial state corresponds to the first chunk in the range
    /// (`begin[d] / chunk_shape[d]` in each dimension).
    ///
    /// The caller is responsible for constructing the [`NdIter`] with the correct chunk-space
    /// bounds — see the struct-level documentation for the full usage pattern.
    pub(crate) fn new(
        shape: &[Ix],
        begin: &[Ix],
        end: &[Ix],
        chunks_layout: &ChunksLayout,
    ) -> Self {
        let ndim = shape.len();
        let chunk_shape = chunks_layout
            .chunk_shape
            .iter()
            .map(|&c| c.try_into().unwrap())
            .collect::<DimArray<Ix>>();
        assert_eq!(ndim, begin.len());
        assert_eq!(ndim, end.len());
        assert_eq!(ndim, chunk_shape.len());

        let mut borders = DimArray::new();
        for dim in 0..ndim {
            assert!(begin[dim] <= end[dim]);
            assert!(end[dim] <= shape[dim]);
            let chunk_len = chunk_shape[dim];

            let low_chunk_inner_offset = begin[dim] % chunk_len;
            let low = ChunksIterBorder {
                index: begin[dim] / chunk_len,
                inner_offset: low_chunk_inner_offset,
                length: Ix::min(chunk_len - low_chunk_inner_offset, end[dim] - begin[dim]),
            };
            let high_chunk_idx = end[dim] / chunk_len;
            let high_chunk_inner_offset = if low.index != high_chunk_idx {
                Ix::ZERO
            } else {
                low_chunk_inner_offset
            };
            let high_chunk_size = end[dim] % chunk_len - high_chunk_inner_offset;

            borders.push((
                low,
                ChunksIterBorder {
                    index: high_chunk_idx,
                    inner_offset: high_chunk_inner_offset,
                    length: high_chunk_size,
                },
            ));
        }

        let inner_offset = borders
            .iter()
            .map(|(low, _high)| low.inner_offset)
            .collect::<DimArray<Ix>>();
        let current_chunk_size = borders
            .iter()
            .map(|(low, _high)| low.length)
            .collect::<DimArray<Ix>>();
        Self {
            chunk_shape: chunk_shape.clone(),
            borders,
            inner_offset,
            current_chunk_size,
        }
    }
}
impl<Ix> NdIterExtension<Ix> for NdIterExtChunkOffsetSize<Ix>
where
    Ix: Idx,
{
    type Item<'a>
        = (&'a [Ix], &'a [Ix])
    where
        Self: 'a;

    /// Called when the chunk index along `dim` increases. Updates the inner offset and active
    /// size for that dimension: border chunks get their partial values; interior chunks get
    /// offset 0 and the full chunk size.
    fn on_increase(&mut self, dim: usize, _before: Ix, after: Ix, _diff: Ix) {
        let (low, high) = &self.borders[dim];
        let (offset, size) = if after != high.index {
            debug_assert_ne!(after, low.index);
            (Idx::ZERO, self.chunk_shape[dim])
        } else {
            (high.inner_offset, high.length)
        };
        self.inner_offset[dim] = offset;
        self.current_chunk_size[dim] = size;
    }

    /// Called when the chunk index along `dim` decreases (e.g. when a higher dimension steps and
    /// this dimension resets toward its start). Mirror of [`on_increase`](Self::on_increase).
    fn on_decrease(&mut self, dim: usize, _before: Ix, after: Ix, _diff: Ix) {
        let (low, high) = &self.borders[dim];
        let (offset, size) = if after != low.index {
            debug_assert_ne!(after, high.index);
            (Idx::ZERO, self.chunk_shape[dim])
        } else {
            (low.inner_offset, low.length)
        };
        self.inner_offset[dim] = offset;
        self.current_chunk_size[dim] = size;
    }

    fn next(&self) -> Self::Item<'_> {
        (&self.inner_offset, &self.current_chunk_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iter::NdIter;
    use crate::storage::ChunksLayout;
    use crate::util::Idx;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Build a `NdIter` over the chunk-space range that corresponds to the element-space
    /// `[begin, end)` range, using `NdIterExtChunkOffsetSize` as the extension.
    fn make_iter<Ix: Idx>(
        shape: &[Ix],
        begin: &[Ix],
        end: &[Ix],
        chunk: &[Ix],
    ) -> NdIter<Ix, NdIterExtChunkOffsetSize<Ix>> {
        let chunk_usize = chunk
            .iter()
            .map(|&c| c.try_into().unwrap())
            .collect::<DimArray<usize>>();
        let shape_usize = shape
            .iter()
            .map(|&s| s.try_into().unwrap())
            .collect::<DimArray<usize>>();
        let layout = ChunksLayout::new(&chunk_usize, &shape_usize);
        let ext = NdIterExtChunkOffsetSize::new(shape, begin, end, &layout);
        let chunk_begin: Vec<Ix> = begin.iter().zip(chunk).map(|(&b, &c)| b / c).collect();
        let chunk_end: Vec<Ix> = end
            .iter()
            .zip(chunk)
            .map(|(&e, &c)| e.div_ceil(c))
            .collect();
        NdIter::new_with_begin(&chunk_begin, &chunk_end, ext)
    }

    #[derive(Debug, PartialEq)]
    struct ChunksIterItemOwned<Ix> {
        chunk_idx: Vec<Ix>,
        inner_offset: Vec<Ix>,
        chunk_size: Vec<Ix>,
    }

    fn collect<Ix: Idx>(
        mut iter: NdIter<Ix, NdIterExtChunkOffsetSize<Ix>>,
    ) -> Vec<ChunksIterItemOwned<Ix>> {
        let mut out = Vec::new();
        while let Some((chunk_idx, (inner_offset, chunk_size))) = iter.next() {
            out.push(ChunksIterItemOwned {
                chunk_idx: chunk_idx.to_vec(),
                inner_offset: inner_offset.to_vec(),
                chunk_size: chunk_size.to_vec(),
            });
        }
        out
    }

    fn item<Ix: Idx>(
        chunk_idx: &[usize],
        inner_offset: &[usize],
        chunk_size: &[usize],
    ) -> ChunksIterItemOwned<Ix> {
        ChunksIterItemOwned {
            chunk_idx: chunk_idx.iter().map(|&x| x.try_into().unwrap()).collect(),
            inner_offset: inner_offset
                .iter()
                .map(|&x| x.try_into().unwrap())
                .collect(),
            chunk_size: chunk_size.iter().map(|&x| x.try_into().unwrap()).collect(),
        }
    }

    // -----------------------------------------------------------------------
    // 1D — full array, chunks divide shape evenly
    // -----------------------------------------------------------------------

    #[test]
    fn full_1d_one_chunk() {
        // shape=[4], chunk=4: one full chunk
        assert_eq!(
            collect(make_iter(&[4usize], &[0], &[4], &[4])),
            vec![item(&[0], &[0], &[4])],
        );
    }

    #[test]
    fn full_1d_two_chunks() {
        // shape=[6], chunk=3
        assert_eq!(
            collect(make_iter(&[6usize], &[0], &[6], &[3])),
            vec![item(&[0], &[0], &[3]), item(&[1], &[0], &[3])],
        );
    }

    #[test]
    fn full_1d_three_chunks() {
        // shape=[9], chunk=3
        assert_eq!(
            collect(make_iter(&[9usize], &[0], &[9], &[3])),
            vec![
                item(&[0], &[0], &[3]),
                item(&[1], &[0], &[3]),
                item(&[2], &[0], &[3]),
            ],
        );
    }

    // -----------------------------------------------------------------------
    // 1D — range confined to a single chunk
    // -----------------------------------------------------------------------

    #[test]
    fn single_chunk_full() {
        // range=[0,3) hits only chunk 0, entire chunk
        assert_eq!(
            collect(make_iter(&[9usize], &[0], &[3], &[3])),
            vec![item(&[0], &[0], &[3])],
        );
    }

    #[test]
    fn single_chunk_offset_start() {
        // range=[1,3): chunk 0, offset 1, size 2
        assert_eq!(
            collect(make_iter(&[9usize], &[1], &[3], &[3])),
            vec![item(&[0], &[1], &[2])],
        );
    }

    #[test]
    fn single_chunk_interior_slice() {
        // range=[1,2): chunk 0, offset 1, size 1
        assert_eq!(
            collect(make_iter(&[9usize], &[1], &[2], &[3])),
            vec![item(&[0], &[1], &[1])],
        );
    }

    #[test]
    fn single_chunk_in_middle_of_array() {
        // range=[3,5): chunk 1, offset 0, size 2
        assert_eq!(
            collect(make_iter(&[9usize], &[3], &[5], &[3])),
            vec![item(&[1], &[0], &[2])],
        );
    }

    #[test]
    fn single_chunk_mid_offset_in_middle_of_array() {
        // range=[4,5): chunk 1, offset 1, size 1
        assert_eq!(
            collect(make_iter(&[9usize], &[4], &[5], &[3])),
            vec![item(&[1], &[1], &[1])],
        );
    }

    // -----------------------------------------------------------------------
    // 1D — non-aligned start, end aligned to shape boundary
    // -----------------------------------------------------------------------

    #[test]
    fn non_aligned_start_two_chunks() {
        // shape=[6], chunk=3, range=[1,6): chunk 0 partial, chunk 1 full
        assert_eq!(
            collect(make_iter(&[6usize], &[1], &[6], &[3])),
            vec![item(&[0], &[1], &[2]), item(&[1], &[0], &[3])],
        );
    }

    #[test]
    fn non_aligned_start_three_chunks() {
        // shape=[9], chunk=3, range=[2,9)
        assert_eq!(
            collect(make_iter(&[9usize], &[2], &[9], &[3])),
            vec![
                item(&[0], &[2], &[1]),
                item(&[1], &[0], &[3]),
                item(&[2], &[0], &[3]),
            ],
        );
    }

    #[test]
    fn start_at_chunk_boundary() {
        // shape=[9], chunk=3, range=[3,9): exactly chunk 1 and 2
        assert_eq!(
            collect(make_iter(&[9usize], &[3], &[9], &[3])),
            vec![item(&[1], &[0], &[3]), item(&[2], &[0], &[3])],
        );
    }

    // -----------------------------------------------------------------------
    // 1D — non-aligned end (range ends before the array boundary)
    // -----------------------------------------------------------------------

    #[test]
    fn non_aligned_end_two_chunks() {
        // shape=[9], chunk=3, range=[0,5): chunk 0 full, chunk 1 partial
        assert_eq!(
            collect(make_iter(&[9usize], &[0], &[5], &[3])),
            vec![item(&[0], &[0], &[3]), item(&[1], &[0], &[2])],
        );
    }

    #[test]
    fn non_aligned_end_three_chunks() {
        // shape=[12], chunk=4, range=[0,10): chunks 0,1 full; chunk 2 has 2 elements
        assert_eq!(
            collect(make_iter(&[12usize], &[0], &[10], &[4])),
            vec![
                item(&[0], &[0], &[4]),
                item(&[1], &[0], &[4]),
                item(&[2], &[0], &[2]),
            ],
        );
    }

    #[test]
    fn end_aligned_to_chunk_boundary_within_array() {
        // shape=[9], chunk=3, range=[0,6): chunks 0 and 1 only
        assert_eq!(
            collect(make_iter(&[9usize], &[0], &[6], &[3])),
            vec![item(&[0], &[0], &[3]), item(&[1], &[0], &[3])],
        );
    }

    // -----------------------------------------------------------------------
    // 1D — non-aligned both ends
    // -----------------------------------------------------------------------

    #[test]
    fn non_aligned_both_two_chunks() {
        // shape=[9], chunk=3, range=[1,5): chunk 0 (off=1,sz=2), chunk 1 (off=0,sz=2)
        assert_eq!(
            collect(make_iter(&[9usize], &[1], &[5], &[3])),
            vec![item(&[0], &[1], &[2]), item(&[1], &[0], &[2])],
        );
    }

    #[test]
    fn non_aligned_both_three_chunks() {
        // shape=[9], chunk=3, range=[1,7): partial first, full middle, partial last
        assert_eq!(
            collect(make_iter(&[9usize], &[1], &[7], &[3])),
            vec![
                item(&[0], &[1], &[2]),
                item(&[1], &[0], &[3]),
                item(&[2], &[0], &[1]),
            ],
        );
    }

    #[test]
    fn non_aligned_both_four_chunks() {
        // shape=[12], chunk=4, range=[1,11)
        // chunk 0: off=1, sz=3; chunk 1: full; chunk 2: off=0,sz=3
        assert_eq!(
            collect(make_iter(&[12usize], &[1], &[11], &[4])),
            vec![
                item(&[0], &[1], &[3]),
                item(&[1], &[0], &[4]),
                item(&[2], &[0], &[3]),
            ],
        );
    }

    // -----------------------------------------------------------------------
    // 1D — chunk larger than the range
    // -----------------------------------------------------------------------

    #[test]
    fn chunk_larger_than_range() {
        // shape=[10], chunk=8, range=[2,6): entirely within chunk 0
        assert_eq!(
            collect(make_iter(&[10usize], &[2], &[6], &[8])),
            vec![item(&[0], &[2], &[4])],
        );
    }

    // -----------------------------------------------------------------------
    // 1D — different index types
    // -----------------------------------------------------------------------

    #[test]
    fn index_type_u32_full_array() {
        assert_eq!(
            collect(make_iter(&[6u32], &[0], &[6], &[3])),
            vec![item(&[0], &[0], &[3]), item(&[1], &[0], &[3])],
        );
    }

    #[test]
    fn index_type_u32_non_aligned_both() {
        // shape=[9], chunk=3, range=[1,7)
        assert_eq!(
            collect(make_iter(&[9u32], &[1], &[7], &[3])),
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
            collect(make_iter(&[9u64], &[0], &[9], &[3])),
            vec![
                item(&[0], &[0], &[3]),
                item(&[1], &[0], &[3]),
                item(&[2], &[0], &[3]),
            ],
        );
    }

    // -----------------------------------------------------------------------
    // 2D — full array, evenly divided
    // -----------------------------------------------------------------------

    #[test]
    fn full_2d_aligned() {
        // shape=[6,4], chunk=[3,2]: 2×2 chunks, all full, row-major order
        assert_eq!(
            collect(make_iter(&[6usize, 4], &[0, 0], &[6, 4], &[3, 2])),
            vec![
                item(&[0, 0], &[0, 0], &[3, 2]),
                item(&[0, 1], &[0, 0], &[3, 2]),
                item(&[1, 0], &[0, 0], &[3, 2]),
                item(&[1, 1], &[0, 0], &[3, 2]),
            ],
        );
    }

    #[test]
    fn full_2d_asymmetric_chunks() {
        // shape=[4,9], chunk=[2,3]
        assert_eq!(
            collect(make_iter(&[4usize, 9], &[0, 0], &[4, 9], &[2, 3])),
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

    // -----------------------------------------------------------------------
    // 2D — non-aligned start, end aligned to shape
    // -----------------------------------------------------------------------

    #[test]
    fn non_aligned_start_2d() {
        // shape=[6,4], chunk=[3,2], range=[1,6)×[1,4)
        // dim0: chunk0 (off=1,sz=2), chunk1 (off=0,sz=3)
        // dim1: chunk0 (off=1,sz=1), chunk1 (off=0,sz=2)
        assert_eq!(
            collect(make_iter(&[6usize, 4], &[1, 1], &[6, 4], &[3, 2])),
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
        // shape=[9,9], chunk=[3,3], range=[1,9)×[2,9)
        assert_eq!(
            collect(make_iter(&[9usize, 9], &[1, 2], &[9, 9], &[3, 3])),
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

    // -----------------------------------------------------------------------
    // 2D — non-aligned both ends
    // -----------------------------------------------------------------------

    #[test]
    fn non_aligned_both_2d() {
        // shape=[9,9], chunk=[3,3], range=[1,7)×[1,7)
        // Each dim: chunk0 (off=1,sz=2), chunk1 (off=0,sz=3), chunk2 (off=0,sz=1)
        assert_eq!(
            collect(make_iter(&[9usize, 9], &[1, 1], &[7, 7], &[3, 3])),
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

    // -----------------------------------------------------------------------
    // 3D — full array, evenly divided
    // -----------------------------------------------------------------------

    #[test]
    fn full_3d_aligned() {
        // shape=[4,6,8], chunk=[2,3,4]: 2×2×2 chunks, row-major
        let got = collect(make_iter(
            &[4usize, 6, 8],
            &[0, 0, 0],
            &[4, 6, 8],
            &[2, 3, 4],
        ));
        let expected: Vec<ChunksIterItemOwned<usize>> = (0..2)
            .flat_map(|i| {
                (0..2).flat_map(move |j| {
                    (0..2).map(move |k| item(&[i, j, k], &[0, 0, 0], &[2, 3, 4]))
                })
            })
            .collect();
        assert_eq!(got, expected);
    }
}
