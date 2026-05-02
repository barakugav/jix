use std::ops::Range;

use crate::codec::ReadContext;
use crate::dtype::Dtype;
use crate::error::{check_get_range, check_ndim, ensure, Result};
use crate::storage::{ArrayStorage, ArrayStorageSpec, BlockShapeTag, BlocksLayout};
use crate::util::DimArray;
use crate::Array;

/// Inserts new length-1 dimensions at specified positions in an array's shape,
/// returned by [`Array::insert_axes`](crate::Array::insert_axes).
///
/// Each element of `axes` is a **gap index** that identifies a position *between* (or outside)
/// the input dimensions:
///
/// ```text
/// gap:   0     1     2       orig_ndim
///         |  d0  |  d1  |  d2  |
/// ```
///
/// * Gap `0` — before the first input dimension.
/// * Gap `k` — between input dimensions `k-1` and `k`.
/// * Gap `orig_ndim` — after the last input dimension.
///
/// Each occurrence of a gap index inserts one new length-1 dimension at that position. Duplicate
/// gap indices are allowed and each adds another dimension at the same gap. The order of values in
/// `axes` does not matter — only the multiset of gap indices matters. Valid gap indices are
/// `0..=orig_ndim`.
///
/// Output dtype equals the input dtype.
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// # Examples
///
/// ```text
/// [N]       axes: [0]     → [1, N]      (insert before first dim)
/// [N]       axes: [1]     → [N, 1]      (append after last dim)
/// [N, M]    axes: [1]     → [N, 1, M]   (insert between dims)
/// [N, M]    axes: [0, 2]  → [1, N, M, 1]
/// ```
///
/// ```
/// use zix::{Array, ArrayParams};
/// use ndarray::array;
///
/// // [3] → [1, 3]
/// let a = Array::compact_array(&array![1i32, 2, 3])?;
/// assert_eq!(a.insert_axes(&[0]).shape(), &[1, 3]);
///
/// // [3] → [3, 1]
/// let b = Array::compact_array(&array![1i32, 2, 3])?;
/// assert_eq!(b.insert_axes(&[1]).shape(), &[3, 1]);
///
/// // [2, 3] → [1, 2, 3, 1]
/// let c = Array::compact_array(&array![[1i32, 2, 3], [4, 5, 6]])?;
/// assert_eq!(c.insert_axes(&[0, 2]).shape(), &[1, 2, 3, 1]);
/// # Ok::<(), zix::Error>(())
/// ```
pub struct InsertAxes<S> {
    array: Array<S>,
    /// `is_inserted[output_dim]` is `true` for every output dimension that was inserted
    /// (length 1, no corresponding input dimension).
    is_inserted: DimArray<bool>,

    dtype: Dtype,
    shape: DimArray<u64>,
    blocks_layout: BlocksLayout,
}

impl<S: ArrayStorage> InsertAxes<S> {
    /// Constructs an `InsertAxes` storage. See [`InsertAxes`] for semantics and examples.
    pub fn new(array: Array<S>, axes: &[usize]) -> Result<Self> {
        let orig_ndim = array.shape().len();
        let new_ndim = orig_ndim + axes.len();

        check_ndim(new_ndim)?;

        // Each value in `axes` is a gap index in the *input* shape: 0 means "before input dim 0",
        // 1 means "before input dim 1" (i.e. between dims 0 and 1), ..., orig_ndim means "after
        // the last input dim".  Duplicates are allowed — each occurrence inserts one additional
        // dim at that gap.  Valid range: 0..=orig_ndim.
        for &ax in axes {
            ensure!(
                ax <= orig_ndim,
                InvalidShapeOperation,
                "axis {ax} out of bounds for array of ndim {orig_ndim} \
                     (gap indices must be in 0..={orig_ndim})"
            );
        }

        // Sort a local copy so we can walk input dims and inserted gaps together in one pass.
        let mut sorted_axes: DimArray<_> = axes.try_into().unwrap();
        sorted_axes.sort_unstable();
        let mut sorted_axes = sorted_axes.iter().peekable();

        // Build is_inserted and shape by interleaving: for each gap `g` (0..=orig_ndim),
        // first emit all inserted dims that belong at gap `g`, then emit input dim `g`
        // (if g < orig_ndim).
        let mut is_inserted = DimArray::new();
        let mut shape = DimArray::new();

        for input_dim in 0..orig_ndim {
            while sorted_axes.peek() == Some(&&input_dim) {
                is_inserted.push(true);
                shape.push(1u64);
                sorted_axes.next();
            }
            is_inserted.push(false);
            shape.push(array.shape()[input_dim]);
        }
        // Remaining axes sit at gap orig_ndim (after the last input dim).
        for _ in sorted_axes {
            is_inserted.push(true);
            shape.push(1u64);
        }

        // Build blocks_layout: inserted dims get block_shape = 1 (Any); non-inserted dims
        // inherit the corresponding input dim's layout unchanged.
        let inner_layout = array.blocks_layout();
        let mut hint = DimArray::new();
        let mut tag = DimArray::new();
        let mut preferred = DimArray::new();
        let mut input_dim = 0usize;
        for &inserted in is_inserted.iter() {
            if inserted {
                hint.push(1);
                tag.push(BlockShapeTag::Any);
                preferred.push(1);
            } else {
                hint.push(inner_layout.block_shape_hint[input_dim]);
                tag.push(inner_layout.block_shape_tag[input_dim]);
                preferred.push(inner_layout.preferred_read_shape[input_dim]);
                input_dim += 1;
            }
        }
        let mut b_layout = inner_layout.clone();
        b_layout.block_shape_hint = hint;
        b_layout.block_shape_tag = tag;
        b_layout.preferred_read_shape = preferred;

        let dtype = array.dtype().clone();
        Ok(Self {
            array,
            is_inserted,
            dtype,
            shape,
            blocks_layout: b_layout,
        })
    }
}

impl<S: ArrayStorage> ArrayStorage for InsertAxes<S> {
    fn read_data(&self, index: &[Range<u64>], buf: &mut [u8], context: &ReadContext) -> Result<()> {
        check_get_range(self.shape(), index)?;

        // Inserted dimensions have size 1 and do not affect the element sequence.
        // Because the output is always C-contiguous, a size-1 dimension is a no-op
        // in the memory layout: its stride equals the stride of the next dimension,
        // and there is exactly one step along it, so the elements appear in the same
        // order as without it.
        //
        // Therefore we simply strip all inserted dims from `index` and forward the
        // remaining ranges to the inner storage unchanged.  No temporary buffer or
        // element rearrangement is needed.
        let mut inner_index = DimArray::new();
        for (dim, index) in index.iter().enumerate() {
            if !self.is_inserted[dim] {
                inner_index.push(index.clone());
            } else {
                if index.start == index.end {
                    return Ok(()); // empty read
                }
                debug_assert_eq!(*index, 0..1);
            }
        }
        self.array.storage.read_data(&inner_index, buf, context)
    }

    fn shape(&self) -> &[u64] {
        &self.shape
    }
    fn dtype(&self) -> &Dtype {
        &self.dtype
    }
    fn _spec(&self) -> ArrayStorageSpec<'_> {
        ArrayStorageSpec {
            blocks_layout: &self.blocks_layout,
            ..self.array.storage._spec()
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::ArrayD;

    use crate::{array::Array, codec::ReadContext, util::arr_params};

    fn make1d(vals: Vec<i32>, block_size: usize) -> Array<crate::storage::Compact> {
        let nd = ArrayD::from_shape_vec(vec![vals.len()], vals).unwrap();
        Array::compact_array_with(&nd, arr_params(&[block_size])).unwrap()
    }

    fn make2d(vals: Vec<i32>, rows: usize, cols: usize) -> Array<crate::storage::Compact> {
        let nd = ArrayD::from_shape_vec(vec![rows, cols], vals).unwrap();
        Array::compact_array_with(&nd, arr_params(&[rows, cols])).unwrap()
    }

    fn make3d(vals: Vec<i32>, d0: usize, d1: usize, d2: usize) -> Array<crate::storage::Compact> {
        let nd = ArrayD::from_shape_vec(vec![d0, d1, d2], vals).unwrap();
        Array::compact_array_with(&nd, arr_params(&[d0, d1, d2])).unwrap()
    }

    fn seq(n: usize) -> Vec<i32> {
        (0..n as i32).collect()
    }

    // -----------------------------------------------------------------------
    // Shape metadata
    // -----------------------------------------------------------------------

    #[test]
    fn shape_insert_before_first_dim() {
        // gap 0 on [6] → [1, 6]
        assert_eq!(make1d(seq(6), 6).insert_axes(&[0]).shape(), &[1, 6]);
    }

    #[test]
    fn shape_insert_after_last_dim() {
        // gap 1 (=orig_ndim) on [6] → [6, 1]
        assert_eq!(make1d(seq(6), 6).insert_axes(&[1]).shape(), &[6, 1]);
    }

    #[test]
    fn shape_insert_between_dims() {
        // gap 1 on [3, 4] → [3, 1, 4]
        assert_eq!(make2d(seq(12), 3, 4).insert_axes(&[1]).shape(), &[3, 1, 4]);
    }

    #[test]
    fn shape_insert_front_and_back() {
        // gaps 0 and 1 on [6] → [1, 6, 1]
        assert_eq!(make1d(seq(6), 6).insert_axes(&[0, 1]).shape(), &[1, 6, 1]);
    }

    #[test]
    fn shape_insert_duplicates_same_gap() {
        // gaps 0, 0 on [3, 4] → [1, 1, 3, 4]
        assert_eq!(
            make2d(seq(12), 3, 4).insert_axes(&[0, 0]).shape(),
            &[1, 1, 3, 4]
        );
    }

    #[test]
    fn shape_insert_user_example() {
        // axes=(0,1,1,1,3) on (N=2, M=3, K=4) → (1, 2, 1, 1, 1, 3, 4, 1)
        let a = make3d(seq(24), 2, 3, 4);
        assert_eq!(
            a.insert_axes(&[0, 1, 1, 1, 3]).shape(),
            &[1, 2, 1, 1, 1, 3, 4, 1]
        );
    }

    #[test]
    fn shape_insert_unsorted_axes_same_result() {
        // Order of axes values should not matter; only the multiset matters.
        let a1 = make3d(seq(24), 2, 3, 4).insert_axes(&[0, 1, 1, 1, 3]);
        let a2 = make3d(seq(24), 2, 3, 4).insert_axes(&[3, 1, 0, 1, 1]);
        assert_eq!(a1.shape(), a2.shape());
    }

    #[test]
    fn shape_empty_axes_is_identity() {
        assert_eq!(make2d(seq(12), 3, 4).insert_axes(&[]).shape(), &[3, 4]);
    }

    // -----------------------------------------------------------------------
    // Full reads
    // -----------------------------------------------------------------------

    #[test]
    fn full_read_insert_before_first() {
        let got: ArrayD<i32> = make1d(seq(6), 6).insert_axes(&[0]).to_ndarray().unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![1, 6], seq(6)).unwrap());
    }

    #[test]
    fn full_read_insert_after_last() {
        let got: ArrayD<i32> = make1d(seq(6), 6).insert_axes(&[1]).to_ndarray().unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![6, 1], seq(6)).unwrap());
    }

    #[test]
    fn full_read_insert_between_dims() {
        let got: ArrayD<i32> = make2d(seq(12), 3, 4)
            .insert_axes(&[1])
            .to_ndarray()
            .unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![3, 1, 4], seq(12)).unwrap());
    }

    #[test]
    fn full_read_insert_front_and_back() {
        let got: ArrayD<i32> = make1d(seq(6), 6).insert_axes(&[0, 1]).to_ndarray().unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![1, 6, 1], seq(6)).unwrap());
    }

    #[test]
    fn full_read_insert_user_example() {
        // axes=(0,1,1,1,3) on (2,3,4) → (1,2,1,1,1,3,4,1), elements unchanged
        let got: ArrayD<i32> = make3d(seq(24), 2, 3, 4)
            .insert_axes(&[0, 1, 1, 1, 3])
            .to_ndarray()
            .unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![1, 2, 1, 1, 1, 3, 4, 1], seq(24)).unwrap()
        );
    }

    #[test]
    fn full_read_identity_empty_axes() {
        let got: ArrayD<i32> = make2d(seq(12), 3, 4).insert_axes(&[]).to_ndarray().unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![3, 4], seq(12)).unwrap());
    }

    // -----------------------------------------------------------------------
    // Sub-region reads
    // -----------------------------------------------------------------------

    #[test]
    fn sub_read_inserted_dim_is_stripped() {
        // [1, 6]: read [0..1, 2..5] → same as reading [2..5] from the 1D inner
        let got: ArrayD<i32> = make1d(seq(6), 6)
            .insert_axes(&[0])
            .to_ndarray_sub(&[0..1, 2..5], &ReadContext::default())
            .unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![1, 3], vec![2, 3, 4]).unwrap()
        );
    }

    #[test]
    fn sub_read_2d_with_inserted_middle() {
        // [3, 1, 4]: read rows 1..3, inserted dim 0..1, cols 0..2
        let got: ArrayD<i32> = make2d(seq(12), 3, 4)
            .insert_axes(&[1])
            .to_ndarray_sub(&[1..3, 0..1, 0..2], &ReadContext::default())
            .unwrap();
        // row1=[4,5], row2=[8,9]
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![2, 1, 2], vec![4, 5, 8, 9]).unwrap()
        );
    }

    // -----------------------------------------------------------------------
    // Error cases
    // -----------------------------------------------------------------------

    #[test]
    fn error_axis_out_of_bounds() {
        let a = make1d(seq(4), 4);
        // orig_ndim=1, valid gaps are 0..=1; axis 2 is out of bounds
        assert!(super::InsertAxes::new(a, &[2]).is_err());
    }
}
