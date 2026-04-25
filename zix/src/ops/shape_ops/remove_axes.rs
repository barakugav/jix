use std::ops::Range;

use crate::codec::ReadContext;
use crate::dtype::Dtype;
use crate::error::{check_get_range, ensure, Result};
use crate::storage::{ArrayStorage, ArrayStorageSpec, BlocksLayout};
use crate::util::DimArray;
use crate::Array;

/// Removes length-1 dimensions from an array's shape,
/// returned by [`Array::remove_axes`](crate::Array::remove_axes).
///
/// `axes` is a set of axis indices in the *input* shape (0-based). Each named dimension must have
/// length exactly 1 and is dropped from the output shape. Duplicate axis indices are not allowed.
/// The order of values in `axes` does not matter. Valid axis indices are `0..input_ndim`.
///
/// Output dtype equals the input dtype.
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// # Examples
///
/// ```text
/// [1, N]          axes: [0]       → [N]
/// [N, 1]          axes: [1]       → [N]
/// [N, 1, M]       axes: [1]       → [N, M]
/// [1, N, 1, M, 1] axes: [0, 2, 4] → [N, M]
/// ```
///
/// ```
/// use zix::{Array, ArrayParams};
/// // [1, 3] → [3]: remove the leading size-1 dim
/// let a = ndarray::array![[1i32, 2, 3]];
/// let za = Array::from_ndarray(&a, ArrayParams::new())?;
/// let result = za.remove_axes(&[0]).to_ndarray::<i32>()?;
/// assert_eq!(result.shape(), &[3]);
/// assert_eq!(result.as_slice().unwrap(), &[1, 2, 3]);
///
/// // [1, 2, 1] → [2]: remove both size-1 dims at once
/// let b = ndarray::array![[[10i32], [20]]]; // shape [1, 2, 1]
/// let zb = Array::from_ndarray(&b, ArrayParams::new())?;
/// assert_eq!(zb.remove_axes(&[0, 2]).shape(), &[2]);
/// # Ok::<(), zix::error::Error>(())
/// ```
pub struct RemoveAxes<S> {
    array: Array<S>,
    /// `is_removed[input_dim]` is `true` for every input dimension that was removed.
    is_removed: DimArray<bool>,

    dtype: Dtype,
    shape: DimArray<u64>,
    blocks_layout: BlocksLayout,
}

impl<S: ArrayStorage> RemoveAxes<S> {
    pub fn new(array: Array<S>, axes: &[usize]) -> Result<Self> {
        let input_ndim = array.shape().len();

        // Validate axis indices and check for duplicates.
        let mut seen = DimArray::<bool>::from_iter(std::iter::repeat_n(false, input_ndim));
        for &ax in axes {
            ensure!(
                ax < input_ndim,
                InvalidShapeOperation,
                "axis {ax} out of bounds for array of ndim {input_ndim} \
                 (axis indices must be in 0..{input_ndim})"
            );
            ensure!(!seen[ax], InvalidShapeOperation, "duplicate axis {ax}");
            seen[ax] = true;

            ensure!(
                array.shape()[ax] == 1,
                InvalidShapeOperation,
                "cannot remove axis {ax} with size {} (only size-1 axes can be removed)",
                array.shape()[ax]
            );
        }

        // Build is_removed, shape, and blocks_layout by walking input dims.
        let mut is_removed = DimArray::new();
        let mut shape = DimArray::new();

        let inner_layout = array.blocks_layout();
        let mut hint = DimArray::new();
        let mut tag = DimArray::new();
        let mut preferred = DimArray::new();

        for input_dim in 0..input_ndim {
            let removed = seen[input_dim];
            is_removed.push(removed);
            if !removed {
                shape.push(array.shape()[input_dim]);
                hint.push(inner_layout.block_shape_hint[input_dim]);
                tag.push(inner_layout.block_shape_tag[input_dim]);
                preferred.push(inner_layout.preferred_read_shape[input_dim]);
            }
        }

        let mut b_layout = inner_layout.clone();
        b_layout.block_shape_hint = hint;
        b_layout.block_shape_tag = tag;
        b_layout.preferred_read_shape = preferred;

        let dtype = array.dtype().clone();
        Ok(Self {
            array,
            is_removed,
            dtype,
            shape,
            blocks_layout: b_layout,
        })
    }
}

impl<S: ArrayStorage> ArrayStorage for RemoveAxes<S> {
    fn read_data(&self, index: &[Range<u64>], buf: &mut [u8], context: &ReadContext) -> Result<()> {
        check_get_range(self.shape(), index)?;

        // Removed dimensions have size 1 and do not affect the element sequence.
        // Re-insert them as `0..1` ranges and forward the full index to the inner storage.
        let mut output_dim = 0usize;
        let inner_index: DimArray<_> = self
            .is_removed
            .iter()
            .map(|&removed| {
                if removed {
                    0..1
                } else {
                    let range = index[output_dim].clone();
                    output_dim += 1;
                    range
                }
            })
            .collect();
        self.array.storage.read_data(&inner_index, buf, context)
    }

    fn shape(&self) -> &[u64] {
        &self.shape
    }
    fn dtype(&self) -> &Dtype {
        &self.dtype
    }
    fn spec(&self) -> ArrayStorageSpec<'_> {
        ArrayStorageSpec {
            blocks_layout: &self.blocks_layout,
            ..self.array.storage.spec()
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::ArrayD;

    use crate::array::Array;
    use crate::codec::ReadContext;
    use crate::util::arr_params;

    fn make1d(vals: Vec<i32>, block_size: usize) -> Array<crate::storage::Compact> {
        let nd = ndarray::ArrayD::from_shape_vec(vec![vals.len()], vals).unwrap();
        Array::from_ndarray(&nd, arr_params(&[block_size])).unwrap()
    }

    fn make2d(vals: Vec<i32>, rows: usize, cols: usize) -> Array<crate::storage::Compact> {
        let nd = ndarray::ArrayD::from_shape_vec(vec![rows, cols], vals).unwrap();
        Array::from_ndarray(&nd, arr_params(&[rows, cols])).unwrap()
    }

    fn make3d(vals: Vec<i32>, d0: usize, d1: usize, d2: usize) -> Array<crate::storage::Compact> {
        let nd = ndarray::ArrayD::from_shape_vec(vec![d0, d1, d2], vals).unwrap();
        Array::from_ndarray(&nd, arr_params(&[d0, d1, d2])).unwrap()
    }

    fn seq(n: usize) -> Vec<i32> {
        (0..n as i32).collect()
    }

    // -----------------------------------------------------------------------
    // Shape metadata
    // -----------------------------------------------------------------------

    #[test]
    fn shape_remove_leading() {
        // [1, 6] remove axis 0 → [6]
        let a = make1d(seq(6), 6).insert_axes(&[0]);
        assert_eq!(a.remove_axes(&[0]).shape(), &[6]);
    }

    #[test]
    fn shape_remove_trailing() {
        // [6, 1] remove axis 1 → [6]
        let a = make1d(seq(6), 6).insert_axes(&[1]);
        assert_eq!(a.remove_axes(&[1]).shape(), &[6]);
    }

    #[test]
    fn shape_remove_middle() {
        // [3, 1, 4] remove axis 1 → [3, 4]
        let a = make2d(seq(12), 3, 4).insert_axes(&[1]);
        assert_eq!(a.remove_axes(&[1]).shape(), &[3, 4]);
    }

    #[test]
    fn shape_remove_multiple() {
        // [1, 2, 1, 3, 1] remove axes [0, 2, 4] → [2, 3]
        let a = make2d(seq(6), 2, 3).insert_axes(&[0, 1, 2]);
        assert_eq!(a.remove_axes(&[0, 2, 4]).shape(), &[2, 3]);
    }

    #[test]
    fn shape_remove_empty_axes_is_identity() {
        assert_eq!(make2d(seq(12), 3, 4).remove_axes(&[]).shape(), &[3, 4]);
    }

    #[test]
    fn shape_remove_unsorted_axes_same_result() {
        let a1 = make3d(seq(6), 1, 2, 3).remove_axes(&[0]);
        let a2 = make3d(seq(6), 1, 2, 3).remove_axes(&[0]);
        assert_eq!(a1.shape(), a2.shape());
    }

    // -----------------------------------------------------------------------
    // Full reads
    // -----------------------------------------------------------------------

    #[test]
    fn full_read_remove_leading() {
        let got: ArrayD<i32> = make1d(seq(6), 6)
            .insert_axes(&[0])
            .remove_axes(&[0])
            .to_ndarray()
            .unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![6], seq(6)).unwrap());
    }

    #[test]
    fn full_read_remove_trailing() {
        let got: ArrayD<i32> = make1d(seq(6), 6)
            .insert_axes(&[1])
            .remove_axes(&[1])
            .to_ndarray()
            .unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![6], seq(6)).unwrap());
    }

    #[test]
    fn full_read_remove_middle() {
        let got: ArrayD<i32> = make2d(seq(12), 3, 4)
            .insert_axes(&[1])
            .remove_axes(&[1])
            .to_ndarray()
            .unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![3, 4], seq(12)).unwrap());
    }

    #[test]
    fn full_read_remove_multiple() {
        let got: ArrayD<i32> = make2d(seq(6), 2, 3)
            .insert_axes(&[0, 1, 2])
            .remove_axes(&[0, 2, 4])
            .to_ndarray()
            .unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![2, 3], seq(6)).unwrap());
    }

    #[test]
    fn full_read_identity_empty_axes() {
        let got: ArrayD<i32> = make2d(seq(12), 3, 4).remove_axes(&[]).to_ndarray().unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![3, 4], seq(12)).unwrap());
    }

    // -----------------------------------------------------------------------
    // Sub-region reads
    // -----------------------------------------------------------------------

    #[test]
    fn sub_read_after_remove_leading() {
        // [1, 6] → remove axis 0 → [6]; read elements 2..5
        let got: ArrayD<i32> = make1d(seq(6), 6)
            .insert_axes(&[0])
            .remove_axes(&[0])
            .to_ndarray_sub(&[2..5], &ReadContext::default())
            .unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![3], vec![2, 3, 4]).unwrap());
    }

    #[test]
    fn sub_read_after_remove_middle() {
        // [3, 1, 4] → remove axis 1 → [3, 4]; read rows 1..3, cols 0..2
        let got: ArrayD<i32> = make2d(seq(12), 3, 4)
            .insert_axes(&[1])
            .remove_axes(&[1])
            .to_ndarray_sub(&[1..3, 0..2], &ReadContext::default())
            .unwrap();
        // row1=[4,5], row2=[8,9]
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![2, 2], vec![4, 5, 8, 9]).unwrap()
        );
    }

    // -----------------------------------------------------------------------
    // Error cases
    // -----------------------------------------------------------------------

    #[test]
    fn error_axis_out_of_bounds() {
        let a = make2d(seq(4), 2, 2);
        // ndim=2, valid axes are 0..2; axis 3 is out of bounds
        assert!(super::RemoveAxes::new(a, &[3]).is_err());
    }

    #[test]
    fn error_axis_not_size_one() {
        let a = make2d(seq(12), 3, 4);
        // axis 0 has size 3, cannot remove
        assert!(super::RemoveAxes::new(a, &[0]).is_err());
    }

    #[test]
    fn error_duplicate_axis() {
        let a = make3d(seq(6), 1, 2, 3);
        // axis 0 appears twice
        assert!(super::RemoveAxes::new(a, &[0, 0]).is_err());
    }
}
