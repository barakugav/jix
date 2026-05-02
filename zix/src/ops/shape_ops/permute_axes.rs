use std::ops::Range;

use crate::codec::ReadContext;
use crate::dtype::Dtype;
use crate::error::{check_get_buffer_size, check_get_range, ensure, Result};
use crate::storage::{ArrayStorage, ArrayStorageSpec, BlocksLayout};
use crate::util::{default_strides, dim_arr, nd_copy, DimArray};
use crate::Array;

/// Reorders the axes of an array, returned by [`Array::permute_axes`](crate::Array::permute_axes).
///
/// The `i`-th output axis corresponds to axis `axes[i]` of the input — identical to the
/// convention used by NumPy's `numpy.transpose`. No data is copied at construction time;
/// elements are rearranged on demand when the result is read.
///
/// `axes` must be a permutation of `0..ndim`: correct length, all values in range, no
/// duplicates.
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// # Examples
///
/// ```
/// use zix::{Array, ArrayParams};
/// use ndarray::array;
///
/// // 2-D transpose: [2, 3] → [3, 2]
/// let a = Array::compact_array(&array![[1i32, 2, 3], [4, 5, 6]])?;
/// let t = a.permute_axes(&[1, 0]);
/// assert_eq!(t.shape(), &[3, 2]);
/// let result = t.to_ndarray::<i32>()?;
/// assert_eq!(result[[0, 0]], 1);
/// assert_eq!(result[[0, 1]], 4);
/// assert_eq!(result[[2, 1]], 6);
///
/// // 3-D cyclic permutation [2, 3, 4] → [4, 2, 3]
/// let b = ndarray::Array::from_shape_fn((2, 3, 4), |(i, j, k)| (i * 12 + j * 4 + k) as i32);
/// let zb = Array::compact_array(&b)?;
/// let p = zb.permute_axes(&[2, 0, 1]);
/// assert_eq!(p.shape(), &[4, 2, 3]);
/// # Ok::<(), zix::Error>(())
/// ```
pub struct PermuteAxes<S> {
    array: Array<S>,
    /// `axes[i]` = index of the input dimension that maps to output dimension `i`.
    axes: DimArray<usize>,
    /// `inv_axes[d]` = index of the output dimension that maps from input dimension `d`.
    inv_axes: DimArray<usize>,

    dtype: Dtype,
    shape: DimArray<u64>,
    blocks_layout: BlocksLayout,
}

impl<S: ArrayStorage> PermuteAxes<S> {
    /// Constructs a `PermuteAxes` storage. See [`PermuteAxes`] for semantics and examples.
    pub fn new(array: Array<S>, axes: &[usize]) -> Result<Self> {
        let ndim = array.shape().len();
        ensure!(
            axes.len() == ndim,
            InvalidShapeOperation,
            "axes length {} does not match array ndim {ndim}",
            axes.len()
        );
        let axes: DimArray<_> = axes.try_into().unwrap();
        let mut seen = dim_arr(ndim, |_| false);
        for &ax in axes.iter() {
            ensure!(
                ax < ndim,
                InvalidShapeOperation,
                "axis {ax} out of bounds for array of ndim {ndim}"
            );
            ensure!(
                !seen[ax],
                InvalidShapeOperation,
                "duplicate axis {ax} in axes {axes:?}"
            );
            seen[ax] = true;
        }

        let mut inv_axes = dim_arr(ndim, |_| 0);
        for (i, &ax) in axes.iter().enumerate() {
            inv_axes[ax] = i;
        }

        let input_shape = array.shape();
        let shape = dim_arr(ndim, |i| input_shape[axes[i]]);

        let mut b_layout = array.blocks_layout().clone();
        b_layout.block_shape_hint = dim_arr(ndim, |i| b_layout.block_shape_hint[axes[i]]);
        b_layout.block_shape_tag = dim_arr(ndim, |i| b_layout.block_shape_tag[axes[i]]);
        b_layout.preferred_read_shape = dim_arr(ndim, |i| b_layout.preferred_read_shape[axes[i]]);

        let dtype = array.dtype().clone();
        Ok(Self {
            dtype,
            shape,
            blocks_layout: b_layout,
            array,
            axes,
            inv_axes,
        })
    }
}

impl<S: ArrayStorage> ArrayStorage for PermuteAxes<S> {
    fn read_data(&self, index: &[Range<u64>], buf: &mut [u8], context: &ReadContext) -> Result<()> {
        check_get_range(self.shape(), index)?;
        let nitems = check_get_buffer_size(index, &self.dtype, buf)?;

        let ndim = self.axes.len();
        let itemsize = self.dtype.itemsize() as usize;

        // Build the index into the underlying (un-permuted) storage.
        // Output dim i reads from input dim axes[i], so input dim d = inv_axes[output dim] needs
        // the range that was requested for output dim inv_axes[d].
        let input_index = dim_arr(ndim, |d| index[self.inv_axes[d]].clone());

        // Read the underlying data contiguously into tmp_buf.
        // tmp_buf is laid out C-contiguous over sub_shape_in (input dim order).
        let sub_shape_in = dim_arr(ndim, |d| {
            (input_index[d].end - input_index[d].start) as usize
        });
        let n_bytes = nitems * itemsize;
        let mut tmp_buf = context.tmp_buf(n_bytes, self.dtype.alignment());
        let tmp_buf = tmp_buf.as_mut_slice();
        self.array
            .storage
            .read_data(&input_index, tmp_buf, context)?;

        // Strides in tmp_buf (C-contiguous over input dims).
        let src_strides_in = default_strides(&sub_shape_in, itemsize);

        // The output buffer is C-contiguous over sub_shape_out (output dim order).
        let sub_shape_out = dim_arr(ndim, |i| (index[i].end - index[i].start) as usize);
        let dst_strides = default_strides(&sub_shape_out, itemsize);

        // When we advance along output dim i, we're advancing along input dim axes[i] in tmp_buf.
        let src_strides_out = dim_arr(ndim, |i| src_strides_in[self.axes[i]]);

        unsafe {
            nd_copy(
                tmp_buf.as_ptr(),
                buf.as_mut_ptr(),
                &sub_shape_out,
                &src_strides_out,
                &dst_strides,
                itemsize,
            )
        };
        Ok(())
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
    use ndarray::array;
    use proptest::prelude::*;

    use crate::array::Array;
    use crate::storage::Compact;
    use crate::util::{shape_strategy, ScalarStrategy};

    // 2D i32: transpose (axes=[1,0])
    #[test]
    fn test_i32_2d_transpose() {
        let a = array![[1i32, 2, 3], [4, 5, 6]];
        let za = Array::compact_array(&a).unwrap();
        let actual = za.permute_axes(&[1, 0]).to_ndarray::<i32>().unwrap();
        let expected = a
            .view()
            .permuted_axes([1, 0])
            .into_dyn()
            .as_standard_layout()
            .into_owned();
        assert_eq!(actual, expected);
    }

    // 2D f32: transpose
    #[test]
    fn test_f32_2d_transpose() {
        let a = array![[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let za = Array::compact_array(&a).unwrap();
        let actual = za.permute_axes(&[1, 0]).to_ndarray::<f32>().unwrap();
        let expected = a
            .view()
            .permuted_axes([1, 0])
            .into_dyn()
            .as_standard_layout()
            .into_owned();
        assert_eq!(actual, expected);
    }

    // 3D i32: axes=[2,0,1]
    #[test]
    fn test_i32_3d_axes_2_0_1() {
        let a = ndarray::Array::from_shape_fn((2, 3, 4), |(i, j, k)| (i * 12 + j * 4 + k) as i32);
        let za = Array::compact_array(&a).unwrap();
        let actual = za.permute_axes(&[2, 0, 1]).to_ndarray::<i32>().unwrap();
        let expected = a
            .view()
            .permuted_axes([2, 0, 1])
            .into_dyn()
            .as_standard_layout()
            .into_owned();
        assert_eq!(actual, expected);
    }

    // 3D i32: swap only last two dims axes=[0,2,1]
    #[test]
    fn test_i32_3d_axes_0_2_1() {
        let a = ndarray::Array::from_shape_fn((2, 3, 4), |(i, j, k)| (i * 12 + j * 4 + k) as i32);
        let za = Array::compact_array(&a).unwrap();
        let actual = za.permute_axes(&[0, 2, 1]).to_ndarray::<i32>().unwrap();
        let expected = a
            .view()
            .permuted_axes([0, 2, 1])
            .into_dyn()
            .as_standard_layout()
            .into_owned();
        assert_eq!(actual, expected);
    }

    // 3D i32: identity permutation
    #[test]
    fn test_i32_3d_identity() {
        let a = ndarray::Array::from_shape_fn((2, 3, 4), |(i, j, k)| (i * 12 + j * 4 + k) as i32);
        let za = Array::compact_array(&a).unwrap();
        let actual = za.permute_axes(&[0, 1, 2]).to_ndarray::<i32>().unwrap();
        assert_eq!(actual, a.into_dyn());
    }

    // Panic: axes length does not match ndim
    #[test]
    #[should_panic]
    fn test_wrong_axes_length_panics() {
        let a = array![[1i32, 2], [3, 4]];
        let za = Array::compact_array(&a).unwrap();
        let _ = za.permute_axes(&[0, 1, 2]);
    }

    // Panic: axis out of bounds
    #[test]
    #[should_panic]
    fn test_axis_out_of_bounds_panics() {
        let a = array![[1i32, 2], [3, 4]];
        let za = Array::compact_array(&a).unwrap();
        let _ = za.permute_axes(&[0, 5]);
    }

    // Panic: duplicate axis
    #[test]
    #[should_panic]
    fn test_duplicate_axis_panics() {
        let a = array![[1i32, 2], [3, 4]];
        let za = Array::compact_array(&a).unwrap();
        let _ = za.permute_axes(&[0, 0]);
    }

    // -----------------------------------------------------------------------
    // Proptest: arbitrary ndim, arbitrary permutation, verified against ndarray
    // -----------------------------------------------------------------------

    fn permute_axes_strategy<T>(
    ) -> impl Strategy<Value = (ndarray::ArrayD<T>, Array<Compact>, Vec<usize>)>
    where
        T: ScalarStrategy,
    {
        shape_strategy()
            .prop_flat_map(|shape| {
                let ndim = shape.len();
                let perm = Just((0..ndim).collect::<Vec<_>>()).prop_shuffle();
                (Just(shape), perm)
            })
            .prop_flat_map(|(shape, perm)| {
                let array_strat =
                    crate::util::carray_strategy_from_shape::<T>(Just(shape), T::any_strategy());
                (array_strat, Just(perm))
            })
            .prop_map(|((nd, za), perm)| (nd, za, perm))
    }

    proptest::proptest! {
        #[test]
        fn proptest_permute_axes((nd, za, perm) in permute_axes_strategy::<i32>()) {
            let expected = nd
                .view()
                .permuted_axes(perm.clone())
                .as_standard_layout()
                .into_owned();
            crate::util::assert_array_matches(&za.permute_axes(&perm), &expected);
        }
    }
}
