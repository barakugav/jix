use std::io;
use std::ops::Range;

use crate::array::{Array, BlocksLayout};
use crate::codec::ReadContext;
use crate::dtype::Dtype;
use crate::iter::NdIter;
use crate::iter::strides::{NdIterExtStridesPtr, NdIterExtStridesPtrMut};
use crate::storage::{ArrayStorage, Ref};
use crate::util::{DimArray, default_strides, dim_arr};

impl<S> Array<S>
where
    S: ArrayStorage,
{
    /// Return a lazy view of the array with its axes reordered.
    ///
    /// The i-th axis of the returned array corresponds to the axis numbered `axes[i]` of the
    /// input — identical to the convention used by NumPy's `numpy.permute_dims` (also known as
    /// `numpy.transpose`).  No data is copied at construction time; elements are read and
    /// rearranged on demand when the result is materialised.
    ///
    /// # Arguments
    ///
    /// * `axes` — a permutation of `0..ndim`.  Must satisfy:
    ///   - `axes.len() == self.ndim()`
    ///   - every value is in `0..self.ndim()`
    ///   - no value appears more than once
    ///
    /// # Panics
    ///
    /// Panics if `axes` is not a valid permutation of `0..ndim`:
    ///
    /// * `axes.len() != self.ndim()`
    /// * any axis value is ≥ `self.ndim()`
    /// * any axis value is repeated
    ///
    /// # Examples
    ///
    /// ```
    /// use zix::array::Array;
    ///
    /// // 2-D transpose — equivalent to np.permute_dims(a, [1, 0])
    /// let a = Array::from_ndarray(&ndarray::array![[1i32, 2, 3], [4, 5, 6]].view().into_dyn(), &[2, 3]).unwrap();
    /// let t = a.permute_dims(&[1, 0]);
    /// assert_eq!(t.shape(), &[3, 2]);
    /// // t[i, j] == a[j, i]
    ///
    /// // 3-D cyclic permutation — equivalent to np.permute_dims(a, [2, 0, 1])
    /// // output axis 0 ← input axis 2
    /// // output axis 1 ← input axis 0
    /// // output axis 2 ← input axis 1
    /// let a = Array::from_ndarray(&ndarray::Array::from_shape_fn((2,3,4), |(i,j,k)| (i*12+j*4+k) as i32).view().into_dyn(), &[2,3,4]).unwrap();
    /// let p = a.permute_dims(&[2, 0, 1]);
    /// assert_eq!(p.shape(), &[4, 2, 3]);
    /// ```
    #[track_caller]
    pub fn permute_dims(&self, axes: &[usize]) -> Array<PermuteDims<Ref<'_, S>>> {
        Array::from_storage(PermuteDims::new(Ref(&self.storage), axes).unwrap())
    }
}

/// Lazy storage type returned by [`permute_dims`](Array::permute_dims).
///
/// Reads the underlying array with permuted axis order. See [`Array::permute_dims`] for the full
/// description.
pub struct PermuteDims<S> {
    inner: S,
    /// `axes[i]` = index of the input dimension that maps to output dimension `i`.
    axes: DimArray<usize>,
    /// `inv_axes[d]` = index of the output dimension that maps from input dimension `d`.
    inv_axes: DimArray<usize>,

    dtype: Dtype,
    shape: DimArray<usize>,
    blocks_layout: BlocksLayout,
}

impl<S: ArrayStorage> PermuteDims<S> {
    pub(crate) fn new(inner: S, axes: &[usize]) -> io::Result<Self> {
        let ndim = inner.shape().len();
        if axes.len() != ndim {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "axes length {} does not match array ndim {}",
                    axes.len(),
                    ndim
                ),
            ));
        }
        let axes: DimArray<usize> = axes.try_into().unwrap();
        let mut seen = dim_arr(ndim, |_| false);
        for &ax in axes.iter() {
            if ax >= ndim {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("axis {ax} out of bounds for array of ndim {ndim}"),
                ));
            }
            if seen[ax] {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("duplicate axis {ax}"),
                ));
            }
            seen[ax] = true;
        }

        let mut inv_axes = dim_arr(ndim, |_| 0);
        for (i, &ax) in axes.iter().enumerate() {
            inv_axes[ax] = i;
        }

        let input_shape = inner.shape();
        let shape = dim_arr(ndim, |i| input_shape[axes[i]]);

        let input_block_shape = &inner.blocks_layout().block_shape;
        let block_shape = dim_arr(ndim, |i| input_block_shape[axes[i]]);

        let dtype = inner.dtype().clone();
        Ok(Self {
            dtype,
            shape,
            blocks_layout: BlocksLayout::new(&block_shape),
            inner,
            axes,
            inv_axes,
        })
    }
}

impl<S: ArrayStorage> ArrayStorage for PermuteDims<S> {
    fn dtype(&self) -> &Dtype {
        &self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn blocks_layout(&self) -> &BlocksLayout {
        &self.blocks_layout
    }

    fn read_data(
        &self,
        index: &[Range<usize>],
        buf: &mut [u8],
        context: &ReadContext,
    ) -> io::Result<()> {
        let ndim = self.axes.len();
        let itemsize = self.dtype.itemsize() as usize;

        // Build the index into the underlying (un-permuted) storage.
        // Output dim i reads from input dim axes[i], so input dim d = inv_axes[output dim] needs
        // the range that was requested for output dim inv_axes[d].
        let input_index: DimArray<Range<usize>> =
            dim_arr(ndim, |d| index[self.inv_axes[d]].clone());

        // Read the underlying data contiguously into tmp_buf.
        // tmp_buf is laid out C-contiguous over sub_shape_in (input dim order).
        let sub_shape_in = dim_arr(ndim, |d| input_index[d].len());
        let n_bytes = sub_shape_in.iter().product::<usize>() * itemsize;
        let mut tmp_buf = context.tmp_buf(n_bytes, self.dtype.alignment());
        let tmp_buf = tmp_buf.as_mut_slice();
        self.inner.read_data(&input_index, tmp_buf, context)?;

        // Strides in tmp_buf (C-contiguous over input dims).
        let src_strides_in = default_strides(&sub_shape_in, itemsize);

        // The output buffer is C-contiguous over sub_shape_out (output dim order).
        let sub_shape_out = dim_arr(ndim, |i| index[i].len());
        let dst_strides = default_strides(&sub_shape_out, itemsize);

        // When we advance along output dim i, we're advancing along input dim axes[i] in tmp_buf.
        let src_strides_out = dim_arr(ndim, |i| src_strides_in[self.axes[i]]);

        let mut iter = NdIter::new(
            &sub_shape_out,
            (
                NdIterExtStridesPtr::new(&src_strides_out, tmp_buf.as_ptr()),
                NdIterExtStridesPtrMut::new(&dst_strides, buf.as_mut_ptr()),
            ),
        );
        while let Some((_, (src_ptr, dst_ptr))) = iter.next() {
            unsafe {
                std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, itemsize);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::array::Array;

    // 2D i32: transpose (axes=[1,0])
    #[test]
    fn test_i32_2d_transpose() {
        let a = ndarray::array![[1i32, 2, 3], [4, 5, 6]];
        let za = Array::from_ndarray(&a.view().into_dyn(), &[2, 3]).unwrap();
        let actual = za.permute_dims(&[1, 0]).data().to_ndarray::<i32>().unwrap();
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
        let a = ndarray::array![[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let za = Array::from_ndarray(&a.view().into_dyn(), &[3, 2]).unwrap();
        let actual = za.permute_dims(&[1, 0]).data().to_ndarray::<f32>().unwrap();
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
        let za = Array::from_ndarray(&a.view().into_dyn(), &[2, 3, 4]).unwrap();
        let actual = za
            .permute_dims(&[2, 0, 1])
            .data()
            .to_ndarray::<i32>()
            .unwrap();
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
        let za = Array::from_ndarray(&a.view().into_dyn(), &[2, 3, 4]).unwrap();
        let actual = za
            .permute_dims(&[0, 2, 1])
            .data()
            .to_ndarray::<i32>()
            .unwrap();
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
        let za = Array::from_ndarray(&a.view().into_dyn(), &[2, 3, 4]).unwrap();
        let actual = za
            .permute_dims(&[0, 1, 2])
            .data()
            .to_ndarray::<i32>()
            .unwrap();
        assert_eq!(actual, a.into_dyn());
    }

    // Panic: axes length does not match ndim
    #[test]
    #[should_panic]
    fn test_wrong_axes_length_panics() {
        let a = ndarray::array![[1i32, 2], [3, 4]];
        let za = Array::from_ndarray(&a.view().into_dyn(), &[2, 2]).unwrap();
        let _ = za.permute_dims(&[0, 1, 2]);
    }

    // Panic: axis out of bounds
    #[test]
    #[should_panic]
    fn test_axis_out_of_bounds_panics() {
        let a = ndarray::array![[1i32, 2], [3, 4]];
        let za = Array::from_ndarray(&a.view().into_dyn(), &[2, 2]).unwrap();
        let _ = za.permute_dims(&[0, 5]);
    }

    // Panic: duplicate axis
    #[test]
    #[should_panic]
    fn test_duplicate_axis_panics() {
        let a = ndarray::array![[1i32, 2], [3, 4]];
        let za = Array::from_ndarray(&a.view().into_dyn(), &[2, 2]).unwrap();
        let _ = za.permute_dims(&[0, 0]);
    }
}
