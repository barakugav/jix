use std::ops::Range;

use crate::codec::ReadContext;
use crate::dtype::Dtype;
use crate::error::{check_get_buffer_size, check_get_range, check_ndim, ensure, Result};
use crate::storage::params::ArraySpecDynamic;
use crate::storage::{ArraySpec, OutBuf};
use crate::util::{default_strides, dim_arr, nd_copy, DimArray};
use crate::{Array, ArrayStorage, Dimension};

/// Reorders the axes of an array, returned by [`Array::permute_axes`](crate::Array::permute_axes).
///
/// The `i`-th output axis corresponds to axis `axes[i]` of the input - identical to the
/// convention used by NumPy's `numpy.transpose`. No data is copied at construction time;
/// elements are rearranged on demand when the result is read.
///
/// `axes` must be a permutation of `0..ndim`: correct length, all values in range, no
/// duplicates.
///
/// Output dtype equals the input dtype. `PermuteAxes<S>` carries `type Dimension = S::Dimension`
/// - permutation does not change the number of axes so the dimension type is preserved unchanged.
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// # Examples
///
/// ```
/// use jix::Array;
/// use ndarray::array;
///
/// // 2-D transpose: [2, 3] -> [3, 2]
/// let a = Array::compact_ndarray(&array![[1i32, 2, 3], [4, 5, 6]])?;
/// let t = a.permute_axes(&[1, 0]);
/// assert_eq!(t.shape(), &[3, 2]);
/// let result = t.to_ndarray()?;
/// assert_eq!(result[[0, 0]], 1);
/// assert_eq!(result[[0, 1]], 4);
/// assert_eq!(result[[2, 1]], 6);
///
/// // 3-D cyclic permutation [2, 3, 4] -> [4, 2, 3]
/// let b = ndarray::Array::from_shape_fn((2, 3, 4), |(i, j, k)| (i * 12 + j * 4 + k) as i32);
/// let zb = Array::compact_ndarray(&b)?;
/// let p = zb.permute_axes(&[2, 0, 1]);
/// assert_eq!(p.shape(), &[4, 2, 3]);
/// # Ok::<(), jix::Error>(())
/// ```
pub struct PermuteAxes<S: ArrayStorage> {
    array: S,
    /// `axes[i]` = index of the input dimension that maps to output dimension `i`.
    axes: DimArray<u8>,
    /// `inv_axes[d]` = index of the output dimension that maps from input dimension `d`.
    inv_axes: DimArray<u8>,

    shape: S::Dimension,
    spec: ArraySpecDynamic,
}

impl<S: ArrayStorage> PermuteAxes<S> {
    /// Constructs a [`PermuteAxes`] storage. See the struct docs for semantics and examples.
    pub fn new(array: S, axes: &[usize]) -> Result<Self> {
        let ndim = array.shape().len();
        ensure!(
            axes.len() == ndim,
            InvalidShapeOperation,
            "axes length {} does not match array ndim {ndim}",
            axes.len()
        );
        let axes = DimArray::from_slice(axes).unwrap();
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
        let shape = S::Dimension::from_fn(ndim, |i| input_shape[axes[i]]);

        let inner_spec = array.spec();
        let block_shape = dim_arr(ndim, |i| inner_spec.block_shape()[axes[i]]);
        let block_shape_tag = dim_arr(ndim, |i| inner_spec.block_shape_tag()[axes[i]]);
        let spec = ArraySpecDynamic {
            block_shape,
            block_shape_tag,
        };
        Ok(Self {
            shape,
            spec,
            array,
            axes: dim_arr(ndim, |i| axes[i] as u8),
            inv_axes: dim_arr(ndim, |i| inv_axes[i] as u8),
        })
    }

    /// Constructs an array with [`PermuteAxes`] storage. See the storage struct docs for semantics and examples.
    pub fn new_array(array: Array<S>, axes: &[usize]) -> Result<Array<Self>> {
        Self::new(array.into_storage(), axes).map(Array::from_storage)
    }
}

impl<S: ArrayStorage> ArrayStorage for PermuteAxes<S> {
    type ElementType = S::ElementType;
    type Dimension = S::Dimension;

    #[inline]
    fn read_data(
        &self,
        index: &[Range<u64>],
        buf: &mut OutBuf,
        context: &ReadContext,
    ) -> Result<()> {
        let dtype = self.dtype();
        check_get_range(self.shape(), index)?;

        let ndim = self.axes.len();
        let itemsize = dtype.itemsize() as usize;

        // Build the index into the underlying (un-permuted) storage.
        // Output dim i reads from input dim axes[i], so input dim d = inv_axes[output dim] needs
        // the range that was requested for output dim inv_axes[d].
        let input_index = dim_arr(ndim, |d| index[self.inv_axes[d] as usize].clone());

        // Read the underlying data contiguously into tmp_buf.
        // tmp_buf is laid out C-contiguous over sub_shape_in (input dim order).
        let sub_shape_in = dim_arr(ndim, |d| {
            (input_index[d].end - input_index[d].start) as usize
        });
        let mut tmp_buf = OutBuf::new_lazy(context);
        self.array.read_data(&input_index, &mut tmp_buf, context)?;
        let tmp_buf = tmp_buf.as_slice().unwrap();
        let buf = buf.get_mut(index, dtype);
        check_get_buffer_size(index, dtype, buf)?;

        // Strides in tmp_buf (C-contiguous over input dims).
        let src_strides_in = default_strides(&sub_shape_in, itemsize);
        // When we advance along output dim i, we're advancing along input dim axes[i] in tmp_buf.
        let src_strides_out = dim_arr(ndim, |i| src_strides_in[self.axes[i] as usize]);

        // The output buffer is C-contiguous over sub_shape_out (output dim order).
        let sub_shape_out = S::Dimension::from_fn(ndim, |i| index[i].end - index[i].start);
        let dst_strides = default_strides(sub_shape_out.as_slice(), itemsize as u64);

        unsafe {
            nd_copy(
                tmp_buf.as_ptr(),
                buf.as_mut_ptr(),
                sub_shape_out,
                &src_strides_out,
                &dst_strides,
                itemsize,
            )
        };
        Ok(())
    }

    #[inline(always)]
    fn shape(&self) -> &[u64] {
        self.shape.as_slice()
    }
    #[inline(always)]
    fn dtype(&self) -> &Dtype {
        self.array.dtype()
    }
    #[inline]
    fn spec(&self) -> ArraySpec<'_> {
        self.array.spec().with_dynamic_spec(&self.spec)
    }

    type DimensionChange<NewD: crate::Dimension> = PermuteAxes<S::DimensionChange<NewD>>;
    #[inline]
    fn dimension_change<NewD: crate::Dimension>(
        self,
    ) -> crate::error::Result<Self::DimensionChange<NewD>> {
        check_ndim::<NewD>(self.shape().len())?;
        let shape = NewD::from_slice(self.shape());
        let array = self.array.dimension_change::<NewD>()?;
        Ok(PermuteAxes {
            shape,
            array,
            axes: self.axes,
            inv_axes: self.inv_axes,
            spec: self.spec,
        })
    }

    type ElementTypeChange<NewET: crate::ElementType> = PermuteAxes<S::ElementTypeChange<NewET>>;
    #[inline]
    fn element_type_change<NewET: crate::ElementType>(
        self,
    ) -> crate::error::Result<Self::ElementTypeChange<NewET>> {
        Ok(PermuteAxes {
            array: self.array.element_type_change()?,
            axes: self.axes,
            inv_axes: self.inv_axes,
            shape: self.shape,
            spec: self.spec,
        })
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;
    use proptest::prelude::*;

    use crate::storage::Compact;
    use crate::util::{shape_strategy, ScalarStrategy};
    use crate::{Array, DimDyn, Ty};

    // 2D i32: transpose (axes=[1,0])
    #[test]
    fn test_i32_2d_transpose() {
        let a = array![[1i32, 2, 3], [4, 5, 6]];
        let za = Array::compact_ndarray(&a).unwrap();
        let actual = za.permute_axes(&[1, 0]).to_ndarray().unwrap();
        let expected = a
            .view()
            .permuted_axes([1, 0])
            .as_standard_layout()
            .into_owned();
        assert_eq!(actual, expected);
    }

    // 2D f32: transpose
    #[test]
    fn test_f32_2d_transpose() {
        let a = array![[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let za = Array::compact_ndarray(&a).unwrap();
        let actual = za.permute_axes(&[1, 0]).to_ndarray().unwrap();
        let expected = a
            .view()
            .permuted_axes([1, 0])
            .as_standard_layout()
            .into_owned();
        assert_eq!(actual, expected);
    }

    // 3D i32: axes=[2,0,1]
    #[test]
    fn test_i32_3d_axes_2_0_1() {
        let a = ndarray::Array::from_shape_fn((2, 3, 4), |(i, j, k)| (i * 12 + j * 4 + k) as i32);
        let za = Array::compact_ndarray(&a).unwrap();
        let actual = za.permute_axes(&[2, 0, 1]).to_ndarray().unwrap();
        let expected = a
            .view()
            .permuted_axes([2, 0, 1])
            .as_standard_layout()
            .into_owned();
        assert_eq!(actual, expected);
    }

    // 3D i32: swap only last two dims axes=[0,2,1]
    #[test]
    fn test_i32_3d_axes_0_2_1() {
        let a = ndarray::Array::from_shape_fn((2, 3, 4), |(i, j, k)| (i * 12 + j * 4 + k) as i32);
        let za = Array::compact_ndarray(&a).unwrap();
        let actual = za.permute_axes(&[0, 2, 1]).to_ndarray().unwrap();
        let expected = a
            .view()
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .into_owned();
        assert_eq!(actual, expected);
    }

    // 3D i32: identity permutation
    #[test]
    fn test_i32_3d_identity() {
        let a = ndarray::Array::from_shape_fn((2, 3, 4), |(i, j, k)| (i * 12 + j * 4 + k) as i32);
        let za = Array::compact_ndarray(&a).unwrap();
        let actual = za.permute_axes(&[0, 1, 2]).to_ndarray().unwrap();
        assert_eq!(actual, a);
    }

    // Panic: axes length does not match ndim
    #[test]
    #[should_panic]
    fn test_wrong_axes_length_panics() {
        let a = array![[1i32, 2], [3, 4]];
        let za = Array::compact_ndarray(&a).unwrap();
        let _ = za.permute_axes(&[0, 1, 2]);
    }

    // Panic: axis out of bounds
    #[test]
    #[should_panic]
    fn test_axis_out_of_bounds_panics() {
        let a = array![[1i32, 2], [3, 4]];
        let za = Array::compact_ndarray(&a).unwrap();
        let _ = za.permute_axes(&[0, 5]);
    }

    // Panic: duplicate axis
    #[test]
    #[should_panic]
    fn test_duplicate_axis_panics() {
        let a = array![[1i32, 2], [3, 4]];
        let za = Array::compact_ndarray(&a).unwrap();
        let _ = za.permute_axes(&[0, 0]);
    }

    // -----------------------------------------------------------------------
    // Proptest: arbitrary ndim, arbitrary permutation, verified against ndarray
    // -----------------------------------------------------------------------

    fn permute_axes_strategy<T>() -> impl Strategy<
        Value = (
            ndarray::ArrayD<T>,
            Array<Compact<Ty<T>, DimDyn>>,
            Vec<usize>,
        ),
    >
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
