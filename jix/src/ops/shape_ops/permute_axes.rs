use std::ops::Range;

use crate::codec::ReadContext;
use crate::dtype::Dtype;
use crate::error::{check_get_range, check_ndim, ensure, Result};
use crate::storage::params::ArraySpecDynamic;
use crate::storage::{
    check_out_buf, read_data_and_map_strides, ArraySpec, ArrayStorageInfo, StridedBuf,
};
use crate::util::{dim_arr, DimIdx};
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
    axes: <S::Dimension as Dimension>::Vec<DimIdx>,
    /// `inv_axes[d]` = index of the output dimension that maps from input dimension `d`.
    inv_axes: <S::Dimension as Dimension>::Vec<DimIdx>,

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
        let mut seen = S::Dimension::vec(ndim, |_| false);
        for &ax in axes.as_ref().iter() {
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
        let mut inv_axes = S::Dimension::vec(ndim, |_| DimIdx::default());
        for (i, &ax) in axes.as_ref().iter().enumerate() {
            inv_axes[ax] = i as DimIdx;
        }

        let input_shape = array.shape();
        let shape = S::Dimension::from_fn(ndim, |i| input_shape[axes[i]]);

        let inner_spec = array.spec();
        let block_shape = dim_arr(ndim, |i| inner_spec.block_shape()[axes[i]]);
        let inner_block_shape_fixed_dims = inner_spec.block_shape_fixed_dims();
        let spec = ArraySpecDynamic {
            block_shape,
            block_shape_fixed_dims: (0..ndim)
                .map(|i| inner_block_shape_fixed_dims.get(axes[i]))
                .collect(),
            element_cost: inner_spec.element_cost(),
            read_shape_scale_order: inner_spec
                .read_shape_scale_order()
                .iter()
                .map(|&old| inv_axes[old as usize])
                .collect(),
            read_layout_order: inner_spec
                .read_layout_order()
                .iter()
                .map(|&old| inv_axes[old as usize])
                .collect(),
        };
        let axes = S::Dimension::vec(ndim, |i| axes[i] as DimIdx);
        Ok(Self {
            shape,
            spec,
            array,
            axes,
            inv_axes,
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
    fn read_data<'a>(
        &'a self,
        index: &[Range<u64>],
        context: &'a ReadContext,
        out: Option<&'a mut StridedBuf<'_>>,
    ) -> Result<StridedBuf<'a>> {
        check_get_range(self.shape(), index)?;
        check_out_buf(out.as_deref(), self.shape())?;
        let ndim = self.axes.as_ref().len();
        unsafe {
            read_data_and_map_strides(
                &self.array,
                S::Dimension::vec(ndim, |d| index[self.inv_axes[d] as usize].clone()).as_ref(),
                context,
                out,
                |inner_strides| dim_arr(ndim, |i| inner_strides[self.axes[i] as usize]),
                |out_strides| dim_arr(ndim, |d| out_strides[self.inv_axes[d] as usize]),
            )
        }
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
        self.array
            .spec()
            .with_dynamic_spec(&self.spec)
            .with_cleared_flags()
    }
    fn info(&self) -> ArrayStorageInfo<'_> {
        ArrayStorageInfo::new_deps("PermuteAxes", [&self.array])
    }

    type DimensionChange<NewD: crate::Dimension> = PermuteAxes<S::DimensionChange<NewD>>;
    #[inline]
    fn dimension_change<NewD: crate::Dimension>(
        self,
    ) -> crate::error::Result<Self::DimensionChange<NewD>> {
        let ndim = self.shape().len();
        check_ndim::<NewD>(ndim)?;
        let shape = NewD::from_slice(self.shape());
        let axes = NewD::vec(ndim, |i| self.axes[i]);
        let inv_axes = NewD::vec(ndim, |i| self.inv_axes[i]);
        let array = self.array.dimension_change::<NewD>()?;
        Ok(PermuteAxes {
            shape,
            array,
            axes,
            inv_axes,
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

    use crate::util::{shape_strategy, ScalarStrategy};
    use crate::Array;

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

    #[test]
    fn test_i32_3d_identity() {
        let a = ndarray::Array::from_shape_fn((2, 3, 4), |(i, j, k)| (i * 12 + j * 4 + k) as i32);
        let za = Array::compact_ndarray(&a).unwrap();
        let actual = za.permute_axes(&[0, 1, 2]).to_ndarray().unwrap();
        assert_eq!(actual, a);
    }

    #[test]
    #[should_panic]
    fn test_wrong_axes_length_panics() {
        let a = array![[1i32, 2], [3, 4]];
        let za = Array::compact_ndarray(&a).unwrap();
        let _ = za.permute_axes(&[0, 1, 2]);
    }

    #[test]
    #[should_panic]
    fn test_axis_out_of_bounds_panics() {
        let a = array![[1i32, 2], [3, 4]];
        let za = Array::compact_ndarray(&a).unwrap();
        let _ = za.permute_axes(&[0, 5]);
    }

    #[test]
    #[should_panic]
    fn test_duplicate_axis_panics() {
        let a = array![[1i32, 2], [3, 4]];
        let za = Array::compact_ndarray(&a).unwrap();
        let _ = za.permute_axes(&[0, 0]);
    }

    #[test]
    fn test_transpose_2d() {
        let a = array![[1i32, 2, 3], [4, 5, 6]];
        let za = Array::compact_ndarray(&a).unwrap();
        let actual = za.transpose().to_ndarray().unwrap();
        let expected = a.t().as_standard_layout().into_owned();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_transpose_3d_reverses_all_axes() {
        let a = ndarray::Array::from_shape_fn((2, 3, 4), |(i, j, k)| (i * 12 + j * 4 + k) as i32);
        let za = Array::compact_ndarray(&a).unwrap();
        let actual = za.transpose().to_ndarray().unwrap();
        let expected = a.t().as_standard_layout().into_owned();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_transpose_1d_is_identity() {
        let a = array![1i32, 2, 3];
        let za = Array::compact_ndarray(&a).unwrap();
        assert_eq!(za.transpose().to_ndarray().unwrap(), a);
    }

    // -----------------------------------------------------------------------
    // Proptest: arbitrary ndim, arbitrary permutation, verified against ndarray
    // -----------------------------------------------------------------------

    #[allow(clippy::type_complexity)]
    fn permute_axes_strategy<T>(
    ) -> impl Strategy<Value = (ndarray::ArrayD<T>, crate::util::TestArray<T>, Vec<usize>)>
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
                    crate::util::array_strategy_from_shape::<T>(Just(shape), T::any_strategy());
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
