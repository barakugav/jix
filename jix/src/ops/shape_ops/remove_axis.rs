use std::ops::Range;

use crate::codec::ReadContext;
use crate::dtype::Dtype;
use crate::error::{check_get_range, check_ndim, ensure, Result};
use crate::ops::AxesArg;
use crate::storage::params::ArraySpecDynamic;
use crate::storage::{
    check_out_buf, read_data_and_map_strides, ArraySpec, ArrayStorageInfo, StridedBuf,
};
use crate::util::{DimArray, DimIdx};
use crate::{dim_arr, Array, ArrayStorage, Dimension};

/// Removes length-1 dimensions from an array's shape,
/// returned by [`Array::remove_axis`](crate::Array::remove_axis). The inverse operation
/// is [`InsertAxis`](crate::ops::InsertAxis).
///
/// `axis` is a set of axis indices in the *input* shape (0-based). Each named dimension must have
/// length exactly 1 and is dropped from the output shape. Duplicate axis indices are not allowed.
/// The order of values in `axis` does not matter. Valid axis indices are `0..input_ndim`.
///
/// Output dtype equals the input dtype.
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// # Dimension tracking
///
/// `RemoveAxis<S, D>` is generic over `D: Dimension`, determined by the axis argument type.
/// Statically-sized arguments encode the output ndim in the type:
///
/// | Argument type | Output `D` |
/// |---|---|
/// | `usize` | `S::Dimension::Smaller` |
/// | `[usize; N]` / `(usize, ...)` N-tuple | `Smaller` applied N times |
/// | `[usize; 0]` / `()` | `S::Dimension` (unchanged) |
/// | `&[usize]` / `&Vec<usize>` | `DimDyn` |
///
/// # Examples
///
/// ```text
/// [1, N]          axis: [0]       -> [N]
/// [N, 1]          axis: [1]       -> [N]
/// [N, 1, M]       axis: [1]       -> [N, M]
/// [1, N, 1, M, 1] axis: [0, 2, 4] -> [N, M]
/// ```
///
/// Different argument types select both the removed axes and the output dimension type:
///
/// ```
/// use jix::{Array, Dim};
/// use ndarray::array;
///
/// let a = Array::compact_ndarray(&array![[[1i32, 2, 3]]])?; // shape [1, 1, 3], Dim<3>
///
/// // usize -> output D = Dim<2> (one fewer than input Dim<3>)
/// assert_eq!(a.view().remove_axis(0).shape(), &[1, 3]);
///
/// // [usize; 2] -> output D = Dim<1> (two fewer than input Dim<3>)
/// assert_eq!(a.view().remove_axis([0, 1]).shape(), &[3]);
///
/// // &[usize] -> output D = DimDyn
/// let axes = vec![0, 1];
/// assert_eq!(a.remove_axis(axes.as_slice()).shape(), &[3]);
/// # Ok::<(), jix::Error>(())
/// ```
pub struct RemoveAxis<S, D: Dimension> {
    array: S,
    axes_mapping: D::Vec<DimIdx>,

    shape: D,
    spec: ArraySpecDynamic,
}

impl<S, D> RemoveAxis<S, D>
where
    S: ArrayStorage,
    D: Dimension,
{
    /// Constructs a [`RemoveAxis`] storage. See the struct docs for semantics and examples.
    pub fn new<Ax>(array: S, axis: Ax) -> Result<Self>
    where
        Ax: AxesArg<ReducedDimension<S::Dimension> = D>,
    {
        let input_ndim = array.shape().len();

        // Validate axis indices and check for duplicates.
        let mut is_removed = DimArray::<bool>::from_iter(std::iter::repeat_n(false, input_ndim));
        let axes = dim_arr(axis.len(), |i| axis.get(i));
        for &ax in &axes {
            ensure!(
                ax < input_ndim,
                InvalidShapeOperation,
                "axis {ax} out of bounds for array of ndim {input_ndim} \
                 (axis indices must be in 0..{input_ndim})"
            );
            ensure!(
                !is_removed[ax],
                InvalidShapeOperation,
                "duplicate axis {ax}"
            );
            is_removed[ax] = true;

            ensure!(
                array.shape()[ax] == 1,
                InvalidShapeOperation,
                "cannot remove axis {ax} with size {} (only size-1 axes can be removed)",
                array.shape()[ax]
            );
        }

        let mut axes_mapping = DimArray::new();
        let mut shape = DimArray::new();

        let inner_spec = array.spec();
        let inner_block_shape = inner_spec.block_shape();
        let mut block_shape = DimArray::new();

        for input_dim in 0..input_ndim {
            if !is_removed[input_dim] {
                axes_mapping.push(input_dim as DimIdx);
                shape.push(array.shape()[input_dim]);
                block_shape.push(inner_block_shape[input_dim]);
            }
        }
        let shape = D::from_slice(&shape);
        let axes_mapping = D::vec(axes_mapping.len(), |d| axes_mapping[d]);

        let block_shape_fixed_dims = inner_spec
            .block_shape_fixed_dims()
            .into_iter()
            .enumerate()
            .filter_map(|(dim, c)| (!is_removed[dim]).then_some(c))
            .collect();
        let out_dim = |d: usize| (d - (0..d).filter(|&j| is_removed[j]).count()) as DimIdx;
        let read_shape_scale_order = inner_spec
            .read_shape_scale_order()
            .iter()
            .filter(|&&d| !is_removed[d as usize])
            .map(|&d| out_dim(d as usize))
            .collect();
        let spec = ArraySpecDynamic {
            block_shape,
            block_shape_fixed_dims,
            element_cost: inner_spec.element_cost(),
            read_shape_scale_order,
        };

        Ok(Self {
            array,
            axes_mapping,
            shape,
            spec,
        })
    }

    /// Constructs an array with [`RemoveAxis`] storage. See the storage struct docs for semantics and examples.
    pub fn new_array<Ax>(array: Array<S>, axis: Ax) -> Result<Array<Self>>
    where
        Ax: AxesArg<ReducedDimension<S::Dimension> = D>,
    {
        Self::new(array.into_storage(), axis).map(Array::from_storage)
    }

    #[inline(always)]
    fn transform_index(
        &self,
        index: &[Range<u64>],
    ) -> Result<<S::Dimension as Dimension>::Vec<Range<u64>>> {
        check_get_range(self.shape(), index)?;

        let inner_ndim = self.array.shape().len();
        let mut inner_index = S::Dimension::vec(inner_ndim, |_| 0..1);
        for (index, &axis) in index.iter().zip(self.axes_mapping.as_ref()) {
            inner_index[axis as usize] = index.clone();
        }
        Ok(inner_index)
    }
}

impl<S, D> ArrayStorage for RemoveAxis<S, D>
where
    S: ArrayStorage,
    D: Dimension,
{
    type ElementType = S::ElementType;
    type Dimension = D;

    #[inline]
    fn read_data<'a>(
        &'a self,
        index: &[Range<u64>],
        context: &'a ReadContext,
        out: Option<&'a mut StridedBuf<'_>>,
    ) -> Result<StridedBuf<'a>> {
        check_out_buf(out.as_deref(), self.shape())?;
        unsafe {
            read_data_and_map_strides(
                &self.array,
                self.transform_index(index)?.as_ref(),
                context,
                out,
                |inner_strides| {
                    dim_arr(index.len(), |i| {
                        inner_strides[self.axes_mapping[i] as usize]
                    })
                },
                |out_strides| {
                    let mut s = dim_arr(self.array.shape().len(), |_| 0usize);
                    for (i, &axis) in self.axes_mapping.as_ref().iter().enumerate() {
                        s[axis as usize] = out_strides[i];
                    }
                    s
                },
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
            .map_flags(|flags| flags.clear_compact())
    }
    fn info(&self) -> ArrayStorageInfo<'_> {
        ArrayStorageInfo::new_deps("RemoveAxis", [&self.array])
    }

    type DimensionChange<NewD: crate::Dimension> = RemoveAxis<S, NewD>;
    #[inline]
    fn dimension_change<NewD: crate::Dimension>(
        self,
    ) -> crate::error::Result<Self::DimensionChange<NewD>> {
        let ndim = self.shape().len();
        check_ndim::<NewD>(ndim)?;
        let shape = NewD::from_slice(self.shape());
        let axes_mapping = NewD::vec(ndim, |d| self.axes_mapping[d]);
        Ok(RemoveAxis {
            array: self.array,
            axes_mapping,
            shape,
            spec: self.spec,
        })
    }

    type ElementTypeChange<NewET: crate::ElementType> = RemoveAxis<S::ElementTypeChange<NewET>, D>;
    #[inline]
    fn element_type_change<NewET: crate::ElementType>(
        self,
    ) -> crate::error::Result<Self::ElementTypeChange<NewET>> {
        Ok(RemoveAxis {
            array: self.array.element_type_change()?,
            axes_mapping: self.axes_mapping,
            shape: self.shape,
            spec: self.spec,
        })
    }
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
mod tests {
    use ndarray::array;
    use proptest::prelude::*;

    use crate::array::Array;
    use crate::codec::ReadContext;
    use crate::storage::Compact;
    use crate::util::{arr_params, shape_strategy, ScalarStrategy};
    use crate::{Dim, DimDyn, Ty, NDIM_MAX};

    fn make1d(vals: Vec<i32>, block_size: usize) -> Array<Compact<Ty<i32>, Dim<1>>> {
        let nd = ndarray::Array::from_shape_vec([vals.len()], vals).unwrap();
        Array::compact_ndarray_with(&nd, arr_params(&[block_size])).unwrap()
    }

    fn make2d(vals: Vec<i32>, rows: usize, cols: usize) -> Array<Compact<Ty<i32>, Dim<2>>> {
        let nd = ndarray::Array::from_shape_vec([rows, cols], vals).unwrap();
        Array::compact_ndarray_with(&nd, arr_params(&[rows, cols])).unwrap()
    }

    fn make3d(vals: Vec<i32>, d0: usize, d1: usize, d2: usize) -> Array<Compact<Ty<i32>, Dim<3>>> {
        let nd = ndarray::Array::from_shape_vec([d0, d1, d2], vals).unwrap();
        Array::compact_ndarray_with(&nd, arr_params(&[d0, d1, d2])).unwrap()
    }

    fn arange(n: usize) -> Vec<i32> {
        (0..n as i32).collect()
    }

    // -----------------------------------------------------------------------
    // Shape metadata
    // -----------------------------------------------------------------------

    #[test]
    fn shape_remove_leading() {
        let a = make1d(arange(6), 6).insert_axis(&[0]);
        assert_eq!(a.remove_axis(&[0]).shape(), &[6]);
    }

    #[test]
    fn shape_remove_trailing() {
        let a = make1d(arange(6), 6).insert_axis(&[1]);
        assert_eq!(a.remove_axis(&[1]).shape(), &[6]);
    }

    #[test]
    fn shape_remove_middle() {
        let a = make2d(arange(12), 3, 4).insert_axis(&[1]);
        assert_eq!(a.remove_axis(&[1]).shape(), &[3, 4]);
    }

    #[test]
    fn shape_remove_multiple() {
        let a = make2d(arange(6), 2, 3).insert_axis(&[0, 1, 2]);
        assert_eq!(a.remove_axis(&[0, 2, 4]).shape(), &[2, 3]);
    }

    #[test]
    fn shape_remove_empty_axes_is_identity() {
        assert_eq!(make2d(arange(12), 3, 4).remove_axis(&[]).shape(), &[3, 4]);
    }

    #[test]
    fn shape_remove_unsorted_axes_same_result() {
        let a1 = make3d(arange(6), 1, 2, 3).remove_axis(&[0]);
        let a2 = make3d(arange(6), 1, 2, 3).remove_axis(&[0]);
        assert_eq!(a1.shape(), a2.shape());
    }

    // -----------------------------------------------------------------------
    // Full reads
    // -----------------------------------------------------------------------

    #[test]
    fn full_read_remove_leading() {
        let got = make1d(arange(6), 6)
            .insert_axis(&[0])
            .remove_axis(&[0])
            .to_ndarray()
            .unwrap();
        assert_eq!(got, ndarray::Array::from_shape_vec([6], arange(6)).unwrap());
    }

    #[test]
    fn full_read_remove_trailing() {
        let got = make1d(arange(6), 6)
            .insert_axis(&[1])
            .remove_axis(&[1])
            .to_ndarray()
            .unwrap();
        assert_eq!(got, ndarray::Array::from_shape_vec([6], arange(6)).unwrap());
    }

    #[test]
    fn full_read_remove_middle() {
        let got = make2d(arange(12), 3, 4)
            .insert_axis(&[1])
            .remove_axis(&[1])
            .to_ndarray()
            .unwrap();
        assert_eq!(
            got,
            ndarray::Array::from_shape_vec([3, 4], arange(12)).unwrap()
        );
    }

    #[test]
    fn full_read_remove_multiple() {
        let got = make2d(arange(6), 2, 3)
            .insert_axis(&[0, 1, 2])
            .remove_axis(&[0, 2, 4])
            .to_ndarray()
            .unwrap();
        assert_eq!(
            got,
            ndarray::Array::from_shape_vec([2, 3], arange(6)).unwrap()
        );
    }

    #[test]
    fn full_read_identity_empty_axes() {
        let got = make2d(arange(12), 3, 4)
            .remove_axis(&[])
            .to_ndarray()
            .unwrap();
        assert_eq!(
            got,
            ndarray::Array::from_shape_vec([3, 4], arange(12)).unwrap()
        );
    }

    // -----------------------------------------------------------------------
    // Sub-region reads
    // -----------------------------------------------------------------------

    #[test]
    fn sub_read_after_remove_leading() {
        let got = make1d(arange(6), 6)
            .insert_axis(&[0])
            .remove_axis(&[0])
            .to_ndarray_sub(&[2..5], &ReadContext::default())
            .unwrap();
        assert_eq!(got, array![2, 3, 4]);
    }

    #[test]
    fn sub_read_after_remove_middle() {
        let got = make2d(arange(12), 3, 4)
            .insert_axis(&[1])
            .remove_axis(&[1])
            .to_ndarray_sub(&[1..3, 0..2], &ReadContext::default())
            .unwrap();
        assert_eq!(got, array![[4, 5], [8, 9]]);
    }

    // -----------------------------------------------------------------------
    // Error cases
    // -----------------------------------------------------------------------

    #[test]
    fn error_axis_out_of_bounds() {
        let a = make2d(arange(4), 2, 2);
        // ndim=2, valid axes are 0..2; axis 3 is out of bounds
        assert!(super::RemoveAxis::new_array(a, &[3]).is_err());
    }

    #[test]
    fn error_axis_not_size_one() {
        let a = make2d(arange(12), 3, 4);
        // axis 0 has size 3, cannot remove
        assert!(super::RemoveAxis::new_array(a, &[0]).is_err());
    }

    #[test]
    fn error_duplicate_axis() {
        let a = make3d(arange(6), 1, 2, 3);
        assert!(super::RemoveAxis::new_array(a, &[0, 0]).is_err());
    }

    #[allow(clippy::type_complexity)]
    fn remove_axes_strategy<T>() -> impl proptest::strategy::Strategy<
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
                let max_dims_to_remove = NDIM_MAX - shape.len();
                (Just(shape), 0..=max_dims_to_remove)
            })
            .prop_flat_map(|(shape, ndims_to_remove)| {
                let dims_to_remove = prop::collection::vec(0..=shape.len(), ndims_to_remove);
                (Just(shape), dims_to_remove)
            })
            .prop_flat_map(|(mut shape, mut dims_to_remove)| {
                dims_to_remove.sort_unstable();
                for (i, dim) in dims_to_remove.iter_mut().enumerate() {
                    let shift = i;
                    shape.insert(shift + *dim, 1);
                    *dim += shift;
                }
                (Just(shape), Just(dims_to_remove).prop_shuffle())
            })
            .prop_flat_map(|(shape, axes)| {
                let array_strat =
                    crate::util::carray_strategy_from_shape::<T>(Just(shape), T::any_strategy());
                (array_strat, Just(axes))
            })
            .prop_map(|((nd, za), axes)| (nd, za, axes))
    }

    proptest::proptest! {
        #[test]
        fn proptest_remove_axes((nd, za, axes) in remove_axes_strategy::<i32>()) {
            // Oracle: removing size-1 axes is a pure reshape - flat order is unchanged.
            let expected_shape = nd
                .shape()
                .iter()
                .enumerate()
                .filter(|(i, _)| !axes.contains(i))
                .map(|(_, &s)| s)
                .collect::<Vec<_>>();
            let expected = ndarray::ArrayD::from_shape_vec(
                expected_shape,
                nd.iter().cloned().collect::<Vec<_>>(),
            )
            .unwrap();
            crate::util::assert_array_matches(&za.remove_axis(&axes), &expected);
        }
    }
}
