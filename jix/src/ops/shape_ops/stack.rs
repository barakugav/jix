use std::ops::{Not, Range};

use crate::codec::ReadContext;
use crate::dtype::Dtype;
use crate::error::{check_get_range, check_ndim, check_shape_overflow, ensure, Result};
use crate::storage::params::{combine_block_layout, combine_select_hints, ArraySpecDynamic};
use crate::storage::{check_out_buf, materialize_out_buf, ArraySpec, ArrayStorageInfo, StridedBuf};
use crate::util::{
    default_strides, ArraySequence, ArraySequenceDimension, ArraySequenceElementType, DimArray,
    DimIdx,
};
use crate::{Array, ArrayStorage, Dimension, IterExt};

/// Joins a sequence of arrays along a new axis. See [`Stack`] for details and examples.
///
/// # Panics
///
/// Panics if `arrays` is empty, `axis` is out of bounds, dtypes differ, or shapes differ.
#[track_caller]
pub fn stack<ArraysT>(arrays: ArraysT, axis: usize) -> Array<Stack<ArraysT>>
where
    ArraysT: ArraySequence + ArraySequenceElementType + ArraySequenceDimension,
{
    Array::from_storage(Stack::new(arrays, axis).unwrap())
}

/// Joins a sequence of arrays along a new axis, returned by [`stack`].
///
/// All input arrays must have identical shapes and the same [`Dtype`]. A new axis of size equal to
/// the number of input arrays is inserted at position `axis` in the output. The output has one more
/// dimension than the inputs - unlike
/// [`Concatenate`](crate::ops::Concatenate), which joins along an existing axis.
///
/// The output dimension type `Stack<ArraysT>::Dimension` is
/// `<ArraysT::Dimension as Dimension>::Larger` (where `ArraysT::Dimension` comes from
/// [`ArraySequenceDimension`]) - one dimension wider than the inputs. This means a static dimension
/// is propagated when all input arrays share a known `Dim<N>`, producing `Dim<N+1>` for the output.
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// # Examples
///
/// ```
/// use jix::Array;
/// use ndarray::array;
///
/// // Stack two 1-D arrays along a new leading axis -> shape [2, N]
/// let a = Array::compact_ndarray(&array![1i32, 2, 3])?;
/// let b = Array::compact_ndarray(&array![4i32, 5, 6])?;
/// let c = jix::ops::stack((a, b), 0);
/// assert_eq!(c.shape(), &[2, 3]);
///
/// // Stack along axis 1 -> shape [N, 2]
/// let a = Array::compact_ndarray(&array![1i32, 2, 3])?;
/// let b = Array::compact_ndarray(&array![4i32, 5, 6])?;
/// let c = jix::ops::stack((a, b), 1);
/// assert_eq!(c.shape(), &[3, 2]);
/// # Ok::<(), jix::Error>(())
/// ```
pub struct Stack<ArraysT>
where
    ArraysT: ArraySequence + ArraySequenceElementType + ArraySequenceDimension,
{
    arrays: ArraysT,
    stack_axis: DimIdx,

    shape: <ArraysT::Dimension as crate::Dimension>::Larger,
    spec: ArraySpecDynamic,
}
impl<ArraysT> Stack<ArraysT>
where
    ArraysT: ArraySequence + ArraySequenceElementType + ArraySequenceDimension,
{
    /// Constructs a [`Stack`] storage. See the struct docs for semantics and examples.
    pub fn new(arrays: ArraysT, axis: usize) -> Result<Self> {
        let narrays = arrays.narrays();
        ensure!(
            narrays > 0,
            InvalidShapeOperation,
            "cannot stack zero arrays"
        );

        let shape0 = arrays.shape(0);
        let dtype = arrays.dtype(0);
        for arr in 1..narrays {
            let shape_i = arrays.shape(arr);
            ensure!(
                shape_i == shape0,
                InvalidShapeOperation,
                "cannot stack arrays of different shapes: {shape0:?} != {shape_i:?}"
            );
            let dtype_i = arrays.dtype(arr);
            ensure!(
                dtype_i == dtype,
                UnsupportedDtype,
                "cannot stack arrays of different dtypes: {dtype} != {dtype_i}"
            );
        }
        ensure!(
            axis <= shape0.len(),
            InvalidShapeOperation,
            "axis out of bounds: axis {axis} >= array ndim {}",
            shape0.len()
        );
        check_ndim::<<Self as ArrayStorage>::Dimension>(shape0.len() + 1)?;
        let mut new_shape = DimArray::from_slice(shape0).unwrap();
        new_shape.insert(axis, narrays as u64);
        check_shape_overflow(new_shape.as_slice(), dtype.itemsize() as _)?;
        let new_shape = <Self as ArrayStorage>::Dimension::from_slice(&new_shape);

        // Combine the block layout over the (equal-shape) inputs, then insert the new stack axis:
        // it carries no data of its own, so its block is 1 and non-fixed.
        let (mut block_shape, mut block_shape_fixed_dims) = {
            let inputs = (0..narrays)
                .map(|i| {
                    let sp = arrays.spec(i);
                    (sp.block_shape().as_slice(), sp.block_shape_fixed_dims())
                })
                .collect::<Vec<_>>();
            combine_block_layout(&inputs)
        };
        block_shape.insert(axis, 1);
        block_shape_fixed_dims.insert(axis, false);
        let (element_cost, shared_order) = {
            let inputs = (0..narrays)
                .map(|i| {
                    let sp = arrays.spec(i);
                    (sp.element_cost(), sp.read_shape_scale_order().as_slice())
                })
                .collect::<Vec<_>>();
            combine_select_hints(&inputs)
        };
        let read_shape_scale_order = shared_order
            .iter()
            .map(|&d| if d as usize >= axis { d + 1 } else { d })
            .chain(std::iter::once(axis as DimIdx))
            .collect::<DimArray<_>>();
        let spec = ArraySpecDynamic {
            block_shape,
            block_shape_fixed_dims,
            element_cost,
            read_shape_scale_order,
        };

        Ok(Self {
            shape: new_shape,
            spec,
            arrays,
            stack_axis: axis as DimIdx,
        })
    }

    /// Constructs an array with [`Stack`] storage. See the storage struct docs for semantics and examples.
    pub fn new_array(arrays: ArraysT, axis: usize) -> Result<Array<Self>> {
        Self::new(arrays, axis).map(Array::from_storage)
    }
}
impl<ArraysT> ArrayStorage for Stack<ArraysT>
where
    ArraysT: ArraySequence + ArraySequenceElementType + ArraySequenceDimension,
{
    type ElementType = ArraysT::ElementType;
    type Dimension = <ArraysT::Dimension as crate::Dimension>::Larger;

    fn read_data<'a>(
        &'a self,
        index: &[Range<u64>],
        context: &'a ReadContext,
        out: Option<&'a mut StridedBuf<'_>>,
    ) -> Result<StridedBuf<'a>> {
        let shape = self.shape();
        let dtype = self.dtype();
        check_get_range(shape, index)?;
        check_out_buf(out.as_deref(), shape)?;
        let output_shape =
            Self::Dimension::vec(index.len(), |d| (index[d].end - index[d].start) as usize);
        let mut out = materialize_out_buf(out, context, output_shape.as_ref(), dtype);
        if output_shape.as_ref().contains(&0) {
            return Ok(out);
        }

        let stack_axis = self.stack_axis as usize;
        let arr_ndim = shape.len() - 1;
        let arr_range = index[..stack_axis]
            .iter()
            .chain(index[stack_axis + 1..].iter())
            .cloned()
            .collect_dim_vec::<ArraysT::Dimension>(arr_ndim);
        let arr_range_shape = ArraysT::Dimension::vec(arr_ndim, |dim| {
            (arr_range[dim].end - arr_range[dim].start) as usize
        });
        let itemsize = dtype.itemsize() as usize;
        let arr_size_bytes = arr_range_shape.as_ref().iter().product::<usize>() * itemsize;
        let n_stack = (index[stack_axis].end - index[stack_axis].start) as usize;

        // In-place fast path (each sub-array a contiguous chunk) is valid only when the destination
        // is contiguous and all dims before stack_axis have size <=1; else scatter.
        let in_place = out.is_contiguous(output_shape.as_ref(), dtype)
            && shape.iter().take(stack_axis).all(|&s| s <= 1);
        let (out_buf, out_strides) = out.data_mut();
        // Stride of the stack axis in the output (offset between consecutive sub-arrays).
        let stack_axis_stride = out_strides[stack_axis];
        // Per-sub-array strides = the output strides with the stack axis removed.
        let out_of_place_strides = in_place.not().then(|| {
            ArraysT::Dimension::vec(arr_ndim, |dim| {
                if dim < stack_axis {
                    out_strides[dim]
                } else {
                    out_strides[dim + 1]
                }
            })
        });

        for arr_idx in 0..n_stack {
            let buf_offset = arr_idx * stack_axis_stride;
            let arr = index[stack_axis].start as usize + arr_idx;
            let mut sub = if in_place {
                let sub_c = default_strides(&arr_range_shape, itemsize);
                // SAFETY: contiguous destination; array `arr` packs into this contiguous chunk.
                unsafe {
                    StridedBuf::from_slice_mut(
                        &mut out_buf[buf_offset..buf_offset + arr_size_bytes],
                        sub_c.as_ref(),
                    )
                }
            } else {
                let strides = out_of_place_strides.as_ref().unwrap().as_ref();
                // SAFETY: `strides` are the output strides minus the stack axis; the sub-region
                // at `buf_offset` is within `dest`.
                unsafe { StridedBuf::from_slice_mut(&mut out_buf[buf_offset..], strides) }
            };
            self.arrays
                .read_data(arr, arr_range.as_ref(), context, Some(&mut sub))?;
        }
        Ok(out)
    }

    #[inline(always)]
    fn shape(&self) -> &[u64] {
        self.shape.as_slice()
    }
    #[inline(always)]
    fn dtype(&self) -> &Dtype {
        self.arrays.dtype(0)
    }
    #[inline]
    fn spec(&self) -> ArraySpec<'_> {
        self.arrays
            .spec(0)
            .with_dynamic_spec(&self.spec)
            .with_cleared_flags()
    }
    fn info(&self) -> ArrayStorageInfo<'_> {
        let deps = (0..self.arrays.narrays())
            .map(|i| self.arrays.as_array_storage(i))
            .collect::<Vec<_>>();
        ArrayStorageInfo::new_deps_dyn("Stack", deps)
    }

    crate::ops::impl_dimension_change_default!();
    crate::ops::impl_element_type_change_default!();
}

#[cfg(test)]
mod tests {
    use ndarray::array;
    use proptest::prelude::*;

    use crate::array::Array;
    use crate::ops::stack;
    use crate::storage::Compact;
    use crate::util::{arr_params, shape_strategy, ScalarStrategy};
    use crate::{DimDyn, Ty, NDIM_MAX};

    #[test]
    fn test_i32_1d_axis0() {
        let a = array![1i32, 2, 3, 4];
        let b = array![5i32, 6, 7, 8];
        let za = Array::compact_ndarray(&a).unwrap();
        let zb = Array::compact_ndarray(&b).unwrap();
        let actual = stack(vec![za, zb], 0).to_ndarray().unwrap();
        let expected = ndarray::stack(ndarray::Axis(0), &[a.view(), b.view()]).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_i32_1d_axis1() {
        let a = array![1i32, 2, 3];
        let b = array![4i32, 5, 6];
        let za = Array::compact_ndarray(&a).unwrap();
        let zb = Array::compact_ndarray(&b).unwrap();
        let actual = stack([za, zb], 1).to_ndarray().unwrap();
        let expected = ndarray::stack(ndarray::Axis(1), &[a.view(), b.view()]).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_i32_2d_axis0() {
        let a = array![[1i32, 2, 3], [4, 5, 6]];
        let b = array![[7i32, 8, 9], [10, 11, 12]];
        let za = Array::compact_ndarray(&a).unwrap();
        let zb = Array::compact_ndarray(&b).unwrap();
        let actual = stack((za, zb.as_ref()), 0).to_ndarray().unwrap();
        let expected = ndarray::stack(ndarray::Axis(0), &[a.view(), b.view()]).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_i32_2d_axis1() {
        let a = array![[1i32, 2, 3], [4, 5, 6]];
        let b = array![[7i32, 8, 9], [10, 11, 12]];
        let za = Array::compact_ndarray(&a).unwrap();
        let zb = Array::compact_ndarray(&b).unwrap();
        let actual = stack(vec![za.as_ref(), zb.as_ref()], 1)
            .to_ndarray()
            .unwrap();
        let expected = ndarray::stack(ndarray::Axis(1), &[a.view(), b.view()]).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_i32_three_arrays() {
        let a = array![1i32, 2, 3];
        let b = array![4i32, 5, 6];
        let c = array![7i32, 8, 9];
        let za = Array::compact_ndarray(&a).unwrap();
        let zb = Array::compact_ndarray(&b).unwrap();
        let zc = Array::compact_ndarray(&c).unwrap();
        let actual = stack([za, zb, zc], 0).to_ndarray().unwrap();
        let expected = ndarray::stack(ndarray::Axis(0), &[a.view(), b.view(), c.view()]).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_f32_1d_axis0() {
        let a = array![1.0f32, 2.0, 3.0, 4.0];
        let b = array![5.0f32, 6.0, 7.0, 8.0];
        let za = Array::compact_ndarray(&a).unwrap();
        let zb = Array::compact_ndarray(&b).unwrap();
        let actual = stack([za, zb], 0).to_ndarray().unwrap();
        let expected = ndarray::stack(ndarray::Axis(0), &[a.view(), b.view()]).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_f32_2d_axis1() {
        let a = array![[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let b = array![[7.0f32, 8.0], [9.0, 10.0], [11.0, 12.0]];
        let za = Array::compact_ndarray(&a).unwrap();
        let zb = Array::compact_ndarray(&b).unwrap();
        let actual = stack([za.as_ref(), zb.as_ref()], 1).to_ndarray().unwrap();
        let expected = ndarray::stack(ndarray::Axis(1), &[a.view(), b.view()]).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_f32_three_arrays_multi_block() {
        let a = array![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = array![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
        let c = array![13.0f32, 14.0, 15.0, 16.0, 17.0, 18.0];
        let za = Array::compact_ndarray_with(&a, arr_params(&[2])).unwrap();
        let zb = Array::compact_ndarray_with(&b, arr_params(&[2])).unwrap();
        let zc = Array::compact_ndarray_with(&c, arr_params(&[2])).unwrap();
        let actual = stack((za, zb, zc.as_ref()), 0).to_ndarray().unwrap();
        let expected = ndarray::stack(ndarray::Axis(0), &[a.view(), b.view(), c.view()]).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    #[should_panic]
    fn test_shape_mismatch_panics() {
        let a = Array::compact_ndarray(&array![1i32, 2]).unwrap();
        let b = Array::compact_ndarray(&array![1i32, 2, 3]).unwrap();
        let _ = stack(vec![a, b], 0);
    }

    #[test]
    #[should_panic]
    fn test_dtype_mismatch_panics() {
        let a = Array::compact_ndarray(&array![1i32, 2])
            .unwrap()
            .into_type_dyn();
        let b = Array::compact_ndarray(&array![1.0f32, 2.0])
            .unwrap()
            .into_type_dyn();
        let _ = stack((a, b), 0);
    }

    #[test]
    #[should_panic]
    fn test_empty_panics() {
        let _ = stack(Vec::<Array<Compact<Ty<i64>, DimDyn>>>::new(), 0);
    }

    // -----------------------------------------------------------------------
    // Proptest: arbitrary ndim, arbitrary axis, arbitrary number of arrays
    // -----------------------------------------------------------------------

    #[allow(clippy::type_complexity)]
    fn stack_strategy<T>() -> impl Strategy<
        Value = (
            Vec<ndarray::ArrayD<T>>,
            Vec<Array<Compact<Ty<T>, DimDyn>>>,
            usize,
        ),
    >
    where
        T: ScalarStrategy,
    {
        // Output ndim = input ndim + 1, so input ndim must be < NDIM_MAX.
        shape_strategy()
            .prop_filter("stack needs ndim < NDIM_MAX", |s| s.len() < NDIM_MAX)
            .prop_flat_map(|shape| {
                let axis = 0..=shape.len();
                let n_arrays = 1usize..=5;
                (Just(shape), axis, n_arrays)
            })
            .prop_flat_map(|(shape, axis, n_arrays)| {
                // All arrays share the same shape; only elements and block shapes vary.
                let per_array_strat =
                    crate::util::carray_strategy_from_shape::<T>(Just(shape), T::any_strategy());
                (prop::collection::vec(per_array_strat, n_arrays), Just(axis))
            })
            .prop_map(|(arrays, axis)| {
                let (nds, zas): (Vec<_>, Vec<_>) = arrays.into_iter().unzip();
                (nds, zas, axis)
            })
    }

    proptest::proptest! {
        #[test]
        fn proptest_stack((nds, zas, axis) in stack_strategy::<i32>()) {
            let nd_views: Vec<_> = nds.iter().map(|nd| nd.view()).collect();
            let expected = ndarray::stack(ndarray::Axis(axis), &nd_views).unwrap();
            crate::util::assert_array_matches(&stack(zas, axis), &expected);
        }
    }
}
