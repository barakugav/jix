use std::ops::{Not, Range};

use crate::codec::ReadContext;
use crate::dtype::Dtype;
use crate::error::{check_get_range, check_ndim, check_shape_overflow, ensure, Result};
use crate::storage::params::{combine_select_hints, normalize_dim_scale_weights, ArraySpecDynamic};
use crate::storage::{ArraySpec, ArrayStorageInfo, OutBuf};
use crate::util::{ArraySequence, ArraySequenceDimension, ArraySequenceElementType, DimArray};
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
    stack_axis: usize,

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

        let spec = arrays.spec(0);
        let mut block_shape = spec.block_shape().clone();
        let mut block_shape_fixed_dims = spec.block_shape_fixed_dims();
        block_shape.insert(axis, 1);
        block_shape_fixed_dims.insert(axis, false);
        let (element_cost, dim_scale_weights) = {
            let inputs = (0..narrays)
                .map(|i| {
                    let sp = arrays.spec(i);
                    (sp.element_cost(), sp.dim_scale_weights().as_slice())
                })
                .collect::<Vec<_>>();
            combine_select_hints(&inputs)
        };
        // The new stack axis carries no data of its own; seed it low but nonzero (scaled down for
        // pricier arrays) and re-normalize alongside the base dims.
        let mut raw = dim_scale_weights
            .iter()
            .map(|&w| w.f64())
            .collect::<DimArray<_>>();
        raw.insert(axis, (1.0 / element_cost as f64).min(0.1));
        let spec = ArraySpecDynamic {
            block_shape,
            block_shape_fixed_dims,
            element_cost,
            dim_scale_weights: normalize_dim_scale_weights(raw.as_slice()),
        };

        Ok(Self {
            shape: new_shape,
            spec,
            arrays,
            stack_axis: axis,
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

    fn read_data(
        &self,
        index: &[Range<u64>],
        buf: &mut OutBuf,
        context: &ReadContext,
    ) -> Result<()> {
        let shape = self.shape();
        let dtype = self.dtype();
        check_get_range(shape, index)?;
        let nitems = index.iter().map(|r| r.end - r.start).product::<u64>();
        if nitems == 0 {
            buf.materialize(0, dtype);
            return Ok(());
        }

        let arr_ndim = shape.len() - 1;
        let arr_range = index[..self.stack_axis]
            .iter()
            .chain(index[self.stack_axis + 1..].iter())
            .cloned()
            .collect_dim_vec::<ArraysT::Dimension>(arr_ndim);
        let arr_range_shape = ArraysT::Dimension::vec(arr_ndim, |dim| {
            (arr_range[dim].end - arr_range[dim].start) as usize
        });
        let itemsize = dtype.itemsize() as usize;
        let arr_size_bytes = arr_range_shape.as_ref().iter().product::<usize>() * itemsize;
        let n_stack = (index[self.stack_axis].end - index[self.stack_axis].start) as usize;

        // Write straight into the (possibly strided) destination using its own strides. The
        // in_place fast path (each sub-array a contiguous chunk of buf) is valid only when buf is
        // contiguous; a strided destination always scatters with the destination's strides.
        let is_strided = buf.strides().is_some();
        let (dst, out_strides) = buf.get_strided_mut::<Self::Dimension>(index, dtype);
        // Stride of the stack axis in the output buffer (offset between consecutive sub-arrays).
        let stack_axis_stride = out_strides[self.stack_axis];
        let in_place = !is_strided && shape.iter().take(self.stack_axis).all(|&s| s <= 1);
        // Per-sub-array strides = the output strides with the stack axis removed.
        let out_of_place_strides = in_place.not().then(|| {
            ArraysT::Dimension::vec(arr_ndim, |dim| {
                if dim < self.stack_axis {
                    out_strides[dim]
                } else {
                    out_strides[dim + 1]
                }
            })
        });

        for arr_idx in 0..n_stack {
            // In-place: each array occupies a contiguous chunk in buf.
            // Out-of-place: each array starts at its column offset within buf.
            let buf_offset = arr_idx * stack_axis_stride;
            let arr = index[self.stack_axis].start as usize + arr_idx;
            let mut out = if in_place {
                let sub = &mut dst[buf_offset..buf_offset + arr_size_bytes];
                OutBuf::new(sub)
            } else {
                let strides = out_of_place_strides.as_ref().unwrap().as_ref();
                unsafe { OutBuf::new_strided(&mut dst[buf_offset..], strides) }
            };
            self.arrays
                .read_data(arr, arr_range.as_ref(), &mut out, context)?;
        }

        Ok(())
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
