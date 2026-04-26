use std::ops::{Not, Range};

use crate::array::Array;
use crate::codec::ReadContext;
use crate::dtype::Dtype;
use crate::error::{check_get_buffer_size, check_get_range, check_ndim, ensure, Result};
use crate::storage::{ArrayStorage, ArrayStorageSpec, BlockShapeTag, BlocksLayout};
use crate::util::{default_strides, dim_arr, nd_copy, ArraySequence, DimArray};

/// Joins a sequence of arrays along a new axis. See [`Stack`] for details and examples.
///
/// # Panics
///
/// Panics if `arrays` is empty, `axis` is out of bounds, dtypes differ, or shapes differ.
#[track_caller]
pub fn stack<ArraysT>(arrays: ArraysT, axis: usize) -> Array<Stack<ArraysT>>
where
    ArraysT: ArraySequence,
{
    Array::from_storage(Stack::new(arrays, axis).unwrap())
}

/// Joins a sequence of arrays along a new axis, returned by [`stack`].
///
/// All input arrays must have identical shapes and the same [`Dtype`]. A new axis of size equal to
/// the number of input arrays is inserted at position `axis` in the output. The output has one more
/// dimension than the inputs — unlike
/// [`Concatenate`](crate::ops::Concatenate), which joins along an existing axis.
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// # Examples
///
/// ```
/// use zix::{Array, ArrayParams};
/// use ndarray::array;
///
/// // Stack two 1-D arrays along a new leading axis → shape [2, N]
/// let a = Array::compact_array(&array![1i32, 2, 3])?;
/// let b = Array::compact_array(&array![4i32, 5, 6])?;
/// let c = zix::ops::stack((a, b), 0);
/// assert_eq!(c.shape(), &[2, 3]);
///
/// // Stack along axis 1 → shape [N, 2]
/// let a = Array::compact_array(&array![1i32, 2, 3])?;
/// let b = Array::compact_array(&array![4i32, 5, 6])?;
/// let c = zix::ops::stack((a, b), 1);
/// assert_eq!(c.shape(), &[3, 2]);
/// # Ok::<(), zix::error::Error>(())
/// ```
pub struct Stack<ArraysT> {
    arrays: ArraysT,
    stack_axis: usize,

    dtype: Dtype,
    shape: DimArray<u64>,
    blocks_layout: BlocksLayout,
}
impl<ArraysT> Stack<ArraysT> {
    pub fn new(arrays: ArraysT, axis: usize) -> Result<Self>
    where
        ArraysT: ArraySequence,
    {
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
                "cannot stack arrays of different dtypes: {dtype:?} != {dtype_i:?}"
            );
        }
        ensure!(
            axis <= shape0.len(),
            InvalidShapeOperation,
            "axis out of bounds: axis {axis} >= array ndim {}",
            shape0.len()
        );
        check_ndim(shape0.len() + 1)?;
        let mut new_shape: DimArray<_> = shape0.try_into().unwrap();
        new_shape.insert(axis, narrays as u64);

        let mut b_layout = arrays._spec(0).blocks_layout.clone();
        b_layout.block_shape_hint.insert(axis, 1);
        b_layout.block_shape_tag.insert(axis, BlockShapeTag::Any);
        b_layout.preferred_read_shape.insert(axis, 1);

        Ok(Self {
            dtype: dtype.clone(),
            shape: new_shape,
            blocks_layout: b_layout,
            arrays,
            stack_axis: axis,
        })
    }
}
impl<ArraysT> ArrayStorage for Stack<ArraysT>
where
    ArraysT: ArraySequence,
{
    fn read_data(&self, index: &[Range<u64>], buf: &mut [u8], context: &ReadContext) -> Result<()> {
        check_get_range(&self.shape, index)?;
        check_get_buffer_size(index, &self.dtype, buf)?;

        let in_place = self.shape.iter().take(self.stack_axis).all(|&s| s <= 1);
        let arr_range = index[..self.stack_axis]
            .iter()
            .chain(index[self.stack_axis + 1..].iter())
            .cloned()
            .collect::<DimArray<_>>();
        let arr_range_shape = dim_arr(arr_range.len(), |dim| {
            (arr_range[dim].end - arr_range[dim].start) as usize
        });
        let itemsize = self.dtype.itemsize() as usize;
        let arr_size_bytes = arr_range_shape.iter().product::<usize>() * itemsize;
        let mut tmp_buf = in_place
            .not()
            .then(|| context.tmp_buf(arr_size_bytes, self.dtype.alignment()));
        // Stride of the stack axis in the output buffer (= size of one sub-array slice).
        let stack_axis_stride =
            arr_range_shape[self.stack_axis..].iter().product::<usize>() * itemsize;
        let n_stack = (index[self.stack_axis].end - index[self.stack_axis].start) as usize;
        let out_of_place_strides = in_place.not().then(|| {
            let arr_strides = default_strides(&arr_range_shape, itemsize);
            // For dims before the stack axis the output stride is n_stack times wider;
            // for dims at or after it the stride is unchanged.
            let mut out_strides = arr_strides.clone();
            for dim in 0..self.stack_axis {
                out_strides[dim] *= n_stack;
            }
            (arr_strides, out_strides)
        });

        for arr_idx in 0..n_stack {
            // In-place: each array occupies a contiguous chunk in buf.
            // Out-of-place: each array starts at its column offset within buf.
            let buf_offset = arr_idx * stack_axis_stride;
            let arr_buf = if in_place {
                &mut buf[buf_offset..buf_offset + arr_size_bytes]
            } else {
                let tmp_buf = tmp_buf.as_mut().unwrap();
                tmp_buf.set_len(arr_size_bytes);
                tmp_buf.as_mut_slice()
            };

            let arr = index[self.stack_axis].start as usize + arr_idx;
            self.arrays.read_data(arr, &arr_range, arr_buf, context)?;

            // copy arr_buf into the correct position in buf, as both buffers have different strides
            if let Some((arr_strides, out_strides)) = &out_of_place_strides {
                unsafe {
                    nd_copy(
                        arr_buf.as_ptr(),
                        buf.as_mut_ptr().add(buf_offset),
                        &arr_range_shape,
                        arr_strides,
                        out_strides,
                        itemsize,
                    )
                };
            }
        }

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
            ..self.arrays._spec(0)
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use crate::array::Array;
    use crate::ops::stack;
    use crate::util::arr_params;

    // stack two 1D i32 arrays along axis 0 → shape [2, N]
    #[test]
    fn test_i32_1d_axis0() {
        let a = array![1i32, 2, 3, 4];
        let b = array![5i32, 6, 7, 8];
        let za = Array::compact_array(&a).unwrap();
        let zb = Array::compact_array(&b).unwrap();
        let actual = stack(vec![za, zb], 0).to_ndarray::<i32>().unwrap();
        let expected = ndarray::stack(ndarray::Axis(0), &[a.view(), b.view()]).unwrap();
        assert_eq!(actual, expected.into_dyn());
    }

    // stack two 1D i32 arrays along axis 1 → shape [N, 2]
    #[test]
    fn test_i32_1d_axis1() {
        let a = array![1i32, 2, 3];
        let b = array![4i32, 5, 6];
        let za = Array::compact_array(&a).unwrap();
        let zb = Array::compact_array(&b).unwrap();
        let actual = stack([za, zb], 1).to_ndarray::<i32>().unwrap();
        let expected = ndarray::stack(ndarray::Axis(1), &[a.view(), b.view()]).unwrap();
        assert_eq!(actual, expected.into_dyn());
    }

    // stack two 2D i32 arrays along axis 0 → shape [2, M, N]
    #[test]
    fn test_i32_2d_axis0() {
        let a = array![[1i32, 2, 3], [4, 5, 6]];
        let b = array![[7i32, 8, 9], [10, 11, 12]];
        let za = Array::compact_array(&a).unwrap();
        let zb = Array::compact_array(&b).unwrap();
        let actual = stack((za, zb.as_ref()), 0).to_ndarray::<i32>().unwrap();
        let expected = ndarray::stack(ndarray::Axis(0), &[a.view(), b.view()]).unwrap();
        assert_eq!(actual, expected.into_dyn());
    }

    // stack two 2D i32 arrays along axis 1 → shape [M, 2, N]
    #[test]
    fn test_i32_2d_axis1() {
        let a = array![[1i32, 2, 3], [4, 5, 6]];
        let b = array![[7i32, 8, 9], [10, 11, 12]];
        let za = Array::compact_array(&a).unwrap();
        let zb = Array::compact_array(&b).unwrap();
        let actual = stack(vec![za.as_ref(), zb.as_ref()], 1)
            .to_ndarray::<i32>()
            .unwrap();
        let expected = ndarray::stack(ndarray::Axis(1), &[a.view(), b.view()]).unwrap();
        assert_eq!(actual, expected.into_dyn());
    }

    // stack three 1D i32 arrays along axis 0 → shape [3, N]
    #[test]
    fn test_i32_three_arrays() {
        let a = array![1i32, 2, 3];
        let b = array![4i32, 5, 6];
        let c = array![7i32, 8, 9];
        let za = Array::compact_array(&a).unwrap();
        let zb = Array::compact_array(&b).unwrap();
        let zc = Array::compact_array(&c).unwrap();
        let actual = stack([za, zb, zc], 0).to_ndarray::<i32>().unwrap();
        let expected = ndarray::stack(ndarray::Axis(0), &[a.view(), b.view(), c.view()]).unwrap();
        assert_eq!(actual, expected.into_dyn());
    }

    // stack two 1D f32 arrays along axis 0 → shape [2, N]
    #[test]
    fn test_f32_1d_axis0() {
        let a = array![1.0f32, 2.0, 3.0, 4.0];
        let b = array![5.0f32, 6.0, 7.0, 8.0];
        let za = Array::compact_array(&a).unwrap();
        let zb = Array::compact_array(&b).unwrap();
        let actual = stack([za, zb], 0).to_ndarray::<f32>().unwrap();
        let expected = ndarray::stack(ndarray::Axis(0), &[a.view(), b.view()]).unwrap();
        assert_eq!(actual, expected.into_dyn());
    }

    // stack two 2D f32 arrays along axis 1 → shape [M, 2, N]
    #[test]
    fn test_f32_2d_axis1() {
        let a = array![[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let b = array![[7.0f32, 8.0], [9.0, 10.0], [11.0, 12.0]];
        let za = Array::compact_array(&a).unwrap();
        let zb = Array::compact_array(&b).unwrap();
        let actual = stack([za.as_ref(), zb.as_ref()], 1)
            .to_ndarray::<f32>()
            .unwrap();
        let expected = ndarray::stack(ndarray::Axis(1), &[a.view(), b.view()]).unwrap();
        assert_eq!(actual, expected.into_dyn());
    }

    // stack three f32 arrays along axis 0 with multi-block layout
    #[test]
    fn test_f32_three_arrays_multi_block() {
        let a = array![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = array![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
        let c = array![13.0f32, 14.0, 15.0, 16.0, 17.0, 18.0];
        let za = Array::compact_array_with(&a, arr_params(&[2])).unwrap();
        let zb = Array::compact_array_with(&b, arr_params(&[2])).unwrap();
        let zc = Array::compact_array_with(&c, arr_params(&[2])).unwrap();
        let actual = stack((za, zb, zc.as_ref()), 0).to_ndarray::<f32>().unwrap();
        let expected = ndarray::stack(ndarray::Axis(0), &[a.view(), b.view(), c.view()]).unwrap();
        assert_eq!(actual, expected.into_dyn());
    }

    #[test]
    #[should_panic]
    fn test_shape_mismatch_panics() {
        let a = Array::compact_array(&array![1i32, 2]).unwrap();
        let b = Array::compact_array(&array![1i32, 2, 3]).unwrap();
        let _ = stack(vec![a, b], 0);
    }

    #[test]
    #[should_panic]
    fn test_dtype_mismatch_panics() {
        let a = Array::compact_array(&array![1i32, 2]).unwrap();
        let b = Array::compact_array(&array![1.0f32, 2.0]).unwrap();
        let _ = stack((a, b), 0);
    }

    #[test]
    #[should_panic]
    fn test_empty_panics() {
        let _ = stack(Vec::<Array<crate::storage::Compact>>::new(), 0);
    }
}
