use std::cell::RefCell;
use std::io;
use std::ops::Range;

use crate::array::{Array, BlocksLayout};
use crate::codec::ReadContext;
use crate::dtype::Dtype;
use crate::iter::NdIter;
use crate::iter::strides::{NdIterExtStridesPtr, NdIterExtStridesPtrMut};
use crate::storage::ArrayStorage;
use crate::util::{AlignedBytes, ArraySequence, DimArray, default_strides, dim_arr};

#[track_caller]
pub fn stack<ArraysT>(arrays: ArraysT, axis: usize) -> Array<Stack<ArraysT>>
where
    ArraysT: ArraySequence,
{
    Array::new(Stack::new(arrays, axis).unwrap())
}

pub struct Stack<ArraysT> {
    arrays: ArraysT,
    stack_axis: usize,

    dtype: Dtype,
    shape: DimArray<usize>,
    blocks_layout: BlocksLayout,

    tmp_buf: RefCell<AlignedBytes>,
}
impl<ArraysT> Stack<ArraysT> {
    pub(crate) fn new(arrays: ArraysT, axis: usize) -> io::Result<Self>
    where
        ArraysT: ArraySequence,
    {
        let narrays = arrays.narrays();
        if narrays == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "cannot stack zero arrays",
            ));
        }
        let shape0 = arrays.shape(0);
        let dtype = arrays.dtype(0);
        for arr in 1..narrays {
            let shape_i = arrays.shape(arr);
            if shape_i != shape0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("cannot stack arrays of different shapes: {shape0:?} != {shape_i:?}",),
                ));
            }
            let dtype_i = arrays.dtype(arr);
            if dtype_i != dtype {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("cannot stack arrays of different dtypes: {dtype:?} != {dtype_i:?}",),
                ));
            }
        }
        if axis > shape0.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "axis out of bounds: axis {axis} >= array ndim {}",
                    shape0.len()
                ),
            ));
        }
        if shape0.len() + 1 > crate::NDIM_MAX {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "stacking arrays would result in too many dimensions: {}",
                    shape0.len() + 1
                ),
            ));
        }
        let mut new_shape: DimArray<_> = shape0.try_into().unwrap();
        new_shape.insert(axis, narrays);

        let mut block_shape = arrays.blocks_layout(0).block_shape.clone();
        block_shape.insert(axis, 1);

        Ok(Self {
            dtype: dtype.clone(),
            shape: new_shape,
            blocks_layout: BlocksLayout::new(&block_shape),
            tmp_buf: RefCell::new(AlignedBytes::new(dtype.alignment() as usize)),
            arrays,
            stack_axis: axis,
        })
    }
}
impl<ArraysT> ArrayStorage for Stack<ArraysT>
where
    ArraysT: ArraySequence,
{
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
        let mut tmp_buf = self.tmp_buf.borrow_mut();

        let in_place = self.shape.iter().take(self.stack_axis).all(|&s| s <= 1);
        let arr_range = index
            .iter()
            .enumerate()
            .filter(|(dim, _)| *dim != self.stack_axis)
            .map(|(_, r)| r.clone())
            .collect::<DimArray<_>>();
        let arr_range_shape = dim_arr(arr_range.len(), |dim| arr_range[dim].len());
        let itemsize = self.dtype.itemsize() as usize;
        let arr_size_bytes = arr_range.iter().map(|r| r.len()).product::<usize>() * itemsize;
        // Stride of the stack axis in the output buffer (= size of one sub-array slice).
        let stack_axis_stride = arr_range[self.stack_axis..]
            .iter()
            .map(|r| r.len())
            .product::<usize>()
            * itemsize;
        for arr_idx in 0..index[self.stack_axis].len() {
            // In-place: each array occupies a contiguous chunk in buf.
            // Out-of-place: each array starts at its column offset within buf.
            let buf_offset = arr_idx * stack_axis_stride;
            let arr_buf = if in_place {
                &mut buf[buf_offset..buf_offset + arr_size_bytes]
            } else {
                tmp_buf.clear();
                tmp_buf.reserve(arr_size_bytes);
                unsafe { tmp_buf.set_len(arr_size_bytes) };
                tmp_buf.as_mut_slice()
            };

            let arr = index[self.stack_axis].start + arr_idx;
            self.arrays.read_data(arr, &arr_range, arr_buf, context)?;

            // copy arr_buf into the correct position in buf, as both buffers have different strides
            if !in_place {
                let n_arrays = index[self.stack_axis].len();
                let arr_strides = default_strides(&arr_range_shape, itemsize);
                // For dims before the stack axis the output stride is n_arrays times wider;
                // for dims at or after it the stride is unchanged.
                let mut out_strides = arr_strides.clone();
                for dim in 0..self.stack_axis {
                    out_strides[dim] *= n_arrays;
                }

                let mut iter = NdIter::new(
                    &arr_range_shape,
                    (
                        NdIterExtStridesPtr::new(&arr_strides, arr_buf.as_ptr()),
                        NdIterExtStridesPtrMut::new(&out_strides, unsafe {
                            buf.as_mut_ptr().add(buf_offset)
                        }),
                    ),
                );
                while let Some((_idx, (src_ptr, dst_ptr))) = iter.next() {
                    unsafe {
                        std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, itemsize);
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::array::Array;
    use crate::ops::stack;

    // stack two 1D i32 arrays along axis 0 → shape [2, N]
    #[test]
    fn test_i32_1d_axis0() {
        let a = ndarray::array![1i32, 2, 3, 4];
        let b = ndarray::array![5i32, 6, 7, 8];
        let za = Array::from_ndarray(&a.view().into_dyn(), &[4]).unwrap();
        let zb = Array::from_ndarray(&b.view().into_dyn(), &[4]).unwrap();
        let actual = stack(vec![za, zb], 0).data().to_ndarray::<i32>().unwrap();
        let expected = ndarray::stack(ndarray::Axis(0), &[a.view(), b.view()]).unwrap();
        assert_eq!(actual, expected.into_dyn());
    }

    // stack two 1D i32 arrays along axis 1 → shape [N, 2]
    #[test]
    fn test_i32_1d_axis1() {
        let a = ndarray::array![1i32, 2, 3];
        let b = ndarray::array![4i32, 5, 6];
        let za = Array::from_ndarray(&a.view().into_dyn(), &[3]).unwrap();
        let zb = Array::from_ndarray(&b.view().into_dyn(), &[3]).unwrap();
        let actual = stack([za, zb], 1).data().to_ndarray::<i32>().unwrap();
        let expected = ndarray::stack(ndarray::Axis(1), &[a.view(), b.view()]).unwrap();
        assert_eq!(actual, expected.into_dyn());
    }

    // stack two 2D i32 arrays along axis 0 → shape [2, M, N]
    #[test]
    fn test_i32_2d_axis0() {
        let a = ndarray::array![[1i32, 2, 3], [4, 5, 6]];
        let b = ndarray::array![[7i32, 8, 9], [10, 11, 12]];
        let za = Array::from_ndarray(&a.view().into_dyn(), &[2, 3]).unwrap();
        let zb = Array::from_ndarray(&b.view().into_dyn(), &[2, 3]).unwrap();
        let actual = stack((za, &zb), 0).data().to_ndarray::<i32>().unwrap();
        let expected = ndarray::stack(ndarray::Axis(0), &[a.view(), b.view()]).unwrap();
        assert_eq!(actual, expected.into_dyn());
    }

    // stack two 2D i32 arrays along axis 1 → shape [M, 2, N]
    #[test]
    fn test_i32_2d_axis1() {
        let a = ndarray::array![[1i32, 2, 3], [4, 5, 6]];
        let b = ndarray::array![[7i32, 8, 9], [10, 11, 12]];
        let za = Array::from_ndarray(&a.view().into_dyn(), &[2, 3]).unwrap();
        let zb = Array::from_ndarray(&b.view().into_dyn(), &[2, 3]).unwrap();
        let actual = stack(vec![&za, &zb], 1).data().to_ndarray::<i32>().unwrap();
        let expected = ndarray::stack(ndarray::Axis(1), &[a.view(), b.view()]).unwrap();
        assert_eq!(actual, expected.into_dyn());
    }

    // stack three 1D i32 arrays along axis 0 → shape [3, N]
    #[test]
    fn test_i32_three_arrays() {
        let a = ndarray::array![1i32, 2, 3];
        let b = ndarray::array![4i32, 5, 6];
        let c = ndarray::array![7i32, 8, 9];
        let za = Array::from_ndarray(&a.view().into_dyn(), &[3]).unwrap();
        let zb = Array::from_ndarray(&b.view().into_dyn(), &[3]).unwrap();
        let zc = Array::from_ndarray(&c.view().into_dyn(), &[3]).unwrap();
        let actual = stack([za, zb, zc], 0).data().to_ndarray::<i32>().unwrap();
        let expected = ndarray::stack(ndarray::Axis(0), &[a.view(), b.view(), c.view()]).unwrap();
        assert_eq!(actual, expected.into_dyn());
    }

    // stack two 1D f32 arrays along axis 0 → shape [2, N]
    #[test]
    fn test_f32_1d_axis0() {
        let a = ndarray::array![1.0f32, 2.0, 3.0, 4.0];
        let b = ndarray::array![5.0f32, 6.0, 7.0, 8.0];
        let za = Array::from_ndarray(&a.view().into_dyn(), &[4]).unwrap();
        let zb = Array::from_ndarray(&b.view().into_dyn(), &[4]).unwrap();
        let actual = stack([za, zb], 0).data().to_ndarray::<f32>().unwrap();
        let expected = ndarray::stack(ndarray::Axis(0), &[a.view(), b.view()]).unwrap();
        assert_eq!(actual, expected.into_dyn());
    }

    // stack two 2D f32 arrays along axis 1 → shape [M, 2, N]
    #[test]
    fn test_f32_2d_axis1() {
        let a = ndarray::array![[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let b = ndarray::array![[7.0f32, 8.0], [9.0, 10.0], [11.0, 12.0]];
        let za = Array::from_ndarray(&a.view().into_dyn(), &[3, 2]).unwrap();
        let zb = Array::from_ndarray(&b.view().into_dyn(), &[3, 2]).unwrap();
        let actual = stack([&za, &zb], 1).data().to_ndarray::<f32>().unwrap();
        let expected = ndarray::stack(ndarray::Axis(1), &[a.view(), b.view()]).unwrap();
        assert_eq!(actual, expected.into_dyn());
    }

    // stack three f32 arrays along axis 0 with multi-block layout
    #[test]
    fn test_f32_three_arrays_multi_block() {
        let a = ndarray::array![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = ndarray::array![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
        let c = ndarray::array![13.0f32, 14.0, 15.0, 16.0, 17.0, 18.0];
        let za = Array::from_ndarray(&a.view().into_dyn(), &[2]).unwrap();
        let zb = Array::from_ndarray(&b.view().into_dyn(), &[2]).unwrap();
        let zc = Array::from_ndarray(&c.view().into_dyn(), &[2]).unwrap();
        let actual = stack((za, zb, &zc), 0).data().to_ndarray::<f32>().unwrap();
        let expected = ndarray::stack(ndarray::Axis(0), &[a.view(), b.view(), c.view()]).unwrap();
        assert_eq!(actual, expected.into_dyn());
    }

    #[test]
    #[should_panic]
    fn test_shape_mismatch_panics() {
        let a = Array::from_ndarray(&ndarray::array![1i32, 2].view().into_dyn(), &[2]).unwrap();
        let b = Array::from_ndarray(&ndarray::array![1i32, 2, 3].view().into_dyn(), &[3]).unwrap();
        let _ = stack(vec![a, b], 0);
    }

    #[test]
    #[should_panic]
    fn test_dtype_mismatch_panics() {
        let a = Array::from_ndarray(&ndarray::array![1i32, 2].view().into_dyn(), &[2]).unwrap();
        let b = Array::from_ndarray(&ndarray::array![1.0f32, 2.0].view().into_dyn(), &[2]).unwrap();
        let _ = stack((a, b), 0);
    }

    #[test]
    #[should_panic]
    fn test_empty_panics() {
        let _ = stack(Vec::<Array<crate::storage::Owned>>::new(), 0);
    }
}
