use std::io;
use std::ops::{Not, Range};

use crate::array::{Array, BlockShapeTag, BlocksLayout};
use crate::codec::{DecoderParams, EncoderParams, ReadContext};
use crate::dtype::Dtype;
use crate::iter::NdIter;
use crate::iter::strides::{NdIterExtStridesPtr, NdIterExtStridesPtrMut};
use crate::storage::ArrayStorage;
use crate::util::{ArraySequence, DimArray, default_strides, dim_arr};

/// Join a sequence of arrays along a *new* axis.
///
/// All input arrays must have identical shapes and the same [`Dtype`].  A new axis is inserted at
/// position `axis` in the output, whose size equals the number of input arrays.  The result is a
/// lazy [`Array`] whose data is read on demand; no copy is made at construction time.
///
/// This is the array-axis analogue of NumPy's `numpy.stack`.  Unlike [`concatenate`], which joins
/// along an *existing* axis, `stack` introduces a *new* dimension, so the output has one more
/// dimension than the inputs.
///
/// [`concatenate`]: crate::ops::concatenate
///
/// # Arguments
///
/// * `arrays` — any [`ArraySequence`]: a `Vec`, a slice, or a tuple of up to ten arrays.
///   All elements must have identical dtypes and identical shapes.
/// * `axis` — the position in the output array at which the new axis is inserted.  Must be in the
///   range `0..=ndim`, where `ndim` is the number of dimensions of the input arrays.
///
/// # Panics
///
/// Panics if any of the following conditions hold (the underlying [`Stack::new`] returns an error,
/// which this function unwraps):
///
/// * `arrays` is empty.
/// * `axis` is out of bounds (> number of dimensions of the input arrays).
/// * Any two arrays differ in dtype.
/// * Any two arrays differ in shape.
/// * Inserting the new axis would exceed the maximum supported number of dimensions.
///
/// # Examples
///
/// ```
/// use zix::array::Array;
/// use zix::ops::stack;
/// use zix::storage::ArrayParams;
///
/// // Stack two 1-D arrays along axis 0 — analogous to np.stack([a, b], axis=0)
/// // Result shape: [2, N]
/// let a = Array::from_ndarray(&ndarray::array![1i32, 2, 3].view().into_dyn(), ArrayParams::default()).unwrap();
/// let b = Array::from_ndarray(&ndarray::array![4i32, 5, 6].view().into_dyn(), ArrayParams::default()).unwrap();
/// let c = stack(vec![a, b], 0);
/// assert_eq!(c.shape(), &[2, 3]);
///
/// // Stack two 1-D arrays along axis 1 — analogous to np.stack([a, b], axis=1)
/// // Result shape: [N, 2]
/// let a = Array::from_ndarray(&ndarray::array![1i32, 2, 3].view().into_dyn(), ArrayParams::default()).unwrap();
/// let b = Array::from_ndarray(&ndarray::array![4i32, 5, 6].view().into_dyn(), ArrayParams::default()).unwrap();
/// let c = stack(vec![a, b], 1);
/// assert_eq!(c.shape(), &[3, 2]);
///
/// // Stack three 2-D arrays along a new leading axis
/// // Result shape: [3, M, N]
/// let a = Array::from_ndarray(&ndarray::array![[1i32, 2], [3, 4]].view().into_dyn(), ArrayParams::default()).unwrap();
/// let b = Array::from_ndarray(&ndarray::array![[5i32, 6], [7, 8]].view().into_dyn(), ArrayParams::default()).unwrap();
/// let c_arr = Array::from_ndarray(&ndarray::array![[9i32, 10], [11, 12]].view().into_dyn(), ArrayParams::default()).unwrap();
/// let out = stack(vec![a, b, c_arr], 0);
/// assert_eq!(out.shape(), &[3, 2, 2]);
/// ```
#[track_caller]
pub fn stack<ArraysT>(arrays: ArraysT, axis: usize) -> Array<Stack<ArraysT>>
where
    ArraysT: ArraySequence,
{
    Array::from_storage(Stack::new(arrays, axis).unwrap())
}

/// Lazy storage type returned by [`stack`].
///
/// Holds the input arrays and the bookkeeping needed to serve arbitrary read requests.  See
/// [`stack`] for the full description, accepted inputs, error conditions, and examples.
pub struct Stack<ArraysT> {
    arrays: ArraysT,
    stack_axis: usize,

    dtype: Dtype,
    shape: DimArray<u64>,
    blocks_layout: BlocksLayout,
}
impl<ArraysT> Stack<ArraysT> {
    pub fn new(arrays: ArraysT, axis: usize) -> io::Result<Self>
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
                    format!("cannot stack arrays of different shapes: {shape0:?} != {shape_i:?}"),
                ));
            }
            let dtype_i = arrays.dtype(arr);
            if dtype_i != dtype {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("cannot stack arrays of different dtypes: {dtype:?} != {dtype_i:?}"),
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
        new_shape.insert(axis, narrays as u64);

        let mut b_layout = arrays.blocks_layout(0).clone();
        b_layout.block_shape_hint.insert(axis, 1);
        b_layout.block_shape_tag.insert(axis, BlockShapeTag::Any);
        b_layout.preferred_read_block_shape.insert(axis, 1);

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
    fn shape(&self) -> &[u64] {
        &self.shape
    }

    fn dtype(&self) -> &Dtype {
        &self.dtype
    }

    fn read_data(
        &self,
        index: &[Range<u64>],
        buf: &mut [u8],
        context: &ReadContext,
    ) -> io::Result<()> {
        assert_eq!(index.len(), self.shape.len());
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
                let mut iter = NdIter::new(
                    &arr_range_shape,
                    (
                        NdIterExtStridesPtr::new(arr_strides, arr_buf.as_ptr()),
                        NdIterExtStridesPtrMut::new(out_strides, unsafe {
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

    fn blocks_layout(&self) -> &BlocksLayout {
        &self.blocks_layout
    }

    fn codec_params(&self) -> (&EncoderParams, &DecoderParams) {
        self.arrays.codec_params(0)
    }
}

#[cfg(test)]
mod tests {
    use crate::array::Array;
    use crate::block::BlockSize;
    use crate::ops::stack;
    use crate::storage::ArrayParams;

    fn arr_params(block_shape: &[usize]) -> ArrayParams {
        ArrayParams {
            block_shape: Some(block_shape.iter().map(|&x| x as BlockSize).collect()),
            ..ArrayParams::default()
        }
    }

    // stack two 1D i32 arrays along axis 0 → shape [2, N]
    #[test]
    fn test_i32_1d_axis0() {
        let a = ndarray::array![1i32, 2, 3, 4];
        let b = ndarray::array![5i32, 6, 7, 8];
        let za = Array::from_ndarray(&a.view().into_dyn(), arr_params(&[4])).unwrap();
        let zb = Array::from_ndarray(&b.view().into_dyn(), arr_params(&[4])).unwrap();
        let actual = stack(vec![za, zb], 0).data().to_ndarray::<i32>().unwrap();
        let expected = ndarray::stack(ndarray::Axis(0), &[a.view(), b.view()]).unwrap();
        assert_eq!(actual, expected.into_dyn());
    }

    // stack two 1D i32 arrays along axis 1 → shape [N, 2]
    #[test]
    fn test_i32_1d_axis1() {
        let a = ndarray::array![1i32, 2, 3];
        let b = ndarray::array![4i32, 5, 6];
        let za = Array::from_ndarray(&a.view().into_dyn(), arr_params(&[3])).unwrap();
        let zb = Array::from_ndarray(&b.view().into_dyn(), arr_params(&[3])).unwrap();
        let actual = stack([za, zb], 1).data().to_ndarray::<i32>().unwrap();
        let expected = ndarray::stack(ndarray::Axis(1), &[a.view(), b.view()]).unwrap();
        assert_eq!(actual, expected.into_dyn());
    }

    // stack two 2D i32 arrays along axis 0 → shape [2, M, N]
    #[test]
    fn test_i32_2d_axis0() {
        let a = ndarray::array![[1i32, 2, 3], [4, 5, 6]];
        let b = ndarray::array![[7i32, 8, 9], [10, 11, 12]];
        let za = Array::from_ndarray(&a.view().into_dyn(), arr_params(&[2, 3])).unwrap();
        let zb = Array::from_ndarray(&b.view().into_dyn(), arr_params(&[2, 3])).unwrap();
        let actual = stack((za, &zb), 0).data().to_ndarray::<i32>().unwrap();
        let expected = ndarray::stack(ndarray::Axis(0), &[a.view(), b.view()]).unwrap();
        assert_eq!(actual, expected.into_dyn());
    }

    // stack two 2D i32 arrays along axis 1 → shape [M, 2, N]
    #[test]
    fn test_i32_2d_axis1() {
        let a = ndarray::array![[1i32, 2, 3], [4, 5, 6]];
        let b = ndarray::array![[7i32, 8, 9], [10, 11, 12]];
        let za = Array::from_ndarray(&a.view().into_dyn(), arr_params(&[2, 3])).unwrap();
        let zb = Array::from_ndarray(&b.view().into_dyn(), arr_params(&[2, 3])).unwrap();
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
        let za = Array::from_ndarray(&a.view().into_dyn(), arr_params(&[3])).unwrap();
        let zb = Array::from_ndarray(&b.view().into_dyn(), arr_params(&[3])).unwrap();
        let zc = Array::from_ndarray(&c.view().into_dyn(), arr_params(&[3])).unwrap();
        let actual = stack([za, zb, zc], 0).data().to_ndarray::<i32>().unwrap();
        let expected = ndarray::stack(ndarray::Axis(0), &[a.view(), b.view(), c.view()]).unwrap();
        assert_eq!(actual, expected.into_dyn());
    }

    // stack two 1D f32 arrays along axis 0 → shape [2, N]
    #[test]
    fn test_f32_1d_axis0() {
        let a = ndarray::array![1.0f32, 2.0, 3.0, 4.0];
        let b = ndarray::array![5.0f32, 6.0, 7.0, 8.0];
        let za = Array::from_ndarray(&a.view().into_dyn(), arr_params(&[4])).unwrap();
        let zb = Array::from_ndarray(&b.view().into_dyn(), arr_params(&[4])).unwrap();
        let actual = stack([za, zb], 0).data().to_ndarray::<f32>().unwrap();
        let expected = ndarray::stack(ndarray::Axis(0), &[a.view(), b.view()]).unwrap();
        assert_eq!(actual, expected.into_dyn());
    }

    // stack two 2D f32 arrays along axis 1 → shape [M, 2, N]
    #[test]
    fn test_f32_2d_axis1() {
        let a = ndarray::array![[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let b = ndarray::array![[7.0f32, 8.0], [9.0, 10.0], [11.0, 12.0]];
        let za = Array::from_ndarray(&a.view().into_dyn(), arr_params(&[3, 2])).unwrap();
        let zb = Array::from_ndarray(&b.view().into_dyn(), arr_params(&[3, 2])).unwrap();
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
        let za = Array::from_ndarray(&a.view().into_dyn(), arr_params(&[2])).unwrap();
        let zb = Array::from_ndarray(&b.view().into_dyn(), arr_params(&[2])).unwrap();
        let zc = Array::from_ndarray(&c.view().into_dyn(), arr_params(&[2])).unwrap();
        let actual = stack((za, zb, &zc), 0).data().to_ndarray::<f32>().unwrap();
        let expected = ndarray::stack(ndarray::Axis(0), &[a.view(), b.view(), c.view()]).unwrap();
        assert_eq!(actual, expected.into_dyn());
    }

    #[test]
    #[should_panic]
    fn test_shape_mismatch_panics() {
        let a = Array::from_ndarray(
            &ndarray::array![1i32, 2].view().into_dyn(),
            arr_params(&[2]),
        )
        .unwrap();
        let b = Array::from_ndarray(
            &ndarray::array![1i32, 2, 3].view().into_dyn(),
            arr_params(&[3]),
        )
        .unwrap();
        let _ = stack(vec![a, b], 0);
    }

    #[test]
    #[should_panic]
    fn test_dtype_mismatch_panics() {
        let a = Array::from_ndarray(
            &ndarray::array![1i32, 2].view().into_dyn(),
            arr_params(&[2]),
        )
        .unwrap();
        let b = Array::from_ndarray(
            &ndarray::array![1.0f32, 2.0].view().into_dyn(),
            arr_params(&[2]),
        )
        .unwrap();
        let _ = stack((a, b), 0);
    }

    #[test]
    #[should_panic]
    fn test_empty_panics() {
        let _ = stack(Vec::<Array<crate::storage::Owned>>::new(), 0);
    }
}
