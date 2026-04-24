use std::ops::{Not, Range};

use crate::array::Array;
use crate::codec::ReadContext;
use crate::dtype::Dtype;
use crate::error::{bail, check_get_buffer_size, check_get_range, ensure, Result};
use crate::storage::{ArrayStorage, ArrayStorageSpec, BlocksLayout};
use crate::util::{default_strides, dim_arr, nd_copy, ArraySequence, DimArray};

/// Join a sequence of arrays along an existing axis.
///
/// All input arrays must have the same number of dimensions and the same size on every axis
/// *except* the concatenation axis, along which their sizes may differ.  All arrays must share
/// the same [`Dtype`].  The result is a lazy [`Array`] whose data is read on demand; no copy is
/// made at construction time.
///
/// This is the array-axis analogue of NumPy's `numpy.concatenate`.  Unlike [`stack`], which
/// introduces a *new* axis, `concatenate` joins along an *existing* one, so the output has the
/// same number of dimensions as the inputs.
///
/// [`stack`]: crate::ops::stack
///
/// # Arguments
///
/// * `arrays` — any [`ArraySequence`]: a `Vec`, a slice, or a tuple of up to ten arrays.
///   All elements must have identical dtypes and identical shapes on every axis other than `axis`.
/// * `axis` — the axis along which to concatenate.  Must be less than the number of dimensions
///   of the input arrays.
///
/// # Panics
///
/// Panics if any of the following conditions hold (the underlying [`Concatenate::new`] returns an
/// error, which this function unwraps):
///
/// * `arrays` is empty.
/// * `axis` is out of bounds (≥ number of dimensions of the arrays).
/// * Any two arrays differ in dtype.
/// * Any two arrays differ in shape on an axis *other* than `axis`.
///
/// # Examples
///
/// ```
/// use zix::{Array, ArrayParams};
/// use zix::ops::concatenate;
///
/// // 1-D concatenation — analogous to np.concatenate([a, b])
/// let a = Array::from_ndarray(&ndarray::array![1i32, 2, 3].view().into_dyn(), ArrayParams::default()).unwrap();
/// let b = Array::from_ndarray(&ndarray::array![4i32, 5].view().into_dyn(), ArrayParams::default()).unwrap();
/// let c = concatenate(vec![a, b], 0);
/// assert_eq!(c.shape(), &[5]);
///
/// // 2-D concatenation along axis 0 (stack rows)
/// let a = Array::from_ndarray(&ndarray::array![[1i32, 2], [3, 4]].view().into_dyn(), ArrayParams::default()).unwrap();
/// let b = Array::from_ndarray(&ndarray::array![[5i32, 6]].view().into_dyn(), ArrayParams::default()).unwrap();
/// let c = concatenate(vec![a, b], 0);
/// assert_eq!(c.shape(), &[3, 2]);
///
/// // 2-D concatenation along axis 1 (stack columns)
/// let a = Array::from_ndarray(&ndarray::array![[1i32, 2], [3, 4]].view().into_dyn(), ArrayParams::default()).unwrap();
/// let b = Array::from_ndarray(&ndarray::array![[5i32, 6, 7], [8, 9, 10]].view().into_dyn(), ArrayParams::default()).unwrap();
/// let c = concatenate(vec![a, b], 1);
/// assert_eq!(c.shape(), &[2, 5]);
/// ```
#[track_caller]
pub fn concatenate<ArraysT>(arrays: ArraysT, axis: usize) -> Array<Concatenate<ArraysT>>
where
    ArraysT: ArraySequence,
{
    Array::from_storage(Concatenate::new(arrays, axis).unwrap())
}

/// Lazy storage type returned by [`concatenate`].
///
/// Holds the input arrays and the bookkeeping needed to serve arbitrary read requests.  See
/// [`concatenate`] for the full description, accepted inputs, error conditions, and examples.
pub struct Concatenate<ArraysT> {
    arrays: ArraysT,
    concat_axis: usize,
    borders: Vec<u64>,

    dtype: Dtype,
    shape: DimArray<u64>,
    blocks_layout: BlocksLayout,
}
impl<ArraysT> Concatenate<ArraysT> {
    pub fn new(arrays: ArraysT, axis: usize) -> Result<Self>
    where
        ArraysT: ArraySequence,
    {
        let narrays = arrays.narrays();
        ensure!(
            narrays > 0,
            InvalidShapeOperation,
            "cannot concatenate zero arrays"
        );

        let shape0 = arrays.shape(0);
        let mut shape: DimArray<_> = shape0.try_into().unwrap();
        ensure!(
            axis < shape.len(),
            InvalidShapeOperation,
            "concat axis {axis} out of bounds for arrays with ndim {}",
            shape.len()
        );
        shape[axis] = 0;

        let mut borders = Vec::with_capacity(narrays);
        let dtype = arrays.dtype(0);
        for arr in 0..narrays {
            let shape_i = arrays.shape(arr);
            if shape.len() != shape_i.len()
                || shape0
                    .iter()
                    .zip(shape_i)
                    .enumerate()
                    .any(|(dim, (&s0, &s_i))| dim != axis && s0 != s_i)
            {
                bail!(
                    InvalidShapeOperation,
                    "cannot stack arrays of different shapes: {shape0:?} != {shape_i:?}"
                );
            }
            let dtype_i = arrays.dtype(arr);
            ensure!(
                dtype_i == dtype,
                UnsupportedDtype,
                "cannot stack arrays of different dtypes: {dtype:?} != {dtype_i:?}"
            );

            shape[axis] += shape_i[axis];
            borders.push(shape[axis]);
        }

        Ok(Self {
            dtype: dtype.clone(),
            shape,
            blocks_layout: arrays.spec(0).blocks_layout.clone(),
            arrays,
            concat_axis: axis,
            borders,
        })
    }
}
impl<ArraysT> ArrayStorage for Concatenate<ArraysT>
where
    ArraysT: ArraySequence,
{
    /// Fills `buf` with a C-order slice of the concatenated array described by `index`.
    ///
    /// `borders` stores the cumulative end positions of each sub-array along `concat_axis`, so
    /// sub-array `i` owns the range `[borders[i-1], borders[i])` (with `borders[-1] == 0`).
    /// Only the sub-arrays that overlap with `index[concat_axis]` are read.
    ///
    /// To skip leading non-overlapping arrays efficiently, the first overlapping sub-array is
    /// located with a linear scan for small `borders` slices or a binary search otherwise.
    /// The loop then runs forward and breaks as soon as an array starts past the requested range.
    ///
    /// Each overlapping sub-array is read with a local (array-relative) index on the concat axis.
    /// When all dimensions before `concat_axis` have size ≤ 1 the output buffer is contiguous for
    /// each sub-array ("in-place"), so the data is written directly at the right byte offset.
    /// Otherwise each sub-array is read into a temporary buffer and scattered into `buf` with
    /// `NdIter`, using the full output strides for dimensions before `concat_axis` and the
    /// sub-array strides for dimensions at and after it.
    fn read_data(&self, index: &[Range<u64>], buf: &mut [u8], context: &ReadContext) -> Result<()> {
        check_get_range(&self.shape, index)?;
        check_get_buffer_size(index, &self.dtype, buf)?;

        const BINARY_SEARCH_THRESHOLD: usize = 32;

        let itemsize = self.dtype.itemsize() as usize;

        let output_shape = dim_arr(index.len(), |d| (index[d].end - index[d].start) as usize);
        let output_strides = default_strides(&output_shape, itemsize);
        let concat_stride = output_strides[self.concat_axis];
        // When all dims before concat_axis have size <=1 each array's data is contiguous in buf.
        let in_place = output_shape.iter().take(self.concat_axis).all(|&s| s <= 1);
        let mut tmp_buf = in_place
            .not()
            .then(|| context.tmp_buf(0, self.dtype.alignment()));

        let req_start = index[self.concat_axis].start;
        let req_end = index[self.concat_axis].end;

        // Find the first sub-array whose end exceeds req_start (i.e. the first that may overlap).
        let first_arr = if self.borders.len() < BINARY_SEARCH_THRESHOLD {
            self.borders
                .iter()
                .position(|&b| b > req_start)
                .unwrap_or(self.borders.len())
        } else {
            self.borders.partition_point(|&b| b <= req_start)
        };

        for arr in first_arr..self.borders.len() {
            let arr_start = if arr == 0 { 0 } else { self.borders[arr - 1] };
            let arr_end = self.borders[arr];
            if arr_start >= req_end {
                break;
            }

            let overlap_start = req_start.max(arr_start);
            let overlap_end = req_end.min(arr_end);
            let local_start = overlap_start - arr_start;
            let local_end = overlap_end - arr_start;
            let buf_concat_offset = (overlap_start - req_start) as usize;

            // Sub-index into array `arr`: same as `index` but concat axis uses local coords.
            let sub_index = dim_arr(index.len(), |d| {
                if d == self.concat_axis {
                    local_start..local_end
                } else {
                    index[d].clone()
                }
            });
            let sub_shape = dim_arr(index.len(), |d| {
                (sub_index[d].end - sub_index[d].start) as usize
            });
            let sub_size_bytes = sub_shape.iter().product::<usize>() * itemsize;
            let buf_offset = buf_concat_offset * concat_stride;

            let read_buf = if in_place {
                // Data lands contiguously at the right offset — read directly into buf.
                &mut buf[buf_offset..buf_offset + sub_size_bytes]
            } else {
                // Read into tmp_buf then scatter into buf using strided copy.
                let tmp_buf = tmp_buf.as_mut().unwrap();
                tmp_buf.set_len(sub_size_bytes);
                tmp_buf.as_mut_slice()
            };
            self.arrays.read_data(arr, &sub_index, read_buf, context)?;

            if !in_place {
                // Scatter from tmp_buf into buf.
                // src: C-strides of sub_shape.
                // dst: output_strides for dims before concat_axis (wider due to full output width),
                //      sub_strides for dims at/after (sizes match the output there).
                let sub_strides = default_strides(&sub_shape, itemsize);
                let dst_strides = dim_arr(index.len(), |d| {
                    if d < self.concat_axis {
                        output_strides[d]
                    } else {
                        sub_strides[d]
                    }
                });

                unsafe {
                    nd_copy(
                        read_buf.as_ptr(),
                        buf.as_mut_ptr().add(buf_offset),
                        &sub_shape,
                        &sub_strides,
                        &dst_strides,
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
    fn spec(&self) -> ArrayStorageSpec<'_> {
        ArrayStorageSpec {
            blocks_layout: &self.blocks_layout,
            ..self.arrays.spec(0)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::array::Array;
    use crate::ops::concatenate;
    use crate::util::arr_params;

    // 1D i32: concatenate two arrays of equal size along axis 0 (in-place path)
    #[test]
    fn test_i32_1d_equal_sizes() {
        let a = ndarray::array![1i32, 2, 3];
        let b = ndarray::array![4i32, 5, 6];
        let za = Array::from_ndarray(&a.view().into_dyn(), arr_params(&[3])).unwrap();
        let zb = Array::from_ndarray(&b.view().into_dyn(), arr_params(&[3])).unwrap();
        let actual = concatenate(vec![za, zb], 0)
            .to_ndarray::<i32>()
            .unwrap();
        let expected = ndarray::concatenate(ndarray::Axis(0), &[a.view(), b.view()]).unwrap();
        assert_eq!(actual, expected.into_dyn());
    }

    // 1D i32: concatenate two arrays of unequal sizes along axis 0
    #[test]
    fn test_i32_1d_unequal_sizes() {
        let a = ndarray::array![1i32, 2];
        let b = ndarray::array![3i32, 4, 5, 6];
        let za = Array::from_ndarray(&a.view().into_dyn(), arr_params(&[2])).unwrap();
        let zb = Array::from_ndarray(&b.view().into_dyn(), arr_params(&[4])).unwrap();
        let actual = concatenate(vec![za, zb], 0)
            .to_ndarray::<i32>()
            .unwrap();
        let expected = ndarray::concatenate(ndarray::Axis(0), &[a.view(), b.view()]).unwrap();
        assert_eq!(actual, expected.into_dyn());
    }

    // 1D i32: concatenate three arrays along axis 0
    #[test]
    fn test_i32_1d_three_arrays() {
        let a = ndarray::array![1i32, 2];
        let b = ndarray::array![3i32, 4, 5];
        let c = ndarray::array![6i32];
        let za = Array::from_ndarray(&a.view().into_dyn(), arr_params(&[2])).unwrap();
        let zb = Array::from_ndarray(&b.view().into_dyn(), arr_params(&[3])).unwrap();
        let zc = Array::from_ndarray(&c.view().into_dyn(), arr_params(&[1])).unwrap();
        let actual = concatenate(vec![za, zb, zc], 0)
            .to_ndarray::<i32>()
            .unwrap();
        let expected =
            ndarray::concatenate(ndarray::Axis(0), &[a.view(), b.view(), c.view()]).unwrap();
        assert_eq!(actual, expected.into_dyn());
    }

    // 2D i32: concatenate along axis 0 — in-place path
    #[test]
    fn test_i32_2d_axis0() {
        let a = ndarray::array![[1i32, 2, 3], [4, 5, 6]];
        let b = ndarray::array![[7i32, 8, 9]];
        let za = Array::from_ndarray(&a.view().into_dyn(), arr_params(&[2, 3])).unwrap();
        let zb = Array::from_ndarray(&b.view().into_dyn(), arr_params(&[1, 3])).unwrap();
        let actual = concatenate(vec![za, zb], 0)
            .to_ndarray::<i32>()
            .unwrap();
        let expected = ndarray::concatenate(ndarray::Axis(0), &[a.view(), b.view()]).unwrap();
        assert_eq!(actual, expected.into_dyn());
    }

    // 2D i32: concatenate along axis 1 — scatter path
    #[test]
    fn test_i32_2d_axis1() {
        let a = ndarray::array![[1i32, 2], [3, 4], [5, 6]];
        let b = ndarray::array![[7i32, 8, 9], [10, 11, 12], [13, 14, 15]];
        let za = Array::from_ndarray(&a.view().into_dyn(), arr_params(&[3, 2])).unwrap();
        let zb = Array::from_ndarray(&b.view().into_dyn(), arr_params(&[3, 3])).unwrap();
        let actual = concatenate(vec![za, zb], 1)
            .to_ndarray::<i32>()
            .unwrap();
        let expected = ndarray::concatenate(ndarray::Axis(1), &[a.view(), b.view()]).unwrap();
        assert_eq!(actual, expected.into_dyn());
    }

    // 2D i32: three arrays along axis 1 with unequal sizes — scatter path
    #[test]
    fn test_i32_2d_axis1_three_unequal() {
        let a = ndarray::array![[1i32], [2]];
        let b = ndarray::array![[3i32, 4, 5], [6, 7, 8]];
        let c = ndarray::array![[9i32, 10], [11, 12]];
        let za = Array::from_ndarray(&a.view().into_dyn(), arr_params(&[2, 1])).unwrap();
        let zb = Array::from_ndarray(&b.view().into_dyn(), arr_params(&[2, 3])).unwrap();
        let zc = Array::from_ndarray(&c.view().into_dyn(), arr_params(&[2, 2])).unwrap();
        let actual = concatenate(vec![za, zb, zc], 1)
            .to_ndarray::<i32>()
            .unwrap();
        let expected =
            ndarray::concatenate(ndarray::Axis(1), &[a.view(), b.view(), c.view()]).unwrap();
        assert_eq!(actual, expected.into_dyn());
    }

    // 1D f32: concatenate two arrays along axis 0
    #[test]
    fn test_f32_1d_axis0() {
        let a = ndarray::array![1.0f32, 2.0, 3.0];
        let b = ndarray::array![4.0f32, 5.0];
        let za = Array::from_ndarray(&a.view().into_dyn(), arr_params(&[3])).unwrap();
        let zb = Array::from_ndarray(&b.view().into_dyn(), arr_params(&[2])).unwrap();
        let actual = concatenate(vec![za, zb], 0)
            .to_ndarray::<f32>()
            .unwrap();
        let expected = ndarray::concatenate(ndarray::Axis(0), &[a.view(), b.view()]).unwrap();
        assert_eq!(actual, expected.into_dyn());
    }

    // 2D f32: concatenate along axis 1 — scatter path
    #[test]
    fn test_f32_2d_axis1() {
        let a = ndarray::array![[1.0f32, 2.0], [3.0, 4.0]];
        let b = ndarray::array![[5.0f32, 6.0, 7.0], [8.0, 9.0, 10.0]];
        let za = Array::from_ndarray(&a.view().into_dyn(), arr_params(&[2, 2])).unwrap();
        let zb = Array::from_ndarray(&b.view().into_dyn(), arr_params(&[2, 3])).unwrap();
        let actual = concatenate(vec![za, zb], 1)
            .to_ndarray::<f32>()
            .unwrap();
        let expected = ndarray::concatenate(ndarray::Axis(1), &[a.view(), b.view()]).unwrap();
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
            &ndarray::array![[1i32, 2]].view().into_dyn(),
            arr_params(&[1, 2]),
        )
        .unwrap();
        let _ = concatenate(vec![a, b], 0);
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
        let _ = concatenate((a, b), 0);
    }

    #[test]
    #[should_panic]
    fn test_empty_panics() {
        let _ = concatenate(Vec::<Array<crate::storage::Compact>>::new(), 0);
    }
}
