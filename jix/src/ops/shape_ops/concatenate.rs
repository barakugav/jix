use std::ops::{Not, Range};

use crate::codec::ReadContext;
use crate::dtype::Dtype;
use crate::error::{bail, check_get_buffer_size, check_get_range, ensure, Result};
use crate::storage::ArrayStorageSpec;
use crate::util::{default_strides, dim_arr, nd_copy, ArraySequence, DimArray};
use crate::{Array, ArrayStorage, Dimension};

/// Joins a sequence of arrays along an existing axis. See [`Concatenate`] for details and examples.
///
/// # Panics
///
/// Panics if `arrays` is empty, `axis` is out of bounds, dtypes differ, or shapes differ on any
/// axis other than `axis`.
#[track_caller]
pub fn concatenate<ArraysT>(arrays: ArraysT, axis: usize) -> Array<Concatenate<ArraysT>>
where
    ArraysT: ArraySequence,
{
    Array::from_storage(Concatenate::new(arrays, axis).unwrap())
}

/// Joins a sequence of arrays along an existing axis, returned by [`concatenate`].
///
/// All input arrays must have the same number of dimensions and the same size on every axis
/// *except* the concatenation axis, along which their sizes may differ. All arrays must share the
/// same [`Dtype`]. The output has the same number of dimensions as the inputs - unlike
/// [`Stack`](crate::ops::Stack), which introduces a new axis.
///
/// The output dimension type `Concatenate<ArraysT>::Dimension` equals
/// `ArraysT::FirstArrayDimension` - it is taken from the first array in the sequence. This
/// means the static dimension is propagated when all input arrays share a known `Dim<N>`.
///
/// The result is a lazy view; no computation occurs until the array is read.
///
/// # Examples
///
/// ```
/// use jix::Array;
/// use ndarray::array;
///
/// // 1-D: join two arrays end-to-end
/// let a = Array::compact_ndarray(&array![1i32, 2, 3])?;
/// let b = Array::compact_ndarray(&array![4i32, 5])?;
/// let c = jix::ops::concatenate((a, b), 0);
/// assert_eq!(c.shape(), &[5]);
///
/// // 2-D: stack rows (axis 0)
/// let a = Array::compact_ndarray(&array![[1i32, 2], [3, 4]])?;
/// let b = Array::compact_ndarray(&array![[5i32, 6]])?;
/// let c = jix::ops::concatenate((a, b), 0);
/// assert_eq!(c.shape(), &[3, 2]);
///
/// // 2-D: append columns (axis 1)
/// let a = Array::compact_ndarray(&array![[1i32, 2], [3, 4]])?;
/// let b = Array::compact_ndarray(&array![[5i32, 6, 7], [8, 9, 10]])?;
/// let c = jix::ops::concatenate((a, b), 1);
/// assert_eq!(c.shape(), &[2, 5]);
/// # Ok::<(), jix::Error>(())
/// ```
pub struct Concatenate<ArraysT>
where
    ArraysT: ArraySequence,
{
    arrays: ArraysT,
    concat_axis: usize,
    borders: Vec<u64>,

    shape: ArraysT::Dimension,
}
impl<ArraysT> Concatenate<ArraysT>
where
    ArraysT: ArraySequence,
{
    /// Constructs a [`Concatenate`] storage. See the struct docs for semantics and examples.
    pub fn new(arrays: ArraysT, axis: usize) -> Result<Self> {
        let narrays = arrays.narrays();
        ensure!(
            narrays > 0,
            InvalidShapeOperation,
            "cannot concatenate zero arrays"
        );

        let shape0 = arrays.shape(0);
        let mut shape = DimArray::from_slice(shape0).unwrap();
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
                || shape
                    .iter()
                    .zip(shape_i)
                    .enumerate()
                    .any(|(dim, (&s0, &s_i))| dim != axis && s0 != s_i)
            {
                bail!(
                    InvalidShapeOperation,
                    "cannot stack arrays of different shapes: {shape:?} != {shape_i:?}"
                );
            }
            let dtype_i = arrays.dtype(arr);
            ensure!(
                dtype_i == dtype,
                UnsupportedDtype,
                "cannot stack arrays of different dtypes: {dtype} != {dtype_i}"
            );

            shape[axis] += shape_i[axis];
            borders.push(shape[axis]);
        }

        let shape = ArraysT::Dimension::from_slice(&shape);
        Ok(Self {
            shape,
            arrays,
            concat_axis: axis,
            borders,
        })
    }

    /// Constructs an array with [`Concatenate`] storage. See the storage struct docs for semantics and examples.
    pub fn new_array(arrays: ArraysT, axis: usize) -> Result<Array<Self>> {
        Self::new(arrays, axis).map(Array::from_storage)
    }
}
impl<ArraysT> ArrayStorage for Concatenate<ArraysT>
where
    ArraysT: ArraySequence,
{
    type ElementType = ArraysT::ElementType;
    type Dimension = ArraysT::Dimension;

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
    /// When all dimensions before `concat_axis` have size <= 1 the output buffer is contiguous for
    /// each sub-array ("in-place"), so the data is written directly at the right byte offset.
    /// Otherwise each sub-array is read into a temporary buffer and scattered into `buf` with
    /// `NdIter`, using the full output strides for dimensions before `concat_axis` and the
    /// sub-array strides for dimensions at and after it.
    fn read_data(&self, index: &[Range<u64>], buf: &mut [u8], context: &ReadContext) -> Result<()> {
        let dtype = self.dtype();
        check_get_range(self.shape(), index)?;
        let nitems = check_get_buffer_size(index, dtype, buf)?;
        if nitems == 0 {
            return Ok(());
        }

        let itemsize = dtype.itemsize() as usize;

        let output_shape = dim_arr(index.len(), |dim| {
            (index[dim].end - index[dim].start) as usize
        });
        let output_strides = default_strides(&output_shape, itemsize);
        let concat_stride = output_strides[self.concat_axis];
        // When all dims before concat_axis have size <=1 each array's data is contiguous in buf.
        let in_place = output_shape.iter().take(self.concat_axis).all(|&s| s <= 1);
        let mut tmp_buf = in_place
            .not()
            .then(|| context.tmp_buf(0, dtype.alignment()));

        let req_start = index[self.concat_axis].start;
        let req_end = index[self.concat_axis].end;

        // Find the first sub-array whose end exceeds req_start (i.e. the first that may overlap).
        const BINARY_SEARCH_THRESHOLD: usize = 32;
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
            let sub_index = dim_arr(index.len(), |dim| {
                if dim == self.concat_axis {
                    local_start..local_end
                } else {
                    index[dim].clone()
                }
            });
            let sub_shape = Self::Dimension::from_fn(index.len(), |dim| {
                (sub_index[dim].end - sub_index[dim].start) as u64
            });
            let sub_size_bytes = sub_shape.as_slice().iter().product::<u64>() as usize * itemsize;
            let buf_offset = buf_concat_offset * concat_stride;

            let read_buf = if in_place {
                // Data lands contiguously at the right offset - read directly into buf.
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
                let sub_strides = default_strides(sub_shape.as_slice(), itemsize as u64);
                let dst_strides = dim_arr(index.len(), |dim| {
                    if dim < self.concat_axis {
                        output_strides[dim] as u64
                    } else {
                        sub_strides[dim]
                    }
                });

                unsafe {
                    nd_copy(
                        read_buf.as_ptr(),
                        buf.as_mut_ptr().add(buf_offset),
                        sub_shape,
                        &sub_strides,
                        &dst_strides,
                        itemsize,
                    )
                };
            }
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
    fn spec(&self) -> ArrayStorageSpec<'_> {
        self.arrays.spec(0)
    }

    crate::ops::impl_dimension_change_default!();
    crate::ops::impl_element_type_change_default!();
}

#[cfg(test)]
mod tests {
    use ndarray::array;
    use proptest::prelude::*;

    use crate::ops::concatenate;
    use crate::storage::Compact;
    use crate::util::{shape_strategy, ScalarStrategy};
    use crate::{Array, DimDyn, Ty};

    // 1D i32: concatenate two arrays of equal size along axis 0 (in-place path)
    #[test]
    fn test_i32_1d_equal_sizes() {
        let a = array![1i32, 2, 3];
        let b = array![4i32, 5, 6];
        let za = Array::compact_ndarray(&a).unwrap();
        let zb = Array::compact_ndarray(&b).unwrap();
        let actual = concatenate((za, zb), 0).to_ndarray().unwrap();
        let expected = ndarray::concatenate(ndarray::Axis(0), &[a.view(), b.view()]).unwrap();
        assert_eq!(actual, expected);
    }

    // 1D i32: concatenate two arrays of unequal sizes along axis 0
    #[test]
    fn test_i32_1d_unequal_sizes() {
        let a = array![1i32, 2];
        let b = array![3i32, 4, 5, 6];
        let za = Array::compact_ndarray(&a).unwrap();
        let zb = Array::compact_ndarray(&b).unwrap();
        let actual = concatenate((za, zb), 0).to_ndarray().unwrap();
        let expected = ndarray::concatenate(ndarray::Axis(0), &[a.view(), b.view()]).unwrap();
        assert_eq!(actual, expected);
    }

    // 1D i32: concatenate three arrays along axis 0
    #[test]
    fn test_i32_1d_three_arrays() {
        let a = array![1i32, 2];
        let b = array![3i32, 4, 5];
        let c = array![6i32];
        let za = Array::compact_ndarray(&a).unwrap();
        let zb = Array::compact_ndarray(&b).unwrap();
        let zc = Array::compact_ndarray(&c).unwrap();
        let actual = concatenate((za, zb, zc), 0).to_ndarray().unwrap();
        let expected =
            ndarray::concatenate(ndarray::Axis(0), &[a.view(), b.view(), c.view()]).unwrap();
        assert_eq!(actual, expected);
    }

    // 2D i32: concatenate along axis 0 - in-place path
    #[test]
    fn test_i32_2d_axis0() {
        let a = array![[1i32, 2, 3], [4, 5, 6]];
        let b = array![[7i32, 8, 9]];
        let za = Array::compact_ndarray(&a).unwrap();
        let zb = Array::compact_ndarray(&b).unwrap();
        let actual = concatenate((za, zb), 0).to_ndarray().unwrap();
        let expected = ndarray::concatenate(ndarray::Axis(0), &[a.view(), b.view()]).unwrap();
        assert_eq!(actual, expected);
    }

    // 2D i32: concatenate along axis 1 - scatter path
    #[test]
    fn test_i32_2d_axis1() {
        let a = array![[1i32, 2], [3, 4], [5, 6]];
        let b = array![[7i32, 8, 9], [10, 11, 12], [13, 14, 15]];
        let za = Array::compact_ndarray(&a).unwrap();
        let zb = Array::compact_ndarray(&b).unwrap();
        let actual = concatenate((za, zb), 1).to_ndarray().unwrap();
        let expected = ndarray::concatenate(ndarray::Axis(1), &[a.view(), b.view()]).unwrap();
        assert_eq!(actual, expected);
    }

    // 2D i32: three arrays along axis 1 with unequal sizes - scatter path
    #[test]
    fn test_i32_2d_axis1_three_unequal() {
        let a = array![[1i32], [2]];
        let b = array![[3i32, 4, 5], [6, 7, 8]];
        let c = array![[9i32, 10], [11, 12]];
        let za = Array::compact_ndarray(&a).unwrap();
        let zb = Array::compact_ndarray(&b).unwrap();
        let zc = Array::compact_ndarray(&c).unwrap();
        let actual = concatenate((za, zb, zc), 1).to_ndarray().unwrap();
        let expected =
            ndarray::concatenate(ndarray::Axis(1), &[a.view(), b.view(), c.view()]).unwrap();
        assert_eq!(actual, expected);
    }

    // 1D f32: concatenate two arrays along axis 0
    #[test]
    fn test_f32_1d_axis0() {
        let a = array![1.0f32, 2.0, 3.0];
        let b = array![4.0f32, 5.0];
        let za = Array::compact_ndarray(&a).unwrap();
        let zb = Array::compact_ndarray(&b).unwrap();
        let actual = concatenate((za, zb), 0).to_ndarray().unwrap();
        let expected = ndarray::concatenate(ndarray::Axis(0), &[a.view(), b.view()]).unwrap();
        assert_eq!(actual, expected);
    }

    // 2D f32: concatenate along axis 1 - scatter path
    #[test]
    fn test_f32_2d_axis1() {
        let a = array![[1.0f32, 2.0], [3.0, 4.0]];
        let b = array![[5.0f32, 6.0, 7.0], [8.0, 9.0, 10.0]];
        let za = Array::compact_ndarray(&a).unwrap();
        let zb = Array::compact_ndarray(&b).unwrap();
        let actual = concatenate((za, zb), 1).to_ndarray().unwrap();
        let expected = ndarray::concatenate(ndarray::Axis(1), &[a.view(), b.view()]).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    #[should_panic]
    fn test_shape_mismatch_panics() {
        let a = Array::compact_ndarray(&array![1i32, 2])
            .unwrap()
            .into_dim_dyn();
        let b = Array::compact_ndarray(&array![[1i32, 2]])
            .unwrap()
            .into_dim_dyn();
        let _ = concatenate((a, b), 0);
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
        let _ = concatenate((a, b), 0);
    }

    #[test]
    #[should_panic]
    fn test_empty_panics() {
        let _ = concatenate(Vec::<Array<Compact<Ty<i32>, DimDyn>>>::new(), 0);
    }

    // -----------------------------------------------------------------------
    // Proptest: arbitrary ndim, arbitrary axis, arbitrary number of arrays
    // -----------------------------------------------------------------------

    fn concat_strategy<T>() -> impl Strategy<
        Value = (
            Vec<ndarray::ArrayD<T>>,
            Vec<Array<Compact<Ty<T>, DimDyn>>>,
            usize,
        ),
    >
    where
        T: ScalarStrategy,
    {
        shape_strategy()
            .prop_filter("concat needs ndim >= 1", |s| !s.is_empty())
            .prop_flat_map(|shape| {
                let ndim = shape.len();
                (Just(shape), 0..ndim, 1usize..=5usize)
            })
            .prop_flat_map(|(shape, axis, n_arrays)| {
                let prefix = shape[..axis].to_vec();
                let suffix = shape[axis + 1..].to_vec();
                // Each array gets the same non-axis dims but an independently drawn axis size.
                let per_array_strat = (0usize..=5).prop_map(move |axis_size| {
                    let mut s = prefix.clone();
                    s.push(axis_size);
                    s.extend_from_slice(&suffix);
                    s
                });
                let per_array_strat = crate::util::carray_strategy_from_shape::<T>(
                    per_array_strat,
                    T::any_strategy(),
                );
                (prop::collection::vec(per_array_strat, n_arrays), Just(axis))
            })
            .prop_map(|(arrays, axis)| {
                let (nds, zas): (Vec<_>, Vec<_>) = arrays.into_iter().unzip();
                (nds, zas, axis)
            })
    }

    proptest::proptest! {
        #[test]
        fn proptest_concatenate((nds, zas, axis) in concat_strategy::<i32>()) {
            let nd_views: Vec<_> = nds.iter().map(|nd| nd.view()).collect();
            let expected = ndarray::concatenate(ndarray::Axis(axis), &nd_views).unwrap();
            crate::util::assert_array_matches(&concatenate(zas, axis), &expected);
        }
    }
}
