use crate::dtype::{Dtype, Dtyped};
use crate::error::{check_ndim, Result};
use crate::storage::params::{ArraySpecFlags, ArraySpecOwned};
use crate::storage::{
    check_out_buf, materialize_out_buf, ArrayStorageInfo, Compact, Index, StridedBuf,
};
use crate::util::dim_arr;
use crate::util::iter::NdIter;
use crate::ReadContext;
use crate::{
    ArrayParams, ArrayStorage, CompactArray, DimArray, DimDyn, ElementType, SliceExt, Ty, TypeDyn,
};

use super::Array;

impl<T> CompactArray<T> {
    /// Compress an ndarray into a block-compressed `Array<Compact>` with default encoding settings.
    ///
    /// The array is partitioned into n-dimensional blocks, each independently compressed. The
    /// block shape is derived automatically according to the CPU cache sizes. Use
    /// [`compact_ndarray_with`](Array::compact_ndarray_with) for explicit control over block shape,
    /// compression level, and other codec settings. If the access pattern is known in advance,
    /// providing a matching block shape can improve read performance significantly.
    ///
    /// The element type `Ty<T>` is inferred from the ndarray argument's type.
    ///
    /// # Errors
    ///
    /// - [`TooManyDimensions`](crate::error::ErrorKind::TooManyDimensions) - `array.ndim()` exceeds
    ///   [`NDIM_MAX`](crate::NDIM_MAX).
    /// - [`InvalidArgument`](crate::error::ErrorKind::InvalidArgument) - `array` has a negative
    ///   stride (e.g. a reversed axis).
    /// - [`CodecError`](crate::error::ErrorKind::CodecError) - compression fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use jix::dtype::Dtyped;
    /// use jix::Array;
    /// use ndarray::{array, Ix2};
    ///
    /// // Compress a 2-D f32 ndarray.
    /// let array = Array::compact_ndarray(&array![[1.5f32, 2.0, -9.0], [3.14, 6.17, 0.0]])?;
    /// assert_eq!(array.shape(), &[2, 3]);
    /// assert_eq!(array.dtype(), &f32::DTYPE);
    ///
    /// // Decompress and compare.
    /// let decompressed = array.to_ndarray::<Ix2>()?;
    /// assert_eq!(decompressed[[0, 0]], 1.5);
    /// assert_eq!(decompressed[[1, 1]], 6.17);
    ///
    /// // Apply operations on a compressed array, creating lazy views
    /// let ones = Array::compact_ndarray(&ndarray::Array2::<f32>::ones((2, 3)))?;
    /// let scaled = array                               // Array<Compact>
    ///     .exp()                                       // Array<Exp<Compact>>
    ///     .floor()                                     // Array<Floor<Exp<Compact>>>
    ///     .map(|x| x * 2.0f32)                         // Array<Map<Floor<...>>>
    ///     + ones;                                      // Array<Add<Map<...>, Compact>>
    /// assert_eq!(scaled.shape(), &[2, 3]);
    /// assert_eq!(scaled.dtype(), &f32::DTYPE);
    /// assert_eq!(scaled.to_ndarray::<Ix2>()?[[1, 1]], 957.0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn compact_ndarray<S, InD>(array: &ndarray::ArrayBase<S, InD>) -> Result<Self>
    where
        InD: ndarray::Dimension,
        S: ndarray::Data<Elem = T>,
        T: Dtyped,
    {
        Array::compact_ndarray_with(array, ArrayParams::default())
    }

    /// Compress an ndarray into a block-compressed `Array<Compact>` with explicit `ArrayParams`.
    ///
    /// See [`compact_ndarray`](Array::compact_ndarray) for the default-parameter version, which has more
    /// documentation and examples.
    ///
    /// Use this method to specify encoding parameters such as block shape, compression level, etc.
    /// See [`ArrayParams`] for details on the available parameters and their effects on performance.
    ///
    /// # Examples
    ///
    /// ```
    /// use jix::{Array, ArrayParams};
    /// use ndarray::Ix2;
    ///
    /// let data = ndarray::Array2::<f32>::zeros((512, 512));
    ///
    /// // Store with 64*64 blocks - good for tile-at-a-time access patterns.
    /// let array = Array::compact_ndarray_with(&data, ArrayParams::new().block_shape(&[64, 64])?)?;
    ///
    /// // Read tiles of 128*128 by decompressing 2*2 blocks at a time.
    /// let context = array.read_ctx();
    /// for tile_row in 0..7 {
    ///     for tile_col in 0..7 {
    ///         let row_range = (tile_row * 64)..((tile_row + 2) * 64);
    ///         let col_range = (tile_col * 64)..((tile_col + 2) * 64);
    ///         let tile = array.to_ndarray_sub::<Ix2>(&[row_range, col_range], &context)?;
    ///         println!("tile ({tile_row},{tile_col}) sum: {}", tile.sum());
    ///     }
    /// }
    /// # Ok::<(), jix::Error>(())
    /// ```
    pub fn compact_ndarray_with<S, InD>(
        array: &ndarray::ArrayBase<S, InD>,
        mut params: ArrayParams,
    ) -> Result<Self>
    where
        InD: ndarray::Dimension,
        S: ndarray::Data<Elem = T>,
        T: Dtyped,
    {
        let array = Array::plain_ndarray_ref(array)?;
        params.tune(array.shape(), array.dtype())?;
        array.compact_with(params, &array.try_read_ctx()?)
    }
}

impl Array<Compact<TypeDyn>> {
    /// Compress a raw n-dimensional buffer into a block-compressed `Array<Compact>`.
    ///
    /// Same as [`compact_ndarray_with`](Array::compact_ndarray_with) but takes a raw pointer and
    /// explicit shape and strides.
    ///
    /// # Arguments
    ///
    /// - `ptr`: pointer to the beginning of the buffer. It needs no particular alignment.
    /// - `shape`: shape of the n-dimensional array. At most [`NDIM_MAX`](crate::NDIM_MAX)
    ///   dimensions are allowed.
    /// - `strides`: strides of the n-dimensional array, in bytes. Must be the same length as
    ///   `shape`.
    /// - `dtype`: element type of the array. The buffer is interpreted as containing elements of
    ///   this type.
    /// - `params`: block layout and codec parameters. See [`ArrayParams`] for details.
    ///
    /// # Safety
    ///
    /// `ptr` must point to a readable buffer, laid out with the given
    /// `strides` (in bytes, one per dimension).
    /// The buffer should contain elements of the given `dtype`.
    /// Accessing the buffer according to the shape, strides and dtype must be memory-safe and yield
    /// valid elements. Every byte from the first to the last element (the strided span, gaps
    /// between elements included) lies in one allocation, and none of it is written, or borrowed
    /// through a `&mut`, during the call.
    pub unsafe fn compact_ndarray_ptr(
        ptr: *const u8,
        shape: &[u64],
        strides: &[usize],
        dtype: Dtype,
        mut params: ArrayParams,
    ) -> Result<Self> {
        check_ndim(shape.len())?;
        params.tune(shape, &dtype)?;
        let array =
            unsafe { Array::plain_ndarray_ptr(ptr, shape, strides, dtype, params.clone())? };
        array.compact_with(params, &array.try_read_ctx()?)
    }
}

impl<T> CompactArray<T> {
    /// Create a block-compressed array by evaluating a function `f` at each index.
    ///
    /// The function `f` is called with the index of each element in the output array, and should
    /// return the value for that element.
    /// Elements are visited in an arbitrary order - this is not some theoretical use case, most of
    /// the time elements will NOT be visited in row-major order, and the function should not rely
    /// on any specific order for correctness or performance.
    ///
    /// # Arguments
    ///
    /// - `shape`: shape of the output array.
    /// - `f`: function that produces the value of each element given its index.
    ///
    /// # Examples
    ///
    /// The index is passed as a `&[u64]` with one entry per dim:
    ///
    /// ```
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::compact_fn(&[5], |i| i[0] * 2)?;
    /// assert_eq!(a.to_ndarray()?, array![0, 2, 4, 6, 8]);
    ///
    /// let a = Array::compact_fn(&[3, 3], |i| i[0] * 10 + i[1])?;
    /// assert_eq!(
    ///     a.to_ndarray()?,
    ///     array![[0, 1, 2], [10, 11, 12], [20, 21, 22]]
    /// );
    /// # Ok::<(), jix::Error>(())
    /// ```
    pub fn compact_fn(shape: &[u64], f: impl Fn(&[u64]) -> T) -> Result<Self>
    where
        T: Dtyped,
    {
        Self::compact_fn_with(shape, ArrayParams::default(), f)
    }

    /// Create a block-compressed array by evaluating a function `f` at each index, with explicit `ArrayParams`.
    ///
    /// The function `f` is called with the index of each element in the output array, and should
    /// return the value for that element.
    /// Elements are visited in an arbitrary order - this is not some theoretical use case, most of
    /// the time elements will NOT be visited in row-major order, and the function should not rely
    /// on any specific order for correctness or performance.
    ///
    /// # Arguments
    ///
    /// - `shape`: shape of the output array.
    /// - `params`: block layout and codec parameters. See [`ArrayParams`] for details.
    /// - `f`: function that produces the value of each element given its index.
    ///
    /// # Examples
    ///
    /// Materialize a large 2D array with an explicit block shape tuned to the
    /// access pattern (here, square 64x64 tiles):
    ///
    /// ```
    /// # use jix::{Array, ArrayParams};
    /// let params = ArrayParams::new().block_shape(&[64, 64])?;
    ///
    /// let a = Array::compact_fn_with(&[256, 256], params, |i| (i[0] as f32) * 0.5 + (i[1] as f32))?;
    /// assert_eq!(a.shape(), &[256, 256]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    pub fn compact_fn_with(
        shape: &[u64],
        mut params: ArrayParams,
        f: impl Fn(&[u64]) -> T,
    ) -> Result<Self>
    where
        T: Dtyped,
    {
        struct FnStorage<T, F> {
            dtype: Ty<T>,
            shape: DimArray<u64>,
            f: F,
            spec: ArraySpecOwned,
        }
        impl<T, F> crate::storage::ArrayStorageImpl for FnStorage<T, F> {}
        impl<T, F> ArrayStorage for FnStorage<T, F>
        where
            T: Dtyped,
            F: Fn(&[u64]) -> T,
        {
            type ElementType = Ty<T>;

            #[inline]
            fn read_data<'a>(
                &'a self,
                index: &Index<'_>,
                context: &'a ReadContext,
                out: Option<&'a mut StridedBuf<'_>>,
            ) -> Result<StridedBuf<'a>> {
                check_out_buf(
                    out.as_deref(),
                    self.shape(),
                    index,
                    self.dtype().itemsize() as usize,
                )?;
                let ndim = index.len();
                let read_shape_usize =
                    dim_arr(ndim, |dim| (index[dim].end - index[dim].start) as usize);
                let mut out = materialize_out_buf(
                    out,
                    context,
                    read_shape_usize.as_ref(),
                    self.dtype(),
                    self.spec().read_layout_order(),
                );
                let (out_buf, out_strides) = out.data_mut();

                let alignment = self.dtype().alignment().as_usize();
                let dst_aligned = (out_buf.as_ptr() as usize).is_multiple_of(alignment)
                    && (out_strides.iter().all(|s| s.is_multiple_of(alignment)));

                let begin = dim_arr(ndim, |dim| index[dim].start);
                let end = dim_arr(ndim, |dim| index[dim].end);
                let base = out_buf.as_mut_ptr();
                let iter = NdIter::builder_with_begin(begin, end)
                    .with_strides_offset_ext(out_strides.to_dim_vec::<DimDyn>(), 0)
                    .build();
                if dst_aligned {
                    iter.for_each(|idx, dst_offset| {
                        let value = (self.f)(idx.as_ref());
                        unsafe { base.add(dst_offset).cast::<T>().write(value) };
                    });
                } else {
                    iter.for_each(|idx, dst_offset| {
                        let value = (self.f)(idx.as_ref());
                        unsafe { base.add(dst_offset).cast::<T>().write_unaligned(value) };
                    });
                }
                Ok(out)
            }

            #[inline(always)]
            fn shape(&self) -> &[u64] {
                self.shape.as_slice()
            }

            #[inline(always)]
            fn dtype(&self) -> &Dtype {
                self.dtype.dtype()
            }

            #[inline]
            fn spec(&self) -> crate::storage::ArraySpec<'_> {
                self.spec.as_ref()
            }

            fn info(&self) -> ArrayStorageInfo<'_> {
                ArrayStorageInfo::new("FnStorage")
            }

            crate::ops::impl_element_type_change_default!();
        }

        let dtype = Ty::<T>::new();

        check_ndim(shape.len())?;
        let shape = DimArray::from_slice(shape).unwrap();
        params.tune(shape.as_slice(), dtype.dtype())?;
        let spec =
            params
                .clone()
                .into_spec(shape.as_slice(), dtype.dtype(), ArraySpecFlags::new())?;
        let array = Array::from_storage(FnStorage {
            dtype,
            shape,
            f,
            spec,
        });

        array.compact_with(params, &array.try_read_ctx()?)
    }
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
mod tests {
    use ndarray::array;

    use super::Array;

    use crate::dtype::Dtyped;

    use crate::storage::block::BlockSize;

    use crate::util::arr_params;
    use crate::{ArrayParams, ArrayStorage};

    use crate::array::tests::*;

    // -----------------------------------------------------------------------
    // compact_ndarray - 1D
    // -----------------------------------------------------------------------

    #[test]
    fn compact_ndarray_1d_single_block() {
        let src = array![0u8, 1, 2, 3];
        assert_eq!(roundtrip(&src, &[4]), src);
    }

    #[test]
    fn compact_ndarray_1d_multi_block() {
        let src = array![0u8, 1, 2, 3, 4, 5];
        assert_eq!(roundtrip(&src, &[3]), src);
    }

    #[test]
    fn compact_ndarray_1d_with_padding() {
        // size 5, block 3 -> padded to 6; shape reported as 5
        let src = array![0u8, 1, 2, 3, 4];
        let a = Array::compact_ndarray_with(&src, arr_params(&[3])).unwrap();
        assert_eq!(a.shape(), &[5]);
        let got = a.to_ndarray().unwrap();
        assert_eq!(got, src);
    }

    #[test]
    fn compact_ndarray_1d_i32() {
        let src = array![0i32, 10, 20, 30, 40, 50, 60, 70];
        assert_eq!(roundtrip(&src, &[4]), src);
    }

    #[test]
    fn compact_ndarray_1d_f32() {
        let src = array![0.0f32, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5];
        assert_eq!(roundtrip(&src, &[4]), src);
    }

    #[test]
    fn compact_ndarray_block_larger_than_shape_is_clamped() {
        // block_shape [10] > array shape [4] is clamped to the array boundary.
        let src = array![0u8, 1, 2, 3];
        let a = Array::compact_ndarray_with(&src, arr_params(&[10])).unwrap();
        assert_eq!(a.storage().spec().block_shape().as_slice(), &[4]);
        assert_eq!(a.to_ndarray().unwrap(), src);
    }

    #[test]
    fn compact_ndarray_1d_noncontiguous() {
        // Step-2 slice of [0..10] -> [0, 2, 4, 6, 8]
        let src = array![0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let view = src.slice(ndarray::s![..;2]);
        let a = Array::compact_ndarray_with(&view, arr_params(&[3])).unwrap();
        assert_eq!(a.shape(), &[5]);
        assert_eq!(a.to_ndarray().unwrap(), array![0u8, 2, 4, 6, 8]);
    }

    // -----------------------------------------------------------------------
    // compact_ndarray - metadata
    // -----------------------------------------------------------------------

    #[test]
    fn compact_ndarray_metadata() {
        let a = Array::compact_ndarray(&array![0i32, 1, 2, 3, 4, 5]).unwrap();
        assert_eq!(a.ndim(), 1);
        assert_eq!(a.shape(), &[6]);
        assert_eq!(a.dtype(), &i32::DTYPE);
    }

    // -----------------------------------------------------------------------
    // compact_ndarray - 2D
    // -----------------------------------------------------------------------

    #[test]
    fn compact_ndarray_2d() {
        #[rustfmt::skip]
        let src = array![
            [0u8,  1,  2,  3,  4,  5],
            [6,    7,  8,  9, 10, 11],
            [12,  13, 14, 15, 16, 17],
            [18,  19, 20, 21, 22, 23],
        ];
        assert_eq!(roundtrip(&src, &[2, 3]), src);
    }

    #[test]
    fn compact_ndarray_2d_with_padding() {
        // shape [3,5], block [2,3] -> padded to [4,6]; shape reported as [3,5]
        #[rustfmt::skip]
        let src = array![
            [0i32,  1,  2,  3,  4],
            [5,     6,  7,  8,  9],
            [10,   11, 12, 13, 14],
        ];
        let a = Array::compact_ndarray_with(&src, arr_params(&[2, 3])).unwrap();
        assert_eq!(a.shape(), &[3, 5]);
        assert_eq!(a.to_ndarray().unwrap(), src);
    }

    #[test]
    fn compact_ndarray_2d_noncontiguous() {
        // Fortran-order (column-major) array
        let src = ndarray::Array2::<u8>::from_shape_vec(
            ndarray::ShapeBuilder::f((3, 4)),
            (0..12).collect(),
        )
        .unwrap();
        assert_eq!(roundtrip(&src, &[2, 2]), src);
    }

    // -----------------------------------------------------------------------
    // compact_ndarray + to_ndarray_sub integration
    // -----------------------------------------------------------------------

    #[test]
    fn compact_ndarray_then_to_ndarray_sub_1d() {
        let src = array![0u8, 1, 2, 3, 4, 5];
        let a = Array::compact_ndarray_with(&src, arr_params(&[3])).unwrap();
        let got = a.to_ndarray_sub(&[1..5], &a.read_ctx()).unwrap();
        assert_eq!(got, array![1u8, 2, 3, 4]);
    }

    #[test]
    fn compact_ndarray_then_to_ndarray_sub_2d() {
        #[rustfmt::skip]
        let src = array![
            [0u8,  1,  2,  3,  4,  5],
            [6,    7,  8,  9, 10, 11],
            [12,  13, 14, 15, 16, 17],
            [18,  19, 20, 21, 22, 23],
        ];
        let a = Array::compact_ndarray_with(&src, arr_params(&[2, 3])).unwrap();
        let got = a.to_ndarray_sub(&[1..3, 2..5], &a.read_ctx()).unwrap();
        assert_eq!(got, array![[8u8, 9, 10], [14, 15, 16]]);
    }

    // -----------------------------------------------------------------------
    // compact_fn
    // -----------------------------------------------------------------------

    #[test]
    fn compact_fn_1d() {
        let a = Array::compact_fn(&[47], |i| i[0] * 4 + 6).unwrap();
        assert_eq!(a.shape(), &[47]);
        assert_eq!(
            a.to_ndarray().unwrap(),
            ndarray::Array1::from_iter((0..47).map(|i| i * 4 + 6))
        );
    }

    #[test]
    fn compact_fn_2d() {
        let a = Array::compact_fn(&[3, 3], |i| i[0] * 10 + i[1]).unwrap();
        assert_eq!(a.shape(), &[3, 3]);
        assert_eq!(
            a.to_ndarray().unwrap(),
            array![[0, 1, 2], [10, 11, 12], [20, 21, 22]]
        );
    }

    /// Params whose small read size makes compaction read the array in many chunks, so `f` is
    /// called for regions that do not start at the origin.
    fn multi_chunk_params(block_shape: &[BlockSize]) -> ArrayParams {
        ArrayParams::new()
            .block_shape(block_shape)
            .unwrap()
            .read_size((8, 64))
    }

    #[test]
    fn compact_fn_multi_chunk_static() {
        let params = multi_chunk_params(&[4, 4]);
        let a = Array::compact_fn_with(&[32, 24], params, |i| i[0] * 100 + i[1]).unwrap();
        let expected = ndarray::Array2::from_shape_fn((32, 24), |(x, y)| (x * 100 + y) as u64);
        assert_eq!(a.to_ndarray().unwrap(), expected);
    }

    #[test]
    fn compact_fn_multi_chunk_high_ndim() {
        let params = multi_chunk_params(&[2, 2, 2, 2, 2]);
        let f = |i: &[u64]| i.iter().fold(0, |acc, &x| acc * 10 + x);
        let a = Array::compact_fn_with(&[6, 4, 6, 4, 2], params, f).unwrap();
        let expected = ndarray::Array5::from_shape_fn((6, 4, 6, 4, 2), |(a, b, c, d, e)| {
            f(&[a, b, c, d, e].map(|x| x as u64))
        });
        assert_eq!(a.to_ndarray().unwrap(), expected);
    }

    #[test]
    fn compact_fn_0d() {
        let a = Array::compact_fn(&[], |_| 42).unwrap();
        assert_eq!(a.shape(), &[]);
        assert_eq!(a.to_ndarray().unwrap(), ndarray::arr0(42));
    }
}
