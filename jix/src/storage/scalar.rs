use crate::dtype::{Dtype, Dtyped};
use crate::error::{Result, check_dtype, check_get_range, check_ndim};
use crate::ops::{ElementwisePipeline, ElementwisePipelineImpl, Operand};
use crate::storage::params::{ArraySpecFlags, ArraySpecOwned, ReadTiling};
use crate::storage::{
    ArraySpec, ArrayStorageInfo, CompactParams, Index, OutBuf, OutBufInner, StorageBase,
    StridedBuf, check_out_buf, write_out_buf_slot,
};
use crate::util::{cast_slice, cast_slice_mut, dim_arr};
use crate::{DimDyn, Dimension, ReadContext};

/// Storage that broadcasts a single scalar value across an arbitrary shape: a zero-copy
/// constant array. A 0-D `Scalar` (empty `shape`) is a plain scalar value.
///
/// # Examples
///
/// ```
/// use jix::Array;
/// # use jix::__private::Scalar;
/// use ndarray::array;
///
/// let arr = Array::compact_ndarray(&array![[1.0f32, 2.0], [3.0, 4.0]])?;
///
/// let scalar_arr = Array::from_storage(Scalar::new(5.0f32, &[2, 2])?);
/// let result = (arr * scalar_arr).to_ndarray()?;
/// assert_eq!(result, array![[5.0f32, 10.0], [15.0, 20.0]]);
/// # Ok::<(), jix::Error>(())
/// ```
pub struct Scalar<T> {
    data: T,
    shape: DimDyn,
    spec: ArraySpecOwned,
}
impl<T> Scalar<T> {
    /// Create a `Scalar` storage that broadcasts `data` across `shape`, which may be empty (0-D).
    ///
    /// # Errors
    ///
    /// Returns an error if `shape.len()` exceeds the maximum supported number
    /// of dimensions, if a dim of `shape` exceeds `i64::MAX` or its byte size
    /// overflows `u64`, or if `T` has a zero itemsize.
    pub fn new(data: T, shape: &[u64]) -> Result<Self>
    where
        T: Dtyped,
    {
        check_ndim(shape.len())?;
        let shape = DimDyn::from_slice(shape);
        let ndim = shape.ndim();

        let mut spec = CompactParams::default().into_spec(
            shape.as_slice(),
            &T::DTYPE,
            ArraySpecFlags::new().with_read_tiling(ReadTiling::ZeroCopy),
        )?;
        spec.dynamic_mut().block_shape = dim_arr(ndim, |_| 1);
        // Reading a scalar element is free, and it broadcasts over every dimension.
        spec.dynamic_mut().element_cost = 0.0;

        Ok(Self { data, shape, spec })
    }

    /// Get the (singular) value of this storage.
    #[inline(always)]
    pub fn data(&self) -> &T {
        &self.data
    }
}

impl<T> crate::storage::ArrayStorageImpl for Scalar<T> {}
impl<T> StorageBase for Scalar<T>
where
    T: Dtyped,
{
    type ElementType = T;

    #[inline]
    fn read_data<'a>(
        &'a self,
        index: &Index<'_>,
        _context: &ReadContext,
        out: OutBuf<'_, 'a>,
    ) -> Result<()> {
        check_out_buf(&out, self.shape(), index, self.dtype().itemsize() as usize)?;
        let ndim = index.ndim();
        let dtype = T::DTYPE;

        match out.0 {
            // Lend a single-element view with all-zero strides: every position reads the same byte.
            OutBufInner::Pull(slot) => {
                let src = unsafe { cast_slice::<T, u8>(std::slice::from_ref(&self.data)) };
                write_out_buf_slot(slot, unsafe {
                    StridedBuf::from_slice(src, dim_arr(ndim, |_| 0usize))
                });
            }
            OutBufInner::Push(out) => {
                let read_shape = index.size();
                if out.is_contiguous_aligned(read_shape, &dtype) {
                    // Fill the region in one tight loop.
                    let region_len = read_shape.iter().product::<usize>() * size_of::<T>();
                    let (out_buf, _) = out.data_mut();
                    for item in unsafe { cast_slice_mut::<u8, T>(&mut out_buf[..region_len]) } {
                        *item = self.data;
                    }
                } else {
                    // Write the scalar to each strided output position
                    let src = unsafe { cast_slice::<T, u8>(std::slice::from_ref(&self.data)) };
                    let src_strides = dim_arr(ndim, |_| 0usize);
                    unsafe { out.copy_from(src, src_strides.as_ref(), read_shape, &dtype) };
                }
            }
        }
        Ok(())
    }

    #[inline]
    fn read_as_elementwise_pipeline<'a, T2>(
        &'a self,
        index: &Index<'_>,
        _context: &ReadContext,
    ) -> Result<impl ElementwisePipeline<T2> + use<'a, T2, T>>
    where
        T2: Dtyped,
    {
        check_dtype(&T2::DTYPE, &T::DTYPE)?;
        check_get_range(self.shape(), index)?;
        // SAFETY: `check_dtype` accepted `T` and `T2` as the same dtype.
        let value = unsafe { std::mem::transmute_copy::<T, T2>(&self.data) };

        struct Constant<T>(T);
        impl<T: Dtyped> ElementwisePipelineImpl<T> for Constant<T> {
            // The value lives in the node, so there is nothing to walk.
            const N_OPERANDS: Option<usize> = Some(0);
            const MAX_ITEMSIZE: usize = size_of::<T>();

            #[inline]
            fn operands<'s>(&'s self) -> impl Iterator<Item = &'s Operand<'s>> + 's {
                std::iter::empty()
            }

            #[inline(always)]
            unsafe fn read_bulk<const N: usize, const CONTIGUOUS: bool, S: fearless_simd::Simd>(
                &self,
                _offset: usize,
                _status: &mut crate::error::ArithmeticStatus,
                _simd: S,
            ) -> [T; N] {
                [self.0; N]
            }
        }

        Ok(Constant(value))
    }

    #[inline(always)]
    fn shape(&self) -> &[u64] {
        self.shape.as_slice()
    }
    #[inline(always)]
    fn dtype(&self) -> &Dtype {
        Dtype::new_ref::<T>()
    }
    #[inline]
    fn spec(&self) -> ArraySpec<'_> {
        self.spec.as_ref()
    }

    fn info(&self) -> ArrayStorageInfo<'_> {
        ArrayStorageInfo::new("Scalar")
    }

    crate::ops::impl_element_type_change_default!();
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use crate::__private::Scalar;
    use crate::dtype::Dtyped;
    use crate::error::Result;
    use crate::storage::{ArrayStorageExt, Index};
    use crate::util::DimArray;
    use crate::{Array, ReadContext};

    pub fn plain_scalar<T>(value: T, shape: &[u64]) -> Result<Array<Scalar<T>>>
    where
        T: Dtyped,
    {
        Ok(Array::from_storage(Scalar::new(value, shape)?))
    }

    #[test]
    fn push_fills_only_the_region() {
        // A destination whose bytes run past the region (a sub-view of a larger buffer, like the
        // ones ops hand to inner reads): the bytes after the region must stay untouched.
        use crate::storage::StridedBuf;

        let a = plain_scalar(5i32, &[3, 4]).unwrap();
        let ctx = ReadContext::default();
        let mut dst = [-1i32; 12];
        {
            let bytes = unsafe { crate::util::cast_slice_mut::<i32, u8>(&mut dst) };
            // Rows 1..3 of a 3x4 row-major buffer, starting at row 1.
            let mut out = unsafe {
                StridedBuf::from_slice_mut(
                    &mut bytes[16..],
                    DimArray::from_slice(&[16, 4]).unwrap(),
                )
            };
            a.storage()
                .read_data_push(&Index::new(&[0, 0], &[2, 4]).unwrap(), &ctx, &mut out)
                .unwrap();
        }
        assert_eq!(dst, [-1, -1, -1, -1, 5, 5, 5, 5, 5, 5, 5, 5]);

        let mut dst = [-1i32; 12];
        {
            let bytes = unsafe { crate::util::cast_slice_mut::<i32, u8>(&mut dst) };
            // Row 0 only, in a destination spanning all three rows.
            let mut out = unsafe {
                StridedBuf::from_slice_mut(bytes, DimArray::from_slice(&[16, 4]).unwrap())
            };
            a.storage()
                .read_data_push(&Index::new(&[0, 0], &[1, 4]).unwrap(), &ctx, &mut out)
                .unwrap();
        }
        assert_eq!(dst, [5, 5, 5, 5, -1, -1, -1, -1, -1, -1, -1, -1]);
    }

    // -----------------------------------------------------------------------
    // plain_scalar - shape [N]
    // -----------------------------------------------------------------------

    #[test]
    fn broadcast_1d_shape() {
        let a = plain_scalar(5i32, &[4]).unwrap();
        assert_eq!(a.shape(), &[4u64]);
    }

    #[test]
    fn broadcast_1d_read_i32() {
        let got = plain_scalar(5i32, &[4]).unwrap().to_ndarray().unwrap();
        assert_eq!(got, array![5i32, 5, 5, 5]);
    }

    #[test]
    fn broadcast_2d_shape() {
        let a = plain_scalar(1u8, &[3, 4]).unwrap();
        assert_eq!(a.shape(), &[3u64, 4]);
    }

    #[test]
    fn broadcast_2d_read_u8() {
        let got = plain_scalar(9u8, &[2, 3]).unwrap().to_ndarray().unwrap();
        assert_eq!(got, array![[9u8, 9, 9], [9, 9, 9]]);
    }

    #[test]
    fn broadcast_2d_subregion_read() {
        let got = plain_scalar(42i32, &[5, 5])
            .unwrap()
            .to_ndarray_sub(&[1..3, 2..4], &ReadContext::default())
            .unwrap();
        assert_eq!(got, array![[42i32, 42], [42, 42]]);
    }

    #[test]
    fn broadcast_3d_read_f32() {
        let got = plain_scalar(1.5f32, &[2, 3, 4])
            .unwrap()
            .to_ndarray()
            .unwrap();
        assert_eq!(
            got,
            ndarray::Array::from_shape_vec([2, 3, 4], vec![1.5f32; 24]).unwrap()
        );
    }

    // -----------------------------------------------------------------------
    // reduction on top of Scalar storage
    // -----------------------------------------------------------------------

    #[test]
    fn max_of_broadcast_scalar() {
        // max of a constant array is the constant itself
        let got = plain_scalar(7.0f64, &[3, 4])
            .unwrap()
            .max(0)
            .to_ndarray()
            .unwrap();
        assert_eq!(got, array![7.0f64, 7.0, 7.0, 7.0]);
    }

    #[test]
    fn sum_of_broadcast_scalar_i32() {
        // sum of [2,2,2] (3 rows, broadcast) over dim 0 = [6,6,6,6] as i64
        let got = plain_scalar(2i32, &[3, 4])
            .unwrap()
            .sum(0)
            .to_ndarray()
            .unwrap();
        assert_eq!(got, array![6i64, 6, 6, 6]);
    }
}
