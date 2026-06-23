use std::ops::Range;

use crate::codec::ReadContext;
use crate::dtype::{Dtype, Dtyped};
use crate::error::{check_dtype, check_get_buffer_size, check_get_range, check_ndim, Result};
use crate::storage::params::ArraySpecOwned;
use crate::storage::{ArraySpec, ArrayStorage, BlockShapeTag, ReadData, Ty};
use crate::util::{cast_slice_mut, dim_arr};
use crate::{ArrayParams, Dimension, ElementType, IntoDimension};

/// Storage type that broadcasts a single scalar value across an arbitrary shape.
///
/// Every element read from a `Scalar<T, D>` array returns the same value regardless
/// of position, making it a zero-copy way to represent constant or filled arrays.
/// A 0-D `Scalar` (empty `shape`) represents a plain scalar value.
///
/// `D: Dimension` tracks the ndim at the type level and follows the same semantics as
/// [`Compact<ET, D>`](crate::storage::Compact): `D` is inferred from the shape argument type.
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
pub struct Scalar<T, D> {
    data: T,
    shape: D,
    element_type: Ty<T>,
    spec: ArraySpecOwned,
}
impl<T, D> Scalar<T, D> {
    /// Create a `Scalar` storage that broadcasts `data` across `shape`.
    ///
    /// `shape` may be empty (producing a 0-D scalar) or any non-empty slice
    /// (producing a constant array of that shape).
    ///
    /// # Errors
    ///
    /// Returns an error if `shape.len()` exceeds the maximum supported number
    /// of dimensions.
    pub fn new<Sh>(data: T, shape: Sh) -> Result<Self>
    where
        T: Dtyped,
        D: Dimension,
        Sh: IntoDimension<Dimension = D>,
    {
        let shape = shape.into_dimension()?;
        let ndim = shape.ndim();

        let params = ArrayParams {
            block_shape: Some(dim_arr(ndim, |_| 1)),
            block_shape_tag: Some(dim_arr(ndim, |_| BlockShapeTag::Any)),
            block_size: None,
            read_size: None,
            encoder_params: None,
            decoder_params: None,
        };
        let spec = params.into_spec(shape.as_slice(), &T::DTYPE)?;

        Ok(Self {
            data,
            shape,
            element_type: Ty::new(),
            spec,
        })
    }

    /// Get the (singular) value of this storage.
    #[inline(always)]
    pub fn data(&self) -> &T {
        &self.data
    }
}

impl<T, D> ArrayStorage for Scalar<T, D>
where
    T: Dtyped,
    D: Dimension,
{
    type ElementType = Ty<T>;
    type Dimension = D;

    #[inline(always)]
    fn read_data(
        &self,
        index: &[Range<u64>],
        buf: &mut [u8],
        _context: &ReadContext,
    ) -> Result<()> {
        check_get_range(self.shape(), index)?;
        check_get_buffer_size(index, &T::DTYPE, buf)?;
        let buf = unsafe { cast_slice_mut::<u8, T>(buf) };
        for item in buf.iter_mut() {
            *item = self.data;
        }
        Ok(())
    }

    #[inline(always)]
    fn read_data_typed<'a, T2>(
        &'a self,
        index: &[Range<u64>],
        _context: &'a ReadContext,
    ) -> Result<impl ReadData<T2> + use<'a, T2, T, D>>
    where
        T2: Dtyped,
    {
        check_dtype(&T2::DTYPE, &T::DTYPE)?;
        check_get_range(self.shape(), index)?;
        let nitems = index.iter().map(|r| r.end - r.start).product::<u64>() as usize;
        struct ScalarReadData<T2> {
            value: T2,
            len_: usize,
        }
        impl<T2> ReadData<T2> for ScalarReadData<T2>
        where
            T2: Dtyped,
        {
            fn len(&self) -> usize {
                self.len_
            }

            fn read_bulk<const N: usize>(&mut self, offset: usize) -> [T2; N] {
                let len = self.len();
                assert!(offset + N <= len);
                [self.value; N]
            }
        }

        // SAFETY: we checked that T and T2 have the same dtype
        let value = unsafe { std::mem::transmute_copy::<T, T2>(&self.data) };

        Ok(ScalarReadData {
            value,
            len_: nitems,
        })
    }

    #[inline(always)]
    fn shape(&self) -> &[u64] {
        self.shape.as_slice()
    }
    #[inline(always)]
    fn dtype(&self) -> &Dtype {
        self.element_type.dtype()
    }
    #[inline]
    fn spec(&self) -> ArraySpec<'_> {
        self.spec.as_ref()
    }

    type DimensionChange<NewD: Dimension> = Scalar<T, NewD>;
    #[inline]
    fn dimension_change<NewD: Dimension>(self) -> Result<Self::DimensionChange<NewD>> {
        check_ndim::<NewD>(self.shape().len())?;
        let shape = NewD::from_slice(self.shape());
        Ok(Scalar {
            data: self.data,
            shape,
            element_type: self.element_type,
            spec: self.spec,
        })
    }

    crate::ops::impl_element_type_change_default!();
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use crate::__private::Scalar;
    use crate::codec::ReadContext;
    use crate::dtype::Dtyped;
    use crate::error::Result;
    use crate::{Array, IntoDimension};

    pub fn plain_scalar<T, Sh>(value: T, shape: Sh) -> Result<Array<Scalar<T, Sh::Dimension>>>
    where
        T: Dtyped,
        Sh: IntoDimension,
    {
        Ok(Array::from_storage(Scalar::new(value, shape)?))
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
        // sum of [2,2,2] (3 rows, broadcast) over axis 0 = [6,6,6,6] as i64
        let got = plain_scalar(2i32, &[3, 4])
            .unwrap()
            .sum(0)
            .to_ndarray()
            .unwrap();
        assert_eq!(got, array![6i64, 6, 6, 6]);
    }
}
