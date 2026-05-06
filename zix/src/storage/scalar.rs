use std::ops::Range;

use crate::codec::ReadContext;
use crate::dtype::{Dtype, Dtyped};
use crate::error::{check_get_buffer_size, check_get_range, check_ndim, Result};
use crate::storage::{ArrayStorage, ArrayStorageSpec, BlockShapeTag, BlocksLayout};
use crate::util::{cast_slice_mut, dim_arr, DimArray};
use crate::Array;

/// Storage type that broadcasts a single scalar value across an arbitrary shape.
///
/// Every element read from a `Scalar` array returns the same value regardless
/// of position, making it a zero-copy way to represent constant or filled arrays.
/// A 0-D `Scalar` (empty `shape`) represents a plain scalar value.
///
/// Prefer the [`Array`] constructor [`Array::plain_scalar`] over constructing this type directly.
///
/// # Examples
///
/// Scalars can be combined with arrays either implicitly, by using a raw Rust
/// value directly as an operand, or explicitly via [`Array::plain_scalar`].
/// Both are equivalent; choose whichever reads more clearly.
///
/// **Implicit** - the operator accepts a raw scalar and broadcasts it automatically:
///
/// ```
/// use zix::{Array, ArrayParams};
/// use ndarray::array;
///
/// let arr = Array::compact_array(&array![[1.0f32, 2.0], [3.0, 4.0]])?;
///
/// let result = (arr * 5.0f32).to_ndarray::<f32>()?; // the scalar is broadcast automatically
/// assert_eq!(result, array![[5.0f32, 10.0], [15.0, 20.0]].into_dyn());
/// # Ok::<(), zix::Error>(())
/// ```
///
/// **Explicit** - construct a `Scalar` array first, then apply the operation:
///
/// ```
/// use zix::{Array, ArrayParams};
/// use ndarray::array;
///
/// let arr = Array::compact_array(&array![[1.0f32, 2.0], [3.0, 4.0]])?;
///
/// let scalar_arr = Array::plain_scalar(5.0f32, &[2, 2])?;
/// let result = (arr * scalar_arr).to_ndarray::<f32>()?;
/// assert_eq!(result, array![[5.0f32, 10.0], [15.0, 20.0]].into_dyn());
/// # Ok::<(), zix::Error>(())
/// ```
pub struct Scalar<T> {
    data: T,
    shape: DimArray<u64>,
    dtype: Dtype,
    blocks_layout: BlocksLayout,
}
impl<T> Scalar<T> {
    /// Create a `Scalar` storage that broadcasts `data` across `shape`.
    ///
    /// `shape` may be empty (producing a 0-D scalar) or any non-empty slice
    /// (producing a constant array of that shape).
    ///
    /// # Errors
    ///
    /// Returns an error if `shape.len()` exceeds the maximum supported number
    /// of dimensions.
    pub fn new(data: T, shape: &[u64]) -> Result<Self>
    where
        T: Dtyped,
    {
        let ndim = shape.len();
        check_ndim(ndim)?;
        let shape: DimArray<_> = shape.try_into().unwrap();

        let dtype = T::DTYPE;

        let blocks_layout = BlocksLayout::new(
            Some(dim_arr(ndim, |_| 1)),
            Some(dim_arr(ndim, |_| BlockShapeTag::Any)),
            None,
            None,
            None,
            &shape,
            dtype.itemsize(),
        )?;

        Ok(Self {
            data,
            shape,
            dtype,
            blocks_layout,
        })
    }

    /// Get the (singular) value of this storage.
    pub fn data(&self) -> &T {
        &self.data
    }
}

impl<T> Array<Scalar<T>> {
    /// Create an array with the given `shape` where every element equals `value`.
    ///
    /// This is a zero-copy broadcast: the scalar is stored once and repeated on
    /// every read.  The resulting array behaves like `np.full(shape, value)`.
    ///
    /// The storage does not compress anything. This function is useful when a scalar need to participate
    /// in an operation with another array.
    ///
    /// # Errors
    ///
    /// Returns an error if `shape.len()` exceeds the maximum supported number
    /// of dimensions.
    pub fn plain_scalar(value: T, shape: &[u64]) -> Result<Self>
    where
        T: Dtyped,
    {
        Ok(Self::from_storage(Scalar::new(value, shape)?))
    }
}

impl<T> ArrayStorage for Scalar<T>
where
    T: Dtyped,
{
    fn read_data(
        &self,
        index: &[Range<u64>],
        buf: &mut [u8],
        _context: &ReadContext,
    ) -> Result<()> {
        check_get_range(self.shape(), index)?;
        check_get_buffer_size(index, self.dtype(), buf)?;
        let buf = unsafe { cast_slice_mut::<u8, T>(buf) };
        for item in buf.iter_mut() {
            *item = self.data;
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
            encoder_params: None,
            decoder_params: None,
            // decoder_config: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::ArrayD;

    use crate::codec::ReadContext;
    use crate::Array;

    // -----------------------------------------------------------------------
    // plain_scalar - shape [N]
    // -----------------------------------------------------------------------

    #[test]
    fn broadcast_1d_shape() {
        let a = Array::plain_scalar(5i32, &[4]).unwrap();
        assert_eq!(a.shape(), &[4u64]);
    }

    #[test]
    fn broadcast_1d_read_i32() {
        let got: ArrayD<i32> = Array::plain_scalar(5i32, &[4])
            .unwrap()
            .to_ndarray()
            .unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![4], vec![5i32; 4]).unwrap());
    }

    #[test]
    fn broadcast_2d_shape() {
        let a = Array::plain_scalar(1u8, &[3, 4]).unwrap();
        assert_eq!(a.shape(), &[3u64, 4]);
    }

    #[test]
    fn broadcast_2d_read_u8() {
        let got: ArrayD<u8> = Array::plain_scalar(9u8, &[2, 3])
            .unwrap()
            .to_ndarray()
            .unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![2, 3], vec![9u8; 6]).unwrap()
        );
    }

    #[test]
    fn broadcast_2d_subregion_read() {
        let got: ArrayD<i32> = Array::plain_scalar(42i32, &[5, 5])
            .unwrap()
            .to_ndarray_sub(&[1..3, 2..4], &ReadContext::default())
            .unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![2, 2], vec![42i32; 4]).unwrap()
        );
    }

    #[test]
    fn broadcast_3d_read_f32() {
        let got: ArrayD<f32> = Array::plain_scalar(1.5f32, &[2, 3, 4])
            .unwrap()
            .to_ndarray()
            .unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![2, 3, 4], vec![1.5f32; 24]).unwrap()
        );
    }

    // -----------------------------------------------------------------------
    // reduction on top of Scalar storage
    // -----------------------------------------------------------------------

    #[test]
    fn max_of_broadcast_scalar() {
        // max of a constant array is the constant itself
        let got: ArrayD<f64> = Array::plain_scalar(7.0f64, &[3, 4])
            .unwrap()
            .max(&[0], false)
            .to_ndarray()
            .unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![4], vec![7.0f64; 4]).unwrap()
        );
    }

    #[test]
    fn sum_of_broadcast_scalar_i32() {
        // sum of [2,2,2] (3 rows, broadcast) over axis 0 = [6,6,6,6] as i64
        let got: ArrayD<i64> = Array::plain_scalar(2i32, &[3, 4])
            .unwrap()
            .sum(&[0], false)
            .to_ndarray()
            .unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![4], vec![6i64; 4]).unwrap());
    }
}
