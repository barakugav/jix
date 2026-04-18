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
/// Prefer the [`Array`] constructors [`Array::from_scalar`] and
/// [`Array::from_scalar_broadcast`] over constructing this type directly.
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
}

impl<T> Array<Scalar<T>> {
    /// Create a 0-D array that holds a single scalar `value`.
    ///
    /// The resulting array has an empty shape (`[]`) and its single element is
    /// `value`.  No heap allocation is made for the element data.
    pub fn from_scalar(value: T) -> Result<Self>
    where
        T: Dtyped,
    {
        Ok(Self::from_storage(Scalar::new(value, &[])?))
    }

    /// Create an array with the given `shape` where every element equals `value`.
    ///
    /// This is a zero-copy broadcast: the scalar is stored once and repeated on
    /// every read.  The resulting array behaves like `np.full(shape, value)`.
    ///
    /// # Errors
    ///
    /// Returns an error if `shape.len()` exceeds the maximum supported number
    /// of dimensions.
    pub fn from_scalar_broadcast(value: T, shape: &[u64]) -> Result<Self>
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
    fn spec(&self) -> ArrayStorageSpec<'_> {
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

    use crate::Array;

    // -----------------------------------------------------------------------
    // from_scalar — 0-D
    // -----------------------------------------------------------------------

    #[test]
    fn scalar_0d_shape() {
        let a = Array::from_scalar(42i32).unwrap();
        assert_eq!(a.shape(), &[] as &[u64]);
    }

    #[test]
    fn scalar_0d_dtype_i32() {
        use crate::dtype::DtypeScalarKind;
        let a = Array::from_scalar(0i32).unwrap();
        assert_eq!(a.dtype().try_to_scalar(), Some(DtypeScalarKind::I32));
    }

    #[test]
    fn scalar_0d_read_i32() {
        let got: ArrayD<i32> = Array::from_scalar(7i32)
            .unwrap()
            .data()
            .to_ndarray()
            .unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![], vec![7i32]).unwrap());
    }

    #[test]
    fn scalar_0d_read_f64() {
        let got: ArrayD<f64> = Array::from_scalar(3.14f64)
            .unwrap()
            .data()
            .to_ndarray()
            .unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![], vec![3.14f64]).unwrap());
    }

    #[test]
    fn scalar_0d_read_bool() {
        let got: ArrayD<bool> = Array::from_scalar(true)
            .unwrap()
            .data()
            .to_ndarray()
            .unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![], vec![true]).unwrap());
    }

    // -----------------------------------------------------------------------
    // from_scalar_broadcast — shape [N]
    // -----------------------------------------------------------------------

    #[test]
    fn broadcast_1d_shape() {
        let a = Array::from_scalar_broadcast(5i32, &[4]).unwrap();
        assert_eq!(a.shape(), &[4u64]);
    }

    #[test]
    fn broadcast_1d_read_i32() {
        let got: ArrayD<i32> = Array::from_scalar_broadcast(5i32, &[4])
            .unwrap()
            .data()
            .to_ndarray()
            .unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![4], vec![5i32; 4]).unwrap());
    }

    #[test]
    fn broadcast_2d_shape() {
        let a = Array::from_scalar_broadcast(1u8, &[3, 4]).unwrap();
        assert_eq!(a.shape(), &[3u64, 4]);
    }

    #[test]
    fn broadcast_2d_read_u8() {
        let got: ArrayD<u8> = Array::from_scalar_broadcast(9u8, &[2, 3])
            .unwrap()
            .data()
            .to_ndarray()
            .unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![2, 3], vec![9u8; 6]).unwrap()
        );
    }

    #[test]
    fn broadcast_2d_subregion_read() {
        let got: ArrayD<i32> = Array::from_scalar_broadcast(42i32, &[5, 5])
            .unwrap()
            .data()
            .to_ndarray_sub(&[1..3, 2..4])
            .unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![2, 2], vec![42i32; 4]).unwrap()
        );
    }

    #[test]
    fn broadcast_3d_read_f32() {
        let got: ArrayD<f32> = Array::from_scalar_broadcast(1.5f32, &[2, 3, 4])
            .unwrap()
            .data()
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
        let got: ArrayD<f64> = Array::from_scalar_broadcast(7.0f64, &[3, 4])
            .unwrap()
            .max(&[0], false)
            .data()
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
        let got: ArrayD<i64> = Array::from_scalar_broadcast(2i32, &[3, 4])
            .unwrap()
            .sum(&[0], false)
            .data()
            .to_ndarray()
            .unwrap();
        assert_eq!(got, ArrayD::from_shape_vec(vec![4], vec![6i64; 4]).unwrap());
    }
}
