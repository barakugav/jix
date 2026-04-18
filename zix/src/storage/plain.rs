use std::io::{self};
use std::marker::PhantomData;
use std::ops::Range;

use crate::codec::ReadContext;
use crate::dtype::{Dtype, Dtyped};
use crate::storage::{ArrayStorage, ArrayStorageSpec, BlockShapeTag, BlocksLayout};
use crate::util::default_strides;
use crate::util::{DimArray, dim_arr, nd_copy};
use crate::{Array, NDIM_MAX};

/// Storage type that provides a zero-copy view into an arbitrary strided buffer.
///
/// `Plain<S>` holds a raw `*const u8` pointer into a buffer owned by `S`,
/// together with a per-dimension shape and byte-stride description.  The
/// buffer may be laid out in any order (C-contiguous, Fortran-contiguous,
/// transposed, sliced with gaps, etc.) — reads use the strides to copy the
/// requested sub-region into a C-contiguous output buffer.
///
/// The type parameter `S` is the *owner* of the underlying memory.  Keeping
/// `S` alive alongside the pointer ensures the data remains valid.  Two
/// concrete owners are provided:
///
/// * `Plain<Vec<T>>` — owns the data (see [`Array::from_ndarray_plain`]).
/// * `Plain<PlainRef<'a, T>>` — borrows from an `ndarray` view
///   (see [`Array::from_ndarray_view_plain`]).
///
/// # Safety
///
/// The raw pointer is not checked after construction.  Callers of
/// [`Plain::new`] must ensure the pointer and strides are valid for the
/// lifetime of the `Plain` value.
pub struct Plain<S> {
    #[allow(unused)]
    storage: S,

    data: *const u8,
    shape: DimArray<u64>,
    strides: DimArray<usize>, // in bytes
    dtype: Dtype,
    blocks_layout: BlocksLayout,
}
impl<S> Plain<S> {
    /// Construct a `Plain` storage from a raw pointer, shape, and byte strides.
    ///
    /// `storage` is any value that owns (or keeps alive) the memory pointed to
    /// by `data`; it is stored alongside the pointer so the borrow checker can
    /// enforce lifetime constraints through `S`'s type parameter.
    ///
    /// # Arguments
    ///
    /// * `storage` — owner of the underlying allocation.
    /// * `data` — pointer to the first element (i.e. already offset to
    ///   `[0, 0, ..., 0]` of the logical view).
    /// * `shape` — number of elements along each dimension.
    /// * `strides` — byte distance between adjacent elements along each
    ///   dimension.  Must have the same length as `shape`.
    /// * `dtype` — element type descriptor; used for itemsize and alignment
    ///   checks.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// * `shape.len()` exceeds the maximum supported number of dimensions.
    /// * `strides.len() != shape.len()`.
    /// * `data` or any stride is not aligned to `dtype.alignment()`.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// * `data` is a valid, non-dangling pointer for the lifetime of the
    ///   returned `Plain`.
    /// * The memory region reachable via `data` and `strides` is valid to
    ///   read for all index combinations in `0..shape[d]` on each dimension
    ///   `d`.
    pub unsafe fn new(
        storage: S,
        data: *const u8,
        shape: &[u64],
        strides: &[usize], // in bytes
        dtype: Dtype,
    ) -> io::Result<Self> {
        let ndim = shape.len();
        if ndim > NDIM_MAX {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("array ndim {ndim} exceeds maximum supported ndim {NDIM_MAX}"),
            ));
        }
        let shape: DimArray<_> = shape.try_into().unwrap();

        if strides.len() != ndim {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "strides has different ndim {} than shape {}",
                    strides.len(),
                    ndim
                ),
            ));
        }
        let strides: DimArray<_> = strides.try_into().unwrap();

        let alignment = dtype.alignment() as usize;
        if (data as usize % alignment != 0) || (strides.iter().any(|&s| s % alignment != 0)) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "data pointer and strides must be aligned to dtype alignment {}",
                    alignment
                ),
            ));
        }

        let blocks_layout = BlocksLayout::new(
            Some(dim_arr(ndim, |_| 1)),
            Some(dim_arr(ndim, |_| BlockShapeTag::Any)),
            None,
            None,
            None,
            &shape,
            dtype.itemsize(),
        );

        Ok(Self {
            storage,
            data,
            shape,
            strides,
            dtype,
            blocks_layout,
        })
    }
}

impl<T> Array<Plain<Vec<T>>> {
    /// Create a [`Plain`] array that takes ownership of an `ndarray` array.
    ///
    /// The ndarray's allocation is moved into the returned `Array`; no element
    /// data is copied.  The resulting array respects the ndarray's existing
    /// memory layout (C-order, Fortran-order, transposed, etc.) and can handle
    /// non-contiguous strides.
    ///
    /// # Errors
    ///
    /// Returns an error if the ndarray's number of dimensions exceeds the
    /// maximum supported ndim.
    pub fn from_ndarray_plain<D>(arr: ndarray::Array<T, D>) -> io::Result<Self>
    where
        T: Dtyped,
        D: ndarray::Dimension,
    {
        let shape = arr.shape();
        if shape.len() > NDIM_MAX {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "array ndim {} exceeds maximum supported ndim {}",
                    shape.len(),
                    NDIM_MAX
                ),
            ));
        }
        let ndim = shape.len();
        let shape = dim_arr(ndim, |dim| shape[dim] as u64);

        let strides = arr
            .strides()
            .iter()
            .map(|&s| s as usize * std::mem::size_of::<T>())
            .collect::<DimArray<_>>();

        let (allocation, allocation_offset) = arr.into_raw_vec_and_offset();
        let mut data_ptr = allocation.as_ptr();
        if let Some(allocation_offset) = allocation_offset {
            data_ptr = unsafe { data_ptr.add(allocation_offset) };
        }
        let data_ptr = data_ptr.cast::<u8>();

        let storage = unsafe { Plain::new(allocation, data_ptr, &shape, &strides, T::DTYPE) }?;
        Ok(Self::from_storage(storage))
    }
}

/// Marker type used as the `S` parameter of [`Plain`] when borrowing from an
/// ndarray view rather than owning the allocation.
///
/// The lifetime `'a` ties the [`Plain`] storage to the ndarray it was created
/// from, so the borrow checker prevents the underlying data from being freed
/// while the `Plain` array is still alive.
pub struct PlainRef<'a, T>(PhantomData<&'a T>);

impl<'a, T> Array<Plain<PlainRef<'a, T>>> {
    /// Create a [`Plain`] array that borrows from an ndarray view.
    ///
    /// No element data is copied.  The resulting array shares memory with
    /// `arr` and is valid for its lifetime `'a`.  Any layout supported by
    /// ndarray (C-order, Fortran-order, non-contiguous slices, transposed
    /// views, etc.) is handled correctly.
    ///
    /// # Errors
    ///
    /// Returns an error if the ndarray's number of dimensions exceeds the
    /// maximum supported ndim.
    pub fn from_ndarray_view_plain<S, D>(arr: &ndarray::ArrayBase<S, D>) -> io::Result<Self>
    where
        T: Dtyped,
        S: ndarray::Data<Elem = T>,
        D: ndarray::Dimension,
    {
        let shape = arr.shape();
        if shape.len() > NDIM_MAX {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "array ndim {} exceeds maximum supported ndim {}",
                    shape.len(),
                    NDIM_MAX
                ),
            ));
        }
        let ndim = shape.len();
        let shape = dim_arr(ndim, |dim| shape[dim] as u64);

        let strides = arr
            .strides()
            .iter()
            .map(|&s| s as usize * std::mem::size_of::<T>())
            .collect::<DimArray<_>>();

        let data_ptr = arr.as_ptr().cast::<u8>();
        let allocation = PlainRef(PhantomData);

        let storage = unsafe { Plain::new(allocation, data_ptr, &shape, &strides, T::DTYPE) }?;
        Ok(Self::from_storage(storage))
    }
}

impl<S> ArrayStorage for Plain<S> {
    fn read_data(
        &self,
        index: &[Range<u64>],
        buf: &mut [u8],
        _context: &ReadContext,
    ) -> io::Result<()> {
        let ndim = self.shape.len();
        assert_eq!(index.len(), ndim);
        let itemsize = self.dtype.itemsize() as usize;
        let out_shape = dim_arr(ndim, |dim| (index[dim].end - index[dim].start) as usize);
        let out_size = out_shape.iter().product::<usize>();
        if buf.len() != out_size * itemsize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "output buffer has incorrect size: expected {} bytes, actual {} bytes",
                    out_size * itemsize,
                    buf.len()
                ),
            ));
        }
        let out_strides = default_strides(&out_shape, itemsize);

        let in_offset = (0..ndim)
            .map(|dim| index[dim].start as usize * self.strides[dim])
            .sum::<usize>();
        let src_ptr = unsafe { self.data.add(in_offset) };
        let dst_ptr = buf.as_mut_ptr();

        unsafe {
            nd_copy(
                src_ptr,
                dst_ptr,
                &out_shape,
                &self.strides,
                &out_strides,
                itemsize,
            )
        };

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
    use ndarray::{ArrayD, IxDyn, s};

    use crate::Array;

    // -----------------------------------------------------------------------
    // from_ndarray_plain (owned)
    // -----------------------------------------------------------------------

    #[test]
    fn owned_1d_shape() {
        let a = Array::from_ndarray_plain(ndarray::array![1i32, 2, 3].into_dyn()).unwrap();
        assert_eq!(a.shape(), &[3u64]);
    }

    #[test]
    fn owned_1d_dtype_i32() {
        use crate::dtype::DtypeScalarKind;
        let a = Array::from_ndarray_plain(ndarray::array![0i32].into_dyn()).unwrap();
        assert_eq!(a.dtype().try_to_scalar(), Some(DtypeScalarKind::I32));
    }

    #[test]
    fn owned_1d_read_i32() {
        let nd = ndarray::array![10i32, 20, 30].into_dyn();
        let got: ArrayD<i32> = Array::from_ndarray_plain(nd.clone())
            .unwrap()
            .data()
            .to_ndarray()
            .unwrap();
        assert_eq!(got, nd);
    }

    #[test]
    fn owned_2d_read_i32() {
        let nd = ndarray::array![[1i32, 2, 3], [4, 5, 6]].into_dyn();
        let got: ArrayD<i32> = Array::from_ndarray_plain(nd.clone())
            .unwrap()
            .data()
            .to_ndarray()
            .unwrap();
        assert_eq!(got, nd);
    }

    #[test]
    fn owned_2d_subregion_read() {
        let nd = ndarray::array![[1i32, 2, 3], [4, 5, 6], [7, 8, 9]].into_dyn();
        let got: ArrayD<i32> = Array::from_ndarray_plain(nd.clone())
            .unwrap()
            .data()
            .to_ndarray_sub(&[1..3, 0..2])
            .unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![2, 2], vec![4i32, 5, 7, 8]).unwrap()
        );
    }

    #[test]
    fn owned_3d_read_f32() {
        let vals: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let nd = ArrayD::from_shape_vec(vec![2, 3, 4], vals.clone()).unwrap();
        let got: ArrayD<f32> = Array::from_ndarray_plain(nd.clone())
            .unwrap()
            .data()
            .to_ndarray()
            .unwrap();
        assert_eq!(got, nd);
    }

    #[test]
    fn owned_read_f64() {
        let nd = ndarray::array![[1.0f64, 2.0], [3.0, 4.0]].into_dyn();
        let got: ArrayD<f64> = Array::from_ndarray_plain(nd.clone())
            .unwrap()
            .data()
            .to_ndarray()
            .unwrap();
        assert_eq!(got, nd);
    }

    #[test]
    fn owned_read_bool() {
        let nd = ndarray::array![[true, false], [false, true]].into_dyn();
        let got: ArrayD<bool> = Array::from_ndarray_plain(nd.clone())
            .unwrap()
            .data()
            .to_ndarray()
            .unwrap();
        assert_eq!(got, nd);
    }

    // Non-contiguous (transposed) array — column-major strides
    #[test]
    fn owned_non_contiguous_transposed() {
        let nd = ndarray::array![[1i32, 2, 3], [4, 5, 6]].into_dyn();
        let transposed = nd.clone().reversed_axes(); // shape [3,2], strides swapped
        let got: ArrayD<i32> = Array::from_ndarray_plain(transposed.clone())
            .unwrap()
            .data()
            .to_ndarray()
            .unwrap();
        assert_eq!(got, transposed);
    }

    // -----------------------------------------------------------------------
    // from_ndarray_view_plain (borrowed)
    // -----------------------------------------------------------------------

    #[test]
    fn view_1d_shape() {
        let nd = ndarray::array![1i32, 2, 3].into_dyn();
        let a =
            Array::<crate::storage::Plain<_>>::from_ndarray_view_plain::<_, IxDyn>(&nd).unwrap();
        assert_eq!(a.shape(), &[3u64]);
    }

    #[test]
    fn view_1d_read_i32() {
        let nd = ndarray::array![10i32, 20, 30].into_dyn();
        let a =
            Array::<crate::storage::Plain<_>>::from_ndarray_view_plain::<_, IxDyn>(&nd).unwrap();
        let got: ArrayD<i32> = a.data().to_ndarray().unwrap();
        assert_eq!(got, nd);
    }

    #[test]
    fn view_2d_read_i32() {
        let nd = ndarray::array![[1i32, 2, 3], [4, 5, 6]].into_dyn();
        let a =
            Array::<crate::storage::Plain<_>>::from_ndarray_view_plain::<_, IxDyn>(&nd).unwrap();
        let got: ArrayD<i32> = a.data().to_ndarray().unwrap();
        assert_eq!(got, nd);
    }

    #[test]
    fn view_2d_subregion_read() {
        let nd = ndarray::array![[1i32, 2, 3], [4, 5, 6], [7, 8, 9]].into_dyn();
        let a =
            Array::<crate::storage::Plain<_>>::from_ndarray_view_plain::<_, IxDyn>(&nd).unwrap();
        let got: ArrayD<i32> = a.data().to_ndarray_sub(&[1..3, 1..3]).unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![2, 2], vec![5i32, 6, 8, 9]).unwrap()
        );
    }

    #[test]
    fn view_non_contiguous_slice() {
        // Take every-other column via an ndarray slice, then read it back.
        let nd = ndarray::array![[1i32, 2, 3, 4], [5, 6, 7, 8]].into_dyn();
        let sliced = nd.slice(s![.., ..;2]).into_dyn(); // columns 0 and 2: [[1,3],[5,7]]
        let a = Array::<crate::storage::Plain<_>>::from_ndarray_view_plain::<_, IxDyn>(&sliced)
            .unwrap();
        let got: ArrayD<i32> = a.data().to_ndarray().unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![2, 2], vec![1i32, 3, 5, 7]).unwrap()
        );
    }

    #[test]
    fn view_transposed_read() {
        let nd = ndarray::array![[1i32, 2, 3], [4, 5, 6]].into_dyn();
        let t = nd.t().into_dyn(); // shape [3, 2]
        let a = Array::<crate::storage::Plain<_>>::from_ndarray_view_plain::<_, IxDyn>(&t).unwrap();
        let got: ArrayD<i32> = a.data().to_ndarray().unwrap();
        // t[[0,0]]=1, t[[0,1]]=4, t[[1,0]]=2, t[[1,1]]=5, t[[2,0]]=3, t[[2,1]]=6
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![3, 2], vec![1i32, 4, 2, 5, 3, 6]).unwrap()
        );
    }

    // -----------------------------------------------------------------------
    // reduction on top of Plain storage
    // -----------------------------------------------------------------------

    #[test]
    fn max_over_plain_2d() {
        let nd = ndarray::array![[1i32, 5, 3], [4, 2, 6]].into_dyn();
        let got: ArrayD<i32> = Array::from_ndarray_plain(nd)
            .unwrap()
            .max(&[0], false)
            .data()
            .to_ndarray()
            .unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![3], vec![4i32, 5, 6]).unwrap()
        );
    }

    #[test]
    fn sum_over_plain_2d() {
        let nd = ndarray::array![[1i32, 2, 3], [4, 5, 6]].into_dyn();
        let got: ArrayD<i64> = Array::from_ndarray_plain(nd)
            .unwrap()
            .sum(&[1], false)
            .data()
            .to_ndarray()
            .unwrap();
        assert_eq!(
            got,
            ArrayD::from_shape_vec(vec![2], vec![6i64, 15]).unwrap()
        );
    }
}
