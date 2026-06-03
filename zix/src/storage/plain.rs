use std::ops::Range;

use crate::codec::ReadContext;
use crate::dtype::{Dtype, Dtyped};
use crate::error::{check_get_buffer_size, check_get_range, ensure, Result};
use crate::ops::ElementTypeChange;
use crate::storage::{
    ArrayStorage, ArrayStorageSpec, BlockShapeTag, BlocksLayout, ElementType, Ty, TypeDyn,
};
use crate::util::{default_strides, dim_arr, nd_copy, DimArray, SendSyncPtr};
use crate::{Array, Dimension, IntoDimension};

/// Storage type that provides a zero-copy view into an arbitrary strided buffer.
///
/// `Plain` is an adapter that allows non-compressed data to be used as the storage of an `Array`,
/// in contrast to the main library [`Compact`](crate::storage::Compact) block-compressed storage.
/// This storage is useful when regular ndarray objects need to behave like `Array`, for example
/// to participate in math operations with compressed arrays.
///
/// `Plain<A, ...>` holds a raw `*const u8` pointer into a buffer owned by `A`,
/// together with a per-dimension shape and byte-stride description. The
/// buffer may be laid out in any order (C-contiguous, Fortran-contiguous,
/// transposed, sliced with gaps, etc.) - reads use the strides to copy the
/// requested sub-region into a C-contiguous output buffer.
///
/// The type parameter `A` is the *owner* of the underlying memory, and potentially can be any type.
/// Keeping `A` alive alongside the pointer ensures the data remains valid. Two
/// concrete owners are provided:
///
/// * `Plain<Vec<T>, ...>` - owns the data, `Vec<T>` is the data allocation (see [`Array::plain_ndarray`]).
/// * `Plain<&'a (), ...>` - borrows the data, `&'a ()` is a marker for the borrowed allocation
///   (see [`Array::plain_ndarray_ref`]).
///
/// `ET: ElementType` tracks the element type at the type level and follows the same semantics as
/// [`Compact<ET, D>`](crate::storage::Compact): `ET` is inferred from the dtype argument type.
///
/// `D: Dimension` tracks the ndim at the type level and follows the same semantics as
/// [`Compact<ET, D>`](crate::storage::Compact): `D` is inferred from the shape argument type.
///
/// # Examples
///
/// Mix a plain array with a compressed array in an element-wise operation:
///
/// ```
/// # use zix::{Array, ArrayParams};
/// use ndarray::array;
///
/// let nd_compact = array![[1.0f32, 2.0], [3.0, 4.0]];
/// let compact = Array::compact_array(&nd_compact)?;
///
/// let nd_plain = array![[10.0f32, 20.0], [30.0, 40.0]];
/// let plain = Array::plain_ndarray(nd_plain)?;
///
/// // The result is computed lazily - no data is read until to_ndarray() is called.
/// let result = (compact + plain).to_ndarray()?;
/// assert_eq!(result, array![[11.0f32, 22.0], [33.0, 44.0]]);
/// # Ok::<(), zix::Error>(())
/// ```
pub struct Plain<A, ET, D> {
    #[allow(unused)]
    allocation: A,

    data: SendSyncPtr<u8>,
    shape: D,
    strides: DimArray<usize>, // in bytes
    element_type: ET,
    blocks_layout: BlocksLayout,
}
impl<A, D> Plain<A, TypeDyn, D> {
    /// Construct a `Plain` storage from a raw pointer, shape, and byte strides.
    ///
    /// `storage` is any value that owns (or keeps alive) the memory pointed to
    /// by `data`; it is stored alongside the pointer so the borrow checker can
    /// enforce lifetime constraints through `A`'s type parameter.
    ///
    /// # Arguments
    ///
    /// * `allocation` - owner of the underlying allocation.
    /// * `data` - pointer to the first element (i.e. already offset to
    ///   `[0, 0, ..., 0]` of the logical view).
    /// * `shape` - number of elements along each dimension.
    /// * `strides` - byte distance between adjacent elements along each
    ///   dimension.  Must have the same length as `shape`.
    /// * `dtype` - element type descriptor; used for itemsize and alignment
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
    pub unsafe fn new<Sh>(
        allocation: A,
        data: *const u8,
        shape: Sh,
        strides: &[usize], // in bytes
        dtype: Dtype,
    ) -> Result<Self>
    where
        D: Dimension,
        Sh: IntoDimension<Dimension = D>,
    {
        let shape = shape.into_dimension()?;
        let ndim = shape.ndim();

        ensure!(
            strides.len() == ndim,
            InvalidArgument,
            "Strides length {} does not match number of dimensions {ndim}",
            strides.len()
        );
        let strides = DimArray::from_slice(strides).unwrap();

        let alignment = dtype.alignment().as_usize();
        ensure!(
            (data as usize).is_multiple_of(alignment)
                && strides.iter().all(|&s| s.is_multiple_of(alignment)),
            InvalidArgument,
            "Data pointer or strides are not aligned to required alignment {alignment}"
        );

        let blocks_layout = BlocksLayout::tune(
            Some(dim_arr(ndim, |_| 1)),
            Some(dim_arr(ndim, |_| BlockShapeTag::Any)),
            None,
            None,
            None,
            shape.as_slice(),
            dtype.itemsize(),
        )?;

        let element_type = TypeDyn::from_dtype(dtype).unwrap();

        Ok(Self {
            allocation,
            data: unsafe { SendSyncPtr::new(data) },
            shape,
            strides,
            element_type,
            blocks_layout,
        })
    }
}

impl<T, D> Array<Plain<Vec<T>, Ty<T>, D>> {
    /// Create a [`Plain`] array that takes ownership of an `ndarray` array.
    ///
    /// The ndarray's allocation is moved into the returned `Array`; no element
    /// data is copied.  The resulting array respects the ndarray's existing
    /// memory layout (C-order, Fortran-order, transposed, etc.) and can handle
    /// non-contiguous strides.
    ///
    /// A `Plain` storage does not compress the data, and is useful when you want to treat regular
    /// ndarrays as `Array`, for example to participate in math operations with compressed arrays.
    ///
    /// # Errors
    ///
    /// Returns an error if the ndarray's number of dimensions exceeds the
    /// maximum supported ndim.
    pub fn plain_ndarray<D2>(arr: ndarray::Array<T, D2>) -> Result<Self>
    where
        T: Dtyped,
        D: Dimension,
        D2: ndarray::Dimension + IntoDimension<Dimension = D>,
    {
        let shape = arr.raw_dim().into_dimension()?;

        let strides = arr
            .strides()
            .iter()
            .map(|&s| usize::try_from(s).unwrap() * std::mem::size_of::<T>())
            .collect::<DimArray<_>>();

        let (allocation, allocation_offset) = arr.into_raw_vec_and_offset();
        let mut data_ptr = allocation.as_ptr();
        if let Some(allocation_offset) = allocation_offset {
            data_ptr = unsafe { data_ptr.add(allocation_offset) };
        }
        let data_ptr = data_ptr.cast::<u8>();

        let storage = unsafe { Plain::new(allocation, data_ptr, shape, &strides, T::DTYPE) }?;
        let array = Array::from_storage(storage);
        let array = array.into_type().unwrap();
        Ok(array)
    }
}

impl<'a, T, D> Array<Plain<&'a (), Ty<T>, D>> {
    /// Internal implementation for creating a `Plain` array that borrows from an ndarray view.
    ///
    /// Note this function does have any lifetime bounds on the input array.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the lifetime `'a` correctly reflects the lifetime of the
    /// borrowed *data* in `arr`.
    unsafe fn plain_ndarray_ref_impl<S, D2>(arr: &ndarray::ArrayBase<S, D2>) -> Result<Self>
    where
        S: ndarray::Data<Elem = T>,
        D: Dimension,
        D2: ndarray::Dimension + IntoDimension<Dimension = D>,
        T: Dtyped,
    {
        let shape = arr.raw_dim().into_dimension()?;

        let strides = arr
            .strides()
            .iter()
            .map(|&s| s as usize * std::mem::size_of::<T>())
            .collect::<DimArray<_>>();

        let data_ptr = arr.as_ptr().cast::<u8>();

        unsafe { Self::plain_ndarray_ptr(data_ptr, shape, &strides, T::DTYPE) }
    }

    /// Create a [`Plain`] array that borrows from an ndarray.
    ///
    /// No element data is copied.  The resulting array shares memory with
    /// `arr` and is valid for its lifetime `'a`.  Any layout supported by
    /// ndarray (C-order, Fortran-order, non-contiguous slices, transposed
    /// views, etc.) is handled correctly.
    ///
    /// A `Plain` storage does not compress the data, and is useful when you want to treat regular
    /// ndarrays as `Array`, for example to participate in math operations with compressed arrays.
    ///
    /// # Errors
    ///
    /// Returns an error if the ndarray's number of dimensions exceeds the
    /// maximum supported ndim.
    pub fn plain_ndarray_ref<S, D2>(arr: &'a ndarray::ArrayBase<S, D2>) -> Result<Self>
    where
        S: ndarray::Data<Elem = T>,
        D: Dimension,
        D2: ndarray::Dimension + IntoDimension<Dimension = D>,
        T: Dtyped,
    {
        // SAFETY: `arr` is 'a, so the returned `Plain` will not outlive the data in `arr`.
        unsafe { Self::plain_ndarray_ref_impl(arr) }
    }

    /// Create a [`Plain`] array that borrows from an ndarray view.
    ///
    /// Similar to [`Self::plain_ndarray_ref`] but takes an `ArrayView` instead of an `ArrayBase`.
    /// See `plain_ndarray_ref` for more details.
    pub fn plain_ndarray_view<D2>(arr: ndarray::ArrayView<'a, T, D2>) -> Result<Self>
    where
        D: Dimension,
        D2: ndarray::Dimension + IntoDimension<Dimension = D>,
        T: Dtyped,
    {
        // SAFETY: `arr` *data* is 'a, so the returned `Plain` will not outlive the data in `arr`.
        unsafe { Self::plain_ndarray_ref_impl(&arr) }
    }
}
impl<ET, D> Array<Plain<&(), ET, D>> {
    /// Create a [`Plain`] array from a raw pointer, shape, and byte strides, borrowing from an external
    /// allocation.
    ///
    /// No element data is copied.  The resulting array shares memory with the external allocation and is valid for its
    /// lifetime `'a`. The buffer may be laid out in any order (C-contiguous, Fortran-contiguous, transposed, sliced
    /// with gaps, etc.) as long as elements are aligned.
    ///
    /// A `Plain` storage does not compress the data, and is useful when you want to treat regular
    /// buffers as `Array`, for example to participate in math operations with compressed arrays.
    ///
    /// # Arguments
    ///
    /// * `data_ptr` - pointer to the first element.
    /// * `shape` - number of elements along each dimension.
    /// * `strides` - the array element strides in **bytes units**. Must have the same length as `shape`.
    /// * `dtype` - element type descriptor; used for itemsize and alignment checks.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// * `data_ptr` is a valid, non-dangling pointer for the lifetime of the returned `Array`.
    /// * The memory region reachable via `data_ptr` and `strides` is valid to read for all index combinations in
    ///   `0..shape[d]` on each dimension `d`.
    /// * `data_ptr` and all strides are aligned to `dtype.alignment()`.
    /// * Elements accessed by the data pointer must be valid for the specified `dtype`.
    pub unsafe fn plain_ndarray_ptr<Sh>(
        data_ptr: *const u8,
        shape: Sh,
        strides: &[usize],
        dtype: Dtype,
    ) -> Result<Self>
    where
        ET: ElementType,
        D: Dimension,
        Sh: IntoDimension<Dimension = D>,
    {
        let allocation = &();
        let storage = unsafe { Plain::new(allocation, data_ptr, shape, strides, dtype) }?;
        let array = Array::from_storage(storage);
        let array = array.into_type()?;
        Ok(array)
    }
}

impl<A, ET, D> ArrayStorage for Plain<A, ET, D>
where
    ET: ElementType,
    D: Dimension,
{
    type ElementType = ET;
    type Dimension = D;

    fn read_data(
        &self,
        index: &[Range<u64>],
        buf: &mut [u8],
        _context: &ReadContext,
    ) -> Result<()> {
        let dtype = self.dtype();
        let itemsize = dtype.itemsize() as usize;
        check_get_range(self.shape(), index)?;
        check_get_buffer_size(index, dtype, buf)?;

        let ndim = self.shape.ndim();
        let out_shape = dim_arr(ndim, |dim| (index[dim].end - index[dim].start) as usize);
        let out_strides = default_strides(&out_shape, itemsize);

        let in_offset = (0..ndim)
            .map(|dim| index[dim].start as usize * self.strides[dim])
            .sum::<usize>();
        let src_ptr = unsafe { self.data.as_ptr().add(in_offset) };
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
        self.shape.as_slice()
    }
    fn dtype(&self) -> &Dtype {
        self.element_type.dtype()
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

impl<A, ET, D> ElementTypeChange for Plain<A, ET, D>
where
    ET: ElementType,
    D: Dimension,
{
    type ElementTypeChange<NewET: ElementType> = Plain<A, NewET, D>;

    fn change_type<NewET: ElementType>(self) -> Result<Self::ElementTypeChange<NewET>> {
        Ok(Plain {
            allocation: self.allocation,
            data: self.data,
            shape: self.shape,
            strides: self.strides,
            element_type: NewET::from_dtype(self.element_type.dtype().clone())?,
            blocks_layout: self.blocks_layout,
        })
    }
}

impl<A, ET, D> crate::ops::DimensionChange for Plain<A, ET, D>
where
    ET: ElementType,
    D: Dimension,
{
    type DimensionChange<NewD: Dimension> = Plain<A, ET, NewD>;

    fn dimension_change<NewD: Dimension>(self) -> Result<Self::DimensionChange<NewD>> {
        let shape = NewD::from_slice(self.shape())?;
        Ok(Plain {
            allocation: self.allocation,
            data: self.data,
            shape,
            strides: self.strides,
            element_type: self.element_type,
            blocks_layout: self.blocks_layout,
        })
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{array, s, ArrayD};

    use crate::codec::ReadContext;
    use crate::Array;

    // -----------------------------------------------------------------------
    // plain_ndarray (owned)
    // -----------------------------------------------------------------------

    #[test]
    fn owned_1d_shape() {
        let a = Array::plain_ndarray(array![1i32, 2, 3]).unwrap();
        assert_eq!(a.shape(), &[3u64]);
    }

    #[test]
    fn owned_1d_dtype_i32() {
        use crate::dtype::DtypeScalarKind;
        let a = Array::plain_ndarray(array![0i32]).unwrap();
        assert_eq!(a.dtype().try_to_scalar(), Some(DtypeScalarKind::I32));
    }

    #[test]
    fn owned_1d_read_i32() {
        let nd = array![10i32, 20, 30];
        let got = Array::plain_ndarray(nd.clone())
            .unwrap()
            .to_ndarray()
            .unwrap();
        assert_eq!(got, nd);
    }

    #[test]
    fn owned_2d_read_i32() {
        let nd = array![[1i32, 2, 3], [4, 5, 6]];
        let got = Array::plain_ndarray(nd.clone())
            .unwrap()
            .to_ndarray()
            .unwrap();
        assert_eq!(got, nd);
    }

    #[test]
    fn owned_2d_subregion_read() {
        let nd = array![[1i32, 2, 3], [4, 5, 6], [7, 8, 9]];
        let got = Array::plain_ndarray(nd.clone())
            .unwrap()
            .to_ndarray_sub(&[1..3, 0..2], &ReadContext::default())
            .unwrap();
        assert_eq!(got, array![[4i32, 5], [7, 8]]);
    }

    #[test]
    fn owned_3d_read_f32() {
        let vals = (0..24).map(|x| x as f32).collect::<Vec<_>>();
        let nd = ArrayD::from_shape_vec(vec![2, 3, 4], vals.clone()).unwrap();
        let got = Array::plain_ndarray(nd.clone())
            .unwrap()
            .to_ndarray()
            .unwrap();
        assert_eq!(got, nd);
    }

    #[test]
    fn owned_read_f64() {
        let nd = array![[1.0f64, 2.0], [3.0, 4.0]];
        let got = Array::plain_ndarray(nd.clone())
            .unwrap()
            .to_ndarray()
            .unwrap();
        assert_eq!(got, nd);
    }

    #[test]
    fn owned_read_bool() {
        let nd = array![[true, false], [false, true]];
        let got = Array::plain_ndarray(nd.clone())
            .unwrap()
            .to_ndarray()
            .unwrap();
        assert_eq!(got, nd);
    }

    // Non-contiguous (transposed) array - column-major strides
    #[test]
    fn owned_non_contiguous_transposed() {
        let nd = array![[1i32, 2, 3], [4, 5, 6]];
        let transposed = nd.clone().reversed_axes(); // shape [3,2], strides swapped
        let got = Array::plain_ndarray(transposed.clone())
            .unwrap()
            .to_ndarray()
            .unwrap();
        assert_eq!(got, transposed);
    }

    // -----------------------------------------------------------------------
    // plain_ndarray_ref (borrowed)
    // -----------------------------------------------------------------------

    #[test]
    fn view_1d_shape() {
        let nd = array![1i32, 2, 3];
        let a = Array::plain_ndarray_ref(&nd).unwrap();
        assert_eq!(a.shape(), &[3u64]);
    }

    #[test]
    fn view_1d_read_i32() {
        let nd = array![10i32, 20, 30];
        let a = Array::plain_ndarray_ref(&nd).unwrap();
        let got = a.to_ndarray().unwrap();
        assert_eq!(got, nd);
    }

    #[test]
    fn view_2d_read_i32() {
        let nd = array![[1i32, 2, 3], [4, 5, 6]];
        let a = Array::plain_ndarray_ref(&nd).unwrap();
        let got = a.to_ndarray().unwrap();
        assert_eq!(got, nd);
    }

    #[test]
    fn view_2d_subregion_read() {
        let nd = array![[1i32, 2, 3], [4, 5, 6], [7, 8, 9]];
        let a = Array::plain_ndarray_ref(&nd).unwrap();
        let got = a.to_ndarray_sub(&[1..3, 1..3], &a.read_ctx()).unwrap();
        assert_eq!(got, array![[5i32, 6], [8, 9]]);
    }

    #[test]
    fn view_non_contiguous_slice() {
        // Take every-other column via an ndarray slice, then read it back.
        let nd = array![[1i32, 2, 3, 4], [5, 6, 7, 8]];
        let sliced = nd.slice(s![.., ..;2]); // columns 0 and 2: [[1,3],[5,7]]
        let a = Array::plain_ndarray_ref(&sliced).unwrap();
        let got = a.to_ndarray().unwrap();
        assert_eq!(got, array![[1i32, 3], [5, 7]]);
    }

    #[test]
    fn view_transposed_read() {
        let nd = array![[1i32, 2, 3], [4, 5, 6]];
        let t = nd.t(); // shape [3, 2]
        let a = Array::plain_ndarray_ref(&t).unwrap();
        let got = a.to_ndarray().unwrap();
        // t[[0,0]]=1, t[[0,1]]=4, t[[1,0]]=2, t[[1,1]]=5, t[[2,0]]=3, t[[2,1]]=6
        assert_eq!(got, array![[1i32, 4], [2, 5], [3, 6]]);
    }

    // -----------------------------------------------------------------------
    // reduction on top of Plain storage
    // -----------------------------------------------------------------------

    #[test]
    fn max_over_plain_2d() {
        let nd = array![[1i32, 5, 3], [4, 2, 6]];
        let got = Array::plain_ndarray(nd)
            .unwrap()
            .max(0)
            .to_ndarray()
            .unwrap();
        assert_eq!(got, array![4i32, 5, 6]);
    }

    #[test]
    fn sum_over_plain_2d() {
        let nd = array![[1i32, 2, 3], [4, 5, 6]];
        let got = Array::plain_ndarray(nd)
            .unwrap()
            .sum(1)
            .to_ndarray()
            .unwrap();
        assert_eq!(got, array![6i64, 15]);
    }
}
