use std::marker::PhantomData;

use crate::dtype::{Dtype, Dtyped};
use crate::element_type::check_element_type;
use crate::error::{bail, check_ndim, ensure, error, Result};
use crate::storage::params::{
    read_layout_order_insert_dont_care_dim, ArraySpecFlags, ArraySpecOwned, ReadTiling,
};
use crate::storage::{
    check_out_buf, write_out_buf_slot, ArraySpec, ArrayStorageInfo, ElementType, Index, OutBuf,
    OutBufInner, StridedBuf, TypeDyn,
};
use crate::util::{dim_arr, DimArray, DimIdx, IterExt, SendSyncPtr};
use crate::{Array, CompactParams, DimDyn, Dimension, PlainArray, ReadContext, StorageBase};

/// Uncompressed nd-array storage that owns its data in a `Vec<T>`.
///
/// Same as [`PlainBase`] (see there for details) with a `Vec<T>` as the owner of the data.
///
/// Created by [`Array::plain_ndarray`] and [`Array::to_plain`].
///
/// Its data is accessible directly, without a copy, through [`PlainArrayMethods`].
pub struct Plain<T>(PlainBase<Vec<T>, T>);

/// Uncompressed nd-array storage that borrows its data.
///
/// Same as [`PlainBase`] (see there for details) but borrows the data for the lifetime `'a`
/// instead of owning it.
///
/// Created by [`Array::plain_ndarray_ref`], [`Array::plain_ndarray_view`] and
/// [`Array::plain_ndarray_ptr`].
///
/// Its data is accessible directly, without a copy, through [`PlainArrayMethods`].
pub struct PlainView<'a, T>(PlainBase<&'a (), T>);

/// Storage type that provides a zero-copy view into an arbitrary strided buffer.
///
/// `PlainBase` lets uncompressed data be the storage of an `Array`, unlike the block-compressed
/// [`Compact`](crate::storage::Compact). It is useful when regular ndarrays need to behave like
/// `Array`, for example to participate in math operations with compressed arrays.
///
/// `PlainBase<A, ...>` holds a raw pointer into a buffer owned by `A`, together with a
/// per-dimension shape and byte strides. The buffer may be laid out in any order (C-contiguous,
/// Fortran-contiguous, transposed, sliced with gaps, etc.): a pull read lends a view of the
/// requested sub-region at these strides, and a push read copies it into the destination.
///
/// The type parameter `A` is the *owner* of the underlying memory and can be any type; keeping it
/// alongside the pointer keeps the data valid. Most code uses one of the two aliases:
///
/// * [`Plain<T>`] - owns the data, `Vec<T>` is the data allocation (see
///   [`Array::plain_ndarray`]).
/// * [`PlainView<'a, ET>`] - borrows the data, `&'a ()` is a marker for the borrowed allocation
///   (see [`Array::plain_ndarray_ref`]).
///
/// Other owners, e.g. a Python object keeping a NumPy buffer alive, use `PlainBase` directly
/// through [`PlainBase::new`].
///
/// Its data is accessible directly, without a copy, through [`PlainArrayMethods`].
///
/// `ET: ElementType` tracks the element type at the type level and follows the same semantics as
/// [`Compact<ET>`](crate::storage::Compact): the ndarray constructors give e.g. `Plain<f32>`,
/// while [`PlainBase::new`] and [`Array::plain_ndarray_ptr`] start from a runtime [`Dtype`].
///
/// # Examples
///
/// Mix a plain array with a compressed array in an element-wise operation:
///
/// ```
/// # use jix::Array;
/// use ndarray::array;
///
/// let nd_compact = array![[1.0f32, 2.0], [3.0, 4.0]];
/// let compact = Array::compact_ndarray(&nd_compact)?;
///
/// let nd_plain = array![[10.0f32, 20.0], [30.0, 40.0]];
/// let plain = Array::plain_ndarray(nd_plain)?;
///
/// // The result is computed lazily - no data is read until to_ndarray() is called.
/// let result = (compact + plain).to_ndarray()?;
/// assert_eq!(result, array![[11.0f32, 22.0], [33.0, 44.0]]);
/// # Ok::<(), jix::Error>(())
/// ```
pub struct PlainBase<A, ET> {
    #[allow(unused)]
    allocation: A,

    data: SendSyncPtr<u8>,
    shape: DimDyn,
    strides: DimArray<usize>, // in bytes
    dtype: Dtype,
    element_type: PhantomData<ET>,
    spec: ArraySpecOwned,
}
impl<A> PlainBase<A, TypeDyn> {
    /// Construct a `PlainBase` storage from a raw pointer, shape, and byte strides.
    ///
    /// # Arguments
    ///
    /// * `allocation` - any value that owns (or keeps alive) the memory behind `data`; stored
    ///   alongside the pointer so the borrow checker enforces its lifetime through `A`.
    /// * `data` - pointer to the first element (i.e. already offset to
    ///   `[0, 0, ..., 0]` of the logical view).
    /// * `shape` - number of elements along each dimension.
    /// * `strides` - byte distance between adjacent elements along each
    ///   dimension. Must have the same length as `shape`.
    /// * `dtype` - element type descriptor; used for itemsize.
    ///
    /// `data` and `strides` need not be aligned to `dtype.alignment()`: a `Plain` read lends a
    /// view of the caller's own bytes, and per the [`read_data`](StorageBase::read_data)
    /// contract a lent view is only as aligned as the bytes behind it. Consumers that need
    /// alignment stage the region themselves.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// * `shape.len()` exceeds the maximum supported number of dimensions.
    /// * `strides.len() != shape.len()`.
    /// * The number of elements exceeds `isize::MAX`.
    /// * `dtype` has itemsize 0, or the byte size of `shape` overflows.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// * `data` is a valid, non-dangling pointer for the lifetime of the
    ///   returned `PlainBase`.
    /// * The memory region reachable via `data` and `strides` is valid to
    ///   read for all index combinations in `0..shape[d]` on each dimension
    ///   `d`.
    /// * Every byte from the first to the last element (the strided span, gaps between elements
    ///   included) lies in one allocation, and none of it is written, or borrowed through a
    ///   `&mut`, while the array is read or while a buffer returned by a pull read of it (or of a
    ///   lazy view over it) is alive.
    pub unsafe fn new(
        allocation: A,
        data: *const u8,
        shape: &[u64],
        strides: &[usize], // in bytes
        dtype: Dtype,
    ) -> Result<Self> {
        check_ndim(shape.len())?;
        let shape = DimDyn::from_slice(shape);
        let ndim = shape.ndim();

        ensure!(
            strides.len() == ndim,
            InvalidArgument,
            "Strides length {} does not match number of dimensions {ndim}",
            strides.len()
        );
        let strides = dim_arr(ndim, |d| strides[d]);

        let nitems = shape
            .as_slice()
            .iter()
            .map(|&len| len.max(1))
            .try_shape_product();
        ensure!(
            nitems.is_some_and(|n| n <= isize::MAX as u64),
            InvalidShapeOperation,
            "Shape {:?} has too many elements",
            shape.as_slice()
        );

        let mut spec = CompactParams::default().into_spec(
            shape.as_slice(),
            &dtype,
            ArraySpecFlags::new().with_read_tiling(ReadTiling::ZeroCopy),
        )?;
        spec.dynamic_mut().block_shape = dim_arr(ndim, |_| 1);

        // Layout order by contiguity: the outermost (largest stride) real dim comes first.
        {
            let shape = shape.as_slice();
            let degenerate = |d: usize| shape[d] <= 1;
            let broadcast = |d: usize| !degenerate(d) && strides[d] == 0;
            let regular = |d: usize| !degenerate(d) && !broadcast(d);

            // sort "regular" dims
            let mut order = (0..ndim)
                .filter(|&d| regular(d))
                .map(|d| d as DimIdx)
                .collect::<DimArray<_>>();
            order.sort_by_key(|&d| (std::cmp::Reverse(strides[d as usize]), d));

            // Insert broadcasted dims at the beginning of the order, outermost
            for broadcasted_dim in (0..ndim)
                .filter(|&d| broadcast(d))
                .map(|d| d as DimIdx)
                .rev()
            {
                order.insert(0, broadcasted_dim);
            }

            // dims with size<=1 are "dont care"
            for d in (0..ndim).filter(|&d| degenerate(d)) {
                read_layout_order_insert_dont_care_dim(&mut order, d);
            }

            let dynamic = spec.dynamic_mut();
            dynamic.read_shape_scale_order = crate::storage::params::scale_order_from_weights(
                &dynamic.read_shape_scale_weight,
                &order,
            );
            dynamic.read_layout_order = order;
        }

        Ok(Self {
            allocation,
            data: unsafe { SendSyncPtr::new(data) },
            shape,
            strides,
            dtype,
            element_type: PhantomData,
            spec,
        })
    }
}

impl<T> PlainArray<T> {
    /// Create a [`Plain`] array that takes ownership of an `ndarray` array.
    ///
    /// The ndarray's allocation is moved into the returned `Array`; no element
    /// data is copied. The resulting array respects the ndarray's existing
    /// memory layout (C-order, Fortran-order, transposed, etc.) and can handle
    /// non-contiguous strides.
    ///
    /// A `Plain` storage does not compress the data, and is useful when you want to treat regular
    /// ndarrays as `Array`, for example to participate in math operations with compressed arrays.
    ///
    /// # Errors
    ///
    /// Returns an error if the ndarray's number of dimensions exceeds the
    /// maximum supported ndim, or if it has a negative stride (e.g. a reversed axis).
    pub fn plain_ndarray<D>(arr: ndarray::Array<T, D>) -> Result<Self>
    where
        T: Dtyped,
        D: ndarray::Dimension,
    {
        let shape = shape_from_ndarray(arr.shape())?;

        let strides = strides_from_ndarray::<T>(arr.shape(), arr.strides())?;

        let (allocation, allocation_offset) = arr.into_raw_vec_and_offset();
        let mut data_ptr = allocation.as_ptr();
        if let Some(allocation_offset) = allocation_offset {
            data_ptr = unsafe { data_ptr.add(allocation_offset) };
        }
        let data_ptr = data_ptr.cast::<u8>();

        let storage =
            unsafe { PlainBase::new(allocation, data_ptr, shape.as_slice(), &strides, T::DTYPE) }?;
        Ok(Array::from_storage(Plain(
            storage.element_type_change::<T>()?,
        )))
    }
}

impl<'a, T> Array<PlainView<'a, T>> {
    /// Internal implementation for creating a `Plain` array that borrows from an ndarray view.
    ///
    /// Note this function does not have any lifetime bounds on the input array.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the lifetime `'a` correctly reflects the lifetime of the
    /// borrowed *data* in `arr`.
    unsafe fn plain_ndarray_ref_impl<S, D>(arr: &ndarray::ArrayBase<S, D>) -> Result<Self>
    where
        S: ndarray::Data<Elem = T>,
        D: ndarray::Dimension,
        T: Dtyped,
    {
        let shape = shape_from_ndarray(arr.shape())?;
        let strides = strides_from_ndarray::<T>(arr.shape(), arr.strides())?;
        let data_ptr = arr.as_ptr().cast::<u8>();
        unsafe { Self::plain_ndarray_ptr(data_ptr, shape.as_slice(), &strides, T::DTYPE) }
    }

    /// Create a [`PlainView`] array that borrows from an ndarray.
    ///
    /// No element data is copied. The resulting array shares memory with
    /// `arr` and is valid for its lifetime `'a`. Any layout supported by
    /// ndarray (C-order, Fortran-order, non-contiguous slices, transposed
    /// views, etc.) is handled correctly, except negative strides.
    ///
    /// A `Plain` storage does not compress the data, and is useful when you want to treat regular
    /// ndarrays as `Array`, for example to participate in math operations with compressed arrays.
    ///
    /// # Errors
    ///
    /// Returns an error if the ndarray's number of dimensions exceeds the
    /// maximum supported ndim, or if it has a negative stride (e.g. a reversed axis).
    pub fn plain_ndarray_ref<S, D>(arr: &'a ndarray::ArrayBase<S, D>) -> Result<Self>
    where
        S: ndarray::Data<Elem = T>,
        D: ndarray::Dimension,
        T: Dtyped,
    {
        // SAFETY: `arr` is 'a, so the returned `Plain` will not outlive the data in `arr`.
        unsafe { Self::plain_ndarray_ref_impl(arr) }
    }

    /// Create a [`PlainView`] array that borrows from an ndarray view.
    ///
    /// Similar to [`Self::plain_ndarray_ref`] but takes an `ArrayView` instead of an `ArrayBase`.
    /// See `plain_ndarray_ref` for more details.
    pub fn plain_ndarray_view<D>(arr: ndarray::ArrayView<'a, T, D>) -> Result<Self>
    where
        D: ndarray::Dimension,
        T: Dtyped,
    {
        // SAFETY: `arr` *data* is 'a, so the returned `Plain` will not outlive the data in `arr`.
        unsafe { Self::plain_ndarray_ref_impl(&arr) }
    }
}
impl<ET> Array<PlainView<'_, ET>> {
    /// Create a [`PlainView`] array from a raw pointer, shape, and byte strides, borrowing from an
    /// external allocation.
    ///
    /// No element data is copied: the resulting array shares memory with the external allocation.
    /// The buffer may be laid out in any order (C-contiguous, Fortran-contiguous, transposed,
    /// sliced with gaps, etc.), and neither `data_ptr` nor `strides` need be aligned to
    /// `dtype.alignment()`.
    ///
    /// A `Plain` storage does not compress the data, and is useful when you want to treat regular
    /// buffers as `Array`, for example to participate in math operations with compressed arrays.
    ///
    /// # Arguments
    ///
    /// * `data_ptr` - pointer to the first element.
    /// * `shape` - number of elements along each dimension.
    /// * `strides` - the element strides in bytes. Must have the same length as `shape`.
    /// * `dtype` - element type descriptor; used for itemsize.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// * `data_ptr` is a valid, non-dangling pointer for the lifetime of the returned `Array`.
    /// * The memory region reachable via `data_ptr` and `strides` is valid to read for all index
    ///   combinations in `0..shape[d]` on each dimension `d`.
    /// * Elements accessed by the data pointer must be valid for the specified `dtype`.
    /// * Every byte from the first to the last element (the strided span, gaps between elements
    ///   included) lies in one allocation, and none of it is written, or borrowed through a
    ///   `&mut`, while the array is read or while a buffer returned by a pull read of it (or of a
    ///   lazy view over it) is alive.
    pub unsafe fn plain_ndarray_ptr(
        data_ptr: *const u8,
        shape: &[u64],
        strides: &[usize],
        dtype: Dtype,
    ) -> Result<Self>
    where
        ET: ElementType,
    {
        let allocation = &();
        let storage = unsafe { PlainBase::new(allocation, data_ptr, shape, strides, dtype) }?;
        Ok(Array::from_storage(PlainView(
            storage.element_type_change::<ET>()?,
        )))
    }
}

impl<A, ET> crate::storage::ArrayStorageImpl for PlainBase<A, ET> {}
impl<A, ET> StorageBase for PlainBase<A, ET>
where
    ET: ElementType,
{
    type ElementType = ET;

    #[inline(always)]
    fn read_data<'a>(
        &'a self,
        index: &Index<'_>,
        _context: &ReadContext,
        out: OutBuf<'_, 'a>,
    ) -> Result<()> {
        let dtype = self.dtype();
        check_out_buf(&out, self.shape(), index, self.dtype().itemsize() as usize)?;

        let mut offset = 0;
        let mut span = dtype.itemsize() as usize;
        let dims = index
            .start()
            .iter()
            .zip(index.size())
            .zip(self.strides.as_ref());
        for ((&start, &size), &stride) in dims {
            debug_assert!(size != 0);
            offset += start as usize * stride;
            span += (size - 1) * stride;
        }
        let src = unsafe { std::slice::from_raw_parts(self.data.as_ptr().add(offset), span) };

        match out.0 {
            OutBufInner::Pull(slot) => {
                _ = write_out_buf_slot(slot, unsafe {
                    StridedBuf::from_slice(src, self.strides.clone())
                })
            }
            OutBufInner::Push(out) => unsafe {
                out.copy_from(src, self.strides.as_ref(), index.size(), dtype)
            },
        }
        Ok(())
    }

    #[inline(always)]
    fn shape(&self) -> &[u64] {
        self.shape.as_slice()
    }
    #[inline(always)]
    fn dtype(&self) -> &Dtype {
        &self.dtype
    }
    #[inline]
    fn spec(&self) -> ArraySpec<'_> {
        self.spec.as_ref()
    }

    fn info(&self) -> ArrayStorageInfo<'_> {
        ArrayStorageInfo::new("Plain")
    }

    type ElementTypeChange<NewET: ElementType> = PlainBase<A, NewET>;
    #[inline]
    fn element_type_change<NewET: ElementType>(self) -> Result<Self::ElementTypeChange<NewET>> {
        check_element_type::<NewET>(&self.dtype)?;
        Ok(PlainBase {
            allocation: self.allocation,
            data: self.data,
            shape: self.shape,
            strides: self.strides,
            dtype: self.dtype,
            element_type: PhantomData,
            spec: self.spec,
        })
    }
}

macro_rules! forward_plain_storage {
    () => {
        #[inline(always)]
        fn read_data<'rd>(
            &'rd self,
            index: &crate::storage::Index<'_>,
            context: &crate::ReadContext,
            out: crate::storage::OutBuf<'_, 'rd>,
        ) -> crate::error::Result<()> {
            self.0.read_data(index, context, out)
        }

        #[inline(always)]
        fn shape(&self) -> &[u64] {
            self.0.shape()
        }
        #[inline(always)]
        fn dtype(&self) -> &crate::dtype::Dtype {
            self.0.dtype()
        }
        #[inline]
        fn spec(&self) -> crate::storage::ArraySpec<'_> {
            self.0.spec()
        }

        fn info(&self) -> crate::storage::ArrayStorageInfo<'_> {
            self.0.info()
        }
    };
}

impl<T> crate::storage::ArrayStorageImpl for Plain<T> {}
impl<T> StorageBase for Plain<T>
where
    T: Dtyped,
{
    type ElementType = T;

    forward_plain_storage!();

    type ElementTypeChange<NewET: ElementType> = PlainBase<Vec<T>, NewET>;
    #[inline]
    fn element_type_change<NewET: ElementType>(self) -> Result<Self::ElementTypeChange<NewET>> {
        self.0.element_type_change()
    }
}

impl<ET> crate::storage::ArrayStorageImpl for PlainView<'_, ET> {}
impl<'a, ET> StorageBase for PlainView<'a, ET>
where
    ET: ElementType,
{
    type ElementType = ET;

    forward_plain_storage!();

    type ElementTypeChange<NewET: ElementType> = PlainView<'a, NewET>;
    #[inline]
    fn element_type_change<NewET: ElementType>(self) -> Result<Self::ElementTypeChange<NewET>> {
        Ok(PlainView(self.0.element_type_change()?))
    }
}

#[inline]
fn shape_from_ndarray(shape: &[usize]) -> Result<DimDyn> {
    check_ndim(shape.len())?;
    Ok(DimDyn::from_fn(shape.len(), |i| shape[i] as u64))
}

fn strides_from_ndarray<T>(shape: &[usize], strides: &[isize]) -> Result<DimArray<usize>> {
    let empty = shape.contains(&0);
    std::iter::zip(shape, strides)
        .map(|(&len, &s)| {
            if len <= 1 || empty {
                return Ok(size_of::<T>()); // no element is addressed through this stride
            }
            let s = usize::try_from(s)
                .map_err(|_| error!(InvalidArgument, "Negative strides are not supported"))?;
            Ok(s * size_of::<T>())
        })
        .collect()
}

impl<A, ET> PlainBase<A, ET> {
    #[inline]
    fn as_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }

    #[inline]
    fn strides(&self) -> &[usize] {
        self.strides.as_ref()
    }

    fn is_standard_layout(&self) -> bool {
        // TODO: keep a pre-computed flag for this
        let shape = self.shape.as_slice();
        if shape.contains(&0) {
            return true;
        }
        let mut expected = self.dtype.itemsize() as u64;
        for (&len, &stride) in std::iter::zip(shape, self.strides.as_ref()).rev() {
            if len > 1 && stride as u64 != expected {
                return false;
            }
            expected *= len;
        }
        true
    }

    fn is_contiguous(&self) -> bool {
        // TODO: keep a pre-computed flag for this
        let shape = self.shape.as_slice();
        if shape.contains(&0) {
            return true;
        }
        // Walk the stepped dims from the innermost (smallest stride) out: each must start right
        // where the dims inside it end.
        let mut dims = (0..shape.len())
            .filter(|&d| shape[d] > 1)
            .collect::<DimArray<usize>>();
        dims.sort_by_key(|&d| self.strides[d]);
        let mut expected = self.dtype.itemsize() as u64;
        for d in dims {
            if self.strides[d] as u64 != expected {
                return false;
            }
            expected *= shape[d];
        }
        true
    }

    fn nitems(&self) -> usize {
        self.shape
            .as_slice()
            .iter()
            .map(|&len| len as usize)
            .product()
    }

    fn as_bytes(&self) -> Option<&[u8]> {
        if !self.is_standard_layout() {
            return None;
        }
        let nbytes = self.nitems() * self.dtype.itemsize() as usize;
        if nbytes == 0 {
            return Some(&[]);
        }
        Some(unsafe { std::slice::from_raw_parts(self.data.as_ptr(), nbytes) })
    }

    fn as_slice(&self) -> Option<&[ET]>
    where
        ET: Dtyped,
    {
        if !self.is_standard_layout() {
            return None;
        }
        let nitems = self.nitems();
        if nitems == 0 {
            return Some(&[]);
        }
        let ptr = self.data.as_ptr().cast::<ET>();
        ptr.is_aligned()
            .then(|| unsafe { std::slice::from_raw_parts(ptr, nitems) })
    }

    fn as_ndarray_view<D>(&self) -> Result<ndarray::ArrayView<'_, ET, D>>
    where
        ET: Dtyped,
        D: ndarray::Dimension,
    {
        use ndarray::ShapeBuilder;

        let shape = self.shape.as_slice();
        let ndim = shape.len();
        if let Some(expected) = D::NDIM {
            ensure!(
                ndim == expected,
                InvalidShapeOperation,
                "Can't create ndarray view of ndim {ndim} as an ndarray of ndim {expected}"
            );
        }
        let mut dim = D::zeros(ndim);
        for d in 0..ndim {
            dim[d] = shape[d] as usize;
        }
        if shape.contains(&0) {
            let ptr = std::ptr::NonNull::<ET>::dangling().as_ptr();
            return Ok(unsafe { ndarray::ArrayView::from_shape_ptr(dim, ptr) });
        }

        let ptr = self.data.as_ptr().cast::<ET>();
        ensure!(
            ptr.is_aligned(),
            InvalidArgument,
            "Can't create ndarray view: data pointer {ptr:?} is not aligned to the element alignment {}",
            align_of::<ET>()
        );
        let mut strides = D::zeros(ndim);
        for d in 0..ndim {
            let stride = self.strides[d];
            strides[d] = if stride.is_multiple_of(size_of::<ET>()) {
                stride / size_of::<ET>()
            } else if shape[d] == 1 {
                0 // never stepped
            } else {
                bail!(
                    InvalidArgument,
                    "Can't create ndarray view: stride {stride} of dim {d} is not a multiple of the itemsize {}",
                    size_of::<ET>()
                );
            };
        }
        Ok(unsafe { ndarray::ArrayView::from_shape_ptr(dim.strides(strides), ptr) })
    }
}

/// Extension methods for arrays with plain (uncompressed) storage.
///
/// Implemented for `Array` of [`Plain`], [`PlainView`] and [`PlainBase`]. These give direct,
/// zero-copy access to the in-memory buffer behind the array, and set its read size.
pub trait PlainArrayMethods {
    /// The compile-time element type: the [`ElementType`](StorageBase::ElementType) of the
    /// storage.
    type ElementType: ElementType;

    /// A raw pointer to the first element, at index `[0, 0, ..., 0]`.
    ///
    /// Element `[i0, i1, ...]` is at byte offset `i0 * strides[0] + i1 * strides[1] + ...` (see
    /// [`strides`](Self::strides)). The pointer may not be aligned to the element type if the
    /// array was made with [`Array::plain_ndarray_ptr`]. Don't write through it.
    ///
    /// # Examples
    ///
    /// ```
    /// use jix::storage::PlainArrayMethods;
    /// use jix::Array;
    /// use ndarray::{array, s};
    ///
    /// let nd = array![[1u8, 2, 3], [4, 5, 6]];
    /// let a = Array::plain_ndarray_ref(&nd)?;
    /// assert_eq!(a.as_ptr(), nd.as_ptr()); // no copy
    ///
    /// let b = Array::plain_ndarray_view(nd.slice(s![1.., 1..]))?;
    /// assert_eq!(unsafe { *b.as_ptr() }, 5);
    /// # Ok::<(), jix::Error>(())
    /// ```
    fn as_ptr(&self) -> *const u8;

    /// The stride of each dimension, **in bytes** (ndarray's strides are in elements).
    ///
    /// Strides are never negative. A stride of 0 is a broadcast dim. The stride of a dim of
    /// length 1 is never used and may be anything.
    ///
    /// # Examples
    ///
    /// ```
    /// use jix::storage::PlainArrayMethods;
    /// use jix::Array;
    /// use ndarray::{s, Array3};
    ///
    /// let nd = Array3::<f64>::zeros((2, 3, 4));
    /// assert_eq!(Array::plain_ndarray_ref(&nd)?.strides(), &[96, 32, 8]);
    ///
    /// let every_other = nd.slice(s![.., .., ..;2]);
    /// assert_eq!(Array::plain_ndarray_view(every_other)?.strides(), &[96, 32, 16]);
    /// # Ok::<(), jix::Error>(())
    /// ```
    fn strides(&self) -> &[usize];

    /// Check if the data is in standard layout: C-contiguous, i.e. row-major with no gaps.
    ///
    /// Dims of length 1 are ignored, and an array with no elements is always in standard layout.
    ///
    /// # Examples
    ///
    /// ```
    /// use jix::storage::PlainArrayMethods;
    /// use jix::Array;
    /// use ndarray::{s, Array2};
    ///
    /// let nd = Array2::<f32>::zeros((3, 4));
    /// assert!(Array::plain_ndarray_ref(&nd)?.is_standard_layout());
    /// assert!(!Array::plain_ndarray_view(nd.t())?.is_standard_layout());
    /// assert!(Array::plain_ndarray_view(nd.slice(s![1..2, ..]))?.is_standard_layout());
    /// assert!(!Array::plain_ndarray_view(nd.slice(s![.., 1..3]))?.is_standard_layout());
    /// # Ok::<(), jix::Error>(())
    /// ```
    fn is_standard_layout(&self) -> bool;

    /// Check if the data is contiguous: the elements fill one block of memory with no gaps, with
    /// the dims in any order (C, Fortran, or any other permutation).
    ///
    /// Dims of length 1 are ignored, and an array with no elements is always contiguous. A
    /// broadcast dim (stride 0) makes the array non-contiguous.
    ///
    /// # Examples
    ///
    /// ```
    /// use jix::storage::PlainArrayMethods;
    /// use jix::Array;
    /// use ndarray::{s, Array2};
    ///
    /// let nd = Array2::<f32>::zeros((3, 4));
    /// assert!(Array::plain_ndarray_ref(&nd)?.is_contiguous());
    /// assert!(Array::plain_ndarray_view(nd.t())?.is_contiguous()); // Fortran layout
    /// assert!(!Array::plain_ndarray_view(nd.slice(s![.., ..;2]))?.is_contiguous());
    /// # Ok::<(), jix::Error>(())
    /// ```
    fn is_contiguous(&self) -> bool;

    /// The raw bytes of the array in row-major order, or `None` if it is not in
    /// [standard layout](Self::is_standard_layout).
    ///
    /// Works for any element type, including [`TypeDyn`], and for unaligned data.
    ///
    /// # Examples
    ///
    /// ```
    /// use jix::storage::PlainArrayMethods;
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::plain_ndarray(array![[1u16, 2], [3, 4]])?;
    /// assert_eq!(a.as_bytes(), Some(&[1u8, 0, 2, 0, 3, 0, 4, 0][..]));
    /// # Ok::<(), jix::Error>(())
    /// ```
    fn as_bytes(&self) -> Option<&[u8]>;

    /// The elements of the array in row-major order, or `None` if it is not in
    /// [standard layout](Self::is_standard_layout) or the data is not aligned to the element
    /// type.
    ///
    /// Data from an ndarray is always aligned; only [`Array::plain_ndarray_ptr`] and
    /// [`PlainBase::new`] can make unaligned arrays. Needs a compile-time element type: for
    /// [`TypeDyn`], call [`Array::into_typed`] first.
    ///
    /// # Examples
    ///
    /// ```
    /// use jix::storage::PlainArrayMethods;
    /// use jix::Array;
    /// use ndarray::array;
    ///
    /// let a = Array::plain_ndarray(array![[1.0f32, 2.0], [3.0, 4.0]])?;
    /// assert_eq!(a.as_slice(), Some(&[1.0f32, 2.0, 3.0, 4.0][..]));
    ///
    /// let t = Array::plain_ndarray(array![[1.0f32, 2.0], [3.0, 4.0]].reversed_axes())?;
    /// assert_eq!(t.as_slice(), None); // not row-major
    /// # Ok::<(), jix::Error>(())
    /// ```
    fn as_slice(&self) -> Option<&[Self::ElementType]>
    where
        Self::ElementType: Dtyped;

    /// Borrow the data as an [`ndarray::ArrayView`], without a copy.
    ///
    /// Works for any layout, including sliced and broadcast ones. `D` (e.g. `Ix2` or `IxDyn`)
    /// is checked against the array's ndim at runtime. Needs a compile-time element type: for
    /// [`TypeDyn`], call [`Array::into_typed`] first.
    ///
    /// # Errors
    ///
    /// - [`InvalidShapeOperation`](crate::error::ErrorKind::InvalidShapeOperation) - `D` has a
    ///   different ndim than the array.
    /// - [`InvalidArgument`](crate::error::ErrorKind::InvalidArgument) - the data is not aligned
    ///   to the element type, or a stride is not a whole number of elements. Only possible for
    ///   arrays made with [`Array::plain_ndarray_ptr`] or [`PlainBase::new`].
    ///
    /// # Examples
    ///
    /// ```
    /// use jix::storage::PlainArrayMethods;
    /// use jix::Array;
    /// use ndarray::{array, s, Ix2};
    ///
    /// let nd = array![[1i32, 2, 3, 4], [5, 6, 7, 8]];
    /// let a = Array::plain_ndarray_view(nd.slice(s![.., ..;2]))?;
    ///
    /// let view = a.as_ndarray_view::<Ix2>()?;
    /// assert_eq!(view, array![[1, 3], [5, 7]]);
    /// assert_eq!(view.as_ptr(), nd.as_ptr()); // same memory
    /// # Ok::<(), jix::Error>(())
    /// ```
    fn as_ndarray_view<D>(&self) -> Result<ndarray::ArrayView<'_, Self::ElementType, D>>
    where
        Self::ElementType: Dtyped,
        D: ndarray::Dimension;

    /// Sets the target byte range `(min, max)` for a single read region of this array and of
    /// the lazy views built on it (a view over several arrays takes it from one of them, usually
    /// the first). [`Array::read_ctx`](crate::Array::read_ctx) starts from it, and
    /// [`Array::compact`](crate::Array::compact) keeps it. See
    /// [`ReadContext::set_read_size`](crate::ReadContext::set_read_size).
    fn set_read_size(&mut self, size_hint: (u64, u64));
}

macro_rules! impl_plain_array_methods {
    ([$($gen:tt)*] $storage:ty, $element_type:ty $(, .$field:tt)?) => {
        impl<$($gen)*> crate::storage::PlainArrayMethods for crate::Array<$storage> {
            type ElementType = $element_type;

            #[inline]
            fn as_ptr(&self) -> *const u8 {
                self.storage $(.$field)? .as_ptr()
            }

            #[inline]
            fn strides(&self) -> &[usize] {
                self.storage $(.$field)? .strides()
            }

            #[inline]
            fn is_standard_layout(&self) -> bool {
                self.storage $(.$field)? .is_standard_layout()
            }

            #[inline]
            fn is_contiguous(&self) -> bool {
                self.storage $(.$field)? .is_contiguous()
            }

            #[inline]
            fn as_bytes(&self) -> Option<&[u8]> {
                self.storage $(.$field)? .as_bytes()
            }

            #[inline]
            fn as_slice(&self) -> Option<&[Self::ElementType]>
            where
                Self::ElementType: crate::dtype::Dtyped,
            {
                self.storage $(.$field)? .as_slice()
            }

            #[inline]
            fn as_ndarray_view<D>(
                &self,
            ) -> crate::error::Result<ndarray::ArrayView<'_, Self::ElementType, D>>
            where
                Self::ElementType: crate::dtype::Dtyped,
                D: ndarray::Dimension,
            {
                self.storage $(.$field)? .as_ndarray_view()
            }

            #[inline]
            fn set_read_size(&mut self, size_hint: (u64, u64)) {
                self.storage $(.$field)? .spec.set_read_size(size_hint);
            }
        }
    };
}

impl_plain_array_methods!([T: Dtyped] Plain<T>, T, .0);
impl_plain_array_methods!(['a, ET: ElementType] PlainView<'a, ET>, ET, .0);
impl_plain_array_methods!([A, ET: ElementType] PlainBase<A, ET>, ET);

#[cfg(test)]
mod tests {
    use ndarray::{array, s, ArrayD, Ix3};

    use crate::scalar::bool8;
    use crate::storage::PlainArrayMethods;
    use crate::{Array, ReadContext};

    // -----------------------------------------------------------------------
    /// Reading an empty region whose start sits at the end of a dim must not compute an
    /// out-of-bounds pointer, even though the resulting slice is empty.
    #[test]
    fn read_empty_subrange_at_end_of_dim() {
        let nd = ndarray::Array3::<i32>::from_shape_vec((2, 3, 4), (0..24).collect()).unwrap();
        let a = Array::plain_ndarray(nd).unwrap();
        let ctx = a.read_ctx();
        // `2..2` is the whole of dim 0, so the start offset alone runs to the end of the
        // allocation; the non-zero start on dim 2 then pushes it past.
        let got = a.to_ndarray_sub::<Ix3>(&[2..2, 0..3, 1..4], &ctx).unwrap();
        assert_eq!(got.shape(), &[0, 3, 3]);
        assert_eq!(got.len(), 0);
    }

    #[test]
    fn negative_strides_are_an_error() {
        use crate::error::ErrorKind;

        let base = array![[1i32, 2, 3], [4, 5, 6]];
        let reversed = base.slice(s![..;-1, ..]);
        let kind = |r: crate::error::Result<()>| r.unwrap_err().kind();
        assert_eq!(
            kind(Array::plain_ndarray_view(reversed).map(|_| ())),
            ErrorKind::InvalidArgument
        );
        assert_eq!(
            kind(Array::plain_ndarray_ref(&reversed).map(|_| ())),
            ErrorKind::InvalidArgument
        );
        assert_eq!(
            kind(Array::compact_ndarray(&reversed).map(|_| ())),
            ErrorKind::InvalidArgument
        );
        let mut owned = array![[1i32, 2, 3]];
        owned.invert_axis(ndarray::Axis(1));
        assert_eq!(
            kind(Array::plain_ndarray(owned).map(|_| ())),
            ErrorKind::InvalidArgument
        );

        // A length-1 dim is never stepped, its negative stride is fine.
        let mut owned = array![[1i32, 2, 3]];
        owned.invert_axis(ndarray::Axis(0));
        let a = Array::plain_ndarray(owned).unwrap();
        assert_eq!(a.to_ndarray().unwrap(), array![[1, 2, 3]]);

        // A zero-element array addresses no element, any stride is fine.
        let empty = base.slice(s![..0, ..;-1]);
        let a = Array::plain_ndarray_view(empty).unwrap();
        assert_eq!(a.shape(), &[0, 3]);
    }

    #[test]
    fn owned_1d_shape() {
        let a = Array::plain_ndarray(array![1i32, 2, 3]).unwrap();
        assert_eq!(a.shape(), &[3u64]);
    }

    #[test]
    fn owned_1d_dtype_i32() {
        use crate::dtype::ScalarKind;
        let a = Array::plain_ndarray(array![0i32]).unwrap();
        assert_eq!(a.dtype().try_to_scalar(), Some(ScalarKind::I32));
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
        let nd = array![[bool8::TRUE, bool8::FALSE], [bool8::FALSE, bool8::TRUE]];
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
        // Take every-other column via an ndarray slice.
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

    // Force the strided read/scatter path (`nd_iter_unordered` in storage/mod.rs): a tiny read window plus
    // an inner dim wider than it makes the read split so each block lands strided in the output.
    // Unique element values mean any mis-scatter (wrong dst position) shows up as a mismatch.
    #[test]
    fn strided_scatter_path_negation() {
        use std::ops::Neg;

        let vals = (0..4 * 10).collect::<Vec<i32>>();
        let nd = ArrayD::from_shape_vec(vec![4, 10], vals).unwrap();
        // for i32: a 4..8 item window, narrower than the 10-wide inner dim
        let mut a = Array::plain_ndarray(nd.clone()).unwrap();
        a.set_read_size((16, 32));

        let got = a.view().neg().to_ndarray().unwrap();
        let expected = nd.mapv(|x: i32| -x);
        assert_eq!(got, expected);
    }

    // Same as `strided_scatter_path_negation`, with the read window set on the context instead of
    // the array.
    #[test]
    fn strided_scatter_path_negation_context_read_size() {
        use std::ops::Neg;

        let vals = (0..4 * 10).collect::<Vec<i32>>();
        let nd = ArrayD::from_shape_vec(vec![4, 10], vals).unwrap();
        let a = Array::plain_ndarray(nd.clone()).unwrap();
        let mut context = a.read_ctx();
        context.set_read_size((16, 32));

        let neg = a.view().neg();
        let got = neg.to_ndarray_sub(&[0..4, 0..10], &context).unwrap();
        let expected = nd.mapv(|x: i32| -x);
        assert_eq!(got, expected);
    }

    // -----------------------------------------------------------------------
    // Under-aligned views
    // -----------------------------------------------------------------------

    /// A `Plain` view may start off its dtype's alignment: a read lends the caller's own bytes,
    /// and every consumer either re-strides the view without reading it or copies it through
    /// `NdCopier`, which handles unaligned source and destination. This walks a deliberately
    /// 1-byte-offset `i32` view through each of those consumers - Miri checks the alignment of
    /// every access.
    #[test]
    fn misaligned_view_reads_through_every_consumer() {
        let (rows, cols) = (4usize, 5);
        let nitems = rows * cols;
        let itemsize = size_of::<i32>();
        let expected = (0..nitems).map(|k| k as i32 * 3 - 7).collect::<Vec<i32>>();

        // An over-aligned backing buffer with element 0 one byte in, so neither the base pointer
        // nor `base + k * itemsize` is 4-byte aligned.
        let mut backing = vec![0u64; (nitems * itemsize + 1).div_ceil(8) + 1];
        let base = unsafe { backing.as_mut_ptr().cast::<u8>().add(1) };
        assert!(!(base as usize).is_multiple_of(itemsize));
        for (k, &v) in expected.iter().enumerate() {
            unsafe { base.byte_add(k * itemsize).cast::<i32>().write_unaligned(v) };
        }

        let strides = [cols * itemsize, itemsize];
        let make = || unsafe {
            Array::plain_ndarray_ptr(
                base.cast_const(),
                &[rows as u64, cols as u64],
                &strides,
                <i32 as crate::dtype::Dtyped>::DTYPE,
            )
            .unwrap()
        };
        fn as_vec(nd: ndarray::Array2<i32>) -> Vec<i32> {
            nd.iter().copied().collect()
        }
        /// The `rows x cols` sub-block of a row-major `all`, in row-major order.
        fn sub(
            all: &[i32],
            cols: usize,
            rows: std::ops::Range<usize>,
            cs: std::ops::Range<usize>,
        ) -> Vec<i32> {
            rows.flat_map(|r| cs.clone().map(move |c| all[r * cols + c]))
                .collect()
        }

        // Straight materialization, and a sub-range of it.
        let arr = make();
        let ctx = arr.read_ctx();
        assert_eq!(as_vec(arr.to_ndarray().unwrap()), expected);
        assert_eq!(
            as_vec(arr.to_ndarray_sub(&[1..3, 1..4], &ctx).unwrap()),
            sub(&expected, cols, 1..3, 1..4)
        );

        // Shape ops: a re-strided lend (slice), a broadcast, and a copying one (flip).
        assert_eq!(
            as_vec(make().slice((.., 1..3)).to_ndarray().unwrap()),
            sub(&expected, cols, 0..rows, 1..3)
        );
        assert_eq!(
            as_vec(make().flip(0).to_ndarray().unwrap()),
            (0..rows)
                .rev()
                .flat_map(|r| sub(&expected, cols, r..r + 1, 0..cols))
                .collect::<Vec<i32>>()
        );

        // An elementwise op chain, which walks the view through the pipeline.
        use core::ops::{Add, Neg};
        assert_eq!(
            as_vec(make().neg().add(make()).to_ndarray().unwrap()),
            vec![0i32; nitems]
        );

        // A reduction, which pushes tiles of the view into its own scratch.
        let row_sums = (0..rows)
            .map(|r| {
                (0..cols)
                    .map(|c| expected[r * cols + c] as i64)
                    .sum::<i64>()
            })
            .collect::<Vec<i64>>();
        assert_eq!(
            make().sum(1).to_ndarray().unwrap(),
            ndarray::Array1::from(row_sums)
        );

        // And compressing it, which reads the view chunk by chunk into aligned block buffers.
        assert_eq!(
            as_vec(make().compact().unwrap().to_ndarray().unwrap()),
            expected
        );
    }

    #[test]
    fn plain_ndarray_view_reads_back_the_borrowed_data() {
        let owned = ndarray::array![[1i32, 2, 3], [4, 5, 6]];
        let array = Array::plain_ndarray_view(owned.view()).unwrap();
        assert_eq!(array.to_ndarray().unwrap(), owned);

        // A transposed view is non-contiguous, so the strides come from the view itself.
        let transposed = Array::plain_ndarray_view(owned.t()).unwrap();
        assert_eq!(
            transposed.to_ndarray().unwrap(),
            owned.t().as_standard_layout().into_owned()
        );
    }

    // -----------------------------------------------------------------------
    // PlainArrayMethods
    // -----------------------------------------------------------------------

    #[test]
    fn methods_standard_layout() {
        let nd = array![[1i32, 2, 3], [4, 5, 6]];
        let a = Array::plain_ndarray_ref(&nd).unwrap();
        assert_eq!(a.as_ptr(), nd.as_ptr().cast::<u8>());
        assert_eq!(a.strides(), &[12, 4]);
        assert!(a.is_standard_layout());
        assert!(a.is_contiguous());
        assert_eq!(a.as_slice(), Some(&[1, 2, 3, 4, 5, 6][..]));
        let bytes = nd.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        assert_eq!(a.as_bytes(), Some(&bytes[..]));
        assert_eq!(a.as_ndarray_view().unwrap(), nd);
    }

    #[test]
    fn methods_fortran_layout() {
        let nd = array![[1i32, 2, 3], [4, 5, 6]];
        let a = Array::plain_ndarray(nd.clone().reversed_axes()).unwrap();
        assert_eq!(a.strides(), &[4, 12]);
        assert!(!a.is_standard_layout());
        assert!(a.is_contiguous());
        assert_eq!(a.as_slice(), None);
        assert_eq!(a.as_bytes(), None);
        assert_eq!(a.as_ndarray_view().unwrap(), nd.t());
    }

    #[test]
    fn methods_permuted_3d_is_contiguous() {
        let nd = ArrayD::from_shape_vec(vec![2, 3, 4], (0..24).collect::<Vec<i64>>()).unwrap();
        let permuted = nd.view().permuted_axes(vec![1, 2, 0]);
        let a = Array::plain_ndarray_view(permuted.clone()).unwrap();
        assert!(!a.is_standard_layout());
        assert!(a.is_contiguous());
        assert_eq!(a.as_ndarray_view::<ndarray::IxDyn>().unwrap(), permuted);
    }

    #[test]
    fn methods_with_gaps() {
        let nd = array![[1i32, 2, 3, 4], [5, 6, 7, 8]];
        for sliced in [nd.slice(s![.., ..;2]), nd.slice(s![.., 1..3])] {
            let a = Array::plain_ndarray_view(sliced).unwrap();
            assert_eq!(a.as_ptr(), sliced.as_ptr().cast::<u8>());
            assert!(!a.is_standard_layout());
            assert!(!a.is_contiguous());
            assert_eq!(a.as_slice(), None);
            assert_eq!(a.as_bytes(), None);
            let view = a.as_ndarray_view::<ndarray::Ix2>().unwrap();
            assert_eq!(view, sliced);
            assert_eq!(view.as_ptr(), sliced.as_ptr());
        }
        // A row slice has no gaps.
        let row = nd.slice(s![1..2, ..]);
        let a = Array::plain_ndarray_view(row).unwrap();
        assert!(a.is_standard_layout());
        assert_eq!(a.as_slice(), Some(&[5, 6, 7, 8][..]));
    }

    #[test]
    fn methods_ignore_length_one_dims() {
        // A length-1 dim with a stride that is neither packed nor a whole number of elements.
        let data = [1u16, 2, 3];
        let a = unsafe {
            Array::<crate::storage::PlainView<'_, u16>>::plain_ndarray_ptr(
                data.as_ptr().cast(),
                &[1, 3],
                &[1001, 2],
                <u16 as crate::dtype::Dtyped>::DTYPE,
            )
        }
        .unwrap();
        assert!(a.is_standard_layout());
        assert!(a.is_contiguous());
        assert_eq!(a.as_slice(), Some(&data[..]));
        assert_eq!(a.as_ndarray_view().unwrap(), array![[1u16, 2, 3]]);
    }

    #[test]
    fn methods_broadcast() {
        let data = [7i32, 8];
        let a = unsafe {
            Array::<crate::storage::PlainView<'_, i32>>::plain_ndarray_ptr(
                data.as_ptr().cast(),
                &[3, 2],
                &[0, 4],
                <i32 as crate::dtype::Dtyped>::DTYPE,
            )
        }
        .unwrap();
        assert!(!a.is_standard_layout());
        assert!(!a.is_contiguous());
        assert_eq!(a.as_slice(), None);
        let view = a.as_ndarray_view::<ndarray::Ix2>().unwrap();
        assert_eq!(view, array![[7, 8], [7, 8], [7, 8]]);
        assert_eq!(view.strides(), &[0, 1]);
    }

    #[test]
    fn methods_empty() {
        let nd = ndarray::Array2::<f32>::zeros((0, 3));
        let a = Array::plain_ndarray(nd).unwrap();
        assert!(a.is_standard_layout());
        assert!(a.is_contiguous());
        assert_eq!(a.as_slice(), Some(&[][..]));
        assert_eq!(a.as_bytes(), Some(&[][..]));
        let view = a.as_ndarray_view::<ndarray::Ix2>().unwrap();
        assert_eq!(view.shape(), &[0, 3]);

        // An empty array may come with a null pointer.
        let a = unsafe {
            Array::<crate::storage::PlainView<'_, f32>>::plain_ndarray_ptr(
                std::ptr::null(),
                &[2, 0],
                &[3, 1],
                <f32 as crate::dtype::Dtyped>::DTYPE,
            )
        }
        .unwrap();
        assert_eq!(a.as_slice(), Some(&[][..]));
        assert_eq!(
            a.as_ndarray_view::<ndarray::Ix2>().unwrap().shape(),
            &[2, 0]
        );
    }

    #[test]
    fn methods_misaligned() {
        let backing = [0u64; 4];
        let base = unsafe { backing.as_ptr().cast::<u8>().add(1) };
        let a = unsafe {
            Array::<crate::storage::PlainView<'_, i32>>::plain_ndarray_ptr(
                base,
                &[2, 3],
                &[12, 4],
                <i32 as crate::dtype::Dtyped>::DTYPE,
            )
        }
        .unwrap();
        assert_eq!(a.as_ptr(), base);
        assert!(a.is_standard_layout());
        assert_eq!(a.as_bytes().map(<[u8]>::len), Some(24));
        assert_eq!(a.as_slice(), None);
        let err = a.as_ndarray_view::<ndarray::Ix2>().unwrap_err();
        assert_eq!(err.kind(), crate::error::ErrorKind::InvalidArgument);
    }

    #[test]
    fn methods_ndarray_view_errors() {
        use crate::error::ErrorKind;

        let a = Array::plain_ndarray(array![[1i32, 2], [3, 4]]).unwrap();
        let err = a.as_ndarray_view::<ndarray::Ix3>().unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidShapeOperation);

        // A stride that is not a whole number of elements on a stepped dim.
        let backing = [0u32; 8];
        let a = unsafe {
            Array::<crate::storage::PlainView<'_, u16>>::plain_ndarray_ptr(
                backing.as_ptr().cast(),
                &[2, 2],
                &[5, 2],
                <u16 as crate::dtype::Dtyped>::DTYPE,
            )
        }
        .unwrap();
        let err = a.as_ndarray_view::<ndarray::Ix2>().unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidArgument);
    }

    #[test]
    fn too_many_elements_is_an_error() {
        use crate::error::ErrorKind;

        // A broadcast array addresses one element, but ndarray can't view more than isize::MAX.
        let data = [0u8];
        let make = |len: u64| unsafe {
            Array::<crate::storage::PlainView<'_, u8>>::plain_ndarray_ptr(
                data.as_ptr(),
                &[len, 2],
                &[0, 0],
                <u8 as crate::dtype::Dtyped>::DTYPE,
            )
        };
        assert!(make(isize::MAX as u64 / 2).is_ok());
        let err = make(isize::MAX as u64 / 2 + 1).map(|_| ()).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidShapeOperation);
    }

    #[test]
    fn methods_plain_base() {
        use crate::dtype::Dtyped;
        use crate::storage::PlainBase;
        use crate::TypeDyn;

        let data = vec![1i32, 2, 3, 4, 5, 6];
        let ptr = data.as_ptr().cast::<u8>();
        let storage = unsafe {
            PlainBase::<Vec<i32>, TypeDyn>::new(data, ptr, &[3, 2], &[4, 12], i32::DTYPE)
        }
        .unwrap();
        let a = Array::from_storage(storage);
        assert_eq!(a.as_ptr(), ptr);
        assert_eq!(a.strides(), &[4, 12]);
        assert!(!a.is_standard_layout());
        assert!(a.is_contiguous());
        assert_eq!(a.as_bytes(), None);

        let a = a.into_typed::<i32>().unwrap();
        assert_eq!(a.as_slice(), None);
        assert_eq!(a.as_ndarray_view().unwrap(), array![[1, 4], [2, 5], [3, 6]]);
    }
}
