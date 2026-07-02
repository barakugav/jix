use std::ops::Range;

use crate::codec::{ReadContext, TmpBuf};
use crate::dtype::{Dtype, Dtyped};
use crate::error::{check_dtype, Result};
use crate::storage::{ArraySpec, CompactBorrowed, ReadData};
use crate::util::assert_unchecked_eq;
use crate::{Dimension, ElementType};

/// The backing data source of an [`Array<S>`](crate::Array).
///
/// `Array<S>` is generic over its storage `S: ArrayStorage`, which provides three pieces
/// of information: the array's shape, its element type, and the ability to read any
/// rectangular sub-region into a byte buffer. Everything else - slicing, reshaping,
/// arithmetic, reductions - is built on top of these three methods.
///
/// # Primary storage backends
///
/// The main concrete storages are the block-compressed backends:
/// `compressed::Compact` (heap-allocated) and `compressed::CompactMmap` (memory-mapped file)
///  These store the array as independently
/// compressed nd-blocks and are the primary on-disk format.
///
/// # Adapter
///
/// An adapter lets plain data participate in the same `Array<S>` world:
/// - `plain::Plain` - wraps a contiguous or strided in-memory ndarray, used when
///   operating on regular Rust/ndarray data alongside compressed arrays.
///
/// # Lazy operation views
///
/// Every operation on an `Array<S>` returns a new `Array` whose storage wraps the
/// original, applying the transformation lazily on each `read_data` call:
///
/// ```text
/// arr.neg()                  -> Array<Neg<S>>
/// arr.reshape(new_shape)     -> Array<Reshape<S>>
/// arr.permute_axes(axes)     -> Array<PermuteAxes<S>>
/// arr1.add(arr2)             -> Array<Add<S1, S2>>
/// arr.sum(axis)              -> Array<Sum<S>>
/// arr.cast::<f32>()          -> Array<Cast<S>>
/// ```
///
/// No data is copied or computed until `read_data` is called. At that point the index
/// transformation propagates inward through the storage chain and only the minimum
/// required data is read from the innermost backend.
///
/// # Flexibility
///
/// Because `Array<S>` is monomorphized over `S` at compile time, chains of operations
/// carry zero heap allocation or virtual dispatch overhead. The full static type of an
/// expression - e.g. `Array<Add<Neg<S1>, Reshape<S2>>>` - is resolved by the compiler,
/// and only the final `read_data` call touches actual bytes.
pub trait ArrayStorage {
    /// The compile-time element type of arrays backed by this storage.
    ///
    /// Either [`Ty<T>`](crate::Ty) - element type `T` is known at compile time - or [`TypeDyn`](crate::TypeDyn) - element
    /// type is only available at runtime via [`dtype()`](ArrayStorage::dtype).
    ///
    /// Operations that require knowing the element type (arithmetic, comparisons, reductions,
    /// cast) are bounded on [`ArrayStorageTyped`](crate::storage::ArrayStorageTyped), a shorthand for
    /// `ArrayStorage<ElementType = Ty<T>>`. Arrays loaded from disk carry `TypeDyn`; call
    /// [`Array::into_typed::<T>()`](crate::Array::into_typed) to assert the expected element
    /// type and re-enable those operations.
    type ElementType: ElementType;

    /// The compile-time dimension of arrays backed by this storage.
    ///
    /// This associated type lets the compiler track how many axes an array has through a chain
    /// of lazy operations. When the dimension is known statically (e.g. arrays created from a
    /// statically-dimensioned ndarray, or after calling
    /// [`Array::into_dim::<Dim<N>>`](crate::Array::into_dim)), it is [`Dim<N>`](crate::Dim);
    /// when it is only known at runtime (e.g. for arrays loaded from a file or created with
    /// slice-based shape arguments) it is [`DimDyn`](crate::DimDyn).
    ///
    /// Operations that change the number of axes determine the output dimension by either using
    /// the input dimension's associated type (e.g. `S::Dimension::Smaller` or `S::Dimension::Larger`)
    /// or by accepting an explicit dimension argument from the caller
    /// (e.g. `reshape()` accept IntoDimension, `max()` accept `AxesArg`).
    type Dimension: Dimension;

    /// Read a sub-region of the array into a caller-supplied byte buffer.
    ///
    /// This is the single I/O method that every storage backend must implement.
    /// All higher-level read operations (`to_ndarray`, `to_ndarray_sub`, etc.) bottom
    /// out here.
    ///
    /// # Arguments
    ///
    /// - `index` - one half-open range per dimension (`start..end`).
    ///   The number of ranges must equal `self.shape().len()`.
    ///   Ranges must be within the array shape bounds; empty ranges are allowed.
    /// - `buf` - destination buffer. Either a caller-provided buffer or a lazily-allocated one.
    ///   If the buffer is provided by the caller, it must be aligned to `dtype.alignment()` and
    ///   has a length of exactly `index.iter().map(|r| r.len()).product() * dtype.itemsize()` bytes.
    ///   See [`OutBuf`](crate::storage::OutBuf) for details.
    ///   On success the elements are written in row-major (C-contiguous) order.
    /// - `context` - read context carrying the decoder state.
    fn read_data(
        &self,
        index: &[Range<u64>],
        buf: &mut OutBuf<'_>,
        context: &ReadContext,
    ) -> Result<()>;

    /// Read a sub-region of the array as a typed `ReadData<T>`.
    ///
    /// # Arguments
    ///
    /// - `index` - one half-open range per dimension (`start..end`).
    ///   The number of ranges must equal `self.shape().len()`.
    ///   Ranges must be within the array shape bounds; empty ranges are allowed.
    /// - `context` - read context carrying the decoder state.
    ///
    /// # Returns
    ///
    /// A `ReadData<T>` that can be used to read the requested region as typed elements.
    #[inline(always)]
    fn read_data_typed<'a, T>(
        &'a self,
        index: &[Range<u64>],
        context: &'a ReadContext,
    ) -> Result<impl ReadData<T> + use<'a, T, Self>>
    where
        T: Dtyped,
        Self: Sized,
    {
        check_dtype(&T::DTYPE, self.dtype())?;

        let nitems = index.iter().map(|r| r.end - r.start).product::<u64>() as usize;
        let mut out = OutBuf::new_lazy(context);
        self.read_data(index, &mut out, context)?;
        let buf = out.unwrap_tmp();

        struct DefaultReadData<'a, T> {
            buf: TmpBuf<'a>,
            len_: usize,
            _phantom: std::marker::PhantomData<T>,
        }
        impl<T> ReadData<T> for DefaultReadData<'_, T>
        where
            T: Dtyped,
        {
            fn len(&self) -> usize {
                let len = self.len_;
                unsafe { assert_unchecked_eq!(self.buf.as_slice().len(), len * size_of::<T>()) };
                len
            }

            fn read_bulk<const N: usize>(&mut self, offset: usize) -> [T; N] {
                let len = self.len();
                assert!(offset + N <= len);
                let ptr = self.buf.as_slice().as_ptr().cast::<T>();
                unsafe { ptr.add(offset).cast::<[T; N]>().read() }
            }
        }
        Ok(DefaultReadData {
            len_: nitems,
            buf,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Returns the shape of the array, one element per dimension.
    fn shape(&self) -> &[u64];

    /// Returns the element type of the array.
    fn dtype(&self) -> &Dtype;

    /// Returns metadata about this storage backend.
    #[doc(hidden)]
    fn spec(&self) -> ArraySpec<'_>;

    /// If this storage is a compact block-compressed backend, return a borrowed view of itself.
    #[doc(hidden)]
    #[inline(always)]
    fn as_compact(&self) -> Option<CompactBorrowed<'_, Self::ElementType, Self::Dimension>> {
        None
    }

    /// The concrete storage type after swapping the dimension to `NewD`.
    type DimensionChange<NewD: Dimension>: ArrayStorage<
        ElementType = Self::ElementType,
        Dimension = NewD,
    >
    where
        Self: Sized;

    /// Consume `self`, validate the new dimension against the runtime ndim, and return the
    /// re-tagged storage.
    ///
    /// Returns an error if `NewD = Dim<N>` and the runtime ndim does not equal `N`; the
    /// concrete [`ErrorKind`](crate::ErrorKind) is backend-dependent
    /// ([`TooManyDimensions`](crate::ErrorKind::TooManyDimensions) for the concrete storage
    /// backends). Always succeeds for `NewD = DimDyn`.
    fn dimension_change<NewD: Dimension>(self) -> Result<Self::DimensionChange<NewD>>
    where
        Self: Sized;

    /// The concrete storage type after swapping the element type to `NewET`.
    type ElementTypeChange<NewET: ElementType>: ArrayStorage<
        ElementType = NewET,
        Dimension = Self::Dimension,
    >
    where
        Self: Sized;

    /// Consume `self`, validate the new element type against the runtime dtype, and return the
    /// re-tagged storage.
    ///
    /// Returns [`ErrorKind::UnsupportedDtype`](crate::ErrorKind::UnsupportedDtype) if
    /// `NewET = Ty<T>` and the runtime dtype does not equal `T::DTYPE`. Always succeeds for
    /// `NewET = TypeDyn`.
    fn element_type_change<NewET: ElementType>(self) -> Result<Self::ElementTypeChange<NewET>>
    where
        Self: Sized;
}

/// Destination for a [`read_data`](ArrayStorage::read_data) call.
///
/// Either a caller-provided byte buffer ([`OutBuf::new`]) or a lazily-allocated pooled buffer
/// ([`OutBuf::new_lazy`]).
/// After a successful `read_data` call, the buffer contents can be accessed via
/// [`as_slice`](OutBuf::as_slice).
pub struct OutBuf<'a>(OutBufInner<'a>);
enum OutBufInner<'a> {
    Lazy(&'a ReadContext),
    Borrowed(&'a mut [u8]),
    Tmp(TmpBuf<'a>),
}
impl<'a> OutBuf<'a> {
    /// Write into a caller-provided buffer.
    #[inline(always)]
    pub fn new(buf: &'a mut [u8]) -> Self {
        Self(OutBufInner::Borrowed(buf))
    }

    /// Defer allocation: the storage allocates a pooled buffer from `context` on the first demand
    /// for a mutable slice.
    #[inline(always)]
    pub fn new_lazy(context: &'a ReadContext) -> Self {
        Self(OutBufInner::Lazy(context))
    }

    #[inline]
    pub(crate) fn get_mut(&mut self, index: &[Range<u64>], dtype: &Dtype) -> &mut [u8] {
        if let OutBufInner::Lazy(context) = &self.0 {
            let nitems = index.iter().map(|r| r.end - r.start).product::<u64>() as usize;
            let tmp = context.tmp_buf(nitems * dtype.itemsize() as usize, dtype.alignment());
            self.0 = OutBufInner::Tmp(tmp);
        }
        self.as_mut_slice().unwrap()
    }

    /// Returns the buffer contents, or `None` if this is a not-yet-materialized lazy `OutBuf`.
    #[inline(always)]
    pub fn as_slice(&self) -> Option<&[u8]> {
        match &self.0 {
            OutBufInner::Lazy(_) => None,
            OutBufInner::Borrowed(buf) => Some(buf),
            OutBufInner::Tmp(tmp) => Some(tmp.as_slice()),
        }
    }

    /// Returns the mutable buffer contents, or `None` if this is a not-yet-materialized lazy `OutBuf`.
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> Option<&mut [u8]> {
        match &mut self.0 {
            OutBufInner::Lazy(_) => None,
            OutBufInner::Borrowed(buf) => Some(buf),
            OutBufInner::Tmp(tmp) => Some(tmp.as_mut_slice()),
        }
    }

    #[track_caller]
    #[inline(always)]
    pub(crate) fn unwrap_tmp(self) -> TmpBuf<'a> {
        match self.0 {
            OutBufInner::Tmp(tmp) => tmp,
            _ => unreachable!(),
        }
    }
}
