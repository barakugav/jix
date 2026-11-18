use crate::dtype::{Dtype, Dtyped};
use crate::error::{check_dtype, Result};
use crate::ops::{ElementwisePipeline, OperandTyped};
use crate::storage::{ArraySpec, ArrayStorageInfo, CompactBorrowed, Index, OutBuf, StridedBuf};
use crate::ReadContext;
use crate::{ElementType, TypeDyn};

mod sealed {
    /// Seals [`StorageBase`](super::StorageBase).
    pub trait ArrayStorageImpl {}
}
pub(crate) use sealed::ArrayStorageImpl;

/// The backing data source of an [`Array<S>`](crate::Array).
///
/// `Array<S>` is generic over its storage `S: StorageBase`, which provides three things: the
/// array's shape, its element dtype, and reading any rectangular sub-region into a byte buffer.
/// Everything else - slicing, reshaping, arithmetic, reductions - is built on these three methods.
///
/// # Primary storage backends
///
/// The block-compressed backends [`Compact`](crate::storage::Compact) (heap-allocated) and
/// [`CompactMmap`](crate::storage::CompactMmap) (memory-mapped file) store the array as
/// independently compressed nd-blocks, the primary on-disk format.
///
/// # Adapter
///
/// [`Plain`](crate::storage::Plain) wraps a contiguous or strided in-memory buffer, so regular
/// Rust/ndarray data can be combined with compact arrays.
///
/// # Lazy operation views
///
/// Every operation on an `Array<S>` returns a new `Array` whose storage wraps the
/// original, applying the transformation lazily on each `read_data` call:
///
/// ```text
/// arr.neg()                  -> Array<Neg<S>>
/// arr.reshape(new_shape)     -> Array<Reshape<S>>
/// arr.permute_dims(dims)     -> Array<PermuteDims<S>>
/// arr1.add(arr2)             -> Array<Add<S1, S2>>
/// arr.sum(dim)               -> Array<Sum<S>>
/// arr.cast::<f32>()          -> Array<Cast<S, f32>>
/// ```
///
/// No data is copied or computed until `read_data` is called. At that point the index
/// transformation propagates inward through the storage chain and only the minimum
/// required data is read from the innermost backend.
///
/// # Static dispatch
///
/// Because `Array<S>` is monomorphized over `S` at compile time, chains of operations
/// carry zero heap allocation or virtual dispatch overhead. The full static type of an
/// expression - e.g. `Array<Add<Neg<S1>, Reshape<S2>>>` - is resolved by the compiler.
pub trait StorageBase: sealed::ArrayStorageImpl {
    /// The compile-time element type of arrays backed by this storage.
    ///
    /// Either the element type itself, e.g. `f32` (known at compile time), or
    /// [`TypeDyn`](crate::TypeDyn) (only available at runtime via [`dtype()`](StorageBase::dtype)).
    ///
    /// Operations that require knowing the element type (arithmetic, comparisons, reductions,
    /// cast) are bounded on [`Storage`](crate::storage::Storage), a shorthand
    /// for `StorageBase<ElementType = T>`. Arrays loaded from disk carry `TypeDyn`; call
    /// [`Array::into_typed::<T>()`](crate::Array::into_typed) to assert the expected element
    /// type and re-enable those operations.
    type ElementType: ElementType
    where
        Self: Sized;

    /// Read a sub-region of the array into `out`, as a [`StridedBuf`](crate::storage::StridedBuf).
    ///
    /// The single I/O method of a storage: all higher-level reads (`to_ndarray`,
    /// `to_ndarray_sub`, etc.) bottom out here.
    ///
    /// The two modes mirror NumPy's `out=`, and are picked by `out` (see [`OutBuf`]):
    /// - **pull**: the impl puts the cheapest buffer it can in the slot of `out` - a borrowed view
    ///   into its own memory (possibly with broadcast/stride-0 dims), or a freshly materialized
    ///   one - and picks the strides. The bytes are only contiguous if the strides say so.
    /// - **push**: the region must be written into the buffer of `out` at its own strides. It must
    ///   be writable and have one stride per array dimension.
    ///
    /// # Arguments
    ///
    /// - `index` - a start and a size per dimension, one per dimension of `self.shape()`.
    ///   The region must be within the array shape bounds; an [`Index`] is never empty.
    /// - `context` - read context carrying reusable buffers and decoder state.
    /// - `out` - the destination, as described above.
    ///
    /// # Alignment
    ///
    /// No alignment is required of a push destination, and none is promised of a pulled buffer:
    /// it is only as aligned as its backing bytes happen to be. Reinterpret the bytes as typed
    /// values with unaligned reads, or stage the region through a buffer of your own.
    fn read_data<'a>(
        &'a self,
        index: &Index<'_>,
        context: &'a ReadContext,
        out: OutBuf<'_, 'a>,
    ) -> Result<()>;

    /// Read a sub-region as an element-wise pipeline over `T`.
    ///
    /// Where [`read_data`](Self::read_data) materializes the region into a single buffer, this
    /// returns a tree of operations with operands at the leaves and ops at the inner nodes. The
    /// default implementation reads the region into a leaf buffer; ops override it to build the
    /// tree recursively from their inner storages.
    ///
    /// # Arguments
    ///
    /// Same as [`read_data`](Self::read_data): the region to read, and the read
    /// context.
    #[doc(hidden)]
    #[inline]
    fn read_as_elementwise_pipeline<'a, T>(
        &'a self,
        index: &Index<'_>,
        context: &'a ReadContext,
    ) -> Result<impl ElementwisePipeline<T> + use<'a, T, Self>>
    where
        T: Dtyped,
        Self: Sized,
    {
        check_dtype(Dtype::new_ref::<T>(), self.dtype())?;
        let data = self.read_data_pull(index, context)?;
        Ok(unsafe { OperandTyped::new_input(data, Dtype::new_ref::<T>()) })
    }

    /// Returns the shape of the array, one element per dimension.
    fn shape(&self) -> &[u64];

    /// Returns the element dtype of the array.
    fn dtype(&self) -> &Dtype;

    /// Returns metadata about this storage backend.
    #[doc(hidden)]
    fn spec(&self) -> ArraySpec<'_>;

    /// Returns info about this storage and its dependencies, for debugging and introspection.
    #[doc(hidden)]
    fn info(&self) -> ArrayStorageInfo<'_>;

    /// If this storage is a compact block-compressed backend, return a borrowed view of itself.
    #[doc(hidden)]
    #[inline(always)]
    fn as_compact(&self) -> Option<CompactBorrowed<'_, TypeDyn>> {
        None
    }

    /// The concrete storage type after swapping the element type to `NewET`.
    type ElementTypeChange<NewET: ElementType>: StorageBase<ElementType = NewET>
    where
        Self: Sized;

    /// Consume `self`, validate the new element type against the runtime dtype, and return the
    /// re-tagged storage.
    ///
    /// Returns [`ErrorKind::UnsupportedDtype`](crate::error::ErrorKind::UnsupportedDtype) if
    /// `NewET = T` and the runtime dtype does not equal `T::DTYPE`. Always succeeds for
    /// `NewET = TypeDyn`.
    #[doc(hidden)]
    fn element_type_change<NewET: ElementType>(self) -> Result<Self::ElementTypeChange<NewET>>
    where
        Self: Sized;
}

pub(crate) trait ArrayStorageExt: StorageBase {
    #[inline(always)]
    fn read_data_pull<'a>(
        &'a self,
        index: &Index<'_>,
        context: &'a ReadContext,
    ) -> Result<StridedBuf<'a>> {
        let mut slot = None;
        self.read_data_pull_in(index, context, &mut slot)?;
        Ok(slot.unwrap())
    }

    #[inline(always)]
    fn read_data_pull_in<'s, 'a>(
        &'a self,
        index: &Index<'_>,
        context: &'a ReadContext,
        slot: &'s mut Option<StridedBuf<'a>>,
    ) -> Result<&'s mut StridedBuf<'a>> {
        self.read_data(index, context, OutBuf::pull(slot))?;
        Ok(slot
            .as_mut()
            .expect("a pull read puts its buffer in the slot"))
    }

    #[inline(always)]
    fn read_data_push<'a>(
        &'a self,
        index: &Index<'_>,
        context: &'a ReadContext,
        out: &mut StridedBuf<'a>,
    ) -> Result<()> {
        self.read_data(index, context, OutBuf::push(out))
    }
}
impl<S: StorageBase + ?Sized> ArrayStorageExt for S {}
