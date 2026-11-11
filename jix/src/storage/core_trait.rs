use std::ops::Range;

use crate::dtype::{Dtype, Dtyped};
use crate::error::{check_dtype, Result};
use crate::ops::{ElementwisePipeline, OperandTyped};
use crate::storage::{ArraySpec, ArrayStorageInfo, CompactBorrowed, StridedBuf};
use crate::ReadContext;
use crate::{ElementType, TypeDyn};

mod sealed {
    /// Seals [`ArrayStorage`](super::ArrayStorage).
    pub trait ArrayStorageImpl {}
}
pub(crate) use sealed::ArrayStorageImpl;

/// The backing data source of an [`Array<S>`](crate::Array).
///
/// `Array<S>` is generic over its storage `S: ArrayStorage`, which provides three things: the
/// array's shape, its element type, and reading any rectangular sub-region into a byte buffer.
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
pub trait ArrayStorage: sealed::ArrayStorageImpl {
    /// The compile-time element type of arrays backed by this storage.
    ///
    /// Either [`Ty<T>`](crate::Ty) (`T` known at compile time) or [`TypeDyn`](crate::TypeDyn)
    /// (only available at runtime via [`dtype()`](ArrayStorage::dtype)).
    ///
    /// Operations that require knowing the element type (arithmetic, comparisons, reductions,
    /// cast) are bounded on [`ArrayStorageTyped`](crate::storage::ArrayStorageTyped), a shorthand
    /// for `ArrayStorage<ElementType = Ty<T>>`. Arrays loaded from disk carry `TypeDyn`; call
    /// [`Array::into_typed::<T>()`](crate::Array::into_typed) to assert the expected element
    /// type and re-enable those operations.
    type ElementType: ElementType
    where
        Self: Sized;

    /// Read a sub-region of the array and return it as a
    /// [`StridedBuf`](crate::storage::StridedBuf).
    ///
    /// The single I/O method of a storage: all higher-level reads (`to_ndarray`,
    /// `to_ndarray_sub`, etc.) bottom out here.
    ///
    /// The two modes mirror NumPy's `out=`:
    /// - **pull** (`out` is `None`): the impl returns the cheapest buffer it can - a borrowed view
    ///   into its own memory (possibly with broadcast/stride-0 dims), or a freshly materialized
    ///   one - and picks the strides. The bytes are only contiguous if the strides say so.
    /// - **push** (`out` is `Some(dst)`): the region must be written into `dst` at `dst`'s own
    ///   strides, and the returned `StridedBuf` is a view of `dst`. `dst` must be writable and have
    ///   one stride per array dimension.
    ///
    /// # Arguments
    ///
    /// - `index` - one half-open range per dimension (`start..end`).
    ///   The number of ranges must equal `self.shape().len()`.
    ///   Ranges must be within the array shape bounds; empty ranges are allowed.
    /// - `context` - read context carrying reusable buffers and decoder state.
    /// - `out` - optional destination, as described above.
    ///
    /// # Alignment
    ///
    /// No alignment is required of `out`, and none is promised of the returned buffer: it is only
    /// as aligned as its backing bytes happen to be. Reinterpret the bytes as typed values with
    /// unaligned reads, or stage the region through a buffer of your own.
    fn read_data<'a>(
        &'a self,
        index: &[Range<u64>],
        context: &'a ReadContext,
        out: Option<&'a mut StridedBuf<'_>>,
    ) -> Result<StridedBuf<'a>>;

    /// Read a sub-region as an element-wise pipeline over `T`.
    ///
    /// Where [`read_data`](Self::read_data) materializes the region into a single buffer, this
    /// returns a tree of operations with operands at the leaves and ops at the inner nodes. The
    /// default implementation reads the region into a leaf buffer; ops override it to build the
    /// tree recursively from their inner storages.
    ///
    /// # Arguments
    ///
    /// Same as [`read_data`](Self::read_data): one half-open range per dimension, and the read
    /// context.
    #[doc(hidden)]
    #[inline]
    fn read_as_elementwise_pipeline<'a, T>(
        &'a self,
        index: &[Range<u64>],
        context: &'a ReadContext,
    ) -> Result<impl ElementwisePipeline<T> + use<'a, T, Self>>
    where
        T: Dtyped,
        Self: Sized,
    {
        check_dtype(Dtype::new_ref::<T>(), self.dtype())?;
        let data = self.read_data(index, context, None)?;
        Ok(unsafe { OperandTyped::new_input(data, Dtype::new_ref::<T>()) })
    }

    /// Returns the shape of the array, one element per dimension.
    fn shape(&self) -> &[u64];

    /// Returns the element type of the array.
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
    type ElementTypeChange<NewET: ElementType>: ArrayStorage<ElementType = NewET>
    where
        Self: Sized;

    /// Consume `self`, validate the new element type against the runtime dtype, and return the
    /// re-tagged storage.
    ///
    /// Returns [`ErrorKind::UnsupportedDtype`](crate::error::ErrorKind::UnsupportedDtype) if
    /// `NewET = Ty<T>` and the runtime dtype does not equal `T::DTYPE`. Always succeeds for
    /// `NewET = TypeDyn`.
    #[doc(hidden)]
    fn element_type_change<NewET: ElementType>(self) -> Result<Self::ElementTypeChange<NewET>>
    where
        Self: Sized;
}
