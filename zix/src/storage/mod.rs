//! Storage backends for [`Array`](crate::Array).
//!
//! Every `Array<S>` is backed by a storage type `S: ArrayStorage`, which exposes three things:
//! the array's shape, its element dtype, and [`read_data`](ArrayStorage::read_data)
//! which reads any rectangular sub-region into a caller-supplied byte buffer.
//! All higher-level operations are built on top of these three methods.
//!
//! # Storage implementations
//!
//! The primary storages are block-compressed backends:
//! - [`Compact`] — heap-allocated
//! - [`CompactMmap`] — memory-mapped file
//!
//! Two adapters let non-compressed data participate in the same `Array` world (such as math operations with compressed arrays):
//! - [`Plain`] — a zero-copy view into a contiguous or strided in-memory buffer.
//! - [`Scalar<T>`] — a single value broadcast to any shape.
//!
//! Operations on `Array` produce lazy views whose storage wraps the original and applies
//! the transformation at read time. These are defined in [`zix::ops`](crate::ops) and include shape
//! operations (`Reshape`, `Slice`, `PermuteAxes`, `Broadcast`, …), element-wise operations
//! (`Neg`, `Add`, `Exp`, `AsType`, …), reductions (`Sum`, `Mean`, …), etc.
//!
//! # Notable items in this module
//!
//! - [`ArrayStorage`] — the trait all storage backends implement.
//! - [`Compact`] — the main block-compressed storage backend.
//! - [`Plain`] and [`Scalar`] — adapters for non-compressed data.
//! - [`BlocksLayout`] — block geometry hints attached to every storage.

use std::ops::Range;

use crate::codec::{DecoderParams, EncoderParams, ReadContext};
use crate::dtype::Dtype;
use crate::error::Result;

mod layout;
pub use layout::*;

mod compressed;
pub use compressed::*;

mod plain;
pub use plain::*;

mod scalar;
pub use scalar::*;

pub(crate) mod block;

/// The backing data source of an [`Array<S>`](crate::Array).
///
/// `Array<S>` is generic over its storage `S: ArrayStorage`, which provides three pieces
/// of information: the array's shape, its element type, and the ability to read any
/// rectangular sub-region into a byte buffer. Everything else — slicing, reshaping,
/// arithmetic, reductions — is built on top of these three methods.
///
/// # Primary storage backends
///
/// The main concrete storages are the block-compressed backends:
/// `compressed::Compact` (heap-allocated) and `compressed::CompactMmap` (memory-mapped file)
///  These store the array as independently
/// compressed nd-blocks and are the primary on-disk format.
///
/// # Adapters
///
/// Two adapters let plain data participate in the same `Array<S>` world:
/// - `plain::Plain` — wraps a contiguous or strided in-memory ndarray, used when
///   operating on regular Rust/ndarray data alongside compressed arrays.
/// - `scalar::Scalar<T>` — represents a single scalar value broadcast to any shape,
///   used as the right-hand side in operations like `array + scalar`.
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
/// arr.astype::<f32>()        -> Array<AsType<S>>
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
/// expression — e.g. `Array<Add<Neg<S1>, Reshape<S2>>>` — is resolved by the compiler,
/// and only the final `read_data` call touches actual bytes.
pub trait ArrayStorage {
    /// Read a sub-region of the array into a caller-supplied byte buffer.
    ///
    /// This is the single I/O method that every storage backend must implement.
    /// All higher-level read operations (`to_ndarray`, `to_ndarray_sub`, etc.) bottom
    /// out here.
    ///
    /// # Arguments
    ///
    /// - `index` — one half-open range per dimension (`start..end`).
    ///   The number of ranges must equal `self.shape().len()`.
    ///   Ranges must be within the array shape bounds; empty ranges are allowed.
    /// - `buf` — destination byte buffer.
    ///   Must be exactly `index.iter().map(|r| r.len()).product() * dtype.itemsize()` bytes.
    ///   Must be aligned to `dtype.alignment()`.
    ///   On success the elements are written in row-major (C-contiguous) order.
    /// - `context` — read context carrying the decoder state.
    fn read_data(&self, index: &[Range<u64>], buf: &mut [u8], context: &ReadContext) -> Result<()>;

    /// Returns the shape of the array, one element per dimension.
    fn shape(&self) -> &[u64];

    /// Returns the element type of the array.
    fn dtype(&self) -> &Dtype;

    /// Returns metadata about this storage backend.
    ///
    /// Used internally by [`Array`](crate::Array) to propagate block geometry and codec
    /// parameters through lazy view operations and when re-encoding via `copy` / `copy_with`.
    /// Not intended to be called directly.
    #[doc(hidden)]
    fn _spec(&self) -> ArrayStorageSpec<'_>;

    #[doc(hidden)]
    fn as_compact(&self) -> Option<CompactBorrowed<'_>> {
        None
    }
}

/// Internal metadata of ArrayStorage.
///
/// Carries the information [`Array`](crate::Array) needs when creating a new storage
/// from an existing one — such as during `copy`, `copy_with`, and lazy view operations.
/// Not intended to be used directly.
pub struct ArrayStorageSpec<'a> {
    pub(crate) blocks_layout: &'a BlocksLayout,
    pub(crate) encoder_params: Option<&'a EncoderParams>,
    pub(crate) decoder_params: Option<&'a DecoderParams>,
    // pub(crate) decoder_config: Option<&'a DecoderCodecConfig>,
}

/// A borrowed reference to an [`ArrayStorage`], itself implementing [`ArrayStorage`].
///
/// Created by [`Array::as_ref`](crate::Array::as_ref) to produce an `Array<Ref<'_, S>>`
/// from `&Array<S>` without cloning the underlying storage.
pub struct Ref<'a, S>(pub(crate) &'a S);
impl_array_storage_forward!(Ref<'a, S> where S: ArrayStorage);

macro_rules! impl_array_storage_forward {
    ($wrapper:ident $(<$($gen:tt),*>)? $(where $($wh:tt)*)?) => {
        impl $(<$($gen),*>)? crate::storage::ArrayStorage for $wrapper $(<$($gen),*>)?
        where
            $($($wh)*)?
        {
            fn read_data(
                &self,
                index: &[core::ops::Range<u64>],
                buf: &mut [u8],
                context: &crate::codec::ReadContext,
            ) -> crate::error::Result<()> {
                self.0.read_data(index, buf, context)
            }
            fn shape(&self) -> &[u64] {
                self.0.shape()
            }
            fn dtype(&self) -> &crate::dtype::Dtype {
                self.0.dtype()
            }
            fn _spec(&self) -> crate::storage::ArrayStorageSpec<'_> {
                self.0._spec()
            }
            fn as_compact(&self) -> Option<crate::storage::CompactBorrowed<'_>> {
                self.0.as_compact()
            }
        }
    };
}
pub(crate) use impl_array_storage_forward;
