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
//! - [`Compact`] - heap-allocated
//! - [`CompactMmap`] - memory-mapped file
//!
//! Two adapters let non-compressed data participate in the same `Array` world (such as math operations with compressed arrays):
//! - [`Plain`] - a zero-copy view into a contiguous or strided in-memory buffer.
//! - [`Scalar<T>`] - a single value broadcast to any shape.
//!
//! Operations on `Array` produce lazy views whose storage wraps the original and applies
//! the transformation at read time. These are defined in [`zix::ops`](crate::ops) and include shape
//! operations (`Reshape`, `Slice`, `PermuteAxes`, `Broadcast`, ...), element-wise operations
//! (`Neg`, `Add`, `Exp`, `Cast`, ...), reductions (`Sum`, `Mean`, ...), etc.
//!
//! # Element types
//!
//! Every storage carries two pieces of compile-time information as associated types:
//!
//! - **[`ElementType`]** — the compile-time element type, accessible via
//!   `S::ElementType`. This is either [`Ty<T>`] (the concrete scalar type `T` is known at
//!   compile time) or [`TypeDyn`] (only known at runtime, e.g. for arrays loaded from disk).
//!
//! - **[`Dimension`](crate::Dimension)** — the compile-time dimension, accessible via
//!   `S::Dimension`. Either [`Dim<N>`](crate::Dim) (known statically) or
//!   [`DimDyn`](crate::DimDyn) (runtime only).
//!
//! The [`ArrayStorageTyped`] supertrait is a shorthand for
//! `ArrayStorage<ElementType = Ty<T>>`. All element-wise operations require it —
//! the element type must be known at compile time so the compiler can dispatch to the
//! correct scalar implementation.
//!
//! Arrays constructed from typed sources (e.g. [`Array::compact_array`](crate::Array::compact_array))
//! are automatically typed. Arrays loaded from disk carry [`TypeDyn`]; call
//! [`Array::to_typed::<T>()`](crate::Array::to_typed) to assert the expected element
//! type and regain compile-time tracking.
//!
//! # Notable items in this module
//!
//! - [`ArrayStorage`] — the trait all storage backends implement.
//! - [`ElementType`], [`Ty<T>`](Ty), [`TypeDyn`] — compile-time element type tracking.
//! - [`Compact`] — the main block-compressed storage backend.
//! - [`Plain`] and [`Scalar`] — adapters for non-compressed data.
//! - [`BlocksLayout`] — block geometry hints attached to every storage.

use crate::codec::{DecoderParams, EncoderParams};
use crate::dtype::Dtyped;
use crate::{ArrayStorage, ElementType, Ty, TypeDyn};

pub(crate) mod core;

mod layout;
pub use layout::*;

mod compressed;
pub use compressed::*;

mod plain;
pub use plain::*;

mod scalar;
pub use scalar::*;

pub(crate) mod block;

/// Internal metadata of ArrayStorage.
///
/// Carries the information [`Array`](crate::Array) needs when creating a new storage
/// from an existing one - such as during `copy`, `copy_with`, and lazy view operations.
/// Not intended to be used directly.
pub struct ArrayStorageSpec<'a> {
    pub(crate) blocks_layout: &'a BlocksLayout,
    pub(crate) encoder_params: Option<&'a EncoderParams>,
    pub(crate) decoder_params: Option<&'a DecoderParams>,
    // pub(crate) decoder_config: Option<&'a DecoderCodecConfig>,
}

/// Supertrait for [`ArrayStorage`] implementations whose element type is statically known.
///
/// `ArrayStorageTyped` is a shorthand for `ArrayStorage<ElementType = Ty<T>>`. It exposes the
/// concrete item type as the associated type `Item`. All element-wise operations — arithmetic,
/// comparisons, reductions, type casts — are bounded on this trait so the compiler can dispatch
/// to the correct scalar implementation without runtime checks.
///
/// To obtain `ArrayStorageTyped` from a `TypeDyn` array (e.g. after loading from disk), use
/// [`Array::to_typed::<T>()`](crate::Array::to_typed).
pub trait ArrayStorageTyped: ArrayStorage<ElementType = Ty<Self::Item>> {
    type Item: Dtyped;
}
impl<S, T> ArrayStorageTyped for S
where
    S: ArrayStorage<ElementType = Ty<T>>,
    T: Dtyped,
{
    type Item = T;
}

/// A borrowed reference to an [`ArrayStorage`], itself implementing [`ArrayStorage`].
///
/// Created by [`Array::as_ref`](crate::Array::as_ref) to produce an `Array<Ref<'_, S>>`
/// from `&Array<S>` without cloning the underlying storage.
pub struct Ref<'a, S>(pub(crate) &'a S);
impl<'a, S> Ref<'a, S> {
    /// Create a new `Ref` wrapper around the given storage reference.
    pub fn new(storage: &'a S) -> Self {
        Self(storage)
    }
}
impl<'a, S> ArrayStorage for Ref<'a, S>
where
    S: ArrayStorage,
{
    type ElementType = S::ElementType;
    type Dimension = S::Dimension;

    impl_array_storage_forward!();
}

macro_rules! impl_array_storage_forward {
    () => {
        fn read_data(
            &self,
            index: &[::core::ops::Range<u64>],
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
        fn as_compact(
            &self,
        ) -> Option<crate::storage::CompactBorrowed<'_, Self::ElementType, Self::Dimension>> {
            self.0.as_compact()
        }
    };
}
pub(crate) use impl_array_storage_forward;
