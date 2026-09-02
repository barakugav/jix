//! Storage backends for [`Array`](crate::Array).
//!
//! Every `Array<S>` is backed by a storage type `S: ArrayStorage`, which exposes three things:
//! the array's shape, its element dtype, and [`read_data`](ArrayStorage::read_data)
//! which reads any rectangular sub-region and returns it as a
//! [`StridedBuf`] - either a borrowed strided view or bytes written into
//! a caller-supplied destination. All higher-level operations are built on top of these three methods.
//!
//! # Storage implementations
//!
//! The primary storages are block-compressed backends:
//! - [`Compact`] - heap-allocated
//! - [`CompactMmap`] - memory-mapped file
//!
//! An adapter lets non-compressed data participate in the same `Array` world (such as math operations with compressed arrays):
//! - [`Plain`] - a zero-copy view into a contiguous or strided in-memory buffer.
//!
//! Operations on `Array` produce lazy views whose storage wraps the original and applies
//! the transformation at read time. These are defined in [`jix::ops`](crate::ops) and include shape
//! operations (`Reshape`, `Slice`, `PermuteAxes`, `Broadcast`, ...), element-wise operations
//! (`Neg`, `Add`, `Exp`, `Cast`, ...), reductions (`Sum`, `Mean`, ...), etc.
//!
//! # Element types
//!
//! Every storage carries two pieces of compile-time information as associated types:
//!
//! - **[`ElementType`]** - the compile-time element type, accessible via
//!   `S::ElementType`. This is either [`Ty<T>`] (the concrete scalar type `T` is known at
//!   compile time) or [`TypeDyn`] (only known at runtime, e.g. for arrays loaded from disk).
//!
//! - **[`Dimension`](crate::Dimension)** - the compile-time dimension, accessible via
//!   `S::Dimension`. Either [`Dim<N>`](crate::Dim) (known statically) or
//!   [`DimDyn`](crate::DimDyn) (runtime only).
//!
//! The [`ArrayStorageTyped`] supertrait is a shorthand for
//! `ArrayStorage<ElementType = Ty<T>>`. All element-wise operations require it -
//! the element type must be known at compile time so the compiler can dispatch to the
//! correct scalar implementation.
//!
//! Arrays constructed from typed sources (e.g. [`Array::compact_ndarray`](crate::Array::compact_ndarray))
//! are automatically typed. Arrays loaded from disk carry [`TypeDyn`]; call
//! [`Array::into_typed::<T>()`](crate::Array::into_typed) to assert the expected element
//! type and regain compile-time tracking.
//!
//! # Notable items in this module
//!
//! - [`ArrayStorage`] - the trait all storage backends implement.
//! - [`Compact`] - the main block-compressed storage backend.
//! - [`Plain`] - adapter for non-compressed data.

use std::ops::Range;

use crate::dtype::Dtyped;
use crate::{ArrayStorage, ElementType, Ty, TypeDyn};

pub(crate) mod core_trait;

mod compact;
pub use compact::*;

mod plain;
pub use plain::*;

pub(crate) mod params;
pub use params::ArraySpec;

mod any;
pub use any::*;

pub(crate) mod block;
pub use block::BlockSize;

pub(crate) mod scalar;

mod buf;
pub use buf::*;

mod elementwise_pipeline;
pub use elementwise_pipeline::*;

mod info;
pub use info::*;

/// Supertrait for [`ArrayStorage`] implementations whose element type is statically known.
///
/// `ArrayStorageTyped` is a shorthand for `ArrayStorage<ElementType = Ty<T>>`. It exposes the
/// concrete item type as the associated type `Item`. All element-wise operations - arithmetic,
/// comparisons, reductions, type casts - are bounded on this trait so the compiler can dispatch
/// to the correct scalar implementation without runtime checks.
///
/// To obtain `ArrayStorageTyped` from a `TypeDyn` array (e.g. after loading from disk), use
/// [`Array::into_typed::<T>()`](crate::Array::into_typed).
pub trait ArrayStorageTyped: ArrayStorage<ElementType = Ty<Self::Item>> + Sized {
    /// The concrete Rust element type stored in this array (e.g. `f32`, `i64`).
    type Item: Dtyped;
}
impl<S, T> ArrayStorageTyped for S
where
    S: ArrayStorage<ElementType = Ty<T>>,
    T: Dtyped,
{
    type Item = T;
}

/// A view reference to an [`ArrayStorage`], itself implementing [`ArrayStorage`].
///
/// Created by [`Array::view`](crate::Array::view) to produce an `Array<View<'_, S>>`
/// from `&Array<S>` without cloning the underlying storage.
pub struct View<'a, S>(pub(crate) &'a S);
impl<'a, S> View<'a, S> {
    /// Create a new `View` wrapper around the given storage reference.
    #[inline(always)]
    pub fn new(storage: &'a S) -> Self {
        Self(storage)
    }
}
impl<'a, S> ArrayStorage for View<'a, S>
where
    S: ArrayStorage,
{
    type ElementType = S::ElementType;
    type Dimension = S::Dimension;

    impl_array_storage_forward!('b, T, <S>);

    fn info(&self) -> ArrayStorageInfo<'_> {
        ArrayStorageInfo::new_deps("View", [self.0])
    }
    crate::ops::impl_dimension_change_default!();
    crate::ops::impl_element_type_change_default!();
}
impl<'a, S> Clone for View<'a, S> {
    fn clone(&self) -> Self {
        Self(self.0)
    }
}

impl ArrayStorage for &dyn ArrayStorage {
    type ElementType = crate::TypeDyn;
    type Dimension = crate::DimDyn;

    #[inline(always)]
    fn read_data<'a>(
        &'a self,
        index: &[Range<u64>],
        context: &'a crate::codec::ReadContext,
        out: Option<&'a mut crate::storage::StridedBuf<'_>>,
    ) -> crate::error::Result<crate::storage::StridedBuf<'a>> {
        (**self).read_data(index, context, out)
    }

    #[inline(always)]
    fn shape(&self) -> &[u64] {
        (**self).shape()
    }

    #[inline(always)]
    fn dtype(&self) -> &crate::dtype::Dtype {
        (**self).dtype()
    }

    #[inline]
    fn spec(&self) -> crate::storage::ArraySpec<'_> {
        (**self).spec()
    }

    fn info(&self) -> crate::storage::ArrayStorageInfo<'_> {
        (**self).info()
    }

    crate::ops::impl_dimension_change_default!();
    crate::ops::impl_element_type_change_default!();
}

macro_rules! impl_array_storage_forward {
    (<$($generics:tt),* $(,)?>) => {
        crate::storage::impl_array_storage_forward!('a, T, <$($generics),*>);
    };

    ($lifetime:tt, $generic:ident, <$($generics:tt),* $(,)?>) => {
        #[inline(always)]
        fn read_data<'rd>(
            &'rd self,
            index: &[::core::ops::Range<u64>],
            context: &'rd crate::codec::ReadContext,
            out: Option<&'rd mut crate::storage::StridedBuf<'_>>,
        ) -> crate::error::Result<crate::storage::StridedBuf<'rd>> {
            self.0.read_data(index, context, out)
        }

        #[allow(refining_impl_trait)]
        #[inline(always)]
        fn read_as_elementwise_pipeline<$lifetime, $generic>(
            &$lifetime self,
            index: &[::core::ops::Range<u64>],
            context: &$lifetime crate::codec::ReadContext,
        ) -> crate::error::Result<impl crate::storage::ElementwisePipeline<$generic> + use<$lifetime, $generic, $($generics),*>>
        where
            $generic: crate::dtype::Dtyped,
        {
            self.0.read_as_elementwise_pipeline(index, context)
        }

        #[inline(always)]
        fn shape(&self) -> &[u64] {
            self.0.shape()
        }
        #[inline(always)]
        fn dtype(&self) -> &crate::dtype::Dtype {
            self.0.dtype()
        }
        #[inline(always)]
        fn spec(&self) -> crate::storage::ArraySpec<'_> {
            self.0.spec()
        }
        #[inline]
        fn as_compact(
            &self,
        ) -> Option<crate::storage::CompactBorrowed<'_, Self::ElementType, Self::Dimension>> {
            self.0.as_compact()
        }
    };
}
pub(crate) use impl_array_storage_forward;
