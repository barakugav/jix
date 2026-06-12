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
//! [`Array::to_typed::<T>()`](crate::Array::to_typed) to assert the expected element
//! type and regain compile-time tracking.
//!
//! # Notable items in this module
//!
//! - [`ArrayStorage`] - the trait all storage backends implement.
//! - [`ElementType`], [`Ty<T>`](Ty), [`TypeDyn`] - compile-time element type tracking.
//! - [`Compact`] - the main block-compressed storage backend.
//! - [`Plain`] and [`Scalar`] - adapters for non-compressed data.
//! - [`BlocksLayout`] - block geometry hints attached to every storage.

use crate::codec::{DecoderParams, EncoderParams};
use crate::dtype::Dtyped;
use crate::error::{check_dtype, ensure, Result};
use crate::ops::bulk_size;
use crate::util::cast_slice_mut;
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

mod any;
pub use any::*;

pub(crate) mod block;

/// Supertrait for [`ArrayStorage`] implementations whose element type is statically known.
///
/// `ArrayStorageTyped` is a shorthand for `ArrayStorage<ElementType = Ty<T>>`. It exposes the
/// concrete item type as the associated type `Item`. All element-wise operations - arithmetic,
/// comparisons, reductions, type casts - are bounded on this trait so the compiler can dispatch
/// to the correct scalar implementation without runtime checks.
///
/// To obtain `ArrayStorageTyped` from a `TypeDyn` array (e.g. after loading from disk), use
/// [`Array::to_typed::<T>()`](crate::Array::to_typed).
pub trait ArrayStorageTyped: ArrayStorage<ElementType = Ty<Self::Item>> {
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
impl ArrayStorageSpec<'_> {
    /// Returns the block layout metadata for this storage.
    ///
    /// Note that if the storage is not block-compressed, rather a view or adapter, the block layout
    /// should be treated as a hint for how to choose block shapes for operations on this storage,
    /// rather than a strict description of how the data is actually laid out on disk.
    #[inline(always)]
    pub fn blocks_layout(&self) -> &BlocksLayout {
        self.blocks_layout
    }

    /// Returns the encoder params of this storage, if any.
    ///
    /// Note that if the storage is not block-compressed, rather a view or adapter, the encoder params
    /// should be treated as a hint for how to choose encoder params for arrays crated from a chain of
    /// operations applied to this storage. This allows propagating encoder params from between arrays
    /// that are in the same context.
    #[inline(always)]
    pub fn encoder_params(&self) -> Option<&EncoderParams> {
        self.encoder_params
    }

    /// Returns the decoder params of this storage, if any.
    ///
    /// Note that if the storage is not block-compressed, rather a view or adapter, the decoder params
    /// should be treated as a hint for how to choose decoder params for arrays crated from a chain of
    /// operations applied to this storage. This allows propagating decoder params from between arrays
    /// that are in the same context.
    #[inline(always)]
    pub fn decoder_params(&self) -> Option<&DecoderParams> {
        self.decoder_params
    }
}

/// A borrowed reference to an [`ArrayStorage`], itself implementing [`ArrayStorage`].
///
/// Created by [`Array::as_ref`](crate::Array::as_ref) to produce an `Array<Ref<'_, S>>`
/// from `&Array<S>` without cloning the underlying storage.
pub struct Ref<'a, S>(pub(crate) &'a S);
impl<'a, S> Ref<'a, S> {
    /// Create a new `Ref` wrapper around the given storage reference.
    #[inline(always)]
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

    impl_array_storage_forward!('b, T, <S>);
}
impl<'a, S> Clone for Ref<'a, S> {
    fn clone(&self) -> Self {
        Self(self.0)
    }
}

macro_rules! impl_array_storage_forward {
    (<$($generics:tt),* $(,)?>) => {
        crate::storage::impl_array_storage_forward!('a, T, <$($generics),*>);
    };

    ($lifetime:tt, $generic:ident, <$($generics:tt),* $(,)?>) => {
        #[inline(always)]
        fn read_data(
            &self,
            index: &[::core::ops::Range<u64>],
            buf: &mut [u8],
            context: &crate::codec::ReadContext,
        ) -> crate::error::Result<()> {
            self.0.read_data(index, buf, context)
        }

        #[allow(refining_impl_trait)]
        #[inline(always)]
        fn read_data_typed<$lifetime, $generic>(
            &$lifetime self,
            index: &[::core::ops::Range<u64>],
            context: &$lifetime crate::codec::ReadContext,
        ) -> crate::error::Result<impl crate::storage::ReadData<$generic> + use<$lifetime, $generic, $($generics),*>>
        where
            $generic: crate::dtype::Dtyped,
        {
            self.0.read_data_typed(index, context)
        }

        #[inline(always)]
        fn shape(&self) -> &[u64] {
            self.0.shape()
        }
        #[inline(always)]
        fn dtype(&self) -> &crate::dtype::Dtype {
            self.0.dtype()
        }
        fn spec(&self) -> crate::storage::ArrayStorageSpec<'_> {
            self.0.spec()
        }
        fn as_compact(
            &self,
        ) -> Option<crate::storage::CompactBorrowed<'_, Self::ElementType, Self::Dimension>> {
            self.0.as_compact()
        }
    };
}
pub(crate) use impl_array_storage_forward;

/// An iterator-like trait for reading items from an `ArrayStorage` in bulk.
pub trait ReadData<T> {
    /// The total number of items available to read.
    fn len(&self) -> usize;

    /// Returns `true` if there are no items to read.
    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Read a contiguous chunk of `N` items starting from the given offset.
    ///
    /// # Panics
    ///
    /// Panics if `offset + N > self.len()`.
    fn read_bulk<const N: usize>(&mut self, offset: usize) -> [T; N];

    /// Read all items into the given buffer.
    ///
    /// The given buffer must have the exact size of `self.len() * size_of::<T>()` and be properly aligned for `T`.
    #[inline]
    fn to_buf(&mut self, buf: &mut [u8]) -> Result<()>
    where
        T: Dtyped,
        Self: Sized,
    {
        let dtype = T::DTYPE;
        let nitems = self.len();
        let required_size = nitems * size_of::<T>();
        let buf_len = buf.len();
        ensure!(
                buf_len == required_size,
                InvalidBufferSize,
                "Unexpected buffer size {buf_len} requested {nitems:?} nitems with dtype {dtype} (required size: {required_size})",
            );
        ensure!(
            (buf.as_ptr() as usize).is_multiple_of(align_of::<T>()),
            InvalidArgument,
            "Buffer pointer is not aligned to required alignment {} for dtype {dtype}",
            align_of::<T>(),
        );
        let buf = unsafe { cast_slice_mut::<u8, T>(buf) };
        assert_eq!(buf.len(), nitems);

        #[inline(always)]
        unsafe fn read_to_buf_impl<T, const BULK: usize>(
            data: &mut impl ReadData<T>,
            buf: &mut [T],
        ) -> Result<()>
        where
            T: Dtyped,
        {
            let nitems = data.len();
            assert_eq!(buf.len(), nitems);
            let mut offset = 0;
            while offset + BULK <= nitems {
                let chunk = data.read_bulk::<BULK>(offset);
                buf[offset..][..BULK].copy_from_slice(&chunk);
                offset += BULK;
            }
            while offset < nitems {
                let item = data.read_bulk::<1>(offset)[0];
                buf[offset] = item;
                offset += 1;
            }
            Ok(())
        }

        let bulk_size = bulk_size::<T>();
        assert!(bulk_size.is_power_of_two());
        // this is a compile time check, the compiler knows the value of `bulk_size::<T>()`
        let read_fn = match bulk_size {
            1 => read_to_buf_impl::<T, 1>,
            2 => read_to_buf_impl::<T, 2>,
            4 => read_to_buf_impl::<T, 4>,
            8 => read_to_buf_impl::<T, 8>,
            16 => read_to_buf_impl::<T, 16>,
            32 => read_to_buf_impl::<T, 32>,
            64 => read_to_buf_impl::<T, 64>,
            128 => read_to_buf_impl::<T, 128>,
            256 => read_to_buf_impl::<T, 256>,
            512 => read_to_buf_impl::<T, 512>,
            _ => read_to_buf_impl::<T, 1024>,
        };
        unsafe { read_fn(self, buf) }
    }
}
pub(crate) trait ReadDataExt<T>: ReadData<T>
where
    T: Copy + Send + Sync + Sized + 'static,
{
    fn map_items<U, F: FnMut(T) -> U>(self, f: F) -> impl ReadData<U>
    where
        Self: Sized,
    {
        struct Map<T, U, R, F> {
            inner: R,
            f: F,
            _phantom: std::marker::PhantomData<(T, U)>,
        }
        impl<T, U, R, F> ReadData<U> for Map<T, U, R, F>
        where
            R: ReadData<T>,
            F: FnMut(T) -> U,
        {
            #[inline(always)]
            fn read_bulk<const N: usize>(&mut self, offset: usize) -> [U; N] {
                self.inner.read_bulk::<N>(offset).map(&mut self.f)
            }
            #[inline(always)]
            fn len(&self) -> usize {
                self.inner.len()
            }
        }
        Map {
            inner: self,
            f,
            _phantom: std::marker::PhantomData,
        }
    }

    fn zip_items<U, R>(self, other: R) -> impl ReadData<(T, U)>
    where
        Self: Sized,
        R: ReadData<U>,
        U: Copy + Send + Sync + Sized + 'static,
    {
        struct Zip<T, U, R1, R2> {
            left: R1,
            right: R2,
            _phantom: std::marker::PhantomData<(T, U)>,
        }
        impl<T, U, R1, R2> ReadData<(T, U)> for Zip<T, U, R1, R2>
        where
            R1: ReadData<T>,
            R2: ReadData<U>,
            T: Copy + Send + Sync + Sized + 'static,
            U: Copy + Send + Sync + Sized + 'static,
        {
            #[inline(always)]
            fn read_bulk<const N: usize>(&mut self, offset: usize) -> [(T, U); N] {
                let left = self.left.read_bulk::<N>(offset);
                let right = self.right.read_bulk::<N>(offset);
                std::array::from_fn(|i| (left[i], right[i]))
            }
            #[inline(always)]
            fn len(&self) -> usize {
                assert_eq!(self.left.len(), self.right.len());
                self.left.len()
            }
        }
        Zip {
            left: self,
            right: other,
            _phantom: std::marker::PhantomData,
        }
    }

    fn transmute_items<U>(self) -> Result<impl ReadData<U>>
    where
        Self: Sized,
        T: Dtyped,
        U: Dtyped,
    {
        check_dtype(&T::DTYPE, &U::DTYPE)?;
        // SAFETY: We checked that `T` has the same dtype as `K::Output`
        Ok(unsafe { self.transmute_items_unsafe() })
    }

    unsafe fn transmute_items_unsafe<U>(self) -> impl ReadData<U>
    where
        Self: Sized,
        U: Copy + Send + Sync + Sized + 'static,
    {
        struct Transmute<T, U, R> {
            inner: R,
            _phantom: std::marker::PhantomData<(T, U)>,
        }
        impl<T, U, R> ReadData<U> for Transmute<T, U, R>
        where
            R: ReadData<T>,
            T: Copy + Send + Sync + Sized + 'static,
            U: Copy + Send + Sync + Sized + 'static,
        {
            #[inline(always)]
            fn read_bulk<const N: usize>(&mut self, offset: usize) -> [U; N] {
                const {
                    assert!(
                        size_of::<T>() == size_of::<U>(),
                        "T and U must have equal size to reinterpret",
                    );
                }
                let chunk: [T; N] = self.inner.read_bulk::<N>(offset);
                unsafe { std::mem::transmute_copy::<[T; N], [U; N]>(&chunk) }
            }
            #[inline(always)]
            fn len(&self) -> usize {
                self.inner.len()
            }
        }
        Transmute {
            inner: self,
            _phantom: std::marker::PhantomData,
        }
    }
}
impl<T, R> ReadDataExt<T> for R
where
    R: ReadData<T>,
    T: Copy + Send + Sync + Sized + 'static,
{
}
