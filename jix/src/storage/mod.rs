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
//! - **[`Dimension`]** - the compile-time dimension, accessible via
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
use crate::error::{check_buffer_aligned, check_dtype, ensure, Result};
use crate::ops::LanesInfo;
use crate::util::cast_slice_mut;
use crate::{
    array_from_fn_inline, assert_unchecked_eq, default_logical_strides_slice, dim_arr,
    nd_iter_unordered, ArrayExt, ArrayStorage, Dimension, ElementType, Ty, TypeDyn,
};

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

    fn info(&self) -> ArrayStorageInfo<'_> {
        ArrayStorageInfo::new_deps("Ref", [self.0])
    }
    crate::ops::impl_dimension_change_default!();
    crate::ops::impl_element_type_change_default!();
}
impl<'a, S> Clone for Ref<'a, S> {
    fn clone(&self) -> Self {
        Self(self.0)
    }
}

impl ArrayStorage for &dyn ArrayStorage {
    type ElementType = crate::TypeDyn;
    type Dimension = crate::DimDyn;

    #[inline(always)]
    fn read_data(
        &self,
        index: &[Range<u64>],
        buf: &mut crate::storage::OutBuf<'_>,
        context: &crate::codec::ReadContext,
    ) -> crate::error::Result<()> {
        (**self).read_data(index, buf, context)
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
        fn read_data(
            &self,
            index: &[::core::ops::Range<u64>],
            buf: &mut crate::storage::OutBuf,
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

/// An interface trait for reading items from an `ArrayStorage` in bulk.
///
/// Returned by [`ArrayStorage::read_data_typed`], used by element-wise operations.
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
}
pub(crate) trait ReadDataExt<T>: ReadData<T>
where
    T: Copy + Send + Sync + Sized + 'static,
{
    /// Write all `self.len()` items into `buf` for the read region `index`.
    ///
    /// A contiguous destination (see [`OutBuf`]) is filled with a bulk copy; a strided destination
    /// receives each item scattered to its position. `D` is the dimension used to walk a strided
    /// destination.
    #[inline(never)]
    fn to_buf<D: Dimension>(&mut self, buf: &mut OutBuf, index: &[Range<u64>]) -> Result<()>
    where
        T: Dtyped,
        Self: Sized,
    {
        let read_fn = match <T as LanesInfo>::LANES {
            1 => read_data_to_buf::<T, Self, 1>,
            2 => read_data_to_buf::<T, Self, 2>,
            4 => read_data_to_buf::<T, Self, 4>,
            8 => read_data_to_buf::<T, Self, 8>,
            16 => read_data_to_buf::<T, Self, 16>,
            32 => read_data_to_buf::<T, Self, 32>,
            64 => read_data_to_buf::<T, Self, 64>,
            128 => read_data_to_buf::<T, Self, 128>,
            256 => read_data_to_buf::<T, Self, 256>,
            512 => read_data_to_buf::<T, Self, 512>,
            _ => read_data_to_buf::<T, Self, 1024>,
        };
        let shape = dim_arr(index.len(), |d| (index[d].end - index[d].start) as usize);
        read_fn(self, buf, &shape)
    }

    #[inline(always)]
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
                self.inner.read_bulk::<N>(offset).map_inline(&mut self.f)
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

    #[inline(always)]
    fn zip_items<U, R>(self, other: R) -> impl ReadData<(T, U)>
    where
        Self: Sized,
        R: ReadData<U>,
        U: Copy + Send + Sync + Sized + 'static,
    {
        assert_eq!(self.len(), other.len());

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
                array_from_fn_inline(|i| (left[i], right[i]))
            }
            #[inline(always)]
            fn len(&self) -> usize {
                unsafe { assert_unchecked_eq!(self.left.len(), self.right.len()) };
                self.left.len()
            }
        }
        Zip {
            left: self,
            right: other,
            _phantom: std::marker::PhantomData,
        }
    }

    #[inline(always)]
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

    #[inline(always)]
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

#[inline(always)]
fn read_data_to_buf<T, R, const LANES: usize>(
    data: &mut R,
    buf: &mut OutBuf,
    shape: &[usize],
) -> Result<()>
where
    T: Dtyped,
    R: ReadData<T>,
{
    let dtype = T::DTYPE;
    let (out, strides) = buf.get_mut(data.len(), &dtype);

    match strides {
        None => read_data_to_buf_contiguous::<T, LANES>(data, out),
        Some(strides) => read_data_to_buf_strided::<T, R, LANES>(data, out, strides, shape),
    }
}

#[inline(never)]
fn read_data_to_buf_contiguous<T, const LANES: usize>(
    data: &mut impl ReadData<T>,
    out: &mut [u8],
) -> Result<()>
where
    T: Dtyped,
{
    let dtype = T::DTYPE;
    let nitems = data.len();
    let required_size = nitems * size_of::<T>();
    let buf_len = out.len();
    ensure!(
        buf_len == required_size,
        InvalidBufferSize,
        "Unexpected buffer size {buf_len} requested {nitems:?} nitems with dtype {dtype} (required size: {required_size})",
    );
    check_buffer_aligned(out.as_ptr(), &dtype)?;

    let buf = unsafe { cast_slice_mut::<u8, T>(out) };
    assert_eq!(buf.len(), nitems);
    let mut offset = 0;
    while offset + LANES <= nitems {
        let chunk = data.read_bulk::<LANES>(offset);
        buf[offset..LANES + offset].copy_from_slice(&chunk);
        offset += LANES;
    }
    while offset < nitems {
        let item = data.read_bulk::<1>(offset)[0];
        buf[offset] = item;
        offset += 1;
    }
    Ok(())
}

fn read_data_to_buf_strided<T, R, const LANES: usize>(
    data: &mut R,
    out: &mut [u8],
    strides: &[usize],
    shape: &[usize],
) -> Result<()>
where
    T: Dtyped,
    R: ReadData<T>,
{
    let itemsize = size_of::<T>();
    // The source is read straight from `data` via `read_bulk`, indexed in *elements*
    // over the region's row-major logical order - so operand 1's strides and offsets
    // are element counts with layout (1, 1).
    let src_offset_strides = default_logical_strides_slice(shape);

    // Operand 0 is the destination byte buffer (`out`); operand 1 is the source, read
    // via `read_bulk` and indexed in elements.
    nd_iter_unordered(
        shape,
        [strides, src_offset_strides.as_ref()],
        [(itemsize, align_of::<T>()), (1, 1)],
        |flags| {
            debug_assert!(flags.aligned[1]);
            let aligned = flags.aligned[0] && out.as_ptr().cast::<T>().is_aligned();
            let contiguous = [flags.contiguous[0], flags.contiguous[1]];
            let inner_loop_fn = match (aligned, contiguous) {
                (true, [true, true]) => inner_loop::<T, R, LANES, true, true, true>,
                (true, [true, false]) => inner_loop::<T, R, LANES, true, true, false>,
                (true, [false, true]) => inner_loop::<T, R, LANES, true, false, true>,
                (true, [false, false]) => inner_loop::<T, R, LANES, true, false, false>,
                (false, [true, true]) => inner_loop::<T, R, LANES, false, true, true>,
                (false, [true, false]) => inner_loop::<T, R, LANES, false, true, false>,
                (false, [false, true]) => inner_loop::<T, R, LANES, false, false, true>,
                (false, [false, false]) => inner_loop::<T, R, LANES, false, false, false>,
            };
            let dst_base = out.as_mut_ptr();
            move |offsets: [usize; 2], len, inner_strides: [usize; 2]| {
                let [dst_offset, src_offset] = offsets;
                let [dst_stride, src_stride] = inner_strides;
                let dst = unsafe { dst_base.add(dst_offset) };
                inner_loop_fn(&mut *data, dst, src_offset, len, dst_stride, src_stride)
            }
        },
    );

    // One inner 1-d run: read `len` elements from `data` starting at logical element
    // `src_index` (stepping `src_stride` elements) and scatter them into `dst`
    // (stepping `dst_stride` bytes). A contiguous source pulls `LANES` consecutive
    // elements per `read_bulk`; otherwise it reads one at a time.
    #[inline(never)]
    fn inner_loop<
        T: Copy,
        R: ReadData<T>,
        const LANES: usize,
        const ALIGNED: bool,
        const DST_CONTIGUOUS: bool,
        const SRC_CONTIGUOUS: bool,
    >(
        data: &mut R,
        dst: *mut u8,
        src_index: usize,
        len: usize,
        dst_stride: usize,
        src_stride: usize,
    ) {
        let dst = dst.cast::<T>();
        if DST_CONTIGUOUS {
            debug_assert_eq!(dst_stride, size_of::<T>());
        }
        if SRC_CONTIGUOUS {
            debug_assert_eq!(src_stride, 1);
        }
        let write = |j: usize, val: T| {
            let elm = if DST_CONTIGUOUS {
                unsafe { dst.add(j) }
            } else {
                unsafe { dst.cast::<u8>().add(j * dst_stride).cast::<T>() }
            };
            if ALIGNED {
                unsafe { elm.write(val) };
            } else {
                unsafe { elm.write_unaligned(val) };
            }
        };
        let mut i = 0;
        if SRC_CONTIGUOUS && len >= LANES {
            while i <= len - LANES {
                let chunk = data.read_bulk::<LANES>(src_index + i);
                #[allow(clippy::needless_range_loop)]
                for k in 0..LANES {
                    write(i + k, chunk[k]);
                }
                i += LANES;
            }

            while i < len {
                write(i, data.read_bulk::<1>(src_index + i)[0]);
                i += 1;
            }
        } else {
            while i < len {
                write(i, data.read_bulk::<1>(src_index + i * src_stride)[0]);
                i += 1;
            }
        }
    }

    Ok(())
}
