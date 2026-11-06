#[cfg(not(miri))]
use std::cell::UnsafeCell;

use crate::buf_pool::{BufferPool, PoolBuf};
use crate::codec::{Decoder, DecoderCodecConfig, DecoderParams};
use crate::dtype::Alignment;
use crate::error::Result;

/// A context with reusable buffers and a long-lived decoder instance.
///
/// Allocating temporary buffers on demand and initializing a codec decoder on every block read
/// internally in the array storage can be expensive, especially for small blocks.
/// `ReadContext` holds reusable buffers and decoder instances that can be shared across multiple
/// reads to amortize these costs.
///
/// A `ReadContext` is tied to a single thread - it is `!Sync`.
/// For concurrent reads from multiple threads, create one `ReadContext` per thread.
///
/// # Obtaining a `ReadContext`
///
/// The preferred way is [`Array::read_ctx()`](crate::Array::read_ctx), which picks up the decoder
/// parameters stored alongside the array data:
///
/// ```
/// use jix::{Array, ArrayParams};
/// use ndarray::array;
///
/// let za = Array::compact_ndarray(&array![1i32, 2, 3, 4])?;
///
/// // read_ctx() inherits the decoder config from the array.
/// let ctx = za.read_ctx();
/// let out = za.to_ndarray_sub(&[1..3], &ctx)?;
/// assert_eq!(out, array![2, 3]);
/// # Ok::<(), jix::Error>(())
/// ```
///
/// `ReadContext::default()` is also available if you need to create a context independently of a
/// specific array.
/// ```
/// use jix::ReadContext;
/// use jix::Array;
/// use ndarray::array;
///
/// let za = Array::compact_ndarray(&array![42i32, 17, 6, 99, 51])?;
/// let out = za.to_ndarray_sub(&[0..3], &ReadContext::default())?;
/// assert_eq!(out, array![42, 17, 6]);
/// # Ok::<(), jix::Error>(())
/// ```
///
/// # Reusing a context
///
/// A single `ReadContext` can be passed to multiple successive reads. Reusing it avoids
/// reinitializing the decompressor and keeps the scratch buffer allocations warm.
pub struct ReadContext {
    #[cfg(not(miri))]
    pub(crate) decompressor: UnsafeCell<zstd::bulk::Decompressor<'static>>,
    pub(crate) buffer_pool: BufferPool,
}
impl ReadContext {
    /// Creates a new `ReadContext` configured with the given decoder parameters.
    ///
    /// Prefer [`Array::read_ctx()`](crate::Array::read_ctx) over calling this directly - it
    /// automatically uses the decoder parameters that match the array's stored codec configuration.
    pub(crate) fn new(#[allow(unused)] decoder_params: &DecoderParams) -> Result<Self> {
        Ok(Self {
            #[cfg(not(miri))]
            decompressor: UnsafeCell::new(zstd::bulk::Decompressor::new().unwrap()),
            buffer_pool: BufferPool::new(),
        })
    }

    #[inline]
    pub(crate) fn decoder<'a>(&'a self, config: &'a DecoderCodecConfig) -> Decoder<'a> {
        Decoder::new(self, config)
    }

    #[inline]
    pub(crate) fn allocate_buf(&self, size: usize, alignment: Alignment) -> PoolBuf<'_> {
        self.buffer_pool.get(size, alignment)
    }
}
const _: fn() = || {
    fn assert_send<T: Send>() {}
    assert_send::<ReadContext>();
};

impl Default for ReadContext {
    fn default() -> Self {
        Self::new(&DecoderParams::default()).unwrap()
    }
}
