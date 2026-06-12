//! Encoder-decoder configuration and implementation for the block-compressed storage backends.
//!
//! Each block of array data is encoded through a two-stage pipeline before being stored:
//!
//! ```text
//! raw block bytes
//!     |
//!     v
//! [ Filter 0 ] -> [ Filter 1 ] -> ...  (optional pre-compression transforms)
//!     |
//!     v
//! [ Codec (e.g. Zstd) ]                (lossless compression)
//!     |
//!     v
//! stored block bytes
//! ```
//!
//! Decoding reverses the pipeline exactly: decompress first, then apply the filters in reverse
//! order.
//!
//! # Configuration
//!
//! The pipeline is split across two separate configuration objects with different lifetimes:
//!
//! - **[`EncoderParams`]** - chosen by the user at write time. Selects the [`Codec`], compression
//!   level, and [`Filter`] pipeline. Stored alongside the data so that the correct decoder can be
//!   reconstructed later.
//!
//! - **[`DecoderParams`]** - chosen by the caller at read time. Currently carries no options, but
//!   is the extension point for future read-time tunables (thread count, cache budget, etc.).
//!
//! The codec, filters, and dtype that were used during encoding are derived from the stored array
//! metadata and passed to the internal decoder; users do not configure it directly.
//!
//! # Filters
//!
//! [`Filter`]s are byte-level transforms that rearrange element data into a layout that compresses
//! more efficiently, then reverse the transform after decompression. For most numeric workloads
//! [`Filter::ByteShuffle`] is the right default. [`Filter::BitShuffle`] can squeeze out more
//! compression for low-entropy data at higher CPU cost.
//!
//! # Read context
//!
//! [`ReadContext`] holds a long-lived decompressor instance and reusable scratch buffers. Create
//! one per thread and pass it to every read call to amortize initialization overhead across many
//! block reads. The preferred way to obtain one is [`Array::read_ctx()`](crate::Array::read_ctx).

mod filter;
pub use filter::*;

use std::cell::UnsafeCell;
use std::marker::PhantomData;

use crate::dtype::{Alignment, Dtype};
use crate::error::{ensure, Error, ErrorKind, Result};
use crate::util::arrayvec::ArrayVec;
use crate::util::cpu_cache::CACHE_LINE_SIZE;
use crate::util::{AlignedBytes, AlternatingBuffers};

/// The compression algorithm applied to each block.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum Codec {
    /// [Zstandard](https://facebook.github.io/zstd/) - a fast general-purpose compressor.
    /// Compression level is controlled by [`EncoderParams::level`].
    Zstd,
}

/// Compression configuration used when encoding array blocks.
///
/// Controls the codec, compression level, and the pre-compression filter pipeline. Filters
/// are applied to the raw element bytes **before** compression and reversed in **after**
/// decompression. For numeric data, filters significantly improve the compression ratio by
/// rearranging bytes or bits into a more compressible layout.
///
/// # Defaults
///
/// [`EncoderParams::default()`] uses Zstd level 3 with [`Filter::ByteShuffle`], which is a
/// good baseline for most numeric workloads: fast encoding, reasonable ratio, and effective
/// at exploiting the byte-level regularity of uniform-dtype arrays.
///
/// # Examples
///
/// Use the defaults (most common case):
///
/// ```
/// use jix::codec::EncoderParams;
/// use jix::{Array, ArrayParams};
/// use ndarray::array;
///
/// let data = array![1.0f32, 2.0, 3.0, 4.0];
/// let mut params = ArrayParams::new();
/// // EncoderParams::default() is equivalent to EncoderParams::new()
/// params.encoder_params(EncoderParams::new());
/// let za = Array::compact_ndarray_with(&data, params)?;
/// # Ok::<(), jix::Error>(())
/// ```
///
/// Increase compression level for archival data:
///
/// ```
/// use jix::codec::{EncoderParams, Filter};
/// use jix::{Array, ArrayParams};
/// use ndarray::array;
///
/// let data = array![1.0f64, 2.0, 3.0, 4.0];
/// let mut enc = EncoderParams::new();
/// enc.level(15)?; // slower encode, better ratio
/// enc.filters(&[Filter::ByteShuffle])?;
/// let mut params = ArrayParams::new();
/// params.encoder_params(enc);
/// let za = Array::compact_ndarray_with(&data, params)?;
/// # Ok::<(), jix::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct EncoderParams {
    pub(crate) codec: Codec,
    pub(crate) level: u8,
    pub(crate) filters: ArrayVec<Filter, 4>,
}
impl Default for EncoderParams {
    fn default() -> Self {
        Self {
            codec: Codec::Zstd,
            level: 3,
            filters: ArrayVec::from_slice([Filter::ByteShuffle].as_slice()).unwrap(),
        }
    }
}
impl EncoderParams {
    /// Create a new `EncoderParams` with the default configuration.
    ///
    /// The default configuration is Zstd level 3 with byte shuffle.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the compression codec.
    pub fn codec(&mut self, codec: Codec) -> &mut Self {
        self.codec = codec;
        self
    }

    /// Get the compression codec.
    pub fn codec_get(&self) -> &Codec {
        &self.codec
    }

    /// Set the compression level (0-19).
    ///
    /// Higher levels trade CPU time for better compression ratios. For zstd, level 3 is the default.
    ///
    /// # Errors
    ///
    /// Returns `InvalidArgument` if `level` is out of the valid range (0-19 for zstd).
    pub fn level(&mut self, level: u32) -> Result<&mut Self> {
        ensure!(
            level <= 19,
            InvalidArgument,
            "Codec level must be between 0 and 19"
        );
        self.level = level.try_into().unwrap();
        Ok(self)
    }

    /// Get the compression level.
    pub fn level_get(&self) -> u8 {
        self.level
    }

    /// Set the pre-compression filter pipeline (up to 4 filters).
    ///
    /// Filters are applied in order before compression and reversed after decompression.
    /// For most numeric dtypes, [`Filter::ByteShuffle`] (the default) provides a good
    /// compression ratio improvement with low overhead. [`Filter::BitShuffle`] can yield
    /// better ratios for low-entropy data at the cost of higher CPU usage. Pass an empty
    /// slice to disable filtering entirely.
    ///
    /// Not all combinations of filters make sense; for example,a byte shuffle followed by a bit
    /// shuffle doesn't make sense because the bit shuffle will operate of the byte-shuffled data,
    /// and the bti shuffle filter will incorrectly assume the data is in the original byte order -
    /// it will not yield incorrect results, but its probably won't improve the compression ratio.
    ///
    /// # Errors
    ///
    /// Returns `InvalidArgument` if `filters` contains more than 4 elements.
    pub fn filters(&mut self, filters: &[Filter]) -> Result<&mut Self> {
        ensure!(
            filters.len() <= 4,
            InvalidArgument,
            "At most 4 filters are supported"
        );
        self.filters = ArrayVec::from_slice(filters).unwrap();
        Ok(self)
    }

    /// Returns the filter pipeline.
    pub fn filters_get(&self) -> &[Filter] {
        &self.filters
    }
}

pub(crate) struct Encoder {
    pub(crate) dtype: Dtype,
    pub(crate) filters: ArrayVec<Filter, 4>,
    pub(crate) compressor: Compressor,
    tmp_buf1: AlignedBytes,
    tmp_buf2: AlignedBytes,
    tmp_buffers: TmpBufferPool,
}
pub(crate) enum Compressor {
    #[cfg(not(miri))]
    Zstd(zstd::bulk::Compressor<'static>),
    #[cfg(miri)]
    Zstd(()),
}
impl Encoder {
    pub(crate) fn new(params: &EncoderParams, dtype: Dtype) -> Result<Self> {
        let tmp_buf1 = AlignedBytes::new_padded(dtype.alignment().as_usize());
        let tmp_buf2 = tmp_buf1.clone();
        Ok(Self {
            dtype,
            filters: params.filters.clone(),
            tmp_buf1,
            tmp_buf2,
            tmp_buffers: TmpBufferPool::new(),
            compressor: match params.codec {
                Codec::Zstd => {
                    #[cfg(not(miri))]
                    let inner = zstd::bulk::Compressor::new(params.level as _).map_err(|e| {
                        Error::new(
                            ErrorKind::CodecError,
                            format!("Failed to create Zstd compressor: {e}"),
                        )
                    })?;
                    #[cfg(miri)]
                    let inner = ();
                    Compressor::Zstd(inner)
                }
            },
        })
    }

    pub(crate) fn encode(&mut self, data: &[u8], dst: &mut [u8]) -> Result<usize> {
        let itemsize = self.dtype.itemsize() as usize;
        ensure!(
            data.len().is_multiple_of(itemsize),
            InvalidArgument,
            "Data length is not a multiple of item size"
        );

        let mut buffers =
            AlternatingBuffers::with_const_src(data, &mut self.tmp_buf1, &mut self.tmp_buf2);
        for filter in &self.filters {
            let (data, buf) = buffers.edit();
            buf.clear();
            buf.reserve(data.len());
            unsafe { buf.set_len(data.len()) };
            filter.encode(data, buf, &self.dtype, &self.tmp_buffers);
        }
        let data = buffers.data();

        match &mut self.compressor {
            Compressor::Zstd(compressor) => {
                #[cfg(not(miri))]
                let result = compressor.compress_to_buffer(data, dst).map_err(|e| {
                    Error::new(
                        ErrorKind::CodecError,
                        format!("Failed to compress data with Zstd: {e}"),
                    )
                });
                #[cfg(miri)]
                let result = {
                    let _ = compressor;
                    dst.copy_from_slice(data);
                    Ok(data.len())
                };
                result
            }
        }
    }

    pub(crate) fn encode_bound(&self, src_size: usize) -> usize {
        match &self.compressor {
            Compressor::Zstd(_) => {
                #[cfg(not(miri))]
                {
                    zstd::zstd_safe::compress_bound(src_size)
                }
                #[cfg(miri)]
                {
                    src_size
                }
            }
        }
    }
}

/// Decoder configuration supplied by the caller at read time.
///
/// Unlike the codec type, filters, and dtype of a compressed block, which are determined by how the
/// data was written and cannot be changed, `DecoderParams` holds settings that the caller can
/// freely choose for each read session. These parameters apply regardless of the codec stored in
/// the data.
///
/// Currently no configuration options are exposed, but future versions may add settings such
/// as:
/// - Number of threads to use for parallel block decompression.
/// - Memory budget for the block cache.
///
/// Use [`DecoderParams::default()`] in the meantime. Pass an explicit instance to
/// [`ReadContext::new`] if you want forward-compatible control over these settings.
#[derive(Clone, Debug, Default)]
pub struct DecoderParams {
    _phantom: PhantomData<()>,
}

/// The codec configuration encoded alongside the array data, required to decode it.
///
/// Every field in `DecoderCodecConfig` is fixed at write time and must match exactly what was
/// used when the data was encoded - none of it can be chosen or overridden at read time.
///
/// This struct is populated from the stored array metadata and passed to [`Decoder`] internally.
/// Users do not construct it directly; it is derived from the array's on-disk representation
/// when a compressed array is read back from an archive.
#[derive(Clone, Debug)]
pub(crate) struct DecoderCodecConfig {
    pub(crate) codec: Codec,
    pub(crate) filters: ArrayVec<Filter, 4>,
    pub(crate) dtype: Dtype,
}

pub(crate) struct Decoder<'a> {
    #[cfg(not(miri))]
    inner: &'a UnsafeCell<zstd::bulk::Decompressor<'static>>,
    context: &'a ReadContext,

    dtype: &'a Dtype,
    filters: &'a [Filter],
}
impl<'a> Decoder<'a> {
    pub(crate) fn new(context: &'a ReadContext, config: &'a DecoderCodecConfig) -> Self {
        Self {
            #[cfg(not(miri))]
            inner: &context.decompressor,
            context,
            dtype: &config.dtype,
            filters: &config.filters,
        }
    }

    pub(crate) fn decode(&self, src: &[u8], dst: &mut [u8]) -> Result<usize> {
        let dst_len = dst.len();
        let dst_ptr = dst.as_mut_ptr();

        let tmp_buf1 = unsafe { &mut *self.context.tmp_buf1.get() };
        let tmp_buf2 = unsafe { &mut *self.context.tmp_buf2.get() };
        let mut buffers = AlternatingBuffers::new(tmp_buf1, tmp_buf2);
        let decompress_out = if self.filters.is_empty() {
            dst
        } else {
            let (_, tmp_buf) = buffers.edit();
            tmp_buf.clear();
            tmp_buf.reserve(dst.len());
            unsafe {
                #[allow(clippy::uninit_vec)]
                tmp_buf.set_len(dst.len())
            };
            tmp_buf.as_mut_slice()
        };

        #[cfg(not(miri))]
        let nbytes = {
            let inner = unsafe { &mut *self.inner.get() };
            inner
                .decompress_to_buffer(src, decompress_out)
                .map_err(|e| {
                    Error::new(
                        ErrorKind::CodecError,
                        format!("Failed to decompress data with Zstd: {e}"),
                    )
                })?
        };
        #[cfg(miri)]
        let nbytes = {
            decompress_out.copy_from_slice(src);
            src.len()
        };

        // Apply filters in reverse order
        for (f_idx, filter) in self.filters.iter().enumerate().rev() {
            let (data, buf) = buffers.edit();
            let buf = if f_idx > 0 {
                buf.clear();
                buf.reserve(data.len());
                unsafe {
                    #[allow(clippy::uninit_vec)]
                    buf.set_len(data.len())
                };
                buf.as_mut_slice()
            } else {
                unsafe { std::slice::from_raw_parts_mut(dst_ptr, dst_len) }
            };
            filter.decode(data, buf, self.dtype, &self.context.tmp_buffers);
        }

        Ok(nbytes)
    }
}

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
/// assert_eq!(out.as_slice().unwrap(), &[2, 3]);
/// # Ok::<(), jix::Error>(())
/// ```
///
/// `ReadContext::default()` (or `ReadContext::new(decoder_params)`) is available for if you need
/// more control over the decoder parameters or want to create a context independently of a specific
/// array.
/// ```
/// use jix::codec::ReadContext;
/// use jix::Array;
///
/// let za = Array::plain_scalar(42i32, &[5])?;
/// let out = za.to_ndarray_sub(&[0..3], &ReadContext::default())?;
/// assert_eq!(out.as_slice().unwrap(), &[42, 42, 42]);
/// # Ok::<(), jix::Error>(())
/// ```
///
/// # Reusing a context
///
/// A single `ReadContext` can be passed to multiple successive reads. Reusing it avoids
/// reinitializing the decompressor and keeps the scratch buffer allocations warm:
pub struct ReadContext {
    tmp_buffers: TmpBufferPool,
    /// Two alternating scratch buffers used by the filter pipeline (encode and decode paths).
    tmp_buf1: UnsafeCell<AlignedBytes>,
    tmp_buf2: UnsafeCell<AlignedBytes>,
    #[cfg(not(miri))]
    decompressor: UnsafeCell<zstd::bulk::Decompressor<'static>>,
}
impl ReadContext {
    /// Creates a new `ReadContext` configured with the given decoder parameters.
    ///
    /// Prefer [`Array::read_ctx()`](crate::Array::read_ctx) over calling this directly - it
    /// automatically uses the decoder parameters that match the array's stored codec configuration.
    pub fn new(#[allow(unused)] decoder_params: &DecoderParams) -> Result<Self> {
        let tmp_buf1 = AlignedBytes::new_exact(CACHE_LINE_SIZE);
        let tmp_buf2 = tmp_buf1.clone();
        Ok(Self {
            tmp_buffers: TmpBufferPool::new(),
            tmp_buf1: UnsafeCell::new(tmp_buf1),
            tmp_buf2: UnsafeCell::new(tmp_buf2),
            #[cfg(not(miri))]
            decompressor: UnsafeCell::new(zstd::bulk::Decompressor::new().unwrap()),
        })
    }

    pub(crate) fn decoder<'a>(&'a self, config: &'a DecoderCodecConfig) -> Decoder<'a> {
        Decoder::new(self, config)
    }

    #[inline]
    pub(crate) fn tmp_buf(&self, size: usize, alignment: Alignment) -> TmpBuf<'_> {
        self.tmp_buffers.get(size, alignment)
    }

    #[inline]
    pub(crate) fn tmp_buf_typed<T>(&self, nitems: usize) -> TmpBuf<'_> {
        self.tmp_buffers
            .get(nitems * size_of::<T>(), Alignment::of::<T>())
    }
}
impl Default for ReadContext {
    /// Creates a `ReadContext` with default [`DecoderParams`].
    ///
    /// Equivalent to `ReadContext::new(&DecoderParams::default()).unwrap()`.
    fn default() -> Self {
        Self::new(&DecoderParams::default()).unwrap()
    }
}

/// A pool of aligned bytes buffers.
///
/// Many components of a storage system (byte shuffle, bit shuffle, arithmetic lazy ops storage views, etc.)
/// need temporary working memory. Allocating fresh buffers on every block is expensive, so
/// `TmpBufferPool` keeps a small free list per alignment class and returns previously allocated
/// buffers when possible.
///
/// Buffers are vended as [`TmpBuf`] RAII guards. When a `TmpBuf` is dropped, its underlying
/// allocation is cleared and pushed back into the pool for reuse.
///
/// The pool is not thread-safe, it is intended to be owned and used by a single thread.
pub(crate) struct TmpBufferPool {
    /// Free list for alignments <= CACHE_LINE_SIZE; all buffers are allocated at CACHE_LINE_SIZE-byte alignment.
    align_standard: UnsafeCell<Vec<AlignedBytes>>,
    /// Free lists for alignments > CACHE_LINE_SIZE, sorted by alignment value.
    align_other: UnsafeCell<Vec<(Alignment, Vec<AlignedBytes>)>>,
}
impl TmpBufferPool {
    fn new() -> Self {
        Self {
            align_standard: UnsafeCell::new(Vec::new()),
            align_other: UnsafeCell::new(Vec::new()),
        }
    }

    /// Borrows a buffer of `size` bytes with at least `alignment` byte alignment.
    ///
    /// Returns a [`TmpBuf`] guard whose contents are initialized to `size` uninitialized bytes.
    /// The buffer is popped from the free list when one is available; otherwise a fresh allocation
    /// is made. The allocation is returned to the pool when the `TmpBuf` is dropped.
    fn get(&self, size: usize, alignment: Alignment) -> TmpBuf<'_> {
        let (pool, pool_align) = self.get_pool(alignment);
        let pool = unsafe { &mut *pool };
        let tmp_buf = pool
            .pop()
            .unwrap_or_else(|| AlignedBytes::with_capacity_exact(pool_align.as_usize(), size));
        let mut buf = TmpBuf {
            buf: tmp_buf,
            buffers: self,
        };
        buf.set_len(size);
        buf
    }

    /// Returns `buf` to the appropriate free list after clearing its length.
    fn return_buf(&self, mut buf: AlignedBytes) {
        buf.clear();
        let (pool, _) = self.get_pool(buf.alignment().try_into().unwrap());
        let pool = unsafe { &mut *pool };
        pool.push(buf);
    }

    /// Returns a raw pointer to the free list for `alignment` together with the actual alignment
    /// that will be used for allocations from that list.
    ///
    /// Alignments <= 16 are folded into the single `align16` list (allocated at 16 bytes).
    /// Larger alignments are looked up (or inserted) in the sorted `align_other` list.
    fn get_pool(&self, alignment: Alignment) -> (*mut Vec<AlignedBytes>, Alignment) {
        let standard_alignment = Alignment::new(CACHE_LINE_SIZE).unwrap();
        let alignment = alignment.max(standard_alignment);

        if alignment.as_usize() <= CACHE_LINE_SIZE {
            (self.align_standard.get(), standard_alignment)
        } else {
            let align_other = unsafe { &mut *self.align_other.get() };
            debug_assert!(align_other
                .iter()
                .zip(align_other.iter().skip(1))
                .all(|((align1, _pool1), (align2, _pool2))| align1 < align2));

            let (idx, exists) = align_other
                .iter()
                .map(|(align, _pool)| *align)
                .enumerate()
                .find(|(_idx, align)| *align >= alignment)
                .map(|(idx, align)| (idx, align == alignment))
                .unwrap_or((align_other.len(), false));

            if !exists {
                align_other.insert(idx, (alignment, Vec::new()));
            }
            let pool = &mut align_other[idx].1;
            (pool, alignment)
        }
    }
}

/// An RAII guard for a temporary scratch buffer borrowed from a [`TmpBufferPool`].
///
/// Obtained via [`TmpBufferPool::get`] (or [`ReadContext::tmp_buf`]). The buffer is
/// pre-sized to the requested length on creation. When `TmpBuf` is dropped, the underlying
/// allocation is cleared and returned to the pool for reuse.
pub(crate) struct TmpBuf<'a> {
    buf: AlignedBytes,
    buffers: &'a TmpBufferPool,
}
impl TmpBuf<'_> {
    /// Resizes the buffer to `new_len` bytes. The new contents are uninitialized.
    #[inline]
    pub(crate) fn set_len(&mut self, new_len: usize) {
        self.buf.clear();
        self.buf.reserve(new_len);
        unsafe { self.buf.set_len(new_len) };
    }

    #[inline(always)]
    pub(crate) fn as_slice(&self) -> &[u8] {
        self.buf.as_slice()
    }

    #[inline(always)]
    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
        self.buf.as_mut_slice()
    }
}
impl Drop for TmpBuf<'_> {
    fn drop(&mut self) {
        // Swap out self.buf so we can pass ownership to return_buf.
        let mut buf = AlignedBytes::new_exact(self.buf.alignment());
        std::mem::swap(&mut self.buf, &mut buf);

        self.buffers.return_buf(buf);
    }
}
