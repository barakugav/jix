//! Internal block codec implementation: filter pipeline plus the codec (Zstd) used to compress and
//! decompress each array block. The user-facing documentation lives on
//! [`Compact`](crate::storage::Compact); configuration is exposed through
//! [`ArrayParams`](crate::ArrayParams).

mod filter;
pub use filter::*;

use std::cell::{Cell, RefCell, UnsafeCell};
use std::marker::PhantomData;
use std::ops::Range;
use std::sync::Arc;

use crate::buf_pool::{BufferPool, PoolBuf, PoolBufShared};
use crate::dtype::{Alignment, Dtype};
#[allow(unused_imports)]
use crate::error::{ensure, error, Result};
use crate::storage::cache_id::{ArrayCacheId, ArrayCacheIdMap, ArrayCacheIdSet};
use crate::storage::StridedBuf;
use crate::util::arrayvec::ArrayVec;
use crate::util::cpu_cache::CACHE_LINE_SIZE;
use crate::util::{AlignedBytes, AlternatingBuffers, DimArray, SliceExt};
use crate::DimDyn;

/// The compression algorithm applied to each block.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum Codec {
    /// [Zstandard](https://facebook.github.io/zstd/) - a fast general-purpose compressor.
    /// Compression level is controlled by [`ArrayParams::level`](crate::ArrayParams::level).
    Zstd,
}

/// Compression configuration used when encoding array blocks.
///
/// Controls the codec, compression level, and the pre-compression filter pipeline. Filters
/// are applied to the raw element bytes **before** compression and reversed **after**
/// decompression. For numeric data, filters significantly improve the compression ratio by
/// rearranging bytes or bits into a more compressible layout.
///
/// # Defaults
///
/// [`EncoderParams::default()`] uses Zstd level 3 with [`Filter::ByteShuffle`], which is a
/// good baseline for most numeric workloads: fast encoding, reasonable ratio, and effective
/// at exploiting the byte-level regularity of uniform-dtype arrays.
#[derive(Debug, Clone)]
pub(crate) struct EncoderParams {
    pub(crate) codec: Codec,
    pub(crate) level: i8,
    pub(crate) filters: ArrayVec<Filter, MAX_FILTERS>,
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
    /// Set the compression codec.
    pub fn codec(&mut self, codec: Codec) -> &mut Self {
        self.codec = codec;
        self
    }

    /// Set the compression level.
    ///
    /// Higher levels trade CPU time for better compression ratios. For zstd, level 3 is the default.
    ///
    /// # Errors
    ///
    /// Returns `InvalidArgument` if `level` is not in the valid range.
    pub fn level(&mut self, level: i32) -> Result<&mut Self> {
        self.level = level.try_into().map_err(|_| {
            error!(
                InvalidArgument,
                "compression level {level} is out of the supported range"
            )
        })?;
        Ok(self)
    }

    /// Set the pre-compression filter pipeline (up to 4 filters).
    ///
    /// Filters are applied in order before compression and reversed after decompression.
    /// For most numeric dtypes, [`Filter::ByteShuffle`] (the default) provides a good
    /// compression ratio improvement with low overhead. [`Filter::BitShuffle`] can yield
    /// better ratios for low-entropy data at the cost of higher CPU usage. Pass an empty
    /// slice to disable filtering entirely.
    ///
    /// Not all combinations of filters make sense; for example, a byte shuffle followed by a bit
    /// shuffle doesn't make sense because the bit shuffle will operate on the byte-shuffled data,
    /// and the bit shuffle filter will incorrectly assume the data is in the original byte order -
    /// it will not yield incorrect results, but it probably won't improve the compression ratio.
    ///
    /// # Errors
    ///
    /// Returns `InvalidArgument` if `filters` contains more than 4 elements.
    pub fn filters(&mut self, filters: &[Filter]) -> Result<&mut Self> {
        ensure!(
            filters.len() <= MAX_FILTERS,
            InvalidArgument,
            "At most {MAX_FILTERS} filters are supported"
        );
        self.filters = ArrayVec::from_slice(filters).unwrap();
        Ok(self)
    }
}

pub(crate) struct Encoder {
    pub(crate) dtype: Dtype,
    pub(crate) filters: ArrayVec<Filter, MAX_FILTERS>,
    pub(crate) compressor: Compressor,
    filters_tmp_buf1: AlignedBytes,
    filters_tmp_buf2: AlignedBytes,
    tmp_buffers: BufferPool,
}
pub(crate) enum Compressor {
    #[cfg(not(miri))]
    Zstd(zstd::bulk::Compressor<'static>),
    #[cfg(miri)]
    Zstd(()),
}
impl Encoder {
    pub(crate) fn new(params: &EncoderParams, dtype: Dtype) -> Result<Self> {
        let filters_tmp_buf1 = AlignedBytes::new_padded(dtype.alignment().as_usize());
        let filters_tmp_buf2 = filters_tmp_buf1.clone();
        Ok(Self {
            dtype,
            filters: params.filters.clone(),
            filters_tmp_buf1,
            filters_tmp_buf2,
            tmp_buffers: BufferPool::new(),
            compressor: match params.codec {
                Codec::Zstd => {
                    #[cfg(not(miri))]
                    let inner = zstd::bulk::Compressor::new(params.level as i32)
                        .map_err(|e| error!(CodecError, "Failed to create Zstd compressor: {e}"))?;
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

        let data = if self.filters.is_empty() {
            data
        } else {
            let tmp_buf1 = &mut self.filters_tmp_buf1;
            tmp_buf1.clear();
            tmp_buf1.reserve(data.len());
            unsafe { tmp_buf1.set_len(data.len()) };
            let tmp_buf1 = tmp_buf1.as_mut_slice();
            if self.filters.len() == 1 {
                self.filters[0].encode(data, tmp_buf1, &self.dtype, &self.tmp_buffers);
                tmp_buf1
            } else {
                let tmp_buf2 = &mut self.filters_tmp_buf2;
                tmp_buf2.clear();
                tmp_buf2.reserve(data.len());
                unsafe { tmp_buf2.set_len(data.len()) };
                let tmp_buf2 = tmp_buf2.as_mut_slice();
                let mut buffers = AlternatingBuffers::with_const_src(data, tmp_buf1, tmp_buf2);
                for filter in &self.filters {
                    let (data, buf) = buffers.edit();
                    filter.encode(data, buf, &self.dtype, &self.tmp_buffers);
                }
                buffers.into_data()
            }
        };

        match &mut self.compressor {
            Compressor::Zstd(compressor) => {
                #[cfg(not(miri))]
                let result = compressor
                    .compress_to_buffer(data, dst)
                    .map_err(|e| error!(CodecError, "Failed to compress data with Zstd: {e}"));
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
pub(crate) struct DecoderParams {
    _phantom: PhantomData<()>,
}

pub(crate) const MAX_FILTERS: usize = 4;

/// The codec configuration encoded alongside the array data, required to decode it.
///
/// Every field in `DecoderCodecConfig` is fixed at write time and must match exactly what was
/// used when the data was encoded - none of it can be chosen or overridden at read time.
///
/// This struct is populated from the stored array metadata and passed to [`Decoder`] internally.
/// Users do not construct it directly; it is derived from the array's on-disk representation
/// when a compressed array is read back from an archive.
#[derive(Debug, Clone)]
pub(crate) struct DecoderCodecConfig {
    pub(crate) codec: Codec,
    pub(crate) filters: ArrayVec<Filter, MAX_FILTERS>,
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
    #[inline]
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
        let mut buffers = None;
        let mut tmp_buf;
        let decompress_out = if self.filters.is_empty() {
            dst
        } else {
            tmp_buf = self
                .context
                .allocate_buf(dst.len(), Alignment::new(CACHE_LINE_SIZE).unwrap());
            let tmp_buf = tmp_buf.as_mut_slice();

            let dst_ptr = dst.as_mut_ptr();
            let dst_buf_odd = self.filters.len().is_multiple_of(2);
            let (buf1, buf2) = if dst_buf_odd {
                // we have 1 decompression + even number of filter, so we are going to write to a
                // total of odd number of buffers. For the last buffer to be dst, he also need to
                // be the first.
                // the 'secondary_buf' of AlternatingBuffers is the one returned as mut on the
                // first edit.
                (tmp_buf, dst)
            } else {
                (dst, tmp_buf)
            };

            buffers = Some(AlternatingBuffers::new(buf1, buf2));

            let (_, tmp_buf) = buffers.as_mut().unwrap().edit();
            debug_assert_eq!(dst_buf_odd, tmp_buf.as_mut_ptr() == dst_ptr);
            tmp_buf
        };

        #[cfg(not(miri))]
        let nbytes = {
            let inner = unsafe { &mut *self.inner.get() };
            inner
                .decompress_to_buffer(src, decompress_out)
                .map_err(|e| error!(CodecError, "Failed to decompress data with Zstd: {e}"))?
        };
        #[cfg(miri)]
        let nbytes = {
            decompress_out.copy_from_slice(src);
            src.len()
        };

        // Apply filters in reverse order
        for filter in self.filters.iter().rev() {
            let (data, buf) = buffers.as_mut().unwrap().edit();
            filter.decode(data, buf, self.dtype, &self.context.buffer_pool);
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
/// `ReadContext::default()` is also available if you need to create a context independently of a
/// specific array.
/// ```
/// use jix::ReadContext;
/// use jix::Array;
/// use ndarray::array;
///
/// let za = Array::compact_ndarray(&array![42i32, 17, 6, 99, 51])?;
/// let out = za.to_ndarray_sub(&[0..3], &ReadContext::default())?;
/// assert_eq!(out.as_slice().unwrap(), &[42, 17, 6]);
/// # Ok::<(), jix::Error>(())
/// ```
///
/// # Reusing a context
///
/// A single `ReadContext` can be passed to multiple successive reads. Reusing it avoids
/// reinitializing the decompressor and keeps the scratch buffer allocations warm:
pub struct ReadContext {
    #[cfg(not(miri))]
    decompressor: UnsafeCell<zstd::bulk::Decompressor<'static>>,
    buffer_pool: BufferPool,
    /// The last pull-mode read of each storage with an [`ArrayCacheId`] whose reads are cached, so
    /// a repeated read (e.g. of `x` in `x + x`) is computed only once. See
    /// [`read_cached`](Self::read_cached).
    read_cache: RefCell<ArrayCacheIdMap<CacheEntry>>,
    /// The storages whose reads are added to `read_cache` in the current scope. See
    /// [`top_level_read`](Self::top_level_read) and [`CacheScope`].
    arrays_to_cache: RefCell<Option<Arc<ArrayCacheIdSet>>>,
    /// Whether a top level read with storages to cache is in progress, so `read_cache` may be used.
    cache_active: Cell<bool>,
}

/// A cached read of the region `index` of a storage, laid out at `strides` from `offset` in
/// `data`.
struct CacheEntry {
    index: DimArray<Range<u64>>,
    strides: DimArray<usize>,
    offset: usize,
    data: Arc<AlignedBytes>,
}

/// The scope in which a storage with an [`ArrayCacheId`] reads its inner storages, which decides
/// the arrays to cache while it does. See [`ReadContext::read_cached`].
#[derive(Clone, Copy)]
pub(crate) enum CacheScope<'a> {
    /// The current scope: the storage reads its inner storages, if any, at the region it is read
    /// at (compact arrays, element-wise ops), so their reads are cached as reads outside of the
    /// storage are.
    Current,
    /// A scope of its own: the storage reads its inner storages at other regions than its own
    /// (reductions), which reads outside of it do not read. Only the arrays the storage reaches
    /// more than once itself (`arrays_to_cache`) are cached while it reads them, and their entries
    /// are dropped once the read is done.
    Own(&'a Option<Arc<ArrayCacheIdSet>>),
}

/// The read of a storage passed to [`ReadContext::read_cached`]: a closure taking the `out` of the
/// read. A trait object of it adds a single function to each storage, where a `dyn Fn` would add
/// the `call_once` and `call_mut` shims too.
pub(crate) trait CachedRead<'a, 'b: 'a> {
    fn read(&self, out: Option<&'a mut StridedBuf<'b>>) -> Result<StridedBuf<'a>>;
}
impl<'a, 'b: 'a, F> CachedRead<'a, 'b> for F
where
    F: Fn(Option<&'a mut StridedBuf<'b>>) -> Result<StridedBuf<'a>>,
{
    #[inline(always)]
    fn read(&self, out: Option<&'a mut StridedBuf<'b>>) -> Result<StridedBuf<'a>> {
        self(out)
    }
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
            read_cache: RefCell::new(ArrayCacheIdMap::default()),
            arrays_to_cache: RefCell::new(None),
            cache_active: Cell::new(false),
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

    /// Run `read`, a top level read of an array, with the reads of `arrays_to_cache` (the storages
    /// reachable more than once from the read array) going through the read cache.
    ///
    /// The read cache is cleared before and after `read`, so cached buffers do not outlive it, and
    /// the previous set is restored afterwards, so a nested top level read does not disturb the
    /// outer one.
    #[inline]
    pub(crate) fn top_level_read<R>(
        &self,
        arrays_to_cache: Option<Arc<ArrayCacheIdSet>>,
        read: impl FnOnce() -> R,
    ) -> R {
        // Nothing is cached outside of a read with a set, so without one the cache is empty.
        if arrays_to_cache.is_none() && !self.cache_active.get() {
            return read();
        }
        self.clear_read_cache();
        let prev_active = self.cache_active.replace(arrays_to_cache.is_some());
        let prev = self.arrays_to_cache.replace(arrays_to_cache);
        let result = read();
        self.clear_read_cache();
        self.arrays_to_cache.replace(prev);
        self.cache_active.set(prev_active);
        result
    }

    fn clear_read_cache(&self) {
        self.read_cache
            .borrow_mut()
            .clear_with(|entry| self.buffer_pool.return_shared(entry.data));
    }

    /// Read the region `index` of the storage `id` with `read` (called with `out`), through the
    /// read cache.
    ///
    /// A pull-mode read (`out` is `None`) of a region in the cache returns a read-only view of the
    /// cached buffer, whatever the arrays to cache. Otherwise `read` reads the region and, if `id` is
    /// one of the arrays to cache, its buffer is added to the cache: a buffer it allocated is
    /// shared with the cache, and replaces the cached read of another region of `id`, if any. See
    /// [`CacheScope`] for the arrays to cache while `read` reads the inner storages.
    ///
    /// The cached buffers are reference counted, so evicting an entry never invalidates views
    /// handed out earlier; an allocation returns to the pool when its last reference is dropped.
    #[inline(never)]
    pub(crate) fn read_cached<'a, 'b>(
        &'a self,
        id: ArrayCacheId,
        index: &[Range<u64>],
        out: Option<&'a mut StridedBuf<'b>>,
        scope: CacheScope<'_>,
        read: &dyn CachedRead<'a, 'b>,
    ) -> Result<StridedBuf<'a>> {
        // Not generic and not inlined, so it adds no code to the many storages that call it.
        if !self.cache_active.get() {
            return read.read(out);
        }
        let mut insert = false;
        if out.is_none() {
            let mut cache = self.read_cache.borrow_mut();
            if let Some(entry) = cache.get(id)
                && entry.index.as_slice() == index
            {
                let buf = PoolBufShared::new(entry.data.clone(), &self.buffer_pool);
                let buf = unsafe { StridedBuf::from_shared(buf, &entry.strides) };
                return Ok(unsafe { buf.with_offset(entry.offset) });
            }
            insert = self.is_cached(id);
            // Evict before reading, so the read can reuse the buffer.
            if insert && let Some(stale) = cache.remove(id) {
                self.buffer_pool.return_shared(stale.data);
            }
        }

        let prev = match scope {
            CacheScope::Current => None,
            CacheScope::Own(ids) => Some(self.arrays_to_cache.replace(ids.clone())),
        };
        let buf = read.read(out);
        if let Some(prev) = prev
            && let Some(ids) = self.arrays_to_cache.replace(prev)
        {
            let mut cache = self.read_cache.borrow_mut();
            for id in ids.iter() {
                if let Some(entry) = cache.remove(id) {
                    self.buffer_pool.return_shared(entry.data);
                }
            }
        }
        let buf = buf?;

        // A view of memory the read did not allocate, e.g. a cached buffer, is not cached.
        if !insert || !buf.owns_pool_buf() {
            return Ok(buf);
        }
        let (buf, offset, strides) = buf.into_pool_buf();
        let data = Arc::new(buf.into_inner());
        let buf = PoolBufShared::new(data.clone(), &self.buffer_pool);
        let buf = unsafe { StridedBuf::from_shared(buf, &strides).with_offset(offset) };
        let entry = CacheEntry {
            index: index.to_dim_vec::<DimDyn>(),
            strides,
            offset,
            data,
        };
        if let Some(stale) = self.read_cache.borrow_mut().insert(id, entry) {
            self.buffer_pool.return_shared(stale.data);
        }
        Ok(buf)
    }

    #[cfg(test)]
    pub(crate) fn read_cache_len(&self) -> usize {
        self.read_cache.borrow().keys().count()
    }

    /// Whether reads of the storage `id` are added to the read cache in the current scope.
    fn is_cached(&self, id: ArrayCacheId) -> bool {
        self.arrays_to_cache
            .borrow()
            .as_ref()
            .is_some_and(|ids| ids.contains(id))
    }
}

impl Default for ReadContext {
    fn default() -> Self {
        Self::new(&DecoderParams::default()).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ErrorKind;

    #[test]
    fn level_out_of_i8_range_errors() {
        for level in [128, 1000, i32::MAX, -129, -1000, i32::MIN] {
            let err = EncoderParams::default().level(level).unwrap_err();
            assert_eq!(err.kind(), ErrorKind::InvalidArgument);
            assert!(err.message().contains("compression level"), "{err}");
        }
    }

    #[test]
    fn level_in_i8_range_is_accepted() {
        for level in [i8::MIN, -5, 0, 3, 19, 22, i8::MAX] {
            let mut params = EncoderParams::default();
            params.level(level as i32).unwrap();
            assert_eq!(params.level, level);
        }
    }

    #[test]
    fn a_two_filter_pipeline_round_trips() {
        use crate::codec::filter::Filter;
        use crate::{Array, ArrayParams};

        let nd = ndarray::Array2::from_shape_fn((16, 16), |(i, j)| (i * 16 + j) as i32);
        let mut params = ArrayParams::new();
        params.block_shape(&[4, 4]);
        params
            .filters(&[Filter::ByteShuffle, Filter::BitShuffle])
            .unwrap();

        let array = Array::compact_ndarray_with(&nd, params).unwrap();
        assert_eq!(array.to_ndarray().unwrap(), nd);
    }
}
