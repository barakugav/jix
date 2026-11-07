use std::cell::{RefCell, UnsafeCell};
use std::ops::Range;
use std::sync::Arc;

use crate::buf_pool::{BufferPool, PoolBuf, PoolBufShared};
use crate::codec::{Decoder, DecoderCodecConfig, DecoderParams};
use crate::dtype::{Alignment, Dtype};
use crate::error::Result;
use crate::storage::cache_id::{ArrayCacheId, ArrayCacheIdMap, ArrayCacheIdSet};
use crate::storage::StridedBuf;
use crate::util::{dim_arr, DimArray, RawAlignedBytes, SliceExt};
use crate::DimDyn;

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

    /// The last pull-mode read of each of the arrays to cache of the current cache scope, so a
    /// repeated read (e.g. of `x` in `x + x`) is computed only once. See
    /// [`read_cached`](Self::read_cached). Boxed, so a cache scope moves it aside by pointer.
    read_cache: RefCell<Box<ArrayCacheIdMap<CacheEntry>>>,
    /// The storages whose reads are cached in the current cache scope, see
    /// [`cache_scope`](Self::cache_scope). Only accessed by [`is_caching`](Self::is_caching),
    /// [`is_cached`](Self::is_cached) and
    /// [`replace_arrays_to_cache`](Self::replace_arrays_to_cache).
    arrays_to_cache: UnsafeCell<Option<Arc<ArrayCacheIdSet>>>,
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
            read_cache: RefCell::new(Box::default()),
            arrays_to_cache: UnsafeCell::new(None),
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

/// A cached read of the region `index` of a storage, laid out at `strides` from `offset` in
/// `data`.
struct CacheEntry {
    index: DimArray<Range<u64>>,
    strides: DimArray<usize>,
    offset: usize,
    data: Arc<RawAlignedBytes>,
}

/// A cache scope, see [`ReadContext::cache_scope`]. The scope ends when it is dropped.
pub(crate) struct CacheScopeGuard<'a> {
    context: &'a ReadContext,
    /// The state of the read cache before the scope, restored when it ends. `None` if the scope
    /// changes nothing, as the read cache is not used in it.
    prev: Option<CacheState>,
}

/// The state of the read cache of the scope enclosing a cache scope, see
/// [`ReadContext::cache_scope`].
struct CacheState {
    arrays_to_cache: Option<Arc<ArrayCacheIdSet>>,
    /// The entries, hidden from the reads in the scope. `None` if the scope does not use the read
    /// cache or there are none.
    entries: Option<Box<ArrayCacheIdMap<CacheEntry>>>,
}
impl Drop for CacheScopeGuard<'_> {
    #[inline]
    fn drop(&mut self) {
        if let Some(prev) = self.prev.take() {
            self.context.exit_cache_scope(prev);
        }
    }
}

impl ReadContext {
    /// Read the region `index` of the storage `id` with `read` (called with `out`), through the
    /// read cache if `id` is one of the arrays to cache of the current cache scope.
    ///
    /// A read of the cached region of `id` is served from the cache: a pull-mode read (`out` is
    /// `None`) returns a read-only view of the cached buffer, and a push-mode read copies the
    /// cached region into `out`. Otherwise `read` reads the region and, for a pull-mode read, its
    /// buffer is added to the cache: a buffer it allocated is shared with the cache, and replaces
    /// the cached read of another region of `id`, if any. A push-mode read is never added, as it
    /// writes to the caller's buffer.
    ///
    /// The cached buffers are reference counted, so evicting an entry never invalidates views
    /// handed out earlier; an allocation returns to the pool when its last reference is dropped.
    #[inline(always)]
    pub(crate) fn read_cached<'a, 'b>(
        &'a self,
        id: ArrayCacheId,
        index: &[Range<u64>],
        dtype: &Dtype,
        out: Option<&'a mut StridedBuf<'b>>,
        read: impl Fn(Option<&'a mut StridedBuf<'b>>) -> Result<StridedBuf<'a>>,
    ) -> Result<StridedBuf<'a>> {
        if !self.is_caching() {
            return read(out);
        }
        self.read_cached_impl(id, index, dtype, out, &read)
    }

    #[inline(never)]
    pub(crate) fn read_cached_impl<'a, 'b>(
        &'a self,
        id: ArrayCacheId,
        index: &[Range<u64>],
        dtype: &Dtype,
        out: Option<&'a mut StridedBuf<'b>>,
        read: &dyn Fn(Option<&'a mut StridedBuf<'b>>) -> Result<StridedBuf<'a>>,
    ) -> Result<StridedBuf<'a>> {
        // Not generic and not inlined, so it adds no code to the many storages that call it.
        if !self.is_cached(id) {
            return read(out);
        }
        let mut cache = self.read_cache.borrow_mut();
        if let Some(entry) = cache.get(id)
            && entry.index.as_slice() == index
        {
            let buf = PoolBufShared::new(entry.data.clone(), &self.buffer_pool);
            let buf =
                unsafe { StridedBuf::from_shared(buf, &entry.strides).with_offset(entry.offset) };
            // Not borrowed while copying, so a cache scope that ends on a panic can borrow it.
            drop(cache);
            let Some(out) = out else {
                return Ok(buf);
            };
            let shape = dim_arr(index.len(), |d| (index[d].end - index[d].start) as usize);
            let (src, src_strides) = buf.data();
            // SAFETY: the cached buffer holds the region at `src_strides`, and `out` was checked to
            // hold it at its own strides.
            unsafe { out.copy_from(src, src_strides, &shape, dtype) };
            return Ok(out.view_mut());
        }
        let pull = out.is_none();
        // Evict before reading, so the read can reuse the buffer.
        if pull && let Some(stale) = cache.remove(id) {
            self.buffer_pool.return_shared(stale.data);
        }
        drop(cache);

        let buf = read(out)?;

        // A view of memory the read did not allocate, e.g. a cached buffer, is not cached.
        if !pull || !buf.owns_pool_buf() {
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

    /// Enter the cache scope of a read of a storage at some region, in which the reads of
    /// `arrays_to_cache`, the storages it reaches more than once at that region, are cached.
    ///
    /// A top level read of an array is the scope of the read array, and a storage that reads its
    /// inner storages at other regions than the one it is read at (shape ops other than
    /// broadcasts, reductions) reads them in a scope of its own (see
    /// [`ArraySpecDynamic`](crate::storage::params::ArraySpecDynamic)).
    ///
    /// A scope is isolated from the enclosing ones: the read cache is used in it only if it has
    /// arrays to cache, and its reads see only its own entries. The scope ends when the returned
    /// guard is dropped, which drops its entries and restores the read cache of the enclosing
    /// scope.
    #[inline]
    pub(crate) fn cache_scope(
        &self,
        arrays_to_cache: &Option<Arc<ArrayCacheIdSet>>,
    ) -> CacheScopeGuard<'_> {
        let prev = (self.is_caching() || arrays_to_cache.is_some())
            .then(|| self.enter_cache_scope(arrays_to_cache));
        CacheScopeGuard {
            context: self,
            prev,
        }
    }

    #[inline(never)]
    fn enter_cache_scope(&self, arrays_to_cache: &Option<Arc<ArrayCacheIdSet>>) -> CacheState {
        // Without arrays to cache the read cache is not used, so its entries need no hiding.
        let mut entries = None;
        if arrays_to_cache.is_some() {
            let mut cache = self.read_cache.borrow_mut();
            if !cache.is_empty() {
                entries = Some(std::mem::take(&mut *cache));
            }
        }
        CacheState {
            arrays_to_cache: self.replace_arrays_to_cache(arrays_to_cache.clone()),
            entries,
        }
    }

    #[inline(never)]
    fn exit_cache_scope(&self, prev: CacheState) {
        let used_cache = self.replace_arrays_to_cache(prev.arrays_to_cache).is_some();
        if used_cache {
            let mut cache = self.read_cache.borrow_mut();
            cache.clear_with(|entry| self.buffer_pool.return_shared(entry.data));
            if let Some(entries) = prev.entries {
                *cache = entries;
            }
        }
    }

    // SAFETY of the accesses to `arrays_to_cache`: `ReadContext` is not `Sync`, and none of these
    // methods keeps a reference to it past its body or runs code that could access it meanwhile.

    /// Whether the current cache scope has arrays to cache.
    #[inline(always)]
    fn is_caching(&self) -> bool {
        unsafe { (*self.arrays_to_cache.get()).is_some() }
    }

    /// Whether the reads of the storage `id` are cached in the current cache scope.
    fn is_cached(&self, id: ArrayCacheId) -> bool {
        let ids = unsafe { &*self.arrays_to_cache.get() };
        ids.as_ref().is_some_and(|ids| ids.contains(id))
    }

    fn replace_arrays_to_cache(
        &self,
        ids: Option<Arc<ArrayCacheIdSet>>,
    ) -> Option<Arc<ArrayCacheIdSet>> {
        unsafe { std::mem::replace(&mut *self.arrays_to_cache.get(), ids) }
    }
}

pub(crate) mod tests {
    #[allow(unused)]
    pub(crate) fn context_read_cache_len(context: &super::ReadContext) -> usize {
        context.read_cache.borrow().keys().count()
    }
}
