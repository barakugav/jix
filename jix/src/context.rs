#[cfg(not(miri))]
use std::cell::UnsafeCell;
use std::cell::{Cell, RefCell};
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

    /// The last pull-mode read of each storage with an [`ArrayCacheId`] whose reads are cached, so
    /// a repeated read (e.g. of `x` in `x + x`) is computed only once. See
    /// [`read_cached`](Self::read_cached).
    read_cache: RefCell<ArrayCacheIdMap<CacheEntry>>,
    /// The storages whose reads are added to `read_cache` in the current cache scope. See
    /// [`cache_scope`](Self::cache_scope).
    arrays_to_cache: RefCell<Option<Arc<ArrayCacheIdSet>>>,
    /// Whether reads use `read_cache`: whether the current cache scope or an enclosing one has
    /// arrays to cache.
    cache_active: Cell<bool>,
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
#[derive(Clone)]
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

/// The state of the read cache that a cache scope changes, see [`ReadContext::cache_scope`].
struct CacheState {
    active: bool,
    arrays_to_cache: Option<Arc<ArrayCacheIdSet>>,
    /// The entries of the arrays to cache of the scope, which the reads in the scope replace.
    entries: Vec<(ArrayCacheId, CacheEntry)>,
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
    /// read cache.
    ///
    /// A read of a region in the cache is served from the cache, whatever the arrays to cache: a
    /// pull-mode read (`out` is `None`) returns a read-only view of the cached buffer, and a
    /// push-mode read copies the cached region into `out`. Otherwise `read` reads the region and,
    /// for a pull-mode read of one of the arrays to cache, its buffer is added to the cache: a
    /// buffer it allocated is shared with the cache, and replaces the cached read of another region
    /// of `id`, if any. A push-mode read is never added, as it writes to the caller's buffer.
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
        if !self.cache_active.get() {
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
        let mut insert = false;
        if out.is_none() {
            insert = self.is_cached(id);
            // Evict before reading, so the read can reuse the buffer.
            if insert && let Some(stale) = cache.remove(id) {
                self.buffer_pool.return_shared(stale.data);
            }
        }
        drop(cache);

        let buf = read(out)?;

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

    /// Enter the cache scope of a read of a storage at some region, in which the reads of
    /// `arrays_to_cache`, the storages it reaches more than once at that region, are cached.
    ///
    /// A top level read of an array is the scope of the read array, and a storage that reads its
    /// inner storages at other regions than the one it is read at (shape ops other than
    /// broadcasts, reductions) reads them in a scope of its own (see
    /// [`ArraySpecDynamic`](crate::storage::params::ArraySpecDynamic)).
    ///
    /// The read cache is used in a scope with arrays to cache and in the scopes it encloses, so
    /// the reads in a scope without arrays to cache within it still hit the cached entries.
    ///
    /// The scope ends when the returned guard is dropped, which restores the read cache to its
    /// state before the scope: the entries of the reads in the scope are dropped, the entries
    /// they replaced are restored, and the arrays to cache of the enclosing scope are restored.
    #[inline]
    pub(crate) fn cache_scope(
        &self,
        arrays_to_cache: &Option<Arc<ArrayCacheIdSet>>,
    ) -> CacheScopeGuard<'_> {
        let prev = (self.cache_active.get() || arrays_to_cache.is_some())
            .then(|| self.enter_cache_scope(arrays_to_cache));
        CacheScopeGuard {
            context: self,
            prev,
        }
    }

    #[inline(never)]
    fn enter_cache_scope(&self, arrays_to_cache: &Option<Arc<ArrayCacheIdSet>>) -> CacheState {
        // The entries the reads in the scope may replace are saved, but stay in the cache for the
        // reads in the scope to hit.
        let mut entries = Vec::new();
        let cache = self.read_cache.borrow();
        if let Some(ids) = arrays_to_cache
            && !cache.is_empty()
        {
            ids.iter().for_each(|id| {
                if let Some(entry) = cache.get(id) {
                    entries.push((id, entry.clone()));
                }
            });
        }
        CacheState {
            active: self.cache_active.replace(true),
            arrays_to_cache: self.arrays_to_cache.replace(arrays_to_cache.clone()),
            entries,
        }
    }

    #[inline(never)]
    fn exit_cache_scope(&self, prev: CacheState) {
        self.cache_active.set(prev.active);
        let ids = self.arrays_to_cache.replace(prev.arrays_to_cache);
        // Only the arrays to cache of the scope are added to the cache in it, enclosed scopes
        // restore their own.
        let mut cache = self.read_cache.borrow_mut();
        if let Some(ids) = ids {
            ids.iter().for_each(|id| {
                if let Some(entry) = cache.remove(id) {
                    self.buffer_pool.return_shared(entry.data);
                }
            });
        }
        for (id, entry) in prev.entries {
            cache.insert(id, entry);
        }
    }

    /// Whether reads of the storage `id` are added to the read cache in the current scope.
    fn is_cached(&self, id: ArrayCacheId) -> bool {
        self.arrays_to_cache
            .borrow()
            .as_ref()
            .is_some_and(|ids| ids.contains(id))
    }
}

pub(crate) mod tests {
    #[allow(unused)]
    pub(crate) fn context_read_cache_len(context: &super::ReadContext) -> usize {
        context.read_cache.borrow().keys().count()
    }
}
