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
use crate::storage::params::ArraySpecDynamic;
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
    /// The storages whose reads are added to `read_cache` in the current scope. See
    /// [`top_level_read`](Self::top_level_read) and [`cache_scope`](Self::cache_scope).
    arrays_to_cache: RefCell<Option<Arc<ArrayCacheIdSet>>>,
    /// Whether a top level read that uses `read_cache` is in progress.
    cache_active: Cell<bool>,
}

/// A cached read of the region `index` of a storage, laid out at `strides` from `offset` in
/// `data`.
struct CacheEntry {
    index: DimArray<Range<u64>>,
    strides: DimArray<usize>,
    offset: usize,
    data: Arc<RawAlignedBytes>,
}

/// A cache scope of a storage that reads its inner storages at other regions than its own, see
/// [`ReadContext::cache_scope`]. The scope ends when it is dropped.
pub(crate) struct CacheScopeGuard<'a> {
    context: &'a ReadContext,
    /// The arrays to cache of the enclosing scope, restored when the scope ends. `None` if no
    /// scope was entered, as the read cache is not used.
    prev: Option<Option<Arc<ArrayCacheIdSet>>>,
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

    /// Run `read`, a top level read of an array of spec `spec`, with the reads of its
    /// [`arrays_to_cache`](ArraySpecDynamic::arrays_to_cache) (the storages reachable more than
    /// once from the read array) going through the read cache.
    ///
    /// The read cache is cleared before and after `read`, so cached buffers do not outlive it, and
    /// the previous set is restored afterwards, so a nested top level read does not disturb the
    /// outer one.
    #[inline]
    pub(crate) fn top_level_read<R>(&self, spec: &ArraySpecDynamic, read: impl FnOnce() -> R) -> R {
        // Nothing is cached outside of a read that uses the cache, so without one it is empty.
        let active = spec.uses_read_cache();
        if !active && !self.cache_active.get() {
            return read();
        }
        self.clear_read_cache();
        let prev_active = self.cache_active.replace(active);
        let prev = self.arrays_to_cache.replace(spec.arrays_to_cache.clone());
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

    /// Enter a cache scope of a storage that reads its inner storages at other regions than the
    /// one it is read at (shape ops other than broadcasts, reductions), which reads outside of it
    /// do not read: only `arrays_to_cache`, the arrays its inner storages reach more than once (see
    /// [`ArraySpecDynamic::with_cache_scope`]), are cached while it reads them.
    ///
    /// The scope ends when the returned guard is dropped, which restores the arrays to cache of
    /// the enclosing scope and drops the entries of `arrays_to_cache`, so the reads in the scope
    /// do not evict entries of the enclosing scope for long.
    #[inline]
    pub(crate) fn cache_scope(
        &self,
        arrays_to_cache: &Option<Arc<ArrayCacheIdSet>>,
    ) -> CacheScopeGuard<'_> {
        let prev = self
            .cache_active
            .get()
            .then(|| self.enter_cache_scope(arrays_to_cache));
        CacheScopeGuard {
            context: self,
            prev,
        }
    }

    #[inline(never)]
    fn enter_cache_scope(
        &self,
        arrays_to_cache: &Option<Arc<ArrayCacheIdSet>>,
    ) -> Option<Arc<ArrayCacheIdSet>> {
        self.arrays_to_cache.replace(arrays_to_cache.clone())
    }

    #[inline(never)]
    fn exit_cache_scope(&self, prev: Option<Arc<ArrayCacheIdSet>>) {
        if let Some(ids) = self.arrays_to_cache.replace(prev) {
            let mut cache = self.read_cache.borrow_mut();
            for id in ids.iter() {
                if let Some(entry) = cache.remove(id) {
                    self.buffer_pool.return_shared(entry.data);
                }
            }
        }
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
const _: fn() = || {
    fn assert_send<T: Send>() {}
    assert_send::<ReadContext>();
};

impl Default for ReadContext {
    fn default() -> Self {
        Self::new(&DecoderParams::default()).unwrap()
    }
}
