use std::cell::{Cell, RefCell, UnsafeCell};
use std::sync::Arc;

use crate::buf_pool::{BufferPool, PoolBuf, PoolBufShared};
use crate::codec::{Decoder, DecoderCodecConfig, DecoderParams};
use crate::dtype::{Alignment, Dtype};
use crate::error::Result;
use crate::storage::cache_id::{ArrayCacheId, ArrayCacheIdMap, ArrayCacheIdSet};
use crate::storage::{check_out_buf_impl, Index, StridedBuf};
use crate::util::{DimArray, RawAlignedBytes, SliceExt};
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

    /// The read cache of the current cache scope, `None` if the scope has no arrays to cache.
    read_cache: RefCell<Option<Cache>>,
    /// equals `read_cache.is_some()`.
    /// So the reads in a scope without arrays to cache never touch the `RefCell`.
    is_cache_enabled: Cell<bool>,
    /// Reuseable cache maps for cache scopes.
    #[allow(clippy::vec_box)]
    temp_cache_maps: RefCell<Vec<Box<CacheMap>>>,
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

            read_cache: RefCell::new(None),
            is_cache_enabled: Cell::new(false),
            temp_cache_maps: RefCell::new(Vec::new()),
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

struct Cache {
    arrays_to_cache: Arc<ArrayCacheIdSet>,
    map: Box<CacheMap>,
}
type CacheMap = ArrayCacheIdMap<CacheEntry>;
struct CacheEntry {
    start: DimArray<u64>,
    size: DimArray<usize>,

    data: Arc<RawAlignedBytes>,
    offset: usize,
    strides: DimArray<usize>,
}

pub(crate) struct CacheScope {
    pub(crate) arrays_to_cache: Option<Arc<ArrayCacheIdSet>>,
}
impl CacheScope {
    #[inline]
    pub(crate) fn enter<'a>(&self, context: &'a ReadContext) -> CacheScopeGuard<'a> {
        context.cache_scope(&self.arrays_to_cache)
    }
}
pub(crate) struct CacheScopeGuard<'a> {
    context: &'a ReadContext,
    /// The read cache of the enclosing scope, hidden from the reads in the scope and restored when
    /// it ends. `None` if the scope changes nothing, as neither it nor the enclosing scope uses
    /// the read cache.
    prev: Option<Option<Cache>>,
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
    /// Pull the region `index` of the storage `id` with `read`, through the read cache if `id` is
    /// one of the arrays to cache of the current cache scope.
    #[inline(always)]
    pub(crate) fn read_cached<'a>(
        &'a self,
        id: ArrayCacheId,
        index: &Index<'_>,
        read: impl Fn() -> Result<StridedBuf<'a>>,
    ) -> Result<StridedBuf<'a>> {
        if !self.is_cache_enabled() {
            return read();
        }
        self.read_cached_impl(id, index, &read)
    }

    /// Push the region `index` of the storage `id` into `out` with `read_into`, or copy it from
    /// the read cache if the cache holds it.
    #[inline(always)]
    pub(crate) fn read_cached_into(
        &self,
        id: ArrayCacheId,
        index: &Index<'_>,
        dtype: &Dtype,
        out: &mut StridedBuf<'_>,
        read_into: impl FnOnce(&mut StridedBuf<'_>) -> Result<()>,
    ) -> Result<()> {
        if self.is_cache_enabled() && self.copy_cached(id, index, dtype, out)? {
            return Ok(());
        }
        read_into(out)
    }

    #[inline(never)]
    fn read_cached_impl<'a>(
        &'a self,
        id: ArrayCacheId,
        index: &Index<'_>,
        read_fn: &dyn CachedReadFn<'a>,
    ) -> Result<StridedBuf<'a>> {
        let mut cache_guard = self.read_cache.borrow_mut();
        // Check if cache is enabled for this array ID
        let cache = cache_guard
            .as_mut()
            .filter(|c| c.arrays_to_cache.contains(id));
        let Some(cache) = cache else {
            drop(cache_guard); // before calling unknown `read()`
            return read_fn.read();
        };
        // Search for a matching cache entry for this array ID and index
        if let Some(entry) = cache.map.get(id) {
            if entry.start.as_slice() == index.start() && entry.size.as_slice() == index.size() {
                let buf = PoolBufShared::new(entry.data.clone(), &self.buffer_pool);
                return Ok(unsafe {
                    StridedBuf::from_shared(buf, entry.strides.clone()).with_offset(entry.offset)
                });
            }
            // Evict before reading, so the read can reuse the buffer.
            if let Some(stale) = cache.map.remove(id) {
                self.buffer_pool.return_shared(stale.data);
            }
        }

        drop(cache_guard); // before calling unknown `read()`
        let buf = read_fn.read()?;
        let (data, offset, strides) = match buf.into_shared() {
            Ok(parts) => parts,
            Err(buf) => return Ok(buf), // A view of memory outside the pool is not cached.
        };
        let buf = PoolBufShared::new(data.clone(), &self.buffer_pool);
        let buf = unsafe { StridedBuf::from_shared(buf, strides.clone()).with_offset(offset) };
        let entry = CacheEntry {
            start: index.start().to_dim_vec::<DimDyn>(),
            size: index.size().to_dim_vec::<DimDyn>(),
            data,
            offset,
            strides,
        };
        let stale = self
            .read_cache
            .borrow_mut()
            .as_mut()
            .unwrap()
            .map
            .insert(id, entry);
        if let Some(stale) = stale {
            self.buffer_pool.return_shared(stale.data);
        }
        Ok(buf)
    }

    /// Copies the region `index` of the storage `id` into `out` if the read cache holds it, and
    /// returns whether it did. A push read is not added to the cache, as it cannot hold `out`.
    #[inline(never)]
    fn copy_cached(
        &self,
        id: ArrayCacheId,
        index: &Index<'_>,
        dtype: &Dtype,
        out: &mut StridedBuf<'_>,
    ) -> Result<bool> {
        let cache_guard = self.read_cache.borrow();
        let entry = cache_guard
            .as_ref()
            .filter(|c| c.arrays_to_cache.contains(id))
            .and_then(|c| c.map.get(id));
        let Some(entry) = entry else {
            return Ok(false);
        };
        if entry.start.as_slice() != index.start() || entry.size.as_slice() != index.size() {
            return Ok(false);
        }
        check_out_buf_impl(out, index, dtype.itemsize() as usize)?;
        let buf = PoolBufShared::new(entry.data.clone(), &self.buffer_pool);
        let buf = unsafe {
            StridedBuf::from_shared(buf, entry.strides.clone()).with_offset(entry.offset)
        };
        drop(cache_guard);
        let (src, src_strides) = buf.data();
        // SAFETY: the cached buffer holds the region at `src_strides`, and `out` was checked to
        // hold it at its own strides.
        unsafe { out.copy_from(src, src_strides, index.size(), dtype) };
        Ok(true)
    }

    #[inline]
    pub(crate) fn cache_scope(
        &self,
        arrays_to_cache: &Option<Arc<ArrayCacheIdSet>>,
    ) -> CacheScopeGuard<'_> {
        let prev = (self.is_cache_enabled() || arrays_to_cache.is_some())
            .then(|| self.enter_cache_scope(arrays_to_cache));
        CacheScopeGuard {
            context: self,
            prev,
        }
    }

    #[inline(never)]
    fn enter_cache_scope(&self, arrays_to_cache: &Option<Arc<ArrayCacheIdSet>>) -> Option<Cache> {
        let cache = arrays_to_cache.as_ref().map(|ids| {
            let map = self.temp_cache_maps.borrow_mut().pop().unwrap_or_default();
            debug_assert!(map.is_empty());
            Cache {
                arrays_to_cache: ids.clone(),
                map,
            }
        });
        self.is_cache_enabled.set(cache.is_some());
        self.read_cache.replace(cache)
    }

    #[inline(never)]
    fn exit_cache_scope(&self, prev: Option<Cache>) {
        self.is_cache_enabled.set(prev.is_some());
        let ended = self.read_cache.replace(prev);
        if let Some(mut ended) = ended {
            ended
                .map
                .clear_with(|entry| self.buffer_pool.return_shared(entry.data));
            self.temp_cache_maps.borrow_mut().push(ended.map);
        }
    }

    #[inline(always)]
    fn is_cache_enabled(&self) -> bool {
        let enabled = self.is_cache_enabled.get();
        debug_assert!(enabled == self.read_cache.borrow().is_some());
        enabled
    }
}

// could be replaced by `Fn`, but a trait generate less binary
trait CachedReadFn<'a> {
    fn read(&self) -> Result<StridedBuf<'a>>;
}
impl<'a, F> CachedReadFn<'a> for F
where
    F: Fn() -> Result<StridedBuf<'a>>,
{
    #[inline(always)]
    fn read(&self) -> Result<StridedBuf<'a>> {
        self()
    }
}

pub(crate) mod tests {
    #[allow(unused)]
    pub(crate) fn context_read_cache_len(context: &super::ReadContext) -> usize {
        match &*context.read_cache.borrow() {
            Some(cache) => cache.map.keys().count(),
            None => 0,
        }
    }
}
