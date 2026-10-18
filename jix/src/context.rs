#[cfg(not(miri))]
use std::cell::UnsafeCell;
use std::cell::{Cell, RefCell};
use std::ops::Range;
use std::sync::Arc;

use crate::buf_pool::{BufferPool, PoolBuf, PoolBufShared};
use crate::codec::{Decoder, DecoderCodecConfig, DecoderParams};
use crate::dtype::Alignment;
use crate::error::Result;
use crate::storage::cache_id::{ArrayCacheId, ArrayCacheIdMap, ArrayCacheIdSet};
use crate::storage::StridedBuf;
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
    data: Arc<RawAlignedBytes>,
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
const _: fn() = || {
    fn assert_send<T: Send>() {}
    assert_send::<ReadContext>();
};

impl Default for ReadContext {
    fn default() -> Self {
        Self::new(&DecoderParams::default()).unwrap()
    }
}
