use std::cell::{Cell, RefCell, UnsafeCell};
use std::sync::Arc;

use crate::DimDyn;
use crate::codec::{Decoder, DecoderCodecConfig};
use crate::dtype::{Alignment, Dtype};
use crate::error::Result;
use crate::storage::cache_id::{ArrayCacheId, ArrayCacheIdMap, ArrayCacheIdSet};
use crate::storage::params::{ReadParams, ReadSize};
use crate::storage::{Index, OutBuf, OutBufInner, StridedBuf, write_out_buf_slot};
use crate::util::buf_pool::{BufferPool, PoolBuf};
use crate::util::{DimArray, SliceExt};

/// A context with reusable buffers and a long-lived decoder instance.
///
/// Pass one `ReadContext` to many successive reads: it saves allocating scratch buffers and
/// initializing a decoder on every block read, a cost that matters most for small blocks.
///
/// `ReadContext` is `!Sync`: for concurrent reads, create one per thread.
///
/// # Obtaining a `ReadContext`
///
/// The preferred way is [`Array::read_ctx()`](crate::Array::read_ctx), which picks up the read
/// parameters of the array, such as its read size (see [`ReadParams`]):
///
/// ```
/// use jix::Array;
/// use ndarray::array;
///
/// let za = Array::compact_ndarray(&array![1i32, 2, 3, 4])?;
///
/// // read_ctx() takes the read size of the array.
/// let ctx = za.read_ctx();
/// let out = za.to_ndarray_sub(&[1..3], &ctx)?;
/// assert_eq!(out, array![2, 3]);
/// # Ok::<(), jix::Error>(())
/// ```
///
/// `ReadContext::default()` creates a context independent of any array:
/// ```
/// use jix::{Array, ReadContext};
/// use ndarray::array;
///
/// let za = Array::compact_ndarray(&array![42i32, 17, 6, 99, 51])?;
/// let out = za.to_ndarray_sub(&[0..3], &ReadContext::default())?;
/// assert_eq!(out, array![42, 17, 6]);
/// # Ok::<(), jix::Error>(())
/// ```
pub struct ReadContext {
    pub(crate) decompressor: UnsafeCell<Option<zstd::bulk::Decompressor<'static>>>,
    pub(crate) buffer_pool: BufferPool,

    /// The read cache of the current cache scope, `None` if the scope has no arrays to cache.
    read_cache: RefCell<Option<Cache>>,
    /// Equals `read_cache.is_some()`, so the reads in a scope without arrays to cache never touch
    /// the `RefCell`.
    is_cache_enabled: Cell<bool>,
    /// Reusable cache maps for cache scopes.
    #[allow(clippy::vec_box)]
    temp_cache_maps: RefCell<Vec<Box<CacheMap>>>,

    params: ReadParams,
}

impl ReadContext {
    /// Creates a new `ReadContext` configured with the given read parameters.
    ///
    /// Prefer [`Array::read_ctx()`](crate::Array::read_ctx) over calling this directly - it
    /// uses the read parameters of the array.
    pub(crate) fn new(params: ReadParams) -> Result<Self> {
        Ok(Self {
            decompressor: UnsafeCell::new(None),
            buffer_pool: BufferPool::new(),

            read_cache: RefCell::new(None),
            is_cache_enabled: Cell::new(false),
            temp_cache_maps: RefCell::new(Vec::new()),

            params,
        })
    }

    /// The read parameters of the reads using this context, e.g. the read size.
    ///
    /// [`Array::read_ctx()`](crate::Array::read_ctx) starts from the array's
    /// [`read_params`](crate::Array::read_params), [`ReadContext::default()`] from the default
    /// ones.
    #[inline]
    pub fn read_params(&self) -> &ReadParams {
        &self.params
    }

    /// Mutable access to the read parameters, see [`read_params`](Self::read_params).
    #[inline]
    pub fn read_params_mut(&mut self) -> &mut ReadParams {
        &mut self.params
    }

    #[inline]
    pub(crate) fn read_size(&self) -> ReadSize {
        self.params.read_size
    }

    #[inline]
    pub(crate) fn decoder<'a>(&'a self, config: &'a DecoderCodecConfig) -> Decoder<'a> {
        Decoder::new(self, config)
    }

    #[inline]
    pub(crate) fn allocate_buf(&self, size: usize, alignment: Alignment) -> PoolBuf {
        self.buffer_pool.allocate(size, alignment)
    }

    #[inline]
    pub(crate) fn return_buf(&self, buf: PoolBuf) {
        self.buffer_pool.return_buf(buf);
    }

    #[inline]
    pub(crate) fn return_buf_strided(&self, buf: StridedBuf<'_>) {
        buf.return_to(&self.buffer_pool);
    }
}
const _: fn() = || {
    fn assert_send<T: Send>() {}
    assert_send::<ReadContext>();
};

impl Default for ReadContext {
    fn default() -> Self {
        Self::new(ReadParams::default()).unwrap()
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

    data: Arc<PoolBuf>,
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
    /// Read the region `index` of the storage `id` into `out` with `read`, through the read cache
    /// if `id` is one of the arrays to cache of the current cache scope.
    #[inline(always)]
    pub(crate) fn read_cached<'a>(
        &self,
        id: ArrayCacheId,
        index: &Index<'_>,
        dtype: &Dtype,
        out: OutBuf<'_, 'a>,
        read: impl Fn(OutBuf<'_, 'a>) -> Result<()>,
    ) -> Result<()> {
        if !self.is_cache_enabled() {
            return read(out);
        }
        self.read_cached_impl(id, index, dtype, out, &read)
    }

    #[inline(never)]
    fn read_cached_impl<'a>(
        &self,
        id: ArrayCacheId,
        index: &Index<'_>,
        dtype: &Dtype,
        out: OutBuf<'_, 'a>,
        read_fn: &dyn CachedReadFn<'a>,
    ) -> Result<()> {
        let mut cache_guard = self.read_cache.borrow_mut();
        // Check if cache is enabled for this array ID
        let cache = cache_guard
            .as_mut()
            .filter(|c| c.arrays_to_cache.contains(id));
        let Some(cache) = cache else {
            drop(cache_guard); // before calling unknown `read()`
            return read_fn.read(out);
        };
        // Search for a matching cache entry for this array ID and index
        if let Some(entry) = cache.map.get(id) {
            if entry.start.as_slice() == index.start() && entry.size.as_slice() == index.size() {
                let buf = entry.data.clone();
                let buf = unsafe {
                    StridedBuf::from_shared(buf, entry.strides.clone()).with_offset(entry.offset)
                };
                drop(cache_guard);
                // return the cached buffer itself if this is a pull read, or a copy in push mode.
                return out.set(buf, index, dtype, self);
            } else {
                // Evict before reading, so the read can reuse the buffer.
                if out.is_pull()
                    && let Some(stale) = cache.map.remove(id)
                {
                    self.buffer_pool.return_shared(stale.data);
                }
            }
        }

        drop(cache_guard); // before calling unknown `read()`
        let slot = match out.0 {
            // user provided an output buffer, we can't hold a ref to it in cache
            OutBufInner::Push(_) => return read_fn.read(out),
            OutBufInner::Pull(slot) => slot,
        };
        read_fn.read(OutBuf::pull(slot))?;
        let buf = slot
            .take()
            .expect("a pull read puts its buffer in the slot");
        let (data, offset, strides) = match buf.into_shared() {
            Ok(parts) => parts,
            Err(buf) => {
                write_out_buf_slot(slot, buf); // A view of memory outside the pool is not cached.
                return Ok(());
            }
        };
        let buf = data.clone();
        write_out_buf_slot(slot, unsafe {
            StridedBuf::from_shared(buf, strides.clone()).with_offset(offset)
        });
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
        Ok(())
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
    fn read(&self, out: OutBuf<'_, 'a>) -> Result<()>;
}
impl<'a, F> CachedReadFn<'a> for F
where
    F: Fn(OutBuf<'_, 'a>) -> Result<()>,
{
    #[inline(always)]
    fn read(&self, out: OutBuf<'_, 'a>) -> Result<()> {
        self(out)
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

#[cfg(test)]
mod read_size_tests {
    use crate::storage::{PlainArrayMethods, ReadParams};
    use crate::{Array, ReadContext};

    #[test]
    fn read_ctx_starts_from_the_array_read_size_and_can_override_it() {
        let mut a = Array::plain_ndarray(ndarray::Array1::<i32>::zeros(100)).unwrap();
        a.read_params_mut().set_read_size((64, 128));
        assert_eq!(a.read_params().read_size(), (64, 128));

        let mut context = a.read_ctx();
        assert_eq!(context.read_params(), a.read_params());
        context.read_params_mut().set_read_size((16, 32));
        assert_eq!(context.read_params().read_size(), (16, 32));
        assert_eq!(a.read_params().read_size(), (64, 128));

        assert_eq!(ReadContext::default().read_params(), &ReadParams::default());
    }
}
