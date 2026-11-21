use std::cell::{Cell, RefCell};
use std::sync::Arc;

use crate::codec::{Decoder, DecoderCodecConfig};
use crate::dtype::{Alignment, Dtype};
use crate::error::Result;
use crate::storage::cache_id::{ArrayCacheId, ArrayCacheIdMap, ArrayCacheIdSet};
use crate::storage::params::{ReadParams, ReadSize};
use crate::storage::{Index, OutBuf, OutBufInner, StridedBuf, write_out_buf_slot};
use crate::util::buf_pool::{BufferPool, PoolBuf};
use crate::util::{DimArray, SliceExt};
use crate::{Array, DimDyn, StorageBase};

/// A context with reusable buffers and a long-lived decoder instance.
///
/// Pass one `ReadContext` to many successive reads (the `to_buf_*` methods of [`Array`] take one):
/// it saves allocating scratch buffers on every read, a cost that matters most for small reads.
/// Reads without one, and the other read methods, create one per call with the read parameters of
/// the array, borrowing the decompressor of the current thread (see [`ReadContext::thread_ctx`]).
/// The read parameters of a context override the ones of the array.
///
/// `ReadContext` is `!Sync`: for concurrent reads, create one per thread.
///
/// # Obtaining a `ReadContext`
///
/// The preferred way is [`ReadContext::new_for_array`], which picks up the read parameters of the
/// array, such as its read size (see [`ReadParams`]):
///
/// ```
/// use jix::{Array, ReadContext};
/// use ndarray::array;
///
/// let za = Array::compact_ndarray(&array![1i32, 2, 3, 4])?;
///
/// // new_for_array() takes the read size of the array.
/// let ctx = ReadContext::new_for_array(&za)?;
/// // Reuse it, and its scratch buffers, for many reads.
/// for start in 0..3 {
///     let mut out = [0u8; 8];
///     za.to_buf_slice(&[start..start + 2], &mut out, Some(&ctx))?;
///     assert_eq!(out[0] as u64, start + 1);
/// }
/// # Ok::<(), jix::Error>(())
/// ```
///
/// A context made by [`new`](ReadContext::new), [`new_for_array`](ReadContext::new_for_array) or
/// [`default`](ReadContext::default) creates its own decompressor on its first block read, and
/// keeps it until dropped. [`ReadContext::thread_ctx`] borrows the one of the current thread
/// instead.
///
/// `ReadContext::default()` creates a context independent of any array:
/// ```
/// use jix::{Array, ReadContext};
/// use ndarray::array;
///
/// let za = Array::compact_ndarray(&array![42i32, 17, 6, 99, 51])?;
/// let mut out = [0u8; 4];
/// za.to_buf_slice(&[2..3], &mut out, Some(&ReadContext::default()))?;
/// assert_eq!(out, [6, 0, 0, 0]);
/// # Ok::<(), jix::Error>(())
/// ```
pub struct ReadContext {
    pub(crate) decompressor: RefCell<Option<zstd::bulk::Decompressor<'static>>>,
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
    /// Prefer [`ReadContext::new_for_array`] over calling this directly - it uses the read
    /// parameters of the array.
    pub fn new(params: ReadParams) -> Result<Self> {
        Ok(Self {
            decompressor: RefCell::new(None),
            buffer_pool: BufferPool::new(),

            read_cache: RefCell::new(None),
            is_cache_enabled: Cell::new(false),
            temp_cache_maps: RefCell::new(Vec::new()),

            params,
        })
    }

    /// Creates a new `ReadContext` with the [read parameters](Array::read_params) of `array`,
    /// e.g. its read size.
    pub fn new_for_array(array: &Array<impl StorageBase>) -> Result<Self> {
        Self::new(array.read_params().clone())
    }

    /// A read context with the default read parameters, borrowing the decompressor of the current
    /// thread until the returned handle is dropped.
    ///
    /// The read methods of [`Array`] create one like it, with the read parameters of the array,
    /// when given no context. The thread keeps its decompressor between reads, so its setup is paid
    /// once per thread. The scratch buffers are freed with the handle, so an idle thread holds no
    /// large allocation.
    ///
    /// Pass the handle to many reads to also reuse its scratch buffers, without creating a second
    /// decompressor. While it is alive, the other reads of the thread find no decompressor to
    /// borrow and create their own.
    pub fn thread_ctx() -> impl AsRef<ReadContext> {
        Self::thread_ctx_with(&ReadParams::default()).unwrap()
    }

    /// A read context with `params`, borrowing the decompressor of the current thread until the
    /// returned handle is dropped.
    #[inline]
    pub(crate) fn thread_ctx_with(params: &ReadParams) -> Result<impl AsRef<ReadContext> + use<>> {
        let decompressor = THREAD_CTX
            .try_with(|state| state.decompressor.take())
            .ok()
            .flatten();
        let context = ReadContext::new(params.clone())?;
        context.decompressor.replace(decompressor);
        Ok(ThreadCtx(context))
    }

    /// The read parameters of the reads using this context, e.g. the read size.
    ///
    /// [`ReadContext::new_for_array`] starts from the array's
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

struct ThreadLocalState {
    decompressor: Cell<Option<zstd::bulk::Decompressor<'static>>>,
}
thread_local! {
    static THREAD_CTX: ThreadLocalState = const { ThreadLocalState {
        decompressor: Cell::new(None),
    } };
}
struct ThreadCtx(ReadContext);
impl AsRef<ReadContext> for ThreadCtx {
    #[inline]
    fn as_ref(&self) -> &ReadContext {
        &self.0
    }
}
impl Drop for ThreadCtx {
    #[inline]
    fn drop(&mut self) {
        if let Some(decompressor) = self.0.decompressor.take() {
            let _ = THREAD_CTX.try_with(|state| state.decompressor.replace(Some(decompressor)));
        }
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
mod thread_ctx_tests {
    use super::THREAD_CTX;
    use crate::storage::{PlainArrayMethods, ReadParams};
    use crate::{Array, ReadContext};

    #[test]
    fn contexts_take_the_read_params() {
        let mut a = Array::plain_ndarray(ndarray::Array1::<i32>::zeros(100)).unwrap();
        a.read_params_mut().set_read_size((64, 128));

        let context = ReadContext::thread_ctx_with(a.read_params()).unwrap();
        assert_eq!(context.as_ref().read_params(), a.read_params());
        assert_eq!(
            ReadContext::thread_ctx().as_ref().read_params(),
            &ReadParams::default()
        );
        assert_eq!(ReadContext::default().read_params(), &ReadParams::default());

        let mut context = ReadContext::new_for_array(&a).unwrap();
        assert_eq!(context.read_params(), a.read_params());
        context.read_params_mut().set_read_size((16, 32));
        assert_eq!(context.read_params().read_size(), (16, 32));
        assert_eq!(a.read_params().read_size(), (64, 128));
    }

    #[test]
    fn thread_ctx_lends_the_thread_decompressor() {
        let has_thread_decompressor = || {
            THREAD_CTX.with(|state| {
                let decompressor = state.decompressor.take();
                let has = decompressor.is_some();
                state.decompressor.set(decompressor);
                has
            })
        };
        let a = Array::compact_ndarray(&ndarray::Array1::<i32>::from_elem(1000, 7)).unwrap();

        // The first read creates the decompressor and hands it to the thread.
        assert!(!has_thread_decompressor());
        a.to_ndarray::<ndarray::Ix1>().unwrap();
        assert!(has_thread_decompressor());
        {
            let context = ReadContext::thread_ctx();
            assert!(!has_thread_decompressor());
            assert!(context.as_ref().decompressor.borrow().is_some());
            // A nested context finds none to borrow, and keeps the thread's when it is dropped.
            let inner = ReadContext::thread_ctx();
            assert!(inner.as_ref().decompressor.borrow().is_none());
        }
        assert!(has_thread_decompressor());
    }
}
