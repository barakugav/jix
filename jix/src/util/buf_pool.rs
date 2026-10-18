use std::cell::UnsafeCell;
use std::mem::ManuallyDrop;
use std::sync::Arc;

use crate::cpu_cache::CACHE_LINE_SIZE;
use crate::dtype::Alignment;
use crate::RawAlignedBytes;

/// A pool of aligned bytes buffers.
///
/// Many components of a storage system (byte shuffle, bit shuffle, arithmetic lazy ops storage views, etc.)
/// need temporary working memory. Allocating fresh buffers on every block is expensive, so
/// `BufferPool` keeps a small free list per alignment class and returns previously allocated
/// buffers when possible.
///
/// Buffers are vended as [`PoolBuf`] RAII guards. When a `PoolBuf` is dropped, its underlying
/// allocation is pushed back into the pool for reuse. The pool keeps bare allocations (pointer and
/// capacity): the contents are scratch, so there is no length to track or bytes to preserve.
///
/// The pool is not thread-safe, it is intended to be owned and used by a single thread.
pub(crate) struct BufferPool {
    /// Free list for alignments <= CACHE_LINE_SIZE; all buffers are allocated at CACHE_LINE_SIZE-byte alignment.
    align_standard: UnsafeCell<Vec<RawAlignedBytes>>,
    /// Free lists for alignments > CACHE_LINE_SIZE, sorted by alignment value.
    align_other: UnsafeCell<Vec<(Alignment, Vec<RawAlignedBytes>)>>,
}
impl BufferPool {
    pub(crate) fn new() -> Self {
        Self {
            align_standard: UnsafeCell::new(Vec::new()),
            align_other: UnsafeCell::new(Vec::new()),
        }
    }

    /// Borrows a buffer of `size` bytes with at least `alignment` byte alignment.
    ///
    /// Returns a [`PoolBuf`] guard over at least `size` uninitialized bytes. The buffer is popped
    /// from the free list when one is available and large enough; otherwise a fresh allocation is
    /// made (a popped buffer that is too small is freed, not grown, so its stale bytes are never
    /// copied). The allocation is returned to the pool when the `PoolBuf` is dropped.
    pub(crate) fn get(&self, size: usize, alignment: Alignment) -> PoolBuf<'_> {
        let (pool, pool_align) = self.get_pool(alignment);
        let pool = unsafe { &mut *pool };
        let buf = match pool.pop() {
            Some(buf) if buf.capacity() >= size => buf,
            _ => unsafe { RawAlignedBytes::with_capacity_unchecked(size, pool_align.as_usize()) },
        };
        PoolBuf {
            buf: ManuallyDrop::new(buf),
            buffers: self,
        }
    }

    fn return_buf(&self, buf: RawAlignedBytes) {
        let (pool, _) = self.get_pool(buf.align().try_into().unwrap());
        let pool = unsafe { &mut *pool };
        pool.push(buf);
    }

    /// Drops one reference to a shared buffer, returning the allocation to the pool if it was the
    /// last one.
    pub(crate) fn return_shared(&self, buf: Arc<RawAlignedBytes>) {
        if let Some(buf) = Arc::into_inner(buf) {
            self.return_buf(buf);
        }
    }

    /// Returns a raw pointer to the free list for `alignment` together with the actual alignment
    /// that will be used for allocations from that list.
    ///
    /// Alignments <= `CACHE_LINE_SIZE` are folded into the single `align_standard` list
    /// (allocated at `CACHE_LINE_SIZE`-byte alignment). Larger alignments are looked up (or
    /// inserted) in the sorted `align_other` list.
    fn get_pool(&self, alignment: Alignment) -> (*mut Vec<RawAlignedBytes>, Alignment) {
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

/// An RAII guard for a temporary scratch buffer borrowed from a [`BufferPool`].
///
/// Obtained via [`BufferPool::get`] (or [`ReadContext::tmp_buf`]). The slices it hands out cover
/// the whole allocation, which may be larger than the size asked for: slice them to that size
/// wherever the length matters. When `PoolBuf` is dropped, the underlying allocation is returned to
/// the pool for reuse.
pub(crate) struct PoolBuf<'a> {
    /// Taken back into the pool on drop.
    buf: ManuallyDrop<RawAlignedBytes>,
    buffers: &'a BufferPool,
}
impl PoolBuf<'_> {
    /// The whole allocation, at least the size requested from [`BufferPool::get`].
    #[inline(always)]
    pub(crate) fn as_slice(&self) -> &[u8] {
        // SAFETY: the allocation holds `capacity` bytes.
        unsafe { std::slice::from_raw_parts(self.buf.as_ptr(), self.buf.capacity()) }
    }

    /// The whole allocation, at least the size requested from [`BufferPool::get`].
    #[inline(always)]
    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: the allocation holds `capacity` bytes.
        unsafe { std::slice::from_raw_parts_mut(self.buf.as_mut_ptr(), self.buf.capacity()) }
    }

    /// Take the allocation out of the guard, which then does not return it to the pool.
    pub(crate) fn into_inner(self) -> RawAlignedBytes {
        let mut this = ManuallyDrop::new(self);
        // SAFETY: `this` is never used or dropped again.
        unsafe { ManuallyDrop::take(&mut this.buf) }
    }
}
impl Drop for PoolBuf<'_> {
    fn drop(&mut self) {
        // SAFETY: `buf` is not touched again after being taken.
        let buf = unsafe { ManuallyDrop::take(&mut self.buf) };
        self.buffers.return_buf(buf);
    }
}

/// A read-only, reference-counted buffer whose allocation returns to its [`BufferPool`] when the
/// last reference is dropped.
pub(crate) struct PoolBufShared<'a> {
    buf: ManuallyDrop<Arc<RawAlignedBytes>>,
    buffers: &'a BufferPool,
}
impl<'a> PoolBufShared<'a> {
    #[inline]
    pub(crate) fn new(buf: Arc<RawAlignedBytes>, buffers: &'a BufferPool) -> Self {
        Self {
            buf: ManuallyDrop::new(buf),
            buffers,
        }
    }

    /// The whole allocation.
    #[inline(always)]
    pub(crate) fn as_slice(&self) -> &[u8] {
        // SAFETY: the allocation holds `capacity` bytes.
        unsafe { std::slice::from_raw_parts(self.buf.as_ptr(), self.buf.capacity()) }
    }
}
impl Drop for PoolBufShared<'_> {
    fn drop(&mut self) {
        // SAFETY: `self.buf` is never used again.
        let buf = unsafe { ManuallyDrop::take(&mut self.buf) };
        self.buffers.return_shared(buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_alignments_share_one_free_list() {
        let pool = BufferPool::new();
        for alignment in [1usize, 8, CACHE_LINE_SIZE] {
            let buf = pool.get(64, Alignment::new(alignment).unwrap());
            assert!(buf.as_slice().len() >= 64);
            assert!((buf.as_slice().as_ptr() as usize).is_multiple_of(CACHE_LINE_SIZE));
        }
        // Each buffer was returned before the next request, so one allocation served all three.
        assert!(unsafe { &*pool.align_other.get() }.is_empty());
        assert_eq!(unsafe { &*pool.align_standard.get() }.len(), 1);
    }

    #[test]
    fn reuses_a_large_enough_buffer_and_replaces_a_small_one() {
        let pool = BufferPool::new();
        let align = Alignment::new(8).unwrap();
        let first = pool.get(256, align);
        let first_ptr = first.as_slice().as_ptr();
        drop(first);

        // Smaller request: the pooled allocation is reused whole.
        let smaller = pool.get(100, align);
        assert_eq!(smaller.as_slice().as_ptr(), first_ptr);
        assert_eq!(smaller.as_slice().len(), 256);
        drop(smaller);

        // Larger request: the pooled allocation is too small and is replaced, not grown.
        let mut larger = pool.get(4096, align);
        assert!(larger.as_mut_slice().len() >= 4096);
        larger.as_mut_slice()[..4096].fill(7);
        drop(larger);
        let free = unsafe { &*pool.align_standard.get() };
        assert_eq!(free.len(), 1);
        assert!(free[0].capacity() >= 4096);
    }

    #[test]
    fn over_aligned_buffers_get_a_free_list_each() {
        let pool = BufferPool::new();
        let wide = Alignment::new(CACHE_LINE_SIZE * 4).unwrap();
        let narrow = Alignment::new(CACHE_LINE_SIZE * 2).unwrap();

        // `wide` first so the second request has to insert *before* it and keep the list sorted.
        for alignment in [wide, narrow, wide] {
            let buf = pool.get(128, alignment);
            assert!((buf.as_slice().as_ptr() as usize).is_multiple_of(alignment.as_usize()));
        }

        let lists = unsafe { &*pool.align_other.get() };
        assert_eq!(lists.len(), 2);
        assert_eq!(lists[0].0, narrow);
        assert_eq!(lists[1].0, wide);
    }
}
