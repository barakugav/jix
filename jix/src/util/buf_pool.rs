use std::cell::UnsafeCell;

use crate::cpu_cache::CACHE_LINE_SIZE;
use crate::dtype::Alignment;
use crate::AlignedBytes;

/// A pool of aligned bytes buffers.
///
/// Many components of a storage system (byte shuffle, bit shuffle, arithmetic lazy ops storage views, etc.)
/// need temporary working memory. Allocating fresh buffers on every block is expensive, so
/// `BufferPool` keeps a small free list per alignment class and returns previously allocated
/// buffers when possible.
///
/// Buffers are vended as [`PoolBuf`] RAII guards. When a `PoolBuf` is dropped, its underlying
/// allocation is cleared and pushed back into the pool for reuse.
///
/// The pool is not thread-safe, it is intended to be owned and used by a single thread.
pub(crate) struct BufferPool {
    /// Free list for alignments <= CACHE_LINE_SIZE; all buffers are allocated at CACHE_LINE_SIZE-byte alignment.
    align_standard: UnsafeCell<Vec<AlignedBytes>>,
    /// Free lists for alignments > CACHE_LINE_SIZE, sorted by alignment value.
    align_other: UnsafeCell<Vec<(Alignment, Vec<AlignedBytes>)>>,
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
    /// Returns a [`PoolBuf`] guard whose contents are initialized to `size` uninitialized bytes.
    /// The buffer is popped from the free list when one is available; otherwise a fresh allocation
    /// is made. The allocation is returned to the pool when the `PoolBuf` is dropped.
    pub(crate) fn get(&self, size: usize, alignment: Alignment) -> PoolBuf<'_> {
        let (pool, pool_align) = self.get_pool(alignment);
        let pool = unsafe { &mut *pool };
        let tmp_buf = pool
            .pop()
            .unwrap_or_else(|| AlignedBytes::with_capacity_exact(pool_align.as_usize(), size));
        let mut buf = PoolBuf {
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
    /// Alignments <= `CACHE_LINE_SIZE` are folded into the single `align_standard` list
    /// (allocated at `CACHE_LINE_SIZE`-byte alignment). Larger alignments are looked up (or
    /// inserted) in the sorted `align_other` list.
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

/// An RAII guard for a temporary scratch buffer borrowed from a [`BufferPool`].
///
/// Obtained via [`BufferPool::get`] (or [`ReadContext::tmp_buf`]). The buffer is
/// pre-sized to the requested length on creation. When `PoolBuf` is dropped, the underlying
/// allocation is cleared and returned to the pool for reuse.
pub(crate) struct PoolBuf<'a> {
    buf: AlignedBytes,
    buffers: &'a BufferPool,
}
impl PoolBuf<'_> {
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
impl Drop for PoolBuf<'_> {
    fn drop(&mut self) {
        // Swap out self.buf so we can pass ownership to return_buf.
        let mut buf = AlignedBytes::new_exact(self.buf.alignment());
        std::mem::swap(&mut self.buf, &mut buf);

        self.buffers.return_buf(buf);
    }
}
