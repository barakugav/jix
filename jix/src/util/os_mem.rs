//! Hints to the OS about big memory regions. Linux only: everywhere else (and under Miri) these
//! do nothing.

/// An allocation at least this large is worth backing with huge pages (NumPy's threshold too).
pub(crate) const HUGE_PAGE_MIN_BYTES: usize = 4 << 20;

/// The granularity [`advise_huge_pages`] works in: the transparent huge page size on x86-64 and
/// on aarch64 with 4K pages, and a multiple of the base page size on every Linux target.
#[cfg(all(target_os = "linux", not(miri)))]
const HUGE_PAGE_ALIGN: usize = 2 << 20;

/// Ask the OS to back the freshly allocated, not yet written `[ptr, ptr + len)` with transparent
/// huge pages.
///
/// A big allocation is otherwise populated one small page fault at a time - tens of thousands of
/// them for a few hundred megabytes, which can cost as much as decoding the data that goes in.
/// Purely a hint: it never changes the contents, and a failure is ignored.
///
/// Only the huge-page-aligned middle of the region is advised. The kernel can only ever put a
/// huge page on an aligned extent that lies wholly inside the advised range, so the unaligned
/// head and tail have nothing to gain - and an aligned start is a valid page boundary whatever
/// the base page size is.
#[inline]
pub(crate) fn advise_huge_pages(ptr: *mut u8, len: usize) {
    #[cfg(all(target_os = "linux", not(miri)))]
    {
        let start = (ptr as usize).next_multiple_of(HUGE_PAGE_ALIGN);
        let end = (ptr as usize + len) & !(HUGE_PAGE_ALIGN - 1);
        if end > start {
            // SAFETY: `[start, end)` lies inside the caller's region, and `MADV_HUGEPAGE` only
            // changes how the kernel backs those pages, never what they hold.
            unsafe { libc::madvise(start as *mut libc::c_void, end - start, libc::MADV_HUGEPAGE) };
        }
    }
    #[cfg(not(all(target_os = "linux", not(miri))))]
    let _ = (ptr, len);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg_attr(miri, ignore)] // a 64MB region
    fn advised_memory_reads_back_as_written() {
        // Big enough to come straight from the OS, untouched, and to hold aligned huge pages.
        let len = 64 << 20;
        let mut buf = Vec::<u8>::with_capacity(len);
        advise_huge_pages(buf.as_mut_ptr(), len);
        unsafe {
            buf.as_mut_ptr().write_bytes(7, len);
            buf.set_len(len);
        }
        assert!(buf.iter().all(|&b| b == 7));
    }

    #[test]
    fn regions_without_an_aligned_huge_page_are_left_alone() {
        let mut small = [1u8; 100];
        advise_huge_pages(small.as_mut_ptr(), small.len());
        advise_huge_pages(small.as_mut_ptr(), 0);
        assert!(small.iter().all(|&b| b == 1));

        // Just under two huge pages: may or may not contain an aligned one, either is fine.
        let len = (4 << 20) - 1;
        let mut buf = vec![3u8; len];
        advise_huge_pages(buf.as_mut_ptr(), len);
        assert!(buf.iter().all(|&b| b == 3));
    }
}
