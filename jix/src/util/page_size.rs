/// Ask the OS to back the freshly allocated, not yet written `[ptr, ptr + len)` with transparent
/// huge pages.
///
/// Purely a hint: it never changes the contents, and a failure is ignored.
/// Helps big allocations by reducing the number of page faults.
#[inline(always)]
pub(crate) fn advise_huge_pages(ptr: *mut u8, len: usize) {
    const THRESHOLD: usize = 4 << 20; // 4 MiB
    if len >= THRESHOLD {
        advise_huge_pages_impl(ptr, len);
    }
}
fn advise_huge_pages_impl(ptr: *mut u8, len: usize) {
    #[cfg(all(any(target_os = "linux", target_os = "android"), not(miri)))]
    {
        static HUGE_PAGES_ENABLED: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
            let env_val = std::env::var("JIX_HUGE_PAGES_ENABLED").ok();
            env_val.map_or(true, |val| val != "0")
        });
        if !*HUGE_PAGES_ENABLED {
            return;
        }

        const HUGE_PAGE_ALIGN: usize = 2 << 20; // 2 MiB
        let start = ptr.addr().next_multiple_of(HUGE_PAGE_ALIGN);
        let end = (ptr.addr() + len) & !(HUGE_PAGE_ALIGN - 1);
        if end > start {
            unsafe {
                let retcode = libc::madvise(
                    ptr.with_addr(start).cast(),
                    end - start,
                    libc::MADV_HUGEPAGE,
                );
                let _ = retcode; // ignored
            }
        }
    }
    #[cfg(not(all(any(target_os = "linux", target_os = "android"), not(miri))))]
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
