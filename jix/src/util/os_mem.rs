//! Hints to and from the OS about big memory regions. Linux only: everywhere else (and under
//! Miri) these do nothing and report nothing.

/// An allocation at least this large is worth backing with huge pages (NumPy's threshold too).
pub(crate) const HUGE_PAGE_MIN_BYTES: usize = 4 << 20;

#[cfg(all(target_os = "linux", not(miri)))]
mod sys {
    use std::ffi::{c_int, c_long, c_void};

    unsafe extern "C" {
        pub(super) fn madvise(addr: *mut c_void, len: usize, advice: c_int) -> c_int;
        pub(super) fn mincore(addr: *mut c_void, len: usize, vec: *mut u8) -> c_int;
        pub(super) fn sysconf(name: c_int) -> c_long;
    }
    pub(super) const MADV_HUGEPAGE: c_int = 14;
    pub(super) const SC_PAGESIZE: c_int = 30;

    pub(super) fn page_size() -> usize {
        static PAGE_SIZE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *PAGE_SIZE.get_or_init(|| {
            let size = unsafe { sysconf(SC_PAGESIZE) };
            usize::try_from(size)
                .ok()
                .filter(|s| s.is_power_of_two())
                .unwrap_or(4096)
        })
    }
}

/// Ask the OS to back the freshly allocated, not yet written `[ptr, ptr + len)` with transparent
/// huge pages.
///
/// A big allocation is otherwise populated one small page fault at a time - tens of thousands of
/// them for a few hundred megabytes, which can cost as much as decoding the data that goes in.
/// Purely a hint: it never changes the contents, and a failure is ignored.
#[inline]
pub(crate) fn advise_huge_pages(ptr: *mut u8, len: usize) {
    #[cfg(all(target_os = "linux", not(miri)))]
    {
        // The whole pages inside the region.
        let page = sys::page_size();
        let start = (ptr as usize).next_multiple_of(page);
        let end = (ptr as usize + len) & !(page - 1);
        if end > start {
            unsafe { sys::madvise(start as *mut _, end - start, sys::MADV_HUGEPAGE) };
        }
    }
    #[cfg(not(all(target_os = "linux", not(miri))))]
    let _ = (ptr, len);
}

/// Whether `[ptr, ptr + len)` is memory that has already been written to - as opposed to a fresh
/// allocation the OS has not yet backed with pages. Checks a few pages across the region; `false`
/// when it cannot tell.
///
/// The first write to a fresh page faults, and the OS hands over a zeroed page that sits in the
/// cache; a non-temporal store then bypasses that cached page and pays for the memory twice.
#[inline]
pub(crate) fn is_resident(ptr: *const u8, len: usize) -> bool {
    #[cfg(all(target_os = "linux", not(miri)))]
    {
        let page = sys::page_size();
        let first = (ptr as usize) & !(page - 1);
        let last = (ptr as usize + len.saturating_sub(1)) & !(page - 1);
        let middle = (first + (last - first) / 2) & !(page - 1);
        [first, middle, last].into_iter().all(|addr| {
            let mut resident = 0u8;
            let rc = unsafe { sys::mincore(addr as *mut _, page, &mut resident) };
            rc == 0 && resident & 1 == 1
        })
    }
    #[cfg(not(all(target_os = "linux", not(miri))))]
    {
        let _ = (ptr, len);
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg_attr(miri, ignore)]
    fn fresh_memory_is_not_resident_until_written() {
        if !cfg!(target_os = "linux") {
            assert!(!is_resident([0u8; 16].as_ptr(), 16));
            return;
        }
        // Big enough to come straight from the OS, untouched.
        let len = 64 << 20;
        let mut buf = Vec::<u8>::with_capacity(len);
        assert!(!is_resident(buf.as_ptr(), len));
        // Huge pages are only a hint: the region reads back as written either way.
        advise_huge_pages(buf.as_mut_ptr(), len);
        unsafe {
            buf.as_mut_ptr().write_bytes(7, len);
            buf.set_len(len);
        }
        assert!(is_resident(buf.as_ptr(), len));
        assert!(buf.iter().all(|&b| b == 7));
    }

    #[test]
    fn tiny_and_unaligned_regions_are_handled() {
        let mut small = [1u8; 100];
        advise_huge_pages(small.as_mut_ptr(), small.len());
        advise_huge_pages(small.as_mut_ptr(), 0);
        let _ = is_resident(small.as_ptr(), small.len());
        let _ = is_resident(small.as_ptr(), 0);
        assert!(small.iter().all(|&b| b == 1));
    }
}
