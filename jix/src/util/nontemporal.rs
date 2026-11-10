//! Non-temporal ("streaming") stores, for writing a destination far larger than the cache.
//!
//! A regular store to a cache line that is not in the cache first reads the line from memory
//! (a read for ownership), so writing a big buffer moves every byte across the memory bus twice.
//! A non-temporal store skips the read and writes whole lines straight to memory - about twice
//! the throughput - at the price of the data *not* being in the cache afterwards. That is the
//! right trade exactly when the destination would not have stayed in the cache anyway.

use crate::util::cpu_cache::cache_sizes;

/// Runs shorter than this are copied with regular stores: a short run is mostly the unaligned
/// head and tail, which take regular stores either way.
pub(crate) const NONTEMPORAL_MIN_RUN_BYTES: usize = 256;

/// The size of a pushed read, in bytes, from which its destination is written with non-temporal
/// stores: half the last-level cache, within sane bounds (virtual machines report odd sizes).
///
/// Small under test, so that modest reads take the non-temporal path.
#[inline]
pub(crate) fn nontemporal_min_read_bytes() -> usize {
    if cfg!(test) {
        return 16 << 10;
    }
    (cache_sizes().l3 / 2).clamp(4 << 20, 32 << 20)
}

/// [`std::ptr::copy_nonoverlapping`] of `len` bytes that writes the cache-line-aligned middle of
/// the destination with non-temporal stores, where the target has them.
///
/// The caller must run [`nontemporal_fence`] before the copied bytes can be observed by another
/// thread.
///
/// # Safety
///
/// Same contract as [`std::ptr::copy_nonoverlapping`] for `len` bytes.
#[inline]
pub(crate) unsafe fn copy_nonoverlapping_nontemporal(src: *const u8, dst: *mut u8, len: usize) {
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    unsafe {
        use std::arch::x86_64::{__m128i, _mm_loadu_si128, _mm_stream_si128};
        const LINE: usize = 64;

        let head = dst.align_offset(LINE).min(len);
        std::ptr::copy_nonoverlapping(src, dst, head);
        let mut off = head;
        while off + LINE <= len {
            let s = src.add(off).cast::<__m128i>();
            let d = dst.add(off).cast::<__m128i>();
            let (v0, v1) = (_mm_loadu_si128(s), _mm_loadu_si128(s.add(1)));
            let (v2, v3) = (_mm_loadu_si128(s.add(2)), _mm_loadu_si128(s.add(3)));
            _mm_stream_si128(d, v0);
            _mm_stream_si128(d.add(1), v1);
            _mm_stream_si128(d.add(2), v2);
            _mm_stream_si128(d.add(3), v3);
            off += LINE;
        }
        std::ptr::copy_nonoverlapping(src.add(off), dst.add(off), len - off);
    }
    #[cfg(not(all(target_arch = "x86_64", not(miri))))]
    unsafe {
        std::ptr::copy_nonoverlapping(src, dst, len)
    };
}

/// Make every non-temporal store issued so far by this thread globally visible, in order with
/// the stores that follow. Run once after a batch of [`copy_nonoverlapping_nontemporal`] calls.
#[inline]
pub(crate) fn nontemporal_fence() {
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    unsafe {
        std::arch::x86_64::_mm_sfence()
    };
}

/// Runs [`nontemporal_fence`] when dropped, so that a read which streams into its destination
/// fences on every way out - an early return and an unwinding panic included.
pub(crate) struct NontemporalFenceGuard;
impl Drop for NontemporalFenceGuard {
    #[inline]
    fn drop(&mut self) {
        nontemporal_fence();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copies_every_length_and_alignment() {
        let src = (0..1024usize)
            .map(|i| (i * 31 + 7) as u8)
            .collect::<Vec<_>>();
        for len in [0, 1, 63, 64, 65, 127, 128, 200, 256, 257, 511, 700] {
            for src_shift in [0, 1, 17] {
                for dst_shift in 0..70 {
                    let mut dst = vec![0xEEu8; len + dst_shift + 80];
                    unsafe {
                        copy_nonoverlapping_nontemporal(
                            src.as_ptr().add(src_shift),
                            dst.as_mut_ptr().add(dst_shift),
                            len,
                        )
                    };
                    nontemporal_fence();
                    assert_eq!(
                        &dst[dst_shift..dst_shift + len],
                        &src[src_shift..src_shift + len]
                    );
                    assert!(dst[..dst_shift].iter().all(|&b| b == 0xEE));
                    assert!(dst[dst_shift + len..].iter().all(|&b| b == 0xEE));
                }
            }
        }
    }
}
