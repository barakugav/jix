// Verifier repro for F010 (run 2026-10-21-pilot).
//
// `AlignedBytes` / `RawAlignedBytes` are pub(crate) and unreachable from the public jix API, so
// this test compiles the UNMODIFIED source files jix/src/util/aligned_vec/{mod.rs,raw.rs} into the
// test crate with `#[path]`, providing the two `crate::util::*` paths they import. The code under
// test is byte-for-byte the library's.
//
// Native (debug) run: a counting #[global_allocator] (thread-local counters, so parallel tests do
// not interfere) records realloc calls with new_size == 0 and live allocations, and can be told to
// fail the next realloc.
// Miri run: the counting allocator is compiled out so Miri's own `__rust_realloc` shim checks the
// request (it rejects size 0 as UB) and its leak checker runs at exit.

#[path = "../src/util/aligned_vec/mod.rs"]
mod aligned_vec;

mod util {
    pub(crate) use crate::aligned_vec;
    pub(crate) mod cpu_cache {
        pub(crate) const CACHE_LINE_SIZE: usize = 128;
    }
}

use aligned_vec::AlignedBytes;

#[cfg(not(miri))]
mod counting {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    thread_local! {
        pub static LIVE: Cell<isize> = const { Cell::new(0) };
        pub static ZERO_SIZE_REALLOCS: Cell<usize> = const { Cell::new(0) };
        pub static FAIL_NEXT_REALLOC: Cell<bool> = const { Cell::new(false) };
    }

    pub struct Counting;

    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, l: Layout) -> *mut u8 {
            let p = unsafe { System.alloc(l) };
            if !p.is_null() {
                let _ = LIVE.try_with(|c| c.set(c.get() + 1));
            }
            p
        }
        unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
            let p = unsafe { System.alloc_zeroed(l) };
            if !p.is_null() {
                let _ = LIVE.try_with(|c| c.set(c.get() + 1));
            }
            p
        }
        unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
            let _ = LIVE.try_with(|c| c.set(c.get() - 1));
            unsafe { System.dealloc(p, l) }
        }
        unsafe fn realloc(&self, p: *mut u8, l: Layout, new_size: usize) -> *mut u8 {
            if new_size == 0 {
                let _ = ZERO_SIZE_REALLOCS.try_with(|c| c.set(c.get() + 1));
            }
            if FAIL_NEXT_REALLOC.try_with(|c| c.replace(false)).unwrap_or(false) {
                // A legal GlobalAlloc::realloc result: "Returns null if ... reallocation fails".
                return std::ptr::null_mut();
            }
            // Net-neutral for LIVE: one block in, one block out.
            unsafe { System.realloc(p, l, new_size) }
        }
    }

    #[global_allocator]
    static GLOBAL: Counting = Counting;

    pub fn live() -> isize {
        LIVE.with(|c| c.get())
    }
    pub fn zero_size_reallocs() -> usize {
        ZERO_SIZE_REALLOCS.with(|c| c.get())
    }
}

/// Control: shrinking to a non-zero length frees everything on drop.
#[cfg(not(miri))]
#[test]
fn f010_control_shrink_to_len3_no_leak() {
    let live0 = counting::live();
    let z0 = counting::zero_size_reallocs();
    {
        let mut v = AlignedBytes::with_capacity_exact(16, 10);
        v.push(0);
        v.push(1);
        v.push(2);
        v.shrink_to_fit();
        assert_eq!(v.capacity(), 3);
    }
    eprintln!(
        "control: zero-size reallocs = {}, leaked blocks = {}",
        counting::zero_size_reallocs() - z0,
        counting::live() - live0
    );
    assert_eq!(counting::zero_size_reallocs() - z0, 0);
    assert_eq!(counting::live() - live0, 0);
}

/// F010 trigger: an allocated but empty vector (len 0, cap 10), then shrink_to_fit().
#[test]
fn f010_shrink_to_fit_empty_allocated_vec() {
    #[cfg(not(miri))]
    let (live0, z0) = (counting::live(), counting::zero_size_reallocs());
    {
        let mut v = AlignedBytes::with_capacity_exact(16, 10);
        assert_eq!(v.len(), 0);
        assert_eq!(v.capacity(), 10);
        v.shrink_to_fit(); // -> RawAlignedBytes::shrink_to(0) -> realloc(ptr, Layout(10, 16), 0)
        eprintln!(
            "after shrink_to_fit: len = {}, cap = {}, ptr = {:p} (cap-0 dangling address would be 0x10)",
            v.len(),
            v.capacity(),
            v.as_ptr()
        );
        // Re-grow: the cap==0 branch of grow_amortized overwrites the old (non-dangling) block.
        v.push(7);
        assert_eq!(&*v, &[7]);
    }
    #[cfg(not(miri))]
    {
        let zero = counting::zero_size_reallocs() - z0;
        let leaked = counting::live() - live0;
        eprintln!("trigger: realloc calls with new_size == 0: {zero}; leaked blocks: {leaked}");
        assert_eq!(zero, 0, "GlobalAlloc::realloc was called with new_size == 0 (contract: > 0)");
        assert_eq!(leaked, 0, "allocation leaked after shrink_to_fit on an empty vector");
    }
}

/// Same trigger through shrink_to(0) with len 0.
#[cfg(not(miri))]
#[test]
fn f010_shrink_to_zero_empty_allocated_vec() {
    let (live0, z0) = (counting::live(), counting::zero_size_reallocs());
    {
        let mut v = AlignedBytes::with_capacity_exact(128, 64);
        v.shrink_to(0);
        eprintln!("after shrink_to(0): cap = {}, ptr = {:p}", v.capacity(), v.as_ptr());
    }
    let zero = counting::zero_size_reallocs() - z0;
    let leaked = counting::live() - live0;
    eprintln!("shrink_to(0): realloc calls with new_size == 0: {zero}; leaked blocks: {leaked}");
    assert_eq!(zero, 0, "GlobalAlloc::realloc was called with new_size == 0");
    assert_eq!(leaked, 0, "allocation leaked after shrink_to(0) on an empty vector");
}

/// Null result on a regular (len > 0) shrink: the allocator legally returns null; raw.rs:158 wraps
/// it in NonNull::new_unchecked. In a debug build the core ub_check aborts the process, so this is
/// #[ignore]d and run on its own:
///   cargo test ... --test repro_2026_10_21_pilot__f010 -- --ignored f010_shrink_null --nocapture
#[cfg(not(miri))]
#[test]
#[ignore]
fn f010_shrink_null_realloc_result_unchecked() {
    let mut v = AlignedBytes::with_capacity_exact(16, 10);
    v.push(1);
    v.push(2);
    v.push(3);
    counting::FAIL_NEXT_REALLOC.with(|c| c.set(true));
    eprintln!("shrinking cap 10 -> 3 with a realloc that returns null");
    v.shrink_to_fit();
    eprintln!(
        "returned normally: cap = {}, ptr = {:p} (null => UB; Drop will dealloc(null))",
        v.capacity(),
        v.as_ptr()
    );
    std::mem::forget(v);
}
