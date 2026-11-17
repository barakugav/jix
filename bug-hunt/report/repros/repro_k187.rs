//! K187 (run 2026-11-16-scan-106-107): 32-bit only. Since #106 a stepped Slice forwards its
//! inner's ReadTiling, so a stepped Slice over a ZeroCopy inner (Plain, Scalar, Broadcast of
//! them, another stepped Slice...) is classified ZeroCopy. Since #107 (the K156 fix) a stepped
//! Slice whose bounding box does not fit in isize::MAX bytes falls back to its general path, which
//! materializes the requested region and reads the inner once per stepped position - so that
//! ZeroCopy Slice is not zero-copy. An OUTER stepped Slice trusts the ZeroCopy flag
//! (slice.rs:161) and pulls its whole bounding box ((n-1)*step+1 elements) from the inner one,
//! which then allocates that box and does one inner read per box element. A pull of 2 elements
//! allocates and computes O(step).
//!
//! The tests assert the correct behavior (a pull of 2 elements makes no allocation proportional
//! to the step), so they fail on i686 while the bug exists. On 64-bit the inner box always fits
//! and the tests pass.
//!
//! Run: bug-hunt/tools/cargo-i686 test --manifest-path jix/Cargo.toml --all-features \
//!          --test repro_k187 -- --nocapture --test-threads=1
use jix::ops::SliceItem;
use jix::Array;
use ndarray::{array, Ix1};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Records the largest single allocation (or reallocation) made while measuring.
struct Tracking;
static MAX_ALLOC: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Tracking {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        MAX_ALLOC.fetch_max(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        MAX_ALLOC.fetch_max(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        MAX_ALLOC.fetch_max(new_size, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Tracking = Tracking;

/// Serializes the measurements of the tests in this binary (the allocator counter is global).
static SERIAL: Mutex<()> = Mutex::new(());

/// Runs `f`, returning its result, the largest single allocation it made and its duration.
fn measure<R>(f: impl FnOnce() -> R) -> (R, usize, Duration) {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    MAX_ALLOC.store(0, Ordering::Relaxed);
    let t = Instant::now();
    let r = f();
    let elapsed = t.elapsed();
    (r, MAX_ALLOC.load(Ordering::Relaxed), elapsed)
}

/// A correct pull of 2 elements allocates at most a small output buffer.
const ALLOC_BOUND: usize = 4096;

/// `u32` value 7 broadcast to [2^33], sliced [::2^16] (shape [2^17]), sliced again [::2^16]
/// (shape [2]). Reading [0..2] of the outer Slice: its box is 2^16+1 elements (256 KiB), the
/// inner's box for that is 2^32+1 elements, which does not fit in a 32-bit usize (K156 fallback).
macro_rules! nested_u32 {
    () => {
        Array::plain_ndarray(ndarray::Array1::from_elem(1, 7u32))
            .unwrap()
            .broadcast(&[1u64 << 33])
            .slice((SliceItem::new(None, None, 1 << 16),))
            .slice((SliceItem::new(None, None, 1 << 16),))
    };
}

#[test]
fn f004_nested_stepped_slice_pull_allocates_outer_box() {
    let s2 = nested_u32!();
    assert_eq!(s2.shape(), &[2]);
    let ctx = s2.read_ctx();

    // Control: push mode (to_buf_slice) takes the general path of both slices.
    let mut dst = [0u32; 2];
    let (r, push_alloc, push_time) = measure(|| {
        let bytes = unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<u8>(), 8) };
        s2.to_buf_slice(&[0..2], bytes, &ctx)
    });
    r.unwrap();
    assert_eq!(dst, [7, 7]);

    // Pull mode (to_buf_strided).
    let (pulled, pull_alloc, pull_time) = measure(|| {
        let buf = s2.to_buf_strided(&[0..2], &ctx).unwrap();
        let values: Vec<u32> = (0..2)
            .map(|i| unsafe {
                buf.data_ptr()
                    .add(i * buf.strides()[0])
                    .cast::<u32>()
                    .read_unaligned()
            })
            .collect();
        values
    });
    assert_eq!(pulled, vec![7, 7]);
    eprintln!(
        "push: max alloc {push_alloc} B, {push_time:?}; pull: max alloc {pull_alloc} B, {pull_time:?}"
    );
    assert!(
        pull_alloc <= ALLOC_BOUND,
        "a pull of 2 u32 elements made a {pull_alloc}-byte allocation (push: {push_alloc} B)"
    );
}

#[test]
fn f004_nested_stepped_slice_as_elementwise_leaf() {
    // The element-wise pipeline pulls its leaves (core_trait.rs read_as_elementwise_pipeline).
    let s2 = nested_u32!();
    let (got, alloc, time) = measure(|| (!s2.view()).to_ndarray::<Ix1>().unwrap());
    assert_eq!(got, array![!7u32, !7u32]);
    eprintln!("!s2 to_ndarray: max alloc {alloc} B, {time:?}");
    assert!(
        alloc <= ALLOC_BOUND,
        "(!s2).to_ndarray() of 2 u32 elements made a {alloc}-byte allocation"
    );
}

#[test]
fn f004_nested_stepped_slice_pull_scaled() {
    // The hunter's scale: [u32; 16] (64 B) broadcast to [2^27], [::64] then [::2^20] (shape [2]).
    // The outer box is 2^20+1 items (64 MiB); the inner box (2^26+1 items * 64 B) exceeds
    // isize::MAX on 32-bit.
    let s2 = Array::plain_ndarray(ndarray::Array1::from_elem(1, [3u32; 16]))
        .unwrap()
        .broadcast(&[1u64 << 27])
        .slice((SliceItem::new(None, None, 64),))
        .slice((SliceItem::new(None, None, 1 << 20),));
    assert_eq!(s2.shape(), &[2]);
    let ctx = s2.read_ctx();
    let (_, alloc, time) = measure(|| s2.to_buf_strided(&[0..2], &ctx).map(|_| ()).unwrap());
    eprintln!("[u32; 16] pull of 2 items: max alloc {alloc} B, {time:?}");
    assert!(
        alloc <= ALLOC_BOUND,
        "a pull of 2 64-byte items made a {alloc}-byte allocation, took {time:?}"
    );
}
