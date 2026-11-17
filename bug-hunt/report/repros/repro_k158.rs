// Verifier repro (run 2026-11-15-batch7-1, finding K158).
//
// Every element-wise op clones its input's `arrays_to_cache_potentially` set (an
// `Arc<ArrayCacheIdSet>` holding the cache id of every storage below it) and then calls
// `Arc::make_mut` on it to insert its own id. The child still holds the Arc, so `make_mut`
// deep-copies the whole set: a left-deep chain of n ops (the Python `x = x + base` loop, here
// through `ArrayAny` like every Python op result) holds n^2/2 ids, so its memory is quadratic in
// the chain depth.
//
// The test counts live heap bytes with a global allocator and asserts the CORRECT behavior: a
// chain 4x deeper holds about 4x the memory (linear). It fails while the bug exists (the ratio is
// about 16).

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicIsize, Ordering};

use jix::{Array, ArrayAny};
use ndarray::Array1;

struct Counting;
static LIVE: AtomicIsize = AtomicIsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            LIVE.fetch_add(layout.size() as isize, Ordering::Relaxed);
        }
        p
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            LIVE.fetch_add(layout.size() as isize, Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        LIVE.fetch_sub(layout.size() as isize, Ordering::Relaxed);
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            LIVE.fetch_add(new_size as isize - layout.size() as isize, Ordering::Relaxed);
        }
        p
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// Live heap bytes held by a left-deep chain `base + base + ... + base` of `n` additions, built
/// one type-erased op at a time (what jix-py does for `x = x + base`).
fn chain_bytes(n: usize) -> isize {
    let base: ArrayAny =
        Array::compact_ndarray(&Array1::from_iter((0..16).map(|i| i as f32))).unwrap().into_any();
    let before = LIVE.load(Ordering::Relaxed);
    let mut x = base.clone();
    for _ in 0..n {
        x = (x.into_typed::<f32>().unwrap() + base.clone().into_typed::<f32>().unwrap()).into_any();
    }
    let held = LIVE.load(Ordering::Relaxed) - before;
    // sanity: the chain is still a valid lazy array
    assert_eq!(x.shape(), &[16]);
    drop(x);
    held
}

#[test]
fn deep_elementwise_chain_metadata_is_linear() {
    // Run on a big stack: dropping a deep type-erased chain recurses once per level (K116).
    let (b1, b4) = std::thread::Builder::new()
        .stack_size(512 << 20)
        .spawn(|| (chain_bytes(1000), chain_bytes(4000)))
        .unwrap()
        .join()
        .unwrap();
    let ratio = b4 as f64 / b1 as f64;
    println!(
        "n=1000: {:.2} MB ({} B/op), n=4000: {:.2} MB ({} B/op), ratio {:.2} (linear = 4, quadratic = 16)",
        b1 as f64 / 1e6,
        b1 / 1000,
        b4 as f64 / 1e6,
        b4 / 4000,
        ratio
    );
    assert!(
        ratio < 6.0,
        "a 4x deeper element-wise chain holds {ratio:.1}x the memory: per-op metadata grows with \
         the chain depth (cache-id sets deep-copied per op)"
    );
}
