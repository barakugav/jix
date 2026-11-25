//! EXPERIMENTAL probe: heap allocations per small read, with and without reusing the scratch
//! buffers between reads. The companion of `benches/thread_alloc.rs`, which has the careful
//! timings; this one counts allocations through a counting global allocator, and prints a rough
//! ns/read next to them.
//!
//! ```text
//! cargo run --release --example thread_alloc_probe
//! ```
//!
//! Modes:
//! - `get/fresh` - `Array::get` as on main: a context per call, fresh scratch buffers.
//! - `get/thread_pool` - `Array::get` with the experimental thread-local buffer pool.
//! - `slice/fresh` - `to_buf_slice(.., None)` of a 1x1 region: a context per call, like `get`.
//! - `slice/explicit_ctx` - `to_buf_slice(.., Some(&ctx))` with one long-lived `ReadContext`.

#[path = "../benches/common/thread_alloc.rs"]
mod scenarios;

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Instant;

use jix::__private::set_thread_pool_retain_bytes;
use jix::{Array, ReadContext, Storage};

use crate::scenarios::{Positions, RETAIN_BYTES, ScenarioVisitor, for_each_scenario};

struct Counting;
static ALLOCS: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        ALLOC_BYTES.fetch_add(layout.size() as u64, Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        ALLOC_BYTES.fetch_add(layout.size() as u64, Relaxed);
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        ALLOC_BYTES.fetch_add(new_size as u64, Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}
#[global_allocator]
static GLOBAL: Counting = Counting;

/// Reads per measured round; the reported time is the best of `ROUNDS`.
const READS: usize = 20_000;
const ROUNDS: usize = 5;

struct Stats {
    allocs: f64,
    kib: f64,
    ns: f64,
}

/// Runs `read(i)` for `READS` positions (after a warm-up round), `ROUNDS` times.
fn measure(mut read: impl FnMut(usize)) -> Stats {
    for i in 0..1000 {
        read(i);
    }
    let (allocs0, bytes0) = (ALLOCS.load(Relaxed), ALLOC_BYTES.load(Relaxed));
    let mut best = f64::INFINITY;
    for round in 0..ROUNDS {
        let start = Instant::now();
        for i in 0..READS {
            read(round * READS + i);
        }
        best = best.min(start.elapsed().as_nanos() as f64 / READS as f64);
    }
    let n = (READS * ROUNDS) as f64;
    Stats {
        allocs: (ALLOCS.load(Relaxed) - allocs0) as f64 / n,
        kib: (ALLOC_BYTES.load(Relaxed) - bytes0) as f64 / n / 1024.0,
        ns: best,
    }
}

struct Probe;

impl ScenarioVisitor for Probe {
    fn visit<S: Storage>(&mut self, name: &str, array: &Array<S>, positions: &Positions) {
        let (indices, regions) = (&positions.indices, &positions.regions);
        let get = |i: usize| {
            std::hint::black_box(array.get(&indices[i % indices.len()]).unwrap());
        };

        set_thread_pool_retain_bytes(0);
        let fresh = measure(get);
        set_thread_pool_retain_bytes(RETAIN_BYTES);
        let pooled = measure(get);
        set_thread_pool_retain_bytes(0);

        let ctx = ReadContext::new_for_array(array).unwrap();
        let itemsize = array.dtype().itemsize() as usize;
        let mut out = [0u64; 2];
        let buf =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), itemsize) };
        let mut slice = |ctx: Option<&ReadContext>| {
            measure(|i| {
                let region = &regions[i % regions.len()];
                array.to_buf_slice(region, buf, ctx).unwrap();
            })
        };
        let slice_fresh = slice(None);
        let explicit = slice(Some(&ctx));

        let row = |mode: &str, s: &Stats| {
            println!(
                "| {name:<38} | {mode:<18} | {:>6.2} | {:>9.1} | {:>8.0} | {:>6.2}x |",
                s.allocs,
                s.kib,
                s.ns,
                fresh.ns / s.ns
            );
        };
        row("get/fresh", &fresh);
        row("get/thread_pool", &pooled);
        row("slice/fresh", &slice_fresh);
        row("slice/explicit_ctx", &explicit);
    }
}

fn main() {
    println!(
        "| {:<38} | {:<18} | {:>6} | {:>9} | {:>8} | {:>7} |",
        "scenario", "mode", "allocs", "KiB alloc", "ns/read", "speedup"
    );
    println!(
        "|{:-<40}|{:-<20}|{:->8}|{:->11}|{:->10}|{:->9}|",
        "", "", ":", ":", ":", ":"
    );
    for_each_scenario(&mut Probe);
}
