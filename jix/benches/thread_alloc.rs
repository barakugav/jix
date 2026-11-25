//! EXPERIMENTAL: how much does reusing scratch buffers between small reads save?
//!
//! Since the reads stopped taking a `ReadContext`, `Array::get` and the `to_buf_*` methods given
//! `None` create a context per call. The zstd decompressor is borrowed from the thread, but the
//! scratch buffers (the decoded block, the filter output, op staging buffers, ...) are allocated
//! anew on every call. For each scenario of `common/thread_alloc.rs` this times:
//!
//! - `get/fresh` - `Array::get` as on main: fresh scratch buffers on every call.
//! - `get/thread_pool` - `Array::get`, with the experimental hack that keeps the buffer pool of the
//!   thread context in a thread-local between calls (`set_thread_pool_retain_bytes`).
//! - `slice/fresh`, `slice/thread_pool` - the same for `to_buf_slice(.., None)` of a 1x1 region.
//! - `slice/explicit_ctx` - `to_buf_slice(.., Some(&ctx))` with one long-lived `ReadContext`, the
//!   reuse a caller can already get today: the reference `thread_pool` should match.
//!
//! Each iteration reads a different random position, so every read decodes a block. Run with
//! `cargo bench --bench thread_alloc`, and `examples/thread_alloc_probe.rs` for the allocation
//! counts behind the numbers.
//!
//! Results (2048x2048 i32, 4-vCPU Xeon @ 2.3GHz, 48 KiB L1d, glibc 2.39 malloc). The thread pool
//! removes every allocation of the read (3 for a byte-shuffled block, 2 unfiltered, 4 bit
//! shuffled, 5 for `a + b`) and matches `explicit_ctx`:
//!
//! ```text
//! scenario                    get/fresh  get/thread_pool  saved
//! plain                           44 ns            52 ns   -8 ns  (no allocation to save)
//! compact [6, 2048] (auto)     14.8 us          14.1 us   0.7 us  (4.8%)
//! compact [32, 32]              2.10 us          1.97 us   0.13 us (6.7%)
//! compact [256, 256]           66.6 us          65.7 us   noise
//! compact [512, 512]            280 us           294 us   noise
//! compact [6, 2048] zeros       5.41 us          5.07 us   0.33 us (6.6%)
//! compact [32, 32] zeros         810 ns           696 ns   114 ns (16%)
//! compact [6, 2048] no_filter  49.1 us          47.8 us   ~1 us  (~3%, noisy)
//! compact [6, 2048] bit_shuf   28.0 us          26.8 us   ~1 us  (~4%, noisy)
//! lazy a + b                   29.5 us          28.6 us   0.9 us  (3.0%)
//! lazy a.sum(1)                14.9 us          15.2 us   noise
//! ```
//!
//! A `get` decodes the whole block it touches, which dwarfs the allocations: reuse saves ~0.1-1
//! us per read, 3-7% for typical blocks and up to ~16% only for tiny blocks of trivially
//! compressible data.

#[path = "common/thread_alloc.rs"]
mod scenarios;

use std::time::Duration;

use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, Criterion, criterion_group, criterion_main};
use jix::__private::set_thread_pool_retain_bytes;
use jix::{Array, ReadContext, Storage};

use crate::scenarios::{Positions, RETAIN_BYTES, ScenarioVisitor, for_each_scenario};

struct Bench<'c> {
    c: &'c mut Criterion,
}

impl ScenarioVisitor for Bench<'_> {
    fn visit<S: Storage>(&mut self, name: &str, array: &Array<S>, positions: &Positions) {
        let mut group = self.c.benchmark_group(format!("thread_alloc {name}"));
        group
            .warm_up_time(Duration::from_millis(500))
            .measurement_time(Duration::from_secs(2));

        for (mode, retain_bytes) in [("fresh", 0), ("thread_pool", RETAIN_BYTES)] {
            bench_get(
                &mut group,
                &format!("get/{mode}"),
                array,
                positions,
                retain_bytes,
            );
            bench_slice(
                &mut group,
                &format!("slice/{mode}"),
                array,
                positions,
                retain_bytes,
                false,
            );
        }
        bench_slice(&mut group, "slice/explicit_ctx", array, positions, 0, true);
        group.finish();
    }
}

fn bench_get<S: Storage>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    id: &str,
    array: &Array<S>,
    positions: &Positions,
    retain_bytes: usize,
) {
    set_thread_pool_retain_bytes(retain_bytes);
    let indices = &positions.indices;
    group.bench_function(id, |b| {
        let mut i = 0;
        b.iter(|| {
            let index = &indices[i % indices.len()];
            i += 1;
            array.get(index).unwrap()
        });
    });
    set_thread_pool_retain_bytes(0);
}

fn bench_slice<S: Storage>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    id: &str,
    array: &Array<S>,
    positions: &Positions,
    retain_bytes: usize,
    explicit_ctx: bool,
) {
    set_thread_pool_retain_bytes(retain_bytes);
    let ctx = explicit_ctx.then(|| ReadContext::new_for_array(array).unwrap());
    let regions = &positions.regions;
    let itemsize = array.dtype().itemsize() as usize;
    group.bench_function(id, |b| {
        let mut out = [0u64; 2];
        let buf =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), itemsize) };
        let mut i = 0;
        b.iter(|| {
            let region = &regions[i % regions.len()];
            i += 1;
            array.to_buf_slice(region, buf, ctx.as_ref()).unwrap();
        });
    });
    set_thread_pool_retain_bytes(0);
}

fn bench_thread_alloc(c: &mut Criterion) {
    for_each_scenario(&mut Bench { c });
}

criterion_group!(benches, bench_thread_alloc);
criterion_main!(benches);
