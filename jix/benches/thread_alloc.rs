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
