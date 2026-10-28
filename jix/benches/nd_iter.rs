//! `NdIter` loops, isolated:
//! - The outer walk of `NdIterUnordered`: copies of many 2-element rows whose strides block every
//!   dim merge, so the walk over the outer dims dominates the inner copies. One case per outer-rank
//!   instantiation: `Dim<2>`, `Dim<3>` and `DimDyn` (5 dims and up).
//! - The per-element loop of `Array::compact_fn`: `NdIter::for_each` with an inlined body.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use jix::__private::bench_util::{nd_copy, nd_iter_fill};
use jix::dtype::Dtyped;

fn bench_nd_iter(c: &mut Criterion) {
    let cases: [(&str, &[usize]); 3] = [
        ("outer Dim<2> 256x256x2", &[256, 256, 2]),
        ("outer Dim<3> 64x64x32x2", &[64, 64, 32, 2]),
        ("outer DimDyn 16x16x16x16x2", &[16, 16, 16, 16, 2]),
    ];

    let itemsize = size_of::<i32>();
    let dtype = i32::DTYPE;
    let mut group = c.benchmark_group("nd_iter outer walk i32");
    group.sample_size(20);

    for (name, shape) in cases {
        // A contiguous source copied into a destination padded by one element on every dim: no two
        // dims merge, and the inner run is the 2-element innermost dim.
        let padded = shape.iter().map(|&d| d + 1).collect::<Vec<_>>();
        let src_strides = default_strides(shape, itemsize);
        let dst_strides = default_strides(&padded, itemsize);
        let src_len = shape.iter().product::<usize>();
        let dst_len = padded.iter().product::<usize>();
        let src = (0..src_len).map(|i| i as i32).collect::<Vec<_>>();
        let mut dst = vec![0i32; dst_len];

        group.throughput(Throughput::Bytes((src_len * itemsize) as u64));
        group.bench_function(BenchmarkId::from_parameter(name), |b| {
            b.iter(|| unsafe {
                nd_copy(
                    std::slice::from_raw_parts(
                        black_box(src.as_ptr()).cast::<u8>(),
                        src_len * itemsize,
                    ),
                    std::slice::from_raw_parts_mut(
                        black_box(dst.as_mut_ptr()).cast::<u8>(),
                        dst_len * itemsize,
                    ),
                    shape,
                    &src_strides,
                    &dst_strides,
                    &dtype,
                );
            });
            black_box(dst.as_ptr());
        });
    }
    group.finish();
}

/// The per-element loop of `Array::compact_fn`: `NdIter::for_each` writing a value computed from
/// the index at every position. A contiguous innermost dim can be vectorized, a strided one cannot.
fn bench_nd_iter_fill(c: &mut Criterion) {
    let cases: [(&str, &[u64], usize); 4] = [
        ("Dim<3> 64x64x64 contiguous", &[64, 64, 64], 1),
        ("Dim<3> 64x64x64 inner-strided", &[64, 64, 64], 2),
        ("DimDyn 16x16x32x32 contiguous", &[16, 16, 32, 32], 1),
        ("DimDyn 16x16x32x32 inner-strided", &[16, 16, 32, 32], 2),
    ];

    let mut group = c.benchmark_group("nd_iter for_each fill u64");
    group.sample_size(20);

    for (name, shape, inner_step) in cases {
        let itemsize = size_of::<u64>();
        // Row-major over `shape`, the innermost dim spread by `inner_step` elements.
        let mut backing = shape.iter().map(|&d| d as usize).collect::<Vec<_>>();
        *backing.last_mut().unwrap() *= inner_step;
        let mut strides = default_strides(&backing, itemsize);
        *strides.last_mut().unwrap() *= inner_step;
        let weights = (0..shape.len() as u64).map(|d| d + 3).collect::<Vec<_>>();
        let nitems = shape.iter().product::<u64>() as usize;
        let mut dst = vec![0u64; backing.iter().product::<usize>()];

        group.throughput(Throughput::Bytes((nitems * itemsize) as u64));
        group.bench_function(BenchmarkId::from_parameter(name), |b| {
            b.iter(|| unsafe {
                nd_iter_fill(
                    std::slice::from_raw_parts_mut(
                        black_box(dst.as_mut_ptr()).cast::<u8>(),
                        dst.len() * itemsize,
                    ),
                    shape,
                    &strides,
                    &weights,
                );
            });
            black_box(dst.as_ptr());
        });
    }
    group.finish();
}

criterion_group!(benches, bench_nd_iter, bench_nd_iter_fill);
criterion_main!(benches);

fn default_strides(shape: &[usize], itemsize: usize) -> Vec<usize> {
    let mut strides = vec![itemsize; shape.len()];
    for dim in (0..shape.len().saturating_sub(1)).rev() {
        strides[dim] = strides[dim + 1] * shape[dim + 1];
    }
    strides
}
