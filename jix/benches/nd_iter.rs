//! The outer `NdIter` walk of `NdIterUnordered`, isolated: copies of many 2-element rows whose
//! strides block every dim merge, so the walk over the outer dims dominates the inner copies.
//! One case per outer-rank instantiation: `Dim<2>`, `Dim<3>` and `DimDyn` (5 dims and up).

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use jix::__private::bench_util::nd_copy;
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

criterion_group!(benches, bench_nd_iter);
criterion_main!(benches);

fn default_strides(shape: &[usize], itemsize: usize) -> Vec<usize> {
    let mut strides = vec![itemsize; shape.len()];
    for dim in (0..shape.len().saturating_sub(1)).rev() {
        strides[dim] = strides[dim + 1] * shape[dim + 1];
    }
    strides
}
