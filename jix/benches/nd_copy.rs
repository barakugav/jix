use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use jix::__private::bench_util::nd_copy;
use jix::dtype::Dtyped;

fn bench_nd_copy(c: &mut Criterion) {
    struct Case {
        name: &'static str,
        shape: &'static [usize],
        src_strides: Vec<usize>,
        dst_strides: Vec<usize>,
    }

    let cases = vec![
        // --- tiny per-call-overhead cases (mirror the compact_read block scatter, which issues
        // many small copies). These are latency/fixed-overhead bound, not bandwidth bound. ---
        Case {
            name: "tiny single 1x1",
            shape: &[1, 1],
            src_strides: default_strides(&[1, 1], size_of::<i32>()),
            dst_strides: default_strides(&[1, 1], size_of::<i32>()),
        },
        Case {
            name: "tiny 1x32",
            shape: &[1, 32],
            src_strides: default_strides(&[1, 32], size_of::<i32>()),
            dst_strides: default_strides(&[1, 32], size_of::<i32>()),
        },
        Case {
            name: "tiny 32x1",
            shape: &[32, 1],
            src_strides: default_strides(&[32, 1], size_of::<i32>()),
            dst_strides: default_strides(&[32, 1], size_of::<i32>()),
        },
        Case {
            name: "tiny 32x32",
            shape: &[32, 32],
            src_strides: default_strides(&[32, 32], size_of::<i32>()),
            dst_strides: default_strides(&[32, 32], size_of::<i32>()),
        },
        // tiny copy into a strided (row-padded) destination, like a block scattered into a wider out
        Case {
            name: "tiny 32x32 gapped-dst",
            shape: &[32, 32],
            src_strides: default_strides(&[32, 32], size_of::<i32>()),
            dst_strides: vec![64 * size_of::<i32>(), size_of::<i32>()],
        },
        // nice, easy, contiguous.
        Case {
            name: "contiguous 4096x64",
            shape: &[4096, 64],
            src_strides: default_strides(&[4096, 64], size_of::<i32>()),
            dst_strides: default_strides(&[4096, 64], size_of::<i32>()),
        },
        // 64 column transpose
        Case {
            name: "transpose 4096x64",
            shape: &[4096, 64],
            src_strides: default_strides(&[4096, 64], size_of::<i32>()),
            dst_strides: transposed_strides(&[4096, 64], size_of::<i32>()),
        },
        // big square transpose: classic cache-hostile case, 4 MB, spills all caches ---
        Case {
            name: "contiguous 1024x1024",
            shape: &[1024, 1024],
            src_strides: default_strides(&[1024, 1024], size_of::<i32>()),
            dst_strides: default_strides(&[1024, 1024], size_of::<i32>()),
        },
        Case {
            name: "transpose 1024x1024",
            shape: &[1024, 1024],
            src_strides: default_strides(&[1024, 1024], size_of::<i32>()),
            dst_strides: transposed_strides(&[1024, 1024], size_of::<i32>()),
        },
        // Small transpose that fits in L2 (256 KB)
        Case {
            name: "transpose 256x256",
            shape: &[256, 256],
            src_strides: default_strides(&[256, 256], size_of::<i32>()),
            dst_strides: transposed_strides(&[256, 256], size_of::<i32>()),
        },
        // Outer axis gapped (backing width 2x), inner axis contiguous on BOTH sides: the inner
        // 64-run coalesces, leaving a strided 1D copy of blocks. Should be unaffected by reordering.
        Case {
            name: "outer-strided 4096x64",
            shape: &[4096, 64],
            src_strides: vec![2 * 64 * size_of::<i32>(), size_of::<i32>()],
            dst_strides: vec![2 * 64 * size_of::<i32>(), size_of::<i32>()],
        },
        // Inner axis gapped on both sides (stride 2*size_of::<i32>() != size_of::<i32>()): nothing
        // coalesces, the general strided nd loop runs with a strided innermost on both operands.
        Case {
            name: "strided-inner 4096x64",
            shape: &[4096, 64],
            src_strides: vec![128 * size_of::<i32>(), 2 * size_of::<i32>()],
            dst_strides: vec![128 * size_of::<i32>(), 2 * size_of::<i32>()],
        },
        // 3D axis-reversing transpose, 64^3 = 1 M elements (4 MB): reordering has more axes to work
        // with.
        Case {
            name: "transpose 64x64x64",
            shape: &[64, 64, 64],
            src_strides: default_strides(&[64, 64, 64], size_of::<i32>()),
            dst_strides: transposed_strides(&[64, 64, 64], size_of::<i32>()),
        },
        // Contiguous source copied into a row-padded destination (each 60-element row sits in a
        // 64-wide slot). The innermost axis is a short both-contiguous run, but the padding stops
        // it from absorbing the outer axes - which ARE stride-compatible on both sides. The old
        // trailing-only coalescing leaves a 2D strided walk ([64, 64]); the general merge collapses
        // the two outer axes into one, turning the whole copy into a single 1D strided run. Shows
        // the outer-axis merge (iterator rank reduction), not a raw-bandwidth case.
        Case {
            name: "padded-rows 64x64x60",
            shape: &[64, 64, 60],
            src_strides: default_strides(&[64, 64, 60], size_of::<i32>()),
            dst_strides: default_strides(&[64, 64, 64], size_of::<i32>()), // innermost 60 of every padded-64 row
        },
    ];

    let dtype = i32::DTYPE;
    let mut group = c.benchmark_group("nd_copy i32");
    group.sample_size(20);

    for case in &cases {
        let nitems: usize = case.shape.iter().product();
        let src_elems = strided_span_bytes(&case.shape, &case.src_strides, size_of::<i32>())
            .div_ceil(size_of::<i32>());
        let dst_elems = strided_span_bytes(&case.shape, &case.dst_strides, size_of::<i32>())
            .div_ceil(size_of::<i32>());
        let src = (0..src_elems).map(|i| i as i32).collect::<Vec<_>>();
        let mut dst = vec![0; dst_elems];

        group.throughput(Throughput::Bytes((nitems * size_of::<i32>()) as u64));
        group.bench_function(BenchmarkId::from_parameter(&case.name), |b| {
            b.iter(|| unsafe {
                nd_copy(
                    std::slice::from_raw_parts(
                        black_box(src.as_ptr()).cast::<u8>(),
                        src_elems * size_of::<i32>(),
                    ),
                    std::slice::from_raw_parts_mut(
                        black_box(dst.as_mut_ptr()).cast::<u8>(),
                        dst_elems * size_of::<i32>(),
                    ),
                    &case.shape,
                    &case.src_strides,
                    &case.dst_strides,
                    &dtype,
                );
            });
            black_box(dst.as_ptr());
        });
    }
    group.finish();
}

criterion_group!(benches, bench_nd_copy);
criterion_main!(benches);

// ==== utils ====

fn default_strides(shape: &[usize], itemsize: usize) -> Vec<usize> {
    let ndim = shape.len();
    let mut strides = vec![itemsize; ndim];
    if ndim > 1 {
        for (i, s) in shape.iter().rev().take(ndim - 1).enumerate() {
            let dim = ndim - i - 1;
            strides[dim - 1] = strides[dim] * s;
        }
    }
    strides
}

fn transposed_strides(shape: &[usize], itemsize: usize) -> Vec<usize> {
    let mut shape = shape.to_vec();
    shape.reverse();
    let mut strides = default_strides(&shape, itemsize);
    strides.reverse();
    strides
}

/// Compute the byte span of a region accessed by `shape` and `strides`.
fn strided_span_bytes(shape: &[usize], strides: &[usize], itemsize: usize) -> usize {
    let mut biggest_offset = 0;
    for (&len, &stride) in shape.iter().zip(strides) {
        if len == 0 {
            return 0;
        }
        biggest_offset += stride * (len - 1);
    }
    biggest_offset + itemsize
}
