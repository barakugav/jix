mod common;

use std::ops::Range;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use fastrand::Rng;

use crate::common::create_compact;

fn bench_compact_read_region(c: &mut Criterion) {
    let mut rng = Rng::with_seed(0xfacc5ed5f7466613);
    let configs: [(_, &[(_, _)]); _] = [
        (
            // read_shape
            [1, 1],
            &[
                // (array_shape, block_shape)
                ([600, 32], [1, 32]),
                ([600, 32], [32, 1]),
                ([600, 32], [32, 32]),
                ([11_000, 460], [32, 32]),
            ],
        ),
        (
            // read_shape
            [1, 32],
            &[
                // (array_shape, block_shape)
                ([600, 32], [1, 32]),
                ([600, 32], [32, 1]),
                ([600, 32], [32, 32]),
            ],
        ),
        (
            // read_shape
            [32, 1],
            &[
                // (array_shape, block_shape)
                ([600, 32], [1, 32]),
                ([600, 32], [32, 1]),
                ([600, 32], [32, 32]),
            ],
        ),
        (
            // read_shape
            [128, 32],
            &[
                // (array_shape, block_shape)
                ([600, 32], [1, 32]),
                ([600, 32], [32, 1]),
                ([600, 32], [32, 32]),
                ([11_000, 460], [32, 32]),
            ],
        ),
    ];

    for (read_shape, shape_cfgs) in configs {
        let mut group = c.benchmark_group(format!("compact_read {:?}", read_shape));
        group.sample_size(20);

        for (shape, block_shape) in shape_cfgs {
            let array = create_compact(shape, block_shape.as_slice(), None, &mut rng);
            let ndim = array.shape().len();

            let mut regions: Vec<Vec<Range<u64>>> = vec![vec![]];
            for dim in 0..ndim {
                regions = regions
                    .into_iter()
                    .flat_map(|region| {
                        let dim_len: u64 = shape[dim];
                        let nblocks: u64 = dim_len.div_ceil(read_shape[dim]);
                        (0..nblocks).map(move |blk| {
                            let start = blk * read_shape[dim];
                            let end = (start + read_shape[dim]).min(dim_len);
                            let mut new_region = region.clone();
                            new_region.push(start..end);
                            new_region
                        })
                    })
                    .collect();
            }

            let rng = &mut rng;
            group.bench_function(
                BenchmarkId::from_parameter(format!("shape={shape:?}, block={block_shape:?}")),
                move |b| {
                    let mut region_iter = (0..u64::MAX).flat_map(|_| {
                        let mut shuffled = regions.clone();
                        rng.shuffle(&mut shuffled);
                        shuffled
                    });

                    let ctx = array.read_ctx();
                    b.iter_batched(
                        || {
                            let region = region_iter.next().unwrap();
                            let region: [_; 2] = region.try_into().unwrap();
                            region
                        },
                        |region| array.to_ndarray_sub(&region, &ctx).unwrap(),
                        criterion::BatchSize::SmallInput,
                    );
                },
            );
        }
    }
}

fn bench_compact_read_full(c: &mut Criterion) {
    let mut rng = Rng::with_seed(0xcbcc8bc6101ce649);
    let configs: [(_, &[_]); _] = [
        // array_shape
        (
            [600, 32],
            &[
                //  block_shape
                [32, 32],
                [64, 8],
            ],
        ),
        // array_shape
        (
            [11_000, 460],
            &[
                //  block_shape
                [32, 32],
                [32, 460],
                [2000, 32],
            ],
        ),
    ];

    for (shape, block_shapes) in configs {
        let mut group = c.benchmark_group(format!("compact_read_full {:?}", shape));
        group.sample_size(20);

        for block_shape in block_shapes {
            let array = create_compact(&shape, block_shape.as_slice(), None, &mut rng);

            group.bench_function(
                BenchmarkId::from_parameter(format!("shape={shape:?}, block={block_shape:?}")),
                move |b| {
                    b.iter_batched(
                        || (),
                        |_| array.to_ndarray().unwrap(),
                        criterion::BatchSize::SmallInput,
                    );
                },
            );
        }
    }
}

/// Reading a transposed view: `hinted` lets the destination follow the source's layout so the
/// decode writes straight through, while `forced_c_order` pushes into a packed row-major buffer
/// and pays a transposing copy for every element - the behavior before `read_layout_order`.
fn bench_compact_read_transposed(c: &mut Criterion) {
    let mut rng = Rng::with_seed(0x7a110c0f0d3ea501);
    // Sized to straddle the cache: the hint's win shrinks as the destination grows past it.
    let configs: &[([u64; 2], [u32; 2])] = &[
        ([600, 32], [32, 32]),
        ([600, 460], [32, 32]),
        ([2_000, 460], [32, 32]),
        ([11_000, 460], [32, 32]),
    ];

    let mut group = c.benchmark_group("compact_read_transposed");
    group.sample_size(20);
    for (shape, block_shape) in configs {
        let array = create_compact(shape, block_shape.as_slice(), None, &mut rng);
        let t = array.view().permute_axes(&[1, 0]);
        let full = [0..shape[1], 0..shape[0]];
        let param = format!("shape={shape:?}, block={block_shape:?}");

        group.bench_function(BenchmarkId::new("hinted", &param), |b| {
            let ctx = t.read_ctx();
            b.iter(|| t.to_ndarray_sub(&full, &ctx).unwrap());
        });
        // The pre-hint baseline: allocate a packed row-major destination per read (exactly what
        // `to_ndarray_sub` used to do) and push into it, paying a transposing copy. Allocating
        // inside the loop keeps this comparable to `hinted`, which allocates internally - at these
        // sizes first-touch page faults on the fresh allocation are a real part of the cost.
        group.bench_function(BenchmarkId::new("forced_c_order", &param), |b| {
            let ctx = t.read_ctx();
            let nitems = (shape[0] * shape[1]) as usize;
            b.iter(|| {
                let mut arr = ndarray::Array::<i32, _>::uninit(ndarray::IxDyn(&[
                    shape[1] as usize,
                    shape[0] as usize,
                ]));
                let buf = unsafe {
                    std::slice::from_raw_parts_mut(
                        arr.as_mut_ptr().cast::<u8>(),
                        nitems * size_of::<i32>(),
                    )
                };
                t.to_ndarray_slice(&full, buf, &ctx).unwrap();
                unsafe { arr.assume_init() }
            });
        });
    }
}

criterion_group!(
    benches,
    bench_compact_read_region,
    bench_compact_read_full,
    bench_compact_read_transposed
);
criterion_main!(benches);
