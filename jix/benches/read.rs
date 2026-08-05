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
                        || region_iter.next().unwrap(),
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

criterion_group!(benches, bench_compact_read_region, bench_compact_read_full);
criterion_main!(benches);
