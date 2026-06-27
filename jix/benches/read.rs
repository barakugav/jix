mod common;

use std::ops::Range;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use fastrand::Rng;

use crate::common::create_compact;

fn bench_compact_read(c: &mut Criterion) {
    let mut rng = Rng::with_seed(0xfacc5ed5f7466613);
    let read_shapes = [[1, 1], [4, 4], [1, 32], [32, 1], [32, 32]];
    for read_shape in read_shapes {
        let mut group = c.benchmark_group(format!("compact_read {:?}", read_shape));
        group.sample_size(40);

        let configs = [
            // (array_shape, block_shape)
            ([600, 32], [4, 4]),
            ([600, 32], [1, 32]),
            ([600, 32], [32, 1]),
            ([600, 32], [32, 32]),
            ([11_000, 460], [32, 32]),
        ];
        for (shape, block_shape) in configs {
            let array = create_compact(&shape, block_shape.as_slice(), None, &mut rng);
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

criterion_group!(benches, bench_compact_read);
criterion_main!(benches);
