mod common;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use jix::{Array, ArrayParams};

use crate::common::{create_data, Profile};

fn bench_compact_array(c: &mut Criterion) {
    let args: [(_, &[_]); _] = [
        (
            // array_shape
            [600, 32],
            // block_shapes
            &[[4, 4], [1, 32], [32, 32]],
        ),
        (
            // array_shape
            [11_000, 460],
            // block_shapes
            &[[4, 4], [1, 32], [512, 32], [1000, 230]],
        ),
    ];
    let levels = [3, 9];

    for (shape, block_shapes) in args {
        let mut group = c.benchmark_group(format!("compact_array {:?}", shape));
        group.sample_size(20);
        for profile in Profile::ALL {
            let data = create_data::<i32>(profile, &shape, 0x52cd98c6eb78ed4b);
            for block_shape in block_shapes {
                for level in levels {
                    let mut params = ArrayParams::new();
                    params.block_shape(block_shape.as_slice());
                    params.level(level).unwrap();
                    group.bench_function(
                        BenchmarkId::from_parameter(format!(
                            "profile={}, block={block_shape:?}, level={level}",
                            profile.name()
                        )),
                        |b| {
                            b.iter_batched(
                                || params.clone(),
                                |params| Array::compact_ndarray_with(&data, params).unwrap(),
                                criterion::BatchSize::LargeInput,
                            );
                        },
                    );
                }
            }
        }
    }
}

criterion_group!(benches, bench_compact_array);
criterion_main!(benches);
