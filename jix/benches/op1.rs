mod common;

use std::ops::Neg;

use criterion::{criterion_group, criterion_main, BenchmarkGroup, Criterion};
use jix::{storage::ArrayStorageTyped, Array};

use crate::common::{create_data, Profile};

fn bench_op1_compact(c: &mut Criterion) {
    let mut group = c.benchmark_group("op1 compact");
    group.sample_size(20);
    bench_op1_impl(&mut group, |data| Array::compact_ndarray(&data).unwrap());
    group.finish();
}
fn bench_op1_plain(c: &mut Criterion) {
    let mut group = c.benchmark_group("op1 plain");
    group.sample_size(20);
    bench_op1_impl(&mut group, |data| Array::plain_ndarray(data).unwrap());
    group.finish();
}

fn bench_op1_impl<S: ArrayStorageTyped<Item = i32>>(
    group: &mut BenchmarkGroup<'_, criterion::measurement::WallTime>,
    create_array: impl Fn(ndarray::ArrayD<i32>) -> Array<S>,
) {
    for size in [512, 4000, 40_000, 400_000, 4_000_000] {
        let shape = [size, 64];
        group.bench_function(&format!("op1 {shape:?}"), |b| {
            let data = create_data::<i32>(Profile::Smooth, &shape, 0xe8f34272be79cb28);
            let array = create_array(data);
            b.iter_batched(
                || {},
                |_| array.as_ref().neg().to_ndarray().unwrap().shape().to_vec(),
                criterion::BatchSize::LargeInput,
            );
        });
    }
}

criterion_group!(benches, bench_op1_compact, bench_op1_plain);
criterion_main!(benches);
