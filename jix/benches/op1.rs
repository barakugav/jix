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
    for size in [4000, 40_000, 400_000, 2_000_000] {
        let shape = [size, 64];
        group.bench_function(&format!("op1 {shape:?}"), |b| {
            let data = create_data::<i32>(Profile::Smooth, &shape, 0x7e5352cebf8b6db2);
            let array = create_array(data);
            b.iter_batched(
                || {},
                |_| array.as_ref().neg().to_ndarray().unwrap().shape().to_vec(),
                criterion::BatchSize::LargeInput,
            );
        });
    }
}

fn bench_op1_plain_transposed(c: &mut Criterion) {
    let mut group = c.benchmark_group("op1 plain transposed");
    group.sample_size(20);

    for shape in [[300, 300], [1200, 1200]] {
        let data = create_data::<i32>(Profile::Smooth, &shape, 0x63fedb38a8565e8c);
        let data = data.t();
        let array = Array::plain_ndarray_ref(&data).unwrap();

        let nitems = shape.iter().product::<u64>() as usize;
        let index = [0..shape[0], 0..shape[1]];
        let neg = array.as_ref().neg();
        let ctx = neg.read_ctx();

        let neg = neg.as_ref().transpose();

        let id = format!("{shape:?}");
        group.bench_function(&id, |b| {
            let mut out = vec![0i32; nitems];
            b.iter(|| {
                let buf = unsafe {
                    std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), nitems * 4)
                };
                neg.to_ndarray_buf(&index, buf, &ctx).unwrap();
                out[0]
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_op1_compact,
    bench_op1_plain,
    bench_op1_plain_transposed,
);
criterion_main!(benches);
