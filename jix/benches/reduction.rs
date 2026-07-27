mod common;

use criterion::{criterion_group, criterion_main, BenchmarkGroup, BenchmarkId, Criterion};
use jix::{dtype::Dtyped, scalar::Sum, Array};

use crate::common::{create_data, Profile};

fn bench_sum_plain(c: &mut Criterion) {
    for size in [40_000_u64, 400_000, 1_000_000] {
        let shape = [size, 300];
        let mut group = c.benchmark_group(format!("sum plain [{size}, 300]"));
        group.sample_size(20);

        let base = create_data::<i32>(Profile::Smooth, &shape, 0)
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap();

        fn bench_sum_impl<T: Dtyped + Sum<Output: Dtyped>>(
            group: &mut BenchmarkGroup<'_, criterion::measurement::WallTime>,
            data: &ndarray::Array2<T>,
            dtype_str: &str,
        ) {
            let array = Array::plain_ndarray_ref(&data).unwrap();
            group.bench_function(BenchmarkId::new(dtype_str, "axis=1 (contiguous)"), |b| {
                b.iter(|| array.as_ref().sum(1usize).to_ndarray().unwrap());
            });
            group.bench_function(BenchmarkId::new(dtype_str, "axis=0 (strided)"), |b| {
                b.iter(|| array.as_ref().sum(0usize).to_ndarray().unwrap());
            });
            group.bench_function(BenchmarkId::new(dtype_str, "all"), |b| {
                b.iter(|| array.as_ref().sum((0usize, 1usize)).to_ndarray().unwrap());
            });
        }

        bench_sum_impl(&mut group, &base, "i32");
        bench_sum_impl(&mut group, &base.mapv(|x| x as i64), "i64");
        bench_sum_impl(&mut group, &base.mapv(|x| x as f32), "f32");
        bench_sum_impl(&mut group, &base.mapv(|x| x as f64), "f64");

        group.finish();
    }
}

fn bench_sum_compact(c: &mut Criterion) {
    for size in [40_000_u64, 400_000, 1_000_000] {
        let shape = [size, 300];
        let mut group = c.benchmark_group(format!("sum compact [{size}, 300]"));
        group.sample_size(20);

        let base = create_data::<i32>(Profile::Smooth, &shape, 0)
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap();

        fn bench_sum_impl<T: Dtyped + Sum<Output: Dtyped>>(
            group: &mut BenchmarkGroup<'_, criterion::measurement::WallTime>,
            data: &ndarray::Array2<T>,
            dtype_str: &str,
        ) {
            let array = Array::compact_ndarray(&data).unwrap();
            group.bench_function(BenchmarkId::new(dtype_str, "axis=1 (contiguous)"), |b| {
                b.iter(|| array.as_ref().sum(1usize).to_ndarray().unwrap());
            });
            group.bench_function(BenchmarkId::new(dtype_str, "axis=0 (strided)"), |b| {
                b.iter(|| array.as_ref().sum(0usize).to_ndarray().unwrap());
            });
            group.bench_function(BenchmarkId::new(dtype_str, "all"), |b| {
                b.iter(|| array.as_ref().sum((0usize, 1usize)).to_ndarray().unwrap());
            });
        }

        bench_sum_impl(&mut group, &base, "i32");

        group.finish();
    }
}

criterion_group!(benches, bench_sum_plain, bench_sum_compact);
criterion_main!(benches);
