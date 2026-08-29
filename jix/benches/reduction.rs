mod common;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use jix::Array;

use crate::common::{create_data, Profile};

fn bench_sum_plain(c: &mut Criterion) {
    fn bench_sum_plain_impl(c: &mut Criterion, size: u64, all_dtypes: bool) {
        let shape = [size, 300];
        let mut group = c.benchmark_group(format!("sum plain [{size}, 300]"));
        group.sample_size(20);

        let base = create_data::<i32>(Profile::Smooth, &shape, 0x2c4f558960d6c384)
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap();

        let arr_i32 = Array::plain_ndarray_ref(&base).unwrap();
        group.bench_function(BenchmarkId::new("i32", "axis=1 (contiguous)"), |b| {
            b.iter(|| arr_i32.as_ref().sum(1).to_ndarray().unwrap());
        });
        group.bench_function(BenchmarkId::new("i32", "axis=0 (strided)"), |b| {
            b.iter(|| arr_i32.as_ref().sum(0).to_ndarray().unwrap());
        });
        group.bench_function(BenchmarkId::new("i32", "all"), |b| {
            b.iter(|| arr_i32.as_ref().sum((0, 1)).to_ndarray().unwrap());
        });

        if all_dtypes {
            let arr_i64 = Array::plain_ndarray(base.mapv(|x| x as i64)).unwrap();
            group.bench_function(BenchmarkId::new("i64", "axis=1 (contiguous)"), |b| {
                b.iter(|| arr_i64.as_ref().sum(1).to_ndarray().unwrap());
            });
            group.bench_function(BenchmarkId::new("i64", "axis=0 (strided)"), |b| {
                b.iter(|| arr_i64.as_ref().sum(0).to_ndarray().unwrap());
            });
            let arr_f32 = Array::plain_ndarray(base.mapv(|x| x as f32)).unwrap();
            group.bench_function(BenchmarkId::new("f32", "axis=1 (contiguous)"), |b| {
                b.iter(|| arr_f32.as_ref().sum(1).to_ndarray().unwrap());
            });
            group.bench_function(BenchmarkId::new("f32", "axis=0 (strided)"), |b| {
                b.iter(|| arr_f32.as_ref().sum(0).to_ndarray().unwrap());
            });
            let arr_f64 = Array::plain_ndarray(base.mapv(|x| x as f64)).unwrap();
            group.bench_function(BenchmarkId::new("f64", "axis=1 (contiguous)"), |b| {
                b.iter(|| arr_f64.as_ref().sum(1).to_ndarray().unwrap());
            });
            group.bench_function(BenchmarkId::new("f64", "axis=0 (strided)"), |b| {
                b.iter(|| arr_f64.as_ref().sum(0).to_ndarray().unwrap());
            });
        }

        group.finish();
    }

    bench_sum_plain_impl(c, 40_000, false);
    bench_sum_plain_impl(c, 400_000, false);
    bench_sum_plain_impl(c, 1_000_000, true);
}

fn bench_sum_compact(c: &mut Criterion) {
    for size in [40_000_u64, 400_000, 1_000_000] {
        let shape = [size, 300];
        let mut group = c.benchmark_group(format!("sum compact [{size}, 300]"));
        group.sample_size(20);

        let data = create_data::<i32>(Profile::Smooth, &shape, 0xddd8f5ce716490be)
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap();

        let array = Array::compact_ndarray(&data).unwrap();
        group.bench_function(BenchmarkId::new("i32", "axis=1 (contiguous)"), |b| {
            b.iter(|| array.as_ref().sum(1).to_ndarray().unwrap());
        });
        group.bench_function(BenchmarkId::new("i32", "axis=0 (strided)"), |b| {
            b.iter(|| array.as_ref().sum(0).to_ndarray().unwrap());
        });
        group.bench_function(BenchmarkId::new("i32", "all"), |b| {
            b.iter(|| array.as_ref().sum((0, 1)).to_ndarray().unwrap());
        });

        group.finish();
    }
}

criterion_group!(benches, bench_sum_plain, bench_sum_compact);
criterion_main!(benches);
