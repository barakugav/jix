mod common;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use jix::storage::Compact;
use jix::{Array, ArrayParams, DimDyn, Ty};

use crate::common::{create_data, Profile};

const SIZES: [(u64, u64); 4] = [(4_000, 200), (40_000, 200), (200_000, 64), (20_000, 2_000)];

fn compact_f32(shape: &[u64], seed: u64) -> Array<Compact<Ty<f32>, DimDyn>> {
    let data = create_data::<f32>(Profile::Smooth, shape, seed);
    Array::compact_ndarray_with(&data, ArrayParams::new()).unwrap()
}

fn bench_normalize(c: &mut Criterion) {
    for axis in [0usize, 1] {
        let mut group = c.benchmark_group(format!("normalize_std axis={axis}"));
        group.sample_size(10);
        for (n, m) in SIZES {
            let shape = [n, m];
            let a = compact_f32(&shape, 0x9e3779b97f4a7c15 ^ n);
            group.bench_function(BenchmarkId::from_parameter(format!("{shape:?}")), |b| {
                b.iter(|| {
                    let d = a.view() / a.view().std(axis, 0.0).insert_axis(axis).broadcast(&shape);
                    d.to_ndarray().unwrap()
                });
            });
        }
        group.finish();
    }
}

criterion_group!(benches, bench_normalize);
criterion_main!(benches);
