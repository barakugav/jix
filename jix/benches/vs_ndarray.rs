//! The benchmarks behind the published report: jix against `ndarray`, on identical work.
//!
//! Kept apart from the rest of `benches/`, which exists to optimize the library and has no place in
//! a report. Every group here is named `report_<section>` and every benchmark id is
//! `<library>/<case>`, which is the contract `benches/report_bars.py` parses. Case ids are
//! filesystem-safe; `report_spec.py` maps them to the labels that appear on the plots.
//!
//! Run just this target with `python jix/benches/run.py --report`.
//!
//! The arms are written out per library rather than behind a generic helper. Each jix operation
//! produces a distinct storage type, so a generic version needs a trait bound per operation and
//! stops being readable long before it stops being possible.

mod common;

use std::ops::Neg;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use jix::storage::Compact;
use jix::{Array, ArrayParams, DimDyn, Filter, Ty};
use ndarray::{Array2, ArrayD, Axis, IxDyn, Slice};

use crate::common::{create_data, Profile};

/// The canonical array for the whole report, matching `report_spec.SHAPE`.
const SHAPE: [u64; 2] = [130_000, 200];
const SEED: u64 = 0x5eed_1234_abcd_ef01;

/// Criterion's default of 100 samples is far too many for 100 MB arrays.
const SAMPLES: usize = 10;

/// Block-compressed storage with the settings the report pins: zstd level 3, byte-shuffle.
fn compact<T>(data: &ArrayD<T>, block_shape: Option<&[u32]>) -> Array<Compact<Ty<T>, DimDyn>>
where
    T: jix::dtype::Dtyped + Copy + Send + Sync + 'static,
{
    let mut params = ArrayParams::new();
    params.level(3).unwrap();
    params.filters(&[Filter::ByteShuffle]).unwrap();
    if let Some(block_shape) = block_shape {
        params.block_shape(block_shape);
    }
    Array::compact_ndarray_with(data, params).unwrap()
}

// ---------------------------------------------------------------------------------------------
// report_rust_read - pulling a region out of a compressed array, against slicing a plain ndarray.
// ---------------------------------------------------------------------------------------------

fn random_regions(
    shape: &[u64],
    read_shape: &[u64],
    count: usize,
    seed: u64,
) -> Vec<Vec<std::ops::Range<u64>>> {
    let mut rng = fastrand::Rng::with_seed(seed);
    (0..count)
        .map(|_| {
            shape
                .iter()
                .zip(read_shape)
                .map(|(&dim, &want)| {
                    let size = want.min(dim);
                    let start = rng.u64(0..=(dim - size));
                    start..start + size
                })
                .collect()
        })
        .collect()
}

fn bench_read(c: &mut Criterion) {
    // (case id, block shape, read shape)
    let cases: [(&str, [u32; 2], [u64; 2]); 3] = [
        ("b32x32_r32x32", [32, 32], [32, 32]),
        ("b32x32_r1x200", [32, 32], [1, 200]),
        ("b512x32_r128x200", [512, 32], [128, 200]),
    ];

    let mut group = c.benchmark_group("report_rust_read");
    group.sample_size(SAMPLES);
    let data = create_data::<i32>(Profile::Smooth, &SHAPE, SEED);
    let plain = Array::plain_ndarray_ref(&data).unwrap();

    for (case, block_shape, read_shape) in cases {
        // Pre-generate the regions so the timed call is a read and nothing else. Cycling through
        // many of them keeps the working set larger than cache, which is the point of the case.
        let regions = random_regions(&SHAPE, &read_shape, 1024, SEED ^ 0x9e37);
        let array = compact(&data, Some(block_shape.as_slice()));

        let mut iter = regions.iter().cycle();
        group.bench_function(BenchmarkId::new("ndarray", case), |b| {
            b.iter(|| {
                let region = iter.next().unwrap();
                data.slice_each_axis(|ax| {
                    let range = &region[ax.axis.index()];
                    Slice::from(range.start as usize..range.end as usize)
                })
                .to_owned()
            });
        });

        let mut iter = regions.iter().cycle();
        group.bench_function(BenchmarkId::new("jix-plain", case), |b| {
            let ctx = plain.read_ctx();
            b.iter(|| plain.to_ndarray_sub(iter.next().unwrap(), &ctx).unwrap());
        });

        let mut iter = regions.iter().cycle();
        group.bench_function(BenchmarkId::new("jix", case), |b| {
            let ctx = array.read_ctx();
            b.iter(|| array.to_ndarray_sub(iter.next().unwrap(), &ctx).unwrap());
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------------------------
// report_rust_op - elementwise operations and reductions.
// ---------------------------------------------------------------------------------------------

/// Negate, three ways. A macro rather than a function so each expansion is monomorphic.
macro_rules! bench_negate {
    ($group:expr, $case:expr, $data:expr) => {{
        let data = $data;
        let plain = Array::plain_ndarray_ref(&data).unwrap();
        let array = compact(&data, None);
        $group.bench_function(BenchmarkId::new("ndarray", $case), |b| {
            b.iter(|| data.map(|&x| x.neg()));
        });
        $group.bench_function(BenchmarkId::new("jix-plain", $case), |b| {
            b.iter(|| plain.view().neg().to_ndarray().unwrap());
        });
        $group.bench_function(BenchmarkId::new("jix", $case), |b| {
            b.iter(|| array.view().neg().to_ndarray().unwrap());
        });
    }};
}

macro_rules! bench_add {
    ($group:expr, $case:expr, $lhs:expr, $rhs:expr) => {{
        let (lhs, rhs) = ($lhs, $rhs);
        let (plain_l, plain_r) = (
            Array::plain_ndarray_ref(&lhs).unwrap(),
            Array::plain_ndarray_ref(&rhs).unwrap(),
        );
        let (compact_l, compact_r) = (compact(&lhs, None), compact(&rhs, None));
        $group.bench_function(BenchmarkId::new("ndarray", $case), |b| {
            // The idiomatic form: a fresh allocation for the result.
            b.iter(|| {
                ndarray::Zip::from(&lhs)
                    .and(&rhs)
                    .map_collect(|&a, &b| a + b)
            });
        });
        $group.bench_function(BenchmarkId::new("jix-plain", $case), |b| {
            b.iter(|| (plain_l.view() + plain_r.view()).to_ndarray().unwrap());
        });
        $group.bench_function(BenchmarkId::new("jix", $case), |b| {
            b.iter(|| (compact_l.view() + compact_r.view()).to_ndarray().unwrap());
        });
    }};
}

/// `sum` over one axis. The all-axes variant is separate because jix spells it differently.
macro_rules! bench_sum_axis {
    ($group:expr, $case:expr, $data:expr, $axis:expr) => {{
        let data = $data;
        let axis = $axis;
        let plain = Array::plain_ndarray_ref(&data).unwrap();
        let array = compact(&data, None);
        $group.bench_function(BenchmarkId::new("ndarray", $case), |b| {
            b.iter(|| data.sum_axis(Axis(axis)));
        });
        $group.bench_function(BenchmarkId::new("jix-plain", $case), |b| {
            b.iter(|| plain.view().sum(axis).to_ndarray().unwrap());
        });
        $group.bench_function(BenchmarkId::new("jix", $case), |b| {
            b.iter(|| array.view().sum(axis).to_ndarray().unwrap());
        });
    }};
}

macro_rules! bench_sum_all {
    ($group:expr, $case:expr, $data:expr) => {{
        let data = $data;
        let plain = Array::plain_ndarray_ref(&data).unwrap();
        let array = compact(&data, None);
        $group.bench_function(BenchmarkId::new("ndarray", $case), |b| {
            b.iter(|| data.sum());
        });
        $group.bench_function(BenchmarkId::new("jix-plain", $case), |b| {
            b.iter(|| plain.view().sum((0, 1)).to_ndarray().unwrap());
        });
        $group.bench_function(BenchmarkId::new("jix", $case), |b| {
            b.iter(|| array.view().sum((0, 1)).to_ndarray().unwrap());
        });
    }};
}

fn bench_op(c: &mut Criterion) {
    let mut group = c.benchmark_group("report_rust_op");
    group.sample_size(SAMPLES);

    bench_negate!(
        group,
        "negate_f32",
        create_data::<f32>(Profile::Smooth, &SHAPE, SEED)
    );
    bench_negate!(
        group,
        "negate_i32",
        create_data::<i32>(Profile::Smooth, &SHAPE, SEED)
    );
    bench_add!(
        group,
        "add_f32",
        create_data::<f32>(Profile::Smooth, &SHAPE, SEED),
        create_data::<f32>(Profile::Smooth, &SHAPE, SEED ^ 1)
    );
    bench_add!(
        group,
        "add_i32",
        create_data::<i32>(Profile::Smooth, &SHAPE, SEED),
        create_data::<i32>(Profile::Smooth, &SHAPE, SEED ^ 1)
    );
    bench_sum_axis!(
        group,
        "sum_f32_axis0",
        create_data::<f32>(Profile::Smooth, &SHAPE, SEED),
        0
    );
    bench_sum_axis!(
        group,
        "sum_f32_axis1",
        create_data::<f32>(Profile::Smooth, &SHAPE, SEED),
        1
    );
    bench_sum_all!(
        group,
        "sum_i32_all",
        create_data::<i32>(Profile::Smooth, &SHAPE, SEED)
    );

    group.finish();
}

// ---------------------------------------------------------------------------------------------
// report_rust_chain - the cost of intermediates, as the chain gets longer.
// ---------------------------------------------------------------------------------------------

// The same alternating multiply/add chain the Python suite runs, so the two halves agree. jix has
// no scalar-operand operators, so each step is a `map` - which is still a separate lazy wrapper,
// and that is exactly the thing being measured.
macro_rules! jix_chain {
    ($a:expr, 1) => {
        $a.map(|x: f32| x * 2.0)
    };
    ($a:expr, 2) => {
        jix_chain!($a, 1).map(|x: f32| x + 1.0)
    };
    ($a:expr, 3) => {
        jix_chain!($a, 2).map(|x: f32| x * 0.5)
    };
    ($a:expr, 4) => {
        jix_chain!($a, 3).map(|x: f32| x - 3.0)
    };
    ($a:expr, 5) => {
        jix_chain!($a, 4).map(|x: f32| x * 2.0)
    };
    ($a:expr, 6) => {
        jix_chain!($a, 5).map(|x: f32| x + 1.0)
    };
    ($a:expr, 7) => {
        jix_chain!($a, 6).map(|x: f32| x * 0.5)
    };
    ($a:expr, 8) => {
        jix_chain!($a, 7).map(|x: f32| x - 3.0)
    };
}

/// `ndarray` has no lazy views, so every step allocates. That is the comparison.
fn ndarray_chain(data: &ArrayD<f32>, steps: usize) -> ArrayD<f32> {
    let mut out = data.mapv(|x| x * 2.0);
    for index in 1..steps {
        out = match index % 4 {
            1 => out.mapv(|x| x + 1.0),
            2 => out.mapv(|x| x * 0.5),
            3 => out.mapv(|x| x - 3.0),
            _ => out.mapv(|x| x * 2.0),
        };
    }
    out
}

macro_rules! bench_chain_len {
    ($group:expr, $steps:tt, $data:expr, $plain:expr, $array:expr) => {{
        let case = concat!($steps, "op");
        $group.bench_function(BenchmarkId::new("ndarray", case), |b| {
            b.iter(|| ndarray_chain($data, $steps));
        });
        $group.bench_function(BenchmarkId::new("jix-plain", case), |b| {
            b.iter(|| jix_chain!($plain.view(), $steps).to_ndarray().unwrap());
        });
        $group.bench_function(BenchmarkId::new("jix", case), |b| {
            b.iter(|| jix_chain!($array.view(), $steps).to_ndarray().unwrap());
        });
    }};
}

fn bench_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("report_rust_chain");
    group.sample_size(SAMPLES);
    let data = create_data::<f32>(Profile::Smooth, &SHAPE, SEED);
    let plain = Array::plain_ndarray_ref(&data).unwrap();
    let array = compact(&data, None);

    bench_chain_len!(group, 1, &data, plain, array);
    bench_chain_len!(group, 2, &data, plain, array);
    bench_chain_len!(group, 4, &data, plain, array);
    bench_chain_len!(group, 8, &data, plain, array);

    // `normalize` mixes elementwise work with a reduction and a broadcast - the shape that is
    // awkward to hand-write as an iterator chain, which is jix's actual pitch in Rust.
    let shape = SHAPE.to_vec();
    let shape_usize = [SHAPE[0] as usize, SHAPE[1] as usize];
    for axis in [0usize, 1] {
        let case = format!("normalize_axis{axis}");
        group.bench_function(BenchmarkId::new("ndarray", &case), |b| {
            let view = data.view().into_dimensionality::<ndarray::Ix2>().unwrap();
            b.iter(|| {
                let std = view.std_axis(Axis(axis), 0.0).insert_axis(Axis(axis));
                let std = std.broadcast(shape_usize).unwrap();
                ndarray::Zip::from(&view)
                    .and(&std)
                    .map_collect(|&a, &s| a / s)
            });
        });
        group.bench_function(BenchmarkId::new("jix-plain", &case), |b| {
            b.iter(|| {
                let std = plain
                    .view()
                    .std(axis, 0.0)
                    .insert_axis(axis)
                    .broadcast(&shape);
                (plain.view() / std).to_ndarray().unwrap()
            });
        });
        group.bench_function(BenchmarkId::new("jix", &case), |b| {
            b.iter(|| {
                let std = array
                    .view()
                    .std(axis, 0.0)
                    .insert_axis(axis)
                    .broadcast(&shape);
                (array.view() / std).to_ndarray().unwrap()
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------------------------
// report_rust_axis_order - jix sorts axes by stride; ndarray classifies layout four ways.
// ---------------------------------------------------------------------------------------------

/// `ndarray`'s `array_layout` returns C, F, first-axis-contiguous, last-axis-contiguous, or
/// nothing, and on nothing it iterates in logical order however the strides actually run. A 3-D
/// rotation lands on nothing with the largest stride innermost; a reversal is exactly F layout and
/// a 2-D transpose is too, so both of those are controls that should show no difference.
macro_rules! bench_permuted {
    ($group:expr, $case:expr, $data:expr, $axes:expr) => {{
        let data = $data;
        let axes: &[usize] = $axes;
        let permuted = data.view().permuted_axes(axes);
        $group.bench_function(BenchmarkId::new("ndarray", $case), |b| {
            b.iter(|| permuted.map(|&x| x.neg()));
        });
        let plain = Array::plain_ndarray_ref(&data).unwrap();
        $group.bench_function(BenchmarkId::new("jix-plain", $case), |b| {
            b.iter(|| plain.view().permute_axes(axes).neg().to_ndarray().unwrap());
        });
    }};
}

fn bench_axis_order(c: &mut Criterion) {
    let mut group = c.benchmark_group("report_rust_axis_order");
    group.sample_size(SAMPLES);

    let shape3 = [300u64, 400, 500];
    bench_permuted!(
        group,
        "rotate_f32",
        create_data::<f32>(Profile::Smooth, &shape3, SEED),
        &[1, 2, 0]
    );
    bench_permuted!(
        group,
        "rotate_i32",
        create_data::<i32>(Profile::Smooth, &shape3, SEED),
        &[1, 2, 0]
    );
    bench_permuted!(
        group,
        "reverse_f32",
        create_data::<f32>(Profile::Smooth, &shape3, SEED),
        &[2, 1, 0]
    );
    bench_permuted!(
        group,
        "transpose2d_f32",
        Array2::<f32>::from_shape_fn((1200, 1200), |(i, j)| (i * 1200 + j) as f32)
            .into_dimensionality::<IxDyn>()
            .unwrap(),
        &[1, 0]
    );

    group.finish();
}

criterion_group!(benches, bench_read, bench_op, bench_chain, bench_axis_order);
criterion_main!(benches);
