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

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use jix::storage::{CompactParams, Filter};
use jix::{Array, CompactArray};
use ndarray::{ArrayD, Axis, IxDyn, Slice};

use crate::common::{Profile, create_data};

/// The canonical array for the whole report, matching `report_spec.SHAPE`.
const SHAPE: [u64; 2] = [130_000, 200];
const SEED: u64 = 0x5eed_1234_abcd_ef01;

/// Criterion's default of 100 samples is far too many for 100 MB arrays.
const SAMPLES: usize = 10;

/// Block-compressed storage with the settings the report pins: zstd level 3, byte-shuffle.
fn compact<T: jix::dtype::Dtyped>(
    data: &ArrayD<T>,
    block_shape: Option<&[u32]>,
) -> CompactArray<T> {
    let mut params = CompactParams::new()
        .level(3)
        .unwrap()
        .filters(&[Filter::ByteShuffle])
        .unwrap();
    if let Some(block_shape) = block_shape {
        params = params.block_shape(block_shape).unwrap();
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
            b.iter(|| plain.to_ndarray_sub::<IxDyn>(iter.next().unwrap()).unwrap());
        });

        let mut iter = regions.iter().cycle();
        group.bench_function(BenchmarkId::new("jix", case), |b| {
            b.iter(|| array.to_ndarray_sub::<IxDyn>(iter.next().unwrap()).unwrap());
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
            b.iter(|| plain.view().neg().to_ndarray::<IxDyn>().unwrap());
        });
        $group.bench_function(BenchmarkId::new("jix", $case), |b| {
            b.iter(|| array.view().neg().to_ndarray::<IxDyn>().unwrap());
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
            b.iter(|| {
                (plain_l.view() + plain_r.view())
                    .to_ndarray::<IxDyn>()
                    .unwrap()
            });
        });
        $group.bench_function(BenchmarkId::new("jix", $case), |b| {
            b.iter(|| {
                (compact_l.view() + compact_r.view())
                    .to_ndarray::<IxDyn>()
                    .unwrap()
            });
        });
    }};
}

/// `sum` over one dim. The all-dims variant is separate because jix spells it differently.
macro_rules! bench_sum_dim {
    ($group:expr, $case:expr, $data:expr, $dim:expr) => {{
        let data = $data;
        let dim = $dim;
        let plain = Array::plain_ndarray_ref(&data).unwrap();
        let array = compact(&data, None);
        $group.bench_function(BenchmarkId::new("ndarray", $case), |b| {
            b.iter(|| data.sum_axis(Axis(dim)));
        });
        $group.bench_function(BenchmarkId::new("jix-plain", $case), |b| {
            b.iter(|| plain.view().sum(dim).to_ndarray::<IxDyn>().unwrap());
        });
        $group.bench_function(BenchmarkId::new("jix", $case), |b| {
            b.iter(|| array.view().sum(dim).to_ndarray::<IxDyn>().unwrap());
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
            b.iter(|| plain.view().sum((0, 1)).to_ndarray::<IxDyn>().unwrap());
        });
        $group.bench_function(BenchmarkId::new("jix", $case), |b| {
            b.iter(|| array.view().sum((0, 1)).to_ndarray::<IxDyn>().unwrap());
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
    bench_sum_dim!(
        group,
        "sum_f32_dim0",
        create_data::<f32>(Profile::Smooth, &SHAPE, SEED),
        0
    );
    bench_sum_dim!(
        group,
        "sum_f32_dim1",
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

// The same chain the Python suite runs, over array operands only - see `array_impls.CHAIN_STEPS`.
// Starting from `a`, the cycle builds up the shape of `a*a + b*b + a*b`. No scalar constants: real
// operands keep the chain comparable with the Python suite, and literal constants can fold away
// (see scripts/bench/DEVELOPER.md).
//
// Unrolled because each step produces a distinct storage type, so the chain cannot be built in a
// loop. `.view()` on every use because the operand types are not `Clone`.
macro_rules! jix_chain {
    ($a:expr, $b:expr, 1) => {
        $a.view() * $a.view()
    };
    ($a:expr, $b:expr, 2) => {
        jix_chain!($a, $b, 1) + $b.view()
    };
    ($a:expr, $b:expr, 3) => {
        jix_chain!($a, $b, 2) * $b.view()
    };
    ($a:expr, $b:expr, 4) => {
        jix_chain!($a, $b, 3) - $a.view()
    };
    ($a:expr, $b:expr, 5) => {
        jix_chain!($a, $b, 4) * $a.view()
    };
    ($a:expr, $b:expr, 6) => {
        jix_chain!($a, $b, 5) + $b.view()
    };
    ($a:expr, $b:expr, 7) => {
        jix_chain!($a, $b, 6) * $b.view()
    };
    ($a:expr, $b:expr, 8) => {
        jix_chain!($a, $b, 7) - $a.view()
    };
    ($a:expr, $b:expr, 9) => {
        jix_chain!($a, $b, 8) * $a.view()
    };
    ($a:expr, $b:expr, 10) => {
        jix_chain!($a, $b, 9) + $b.view()
    };
    ($a:expr, $b:expr, 11) => {
        jix_chain!($a, $b, 10) * $b.view()
    };
    ($a:expr, $b:expr, 12) => {
        jix_chain!($a, $b, 11) - $a.view()
    };
    ($a:expr, $b:expr, 13) => {
        jix_chain!($a, $b, 12) * $a.view()
    };
    ($a:expr, $b:expr, 14) => {
        jix_chain!($a, $b, 13) + $b.view()
    };
    ($a:expr, $b:expr, 15) => {
        jix_chain!($a, $b, 14) * $b.view()
    };
    ($a:expr, $b:expr, 16) => {
        jix_chain!($a, $b, 15) - $a.view()
    };
}

/// `ndarray` in its efficient form: an owned left operand reuses its buffer, so this does not
/// allocate per step - but it still makes one full read-write pass per step, which is what fusing
/// the chain avoids.
fn ndarray_chain(a: &ArrayD<f32>, b: &ArrayD<f32>, steps: usize) -> ArrayD<f32> {
    let mut out = a * a;
    for index in 1..steps {
        out = match index % 4 {
            1 => out + b,
            2 => out * b,
            3 => out - a,
            _ => out * a,
        };
    }
    out
}

macro_rules! bench_chain_len {
    ($group:expr, $steps:tt, $a_data:expr, $b_data:expr, $a_plain:expr, $b_plain:expr, $a_arr:expr, $b_arr:expr) => {{
        let case = concat!($steps, "op");
        $group.bench_function(BenchmarkId::new("ndarray", case), |b| {
            b.iter(|| ndarray_chain($a_data, $b_data, $steps));
        });
        $group.bench_function(BenchmarkId::new("jix-plain", case), |b| {
            b.iter(|| {
                jix_chain!($a_plain, $b_plain, $steps)
                    .to_ndarray::<IxDyn>()
                    .unwrap()
            });
        });
        $group.bench_function(BenchmarkId::new("jix", case), |b| {
            b.iter(|| {
                jix_chain!($a_arr, $b_arr, $steps)
                    .to_ndarray::<IxDyn>()
                    .unwrap()
            });
        });
    }};
}

fn bench_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("report_rust_chain");
    group.sample_size(SAMPLES);
    let data = create_data::<f32>(Profile::Smooth, &SHAPE, SEED);
    let other = create_data::<f32>(Profile::Smooth, &SHAPE, SEED ^ 1);
    let plain = Array::plain_ndarray_ref(&data).unwrap();
    let plain_other = Array::plain_ndarray_ref(&other).unwrap();
    let array = compact(&data, None);
    let array_other = compact(&other, None);

    bench_chain_len!(
        group,
        1,
        &data,
        &other,
        plain,
        plain_other,
        array,
        array_other
    );
    bench_chain_len!(
        group,
        2,
        &data,
        &other,
        plain,
        plain_other,
        array,
        array_other
    );
    bench_chain_len!(
        group,
        4,
        &data,
        &other,
        plain,
        plain_other,
        array,
        array_other
    );
    bench_chain_len!(
        group,
        8,
        &data,
        &other,
        plain,
        plain_other,
        array,
        array_other
    );
    bench_chain_len!(
        group,
        16,
        &data,
        &other,
        plain,
        plain_other,
        array,
        array_other
    );

    // `normalize` mixes elementwise work with a reduction and a broadcast - the shape that is
    // awkward to hand-write as an iterator chain, which is jix's actual pitch in Rust.
    let shape = SHAPE.to_vec();
    let shape_usize = [SHAPE[0] as usize, SHAPE[1] as usize];
    for dim in [0usize, 1] {
        let case = format!("normalize_dim{dim}");
        group.bench_function(BenchmarkId::new("ndarray", &case), |b| {
            let view = data.view().into_dimensionality::<ndarray::Ix2>().unwrap();
            b.iter(|| {
                let std = view.std_axis(Axis(dim), 0.0).insert_axis(Axis(dim));
                let std = std.broadcast(shape_usize).unwrap();
                ndarray::Zip::from(&view)
                    .and(&std)
                    .map_collect(|&a, &s| a / s)
            });
        });
        group.bench_function(BenchmarkId::new("jix-plain", &case), |b| {
            b.iter(|| {
                let std = plain.view().std(dim, 0.0).insert_dim(dim).broadcast(&shape);
                (plain.view() / std).to_ndarray::<IxDyn>().unwrap()
            });
        });
        group.bench_function(BenchmarkId::new("jix", &case), |b| {
            b.iter(|| {
                let std = array.view().std(dim, 0.0).insert_dim(dim).broadcast(&shape);
                (array.view() / std).to_ndarray::<IxDyn>().unwrap()
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------------------------
// report_rust_dim_order - jix sorts dims by stride; ndarray classifies layout four ways.
// ---------------------------------------------------------------------------------------------

/// jix sorts every dim by descending stride before iterating. `ndarray` has two paths, and they
/// treat layouts differently:
///
/// - `map` (unary) takes a flat fast path whenever the view is contiguous *in memory order* -
///   `is_contiguous` sorts the strides first - so any pure permutation of a contiguous array is
///   fast, and the output keeps the input's strides. Only a genuinely non-contiguous view, such as
///   a slice, falls back to `iter()`, which walks logical index order however the strides run.
/// - `Zip` (binary ops, `assign`, `map_collect`) only knows C and F layouts. Anything else, such as
///   a `[0,2,1]` permutation, takes the strided path, which unrolls over the last logical axis
///   whatever its stride - for `[0,2,1]` that is a full row per element.
///
/// `-a` on a permuted view.
macro_rules! bench_dim_order_neg {
    ($group:expr, $case:expr, $data:expr, $keep:expr, $dims:expr) => {{
        let data = $data;
        let keep: usize = $keep;
        let dims: &[usize] = $dims;
        let sliced = data.slice_each_axis(|ax| {
            if ax.axis.index() == 2 {
                Slice::from(0..keep)
            } else {
                Slice::from(0..ax.len)
            }
        });
        let permuted = sliced.permuted_axes(dims);
        $group.bench_function(BenchmarkId::new("ndarray", $case), |b| {
            b.iter(|| permuted.map(|&x| x.neg()));
        });

        let plain = Array::plain_ndarray_ref(&data).unwrap();
        let shape = data.shape().to_vec();
        $group.bench_function(BenchmarkId::new("jix-plain", $case), |b| {
            b.iter(|| {
                plain
                    .view()
                    .slice((0..shape[0] as i64, 0..shape[1] as i64, 0..keep as i64))
                    .permute_dims(dims)
                    .neg()
                    .to_ndarray::<IxDyn>()
                    .unwrap()
            });
        });
    }};
}

/// `a + b` on permuted views, with the same `Zip` call `report_rust_op` uses for its `add`.
macro_rules! bench_dim_order_add {
    ($group:expr, $case:expr, $lhs:expr, $lhs_dims:expr, $rhs:expr, $rhs_dims:expr) => {{
        let (lhs, rhs) = ($lhs, $rhs);
        let (lhs_dims, rhs_dims): (&[usize], &[usize]) = ($lhs_dims, $rhs_dims);
        let (lhs_view, rhs_view) = (
            lhs.view().permuted_axes(lhs_dims),
            rhs.view().permuted_axes(rhs_dims),
        );
        $group.bench_function(BenchmarkId::new("ndarray", $case), |b| {
            b.iter(|| {
                ndarray::Zip::from(&lhs_view)
                    .and(&rhs_view)
                    .map_collect(|&a, &b| a + b)
            });
        });

        let (plain_l, plain_r) = (
            Array::plain_ndarray_ref(&lhs).unwrap(),
            Array::plain_ndarray_ref(&rhs).unwrap(),
        );
        $group.bench_function(BenchmarkId::new("jix-plain", $case), |b| {
            b.iter(|| {
                (plain_l.view().permute_dims(lhs_dims) + plain_r.view().permute_dims(rhs_dims))
                    .to_ndarray::<IxDyn>()
                    .unwrap()
            });
        });
    }};
}

fn bench_dim_order(c: &mut Criterion) {
    let mut group = c.benchmark_group("report_rust_dim_order");
    group.sample_size(SAMPLES);

    let shape3 = [300u64, 400, 500];
    let data = || create_data::<i32>(Profile::Smooth, &shape3, SEED);
    let other = || create_data::<i32>(Profile::Smooth, &shape3, SEED ^ 1);

    // Control: C layout, which `Zip` runs as one flat loop.
    bench_dim_order_add!(group, "add_c", data(), &[0, 1, 2], other(), &[0, 1, 2]);
    // Neither C nor F: `Zip` walks the last logical axis, a 500-element stride.
    bench_dim_order_add!(group, "add_021", data(), &[0, 2, 1], other(), &[0, 2, 1]);
    // Control: `map` sees a contiguous array in memory order and keeps its strides.
    bench_dim_order_neg!(group, "neg_021", data(), 500, &[0, 2, 1]);
    // Slicing the last dim makes the view non-contiguous, so `map` falls back to `iter()`.
    bench_dim_order_neg!(group, "neg_sliced_120", data(), 250, &[1, 2, 0]);

    group.finish();
}

criterion_group!(benches, bench_read, bench_op, bench_chain, bench_dim_order);
criterion_main!(benches);
