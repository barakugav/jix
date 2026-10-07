//! Static `Dim<N>` vs `DimDyn` on identical pipelines and data.
//!
//! Every case is run twice: once with sources whose storage carries a static `Dim<N>`, and once
//! with the same data re-tagged as `DimDyn`. Any gap is the cost of dropping the compile-time
//! dimension from `ArrayStorage`.

use std::ops::Range;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use jix::ops::SliceItem;
use jix::{Array, ArrayParams, ReadContext};
use ndarray::{Array2, ArrayD};

fn nd2(n: usize, m: usize, seed: u32) -> Array2<f32> {
    Array2::from_shape_fn((n, m), |(i, j)| {
        ((i as u32).wrapping_mul(31).wrapping_add(j as u32 ^ seed) % 1000) as f32
    })
}

fn out_buf(nitems: usize) -> Vec<u8> {
    vec![0u8; nitems * 4]
}

/// Run `$body` once with each name in `$srcs` bound to the static-dim array and once with it
/// bound to the `DimDyn` array of the same data.
macro_rules! pair {
    ($g:ident, $name:expr, [$(($s:ident, $d:ident)),*], |$($a:ident),*| $body:expr) => {{
        {
            $(let $a = &$s;)*
            $g.bench_function(BenchmarkId::new($name, "static"), |b| b.iter(|| $body));
        }
        {
            $(let $a = &$d;)*
            $g.bench_function(BenchmarkId::new($name, "dyn"), |b| b.iter(|| $body));
        }
    }};
}

fn bench_dim_generic(c: &mut Criterion) {
    let ctx = ReadContext::default();
    let (n, m) = (1024usize, 1024usize);
    let a_nd = nd2(n, m, 1);
    let b_nd = nd2(n, m, 2);
    let col_nd = nd2(n, 1, 3);

    let a_s = Array::plain_ndarray(a_nd.clone()).unwrap();
    let a_d = Array::plain_ndarray(a_nd.clone().into_dyn()).unwrap();
    let b_s = Array::plain_ndarray(b_nd.clone()).unwrap();
    let b_d = Array::plain_ndarray(b_nd.clone().into_dyn()).unwrap();
    let col_s = Array::plain_ndarray(col_nd.clone()).unwrap();
    let col_d = Array::plain_ndarray(col_nd.clone().into_dyn()).unwrap();
    let full: Vec<Range<u64>> = vec![0..n as u64, 0..m as u64];
    let mut buf = out_buf(n * m);

    // -------- whole-array, bulk pipelines (per-element work is ndim-agnostic) --------
    let mut g = c.benchmark_group("bulk [1024,1024] f32");
    g.sample_size(20);
    pair!(g, "a*b+a", [(a_s, a_d), (b_s, b_d)], |a, b| {
        let e = a.view() * b.view() + a.view();
        e.to_ndarray_slice(&full, &mut buf, &ctx).unwrap()
    });
    pair!(
        g,
        "bcast col + a",
        [(a_s, a_d), (col_s, col_d)],
        |a, col| {
            let e = col.view().broadcast(&[n as u64, m as u64]) + a.view();
            e.to_ndarray_slice(&full, &mut buf, &ctx).unwrap()
        }
    );
    pair!(g, "permute(a)+b", [(a_s, a_d), (b_s, b_d)], |a, b| {
        let e = a.view().permute_axes(&[1, 0]) + b.view();
        e.to_ndarray_slice(&full, &mut buf, &ctx).unwrap()
    });
    pair!(g, "sum axis=0", [(a_s, a_d)], |a| a
        .view()
        .sum(0)
        .to_ndarray()
        .unwrap());
    pair!(g, "sum axis=1", [(a_s, a_d)], |a| a
        .view()
        .sum(1)
        .to_ndarray()
        .unwrap());
    pair!(g, "flip axis=0", [(a_s, a_d)], |a| {
        a.view()
            .flip(0)
            .to_ndarray_slice(&full, &mut buf, &ctx)
            .unwrap()
    });
    g.finish();

    // -------- ops whose read loops issue one inner read / copy per element or row --------
    let mut g = c.benchmark_group("per-element loops [1024,1024] f32");
    g.sample_size(10);
    let half: Vec<Range<u64>> = vec![0..n as u64, 0..(m / 2) as u64];
    pair!(g, "slice [:, ::2]", [(a_s, a_d)], |a| {
        let e = a.view().slice((.., SliceItem::new(None, None, 2)));
        e.to_ndarray_slice(&half, &mut buf[..n * m * 2], &ctx)
            .unwrap()
    });
    pair!(g, "flip axis=1", [(a_s, a_d)], |a| {
        a.view()
            .flip(1)
            .to_ndarray_slice(&full, &mut buf, &ctx)
            .unwrap()
    });
    #[allow(clippy::single_range_in_vec_init)]
    let flat = [0..(n * m) as u64];
    {
        let a = &a_s;
        g.bench_function(BenchmarkId::new("permute+reshape flat", "static"), |b| {
            b.iter(|| {
                let e = a.view().permute_axes(&[1, 0]).reshape([(n * m) as u64]);
                e.to_ndarray_slice(&flat, &mut buf, &ctx).unwrap()
            })
        });
    }
    {
        let a = &a_d;
        g.bench_function(BenchmarkId::new("permute+reshape flat", "dyn"), |b| {
            b.iter(|| {
                let e = a
                    .view()
                    .permute_axes(&[1, 0])
                    .reshape(&[(n * m) as u64][..]);
                e.to_ndarray_slice(&flat, &mut buf, &ctx).unwrap()
            })
        });
    }
    g.finish();

    // -------- many tiny reads through a lazy chain (per-read_data-call overhead) --------
    let mut g = c.benchmark_group("tiny reads x10000");
    g.sample_size(20);
    let positions: Vec<(u64, u64)> = (0..10_000u64)
        .map(|k| ((k * 7919) % 1000, (k * 104_729) % 1000))
        .collect();
    let mut small = out_buf(16);
    for side in [1u64, 4] {
        pair!(g, format!("plain {side}x{side}"), [(a_s, a_d)], |a| {
            for &(i, j) in &positions {
                let idx = [i..i + side, j..j + side];
                a.to_ndarray_slice(&idx, &mut small[..(side * side * 4) as usize], &ctx)
                    .unwrap();
            }
        });
        pair!(
            g,
            format!("chain {side}x{side}"),
            [(a_s, a_d), (b_s, b_d), (col_s, col_d)],
            |a, b, col| {
                let e = (a.view().permute_axes(&[1, 0]) + b.view()).slice((1..1020, 2..1022))
                    * col
                        .view()
                        .broadcast(&[n as u64, m as u64])
                        .slice((1..1020, 2..1022));
                for &(i, j) in &positions {
                    let idx = [i..i + side, j..j + side];
                    e.to_ndarray_slice(&idx, &mut small[..(side * side * 4) as usize], &ctx)
                        .unwrap();
                }
            }
        );
    }
    g.finish();

    // -------- compressed storage: read, write, compact_fn --------
    let mut g = c.benchmark_group("compact");
    g.sample_size(10);
    let mut params = ArrayParams::new();
    params.block_shape(&[64, 64]);
    let c_s = Array::compact_ndarray_with(&a_nd, params.clone()).unwrap();
    let c_d = Array::compact_ndarray_with(&a_nd.clone().into_dyn(), params.clone()).unwrap();
    pair!(g, "read 2d full", [(c_s, c_d)], |a| {
        a.to_ndarray_slice(&full, &mut buf, &ctx).unwrap()
    });
    pair!(g, "write 2d", [(a_s, a_d)], |a| a
        .view()
        .compact_with(params.clone(), &ctx)
        .unwrap());

    let sh5 = [16usize, 16, 16, 16, 16];
    let nd5 = ArrayD::from_shape_fn(sh5.as_slice(), |i| (i[0] * 3 + i[4] * 7 + i[2]) as f32);
    let nd5_static = nd5.clone().into_dimensionality::<ndarray::Ix5>().unwrap();
    let mut p5 = ArrayParams::new();
    p5.block_shape(&[4, 4, 4, 4, 4]);
    let c5_s = Array::compact_ndarray_with(&nd5_static, p5.clone()).unwrap();
    let c5_d = Array::compact_ndarray_with(&nd5, p5.clone()).unwrap();
    let full5: Vec<Range<u64>> = sh5.iter().map(|&s| 0..s as u64).collect();
    let mut buf5 = out_buf(sh5.iter().product());
    let sub5: Vec<Range<u64>> = vec![1..15, 2..14, 1..15, 3..13, 1..15];
    pair!(g, "read 5d full", [(c5_s, c5_d)], |a| {
        a.to_ndarray_slice(&full5, &mut buf5, &ctx).unwrap()
    });
    pair!(g, "read 5d unaligned", [(c5_s, c5_d)], |a| {
        let mut b = out_buf(14 * 12 * 14 * 10 * 14);
        a.to_ndarray_slice(&sub5, &mut b, &ctx).unwrap()
    });

    g.bench_function(BenchmarkId::new("compact_fn 2d", "static"), |b| {
        b.iter(|| {
            Array::compact_fn_with([n as u64, m as u64], params.clone(), |(i, j)| {
                (i * 3 + j) as f32
            })
            .unwrap()
        })
    });
    g.bench_function(BenchmarkId::new("compact_fn 2d", "dyn"), |b| {
        b.iter(|| {
            Array::compact_fn_with(&[n as u64, m as u64][..], params.clone(), |i: &[u64]| {
                (i[0] * 3 + i[1]) as f32
            })
            .unwrap()
        })
    });
    g.finish();
}

criterion_group!(benches, bench_dim_generic);
criterion_main!(benches);
