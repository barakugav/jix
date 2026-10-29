//! F029 (run 2026-10-29-batch6): no test strategy produces a `bool8` byte other than 0/1, and no
//! SIMD test covers `bool8`. That is a test-infrastructure GAP; the library itself is correct
//! today. This file is the regression test the gap leaves missing: every op that takes `bool8`
//! input runs on LONG arrays (long enough for the 256/512-lane contiguous bodies of the
//! elementwise pipeline and the vectorized reduction folds, plus a tail) whose bytes include
//! 2, 0x80, 0xFE, 0xFF..., and must agree with the same op on the canonicalized (0/1) array.
//! It passes on the current code; a regression that treats `bool8` as a raw byte in a fast path
//! (bitwise `2 & 1 == 0`, a raw widening cast, a SIMD mask/sum of raw bytes) makes it fail.
//!
//! The gap itself (strategies drawing only 0/1) is crate-internal (`pub(crate)` ScalarStrategy);
//! see bug-hunt/runs/2026-10-29-batch6/repros/F029_internal.rs and F029.py.
use jix::ops::where_condition;
use jix::scalar::bool8;
use jix::{Array, ArrayParams};
use ndarray::{Array1, Array2, Ix0, Ix1, Ix2};

/// Deterministic bytes: ~40% zeros, the rest spread over 1..=255 with the nasty ones forced in.
fn noncanonical_bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed;
    let special = [2u8, 0x80, 0xFE, 0xFF, 4, 0x40, 1, 0];
    (0..n)
        .map(|i| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let r = (s >> 33) as u32;
            if i % 13 == 5 {
                special[(i / 13) % special.len()]
            } else if r % 10 < 4 {
                0
            } else {
                1 + (r % 255) as u8
            }
        })
        .collect()
}

fn canon(v: &[bool8]) -> Vec<bool8> {
    v.iter().map(|&b| bool8::from(bool::from(b))).collect()
}

fn truth(v: &[bool8]) -> Vec<bool> {
    v.iter().map(|&b| bool::from(b)).collect()
}

macro_rules! cast_check {
    ($label:expr, $mk:expr, $a_nd:expr; $($t:ty => $one:expr),*) => {$(
        let got = $mk(&$a_nd).cast::<$t>().to_ndarray::<Ix1>().unwrap();
        let want: Vec<$t> = $a_nd
            .iter()
            .map(|&b| if bool::from(b) { $one } else { <$t>::default() })
            .collect();
        assert_eq!(got.to_vec(), want, "{}: cast to {}", $label, stringify!($t));
    )*};
}

macro_rules! where_check {
    ($label:expr, $mk:expr, $a_nd:expr, $n:expr; $($t:ty),*) => {$(
        let xs: Array1<$t> = (0..$n).map(|i| (i % 100) as $t + (1 as $t)).collect();
        let ys: Array1<$t> = (0..$n).map(|_| <$t>::default()).collect();
        let got = where_condition(
            $mk(&$a_nd),
            Array::compact_ndarray(&xs).unwrap(),
            Array::compact_ndarray(&ys).unwrap(),
        )
        .to_ndarray::<Ix1>()
        .unwrap();
        let want: Vec<$t> = (0..$n)
            .map(|i| if bool::from($a_nd[i]) { xs[i] } else { ys[i] })
            .collect();
        assert_eq!(got.to_vec(), want, "{}: where -> {}", $label, stringify!($t));
    )*};
}

/// Run the whole battery on two equally shaped bool8 arrays built by `$mk` from 1-D ndarrays,
/// once on the non-canonical data and once on its canonical (0/1) copy, and compare.
macro_rules! battery_1d {
    ($label:expr, $mk:expr, $a:expr, $b:expr) => {{
        let label: &str = $label;
        let a_nd: Array1<bool8> = $a;
        let b_nd: Array1<bool8> = $b;
        let ca_nd = Array1::from(canon(a_nd.as_slice().unwrap()));
        let cb_nd = Array1::from(canon(b_nd.as_slice().unwrap()));
        let n = a_nd.len();
        assert!(a_nd.iter().any(|b| b.0 > 1), "input must hold non-canonical bytes");

        macro_rules! cmp_bool_op2 {
            ($name:expr, |$x:ident, $y:ident| $op:expr, $reference:expr) => {{
                let got = {
                    let $x = $mk(&a_nd);
                    let $y = $mk(&b_nd);
                    $op.to_ndarray::<Ix1>().unwrap()
                };
                let want = {
                    let $x = $mk(&ca_nd);
                    let $y = $mk(&cb_nd);
                    $op.to_ndarray::<Ix1>().unwrap()
                };
                let reference: Vec<bool> = (0..n)
                    .map(|i| {
                        let f: &dyn Fn(bool, bool) -> bool = &$reference;
                        f(bool::from(a_nd[i]), bool::from(b_nd[i]))
                    })
                    .collect();
                assert_eq!(
                    truth(got.as_slice().unwrap()),
                    reference,
                    "{label}: {} on non-canonical bytes",
                    $name
                );
                assert_eq!(
                    truth(want.as_slice().unwrap()),
                    reference,
                    "{label}: {} on canonical bytes",
                    $name
                );
            }};
        }
        cmp_bool_op2!("bitand", |x, y| (x & y), |p, q| p & q);
        cmp_bool_op2!("bitor", |x, y| (x | y), |p, q| p | q);
        cmp_bool_op2!("bitxor", |x, y| (x ^ y), |p, q| p ^ q);
        cmp_bool_op2!("equal", |x, y| x.equal(y), |p, q| p == q);
        cmp_bool_op2!("not_equal", |x, y| x.not_equal(y), |p, q| p != q);
        cmp_bool_op2!("less", |x, y| x.less(y), |p, q| !p & q);
        cmp_bool_op2!("less_equal", |x, y| x.less_equal(y), |p, q| !p | q);
        cmp_bool_op2!("greater", |x, y| x.greater(y), |p, q| p & !q);
        cmp_bool_op2!("greater_equal", |x, y| x.greater_equal(y), |p, q| p | !q);
        cmp_bool_op2!("maximum", |x, y| x.maximum(y), |p, q| p | q);
        cmp_bool_op2!("minimum", |x, y| x.minimum(y), |p, q| p & q);

        // Unary NOT.
        let not = (!$mk(&a_nd)).to_ndarray::<Ix1>().unwrap();
        let want: Vec<bool> = a_nd.iter().map(|&b| !bool::from(b)).collect();
        assert_eq!(truth(not.as_slice().unwrap()), want, "{label}: not");

        // Casts from bool8: every non-zero byte must become 1.
        cast_check!(label, $mk, a_nd; i8 => 1, i16 => 1, i32 => 1, i64 => 1, u8 => 1, u16 => 1,
                    u32 => 1, u64 => 1, f32 => 1.0, f64 => 1.0);
        let got = $mk(&a_nd).cast::<bool8>().to_ndarray::<Ix1>().unwrap();
        assert_eq!(truth(got.as_slice().unwrap()), truth(a_nd.as_slice().unwrap()), "{label}: cast to bool8");

        // Reductions.
        let nnz = a_nd.iter().filter(|&&b| bool::from(b)).count() as u64;
        let sum = $mk(&a_nd).sum(0usize).to_ndarray::<Ix0>().unwrap()[[]];
        assert_eq!(sum, nnz, "{label}: sum");
        let mean = $mk(&a_nd).mean(0usize).to_ndarray::<Ix0>().unwrap()[[]];
        let cmean = $mk(&ca_nd).mean(0usize).to_ndarray::<Ix0>().unwrap()[[]];
        assert_eq!(mean, cmean, "{label}: mean");
        let var = $mk(&a_nd).var(0usize, 0.0).to_ndarray::<Ix0>().unwrap()[[]];
        let cvar = $mk(&ca_nd).var(0usize, 0.0).to_ndarray::<Ix0>().unwrap()[[]];
        assert_eq!(var, cvar, "{label}: var");
        let any = $mk(&a_nd).any(0usize).to_ndarray::<Ix0>().unwrap()[[]];
        assert_eq!(bool::from(any), nnz > 0, "{label}: any");
        let all = $mk(&a_nd).all(0usize).to_ndarray::<Ix0>().unwrap()[[]];
        assert_eq!(bool::from(all), nnz == n as u64, "{label}: all");
        let max = $mk(&a_nd).max(0usize).to_ndarray::<Ix0>().unwrap()[[]];
        assert_eq!(bool::from(max), nnz > 0, "{label}: max");
        let min = $mk(&a_nd).min(0usize).to_ndarray::<Ix0>().unwrap()[[]];
        assert_eq!(bool::from(min), nnz == n as u64, "{label}: min");
        let argmax = $mk(&a_nd).argmax(0usize).to_ndarray::<Ix0>().unwrap()[[]];
        let first_true = a_nd.iter().position(|&b| bool::from(b)).unwrap_or(0) as u64;
        assert_eq!(argmax, first_true, "{label}: argmax");
        let argmin = $mk(&a_nd).argmin(0usize).to_ndarray::<Ix0>().unwrap()[[]];
        let first_false = a_nd.iter().position(|&b| !bool::from(b)).unwrap_or(0) as u64;
        assert_eq!(argmin, first_false, "{label}: argmin");

        // where with the non-canonical array as the condition, several value widths.
        where_check!(label, $mk, a_nd, n; u8, i16, i32, f32, f64, u64);
        // where over bool8 values (x, y both non-canonical).
        let got = where_condition($mk(&a_nd), $mk(&b_nd), $mk(&a_nd))
            .to_ndarray::<Ix1>()
            .unwrap();
        let want: Vec<bool> = (0..n)
            .map(|i| bool::from(if bool::from(a_nd[i]) { b_nd[i] } else { a_nd[i] }))
            .collect();
        assert_eq!(truth(got.as_slice().unwrap()), want, "{label}: where over bool8");

        // clamp(a, b, b) == b in truthiness.
        let got = $mk(&a_nd).clamp($mk(&b_nd), $mk(&b_nd)).to_ndarray::<Ix1>().unwrap();
        assert_eq!(truth(got.as_slice().unwrap()), truth(b_nd.as_slice().unwrap()), "{label}: clamp");

        // A fused chain: (a & b) | !a, summed - reads through the pipeline, then reduces.
        let fused = (($mk(&a_nd) & $mk(&b_nd)) | !$mk(&a_nd))
            .sum(0usize)
            .to_ndarray::<Ix0>()
            .unwrap()[[]];
        let want = (0..n)
            .filter(|&i| {
                let (p, q) = (bool::from(a_nd[i]), bool::from(b_nd[i]));
                (p & q) | !p
            })
            .count() as u64;
        assert_eq!(fused, want, "{label}: fused (a & b) | !a summed");
    }};
}

fn inputs(n: usize) -> (Array1<bool8>, Array1<bool8>) {
    let a = noncanonical_bytes(n, 1).into_iter().map(bool8).collect();
    let b = noncanonical_bytes(n, 2).into_iter().map(bool8).collect();
    (a, b)
}

#[test]
fn noncanonical_bool8_long_plain() {
    // 1031 = 2 * 512 + 7: two full 512-lane bodies (AVX-512, 1-byte items) and a scalar tail.
    for n in [1031usize, 4099] {
        let (a, b) = inputs(n);
        battery_1d!(
            &format!("plain n={n}"),
            |x: &Array1<bool8>| Array::plain_ndarray(x.clone()).unwrap(),
            a,
            b
        );
    }
}

#[test]
fn noncanonical_bool8_long_compact() {
    for n in [1031usize, 4099] {
        let (a, b) = inputs(n);
        battery_1d!(
            &format!("compact n={n}"),
            |x: &Array1<bool8>| Array::compact_ndarray(x).unwrap(),
            a,
            b
        );
    }
}

#[test]
fn noncanonical_bool8_long_compact_small_blocks() {
    let (a, b) = inputs(1031);
    battery_1d!(
        "compact block 7",
        |x: &Array1<bool8>| {
            let mut p = ArrayParams::new();
            p.block_shape(&[7]);
            Array::compact_ndarray_with(x, p).unwrap()
        },
        a,
        b
    );
}

#[test]
fn noncanonical_bool8_strided_2d_reductions() {
    // Permuted (strided) reads and reductions along each dim.
    let (r, c) = (37usize, 61usize);
    let bytes = noncanonical_bytes(r * c, 3);
    let nd = Array2::from_shape_vec((r, c), bytes.iter().map(|&x| bool8(x)).collect()).unwrap();
    let cnd = nd.mapv(|b| bool8::from(bool::from(b)));
    let a =Array::compact_ndarray(&nd).unwrap();
    let ca = Array::compact_ndarray(&cnd).unwrap();
    let t = a.view().permute_dims(&[1, 0]);
    let ct = ca.view().permute_dims(&[1, 0]);
    for dim in [0usize, 1] {
        let s = t.view().sum(dim).to_ndarray::<Ix1>().unwrap();
        let cs = ct.view().sum(dim).to_ndarray::<Ix1>().unwrap();
        assert_eq!(s, cs, "strided sum dim {dim}");
        let any = t.view().any(dim).to_ndarray::<Ix1>().unwrap();
        let cany = ct.view().any(dim).to_ndarray::<Ix1>().unwrap();
        assert_eq!(truth(&any.to_vec()), truth(&cany.to_vec()), "strided any dim {dim}");
        let all = t.view().all(dim).to_ndarray::<Ix1>().unwrap();
        let call = ct.view().all(dim).to_ndarray::<Ix1>().unwrap();
        assert_eq!(truth(&all.to_vec()), truth(&call.to_vec()), "strided all dim {dim}");
        let am = t.view().argmax(dim).to_ndarray::<Ix1>().unwrap();
        let cam = ct.view().argmax(dim).to_ndarray::<Ix1>().unwrap();
        assert_eq!(am, cam, "strided argmax dim {dim}");
        let m = t.view().mean(dim).to_ndarray::<Ix1>().unwrap();
        let cm = ct.view().mean(dim).to_ndarray::<Ix1>().unwrap();
        assert_eq!(m, cm, "strided mean dim {dim}");
    }
    let n = (!t.view()).to_ndarray::<Ix2>().unwrap();
    let cn = (!ct.view()).to_ndarray::<Ix2>().unwrap();
    assert_eq!(n, cn, "strided not");
    let x = (t.view() & ct.view()).cast::<u32>().to_ndarray::<Ix2>().unwrap();
    let cx = (ct.view() & ct.view()).cast::<u32>().to_ndarray::<Ix2>().unwrap();
    assert_eq!(x, cx, "strided and+cast");
}
