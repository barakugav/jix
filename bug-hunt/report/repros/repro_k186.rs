//! K186 (run 2026-11-16-scan-106-107): since the K168 fix (#107), `approx_equal` ANDs
//! `diff.is_finite()` into the result, so finite operands whose difference overflows are never
//! close - even when the documented formula `|a - b| <= max(atol, rtol * max(|a|, |b|))` holds
//! exactly: atol = inf, rtol = inf, or ANY finite rtol >= 2 (|a - b| <= |a| + |b| <= 2 * max).
//! Asserts the documented (correct) result, so it fails while the bug exists.
use jix::scalar::{bool8, f16};
use jix::Array;
use ndarray::array;

const T: bool8 = bool8::TRUE;
const F: bool8 = bool8::FALSE;

#[test]
fn f003_f64_overflowing_diff_with_tolerance_that_covers_it() {
    let a = || Array::compact_ndarray(&array![f64::MAX, 1.0, f64::MAX]).unwrap();
    let b = || Array::compact_ndarray(&array![-f64::MAX, -1.0, f64::MAX / 2.0]).unwrap();
    // (rtol, atol, expected): exact |MAX - (-MAX)| = 2 * MAX.
    let cases = [
        (0.0, f64::INFINITY, [T, T, T]), // 2 * MAX <= inf
        (f64::INFINITY, 0.0, [T, T, T]), // 2 * MAX <= inf * MAX
        (2.0, 0.0, [T, T, T]),           // 2 * MAX <= 2 * MAX (finite rtol!)
        (3.0, 0.0, [T, T, T]),           // 2 * MAX <= 3 * MAX
        (1.5, 0.0, [F, F, T]),           // K168 stays fixed: 2 * MAX > 1.5 * MAX (and 2 > 1.5)
    ];
    let mut bad = Vec::new();
    for (rtol, atol, expected) in cases {
        let got = a().approx_equal(b(), rtol, atol).to_ndarray().unwrap();
        println!("f64 rtol={rtol} atol={atol}: got {got:?}, expected {expected:?}");
        if got != ndarray::Array1::from(expected.to_vec()) {
            bad.push((rtol, atol));
        }
    }
    assert!(bad.is_empty(), "wrong for (rtol, atol) = {bad:?}");
}

#[test]
fn f003_f16_ordinary_magnitudes() {
    // In f16 the difference of +-40000 (80000 > 65504) already overflows.
    let h = f16::from_f32;
    let a = || Array::compact_ndarray(&array![h(40000.0), h(1.0)]).unwrap();
    let b = || Array::compact_ndarray(&array![h(-40000.0), h(-1.0)]).unwrap();
    let cases = [
        (h(2.0), h(0.0), [T, T]),           // 80000 <= 2 * 40000
        (h(3.0), h(0.0), [T, T]),           // 80000 <= 3 * 40000
        (h(0.0), f16::INFINITY, [T, T]),    // 80000 <= inf (a Python atol >= 65520 casts to inf)
        (h(1.5), h(0.0), [F, F]),           // K168 stays fixed: 80000 > 60000 (and 2 > 1.5)
    ];
    let mut bad = Vec::new();
    for (rtol, atol, expected) in cases {
        let got = a().approx_equal(b(), rtol, atol).to_ndarray().unwrap();
        println!("f16 rtol={rtol} atol={atol}: got {got:?}, expected {expected:?}");
        if got != ndarray::Array1::from(expected.to_vec()) {
            bad.push((rtol, atol));
        }
    }
    assert!(bad.is_empty(), "wrong for (rtol, atol) = {bad:?}");
}

#[test]
fn f003_rtol_between_1_and_2_when_exactly_close() {
    // An overflowing |a - b| is not always outside rtol < 2: exact |a - b| = |a| + |b| is within
    // rtol * max(|a|, |b|) iff rtol >= 1 + min/max. MAX vs -MAX/4 needs rtol >= 1.25.
    let got = Array::compact_ndarray(&array![f64::MAX, f64::MAX])
        .unwrap()
        .approx_equal(
            Array::compact_ndarray(&array![-f64::MAX / 4.0, -f64::MAX]).unwrap(),
            1.5,
            0.0,
        )
        .to_ndarray()
        .unwrap();
    println!("f64 MAX vs [-MAX/4, -MAX] rtol=1.5: {got:?}");
    // f16: 50000 vs -20000 -> exact 70000 <= 1.5 * 50000 = 75000.
    let h = f16::from_f32;
    let got16 = Array::compact_ndarray(&array![h(50000.0)])
        .unwrap()
        .approx_equal(Array::compact_ndarray(&array![h(-20000.0)]).unwrap(), h(1.5), h(0.0))
        .to_ndarray()
        .unwrap();
    println!("f16 50000 vs -20000 rtol=1.5: {got16:?}");
    assert_eq!(got, array![T, F]);
    assert_eq!(got16, array![T]);
}
