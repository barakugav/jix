//! F009 repro: complex `mean` finalizes with num-complex's textbook division
//! `Complex(sum) / Complex(n, 0)` (jix/src/ops/reduction.rs:1766), which forms `sum.re * n` and
//! `sum.im * n` before dividing by `n * n`. Those products overflow to inf whenever
//! |sum component| > f64::MAX / n, although the mean itself is finite.
//! These tests assert the CORRECT behavior (the finite mean NumPy and jix's own f64 mean return),
//! so they fail while the bug exists.
use jix::Array;
use jix::scalar::Complex;
use ndarray::{Ix0, array};

#[test]
fn complex_mean_of_two_large_values_is_finite() {
    let nd = array![Complex::new(1.7e308f64, -1.7e308), Complex::new(0.0, 0.0)];
    let m = Array::compact_ndarray(&nd).unwrap().mean(0).to_ndarray::<Ix0>().unwrap()[[]];
    // control: the real-valued mean of the same components is finite
    let re = Array::compact_ndarray(&array![1.7e308f64, 0.0]).unwrap().mean(0);
    let re = re.to_ndarray::<Ix0>().unwrap()[[]];
    println!("complex mean = {m:e}, f64 mean of the real parts = {re:e}");
    assert_eq!(re, 8.5e307);
    assert_eq!(m, Complex::new(8.5e307, -8.5e307));
}

#[test]
fn complex_mean_of_constant_array_is_the_constant() {
    // 1e6 copies of 1e302+1e302j: the sum (1e308) is finite, the mean is 1e302+1e302j,
    // but sum * n = 1e314 overflows.
    let n = 1_000_000u64;
    let x = Complex::new(1e302f64, 1e302);
    let a = Array::compact_ndarray(&array![x]).unwrap();
    let a = jix::ops::Broadcast::new_array(a, &[n]).unwrap();
    let m = a.mean(0).to_ndarray::<Ix0>().unwrap()[[]];
    println!("mean of {n} x ({x:e}) = {m:e}");
    assert!(m.re.is_finite() && m.im.is_finite(), "mean = {m:e}");
    assert!((m.re / 1e302 - 1.0).abs() < 1e-9 && (m.im / 1e302 - 1.0).abs() < 1e-9, "mean = {m:e}");
}
