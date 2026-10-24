//! F002 repro: Welford variance goes negative / NaN when |x - mean| overflows f64, and std
//! then trips `debug_assert!(status.is_ok())` in StandardDeviationKernel::finalize_state.
//!
//! Each test asserts the CORRECT result (the variance of values whose squared deviations
//! overflow is +inf, as NumPy returns), so it fails while the bug exists.
use jix::Array;
use jix::scalar::Complex;
use ndarray::{Array1, Ix0, Ix1, array};

fn var_of(v: &[f64]) -> f64 {
    let nd = Array1::from(v.to_vec());
    Array::compact_ndarray(&nd).unwrap().var(0, 0.0).to_ndarray::<Ix0>().unwrap()[[]]
}

#[test]
fn var_two_opposite_huge_is_inf() {
    let r = var_of(&[1e308, -1e308]);
    println!("var([1e308, -1e308]) = {r}");
    assert_eq!(r, f64::INFINITY);
}

#[test]
fn var_three_opposite_huge_is_inf() {
    let r = var_of(&[1e308, -1e308, 0.0]);
    println!("var([1e308, -1e308, 0]) = {r}");
    assert_eq!(r, f64::INFINITY);
}

#[test]
fn var_blocks_of_opposite_huge_is_inf() {
    let mut v = vec![1e308; 40];
    v.extend(vec![-1e308; 40]);
    let r = var_of(&v);
    println!("var([1e308]*40 + [-1e308]*40) = {r}");
    assert_eq!(r, f64::INFINITY);
}

#[test]
fn var_2d_dim1_is_inf() {
    let nd = array![[1.7e308f64, -1.7e308], [1.7e308, -1.7e308], [1.7e308, -1.7e308]];
    let r = Array::compact_ndarray(&nd).unwrap().var(1, 0.0).to_ndarray::<Ix1>().unwrap();
    println!("var([[1.7e308, -1.7e308]]*3, dim=1) = {r}");
    assert!(r.iter().all(|&x| x == f64::INFINITY), "{r}");
}

#[test]
fn var_complex128_is_inf() {
    let nd = array![Complex::new(0.0f64, 1e308), Complex::new(0.0, -1e308)];
    let r = Array::compact_ndarray(&nd).unwrap().var(0, 0.0).to_ndarray::<Ix0>().unwrap()[[]];
    println!("var([1e308j, -1e308j]) = {r}");
    assert_eq!(r, f64::INFINITY);
}

#[test]
fn std_two_opposite_huge_is_inf() {
    // Debug build: panics at the debug_assert in StandardDeviationKernel::finalize_state.
    // Release build: returns NaN.
    let nd = array![1e308f64, -1e308];
    let r = Array::compact_ndarray(&nd).unwrap().std(0, 0.0).to_ndarray::<Ix0>().unwrap()[[]];
    println!("std([1e308, -1e308]) = {r}");
    assert_eq!(r, f64::INFINITY);
}
