// Repro for bug-hunt run 2026-10-27-batch5, finding F010.
//
// std of f16 / f32 / Complex<f32> rounds the f64 variance to the narrow output dtype before
// taking the square root, so the result is inf, 0 or many ulps off even though the correctly
// rounded std is representable. Each test asserts the CORRECT result (within 1 ulp of
// sqrt(f64 variance) rounded once to the output dtype) and fails while the bug exists.
//
// cargo test --manifest-path jix/Cargo.toml --all-features --test repro_2026_10_27_batch5__f010 -- --nocapture
#![cfg(all(feature = "half", feature = "num-complex"))]

use jix::Array;
use jix::scalar::{Complex, f16};
use ndarray::{Array1, Ix0, array};

/// Population std in f64. Inputs are chosen so that every step is exact or nearly so.
fn std_ref(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let m = x.iter().sum::<f64>() / n;
    (x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / n).sqrt()
}

fn f16_ulps(a: f16, b: f16) -> i32 {
    // Both values are finite and non-negative here, so bit patterns are ordered.
    (a.to_bits() as i32 - b.to_bits() as i32).abs()
}

fn f32_ulps(a: f32, b: f32) -> i64 {
    (a.to_bits() as i64 - b.to_bits() as i64).abs()
}

fn std_f16(x: &[f64]) -> (f16, f16) {
    let xs: Vec<f16> = x.iter().map(|&v| f16::from_f64(v)).collect();
    let exact: Vec<f64> = xs.iter().map(|v| v.to_f64()).collect();
    let want = f16::from_f64(std_ref(&exact));
    let a = Array::compact_ndarray(&Array1::from(xs)).unwrap();
    let got = a.std(0usize, 0.0).to_ndarray::<Ix0>().unwrap()[[]];
    (got, want)
}

#[test]
fn f16_std_near_one_is_correctly_rounded() {
    // Three consecutive f16 values near 1.0: var = 2^-19 / 3 = 6.36e-7 is an f16 subnormal.
    let u = 2f64.powi(-10);
    let (got, want) = std_f16(&[1.0, 1.0 + u, 1.0 + 2.0 * u]);
    println!("f16 [1, 1+2^-10, 1+2^-9]: got {got}, want {want}");
    assert!(f16_ulps(got, want) <= 1, "got {got}, want {want}");
}

#[test]
fn f16_std_near_0_01_is_not_zero() {
    // Two adjacent f16 values near 0.01: var = 1.4e-11 underflows f16 to 0.
    let a = f16::from_f64(0.01);
    let b = f16::from_bits(a.to_bits() + 1);
    let (got, want) = std_f16(&[a.to_f64(), b.to_f64()]);
    println!("f16 [0.01, next]: got {got}, want {want}");
    assert!(f16_ulps(got, want) <= 1, "got {got}, want {want}");
}

#[test]
fn f16_std_does_not_overflow() {
    // var = 90000 > f16::MAX (65504), std = 300 is exact in f16.
    let (got, want) = std_f16(&[-300.0, 300.0]);
    println!("f16 [-300, 300]: got {got}, want {want}");
    assert_eq!(got.to_f64(), 300.0, "got {got}, want {want}");
}

#[test]
fn f32_std_does_not_overflow() {
    // var = 4e38 > f32::MAX, std = 2e19.
    let a = Array::compact_ndarray(&array![-2e19f32, 2e19]).unwrap();
    let got = a.std(0usize, 0.0).to_ndarray::<Ix0>().unwrap()[[]];
    println!("f32 [-2e19, 2e19]: got {got}, want 2e19");
    assert_eq!(got, 2e19f32);
}

#[test]
fn f32_std_does_not_underflow() {
    // var = 2.5e-51 < f32 smallest subnormal, std = 5e-26.
    let x = [1e-25f32, 2e-25];
    let want = std_ref(&x.map(|v| v as f64)) as f32;
    let a = Array::compact_ndarray(&Array1::from(x.to_vec())).unwrap();
    let got = a.std(0usize, 0.0).to_ndarray::<Ix0>().unwrap()[[]];
    println!("f32 [1e-25, 2e-25]: got {got}, want {want}");
    assert!(f32_ulps(got, want) <= 1, "got {got}, want {want}");
}

#[test]
fn complex64_std_does_not_overflow() {
    let a = Array::compact_ndarray(&array![
        Complex::new(-2e19f32, 0.0),
        Complex::new(2e19f32, 0.0)
    ])
    .unwrap();
    let got = a.std(0usize, 0.0).to_ndarray::<Ix0>().unwrap()[[]];
    println!("complex64 [-2e19, 2e19]: got {got}, want 2e19");
    assert_eq!(got, 2e19f32);
}

#[test]
fn f16_std_agrees_with_f64_std_rounded() {
    // Cross-dtype consistency: std on the f16 array vs std on the same values as f64,
    // rounded once to f16.
    let u = 2f64.powi(-10);
    let xs: Vec<f16> = [1.0, 1.0 + u, 1.0 + 2.0 * u]
        .iter()
        .map(|&v| f16::from_f64(v))
        .collect();
    let x64: Vec<f64> = xs.iter().map(|v| v.to_f64()).collect();
    let got16 = Array::compact_ndarray(&Array1::from(xs))
        .unwrap()
        .std(0usize, 0.0)
        .to_ndarray::<Ix0>()
        .unwrap()[[]];
    let got64 = Array::compact_ndarray(&Array1::from(x64))
        .unwrap()
        .std(0usize, 0.0)
        .to_ndarray::<Ix0>()
        .unwrap()[[]];
    let via64 = f16::from_f64(got64);
    println!("f16 std {got16} vs f64 std rounded to f16 {via64}");
    assert!(f16_ulps(got16, via64) <= 1, "f16 std {got16}, f64 std -> f16 {via64}");
}
