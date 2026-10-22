//! F014 repro: complex division (`Div` for `Complex<f32>` / `Complex<f64>`, op2.rs:781) is
//! num-complex's textbook formula, whose `c*c + d*d` denominator overflows / underflows for
//! finite divisors far from 1. Asserts the CORRECT quotient, so it fails while the bug exists.
#![cfg(feature = "num-complex")]

use jix::Array;
use jix::scalar::Complex;
use ndarray::{Ix1, array};

// num-complex is built without `std` here, so no `Complex::norm`.
fn close64(got: Complex<f64>, want: Complex<f64>, rtol: f64) -> bool {
    let err = (got.re - want.re).hypot(got.im - want.im);
    got.re.is_finite() && got.im.is_finite() && err <= rtol * want.re.hypot(want.im)
}

fn close32(got: Complex<f32>, want: Complex<f32>, rtol: f32) -> bool {
    let err = (got.re - want.re).hypot(got.im - want.im);
    got.re.is_finite() && got.im.is_finite() && err <= rtol * want.re.hypot(want.im)
}

#[test]
fn complex128_division_large_and_small_divisors() {
    // (dividend, divisor, exact quotient)
    let cases = [
        (Complex::new(1e160, 1e160), Complex::new(1e160, 1e160), Complex::new(1.0, 0.0)),
        (Complex::new(1.0, 1.0), Complex::new(1e160, 1e160), Complex::new(1e-160, 0.0)),
        (Complex::new(1.0, 1.0), Complex::new(1e-170, 1e-170), Complex::new(1e170, 0.0)),
        (Complex::new(3.0, 4.0), Complex::new(1e-160, 0.0), Complex::new(3e160, 4e160)),
    ];
    let a = Array::compact_ndarray(&array![cases[0].0, cases[1].0, cases[2].0, cases[3].0])
        .unwrap();
    let b = Array::compact_ndarray(&array![cases[0].1, cases[1].1, cases[2].1, cases[3].1])
        .unwrap();
    let got = (a / b).to_ndarray::<Ix1>().unwrap();
    let mut bad = Vec::new();
    for (i, &(x, y, want)) in cases.iter().enumerate() {
        println!("complex128 {x} / {y}: want {want}, got {}", got[i]);
        if !close64(got[i], want, 1e-15) {
            bad.push(i);
        }
    }
    assert!(bad.is_empty(), "wrong complex128 quotients at {bad:?}: {got:?}");
}

#[test]
fn complex64_division_large_and_small_divisors() {
    let cases = [
        (Complex::new(1e20f32, 1e20), Complex::new(1e20f32, 1e20), Complex::new(1.0f32, 0.0)),
        (Complex::new(1.0f32, 1.0), Complex::new(1e20f32, 1e20), Complex::new(1e-20f32, 0.0)),
        (Complex::new(1.0f32, 1.0), Complex::new(1e-23f32, 1e-23), Complex::new(1e23f32, 0.0)),
        (Complex::new(3.0f32, 4.0), Complex::new(1e-20f32, 0.0), Complex::new(3e20f32, 4e20)),
    ];
    let a = Array::compact_ndarray(&array![cases[0].0, cases[1].0, cases[2].0, cases[3].0])
        .unwrap();
    let b = Array::compact_ndarray(&array![cases[0].1, cases[1].1, cases[2].1, cases[3].1])
        .unwrap();
    let got = (a / b).to_ndarray::<Ix1>().unwrap();
    let mut bad = Vec::new();
    for (i, &(x, y, want)) in cases.iter().enumerate() {
        println!("complex64 {x} / {y}: want {want}, got {}", got[i]);
        if !close32(got[i], want, 1e-6) {
            bad.push(i);
        }
    }
    assert!(bad.is_empty(), "wrong complex64 quotients at {bad:?}: {got:?}");
}

/// `Complex<f16>` has no dtype (no Array of it), but the public scalar trait
/// `jix::scalar::Div` is implemented for it (op2.rs:786) with the same formula; there
/// `sqrt(f16::MAX)` is only ~256, so ordinary magnitudes already overflow.
#[cfg(feature = "half")]
#[test]
fn complex_f16_scalar_division() {
    use jix::scalar::{Div, f16};
    let mut status = Default::default();
    let x = Complex::new(f16::from_f32(300.0), f16::from_f32(0.0));
    let q = Div::div(x, x, &mut status);
    println!("Complex<f16> {x} / {x} = {q}");
    assert_eq!((q.re.to_f32(), q.im.to_f32()), (1.0, 0.0), "got {q}");
}
