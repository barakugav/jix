//! F021: an empty float / complex `sum` returns -0.0 (sign bit set) instead of +0.0.
//!
//! `Sum::init()` is `cast(-0.0)` (jix/src/ops/reduction.rs:1482-1483), and an empty reduction
//! writes `finalize(init_state(None), 0)` = `init()` (reduction.rs:1260-1269). The doc
//! (reduction.rs:2381) says an empty sum returns `0`; NumPy gives +0.0. These tests assert the
//! correct (+0.0) result, so they FAIL while the bug exists.

use jix::Array;
use jix::scalar::{Complex, f16};
use ndarray::{Array1, Array2, Ix0, Ix1};

#[test]
fn empty_f64_sum_is_positive_zero() {
    let a = Array::compact_ndarray(&Array2::<f64>::zeros((3, 0))).unwrap();
    let s = a.sum(1).to_ndarray::<Ix1>().unwrap();
    println!("f64 sum over empty dim: {s:?}, 1/x = {:?}", s.mapv(|x| 1.0 / x));
    for v in s.iter() {
        assert_eq!(v.to_bits(), 0.0f64.to_bits(), "empty f64 sum gave {v:?}");
    }
}

#[test]
fn empty_f32_sum_all_dims_is_positive_zero() {
    let a = Array::compact_ndarray(&Array1::<f32>::zeros(0)).unwrap();
    let s = a.sum(0).to_ndarray::<Ix0>().unwrap()[()];
    println!("f32 sum of empty 1-d array: {s:?}");
    assert_eq!(s.to_bits(), 0.0f32.to_bits(), "empty f32 sum gave {s:?}");
}

#[test]
fn empty_f16_sum_is_positive_zero() {
    let a = Array::compact_ndarray(&Array1::<f16>::from(vec![])).unwrap();
    let s = a.sum(0).to_ndarray::<Ix0>().unwrap()[()];
    println!("f16 sum of empty 1-d array: {s:?}");
    assert_eq!(s.to_bits(), f16::from_f32(0.0).to_bits(), "empty f16 sum gave {s:?}");
}

#[test]
fn empty_complex_sum_is_positive_zero() {
    let a = Array::compact_ndarray(&Array2::<Complex<f64>>::zeros((2, 0))).unwrap();
    let s = a.sum(1).to_ndarray::<Ix1>().unwrap();
    println!("Complex<f64> sum over empty dim: {s:?}");
    for v in s.iter() {
        assert_eq!((v.re.to_bits(), v.im.to_bits()), (0, 0), "empty complex sum gave {v:?}");
    }
}

#[test]
fn integer_and_nonempty_sums_are_fine() {
    // Control: integer empty sums are 0, and a non-empty sum of +0.0 is +0.0.
    let a = Array::compact_ndarray(&Array2::<i32>::zeros((2, 0))).unwrap();
    assert!(a.sum(1).to_ndarray::<Ix1>().unwrap().iter().all(|&v| v == 0));
    let a = Array::compact_ndarray(&Array1::<f64>::zeros(4)).unwrap();
    let s = a.sum(0).to_ndarray::<Ix0>().unwrap()[()];
    assert_eq!(s.to_bits(), 0.0f64.to_bits(), "sum of +0.0s gave {s:?}");
}
