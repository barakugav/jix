//! K108: var/std of finite data whose squared deviations overflow returns NaN for 3+ elements,
//! where numpy (and jix for 2 elements) returns +inf. Written while consolidating the scans at main
//! 1169965 (#106); see bug-hunt/report/findings/K108.md.
//!
//! Asserts the CORRECT result (+inf), so it fails while the bug exists. Run:
//!   cargo test --manifest-path jix/Cargo.toml --all-features --test repro_k108 -- --nocapture
use jix::Array;
use ndarray::{Ix0, array};

#[test]
fn k108_var_of_overflowing_deviations_is_inf() {
    let a = Array::compact_ndarray(&array![1e308f64, -1e308, 0.0]).unwrap();
    let v = a.var(0usize, 0.0).to_ndarray::<Ix0>().unwrap()[[]];
    let mut big = vec![1e308f64; 40];
    big.extend(vec![-1e308f64; 40]);
    let b = Array::compact_ndarray(&ndarray::Array1::from(big)).unwrap();
    let w = b.var(0usize, 0.0).to_ndarray::<Ix0>().unwrap()[[]];
    println!("var([1e308, -1e308, 0]) = {v}, var([1e308]*40 + [-1e308]*40) = {w} (numpy: inf, inf)");
    assert_eq!((v, w), (f64::INFINITY, f64::INFINITY));
}
