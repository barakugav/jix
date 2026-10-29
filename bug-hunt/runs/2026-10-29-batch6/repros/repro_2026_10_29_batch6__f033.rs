//! F033: an ndarray with ZERO elements is rejected by the Plain/compact constructors when a dim of
//! length > 1 has a negative stride, although no element is ever addressed (gap in the K80 fix,
//! which only skips the check on dims of length <= 1).
//!
//! Asserts the CORRECT behavior, so every test fails while the bug exists.
//! Run: cargo test --manifest-path jix/Cargo.toml --all-features --test repro_2026_10_29_batch6__f033 -- --nocapture

use jix::Array;
use ndarray::{s, Array2, Axis, Ix2};

/// `a[::-1, 1:1]` of a (4, 3) array: shape (4, 0), strides [-3, 0].
fn empty_reversed_view(base: &Array2<f64>) -> ndarray::ArrayView2<'_, f64> {
    let v = base.slice(s![..;-1, 1..1]);
    println!(
        "view shape {:?} strides {:?} standard_layout {}",
        v.shape(),
        v.strides(),
        v.is_standard_layout()
    );
    v
}

#[test]
fn plain_ndarray_view_empty_reversed() {
    let base = Array2::<f64>::zeros((4, 3));
    let v = empty_reversed_view(&base);
    let r = Array::plain_ndarray_view(v);
    println!("plain_ndarray_view: {:?}", r.as_ref().map(|a| a.shape().to_vec()));
    let a = r.expect("an empty view must be accepted");
    assert_eq!(a.shape(), &[4u64, 0]);
    assert_eq!(a.to_ndarray::<Ix2>().unwrap().shape(), &[4, 0]);
}

#[test]
fn plain_ndarray_ref_empty_reversed() {
    let base = Array2::<f64>::zeros((4, 3));
    let v = empty_reversed_view(&base);
    let r = Array::plain_ndarray_ref(&v);
    println!("plain_ndarray_ref: {:?}", r.as_ref().map(|a| a.shape().to_vec()));
    let a = r.expect("an empty view must be accepted");
    assert_eq!(a.shape(), &[4u64, 0]);
}

#[test]
fn compact_ndarray_empty_reversed() {
    let base = Array2::<f64>::zeros((4, 3));
    let v = empty_reversed_view(&base);
    let r = Array::compact_ndarray(&v);
    println!("compact_ndarray: {:?}", r.as_ref().map(|a| a.shape().to_vec()));
    let a = r.expect("an empty view must be accepted");
    assert_eq!(a.to_ndarray::<Ix2>().unwrap().shape(), &[4, 0]);
}

#[test]
fn plain_ndarray_owned_empty_inverted() {
    // An owned (4, 3) array with its length-4 axis inverted, then sliced down to (4, 0) in place.
    let mut owned = Array2::<f64>::zeros((4, 3));
    owned.invert_axis(Axis(0));
    owned.slice_collapse(s![.., 1..1]);
    println!("owned shape {:?} strides {:?}", owned.shape(), owned.strides());
    let r = Array::plain_ndarray(owned);
    println!("plain_ndarray: {:?}", r.as_ref().map(|a| a.shape().to_vec()));
    let a = r.expect("an empty owned array must be accepted");
    assert_eq!(a.shape(), &[4u64, 0]);
}
