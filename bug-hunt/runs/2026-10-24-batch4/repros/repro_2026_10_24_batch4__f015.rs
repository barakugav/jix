//! F015: a negative stride on a never-stepped (length 0/1) ndarray axis is rejected.
//!
//! `strides_from_ndarray` (jix/src/storage/plain.rs:516-524) rejects every negative stride, even on
//! an axis of length <= 1 where the stride is never used. ndarray's `invert_axis` (also used by the
//! `numpy` crate's `as_array` for every negative-stride axis) negates the stride of a length-1
//! axis, so these layouts arise from ordinary safe code. The same function multiplies the stride of
//! a length-1 axis by `size_of::<T>()` unchecked, so a huge (never-stepped) stride panics in debug.
//!
//! Each test asserts the CORRECT behavior (accepted, values equal), so it fails while the bug
//! exists.

use jix::Array;
use ndarray::{array, Array2, ArrayView2, Axis, ShapeBuilder};

/// `invert_axis` on a length-1 axis: logically and physically the same array as before.
#[test]
fn inverted_len1_axis_view_is_accepted() {
    let mut v = array![[1i32, 2, 3]];
    v.invert_axis(Axis(0));
    println!("strides after invert_axis on a length-1 axis: {:?}", v.strides());
    assert!(v.is_standard_layout(), "ndarray itself reports standard layout");

    let a = Array::plain_ndarray_view(v.view());
    println!("plain_ndarray_view: {:?}", a.as_ref().map(|_| ()));
    let a = a.expect("plain_ndarray_view of a length-1 inverted axis should be accepted");
    assert_eq!(a.to_ndarray().unwrap(), array![[1i32, 2, 3]]);
}

#[test]
fn inverted_len1_axis_compact_is_accepted() {
    let mut v = array![[1i32, 2, 3]];
    v.invert_axis(Axis(0));
    let a = Array::compact_ndarray(&v);
    println!("compact_ndarray: {:?}", a.as_ref().map(|_| ()));
    let a = a.expect("compact_ndarray of a length-1 inverted axis should be accepted");
    assert_eq!(a.to_ndarray().unwrap(), array![[1i32, 2, 3]]);
}

#[test]
fn inverted_len1_axis_owned_is_accepted() {
    let mut v: Array2<i32> = array![[1i32, 2, 3]];
    v.invert_axis(Axis(0));
    let a = Array::plain_ndarray(v);
    println!("plain_ndarray: {:?}", a.as_ref().map(|_| ()));
    let a = a.expect("plain_ndarray of a length-1 inverted axis should be accepted");
    assert_eq!(a.to_ndarray().unwrap(), array![[1i32, 2, 3]]);
}

/// A 1-element 1-d array reversed with `invert_axis`.
#[test]
fn inverted_single_element_1d_is_accepted() {
    let mut v = array![7.0f64];
    v.invert_axis(Axis(0));
    let a = Array::plain_ndarray_view(v.view())
        .expect("a reversed 1-element array should be accepted");
    assert_eq!(a.to_ndarray().unwrap(), array![7.0f64]);
}

/// Related report: a huge positive stride on a length-1 axis (accepted by ndarray, never stepped)
/// overflows `s * size_of::<T>()` at plain.rs:522 - a debug panic on a valid view.
#[test]
fn huge_stride_on_len1_axis_does_not_panic() {
    let data = [1i32, 2, 3];
    let v = ArrayView2::from_shape((1, 3).strides((isize::MAX as usize, 1)), &data[..])
        .expect("ndarray accepts any stride on a length-1 axis");
    println!("ndarray view strides: {:?}", v.strides());
    let a = Array::plain_ndarray_view(v).expect("a never-stepped stride should be accepted");
    assert_eq!(a.to_ndarray().unwrap(), array![[1i32, 2, 3]]);
}
