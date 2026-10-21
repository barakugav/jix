//! F012: ArrayParams::block_shape / block_shape_fixed_dims panic on > NDIM_MAX entries
//! instead of reporting TooManyDimensions through the Result-based API.
use jix::{Array, ArrayParams};
use ndarray::array;
use std::panic::{AssertUnwindSafe, catch_unwind};

#[test]
fn block_shape_9_entries_panics() {
    let r = catch_unwind(AssertUnwindSafe(|| {
        let mut p = ArrayParams::new();
        p.block_shape(&[1u32; 9]);
    }));
    println!("block_shape(&[1; 9]) panicked: {}", r.is_err());
    assert!(r.is_err());
}

#[test]
fn block_shape_fixed_dims_9_entries_panics() {
    let r = catch_unwind(AssertUnwindSafe(|| {
        let mut p = ArrayParams::new();
        p.block_shape_fixed_dims(&[true; 9]);
    }));
    println!("block_shape_fixed_dims(&[true; 9]) panicked: {}", r.is_err());
    assert!(r.is_err());
}

#[test]
fn mismatched_but_small_block_shape_is_a_clean_error() {
    // Contrast: a wrong-length (but <= 8) block shape is reported as an Err by tune().
    let a = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let mut p = ArrayParams::new();
    p.block_shape(&[1, 1, 1]);
    let r = Array::compact_ndarray_with(&a, p);
    println!("3-entry block_shape on 2-D array: {:?}", r.as_ref().err());
    assert!(r.is_err());
    let mut p = ArrayParams::new();
    p.block_shape(&[1, 1]).block_shape_fixed_dims(&[true; 3]);
    let r = Array::compact_ndarray_with(&a, p);
    println!("3-entry block_shape_fixed_dims on 2-D array: {:?}", r.as_ref().err());
    assert!(r.is_err());
}
