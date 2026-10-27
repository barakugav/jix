//! F011: RemoveDim::new / new_array must return Err (not panic) for a DimsArg with more than
//! NDIM_MAX (8) entries. Fails (panics at dimension.rs dim_arr assert) while the bug exists.

use jix::Array;
use ndarray::array;

#[test]
fn remove_dim_nine_repeated_dims_is_err() {
    let a = Array::compact_ndarray(&array![[1.0f32]]).unwrap();
    let r = jix::ops::RemoveDim::new_array(a, &[0usize; 9][..]);
    assert!(r.is_err(), "expected Err for 9 (duplicate) dims");
}

#[test]
fn remove_dim_nine_dims_vec_is_err() {
    // Same via Vec<usize> (a DimsArg impl provided by the library).
    let a = Array::compact_ndarray(&array![[1i32, 2, 3]]).unwrap();
    let r = jix::ops::RemoveDim::new_array(a, vec![0usize, 1, 2, 3, 4, 5, 6, 7, 8]);
    assert!(r.is_err(), "expected Err for 9 dims");
}

#[test]
fn control_flip_and_insert_dim_return_err() {
    // Sibling DimsArg consumers already return Err on the same input.
    let a = || Array::compact_ndarray(&array![[1.0f32]]).unwrap();
    let f = jix::ops::Flip::new_array(a(), &[0usize; 9][..]);
    println!("flip: {:?}", f.as_ref().err().map(|e| e.to_string()));
    assert!(f.is_err());
    let i = jix::ops::InsertDim::new_array(a(), &[0usize; 9][..]);
    println!("insert_dim: {:?}", i.as_ref().err().map(|e| e.to_string()));
    assert!(i.is_err());
}
