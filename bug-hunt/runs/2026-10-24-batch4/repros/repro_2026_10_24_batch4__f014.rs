//! F014: SliceSpec::new / From<&[I]> collect into a DimArray (capacity NDIM_MAX = 8) and panic
//! with "ArrayVec: capacity exceeded" on 9+ items, so the Result-returning Slice::new never gets
//! to report its InvalidIndex for an item count != ndim.
//!
//! The tests assert the CORRECT behavior (Slice::new returns Err(InvalidIndex)), so they fail
//! (panic inside SliceSpec construction) while the bug exists.

use jix::error::ErrorKind;
use jix::ops::{Slice, SliceItem, SliceSpec};
use jix::Array;
use ndarray::array;

fn nine_items() -> Vec<SliceItem> {
    vec![SliceItem::new(None, None, 1); 9]
}

#[test]
fn slice_new_with_9_items_via_slicespec_new_returns_invalid_index() {
    let a = Array::compact_ndarray(&array![[1i32, 2, 3], [4, 5, 6]]).unwrap();
    // Panics here today: ArrayVec: capacity exceeded in extend/from_iter (slice.rs:252).
    let spec = SliceSpec::new(&nine_items());
    let err = match Slice::new_array(a, spec) {
        Ok(_) => panic!("9 slice items on a 2-d array must be rejected"),
        Err(e) => e,
    };
    assert_eq!(err.kind(), ErrorKind::InvalidIndex);
}

#[test]
fn slice_new_with_9_items_via_from_slice_returns_invalid_index() {
    let a = Array::compact_ndarray(&array![[1i32, 2, 3], [4, 5, 6]]).unwrap();
    let ranges: Vec<std::ops::Range<u64>> = (0..9).map(|_| 0u64..1).collect();
    // Panics here today: ArrayVec: capacity exceeded in extend/from_iter (slice.rs:263).
    let spec: SliceSpec = ranges.as_slice().into();
    let err = match Slice::new_array(a, spec) {
        Ok(_) => panic!("9 slice items on a 2-d array must be rejected"),
        Err(e) => e,
    };
    assert_eq!(err.kind(), ErrorKind::InvalidIndex);
}

#[test]
fn contrast_slice_new_with_3_items_returns_invalid_index() {
    // Wrong count <= 8: the documented Err path works.
    let a = Array::compact_ndarray(&array![[1i32, 2, 3], [4, 5, 6]]).unwrap();
    let spec = SliceSpec::new(&vec![SliceItem::new(None, None, 1); 3]);
    let err = match Slice::new_array(a, spec) {
        Ok(_) => panic!("3 slice items on a 2-d array must be rejected"),
        Err(e) => e,
    };
    println!("3 items -> {err:?}");
    assert_eq!(err.kind(), ErrorKind::InvalidIndex);
}
