// Repro for bug-hunt run 2026-10-27-batch5, finding F003.
//
// `MapMultiple::new` is fallible (returns `Result`) but panics with
// "index out of bounds: the len is 0 but the index is 0" (storage/params.rs, `combine_hints`
// indexing `inputs[0]`) when given an empty array sequence. `Stack::new` / `Concatenate::new`
// return an Err for the same input. These tests assert the correct behavior (an Err), so they
// FAIL (by panicking) while the bug exists.
use jix::storage::Compact;
use jix::{Array, Ty};

type A = Array<Compact<Ty<f32>>>;

#[test]
fn map_multiple_new_empty_vec_is_err() {
    let arrays: Vec<A> = Vec::new();
    let r = jix::ops::MapMultiple::new(arrays, |xs: &[f32]| xs.len() as f32);
    assert!(r.is_err(), "MapMultiple::new(empty Vec) must return Err");
}

#[test]
fn map_multiple_new_empty_slice_is_err() {
    let arrays: Vec<A> = Vec::new();
    let r = jix::ops::MapMultiple::new(&arrays[..], |xs: &[f32]| xs.len() as f32);
    assert!(r.is_err(), "MapMultiple::new(empty slice) must return Err");
}

#[test]
fn map_multiple_new_empty_fixed_array_is_err() {
    let arrays: [A; 0] = [];
    let r = jix::ops::MapMultiple::new(arrays, |_xs: [f32; 0]| 1.0f32);
    assert!(r.is_err(), "MapMultiple::new([A; 0]) must return Err");
}

// Control: the sibling ArraySequence consumers already reject an empty sequence with an Err.
#[test]
fn control_stack_and_concatenate_empty_are_err() {
    let arrays: Vec<A> = Vec::new();
    assert!(jix::ops::Stack::new(arrays, 0).is_err());
    let arrays: Vec<A> = Vec::new();
    assert!(jix::ops::Concatenate::new(arrays, 0).is_err());
}
