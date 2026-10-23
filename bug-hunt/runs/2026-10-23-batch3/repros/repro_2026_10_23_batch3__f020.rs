//! F020 repro: a reversed index range passed straight to the safe `ArrayStorage::read_data` of an
//! op that calls `check_out_buf` (-> `check_region_size`, which computes `r.end - r.start`)
//! before validating `start <= end`. On 32-bit targets this panics with "attempt to subtract with
//! overflow" in a debug build and returns `InvalidShapeOperation` instead of `InvalidIndex` in a
//! release build. 64-bit is unaffected (the check is compiled out there).
//!
//! Asserts the CORRECT behavior (`Err(InvalidIndex)`), so it fails on i686 while the bug exists:
//!   bug-hunt/tools/cargo-i686 test --manifest-path jix/Cargo.toml --all-features \
//!       --test repro_2026_10_23_batch3__f020 -- --nocapture
#![allow(clippy::single_range_in_vec_init, clippy::reversed_empty_ranges)]

use std::ops::Range;

use jix::error::ErrorKind;
use jix::ops::where_condition;
use jix::{Array, ArrayStorage, ReadContext};
use ndarray::array;

fn kind_of<S: ArrayStorage>(name: &str, a: &Array<S>, index: &[Range<u64>]) -> Option<ErrorKind> {
    let ctx = ReadContext::default();
    let r = a.storage().read_data(index, &ctx, None).map(|_| ());
    let k = r.err().map(|e| e.kind());
    println!("{name}: {k:?}");
    k
}

#[test]
fn op1_neg_reversed_range() {
    let a = Array::compact_ndarray(&array![1i32, 2, 3, 4]).unwrap();
    assert_eq!(kind_of("neg", &(-a.view()), &[3..1]), Some(ErrorKind::InvalidIndex));
}

#[test]
fn op2_add_reversed_range() {
    let a = Array::compact_ndarray(&array![1i32, 2, 3, 4]).unwrap();
    let s = a.view() + a.view();
    assert_eq!(kind_of("add", &s, &[3..1]), Some(ErrorKind::InvalidIndex));
}

#[test]
fn insert_dim_reversed_range() {
    let a = Array::compact_ndarray(&array![1i32, 2, 3, 4]).unwrap();
    let s = a.view().insert_dim(0);
    assert_eq!(kind_of("insert_dim", &s, &[0..1, 3..1]), Some(ErrorKind::InvalidIndex));
}

#[test]
fn permute_dims_reversed_range() {
    let a = Array::compact_ndarray(&array![[1i32, 2], [3, 4]]).unwrap();
    let s = a.view().permute_dims(&[1, 0]);
    assert_eq!(kind_of("permute_dims", &s, &[0..2, 2..1]), Some(ErrorKind::InvalidIndex));
}

#[test]
fn remove_dim_reversed_range() {
    let a = Array::compact_ndarray(&array![[1i32, 2, 3, 4]]).unwrap();
    let s = a.view().remove_dim(0);
    assert_eq!(kind_of("remove_dim", &s, &[3..1]), Some(ErrorKind::InvalidIndex));
}

#[test]
fn where_reversed_range() {
    let c = Array::compact_ndarray(&array![true, false, true, false]).unwrap();
    let a = Array::compact_ndarray(&array![1i32, 2, 3, 4]).unwrap();
    let s = where_condition(c, a.view(), a.view());
    assert_eq!(kind_of("where", &s, &[3..1]), Some(ErrorKind::InvalidIndex));
}

#[test]
fn clamp_reversed_range() {
    let a = Array::compact_ndarray(&array![1i32, 2, 3, 4]).unwrap();
    let s = a.view().clamp(a.view(), a.view());
    assert_eq!(kind_of("clamp", &s, &[3..1]), Some(ErrorKind::InvalidIndex));
}

/// Control: a leaf (Compact) validates the index before `check_out_buf`, so it already gives
/// InvalidIndex on every target.
#[test]
fn control_compact_reversed_range() {
    let a = Array::compact_ndarray(&array![1i32, 2, 3, 4]).unwrap();
    assert_eq!(kind_of("compact", &a, &[3..1]), Some(ErrorKind::InvalidIndex));
}
