//! F018: `Dtype::from_fields` computes the struct itemsize as `last.offset + last.itemsize` with a
//! plain u16 `+` (jix/src/dtype.rs:491) BEFORE `check_fields` validates the layout with checked
//! adds. In a debug build an oversized (or simply malformed) field list panics with "attempt to add
//! with overflow" instead of returning `Err`.
//!
//! These tests assert the CORRECT behavior (an `Err`), so they fail while the bug exists.
use jix::dtype::{Dtype, Dtyped};

/// Fields that tile back to back but end past u16::MAX (65000 + 1000 = 66000).
#[test]
fn from_fields_total_size_over_u16_max_is_err() {
    let r = Dtype::from_fields(vec![
        ("a".to_string(), 0, <[u8; 65000]>::DTYPE),
        ("b".to_string(), 65000, <[u8; 1000]>::DTYPE),
    ]);
    assert!(r.is_err(), "expected Err for a struct larger than u16::MAX bytes, got {r:?}");
}

/// A single tiny field at a bogus offset: invalid regardless of overflow (offset must be 0), but
/// the unchecked add of offset 65535 + itemsize 2 runs first.
#[test]
fn from_fields_single_field_at_huge_offset_is_err() {
    let r = Dtype::from_fields(vec![("x".to_string(), u16::MAX, u16::DTYPE)]);
    assert!(r.is_err(), "expected Err for a field at offset 65535, got {r:?}");
}

/// Control: the same API with an in-range layout works.
#[test]
fn from_fields_control_in_range_ok() {
    let d = Dtype::from_fields(vec![
        ("a".to_string(), 0, <[u8; 65000]>::DTYPE),
        ("b".to_string(), 65000, <[u8; 535]>::DTYPE),
    ])
    .unwrap();
    assert_eq!(d.itemsize(), 65535);
}
