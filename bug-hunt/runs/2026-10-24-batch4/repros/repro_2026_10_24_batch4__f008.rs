//! F008 repro: ReductionOp::new computes `reduced_prod + 4` unchecked (jix/src/ops/reduction.rs:177).
//! A valid lazy shape (each dim <= i64::MAX, byte size fits in u64) whose reduced product is
//! > u64::MAX - 4 makes the reduction constructors panic in debug builds instead of returning Ok.
//! These tests assert the CORRECT behavior (Ok, 0-d lazy result), so they fail while the bug exists.
use jix::Array;
use jix::scalar::bool8;
use ndarray::array;

// 1-byte dtype, shape [i64::MAX, 2]: 2^64 - 2 elements / bytes, accepted by check_shape_overflow.
fn huge_u8() -> Array<impl jix::storage::ArrayStorageTyped<Item = u8>> {
    let a = Array::compact_ndarray(&array![[1u8]]).unwrap();
    jix::ops::Broadcast::new_array(a, &[i64::MAX as u64, 2]).unwrap()
}

#[test]
fn sum_u8_over_huge_broadcast_constructs() {
    let s = jix::ops::Sum::new_array(huge_u8(), (0, 1));
    assert!(s.is_ok(), "{:?}", s.err());
    assert_eq!(s.unwrap().shape(), &[] as &[u64]);
}

#[test]
fn max_u8_over_huge_broadcast_constructs() {
    let s = jix::ops::Max::new_array(huge_u8(), (0, 1));
    assert!(s.is_ok(), "{:?}", s.err());
}

#[test]
fn all_bool8_over_huge_broadcast_constructs() {
    let a = Array::compact_ndarray(&array![[bool8::TRUE]]).unwrap();
    let a = jix::ops::Broadcast::new_array(a, &[i64::MAX as u64, 2]).unwrap();
    let s = jix::ops::All::new_array(a, (0, 1));
    assert!(s.is_ok(), "{:?}", s.err());
}
