// Repro for F003 (bug-hunt run 2026-10-24-batch4): `impl_from_range!` in
// jix/src/ops/shape_ops/slice.rs turns an inclusive end `e` into `e as i64 + 1`, so an
// inclusive end of -1 ("through the last element") becomes the absolute end 0, and every
// integer bound is cast with a plain `as i64`, so u64/usize bounds above i64::MAX turn into
// end-relative negatives. Each test asserts the CORRECT behavior and fails while the bug exists.

use jix::Array;
use jix::ops::{Slice, SliceItem};
use ndarray::array;

fn arr5() -> Array<impl jix::ArrayStorage<ElementType = jix::Ty<i32>>> {
    Array::compact_ndarray(&array![0i32, 1, 2, 3, 4]).unwrap()
}

/// `..=-1` (end-relative, inclusive) selects the whole dim, like ndarray's `s![..=-1]`.
#[test]
fn to_inclusive_minus_one_is_whole_dim() {
    let item: SliceItem = (..=-1i64).into();
    println!("..=-1 -> {item:?}");
    let s = Slice::new_array(arr5(), (..=-1i64,).into()).unwrap();
    println!("..=-1 shape: {:?}", s.shape());
    assert_eq!(s.to_ndarray().unwrap(), array![0, 1, 2, 3, 4]);
}

/// `0..=-1` is the same selection with an explicit start.
#[test]
fn zero_to_inclusive_minus_one_is_whole_dim() {
    let s = Slice::new_array(arr5(), (0i64..=-1,).into()).unwrap();
    println!("0..=-1 shape: {:?}", s.shape());
    assert_eq!(s.to_ndarray().unwrap(), array![0, 1, 2, 3, 4]);
}

/// `-3..=-1` is the last three elements. Today `Slice::new` returns Err and `Array::slice` panics.
#[test]
fn neg3_to_inclusive_minus_one_is_last_three() {
    let r = Slice::new_array(arr5(), (-3i64..=-1,).into());
    if let Err(e) = &r {
        println!("-3..=-1 err: {e}");
    }
    assert_eq!(r.unwrap().to_ndarray().unwrap(), array![2, 3, 4]);
}

/// Same, through the documented `Array::slice` convenience method (panics today).
#[test]
fn array_slice_neg3_to_inclusive_minus_one() {
    let s = arr5().slice((-3i32..=-1,));
    assert_eq!(s.to_ndarray().unwrap(), array![2, 3, 4]);
}

/// A u64 end above i64::MAX is out of bounds for any dim (dims are capped at i64::MAX) and must be
/// rejected; today it wraps to -1 and silently drops the last element.
#[test]
fn huge_u64_end_is_rejected() {
    let r = Slice::new_array(arr5(), (0u64..u64::MAX,).into());
    match &r {
        Ok(s) => println!("0..u64::MAX accepted, shape {:?}, values {:?}", s.shape(), s.to_ndarray::<ndarray::Ix1>().unwrap()),
        Err(e) => println!("0..u64::MAX err: {e}"),
    }
    assert!(r.is_err(), "0..u64::MAX must be out of bounds");
}

/// `..=usize::MAX` wraps to `-1 + 1 = 0`: an empty slice instead of an out-of-bounds error.
#[test]
fn huge_usize_inclusive_end_is_rejected() {
    let r = Slice::new_array(arr5(), (..=usize::MAX,).into());
    match &r {
        Ok(s) => println!("..=usize::MAX accepted, shape {:?}", s.shape()),
        Err(e) => println!("..=usize::MAX err: {e}"),
    }
    assert!(r.is_err(), "..=usize::MAX must be out of bounds");
}

/// `..=i64::MAX` is out of bounds and should be an Err; today `e as i64 + 1` overflows inside the
/// `From` conversion (debug panic; release wraps to i64::MIN, which then errors).
#[test]
fn inclusive_i64_max_end_is_rejected_without_panic() {
    let r = std::panic::catch_unwind(|| -> SliceItem { (..=i64::MAX).into() });
    let item = r.expect("converting ..=i64::MAX into a SliceItem must not panic");
    assert!(Slice::new_array(arr5(), (item,).into()).is_err());
}
