//! F004 (run 2026-10-23-batch3): Plain construction from an ndarray with negative strides.
//!
//! `plain_ndarray_ref_impl` converts each ndarray stride with `s as usize * size_of::<T>()`
//! (jix/src/storage/plain.rs:289); `plain_ndarray_with` uses `usize::try_from(s).unwrap()`
//! (plain.rs:248). Debug builds panic (multiply overflow / unwrap). Release builds wrap the stride
//! to 2^64 - k*itemsize; `Plain::read_data` then computes a wrapped span, and when it wraps to
//! exactly 0 it reads from the BASE pointer, ignoring the window start (plain.rs:468).
//!
//! Each test asserts the CORRECT behavior: either the right values or a clean `Err`.
//! Run in debug and with `--release`.

use jix::Array;
use ndarray::{array, s, Axis, Ix1, Ix2};

/// Correct: either `Ok(exp)` or a clean `Err` rejecting negative strides.
fn ok_or_err<T: PartialEq + std::fmt::Debug>(res: Result<T, jix::Error>, exp: T, what: &str) {
    match res {
        Ok(v) => assert_eq!(v, exp, "{what}: wrong value"),
        Err(e) => eprintln!("{what}: rejected with error (acceptable): {e}"),
    }
}

/// 2-element window of a reversed 1-d view: span = -4 * 1 + 4 == 0 (mod 2^64).
#[test]
fn reversed_1d_window_of_two() {
    let base = array![1i32, 2, 3, 4];
    let rev = base.slice(s![..;-1]); // logical [4, 3, 2, 1]
    let res = Array::plain_ndarray_view(rev).and_then(|a| {
        let ctx = a.read_ctx();
        a.to_ndarray_sub::<Ix1>(&[1..3], &ctx)
    });
    ok_or_err(res, array![3, 2], "reversed[1..3]");
}

/// Two full rows of a row-flipped (`flipud`) matrix: strides [-12, 4],
/// span = -12 * 1 + 4 * 2 + 4 == 0 (mod 2^64).
#[test]
fn flipud_matrix_two_rows() {
    let base = array![[1i32, 2, 3], [4, 5, 6], [7, 8, 9]];
    let flip = base.slice(s![..;-1, ..]); // logical [[7,8,9],[4,5,6],[1,2,3]]
    let res = Array::plain_ndarray_view(flip)
        .and_then(|a| a.slice((1..3, 0..3)).to_ndarray::<Ix2>());
    ok_or_err(res, array![[4, 5, 6], [1, 2, 3]], "flipud[1..3, :]");
}

/// Owned array with a negative stride on a length-1 axis (ndarray reports it as standard layout).
#[test]
fn owned_len1_inverted_axis() {
    let mut a = ndarray::Array2::<i32>::from_shape_vec((1, 3), vec![1, 2, 3]).unwrap();
    a.invert_axis(Axis(0));
    assert!(a.is_standard_layout());
    let res = Array::plain_ndarray(a).and_then(|a| a.to_ndarray::<Ix2>());
    ok_or_err(res, array![[1, 2, 3]], "plain_ndarray(len-1 inverted)");
}

/// Full read of a reversed view. Release returns the right values by accident, but the wrapped span
/// (-4 * 3 + 4 = 2^64 - 8) reaches `slice::from_raw_parts` with len > isize::MAX (UB). A debug build
/// with overflow checks OFF (std UB checks on) aborts here with "unsafe precondition(s) violated".
/// That aborts the whole process, so run this test on its own: `-- reversed_full_read --exact`.
#[test]
fn reversed_full_read() {
    let base = array![1i32, 2, 3, 4];
    let rev = base.slice(s![..;-1]);
    let res = Array::plain_ndarray_view(rev).and_then(|a| a.to_ndarray::<Ix1>());
    ok_or_err(res, array![4, 3, 2, 1], "reversed full read");
}

/// compact_ndarray goes through plain_ndarray_ref.
#[test]
fn compact_ndarray_reversed() {
    let base = array![1i32, 2, 3, 4];
    let rev = base.slice(s![..;-1]);
    let res = Array::compact_ndarray(&rev).and_then(|a| a.to_ndarray::<Ix1>());
    ok_or_err(res, array![4, 3, 2, 1], "compact_ndarray(reversed)");
}
