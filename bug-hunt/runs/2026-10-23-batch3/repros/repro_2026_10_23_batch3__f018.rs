#![allow(clippy::single_range_in_vec_init)]
//! F018 (run 2026-10-23-batch3): the `# Safety` sections of the public unsafe constructors
//! `StridedBuf::from_raw_parts` / `from_raw_parts_mut` (jix/src/storage/buf.rs:59-82) omit two
//! requirements the implementation relies on:
//!   (1) the pointer must be non-null even for an empty (0-byte) region, and
//!   (2) the bytes must not be accessed through any other pointer for `'a` - in particular an
//!       `out` must not overlap memory that an input of the read (e.g. a `Plain` made with the
//!       unsafe `plain_ndarray_ptr`) covers.
//! Every call below meets the documented contracts of both unsafe constructors, so each test
//! asserts the behavior a caller following the docs may expect.
//!
//! NOTE: `null_*` and `in_place_identity_push_5` ABORT THE PROCESS in a debug build (std's
//! `ub_checks`), which also kills every other test in this binary. Run each by name with
//! `--exact`. `in_place_neg_4` passes natively and fails only under Miri (Stacked Borrows):
//! `python jix/run_miri.py --all-features --test repro_2026_10_23_batch3__f018 -- in_place_neg_4`.

use jix::dtype::Dtyped;
use jix::storage::{Plain, StridedBuf};
use jix::{Array, ArrayParams, ReadContext, Ty};
use ndarray::array;

/// (1) Zero-sized access: every pointer is "valid for reads" of 0 bytes (std::ptr docs), null
/// included, so this meets the documented contract.
#[test]
fn null_ptr_for_empty_region() {
    let buf = unsafe { StridedBuf::from_raw_parts(std::ptr::null(), &[0], &[4], 4) };
    assert_eq!(buf.strides(), &[4]);
}

/// (1) Same for the writable constructor, used as `out` of an empty read (e.g. a C API that
/// reports an empty output buffer as a null pointer).
#[test]
fn null_ptr_mut_for_empty_out() {
    let a = Array::plain_ndarray(array![1.0f32, 2.0]).unwrap();
    let ctx = ReadContext::default();
    let mut out = unsafe { StridedBuf::from_raw_parts_mut(std::ptr::null_mut(), &[0], &[4], 4) };
    a.to_ndarray_buf(&[0..0], &ctx, Some(&mut out)).map(|_| ()).unwrap();
}

/// (2) In-place identity read: `out` is exactly the memory the Plain input views. Both unsafe
/// contracts hold (valid pointer for the lifetime, non-aliasing strides, valid f32 values).
#[test]
fn in_place_identity_push_5() {
    let mut buf = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let p = buf.as_mut_ptr().cast::<u8>();
    let a = unsafe {
        Array::<Plain<&(), Ty<f32>>>::plain_ndarray_ptr(
            p,
            [5u64],
            &[4],
            f32::DTYPE,
            ArrayParams::default(),
        )
    }
    .unwrap();
    let ctx = ReadContext::default();
    {
        let mut out = unsafe { StridedBuf::from_raw_parts_mut(p, &[5], &[4], 4) };
        a.to_ndarray_buf(&[0..5], &ctx, Some(&mut out)).map(|_| ()).unwrap();
    }
    drop(a);
    assert_eq!(buf, vec![1.0f32, 2.0, 3.0, 4.0, 5.0]);
}

/// (2) In-place negate, the Rust analogue of `np.negative(x, out=x)`. Natively prints the
/// expected values; Miri reports Stacked Borrows UB.
#[test]
fn in_place_neg_4() {
    let mut buf = vec![1.0f32, 2.0, 3.0, 4.0];
    let p = buf.as_mut_ptr().cast::<u8>();
    let a = unsafe {
        Array::<Plain<&(), Ty<f32>>>::plain_ndarray_ptr(
            p,
            [4u64],
            &[4],
            f32::DTYPE,
            ArrayParams::default(),
        )
    }
    .unwrap();
    let ctx = ReadContext::default();
    {
        let mut out = unsafe { StridedBuf::from_raw_parts_mut(p, &[4], &[4], 4) };
        let neg = -a.view();
        neg.to_ndarray_buf(&[0..4], &ctx, Some(&mut out)).map(|_| ()).unwrap();
    }
    drop(a);
    println!("{buf:?}");
    assert_eq!(buf, vec![-1.0f32, -2.0, -3.0, -4.0]);
}
