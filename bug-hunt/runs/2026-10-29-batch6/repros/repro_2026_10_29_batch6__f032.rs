//! Repro for F032 (run 2026-10-29-batch6): the Safety sections of `Plain::new`,
//! `Array::plain_ndarray_ptr` and `Array::compact_nd_ptr` do not say the memory must not be
//! written while jix reads it or while a view lent by a pull read is alive. Both tests below meet
//! every requirement those sections state (pointer valid and non-dangling for the array's
//! lifetime, every element readable, every element a valid dtype value), yet jix turns the region
//! into a `&[u8]` (plain.rs:468) and lends it out, so the calls are UB.
//!
//! The values are right natively; the UB is visible under Miri only:
//!   python jix/run_miri.py --all-features --test repro_2026_10_29_batch6__f032 -- lent_view_then_owner_write
//! This is a documentation gap: once the Safety sections forbid these writes, these tests become
//! examples of contract violations rather than regression tests.
use std::sync::atomic::{AtomicU32, Ordering};

use jix::dtype::Dtyped;
use jix::storage::Plain;
use jix::{Array, ArrayParams, ReadContext, Ty};
use ndarray::Ix1;

/// Single thread, no data race. The owner keeps using its own raw pointer (which the documented
/// contract allows: the pointer jix holds stays valid) and writes element 0 while it holds the
/// pull view `to_ndarray_buf(.., None)` lent. Reading that view afterwards should see the new
/// value; under Miri it is UB, because the view is a `&[u8]` the write invalidated.
#[test]
fn lent_view_then_owner_write() {
    let mut data = vec![1u8, 2, 3];
    let p = data.as_mut_ptr();
    // SAFETY (as documented): `data` outlives `a` and is never moved or reallocated, every element
    // is readable through `p`, and every byte is a valid u8.
    let a = unsafe {
        Array::<Plain<&(), Ty<u8>>>::plain_ndarray_ptr(
            p.cast_const(),
            [3u64],
            &[1],
            <u8 as Dtyped>::DTYPE,
            ArrayParams::default(),
        )
    }
    .unwrap();
    let ctx = ReadContext::default();
    let view = a.to_ndarray_buf(&[0..3], &ctx, None).unwrap();
    // The owner updates its buffer through its own pointer (no other thread involved).
    unsafe { p.write(9) };
    let first = unsafe { view.data_ptr().read() };
    assert_eq!(first, 9);
    drop(view);
    drop(a);
    drop(data);
}

/// Same contract, with a concurrent writer: another thread stores to element 1 (an atomic) while
/// jix reads the region. `Plain` is Send + Sync through `SendSyncPtr`, and nothing in the Safety
/// section forbids writes, but jix's non-atomic read through a `&[u8]` is a data race.
#[test]
fn concurrent_writer_during_read() {
    let data: Vec<AtomicU32> = (0..4).map(AtomicU32::new).collect();
    // SAFETY (as documented): `data` outlives `arr`, every element is readable and a valid u32.
    let arr = unsafe {
        Array::<Plain<&(), Ty<u32>>>::plain_ndarray_ptr(
            data.as_ptr().cast::<u8>(),
            [4u64],
            &[4],
            <u32 as Dtyped>::DTYPE,
            ArrayParams::default(),
        )
    }
    .unwrap();
    std::thread::scope(|s| {
        s.spawn(|| data[1].store(100, Ordering::Relaxed));
        let got = arr.to_ndarray::<Ix1>().unwrap();
        assert!(got[1] == 1 || got[1] == 100);
    });
}
