// F001 repro: a writable pull result (PoolBuf-backed StridedBuf) handed back as `out` of a larger
// read is accepted; nothing checks that `out` covers the region, so the read writes past it.
//
// Each test asserts the CORRECT behavior (an Err for an undersized `out`), so it fails while the
// bug exists. WARNING: `flip_push_into_undersized_pull_result` ABORTS the test process in a debug
// build (std's ub_checks catch `get_unchecked_mut` out of range in flip.rs); run the tests one at
// a time by name. `plain_push_into_undersized_pull_result` does not crash in a debug build - it
// silently writes 3 bytes past a 1-byte heap allocation - and is the one to run under Miri.
use jix::{Array, ReadContext};
use ndarray::array;

#[test]
fn flip_push_into_undersized_pull_result() {
    let small = Array::plain_ndarray(array![1u8, 2]).unwrap().flip(0);
    let ctx = ReadContext::default();
    // Pull: Flip materializes a 1-byte pool buffer; the returned StridedBuf is writable.
    let mut buf = small.to_ndarray_buf(&[0..1], &ctx, None).unwrap();
    assert_eq!(buf.strides(), &[1]);
    assert!(buf.data_ptr_mut().is_some());

    let big = Array::plain_ndarray(array![1u8, 2, 3, 4]).unwrap().flip(0);
    let ctx2 = ReadContext::default();
    // 4-element region into a 1-byte destination: must be rejected.
    let r = big.to_ndarray_buf(&[0..4], &ctx2, Some(&mut buf));
    assert!(r.is_err(), "push of 4 elements into a 1-byte out buffer was accepted");
}

#[test]
fn plain_push_into_undersized_pull_result() {
    let small = Array::plain_ndarray(array![1u8, 2]).unwrap().flip(0);
    let ctx = ReadContext::default();
    let mut buf = small.to_ndarray_buf(&[0..1], &ctx, None).unwrap();
    assert_eq!(buf.strides(), &[1]);

    // No materializing op at all: a plain array's push read copies straight into `out`.
    let big = Array::plain_ndarray(array![1u8, 2, 3, 4]).unwrap();
    let ctx2 = ReadContext::default();
    let r = big.to_ndarray_buf(&[0..4], &ctx2, Some(&mut buf));
    assert!(r.is_err(), "push of 4 elements into a 1-byte out buffer was accepted");
}
