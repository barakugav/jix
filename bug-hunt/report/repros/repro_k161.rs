// Repro for K161 (run 2026-11-15-batch7-2).
//
// Scalar::new and Plain::new build their spec through ArrayParams::into_spec -> tune, whose last
// check rejects a shape whose product, padded to the (auto) block shape, overflows u64. Neither
// storage has blocks: both overwrite the tuned block shape with all-ones right after. So a lazy
// shape that Broadcast accepts (element product fits u64) is rejected for a Scalar / Plain leaf.
//
// The tests assert the CORRECT behavior (construction succeeds), so they fail while the bug
// exists. block_size is pinned so the auto block shape, and so the outcome, does not depend on
// the host's L1d size (the default path takes block_size from L1d, see params.rs:311-313).
use jix::dtype::Dtyped;
use jix::{Array, ArrayParams, ReadContext};
use ndarray::Ix2;

// B is odd, so no power-of-two block dim divides it; A * B fits u64 with B to spare, so padding
// B by even one item makes the product overflow.
const B: u64 = (1 << 31) + 1;
const A: u64 = u64::MAX / B;

fn params() -> ArrayParams {
    // 32 KiB blocks: the auto block shape for u8 [A, B] is [1, 32768].
    ArrayParams::new().block_size(1 << 15)
}

#[test]
fn control_broadcast_of_same_shape_is_accepted() {
    assert!(A.checked_mul(B).is_some());
    let x = Array::compact_ndarray(&ndarray::array![[7u8]]).unwrap();
    let y = x.broadcast(&[A, B]);
    assert_eq!(y.shape(), &[A, B]);
    let got = y
        .to_ndarray_sub::<Ix2>(&[0..2, B - 3..B], &ReadContext::default())
        .unwrap();
    assert_eq!(got, ndarray::Array2::from_elem((2, 3), 7u8));
}

#[test]
fn scalar_of_huge_lazy_shape_is_accepted() {
    let s = jix::__private::Scalar::new(1u8, &[A, B], params());
    let s = match s {
        Ok(s) => s,
        Err(e) => panic!("Scalar::new rejected a valid lazy shape: {e:?}"),
    };
    let a = Array::from_storage(s);
    let got = a
        .to_ndarray_sub::<Ix2>(&[A - 2..A, 0..3], &ReadContext::default())
        .unwrap();
    assert_eq!(got, ndarray::Array2::from_elem((2, 3), 1u8));
}

#[test]
fn zero_stride_plain_of_huge_shape_is_accepted() {
    let v = 9u8;
    // Every element reads the same byte: the strided span is one byte, valid for the whole call.
    let p: Result<jix::PlainArrayRef<u8>, jix::Error> = unsafe {
        Array::plain_ndarray_ptr(&v as *const u8, &[A, B], &[0, 0], u8::DTYPE, params())
    };
    let p = match p {
        Ok(p) => p,
        Err(e) => panic!("plain_ndarray_ptr rejected a valid zero-stride shape: {e:?}"),
    };
    let got = p
        .to_ndarray_sub::<Ix2>(&[0..2, B - 3..B], &ReadContext::default())
        .unwrap();
    assert_eq!(got, ndarray::Array2::from_elem((2, 3), 9u8));
}
