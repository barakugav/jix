// K194 (run 2026-11-17-scan-106-107-b): 32-bit only.
//
// A stepped Slice over a SubDtype checks its zero-copy bounding box with
// check_region_size(box, FIELD itemsize) (slice.rs:153-160) and, since SubDtype over a ZeroCopy
// parent is ZeroCopy (#106, sub_dtype.rs:65-66), takes the box path. SubDtype then pulls the same
// box from its parent (sub_dtype.rs:98), whose check_out_buf runs at the PARENT itemsize and fails.
// A 2-element pull returns Err(InvalidShapeOperation) instead of taking the K156 fallback.
//
// These tests assert the CORRECT behavior: they fail on i686 while the bug exists and pass on
// x86_64 (check_region_size is a no-op on 64-bit).
//
//   bug-hunt/tools/cargo-i686 test --manifest-path jix/Cargo.toml --all-features \
//       --test repro_k194 -- --nocapture
use jix::Array;
use jix::dtype::Dtyped;
use jix::ops::SliceItem;
use ndarray::{Ix1, array};

#[derive(Copy, Clone, Dtyped, Debug, PartialEq)]
#[repr(C)]
struct S16 {
    a: u8,
    rest: [u8; 15],
}

// Box = STEP + 1 elements: (2^27 + 1) * 1 B fits in isize::MAX on 32-bit, (2^27 + 1) * 16 B does not.
const STEP: i64 = 1 << 27;
const LEN: u64 = (1 << 27) + 2;

fn field_slice() -> Array<impl jix::storage::ArrayStorageTyped<Item = u8>> {
    let one = Array::plain_ndarray(array![S16 { a: 7, rest: [0; 15] }]).unwrap();
    one.broadcast(&[LEN])
        .dtype_sub_field::<u8>("a")
        .slice((SliceItem::new(None, None, STEP),))
}

#[test]
fn f003_pull_two_elements_of_stepped_sub_field() {
    let s = field_slice();
    assert_eq!(s.shape(), &[2]);
    let ctx = s.read_ctx();
    let buf = s.to_buf_strided(&[0..2], &ctx);
    println!("pull: {:?}", buf.as_ref().map(|b| b.strides().to_vec()));
    let buf = buf.expect("2-element pull of a stepped Slice over a SubDtype");
    let stride = buf.strides()[0];
    let vals: Vec<u8> = (0..2)
        .map(|i| unsafe { *buf.data_ptr().add(i * stride) })
        .collect();
    assert_eq!(vals, vec![7u8, 7]);
}

#[test]
fn f003_elementwise_op_over_stepped_sub_field() {
    let s = field_slice();
    let two = Array::plain_ndarray(array![1u8, 1]).unwrap();
    let got = (s + two).to_ndarray::<Ix1>();
    println!("op: {got:?}");
    assert_eq!(got.unwrap(), array![8u8, 8]);
}

// Controls (pass on both targets): the push read of the same view, and the same stepped slice on
// the parent dtype (the Slice's own check then sees itemsize 16 and falls back).
#[test]
fn control_push_two_elements_of_stepped_sub_field() {
    let got = field_slice().to_ndarray::<Ix1>();
    assert_eq!(got.unwrap(), array![7u8, 7]);
}

#[test]
fn control_pull_two_elements_of_stepped_parent() {
    let one = Array::plain_ndarray(array![S16 { a: 7, rest: [0; 15] }]).unwrap();
    let s = one.broadcast(&[LEN]).slice((SliceItem::new(None, None, STEP),));
    let ctx = s.read_ctx();
    let buf = s.to_buf_strided(&[0..2], &ctx);
    assert!(buf.is_ok(), "{:?}", buf.err());
}
