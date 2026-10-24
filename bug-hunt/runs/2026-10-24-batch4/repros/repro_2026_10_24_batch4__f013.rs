// F013: Slice's plain fast path (pull, PLAIN_READ inner) computes `inner_stride * step`
// unchecked (jix/src/ops/shape_ops/slice.rs:161). A step larger than the dim gives an output
// dim of length 1, a valid request, but the product overflows: a debug build panics with
// 'attempt to multiply with overflow'. These tests assert the correct behavior (no panic,
// right values), so they fail while the bug exists.
use jix::ops::SliceItem;
use jix::{Array, ReadContext};
use ndarray::{array, Ix1, Ix2};
use std::ops::Neg;

// Pull read through the public to_ndarray_buf (Plain inner carries PLAIN_READ -> fast path).
#[test]
fn f013_pull_huge_step() {
    let a = Array::plain_ndarray(array![0i32, 1, 2, 3, 4]).unwrap();
    let s = a.slice((SliceItem::new(Some(3), None, i64::MAX),));
    assert_eq!(s.shape(), &[1]);
    let ctx = ReadContext::default();
    let buf = s.to_ndarray_buf(&[0..1], &ctx, None).unwrap();
    // SAFETY: the pull buffer is valid for the 1-element region (read_data contract).
    let v = unsafe { buf.data_ptr().cast::<i32>().read_unaligned() };
    println!("strides {:?}", buf.strides());
    assert_eq!(v, 3);
}

// An elementwise op pulls its input (default read_as_elementwise_pipeline) -> same fast path.
#[test]
fn f013_neg_huge_step() {
    let a = Array::plain_ndarray(array![0i32, 1, 2, 3, 4]).unwrap();
    let s = a.slice((SliceItem::new(Some(2), None, i64::MAX),));
    assert_eq!(s.neg().to_ndarray::<Ix1>().unwrap(), array![-2]);
}

// 2-d: the product is stride(dim 0) * step, so a wider row lowers the overflowing step
// (64-bit: row stride 4000 B -> any step > ~4.6e15; 32-bit: any step > ~1.07e6).
#[test]
fn f013_neg_2d_large_step() {
    let nd = ndarray::Array::from_shape_fn((5, 1000), |(i, j)| (i * 1000 + j) as f32);
    let a = Array::plain_ndarray(nd).unwrap();
    let step: i64 = if usize::BITS == 32 { 2_000_000 } else { 1 << 53 };
    let s = a.slice((SliceItem::new(Some(1), None, step), ..));
    assert_eq!(s.shape(), &[1, 1000]);
    let got = s.neg().to_ndarray::<Ix2>().unwrap();
    let want = ndarray::Array::from_shape_fn((1, 1000), |(_, j)| -((1000 + j) as f32));
    assert_eq!(got, want);
}
