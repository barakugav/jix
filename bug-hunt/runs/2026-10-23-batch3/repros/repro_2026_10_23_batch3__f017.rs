//! F017 repro: safe code receives writable pull results whose strides alias (a 0 stride on a dim
//! longer than 1), and `check_out_buf` accepts them back as `out` of a push read. The push then
//! "succeeds" while writing several elements onto the same bytes.
//!
//! Each test asserts the correct behavior: a push into such an `out` must either be rejected with
//! `Err`, or (if accepted) the region must be readable back through `out`'s strides. Both fail
//! while the bug exists.
#![allow(clippy::single_range_in_vec_init)]

use jix::storage::StridedBuf;
use jix::{Array, ReadContext};
use ndarray::array;

/// Read back `shape` f32 elements from `buf` at its own strides, row-major.
fn read_back_f32(buf: &StridedBuf<'_>, shape: &[usize]) -> Vec<f32> {
    let base = buf.data_ptr();
    let strides = buf.strides();
    let mut vals = Vec::new();
    match shape.len() {
        1 => {
            for i in 0..shape[0] {
                vals.push(unsafe { base.add(i * strides[0]).cast::<f32>().read_unaligned() });
            }
        }
        2 => {
            for i in 0..shape[0] {
                for j in 0..shape[1] {
                    let off = i * strides[0] + j * strides[1];
                    vals.push(unsafe { base.add(off).cast::<f32>().read_unaligned() });
                }
            }
        }
        _ => unreachable!(),
    }
    vals
}

/// Path 1: Broadcast pull relabels a writable PoolBuf (from Flip) with a 0 stride
/// (broadcast.rs:156). Reused as `out` of a Plain push read.
#[test]
fn broadcast_pull_result_as_out_of_plain_push() {
    let ctx = ReadContext::default();
    let b = Array::plain_ndarray(array![[1.0f32], [2.0]])
        .unwrap()
        .flip(0)
        .broadcast(&[2, 4]);
    let mut out = b.to_ndarray_buf(&[0..2, 0..4], &ctx, None).unwrap();
    let writable = out.data_ptr_mut().is_some();
    println!("pull strides: {:?}, writable: {writable}", out.strides());

    let x = Array::plain_ndarray(array![[1.0f32, 2., 3., 4.], [5., 6., 7., 8.]]).unwrap();
    let ctx2 = ReadContext::default();
    let r = x
        .view()
        .to_ndarray_buf(&[0..2, 0..4], &ctx2, Some(&mut out))
        .map(|_| ());
    println!("push ok: {}", r.is_ok());
    if r.is_ok() {
        let got = read_back_f32(&out, &[2, 4]);
        println!("read back through out strides: {got:?}");
        assert_eq!(
            got,
            vec![1.0f32, 2., 3., 4., 5., 6., 7., 8.],
            "push into an aliasing out returned Ok but the region is not in out"
        );
    }
}

/// Path 2: the elementwise pipeline's pull buffer (`materialize_pipeline_out_buf`,
/// elementwise_pipeline.rs:876-885) gets stride 0 on dims where every operand has stride 0.
/// Reused as `out` of an elementwise (Neg) push read.
#[test]
fn pipeline_pull_result_as_out_of_neg_push() {
    let ctx = ReadContext::default();
    let n = -Array::plain_ndarray(array![[1.0f32]]).unwrap().broadcast(&[2, 4]);
    let mut out = n.to_ndarray_buf(&[0..2, 0..4], &ctx, None).unwrap();
    let writable = out.data_ptr_mut().is_some();
    println!("pull strides: {:?}, writable: {writable}", out.strides());

    let x = Array::plain_ndarray(array![[1.0f32, 2., 3., 4.], [5., 6., 7., 8.]]).unwrap();
    let ctx2 = ReadContext::default();
    let r = (-x.view())
        .to_ndarray_buf(&[0..2, 0..4], &ctx2, Some(&mut out))
        .map(|_| ());
    println!("push ok: {}", r.is_ok());
    if r.is_ok() {
        let got = read_back_f32(&out, &[2, 4]);
        println!("read back through out strides: {got:?}");
        assert_eq!(
            got,
            vec![-1.0f32, -2., -3., -4., -5., -6., -7., -8.],
            "push into an aliasing out returned Ok but the region is not in out"
        );
    }
}

/// Path 3: a reduction pushed into an aliasing `out`. With `state_in_out_buf` (reduction.rs:497)
/// the two output cells share one state, so their streams are folded together.
#[test]
fn reduction_into_aliasing_out() {
    let ctx = ReadContext::default();
    let b = Array::plain_ndarray(array![1.0f32])
        .unwrap()
        .flip(0)
        .broadcast(&[2]);
    let mut out = b.to_ndarray_buf(&[0..2], &ctx, None).unwrap();
    let writable = out.data_ptr_mut().is_some();
    println!("pull strides: {:?}, writable: {writable}", out.strides());

    let x = Array::plain_ndarray(array![[1.0f32, 2., 3., 4.], [5., 6., 7., 8.]]).unwrap();
    let ctx2 = ReadContext::default();
    let r = x
        .view()
        .sum(1)
        .to_ndarray_buf(&[0..2], &ctx2, Some(&mut out))
        .map(|_| ());
    println!("push ok: {}", r.is_ok());
    if r.is_ok() {
        let got = read_back_f32(&out, &[2]);
        println!("read back through out strides: {got:?}");
        assert_eq!(
            got,
            vec![10.0f32, 26.0],
            "reduction push into an aliasing out returned Ok but the region is not in out"
        );
    }
}
