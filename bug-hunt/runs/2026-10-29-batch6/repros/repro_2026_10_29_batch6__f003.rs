//! Repro for F003 (run 2026-10-29-batch6): Plain::read_data builds a `&[u8]` over the whole
//! strided SPAN of a non-contiguous view (gap bytes included), and jix-py's numpy(out=) builds a
//! `&mut [u8]` over the whole span of `out`. When an input view and the destination share no
//! element but their spans interleave, those slices cover bytes that another live `&mut` /
//! `PtrMutNoalias` writes - an aliasing violation (UB) on the Rust abstract machine.
//!
//! The values come out right natively (the touched bytes are disjoint), so these tests PASS in a
//! normal `cargo test`; the bug shows as a Miri error:
//!   python jix/run_miri.py --all-features --test repro_2026_10_29_batch6__f003
//! Each test asserts the correct values; under Miri it fails with "Undefined Behavior" while the
//! bug exists, and passes once Plain / numpy(out=) stop claiming the gap bytes.
use jix::dtype::Dtyped;
use jix::storage::{Plain, StridedBuf};
use jix::{Array, ArrayParams, ReadContext, TypeDyn};
use ndarray::{Array2, s};

/// 100% safe code. `big` is 3x4 u8, values 0..12 at byte offsets 0..12.
/// col = column 0 (offsets 0, 4, 8; span [0, 9)); row = row 1, cols 1..4 (offsets 5, 6, 7).
/// No shared element, but the column's span contains the whole row. Plain push copy:
/// to_ndarray_slice(buf: &mut [u8] = row) -> Plain::read_data -> slice::from_raw_parts(col, 9)
/// -> copy_from -> NdCopier::copy(PtrNoalias(0..9), PtrMutNoalias(5..8)).
#[test]
fn safe_plain_push_out_inside_input_span() {
    let mut big = Array2::<u8>::from_shape_fn((3, 4), |(r, c)| (r * 4 + c) as u8);
    let (mut row, col) = big.multi_slice_mut((s![1, 1..4], s![.., 0]));
    let a = Array::plain_ndarray_ref(&col).unwrap();
    let out = row.as_slice_mut().unwrap();
    a.to_ndarray_slice(&[0..3], out, &ReadContext::default())
        .unwrap();
    assert_eq!(out, &[0u8, 4, 8]);
}

/// Same views, through the element-wise pipeline: the Plain operand's `&[u8]` over [0, 9) stays
/// live for the whole run while the pipeline writes bytes 5..8 through `PtrMutNoalias`.
#[test]
fn safe_pipeline_out_inside_input_span() {
    let mut big = Array2::<u8>::from_shape_fn((3, 4), |(r, c)| (r * 4 + c) as u8);
    let (mut row, col) = big.multi_slice_mut((s![1, 1..4], s![.., 0]));
    let a = Array::plain_ndarray_ref(&col).unwrap().map(|x: u8| x + 1);
    let out = row.as_slice_mut().unwrap();
    a.to_ndarray_slice(&[0..3], out, &ReadContext::default())
        .unwrap();
    assert_eq!(out, &[1u8, 5, 9]);
}

/// Mirrors, step by step, what jix-py does for
/// `x = np.arange(12, dtype=np.uint8).reshape(3, 4); jix.asarray(x[:, 0]).numpy(out=x[:, 1])`
/// (np.shares_memory(x[:, 0], x[:, 1]) is False; the numpy() docstring only forbids `out`
/// overlapping memory the array reads):
/// - jix.asarray(x[:, 0]) -> Plain::new over numpy's pointer, shape [3], strides [4]
///   (jix-py/src/ops/common/operand.rs:486);
/// - numpy(out=x[:, 1]) -> `data = slice::from_raw_parts_mut(out_ptr, strided_span)` = bytes
///   [1, 10) (jix-py/src/array.rs:651), then `StridedBuf::from_raw_parts_mut(data.as_mut_ptr(),
///   [3], [4], 1)` (array.rs:665) and `to_ndarray_buf(.., Some(&mut out))`.
/// The `&mut [u8]` over [1, 10) covers input elements 4 and 8; the input `&[u8]` over [0, 9)
/// covers output elements 1 and 5.
#[test]
fn jixpy_numpy_out_column_into_column_pattern() {
    let mut x: Vec<u8> = (0..12).collect();
    let base: *mut u8 = x.as_mut_ptr(); // stands in for numpy's data pointer
    let a = unsafe {
        Array::<Plain<&(), TypeDyn>>::plain_ndarray_ptr(
            base.cast_const(),
            [3u64],
            &[4],
            u8::DTYPE,
            ArrayParams::default(),
        )
    }
    .unwrap();
    let ctx = ReadContext::default();
    {
        let data = unsafe { std::slice::from_raw_parts_mut(base.add(1), 9) };
        let mut out = unsafe { StridedBuf::from_raw_parts_mut(data.as_mut_ptr(), &[3], &[4], 1) };
        a.to_ndarray_buf(&[0..3], &ctx, Some(&mut out)).unwrap();
    }
    drop(a);
    assert_eq!(x, vec![0u8, 0, 2, 3, 4, 4, 6, 7, 8, 8, 10, 11]);
}
