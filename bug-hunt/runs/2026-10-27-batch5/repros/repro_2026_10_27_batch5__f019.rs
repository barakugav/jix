// Repro for F019 (run 2026-10-27-batch5): check_out_buf does not reject an `out` whose strides
// alias over the requested region. Safe code gets such a writable `out` by reusing a pull result
// (whose strides are only non-aliasing for ITS OWN region) as the `out` of a read of a different
// shape. The push then returns Ok while `out` does not hold the region.
//
// Each test asserts the CORRECT behavior: the push is rejected with Err, or the region reads back
// correctly through out's strides. They fail while the bug exists.
#![allow(clippy::single_range_in_vec_init)]

use jix::storage::StridedBuf;
use jix::{Array, ReadContext};
use ndarray::{array, Array1};

/// The i32 elements of `buf` over `shape` (C order), read through its strides.
fn read_region(buf: &StridedBuf<'_>, shape: &[usize]) -> Vec<i32> {
    let n: usize = shape.iter().product();
    let mut idx = vec![0usize; shape.len()];
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        let off: usize = idx.iter().zip(buf.strides()).map(|(i, s)| i * s).sum();
        v.push(unsafe { buf.data_ptr().add(off).cast::<i32>().read_unaligned() });
        for d in (0..shape.len()).rev() {
            idx[d] += 1;
            if idx[d] < shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    v
}

/// Push `src[region]` into `out`; expect Err, or the region read back correctly.
fn push_and_check(
    src: &Array<impl jix::ArrayStorage>,
    region: &[std::ops::Range<u64>],
    out: &mut StridedBuf<'_>,
    expected: &[i32],
) {
    let shape: Vec<usize> = region.iter().map(|r| (r.end - r.start) as usize).collect();
    let ctx = ReadContext::default();
    let res = src.to_ndarray_buf(region, &ctx, Some(out));
    println!("push ok: {}", res.is_ok());
    if res.is_ok() {
        drop(res);
        let got = read_region(out, &shape);
        println!("read back through out strides {:?}: {got:?}", out.strides());
        assert_eq!(got, expected, "push returned Ok but out does not hold the region");
    }
}

/// InsertDim relabels its inner pull buffer with stride 0 on the inserted length-1 dim and keeps
/// it writable (Flip materialized it into a pool buffer). No pool warm-up needed: the [3, 4] region
/// at strides [0, 4] spans exactly the 16 bytes the pull allocated.
#[test]
fn insert_dim_pull_result_reused_as_out() {
    let ctx = ReadContext::default();
    let a = Array::plain_ndarray(array![1i32, 2, 3, 4]).unwrap().flip(0).insert_dim(0);
    let mut out = a.to_ndarray_buf(&[0..1, 0..4], &ctx, None).unwrap();
    let writable = out.data_ptr_mut().is_some();
    println!("pull strides {:?}, writable {writable}", out.strides());
    let big = Array::plain_ndarray(array![
        [10i32, 11, 12, 13],
        [20, 21, 22, 23],
        [30, 31, 32, 33]
    ])
    .unwrap();
    push_and_check(
        &big,
        &[0..3, 0..4],
        &mut out,
        &[10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33],
    );
}

/// Not specific to 0 strides: a packed [2, 2] pull (strides [8, 4]) reused for a [2, 3] region
/// puts (0, 2) and (1, 0) on the same byte 8. A warmed pool makes the buffer large enough to pass
/// the extent check.
#[test]
fn packed_pull_result_reused_as_out_of_wider_region() {
    let ctx = ReadContext::default();
    {
        let warm = Array::plain_ndarray(Array1::<i32>::zeros(1000)).unwrap().flip(0);
        let _b = warm.to_ndarray_buf(&[0..1000], &ctx, None).unwrap();
    }
    let a = Array::plain_ndarray(array![[1i32, 2], [3, 4]]).unwrap().flip(0);
    let mut out = a.to_ndarray_buf(&[0..2, 0..2], &ctx, None).unwrap();
    let writable = out.data_ptr_mut().is_some();
    println!("pull strides {:?}, writable {writable}", out.strides());
    let big = Array::plain_ndarray(array![[10i32, 11, 12], [20, 21, 22]]).unwrap();
    push_and_check(&big, &[0..2, 0..3], &mut out, &[10, 11, 12, 20, 21, 22]);
}
