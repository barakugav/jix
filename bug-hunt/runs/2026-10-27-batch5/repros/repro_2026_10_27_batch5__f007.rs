// Repro for bug-hunt run 2026-10-27-batch5, finding F007:
// Broadcast::read_data maps every broadcast dim to the inner range 0..1 and pulls the inner
// BEFORE noticing that the requested region is empty, so an empty read evaluates the inner row.
//
// Each test asserts the CORRECT behavior (an empty read returns Ok without evaluating anything),
// so it fails while the bug exists.
//
// `empty_broadcast_of_huge_lazy_row` ABORTS THE PROCESS (allocation failure, SIGABRT) while the
// bug exists, so it is #[ignore]d; run it alone with `-- --ignored empty_broadcast_of_huge`.
#![allow(clippy::single_range_in_vec_init)]

use jix::Array;
use ndarray::{array, IxDyn};

/// q = [[6, 7, 8]] / [[2, 0, 4]]: element [0, 1] divides by zero (an Error by default).
fn quotient() -> Array<impl jix::ArrayStorage<ElementType = jix::Ty<i32>>> {
    let a = Array::plain_ndarray(array![[6i32, 7, 8]]).unwrap();
    let b = Array::plain_ndarray(array![[2i32, 0, 4]]).unwrap();
    a / b
}

#[test]
fn baseline_other_empty_reads_do_not_evaluate() {
    // Sanity: empty reads through other views of the same failing pipeline are Ok.
    let q = quotient();
    let ctx = q.read_ctx();
    let r = q.view().to_ndarray_sub::<IxDyn>(&[0..1, 1..1], &ctx);
    assert_eq!(r.map(|a| a.shape().to_vec()).unwrap(), vec![1, 0]);
    let r = q.view().slice((0..1, 1..1)).to_ndarray::<IxDyn>();
    assert_eq!(r.map(|a| a.shape().to_vec()).unwrap(), vec![1, 0]);
    let r = q.view().insert_dim(0).to_ndarray_sub::<IxDyn>(&[0..0, 0..1, 0..3], &ctx);
    assert_eq!(r.map(|a| a.shape().to_vec()).unwrap(), vec![0, 1, 3]);
    // And the full (non-empty) read really fails.
    assert!(q.view().to_ndarray::<IxDyn>().is_err());
}

#[test]
fn broadcast_to_zero_length_full_read() {
    // The array has ZERO elements, yet reading it fails with a divide-by-zero error.
    let e = quotient().broadcast(&[0, 3]);
    let r = e.to_ndarray::<IxDyn>();
    let r = r.map(|a| a.shape().to_vec()).map_err(|e| format!("{e}"));
    println!("q.broadcast([0, 3]).to_ndarray() = {r:?}");
    assert_eq!(r, Ok(vec![0, 3]));
}

#[test]
fn broadcast_empty_sub_read() {
    // An empty sub-range of a broadcast dim.
    let b = quotient().broadcast(&[5, 3]);
    let ctx = b.read_ctx();
    let r = b.to_ndarray_sub::<IxDyn>(&[2..2, 0..3], &ctx);
    let r = r.map(|a| a.shape().to_vec()).map_err(|e| format!("{e}"));
    println!("q.broadcast([5, 3]).to_ndarray_sub([2..2, 0..3]) = {r:?}");
    assert_eq!(r, Ok(vec![0, 3]));
}

#[test]
fn concatenate_with_empty_broadcast_member_in_the_middle() {
    // concatenate([ones(2,3), q.broadcast([0,3]), ones(2,3)]) - no element of the result comes
    // from q, but Concatenate reads the zero-length middle member with local range 0..0.
    let o1 = Array::plain_ndarray(ndarray::Array2::<i32>::ones((2, 3))).unwrap();
    let o2 = Array::plain_ndarray(ndarray::Array2::<i32>::ones((2, 3))).unwrap();
    let mid = quotient().broadcast(&[0, 3]);
    let c = jix::ops::concatenate((o1, mid, o2), 0);
    let r = c.to_ndarray::<IxDyn>().map_err(|e| format!("{e}"));
    println!("concatenate((ones, q.broadcast([0,3]), ones)) = {r:?}");
    assert_eq!(r, Ok(ndarray::ArrayD::<i32>::ones(IxDyn(&[4, 3]))));

    // Same members, empty one LAST: never read, so this one is Ok (position-dependent result).
    let o1 = Array::plain_ndarray(ndarray::Array2::<i32>::ones((2, 3))).unwrap();
    let o2 = Array::plain_ndarray(ndarray::Array2::<i32>::ones((2, 3))).unwrap();
    let last = jix::ops::concatenate((o1, o2, quotient().broadcast(&[0, 3])), 0);
    let r = last.to_ndarray::<IxDyn>().map(|a| a.shape().to_vec());
    println!("concatenate((ones, ones, q.broadcast([0,3]))) = {:?}", r.as_ref().map_err(|e| e.kind()));
}

#[test]
fn compact_vs_to_ndarray_disagree() {
    // compact() iterates no chunks for an empty array and succeeds; to_ndarray() fails.
    let e = quotient().broadcast(&[0, 3]);
    let c = e.compact().map(|c| c.shape().to_vec());
    let r = e.to_ndarray::<IxDyn>().map(|a| a.shape().to_vec());
    println!("compact: {:?}, to_ndarray: {:?}", c.as_ref().map_err(|e| e.kind()), r.as_ref().map_err(|e| e.kind()));
    assert_eq!(c.unwrap(), vec![0, 3]);
    assert_eq!(r.map_err(|e| format!("{e}")), Ok(vec![0, 3]));
}

#[test]
fn empty_broadcast_read_cost() {
    // An empty read of a [0, 2^24] broadcast of (a + a) evaluates the whole 2^24-element row.
    let n = 1usize << 24;
    let a = Array::compact_ndarray(&ndarray::Array2::<f32>::zeros((1, n))).unwrap();
    let e = (a.view() + a.view()).broadcast(&[0, n as u64]);
    let t = std::time::Instant::now();
    let r = e.to_ndarray::<IxDyn>().unwrap();
    let dt = t.elapsed();
    let t2 = std::time::Instant::now();
    let r2 = (a.view() + a.view()).slice((0..1, 0..0)).to_ndarray::<IxDyn>().unwrap();
    let dt2 = t2.elapsed();
    println!("empty broadcast read {:?} in {dt:?}; empty slice read {:?} in {dt2:?}", r.shape(), r2.shape());
}

// ABORTS the process while the bug exists (memory allocation of 2^50 bytes failed).
#[test]
#[ignore]
fn empty_broadcast_of_huge_lazy_row() {
    let n: u64 = 1 << 50;
    // [1, 2^50] u8: an O(1) lazy broadcast, flipped (Flip materializes its pull read).
    let row = Array::compact_ndarray(&array![[1u8]]).unwrap().broadcast(&[1, n]).flip(1);
    let e = row.broadcast(&[0, n]); // zero elements
    let ctx = e.read_ctx();
    let r = e.to_ndarray_sub::<IxDyn>(&[0..0, 0..n], &ctx).map(|a| a.shape().to_vec());
    println!("{:?}", r.as_ref().map_err(|e| e.kind()));
}
