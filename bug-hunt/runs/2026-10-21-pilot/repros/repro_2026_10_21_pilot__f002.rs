// Verifier repro for F002: an itemsize-widening op (cast u8 -> f64) after a broadcast is never
// re-checked with check_shape_overflow, so a pull read sizes its buffer with a wrapping
// `product(shape) * itemsize` (storage/buf.rs:312). Debug build: overflow panic at that line.
use jix::Array;
use ndarray::array;

#[test]
fn widening_cast_pull_read_size_wraps() {
    let n: u64 = (1 << 61) + 1; // n * 1 fits u64 (broadcast check passes); n * 8 does not
    let big = Array::compact_ndarray(&array![7u8]).unwrap().broadcast(&[n]).cast::<f64>();
    println!("shape {:?} itemsize {}", big.shape(), big.dtype().itemsize());
    let ctx = big.read_ctx();
    let r = big.to_ndarray_buf(&[0..n], &ctx, None);
    println!("returned: is_err = {}", r.is_err());
}
