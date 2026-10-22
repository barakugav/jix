//! F010: Concatenate::new sums concat-dim lengths unchecked (concatenate.rs:111).
//! Asserts the CORRECT behavior (an Err for an overflowing shape), so it fails while the bug exists.
use jix::Array;
use jix::ops::Concatenate;
use ndarray::array;

#[test]
fn concatenate_overflowing_concat_dim_is_an_error() {
    // u8 element, lazy broadcast: shape [2^63 + 2] passes check_shape_overflow (itemsize 1).
    let len = (1u64 << 63) + 2;
    let a = Array::compact_ndarray(&array![7u8]).unwrap().broadcast(&[len]);
    let b = Array::compact_ndarray(&array![9u8]).unwrap().broadcast(&[len]);
    // 2 * (2^63 + 2) = 2^64 + 4 overflows u64: must be rejected, not panic / wrap to [4].
    let res = Concatenate::new_array((a, b), 0);
    match res {
        Err(e) => println!("got expected error: {e}"),
        Ok(c) => {
            let shape = c.shape().to_vec();
            let data = if shape[0] <= 16 {
                format!("{:?}", c.to_ndarray::<ndarray::Ix1>())
            } else {
                "<not read>".to_string()
            };
            panic!("expected Err, got Ok with shape {shape:?}, data = {data}")
        }
    }
}
