// F017 (run 2026-10-24-batch4): the first paragraph of `Array::to_ndarray`'s docs
// (jix/src/array.rs:576-579) promises "a contiguous row-major (C-order) ndarray". Since #52
// (217ced2) the result is allocated in the storage's `read_layout_order`, so a transposed view or
// an F-order Plain source comes back F-contiguous. This test asserts the documented C-order
// contract, so it fails today. If the fix is doc-only (drop the C-order promise), delete this test.
use jix::Array;
use ndarray::{Ix2, ShapeBuilder, array};

fn check_c_order(label: &str, nd: ndarray::Array2<i32>, expected: ndarray::Array2<i32>) {
    println!(
        "{label:<28} values_ok={} strides={:?} as_slice_is_some={}",
        nd == expected,
        nd.strides(),
        nd.as_slice().is_some()
    );
    assert_eq!(nd, expected, "{label}: values differ");
    // A caller following "contiguous row-major (C-order)" reads the raw buffer in C order.
    let (raw, _) = nd.clone().into_raw_vec_and_offset();
    let c_order: Vec<i32> = expected.iter().copied().collect();
    println!("{label:<28} raw buffer = {raw:?}, row-major = {c_order:?}");
    assert!(
        nd.as_slice().is_some(),
        "{label}: to_ndarray result is not C-contiguous (strides {:?}), docs promise C-order",
        nd.strides()
    );
    assert_eq!(raw, c_order, "{label}: raw buffer is not in row-major order");
}

#[test]
fn to_ndarray_is_c_order_for_permuted_view() {
    let a = Array::compact_ndarray(&array![[1i32, 2, 3], [4, 5, 6]]).unwrap();
    let nd = a.view().permute_dims(&[1, 0]).to_ndarray::<Ix2>().unwrap();
    check_c_order("permute_dims(compact,[1,0])", nd, array![[1i32, 4], [2, 5], [3, 6]]);
}

#[test]
fn to_ndarray_is_c_order_for_f_order_plain() {
    let mut f = ndarray::Array2::<i32>::zeros((2, 3).f());
    f.assign(&array![[1i32, 2, 3], [4, 5, 6]]);
    let p = Array::plain_ndarray_ref(&f).unwrap();
    let nd = p.to_ndarray::<Ix2>().unwrap();
    check_c_order("plain_ndarray_ref(F-order)", nd, array![[1i32, 2, 3], [4, 5, 6]]);
}

#[test]
fn to_ndarray_is_c_order_for_f_order_plain_plus_op() {
    let mut f = ndarray::Array2::<i32>::zeros((2, 3).f());
    f.assign(&array![[1i32, 2, 3], [4, 5, 6]]);
    let p = Array::plain_ndarray_ref(&f).unwrap();
    let nd = p.map(|x| x + 1i32).to_ndarray::<Ix2>().unwrap();
    check_c_order("map(plain F-order, +1)", nd, array![[2i32, 3, 4], [5, 6, 7]]);
}
