// Repro for F014 (run 2026-10-29-batch6): an explicit block_shape passed to the Plain
// constructors (plain_ndarray_with / plain_ndarray_view_with / plain_ndarray_ptr) is silently
// discarded, although their docs say the params "seed arrays derived from this one (e.g. the
// block shape and codec used by Array::compact/compact_with)".
//
// Each test asserts the documented behavior, so it FAILS while the bug exists.
// The block shape is observed through the public `ArrayParams::override_from_array` plus the
// derived Debug output of `ArrayParams`.
use jix::dtype::Dtyped;
use jix::{Array, ArrayParams, ArrayStorage};
use ndarray::Array2;

/// "block_shape: Some([..]), block_shape_fixed_dims: Some(..)" of an array's storage spec.
fn block_layout_of<S: ArrayStorage>(a: &Array<S>) -> String {
    let mut p = ArrayParams::new();
    p.override_from_array(a);
    let s = format!("{p:?}");
    let start = s.find("block_shape:").unwrap();
    let end = s.find(", block_size").unwrap();
    s[start..end].to_string()
}

fn params_64x64() -> ArrayParams {
    let mut p = ArrayParams::new();
    p.block_shape(&[64, 64]);
    p
}

/// Reference: the same params given straight to compact_ndarray_with.
fn reference_layout(data: &Array2<f32>) -> String {
    let c = Array::compact_ndarray_with(data, params_64x64()).unwrap();
    block_layout_of(&c)
}

#[test]
fn plain_ndarray_with_block_shape_seeds_compact() {
    let data = Array2::<f32>::zeros((1024, 1024));
    let expected = reference_layout(&data);
    println!("reference compact_ndarray_with: {expected}");

    let p = Array::plain_ndarray_with(data, params_64x64()).unwrap();
    println!("plain spec:                     {}", block_layout_of(&p));
    let c = p.compact().unwrap();
    let got = block_layout_of(&c);
    println!("plain_ndarray_with(..).compact: {got}");
    assert_eq!(got, expected);
}

#[test]
fn plain_ndarray_view_with_block_shape_seeds_compact() {
    let data = Array2::<f32>::zeros((1024, 1024));
    let expected = reference_layout(&data);
    let p = Array::plain_ndarray_view_with(data.view(), params_64x64()).unwrap();
    let got = block_layout_of(&p.compact().unwrap());
    println!("plain_ndarray_view_with(..).compact: {got}");
    assert_eq!(got, expected);
}

#[test]
fn plain_ndarray_ptr_block_shape_seeds_compact() {
    let data = Array2::<f32>::zeros((1024, 1024));
    let expected = reference_layout(&data);
    let p: Array<jix::storage::Plain<&(), jix::TypeDyn>> = unsafe {
        Array::plain_ndarray_ptr(
            data.as_ptr() as *const u8,
            [1024u64, 1024],
            &[1024 * 4, 4],
            f32::DTYPE,
            params_64x64(),
        )
        .unwrap()
    };
    let got = block_layout_of(&p.compact().unwrap());
    println!("plain_ndarray_ptr(..).compact: {got}");
    assert_eq!(got, expected);
}

/// Control (passes today): with block_shape_fixed_dims set explicitly, Plain::new does not
/// override the fixed dims and the block shape is kept. This isolates the root cause to the
/// default fixed-dims override in Plain::new.
#[test]
fn control_explicit_fixed_dims_is_kept() {
    let data = Array2::<f32>::zeros((1024, 1024));
    let mut params = params_64x64();
    params.block_shape_fixed_dims(&[true, true]);
    let p = Array::plain_ndarray_with(data, params).unwrap();
    let got = block_layout_of(&p.compact().unwrap());
    println!("with explicit fixed dims: {got}");
    assert!(got.starts_with("block_shape: Some([64, 64])"), "got {got}");
}
