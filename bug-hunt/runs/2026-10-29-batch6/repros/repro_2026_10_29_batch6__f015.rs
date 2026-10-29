// Repro for F015 (run 2026-10-29-batch6): a caller's `block_shape_fixed_dims` given without
// `block_shape` is silently replaced by the source's flags in `override_from_array` /
// `compact_with`, while `compact_ndarray_with` rejects the same params and archive reads honor them.
// Each test asserts the CORRECT behavior, so it fails while the bug exists.
//
// The block layout is observed through the public `ArrayParams::override_from_array` plus the
// derived `Debug` of `ArrayParams` (the fields are pub(crate)).
use jix::{Array, ArrayParams};
use ndarray::Array2;

fn block_layout_of<S: jix::ArrayStorage>(a: &Array<S>) -> String {
    let mut p = ArrayParams::new();
    p.override_from_array(a);
    let s = format!("{p:?}");
    let start = s.find("block_shape:").unwrap();
    let end = s.find(", block_size").unwrap();
    s[start..end].to_string()
}

fn source() -> Array<impl jix::ArrayStorage> {
    // Explicit block shape [8, 32]: all dims fixed by default.
    let data = Array2::from_shape_fn((64, 64), |(i, j)| (i * 64 + j) as i32);
    let mut p = ArrayParams::new();
    p.block_shape(&[8, 32]);
    Array::compact_ndarray_with(&data, p).unwrap()
}

/// `override_from_array` doc: "Fields that are already set in self are not overwritten."
#[test]
fn override_from_array_keeps_caller_fixed_dims() {
    let a = source();
    let mut p = ArrayParams::new();
    p.block_shape_fixed_dims(&[true, false]);
    p.override_from_array(&a);
    let got = format!("{p:?}");
    // [true, false] is DimBitmap bits 0b01.
    assert!(
        got.contains("block_shape_fixed_dims: Some(DimBitmap { bits: 1, len: 2 })"),
        "override_from_array overwrote the caller's block_shape_fixed_dims: {got}"
    );
}

/// `compact_with` doc: "Any optional field in `params` that is not set will be inherited from the
/// source storage". The caller's flags are set, so they must be honored (dim 0 pinned at 8, dim 1
/// rescaled to block_size) or rejected like `compact_ndarray_with` does - not silently dropped.
#[test]
fn compact_with_honors_or_rejects_caller_fixed_dims() {
    let a = source();
    let mut p = ArrayParams::new();
    p.block_shape_fixed_dims(&[true, false]);
    p.block_size(8 * 64 * 4); // with dim 0 pinned at 8, dim 1 scales to 64

    // The same params are rejected on the ndarray path.
    let data = Array2::<i32>::zeros((64, 64));
    let e = Array::compact_ndarray_with(&data, p.clone()).unwrap_err();
    println!("compact_ndarray_with: Err({e})");

    match a.compact_with(p, &a.read_ctx()) {
        Err(e) => println!("compact_with: Err({e})"), // acceptable: consistent with the ndarray path
        Ok(c) => {
            let got = block_layout_of(&c);
            println!("compact_with: Ok, {got}");
            assert!(
                got.contains("block_shape_fixed_dims: Some(DimBitmap { bits: 1, len: 2 })")
                    && got.starts_with("block_shape: Some([8, 64])"),
                "compact_with dropped the caller's block_shape_fixed_dims [true, false]: {got}"
            );
        }
    }
}
