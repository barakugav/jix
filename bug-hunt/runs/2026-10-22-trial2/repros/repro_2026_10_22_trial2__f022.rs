// F022 (core part): DimSlice::resolve compares slice endpoints against `dim_len as i64`
// (jix/src/ops/shape_ops/slice.rs:365-367). For a dim longer than i64::MAX (valid for 1-byte
// dtypes: tile/repeat/broadcast), `dim_len as i64` is negative, so every explicit endpoint -
// even `0..2` - is rejected as out of bounds. The Python index parser hands core exactly such
// explicit endpoints (`end: Some(shape as i64)` for ellipsis / trailing dims), so `jix.slice` on
// any array with a huge dim fails too. Asserts the CORRECT behavior, so it fails while the bug
// exists.
use jix::{Array, ReadContext};
use ndarray::{Ix1, array};

fn huge() -> Array<impl jix::ArrayStorage<ElementType = jix::Ty<u8>>> {
    // [0,1,2] tiled r times: len = 3r = 2^63 + 1 > i64::MAX; in[j] = j % 3.
    let r: u64 = 3074457345618258603;
    let t = Array::compact_ndarray(&array![0u8, 1, 2]).unwrap().tile(r, 0);
    assert_eq!(t.shape(), &[(1u64 << 63) + 1]);
    t
}

#[test]
fn reference_full_range_slice_works() {
    // `..` resolves to (None, None) -> 0..dim_len without the i64 cast, and reading works.
    let s = jix::ops::Slice::new_array(huge(), (..,).into()).unwrap();
    let got = s.to_ndarray_sub::<Ix1>(&[0..4], &ReadContext::default()).unwrap();
    assert_eq!(got, array![0u8, 1, 2, 0]);
}

#[test]
fn slice_explicit_small_range_on_huge_dim() {
    let s = jix::ops::Slice::new_array(huge(), (0u64..2,).into());
    let s = s.expect("slice 0..2 of a dim of length 2^63+1 must be Ok");
    assert_eq!(s.to_ndarray::<Ix1>().unwrap(), array![0u8, 1]);
}

#[test]
fn slice_negative_start_on_huge_dim() {
    // -2.. is the last two elements: j = 2^63 - 1 and 2^63, values (2^63-1)%3 = 1, 2^63%3 = 2.
    let s = jix::ops::Slice::new_array(huge(), (-2i64..,).into());
    let s = s.expect("slice -2.. of a dim of length 2^63+1 must be Ok");
    assert_eq!(s.to_ndarray::<Ix1>().unwrap(), array![1u8, 2]);
}
