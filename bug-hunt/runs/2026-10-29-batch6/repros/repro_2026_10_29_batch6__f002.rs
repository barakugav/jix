// F002 (run 2026-10-29-batch6): #[derive(Dtyped)] is not hygienic. The generated `const DTYPE`
// calls unqualified `size_of` / `align_of` / `assert!` and declares block-local items
// (`BASE_OFFSET`, `FIELDS`, ...) in the same block where it re-emits the field types, so names in
// the user's scope change the layout the derive computes AND the asserts that should catch it.
// The DTYPE then disagrees with the real Rust layout, from code with no `unsafe`.
//
// These tests assert the CORRECT behavior and compile both before and after a fix.
// The UB consequences (uninit padding read, accepted padded struct) are in
// repro_2026_10_29_batch6__f002_ub.rs, which must stop compiling once fixed.

#![allow(dead_code)]

use jix::dtype::Dtyped;

mod user_size_of {
    // An ordinary safe helper that happens to be called `size_of`.
    pub const fn size_of<T>() -> usize {
        16
    }

    #[derive(Copy, Clone, Debug, jix::dtype::Dtyped)]
    #[repr(C)]
    pub struct S {
        pub a: u8,
    }
}

const BASE_OFFSET: u16 = 4;

#[derive(Copy, Clone, Debug, Dtyped)]
#[repr(C)]
struct UsesBaseOffset {
    a: [u8; BASE_OFFSET as usize + 1], // real: [u8; 5]
    b: [u8; 9 - BASE_OFFSET as usize], // real: [u8; 5]
}

#[test]
fn shadowed_size_of_dtype_matches_layout() {
    let d = user_size_of::S::DTYPE;
    eprintln!(
        "S: size_of = {}, DTYPE = {d:?}",
        std::mem::size_of::<user_size_of::S>()
    );
    assert_eq!(d.itemsize() as usize, std::mem::size_of::<user_size_of::S>());
}

#[test]
fn shadowed_size_of_plain_read_stays_in_array() {
    // A 32-element array; the jix Plain view covers only its first 2 elements. Reading element 1
    // must return exactly that element's byte. With the inflated DTYPE, jix reads DTYPE.itemsize
    // bytes from element 1 on - bytes of elements 2.. that are outside the viewed array (with a
    // 2-element allocation these bytes are past the end of the heap allocation).
    let big = ndarray::Array1::from_shape_fn(32, |i| user_size_of::S { a: i as u8 });
    let view = big.slice(ndarray::s![0..2]);
    let plain = jix::Array::plain_ndarray_ref(&view).unwrap();
    let itemsize = plain.dtype().itemsize() as usize;
    let mut buf = vec![0u8; itemsize];
    plain
        .to_ndarray_slice(&[1..2], &mut buf, &jix::ReadContext::default())
        .unwrap();
    eprintln!("element 1 read as {itemsize} bytes: {buf:?}");
    assert_eq!(buf, vec![1u8], "read bytes outside the 2-element array");
}

#[test]
fn shadowed_base_offset_field_offsets_match_layout() {
    let d = UsesBaseOffset::DTYPE;
    let fields = d.fields().unwrap();
    let got: Vec<(String, usize, usize)> = fields
        .iter()
        .map(|(n, off, dt)| (n.to_string(), *off as usize, dt.itemsize() as usize))
        .collect();
    eprintln!("UsesBaseOffset fields (name, offset, itemsize): {got:?}");
    let want = vec![
        ("a".to_string(), std::mem::offset_of!(UsesBaseOffset, a), 5),
        ("b".to_string(), std::mem::offset_of!(UsesBaseOffset, b), 5),
    ];
    assert_eq!(got, want);
}
