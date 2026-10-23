//! F022: `Owned::read_section` (jix/src/archive/block.rs:439) converts the zero-filled
//! `Vec<MaybeUninit<T>>` into `Vec<T>` with `std::mem::transmute`, which relies on the
//! unspecified layout of `Vec` being identical for the two instantiations.
//!
//! This test drives that exact line through the public API (an in-memory round trip read back
//! with `Array::read_from_reader`, i.e. the `Owned` backend, which calls `read_section::<u8>`
//! for the block data and `read_section::<BlockLocation>` for the block table). It asserts the
//! correct result. It PASSES today: with the current rustc - including `-Zrandomize-layout`
//! with `-Zbuild-std`, and under Miri - `Vec<MaybeUninit<T>>` and `Vec<T>` get the same layout,
//! so the defect is latent (a reliance on unguaranteed layout), not an observable failure.
use std::io::Cursor;

use jix::{ArchiveValidation, Array, ArrayParams};
use ndarray::array;

#[test]
fn owned_read_section_round_trip() {
    let original = Array::compact_ndarray(&array![[1i32, -2, 3], [4, 5, -6]]).unwrap();
    let mut buf = Cursor::new(Vec::new());
    original.write_to(&mut buf).unwrap();

    let loaded = Array::read_from_reader(
        Cursor::new(buf.into_inner()),
        None,
        ArrayParams::default(),
        ArchiveValidation::default(),
    )
    .unwrap()
    .into_typed::<i32>()
    .unwrap();
    assert_eq!(
        loaded.to_ndarray().unwrap(),
        array![[1i32, -2, 3], [4, 5, -6]].into_dyn()
    );
}
