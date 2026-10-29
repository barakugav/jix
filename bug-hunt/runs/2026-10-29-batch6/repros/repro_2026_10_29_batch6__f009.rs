// Repro for F009 (run 2026-10-29-batch6): #[derive(Dtyped)] on a bare #[repr(packed)] struct
// (= repr(Rust, packed), field order unspecified) computes field offsets in declaration order and
// never checks them against offset_of!.
//
// Asserts the CORRECT behavior: the DTYPE field offsets equal the real Rust layout, and
// dtype_sub_field reads back the values that were stored.
//
// On stable rustc today the layout of a packed struct with niche-free fields happens to follow
// declaration order, so this test PASSES there (the bug is latent). It FAILS once rustc uses its
// freedom to reorder, e.g. with layout randomization applied to this test crate only (a
// RUSTFLAGS-wide -Zrandomize-layout cannot build jix itself):
//
//   CARGO_TARGET_DIR=bug-hunt/runs/2026-10-29-batch6/scratch/target-f009-rand \
//   cargo +nightly rustc --manifest-path jix/Cargo.toml --all-features --profile test \
//       --test repro_2026_10_29_batch6__f009 -- -Zrandomize-layout
//   bug-hunt/runs/2026-10-29-batch6/scratch/target-f009-rand/debug/deps/repro_2026_10_29_batch6__f009-* --nocapture

#![allow(dead_code)]

use jix::Array;
use jix::dtype::Dtyped;
use ndarray::{Ix1, array};
use std::mem::offset_of;

#[allow(clippy::repr_packed_without_abi)]
#[derive(Copy, Clone, Debug, PartialEq, Dtyped)]
#[repr(packed)]
struct P {
    a: u8,
    b: u32,
    c: u16,
    d: u64,
}

#[test]
fn bare_packed_dtype_offsets_match_rust_layout() {
    let real = [
        ("a", offset_of!(P, a)),
        ("b", offset_of!(P, b)),
        ("c", offset_of!(P, c)),
        ("d", offset_of!(P, d)),
    ];
    let dtype = P::DTYPE;
    let fields = dtype.fields().unwrap();
    for ((name, off), f) in real.iter().zip(fields) {
        eprintln!("{name}: offset_of={off} DTYPE offset={}", f.1);
    }

    let src = array![
        P { a: 1, b: 2, c: 3, d: 4 },
        P { a: 5, b: 6, c: 7, d: 8 }
    ];
    let arr = Array::compact_ndarray(&src).unwrap();
    // The whole element round-trips (raw bytes), whatever the layout.
    assert_eq!(arr.view().to_ndarray::<Ix1>().unwrap(), src);

    let a = arr.view().dtype_sub_field::<u8>("a").to_ndarray::<Ix1>().unwrap();
    let b = arr.view().dtype_sub_field::<u32>("b").to_ndarray::<Ix1>().unwrap();
    let c = arr.view().dtype_sub_field::<u16>("c").to_ndarray::<Ix1>().unwrap();
    let d = arr.view().dtype_sub_field::<u64>("d").to_ndarray::<Ix1>().unwrap();
    eprintln!(
        "field a = {:?}, b = {:?}, c = {:?}, d = {:?}",
        a.to_vec(),
        b.to_vec(),
        c.to_vec(),
        d.to_vec()
    );

    assert_eq!(a, array![1u8, 5]);
    assert_eq!(b, array![2u32, 6]);
    assert_eq!(c, array![3u16, 7]);
    assert_eq!(d, array![4u64, 8]);
    for ((name, off), f) in real.iter().zip(fields) {
        assert_eq!(*off, f.1 as usize, "DTYPE offset of `{name}` does not match the Rust layout");
    }
}
