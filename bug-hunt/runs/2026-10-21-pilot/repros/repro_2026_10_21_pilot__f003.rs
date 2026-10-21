// Verifier repro for F003 (run 2026-10-21-pilot).
//
// A padded `#[repr(C)]` struct dtype (`{a: u8, b: u32}`, itemsize 8, alignment 4) is accepted by
// `#[derive(Dtyped)]`. `NdCopier::new` maps (8, 4) to `scalar_fn::<[u32; 2]>`, so every element is
// moved with a typed `[u32; 2]` load. Bytes 1..4 of each element are padding (uninitialized in the
// Rust abstract machine), so the load is UB. Natively the test passes; run it under Miri:
//
//   python jix/run_miri.py --all-features --test repro_2026_10_21_pilot__f003 -- padded_plain_to_ndarray
use jix::Array;
use jix::dtype::Dtyped;
use ndarray::{Array1, Ix1};

#[derive(Clone, Copy, Debug, PartialEq, Dtyped)]
#[repr(C)]
struct Padded {
    a: u8,
    b: u32,
}

#[test]
fn padded_plain_to_ndarray() {
    assert_eq!(Padded::DTYPE.itemsize(), 8);
    assert_eq!(Padded::DTYPE.alignment().as_usize(), 4);
    let nd = Array1::from(vec![Padded { a: 1, b: 2 }; 4]);
    let a = Array::plain_ndarray(nd).unwrap();
    let out = a.to_ndarray::<Ix1>().unwrap();
    assert_eq!(out.to_vec(), vec![Padded { a: 1, b: 2 }; 4]);
}

// Same dtype through the Compact path (gather on compress, scatter on read). Native only: under
// Miri the first UB aborts the process, so only `padded_plain_to_ndarray` is run there.
#[test]
fn padded_compact_roundtrip() {
    let nd = Array1::from(vec![Padded { a: 7, b: 9 }; 4]);
    let a = Array::compact_ndarray(&nd).unwrap();
    let out = a.to_ndarray::<Ix1>().unwrap();
    assert_eq!(out.to_vec(), vec![Padded { a: 7, b: 9 }; 4]);
}
