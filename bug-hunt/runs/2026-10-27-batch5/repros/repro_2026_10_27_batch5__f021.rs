// Repro for F021 (run 2026-10-27-batch5): a widening element-wise view (Cast / Map / Op2)
// inherits the source's explicit, all-fixed block shape counted in ELEMENTS. compact() /
// write_to() with default params then re-checks that block against the 1 GiB per-block limit at
// the WIDER itemsize and fails, although the source array itself is valid and compacted fine.
//
// The assertions state the expected behavior (default-param materialization of a valid view
// succeeds), so these tests FAIL while the bug exists.
//
// Run: cargo test --manifest-path jix/Cargo.toml --all-features --test repro_2026_10_27_batch5__f021 -- --nocapture
use std::io::Cursor;

use jix::scalar::Complex;
use jix::{Array, ArrayParams};

// 2^26 + 8 u8 elements in ONE explicit block: 64 MiB at itemsize 1 (well under the 1 GiB limit),
// but 1 GiB + 128 bytes at itemsize 16.
const N: usize = (1 << 26) + 8;

fn source() -> Array<jix::storage::Compact<jix::Ty<u8>>> {
    let data = ndarray::Array1::<u8>::zeros(N);
    let mut params = ArrayParams::new();
    params.block_shape(&[N as u32]); // explicit => all dims fixed by default
    Array::compact_ndarray_with(&data, params).expect("source with a 64 MiB block is valid")
}

#[test]
fn widening_cast_compact_default_params() {
    let a = source();
    // Built-in op: u8 -> Complex<f64> (itemsize 1 -> 16).
    let r = a.view().cast::<Complex<f64>>().compact();
    println!("cast::<Complex<f64>>().compact(): {:?}", r.as_ref().err());
    assert!(
        r.is_ok(),
        "compact() of a valid widened view failed: {:?}",
        r.err()
    );
}

#[test]
fn widening_cast_write_to_default_params() {
    let a = source();
    let mut buf = Cursor::new(Vec::new());
    let r = a.view().cast::<Complex<f64>>().write_to(&mut buf);
    println!("cast::<Complex<f64>>().write_to(): {:?}", r.as_ref().err());
    assert!(
        r.is_ok(),
        "write_to() of a valid widened view failed: {:?}",
        r.err()
    );
}
