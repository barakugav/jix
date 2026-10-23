// Repro for bug-hunt run 2026-10-23-batch3, finding F012.
//
// A derived #[repr(C)] struct with a zero-element array field (`[u32; 0]`) or a zero-sized array
// field whose shape product exceeds u16 (`[[Empty; 300]; 300]`) has a nonzero itemsize, so arrays
// of it can be built and written to a .jix archive. Reading the archive back fails in
// `Dtype::from_proto` (jix/src/archive/schema/conversions.rs), so the library writes files that it
// refuses to read.
//
// Correct behavior asserted here: either `write_to` refuses the dtype up front, or the round trip
// succeeds with the same dtype and values. Also: `Dtype::set_shape` must not panic (debug
// overflow in the unchecked `product::<Itemsize>()` of the current shape, jix/src/dtype.rs).
//
// Run: cargo test --manifest-path jix/Cargo.toml --all-features --test repro_2026_10_23_batch3__f012 -- --nocapture
use std::io::Cursor;

use jix::dtype::Dtyped;
use jix::{ArchiveValidation, Array, ArrayParams};

#[derive(Copy, Clone, Dtyped)]
#[repr(C)]
struct WithZeroLen {
    a: u32,
    z: [u32; 0],
}

#[derive(Copy, Clone, Dtyped)]
#[repr(C)]
struct Empty {}

#[derive(Copy, Clone, Dtyped)]
#[repr(C)]
struct WithBigZst {
    a: u32,
    z: [[Empty; 300]; 300],
}

/// Write `arr` and read it back. Writing may refuse the dtype; if it accepts it, reading must work
/// and give back the same dtype and bytes.
fn assert_roundtrip_or_refused<T: Dtyped>(elems: &[T], first_u32: impl Fn(&T) -> u32) {
    let a = Array::compact_ndarray(&ndarray::arr1(elems)).unwrap();
    let mut buf = Cursor::new(Vec::new());
    match a.write_to(&mut buf) {
        Err(e) => {
            println!("write refused up front (acceptable): {e}");
        }
        Ok(_) => {
            let r = Array::read_from_reader(
                Cursor::new(buf.into_inner()),
                None,
                ArrayParams::default(),
                ArchiveValidation::Strict,
            );
            let back = match r {
                Ok(b) => b,
                Err(e) => panic!(
                    "write_to accepted dtype {:?} but read_from_reader rejects the file: {e}",
                    T::DTYPE
                ),
            };
            assert_eq!(back.dtype(), &T::DTYPE);
            let typed = back.into_typed::<T>().unwrap();
            let got = typed.to_ndarray::<ndarray::Ix1>().unwrap();
            assert_eq!(got.len(), elems.len());
            for (g, e) in got.iter().zip(elems) {
                assert_eq!(first_u32(g), first_u32(e));
            }
        }
    }
}

#[test]
fn derived_zero_len_field_roundtrip() {
    println!("WithZeroLen::DTYPE = {:?}", WithZeroLen::DTYPE);
    assert_roundtrip_or_refused(&[WithZeroLen { a: 7, z: [] }], |x| x.a);
}

#[test]
fn derived_big_zst_field_roundtrip() {
    assert_roundtrip_or_refused(&[WithBigZst { a: 7, z: [[Empty {}; 300]; 300] }], |x| x.a);
}

#[test]
fn set_shape_on_big_zst_array_dtype_does_not_panic() {
    // Current shape [300, 300]: product 90000 does not fit Itemsize (u16).
    let mut d = <[[Empty; 300]; 300] as Dtyped>::DTYPE;
    let r = d.set_shape(&[2]);
    println!("set_shape 300x300 -> [2]: {:?}", r.as_ref().map_err(|e| e.to_string()));
    if r.is_ok() {
        assert_eq!(d.shape(), &[2]);
        assert_eq!(d.itemsize(), 0);
    }
}

#[test]
fn set_shape_on_256x256_zst_array_dtype_does_not_panic() {
    // Current shape [256, 256]: product 65536 wraps to exactly 0 in u16.
    let mut d = <[[Empty; 256]; 256] as Dtyped>::DTYPE;
    let r = d.set_shape(&[2]);
    println!("set_shape 256x256 -> [2]: {:?}", r.as_ref().map_err(|e| e.to_string()));
    if r.is_ok() {
        assert_eq!(d.shape(), &[2]);
        assert_eq!(d.itemsize(), 0);
    }
}
