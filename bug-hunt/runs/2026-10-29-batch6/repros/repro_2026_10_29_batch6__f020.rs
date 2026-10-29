//! F020: a struct dtype nested 33+ levels deep is written to a .jix archive without error, but the
//! archive cannot be read back - prost's default decode recursion limit (100) rejects the header
//! (BlockTableHeader -> Dtype -> DtypeStruct -> Field -> Dtype -> ... is 3 levels per struct).
//!
//! Safe public API only: nested `#[derive(Dtyped)]` structs, `compact_ndarray`, `write_to`,
//! `read_from_reader`. The test asserts the CORRECT behavior (either write_to rejects the dtype,
//! or the file round-trips), so it fails while the bug exists.

use std::io::Cursor;

use jix::dtype::Dtyped;
use jix::{ArchiveValidation, Array, ArrayParams};

#[derive(Copy, Clone, Default, Debug, PartialEq, Dtyped)]
#[repr(C)]
struct S1 {
    a: u8,
}

impl From<u8> for S1 {
    fn from(a: u8) -> Self {
        S1 { a }
    }
}

macro_rules! nest {
    ($($name:ident($inner:ident)),* $(,)?) => {
        $(
            #[derive(Copy, Clone, Default, Debug, PartialEq, Dtyped)]
            #[repr(C)]
            struct $name {
                a: $inner,
            }

            impl From<u8> for $name {
                fn from(x: u8) -> Self {
                    $name { a: $inner::from(x) }
                }
            }
        )*
    };
}

nest!(
    S2(S1), S3(S2), S4(S3), S5(S4), S6(S5), S7(S6), S8(S7), S9(S8), S10(S9),
    S11(S10), S12(S11), S13(S12), S14(S13), S15(S14), S16(S15), S17(S16), S18(S17), S19(S18),
    S20(S19), S21(S20), S22(S21), S23(S22), S24(S23), S25(S24), S26(S25), S27(S26), S28(S27),
    S29(S28), S30(S29), S31(S30), S32(S31), S33(S32),
);

/// Write `vals` to an in-memory .jix and read it back. Returns `Ok(None)` if the writer
/// rejected the dtype up front (acceptable), `Ok(Some(..))` with the values read back otherwise.
fn roundtrip<T: Dtyped + Copy + Send + Sync + 'static>(
    vals: Vec<T>,
) -> Result<Option<Vec<T>>, String> {
    let a = Array::compact_ndarray(&ndarray::Array1::from(vals)).map_err(|e| format!("{e:?}"))?;
    let mut file = Cursor::new(Vec::new());
    if let Err(e) = a.write_to(&mut file) {
        println!("write_to rejected the dtype: {e:?}");
        return Ok(None);
    }
    let bytes = file.into_inner();
    println!("write_to: Ok, {} bytes", bytes.len());
    for validation in [ArchiveValidation::Strict, ArchiveValidation::Minimal] {
        let r = Array::read_from_reader(
            Cursor::new(&bytes),
            None,
            ArrayParams::default(),
            validation,
        )
        .map_err(|e| format!("read_from_reader({validation:?}) failed: {e:?}"))?;
        let r = r.into_typed::<T>().map_err(|e| format!("into_typed: {e:?}"))?;
        let got: Vec<T> = r
            .to_ndarray::<ndarray::IxDyn>()
            .map_err(|e| format!("{e:?}"))?
            .into_iter()
            .collect();
        if validation == ArchiveValidation::Minimal {
            return Ok(Some(got));
        }
    }
    unreachable!()
}

#[test]
fn nested_struct_depth_32_roundtrips() {
    let v: Vec<S32> = (1..=4u8).map(S32::from).collect();
    let got = roundtrip(v.clone()).expect("depth 32 must round-trip");
    assert_eq!(got, Some(v));
}

#[test]
fn nested_struct_depth_33_roundtrips_or_is_rejected_on_write() {
    let v: Vec<S33> = (1..=4u8).map(S33::from).collect();
    match roundtrip(v.clone()) {
        Ok(None) => {} // writer refused: fine, no unreadable file was produced
        Ok(Some(got)) => assert_eq!(got, v),
        Err(e) => panic!("write_to returned Ok but the file cannot be read back: {e}"),
    }
}
