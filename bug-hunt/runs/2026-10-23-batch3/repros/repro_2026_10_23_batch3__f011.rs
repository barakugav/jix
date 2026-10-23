// Bug hunt 2026-10-23-batch3, finding F011: `Dtype::new_struct` (and so the archive reader, which
// builds struct dtypes through it) accepts duplicate field names, while `Dtype::from_fields`
// rejects them. With duplicates, `dtype_sub_field` silently picks the first match and the second
// field can never be selected.
//
// The repro asserts the CORRECT behavior (duplicate names rejected), so it fails while the bug
// exists.
use std::io::Cursor;

use jix::dtype::{Alignment, Dtype, Dtyped};
use jix::{ArchiveValidation, Array, ArrayParams};
use ndarray::{Ix1, array};

fn dup_fields() -> Vec<(String, u16, Dtype)> {
    vec![
        ("x".to_string(), 0, u8::DTYPE),
        ("x".to_string(), 1, u8::DTYPE),
    ]
}

#[test]
fn new_struct_rejects_duplicate_field_names_like_from_fields() {
    let from_fields = Dtype::from_fields(dup_fields());
    println!(
        "from_fields: {:?}",
        from_fields.as_ref().map(|d| d.itemsize()).map_err(|e| e.to_string())
    );
    let new_struct = Dtype::new_struct(dup_fields(), &[], 2, Alignment::new(1).unwrap());
    println!(
        "new_struct:  {:?}",
        new_struct.as_ref().map(|d| d.to_string()).map_err(|e| e.to_string())
    );
    assert!(from_fields.is_err());
    // The two constructors validate the same field list differently.
    assert!(
        new_struct.is_err(),
        "new_struct accepted duplicate field names: {:?}",
        new_struct.map(|d| d.to_string())
    );
}

#[test]
fn archive_with_duplicate_field_names_is_rejected() {
    // Build the dtype through the reader's own constructor path. If new_struct ever starts
    // rejecting it, the test passes here.
    let Ok(dup) = Dtype::new_struct(dup_fields(), &[], 2, Alignment::new(1).unwrap()) else {
        return;
    };
    // Bytes 01 02 | 03 04 viewed as two elements {x: u8 @0, x: u8 @1}.
    let a = Array::compact_ndarray(&array![0x0201u16, 0x0403]).unwrap();
    // SAFETY: the target dtype only has u8 fields, so every byte pattern is valid.
    let a = unsafe { a.transmute_elements_dyn(dup) };
    let mut buf = Cursor::new(Vec::new());
    a.write_to(&mut buf).unwrap();

    let r = Array::read_from_reader(
        Cursor::new(buf.into_inner()),
        None,
        ArrayParams::default(),
        ArchiveValidation::Strict,
    );
    match &r {
        Ok(arr) => {
            let x = arr.view().dtype_sub_field::<u8>("x").to_ndarray::<Ix1>().unwrap();
            println!(
                "archive loaded, dtype {}; field \"x\" = {x:?}; bytes 2 and 4 cannot be selected",
                arr.dtype()
            );
        }
        Err(e) => println!("archive rejected: {e}"),
    }
    assert!(
        r.is_err(),
        "archive whose struct dtype has two fields named \"x\" was accepted"
    );
}
