// Repro for F003 (run 2026-10-23-batch3).
//
// A `.jix` archive whose bool array holds bytes other than 0/1 passes Strict validation, and the
// decoded bytes reach typed Rust `bool` through the safe API (`to_ndarray`, `where_condition`,
// `cast`, `!`, `sum`). A `bool` byte other than 0/1 is an invalid value: UB from safe code.
//
// The archive is built by writing a u8 array and patching the single header byte that encodes
// the scalar kind (U8 = 5) to the bool kind (Bool = 14). The block bytes are left untouched.
//
// Each test asserts the CORRECT behavior: the load or the read fails, or the values come back
// normalized to 0/1 with nonzero-is-true semantics. So every test FAILS while the bug exists.
// `miri_cast_reads_invalid_bool` is the tiny test meant for `python jix/run_miri.py`.
use std::io::Cursor;

use jix::dtype::Dtyped;
use jix::ops::where_condition;
use jix::{ArchiveValidation, Array, ArrayParams};
use ndarray::{Array1, Ix0, Ix1, array};

fn archive_bytes<T: Dtyped + Clone>(v: Vec<T>) -> Vec<u8> {
    let a = Array::compact_ndarray(&Array1::from(v)).unwrap();
    let mut buf = Cursor::new(Vec::new());
    a.write_to(&mut buf).unwrap();
    buf.into_inner()
}

/// Archive of `raw` written as u8, with the header's scalar kind patched to bool. Asserts that the
/// patch is exactly one byte, U8 (5) -> Bool (14), and that it sits before the block data.
fn bool_archive_with_bytes(raw: &[u8]) -> Vec<u8> {
    let mut u = archive_bytes(raw.to_vec());
    let b = archive_bytes(vec![false; raw.len()]);
    assert_eq!(u.len(), b.len());
    let i = u.iter().zip(&b).position(|(x, y)| x != y).unwrap();
    assert_eq!((u[i], b[i]), (5, 14), "expected the scalar-kind byte at {i}");
    u[i] = b[i];
    u
}

fn load(bytes: Vec<u8>) -> Result<Array<jix::storage::Compact<jix::TypeDyn>>, jix::Error> {
    Array::read_from_reader(
        Cursor::new(bytes),
        None,
        ArrayParams::default(),
        ArchiveValidation::Strict,
    )
}

/// Raw bytes of a bool slice, read as u8 (never as a typed `bool`).
fn bytes_of(v: &Array1<bool>) -> Vec<u8> {
    let s = v.as_slice().unwrap();
    unsafe { std::slice::from_raw_parts(s.as_ptr().cast::<u8>(), s.len()) }.to_vec()
}

const RAW: [u8; 4] = [0, 1, 2, 255];

/// `to_ndarray` of the loaded array must not hand out `bool`s whose byte is not 0/1.
#[test]
fn decoded_bools_are_valid() {
    let Ok(a) = load(bool_archive_with_bytes(&RAW)) else {
        return; // rejecting the archive at load time is a correct fix
    };
    let a = a.into_typed::<bool>().unwrap();
    let Ok(got) = a.to_ndarray::<Ix1>() else {
        return; // rejecting it at read time is a correct fix too
    };
    let bytes = bytes_of(&got);
    println!("decoded bool bytes: {bytes:?}");
    assert!(bytes.iter().all(|&b| b <= 1), "invalid bool bytes {bytes:?}");
}

/// The kernels must agree with each other (and with the nonzero-is-true reading of the file).
/// Today: byte 2 is false and byte 255 is true in where/cast/sum.
#[test]
fn kernels_agree_on_archive_bools() {
    let Ok(a) = load(bool_archive_with_bytes(&RAW)) else {
        return;
    };
    let a = a.into_typed::<bool>().unwrap();
    let x = Array::compact_ndarray(&array![10i32, 11, 12, 13]).unwrap();
    let y = Array::compact_ndarray(&array![20i32, 21, 22, 23]).unwrap();
    let w = where_condition(a.view(), x, y).to_ndarray::<Ix1>();
    let c = a.view().cast::<i32>().to_ndarray::<Ix1>();
    let n = (!a.view()).to_ndarray::<Ix1>().map(|n| bytes_of(&n));
    let s = a.view().sum(0).to_ndarray::<Ix0>().map(|s| s[[]]);
    println!("where: {w:?}");
    println!("cast to i32: {c:?}");
    println!("not (bytes): {n:?}");
    println!("sum: {s:?}");
    let (Ok(w), Ok(c), Ok(n), Ok(s)) = (w, c, n, s) else {
        return; // the read was rejected: correct
    };
    assert_eq!(w, array![20, 11, 12, 13], "where");
    assert_eq!(c, array![0, 1, 1, 1], "cast to i32");
    assert_eq!(n, vec![1, 0, 0, 0], "not");
    assert_eq!(s, 3, "sum");
}

/// The same for a struct dtype with a bool field: the struct values handed out by `to_ndarray`
/// must not carry an invalid `bool`.
#[derive(Copy, Clone, Debug, Dtyped)]
#[repr(C)]
struct WithBool {
    a: u8,
    b: bool,
}
#[derive(Copy, Clone, Debug, Dtyped)]
#[repr(C)]
#[allow(dead_code)]
struct WithU8 {
    a: u8,
    b: u8,
}

#[test]
fn struct_bool_field_is_valid() {
    let mut u = archive_bytes(vec![WithU8 { a: 7, b: 2 }, WithU8 { a: 8, b: 255 }]);
    let t = archive_bytes(vec![WithBool { a: 0, b: false }; 2]);
    assert_eq!(u.len(), t.len());
    let i = u.iter().zip(&t).position(|(x, y)| x != y).unwrap();
    assert_eq!((u[i], t[i]), (5, 14), "expected the field scalar-kind byte at {i}");
    u[i] = t[i];
    let Ok(a) = load(u) else {
        return;
    };
    let a = a.into_typed::<WithBool>().unwrap();
    let Ok(got) = a.to_ndarray::<Ix1>() else {
        return;
    };
    let s = got.as_slice().unwrap();
    // Look at the `b` bytes as u8 (offset 1 in the repr(C) layout).
    let raw = unsafe { std::slice::from_raw_parts(s.as_ptr().cast::<u8>(), 2 * s.len()) };
    let b_bytes: Vec<u8> = raw.chunks(2).map(|c| c[1]).collect();
    println!("struct b bytes: {b_bytes:?}");
    assert!(b_bytes.iter().all(|&b| b <= 1), "invalid bool field bytes {b_bytes:?}");
}

/// Tiny version for Miri: one typed read of the decoded bools (bool -> i32 cast).
#[test]
fn miri_cast_reads_invalid_bool() {
    let Ok(a) = load(bool_archive_with_bytes(&[2])) else {
        return;
    };
    let a = a.into_typed::<bool>().unwrap();
    let Ok(c) = a.cast::<i32>().to_ndarray::<Ix1>() else {
        return;
    };
    println!("cast [2] to i32: {c:?}");
    assert_eq!(c, array![1]);
}
