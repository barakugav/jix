//! F038: `write_to_file` of a lazy view whose read fails part-way (here an integer divide by zero
//! in the last block, an error under the default arithmetic modes) returns `Err` but leaves a
//! partially written, unreadable `.jix` file at `path`. Because `write_to_file` uses
//! `create_new`, a retry at the same path then fails with `AlreadyExists`.
//!
//! The test asserts the CORRECT behavior (no file left behind after a failed write), so it fails
//! while the bug exists.
use jix::error::ErrorKind;
use jix::{Array, ArrayParams};
use ndarray::Array1;

#[test]
fn failed_write_to_file_leaves_no_file() {
    let n = 200_000i32;
    let mut params = ArrayParams::new();
    params.block_shape(&[1000]);
    let a = Array::compact_ndarray_with(&Array1::from_iter(1..=n), params.clone()).unwrap();
    let mut b = Array1::from_elem(n as usize, 1i32);
    b[n as usize - 1] = 0; // divide by zero in the LAST block only
    let b = Array::compact_ndarray_with(&b, params).unwrap();
    let q = a / b;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("q.jix");
    let err = q.write_to_file(&path).unwrap_err();
    println!("write_to_file error: {err:?}");
    assert_eq!(err.kind(), ErrorKind::Arithmetic);

    let exists = path.exists();
    let size = std::fs::metadata(&path).map(|m| m.len()).ok();
    println!("file exists after failed write: {exists}, size: {size:?}");
    if exists {
        let read_back = Array::read_from_file(&path, ArrayParams::default());
        println!("read back partial file: {:?}", read_back.as_ref().err());
        let retry = q.write_to_file(&path);
        println!("retry at the same path: {:?}", retry.as_ref().err());
    }
    assert!(
        !exists,
        "failed write_to_file left a {size:?}-byte partial file behind at the new path"
    );
}
