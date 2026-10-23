// Repro for F008 (run 2026-10-23-batch3): writing an archive into an append-mode (O_APPEND)
// file returns Ok, but the archive cannot be read back.
//
// BlockArchiveWriter::start writes a zero TOC placeholder and seeks forward to the data section;
// finalize seeks back to patch the block locations and the TOC. With O_APPEND every write goes
// to EOF, so the placeholder stays zero and the real locations / TOC land after the data.
//
// Correct behavior (asserted here): either write_to returns an Err, or the file reads back with
// the written values. The test fails while the bug exists.
use std::fs::OpenOptions;
use std::path::Path;

use jix::{ArchiveValidation, Array, ArrayParams};
use ndarray::array;

fn read_back_i32(path: &Path, offset: u64, len: u64) -> ndarray::ArrayD<i32> {
    let r = Array::read_from_file_section(
        path,
        offset,
        len,
        ArrayParams::default(),
        ArchiveValidation::Strict,
    );
    match r {
        Ok(b) => b.into_typed::<i32>().unwrap().to_ndarray().unwrap(),
        Err(e) => panic!("write_to returned Ok(()) but the archive does not read back: {e}"),
    }
}

/// A fresh file opened with `append(true)`.
#[test]
fn append_mode_fresh_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fresh.jix");
    let a = Array::compact_ndarray(&array![1i32, 2, 3, 4, 5]).unwrap();
    let res = {
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .unwrap();
        a.write_to(&mut f)
    };
    if let Err(e) = res {
        // Rejecting a writer whose writes ignore seeks is an acceptable outcome.
        eprintln!("write_to rejected the append-mode writer: {e}");
        return;
    }
    let len = std::fs::metadata(&path).unwrap().len();
    eprintln!("write_to returned Ok; file is {len} bytes");
    assert_eq!(read_back_i32(&path, 0, len), array![1i32, 2, 3, 4, 5].into_dyn());
}

/// The documented "pack two arrays back-to-back" use case, with the second array appended
/// through an append-mode handle.
#[test]
fn append_mode_second_packed_array() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("packed.jix");
    let a = Array::compact_ndarray(&array![1i32, 2, 3]).unwrap();
    let b = Array::compact_ndarray(&array![10i32, 20, 30, 40]).unwrap();
    a.write_to_file(&path).unwrap();
    let offset = std::fs::metadata(&path).unwrap().len();
    let res = {
        let mut f = OpenOptions::new().append(true).open(&path).unwrap();
        b.write_to(&mut f)
    };
    if let Err(e) = res {
        eprintln!("write_to rejected the append-mode writer: {e}");
        return;
    }
    let total = std::fs::metadata(&path).unwrap().len();
    eprintln!("write_to returned Ok; first array {offset} bytes, file {total} bytes");
    // The first array is untouched.
    assert_eq!(read_back_i32(&path, 0, offset), array![1i32, 2, 3].into_dyn());
    assert_eq!(
        read_back_i32(&path, offset, total - offset),
        array![10i32, 20, 30, 40].into_dyn()
    );
}
