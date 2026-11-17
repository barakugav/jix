//! K184 repro: an archive written by the published jix 0.0.7 release (2b229d2) no longer loads.
//!
//! #106 (1169965) reused FileMetadata field 2: it was `uint64 lib_version_semver` (a varint) and
//! is now `Semver lib_version` (a length-delimited message). prost rejects a known field that
//! arrives with the wrong wire type, so decoding the FileMetadata of every pre-#106 archive fails.
//!
//! `JIX_0_0_7_ARCHIVE` is NOT synthetic: it is the exact file written by jix 0.0.7 (built from
//! 2b229d2 in a scratch copy) for `array![[1i32, 2, 3], [4, 5, 6]]` via `write_to_file`. Its
//! FileMetadata is the 7 bytes at offset 4: len=6, `08 02` (archive_type = ARRAY_V1),
//! `10 80 80 1c` (field 2, varint, 458752 = 7 << 16 = "0.0.7" in the old packed layout).
//!
//! The repro asserts the CORRECT behavior (the file loads), so it fails while the bug exists.
//! `control_*` tests pass today and show that field 2 is the only obstacle.

use std::io::Cursor;

use jix::{ArchiveValidation, Array, ArrayParams};
use ndarray::array;

#[rustfmt::skip]
const JIX_0_0_7_ARCHIVE: [u8; 113] = [
    0x6a, 0x69, 0x78, 0x31, 0x06, 0x08, 0x02, 0x10, 0x80, 0x80, 0x1c, 0x08, 0x0a, 0x02, 0x02, 0x03,
    0x12, 0x02, 0x02, 0x03, 0x1a, 0x0a, 0x0a, 0x0a, 0x04, 0x08, 0x03, 0x10, 0x01, 0x20, 0x04, 0x28,
    0x04, 0x10, 0x06, 0x18, 0x06, 0x22, 0x02, 0x0a, 0x00, 0x2a, 0x02, 0x0a, 0x00, 0x3a, 0x00, 0x4f,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x5b,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x16, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x16, 0x00, 0x00, 0x00, 0x28, 0xb5, 0x2f, 0xfd, 0x20,
    0x18, 0x6d, 0x00, 0x00, 0x38, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x00, 0x01, 0x00, 0x33, 0x30,
    0x02,
];

fn expected() -> ndarray::ArrayD<i32> {
    array![[1i32, 2, 3], [4, 5, 6]].into_dyn()
}

fn read(bytes: Vec<u8>) -> Result<ndarray::ArrayD<i32>, jix::Error> {
    let a = Array::read_from(
        Cursor::new(bytes),
        None,
        ArrayParams::default(),
        ArchiveValidation::Strict,
    )?;
    a.into_typed::<i32>()?.to_ndarray()
}

/// The bug: a file written by the previous release does not load.
#[test]
fn archive_written_by_jix_0_0_7_loads() {
    let res = read(JIX_0_0_7_ARCHIVE.to_vec());
    if let Err(e) = &res {
        println!("read_from error, kind {:?}: {e}", e.kind());
    }
    assert_eq!(res.expect("an archive written by jix 0.0.7 should load"), expected());
}

/// Same through the file entry point (the one `jix.read_array` uses).
#[test]
fn archive_file_written_by_jix_0_0_7_loads() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("old.jix");
    std::fs::write(&path, JIX_0_0_7_ARCHIVE).unwrap();
    let res = Array::read_from_file(&path).and_then(|a| a.into_typed::<i32>()?.to_ndarray());
    if let Err(e) = &res {
        println!("read_from_file error, kind {:?}: {e}", e.kind());
    }
    assert_eq!(res.expect("an archive file written by jix 0.0.7 should load"), expected());
}

/// Control: the same 0.0.7 file with only field 2 of its FileMetadata re-encoded in the new
/// form, `10 80 80 1c` (varint 7 << 16) -> `12 02 18 07` (`Semver { patch: 7 }`, same length so
/// the section offsets stay valid), loads with the right contents. So the rest of the 0.0.7
/// format is still compatible and the reused field number is the only obstacle.
#[test]
fn control_0_0_7_archive_with_new_field_2_loads() {
    let mut bytes = JIX_0_0_7_ARCHIVE.to_vec();
    assert_eq!(&bytes[4..11], &[0x06, 0x08, 0x02, 0x10, 0x80, 0x80, 0x1c]);
    bytes[7..11].copy_from_slice(&[0x12, 0x02, 0x18, 0x07]);
    assert_eq!(read(bytes).unwrap(), expected());
}
