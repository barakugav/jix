// Repro for bug-hunt run 2026-10-23-batch3, finding F005.
//
// `ArchiveReader::check_section_bounds` (jix/src/archive/common.rs) evaluates
// `section.offset as u64 + section.size <= total_size` with a plain `+`. A crafted TOC whose
// section offset + size exceeds u64::MAX must be rejected with an InvalidArchive error. Instead:
// - debug build: panic "attempt to add with overflow" in check_section_bounds (Owned and Mmap);
// - release build: the add wraps to a small value, the check passes, and Owned::read_section
//   allocates `vec![_; size as usize]` -> panic "capacity overflow".
//
// Each test asserts the CORRECT behavior (an `Err`), so it fails (panics) while the bug exists.
//
// Run: cargo test --manifest-path jix/Cargo.toml --all-features \
//          --test repro_2026_10_23_batch3__f005 -- --nocapture
use std::io::Cursor;

use jix::{ArchiveValidation, Array, ArrayParams};
use ndarray::array;

/// Serialize a tiny valid archive and return (bytes, byte position of the TOC `[Section; 2]`).
fn valid_archive() -> (Vec<u8>, usize) {
    let a = Array::compact_ndarray(&array![1u8, 2, 3, 4, 5, 6, 7, 8]).unwrap();
    let mut buf = Cursor::new(Vec::new());
    a.write_to(&mut buf).unwrap();
    let bytes = buf.into_inner();
    assert_eq!(&bytes[..4], b"jix1");
    // Layout: "jix1" | uleb128-len FileMetadata | uleb128-len ArrayHeader
    //         | uleb128-len BlockTableHeader | TOC [Section{offset: i64, size: u64}; 2]
    let mut pos = 4;
    for _ in 0..3 {
        let mut len = 0u64;
        let mut shift = 0;
        loop {
            let b = bytes[pos];
            pos += 1;
            len |= u64::from(b & 0x7f) << shift;
            shift += 7;
            if b < 0x80 {
                break;
            }
        }
        pos += len as usize;
    }
    // Sanity check: the second TOC row is block_data and ends exactly at the end of the archive.
    let off = i64::from_le_bytes(bytes[pos + 16..pos + 24].try_into().unwrap());
    let size = u64::from_le_bytes(bytes[pos + 24..pos + 32].try_into().unwrap());
    assert_eq!(off as u64 + size, bytes.len() as u64, "TOC not where expected");
    (bytes, pos)
}

/// Overwrite TOC row `row` (0 = blocks_loc, 1 = block_data) with (offset, size).
fn patch_section(bytes: &mut [u8], toc: usize, row: usize, offset: i64, size: u64) {
    let p = toc + 16 * row;
    bytes[p..p + 8].copy_from_slice(&offset.to_le_bytes());
    bytes[p + 8..p + 16].copy_from_slice(&size.to_le_bytes());
}

fn read_owned(bytes: Vec<u8>) -> Result<(), jix::Error> {
    let len = bytes.len() as u64;
    Array::read_from_reader(
        Cursor::new(bytes),
        Some(len),
        ArrayParams::default(),
        ArchiveValidation::Strict,
    )
    .map(|_| ())
}

/// Sanity: the untouched archive reads back fine.
#[test]
fn unmodified_archive_reads() {
    let (bytes, _) = valid_archive();
    read_owned(bytes).unwrap();
}

/// Trigger 1 (storage-compact-read-path): offset = i64::MAX, size = 2^63 + 2.
/// offset + size = 2^64 + 1, wraps to 1 in release.
#[test]
fn owned_reader_section_offset_plus_size_overflow_is_an_error() {
    let (mut bytes, toc) = valid_archive();
    patch_section(&mut bytes, toc, 1, i64::MAX, (1u64 << 63) + 2);
    let r = read_owned(bytes);
    println!("owned, offset=i64::MAX size=2^63+2: {:?}", r.as_ref().map_err(|e| e.to_string()));
    let err = r.expect_err("crafted section must be rejected");
    assert!(err.to_string().contains("out of archive bounds"), "{err}");
}

/// Trigger 2 (codec-archive-archive-io): offset = 1, size = u64::MAX.
/// offset + size = 2^64, wraps to 0 in release.
#[test]
fn owned_reader_section_size_u64_max_is_an_error() {
    let (mut bytes, toc) = valid_archive();
    patch_section(&mut bytes, toc, 1, 1, u64::MAX);
    let r = read_owned(bytes);
    println!("owned, offset=1 size=u64::MAX: {:?}", r.as_ref().map_err(|e| e.to_string()));
    let err = r.expect_err("crafted section must be rejected");
    assert!(err.to_string().contains("out of archive bounds"), "{err}");
}

/// Same crafted bytes through the mmap reader (which re-checks with checked_add against the
/// mapping, but only after check_section_bounds has already panicked in a debug build).
#[test]
fn mmap_reader_section_offset_plus_size_overflow_is_an_error() {
    let (mut bytes, toc) = valid_archive();
    patch_section(&mut bytes, toc, 1, i64::MAX, (1u64 << 63) + 2);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("crafted.jix");
    std::fs::write(&path, &bytes).unwrap();
    // Safety: the file is private to this test and is not modified while mapped.
    let r = unsafe {
        Array::read_from_file_mmap(
            &path,
            0,
            bytes.len() as u64,
            ArrayParams::default(),
            ArchiveValidation::Strict,
        )
    }
    .map(|_| ());
    println!("mmap, offset=i64::MAX size=2^63+2: {:?}", r.as_ref().map_err(|e| e.to_string()));
    assert!(r.is_err(), "crafted section must be rejected");
}
