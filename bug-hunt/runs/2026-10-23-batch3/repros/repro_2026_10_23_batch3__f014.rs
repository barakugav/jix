// Verifier repro for F014 (run 2026-10-23-batch3).
//
// ArchiveReader::read_uleb128 (jix/src/archive/common.rs:184) has no cap on the number of
// bytes and shifts by `i * 7` unchecked. The archive's length prefixes are written by prost's
// `encode_length_delimited`, whose reader counterpart (`decode_varint`) rejects a varint longer
// than 10 bytes. These tests assert that jix rejects an 11-byte length prefix with an Err.
//
// While the bug exists:
// - debug build: panics "attempt to shift left with overflow" at common.rs:184;
// - release build: the 11-byte prefix is accepted (shift 70 is masked to 6), so the archive
//   with an overlong but otherwise valid prefix reads successfully, and high groups are OR-ed
//   into low bits.
use std::io::Cursor;

use jix::{ArchiveValidation, Array, ArrayParams};
use ndarray::{Ix1, array};

fn read(bytes: Vec<u8>) -> Result<jix::ArrayAny, jix::Error> {
    let len = bytes.len() as u64;
    Array::read_from_reader(
        Cursor::new(bytes),
        Some(len),
        ArrayParams::default(),
        ArchiveValidation::Strict,
    )
    .map(|a| a.into_any())
}

/// Position just past the three length-delimited messages (FileMetadata, ArrayHeader,
/// BlockTableHeader), i.e. the start of the raw TOC `[Section; 2]` ({offset: i64, size: u64}).
fn toc_pos(bytes: &[u8]) -> usize {
    let mut pos = 4;
    for _ in 0..3 {
        let (mut len, mut shift) = (0u64, 0);
        loop {
            let b = bytes[pos];
            pos += 1;
            len |= u64::from(b & 0x7f) << shift;
            if b < 0x80 {
                break;
            }
            shift += 7;
        }
        pos += len as usize;
    }
    pos
}

fn valid_archive() -> Vec<u8> {
    let a = Array::compact_ndarray(&array![1u8, 2, 3, 4, 5, 6, 7, 8]).unwrap();
    let mut buf = Cursor::new(Vec::new());
    a.write_to(&mut buf).unwrap();
    buf.into_inner()
}

/// Hunter's trigger: magic + 10 continuation bytes + a terminating 0x00 as the FileMetadata
/// length. Expected: Err (varint longer than 10 bytes).
#[test]
fn uleb128_eleven_bytes_is_err() {
    let mut bytes = b"jix1".to_vec();
    bytes.extend([0x80u8; 10]);
    bytes.push(0x00);
    let r = read(bytes);
    assert!(r.is_err());
}

/// A valid archive whose FileMetadata length prefix (one byte, L < 128) is rewritten as the
/// 11-byte overlong form [L | 0x80, 0x80 x 9, 0x00]. Expected: Err. Release: reads fine.
#[test]
fn valid_archive_with_overlong_length_prefix_is_rejected() {
    let orig = valid_archive();
    let l = orig[4];
    assert!(l < 0x80, "FileMetadata length should be a one-byte uleb128");
    // Canonical file reads fine.
    let a = Array::read_from_reader(
        Cursor::new(orig.clone()),
        None,
        ArrayParams::default(),
        ArchiveValidation::Strict,
    )
    .unwrap()
    .into_typed::<u8>()
    .unwrap();
    assert_eq!(
        a.to_ndarray::<Ix1>().unwrap(),
        array![1u8, 2, 3, 4, 5, 6, 7, 8]
    );

    let mut bytes = b"jix1".to_vec();
    bytes.push(l | 0x80);
    bytes.extend([0x80u8; 9]);
    bytes.push(0x00);
    bytes.extend_from_slice(&orig[5..]);
    // The prefix grew by 10 bytes: shift both TOC section offsets (relative to the archive
    // start) by 10 so that only the length encoding differs from the canonical file.
    let toc = toc_pos(&orig) + 10;
    for k in 0..2 {
        let p = toc + 16 * k;
        let off = i64::from_le_bytes(bytes[p..p + 8].try_into().unwrap());
        bytes[p..p + 8].copy_from_slice(&(off + 10).to_le_bytes());
    }
    let r = read(bytes);
    assert!(
        r.is_err(),
        "an 11-byte uleb128 length prefix was accepted: {:?}",
        r.map(|a| a.shape().to_vec())
    );
}
