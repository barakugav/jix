// Verifier repro for F015 (run 2026-10-23-batch3).
//
// 32-bit only: `Owned::read_section` (jix/src/archive/block.rs:430) truncates `section.size as
// usize`, so `read_section_into` hits `assert_eq!(buf.len() as u64, section.size)`
// (jix/src/archive/common.rs:240) - a panic instead of an Err. Sizes in (isize::MAX, 2^32) panic
// with "capacity overflow" in `vec!` instead. Relatedly, `read_slice(length as usize)`
// (common.rs:161) truncates a uleb128 message length, so a 32-bit build silently accepts a
// malformed archive that a 64-bit build rejects.
//
// Each test asserts the CORRECT behavior (an Err), so it fails while the bug exists. The tests
// are ignored on 64-bit targets because there the same inputs make the reader allocate >= 2 GiB
// before it reports UnexpectedEof (correct, but expensive).
//
// Run (32-bit, where the bug is):
//   bug-hunt/tools/cargo-i686 test --manifest-path jix/Cargo.toml --all-features \
//       --test repro_2026_10_23_batch3__f015 -- --nocapture
use std::io::Cursor;

use jix::{ArchiveValidation, Array, ArrayParams};
use ndarray::array;

fn read_uleb(bytes: &[u8], pos: &mut usize) -> u64 {
    let mut x = 0u64;
    let mut shift = 0;
    loop {
        let b = bytes[*pos];
        *pos += 1;
        x |= u64::from(b & 0x7f) << shift;
        if b < 0x80 {
            return x;
        }
        shift += 7;
    }
}

fn write_uleb(mut x: u64, out: &mut Vec<u8>) {
    loop {
        let b = (x & 0x7f) as u8;
        x >>= 7;
        if x == 0 {
            out.push(b);
            return;
        }
        out.push(b | 0x80);
    }
}

/// A valid 8-element u8 archive. Returns the bytes and the position of the TOC
/// (`[Section; 2]`: blocks_loc row then block_data row, each `{offset: i64, size: u64}`).
fn valid_archive() -> (Vec<u8>, usize) {
    let a = Array::compact_ndarray(&array![1u8, 2, 3, 4, 5, 6, 7, 8]).unwrap();
    let mut buf = Cursor::new(Vec::new());
    a.write_to(&mut buf).unwrap();
    let bytes = buf.into_inner();
    let mut pos = 4; // magic "jix1"
    for _ in 0..3 {
        // FileMetadata, ArrayHeader, BlockTableHeader
        let len = read_uleb(&bytes, &mut pos) as usize;
        pos += len;
    }
    (bytes, pos)
}

fn with_section_size(row: usize, size: u64) -> Vec<u8> {
    let (mut bytes, toc) = valid_archive();
    let p = toc + 16 * row + 8;
    bytes[p..p + 8].copy_from_slice(&size.to_le_bytes());
    bytes
}

fn read(bytes: Vec<u8>, len: Option<u64>) -> Result<(), jix::Error> {
    Array::read_from_reader(
        Cursor::new(bytes),
        len,
        ArrayParams::default(),
        ArchiveValidation::Strict,
    )
    .map(|_| ())
}

#[test]
fn control_valid_archive_reads() {
    let (bytes, _) = valid_archive();
    println!("usize bits: {}, archive bytes: {}", usize::BITS, bytes.len());
    read(bytes, None).unwrap();
}

/// block_data size 2^32 + 1 truncates to 1 on 32-bit: assert_eq panic at common.rs:240.
#[test]
#[cfg_attr(not(target_pointer_width = "32"), ignore = "32-bit only (allocates 4 GiB on 64-bit)")]
fn block_data_size_2_pow_32_plus_1_is_an_error() {
    let r = read(with_section_size(1, (1u64 << 32) + 1), None);
    println!("{r:?}");
    assert!(r.is_err());
}

/// The same with an explicit (too large) caller `len`, as Python `read_array(path, len=...)` or
/// `read_from_file_section` would pass it.
#[test]
#[cfg_attr(not(target_pointer_width = "32"), ignore = "32-bit only (allocates 4 GiB on 64-bit)")]
fn block_data_size_2_pow_32_plus_1_explicit_len_is_an_error() {
    let r = read(with_section_size(1, (1u64 << 32) + 1), Some(1 << 40));
    println!("{r:?}");
    assert!(r.is_err());
}

/// block_data size 2^31 does not truncate but exceeds isize::MAX on 32-bit: `vec!` panics with
/// "capacity overflow".
#[test]
#[cfg_attr(not(target_pointer_width = "32"), ignore = "32-bit only (allocates 2 GiB on 64-bit)")]
fn block_data_size_2_pow_31_is_an_error() {
    let r = read(with_section_size(1, 1u64 << 31), None);
    println!("{r:?}");
    assert!(r.is_err());
}

/// blocks_loc size 2^32 + 8 (a multiple of size_of::<BlockLocation>() == 12) truncates to 8,
/// i.e. 0 elements: assert_eq panic at common.rs:240 (left: 0).
#[test]
#[cfg_attr(not(target_pointer_width = "32"), ignore = "32-bit only (allocates 4 GiB on 64-bit)")]
fn blocks_loc_size_2_pow_32_plus_8_is_an_error() {
    let r = read(with_section_size(0, (1u64 << 32) + 8), None);
    println!("{r:?}");
    assert!(r.is_err());
}

/// The FileMetadata length prefix is rewritten to `L + 2^32` (5-byte uleb128). The archive does
/// not contain 4 GiB of metadata, so the read must fail. On 32-bit `length as usize` truncates
/// to `L` (common.rs:161) and the malformed archive loads successfully.
#[test]
#[cfg_attr(not(target_pointer_width = "32"), ignore = "32-bit only (allocates 4 GiB on 64-bit)")]
fn message_length_over_2_pow_32_is_an_error() {
    let (bytes, toc) = valid_archive();
    let mut pos = 4;
    let meta_len = read_uleb(&bytes, &mut pos);
    let old_prefix_len = pos - 4;

    let mut out = bytes[..4].to_vec();
    write_uleb(meta_len + (1u64 << 32), &mut out);
    let delta = (out.len() - 4 - old_prefix_len) as i64;
    out.extend_from_slice(&bytes[pos..]);
    // Section offsets are relative to the archive start: shift both by the prefix growth.
    let toc = toc + delta as usize;
    for row in 0..2 {
        let p = toc + 16 * row;
        let off = i64::from_le_bytes(out[p..p + 8].try_into().unwrap()) + delta;
        out[p..p + 8].copy_from_slice(&off.to_le_bytes());
    }

    let r = read(out.clone(), None);
    println!("len=None: {r:?}");
    let r2 = read(out.clone(), Some(out.len() as u64));
    println!("len=Some({}): {r2:?}", out.len());
    assert!(r.is_err(), "4 GiB message length accepted with len=None");
    assert!(r2.is_err(), "4 GiB message length accepted with exact len");
}
