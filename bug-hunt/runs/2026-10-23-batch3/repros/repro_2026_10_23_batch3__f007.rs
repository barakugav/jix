// Repro for bug-hunt run 2026-10-23-batch3, finding F007.
//
// NOTE: while the bug exists, the `*_is_an_error` tests ABORT the whole test process
// ("memory allocation of N bytes failed", SIGABRT). Run them one at a time:
//
//   cargo test --manifest-path jix/Cargo.toml --all-features \
//       --test repro_2026_10_23_batch3__f007 -- --nocapture --test-threads 1 <test_name>
//
// `Array::read_from_reader(reader, None, ..)` (len = None, the Python default for file-like
// readers) sets `total_size = u64::MAX` (jix/src/archive/common.rs:116), so
// `check_section_bounds` accepts any section size whose end does not overflow u64.
// `Owned::read_section` (jix/src/archive/block.rs:433) then allocates and zero-fills
// `section.size` bytes BEFORE reading a single byte of the section. A ~100-byte archive whose TOC
// declares a 16 TiB section aborts the process; one that declares 1 GiB commits 1 GiB before
// failing with UnexpectedEof.
//
// Each test asserts the CORRECT behavior (an `Err`, with memory use bounded by the bytes that
// actually exist), so it fails (aborts / assertion failure) while the bug exists.
use std::io::Cursor;

use jix::{ArchiveValidation, Array, ArrayParams};
use ndarray::array;

/// Serialize a tiny valid archive and return (bytes, byte position of the TOC `[Section; 2]`).
fn valid_archive() -> (Vec<u8>, usize) {
    let a = Array::compact_ndarray(&array![0u8, 1, 2, 3, 4, 5, 6, 7]).unwrap();
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

/// Overwrite the size of TOC row `row` (0 = blocks_loc, 1 = block_data).
fn patch_size(bytes: &mut [u8], toc: usize, row: usize, size: u64) {
    let p = toc + 16 * row + 8;
    bytes[p..p + 8].copy_from_slice(&size.to_le_bytes());
}

/// `Array::read_from_reader` with `len = None` (no explicit archive length).
fn read_no_len(bytes: Vec<u8>) -> Result<(), jix::Error> {
    Array::read_from_reader(
        Cursor::new(bytes),
        None,
        ArrayParams::default(),
        ArchiveValidation::Strict,
    )
    .map(|_| ())
}

/// Peak resident set size of this process in KiB (Linux `/proc/self/status` VmHWM).
fn peak_rss_kib() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let line = status.lines().find(|l| l.starts_with("VmHWM:")).unwrap();
    line.split_whitespace().nth(1).unwrap().parse().unwrap()
}

/// Sanity: the untouched archive reads back fine with `len = None`.
#[test]
fn unmodified_archive_reads_without_len() {
    let (bytes, _) = valid_archive();
    let n = bytes.len();
    read_no_len(bytes).unwrap();
    println!("valid archive: {n} bytes");
}

/// block_data section size patched to 2^44 (16 TiB) in a ~100-byte archive.
/// Expected: Err (the section is longer than the stream). Actual: process abort.
#[test]
fn huge_block_data_section_without_len_is_an_error() {
    let (mut bytes, toc) = valid_archive();
    patch_size(&mut bytes, toc, 1, 1u64 << 44);
    println!("archive: {} bytes, block_data size = 2^44", bytes.len());
    let r = read_no_len(bytes);
    println!("result: {r:?}");
    assert!(r.is_err());
}

/// blocks_loc section size patched to 12 * 2^40 (12 TiB, a multiple of size_of::<BlockLocation>()).
/// block_data is read first and is valid; blocks_loc then allocates before any check against
/// nblocks (which is known from the header). Expected: Err. Actual: process abort.
#[test]
fn huge_blocks_loc_section_without_len_is_an_error() {
    let (mut bytes, toc) = valid_archive();
    patch_size(&mut bytes, toc, 0, 12u64 << 40);
    println!("archive: {} bytes, blocks_loc size = 12 * 2^40", bytes.len());
    let r = read_no_len(bytes);
    println!("result: {r:?}");
    assert!(r.is_err());
}

/// block_data section size patched to 1 GiB: the allocation succeeds, so the read returns
/// UnexpectedEof - but only after committing (zero-filling) 1 GiB for a ~100-byte input.
/// Expected: Err with peak RSS growth bounded by the bytes that exist. Actual: ~1 GiB growth.
#[test]
fn gib_block_data_section_without_len_does_not_commit_the_declared_size() {
    let (mut bytes, toc) = valid_archive();
    patch_size(&mut bytes, toc, 1, 1u64 << 30);
    let before = peak_rss_kib();
    let r = read_no_len(bytes);
    let after = peak_rss_kib();
    println!(
        "result: {r:?}; peak RSS before = {before} KiB, after = {after} KiB, growth = {} MiB",
        (after - before) / 1024
    );
    assert!(r.is_err());
    assert!(
        after - before < 64 * 1024,
        "a ~100-byte archive grew peak RSS by {} MiB",
        (after - before) / 1024
    );
}
