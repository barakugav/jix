//! Repro for K189 (run 2026-11-16-scan-106-107), 32-bit only.
//!
//! `Owned::read_section` (jix/src/archive/block.rs:394-410) only checks that the section's
//! ELEMENT count fits `usize`, then grows a `Vec<MaybeUninit<T>>` towards that many elements in
//! doubling chunks. On 32-bit a section of more than isize::MAX bytes passes the check, the
//! section bounds check passes too (since #107 the bound is the real file size, and the file
//! really is that large), and the next `Vec::resize` past isize::MAX bytes panics with
//! "capacity overflow" after 1 GiB (block_data, u8) or 1.5 GiB (blocks_loc, 12-byte locations)
//! has been read into memory. The correct result is an `Err` (the section cannot fit in memory
//! on this target).
//!
//! The archive is a valid 4-element i32 archive whose TOC is patched so one section is just over
//! 2 GiB; the file is extended sparsely so the section lies inside it.
//!
//! Run (the tests are ignored on 64-bit, where a 2 GiB section fits):
//!   bug-hunt/tools/cargo-i686 test --manifest-path jix/Cargo.toml --all-features \
//!     --test repro_k189 -- --nocapture --include-ignored --test-threads 1
//!
//! Both tests FAIL (panic "capacity overflow") while the bug exists.

use std::io::Write;
use std::path::Path;

use jix::Array;
use ndarray::array;

fn read_uleb(bytes: &[u8], pos: &mut usize) -> u64 {
    let mut x = 0u64;
    for i in 0..10 {
        let b = bytes[*pos];
        *pos += 1;
        x |= u64::from(b & 0x7f) << (7 * i);
        if b < 0x80 {
            return x;
        }
    }
    panic!("bad uleb");
}

/// Write a valid 4-element i32 archive to `path`, with TOC entry `which` (0 = blocks_loc,
/// 1 = block_data) resized to `new_size` bytes and the file sparsely extended to contain it.
fn write_patched(path: &Path, which: usize, new_size: u64) {
    let a = Array::compact_ndarray(&array![1i32, 2, 3, 4]).unwrap();
    let mut buf = std::io::Cursor::new(Vec::new());
    a.write_to(&mut buf, jix::ArrayParams::default(), &a.read_ctx())
        .unwrap();
    let mut bytes = buf.into_inner();

    // magic, then FileMetadata, ArrayHeader, BlockTableHeader (uleb128-length-delimited), then
    // the TOC: [Section { offset: i64, size: u64 }; 2] for blocks_loc, block_data.
    let mut pos = 4;
    for _ in 0..3 {
        let len = read_uleb(&bytes, &mut pos) as usize;
        pos += len;
    }
    let entry = pos + 16 * which;
    let off = i64::from_le_bytes(bytes[entry..entry + 8].try_into().unwrap());
    let size = u64::from_le_bytes(bytes[entry + 8..entry + 16].try_into().unwrap());
    println!("section {which}: offset {off} size {size} -> {new_size}");
    bytes[entry + 8..entry + 16].copy_from_slice(&new_size.to_le_bytes());

    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(&bytes).unwrap();
    // Sparse extension so the patched section lies inside the file (no disk space used).
    f.set_len((off as u64 + new_size).max(bytes.len() as u64) + 64)
        .unwrap();
}

fn check(which: usize, new_size: u64) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("huge.jix");
    write_patched(&path, which, new_size);
    // Correct behavior: a clean Err (the section is larger than this target can hold).
    // Bug: Vec::resize panics with "capacity overflow" (block.rs:404).
    let res = Array::read_from_file(&path);
    match &res {
        Ok(a) => println!("loaded, shape {:?}", a.shape()),
        Err(e) => println!("Err kind {:?}: {e}", e.kind()),
    }
    assert!(
        res.is_err(),
        "a section of {new_size} bytes cannot be held in memory on this target"
    );
}

/// block_data (u8) section of 2^31 + 4096 bytes: panics after reading 1 GiB.
#[test]
#[cfg_attr(
    not(target_pointer_width = "32"),
    ignore = "32-bit only: on 64-bit the section fits"
)]
fn f006_huge_block_data_section_is_an_err() {
    check(1, (1u64 << 31) + 4096);
}

/// blocks_loc (12-byte BlockLocation) section of 12 * 178956971 = 2^31 + 4 bytes: the element
/// count fits usize, the byte size does not fit isize; panics after reading 1.5 GiB.
#[test]
#[cfg_attr(
    not(target_pointer_width = "32"),
    ignore = "32-bit only: on 64-bit the section fits"
)]
fn f006_huge_blocks_loc_section_is_an_err() {
    check(0, 12 * ((1u64 << 31) / 12 + 1));
}
